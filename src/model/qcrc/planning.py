"""QCRC step (i): question-conditioned planning (paper Fig. 9).

Given the full schema of D_m and one question, the planner produces:
    {needed: [{table, columns, new_columns: [{name, type, description}, ...]}]}
- `columns` are existing columns the question needs (NULL cells will be backfilled
  in step iii).
- `new_columns` are attributes proposed for materialization (added to side tables
  only -- source tables don't gain columns).
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import compact_schema, corpus, profiler
from src.model.llm_client import make_client
from src.prompt import qcrc as prompt

SRC_DB_DIR = config.DB_DIR
PLAN_DIR = config.DATA_DIR / "plan"


def _profile_conn(conn, tables):
    profiles = {}
    for t in tables:
        df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
        for col in df.columns:
            if df[col].dtype == "float64":
                nn = df[col].dropna()
                if not nn.empty and (nn == nn.astype(int)).all():
                    df[col] = df[col].astype("Int64")
        profiles[t] = profiler.profile_table(df, top_k=config.TOP_K_IN_PROFILE)
    return profiles


def build_full_schema(conn, joinmap):
    side_set = set((joinmap or {}).keys())
    all_tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    source_tables = [t for t in all_tables if t not in side_set]
    side_tables = [t for t in all_tables if t in side_set]
    ordered = source_tables + side_tables
    row_counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in ordered}
    profiles = _profile_conn(conn, ordered)
    base = compact_schema.build_compact_schema(conn, profiles, ordered)
    join_refs = {tbl: refs for tbl, refs in (joinmap or {}).items() if refs}
    out = []
    for line in base.split("\n"):
        m = re.match(r'CREATE TABLE "(\w+)"', line)
        if m and m.group(1) in row_counts:
            tname = m.group(1)
            kind = "side" if tname in side_set else "source"
            out.append(f'-- table "{tname}" ({kind}): {row_counts[tname]} rows')
            if tname in join_refs:
                out.append(f'-- linked_cell joinable with: {", ".join(join_refs[tname])}')
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_fenced_json(raw: str):
    m = _FENCED_JSON_RE.search(raw)
    if m:
        return json.loads(m.group(1))
    return json.loads(raw.strip())


def _resolve_db_path(qid: str) -> Path:
    if config.BENCHMARK == "hybridqa":
        return SRC_DB_DIR / f"{qid}.db"
    if config.BENCHMARK == "sparta":
        config.require_domain()
        return SRC_DB_DIR / f"{config.DOMAIN}.db"
    raise NotImplementedError(config.BENCHMARK)


def _resolve_question(qid: str) -> str:
    if config.BENCHMARK == "hybridqa":
        return corpus.get_record(qid).get("question", "")
    return corpus.get_question(qid).get("question", "")


def _call_plan(client, schema: str, question: str):
    msgs = [
        {"role": "system", "content": prompt.planning_system_prompt.format().strip()},
        {"role": "user",   "content": prompt.planning_user_prompt.format(
            question=question, schema=schema).strip()},
    ]
    t0 = time.perf_counter()
    raw = client.inference(messages=msgs, max_new_tokens=1024, temperature=0.0,
                           response_format={"type": "json_object"})
    dt_ms = (time.perf_counter() - t0) * 1000
    u = client.get_last_call_usage() or {}
    usage = {"prompt_tokens": u.get("prompt_tokens", 0),
             "completion_tokens": u.get("completion_tokens", 0)}
    return raw, usage, dt_ms


def plan_one(qid: str, *, client=None, src_db_path: Path = None, question: str = None) -> dict:
    if client is None:
        client = make_client()
    if src_db_path is None:
        src_db_path = _resolve_db_path(qid)
    if question is None:
        question = _resolve_question(qid)

    wall0 = time.perf_counter()
    if not src_db_path.exists():
        meta = {"qid": qid, "error": "no_db", "src_db": str(src_db_path)}
        PLAN_DIR.mkdir(parents=True, exist_ok=True)
        (PLAN_DIR / f"{qid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    conn = sqlite3.connect(src_db_path)
    joinmap_path = src_db_path.parent / f"{src_db_path.stem}.joinmap.json"
    joinmap = json.loads(joinmap_path.read_text()) if joinmap_path.exists() else {}
    schema = build_full_schema(conn, joinmap)
    conn.close()

    raw, usage, dt_ms = _call_plan(client, schema, question)
    plan_err = None
    try:
        plan = parse_fenced_json(raw)
    except Exception as e:
        plan = {"needed": []}; plan_err = str(e)

    meta = {
        "qid": qid, "src_db": str(src_db_path), "question": question, "schema": schema,
        "plan_raw": raw, "plan": plan, "plan_err": plan_err,
        "usage": usage, "llm_ms": dt_ms,
        "wall_ms": (time.perf_counter() - wall0) * 1000,
    }
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    (PLAN_DIR / f"{qid}.json").write_text(
        json.dumps(meta, ensure_ascii=False, default=str, indent=2))
    return meta


def load_plan(qid: str):
    fp = PLAN_DIR / f"{qid}.json"
    return json.loads(fp.read_text()) if fp.exists() else None

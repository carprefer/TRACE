"""QCRC step (ii)+(iii): embedding-similarity gate + targeted extraction (Fig. 10).

Inputs:
    - plan from step (i)
    - baseline DB D_m at config.DB_DIR/<name>.db (+ urlmap/joinmap sidecars)
    - passages corpus (from src.model.corpus)

Pipeline per question:
    1. Copy D_m to dbs_completed_th{TAG}/<qid>.db
    2. ALTER TABLE for plan's new_columns
    3. For each (table, col) the plan touches:
         backfill: pick NULL rows of an existing column
         new_col : pick all rows
       For each such cell, build a (cell_value, url->passage) extract task.
    4. Apply the embedding-similarity gate (config.SIM_THRESHOLD): cosine(passage, attr_text)
       must exceed threshold or the candidate stays NULL.
    5. For surviving candidates, call the targeted-extraction prompt (Fig. 10);
       UPDATE the DB with the result.
    6. Drop side-table columns that ended up entirely NULL (so SQL never sees them).

Output: dbs_completed_th{TAG}/<qid>.db (D_q) + per-qid meta JSON.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus, retriever as rt
from src.model.llm_client import make_client
from src.model.qcrc import planning as plan_mod
from src.prompt import qcrc as prompt

ENRICHED_DIR = config.DB_COMPLETED
META_DIR     = config.DATA_DIR / f"completion_th{int(round(config.SIM_THRESHOLD * 100)):02d}"

SQL_KIND_OF = {"int": "INTEGER", "float": "REAL", "date": "TEXT", "text": "TEXT"}

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _parse_fenced_json(raw: str):
    m = _FENCED_JSON_RE.search(raw)
    if m:
        return json.loads(m.group(1))
    return json.loads(raw.strip())


def _coerce_value(v, t: str):
    if v is None:
        return None
    if t == "int":
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v) if v == int(v) else None
        try:
            s = str(v).replace(",", "").strip()
            return int(float(s)) if s else None
        except Exception:
            return None
    if t == "float":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return None
    if t == "date":
        s = str(v).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        return None
    return None if v is None else str(v)


def _extract_one(client, question: str, name: str, typ: str, desc: str, cell_value, passage: str):
    u = prompt.completion_user_prompt.format(
        question=question, name=name, type=typ, description=desc or "",
        cell_value=cell_value, passage=passage,
    )
    msgs = [{"role": "system", "content": prompt.completion_system_prompt.format().strip()},
            {"role": "user",   "content": u}]
    try:
        raw = client.inference(messages=msgs, max_new_tokens=256, temperature=0.0,
                               response_format={"type": "json_object"})
        obj = _parse_fenced_json(raw)
        return _coerce_value(obj.get("value"), typ)
    except Exception:
        return None


def _resolve_src_db(qid: str) -> Path:
    if config.BENCHMARK == "hybridqa":
        return config.DB_DIR / f"{qid}.db"
    if config.BENCHMARK == "sparta":
        config.require_domain()
        return config.DB_DIR / f"{config.DOMAIN}.db"
    raise NotImplementedError(config.BENCHMARK)


def _resolve_inputs(qid: str):
    src_db = _resolve_src_db(qid)
    if config.BENCHMARK == "hybridqa":
        rec = corpus.get_record(qid)
        return src_db, (rec.get("text") or {}), rec.get("question", "")
    rec = corpus.get_question(qid)
    return src_db, (corpus.get_text_data() or {}), rec.get("question", "")


def _sim_filter(tasks: list, threshold: float, idx_name: str):
    """Filter tasks by cosine(passage, attr_text) >= threshold. Encodes only what's needed."""
    if not tasks:
        return [], {"n_in": 0, "n_kept": 0}
    import numpy as np
    idx = rt.load_index(idx_name)
    url_to_emb = {u: e for u, e in zip(idx["urls"], idx["emb"])} if idx is not None else {}
    fb_psgs = []
    fb_idx = {}
    for t in tasks:
        if t["url"] not in url_to_emb and t["passage"] and t["passage"] not in fb_idx:
            fb_idx[t["passage"]] = len(fb_idx); fb_psgs.append(t["passage"])
    fb_emb = rt._encode(fb_psgs, max_len=config.RETRIEVER_DOC_LEN) if fb_psgs else None

    queries = sorted({t["attr_text"] for t in tasks})
    q_idx = {q: i for i, q in enumerate(queries)}
    q_emb = rt._encode(queries, max_len=128)
    qn = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    kept = []
    for t in tasks:
        psg_emb = url_to_emb.get(t["url"])
        if psg_emb is None and fb_emb is not None and t["passage"] in fb_idx:
            psg_emb = fb_emb[fb_idx[t["passage"]]]
        if psg_emb is None:
            continue
        psg_n = psg_emb / (np.linalg.norm(psg_emb) + 1e-12)
        sim = float(psg_n @ qn[q_idx[t["attr_text"]]])
        t["sim"] = sim
        if sim >= threshold:
            kept.append(t)
    return kept, {"n_in": len(tasks), "n_kept": len(kept), "threshold": threshold}


def _drop_allnull_columns(conn: sqlite3.Connection) -> dict:
    """After completion, side-table columns that ended up all-NULL would distract
    the agent (SELECT empty_col returns nothing and the LLM gives up). Drop them.
    `linked_cell` is the join key and is preserved.
    """
    dropped: dict = {}
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    for t in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
        rm = []
        for c in cols:
            if c == "linked_cell":
                continue
            n_nonnull = conn.execute(
                f'SELECT COUNT(*) FROM "{t}" WHERE "{c}" IS NOT NULL'
            ).fetchone()[0]
            if n_nonnull == 0:
                rm.append(c)
        for c in rm:
            try:
                conn.execute(f'ALTER TABLE "{t}" DROP COLUMN "{c}"')
            except sqlite3.Error:
                continue
        if rm:
            dropped[t] = rm
    if dropped:
        conn.commit()
    return dropped


def complete_one(qid: str, *, client=None) -> dict:
    """Build D_q for one question. Persists per-qid meta + the completed DB."""
    if client is None:
        client = make_client()
    src_db, passages, question = _resolve_inputs(qid)
    wall0 = time.perf_counter()
    if not src_db.exists():
        return {"qid": qid, "error": "no_db", "src_db": str(src_db)}

    joinmap_path = src_db.parent / f"{src_db.stem}.joinmap.json"
    joinmap = json.loads(joinmap_path.read_text()) if joinmap_path.exists() else {}
    side_set = set(joinmap.keys())

    plan_meta = plan_mod.load_plan(qid)
    if plan_meta is None:
        plan_meta = plan_mod.plan_one(qid, client=client, src_db_path=src_db, question=question)
    plan = plan_meta.get("plan") or {"needed": []}
    plan_raw = plan_meta.get("plan_raw"); plan_err = plan_meta.get("plan_err")
    llm_usage = {"plan": plan_meta.get("usage") or {}, "plan_ms": plan_meta.get("llm_ms", 0.0),
                 "extract_calls": 0, "extract_ms": 0.0}

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    dst_db = ENRICHED_DIR / f"{qid}.db"
    shutil.copyfile(src_db, dst_db)
    for suffix in (".urlmap.json", ".joinmap.json"):
        side = src_db.parent / f"{src_db.stem}{suffix}"
        if side.exists():
            shutil.copyfile(side, ENRICHED_DIR / f"{qid}{suffix}")

    urlmap = {}
    urlmap_path = src_db.parent / f"{src_db.stem}.urlmap.json"
    if urlmap_path.exists():
        urlmap = json.loads(urlmap_path.read_text())

    side_touched = []
    new_col_specs = []
    tasks: list = []

    dconn = sqlite3.connect(dst_db)
    for item in (plan.get("needed") or []):
        tname = item.get("table")
        if not tname or tname not in side_set:
            continue
        side_touched.append(tname)
        urls = urlmap.get(tname) or []
        try:
            pragma_cols = {r[1]: r[2] for r in dconn.execute(
                f'PRAGMA table_info("{tname}")').fetchall()}
        except sqlite3.Error:
            continue
        subject_col = "linked_cell" if "linked_cell" in pragma_cols else next(iter(pragma_cols.keys()))

        # backfill existing columns -- only NULL rows
        for col in (item.get("columns") or []):
            if col not in pragma_cols or col == subject_col:
                continue
            try:
                rows = list(dconn.execute(
                    f'SELECT rowid, "{subject_col}" FROM "{tname}" WHERE "{col}" IS NULL'))
            except sqlite3.Error:
                continue
            ptype = pragma_cols[col]
            t = "int" if ptype == "INTEGER" else "float" if ptype == "REAL" else "text"
            for rowid, cell_value in rows:
                ri = rowid - 1
                if ri < 0 or ri >= len(urls):
                    continue
                u = urls[ri]
                if not u:
                    continue
                psg = (passages or {}).get(u)
                if not psg:
                    continue
                tasks.append({
                    "kind": "backfill", "table": tname, "col": col, "type": t, "desc": "",
                    "cell_value": cell_value, "passage": psg, "rowid": rowid, "url": u,
                    "null_rows_in_col": len(rows),
                    "attr_text": f"{cell_value}: {col}",
                })

        # materialize new_columns
        for nc in (item.get("new_columns") or []):
            name = nc.get("name"); typ = nc.get("type") or "text"
            desc = nc.get("description") or ""
            if not name or name in pragma_cols:
                continue
            sql_type = SQL_KIND_OF.get(typ, "TEXT")
            try:
                dconn.execute(f'ALTER TABLE "{tname}" ADD COLUMN "{name}" {sql_type}')
            except sqlite3.Error:
                continue
            pragma_cols[name] = sql_type
            new_col_specs.append((tname, name, typ))
            try:
                rows = list(dconn.execute(f'SELECT rowid, "{subject_col}" FROM "{tname}"'))
            except sqlite3.Error:
                continue
            for rowid, cell_value in rows:
                ri = rowid - 1
                if ri < 0 or ri >= len(urls):
                    continue
                u = urls[ri]
                if not u:
                    continue
                psg = (passages or {}).get(u)
                if not psg:
                    continue
                tasks.append({
                    "kind": "new_col", "table": tname, "col": name, "type": typ, "desc": desc,
                    "cell_value": cell_value, "passage": psg, "rowid": rowid, "url": u,
                    "rows_in_col": len(rows),
                    "attr_text": f"{cell_value}: {name} - {desc}",
                })
    dconn.commit()

    # embedding gate
    idx_name = qid if config.BENCHMARK == "hybridqa" else "all"
    kept_tasks, sim_stats = _sim_filter(tasks, config.SIM_THRESHOLD, idx_name)
    llm_usage["sim_stats"] = sim_stats

    # targeted extraction (Fig. 10)
    results_by_id = {}
    if kept_tasks:
        t_ext0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=config.EXTRACT_WORKERS) as ex:
            futs = {ex.submit(_extract_one, client, question, t["col"], t["type"], t["desc"],
                              t["cell_value"], t["passage"]): id(t) for t in kept_tasks}
            for fut in as_completed(futs):
                tid = futs[fut]
                results_by_id[tid] = fut.result()
        llm_usage["extract_calls"] = len(kept_tasks)
        llm_usage["extract_ms"] = (time.perf_counter() - t_ext0) * 1000

    # apply updates
    stats = {"side_tables_touched": sorted(set(side_touched)),
             "backfilled": {}, "new_columns": {}}
    grouped: dict = {}
    for t in kept_tasks:
        grouped.setdefault((t["table"], t["col"], t["kind"]), []).append(t)
    for (tname, col, kind), col_tasks in grouped.items():
        n_filled = 0
        for t in col_tasks:
            v = results_by_id.get(id(t))
            if v is None:
                continue
            try:
                dconn.execute(f'UPDATE "{tname}" SET "{col}" = ? WHERE rowid = ?', (v, t["rowid"]))
                n_filled += 1
            except sqlite3.Error:
                pass
        if kind == "backfill":
            null_total = col_tasks[0].get("null_rows_in_col", len(col_tasks))
            stats["backfilled"].setdefault(tname, {})[col] = {
                "null_rows": null_total, "filled": n_filled,
            }
        else:
            typ = col_tasks[0]["type"]
            stats["new_columns"].setdefault(tname, {})[col] = {
                "type": typ, "rows_attempted": len(col_tasks), "rows_filled": n_filled,
            }
    for tname, name, typ in new_col_specs:
        stats["new_columns"].setdefault(tname, {}).setdefault(name, {
            "type": typ, "rows_attempted": 0, "rows_filled": 0,
        })
    dconn.commit()
    dropped = _drop_allnull_columns(dconn)
    if dropped:
        stats["dropped_allnull_columns"] = dropped
    dconn.close()

    meta = {
        "qid": qid, "src_db": str(src_db), "completed_db": str(dst_db),
        "plan_raw": plan_raw, "plan": plan, "plan_err": plan_err,
        "completion_stats": stats, "llm_usage": llm_usage,
        "wall_ms": (time.perf_counter() - wall0) * 1000,
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / f"{qid}.json").write_text(
        json.dumps(meta, ensure_ascii=False, default=str, indent=2))
    return meta

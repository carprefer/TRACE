"""QCRC step (i) runner -- question-conditioned planning (Fig. 9)."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.model.llm_client import make_client
from src.model.qcrc import planning as plan_mod
from src.prompt import qcrc as prompt


def _resolve_db_path(qid: str) -> Path:
    if config.BENCHMARK == "hybridqa":
        return plan_mod.SRC_DB_DIR / f"{qid}.db"
    if config.BENCHMARK == "sparta":
        config.require_domain()
        return plan_mod.SRC_DB_DIR / f"{config.DOMAIN}.db"
    raise NotImplementedError(config.BENCHMARK)


def _build_schema_cached(db_path: Path, cache: dict):
    key = str(db_path)
    if key in cache:
        return cache[key]
    if not db_path.exists():
        cache[key] = None
        return None
    conn = sqlite3.connect(db_path)
    joinmap_path = db_path.parent / f"{db_path.stem}.joinmap.json"
    joinmap = json.loads(joinmap_path.read_text()) if joinmap_path.exists() else {}
    schema = plan_mod.build_full_schema(conn, joinmap)
    conn.close()
    cache[key] = schema
    return schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    corpus.preload()
    workers = args.workers or config.default_workers()
    qids = corpus.load_qids(args.qids)
    if not args.force:
        qids = [q for q in qids if not (plan_mod.PLAN_DIR / f"{q}.json").exists()]
    if not qids:
        print("## nothing to do."); return
    print(f"## qids={len(qids)}  workers={workers}", flush=True)

    plan_mod.PLAN_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()
    schema_cache = {}
    tasks = []
    for qid in qids:
        db_path = _resolve_db_path(qid)
        schema = _build_schema_cached(db_path, schema_cache)
        if schema is None:
            (plan_mod.PLAN_DIR / f"{qid}.json").write_text(
                json.dumps({"qid": qid, "error": "no_db", "src_db": str(db_path)},
                           ensure_ascii=False, indent=2))
            continue
        question = plan_mod._resolve_question(qid)
        msgs = [
            {"role": "system", "content": prompt.planning_system_prompt.format().strip()},
            {"role": "user",   "content": prompt.planning_user_prompt.format(
                question=question, schema=schema).strip()},
        ]
        tasks.append({"qid": qid, "src_db": str(db_path), "question": question,
                      "schema": schema, "msgs": msgs})
    print(f"## prepared {len(tasks)} tasks", flush=True)
    if not tasks:
        return

    def _call(task):
        t0 = time.perf_counter()
        try:
            raw = client.inference(messages=task["msgs"], max_new_tokens=1024,
                                   temperature=0.0,
                                   response_format={"type": "json_object"})
        except Exception as e:
            return task, None, str(e), 0, {}
        dt_ms = (time.perf_counter() - t0) * 1000
        u = client.get_last_call_usage() or {}
        usage = {"prompt_tokens": u.get("prompt_tokens", 0),
                 "completion_tokens": u.get("completion_tokens", 0)}
        return task, raw, None, dt_ms, usage

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_call, t) for t in tasks]
        for fut in as_completed(futs):
            task, raw, err, dt_ms, usage = fut.result()
            if err:
                (plan_mod.PLAN_DIR / f"{task['qid']}.json").write_text(
                    json.dumps({"qid": task["qid"], "src_db": task["src_db"],
                                "question": task["question"], "schema": task["schema"],
                                "error": "llm_call_error", "llm_err": err},
                               ensure_ascii=False, default=str, indent=2))
                continue
            try:
                plan = plan_mod.parse_fenced_json(raw); plan_err = None
            except Exception as e:
                plan = {"needed": []}; plan_err = str(e)
            (plan_mod.PLAN_DIR / f"{task['qid']}.json").write_text(json.dumps({
                "qid": task["qid"], "src_db": task["src_db"], "question": task["question"],
                "schema": task["schema"], "plan_raw": raw, "plan": plan,
                "plan_err": plan_err, "usage": usage, "llm_ms": dt_ms,
            }, ensure_ascii=False, default=str, indent=2))
    print(f"## done  wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

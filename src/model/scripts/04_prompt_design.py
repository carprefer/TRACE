"""SGR step (ii.b) runner -- per-side-table extraction-prompt design (Fig. 6)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.model.llm_client import make_client
from src.model.sgr import grouping, schema_induction as ss, prompt_design as pd_


def _llm_call(client, system, user, max_new_tokens=2048):
    return client.inference(
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        max_new_tokens=max_new_tokens, temperature=0.0,
        response_format={"type": "json_object"},
    )


def _llm_task(t, client):
    prep = t["prep"]
    try:
        raw = _llm_call(client, prep["system"], prep["user"])
    except Exception as e:
        return {**t, "raw": None, "overview": None, "attrs": [], "error": str(e)}
    try:
        parsed = pd_.parse_fenced_json(raw)
    except Exception:
        parsed = {}
    overview = parsed.get("overview") if isinstance(parsed, dict) else None
    attrs = parsed.get("attrs") if isinstance(parsed, dict) else []
    if not isinstance(attrs, list):
        attrs = []
    return {**t, "raw": raw, "overview": overview, "attrs": attrs, "error": None}


def _build_tasks_for(name: str):
    grp = grouping.load(name)
    sch = ss.load(name)
    if grp is None or sch is None:
        return [], []
    link_hits = pd_._link_hits_from_grouping(grp)
    passages = pd_._passages_from_corpus(grp)
    src = grp.get("source_table")
    tasks, tables_info = [], []
    for ti, t in enumerate(sch.get("tables") or []):
        attrs = t.get("attr") or []
        if not attrs:
            continue
        prep = pd_.prepare({"table_name": t["table_name"],
                            "linked_columns": t.get("linked_columns") or [],
                            "attr": attrs},
                           link_hits, passages, src)
        if prep is None:
            continue
        tasks.append({"name": name, "table_idx": ti, "prep": prep})
        tables_info.append({"table_idx": ti, "table_name": t["table_name"], "attrs": attrs})
    return tasks, tables_info


def main_hybridqa(args):
    out_dir = config.DATA_DIR / "prompt_design"
    out_dir.mkdir(parents=True, exist_ok=True)
    qids = corpus.load_qids(args.qids)
    if not args.force:
        qids = [q for q in qids if not (out_dir / f"{q}.json").exists()]
    if not qids:
        print("## nothing to do."); return

    workers = args.workers or config.default_workers()
    all_tasks = []; qid_tables = {}
    for q in qids:
        ts, ti = _build_tasks_for(q)
        if ts:
            qid_tables[q] = ti; all_tasks.extend(ts)
    if not all_tasks:
        print("## nothing to dispatch."); return
    print(f"## qids={len(qid_tables)}  tasks={len(all_tasks)}  workers={workers}", flush=True)

    client = make_client()
    idx_map = {q: {ti["table_idx"]: i for i, ti in enumerate(qid_tables[q])} for q in qid_tables}
    results_by_qid = {q: [None] * len(ts) for q, ts in qid_tables.items()}
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_llm_task, t, client) for t in all_tasks]
        for fut in as_completed(futs):
            r = fut.result()
            q = r["name"]; i = idx_map[q][r["table_idx"]]
            ti = qid_tables[q][i]
            results_by_qid[q][i] = {
                "table_name": ti["table_name"], "overview": r["overview"],
                "attrs": r["attrs"], "raw": r["raw"],
            }
    saved = 0
    for q, tables in results_by_qid.items():
        pd_.save({"qid": q, "benchmark": "hybridqa",
                  "tables": [t for t in tables if t is not None]},
                 name=q, out_dir=out_dir)
        saved += 1
    print(f"## saved {saved}  wall={time.time()-t1:.1f}s", flush=True)


def main_sparta(args):
    config.require_domain()
    out_dir = config.DATA_DIR / "prompt_design"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all.json"
    if out_path.exists() and not args.force:
        print("## output exists; use --force."); return
    tasks, tables_info = _build_tasks_for("all")
    if not tasks:
        print("## no tasks."); return
    workers = args.workers or config.default_workers()
    client = make_client()
    idx_map = {ti["table_idx"]: i for i, ti in enumerate(tables_info)}
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_llm_task, t, client) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            i = idx_map[r["table_idx"]]
            results[i] = {"table_name": tables_info[i]["table_name"], "overview": r["overview"],
                          "attrs": r["attrs"], "raw": r["raw"]}
    p = pd_.save({"qid": None, "benchmark": "sparta",
                  "tables": [t for t in results if t is not None]},
                 name="all", out_dir=out_dir)
    print(f"## saved {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path. Ignored for sparta.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    corpus.preload()
    if config.BENCHMARK == "hybridqa":
        main_hybridqa(args)
    elif config.BENCHMARK == "sparta":
        main_sparta(args)
    else:
        raise SystemExit(config.BENCHMARK)


if __name__ == "__main__":
    main()

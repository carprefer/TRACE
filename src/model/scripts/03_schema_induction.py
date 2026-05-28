"""SGR step (ii.a) runner -- side-table schema induction (Fig. 5)."""
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
from src.model.sgr import grouping, schema_induction as ss


def _llm_call(client, system, user, max_new_tokens=4096):
    return client.inference(
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        max_new_tokens=max_new_tokens, temperature=0.0,
        response_format={"type": "json_object"},
    )


def _llm_task(t, client):
    prep = t["prep"]
    if prep is None:
        return {**t, "raw": None, "attr": [], "error": None}
    try:
        raw = _llm_call(client, prep["system"], prep["user"])
    except Exception as e:
        return {**t, "raw": None, "attr": [], "error": str(e)}
    try:
        parsed = ss.parse_fenced_json(raw)
    except Exception:
        parsed = {}
    attr = parsed.get("attr") if isinstance(parsed, dict) else []
    if not isinstance(attr, list):
        attr = []
    return {**t, "raw": raw, "attr": attr, "error": None}


def _build_tasks_for(name: str):
    grp_art = grouping.load(name)
    if grp_art is None:
        return [], []
    link_hits = ss._link_hits_from_grouping(grp_art)
    passages = ss._passages_from_corpus(grp_art)
    source_table = grp_art.get("source_table")
    groups_list = grp_art.get("groups") or []
    tasks = []
    for gi, g in enumerate(groups_list):
        prep = ss.prepare(g, link_hits, passages, source_table)
        tasks.append({"name": name, "group_idx": gi, "group": g, "prep": prep})
    return tasks, groups_list


def main_hybridqa(args):
    out_dir = config.DATA_DIR / "schema_induction"
    out_dir.mkdir(parents=True, exist_ok=True)
    qids = corpus.load_qids(args.qids)
    grp_dir = config.DATA_DIR / "grouping"
    qids = [q for q in qids if (grp_dir / f"{q}.json").exists()]
    if not args.force:
        qids = [q for q in qids if not (out_dir / f"{q}.json").exists()]
    if not qids:
        print("## nothing to do."); return
    workers = args.workers or config.default_workers()
    print(f"## qids={len(qids)}  workers={workers}", flush=True)

    all_tasks = []
    qid_groups = {}
    for q in qids:
        tasks, groups_list = _build_tasks_for(q)
        qid_groups[q] = groups_list
        all_tasks.extend(tasks)
    if not all_tasks:
        print("## nothing to dispatch."); return

    client = make_client()
    results_by_qid = {q: [None] * len(gs) for q, gs in qid_groups.items()}
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_llm_task, t, client) for t in all_tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results_by_qid[r["name"]][r["group_idx"]] = {
                "table_name": r["group"]["table_name"],
                "linked_columns": r["group"]["linked_columns"],
                "attr": r["attr"],
                "stage2_raw": r["raw"],
            }
    saved = 0
    for q, tables in results_by_qid.items():
        kept = [t for t in tables if t is not None and t.get("attr")]
        ss.save({"qid": q, "benchmark": "hybridqa", "tables": kept}, name=q, out_dir=out_dir)
        saved += 1
    print(f"## done  saved={saved}  wall={time.time()-t1:.1f}s", flush=True)


def main_sparta(args):
    config.require_domain()
    out_dir = config.DATA_DIR / "schema_induction"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all.json"
    if out_path.exists() and not args.force:
        print(f"## output exists; use --force."); return
    tasks, groups_list = _build_tasks_for("all")
    if not tasks:
        print("## no tasks."); return
    workers = args.workers or config.default_workers()
    print(f"## groups={len(groups_list)}  workers={workers}", flush=True)
    client = make_client()
    out = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_llm_task, t, client): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            i = futs[fut]; r = fut.result()
            out[i] = {
                "table_name": r["group"]["table_name"],
                "linked_columns": r["group"]["linked_columns"],
                "attr": r["attr"],
                "stage2_raw": r["raw"],
            }
    kept = [t for t in out if t and t.get("attr")]
    p = ss.save({"qid": None, "benchmark": "sparta", "tables": kept}, name="all", out_dir=out_dir)
    print(f"## saved {p}  tables_w_attr={len(kept)}/{len(out)}", flush=True)


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

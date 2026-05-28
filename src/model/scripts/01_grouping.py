"""SGR step (i) runner -- cluster link-bearing columns into entity groups (Fig. 4).

HybridQA: per-qid (--qids JSON list).
SPARTA  : single corpus-wide call per domain (writes `all.json`).
"""
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
from src.model.sgr import grouping


def _llm_call(client, system, user, max_new_tokens=2048, temperature=0.0):
    return client.inference(
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        max_new_tokens=max_new_tokens, temperature=temperature,
        response_format={"type": "json_object"},
    )


def _llm_task(prep, client):
    try:
        raw = _llm_call(client, prep["system"], prep["user"])
    except Exception as e:
        return prep, None, None, [], str(e)
    try:
        parsed = grouping.parse_fenced_json(raw)
    except Exception as e:
        parsed = {"parse_error": str(e), "raw": raw}
    groups = grouping.groups_from_labels(parsed, prep)
    return prep, raw, parsed, groups, None


def main_hybridqa(args):
    out_dir = config.DATA_DIR / "grouping"
    out_dir.mkdir(parents=True, exist_ok=True)
    qids = corpus.load_qids(args.qids)
    if not args.force:
        qids = [q for q in qids if not (out_dir / f"{q}.json").exists()]
    if not qids:
        print("## nothing to do."); return

    workers = args.workers or config.default_workers()
    print(f"## tag={config.run_tag()}  qids={len(qids)}  workers={workers}", flush=True)

    preps = []
    for q in qids:
        p = grouping.prepare_hybridqa(q)
        if p is not None:
            preps.append(p)
    if not preps:
        print("## nothing to dispatch."); return
    print(f"## prep done: {len(preps)} prompts ready", flush=True)

    client = make_client()
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_llm_task, p, client) for p in preps]
        for fut in as_completed(futs):
            prep, raw, parsed, groups, err = fut.result()
            if err:
                print(f"  ERR {prep['qid']}: {err}", flush=True); continue
            art = {
                "qid": prep["qid"], "benchmark": "hybridqa",
                "source_table": prep["source_table"],
                "link_hits_by_col": {h: [{"value": v, "url": u} for v, u in hits]
                                     for h, hits in prep["link_hits_by_col"].items()},
                "col_samples": prep["col_samples"],
                "stage1_system": prep["system"],
                "stage1_user":   prep["user"],
                "stage1_raw":    raw,
                "stage1_parsed": parsed,
                "groups":        groups,
            }
            grouping.save(art, name=prep["qid"], out_dir=out_dir)
    print(f"## done  wall={time.time()-t1:.1f}s", flush=True)


def main_sparta(args):
    config.require_domain()
    out_dir = config.DATA_DIR / "grouping"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all.json"
    if out_path.exists() and not args.force:
        print(f"## output exists ({out_path}); use --force to overwrite.")
        return
    prep = grouping.prepare_sparta()
    if prep is None:
        print("## no link columns; nothing to do."); return
    client = make_client()
    t0 = time.time()
    _, raw, parsed, groups, err = _llm_task(prep, client)
    if err:
        print(f"## ERR {err}"); return
    art = {
        "qid": None, "benchmark": "sparta", "sources": prep["sources"],
        "link_hits_by_col": {h: [{"value": v, "url": u} for v, u in hits]
                             for h, hits in prep["link_hits_by_col"].items()},
        "col_samples": prep["col_samples"],
        "stage1_system": prep["system"], "stage1_user": prep["user"],
        "stage1_raw": raw, "stage1_parsed": parsed, "groups": groups,
    }
    p = grouping.save(art, name="all", out_dir=out_dir)
    print(f"## saved {p}  groups={len(groups)}  wall={time.time()-t0:.1f}s", flush=True)


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

"""BGE-M3 index builder (used by the QCRC residual-evidence fallback).

HybridQA: per-qid index of `corpus.get_record(qid)['text']`.
SPARTA  : per-domain index of every URL appearing in any source-table cell.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus, retriever as rt


def _collect_sparta_linked_urls() -> set:
    urls = set()
    for t in corpus.get_source_tables().values():
        for row in t.get("data") or []:
            for cell in row:
                for u in (cell[1] or []):
                    if u:
                        urls.add(u.replace("/id/", ""))
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path. Ignored for sparta.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None,
                    help="Accepted for run_pipeline.sh passthrough; encoding parallelism "
                         "is GPU-batched via TRACE_RETRIEVER_BATCH.")
    args = ap.parse_args()

    corpus.preload()
    out_dir = config.SEARCH_INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rt.get_encoder()  # warm-up

    if config.BENCHMARK == "hybridqa":
        qids = corpus.load_qids(args.qids)
        print(f"## qids={len(qids)}", flush=True)
        t0 = time.time(); n_built = 0
        for i, q in enumerate(qids, 1):
            p = out_dir / f"{q}.npz"
            if p.exists() and not args.force:
                continue
            passages = corpus.get_record(q).get("text") or {}
            idx = rt.build_index(q, passages)
            rt.save_index(q, idx, out_dir=out_dir)
            n_built += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(qids)}]  built={n_built}", flush=True)
        print(f"## done  wall={time.time()-t0:.1f}s  built={n_built}", flush=True)
    elif config.BENCHMARK == "sparta":
        config.require_domain()
        p = out_dir / "all.npz"
        if p.exists() and not args.force:
            print("## index exists; use --force."); return
        text_data = corpus.get_text_data()
        passages = {u: text_data[u] for u in sorted(text_data.keys())}
        print(f"## domain={config.DOMAIN}  passages={len(passages)}", flush=True)
        t0 = time.time()
        idx = rt.build_index("all", passages)
        rt.save_index("all", idx, out_dir=out_dir)
        print(f"## done  wall={time.time()-t0:.1f}s", flush=True)
    else:
        raise SystemExit(config.BENCHMARK)


if __name__ == "__main__":
    main()

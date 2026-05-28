"""QCRC step (ii)+(iii) runner -- embedding gate + targeted extraction (Fig. 10).

Per qid: copy D_m -> D_q skeleton, run the gate, run targeted extractions in
a single ThreadPool, UPDATE the DB, drop all-NULL columns.
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
from src.model.qcrc import completion as cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    corpus.preload()
    qids = corpus.load_qids(args.qids)
    if not args.force:
        qids = [q for q in qids if not (cm.META_DIR / f"{q}.json").exists()]
    if not qids:
        print("## nothing to do."); return
    workers = args.workers or 4   # each qid spawns its own intra-qid LLM pool
    print(f"## qids={len(qids)}  workers={workers}  out={cm.ENRICHED_DIR}", flush=True)

    client = make_client()
    t0 = time.time(); n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cm.complete_one, q, client=client): q for q in qids}
        for fut in as_completed(futs):
            try:
                m = fut.result()
                if m.get("error"):
                    n_err += 1; print(f"  ERR {m.get('qid')}: {m.get('error')}", flush=True)
                else:
                    n_ok += 1
            except Exception as e:
                n_err += 1; print(f"  ERR: {e}", flush=True)
    print(f"## done  ok={n_ok}  err={n_err}  wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

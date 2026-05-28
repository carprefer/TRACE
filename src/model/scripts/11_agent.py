"""QCRC step (iv)+(v) runner -- SQL execution agent + residual-evidence fallback.

Notable behavior (matches the implementation lock described in agent.py):
  * Residual-search runs ONLY when the prior SQL turn returned zero rows.
  * Tool-set toggle: {sql, answer} by default; switches to {sql, answer_sql}
    when the most recent SQL was truncated (n_rows > MAX_ROWS_RETURNED).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.model.llm_client import make_client
from src.model.qcrc import agent as ag


def _norm(s):
    if s is None:
        return ""
    return str(s).lower().strip().rstrip(".,!?;:").strip()


def _to_set(v):
    if v is None:
        return set()
    items = v if isinstance(v, list) else [v]
    return {_norm(x) for x in items}


def _em(pred, gold):
    if pred is None:
        return False
    return _to_set(pred) == _to_set(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--force", action="store_true",
                    help="No-op for the agent (it always writes a fresh timestamped "
                         "summary). Accepted for run_pipeline.sh passthrough.")
    args = ap.parse_args()

    corpus.preload()
    if config.BENCHMARK == "sparta":
        config.require_domain()
    workers = args.workers or config.default_workers()

    qids = corpus.load_qids(args.qids)
    print(f"## qids={len(qids)}  workers={workers}  db_dir={ag.DB_DIR}", flush=True)

    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tag = ag.DB_DIR.name
    if args.run_tag:
        tag = f"{tag}_{args.run_tag}"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"agent_{tag}_{stamp}.summary.jsonl"
    full_dir = config.RESULTS_DIR / f"agent_full_{tag}_{stamp}"
    full_dir.mkdir(parents=True, exist_ok=True)

    client = make_client()
    out_f = open(out_path, "w")
    em_count = 0; done = 0
    t0 = time.time()

    def _do(qid):
        try:
            r = ag.run(qid, client=client)
            return r, None
        except Exception:
            return {"qid": qid, "pred": None}, traceback.format_exc()[-300:]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_do, q): q for q in qids}
        for fut in as_completed(futs):
            r, err = fut.result()
            done += 1
            r["em"] = _em(r.get("pred"), r.get("gold"))
            r["err"] = err or ""
            if r["em"]:
                em_count += 1
            (full_dir / f"{r['qid']}.json").write_text(
                json.dumps(r, ensure_ascii=False, default=str, indent=2))
            out_f.write(json.dumps({
                "qid": r["qid"], "pred": r.get("pred"), "gold": r.get("gold"),
                "em": r["em"], "n_attempts": len(r.get("transcript") or []),
                "err": r["err"],
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            if done % 25 == 0 or done == len(qids):
                print(f"[{done}/{len(qids)}]  em_so_far={em_count}", flush=True)
    out_f.close()
    print(f"## done  wall={time.time()-t0:.1f}s  EM={em_count}/{len(qids)} "
          f"({100*em_count/len(qids):.1f}%)", flush=True)
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()

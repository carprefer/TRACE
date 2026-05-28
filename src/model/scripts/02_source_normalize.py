"""Source-table normalization runner (HybridQA only).

SPARTA source tables ship already typed via dtype_dict so this stage is a
no-op for sparta.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.model.llm_client import make_client
from src.model.normalize_util import _is_null
from src.model.sgr import source_normalize as sn

OUT_DIR = config.DATA_DIR / "source_table"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="HybridQA qid list: 'all' (default), or a JSON file with a list of qids.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if config.skip("gold_normalize"):
        print(f"## benchmark={config.BENCHMARK}: source_normalize is a no-op.")
        return

    corpus.preload()
    qids = corpus.load_qids(args.qids)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.force:
        qids = [q for q in qids if not (OUT_DIR / f"{q}.json").exists()]
    if not qids:
        print("## nothing to do."); return
    workers = args.workers or config.default_workers()
    print(f"## tag={config.run_tag()}  qids={len(qids)}  workers={workers}", flush=True)

    t0 = time.time()
    states = {}; headers_orig_by_qid = {}; table_id_by_qid = {}; all_rescue = []
    for q in qids:
        try:
            cs, rescue, hdr, _rows, t_orig = sn.algo_phase(q)
        except Exception as e:
            print(f"  ERR {q}: {e}", flush=True); continue
        states[q] = cs; headers_orig_by_qid[q] = hdr; table_id_by_qid[q] = t_orig
        all_rescue.extend(rescue)
    print(f"## algo done. rescue_tasks={len(all_rescue)}", flush=True)

    if all_rescue:
        client = make_client()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(sn.call_llm, t, client) for t in all_rescue]
            for fut in as_completed(futs):
                _ = fut.result()
        for t in all_rescue:
            v = t.get("value")
            if v is None:
                continue
            state = states.get(t["qid"])
            if not state:
                continue
            col = state[t["col_idx"]]
            target = col["target"]; i = t["cell_idx"]
            if target == "date":
                import re as _re
                if isinstance(v, str) and _re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                    col["parsed"][i] = ["date", v, ""]
            else:
                from src.model.normalize_util import _coerce_num
                num = _coerce_num(v)
                if num is None:
                    continue
                col["parsed"][i] = ["num", num, ""]

    kind_total = Counter(); saved = 0
    for q, cs in states.items():
        cols_out = []
        for state in cs:
            kind, sqltype, unit, nx, vals = sn.finalize_column(state)
            new_name = state["header"] + nx
            cols_out.append({
                "header_orig": state["header_orig"],
                "header": state["header"],
                "new_name": new_name,
                "kind": kind, "sqlite_type": sqltype,
                "unit_suffix": unit,
                "values": vals,
                "raw_cells": state["raw"],
            })
            kind_total[kind] += 1
        art = {
            "qid": q,
            "table_id_orig": table_id_by_qid[q],
            "table_id": sn.sanitize_name(table_id_by_qid[q]),
            "headers_orig": headers_orig_by_qid[q],
            "columns": cols_out,
        }
        (OUT_DIR / f"{q}.json").write_text(
            json.dumps(art, indent=2, ensure_ascii=False, default=str))
        saved += 1
    print(f"## saved {saved}  wall={time.time()-t0:.1f}s  kinds={dict(kind_total)}",
          flush=True)


if __name__ == "__main__":
    main()

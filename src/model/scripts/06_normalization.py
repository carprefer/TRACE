"""SGR step (iv) runner -- side-table column re-typing (Fig. 8 outlier rescue)."""
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
from src.model.sgr import normalization as sn


def _renormalize_file(in_fp: Path, out_fp: Path, llm_fn=None):
    art = json.loads(in_fp.read_text())
    tables = art.get("tables") if isinstance(art.get("tables"), list) else [art]
    for t in tables:
        sn.renormalize_table(t, llm_fn=llm_fn)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(art, indent=2, ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path. Ignored for sparta.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--llm", action="store_true", help="enable LLM outlier rescue (Fig. 8)")
    args = ap.parse_args()

    corpus.preload()
    in_dir = config.DATA_DIR / "extraction"
    out_dir = config.DATA_DIR / "normalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or config.default_workers()

    if config.BENCHMARK == "hybridqa":
        qids = corpus.load_qids(args.qids)
        cands = [(q, in_dir / f"{q}.json", out_dir / f"{q}.json")
                 for q in qids if (in_dir / f"{q}.json").exists()]
    else:
        config.require_domain()
        cands = [(fp.stem, fp, out_dir / fp.name) for fp in sorted(in_dir.glob("*.json"))]
    if not args.force:
        cands = [(n, i, o) for n, i, o in cands if not o.exists()]
    if not cands:
        print("## nothing to do."); return

    llm_fn = sn.make_llm_fn(make_client()) if args.llm else None
    print(f"## files={len(cands)}  workers={workers}  llm_rescue={'on' if args.llm else 'off'}",
          flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_renormalize_file, ifp, ofp, llm_fn): n for n, ifp, ofp in cands}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"  ERR {futs[fut]}: {e}", flush=True)
    print(f"## done  wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

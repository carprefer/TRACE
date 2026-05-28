"""SPARTA post-process: rename imdb -> movie in folder + dtype_dict outer key,
then convert HuggingFace workload parquets into per-domain `workload.json` files.

Inputs (assumed already downloaded by download_data.sh):
    <work>/Corpus_SPARTA/{imdb,medical,nba}/{source_tables/,text_data.json,dtype_dict.json}
    <work>/hf_workload/workload_{movie,medical,nba}/validation/*.parquet

Output:
    <repo>/benchmark/sparta/{movie,medical,nba}/{source_tables/, text_data.json, dtype_dict.json, workload.json}
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DOMAIN_MAP = {"imdb": "movie", "medical": "medical", "nba": "nba"}


def normalize_dtype_dict(d: dict, new_outer: str) -> dict:
    """Rename outer key (imdb -> movie) without touching inner content."""
    if not isinstance(d, dict) or not d:
        return d
    outer = next(iter(d))
    return {new_outer: d[outer]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True,
                    help="Extracted Corpus_SPARTA root (contains imdb/, medical/, nba/)")
    ap.add_argument("--workload-dir", required=True,
                    help="HF SPARTA download root (contains workload_movie/, workload_medical/, workload_nba/)")
    ap.add_argument("--out-dir", required=True,
                    help="benchmark/sparta/ destination")
    args = ap.parse_args()

    corpus_root = Path(args.corpus_dir)
    work_root = Path(args.workload_dir)
    out_root = Path(args.out_dir)

    for src_domain, dst_domain in DOMAIN_MAP.items():
        src = corpus_root / src_domain
        dst = out_root / dst_domain
        if not src.exists():
            print(f"  SKIP corpus/{src_domain}: not found", file=sys.stderr)
            continue
        dst.mkdir(parents=True, exist_ok=True)

        # source_tables/ -- straight copy
        src_tables = src / "source_tables"
        dst_tables = dst / "source_tables"
        if dst_tables.exists():
            shutil.rmtree(dst_tables)
        shutil.copytree(src_tables, dst_tables)

        # text_data.json -- straight copy
        shutil.copyfile(src / "text_data.json", dst / "text_data.json")

        # dtype_dict.json -- rename outer key if needed
        d = json.loads((src / "dtype_dict.json").read_text())
        (dst / "dtype_dict.json").write_text(
            json.dumps(normalize_dtype_dict(d, dst_domain), indent=2, ensure_ascii=False))

        # workload from HF parquet -> workload.json
        # HF SPARTA layout: workload_<domain>/validation-00000-of-XXXXX.parquet (flat)
        wl_dir = work_root / f"workload_{dst_domain}"
        pq_files = sorted(wl_dir.glob("validation-*.parquet"))
        if not pq_files:
            print(f"  WARN workload {wl_dir}/validation-*.parquet missing", file=sys.stderr)
            continue
        import pyarrow.parquet as pq_
        rows = []
        for pq_file in pq_files:
            t = pq_.read_table(pq_file)
            rows.extend(t.to_pylist())
        # answer column on HF is a list of strings (or array); ensure JSON-compatible
        for r in rows:
            for k, v in list(r.items()):
                if hasattr(v, "tolist"):
                    r[k] = v.tolist()
        (dst / "workload.json").write_text(json.dumps(rows, ensure_ascii=False))

        print(f"  ok  sparta/{dst_domain}: tables={len(list(dst_tables.glob('*.json')))}  "
              f"workload={len(rows)} qs", flush=True)


if __name__ == "__main__":
    main()

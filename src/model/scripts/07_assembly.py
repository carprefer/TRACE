"""SGR step (iv) runner -- relational database assembly (produces D_m).

HybridQA: per-qid (one main from source_table + side tables from extraction/normalization).
SPARTA  : per-domain (all source tables + side tables).
"""
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
from src.model.sgr import assembly as asm, grouping


def build_qid(qid: str, out_dir: Path):
    """HybridQA: build D_m for a single qid."""
    grp = grouping.load(qid)
    norm_path = config.DATA_DIR / "normalization" / f"{qid}.json"
    ext_path = norm_path if norm_path.exists() else config.DATA_DIR / "extraction" / f"{qid}.json"
    src_path = config.DATA_DIR / "source_table" / f"{qid}.json"
    if grp is None or not ext_path.exists() or not src_path.exists():
        return None
    src_art = json.loads(src_path.read_text())
    ext_art = json.loads(ext_path.read_text())

    db_path = out_dir / f"{qid}.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    main_sql, main_col_sql_by_raw, n_main_rows = asm.create_main_from_source_table(conn, src_art)
    joinmap_in = {g["table_name"]: list(g.get("linked_columns") or [])
                  for g in (grp.get("groups") or [])}

    def resolve(qkey: str):
        if "." not in qkey:
            return None
        _src, col = qkey.rsplit(".", 1)
        col_sql = main_col_sql_by_raw.get(col) or asm._san(col)
        return f"{main_sql}.{col_sql}"

    seen_tables = {main_sql}
    urlmap, joinmap, n_sides, n_side_rows = asm.create_side_tables(
        conn, ext_art.get("tables") or [], joinmap_in, seen_tables, resolve)
    conn.commit(); conn.close()
    (out_dir / f"{qid}.urlmap.json").write_text(json.dumps(urlmap, ensure_ascii=False))
    (out_dir / f"{qid}.joinmap.json").write_text(json.dumps(joinmap, ensure_ascii=False))
    return (qid, n_main_rows, n_sides, n_side_rows)


def build_domain(out_dir: Path, domain: str):
    """SPARTA: build D_m for one domain."""
    grp = grouping.load("all")
    if grp is None:
        return None
    dtype_path = config.BENCH_PATHS["sparta"]["corpus_dir"]
    dtype_path = Path(str(dtype_path).format(domain=domain)) / "dtype_dict.json"
    dtype_outer = json.loads(dtype_path.read_text())
    dtype_dict = (dtype_outer.get(domain) or {}).get("dtype_dict", {})

    db_path = out_dir / f"{domain}.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    src = corpus.get_source_tables()
    table_sql_by_name, col_sql_by_qualified = asm.create_main_from_sparta_corpus(
        conn, src, dtype_dict)

    norm_dir = config.DATA_DIR / "normalization"
    ext_dir  = config.DATA_DIR / "extraction"
    pick_dir = norm_dir if any(norm_dir.glob("*.json")) else ext_dir
    side_arts = []
    for fp in sorted(pick_dir.glob("*.json")):
        side_arts.append(json.loads(fp.read_text()))

    joinmap_in = {g["table_name"]: list(g.get("linked_columns") or [])
                  for g in (grp.get("groups") or [])}

    def resolve(qkey: str):
        col_sql = col_sql_by_qualified.get(qkey)
        if col_sql is None:
            return None
        src_name, _col = qkey.rsplit(".", 1)
        tsql = table_sql_by_name.get(src_name)
        return f"{tsql}.{col_sql}" if tsql else None

    seen_tables = set(table_sql_by_name.values())
    urlmap, joinmap, n_sides, n_side_rows = asm.create_side_tables(
        conn, side_arts, joinmap_in, seen_tables, resolve)
    conn.commit(); conn.close()
    (out_dir / f"{domain}.urlmap.json").write_text(json.dumps(urlmap, ensure_ascii=False))
    (out_dir / f"{domain}.joinmap.json").write_text(json.dumps(joinmap, ensure_ascii=False))
    return (domain, len(table_sql_by_name), n_sides, n_side_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path. Ignored for sparta.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    corpus.preload()
    out_dir = asm.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.BENCHMARK == "hybridqa":
        qids = corpus.load_qids(args.qids)
        if not args.force:
            qids = [q for q in qids if not (out_dir / f"{q}.db").exists()]
        if not qids:
            print("## nothing to do."); return
        workers = args.workers or 16
        print(f"## qids={len(qids)}  workers={workers}", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(build_qid, q, out_dir): q for q in qids}
            for fut in as_completed(futs):
                _ = fut.result()
        print(f"## done  wall={time.time()-t0:.1f}s", flush=True)
    elif config.BENCHMARK == "sparta":
        config.require_domain()
        if (out_dir / f"{config.DOMAIN}.db").exists() and not args.force:
            print("## output exists; use --force."); return
        t0 = time.time()
        r = build_domain(out_dir, config.DOMAIN)
        if r is None:
            print("## failed."); return
        _, n_main, n_sides, n_side_rows = r
        print(f"## done  main_tables={n_main}  side_tables={n_sides}  "
              f"side_rows={n_side_rows}  wall={time.time()-t0:.1f}s", flush=True)
    else:
        raise SystemExit(config.BENCHMARK)


if __name__ == "__main__":
    main()

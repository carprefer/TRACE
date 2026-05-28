"""HybridQA post-process:
1. Consolidate WikiTables-WithLinks/tables_tok/*.json    -> all_tables.json
2. Consolidate WikiTables-WithLinks/request_tok/*.json   -> all_passages.json  (intermediate)
3. Join dev.traced.json + all_tables + all_passages      -> dev_qa.json
   (one record per qid, with `text` = passages linked from the qid's table)

Outputs (under <repo>/benchmark/hybridqa/):
    all_tables.json     -- {table_id: {url,title,header,data,...}, ...}
    dev_qa.json         -- [{question_id,question,table_id,table,text,answer}, ...]

The all_passages.json is intermediate and not shipped (too big, ~190 MB; dev_qa.json
already contains the needed passages per-qid).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def consolidate(in_dir: Path) -> dict:
    """Read every *.json under in_dir and key by stem (== table_id or url-key)."""
    out = {}
    n_err = 0
    for fp in in_dir.iterdir():
        if fp.suffix != ".json":
            continue
        try:
            out[fp.stem] = json.loads(fp.read_text())
        except Exception:
            n_err += 1
    if n_err:
        print(f"  WARN {n_err} files failed to parse under {in_dir}", file=sys.stderr)
    return out


def build_dev_qa(dev_traced: list, all_tables: dict, request_tok_dir: Path) -> list:
    """For each dev record, attach the passages linked from its table cells."""
    out = []
    for rec in dev_traced:
        qid = rec["question_id"]
        table_id = rec["table_id"]
        table = all_tables.get(table_id)
        if table is None:
            continue
        # passages linked from this table's cells
        req_fp = request_tok_dir / f"{table_id}.json"
        text = {}
        if req_fp.exists():
            try:
                text = json.loads(req_fp.read_text())
            except Exception:
                text = {}
        answer_text = rec.get("answer-text")
        answer = [answer_text] if isinstance(answer_text, str) else (answer_text or [])
        out.append({
            "question_id": qid,
            "question":    rec.get("question", ""),
            "table_id":    table_id,
            "table":       [table_id],
            "text":        text,
            "answer":      answer,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitables-dir", required=True,
                    help="WikiTables-WithLinks root (contains tables_tok/ and request_tok/)")
    ap.add_argument("--dev-traced", required=True,
                    help="dev.traced.json path (from github.com/wenhuchen/HybridQA)")
    ap.add_argument("--out-dir", required=True,
                    help="benchmark/hybridqa destination")
    args = ap.parse_args()

    wt = Path(args.wikitables_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"## consolidating tables from {wt/'tables_tok'} ...", flush=True)
    all_tables = consolidate(wt / "tables_tok")
    (out / "all_tables.json").write_text(json.dumps(all_tables, ensure_ascii=False))
    print(f"   wrote all_tables.json  n={len(all_tables)}", flush=True)

    print(f"## reading dev.traced.json ...", flush=True)
    dev_traced = json.loads(Path(args.dev_traced).read_text())

    print(f"## building dev_qa.json (joining tables + passages) ...", flush=True)
    dev_qa = build_dev_qa(dev_traced, all_tables, wt / "request_tok")
    (out / "dev_qa.json").write_text(json.dumps(dev_qa, ensure_ascii=False))
    print(f"   wrote dev_qa.json     n={len(dev_qa)}", flush=True)


if __name__ == "__main__":
    main()

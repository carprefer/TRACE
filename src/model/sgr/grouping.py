"""SGR step (i): schema-guided passage grouping (paper Fig. 4).

For each link-bearing source column, decide which entity its linked passages
collectively describe -- the cluster name becomes a side-table name.

Inputs:
    HybridQA: per-qid source table + per-qid linked passages
    SPARTA  : per-domain source tables + shared passages

Output artifact (per name):
    {qid|None, benchmark, source_table|sources, link_hits_by_col,
     col_samples, system, user, raw, parsed, groups}
where groups = [{table_name, linked_columns: [<src>.<col>, ...]}].
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.prompt import sgr as prompt

OUT_DIR = config.DATA_DIR / "grouping"
TOP_K = 10


def parse_fenced_json(raw: str):
    """response_format=json_object guarantees clean JSON -- no fence handling needed."""
    return json.loads(raw)


def _strip_id(u: str) -> str:
    return u.replace("/id/", "") if u else u


def _top_k_for_column(hits, passages, k=TOP_K):
    freq = Counter(); first_value = {}
    for v, u in hits:
        if u not in passages:
            continue
        freq[u] += 1
        first_value.setdefault(u, v)
    return [{"value": first_value[u], "url": u, "passage": passages[u], "freq": c}
            for u, c in freq.most_common(k)]


# -----------------------------------------------------------------------------
# HybridQA helpers
# -----------------------------------------------------------------------------

def _collect_link_hits_hybridqa(headers, rows):
    by_col = {h: [] for h in headers}
    for row in rows:
        for ci, cell in enumerate(row):
            urls = [u for u in (cell[1] or []) if u]
            for url in urls:
                by_col[headers[ci]].append((cell[0], url))
    return {h: v for h, v in by_col.items() if v}


def _resolve_col_key_hybridqa(key, source_table, link_hits_by_col):
    """Tolerant resolver. Handles <src>.<col>, <col>, and '!' glitches."""
    candidates = [key]
    if key.startswith(source_table + "."):
        candidates.append(key[len(source_table) + 1:])
    if "." in key:
        candidates.append(key.split(".", 1)[-1])
        candidates.append(key.rsplit(".", 1)[-1])
    expanded = []
    for c in candidates:
        expanded.append(c)
        for sep in ("!", "."):
            if sep in c:
                expanded.append(c.split(sep, 1)[0])
    lower_map = {h.lower(): h for h in link_hits_by_col}
    for c in expanded:
        if c in link_hits_by_col:
            return c
        c_clean = c.replace("!", "")
        if c_clean and c_clean in link_hits_by_col:
            return c_clean
        if c.lower() in lower_map:
            return lower_map[c.lower()]
        if c_clean.lower() in lower_map:
            return lower_map[c_clean.lower()]
    return None


def prepare_hybridqa(qid: str):
    """Build the grouping LLM prompt for one HybridQA qid.

    Requires `data/source_table/<qid>.json` (the source-normalize artifact)
    to be present. Returns None if the question has no link-bearing columns.
    """
    src_dir = config.DATA_DIR / "source_table"
    fp = src_dir / f"{qid}.json"
    if not fp.exists():
        return None
    g = json.loads(fp.read_text())

    headers = [c["header"] for c in g["columns"]]
    raw_t = corpus.get_raw_table(qid)
    rows = raw_t["data"]
    by_col = _collect_link_hits_hybridqa(headers, rows) if rows else {}
    if not by_col:
        raw_headers = [h[0] for h in raw_t["header"]]
        by_col = _collect_link_hits_hybridqa(raw_headers, rows)
    if not by_col:
        return None

    passages = corpus.get_record(qid).get("text") or {}
    col_samples = {h: _top_k_for_column(hits, passages, TOP_K) for h, hits in by_col.items()}
    col_samples = {h: s for h, s in col_samples.items() if s}
    if not col_samples:
        return None

    source_table = g.get("table_id") or raw_t.get("uid") or "main"
    cols_info = g.get("columns") or []
    schema_parts = []
    for c in cols_info:
        ty = c.get("sqlite_type") or c.get("kind") or ""
        nm = c.get("header")
        schema_parts.append(f"{nm}: {ty}" if ty else nm)
    source_tables_block = f"- {source_table}({', '.join(schema_parts)})"
    linked_block_parts = []
    for col, samples in col_samples.items():
        linked_block_parts.append(f'### "{source_table}.{col}"')
        for s in samples:
            linked_block_parts.append(f"- {s['passage']}")
        linked_block_parts.append("")
    user = prompt.grouping_user_prompt.format(
        source_tables_block=source_tables_block,
        linked_columns_block="\n".join(linked_block_parts).rstrip(),
    )
    return {
        "qid": qid,
        "benchmark": "hybridqa",
        "source_table": source_table,
        "link_hits_by_col": by_col,
        "col_samples": col_samples,
        "user": user,
        "system": prompt.grouping_system_prompt,
    }


# -----------------------------------------------------------------------------
# SPARTA helpers
# -----------------------------------------------------------------------------

def _collect_link_hits_sparta(source_tables: dict) -> dict:
    out: dict = {}
    for src_name, t in source_tables.items():
        headers = [h[0] for h in t["header"]]
        for row in t["data"]:
            for ci, cell in enumerate(row):
                urls = [_strip_id(u) for u in (cell[1] or []) if u]
                if not urls:
                    continue
                key = f"{src_name}.{headers[ci]}"
                bucket = out.setdefault(key, [])
                for u in urls:
                    bucket.append((cell[0], u))
    return {k: v for k, v in out.items() if v}


def _resolve_col_key_sparta(key: str, link_hits_by_col: dict):
    candidates = [key]
    parts = key.split(".")
    for length in range(len(parts), 0, -1):
        candidates.append(".".join(parts[:length]))
    expanded = list(candidates)
    for c in candidates:
        if "!" in c:
            expanded.append(c.split("!", 1)[0])
    lower_map = {k.lower(): k for k in link_hits_by_col}
    for c in expanded:
        if c in link_hits_by_col:
            return c
        c_clean = c.replace("!", "")
        if c_clean and c_clean in link_hits_by_col:
            return c_clean
        if c.lower() in lower_map:
            return lower_map[c.lower()]
        if c_clean.lower() in lower_map:
            return lower_map[c_clean.lower()]
    return None


def prepare_sparta():
    src = corpus.get_source_tables()
    passages = corpus.get_text_data()
    by_col = _collect_link_hits_sparta(src)
    if not by_col:
        return None
    col_samples = {h: _top_k_for_column(hits, passages, TOP_K) for h, hits in by_col.items()}
    col_samples = {h: s for h, s in col_samples.items() if s}
    if not col_samples:
        return None

    source_tables_block = []
    for src_name in sorted(src.keys()):
        cols = [h[0] for h in src[src_name]["header"]]
        source_tables_block.append(f"- {src_name}({', '.join(cols)})")
    linked_block_parts = []
    for col_key, samples in col_samples.items():
        linked_block_parts.append(f'### "{col_key}"')
        for s in samples:
            linked_block_parts.append(f"- {s['passage']}")
        linked_block_parts.append("")
    user = prompt.grouping_user_prompt.format(
        source_tables_block="\n".join(source_tables_block),
        linked_columns_block="\n".join(linked_block_parts).rstrip(),
    )
    return {
        "qid": None,
        "benchmark": "sparta",
        "sources": sorted(src.keys()),
        "link_hits_by_col": by_col,
        "col_samples": col_samples,
        "user": user,
        "system": prompt.grouping_system_prompt,
    }


def groups_from_labels(parsed, prep1):
    if not isinstance(parsed, dict):
        return []
    labels = parsed.get("labels")
    if not isinstance(labels, list):
        return []
    is_hybridqa = prep1.get("benchmark") == "hybridqa"
    source_table = prep1.get("source_table")
    link_hits = prep1["link_hits_by_col"]
    by_tn: dict = {}
    order: list = []
    source_set = ({source_table} if source_table else set(prep1.get("sources") or []))
    for lab in labels:
        if not isinstance(lab, dict):
            continue
        col_key = lab.get("column")
        tn = lab.get("new_table_name") or lab.get("table_name")
        if not isinstance(col_key, str) or not isinstance(tn, str) or not tn:
            continue
        tn = tn.replace("!", "_").strip("_") or tn
        if tn in source_set:
            tn = f"{tn}_side"
        if is_hybridqa:
            col = _resolve_col_key_hybridqa(col_key, source_table, link_hits)
            canonical = f"{source_table}.{col}" if col else None
        else:
            col = _resolve_col_key_sparta(col_key, link_hits)
            canonical = col
        if not canonical:
            continue
        if tn not in by_tn:
            by_tn[tn] = []
            order.append(tn)
        if canonical not in by_tn[tn]:
            by_tn[tn].append(canonical)
    return [{"table_name": tn, "linked_columns": by_tn[tn]} for tn in order]


def save(artifact: dict, name: str, out_dir: Path = None) -> Path:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, default=str))
    return p


def load(name: str, out_dir: Path = None):
    out_dir = out_dir or OUT_DIR
    fp = out_dir / f"{name}.json"
    return json.loads(fp.read_text()) if fp.exists() else None

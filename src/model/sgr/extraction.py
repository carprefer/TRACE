"""SGR step (iii): attribute-value extraction (paper Fig. 7).

For each row of a side table (one passage + cell_value), extract the typed
attribute values defined in step (ii.a) using the per-table format hints
designed in step (ii.b).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.prompt import sgr as prompt

OUT_DIR = config.DATA_DIR / "extraction"


def _link_hits_from_grouping(g: dict) -> dict:
    out = {}
    raw = g.get("link_hits_by_col") or g.get("link_cols") or {}
    for col, lst in raw.items():
        if not lst:
            continue
        if isinstance(lst[0], (list, tuple)):
            out[col] = [tuple(p) for p in lst]
        else:
            out[col] = [(d.get("value"), d.get("url")) for d in lst]
    return out


def _passages_from_corpus(grouping_art: dict) -> dict:
    bench = grouping_art.get("benchmark")
    if bench == "hybridqa":
        return corpus.get_record(grouping_art["qid"]).get("text") or {}
    if bench == "sparta":
        return corpus.get_text_data()
    return {}


def _resolve_col_key(key: str, source_table, link_hits_by_col: dict):
    if key in link_hits_by_col:
        return key
    parts = key.split(".")
    for length in range(len(parts), 0, -1):
        cand = ".".join(parts[:length])
        if cand in link_hits_by_col:
            return cand
    if source_table:
        rest = key
        if rest.startswith(source_table + "."):
            rest = rest[len(source_table) + 1:]
        if rest in link_hits_by_col:
            return rest
        head = rest.split(".", 1)[0]
        if head in link_hits_by_col:
            return head
    lower_map = {k.lower(): k for k in link_hits_by_col}
    return lower_map.get(key.lower())


def _url_cv_pairs(link_hits_by_col: dict, linked_columns: list, source_table) -> list:
    cols = []
    for c in linked_columns or []:
        m = _resolve_col_key(c, source_table, link_hits_by_col)
        if m and m not in cols:
            cols.append(m)
    seen, out = set(), []
    for col in cols:
        for v, u in link_hits_by_col.get(col) or []:
            if not u:
                continue
            key = (u, str(v))
            if key in seen:
                continue
            seen.add(key)
            out.append((u, v))
    return out


def _attrs_block(attrs: list, meta_attrs: list) -> str:
    fmt = {a.get("name"): a.get("format", "") for a in (meta_attrs or []) if isinstance(a, dict)}
    lines = []
    for a in attrs:
        if not isinstance(a, dict):
            continue
        n, t = a.get("name"), a.get("type")
        if not n:
            continue
        f = fmt.get(n, "")
        lines.append(f"- {n} ({t}): {f}" if f else f"- {n} ({t})")
    return "\n".join(lines)


def build_user(table_name: str, attrs: list, prompt_meta: dict, cell_value, passage: str) -> str:
    overview = (prompt_meta or {}).get("overview", "")
    m_attrs = (prompt_meta or {}).get("attrs", [])
    return prompt.extraction_user_prompt.format(
        table_name=table_name,
        overview=overview,
        attrs_block=_attrs_block(attrs, m_attrs),
        cell_value=cell_value,
        passage=passage,
    )


SYSTEM_EXTRACT = prompt.extraction_system_prompt


def coerce_values(raw_obj):
    if not isinstance(raw_obj, dict):
        return {}, f"top-level not object: {type(raw_obj).__name__}"
    null_strs = {"null", "NULL", "Null", "None", "none", "NONE", "N/A", "n/a"}
    return {k: (None if isinstance(v, str) and v.strip() in null_strs else v)
            for k, v in raw_obj.items()}, None


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

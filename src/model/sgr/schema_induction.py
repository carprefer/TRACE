"""SGR step (ii.a): side-table schema induction (paper Fig. 5).

For each group from step (i), propose a list of typed attributes.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.prompt import sgr as prompt

OUT_DIR = config.DATA_DIR / "schema_induction"
TOP_K = 10


def parse_fenced_json(raw: str):
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.S)
    if not m:
        m = re.search(r"(\{.*\})", raw, re.S)
    if not m:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    txt = m.group(1)
    txt = re.sub(r'([,\[\{])\s*!\s*', r'\1', txt)
    txt = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)!type"\s*:', r'"\1", "type":', txt)
    return json.loads(txt)


def _top_k_for_group(combined_freq, passages, k=TOP_K):
    out = []
    for url, cnt in combined_freq.most_common():
        if url not in passages:
            continue
        out.append({"url": url, "passage": passages[url], "freq": cnt})
        if len(out) >= k:
            break
    return out


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


def prepare(group: dict, link_hits_by_col: dict, passages: dict, source_table):
    combined = Counter()
    for key in group["linked_columns"]:
        canonical = _resolve_col_key(key, source_table, link_hits_by_col)
        if canonical is None:
            continue
        for _v, u in link_hits_by_col[canonical]:
            combined[u] += 1
    top = _top_k_for_group(combined, passages, TOP_K)
    if not top:
        return None
    passages_block = "\n".join(f"- {s['passage']}" for s in top)
    user = prompt.schema_induction_user_prompt.format(
        table_name=group["table_name"],
        passages_block=passages_block,
    )
    return {
        "table_name": group["table_name"],
        "linked_columns": group["linked_columns"],
        "group_samples": top,
        "user": user,
        "system": prompt.schema_induction_system_prompt,
    }


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

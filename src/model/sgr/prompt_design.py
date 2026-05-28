"""SGR step (ii.b): per-side-table extraction-prompt design (paper Fig. 6).

For each table designed in step (ii.a), ask the LLM to author an `overview` plus
per-attr `format` hints. These feed step (iii) extraction calls.
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

OUT_DIR = config.DATA_DIR / "prompt_design"
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


def _top_k_for_group(combined_freq, passages, k=TOP_K):
    out = []
    for url, cnt in combined_freq.most_common():
        if url not in passages:
            continue
        out.append(passages[url])
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


def prepare(table: dict, link_hits_by_col: dict, passages_corpus: dict, source_table):
    attrs = table.get("attr") or []
    if not attrs:
        return None
    combined = Counter()
    for key in table.get("linked_columns") or []:
        canonical = _resolve_col_key(key, source_table, link_hits_by_col)
        if canonical is None:
            continue
        for _v, u in link_hits_by_col[canonical]:
            combined[u] += 1
    top_passages = _top_k_for_group(combined, passages_corpus, TOP_K)
    if not top_passages:
        return None
    attrs = [a for a in attrs if isinstance(a, dict) and a.get("name") and a.get("type")]
    if not attrs:
        return None
    schema_block = "\n".join(f'- {a["name"]}: {a["type"]}' for a in attrs)
    passages_block = "\n".join(f"- {p}" for p in top_passages)
    user = prompt.prompt_design_user_prompt.format(
        table_name=table["table_name"],
        schema_block=schema_block,
        passages_block=passages_block,
    )
    return {
        "table_name": table["table_name"],
        "attrs": attrs,
        "user": user,
        "system": prompt.prompt_design_system_prompt,
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

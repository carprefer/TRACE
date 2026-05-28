"""SGR step (iv) -- side-table column re-typing using the column-level normalizer
(`normalize_util.normalize_column`), with optional LLM outlier rescue (Fig. 8).

Reads `extraction/<name>.json` (raw extracted JSON) and rewrites every attribute
column to its best-supported sortable type, repairing outliers via the LLM
when needed. Any cell that remains unparseable collapses the column to TEXT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import normalize_util as nu
from src.prompt import sgr as prompt

OUT_DIR = config.DATA_DIR / "normalization"

# normalize_util kind -> side-table attr type
TYPE_MAP = {"integer": "int", "real": "float", "date": "date", "text": "text"}


def _stringify(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else str(v)
    return str(v)


def make_llm_fn(client):
    """Returns llm_fn(value, kind, examples) -> value|None. kind in {'integer','real','date'}."""
    def fn(value, kind, examples):
        sys_p = nu.SYS_PROMPTS[kind]
        ex_lines = [f"  {raw!r} -> {p!r}" for raw, p in examples]
        ex_block = "Examples:\n" + "\n".join(ex_lines) + "\n\n" if ex_lines else ""
        user_p = prompt.outlier_rescue_user_prompt.format(examples_block=ex_block, cell=value)
        try:
            raw = client.inference(
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user",   "content": user_p}],
                max_new_tokens=64, temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(raw).get("value")
        except Exception:
            return None
    return fn


def renormalize_table(table: dict, llm_fn=None) -> dict:
    attrs = table.get("attr") or []
    rows = table.get("rows") or []
    new_attrs = []
    for a in attrs:
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        if not name:
            continue
        raw_col = [_stringify((r.get("values") or {}).get(name)) for r in rows]
        result = nu.normalize_column(raw_col, llm_fn=llm_fn)
        new_attrs.append({"name": name, "type": TYPE_MAP.get(result["kind"], "text")})
        for ri, v in enumerate(result["values"]):
            rows[ri].setdefault("values", {})[name] = v
    table["attr"] = new_attrs
    return table


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

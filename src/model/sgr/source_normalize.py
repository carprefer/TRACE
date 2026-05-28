"""SGR step (iv) -- HybridQA source-table normalization.

HybridQA ships untyped string cells; before SGR can build SQLite tables, the
source columns must be typed (int / real / date / text). SPARTA source tables
are already typed via dtype_dict, so this step is skipped there.

Per-column rule: algo pass first (number with optional unit, then date), then
LLM rescue on any algo-missed cell (paper Fig. 8). Any leftover unparseable
non-null cell collapses the column to TEXT.
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
from src.model.normalize_util import (
    THRESHOLD, _algo_parse, _coerce_num, _is_null, SYS_PROMPTS,
)
from src.prompt import sgr as prompt

OUT_DIR = config.DATA_DIR / "source_table"


def sanitize_name(name: str) -> str:
    n = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_").lower()
    if not n:
        return "col"
    if not n[0].isalpha():
        n = "c_" + n
    return n


def algo_phase(qid: str):
    rec = corpus.get_record(qid)
    t_orig = rec["table_id"]
    t_raw = corpus.get_tables()[t_orig]
    headers_orig = [h[0] for h in t_raw["header"]]
    raw_rows = t_raw["data"]

    col_states = []
    rescue = []
    for ci, oh in enumerate(headers_orig):
        raw_col = [str(r[ci][0]) if r[ci][0] is not None else "" for r in raw_rows]
        parsed = _algo_parse(raw_col)
        n_non = sum(1 for v in raw_col if not _is_null(v))
        target = None
        examples = []
        if n_non:
            n_num = sum(1 for k, _, _ in parsed if k == "num")
            n_date = sum(1 for k, _, _ in parsed if k == "date")
            if n_num >= n_date and n_num / n_non >= THRESHOLD:
                target = "num"
            elif n_date / n_non >= THRESHOLD:
                target = "date"
        if target:
            for i, (k, p, _) in enumerate(parsed):
                if k == target and len(examples) < 5:
                    examples.append((raw_col[i], p))
            llm_kind = "date" if target == "date" else "integer"
            for i, (k, _, _) in enumerate(parsed):
                if _is_null(raw_col[i]) or k == target:
                    continue
                rescue.append({
                    "qid": qid, "col_idx": ci, "cell_idx": i,
                    "target": target, "llm_kind": llm_kind,
                    "raw": raw_col[i], "examples": examples,
                })
        col_states.append({
            "header_orig": oh,
            "header": sanitize_name(oh),
            "raw": raw_col,
            "parsed": [list(t) for t in parsed],
            "target": target,
            "examples": examples,
        })
    return col_states, rescue, headers_orig, raw_rows, t_orig


def call_llm(task, client):
    sys_p = SYS_PROMPTS[task["llm_kind"]]
    ex_block = ""
    if task["examples"]:
        lines = [f"  {raw!r} -> {p!r}" for raw, p in task["examples"]]
        ex_block = "Examples from this column (raw -> normalized):\n" + "\n".join(lines) + "\n\n"
    user_p = prompt.outlier_rescue_user_prompt.format(examples_block=ex_block, cell=task["raw"])
    try:
        raw = client.inference(
            messages=[{"role": "system", "content": sys_p},
                      {"role": "user",   "content": user_p}],
            max_new_tokens=128, temperature=0.0,
            response_format={"type": "json_object"},
        )
        task["value"] = json.loads(raw).get("value")
    except Exception:
        task["value"] = None
    return task


def _text(raw):
    values = [None if _is_null(v) else (v.strip() if isinstance(v, str) else str(v)) for v in raw]
    return ("text", "TEXT", None, "", values)


def finalize_column(state):
    raw = state["raw"]
    parsed = state["parsed"]
    target = state["target"]
    if target is None:
        return _text(raw)
    for v, (k, p, _) in zip(raw, parsed):
        if _is_null(v):
            continue
        if k != target or p is None:
            return _text(raw)
    if target == "date":
        values = [None if _is_null(v) else parsed[i][1] for i, v in enumerate(raw)]
        return ("date", "TEXT", None, "", values)
    has_real = any(isinstance(parsed[i][1], float) and not parsed[i][1].is_integer()
                   for i, v in enumerate(raw) if not _is_null(v))
    if has_real:
        kind, sqltype = "real", "REAL"
        values = [None if _is_null(v) else float(parsed[i][1]) for i, v in enumerate(raw)]
    else:
        kind, sqltype = "integer", "INTEGER"
        values = [None if _is_null(v) else int(parsed[i][1]) for i, v in enumerate(raw)]
    n_non = sum(1 for v in raw if not _is_null(v))
    units = [parsed[i][2] for i, v in enumerate(raw) if not _is_null(v) and parsed[i][2]]
    uc = Counter(units)
    unit_suffix = None
    name_extra = ""
    if uc:
        du, dn = uc.most_common(1)[0]
        if dn / n_non >= THRESHOLD:
            unit_suffix = du
            name_extra = f"_{du}"
    return (kind, sqltype, unit_suffix, name_extra, values)

"""Column normalization utilities -- numbers (with optional units) and dates.

Used by SGR step iv (normalization, §4.1.iv) and by the source-table normalizer
when a benchmark ships untyped string cells (HybridQA).
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.prompt import sgr as _sgr_prompt

THRESHOLD = 0.6

NULL_TOKENS = {"", "-", "–", "—", "n/a", "na", "null", "none"}


def _is_null(v) -> bool:
    if v is None:
        return True
    s = v if isinstance(v, str) else str(v)
    return s.strip().lower() in NULL_TOKENS


# ----- numbers ---------------------------------------------------------------

_NUM_UNIT_RE = re.compile(
    r"^\s*(-?\d{1,3}(?:,\d{3})+|-?\d+)(?:\.(\d+))?\s*([A-Za-z][A-Za-z/]*)?\s*$"
)


def parse_number(s):
    if _is_null(s):
        return None
    m = _NUM_UNIT_RE.match(s if isinstance(s, str) else str(s))
    if not m:
        return None
    int_part = m.group(1).replace(",", "")
    frac = m.group(2)
    unit = (m.group(3) or "").lower()
    try:
        val = float(f"{int_part}.{frac}") if frac else int(int_part)
    except ValueError:
        return None
    return val, unit


# ----- dates -----------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = "|".join(_MONTHS.keys())
_SEP = r"[\s.,]+"
_ISO = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$")
_MDY = re.compile(rf"^\s*({_MONTH_RE}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}})\s*$", re.I)
_DMY = re.compile(rf"^\s*(\d{{1,2}}){_SEP}({_MONTH_RE}){_SEP}(\d{{4}})\s*$", re.I)
_MY  = re.compile(rf"^\s*({_MONTH_RE}){_SEP}(\d{{4}})\s*$", re.I)


def parse_date(s):
    if _is_null(s):
        return None
    s = s if isinstance(s, str) else str(s)
    m = _ISO.match(s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = _MDY.match(s)
    if m:
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"
    m = _DMY.match(s)
    if m:
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(1)):02d}"
    m = _MY.match(s)
    if m:
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            return f"{int(m.group(2)):04d}-{mon:02d}-01"
    return None


def _algo_parse(values):
    out = []
    for v in values:
        if _is_null(v):
            out.append((None, None, ""))
            continue
        d = parse_date(v)
        if d is not None:
            out.append(("date", d, ""))
            continue
        n = parse_number(v)
        if n is not None:
            out.append(("num", n[0], n[1]))
            continue
        out.append((None, None, ""))
    return out


def _text_result(values):
    return {
        "kind": "text",
        "sqlite_type": "TEXT",
        "values": [None if _is_null(v) else (v.strip() if isinstance(v, str) else str(v))
                   for v in values],
        "unit_suffix": None,
        "name_extra": "",
    }


def _coerce_num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        try:
            return int(v) if v.lstrip("-").isdigit() else float(v)
        except ValueError:
            return None
    return None


# LLM rescue prompts -- one per target kind. Tied to /src/prompt/sgr.py Fig. 8.
SYS_PROMPTS = {
    "integer": _sgr_prompt.outlier_rescue_integer_system_prompt,
    "real":    _sgr_prompt.outlier_rescue_real_system_prompt,
    "date":    _sgr_prompt.outlier_rescue_date_system_prompt,
}


def normalize_column(values, llm_fn=None, examples_n: int = 5) -> dict:
    """Return {kind, sqlite_type, values, unit_suffix, name_extra}.

    Per-column rule: target family (num | date) requires THRESHOLD coverage,
    then each non-null cell that the algo missed gets one LLM rescue attempt
    (if llm_fn provided). Any remaining unparseable cell collapses to TEXT.
    """
    parsed = _algo_parse(values)
    n_non = sum(1 for v in values if not _is_null(v))
    if not n_non:
        return _text_result(values)

    n_num  = sum(1 for k, _, _ in parsed if k == "num")
    n_date = sum(1 for k, _, _ in parsed if k == "date")

    if n_num >= n_date and n_num / n_non >= THRESHOLD:
        target = "num"
    elif n_date / n_non >= THRESHOLD:
        target = "date"
    else:
        return _text_result(values)

    examples = [(values[i], p) for i, (k, p, _) in enumerate(parsed) if k == target][:examples_n]
    out = [list(t) for t in parsed]
    llm_kind = "date" if target == "date" else "integer"

    for i, (k, _, _) in enumerate(parsed):
        if _is_null(values[i]) or k == target:
            continue
        if llm_fn is None:
            return _text_result(values)
        v = llm_fn(values[i], llm_kind, examples)
        if v is None:
            return _text_result(values)
        if target == "date":
            if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                out[i] = ["date", v, ""]
            else:
                return _text_result(values)
        else:
            num = _coerce_num(v)
            if num is None:
                return _text_result(values)
            out[i] = ["num", num, ""]

    vals_out = []
    units = []
    for v, (k, p, u) in zip(values, out):
        if _is_null(v):
            vals_out.append(None)
            continue
        vals_out.append(p)
        if target == "num":
            units.append(u)

    if target == "date":
        return {"kind": "date", "sqlite_type": "TEXT", "values": vals_out,
                "unit_suffix": None, "name_extra": ""}

    has_real = any(isinstance(v, float) and not v.is_integer()
                   for v in vals_out if v is not None)
    if has_real:
        kind, sqltype = "real", "REAL"
        vals_out = [None if v is None else float(v) for v in vals_out]
    else:
        kind, sqltype = "integer", "INTEGER"
        vals_out = [None if v is None else int(v) for v in vals_out]

    uc = Counter(u for u in units if u)
    unit_suffix = None
    name_extra = ""
    if uc:
        du, dn = uc.most_common(1)[0]
        if dn / n_non >= THRESHOLD:
            unit_suffix = du
            name_extra = f"_{du}"

    return {"kind": kind, "sqlite_type": sqltype, "values": vals_out,
            "unit_suffix": unit_suffix, "name_extra": name_extra}

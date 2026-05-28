"""SGR step (iv) -- relational database assembly (produces D_m).

Combines:
    - main (source) table(s) -- from source_table/ (HybridQA) or corpus (SPARTA)
    - side table(s)          -- from normalization/<name>.json (or extraction/ if no normalize step ran)
    - per-side joinmap       -- carries `linked_cell joinable with: <src>.<col>` hints
    - per-side urlmap        -- row-index -> source URL (used by the QCRC agent's auto-attach)

Outputs (under config.DB_DIR):
    HybridQA: <qid>.db + <qid>.urlmap.json + <qid>.joinmap.json
    SPARTA  : <domain>.db + .urlmap.json + .joinmap.json
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config

OUT_DIR = config.DB_DIR

SQL_RES = {"select", "from", "where", "order", "group", "by", "having", "limit", "offset",
           "table", "index", "view", "join", "on", "as", "in", "is", "and", "or", "not", "null",
           "between", "like", "case", "when", "then", "else", "end", "distinct", "union", "all",
           "insert", "update", "delete", "create", "drop", "alter", "into", "values", "set", "with"}

DTYPE_OF = {"int": "INTEGER", "float": "REAL", "date": "TEXT", "text": "TEXT", "time": "TEXT"}
KIND_TO_SQL = {"integer": "INTEGER", "real": "REAL", "date": "TEXT", "text": "TEXT"}
SPARTA_DTYPE_OF = {"int": "INTEGER", "float": "REAL", "date": "TEXT", "time": "TEXT", "str": "TEXT"}


def _san(name: str, prefix: str = "c") -> str:
    n = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "")).strip("_").lower()
    if not n:
        n = prefix
    if not n[0].isalpha():
        n = f"{prefix}_{n}"
    if n in SQL_RES:
        n = f"{prefix}_{n}"
    return n


def _to_text(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _coerce(v, t: str):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a"}:
        return None
    if t == "int":
        SQLITE_INT_MAX = (1 << 63) - 1
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v if abs(v) <= SQLITE_INT_MAX else None
        if isinstance(v, float):
            if v != int(v):
                return None
            iv = int(v)
            return iv if abs(iv) <= SQLITE_INT_MAX else None
        try:
            s = str(v).replace(",", "").strip()
            if not s:
                return None
            iv = int(float(s))
            return iv if abs(iv) <= SQLITE_INT_MAX else None
        except Exception:
            return None
    if t == "float":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return None
    if t == "date":
        s = str(v).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        return None
    return _to_text(v)


def _alloc_name(base: str, seen: set, fallback: str = "c") -> str:
    n = _san(base, prefix=fallback); b = n; i = 2
    while n in seen:
        n = f"{b}_{i}"; i += 1
    seen.add(n); return n


# ---------------- main table builders ----------------

def create_main_from_source_table(conn, source_table_art: dict):
    """HybridQA: load the typed source_table artifact, build the main table."""
    table_sql = _san(source_table_art["table_id"], prefix="t")
    cols = source_table_art["columns"]
    seen = set()
    main_col_sql_by_raw: dict = {}
    sql_cols = []
    for c in cols:
        kind = c.get("kind", "text")
        sqltype = c.get("sqlite_type") or KIND_TO_SQL.get(kind, "TEXT")
        nn = _alloc_name(c.get("new_name") or c.get("header") or "c", seen, "c")
        sql_cols.append((nn, sqltype, c.get("values") or []))
        for raw in (c.get("header"), c.get("header_orig"), c.get("new_name")):
            if raw is not None:
                main_col_sql_by_raw[str(raw)] = nn
    col_defs = ", ".join(f'"{n}" {ty}' for n, ty, _ in sql_cols)
    conn.execute(f'CREATE TABLE "{table_sql}" ({col_defs})')
    n_rows = len(sql_cols[0][2]) if sql_cols else 0
    ph = ", ".join("?" * len(sql_cols))
    for ri in range(n_rows):
        conn.execute(f'INSERT INTO "{table_sql}" VALUES ({ph})',
                     [_to_text(c[2][ri]) for c in sql_cols])
    return table_sql, main_col_sql_by_raw, n_rows


def create_main_from_sparta_corpus(conn, source_tables: dict, dtype_dict: dict):
    table_sql_by_name = {}
    col_sql_by_qualified: dict = {}
    seen_tables = set()
    for src_name, t in source_tables.items():
        tsql = _alloc_name(src_name, seen_tables, "t")
        table_sql_by_name[src_name] = tsql
        headers = [h[0] for h in t["header"]]
        col_seen = set()
        sql_cols = []
        for ci, h in enumerate(headers):
            ty = dtype_dict.get(f"{src_name}.{h}", "str")
            sqltype = SPARTA_DTYPE_OF.get(ty, "TEXT")
            nn = _alloc_name(h, col_seen, "c")
            sql_cols.append((nn, sqltype, ty))
            col_sql_by_qualified[f"{src_name}.{h}"] = nn
        col_defs = ", ".join(f'"{n}" {ty}' for n, ty, _ in sql_cols)
        conn.execute(f'CREATE TABLE "{tsql}" ({col_defs})')
        ph = ", ".join("?" * len(sql_cols))
        for row in t["data"]:
            vals = []
            for ci, cell in enumerate(row):
                raw_v = cell[0]
                ty = sql_cols[ci][2]
                vals.append(_coerce(raw_v, ty) if ty in ("int", "float", "date") else _to_text(raw_v))
            conn.execute(f'INSERT INTO "{tsql}" VALUES ({ph})', vals)
    return table_sql_by_name, col_sql_by_qualified


# ---------------- side table builder ----------------

def create_side_tables(conn, side_art_tables: list, joinmap_in: dict,
                       seen_table_names: set, main_col_resolver):
    """Each side table has columns: linked_cell + attrs.

    `joinmap_in` is from the grouping artifact ({table_name: [<src.col>, ...]}).
    `main_col_resolver(qualified_key)` returns "<sql_table>.<sql_col>" or None.
    """
    urlmap: dict = {}
    joinmap: dict = {}
    n_sides = 0; n_rows_total = 0
    for st in side_art_tables:
        attrs = st.get("attr") or []
        rows = st.get("rows") or []
        if not attrs or not rows:
            continue
        tn_raw = st.get("table_name") or "t"
        tn_sql = _alloc_name(tn_raw, seen_table_names, "t")

        attr_seen = {"linked_cell"}
        sql_attrs = []
        for a in attrs:
            if not isinstance(a, dict):
                continue
            an = a.get("name"); ty = (a.get("type") or "text").lower()
            if not an:
                continue
            nn = _alloc_name(an, attr_seen, "a")
            sql_attrs.append((nn, an, ty))
        if not sql_attrs:
            continue

        col_defs = ['"linked_cell" TEXT']
        for nn, _, ty in sql_attrs:
            col_defs.append(f'"{nn}" {DTYPE_OF.get(ty, "TEXT")}')
        conn.execute(f'CREATE TABLE "{tn_sql}" ({", ".join(col_defs)})')

        ph = ", ".join("?" * (1 + len(sql_attrs)))
        url_list = []
        for r in rows:
            vals = r.get("values") or {}
            row_vals = [_to_text(r.get("cell_value"))]
            for nn, an, ty in sql_attrs:
                v = _coerce(vals.get(an), ty)
                row_vals.append(v if (v is None or isinstance(v, (int, float))) else _to_text(v))
            conn.execute(f'INSERT INTO "{tn_sql}" VALUES ({ph})', row_vals)
            url_list.append(r.get("url"))
        urlmap[tn_sql] = url_list

        join_refs = []
        for src_col in joinmap_in.get(tn_raw, []) or []:
            ref = main_col_resolver(src_col)
            if ref and ref not in join_refs:
                join_refs.append(ref)
        joinmap[tn_sql] = join_refs
        n_sides += 1; n_rows_total += len(rows)
    return urlmap, joinmap, n_sides, n_rows_total

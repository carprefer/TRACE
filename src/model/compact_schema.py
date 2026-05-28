"""Compact CREATE TABLE schema text builder -- the format the QCRC planner and
agent reason over. Each table block has the SQLite DDL plus a column-profile
footer with the top-K most-frequent values.
"""
import sqlite3


def _table_ddl(conn: sqlite3.Connection, table: str) -> str:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    col_strs = [f'  "{r[1]}" {r[2]}' for r in rows]
    return f'CREATE TABLE "{table}" (\n' + ",\n".join(col_strs) + "\n);"


def _profile_lines(table_profile: dict, top_k_show: int = 10) -> list[str]:
    out = []
    for col_name, info in table_profile.items():
        stats = info["raw_stats"]
        line = f"  {col_name}: {stats['null_count']} nulls, {stats['distinct_count']} distinct"
        top_k = list(stats.get("top_k_values", {}).keys())[:top_k_show]
        if top_k:
            line += f", values: {top_k}"
        out.append(line)
    return out


def build_compact_schema(
    conn: sqlite3.Connection,
    profiles: dict,
    tables: list,
    top_k_show: int = 10,
) -> str:
    db_tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    parts = []
    for t in tables:
        if t not in db_tables:
            continue
        ddl = _table_ddl(conn, t)
        tp = profiles.get(t)
        if tp:
            parts.append(ddl + "\n-- Column profiles:\n" + "\n".join(_profile_lines(tp, top_k_show)))
        else:
            parts.append(ddl)
    return "\n\n".join(parts)

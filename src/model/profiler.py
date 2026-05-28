"""Pandas column profiler -- shape/distinct/top-K summaries used in CREATE TABLE blocks.

Output: per-column {description, raw_stats, total_rows} consumed by
compact_schema.build_compact_schema (which only reads raw_stats).
"""
import os
import string

import pandas as pd


def _describe(col_name: str, total_rows: int, stats: dict) -> str:
    parts = [f"Column {col_name} has {stats['null_count']} NULL values out of {total_rows} records.",
             f"There are {stats['distinct_count']} distinct values."]
    shape = stats["shape"]
    if shape.get("min") is not None:
        parts.append(f"The minimum value is '{shape['min']}' and the maximum value is '{shape['max']}'.")
    if stats["top_k_values"]:
        top_vals = ", ".join(f"'{v}'" for v in stats["top_k_values"].keys())
        parts.append(f"Most common non-NULL column values are {top_vals}.")
    if "min_len" in shape:
        if shape["min_len"] == shape["max_len"]:
            parts.append(f"The values are always {shape['min_len']} characters long.")
        else:
            parts.append(f"The values range from {shape['min_len']} to {shape['max_len']} characters long.")
    if shape.get("common_prefix") and len(shape["common_prefix"]) > 1:
        parts.append(f"All values start with the prefix '{shape['common_prefix']}'.")
    pattern = shape.get("pattern")
    if pattern == "digits_only":
        parts.append("Every column value looks like a number.")
    elif pattern == "alpha_only":
        parts.append("Every column value looks like text (letters only).")
    abc = shape.get("alphabet_stats", {})
    feats = []
    if abc.get("upper_ratio", 0) > 0: feats.append("uppercase letters")
    if abc.get("lower_ratio", 0) > 0: feats.append("lowercase letters")
    if abc.get("punct_ratio", 0) > 0: feats.append("punctuation")
    if pattern != "digits_only" and feats:
        parts.append(f"It contains {', '.join(feats)}.")
    return " ".join(parts)


def profile_table(df: pd.DataFrame, top_k: int = 10) -> dict:
    total_rows = len(df)
    out = {}
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        non_null_count = total_rows - null_count
        n_unique = int(series.nunique())
        str_series = series.dropna().astype(str)
        shape = {}
        if not str_series.empty:
            lengths = str_series.str.len()
            shape["min_len"] = int(lengths.min())
            shape["max_len"] = int(lengths.max())
            shape["min"] = str_series.min()
            shape["max"] = str_series.max()
            prefix = os.path.commonprefix(str_series.tolist())
            shape["common_prefix"] = prefix if prefix else None
            all_text = "".join(str_series.values)
            n = len(all_text)
            if n > 0:
                u = sum(1 for c in all_text if c.isupper())
                l = sum(1 for c in all_text if c.islower())
                d = sum(1 for c in all_text if c.isdigit())
                p = sum(1 for c in all_text if c in string.punctuation)
                s = sum(1 for c in all_text if c.isspace())
                shape["alphabet_stats"] = {
                    "upper_ratio": round(u/n, 4),
                    "lower_ratio": round(l/n, 4),
                    "digit_ratio": round(d/n, 4),
                    "punct_ratio": round(p/n, 4),
                    "space_ratio": round(s/n, 4),
                }
            else:
                shape["alphabet_stats"] = {}
            if str_series.str.isdigit().all():
                shape["pattern"] = "digits_only"
            elif str_series.str.isalpha().all():
                shape["pattern"] = "alpha_only"
            elif str_series.str.isalnum().all():
                shape["pattern"] = "alphanumeric"
            else:
                shape["pattern"] = "mixed"
        else:
            shape["min"] = shape["max"] = shape["common_prefix"] = None
            shape["alphabet_stats"] = {}
        if non_null_count > 0:
            counts = series.value_counts().head(top_k)
            top_k_values = {str(k): int(v) for k, v in counts.items()}
        else:
            top_k_values = {}
        col_stats = {
            "null_count": null_count,
            "distinct_count": n_unique,
            "shape": shape,
            "top_k_values": top_k_values,
        }
        out[col] = {
            "description": _describe(col, total_rows, col_stats),
            "raw_stats": col_stats,
            "total_rows": total_rows,
        }
    return out

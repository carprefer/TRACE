"""Benchmark-aware corpus loader.

HybridQA: per-qid -- one gold table + one passage dict, both keyed by qid.
SPARTA  : per-domain -- many source tables + shared passages text_data.

Call `preload()` once from the main thread before fan-out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config


# HybridQA caches
_DEV: dict | None = None
_TABLES: dict | None = None

# SPARTA caches
_WL: list | None = None
_WL_MAP: dict | None = None
_SRC: dict | None = None
_TEXT: dict | None = None


def preload() -> None:
    global _DEV, _TABLES, _WL, _WL_MAP, _SRC, _TEXT
    if config.BENCHMARK == "hybridqa":
        paths = config.BENCH_PATHS["hybridqa"]
        if _DEV is None:
            _DEV = {d["question_id"]: d for d in json.loads(paths["dev"].read_text())}
        if _TABLES is None:
            _TABLES = json.loads(paths["all_tables"].read_text())
        return
    if config.BENCHMARK == "sparta":
        config.require_domain()
        sp = config.BENCH_PATHS["sparta"]
        domain = config.DOMAIN
        wl_path = Path(str(sp["workload"]).format(domain=domain))
        cor_dir = Path(str(sp["corpus_dir"]).format(domain=domain))
        if _WL is None:
            _WL = json.loads(wl_path.read_text())
            _WL_MAP = {q["question_id"]: q for q in _WL}
        src_dir = cor_dir / "source_tables"
        if _SRC is None:
            _SRC = {p.stem: json.loads(p.read_text()) for p in sorted(src_dir.glob("*.json"))}
        if _TEXT is None:
            _TEXT = json.loads((cor_dir / "text_data.json").read_text())
        return
    raise NotImplementedError(f"corpus not wired for benchmark {config.BENCHMARK!r}")


# ---- HybridQA accessors ----
def get_dev() -> dict:
    preload(); return _DEV


def get_tables() -> dict:
    preload(); return _TABLES


def get_record(qid: str) -> dict:
    return get_dev()[qid]


def get_raw_table(qid: str) -> dict:
    return get_tables()[get_dev()[qid]["table_id"]]


# ---- SPARTA accessors ----
def get_workload() -> list:
    preload(); return _WL


def get_question(qid: str) -> dict:
    preload(); return _WL_MAP[qid]


def get_source_tables() -> dict:
    preload(); return _SRC


def get_text_data() -> dict:
    preload(); return _TEXT


# ---- qid helpers (used by pipeline runners) ----

def all_qids() -> list:
    """Every qid for the current (benchmark, domain)."""
    preload()
    if config.BENCHMARK == "hybridqa":
        return list(_DEV.keys())
    if config.BENCHMARK == "sparta":
        return [q["question_id"] for q in _WL]
    raise NotImplementedError(config.BENCHMARK)


def load_qids(arg: str) -> list:
    """Parse the --qids CLI value into a qid list.

    Accepts:
      "all" or empty       -> every qid for the current benchmark/domain
      <path to JSON list>  -> read it (also accepts {"all": [...]})
    """
    if not arg or arg.strip().lower() == "all":
        return all_qids()
    qj = json.loads(Path(arg).read_text())
    return qj["all"] if isinstance(qj, dict) else qj

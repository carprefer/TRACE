"""Central config for TRACE. Single source of truth for all parameters.

Override at call time via environment variables, listed alongside each setting
below. Pipeline scripts import from this module exclusively -- editing any
value here propagates to every stage.

Layout (default):

    /root/trace/                                  -- repo root
        benchmark/<benchmark>/[<domain>/]         -- raw benchmark inputs (source tables, text, workload)
        out/<benchmark>/<llm>/[<domain>/]
            data/                                 -- SGR + QCRC intermediate artifacts
            dbs/                                  -- baseline SQLite databases (D_m)
            dbs_completed/                        -- per-question completed databases (D_q)
            results/                              -- QCRC agent outputs

"""
from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Run-time selection -- override via env
# -----------------------------------------------------------------------------

LLM       = os.environ.get("TRACE_LLM",       "gpt-4o-mini")
BENCHMARK = os.environ.get("TRACE_BENCHMARK", "hybridqa")          # hybridqa | sparta
DOMAIN    = os.environ.get("TRACE_DOMAIN",    "")                  # sparta only: medical | movie | nba

# -----------------------------------------------------------------------------
# Path layout
# -----------------------------------------------------------------------------

REPO_ROOT       = Path(os.environ.get("TRACE_REPO_ROOT",       "/root/trace"))
BENCHMARK_ROOT  = Path(os.environ.get("TRACE_BENCHMARK_ROOT",  str(REPO_ROOT / "benchmark")))
OUT_ROOT        = Path(os.environ.get("TRACE_OUT_ROOT",        str(REPO_ROOT / "out")))

# Per-run output namespace: <benchmark>/<llm>[/<domain>]
_SCOPE          = (DOMAIN,) if BENCHMARK == "sparta" else ()
RUN_ROOT        = OUT_ROOT.joinpath(BENCHMARK, LLM, *_SCOPE)
DATA_DIR        = RUN_ROOT / "data"                                # SGR/QCRC intermediates
DB_DIR          = RUN_ROOT / "dbs"                                 # D_m (relationalized DB, before per-q completion)
DB_COMPLETED    = RUN_ROOT / f"dbs_completed_th{int(round(float(os.environ.get('TRACE_SIM_THRESHOLD', '0.65')) * 100)):02d}"
RESULTS_DIR     = RUN_ROOT / "results"
SEARCH_INDEX_DIR = DATA_DIR / "search_index"

# Raw benchmark inputs (shipped with the repo under benchmark/)
BENCH_PATHS = {
    "hybridqa": {
        "dev":        BENCHMARK_ROOT / "hybridqa" / "dev_qa.json",
        "all_tables": BENCHMARK_ROOT / "hybridqa" / "all_tables.json",
    },
    "sparta": {
        # domain-templated; resolved at load time using DOMAIN
        "workload":   BENCHMARK_ROOT / "sparta" / "{domain}" / "workload.json",
        "corpus_dir": BENCHMARK_ROOT / "sparta" / "{domain}",
    },
}

# -----------------------------------------------------------------------------
# Pipeline hyperparameters (paper §4.1, §4.2)
# -----------------------------------------------------------------------------

# QCRC embedding-similarity gate (Fig. 1 of paper, §4.2.ii)
SIM_THRESHOLD       = float(os.environ.get("TRACE_SIM_THRESHOLD", "0.65"))

# QCRC agent loop (paper §4.2.iv-v)
MAX_STEPS           = int(os.environ.get("TRACE_MAX_STEPS",          "5"))
MAX_ROWS_RETURNED   = int(os.environ.get("TRACE_MAX_ROWS_RETURNED", "10"))
MAX_COLLECTED_SHOWN = int(os.environ.get("TRACE_MAX_COLLECTED_SHOWN","10"))
ANSWER_SQL_ROW_CAP  = int(os.environ.get("TRACE_ANSWER_SQL_ROW_CAP","10000"))

# Residual evidence fallback (paper §4.2.v): top-K passages when SQL returns 0 rows
SEARCH_TOP_K        = int(os.environ.get("TRACE_SEARCH_TOP_K",       "3"))

# Schema profiles (column samples shown per CREATE TABLE)
TOP_K_IN_PROFILE    = int(os.environ.get("TRACE_TOP_K_IN_PROFILE",   "8"))

# Concurrency
DEFAULT_WORKERS = {
    "gpt-4o-mini":     int(os.environ.get("TRACE_WORKERS_GPT",  "32")),
    "qwen3.5-35b-a3b": int(os.environ.get("TRACE_WORKERS_QWEN", "32")),
}

EXTRACT_WORKERS = int(os.environ.get("TRACE_EXTRACT_WORKERS", "8"))

# -----------------------------------------------------------------------------
# LLM presets (the only place model-side parameters live)
# -----------------------------------------------------------------------------

LLM_PRESETS = {
    "gpt-4o-mini": {
        "name":           "gpt-4o-mini",
        "max_tokens":     120_000,
        "max_new_tokens":   4_096,
        "temperature":          0,
    },
    "qwen3.5-35b-a3b": {
        "name":                "qwen3.5-35b-a3b",
        "max_tokens":          32_768,
        "max_new_tokens":       4_096,
        "temperature":              0,
        "thinking_budget":          0,
        "base_url":           os.environ.get("TRACE_QWEN_BASE_URL", "http://localhost:8000/v1"),
        "api_key":            os.environ.get("TRACE_QWEN_API_KEY",  "EMPTY"),
        "served_model_name":  os.environ.get("TRACE_QWEN_SERVED",   "Qwen/Qwen3.5-35B-A3B"),
    },
}

# -----------------------------------------------------------------------------
# Retriever (BGE-M3) -- used only by the residual-fallback search in QCRC agent
# -----------------------------------------------------------------------------

BGE_MODEL_PATH    = Path(os.environ.get(
    "TRACE_BGE_MODEL_PATH",
    "./models/bge-m3",
))
RETRIEVER_GPU     = os.environ.get("TRACE_RETRIEVER_GPU", "0")
RETRIEVER_BATCH   = int(os.environ.get("TRACE_RETRIEVER_BATCH", "32"))
RETRIEVER_DOC_LEN = int(os.environ.get("TRACE_RETRIEVER_DOC_LEN", "512"))
RETRIEVER_QRY_LEN = int(os.environ.get("TRACE_RETRIEVER_QRY_LEN", "512"))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

SPARTA_DOMAINS = ("medical", "movie", "nba")


def require_domain() -> None:
    if BENCHMARK == "sparta" and not DOMAIN:
        raise RuntimeError(
            "TRACE_DOMAIN required for benchmark=sparta (one of: "
            + ", ".join(SPARTA_DOMAINS)
            + ")"
        )


def default_workers() -> int:
    return DEFAULT_WORKERS.get(LLM, 32)


def run_tag() -> str:
    parts = [LLM, BENCHMARK]
    if BENCHMARK == "sparta" and DOMAIN:
        parts.append(DOMAIN)
    return "_".join(parts)


# Stages that do not apply to a given benchmark (skipped as no-ops)
SKIP_STAGES_PER_BENCHMARK = {
    "sparta": {"gold_normalize"},  # sparta source tables ship already typed
}


def skip(stage: str) -> bool:
    return stage in SKIP_STAGES_PER_BENCHMARK.get(BENCHMARK, set())

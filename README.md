# TRACE — Table–Text Relationalization and Completion of Evidence for Coverage-Sensitive QA

Reference implementation accompanying the ACL submission *TRACE: Table–Text
Relationalization And Completion of Evidence*.

TRACE reframes coverage-sensitive Table–Text QA as query processing over a
**repairable relational state**:

* **Schema-Guided Relationalization (SGR, §4.1)** runs **once per domain
  corpus**: text-derived facts are extracted from linked passages, typed, and
  joined back to the source schema, producing the offline relationalized
  database `D_m`.
* **Question-Conditioned Relational Completion (QCRC, §4.2)** runs **per
  question**: the planner reads the schema of `D_m`, decides which
  side-table columns the question needs, fills missing cells via a tight
  similarity-gated extraction, and an SQL agent reasons over the
  question-completed database `D_q`.

The agent variant exposed here matches the locked configuration described in
the paper appendix:

* **Tool selection.** Each turn the model may emit `sql` plus exactly one
  terminator. The terminator is `answer` by default, but if the immediately
  preceding SQL turn returned more than `MAX_ROWS_RETURNED` rows (so the
  observation was truncated in the prompt) the terminator switches to
  `answer_sql` — a SELECT whose result rows *are* the answer (capped at
  `ANSWER_SQL_ROW_CAP`).
* **Residual-evidence fallback (§4.2.v).** Dense retrieval over the passage
  corpus is invoked **only when the prior SQL turn returned zero rows**.
  The top-`SEARCH_TOP_K` passages are appended to the agent's `collected`
  buffer.

---

## Repository layout

```
trace/
├── README.md                       — this file
├── requirements.txt                — pip dependencies (excluding torch)
├── scripts/
│   ├── download_data.sh            — fetch SPARTA + HybridQA from official sources
│   ├── run_pipeline.sh             — run all 11 stages (Phase A + B) in one command
│   ├── _build_sparta.py            — OSF zip + HF parquet  → benchmark/sparta/...
│   └── _build_hybridqa.py          — WikiTables-WithLinks + dev.traced.json → benchmark/hybridqa/...
├── benchmark/                      — populated by scripts/download_data.sh (NOT in repo)
│   ├── hybridqa/
│   │   ├── dev_qa.json             — 3,466 dev questions with linked passages
│   │   └── all_tables.json         — 15k tables
│   └── sparta/{medical,movie,nba}/
│       ├── workload.json           — 565 validation questions
│       ├── source_tables/          — typed source tables
│       ├── text_data.json          — passage text by URL
│       └── dtype_dict.json
├── out/                            — all run artifacts (intermediates + DBs + agent results)
└── src/
    ├── config.py                   — central config (single source of truth)
    ├── llm/                        — minimal LLM clients (gpt-4o-mini, qwen3.5-35b-a3b)
    ├── prompt/                     — prompt templates verbatim from the paper
    │   ├── sgr.py                  — Figures 4–8 (SGR)
    │   └── qcrc.py                 — Figures 9–11 (QCRC)
    └── model/
        ├── corpus.py               — benchmark loaders
        ├── retriever.py            — BGE-M3 dense retriever (fp32, GPU-pinned)
        ├── profiler.py             — pandas column profiler
        ├── compact_schema.py       — CREATE TABLE schema text builder
        ├── normalize_util.py       — number / date parsers
        ├── llm_client.py           — client factory bound to config.LLM_PRESETS
        ├── sgr/                    — Phase A modules (paper §4.1)
        │   ├── grouping.py             (i)
        │   ├── schema_induction.py     (ii.a)
        │   ├── prompt_design.py        (ii.b)
        │   ├── extraction.py           (iii)
        │   ├── source_normalize.py     (iv, HybridQA only)
        │   ├── normalization.py        (iv)
        │   └── assembly.py             (iv → D_m)
        ├── qcrc/                   — Phase B modules (paper §4.2)
        │   ├── planning.py             (i)
        │   ├── completion.py           (ii + iii → D_q)
        │   └── agent.py                (iv + v, SQL agent)
        └── scripts/                — pipeline runners
            ├── 01_grouping.py
            ├── 02_source_normalize.py
            ├── 03_schema_induction.py
            ├── 04_prompt_design.py
            ├── 05_extraction.py
            ├── 06_normalization.py
            ├── 07_assembly.py
            ├── 08_build_search_index.py
            ├── 09_planning.py
            ├── 10_completion.py
            └── 11_agent.py
```

---

## Setup from scratch

### 0. Get the code

The anonymized repository for double-blind review is hosted at:

    https://anonymous.4open.science/r/anonymous-paper-69B9/

Use the "Download Repository" button there to obtain a zip, then unpack it:

```bash
unzip anonymous-paper-69B9.zip -d trace
cd trace
```

### 1. Python environment

Either Conda (recommended — the BGE-M3 retriever needs a CUDA build of
PyTorch compatible with the host driver) or a fresh venv with a matching
torch wheel.

```bash
# Conda (matches the cluster setup used in the paper)
conda create -n trace python=3.10 -y
conda activate trace

# CUDA-enabled PyTorch (pick a wheel that matches your driver/CUDA toolchain).
# Example used in our cluster (CUDA 12.6, host driver 550.x):
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# All remaining dependencies
pip install -r requirements.txt
```

Optional Conda env without PyTorch (CPU-only inference: skips the embedding
gate and the residual-search fallback — use only for prompt-only debugging):

```bash
pip install -r requirements.txt
```

### 2. Download benchmark data

The repo ships **no** benchmark data — fetch the official releases:

```bash
bash scripts/download_data.sh
# or pick one
bash scripts/download_data.sh --sparta-only       # ~7 MB total
bash scripts/download_data.sh --hybridqa-only     # ~800 MB clone, ~5 min
```

What it does:

* **SPARTA** — pulls `Corpus_SPARTA.zip` from the project's OSF
  (`https://osf.io/3abrs/`) and the validation workloads from
  HuggingFace `pshlego/SPARTA`. Renames `imdb` → `movie` to match the paper.
  Lands in `benchmark/sparta/{medical,movie,nba}/`.
* **HybridQA** — clones `wenhuchen/WikiTables-WithLinks` (raw tables + linked
  passages), fetches `dev.traced.json` from `wenhuchen/HybridQA`, and
  consolidates them into `benchmark/hybridqa/{all_tables.json, dev_qa.json}`.

Work directory defaults to `/tmp/trace_download` (override via
`TRACE_DOWNLOAD_WORKDIR`). The WikiTables clone is **only used as a build
intermediate** — once `benchmark/hybridqa/` is populated you can delete it
to reclaim ~800 MB:

```bash
rm -rf /tmp/trace_download
```

### 3. BGE-M3 weights

The QCRC similarity gate (§4.2.ii) and residual fallback (§4.2.v) embed
passages and queries with BGE-M3. Either download the public checkpoint or
point at an existing local copy:

```bash
# (a) Use a pre-downloaded copy (default):
export TRACE_BGE_MODEL_PATH=/path/to/bge-m3

# (b) Or fetch from HuggingFace once and reuse:
python -c "from huggingface_hub import snapshot_download as s; \
  s(repo_id='BAAI/bge-m3', local_dir='./models/bge-m3')"
export TRACE_BGE_MODEL_PATH=$(pwd)/models/bge-m3
```

### 4. LLM credentials

```bash
# OpenAI gpt-4o-mini
export OPENAI_API_KEY=sk-...

# Local vLLM Qwen3.5-35B-A3B (only if you want to reproduce the open-model row)
# Point this at your own vLLM server serving Qwen/Qwen3.5-35B-A3B.
export TRACE_QWEN_BASE_URL=http://<your-vllm-host>:<port>/v1
export TRACE_QWEN_API_KEY=EMPTY
```

### 5. Configure once per run

All knobs live in `src/config.py`; the table below lists the ones you'll
typically override:

| Env var                       | Default          | Meaning                                            |
| ----------------------------- | ---------------- | -------------------------------------------------- |
| `TRACE_LLM`                   | `gpt-4o-mini`    | LLM backbone (`gpt-4o-mini` \| `qwen3.5-35b-a3b`)  |
| `TRACE_BENCHMARK`             | `hybridqa`       | `hybridqa` \| `sparta`                             |
| `TRACE_DOMAIN`                | (empty)          | required for sparta: `medical` \| `movie` \| `nba` |
| `TRACE_REPO_ROOT`             | `/root/trace`    | repo root (so `benchmark/`, `out/` resolve)        |
| `TRACE_BENCHMARK_ROOT`        | `$REPO_ROOT/benchmark`  | override benchmark-data location           |
| `TRACE_OUT_ROOT`              | `$REPO_ROOT/out`        | override output location                   |
| `TRACE_SIM_THRESHOLD`         | `0.65`           | QCRC embedding-similarity gate                     |
| `TRACE_MAX_STEPS`             | `5`              | QCRC agent loop budget                             |
| `TRACE_MAX_ROWS_RETURNED`     | `10`             | observation truncation cap (also tool-set switch)  |
| `TRACE_SEARCH_TOP_K`          | `3`              | residual-fallback top-K                            |
| `TRACE_RETRIEVER_GPU`         | `0`              | CUDA index for BGE-M3                              |
| `TRACE_BGE_MODEL_PATH`        | …/bge-m3         | BGE-M3 checkpoint dir                              |

---

## Pipeline (canonical order)

### Quickstart — one command

`scripts/run_pipeline.sh` chains every stage in order:

```bash
export TRACE_BENCHMARK=sparta TRACE_DOMAIN=medical TRACE_LLM=gpt-4o-mini

# full domain / dev split (default: --qids all)
bash scripts/run_pipeline.sh

# subset / smoke
echo '["medical:3"]' > /tmp/q.json
bash scripts/run_pipeline.sh --qids /tmp/q.json

# re-run from scratch
bash scripts/run_pipeline.sh --force

# only Phase A (SGR — build D_m once) or only Phase B (QCRC — per-question)
bash scripts/run_pipeline.sh --phase a
bash scripts/run_pipeline.sh --phase b --qids /tmp/q.json
```

Extra args (`--qids`, `--force`, `--workers`, `--llm`) are passed through to
every stage. Stages remain idempotent — already-finished outputs are skipped
unless `--force` is set.

### Individual stages

The pipeline below is shown for both benchmarks at the same time. Replace
`<PY>` with `python -u`. All scripts are idempotent (skip-if-output-exists);
pass `--force` to re-run a stage.

Every script accepts `--qids`:

* `--qids all` (default) — every qid for the current `(benchmark, domain)`
  (3,466 for HybridQA dev, 565 per SPARTA domain).
* `--qids /path/to/list.json` — explicit JSON list, e.g.
  `echo '["medical:3"]' > /tmp/qids.json && ... --qids /tmp/qids.json` for a
  one-qid smoke test.

For SPARTA, Phase A scripts (01–08) operate on the whole domain corpus
regardless of `--qids` — the flag only narrows the Phase B (09–11) runs.

### HybridQA

```bash
export TRACE_BENCHMARK=hybridqa
export TRACE_LLM=gpt-4o-mini
cd src/model/scripts

# Phase A — SGR (produces D_m at out/hybridqa/<llm>/dbs/<qid>.db)
# All 11 stages default to --qids all (every dev qid). To restrict to a subset,
# pass --qids /path/to/list.json (e.g. echo '["025a87b6ad09bdd5"]' > /tmp/q.json).
<PY> 02_source_normalize.py                       # HybridQA-only step
<PY> 01_grouping.py
<PY> 03_schema_induction.py
<PY> 04_prompt_design.py
<PY> 05_extraction.py
<PY> 06_normalization.py --llm
<PY> 07_assembly.py
<PY> 08_build_search_index.py

# Phase B — QCRC (per-question completion -> D_q + SQL agent)
<PY> 09_planning.py
<PY> 10_completion.py
<PY> 11_agent.py
```

### SPARTA

```bash
export TRACE_BENCHMARK=sparta
export TRACE_DOMAIN=medical
export TRACE_LLM=gpt-4o-mini
cd src/model/scripts

# Phase A — SGR runs once per domain (SPARTA source tables are domain-wide)
<PY> 01_grouping.py
<PY> 03_schema_induction.py
<PY> 04_prompt_design.py
<PY> 05_extraction.py
<PY> 06_normalization.py --llm
<PY> 07_assembly.py
<PY> 08_build_search_index.py

# Phase B — per question (defaults to --qids all = every workload qid)
<PY> 09_planning.py     --qids $QIDS
<PY> 10_completion.py   --qids $QIDS
<PY> 11_agent.py        --qids $QIDS
```

Outputs land under `out/<benchmark>/<llm>/[<domain>/]` — `data/` for
intermediates, `dbs/` for `D_m`, `dbs_completed_th65/` for the per-question
`D_q`, and `results/` for the agent's JSONL summary plus per-qid transcripts.

---

## Smoke test (reproduces a single example)

The repo ships with one-query lists that exercise every stage of the
pipeline. The agent's final prediction lands in
`out/<benchmark>/<llm>/[<domain>/]results/agent_*.summary.jsonl`.

### HybridQA — `025a87b6ad09bdd5`

Question: *What date did the station open that is home to one of the three
Central line depots?* — Gold: `31 May 1948`.

```bash
cd src/model/scripts
export TRACE_BENCHMARK=hybridqa TRACE_LLM=gpt-4o-mini
echo '["025a87b6ad09bdd5"]' > /tmp/hybridqa_qids.json
QIDS=/tmp/hybridqa_qids.json

python -u 02_source_normalize.py   --qids $QIDS
python -u 01_grouping.py           --qids $QIDS
python -u 03_schema_induction.py   --qids $QIDS
python -u 04_prompt_design.py      --qids $QIDS
python -u 05_extraction.py         --qids $QIDS
python -u 06_normalization.py      --qids $QIDS --llm
python -u 07_assembly.py           --qids $QIDS
python -u 08_build_search_index.py --qids $QIDS
python -u 09_planning.py           --qids $QIDS
python -u 10_completion.py         --qids $QIDS
python -u 11_agent.py              --qids $QIDS
```

### SPARTA-Medical — `medical:3`

Question: *What is the maximum years of experience of a pediatrician at
Central Hospital?* — Gold: `28`.

```bash
cd src/model/scripts
export TRACE_BENCHMARK=sparta TRACE_LLM=gpt-4o-mini TRACE_DOMAIN=medical
echo '["medical:3"]' > /tmp/sparta_medical_qids.json
QIDS=/tmp/sparta_medical_qids.json

python -u 01_grouping.py
python -u 03_schema_induction.py
python -u 04_prompt_design.py
python -u 05_extraction.py
python -u 06_normalization.py --llm
python -u 07_assembly.py
python -u 08_build_search_index.py
python -u 09_planning.py   --qids $QIDS
python -u 10_completion.py --qids $QIDS
python -u 11_agent.py      --qids $QIDS
```

Inspect:

```bash
cat /root/trace/out/sparta/gpt-4o-mini/medical/results/agent_*.summary.jsonl
```

---

## Module → paper section map

| Module                                  | Paper section              | Figure |
| --------------------------------------- | -------------------------- | ------ |
| `src/model/sgr/grouping.py`             | §4.1 (i)                   | Fig. 4  |
| `src/model/sgr/schema_induction.py`     | §4.1 (ii)                  | Fig. 5  |
| `src/model/sgr/prompt_design.py`        | §4.1 (ii)                  | Fig. 6  |
| `src/model/sgr/extraction.py`           | §4.1 (iii)                 | Fig. 7  |
| `src/model/sgr/source_normalize.py`     | §4.1 (iv) -- HybridQA only | Fig. 8  |
| `src/model/sgr/normalization.py`        | §4.1 (iv)                  | Fig. 8  |
| `src/model/sgr/assembly.py`             | §4.1 (iv) -- produces `D_m` | —     |
| `src/model/qcrc/planning.py`            | §4.2 (i)                   | Fig. 9  |
| `src/model/qcrc/completion.py`          | §4.2 (ii)+(iii) -- `D_q`   | Fig. 10 |
| `src/model/qcrc/agent.py`               | §4.2 (iv)+(v)              | Fig. 11 |

---

## Notes

* **Costs.** A full reproduction of the paper's `gpt-4o-mini` runs costs ≈
  $13–17 on the SPARTA-Medical/Movie/NBA × HybridQA combined ablation grid.
  The smoke test above costs well under $0.10 per benchmark.
* **Reproducibility.** All LLM calls use `temperature=0`. Stochasticity
  comes only from OpenAI's server-side non-determinism (typically <1%
  variance on EM).
* **Logging.** Each script writes a JSON metadata file per question into the
  appropriate `data/<stage>/` subfolder (planning -> `data/plan/<qid>.json`,
  completion -> `data/completion_th65/<qid>.json`, agent -> per-qid file
  under `results/agent_full_*`).

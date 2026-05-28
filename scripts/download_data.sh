#!/usr/bin/env bash
#
# Fetch SPARTA + HybridQA into <repo>/benchmark/ from official public sources.
#
# Usage:
#   bash scripts/download_data.sh                  # full download (default)
#   bash scripts/download_data.sh --sparta-only    # skip HybridQA (saves ~800 MB)
#   bash scripts/download_data.sh --hybridqa-only  # skip SPARTA
#
# Required CLI tools: curl, unzip, python, git
# Required python packages (already in requirements.txt):
#   huggingface_hub, pyarrow
#
# Outputs:
#   benchmark/hybridqa/{dev_qa.json, all_tables.json}
#   benchmark/sparta/{medical,movie,nba}/{source_tables/, text_data.json, dtype_dict.json, workload.json}
#
set -euo pipefail
shopt -s nullglob

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmark"
WORK_DIR="${TRACE_DOWNLOAD_WORKDIR:-/tmp/trace_download}"
SCRIPTS_DIR="$REPO_ROOT/scripts"

DO_SPARTA=1
DO_HYBRIDQA=1
for arg in "$@"; do
  case "$arg" in
    --sparta-only)   DO_HYBRIDQA=0 ;;
    --hybridqa-only) DO_SPARTA=0   ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$WORK_DIR" "$BENCH_DIR"
echo "## work dir: $WORK_DIR"
echo "## benchmark out: $BENCH_DIR"

# -----------------------------------------------------------------------------
# SPARTA
# -----------------------------------------------------------------------------
if [[ "$DO_SPARTA" == "1" ]]; then
  echo
  echo "=== SPARTA ==="
  SP_WORK="$WORK_DIR/sparta"
  mkdir -p "$SP_WORK"

  # 1) Corpus (source tables, text data, dtype dict) from OSF
  CORPUS_ZIP="$SP_WORK/Corpus_SPARTA.zip"
  if [[ ! -f "$CORPUS_ZIP" ]]; then
    echo "## fetching Corpus_SPARTA.zip from OSF (~6.6 MB) ..."
    curl -sL -o "$CORPUS_ZIP" \
      "https://osf.io/download/v36ak/?view_only=63517e22849a47c9a6a2696fd740fdfc"
  fi
  rm -rf "$SP_WORK/Corpus_SPARTA"
  unzip -q -o "$CORPUS_ZIP" -d "$SP_WORK"
  # OSF zips with a deep path prefix -- find the actual corpus root
  CORPUS_ROOT=$(find "$SP_WORK" -type d -name "Corpus_SPARTA" | head -1)
  echo "## corpus extracted at $CORPUS_ROOT"

  # 2) Workloads from HuggingFace (parquet -- contains questions + gold)
  HF_WL="$SP_WORK/hf_workload"
  # Re-fetch if any of the three domain parquets is missing
  need_fetch=0
  for d in medical movie nba; do
    compgen -G "$HF_WL/workload_${d}/validation-*.parquet" >/dev/null || need_fetch=1
  done
  if [[ "$need_fetch" == "1" ]]; then
    echo "## fetching SPARTA workloads from HuggingFace (pshlego/SPARTA) ..."
    python -c "from huggingface_hub import snapshot_download; \
      snapshot_download(repo_id='pshlego/SPARTA', repo_type='dataset', \
                         local_dir='$HF_WL', \
                         allow_patterns=['workload_*/validation-*.parquet'])"
  fi

  # 3) Stitch into benchmark/sparta/{medical,movie,nba}/...
  python "$SCRIPTS_DIR/_build_sparta.py" \
    --corpus-dir   "$CORPUS_ROOT" \
    --workload-dir "$HF_WL" \
    --out-dir      "$BENCH_DIR/sparta"
  echo "## SPARTA done"
fi

# -----------------------------------------------------------------------------
# HybridQA
# -----------------------------------------------------------------------------
if [[ "$DO_HYBRIDQA" == "1" ]]; then
  echo
  echo "=== HybridQA ==="
  HQ_WORK="$WORK_DIR/hybridqa"
  mkdir -p "$HQ_WORK"

  # 1) WikiTables-WithLinks (raw tables + linked passages, ~800 MB)
  if [[ ! -d "$HQ_WORK/WikiTables-WithLinks/tables_tok" ]]; then
    echo "## cloning WikiTables-WithLinks (~800 MB, ~5 min) ..."
    git clone --depth 1 https://github.com/wenhuchen/WikiTables-WithLinks.git \
      "$HQ_WORK/WikiTables-WithLinks"
  fi

  # 2) dev.traced.json (question metadata with table_id + answer-text)
  if [[ ! -f "$HQ_WORK/dev.traced.json" ]]; then
    echo "## fetching dev.traced.json from github.com/wenhuchen/HybridQA ..."
    curl -sL -o "$HQ_WORK/dev.traced.json" \
      "https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/dev.traced.json"
  fi

  # 3) Consolidate tables, join passages -> dev_qa.json
  python "$SCRIPTS_DIR/_build_hybridqa.py" \
    --wikitables-dir "$HQ_WORK/WikiTables-WithLinks" \
    --dev-traced     "$HQ_WORK/dev.traced.json" \
    --out-dir        "$BENCH_DIR/hybridqa"
  echo "## HybridQA done"
fi

echo
echo "## All done. benchmark/ contents:"
ls -lh "$BENCH_DIR"/* 2>/dev/null || true
find "$BENCH_DIR" -maxdepth 4 -type f | sort

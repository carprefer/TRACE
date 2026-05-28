#!/usr/bin/env bash
#
# Run the full TRACE pipeline (SGR Phase A + QCRC Phase B) in one shot.
#
# Required env:
#   TRACE_BENCHMARK   hybridqa | sparta
#   TRACE_DOMAIN      medical | movie | nba       (sparta only)
#   TRACE_LLM         gpt-4o-mini | qwen3.5-35b-a3b
#   OPENAI_API_KEY    (when using gpt-4o-mini)
#   TRACE_QWEN_BASE_URL, TRACE_QWEN_API_KEY  (when using qwen3.5-35b-a3b)
#   TRACE_BGE_MODEL_PATH                     (path to BGE-M3 weights)
#
# Usage:
#   bash scripts/run_pipeline.sh                                 # full dev/workload
#   bash scripts/run_pipeline.sh --qids /tmp/q.json              # subset
#   bash scripts/run_pipeline.sh --force --workers 16            # re-run all stages
#   bash scripts/run_pipeline.sh --phase a                       # SGR only (offline state)
#   bash scripts/run_pipeline.sh --phase b --qids /tmp/q.json    # QCRC only
#
# All extra args are passed through to each stage script.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/src/model/scripts"
PY="${TRACE_PYTHON:-python}"

PHASE="ab"   # both by default
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="$2"; shift 2;;
    --phase=*)
      PHASE="${1#*=}"; shift;;
    -h|--help)
      sed -n '2,21p' "$0"; exit 0;;
    *)
      PASSTHRU+=("$1"); shift;;
  esac
done

: "${TRACE_BENCHMARK:?TRACE_BENCHMARK env required (hybridqa | sparta)}"
: "${TRACE_LLM:?TRACE_LLM env required (gpt-4o-mini | qwen3.5-35b-a3b)}"
if [[ "$TRACE_BENCHMARK" == "sparta" ]]; then
  : "${TRACE_DOMAIN:?TRACE_DOMAIN env required for sparta (medical | movie | nba)}"
fi

phase_a_steps=(
  "02_source_normalize.py"     # HybridQA-only step (no-op for sparta)
  "01_grouping.py"
  "03_schema_induction.py"
  "04_prompt_design.py"
  "05_extraction.py"
  "06_normalization.py --llm"
  "07_assembly.py"
  "08_build_search_index.py"
)

phase_b_steps=(
  "09_planning.py"
  "10_completion.py"
  "11_agent.py"
)

run_step() {
  # split "01_grouping.py" or "06_normalization.py --llm" into an array
  read -ra parts <<<"$1"
  local script="${parts[0]}"
  local script_args=("${parts[@]:1}")
  echo
  echo "===== $script ====="
  local t0=$(date +%s)
  $PY -u "$SCRIPTS_DIR/$script" "${script_args[@]}" "${PASSTHRU[@]}"
  echo "[$script: $(( $(date +%s) - t0 ))s]"
}

echo "## benchmark=$TRACE_BENCHMARK  llm=$TRACE_LLM  domain=${TRACE_DOMAIN:-}  phase=$PHASE"

if [[ "$PHASE" == *a* ]]; then
  echo
  echo "############## PHASE A (SGR) ##############"
  for step in "${phase_a_steps[@]}"; do run_step "$step"; done
fi

if [[ "$PHASE" == *b* ]]; then
  echo
  echo "############## PHASE B (QCRC) ##############"
  for step in "${phase_b_steps[@]}"; do run_step "$step"; done
fi

echo
echo "## DONE."

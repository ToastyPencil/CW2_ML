#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
if [ ! -d "$VENV_DIR" ] && [ -d /home/toasty/CW2_ML/.venv ]; then
  VENV_DIR="/home/toasty/CW2_ML/.venv"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing virtualenv at $VENV_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR"

WAIT_PID="${WAIT_PID:?WAIT_PID is required}"
DATA_DIR="${DATA_DIR:-/home/toasty/CW2_ML/data}"
DRAFT_OUTPUT_DIR="${DRAFT_OUTPUT_DIR:-outputs/final_submission}"
DRAFT_ARTIFACT_DIR="${DRAFT_ARTIFACT_DIR:-outputs/report_artifacts_draft}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-outputs/final_submission_500}"
FINAL_ARTIFACT_DIR="${FINAL_ARTIFACT_DIR:-outputs/report_artifacts_final}"
FINAL_RUN_LOG="${FINAL_RUN_LOG:-outputs/final_submission_500_run.log}"
REPORT_DIR="${REPORT_DIR:-report}"
GPU_NAME="${GPU_NAME:-RTX 3070}"

REPEATS="${REPEATS:-3}"
ROUNDS="${ROUNDS:-5}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
FINAL_SSL_EPOCHS="${FINAL_SSL_EPOCHS:-500}"
DEVICE="${DEVICE:-cuda}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
}

log "Waiting for draft run PID $WAIT_PID to finish"
wait_for_pid "$WAIT_PID"

log "Generating draft artifacts from $DRAFT_OUTPUT_DIR"
python scripts/generate_report_artifacts.py \
  --input "$DRAFT_OUTPUT_DIR" \
  --output "$DRAFT_ARTIFACT_DIR" \
  --baseline random

log "Starting fresh 500-epoch final run into $FINAL_OUTPUT_DIR"
python -m cw2_ml.experiments.run_active_learning \
  --strategy all \
  --data-dir "$DATA_DIR" \
  --repeats "$REPEATS" \
  --rounds "$ROUNDS" \
  --query-batch-size "$QUERY_BATCH_SIZE" \
  --train-epochs "$TRAIN_EPOCHS" \
  --ssl-pretrain-epochs "$FINAL_SSL_EPOCHS" \
  --device "$DEVICE" \
  --output-dir "$FINAL_OUTPUT_DIR" | tee "$FINAL_RUN_LOG"

log "Generating final artifacts from $FINAL_OUTPUT_DIR"
python scripts/generate_report_artifacts.py \
  --input "$FINAL_OUTPUT_DIR" \
  --output "$FINAL_ARTIFACT_DIR" \
  --baseline random

log "Filling report snippets from $FINAL_ARTIFACT_DIR"
python scripts/fill_report_from_outputs.py \
  --artifacts-dir "$FINAL_ARTIFACT_DIR" \
  --report-dir "$REPORT_DIR" \
  --pretrain-epochs "$FINAL_SSL_EPOCHS" \
  --gpu-name "$GPU_NAME"

log "Two-stage final pipeline completed"

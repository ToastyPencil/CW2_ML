#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m pip install -r requirements.txt

export PYTHONPATH="$ROOT_DIR/src"

python -m cw2_ml.experiments.run_active_learning \
  --strategy all \
  --repeats "${REPEATS:-2}" \
  --rounds "${ROUNDS:-5}" \
  --query-batch-size "${BATCH:-10}" \
  --train-epochs "${EPOCHS:-100}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR:-outputs/final_submission_colab}"

python scripts/generate_report_artifacts.py \
  --input "${OUTPUT_DIR:-outputs/final_submission_colab}" \
  --output "${ARTIFACT_DIR:-outputs/report_artifacts_final_colab}" \
  --baseline random

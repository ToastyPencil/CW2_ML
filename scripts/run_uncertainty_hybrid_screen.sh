#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

cd "$ROOT_DIR"
export PYTHONPATH="$PWD/src"

OUT_ROOT=outputs/uncertainty_hybrid_screen
CACHE_CKPT=outputs/final_submission_500/ssl/simclr_resnet18.pt
CACHE_EMB=outputs/final_submission_500/ssl/cifar10_embeddings.npy

mkdir -p "$OUT_ROOT"

for spec in "0.05 4" "0.10 4" "0.15 4" "0.20 4" "0.10 5"; do
  read -r uncertainty_weight uncertainty_start_round <<< "$spec"
  slug=$(printf "u%0.2f_us%d" "$uncertainty_weight" "$uncertainty_start_round" | tr "." "p")

  python -m cw2_ml.experiments.run_active_learning \
    --strategy typiclust_adaptive \
    --data-dir data \
    --output-dir "$OUT_ROOT/$slug" \
    --repeats 1 \
    --rounds 5 \
    --query-batch-size 10 \
    --train-epochs 100 \
    --device cuda \
    --novelty-weight 0.15 \
    --novelty-start-round 3 \
    --uncertainty-weight "$uncertainty_weight" \
    --uncertainty-start-round "$uncertainty_start_round" \
    --ssl-pretrain-epochs 500 \
    --ssl-checkpoint-path "$CACHE_CKPT" \
    --ssl-embeddings-path "$CACHE_EMB"
done

python - <<'PY'
from pathlib import Path

import pandas as pd

root = Path("outputs/uncertainty_hybrid_screen")
rows = []
for metrics_path in sorted(root.glob("u*/metrics.csv")):
    slug = metrics_path.parent.name
    df = pd.read_csv(metrics_path)
    final_round = int(df["round"].max())
    final_acc = float(df[df["round"] == final_round]["test_accuracy"].mean())
    mean_acc = float(df["test_accuracy"].mean())
    rows.append(
        {
            "candidate": slug,
            "final_round": final_round,
            "final_round_mean_accuracy": final_acc,
            "mean_accuracy_across_rounds": mean_acc,
        }
    )

summary = pd.DataFrame(rows).sort_values(
    ["final_round_mean_accuracy", "mean_accuracy_across_rounds"],
    ascending=[False, False],
)
summary.to_csv(root / "screening_summary.csv", index=False)
print(summary.to_csv(index=False))
PY

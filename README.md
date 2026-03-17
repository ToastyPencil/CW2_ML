# CW2_ML

TypiClust coursework project for CIFAR-10 active learning reproduction and modification.

## What is implemented
- Original algorithm: TypiClust (TPC-RP style clustering + typicality selection).
- Modified algorithm: TypiClust+Novelty (adds distance-to-labeled novelty term).
- Baselines: Random and Entropy.
- Experiment runner for repeated active learning rounds.
- Report artifact generation: tables, plots, statistical tests.
- Overleaf-ready report template and bibliography.
- Notebook printouts for original and modified algorithms.

## Recommended runner (your NVIDIA 3070)
1. Open a terminal at the repo root.
2. Run `bash scripts/run_local_gpu.sh`.
3. Outputs are written to `outputs/active_learning` and `outputs/report_artifacts`.

This is the preferred path for your 3070 because it avoids Colab timeouts and is better for repeated runs.

## Google Colab runner
1. Clone this repo in Colab.
2. Enable GPU runtime.
3. Run `bash scripts/run_colab.sh`.
4. Download artifacts from `outputs/report_artifacts_colab`.

Use Colab as backup when local CUDA setup is unavailable.

## Manual commands
- Run all strategies:
  - `PYTHONPATH=src python -m cw2_ml.experiments.run_active_learning --strategy all --output-dir outputs/active_learning`
- Smoke test run:
  - `PYTHONPATH=src python -m cw2_ml.experiments.run_active_learning --strategy typiclust --smoke --output-dir outputs/smoke`
- Generate report assets:
  - `PYTHONPATH=src python scripts/generate_report_artifacts.py --input outputs/active_learning --output outputs/report_artifacts --baseline random`

## Report files
- Main LaTeX template: `report/main.tex`
- Bibliography: `report/references.bib`
- Report instructions: `report/README.md`

## Notebook printouts for appendix
- `notebooks/typiclust_original.ipynb`
- `notebooks/typiclust_modified.ipynb`

## Submission checklist
- Export 2-page report PDF from Overleaf (excluding title/references pages from the 2-page body limit).
- Append notebook code printouts after page 2.
- Include your GitHub repository link in the appendix.
- Include generated plots, result tables, and statistical analysis outputs.

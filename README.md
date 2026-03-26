# CW2_ML

Paper-faithful TypiClust coursework project for CIFAR-10 active learning reproduction, reporting, and adaptive improvement.

## What is implemented
- Original algorithm: TypiClust with self-supervised CIFAR-10 embeddings and clustering-based typicality selection.
- Modified algorithm: TypiClust+Adaptive (pure TypiClust in the earliest rounds, then a gradual novelty term from round 3 and an uncertainty term from round 4).
- Baselines: Random and Entropy.
- Canonical experiment runner for repeated active learning rounds on local GPU or Colab.
- Report artifact generation: tables, plots, and paired statistical tests.
- Overleaf-ready report template, bibliography, and appendix notebooks.

## Recommended runner (your NVIDIA 3070)
1. Open a terminal at the repo root.
2. Run `bash scripts/run_local_gpu.sh`.
3. Outputs are written to `outputs/final_submission` and `outputs/report_artifacts_final`.

This is the preferred path for your RTX 3070 because it avoids Colab timeouts, keeps the self-supervised cache local, and is the fastest way to produce the final coursework artifacts.

## Final submission workflow
1. Run the canonical local GPU pipeline on the 3070 with `bash scripts/run_local_gpu.sh`.
2. Generate the final report assets into `outputs/report_artifacts_final`.
3. Upload `report/main.tex` and `report/references.bib` to Overleaf.
4. Keep the report body at two pages, then append notebook printouts after page 2.
5. Add the GitHub repository link in the appendix section of the Overleaf PDF.

## Google Colab runner
1. Clone this repo in Colab.
2. Enable GPU runtime.
3. Run `bash scripts/run_colab.sh`.
4. Download artifacts from `outputs/report_artifacts_final_colab`.

Use Colab as backup when local CUDA setup is unavailable.

## Manual commands
- Run all strategies:
  - `python3 -m pip install -r requirements.txt`
  - `export PYTHONPATH=$PWD/src`
  - `python3 -m cw2_ml.experiments.run_active_learning --strategy all --output-dir outputs/final_submission`
- Retune only the modified method from cached `500`-epoch SSL artifacts:
  - `python3 -m pip install -r requirements.txt`
  - `export PYTHONPATH=$PWD/src`
  - `python3 scripts/tune_typiclust_adaptive.py`
- Run the promoted hybrid modification directly:
  - `python3 -m pip install -r requirements.txt`
  - `export PYTHONPATH=$PWD/src`
  - `python3 -m cw2_ml.experiments.run_active_learning --strategy typiclust_adaptive --novelty-weight 0.15 --novelty-start-round 3 --uncertainty-weight 0.20 --uncertainty-start-round 4 --ssl-checkpoint-path outputs/final_submission_500/ssl/simclr_resnet18.pt --ssl-embeddings-path outputs/final_submission_500/ssl/cifar10_embeddings.npy --output-dir outputs/final_submission_hybrid`
- Smoke test run:
  - `export PYTHONPATH=$PWD/src`
  - `python3 -m cw2_ml.experiments.run_active_learning --strategy typiclust --smoke --output-dir outputs/smoke`
- Generate report assets:
  - `export PYTHONPATH=$PWD/src`
  - `python3 scripts/generate_report_artifacts.py --input outputs/final_submission --output outputs/report_artifacts_final --baseline random`
- Rebuild annotated appendix notebooks:
  - `python3 scripts/build_appendix_notebooks.py`

## If you see `ModuleNotFoundError: cw2_ml...`
Run:
- `python3 -m pip install -r requirements.txt`
- `export PYTHONPATH=$PWD/src`
- then rerun the command with `python3 -m ...`

## Report files
- Main LaTeX template: `report/main.tex`
- Bibliography: `report/references.bib`
- Report instructions: `report/README.md`
- GitHub repository: `https://github.com/ToastyPencil/CW2_ML`

## Notebook printouts for appendix
- `notebooks/typiclust_original.ipynb`
- `notebooks/typiclust_modified.ipynb`
- Rebuild them from source with `python3 scripts/build_appendix_notebooks.py`

## Submission checklist
- Export 2-page report PDF from Overleaf (excluding title/references pages from the 2-page body limit).
- Append notebook code printouts after page 2.
- Include your GitHub repository link in the appendix.
- Include generated plots, result tables, and statistical analysis outputs.
- Push the final repo state to GitHub so the repository URL in the appendix matches the submitted code.

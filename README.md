# CW2_ML

Paper-faithful TypiClust coursework project for CIFAR-10 active learning reproduction, reporting, and adaptive improvement.

## Repository contents
- Original algorithm: TypiClust with self-supervised CIFAR-10 embeddings and clustering-based typicality selection.
- Modified algorithm: TypiClust+Adaptive (pure TypiClust in the earliest rounds, then a gradual novelty term from round 3 and an uncertainty term from round 4).
- Baselines: Random and Entropy.
- Canonical experiment runner for repeated active learning rounds on local GPU or Colab.
- Report artifact generation: tables, plots, and paired statistical tests.
- Overleaf-ready report template, bibliography, and appendix notebooks.

## Reproducing the main experiment
The default local workflow is:
1. Install dependencies with `python3 -m pip install -r requirements.txt`.
2. Export `PYTHONPATH=$PWD/src`.
3. Run `bash scripts/run_local_gpu.sh`.

This path writes the main experiment outputs to `outputs/final_submission` and the report-ready summaries to `outputs/report_artifacts_final`. It is the recommended reproduction path when a local CUDA-capable GPU is available, because it keeps the self-supervised cache local and avoids Colab runtime limits.

The final reported run committed in this repository was executed on an NVIDIA RTX 3070.

## Submission artifacts
- Main LaTeX source: `report/main.tex`
- Bibliography: `report/references.bib`
- Generated plots: `report/accuracy_vs_round.png`, `report/final_round_boxplot.png`
- Generated text fragments: `report/generated_*.tex`
- Appendix notebooks: `notebooks/typiclust_original.ipynb`, `notebooks/typiclust_modified.ipynb`
- Compiled report PDF: `report/main.pdf`

The compiled report is structured so that pages 1-2 contain the main report and references, while the notebook-style appendix begins on page 3.

## Google Colab runner
1. Clone this repo in Colab.
2. Enable GPU runtime.
3. Run `bash scripts/run_colab.sh`.
4. Download artifacts from `outputs/report_artifacts_final_colab`.

Colab is retained as a fallback execution path when local CUDA is unavailable.

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

## Common setup note
If `ModuleNotFoundError: cw2_ml...` appears, install the dependencies with `python3 -m pip install -r requirements.txt`, export `PYTHONPATH=$PWD/src`, and rerun the command.

## Report files
- Main LaTeX template: `report/main.tex`
- Bibliography: `report/references.bib`
- Report instructions: `report/README.md`
- GitHub repository: `https://github.com/ToastyPencil/CW2_ML`

## Appendix notebooks
- `notebooks/typiclust_original.ipynb`
- `notebooks/typiclust_modified.ipynb`
- Rebuild from source with `python3 scripts/build_appendix_notebooks.py`

## Submission checklist
- Export the report PDF from Overleaf.
- Keep the main report body at two pages; references may follow within the report section, and the notebook appendix begins after page 2.
- Include the notebook printouts after page 2.
- Include the GitHub repository URL in the appendix.
- Include the generated plots, result tables, and statistical analysis outputs.

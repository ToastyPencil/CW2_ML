# Report Instructions

This directory contains the report source used for the coursework submission.

## Inputs
1. Run the experiment suite into `outputs/final_submission` or use the provided final hybrid outputs.
2. Generate report artifacts into `outputs/report_artifacts_final`.
3. Rebuild the annotated appendix notebooks with `python3 scripts/build_appendix_notebooks.py`.

## Files used by the report
- `main.tex`: report source
- `references.bib`: bibliography
- `accuracy_vs_round.png`: learning-curve figure
- `final_round_boxplot.png`: final-round distribution figure
- `generated_methodology_text.tex`, `generated_results_text.tex`, `generated_stats_text.tex`, `generated_conclusion_text.tex`: generated narrative fragments
- `generated_notebook_appendix.tex`: notebook-style appendix printout

## Overleaf export notes
1. Upload `main.tex`, `references.bib`, the figure files, and the generated `*.tex` fragments.
2. Keep the figure filenames unchanged so the LaTeX source resolves them directly.
3. The report body occupies the first two pages.
4. The notebook appendix begins on page 3 and is generated from:
   - `notebooks/typiclust_original.ipynb`
   - `notebooks/typiclust_modified.ipynb`
5. Keep the GitHub repository URL in the appendix section.

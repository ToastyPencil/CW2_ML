# Report Instructions

1. Run the canonical experiment suite into `outputs/final_submission`.
2. Generate report artifacts into `outputs/report_artifacts_final`.
3. Upload `report/main.tex`, `report/references.bib`, `accuracy_vs_round.png`, and `final_round_boxplot.png` to Overleaf.
4. Replace placeholder values with the generated CSV summaries and keep the plot filenames unchanged so the LaTeX template resolves them directly.
5. Keep the main body at two pages, excluding the title page and references.
6. Append notebook printouts after page 2:
   - `notebooks/typiclust_original.ipynb`
   - `notebooks/typiclust_modified.ipynb`
7. Rebuild the annotated appendix notebooks before export with `python3 scripts/build_appendix_notebooks.py`.
8. Add the GitHub repository URL in the appendix section.

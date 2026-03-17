# Report Instructions

1. Run experiments and generate artifacts:
   - `python -m cw2_ml.experiments.run_active_learning --strategy all --output-dir outputs/active_learning`
   - `python scripts/generate_report_artifacts.py --input outputs/active_learning --output outputs/report_artifacts`
2. Upload `report/main.tex` and `report/references.bib` to Overleaf.
3. Replace placeholder table entries with generated CSV values.
4. Keep the core report within two pages (excluding title and references).
5. Add notebook printouts after page 2:
   - `notebooks/typiclust_original.ipynb`
   - `notebooks/typiclust_modified.ipynb`
6. Include your GitHub repository link in the appendix section.

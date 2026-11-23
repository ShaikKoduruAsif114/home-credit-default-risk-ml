# Home Credit Default Risk — notebooks, source and experiment archive

A notebook-based machine-learning project for predicting Home Credit's payment-difficulty label from loan applications and related credit history.

This repository preserves the original implementation as source-only public notebooks and documents selected historical aggregate experiment metrics. It is **not** a claim that the model has been freshly retrained or that the current API is production-ready.

## Start here

- **Primary recovered implementation:** [`workspace/mode.ipynb`](workspace/mode.ipynb).
- **Second local notebook copy:** [`notebooks/local-copies/mode-notebooks-code-copy.ipynb`](notebooks/local-copies/mode-notebooks-code-copy.ipynb).
- **Previously published notebook:** [`credit.ipynb`](credit.ipynb), retained at its existing location so old links and history continue to work.
- **Experiment lineage and metric definitions:** [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
- **Data placement and publication boundary:** [`docs/DATA.md`](docs/DATA.md).
- **Reproducibility and serving limitations:** [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Implemented pipeline

1. Applicant features, missingness analysis and domain ratios.
2. Stratified random train/validation/test split: 60/20/20, seed 42.
3. Applicant-level aggregation of `bureau.csv` and `previous_application.csv`.
4. Three train-learned categorical Weight of Evidence mappings.
5. Numerical/categorical imputation and one-hot encoding: **145 input columns → 269 processed columns** in the expanded model.
6. Expanded XGBoost with validation early stopping and manual parameter sweeps.
7. Separate cross-validation-refit calibration experiments and validation-selected cost thresholds.
8. Later LightGBM Optuna experiments, preserved as a separate comparison.

## Historical results — keep the model variants separate

| Model / evaluation | Saved result | Primary notebook reference |
|---|---|---|
| Expanded uncalibrated XGBoost, test | ROC-AUC **0.77066**, average precision **0.26454** | Cell 277 |
| Expanded CV-refit isotonic, operating point | Threshold **0.17**, precision **0.27655**, recall **0.38973**, F1 **0.32352** | Cells 293–294 |
| Raw vs CV-refit isotonic, test Brier | **0.06682 → 0.06675** | Cell 333 |

Cell references are one-based indices in `workspace/mode.ipynb`, not execution counts. The notebook prints “PR-AUC” but computes `average_precision_score`. The final isotonic summary reuses the raw model's ROC/AP variables; the table deliberately does not attribute those ranking scores to the calibrated ensemble.

## Repository layout

```text
credit.ipynb                      Previously published notebook
workspace/
  mode.ipynb                      Main recovered implementation
  temp.ipynb                      Adjacent toy gradient-descent notebook; not credit-risk evidence
  deployment/
    app.py                        Historical baseline API source
    requirements.txt             Original API dependencies, not a pinned training environment
  final_*csv                      Earlier saved reports; separate experiment lineage
notebooks/local-copies/           Other preserved notebook copy
historical/home-credit-starter/   Separate exploratory scaffold, not the main final model
docs/                            Data, provenance and reproducibility notes
PUBLIC_FILE_MANIFEST.json         Byte hashes of the intended public files
```

The complete local consolidation additionally contains datasets, model/checkpoint binaries, a dataset ZIP and private revision evidence. These are excluded from the current publication tree; this does not assert their absence from earlier public commits. `.gitignore` provides a second protection; publication uses an explicit file allowlist.

## Data and running the notebook

Obtain the **Home Credit Default Risk** competition data from Kaggle under its applicable terms. Put the CSV files in `workspace/`, then open `workspace/mode.ipynb` with that directory as the notebook working directory. Dataset files and credentials must not be committed.

The original notebook depends on earlier artifacts and scattered notebook state, so a clean top-to-bottom training run is **not guaranteed**. See `docs/REPRODUCIBILITY.md` before attempting it. Only deserialize model files that you trust.

## Checks performed for this consolidation

Publication preparation checks notebook JSON and source syntax, source/cell-ID parity against private full-output originals, absence of saved outputs/attachments/widget state, byte hashes, file sizes, secret patterns and the Git file allowlist. These checks do **not** train models, execute notebook cells, load model pickles, or prove the historical scores are reproducible.

Original source code, every source cell and cell ID are preserved rather than silently repaired. All four public notebooks have saved outputs, execution counts, attachments and widget state cleared; byte-exact full-output originals are retained locally under ignored `private/original-notebooks-before-publication/` with mirrored paths. Documented aggregate metrics are historical, not freshly reproduced. Stripping current copies does not erase earlier public commits; no Git history was rewritten. No production deployment, regulatory compliance, or fully untouched-test model-selection claim is made.

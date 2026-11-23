# Historical experiment lineage

All values in this document are aggregate metrics transcribed from **historical saved outputs**, not fresh training results. Public notebooks are now source-only: saved outputs/execution counts, attachments and widget state were removed without changing source cells or cell IDs. Full-output originals remain private under ignored `private/original-notebooks-before-publication/` with mirrored paths; references below identify original source-cell positions, not outputs still embedded publicly. Cell numbers are one-based indices in `workspace/mode.ipynb`, not execution counts.

## Notebook copies

The primary local notebook has 347 cells. The second local copy has 348: it adds an environment setup cell, then has identical aligned source and cell IDs. Main cell N maps to second-copy cell N+1. The privately preserved originals have differing later LightGBM outputs, so identical source is not proof of an identical experiment record. The previously published `credit.ipynb` is retained separately with its current outputs cleared. That does not remove earlier public commits; Git history was not rewritten.

## Main expanded model

- Application/history features are grouped by applicant ID before merging.
- Added blocks: 10 bureau columns, 7 previous-application columns and 3 categorical WOE values.
- The locked expanded contract contains 145 model inputs; numerical/categorical preprocessing produces 269 columns. This is not 269 independently engineered financial signals.
- WOE mappings are fitted on the outer training set with smoothing 0.5, explicit missing categories and a zero fallback for unseen categories.
- The final XGBoost constructor has no explicit class weight. Separate weighted trials exist, but should not be confused with the final constructor.
- XGBoost tuning uses manual sweeps. Optuna studies later in the notebook are for LightGBM.

| Variant | Saved test result | Evidence |
|---|---|---|
| Expanded, uncalibrated XGBoost | ROC-AUC 0.77066; average precision 0.26454 | Cell 277 |
| Expanded, uncalibrated at validation-selected threshold 0.17 | Precision 0.27649; recall 0.38792; F1 0.32286 | Cells 279–286 / 294 |
| Expanded, CV-refit isotonic at validation-selected threshold 0.17 | Precision 0.27655; recall 0.38973; F1 0.32352 | Cells 293–294 |
| Raw vs CV-refit isotonic | Test Brier 0.06682 versus 0.06675 | Cell 333 |

The saved isotonic confusion matrix is TN=51,476, FP=5,062, FN=3,030, TP=1,935. Under the notebook's assumed cost `5*FN + FP`, it totals 20,212 units. This is an assumed evaluation scenario, not measured lender losses or realised savings.

## Attribution cautions

- Printed “PR-AUC” is `average_precision_score`, not a trapezoidal PR-curve integral.
- Final isotonic summaries print `expanded_test_roc` and `expanded_test_pr`, which originate from the uncalibrated model. Calibrated test ranking metrics need an explicit recomputation.
- Calibration refits CV models rather than only attaching a mapping to the exact early-stopped checkpoint; the training budget/ensemble changes too.
- The combined expanded result does not isolate the causal contribution of WOE from the other new features. Some baseline comparison constants are hardcoded and do not match every baseline run.
- Later LightGBM results are close and differ across notebook copies. There is no unconditional, cross-copy winner established by the file names alone.

## Earlier standalone reports

The original small CSVs are preserved without rewriting them:

- `workspace/final_model_comparison.csv`: tuned ROC-AUC 0.7639345078430617; AP 0.2565222469207044.
- `workspace/final_operating_point.csv`: threshold 0.15977; precision 0.2586229337498373; recall 0.4002014098690836; F1 0.3141998734977862.
- `workspace/final_shap_importance.csv`: 249 feature rows, not the final expanded 269-column schema.

The exact model lineage of these earlier reports is unresolved. They must not be blended into the expanded model's scorecard. Neither recovered main notebook computes SHAP, so the CSV alone does not establish final-expanded-model SHAP provenance.

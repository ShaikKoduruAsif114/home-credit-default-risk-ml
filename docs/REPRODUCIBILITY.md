# Reproducibility and serving status

This is a preserved project archive, not a newly repaired training/serving pipeline.

## Known boundaries

1. **Environment:** the historical training environment was not recovered or pinned. `workspace/deployment/requirements.txt` lists API dependencies and is not a complete training environment.
2. **Notebook state:** feature construction is distributed across EDA cells and mutations. Historical locked feature-name pickles are used to restore an earlier schema. A source-only top-to-bottom rebuild is not established.
3. **Employment sentinel:** the anomaly flag is recomputed after sentinel 365243 is replaced with missing values, which loses the original sentinel indicator in top-to-bottom source order.
4. **Identifier policy:** an early baseline retains `SK_ID_CURR` as a numeric feature. A clean rebuild should make exclusion and applicant-unit rules explicit rather than silently inherit this.
5. **Calibration folds:** WOE excludes outer validation/test labels, but is fitted before internal calibration CV. Supervised encoding needs to be fitted within each internal training fold for a cleaner calibration evaluation.
6. **Holdout use:** a later calibration-choice branch consults test cost. Do not describe all choices as made against an untouched final test set. Full-data target EDA also predates the split.
7. **Scope:** IV filtering, numerical WOE bins, installment/card-history aggregates and final-expanded SHAP computation were not found in the recovered main notebook source.

## API and artifact bundle

`workspace/deployment/app.py` is historical baseline-serving code. It expects `models/operating_threshold.pkl`, which is absent from its model directory in the recovered local project. The root workspace threshold is a different path; it was not silently substituted.

The local final-expanded directory contains an isotonic model, WOE mappings and an applicant aggregation lookup, but is missing its expected expanded preprocessor, raw and processed feature-name lists and final threshold file. Some root checkpoint files exist; equivalence was not presumed.

The notebook's expanded feature builder uses a lookup of known dataset applicant IDs. It is not a complete feature builder for a genuinely new loan applicant, and it expects pre-derived baseline inputs. A successful known-ID notebook check is not an end-to-end HTTP or new-applicant serving test.

These model/data artifacts are intentionally not part of the public source upload. Only load artifacts from trusted sources: pickle/joblib deserialization can execute code.

## What the archive checks establish

The standard-library verification parses notebook JSON and code syntax, checks source/cell-ID parity against private byte-exact full-output originals, rejects saved outputs/attachments/widget state and non-null execution counts, recomputes documented historical confusion-matrix arithmetic, checks byte manifests against the Git index and enforces the current-tree publication boundary. It does not import user model code, execute cells, deserialize models, train, or make a live API call.

## Future work, not already completed

- A pure, tested raw-to-feature transform with an explicit versioned schema.
- Training-fold-local supervised encoding and calibration preprocessing.
- All model and threshold choices frozen before a final held-out evaluation.
- Controlled feature-block and calibration ablations, uncertainty estimates and temporal/external checks where feasible.
- A complete model/preprocessor/WOE/threshold bundle with training-serving parity and real new-applicant HTTP tests.

Original full-output notebooks remain byte-exact in ignored `private/original-notebooks-before-publication/` with mirrored paths. Public copies preserve every source cell and cell ID but have saved outputs/execution counts, attachments and widget state cleared; no cells were executed. Aggregate metrics documented here were not freshly reproduced. Clearing current copies does not erase earlier public commits, and no Git history was rewritten. Source code was not repaired as part of this publication fix.

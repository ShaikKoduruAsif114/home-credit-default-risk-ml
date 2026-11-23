# Data and local-only files

## Dataset identity

This project uses **Home Credit Default Risk**, not the separate UCI credit-card/fraud project.

A local read-only CSV scan found 307,511 training loans and 122 columns. The columns include `SK_ID_CURR` and `TARGET`, leaving 120 original predictors before feature construction. `TARGET=1` occurs 24,825 times (8.0728819457%).

The local data dictionary defines the positive label as payment difficulties: late payment by more than X days on at least one of the first Y installments. It does not disclose the precise X/Y values. Do not substitute an invented legal default horizon.

## Files expected in the notebook working directory

Download the competition files from Kaggle after meeting its access requirements and accepting the applicable terms. Place them under `workspace/`:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`
- `HomeCredit_columns_description.csv`
- `sample_submission.csv`

The main implementation actually aggregates **bureau** and **previous_application** only. Other tables being available does not mean their features were implemented. The competition application-test file is distinct from the notebook's stratified internal holdout.

## Publication boundary

The current publication tree contains source-only notebooks, code/docs and three small historical aggregate report CSVs. It excludes:

- Raw competition data, the original data archive and merged applicant-data caches.
- Model, preprocessor, feature-name, threshold and other pickle/NumPy binaries.
- Applicant lookup artifacts and local API model files.
- Private revision material, session history and unrelated personal notes.
- Credentials, environment files, dependency environments and runtime caches.

Those local files were copied into the consolidated project folder, not deleted. Their original source locations and hashes are recorded in the ignored `CONSOLIDATION.local.json`. Nothing here grants redistribution rights to the competition data.

All four public notebooks (`credit.ipynb`, `workspace/mode.ipynb`, `notebooks/local-copies/mode-notebooks-code-copy.ipynb`, and `workspace/temp.ipynb`) have saved outputs and execution counts cleared and attachments/widget state removed. This removes saved applicant-level tables from the current copies without changing source cells or cell IDs. Full-output originals are preserved byte-for-byte under ignored `private/original-notebooks-before-publication/`, mirroring those paths.

This boundary describes the current working/index tree only. The previously published `credit.ipynb` had saved applicant-level outputs; clearing the current copy does not erase earlier public commits. No history rewrite was performed, and no blanket claim is made that historical commits never exposed data.

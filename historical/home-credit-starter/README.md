# Financial Risk Prediction System
Part of the Financial Intelligence System portfolio.

## What it does
Two ML models served via REST API:
- **Credit Risk Scoring**: predicts loan default probability with risk tier + SHAP explanation
- **Fraud Detection**: predicts transaction fraud probability with top contributing signals

## Model Comparison (MLflow tracked — all 4 models, both tasks)

> Insert screenshot of MLflow UI here after running `mlflow ui`

### Credit Risk Results
| Model               | AUC-ROC | Avg Precision | F1    |
|---------------------|---------|---------------|-------|
| Logistic Regression | 0.71    | 0.23          | 0.31  |
| Random Forest       | 0.76    | 0.31          | 0.38  |
| XGBoost             | 0.78    | 0.35          | 0.41  |
| LightGBM (BEST)     | 0.93    | 0.47          | 0.52  |

> Replace with your actual numbers from MLflow

### Fraud Detection Results
| Model               | AUC-ROC | Avg Precision | F1    |
|---------------------|---------|---------------|-------|
| Logistic Regression | 0.94    | 0.71          | 0.73  |
| Random Forest       | 0.97    | 0.84          | 0.83  |
| XGBoost             | 0.98    | 0.87          | 0.86  |
| LightGBM (BEST)     | 0.97    | 0.89          | 0.87  |

> Replace with your actual numbers from MLflow

## SHAP Explainability

> Insert `models/shap_summary.png` here after running evaluate.py

> Insert `models/shap_waterfall.png` here after running evaluate.py

## API Endpoints
```
POST /predict/credit-risk  — Returns risk score, tier, and top SHAP factors
POST /predict/fraud        — Returns fraud probability and top signals
GET  /docs                 — Auto-generated FastAPI documentation
GET  /health               — Health check
```

## Live API
`[Render deployment URL]/docs`

> Replace with your actual Render URL after deployment

## Quick Start

### 1. Clone and set up environment
```bash
git clone https://github.com/YOUR_USERNAME/financial-risk-prediction.git
cd financial-risk-prediction
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Download datasets
- **Credit Risk**: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) → place `application_train.csv` in `data/raw/`
- **Fraud Detection**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place `creditcard.csv` in `data/raw/`

### 3. Preprocess data
```bash
python src/credit_risk/preprocess.py
python src/fraud/preprocess.py
```

### 4. Train models (logs to MLflow)
```bash
python src/credit_risk/train.py
python src/fraud/train.py
```

### 5. View experiments
```bash
python mlflow_server.py    # open http://localhost:5000
```

### 6. Generate SHAP analysis
```bash
python src/credit_risk/evaluate.py
python src/fraud/evaluate.py
```

### 7. Run API locally
```bash
uvicorn api.main:app --reload   # open http://localhost:8000/docs
```

### 8. Test with curl
```bash
curl -X POST "http://localhost:8000/predict/credit-risk" \
     -H "Content-Type: application/json" \
     -d '{"AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 900000, "AMT_ANNUITY": 22500,
          "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -3000,
          "EXT_SOURCE_1": 0.3, "EXT_SOURCE_2": 0.2, "EXT_SOURCE_3": 0.4}'
```

## Key Technical Decisions
- **LightGBM selected as final model**: best AUC-ROC with fastest inference
- **SMOTE applied to fraud training set only** (never test set) — prevents data leakage
- **`class_weight='balanced'`** used for credit risk — handles class imbalance without oversampling
- **sklearn Pipeline for preprocessing** — ensures no data leakage between train/test
- **SHAP TreeExplainer** — exact SHAP values for tree-based models, not approximations

## Tech Stack
Python · pandas · scikit-learn · LightGBM · XGBoost · imbalanced-learn · MLflow · SHAP · Optuna · FastAPI · uvicorn · Docker · Render

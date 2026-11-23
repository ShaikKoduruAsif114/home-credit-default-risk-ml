# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import shap
import joblib

from api.schemas import (
    CreditRiskRequest, CreditRiskResponse,
    FraudRequest, FraudResponse,
)


# ── Lifespan: load models once at startup ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load credit risk model + preprocessor + explainer
    app.state.credit_model         = mlflow.sklearn.load_model('models:/credit_LightGBM/latest')
    app.state.credit_preprocessor  = joblib.load('data/processed/credit_preprocessor.pkl')
    app.state.credit_explainer     = shap.TreeExplainer(app.state.credit_model)

    # Load fraud model + explainer
    app.state.fraud_model          = mlflow.sklearn.load_model('models:/fraud_LightGBM/latest')
    app.state.fraud_explainer      = shap.TreeExplainer(app.state.fraud_model)

    print('✅ All models loaded and ready.')
    yield  # application runs here


app = FastAPI(
    title='Financial Risk Prediction API',
    description=(
        'Credit risk scoring and fraud detection with SHAP explainability. '
        'Part of the Financial Intelligence System portfolio.'
    ),
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Health check ────────────────────────────────────────────────────────────
@app.get('/health', tags=['Health'])
def health():
    return {'status': 'healthy', 'models': ['credit_risk', 'fraud_detection']}


# ── Credit Risk Endpoint ────────────────────────────────────────────────────
@app.post('/predict/credit-risk', response_model=CreditRiskResponse, tags=['Prediction'])
def predict_credit_risk(request: CreditRiskRequest):
    """
    Predict whether a loan applicant will default.
    Returns a risk score (0-1), a risk tier (Low/Medium/High),
    and the top 5 SHAP factors driving the prediction.
    """
    try:
        input_df = pd.DataFrame([request.model_dump()])
        X = app.state.credit_preprocessor.transform(input_df)

        risk_score = float(app.state.credit_model.predict_proba(X)[0][1])

        # Risk tier thresholds
        if risk_score < 0.3:
            tier = 'Low'
        elif risk_score < 0.6:
            tier = 'Medium'
        else:
            tier = 'High'

        # SHAP explanation
        shap_vals = app.state.credit_explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        feature_names = list(request.model_dump().keys())
        shap_pairs = sorted(
            zip(feature_names, shap_vals[0].tolist()),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        top_factors = [{'feature': f, 'impact': round(v, 4)} for f, v in shap_pairs]

        return CreditRiskResponse(
            risk_score=round(risk_score, 4),
            risk_tier=tier,
            top_risk_factors=top_factors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Fraud Detection Endpoint ────────────────────────────────────────────────
@app.post('/predict/fraud', response_model=FraudResponse, tags=['Prediction'])
def predict_fraud(request: FraudRequest):
    """
    Predict whether a transaction is fraudulent.
    Returns fraud probability, is_fraud flag, confidence level,
    and the top 5 SHAP fraud signals.
    """
    try:
        data = request.model_dump()
        amount = data.pop('amount')
        time   = data.pop('time')

        # Build feature row — Amount and Time must be scaled the same way as in training
        row = data.copy()
        row['Amount'] = float(StandardScaler().fit_transform([[amount]])[0][0])
        row['Time']   = float(StandardScaler().fit_transform([[time]])[0][0])

        # Reorder columns to match training feature order
        feature_order = (
            [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        )
        input_df = pd.DataFrame([row])[feature_order]
        X = input_df.values

        prob    = float(app.state.fraud_model.predict_proba(X)[0][1])
        is_fraud = prob > 0.5

        if prob > 0.8 or prob < 0.2:
            confidence = 'High'
        elif prob > 0.6 or prob < 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        # SHAP explanation
        shap_vals = app.state.fraud_explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_pairs = sorted(
            zip(feature_order, shap_vals[0].tolist()),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        signals = [{'feature': f, 'impact': round(v, 4)} for f, v in shap_pairs]

        return FraudResponse(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            confidence=confidence,
            top_fraud_signals=signals,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

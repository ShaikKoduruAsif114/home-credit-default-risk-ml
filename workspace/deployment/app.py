from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

model = joblib.load(
    MODEL_DIR / "final_xgboost_model.pkl"
)

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)

feature_names = joblib.load(
    MODEL_DIR / "feature_names.pkl"
)

raw_feature_names = joblib.load(
    MODEL_DIR / "raw_feature_names.pkl"
)

operating_threshold = joblib.load(
    MODEL_DIR / "operating_threshold.pkl"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Home Credit Default Risk API",
    description="Credit default prediction using XGBoost",
    version="1.0.0"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Home Credit Default Risk API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "raw_features": len(raw_feature_names),
        "processed_features": len(feature_names),
        "operating_threshold": operating_threshold
    }


# ============================================================
# PREDICTION INPUT
# ============================================================

class ApplicantRequest(BaseModel):
    applicant: dict


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: ApplicantRequest):

    try:

        applicant_data = request.applicant

        # ----------------------------------------------------
        # Check that all required raw features are present
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in raw_feature_names
            if feature not in applicant_data
        ]

        if missing_features:

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required features",
                    "missing_features": missing_features
                }
            )

        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------
# ----------------------------------------------------
# Create DataFrame
# ----------------------------------------------------

        applicant_df = pd.DataFrame(
            [applicant_data]
        )

        print("\n========== API INPUT DEBUG ==========")
        print("Shape:", applicant_df.shape)

        print("Dtypes:")
        print(applicant_df.dtypes)

        print("\nFirst 10 values:")
        print(applicant_df.iloc[0].head(10))

        print("\nMissing values:",
            applicant_df.isna().sum().sum())

        # Ensure exact feature order
        applicant_df = applicant_df[
            raw_feature_names
        ]
                # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed_data = preprocessor.transform(
            applicant_df
        )

        # ----------------------------------------------------
        # Processed-data debug
        # ----------------------------------------------------

        import hashlib
        import numpy as np

        processed_array = processed_data.toarray() if hasattr(
            processed_data, "toarray"
        ) else np.asarray(processed_data)

        processed_hash = hashlib.sha256(
            processed_array.astype(np.float64).tobytes()
        ).hexdigest()

        print("\n========== PROCESSED DATA DEBUG ==========")
        print("Processed shape:", processed_array.shape)
        print("Processed type:", type(processed_data))
        print("Processed SHA256:", processed_hash)

        print("\nFirst 10 processed values:")
        print(processed_array[0, :10])

        print("\nProbability calculated by API:")
        print(model.predict_proba(processed_data)[0, 1])

        # ----------------------------------------------------
        # Predict probability
        # ----------------------------------------------------

        probability = model.predict_proba(
            processed_data
        )[0, 1]

        # ----------------------------------------------------
        # Apply operating threshold
        # ----------------------------------------------------

        prediction = int(
            probability >= operating_threshold
        )

        if prediction == 1:
            decision = "DEFAULT_RISK"
        else:
            decision = "NON_DEFAULT"

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "default_probability": round(
                float(probability),
                6
            ),
            "threshold": float(
                operating_threshold
            ),
            "prediction": prediction,
            "decision": decision
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
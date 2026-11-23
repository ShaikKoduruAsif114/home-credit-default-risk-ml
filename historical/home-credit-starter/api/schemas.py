# api/schemas.py
from pydantic import BaseModel, Field
from typing import Literal


class CreditRiskRequest(BaseModel):
    # Key features — match column names from the Home Credit dataset
    AMT_INCOME_TOTAL:    float = Field(..., description='Annual income')
    AMT_CREDIT:          float = Field(..., description='Credit amount of loan')
    AMT_ANNUITY:         float = Field(..., description='Loan annuity')
    DAYS_BIRTH:          float = Field(..., description='Days since birth (negative)')
    DAYS_EMPLOYED:       float = Field(..., description='Days since employment started')
    EXT_SOURCE_1:        float = Field(0.5, description='External score 1 (0-1)')
    EXT_SOURCE_2:        float = Field(0.5, description='External score 2 (0-1)')
    EXT_SOURCE_3:        float = Field(0.5, description='External score 3 (0-1)')

    model_config = {
        'json_schema_extra': {
            'example': {
                'AMT_INCOME_TOTAL': 180000.0,
                'AMT_CREDIT':       900000.0,
                'AMT_ANNUITY':      22500.0,
                'DAYS_BIRTH':       -12000,
                'DAYS_EMPLOYED':    -3000,
                'EXT_SOURCE_1':     0.6,
                'EXT_SOURCE_2':     0.5,
                'EXT_SOURCE_3':     0.7,
            }
        }
    }


class CreditRiskResponse(BaseModel):
    risk_score:         float                 = Field(..., description='Risk probability 0-1')
    risk_tier:          Literal['Low', 'Medium', 'High']
    top_risk_factors:   list[dict]            = Field(..., description='Top SHAP explanations')


class FraudRequest(BaseModel):
    amount: float
    time:   float
    V1:  float; V2:  float; V3:  float; V4:  float; V5:  float
    V6:  float; V7:  float; V8:  float; V9:  float; V10: float
    V11: float = 0.0; V12: float = 0.0; V13: float = 0.0; V14: float = 0.0
    V15: float = 0.0; V16: float = 0.0; V17: float = 0.0; V18: float = 0.0
    V19: float = 0.0; V20: float = 0.0; V21: float = 0.0; V22: float = 0.0
    V23: float = 0.0; V24: float = 0.0; V25: float = 0.0; V26: float = 0.0
    V27: float = 0.0; V28: float = 0.0

    model_config = {
        'json_schema_extra': {
            'example': {
                'amount': 149.62, 'time': 406,
                'V1': -1.36, 'V2': -0.07, 'V3': 2.54, 'V4': 1.38,
                'V5': -0.34, 'V6': 0.46, 'V7': 0.24, 'V8': 0.10,
                'V9': 0.36, 'V10': 0.09,
            }
        }
    }


class FraudResponse(BaseModel):
    fraud_probability:  float
    is_fraud:           bool
    confidence:         Literal['Low', 'Medium', 'High']
    top_fraud_signals:  list[dict]

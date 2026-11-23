# src/credit_risk/preprocess.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = 'data/raw/data.csv'   # Home Credit dataset
PROCESSED_DIR = 'data/processed'


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop columns with >40% missing values
    threshold = 0.4
    missing_frac = df.isnull().mean()
    cols_to_keep = missing_frac[missing_frac < threshold].index
    df = df[cols_to_keep]
    # Drop ID column
    df = df.drop(columns=['SK_ID_CURR'], errors='ignore')
    # Encode binary categoricals
    for col in df.select_dtypes('object').columns:
        if df[col].nunique() <= 2:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df = df.drop(columns=[col])  # drop high-cardinality categoricals for now
    return df


def build_pipeline(numeric_cols: list) -> ColumnTransformer:
    '''
    sklearn Pipeline ensures: fit on train only, transform applied to test.
    This prevents data leakage — a common beginner mistake.
    '''
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # fill missing with median
        ('scaler', StandardScaler())                    # normalize
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols)
    ])
    return preprocessor


def prepare_data():
    print('Loading data...')
    df = load_and_clean(DATA_PATH)
    target = 'TARGET'
    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = build_pipeline(numeric_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Fit on train, transform both — correct way
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(preprocessor, f'{PROCESSED_DIR}/credit_preprocessor.pkl')
    joblib.dump(
        (X_train_proc, X_test_proc, y_train, y_test, numeric_cols),
        f'{PROCESSED_DIR}/credit_data.pkl'
    )
    print(f'Train: {X_train_proc.shape}, Test: {X_test_proc.shape}')
    print(f'Default rate: {y.mean():.3f}')
    return X_train_proc, X_test_proc, y_train, y_test


if __name__ == '__main__':
    prepare_data()

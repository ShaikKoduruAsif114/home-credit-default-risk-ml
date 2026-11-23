# src/fraud/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

DATA_PATH = 'data/raw/creditcard.csv'
PROCESSED_DIR = 'data/processed'


def prepare_fraud_data():
    print('Loading fraud data...')
    df = pd.read_csv(DATA_PATH)
    print(f'Class distribution:\n{df["Class"].value_counts()}')

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Scale Amount and Time (V1-V28 are already PCA-transformed)
    X = X.copy()
    X['Amount'] = StandardScaler().fit_transform(X[['Amount']])
    X['Time']   = StandardScaler().fit_transform(X[['Time']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE: oversample minority class in TRAINING set only
    # CRITICAL: Never apply SMOTE to test set — only train set
    print('Applying SMOTE oversampling...')
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f'After SMOTE — Train fraud rate: {y_train_res.mean():.3f}')

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(
        (X_train_res, X_test, y_train_res, y_test),
        f'{PROCESSED_DIR}/fraud_data.pkl'
    )
    print(f'Train: {X_train_res.shape}, Test: {X_test.shape}')
    return X_train_res, X_test, y_train_res, y_test


if __name__ == '__main__':
    prepare_fraud_data()

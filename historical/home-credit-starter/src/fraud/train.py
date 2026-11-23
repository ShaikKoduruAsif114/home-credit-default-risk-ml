# src/fraud/train.py
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     classification_report,
                                     ConfusionMatrixDisplay)
import shap
import os

EXPERIMENT_NAME = 'fraud_detection'


def evaluate_model(model, X_test, y_test, model_name):
    '''Compute all metrics. Returns dict.'''
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        'auc_roc':       roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'f1':            float(report.get('1', report.get('1.0', {})).get('f1-score', 0.0)),
        'precision':     float(report.get('1', report.get('1.0', {})).get('precision', 0.0)),
        'recall':        float(report.get('1', report.get('1.0', {})).get('recall', 0.0)),
    }


def save_confusion_matrix(model, X_test, y_test, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}')
    path = f'models/{model_name}_fraud_confusion_matrix.png'
    os.makedirs('models', exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def save_shap_plot(model, X_test, model_name, feature_names):
    '''Generate SHAP summary plot and save as PNG.'''
    explainer = shap.TreeExplainer(model)
    X_sample = X_test[:200] if hasattr(X_test, '__len__') else X_test
    if hasattr(X_sample, 'values'):
        X_arr = X_sample.values
    else:
        X_arr = X_sample
    shap_values = explainer.shap_values(X_arr)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_arr, feature_names=feature_names, show=False)
    path = f'models/{model_name}_fraud_shap_summary.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def train_and_log(model, model_name, params, X_train, X_test,
                  y_train, y_test, feature_names):
    '''Train one model, evaluate, log everything to MLflow.'''
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        cm_path = save_confusion_matrix(model, X_test, y_test, model_name)
        mlflow.log_artifact(cm_path)
        if model_name != 'LogisticRegression':
            shap_path = save_shap_plot(model, X_test, model_name, feature_names)
            mlflow.log_artifact(shap_path)
        mlflow.sklearn.log_model(
            model,
            artifact_path='model',
            registered_model_name=f'fraud_{model_name}'
        )
        print(
            f'{model_name:25s}  AUC-ROC: {metrics["auc_roc"]:.4f}  |  '
            f'AvgPrec: {metrics["avg_precision"]:.4f}'
        )
        return metrics


def main():
    # Load preprocessed data (SMOTE already applied to training set)
    X_train, X_test, y_train, y_test = joblib.load('data/processed/fraud_data.pkl')

    # Get feature names
    if hasattr(X_test, 'columns'):
        feature_names = X_test.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

    # Convert to numpy if needed for sklearn compatibility
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_arr  = X_test.values  if hasattr(X_test,  'values') else X_test

    print(f'Training on {X_train_arr.shape[0]} samples, {X_train_arr.shape[1]} features\n')
    print(f'Fraud rate in training: {y_train.mean():.3f}')
    print(f'Fraud rate in test:     {y_test.mean():.3f}\n')

    # For fraud, scale_pos_weight=1 because SMOTE already balanced training set
    models = [
        (
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'LogisticRegression',
            {'max_iter': 1000, 'solver': 'lbfgs', 'class_weight': 'balanced'}
        ),
        (
            RandomForestClassifier(
                n_estimators=200, max_depth=8,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'RandomForest',
            {'n_estimators': 200, 'max_depth': 8, 'class_weight': 'balanced'}
        ),
        (
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=1,  # SMOTE balanced training — no extra weight needed
                eval_metric='auc', random_state=42, n_jobs=-1
            ),
            'XGBoost',
            {'n_estimators': 300, 'max_depth': 6, 'lr': 0.05, 'scale_pos_weight': 1}
        ),
        (
            LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
            ),
            'LightGBM',
            {'n_estimators': 300, 'max_depth': 6, 'lr': 0.05, 'class_weight': 'balanced'}
        ),
    ]

    print('=== Fraud Detection Model Training ===\n')
    results = {}
    for model, name, params in models:
        metrics = train_and_log(
            model, name, params,
            X_train_arr, X_test_arr, y_train, y_test, feature_names
        )
        results[name] = metrics

    # Print comparison table — use avg_precision as primary metric for imbalanced fraud data
    print('\n=== RESULTS SUMMARY ===')
    print(f'{"Model":25s}  {"AUC-ROC":10s}  {"Avg Precision":15s}  {"F1":8s}')
    print('-' * 65)
    best_model = max(results, key=lambda x: results[x]['avg_precision'])
    for name, m in results.items():
        marker = '  <- BEST' if name == best_model else ''
        print(
            f'{name:25s}  {m["auc_roc"]:.4f}      '
            f'{m["avg_precision"]:.4f}           {m["f1"]:.4f}{marker}'
        )


if __name__ == '__main__':
    main()

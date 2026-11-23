# src/credit_risk/train.py
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
                                     classification_report, confusion_matrix,
                                     ConfusionMatrixDisplay)
import shap
import os

EXPERIMENT_NAME = 'credit_risk_prediction'


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
    path = f'models/{model_name}_confusion_matrix.png'
    os.makedirs('models', exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def save_shap_plot(model, X_test, model_name, feature_names):
    '''Generate SHAP summary plot and save as PNG.'''
    explainer = shap.TreeExplainer(model)
    # Use 200 samples for speed during training
    X_sample = X_test[:200]
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (default) for binary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names, show=False)
    path = f'models/{model_name}_shap_summary.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def train_and_log(model, model_name, params, X_train, X_test,
                  y_train, y_test, feature_names):
    '''Train one model, evaluate, log everything to MLflow.'''
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, model_name)
        # Log params and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Log artifacts
        cm_path = save_confusion_matrix(model, X_test, y_test, model_name)
        mlflow.log_artifact(cm_path)
        if model_name != 'LogisticRegression':
            shap_path = save_shap_plot(model, X_test, model_name, feature_names)
            mlflow.log_artifact(shap_path)
        # Log model to MLflow model registry
        mlflow.sklearn.log_model(
            model,
            artifact_path='model',
            registered_model_name=f'credit_{model_name}'
        )
        print(f'{model_name:25s}  AUC-ROC: {metrics["auc_roc"]:.4f}  |  AvgPrec: {metrics["avg_precision"]:.4f}')
        return metrics


def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names = joblib.load(
        'data/processed/credit_data.pkl'
    )
    print(f'Training on {X_train.shape[0]} samples, {X_train.shape[1]} features\n')

    # Define all 4 models with their params
    models = [
        (
            LogisticRegression(max_iter=1000, random_state=42),
            'LogisticRegression',
            {'max_iter': 1000, 'solver': 'lbfgs'}
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
                scale_pos_weight=10,
                eval_metric='auc', random_state=42, n_jobs=-1
            ),
            'XGBoost',
            {'n_estimators': 300, 'max_depth': 6, 'lr': 0.05, 'scale_pos_weight': 10}
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

    print('=== Credit Risk Model Training ===\n')
    results = {}
    for model, name, params in models:
        metrics = train_and_log(
            model, name, params,
            X_train, X_test, y_train, y_test, feature_names
        )
        results[name] = metrics

    # Print comparison table
    print('\n=== RESULTS SUMMARY ===')
    print(f'{"Model":25s}  {"AUC-ROC":10s}  {"Avg Precision":15s}  {"F1":8s}')
    print('-' * 65)
    best_model = max(results, key=lambda x: results[x]['auc_roc'])
    for name, m in results.items():
        marker = '  <- BEST' if name == best_model else ''
        print(
            f'{name:25s}  {m["auc_roc"]:.4f}      '
            f'{m["avg_precision"]:.4f}           {m["f1"]:.4f}{marker}'
        )


if __name__ == '__main__':
    main()

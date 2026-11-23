# src/fraud/evaluate.py
import shap
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np


def generate_shap_analysis():
    # Load best model from MLflow (LightGBM)
    model = mlflow.sklearn.load_model('models:/fraud_LightGBM/latest')
    X_train, X_test, y_train, y_test = joblib.load('data/processed/fraud_data.pkl')

    # Get feature names
    if hasattr(X_test, 'columns'):
        feature_names = X_test.columns.tolist()
        X_test_arr = X_test.values
    else:
        X_test_arr = X_test
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

    print('Computing SHAP values for fraud model...')
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_arr[:500])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ── Plot 1: Summary Plot (global feature importance) ──────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_arr[:500],
                      feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance — Fraud Detection Model')
    plt.savefig('models/fraud_shap_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: models/fraud_shap_summary.png')

    # ── Plot 2: Waterfall Plot (highest fraud probability prediction) ─────
    y_prob = model.predict_proba(X_test_arr)[:, 1]
    fraud_idx = np.argmax(y_prob)

    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value
    )

    explanation = shap.Explanation(
        values=shap_values[fraud_idx],
        base_values=expected_value,
        data=X_test_arr[fraud_idx],
        feature_names=feature_names
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f'Why is this transaction flagged as fraud? (score={y_prob[fraud_idx]:.3f})')
    plt.savefig('models/fraud_shap_waterfall.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: models/fraud_shap_waterfall.png')

    # ── Top 10 features by mean |SHAP| ───────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features  = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1], reverse=True
    )[:10]
    print('\nTop 10 most important fraud signals:')
    for feat, imp in top_features:
        print(f'  {feat:40s}: {imp:.4f}')


if __name__ == '__main__':
    generate_shap_analysis()

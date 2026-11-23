# src/credit_risk/evaluate.py
import shap
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np


def generate_shap_analysis():
    # Load best model from MLflow (LightGBM)
    model = mlflow.sklearn.load_model('models:/credit_LightGBM/latest')
    X_train, X_test, y_train, y_test, feature_names = joblib.load(
        'data/processed/credit_data.pkl'
    )

    print('Computing SHAP values...')
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])  # 500 samples
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ── Plot 1: Summary Plot (global feature importance) ──────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:500],
                      feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance — Credit Risk Model')
    plt.savefig('models/shap_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: models/shap_summary.png')

    # ── Plot 2: Waterfall Plot (single prediction explanation) ────────────
    # Pick the highest-risk prediction to explain
    y_prob = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(y_prob)

    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value
    )

    explanation = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=expected_value,
        data=X_test[high_risk_idx],
        feature_names=feature_names
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f'Why is this applicant high-risk? (score={y_prob[high_risk_idx]:.3f})')
    plt.savefig('models/shap_waterfall.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: models/shap_waterfall.png')

    # ── Top 10 features by mean |SHAP| ───────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features  = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1], reverse=True
    )[:10]
    print('\nTop 10 most important features:')
    for feat, imp in top_features:
        print(f'  {feat:40s}: {imp:.4f}')


if __name__ == '__main__':
    generate_shap_analysis()

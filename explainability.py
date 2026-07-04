"""
explainability.py -- SHAP values for the tree-based models (Random Forest,
XGBoost), addressing mentor review Comment 12 ("strengthen the predictive
component with explainable AI techniques such as feature importance or SHAP
values").

Design notes:
  - Only Random Forest and XGBoost get SHAP here. shap.TreeExplainer needs a
    tree-based model; Ridge/Lasso/SVM already expose |coefficient|-based
    importance via ml_features.get_fi(), and LSTM/GRU have no direct SHAP
    support in this codebase. This mirrors exactly what Comment 12 asked for
    ("SHAP values specifically for the tree-based models") rather than
    over-claiming SHAP support platform-wide.
  - Degrades gracefully: if the `shap` package isn't installed, every function
    here returns None/empty rather than raising, so the app runs fine without
    it (the existing feature-importance panel is the fallback). This module
    is intentionally the ONLY place that imports `shap`, so the rest of the
    app has zero dependency on it being present.
  - Consumes the `_model_final` / `_X` fields that ml_models.run_sklearn()
    attaches to its (pre-jsonify) return dict for the two tree models --
    these are stripped by forecast_service.strip_internal() before any JSON
    response is built, so this must be called on the RAW result dict.
"""
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[explainability] WARNING: 'shap' package not installed -- SHAP explanations "
          "will be unavailable (the model-derived Feature Weights panel still works "
          "fine without it). Run: pip install shap")

from ml_features import DETREND_FEATURE_NAMES

SHAP_MODELS = {'Random Forest', 'XGBoost'}


def shap_available_for(model_name):
    return SHAP_AVAILABLE and model_name in SHAP_MODELS


def shap_for_result(model_name, raw_result, n_display=4):
    """Given the RAW (un-stripped) result dict from ml_models.run_sklearn() for
       Random Forest or XGBoost, compute per-feature mean |SHAP value| for the
       most recent prediction row, normalized to sum to 1 (same shape/contract
       as ml_features.get_fi()). Returns None if shap isn't installed, the model
       isn't tree-based, or the computation fails for any reason -- callers
       should treat None as "not available" and fall back to feature_importances."""
    if not shap_available_for(model_name):
        return None
    model = raw_result.get('_model_final')
    X = raw_result.get('_X')
    if model is None or X is None or len(X) == 0:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        # Explain the most recent rows (most relevant to "why this forecast"), capped
        # at 10 for speed -- this runs synchronously inside an API request.
        sample = X[-min(10, len(X)):]
        sv = np.array(explainer.shap_values(sample))
        mean_abs = np.mean(np.abs(sv), axis=0)
        total = float(np.sum(mean_abs))
        if total < 1e-9:
            # The model predicts (near-)identically across this whole recent window
            # regardless of input features -- not a SHAP failure, but a real
            # characteristic of heavily-regularized models on a short series (the
            # residual correction has collapsed to ~constant, so there's genuinely
            # nothing for any feature to explain here). Reporting a normalized
            # 0.0000/0.0000/... block would look like a broken computation rather
            # than what it actually is, so this is surfaced as "not available" with
            # an honest reason instead.
            print(f"[explainability] {model_name}: SHAP values are all ~0 -- the model's "
                  f"prediction doesn't vary across the recent window for this series.")
            return None
        names = DETREND_FEATURE_NAMES
        return {names[i]: round(float(mean_abs[i] / total), 4)
                for i in range(min(len(mean_abs), n_display, len(names)))}
    except Exception as e:
        print(f"[explainability] SHAP computation failed for {model_name}: {e}")
        return None

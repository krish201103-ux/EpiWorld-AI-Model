"""
forecast_service.py -- Orchestration layer between Flask routes and the ML models.

Extracted from server.py as part of the modularization requested in mentor
review Comments 1, 3, and 13. This is where series-selection, zero-trimming,
model-dispatch, and best-model-scoring logic lives -- previously duplicated
almost verbatim between the /api/predict and /api/all_models route handlers
in server.py.

Bug fixed by this consolidation: the old /api/all_models had its own,
independently-drifted copy of the series-selection logic that never checked
the `metric` parameter at all, so selecting "Deaths" in the Forecast Settings
dropdown and clicking "Run All Models" silently kept using case data (Run
Forecast, hitting /api/predict, worked correctly). Both endpoints now share
select_series() below, so this inconsistency is gone.
"""
import numpy as np

import config
from ml_models import run_sklearn, run_rnn

ALL_MODEL_NAMES = ['Random Forest', 'Ridge', 'Lasso', 'SVM', 'XGBoost', 'LSTM', 'GRU']
UPLOAD_MODEL_NAMES = ['Random Forest', 'Ridge', 'Lasso', 'XGBoost', 'LSTM']


class SeriesError(ValueError):
    """Raised when a disease/metric combination has no usable series (too few
       points, or every value is zero). Message is safe to show to the user."""


def select_series(disease_entry, metric='auto'):
    """Pick which series (cases/deaths/immunization) to use, then trim leading
       zeros (some diseases have years of all-zero data before reporting began).
       Returns (y: np.ndarray, years: list[str], label: str). Raises SeriesError
       with a user-facing message if nothing usable is found."""
    d = disease_entry
    if metric == 'deaths' and any(v > 0 for v in d['death_series']):
        raw, lbl = d['death_series'], 'deaths'
    elif metric == 'immunization' and any(v > 0 for v in d.get('immunization_series', [])):
        raw, lbl = d['immunization_series'], 'immunization'
    elif any(v > 0 for v in d['case_series']):
        raw, lbl = d['case_series'], 'cases'
    elif any(v > 0 for v in d['death_series']):
        raw, lbl = d['death_series'], 'deaths'
    else:
        raise SeriesError('No usable data series found')

    raw = np.array(raw, dtype=float)
    s = 0
    while s < len(raw) - 2 and raw[s] == 0:
        s += 1
    y = raw[s:]
    years = d['years'][s:]
    if len(y) < 3:
        raise SeriesError(f'Only {len(y)} data points after trimming zeros, need at least 3')
    return y, years, lbl


def run_one_model(model_name, y, horizon, n_lags):
    """Dispatch to the sklearn or RNN runner based on model family. Returns None
       if the model isn't recognized or fails to fit (e.g. too little data)."""
    if model_name in ('LSTM', 'GRU'):
        return run_rnn(model_name, y, horizon, n_lags)
    return run_sklearn(model_name, y, horizon, n_lags)


def score_result(result):
    """Ranking key for auto-selecting the best model: prefer walk-forward test R2;
       fall back to a discounted training R2 only when a series is too short to
       hold out a genuine test fold (test_metrics.r2 will be None/<=0 in that case)."""
    te = result['test_metrics'].get('r2')
    tr = result['train_metrics'].get('r2', 0) or 0
    return te if (te is not None and te > 0) else tr


def strip_internal(result):
    """Remove leading-underscore keys (e.g. _model_final, _X) that ml_models attaches
       for internal use by the SHAP explainability module -- these hold live sklearn
       model objects / raw numpy arrays and are NOT JSON-serializable, so they must
       never reach jsonify(). Routes call this before building any HTTP response;
       explainability.py reads the un-stripped dict directly instead."""
    return {k: v for k, v in result.items() if not k.startswith('_')}


def _horizon_and_years(years, horizon):
    horizon = max(1, min(config.MAX_HORIZON, int(horizon)))
    last_yr = int(years[-1]) if len(years) else 2024
    fc_years = [str(last_yr + i + 1) for i in range(horizon)]
    return horizon, fc_years


def predict_one(disease_data, disease, model_name, metric='auto', horizon=5):
    """Full single-model prediction pipeline: series selection -> model dispatch ->
       response dict (JSON-safe). Raises KeyError if the disease is unknown, or
       SeriesError if there's no usable data. Returns None if the specific model
       failed to fit (caller should treat that as a 400, not a 500)."""
    if disease not in disease_data:
        raise KeyError(disease)
    y, years, lbl = select_series(disease_data[disease], metric)
    n = len(y)
    n_lags = min(3, max(1, n - 2))
    horizon, fc_years = _horizon_and_years(years, horizon)

    result = run_one_model(model_name, y, horizon, n_lags)
    if not result:
        return None
    result = strip_internal(result)
    return {
        **result,
        'disease': disease, 'model': model_name, 'metric': lbl,
        'years': list(years), 'values': [float(v) for v in y.tolist()],
        'fc_years': fc_years, 'n_samples': n,
        'too_small': n < 8, 'n_lags': n_lags,
    }


def predict_all(disease_data, disease, metric='auto', horizon=5, model_names=None):
    """Run every model (default: all 7) for one disease and auto-select the best.
       Returns (best_model_name, {model_name: response_dict}) for models that
       succeeded -- models that errored are simply omitted (matches prior behavior).
       Raises KeyError if the disease is unknown, SeriesError if there's no usable data."""
    if disease not in disease_data:
        raise KeyError(disease)
    y, years, lbl = select_series(disease_data[disease], metric)
    n = len(y)
    n_lags = min(3, max(1, n - 2))
    horizon, fc_years = _horizon_and_years(years, horizon)

    names = model_names or ALL_MODEL_NAMES
    good = {}
    for mn in names:
        try:
            r = run_one_model(mn, y, horizon, n_lags)
            if r:
                r = strip_internal(r)
                good[mn] = {**r, 'disease': disease, 'model': mn, 'metric': lbl,
                            'years': list(years), 'values': [float(v) for v in y.tolist()],
                            'fc_years': fc_years, 'n_samples': n,
                            'too_small': n < 8, 'n_lags': n_lags}
        except Exception:
            continue  # one model failing (e.g. too little data for sklearn) shouldn't
                       # take down the whole comparison -- matches prior behavior

    if not good:
        return None, {}
    best = max(good, key=lambda m: score_result(good[m]))
    return best, good

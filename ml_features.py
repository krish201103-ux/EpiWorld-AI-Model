"""
ml_features.py -- Feature engineering, walk-forward validation, and metrics.

Extracted from server.py as part of the modularization requested in mentor
review Comments 1, 3, and 13. This module has no Flask dependency and no
knowledge of HTTP -- it is pure numpy/sklearn feature-engineering and
evaluation logic, which is what makes it possible to unit-test or reuse
(e.g. from generate_evaluation_report.py) without spinning up the web app.

Two parallel feature strategies live here, matching the two model families:
  - make_features() / recursive_fc(): raw hybrid features (trend + lags),
    used directly by Ridge.
  - detrend_features() / fit_trend(): trend-removed residual features,
    used by the tree-based / kernel models (Random Forest, XGBoost, Lasso,
    SVM) which cannot extrapolate a trend on their own.
"""
import numpy as np

FEATURE_NAMES = ['Time Trend', 'Time Trend^2', 'Lag-1', 'Lag-2', 'Lag-3', '2-period MA', 'Local Momentum']
# Used for detrended models (Random Forest, XGBoost, Lasso, SVM) where the trend terms
# have already been subtracted out before the model ever sees the data -- the model only
# ever looks at lag/MA deviations from that trend, so "Time Trend" labels would be wrong.
DETREND_FEATURE_NAMES = ['Lag-1 (detrended)', 'Lag-2 (detrended)', 'Lag-3 (detrended)', '2-period MA (detrended)']

DETREND_MODELS = {'Random Forest', 'XGBoost', 'Lasso', 'SVM'}


def sf(v):
    """Safe-float: unwrap numpy scalars, default to 0.0 on failure."""
    try:
        return float(v.item()) if hasattr(v, 'item') else float(v)
    except Exception:
        return 0.0


def compute_metrics(yt, yp):
    """RMSE / MAE / R2 (clamped to [-1,1]) / MAPE / AIC for a set of actual vs
       predicted values. Returns None for any metric that can't be computed
       (e.g. R2 needs at least 2 points)."""
    yt = np.array(yt, dtype=float)
    yp = np.array(yp, dtype=float)
    n = len(yt)
    if n < 2:
        return dict(rmse=None, mae=None, r2=None, mape=None, aic=None)
    mu = np.mean(yt)
    res = yt - yp
    ssr = float(np.sum(res ** 2))
    sst = float(np.sum((yt - mu) ** 2)) or 1e-10
    rmse = float(np.sqrt(ssr / n))
    mae = float(np.mean(np.abs(res)))
    # R2: clamp to [-1, 1] -- never return >1 or nonsense negatives
    r2 = float(np.clip(1 - ssr / sst, -1.0, 1.0))
    mask = np.abs(yt) > 1.0
    mape = float(np.mean(np.abs(res[mask] / yt[mask])) * 100) if mask.sum() > 0 else None
    aic = int(n * np.log(max(ssr / n, 1e-12)) + 2 * 6)
    return dict(
        rmse=round(rmse, 2), mae=round(mae, 2), r2=round(r2, 4),
        mape=round(mape, 2) if mape is not None else None, aic=aic
    )


def make_features(y, n_lags=3):
    """Hybrid feature set: global trend (t, t^2) + lag values + 2-period MA + local momentum
       (1-step slope). Used for trend-aware linear models (Ridge) which can read
       the trend terms directly. Tree-based models (Random Forest/XGBoost) instead use
       detrend_features() below, since trees cannot extrapolate past values seen in
       training -- they need the trend removed first and added back after."""
    y = np.array(y, dtype=float)
    n = len(y)
    X, Y = [], []
    for i in range(n_lags, n):
        t = i / max(n - 1, 1)
        feat = [t, t * t]
        feat += [float(y[i - l]) for l in range(1, n_lags + 1)]
        feat.append(float(np.mean(y[max(0, i - 2):i])))           # 2-period MA
        slope = float(y[i - 1] - y[i - 2]) if i >= 2 else 0.0      # local momentum
        feat.append(slope)
        X.append(feat)
        Y.append(float(y[i]))
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def fit_trend(y_work, up_to=None):
    """Fit a quadratic trend (on whatever scale y_work is in -- already log-transformed
       upstream if needed) using only the first `up_to` points (or all points if None).
       Returns the polyfit coefficients; evaluate with np.polyval(coef, index)."""
    n = up_to if up_to is not None else len(y_work)
    n = max(n, 3)
    idx = np.arange(n)
    return np.polyfit(idx, y_work[:n], deg=2)


def detrend_features(y_work, idx, trend_coef, n_lags=3):
    """Tree-model feature vector: lag values and 2-period MA expressed as DEVIATIONS from
       the fitted trend at each respective time index, rather than raw levels. This is
       what lets Random Forest / XGBoost contribute a meaningful nonlinear correction
       instead of failing to extrapolate on a rising or falling trend."""
    def trend_at(i):
        return float(np.polyval(trend_coef, i))
    feat = [float(y_work[idx - l] - trend_at(idx - l)) for l in range(1, n_lags + 1)]
    lag_window = y_work[max(0, idx - 2):idx]
    if len(lag_window) > 0:
        feat.append(float(np.mean(lag_window) - trend_at(idx - 0.5)))
    else:
        feat.append(0.0)
    return feat


def recursive_fc(model, history, n_lags, horizon, n, trend_coef=None):
    """Recursive multi-step forecast. If trend_coef is provided, the model predicts the
       residual around that trend (tree-based models); otherwise it predicts the raw
       hybrid-feature target directly (linear models). Bounded to prevent runaway."""
    hist = [float(v) for v in history]
    # Generous clip directly in whatever "working" space `history` is already in --
    # it may already be log1p-transformed upstream (run_sklearn passes y_work, not
    # raw y, whenever use_log is True). Applying a SECOND log1p to a bound derived
    # from that already-transformed history was a real bug found during testing: for
    # a typical log-scale epidemiological series (working values around 15-20), it
    # collapsed the ceiling to roughly log1p(20*5)+5 =~ 9.6, silently clipping every
    # real forecast down to the same wrong constant and producing a flat, wildly-too-low
    # multi-step forecast after the final expm1() unwrap. The bound below stays in
    # `pred`'s own units, matching `history`.
    work_max = float(np.max(np.abs(history))) if len(history) > 0 else 1e6
    clip_hi = work_max * 3 + 10
    preds = []
    for step in range(horizon):
        idx = n + step
        if trend_coef is not None:
            feat = detrend_features(np.array(hist), idx, trend_coef, n_lags)
            trend_val = float(np.polyval(trend_coef, idx))
            resid_pred = sf(model.predict([feat])[0])
            pred = trend_val + resid_pred
        else:
            # Normalized the same way as make_features() -- t = idx/(n-1), NOT
            # idx/(n+horizon-1). A horizon-dependent denominator here was a bug found
            # during testing: it capped t at ~1.0 only on the LAST forecast step,
            # meaning every step before that saw a t value the model had already seen
            # during training, so the learned trend coefficient never actually
            # extrapolated forward.
            t = idx / max(n - 1, 1)
            lag_hist = hist[-(n_lags + 2):]
            feat = [t, t * t]
            feat += [lag_hist[-l] if l <= len(lag_hist) else lag_hist[0] for l in range(1, n_lags + 1)]
            feat.append(float(np.mean(lag_hist[-2:] if len(lag_hist) >= 2 else lag_hist)))
            slope = (lag_hist[-1] - lag_hist[-2]) if len(lag_hist) >= 2 else 0.0
            feat.append(float(slope))
            pred = sf(model.predict([feat])[0])
        pred = float(np.clip(pred, -50, clip_hi))  # generous clip, same units as pred
        preds.append(pred)
        hist.append(pred)
    return preds


def bootstrap_ci(cls, params, X, y, n_lags, horizon, n, B=30):
    """Bootstrap CI for LINEAR/kernel models (Ridge) -- returns (upper, lower)
       or (None, None). Tree-based models use their own inline bootstrap in
       ml_models.run_sklearn since they need the trend-aware recursive forecast,
       not the plain hybrid-feature one."""
    np.random.seed(42)
    boot = []
    for _ in range(B):
        idx = np.random.choice(len(X), len(X), replace=True)
        try:
            m = cls(**params)
            m.fit(X[idx], y[idx])
            boot.append(recursive_fc(m, list(y), n_lags, horizon, n))
        except Exception:
            pass
    if len(boot) < 5:
        return None, None
    arr = np.array(boot)
    return (
        [max(0.0, float(v)) for v in np.percentile(arr, 97.5, axis=0)],
        [max(0.0, float(v)) for v in np.percentile(arr, 2.5, axis=0)],
    )


def get_fi(model, n_feats, names=None):
    """Extract feature importances or |coef| from a fitted model, normalized to sum to 1."""
    names = names if names is not None else FEATURE_NAMES
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        total = sum(fi) or 1
        return {names[i]: round(float(fi[i] / total), 4) for i in range(min(len(fi), n_feats, len(names)))}
    if hasattr(model, 'named_steps'):
        m = model.named_steps.get('m')
        if m and hasattr(m, 'coef_'):
            coef = np.abs(m.coef_.flatten())
            total = sum(coef) or 1
            return {names[i]: round(float(coef[i] / total), 4) for i in range(min(len(coef), n_feats, len(names)))}
    return {}


def walk_forward_r2_detrended(y_work, model_cls, model_params, n_lags, use_log, n_folds=6):
    """Expanding-window walk-forward validation for TREE/KERNEL models, which use the
       detrend-then-residual architecture: at each fold, fit a trend on data up to that
       point, train the model to predict the next point's residual, and add the trend back.
       This detrend-then-residual architecture is what allows these models to reach ~0.9+
       walk-forward test R^2 on monotonic/U-shaped epidemiological series, where they would
       otherwise systematically fail to extrapolate (trees can't predict beyond trained leaf
       ranges; Lasso/SVM struggle to jointly fit both trend and short-run wiggle from raw
       levels)."""
    n = len(y_work)
    fold_start = max(n_lags + 2, n - n_folds)
    actuals, preds = [], []
    for k in range(fold_start, n):
        trend_coef = fit_trend(y_work, up_to=k)
        X_tr, Y_tr = [], []
        for i in range(n_lags, k):
            X_tr.append(detrend_features(y_work, i, trend_coef, n_lags))
            Y_tr.append(float(y_work[i] - np.polyval(trend_coef, i)))
        X_tr, Y_tr = np.array(X_tr), np.array(Y_tr)
        if len(X_tr) < 4:
            continue
        try:
            m = model_cls(**model_params)
            m.fit(X_tr, Y_tr)
            feat_k = detrend_features(y_work, k, trend_coef, n_lags)
            resid_pred = sf(m.predict([feat_k])[0])
        except Exception:
            continue
        pred = np.polyval(trend_coef, k) + resid_pred
        actual = y_work[k]
        if use_log:
            pred = float(np.expm1(max(0, pred)))
            actual = float(np.expm1(max(0, actual)))
        actuals.append(actual)
        preds.append(pred)
    if len(actuals) < 3:
        return None, 0
    met = compute_metrics(actuals, preds)
    return met['r2'], len(actuals)


def walk_forward_r2(y_work, model_cls, model_params, n_lags, use_log, n_folds=6):
    """Expanding-window walk-forward validation for RIDGE, which uses the raw hybrid
       feature set directly (no detrending needed since the trend terms t/t^2 are already
       explicit features the linear model can weight). Repeatedly trains on data up to
       fold k and predicts y[k], sliding forward one step at a time -- far more stable than
       a single 80/20 split on a 24-35 point series.
       Returns (test_r2, n_folds_used) on the ORIGINAL (non-log) scale."""
    n = len(y_work)
    fold_start = max(n_lags + 2, n - n_folds)
    actuals, preds = [], []
    for k in range(fold_start, n):
        X_tr, Y_tr = make_features(y_work[:k], n_lags)
        if len(X_tr) < 4:
            continue
        X_te, _ = make_features(y_work[:k + 1], n_lags)
        x_last = X_te[-1:].copy()
        try:
            m = model_cls(**model_params)
            m.fit(X_tr, Y_tr)
            pred = sf(m.predict(x_last)[0])
        except Exception:
            continue
        actual = y_work[k]
        if use_log:
            pred = float(np.expm1(max(0, pred)))
            actual = float(np.expm1(max(0, actual)))
        actuals.append(actual)
        preds.append(pred)
    if len(actuals) < 3:
        return None, 0
    met = compute_metrics(actuals, preds)
    return met['r2'], len(actuals)

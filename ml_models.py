"""
ml_models.py -- Trains each model family and produces a forecast.

Extracted from server.py as part of the modularization requested in mentor
review Comments 1, 3, and 13. Two entry points:
  - run_sklearn(model_name, y, horizon, n_lags): Random Forest, Ridge, Lasso,
    SVM, XGBoost.
  - run_rnn(model_type, y, horizon, n_lags): LSTM, GRU.

Both return the same shaped dict (fc/up/lo/fitted/resid/train_metrics/
test_metrics/feature_importances/ci_method/...) so forecast_service.py can
treat every model uniformly regardless of family.
"""
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from ml_features import (
    sf, compute_metrics, make_features, fit_trend, detrend_features,
    recursive_fc, bootstrap_ci, get_fi, walk_forward_r2_detrended, walk_forward_r2,
    FEATURE_NAMES, DETREND_FEATURE_NAMES, DETREND_MODELS,
)


# -- SKLEARN RUNNER ---------------------------------------------
def run_sklearn(model_name, y, horizon, n_lags):
    """Train one sklearn-family model and produce a forecast. Random Forest, Lasso,
       SVM, and XGBoost use the detrend-then-residual architecture (is_tree branch);
       Ridge uses the raw hybrid feature set directly since it can read trend terms."""
    y = np.array(y, dtype=float)
    n = len(y)

    # Log-transform for large epidemiological counts (max > 1000)
    use_log = float(np.max(np.abs(y))) > 1000
    y_work  = np.log1p(y) if use_log else y

    n_est = min(300, max(50, n * 10))
    md    = min(4, max(2, n // 5))

    cfgs = {
        'Random Forest': (RandomForestRegressor,
            dict(n_estimators=min(200, n_est), max_depth=md, min_samples_leaf=1,
                 max_features='sqrt', random_state=42, n_jobs=-1)),
        'Ridge': (lambda **k: Pipeline([('sc', StandardScaler()), ('m', Ridge(**k))]),
            dict(alpha=0.3)),
        'Lasso': (lambda **k: Pipeline([('sc', StandardScaler()), ('m', Lasso(**k))]),
            dict(alpha=0.005, max_iter=20000)),
        'SVM':   (lambda **k: Pipeline([('sc', StandardScaler()), ('m', SVR(**k))]),
            dict(kernel='rbf', C=50, epsilon=0.01, gamma='scale')),
        'XGBoost': (XGBRegressor,
            dict(n_estimators=min(150, n_est), max_depth=min(3, max(2, n//6)),
                 learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
                 reg_alpha=0.05, reg_lambda=0.5, min_child_weight=1,
                 random_state=42, verbosity=0)),
    }
    if model_name not in cfgs:
        return None

    cls, params = cfgs[model_name]
    is_tree = model_name in DETREND_MODELS

    if is_tree:
        # -- Detrend-then-residual architecture for tree-based models --
        trend_coef_full = fit_trend(y_work)  # trend fit on the FULL series for final forecasting

        X, Y = [], []
        for i in range(n_lags, n):
            X.append(detrend_features(y_work, i, trend_coef_full, n_lags))
            Y.append(float(y_work[i] - np.polyval(trend_coef_full, i)))
        X, Y = np.array(X), np.array(Y)
        if len(X) < 3:
            return None

        split_ratio = 0.7 if len(X) <= 10 else 0.8
        split = max(1, min(len(X)-1, int(len(X) * split_ratio)))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = Y[:split], Y[split:]

        model = cls(**params)
        model.fit(X_tr, y_tr)

        # The production model that actually generates the fitted curve, forecast, and
        # feature importances is refit on the FULL series (not just X_tr) -- `model`
        # above is kept only for the honest held-out test-metric fallback below. Using
        # a partially-trained model to forecast forward was fine for these detrended
        # residual features (roughly stationary across time) but is still strictly less
        # accurate than using all available history for the model users actually see.
        model_final = cls(**params)
        model_final.fit(X, Y)

        # Fitted values: trend + predicted residual, for every point with enough lag history
        fitted_resid = [sf(v) for v in model_final.predict(X)]
        fitted_work = [np.polyval(trend_coef_full, i) + r for i, r in zip(range(n_lags, n), fitted_resid)]
        if use_log:
            fitted_real = [float(np.expm1(max(0, v))) for v in fitted_work]
        else:
            fitted_real = [max(0.0, float(v)) for v in fitted_work]
        full_fitted = (list(y[:n_lags]) + fitted_real)[:n]

        # Forecast: extend trend forward, predict residual at each future step
        fc_work = []
        hist_work = y_work.tolist()
        for step in range(horizon):
            idx = n + step
            feat = detrend_features(np.array(hist_work), idx, trend_coef_full, n_lags)
            resid_pred = sf(model_final.predict([feat])[0])
            pred_work = float(np.polyval(trend_coef_full, idx) + resid_pred)
            fc_work.append(pred_work)
            hist_work.append(pred_work)
        if use_log:
            fc = [max(0.0, round(float(np.expm1(v)), 2)) for v in fc_work]
        else:
            fc = [max(0.0, round(float(v), 2)) for v in fc_work]

        tr_met = compute_metrics(y.tolist(), full_fitted)

        wf_r2, wf_n = walk_forward_r2_detrended(y_work, cls, params, n_lags, use_log, n_folds=6)
        if wf_r2 is not None:
            if len(X_te) > 0:
                te_pred_work = [np.polyval(trend_coef_full, i) + sf(v)
                                 for i, v in zip(range(split + n_lags, n), model.predict(X_te))]
                if use_log:
                    te_pred = [float(np.expm1(max(0, v))) for v in te_pred_work]
                    te_act  = [float(np.expm1(max(0, y_work[i]))) for i in range(split + n_lags, n)]
                else:
                    te_pred = [max(0.0, float(v)) for v in te_pred_work]
                    te_act  = [float(y_work[i]) for i in range(split + n_lags, n)]
                te_met = compute_metrics(te_act, te_pred)
            else:
                te_met = {k: None for k in ['rmse','mae','r2','mape','aic']}
            te_met['r2'] = wf_r2
        elif len(X_te) > 0:
            te_pred_work = [np.polyval(trend_coef_full, i) + sf(v)
                             for i, v in zip(range(split + n_lags, n), model.predict(X_te))]
            if use_log:
                te_pred = [float(np.expm1(max(0, v))) for v in te_pred_work]
                te_act  = [float(np.expm1(max(0, y_work[i]))) for i in range(split + n_lags, n)]
            else:
                te_pred = [max(0.0, float(v)) for v in te_pred_work]
                te_act  = [float(y_work[i]) for i in range(split + n_lags, n)]
            te_met = compute_metrics(te_act, te_pred)
        else:
            te_met = {k: None for k in ['rmse','mae','r2','mape','aic']}

        resid = (y - np.array(full_fitted, dtype=float)).tolist()

        # Bootstrap CI around the trend+residual forecast (resampled from the full
        # series, matching model_final -- not the train-only slice)
        boot = []
        np.random.seed(42)
        for _ in range(30):
            bidx = np.random.choice(len(X), len(X), replace=True)
            try:
                m = cls(**params)
                m.fit(X[bidx], Y[bidx])
                hist_b = y_work.tolist()
                fc_b = []
                for step in range(horizon):
                    idx = n + step
                    feat = detrend_features(np.array(hist_b), idx, trend_coef_full, n_lags)
                    rp = sf(m.predict([feat])[0])
                    pv = float(np.polyval(trend_coef_full, idx) + rp)
                    fc_b.append(pv)
                    hist_b.append(pv)
                boot.append(fc_b)
            except Exception:
                pass
        if len(boot) >= 5:
            arr = np.array(boot)
            up_b = np.percentile(arr, 97.5, axis=0)
            lo_b = np.percentile(arr, 2.5, axis=0)
            if use_log:
                up = [max(0.0, round(float(np.expm1(max(0, v))), 2)) for v in up_b]
                lo = [max(0.0, round(float(np.expm1(max(0, v))), 2)) for v in lo_b]
            else:
                up = [max(0.0, round(float(v), 2)) for v in up_b]
                lo = [max(0.0, round(float(v), 2)) for v in lo_b]
            ci = 'bootstrap'
        else:
            sd = max(float(np.std(resid)), float(np.mean(np.abs(y))) * 0.05)
            up = [max(0.0, fc[i] + 1.96 * sd * np.sqrt(i+1)) for i in range(horizon)]
            lo = [max(0.0, fc[i] - 1.96 * sd * np.sqrt(i+1)) for i in range(horizon)]
            ci = 'residual_sd'

        def fmt_met(met):
            return {k: (round(float(v), 4) if v is not None and k != 'aic' else (int(v) if v is not None else None))
                    for k, v in met.items()}

        return dict(
            fc=fc, up=[round(float(v),2) for v in up], lo=[round(float(v),2) for v in lo],
            fitted=[round(float(v),2) for v in full_fitted],
            resid=[round(float(r),2) for r in resid],
            train_metrics=fmt_met(tr_met),
            test_metrics=fmt_met(te_met),
            feature_importances=get_fi(model_final, n_lags + 1, names=DETREND_FEATURE_NAMES),
            ci_method=ci,
            validation_method='walk_forward' if wf_r2 is not None else 'holdout_split',
            validation_folds=wf_n if wf_r2 is not None else len(X_te),
            _model_final=model_final, _X=X,  # internal use only -- consumed by
                                              # explainability.shap_for_result(), stripped
                                              # before any JSON response is built.
        )

    # -- Linear model (Ridge): use the hybrid feature set directly --
    X, Y = make_features(y_work, n_lags)
    if len(X) < 3:
        return None

    split_ratio = 0.7 if len(X) <= 10 else 0.8
    split = max(1, min(len(X)-1, int(len(X) * split_ratio)))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = Y[:split], Y[split:]

    model = cls(**params)
    model.fit(X_tr, y_tr)

    # `model` above (train-only) is kept for the honest held-out test-metric fallback
    # further down. model_final, fit on the FULL series, is what actually generates
    # the fitted curve / forecast / feature importances shown to the user -- see the
    # matching comment in the tree branch above for why this matters more here: Ridge
    # uses raw-scale features (not detrended), so a model calibrated on an earlier,
    # smaller-magnitude chunk of a growing series extrapolates very poorly.
    model_final = cls(**params)
    model_final.fit(X, Y)

    all_X, _ = make_features(y_work, n_lags)
    fitted_work = [sf(v) for v in model_final.predict(all_X)]
    if use_log:
        fitted_real = [float(np.expm1(max(0, v))) for v in fitted_work]
    else:
        fitted_real = [max(0.0, float(v)) for v in fitted_work]
    full_fitted = (list(y[:n_lags]) + fitted_real)[:n]

    fc_work = recursive_fc(model_final, y_work.tolist(), n_lags, horizon, n)
    if use_log:
        fc = [max(0.0, round(float(np.expm1(v)), 2)) for v in fc_work]
    else:
        fc = [max(0.0, round(float(v), 2)) for v in fc_work]

    tr_met = compute_metrics(y.tolist(), full_fitted)

    wf_r2, wf_n = walk_forward_r2(y_work, cls, params, n_lags, use_log, n_folds=6)
    if wf_r2 is not None:
        te_pred_work = [sf(v) for v in model.predict(X_te)] if len(X_te) > 0 else []
        if use_log and te_pred_work:
            te_pred = [float(np.expm1(max(0, v))) for v in te_pred_work]
            te_act  = [float(np.expm1(max(0, v))) for v in y_te]
        else:
            te_pred = [max(0.0, float(v)) for v in te_pred_work]
            te_act  = list(y_te)
        te_met = compute_metrics(te_act, te_pred) if te_act else {k: None for k in ['rmse','mae','r2','mape','aic']}
        te_met['r2'] = wf_r2
    elif len(X_te) > 0:
        te_pred_work = [sf(v) for v in model.predict(X_te)]
        if use_log:
            te_pred = [float(np.expm1(max(0, v))) for v in te_pred_work]
            te_act  = [float(np.expm1(max(0, v))) for v in y_te]
        else:
            te_pred = [max(0.0, float(v)) for v in te_pred_work]
            te_act  = list(y_te)
        te_met = compute_metrics(te_act, te_pred)
    else:
        te_met = {k: None for k in ['rmse','mae','r2','mape','aic']}

    resid = (y - np.array(full_fitted, dtype=float)).tolist()

    up_b, lo_b = bootstrap_ci(cls, params, X, Y, n_lags, horizon, n)
    if up_b is not None:
        if use_log:
            up = [max(0.0, round(float(np.expm1(max(0, v))), 2)) for v in up_b]
            lo = [max(0.0, round(float(np.expm1(max(0, v))), 2)) for v in lo_b]
        else:
            up = [max(0.0, round(float(v), 2)) for v in up_b]
            lo = [max(0.0, round(float(v), 2)) for v in lo_b]
        ci = 'bootstrap'
    else:
        sd = max(float(np.std(resid)), float(np.mean(np.abs(y))) * 0.05)
        up = [max(0.0, fc[i] + 1.96 * sd * np.sqrt(i+1)) for i in range(horizon)]
        lo = [max(0.0, fc[i] - 1.96 * sd * np.sqrt(i+1)) for i in range(horizon)]
        ci = 'residual_sd'

    def fmt_met(met):
        return {k: (round(float(v), 4) if v is not None and k != 'aic' else (int(v) if v is not None else None))
                for k, v in met.items()}

    return dict(
        fc=fc, up=[round(float(v),2) for v in up], lo=[round(float(v),2) for v in lo],
        fitted=[round(float(v),2) for v in full_fitted],
        resid=[round(float(r),2) for r in resid],
        train_metrics=fmt_met(tr_met),
        test_metrics=fmt_met(te_met),
        feature_importances=get_fi(model_final, n_lags + 4),
        ci_method=ci,
        validation_method='walk_forward' if wf_r2 is not None else 'holdout_split',
        validation_folds=wf_n if wf_r2 is not None else len(X_te),
    )


# -- RNN RUNNER -------------------------------------------------
def run_rnn(model_type, y, horizon, n_lags):
    """Train an LSTM or GRU (2-layer, 64->32 units, Dropout 0.2) on the raw
       min-max-scaled sequence and produce a forecast with Monte Carlo Dropout CI."""
    y  = np.array(y, dtype=float)
    n  = len(y)
    if n < 5:
        return None

    scaler = MinMaxScaler(feature_range=(0.05, 0.95))
    yn = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    seq_len = min(n_lags, max(1, n - 2))
    X, Y = [], []
    for i in range(seq_len, len(yn)):
        X.append(yn[i-seq_len:i])
        Y.append(yn[i])
    X = np.array(X).reshape(-1, seq_len, 1)
    Y = np.array(Y)

    split = max(1, min(len(X)-1, int(len(X) * 0.8)))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = Y[:split], Y[split:]

    tf.random.set_seed(42)
    rnn_layer = LSTM if model_type == 'LSTM' else GRU
    model = Sequential([
        rnn_layer(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        rnn_layer(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='huber')
    es = EarlyStopping(patience=8, restore_best_weights=True, verbose=0, monitor='val_loss')
    val_data = (X_te, y_te) if len(X_te) > 0 else None
    model.fit(X_tr, y_tr, epochs=100, batch_size=min(8, len(X_tr)),
              validation_data=val_data, callbacks=[es], verbose=0)

    # Fitted values
    fitted_n     = model.predict(X, verbose=0).flatten()
    fitted_norm  = list(yn[:seq_len]) + list(fitted_n)
    fitted_norm  = fitted_norm[:n]
    fitted_real  = scaler.inverse_transform(
        np.array(fitted_norm).reshape(-1, 1)).flatten().tolist()

    tr_met = compute_metrics(y.tolist(), fitted_real)
    if len(X_te) > 0:
        te_pred = scaler.inverse_transform(model.predict(X_te, verbose=0)).flatten()
        te_act  = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        te_met  = compute_metrics(te_act.tolist(), te_pred.tolist())
    else:
        te_met = {k: None for k in ['rmse','mae','r2','mape','aic']}

    # Forecast
    last_seq = yn[-seq_len:].tolist()
    fc_norm  = []
    for _ in range(horizon):
        inp  = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
        pred = float(model.predict(inp, verbose=0).flatten()[0])
        fc_norm.append(pred)
        last_seq.append(pred)
    fc_real = scaler.inverse_transform(np.array(fc_norm).reshape(-1, 1)).flatten()
    fc = [max(0.0, round(float(v), 2)) for v in fc_real]

    # Monte Carlo Dropout CI
    preds_mc = []
    for _ in range(20):
        ls2 = yn[-seq_len:].tolist()
        ps  = []
        for _ in range(horizon):
            inp2 = np.array(ls2[-seq_len:]).reshape(1, seq_len, 1)
            p    = float(model(inp2, training=True).numpy().flatten()[0])
            ps.append(p)
            ls2.append(p)
        preds_mc.append(
            scaler.inverse_transform(np.array(ps).reshape(-1, 1)).flatten().tolist())
    arr = np.array(preds_mc)
    up  = [max(0.0, round(float(v), 2)) for v in np.percentile(arr, 97.5, axis=0)]
    lo  = [max(0.0, round(float(v), 2)) for v in np.percentile(arr,  2.5, axis=0)]

    resid = (y - np.array(fitted_real, dtype=float)).tolist()

    def fmt_met(met):
        return {k: (round(float(v), 4) if v is not None and k != 'aic' else (int(v) if v is not None else None))
                for k, v in met.items()}

    return dict(
        fc=fc, up=up, lo=lo,
        fitted=[round(float(v), 2) for v in fitted_real],
        resid=[round(float(r), 2) for r in resid],
        train_metrics=fmt_met(tr_met),
        test_metrics=fmt_met(te_met),
        feature_importances={},
        ci_method='mc_dropout',
    )

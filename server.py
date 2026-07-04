#!/usr/bin/env python3
"""
Epiworld Platform - Flask Server
Live sklearn + RNN/LSTM predictions | EpiGem AI Chatbot | File Upload
Run: python server.py  ->  http://127.0.0.1:5000

MODULE MAP (see README.md for the full picture):
  config.py            Environment-driven settings (host/port/API keys)
  data_loader.py        Loads disease_data.json
  ml_features.py        Feature engineering, walk-forward validation, metrics
  ml_models.py          run_sklearn() / run_rnn() -- trains one model, one disease
  forecast_service.py   Orchestration: series selection, model dispatch, best-model pick
  explainability.py     SHAP values for the tree-based models (optional dependency)
  chatbot.py            EpiGem AI: scope filter, local RAG fallback, Groq system prompt
  server.py (this file) Flask routes only -- parses requests, calls the modules above,
                         and turns their results/exceptions into HTTP responses.
"""
import os
import json
import logging
import warnings
import traceback

import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, request, render_template

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import config
from data_loader import DISEASE_DATA
import forecast_service as fsvc
from forecast_service import SeriesError
import explainability
import chatbot

# -- APP SETUP --------------------------------------------------
BASE = Path(__file__).parent
app  = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True  # edits to templates/*.html show up on next
                                             # request without restarting the server

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = app.logger


# -- ERROR HANDLING (Comment 8) ----------------------------------
# Previously, every route's except block returned 'trace': traceback.format_exc()
# directly in the JSON response -- meaning any unexpected error leaked full Python
# stack traces (file paths, internals) to whatever called the API. Tracebacks are
# now always logged server-side only; clients get a clean, safe message.
def log_and_hide(e, where):
    log.error("Unhandled error in %s: %s\n%s", where, e, traceback.format_exc())


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('home.html'), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large -- max {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413


@app.errorhandler(Exception)
def handle_unexpected(e):
    """Last-resort catch-all so a truly unexpected error never surfaces as a raw
       Flask/Werkzeug error page or an unhandled 500 with no body."""
    log_and_hide(e, request.path)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error. Please try again.'}), 500
    return render_template('home.html'), 500


# -- PAGE ROUTES --------------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
        disease_data=json.dumps(DISEASE_DATA, separators=(',', ':')))

@app.route('/upload-page')
def upload_page():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')


# -- FORECAST API -------------------------------------------------
@app.route('/api/diseases')
def api_diseases():
    summary = {k: {'years': d['years'], 'total_cases': d['total_cases'],
                   'total_deaths': d['total_deaths'], 'n_locations': d['n_locations']}
               for k, d in DISEASE_DATA.items()}
    return jsonify(summary)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    body     = request.get_json(silent=True) or {}
    disease  = body.get('disease', 'COVID-19')
    model_nm = body.get('model', 'Random Forest')
    horizon  = body.get('horizon', config.DEFAULT_HORIZON)
    metric   = body.get('metric', 'auto')

    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        return jsonify({'error': 'horizon must be a number'}), 400

    try:
        result = fsvc.predict_one(DISEASE_DATA, disease, model_nm, metric, horizon)
    except KeyError:
        return jsonify({'error': f'Unknown disease: {disease}'}), 400
    except SeriesError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        log_and_hide(e, '/api/predict')
        return jsonify({'error': 'Forecast failed unexpectedly. Please try a different model or disease.'}), 500

    if result is None:
        return jsonify({'error': 'Model training failed, insufficient data points'}), 400
    return jsonify(result)


@app.route('/api/all_models', methods=['POST'])
def api_all_models():
    body    = request.get_json(silent=True) or {}
    disease = body.get('disease', 'COVID-19')
    horizon = body.get('horizon', config.DEFAULT_HORIZON)
    metric  = body.get('metric', 'auto')

    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        return jsonify({'error': 'horizon must be a number'}), 400

    try:
        best, good = fsvc.predict_all(DISEASE_DATA, disease, metric, horizon)
    except KeyError:
        return jsonify({'error': f'Unknown disease: {disease}'}), 400
    except SeriesError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        log_and_hide(e, '/api/all_models')
        return jsonify({'error': 'Forecast failed unexpectedly. Please try a different disease.'}), 500

    if not good:
        return jsonify({'error': 'All models failed'}), 500
    return jsonify({'disease': disease, 'best_model': best, 'models': good})


@app.route('/api/explain/<model_name>', methods=['POST'])
def api_explain(model_name):
    """SHAP explanation for a tree-based model (Random Forest / XGBoost) -- addresses
       mentor review Comment 12. Separate from /api/predict because SHAP computation
       is extra work most callers don't need; the dashboard's Explainability section
       calls this only when the user has that panel open. Returns
       {'available': False, 'reason': ...} rather than an error when shap isn't
       installed or the model isn't tree-based, since that's an expected, non-fatal
       state -- the existing model-derived Feature Weights panel is the fallback."""
    body    = request.get_json(silent=True) or {}
    disease = body.get('disease', 'COVID-19')
    horizon = body.get('horizon', config.DEFAULT_HORIZON)
    metric  = body.get('metric', 'auto')

    if not explainability.shap_available_for(model_name):
        reason = ('shap package not installed on the server' if not explainability.SHAP_AVAILABLE
                  else f'SHAP is only available for {sorted(explainability.SHAP_MODELS)}, not {model_name}')
        return jsonify({'available': False, 'reason': reason})

    try:
        horizon = int(horizon)
        if disease not in DISEASE_DATA:
            return jsonify({'error': f'Unknown disease: {disease}'}), 400
        y, years, lbl = fsvc.select_series(DISEASE_DATA[disease], metric)
        n = len(y)
        n_lags = min(3, max(1, n - 2))
        from ml_models import run_sklearn
        raw = run_sklearn(model_name, y, horizon, n_lags)  # un-stripped -- keeps _model_final/_X
        if not raw:
            return jsonify({'error': 'Model training failed'}), 400
        shap_vals = explainability.shap_for_result(model_name, raw)
    except SeriesError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        log_and_hide(e, '/api/explain')
        return jsonify({'available': False, 'reason': 'SHAP computation failed for this series'})

    if shap_vals is None:
        return jsonify({'available': False,
                        'reason': 'This model\'s prediction does not vary across the recent '
                                  'history for this series, so there is nothing for SHAP to '
                                  'attribute -- try a different disease or model.'})
    return jsonify({'available': True, 'model': model_name, 'disease': disease,
                     'shap_importances': shap_vals})


# -- FILE UPLOAD ----------------------------------------------------
ALLOWED_UPLOAD_EXTS = {'.csv', '.xlsx', '.xls', '.json'}


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f   = request.files['file']
    if not f.filename:
        return jsonify({'error': 'No file selected'}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        return jsonify({'error': f'Unsupported type: {ext or "(none)"}. Use .csv, .xlsx, or .json'}), 400

    try:
        horizon = int(request.form.get('horizon', config.DEFAULT_HORIZON))
    except (TypeError, ValueError):
        return jsonify({'error': 'horizon must be a number'}), 400
    horizon = max(1, min(config.MAX_HORIZON, horizon))  # was previously uncapped

    try:
        if ext == '.csv':
            df = pd.read_csv(f)
        elif ext in ('.xlsx', '.xls'):
            df = pd.read_excel(f)
        else:  # .json
            data = json.load(f)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    except Exception:
        return jsonify({'error': f'Could not parse this {ext} file -- check it is well-formed and try again'}), 400

    if df.empty:
        return jsonify({'error': 'The uploaded file has no rows'}), 400

    df.columns = [c.lower().strip() for c in df.columns]
    year_col  = next((c for c in df.columns if 'year' in c or 'date' in c or 'time' in c), '')
    case_col  = next((c for c in df.columns if 'case' in c or 'count' in c or 'value' in c or 'incident' in c), '')
    death_col = next((c for c in df.columns if 'death' in c or 'mortal' in c or 'fatal' in c), '')
    ctry_col  = next((c for c in df.columns if 'country' in c or 'nation' in c or 'region' in c or 'location' in c), '')

    if not year_col or not case_col:
        return jsonify({'error': f'Could not detect year and case columns. Found: {list(df.columns)}. '
                                  f'Add columns named year, cases.'}), 400

    # Validate the year column actually contains parseable values before going further --
    # previously a non-numeric year column would silently coerce to NaN and get dropped,
    # producing a confusing "need at least 3 points" error with no indication why.
    year_numeric = pd.to_numeric(df[year_col], errors='coerce')
    if year_numeric.notna().sum() == 0:
        return jsonify({'error': f"Column '{year_col}' doesn't contain recognizable year/date values"}), 400

    cols = [year_col, case_col]
    if death_col: cols.append(death_col)
    if ctry_col:  cols.append(ctry_col)
    df = df[cols].dropna(subset=[year_col, case_col]).sort_values(year_col)
    df[case_col] = pd.to_numeric(df[case_col], errors='coerce').fillna(0)

    # Country aggregation
    country_data = {}
    if ctry_col:
        if death_col:
            df[death_col] = pd.to_numeric(df[death_col], errors='coerce').fillna(0)
        for ctry, grp in df.groupby(ctry_col):
            ctry = str(ctry).strip()
            if not ctry:
                continue
            country_data[ctry] = {
                'cases':  float(grp[case_col].sum()),
                'deaths': float(grp[death_col].sum()) if death_col else 0.0,
            }

    # Aggregate to annual totals
    agg = {case_col: 'sum'}
    if death_col:
        df[death_col] = pd.to_numeric(df[death_col], errors='coerce').fillna(0)
        agg[death_col] = 'sum'
    annual = df.groupby(year_col).agg(agg).reset_index().sort_values(year_col)

    y     = annual[case_col].values.astype(float)
    years = [str(int(float(v))) if pd.notna(v) else str(v) for v in annual[year_col].tolist()]
    death_values = [float(v) for v in annual[death_col].values] if death_col else []

    if len(y) < 3:
        return jsonify({'error': f'Need at least 3 annual data points after aggregation. Got {len(y)}.'}), 400

    n_lags   = min(3, max(1, len(y) - 2))
    last_yr  = int(float(years[-1])) if years else 2024
    fc_years = [str(last_yr + i + 1) for i in range(horizon)]

    results = {}
    for mn in fsvc.UPLOAD_MODEL_NAMES:
        try:
            r = fsvc.run_one_model(mn, y, horizon, n_lags)
            if r:
                results[mn] = fsvc.strip_internal(r)
        except Exception as e:
            log.warning("Upload model %s failed: %s", mn, e)

    if not results:
        return jsonify({'error': 'All models failed on this dataset -- try a longer or cleaner series'}), 400

    best = max(results, key=lambda m: fsvc.score_result(results[m]))

    return jsonify({
        'filename':      f.filename,
        'rows':          int(len(df)),
        'columns':       list(df.columns),
        'detected_cols': {'year': year_col, 'cases': case_col,
                           'deaths': death_col, 'country': ctry_col},
        'years':         years,
        'values':        [float(v) for v in y],
        'death_values':  death_values,
        'country_data':  country_data,
        'fc_years':      fc_years,
        'best_model':    best,
        'models':        results,
    })


# -- EPIGEM AI (GROQ) CHATBOT -----------------------------------
@app.route('/api/chat-groq', methods=['POST'])
def api_chat_groq():
    body     = request.get_json(silent=True) or {}
    messages = body.get('messages', [])
    disease  = body.get('disease', '')
    last_q   = messages[-1]['content'] if messages else ''

    # Fast, free, consistent pre-filter for clearly off-topic questions -- skips the
    # API call entirely so refusals are instant even if Groq is slow or unreachable.
    if not chatbot.is_on_topic(last_q):
        return jsonify({'answer': chatbot.off_topic_reply(), 'model': 'scope_filter',
                        'source': 'scope_filter', 'tokens': 0})

    if not config.GROQ_ENABLED:
        # No GROQ_API_KEY configured (see .env.example) -- skip the network
        # call entirely rather than making a guaranteed-401 request.
        return jsonify({'answer': chatbot.rag_answer(last_q) or
                        'EpiGem AI chat is running in local mode (no Groq API key configured). '
                        'Ask me about diseases, models, or the Epiworld platform.',
                        'model': 'local_rag', 'source': 'local'})

    try:
        answer, model, source, tokens = chatbot.call_groq(messages, disease, DISEASE_DATA)
        return jsonify({'answer': answer, 'model': model, 'source': source, 'tokens': tokens})
    except Exception as e:
        log.warning("Groq call failed, falling back to local RAG: %s", e)
        return jsonify({'answer': f'[Groq unavailable] {chatbot.rag_answer(last_q)}',
                        'model': 'local_rag', 'source': 'local', 'error': str(e)})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    body     = request.get_json(silent=True) or {}
    question = body.get('message', '')
    disease  = body.get('disease', '')

    if not chatbot.is_on_topic(question):
        return jsonify({'answer': chatbot.off_topic_reply(), 'source': 'scope_filter'})

    context = chatbot.rag_answer(question)
    if disease and disease in DISEASE_DATA:
        d = DISEASE_DATA[disease]
        context += (f" {disease}: {d['years'][0]}-{d['years'][-1]}, "
                    f"cases={d['total_cases']:,}, deaths={d['total_deaths']:,}.")
    return jsonify({'answer': context or 'Ask me about diseases, models, or the Epiworld platform.',
                    'source': 'local'})


@app.route('/api/chat-status')
def chat_status():
    if not config.GROQ_ENABLED:
        return jsonify({'status': 'not_configured',
                        'message': 'GROQ_API_KEY not set -- EpiGem AI is running in local mode. '
                                   'See .env.example.'})
    try:
        code = chatbot.groq_ping()
        if code == 200:
            return jsonify({'status': 'connected', 'model': config.GROQ_MODEL})
        return jsonify({'status': 'error', 'code': code})
    except Exception as e:
        return jsonify({'status': 'unreachable', 'error': str(e)})


if __name__ == '__main__':
    print("=" * 52)
    print("  Epiworld Platform - AI Epidemic Intelligence")
    print(f"  http://{config.HOST}:{config.PORT}")
    print(f"  EpiGem AI: {'enabled' if config.GROQ_ENABLED else 'local mode (no API key)'}"
          f"  |  SHAP: {'enabled' if explainability.SHAP_AVAILABLE else 'not installed'}"
          f"  |  Live ML  |  World Map  |  Upload")
    print("=" * 52)
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT, threaded=True)

# Epiworld

AI-powered epidemic intelligence and forecasting platform. Applies seven
machine learning models (Random Forest, Ridge, Lasso, SVM, XGBoost, LSTM,
GRU) to seven infectious diseases across 237+ countries, with an
integrated AI assistant (EpiGem, via Groq LLaMA 3.1) for platform Q&A.

Built with Flask, scikit-learn, TensorFlow/Keras, XGBoost, and Chart.js.
Research and educational use only.

## Objectives

- Forecast future case/death trends per disease using multiple ML models,
  auto-selecting the best performer per series via walk-forward validation.
- Present results through a plain-language dashboard: trend direction and a
  qualitative confidence rating, rather than raw statistical output, so the
  platform is usable by non-technical stakeholders.
- Let users upload their own epidemiological data (CSV/Excel/JSON) and get
  the same analysis and a downloadable PDF report.
- Provide an in-app assistant scoped strictly to the platform's own data and
  methodology.

## Project Structure

```
epiworld/
├── server.py               # Flask routes only -- parses requests, delegates to the
│                           #   modules below, turns results/exceptions into HTTP responses
├── config.py               # Central config (env-driven; see Configuration)
├── data_loader.py           # Loads disease_data.json
├── ml_features.py           # Feature engineering, walk-forward validation, metrics
├── ml_models.py             # run_sklearn() / run_rnn() -- trains one model, one disease
├── forecast_service.py      # Orchestration: series selection, model dispatch, best-model pick
├── explainability.py        # SHAP values for the tree-based models (optional dependency)
├── chatbot.py               # EpiGem AI: scope filter, local RAG fallback, Groq system prompt
├── disease_data.json        # Bundled reference dataset (WHO/UNAIDS/OWID)
├── requirements.txt
├── .env.example             # Copy to .env and fill in your Groq key
├── static/
│   ├── css/main.css
│   └── js/dashboard.js
└── templates/
    ├── base.html            # Shared layout, nav, design tokens
    ├── home.html
    ├── dashboard.html
    ├── upload.html
    └── about.html
```

Each module has a single responsibility, so adding a new model only means
touching `ml_models.py` + registering it in `forecast_service.ALL_MODEL_NAMES`
-- not editing a single 1,100+ line file that also handles HTTP routing.

## Installation

Requires Python 3.11 or 3.12.

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd epiworld

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then open .env and paste in a Groq API key (free at https://console.groq.com)
# -- everything except the EpiGem chatbot works without this key.
```

If `tensorflow` or `xgboost` fail to build on Windows, installing inside the
venv above (rather than globally) resolves most version-conflict issues.

## Running

```bash
python server.py
```

The server starts at `http://127.0.0.1:5000` by default. Console output
confirms the port and whether the EpiGem chatbot is enabled.

To change host/port, set `EPIWORLD_HOST` / `EPIWORLD_PORT` in `.env`
(see `.env.example` for all available options).

## Workflow

1. **Home** -- landing page with an interactive world map and platform stats.
2. **Dashboard** -- opens directly on the Live Summary: the best model is
   auto-selected and forecast automatically for the current disease.
   Sidebar sections: Trends & Overview, Risk Map, Early Warning,
   Explainability, Simulation, and Forecast Settings (for picking a specific
   model/horizon manually).
3. **Upload Data** -- upload your own dataset for the same analysis, with a
   downloadable PDF report.
4. **About** -- methodology, data sources, and contact information.

## Methodology (summary)

- Series are detrended (log-transform for large counts) before
  feature-engineered models (Random Forest, XGBoost, Lasso, SVM) see them;
  Ridge uses trend + lag + moving-average + momentum features directly.
- LSTM/GRU (2-layer RNN, 64→32 units, Dropout 0.2) are trained on raw
  sequences for comparison.
- **Validation:** walk-forward (rolling-origin, expanding-window)
  cross-validation across 6 folds -- not a single random train/test split --
  because epidemiological series are short and non-stationary, and ordinary
  k-fold CV would leak future information into training.
- **Model selection:** for each disease, all 7 models are trained; the one
  with the highest out-of-sample (test) R² is auto-selected as the default,
  falling back to a discounted training R² only when a series is too short
  to hold out a genuine test fold.
- **What the UI shows:** the dashboard deliberately does not display raw
  R²/RMSE/MAE/MAPE/AIC to end users. These are computed and used internally
  for model selection, but surfaced only as a plain-language Trend
  (Rising/Falling/Stable) and Forecast Confidence (High/Moderate/Limited)
  rating, so a non-technical viewer isn't left to misinterpret a bare
  statistic. The full numeric metrics remain available via the JSON API
  (`/api/predict`, `/api/all_models`) for evaluation/reporting purposes.

Full methodology detail is on the **About** page in the running app.

## Data & Preprocessing

- **Sources:** WHO, UNAIDS, and Our World in Data, covering 1980-2024 across
  237+ countries for the 7 bundled diseases (COVID-19, AIDS/HIV, Malaria,
  Dengue, Measles, Hepatitis B, Polio).
- **Snapshot, not a live feed:** `disease_data.json` is a static bundle
  captured at build time. There is no scheduled refresh job -- updating the
  dataset means regenerating and replacing that file.
- **Leading-zero trimming:** Some diseases have years of all-zero data before
  reporting began for that disease/region. `forecast_service.select_series()`
  trims those leading zeros before any model sees the series (implemented as
  a simple leading-edge scan, not interpolation -- zeros appearing after the
  series has started are left as real data points).
- **Missing values (uploaded data):** on the Upload tab, rows missing a year
  or case value are dropped (`dropna`); non-numeric case/death values are
  coerced to 0 rather than crashing the request. The year column is validated
  up front (Comment 8) -- if it contains no parseable numeric/date values at
  all, the upload is rejected with a clear message instead of silently
  aggregating to nothing.
- **Log-transform:** applied automatically whenever a series' maximum
  absolute value exceeds 1,000, to stabilize variance across the large swings
  typical of raw case/death counts. This is undone (`expm1`) before any value
  is returned to the client -- the API never returns log-scale numbers.
- **Minimum series length:** at least 3 data points after trimming are
  required for any model to run; fewer than 8 triggers the "small sample"
  warning in the UI. COVID-19 in the bundled dataset has only 5 usable points
  after trimming, which is enough for the RNN models but too few for the
  sklearn-family models to fit at all (see Known limitations below).

See `Epiworld_Model_Evaluation_Report.xlsx` (Methodology sheet) for the
validation-methodology detail that complements this section -- that document
covers walk-forward CV and per-model architecture choices, which are about
*evaluating* the models rather than preprocessing the data.



| Endpoint | Method | Purpose |
|---|---|---|
| `/api/predict` | POST | Run one model for one disease |
| `/api/all_models` | POST | Run all 7 models, return best + full comparison |
| `/api/upload` | POST | Analyze an uploaded CSV/Excel/JSON file |
| `/api/chat-groq` | POST | EpiGem AI chat (falls back to local answers if Groq is unreachable/unconfigured) |
| `/api/chat-status` | GET | Whether the Groq-backed chatbot is currently configured/reachable |
| `/api/explain/<model_name>` | POST | SHAP feature attribution for Random Forest/XGBoost (Comment 12). Returns `{"available": false, "reason": ...}` gracefully if `shap` isn't installed or the model isn't tree-based |

All ML endpoints return train/test metrics (R², RMSE, MAE, MAPE, AIC) in
the JSON response even though the dashboard UI doesn't render them --
useful for building the consolidated evaluation report mentioned above.

## Configuration

All configuration lives in `config.py` and is read from environment
variables (via `.env`, see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Required for EpiGem AI chat |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EPIWORLD_HOST` | `127.0.0.1` | Flask bind host |
| `EPIWORLD_PORT` | `5000` | Flask bind port |
| `EPIWORLD_DEBUG` | `false` | Flask debug mode -- keep `false` in production |
| `EPIWORLD_DEFAULT_HORIZON` | `5` | Default forecast horizon (years) |
| `EPIWORLD_MAX_HORIZON` | `20` | Maximum forecast horizon allowed via the API |

**Do not commit `.env`.** It's already listed in `.gitignore`. Only
`.env.example` (no real values) should ever be committed.

## Known limitations

- Diseases with very few historical data points (e.g. a series with only
  5 years of usable data) can cause the sklearn-family models to fail to
  fit; in that case `/api/all_models` returns whichever models succeeded
  (typically the RNN models) rather than all 7.
- On extremely short series (COVID-19's 5 usable points is the case in the
  bundled dataset), the RNN models can forecast down to 0 in later horizon
  steps. This is a genuine data-volume limitation rather than a code defect --
  the UI already surfaces it honestly via the "Limited" confidence rating and
  small-sample warning rather than hiding it.
- The bundled `disease_data.json` is a static snapshot, not a live feed.
- SHAP explanations (`/api/explain`) are only implemented for the two
  tree-based models (Random Forest, XGBoost); Ridge/Lasso/SVM/LSTM/GRU fall
  back to the existing model-derived Feature Weights panel.

## Evaluation Report

`Epiworld_Model_Evaluation_Report.xlsx` (generated separately, not part of
this repo's runtime) contains the full walk-forward-validated R2/RMSE/MAE/
MAPE/AIC for every model x every disease combination -- this is the
consolidated evaluation Comments 5 and 14 asked for. Regenerate it any time
by calling `forecast_service.predict_all()` for each disease in
`data_loader.DISEASE_DATA` and exporting the result dicts; it is intentionally
a separate artifact rather than a live endpoint, since these are exactly the
raw numbers the dashboard UI deliberately keeps out of the end-user view.

## License / Use

Research and educational use only.
#   E p i W o r l d - A I - M o d e l  
 
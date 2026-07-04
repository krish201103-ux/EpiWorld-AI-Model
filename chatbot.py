"""
chatbot.py -- EpiGem AI: scope filtering, local RAG fallback, and the Groq
system prompt.

Extracted from server.py as part of the modularization requested in mentor
review Comments 1, 3, and 13. server.py's /api/chat-groq route imports
call_groq() and rag_answer()/is_on_topic()/off_topic_reply() from here
rather than embedding ~150 lines of prompt/keyword-list content inline
among the Flask route handlers.
"""
import re
import itertools

import requests as _req

import config

GROQ_KEY   = config.GROQ_API_KEY
GROQ_URL   = config.GROQ_URL
GROQ_MODEL = config.GROQ_MODEL

GROQ_SYSTEM = """You are EpiGem AI, the assistant built into the Epiworld platform. You ONLY discuss
Epiworld itself: its 7 tracked diseases, its ML models and methodology, its dashboard features,
the world map, risk classification, early warning system, file uploads, and how to use the site.

PLATFORM KNOWLEDGE:
- Project: Epiworld - AI-powered epidemic intelligence and decision support system
- 7 diseases: COVID-19 (2020-2024, 237 countries, 775M+ cases), AIDS (1990-2023), Malaria (2000-2023, 5.67B cases), Dengue (1990-2023, 3.74B cases), Measles (1980-2023), Hepatitis B (1990-2022), Polio (1980-2023)
- ML Models: Random Forest, Ridge (L2, alpha=0.3), Lasso (L1, alpha=0.005), SVM (RBF), XGBoost, LSTM (2-layer 64->32 units, Dropout 0.2), GRU
- Methodology: hybrid features (time trend, lags, moving average, momentum) for Ridge; detrend-then-residual architecture for Random Forest/XGBoost/Lasso/SVM, since those models cannot extrapolate a rising or falling trend on their own
- Validation: walk-forward (rolling-origin, expanding window) cross-validation over 6 folds gives the honest test R2, far more stable than a single train/test split on a 24-35 point series
- Key: internally, models are ranked by walk-forward Test R2 (RMSE/MAE/MAPE also tracked) to auto-select the best one. The dashboard UI intentionally does NOT display these raw numbers to users -- it shows a plain-language Forecast Confidence rating instead (High / Moderate / Limited). When a user asks how reliable a forecast is, lead with that same High/Moderate/Limited framing and a plain-language reason, the way the dashboard does. Only give the underlying raw R2/RMSE numbers if the user explicitly asks for the statistical/technical metric itself.
- Log-transform applied automatically for large count data (max > 1000) to stabilise variance
- Bootstrap CI (30 resamples) for sklearn-family models. Monte Carlo Dropout (20 passes) for RNN models
- LSTM/GRU are experimental: with only 24-35 annual data points they cannot reliably beat the feature-engineered models; they are better suited to larger uploaded datasets
- Risk: High >12%, Medium 4-12%, Low <4% country burden share
- Early Warning: Rolling 3-year Z-score. |Z| > 2 = High Alert, |Z| > 1.5 = Medium
- Upload: CSV/Excel/JSON accepted. Auto-detects year, cases, deaths, country columns. Generates a downloadable PDF report
- World map: appears on Home, Dashboard, and Upload. Click any shaded country to open a side-panel analysis report (no page navigation)

NAVIGATION STEPS (Dashboard):
- Opening the Dashboard nav link lands directly on Live Summary -- forecast, risk, alerts, explainability, and simulation together, with the best model auto-run
- Sidebar section: Live Summary (default landing view, consolidated)
- Sidebar section: Trends & Overview (historical charts, CFR, country ranking, world map)
- Sidebar section: Risk Map (all countries, searchable)
- Sidebar section: Early Warning (Z-score anomaly detection)
- Sidebar section: Explainability (model-derived feature weights, not SHAP)
- Sidebar section: Simulation (what-if intervention sliders)
- Sidebar section: Forecast Settings (pick a specific model, horizon, Run Forecast or Run All Models)
- Upload tab: Analyze your own data with the same visualizations, plus PDF report download

STRICT SCOPE RULE -- read carefully:
You must ONLY answer questions about Epiworld, the diseases it tracks, its ML methodology, or how
to use its features. This includes general epidemiology questions ONLY when they directly relate to
interpreting Epiworld's own data or output (e.g. "what does CFR mean" is fine; "what's the CFR of a
disease not on this platform" is not).

If a question is unrelated to Epiworld -- general chit-chat, unrelated coding help, other websites or
products, news, opinions, math homework, personal advice, or any topic with no connection to this
platform -- do NOT answer it. Instead, briefly and warmly say you're scoped to Epiworld questions,
and pivot with ONE specific, relevant suggestion of something they could ask instead (pick whichever
of: disease forecasts, the world map, model accuracy, risk classification, or uploading data, seems
closest to what they were asking about, or default to "Try asking about a disease forecast or how
the world map works" if nothing is close). Keep the decline to 1-2 sentences. Never lecture, never
repeat the rule back to them, never say "I am not able to" more than once in the same reply.

Be helpful, concise (2-4 sentences for normal answers), and accurate. Never invent statistics."""

# -- DOMAIN RESTRICTION (heuristic pre-filter) -------------------
# Catches clearly off-topic questions BEFORE calling the LLM, so refusals are instant, free,
# and consistent even if Groq is unreachable. The LLM's own system-prompt instructions act as
# a second layer of defense for borderline cases this heuristic doesn't catch.
ON_TOPIC_KEYWORDS = [
    'epiworld', 'epigem', 'disease', 'diseas', 'covid', 'coronavirus', 'aids', 'hiv',
    'malaria', 'dengue', 'measles', 'hepatitis', 'polio', 'pandemic', 'epidemic', 'outbreak',
    'forecast', 'predict', 'projection', 'random forest', 'ridge', 'lasso', 'svm',
    'xgboost', 'lstm', 'gru', 'rnn', 'r2', 'r-squared', 'rmse', 'mae', 'mape', 'accuracy',
    'overfit', 'walk-forward', 'walk forward', 'cross-validation', 'cross validation',
    'bootstrap', 'confidence interval', 'validation', 'feature import',
    'shap', 'risk', 'classification', 'z-score', 'zscore', 'anomaly', 'early warning',
    'alert', 'simulation', 'intervention', 'mobility', 'vaccin', 'world map',
    'countries', 'upload', 'csv', 'excel', 'json', 'pdf', 'report', 'dashboard',
    'cfr', 'case fatality', 'mortality', 'unaids', 'owid',
    'data source', 'methodology', 'preprocess', 'feature engineer', 'log-transform',
    'log transform', 'detrend', 'trend', 'horizon', 'chatbot', 'groq', 'llama',
    'this platform', 'this website', 'this site', 'this dashboard', 'this app',
]
# Short or generic words that need WHOLE-WORD matching, since as plain substrings they would
# false-positive on unrelated text (e.g. "who" inside "who won", "test" inside "I'm testing
# you"). Excludes "map", "model", "train" -- those are genuinely ambiguous in everyday English
# (map app, role model, train schedule) and are instead handled by ON_TOPIC_CONTEXT_PAIRS below,
# which only counts them as on-topic when they co-occur with an unambiguous platform signal.
ON_TOPIC_WHOLE_WORDS = [
    'who', 'case', 'cases', 'death', 'deaths', 'test', 'country', 'site', 'tab', 'step',
    'feature', 'platform',
]
# Ambiguous word -> list of companion words that must ALSO appear for it to count as on-topic.
ON_TOPIC_CONTEXT_PAIRS = {
    'map': ['world', 'country', 'click', 'dashboard', 'epiworld', 'global', 'disease'],
    'model': ['forest', 'ridge', 'lasso', 'svm', 'xgboost', 'lstm', 'gru', 'ml', 'machine',
              'accuracy', 'forecast', 'predict', 'r2', 'train', 'test', 'epiworld'],
    'train': ['model', 'test', 'r2', 'accuracy', 'data', 'set', 'split'],
}
OFF_TOPIC_SIGNALS = [
    'write me a', 'write a poem', 'write code for', 'translate this', 'recipe for',
    'weather in', 'stock price', 'movie recommend', 'song lyrics', 'tell me a joke',
    'who won the', 'capital of', 'solve this equation', 'homework', 'essay about',
    'love advice', 'relationship advice', 'medical diagnosis', 'prescribe', 'symptom of',
    'legal advice', 'tax advice', 'is it true that', 'current president', 'crypto',
    'bitcoin', 'stock market',
]


def is_on_topic(question):
    """Heuristic check: does this question plausibly relate to Epiworld? Errs toward YES
       (on-topic) for short or ambiguous questions, since a false negative just means the LLM's
       own scope instructions handle it -- but a false positive (blocking something legitimate)
       is more annoying for users. This is a fast first-pass filter only, not the final word."""
    q = question.lower().strip()
    if len(q) < 2:
        return True  # too short to classify, let it through
    if any(sig in q for sig in OFF_TOPIC_SIGNALS):
        return False
    if any(kw in q for kw in ON_TOPIC_KEYWORDS):
        return True
    # Word-boundary check for short/generic words that would false-positive as substrings
    if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ON_TOPIC_WHOLE_WORDS):
        return True
    # Ambiguous words (map/model/train) only count when paired with an unambiguous companion
    for ambiguous_word, companions in ON_TOPIC_CONTEXT_PAIRS.items():
        if re.search(r'\b' + ambiguous_word + r'\b', q) and any(c in q for c in companions):
            return True
    # Exact greetings/acknowledgements only -- NOT a generic "2 words or fewer" pass-through,
    # since short off-topic questions ("weather today", "stock price") must still be caught
    greetings = {'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'sure',
                 'yes', 'no', 'bye', 'cool', 'nice', 'great', 'got it'}
    if q.rstrip('!.?') in greetings:
        return True
    return False


OFF_TOPIC_REPLIES = [
    "That's outside what I can help with here -- I'm scoped to Epiworld. Try asking about a disease forecast, the world map, or how model accuracy is calculated.",
    "I'm built specifically for Epiworld questions, so I can't help with that. Want to know how the risk classification or early warning system works instead?",
    "That falls outside my scope -- I only answer questions about Epiworld's diseases, models, and features. I'd be glad to explain how to upload your own dataset, for example.",
]
_off_topic_cycle = itertools.cycle(OFF_TOPIC_REPLIES)


def off_topic_reply():
    """Rotate through a few varied decline messages so repeated off-topic attempts don't get
       the exact same canned sentence every time."""
    return next(_off_topic_cycle)


# -- RAG FALLBACK -----------------------------------------------
RAG_KB = {
    "dashboard": "Opening Dashboard takes you straight to the Live Summary -- forecast, risk, alerts, explainability, and simulation in one view, with the best model auto-selected. Trends & Overview, Risk Map, Early Warning, Explainability, Simulation, and Forecast Settings are all in the sidebar.",
    "forecast":  "Forecast Settings lets you pick a specific model -- Random Forest, Ridge, Lasso, SVM, XGBoost, LSTM, or GRU -- then Run Forecast or Run All Models. Each forecast shows a plain-language Trend and Forecast Confidence (High/Moderate/Limited) rating instead of raw accuracy numbers.",
    "rnn":       "LSTM/GRU use 2-layer RNN (64 -> 32 units) with Dropout 0.2, Huber loss, EarlyStopping. CI is Monte Carlo Dropout over 20 passes.",
    "risk":      "Risk classification: High >12%, Medium 4-12%, Low <4% of total disease burden. Click any country for detailed breakdown.",
    "upload":    "Upload CSV, Excel, or JSON with columns: year, cases, deaths, country. Epiworld auto-detects columns, runs analysis, and lets you download a PDF report.",
    "chatbot":   "I am EpiGem AI, the Epiworld epidemic intelligence assistant powered by Groq LLaMA 3.1.",
    "covid":     "COVID-19: 2020-2024, 237 countries, 775M+ cases, 7M+ deaths. Data from WHO/OWID.",
    "aids":      "AIDS/HIV: 1990-2023, UNAIDS data, 9 regions.",
    "malaria":   "Malaria: 2000-2023, WHO World Malaria Report, 5.67B cumulative cases, 15.9M deaths.",
    "map":       "The interactive world map shows country-wise disease burden. Click any country to see annual trend, total cases, deaths, and CFR in a detailed popup.",
    "metrics":   "Under the hood, models are ranked by walk-forward validated Test R2 (with RMSE/MAE/MAPE also tracked) to pick the best one automatically. The dashboard itself shows this as a plain-language Forecast Confidence rating (High/Moderate/Limited) rather than raw numbers, so it's easy to interpret at a glance.",
    "pdf":       "On the Upload tab, after analyzing your data, click Download PDF Report to get a full formatted report with stats, trends, and country breakdown.",
}


def rag_answer(question):
    """Local knowledge-base fallback used when Groq is unreachable. Always gated behind
       is_on_topic() first, so a network failure never becomes a backdoor around the scope
       restriction -- if Groq is down AND the question is off-topic, the user still gets a
       proper decline rather than generic platform trivia."""
    if not is_on_topic(question):
        return off_topic_reply()
    q = question.lower()
    hits = []
    kw = {
        'rnn': ['rnn', 'lstm', 'gru', 'deep learning', 'neural', 'recurrent'],
        'forecast': ['forecast', 'predict', 'horizon', 'projection'],
        'risk': ['risk', 'classify', 'risk zone', 'risk level'],
        'upload': ['upload', 'own data', 'my data', 'csv', 'excel', 'json', 'file'],
        'chatbot': ['who are you', 'what are you', 'epigem', 'assistant', 'epiworld'],
        'metrics': ['r2', 'rmse', 'mae', 'mape', 'accuracy', 'overfit'],
        'map': ['world map', 'interactive map', 'click', 'global'],
        'pdf': ['pdf', 'report', 'download'],
        'covid': ['covid', 'coronavirus', 'pandemic'],
        'aids': ['aids', 'hiv'],
        'malaria': ['malaria'],
        'dashboard': ['dashboard', 'navigate', 'step', 'overview'],
    }
    for key, words in kw.items():
        if any(w in q for w in words):
            hits.append(RAG_KB.get(key, ''))
    return ' '.join(set(hits)) if hits else RAG_KB['dashboard'] + ' ' + RAG_KB['forecast']


def call_groq(messages, disease=None, disease_data=None):
    """Call the Groq chat completion API with the EpiGem system prompt. Returns
       (answer, model_name, source, extra) on success, or raises so the caller
       (server.py) can fall back to rag_answer(). `extra` carries token usage."""
    system = GROQ_SYSTEM
    if disease and disease_data and disease in disease_data:
        d = disease_data[disease]
        system += (f"\n\nCURRENT VIEW: {disease} -- "
                   f"Years {d['years'][0]}-{d['years'][-1]}, "
                   f"Cases {d['total_cases']:,}, Deaths {d['total_deaths']:,}, "
                   f"{d['n_locations']} countries.")

    payload = {
        'model': GROQ_MODEL,
        'messages': [{'role': 'system', 'content': system}] + messages[-12:],
        'max_tokens': 350,
        'temperature': 0.65,
        'stream': False,
    }
    resp = _req.post(
        GROQ_URL,
        headers={'Authorization': f'Bearer {GROQ_KEY}', 'Content-Type': 'application/json'},
        json=payload, timeout=22,
    )
    if resp.status_code == 200:
        data = resp.json()
        answer = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        return answer, GROQ_MODEL, 'groq', usage.get('total_tokens', 0)
    err = resp.json().get('error', {}).get('message', resp.text[:120])
    raise RuntimeError(err)


def groq_ping():
    """Lightweight connectivity check used by /api/chat-status. Raises on failure;
       caller decides how to report that."""
    resp = _req.post(
        GROQ_URL,
        headers={'Authorization': f'Bearer {GROQ_KEY}', 'Content-Type': 'application/json'},
        json={'model': GROQ_MODEL, 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 5},
        timeout=8,
    )
    return resp.status_code

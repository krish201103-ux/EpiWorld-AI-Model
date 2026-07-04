"""
config.py -- Centralized configuration for Epiworld.

Addresses mentor review Comment 9 (Configuration Management): values that
were previously hardcoded inside server.py (host, port, and the Groq API
key) now live here and are read from environment variables, with safe
defaults for local development.

Setup:
  1. Copy .env.example to .env
  2. Put your real Groq API key in .env
  3. .env is in .gitignore -- it will never be committed

server.py reads everything through this module; nothing downstream should
read os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present. No-op in prod
                # environments where real env vars are already set.


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# -- Flask server -----------------------------------------------------
HOST = os.environ.get("EPIWORLD_HOST", "127.0.0.1")
PORT = int(os.environ.get("EPIWORLD_PORT", "5000"))
DEBUG = _get_bool("EPIWORLD_DEBUG", False)

# -- Forecasting defaults ----------------------------------------------
DEFAULT_HORIZON = int(os.environ.get("EPIWORLD_DEFAULT_HORIZON", "5"))
MAX_HORIZON = int(os.environ.get("EPIWORLD_MAX_HORIZON", "20"))

# -- EpiGem AI chatbot (Groq) -------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

GROQ_ENABLED = bool(GROQ_API_KEY)
if not GROQ_ENABLED:
    # Do not crash the whole app if the key is missing -- the dashboard,
    # forecasting, and every other feature work fine without EpiGem.
    # server.py checks GROQ_ENABLED before calling the chatbot endpoints
    # and returns a clear "chatbot not configured" message instead.
    print(
        "[config] WARNING: GROQ_API_KEY is not set. EpiGem AI chat will be "
        "disabled until it is configured in your .env file. All other "
        "features (forecasting, dashboard, upload) are unaffected."
    )

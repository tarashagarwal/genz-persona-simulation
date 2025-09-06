# constants.py
# Central place for config. Reads from env but keeps sane defaults.

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project base (folder that contains this file)
BASE_DIR = Path(__file__).resolve().parent

# Load .env in BASE_DIR (if present), then load any parent/global env
load_dotenv(BASE_DIR / ".env")
load_dotenv()

# ------------ Defaults (override via .env) -------------
DEFAULT_PERSONA_ID  = int(os.getenv("DEFAULT_PERSONA_ID", "3"))
DEFAULT_TEXT        = os.getenv(
    "DEFAULT_TEXT",
    "An Arkansas dad faces murder charges for killing his daughter’s alleged abuser. His wife says he saved their child",
)
PERSONA_CARD_PATH   = os.getenv("PERSONA_CARD_PATH", str(BASE_DIR / "persona_cards.json"))

INDEX_DIR           = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "indexes")))
EMB_MODEL           = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_COL            = os.getenv("TEXT_COL", "text")
ID_COL              = os.getenv("ID_COL", "id")

SIM_THRESHOLD       = float(os.getenv("SIM_THRESHOLD", "0.40"))   # 0..1
EMO_CONF_THRESHOLD  = float(os.getenv("EMO_CONF_THRESHOLD", "0.85"))

GEN_MODEL           = os.getenv("GEN_MODEL", "gpt-4o-mini")

# Verbose default: on unless VERBOSE=0/false
VERBOSE_DEFAULT     = (os.getenv("VERBOSE", "1").lower() not in ["0", "false"])

def get_openai_key() -> str:
    # Helper to fetch OpenAI key consistently
    return os.getenv("OPENAI_API_KEY", "")

from pathlib import Path

# Project root (assumes scripts executed from project root)
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

OUT = ROOT / "outputs"
OUT_FIG = OUT / "figures"
OUT_TAB = OUT / "tables"
OUT_MODELS = OUT / "models"
OUT_LOGS = OUT / "logs"

RANDOM_STATE = 42

def ensure_dirs():
    for p in [DATA_INTERIM, DATA_PROCESSED, OUT_FIG, OUT_TAB, OUT_MODELS, OUT_LOGS]:
        p.mkdir(parents=True, exist_ok=True)
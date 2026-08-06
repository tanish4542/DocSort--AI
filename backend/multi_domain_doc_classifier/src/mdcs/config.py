"""
Central configuration. Add Legal (or any domain) by extending DOMAINS and DOMAIN_KEYWORDS.
"""
from __future__ import annotations

from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Canonical domain order — MUST match sklearn's sorted class order (np.unique on labels).
# Legal will extend this list later; keep alphabetical or retrain after changing order.
DOMAINS: tuple[str, ...] = ("Finance", "Medical", "Sports", "Technology")

# Default source paths (override via env or CLI)
DEFAULT_PATHS = {
    "finance_csv": Path("/Users/prachigoyal/Desktop/Financial.csv"),
    "medical_csv": Path("/Users/prachigoyal/Desktop/train.csv"),
    "technology_numbers": Path(
        "/Users/prachigoyal/Library/Mobile Documents/com~apple~Numbers/Documents/tech.numbers"
    ),
    "sports_numbers": Path("/Users/prachigoyal/Desktop/sports.numbers"),
}

RANDOM_STATE = 42
MIN_DOC_CHARS = 40
MAX_ABSTRACTS_MEDICAL = 12_000
MAX_ROWS_FINANCE = 15_000
BALANCE_MAX_PER_CLASS = 8_000

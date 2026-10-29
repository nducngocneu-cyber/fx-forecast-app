"""Project-wide constants. Kept in one place so the app, scripts and notebook agree."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SNAPSHOT_PATH = DATA_DIR / "usdvnd_daily.csv"

# Yahoo Finance symbol for USD/VND (VND per 1 USD).
TICKER = "VND=X"
HISTORY_START = "2015-01-01"

TRADING_DAYS_PER_YEAR = 252
MAX_HORIZON = 30  # business days

# Spike filter: a quote deviating more than this (in log terms) from its
# centred 11-day rolling median is treated as a bad tick (see data_loader.clean_spikes).
SPIKE_WINDOW = 11
SPIKE_THRESHOLD = 0.006  # 0.6%

# Training windows offered in the app (label -> business days).
TRAIN_WINDOWS = {"1 year": 252, "2 years": 504, "3 years": 756, "5 years": 1260}
DEFAULT_TRAIN_WINDOW = "3 years"

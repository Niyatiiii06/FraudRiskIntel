"""Project paths and settings in one place, so every script uses the same values."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "paysim.csv"   # rename the Kaggle file to this
DB_PATH = DATA_DIR / "fraud.duckdb"         # created by load_data.py

SQL_DIR = PROJECT_ROOT / "sql"

# A fraud label is only "known" this many hours after the transaction.
# Banks confirm fraud later (customer complaint, investigation), so features
# may only use labels that would already be confirmed at scoring time.
LABEL_DELAY_HOURS = 24

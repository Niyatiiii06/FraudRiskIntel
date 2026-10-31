"""Project paths in one place, so every script uses the same files."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "paysim.csv"   # rename the Kaggle file to this
DB_PATH = DATA_DIR / "fraud.duckdb"         # created by load_data.py

SQL_DIR = PROJECT_ROOT / "sql"

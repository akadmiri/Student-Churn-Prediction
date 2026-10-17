from pathlib import Path

# Define the base directory of the project
ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA = ROOT / "data"
RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"

# Raw data files
RAW_INSCRIPTIONS = RAW_DATA / "inscriptions.csv"
RAW_ETUDIANTS = RAW_DATA / "etudiants.csv"
RAW_PROGRAMMES = RAW_DATA / "programmes.csv"
RAW_EVALUATIONS = RAW_DATA / "evaluations.csv"
RAW_ACTIVITES = RAW_DATA / "activites_virtuelles.csv"

# Processed data files
CLEAN_INSCRIPTIONS = PROCESSED_DATA / "clean_inscriptions.parquet"
CLEAN_EVALUATIONS = PROCESSED_DATA / "clean_evaluations.parquet"
CLEAN_ACTIVITES = PROCESSED_DATA / "clean_activities.parquet"

FEATURES = PROCESSED_DATA / "features.parquet"
DATASET = PROCESSED_DATA / "churn.parquet"
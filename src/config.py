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
CLEAN_INSCRIPTIONS = PROCESSED_DATA / "clean_data.csv"
CLEAN_EVALUATIONS = PROCESSED_DATA / "evaluations_clean.csv"
CLEAN_ACTIVITES = PROCESSED_DATA / "activites_virtuelles_clean.csv"

# Notebooks directory
NOTEBOOKS = ROOT / "notebooks"

"""Repository-relative paths for confidential local data and aggregate outputs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"

CLEAN_INSCRIPTIONS = PROCESSED_DATA / "clean_inscriptions.parquet"
CLEAN_EVALUATIONS = PROCESSED_DATA / "clean_evaluations.parquet"
CLEAN_ACTIVITES = PROCESSED_DATA / "clean_activities.parquet"

LONGITUDINAL_PANEL = PROCESSED_DATA / "longitudinal_panel.parquet"
MODEL_FEATURES = PROCESSED_DATA / "model_features.parquet"
OUTPUT = ROOT / "output"
DATA_SUMMARY = OUTPUT / "data_summary.json"
J30_RESULTS = OUTPUT / "j30_results.json"

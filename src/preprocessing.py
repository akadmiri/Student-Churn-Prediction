"""J30 eligibility, student-grouped partitions and model preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLEAN_EVALUATIONS, CLEAN_INSCRIPTIONS, LONGITUDINAL_PANEL

BASE_NUM = [
    "age", "volume_per_active_day_30d", "days_since_last_activity",
    "n_evaluations_30d", "avg_note_cum", "avg_lateness_cum",
]
CAT = ["niveau_scolaire"]
FLAGS = ["grade_missing", "lateness_missing"]
FEATURES = [*BASE_NUM, *CAT, *FLAGS]
TARGET = "churn"
SEED = 10


def split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition the J30 panel by student before applying enrollment eligibility."""
    outer = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(data, groups=data.id_etudiant))
    combined = data.iloc[train_val_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED)
    train_idx, val_idx = next(inner.split(combined, groups=combined.id_etudiant))
    return (
        combined.iloc[train_idx].copy(),
        combined.iloc[val_idx].copy(),
        data.iloc[test_idx].copy(),
    )


def j30_eligibility(data: pd.DataFrame) -> pd.Series:
    """Identify enrollments started and not yet cancelled at programme J30."""
    cutoff = data.date_debut_programme + pd.Timedelta(days=30)
    return (
        data.date_debut.notna() & data.date_debut.le(cutoff)
        & (data.date_annulation.isna() | data.date_annulation.gt(cutoff))
    )


def prepare(data: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    """Filter an existing partition and compute assessment means with missingness.

    Cumulative means retain pre-programme submissions and use the exact J30
    timestamp cutoff, rather than the panel's integer relative-day comparison.
    """
    eligible = data.loc[j30_eligibility(data)].copy()
    submitted = evaluations.merge(
        eligible[["id_inscription", "date_debut_programme"]],
        on="id_inscription", how="inner",
    )
    submitted = submitted.loc[
        submitted.date_soumission.notna()
        & submitted.date_soumission.le(
            submitted.date_debut_programme + pd.Timedelta(days=30)
        )
    ].copy()
    submitted["lateness"] = (
        submitted.date_soumission - submitted.date_echeance
    ).dt.days
    grades = submitted.groupby("id_inscription").note.mean()
    delays = submitted.groupby("id_inscription").lateness.mean()
    eligible["avg_note_cum"] = eligible.id_inscription.map(grades)
    eligible["avg_lateness_cum"] = eligible.id_inscription.map(delays)
    eligible["grade_missing"] = eligible.avg_note_cum.isna().astype("int8")
    eligible["lateness_missing"] = eligible.avg_lateness_cum.isna().astype("int8")
    return eligible


def assemble_j30(panel: pd.DataFrame, inscriptions: pd.DataFrame) -> pd.DataFrame:
    """Attach final labels and eligibility dates without filtering or sorting rows."""
    columns = [
        "id_inscription", "churn", "date_debut", "date_debut_programme",
        "date_annulation",
    ]
    return panel.loc[panel.checkpoint_day.eq(30)].drop(
        columns=["target_next_30d"]
    ).merge(inscriptions[columns], on="id_inscription", validate="one_to_one")


def load_j30() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load J30 rows and evaluations from the confidential local tables."""
    inscriptions = pd.read_parquet(CLEAN_INSCRIPTIONS)
    evaluations = pd.read_parquet(CLEAN_EVALUATIONS)
    panel = pd.read_parquet(LONGITUDINAL_PANEL)
    return assemble_j30(panel, inscriptions), evaluations


def make_preprocessor(
    scale_numeric: bool = False, *, include_missingness: bool = True,
) -> ColumnTransformer:
    """Transform numeric, dense categorical, then missingness columns in order."""
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric.append(("scale", StandardScaler()))
    transformers = [
        ("num", Pipeline(numeric), BASE_NUM),
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            CAT,
        ),
    ]
    if include_missingness:
        transformers.append(("flags", "passthrough", FLAGS))
    return ColumnTransformer(transformers)

"""Select features, split students, and prepare model preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import PROCESSED_DATA

NUM: list[str] = [
    "age",
    "volume_per_active_day_30d",
    "days_since_last_activity",
    "n_evaluations_30d",
    "avg_note_cum",
    "avg_lateness_cum",
]
CAT: list[str] = ["niveau_scolaire"]
FEATURES: list[str] = ["checkpoint_day", *NUM, "active_days_diff", *CAT]
TARGET: str = "target_next_30d"


def load_and_split(
    test_size: float = 0.25,
    val_size: float = 0.20,
    random_state: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load the panel and split by student (60% train, 15% validation, 25% test)."""
    data = pd.read_parquet(PROCESSED_DATA / "longitudinal_panel.parquet")

    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(split.split(data, groups=data["id_etudiant"]))
    train_val = data.iloc[train_val_idx]
    test = data.iloc[test_idx]

    split_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(split_val.split(train_val, groups=train_val["id_etudiant"]))
    train = train_val.iloc[train_idx]
    val = train_val.iloc[val_idx]

    return (
        train[FEATURES].copy(),
        val[FEATURES].copy(),
        test[FEATURES].copy(),
        train[TARGET].copy(),
        val[TARGET].copy(),
        test[TARGET].copy(),
    )


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build transformations fitted only on the training data by each model pipeline."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    activity_diff_steps = [("impute", SimpleImputer(strategy="constant", fill_value=0))]
    if scale_numeric:
        activity_diff_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), ["checkpoint_day", *NUM]),
            ("diff", Pipeline(activity_diff_steps), ["active_days_diff"]),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CAT,
            ),
        ]
    )

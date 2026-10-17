"""Build a J30/J60/J90 panel for prospective 30-day abandonment prediction."""

import numpy as np
import pandas as pd
from config import CLEAN_ACTIVITES, CLEAN_EVALUATIONS, CLEAN_INSCRIPTIONS, PROCESSED_DATA

CHECKPOINTS: tuple[int, ...] = (30, 60, 90)
HORIZON: int = 30
STATIC: list[str] = [
    "niveau_scolaire", "annee_scolaire", "semestre",
    "nb_inscriptions_precedentes", "age", "genre", "region", "situation_handicap",
]
FEATURES: list[str] = [
    "checkpoint_day", *STATIC, "volume_per_active_day_30d",
    "days_since_last_activity", "active_days_diff",
    "n_evaluations_30d", "avg_note_cum", "avg_lateness_cum",
]


def build_longitudinal_features(
    inscriptions: pd.DataFrame,
    activities: pd.DataFrame,
    evaluations: pd.DataFrame,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Build checkpoint features using only records dated by the prediction cutoff."""
    ins = inscriptions.copy()
    ins["date_debut_programme"] = pd.to_datetime(ins["date_debut_programme"])
    ins["date_annulation"] = pd.to_datetime(ins["date_annulation"])
    ins["cancel_day"] = (ins["date_annulation"] - ins["date_debut_programme"]).dt.days

    act = activities.merge(
        ins[["id_inscription", "date_debut_programme"]], on="id_inscription", how="inner"
    )
    act["rel_day"] = (pd.to_datetime(act["horodatage"]) - act["date_debut_programme"]).dt.days

    ev = evaluations.merge(
        ins[["id_inscription", "date_debut_programme"]], on="id_inscription", how="inner"
    )
    ev["sub_day"] = (pd.to_datetime(ev["date_soumission"]) - ev["date_debut_programme"]).dt.days
    ev["lateness_days"] = (
        pd.to_datetime(ev["date_soumission"]) - pd.to_datetime(ev["date_echeance"])
    ).dt.days

    frames: list[pd.DataFrame] = []
    for t in checkpoints:
        risk = ins[ins["cancel_day"].isna() | (ins["cancel_day"] > t)].copy()
        if risk.empty:
            continue

        risk["checkpoint_day"] = t
        risk["target_next_30d"] = (
            (risk["resultat_final"] == "Abandon")
            & risk["cancel_day"].between(t + 1, t + horizon)
        ).astype("int8")
        panel = risk[["id_inscription", "id_etudiant", "id_programme", "checkpoint_day", *STATIC, "target_next_30d"]].copy()
        ids = panel["id_inscription"]

        history = act[act["id_inscription"].isin(ids) & act["rel_day"].between(0, t)]
        recent = history[history["rel_day"] > t - 30]
        n_days = recent.groupby("id_inscription")["rel_day"].nunique()
        volume = recent.groupby("id_inscription")["volume"].sum()
        intensity = volume.div(n_days).rename("volume_per_active_day_30d")
        last_day = history.groupby("id_inscription")["rel_day"].max()

        panel["volume_per_active_day_30d"] = ids.map(intensity).fillna(0.0)
        panel["days_since_last_activity"] = (t - ids.map(last_day)).fillna(t).astype("int16")

        if t == 30:
            panel["active_days_diff"] = np.nan
        else:
            previous = history[history["rel_day"].between(t - 60, t - 30, inclusive="right")]
            prev_days = previous.groupby("id_inscription")["rel_day"].nunique()
            panel["active_days_diff"] = (
                ids.map(n_days).fillna(0) - ids.map(prev_days).fillna(0)
            )

        submitted = ev[
            ev["id_inscription"].isin(ids)
            & ev["sub_day"].notna()
            & (ev["sub_day"] <= t)
        ]
        recent_ev = submitted[submitted["sub_day"] > t - 30]
        n_evals = recent_ev.groupby("id_inscription")["id_evaluation"].count()
        grades = submitted.groupby("id_inscription")["note"].mean()
        lateness = submitted.groupby("id_inscription")["lateness_days"].mean()

        panel["n_evaluations_30d"] = ids.map(n_evals).fillna(0).astype("int16")
        panel["avg_note_cum"] = ids.map(grades).fillna(0.0)
        panel["avg_lateness_cum"] = ids.map(lateness).fillna(0.0)
        frames.append(panel)

    if not frames:
        raise ValueError("No eligible enrollments at the selected checkpoints")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Create the panel and a feature-only matrix for modeling."""
    inscriptions = pd.read_parquet(CLEAN_INSCRIPTIONS)
    activities = pd.read_parquet(CLEAN_ACTIVITES)
    evaluations = pd.read_parquet(CLEAN_EVALUATIONS)

    panel = build_longitudinal_features(inscriptions, activities, evaluations)
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DATA / "longitudinal_panel.parquet", index=False)
    panel[FEATURES].to_parquet(PROCESSED_DATA / "model_features.parquet", index=False)
    print(f"Panel: {panel.shape} | positive prevalence: {panel['target_next_30d'].mean():.4%}")


if __name__ == "__main__":
    main()

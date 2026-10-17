"""Longitudinal person-period feature engineering for discrete-time hazard churn modeling."""

import numpy as np
import pandas as pd
from typing import Final
from pathlib import Path

from config import (
    CLEAN_ACTIVITES,
    CLEAN_EVALUATIONS,
    CLEAN_INSCRIPTIONS,
    PROCESSED_DATA,
)

CHECKPOINTS: Final[tuple[int, ...]] = (30, 60, 90)
HORIZON: Final[int] = 30
HIGH_INTENT_ACTIVITIES: Final[frozenset[str]] = frozenset(
    {"Classe virtuelle", "Quiz", "Contenu de cours", "Video", "Ressource", "pdf"}
)


def build_longitudinal_features(
    inscriptions: pd.DataFrame,
    activities: pd.DataFrame,
    evaluations: pd.DataFrame,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Construct person-period panel dataset at checkpoints t_k with prospective binary target.
    """
    ins = inscriptions.copy()
    ins["date_debut_programme"] = pd.to_datetime(ins["date_debut_programme"])
    ins["date_annulation"] = pd.to_datetime(ins["date_annulation"])
    ins["cancel_day"] = (ins["date_annulation"] - ins["date_debut_programme"]).dt.days

    act = activities.merge(ins[["id_inscription", "date_debut_programme"]], on="id_inscription", how="inner")
    act["rel_day"] = (pd.to_datetime(act["horodatage"]) - act["date_debut_programme"]).dt.days
    act["is_high_intent"] = act["type_activite"].isin(HIGH_INTENT_ACTIVITIES)

    ev = evaluations.merge(ins[["id_inscription", "date_debut_programme"]], on="id_inscription", how="inner")
    ev["sub_day"] = (pd.to_datetime(ev["date_soumission"]) - ev["date_debut_programme"]).dt.days
    ev["due_day"] = (pd.to_datetime(ev["date_echeance"]) - ev["date_debut_programme"]).dt.days
    ev["lateness_days"] = ev["sub_day"] - ev["due_day"]

    panel_rows = []

    for tk in checkpoints:
        # Risk set: active at tk (cancel_day is null or strictly greater than tk)
        at_risk = ins[ins["cancel_day"].isna() | (ins["cancel_day"] > tk)].copy()
        if at_risk.empty:
            continue

        # Prospective target: churn in (tk, tk + horizon]
        at_risk["target_next_30d"] = (
            (at_risk["resultat_final"] == "Abandon")
            & (at_risk["cancel_day"].notna())
            & (at_risk["cancel_day"] <= tk + horizon)
        ).astype("int8")

        at_risk["checkpoint_day"] = tk

        # Window 1: recent 30d (tk - 30, tk]
        w_start, w_end = tk - 30, tk
        act_w1 = act[act["id_inscription"].isin(at_risk["id_inscription"]) & act["rel_day"].between(w_start, w_end, inclusive="right")]
        
        daily_w1 = act_w1.groupby(["id_inscription", "rel_day"], as_index=False)["volume"].sum()
        agg_w1 = daily_w1.groupby("id_inscription").agg(
            active_days_30d=("rel_day", "nunique"),
            volume_30d=("volume", "sum"),
        )
        
        high_vol_w1 = act_w1[act_w1["is_high_intent"]].groupby("id_inscription")["volume"].sum().rename("high_intent_vol_30d")
        agg_w1 = agg_w1.join(high_vol_w1, how="left").fillna({"high_intent_vol_30d": 0.0})
        agg_w1["high_intent_share_30d"] = np.where(agg_w1["volume_30d"] > 0, agg_w1["high_intent_vol_30d"] / agg_w1["volume_30d"], 0.0)
        agg_w1["volume_per_active_day_30d"] = np.where(agg_w1["active_days_30d"] > 0, agg_w1["volume_30d"] / agg_w1["active_days_30d"], 0.0)

        # Window 2: previous 30d (tk - 60, tk - 30] for trajectory comparison
        pw_start, pw_end = tk - 60, tk - 30
        act_w2 = act[act["id_inscription"].isin(at_risk["id_inscription"]) & act["rel_day"].between(pw_start, pw_end, inclusive="right")]
        agg_w2 = act_w2.groupby("id_inscription")["rel_day"].nunique().rename("active_days_prev_30d")
        agg_w1 = agg_w1.join(agg_w2, how="left").fillna({"active_days_prev_30d": 0.0})
        agg_w1["active_days_diff"] = agg_w1["active_days_30d"] - agg_w1["active_days_prev_30d"]

        # Intra-month velocity ratio (second half vs first half of recent 30d)
        mid_w = w_start + 15
        vol_h1 = act_w1[act_w1["rel_day"] <= mid_w].groupby("id_inscription")["volume"].sum()
        vol_h2 = act_w1[act_w1["rel_day"] > mid_w].groupby("id_inscription")["volume"].sum()
        vel_df = pd.DataFrame({"vol_h1": vol_h1, "vol_h2": vol_h2}).fillna(0.0)
        agg_w1["velocity_ratio_30d"] = (vel_df["vol_h2"] + 1.0) / (vel_df["vol_h1"] + 1.0)

        # Evaluations cumulative up to tk
        ev_cum = ev[ev["id_inscription"].isin(at_risk["id_inscription"]) & (ev["sub_day"].notna()) & (ev["sub_day"] <= tk)]
        ev_agg = ev_cum.groupby("id_inscription").agg(
            n_evals_cum=("id_evaluation", "count"),
            avg_note_cum=("note", "mean"),
            avg_lateness_cum=("lateness_days", "mean"),
        )

        due_cum = ev[ev["id_inscription"].isin(at_risk["id_inscription"]) & (ev["due_day"].notna()) & (ev["due_day"] <= tk)]
        missed_cum = due_cum.assign(missed=due_cum["sub_day"].isna() | (due_cum["sub_day"] > tk)).groupby("id_inscription")["missed"].sum().rename("missed_evals_cum")
        ev_agg = ev_agg.join(missed_cum, how="left").fillna({"missed_evals_cum": 0})

        # Merge features onto at_risk cohort
        sub_df = at_risk.merge(agg_w1[["active_days_30d", "volume_30d", "volume_per_active_day_30d", "high_intent_share_30d", "active_days_diff", "velocity_ratio_30d"]], on="id_inscription", how="left")
        sub_df = sub_df.merge(ev_agg, on="id_inscription", how="left")

        # Fill defaults for zero activity/evaluations
        sub_df["active_days_30d"] = sub_df["active_days_30d"].fillna(0.0)
        sub_df["volume_30d"] = sub_df["volume_30d"].fillna(0.0)
        sub_df["volume_per_active_day_30d"] = sub_df["volume_per_active_day_30d"].fillna(0.0)
        sub_df["high_intent_share_30d"] = sub_df["high_intent_share_30d"].fillna(0.0)
        sub_df["active_days_diff"] = sub_df["active_days_diff"].fillna(0.0)
        sub_df["velocity_ratio_30d"] = sub_df["velocity_ratio_30d"].fillna(1.0)
        sub_df["n_evals_cum"] = sub_df["n_evals_cum"].fillna(0).astype("int16")
        sub_df["missed_evals_cum"] = sub_df["missed_evals_cum"].fillna(0).astype("int16")
        sub_df["avg_note_cum"] = sub_df["avg_note_cum"].fillna(0.0)
        sub_df["avg_lateness_cum"] = sub_df["avg_lateness_cum"].fillna(0.0)
        sub_df["has_submitted_eval_cum"] = (sub_df["n_evals_cum"] > 0).astype("int8")

        panel_rows.append(sub_df)

    panel_df = pd.concat(panel_rows, ignore_index=True)
    return panel_df


def main() -> None:
    inscriptions = pd.read_parquet(CLEAN_INSCRIPTIONS)
    activities = pd.read_parquet(CLEAN_ACTIVITES)
    evaluations = pd.read_parquet(CLEAN_EVALUATIONS)

    panel = build_longitudinal_features(inscriptions, activities, evaluations)
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA / "longitudinal_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Longitudinal panel saved with shape: {panel.shape} | Positive prevalence: {panel['target_next_30d'].mean():.4f}")


if __name__ == "__main__":
    main()
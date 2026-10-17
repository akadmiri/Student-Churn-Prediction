"""Leakage-conscious 30-day person-period features for YOOL abandonment hazards.

CAVEAT: Non-abandonment is treated as observed through J+120 because reliable
individual completion/end timestamps are not available.
"""

from typing import Final
import numpy as np
import pandas as pd
from config import CLEAN_ACTIVITES, CLEAN_EVALUATIONS, CLEAN_INSCRIPTIONS, PROCESSED_DATA

CHECKPOINTS: Final[tuple[int, ...]] = (30, 60, 90)
HORIZON: Final[int] = 30
HIGH_INTENT: Final[frozenset[str]] = frozenset({"Classe virtuelle", "Quiz", "Contenu de cours", "Video", "Ressource", "pdf"})
KEYS: Final[list[str]] = ["id_inscription", "id_etudiant", "id_programme", "checkpoint_day"]
STATIC: Final[list[str]] = ["niveau_scolaire", "annee_scolaire", "semestre", "nb_inscriptions_precedentes", "age", "genre", "region", "situation_handicap"]
NUMERIC: Final[list[str]] = [
    "delai_inscription", "pre_start_volume", "active_days_30d", "volume_30d",
    "volume_per_active_day_30d", "high_intent_share_30d", "days_since_last_activity",
    "max_inactive_streak_30d", "active_days_prev_30d", "active_days_diff",
    "second_half_share_30d", "n_evals_cum", "n_scored_evals_cum",
    "avg_note_cum_experimental", "n_dated_submissions_cum",
    "avg_positive_lateness_cum", "late_submission_rate_cum", "overdue_recorded_evals_cum",
    "has_submitted_eval_cum", "has_graded_eval_cum",
]
FEATURE_COLUMNS: Final[list[str]] = ["checkpoint_day", *STATIC, *NUMERIC]


def _longest_zero_run(days: set[int], start: int, end: int) -> int:
    """Longest inactive streak in a 30-day window, with integer calendar days."""
    streak = maximum = 0
    for day in range(start + 1, end + 1):
        streak = 0 if day in days else streak + 1
        maximum = max(maximum, streak)
    return maximum


def build_longitudinal_features(
    inscriptions: pd.DataFrame,
    activities: pd.DataFrame,
    evaluations: pd.DataFrame,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Create snapshot features using records known by each checkpoint only.

    Graduation/completion time is unavailable; documented assumption: the
    non-abandon cohort stays at risk through all specified prediction horizons.
    Scores are tagged experimental as publication timestamps are unavailable.
    """
    if not checkpoints or horizon <= 0 or any(t < 30 for t in checkpoints):
        raise ValueError("Checkpoints must start at J+30; horizon must be positive")
    ins = inscriptions.copy()
    ins["date_debut_programme"] = pd.to_datetime(ins["date_debut_programme"])
    ins["date_annulation"] = pd.to_datetime(ins["date_annulation"])
    ins["date_debut"] = pd.to_datetime(ins["date_debut"])
    ins["cancel_day"] = (ins["date_annulation"] - ins["date_debut_programme"]).dt.days
    if ((ins["resultat_final"] == "Abandon") & ins["cancel_day"].isna()).any():
        raise ValueError("Unconfirmed abandonment dates must be excluded or censored in cleaning")
    if ins["id_inscription"].duplicated().any():
        raise ValueError("Duplicate enrollment IDs")
    if not set(STATIC).issubset(ins.columns):
        raise ValueError(f"Missing static variables: {set(STATIC)-set(ins.columns)}")

    act = activities.merge(ins[["id_inscription", "date_debut_programme"]], on="id_inscription", how="inner", validate="many_to_one")
    act["rel_day"] = (pd.to_datetime(act["horodatage"]) - act["date_debut_programme"]).dt.days
    act["high_intent_volume"] = act["volume"].where(act["type_activite"].isin(HIGH_INTENT), 0)
    ev = evaluations.merge(ins[["id_inscription", "date_debut_programme"]], on="id_inscription", how="inner", validate="many_to_one")
    ev["sub_day"] = (pd.to_datetime(ev["date_soumission"]) - ev["date_debut_programme"]).dt.days
    ev["due_day"] = (pd.to_datetime(ev["date_echeance"]) - ev["date_debut_programme"]).dt.days
    ev["lateness_days"] = ev["sub_day"] - ev["due_day"]

    frames: list[pd.DataFrame] = []
    for t in checkpoints:
        # Use outcome metadata only to define risk set and label; do not export it.
        risk = ins.loc[ins["cancel_day"].isna() | (ins["cancel_day"] > t)].copy()
        if risk.empty:
            continue
        risk["target_next_30d"] = ((risk["resultat_final"] == "Abandon") & risk["cancel_day"].between(t + 1, t + horizon)).astype("int8")
        risk["checkpoint_day"] = t
        risk["delai_inscription"] = (risk["date_debut_programme"] - risk["date_debut"]).dt.days
        # Whitelist prevents cancellation dates, final outcome and enrollment churn leakage.
        panel = risk[KEYS + STATIC + ["target_next_30d", "delai_inscription"]].copy()
        index = pd.Index(panel["id_inscription"], name="id_inscription")
        a = act[act["id_inscription"].isin(index) & (act["rel_day"] <= t)]
        pre = a[a["rel_day"] < 0].groupby("id_inscription")["volume"].sum()
        recent = a[(a["rel_day"] > t - 30) & (a["rel_day"] <= t)]
        previous = a[(a["rel_day"] > t - 60) & (a["rel_day"] <= t - 30)] if t >= 60 else a.iloc[:0]
        recent_days = recent.groupby("id_inscription")["rel_day"].agg(lambda x: set(x.astype(int)))
        n_active = recent.groupby("id_inscription")["rel_day"].nunique().reindex(index).fillna(0)
        recent_vol = recent.groupby("id_inscription")["volume"].sum().reindex(index).fillna(0)
        high_vol = recent.groupby("id_inscription")["high_intent_volume"].sum().reindex(index).fillna(0)
        prev_active = previous.groupby("id_inscription")["rel_day"].nunique().reindex(index).fillna(0)
        last_activity = recent.groupby("id_inscription")["rel_day"].max().reindex(index)
        half_1 = recent[recent["rel_day"] <= t - 15].groupby("id_inscription")["volume"].sum().reindex(index).fillna(0)
        half_2 = recent[recent["rel_day"] > t - 15].groupby("id_inscription")["volume"].sum().reindex(index).fillna(0)
        total_half = half_1 + half_2
        # No activity: share is undefined, not stable trend.
        half_share = half_2.div(total_half.replace(0, np.nan))
        # No activity: exactly 30 days of within-window inactivity.
        recency = (t - last_activity).fillna(30)
        streak = pd.Series({idx: _longest_zero_run(recent_days.get(idx, set()), t - 30, t) for idx in index}, name="max_inactive_streak_30d")
        for col, series in {
            "pre_start_volume": pre.reindex(index).fillna(0),
            "active_days_30d": n_active,
            "volume_30d": recent_vol,
            "volume_per_active_day_30d": recent_vol.div(n_active.replace(0, np.nan)).fillna(0),
            "high_intent_share_30d": high_vol.div(recent_vol.replace(0, np.nan)).fillna(0),
            "days_since_last_activity": recency,
            "max_inactive_streak_30d": streak.reindex(index),
            "active_days_prev_30d": prev_active if t >= 60 else pd.Series(np.nan, index=index),
            "active_days_diff": n_active - prev_active if t >= 60 else pd.Series(np.nan, index=index),
            "second_half_share_30d": half_share,
        }.items():
            panel[col] = series.to_numpy()

        relevant_ev = ev[ev["id_inscription"].isin(index)]
        submitted = relevant_ev[relevant_ev["sub_day"].notna() & (relevant_ev["sub_day"] <= t)]
        dated = submitted[submitted["due_day"].notna()].copy()
        due = relevant_ev[relevant_ev["due_day"].notna() & (relevant_ev["due_day"] <= t)].copy()
        due["still_overdue"] = due["sub_day"].isna() | (due["sub_day"] > t)
        graded = submitted[submitted["note"].notna()]
        n_submit = submitted.groupby("id_inscription")["id_evaluation"].count().reindex(index).fillna(0)
        n_graded = graded.groupby("id_inscription")["id_evaluation"].count().reindex(index).fillna(0)
        n_dated = dated.groupby("id_inscription")["id_evaluation"].count().reindex(index).fillna(0)
        late_rate = dated.assign(late=dated["lateness_days"] > 0).groupby("id_inscription")["late"].mean().reindex(index)
        avg_positive = dated.assign(late_positive=dated["lateness_days"].clip(lower=0)).groupby("id_inscription")["late_positive"].mean().reindex(index)
        averages = graded.groupby("id_inscription")["note"].mean().reindex(index)
        overdue = due.groupby("id_inscription")["still_overdue"].sum().reindex(index).fillna(0)
        for col, series in {
            "n_evals_cum": n_submit,
            "n_scored_evals_cum": n_graded,
            "avg_note_cum_experimental": averages,
            "n_dated_submissions_cum": n_dated,
            "avg_positive_lateness_cum": avg_positive,
            "late_submission_rate_cum": late_rate,
            "overdue_recorded_evals_cum": overdue,
            "has_submitted_eval_cum": (n_submit > 0).astype("int8"),
            "has_graded_eval_cum": (n_graded > 0).astype("int8"),
        }.items():
            panel[col] = series.to_numpy()
        frames.append(panel)

    if not frames:
        raise ValueError("No eligible enrollments")
    output = pd.concat(frames, ignore_index=True)
    output["target_next_30d"] = output["target_next_30d"].astype("int8")
    return output


def validate_panel(panel: pd.DataFrame, inscriptions: pd.DataFrame) -> None:
    """Raise if the panel violates basic grain, target or activity invariants."""
    assert not panel.duplicated(["id_inscription", "checkpoint_day"]).any(), "Duplicate panel keys"
    assert set(panel["target_next_30d"].unique()) <= {0, 1}
    assert not {"resultat_final", "churn", "cancel_day", "date_annulation"}.intersection(panel.columns)
    assert panel["active_days_30d"].between(0, 30).all()
    assert panel["days_since_last_activity"].between(0, 30).all()
    assert panel["max_inactive_streak_30d"].between(0, 30).all()
    assert (panel.loc[panel["active_days_30d"] == 0, "volume_30d"] == 0).all()
    assert panel["high_intent_share_30d"].between(0, 1).all()
    assert panel["second_half_share_30d"].dropna().between(0, 1).all()
    assert panel.loc[panel["checkpoint_day"] == 30, "active_days_diff"].isna().all()
    assert (panel.loc[panel["checkpoint_day"] >= 60, "active_days_diff"] == (panel.loc[panel["checkpoint_day"] >= 60, "active_days_30d"] - panel.loc[panel["checkpoint_day"] >= 60, "active_days_prev_30d"])).all()
    meta = inscriptions[["id_inscription", "resultat_final", "date_debut_programme", "date_annulation"]].copy()
    meta["cancel_day"] = (pd.to_datetime(meta["date_annulation"]) - pd.to_datetime(meta["date_debut_programme"])).dt.days
    merged = panel[["id_inscription", "checkpoint_day", "target_next_30d"]].merge(meta, on="id_inscription", validate="many_to_one")
    expected = ((merged["resultat_final"] == "Abandon") & (merged["cancel_day"] > merged["checkpoint_day"]) & (merged["cancel_day"] <= merged["checkpoint_day"] + HORIZON)).astype("int8")
    assert (merged["target_next_30d"] == expected).all(), "Target differs from confirmed cancellation time"
    assert (merged["cancel_day"].isna() | (merged["cancel_day"] > merged["checkpoint_day"])).all(), "Ineligible at-risk row"


def main() -> None:
    """Build and save the panel plus a strict predictor-column manifest."""
    inscriptions = pd.read_parquet(CLEAN_INSCRIPTIONS)
    activities = pd.read_parquet(CLEAN_ACTIVITES)
    evaluations = pd.read_parquet(CLEAN_EVALUATIONS)
    panel = build_longitudinal_features(inscriptions, activities, evaluations)
    validate_panel(panel, inscriptions)
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DATA / "longitudinal_panel.parquet", index=False)
    # Safe feature matrix without identifiers, labels, and experimental grades.
    safe_features = [c for c in FEATURE_COLUMNS if c != "avg_note_cum_experimental"]
    panel[safe_features].to_parquet(PROCESSED_DATA / "model_features_safe.parquet", index=False)
    print(f"Panel: {panel.shape} | positive labels: {int(panel['target_next_30d'].sum())} | prevalence: {panel['target_next_30d'].mean():.4%}")
    print("NOTE: negative follow-up assumes non-abandon enrollments remained at risk through J+120; completion dates unavailable.")
    print("NOTE: grades are experimental and excluded from model_features_safe.parquet.")


if __name__ == "__main__":
    main()

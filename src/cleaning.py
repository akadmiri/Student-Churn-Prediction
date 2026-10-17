from pathlib import Path
import numpy as np
import pandas as pd
from config import (
    RAW_DATA,
    PROCESSED_DATA,
    CLEAN_INSCRIPTIONS,
    CLEAN_EVALUATIONS,
    CLEAN_ACTIVITES,
)


def load_tables(path: Path = RAW_DATA) -> dict[str, pd.DataFrame]:
    """Load all raw CSV files with explicit schema dtypes and parsed dates."""
    etudiants = pd.read_csv(
        path / "etudiants.csv",
        dtype={
            "id_etudiant": "int32",
            "genre": "category",
            "region": "category",
            "situation_handicap": "int8",
            "age": "int8",
        },
    )
    programmes = pd.read_csv(
        path / "programmes.csv",
        dtype={
            "id_programme": "string",
            "niveau_scolaire": "category",
            "annee_scolaire": "category",
            "semestre": "category",
            "filiere": "string",
        },
    )
    inscriptions = pd.read_csv(
        path / "inscriptions.csv",
        parse_dates=["date_debut", "date_annulation"],
        dtype={
            "id_inscription": "int32",
            "id_etudiant": "int32",
            "id_programme": "string",
            "resultat_final": "category",
            "nb_inscriptions_precedentes": "int8",
        },
    )
    evaluations = pd.read_csv(
        path / "evaluations.csv",
        parse_dates=["date_echeance", "date_soumission"],
        dtype={
            "id_evaluation": "int32",
            "id_inscription": "int32",
            "type_evaluation": "category",
            "note": "float32",
        },
    )
    activites_virtuelles = pd.read_csv(
        path / "activites_virtuelles.csv",
        parse_dates=["horodatage"],
        dtype={
            "id_activite": "int32",
            "id_inscription": "int32",
            "type_activite": "category",
            "volume": "int32",
        },
    )
    return {
        "etudiants": etudiants,
        "programmes": programmes,
        "inscriptions": inscriptions,
        "evaluations": evaluations,
        "activites_virtuelles": activites_virtuelles,
    }


def merge_tables(
    inscriptions: pd.DataFrame,
    etudiants: pd.DataFrame,
    programmes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drop 'filiere', derive 'date_debut_programme' (T_0), and merge inscriptions
    with etudiants and programmes.
    """
    programmes = programmes.drop(columns=["filiere"]).copy()

    annees = programmes["annee_scolaire"].astype(str).str.split("/", expand=True)
    programmes["date_debut_programme"] = pd.to_datetime(
        np.where(
            programmes["semestre"] == "Semestre 1",
            annees[0] + "-09-01",
            annees[1] + "-01-01",
        )
    )

    merged = inscriptions.merge(etudiants, on="id_etudiant", how="left")
    merged = merged.merge(programmes, on="id_programme", how="left")
    return merged


def get_ids(inscriptions: pd.DataFrame) -> pd.Index:
    """
    Get inscription IDs to exclude:
    1. Pre-start churn or temporal impossibility: date_annulation < date_debut_programme
       or date_annulation < date_debut.
    2. Label/date mismatch: resultat_final != 'Abandon' with a date_annulation present.
    3. Unconfirmed event time: resultat_final == 'Abandon' with missing date_annulation.
    """
    has_cancel_date = inscriptions["date_annulation"].notna()
    abandon = inscriptions["resultat_final"] == "Abandon"

    temporal_impossibility = has_cancel_date & (
        (inscriptions["date_annulation"] < inscriptions["date_debut"])
        | (inscriptions["date_annulation"] < inscriptions["date_debut_programme"])
    )
    label_date_mismatch = (~abandon) & has_cancel_date
    missing_abandon_date = abandon & (~has_cancel_date)

    exclude_mask = temporal_impossibility | label_date_mismatch | missing_abandon_date
    excluded_ids = inscriptions.loc[exclude_mask, "id_inscription"]
    print(
        f"Excluding {len(excluded_ids)} inscriptions "
        f"(pre-start/impossible: {temporal_impossibility.sum()}, "
        f"label mismatch: {label_date_mismatch.sum()}, "
        f"unconfirmed abandon date: {missing_abandon_date.sum()})."
    )
    return pd.Index(excluded_ids.unique())


def get_evaluation_ids(
    evaluations: pd.DataFrame, inscriptions: pd.DataFrame
) -> pd.Index:
    """
    Identify evaluation IDs with temporal anomalies relative to date_debut:
    1. date_soumission < date_debut
    2. date_echeance < date_debut
    """
    dates = inscriptions[["id_inscription", "date_debut"]]
    merged = evaluations.merge(dates, on="id_inscription", how="left")

    soumission_anomaly = merged["date_soumission"] < merged["date_debut"]
    echeance_anomaly = merged["date_echeance"] < merged["date_debut"]
    exclude_mask = soumission_anomaly | echeance_anomaly
    excluded_ids = merged.loc[exclude_mask, "id_evaluation"]
    print(f"Excluding {len(excluded_ids)} evaluations due to temporal anomalies.")
    return pd.Index(excluded_ids.unique())


def exclude(
    inscriptions: pd.DataFrame,
    evaluations: pd.DataFrame,
    activites_virtuelles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclude invalid inscriptions across all tables and remove anomalous evaluations."""
    excluded_ids = get_ids(inscriptions)

    inscriptions_filtered = inscriptions[
        ~inscriptions["id_inscription"].isin(excluded_ids)
    ].copy()
    evaluations_filtered = evaluations[
        ~evaluations["id_inscription"].isin(excluded_ids)
    ].copy()
    activites_virtuelles_filtered = activites_virtuelles[
        ~activites_virtuelles["id_inscription"].isin(excluded_ids)
    ].copy()

    evaluation_ids = get_evaluation_ids(evaluations_filtered, inscriptions_filtered)
    evaluations_filtered = evaluations_filtered[
        ~evaluations_filtered["id_evaluation"].isin(evaluation_ids)
    ].copy()

    return (
        inscriptions_filtered,
        evaluations_filtered,
        activites_virtuelles_filtered,
    )


def target(inscriptions: pd.DataFrame) -> pd.DataFrame:
    """Create enrollment-level binary indicator 'churn' (1 = Abandon, 0 = Persisted)."""
    inscriptions = inscriptions.copy()
    inscriptions["churn"] = (
        inscriptions["resultat_final"] == "Abandon"
    ).astype("int8")
    return inscriptions


def main() -> None:
    tables = load_tables()

    merged_inscriptions = merge_tables(
        tables["inscriptions"],
        tables["etudiants"],
        tables["programmes"],
    )

    inscriptions_filtered, evaluations_filtered, activites_virtuelles_filtered = (
        exclude(
            merged_inscriptions,
            tables["evaluations"],
            tables["activites_virtuelles"],
        )
    )

    clean_data = target(inscriptions_filtered)

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    clean_data.to_parquet(CLEAN_INSCRIPTIONS, index=False)
    evaluations_filtered.to_parquet(CLEAN_EVALUATIONS, index=False)
    activites_virtuelles_filtered.to_parquet(CLEAN_ACTIVITES, index=False)
    print(
        f"Saved clean tables: inscriptions={clean_data.shape}, "
        f"evaluations={evaluations_filtered.shape}, "
        f"activites={activites_virtuelles_filtered.shape}"
    )


if __name__ == "__main__":
    main()
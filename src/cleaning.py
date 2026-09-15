import pandas as pd
from pathlib import Path
from config import (
    RAW_DATA,
    PROCESSED_DATA,
    CLEAN_INSCRIPTIONS,
    CLEAN_EVALUATIONS,
    CLEAN_ACTIVITES,
)

def load_tables(path: Path = RAW_DATA) -> dict[str, pd.DataFrame]:
    """Load all CSV files in the specified directory into a dictionary of DataFrames."""
    tables = {}
    for file in path.glob("*.csv"):
        tables[file.stem] = pd.read_csv(file)
    return tables

def get_ids(inscription: pd.DataFrame) -> pd.Index:
    '''
    Get the ids to exclude from the inscription table according the conclusion
     found in the inspection notebook.
           1. Temporal impossibility: date_annulation < date_debut.
           2. Label/date mismatch: resultat_final != 'Abandon' with a
           date_annulation present.
    '''
    inscriptions = inscription.copy()

    # Transform the date columns to datetime format for comparison
    inscriptions['date_debut'] = pd.to_datetime(inscriptions['date_debut'])
    inscriptions['date_annulation'] = pd.to_datetime(inscriptions['date_annulation'])

    # Define the conditions for exclusion
    has_cancel_date = inscriptions['date_annulation'].notna()
    abandon = inscriptions['resultat_final'] == 'Abandon'

    temporal_impossibility = has_cancel_date & (inscriptions['date_annulation'] < inscriptions['date_debut'])
    label_date_mismatch = (~abandon & has_cancel_date)

    exclude = temporal_impossibility | label_date_mismatch
    excluded_ids = inscriptions.loc[exclude, 'id_inscription']
    print(f"Excluding {len(excluded_ids)} inscriptions due to temporal impossibility or label/date mismatch.")
    return pd.Index(excluded_ids.unique())

def get_evaluation_ids(evaluations: pd.DataFrame, inscriptions: pd.DataFrame) -> pd.Index:
    '''
    Identify evaluation ids with temporal anomalies relative to date_debut,
    found in 02_observation_window.ipynb:
          1. date_soumission < date_debut
          2. date_echeance < date_debut
    '''
    evaluations = evaluations.copy()
    evaluations['date_soumission'] = pd.to_datetime(evaluations['date_soumission'])
    evaluations['date_echeance'] = pd.to_datetime(evaluations['date_echeance'])
    
    dates = inscriptions[['id_inscription', 'date_debut']].copy()
    dates['date_debut'] = pd.to_datetime(dates['date_debut'])
    merged = evaluations.merge(dates, on='id_inscription', how='left')

    soumission_anomaly = merged['date_soumission'] < merged['date_debut']
    echeance_anomaly = merged['date_echeance'] < merged['date_debut']
    exclude = soumission_anomaly | echeance_anomaly
    excluded_ids = merged.loc[exclude, 'id_evaluation']
    print(f"Excluding {len(excluded_ids)} evaluations due to temporal anomalies.")
    return pd.Index(excluded_ids.unique())

def exclude(
    inscriptions: pd.DataFrame,
    evaluations: pd.DataFrame,
    activites_virtuelles:pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Exclude the inscriptions that are not valid according to the inspection notebook.
    and the evaluations with temporal anomalies found in the observation window notebook.
    '''
    excluded_ids = get_ids(inscriptions)

    # Exclude the invalid inscriptions from all relevant tables
    inscriptions_filtered = inscriptions[~inscriptions['id_inscription'].isin(excluded_ids)]
    evaluations_filtered = evaluations[~evaluations['id_inscription'].isin(excluded_ids)]
    activites_virtuelles_filtered = activites_virtuelles[~activites_virtuelles['id_inscription'].isin(excluded_ids)]

    # Exclude evaluations with temporal anomalies
    evaluation_ids = get_evaluation_ids(evaluations_filtered, inscriptions_filtered)
    evaluations_filtered = evaluations_filtered[~evaluations_filtered['id_evaluation'].isin(evaluation_ids)]
    return inscriptions_filtered, evaluations_filtered, activites_virtuelles_filtered

def merge_tables(
    inscriptions: pd.DataFrame,
    etudiants: pd.DataFrame,
    programmes: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Merge the inscriptions table with the etudiants and programmes tables.
    '''
    # Drop 'filiere' column from programmes first according to conclusion of the inspection notebook
    programmes = programmes.drop(columns=['filiere'])

    merged = inscriptions.merge(etudiants, on='id_etudiant', how='left')
    merged = merged.merge(programmes, on='id_programme', how='left')

    return merged

def target(inscriptions: pd.DataFrame) -> pd.DataFrame:
    '''
    Create a target variable based on the 'resultat_final' column in the inscriptions table.
    The target variable will be 1 for 'Abandon' and 0 for all other values.
    '''
    inscriptions = inscriptions.copy()
    inscriptions['churn'] = (inscriptions['resultat_final'] == 'Abandon').astype(int)
    return inscriptions

def main():
    # Load all tables from the raw data directory
    tables = load_tables()

    # Exclude invalid inscriptions and related records
    inscriptions_filtered, evaluations_filtered, activites_virtuelles_filtered = exclude(
        tables['inscriptions'],
        tables['evaluations'],
        tables['activites_virtuelles']
    )

    # Merge the filtered inscriptions with etudiants and programmes
    merged_data = merge_tables(
        inscriptions_filtered,
        tables['etudiants'],
        tables['programmes']
    )

    # Create the target variable
    clean_data = target(merged_data)

    # Save the processed data to the processed data directory
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    clean_data.to_csv(CLEAN_INSCRIPTIONS, index=False)
    evaluations_filtered.to_csv(CLEAN_EVALUATIONS, index=False)
    activites_virtuelles_filtered.to_csv(CLEAN_ACTIVITES, index=False)

if __name__ == "__main__":
    main()
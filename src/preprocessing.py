import pandas as pd
from pathlib import Path

raw_directory = Path("data/raw")
processed_directory = Path("data/processed")

def load_tables(path: Path = raw_directory) -> dict[str, pd.DataFrame]:
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
           2. Label/date mismatch: resultat_final == 'Abandon' with no
           date_annulation, or resultat_final != 'Abandon' with a
           date_annulation present.
    '''
    inscriptions = inscription.copy()

    # Transform the date columns to datetime format for comparison
    inscriptions['date_debut'] = pd.to_datetime(inscriptions['date_debut'])
    inscriptions['date_annulation'] = pd.to_datetime(inscriptions['date_annulation'])

    # Define the conditions for exclusion
    has_cancel_date = inscriptions['date_annulation'].notna()
    abandon = inscriptions['resultat_final'] == 'Abandon'

    temporal_impossibility = inscriptions['date_annulation'] < inscriptions['date_debut']
    label_date_mismatch = (abandon & ~has_cancel_date) | (~abandon & has_cancel_date)

    exclude = temporal_impossibility | label_date_mismatch
    excluded_ids = inscriptions.loc[exclude, 'id_inscription']
    print(f"Excluding {len(excluded_ids)} inscriptions due to temporal impossibility or label/date mismatch.")
    return pd.Index(excluded_ids.unique())

def exclude(
    inscriptions: pd.DataFrame,
    evaluations: pd.DataFrame,
    activites_virtuelles:pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Exclude the inscriptions that are not valid according to the inspection notebook.
    '''
    excluded_ids = get_ids(inscriptions)

    # Exclude the invalid inscriptions from all relevant tables
    inscriptions_filtered = inscriptions[~inscriptions['id_inscription'].isin(excluded_ids)]
    evaluations_filtered = evaluations[~evaluations['id_inscription'].isin(excluded_ids)]
    activites_virtuelles_filtered = activites_virtuelles[~activites_virtuelles['id_inscription'].isin(excluded_ids)]

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
    inscriptions['target'] = (inscriptions['resultat_final'] == 'Abandon').astype(int)
    return inscriptions

def main():
    # Load all tables from the raw data directory
    tables = load_tables(raw_directory)

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
    processed_directory.mkdir(parents=True, exist_ok=True)
    clean_data.to_csv(processed_directory / "clean_data.csv", index=False)
    evaluations_filtered.to_csv(processed_directory / "evaluations_clean.csv", index=False)
    activites_virtuelles_filtered.to_csv(processed_directory / "activites_virtuelles_clean.csv", index=False)
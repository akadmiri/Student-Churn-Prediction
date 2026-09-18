import pandas as pd
from config import (
    FEATURES,
    DATASET,

)

def main():
    data = pd.read_csv(FEATURES)
    # Constants
    insignificant = ['genre','cohorte']
    redundant_cols = ['id_programme','resultat_final','date_annulation']
    missing_cols = ['avg_note','avg_lateness_days']
    correlated = ['distinct_active_days']
    cols_to_drop = insignificant + redundant_cols + missing_cols + correlated

    # Drop insignificant and correlated columns
    data.drop(columns=cols_to_drop, inplace=True)
    # Save the final dataset
    data.to_csv(DATASET, index=False)
    print(f"Dataset saved with shape: {data.shape}")

if __name__ == "__main__":
    main()
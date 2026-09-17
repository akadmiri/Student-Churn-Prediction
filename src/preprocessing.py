import pandas as pd
from config import CLEAN_INSCRIPTIONS, FINAL_INSCRIPTIONS


# Drop non significant variables according to the descriptive stats notebook
data = pd.read_csv(CLEAN_INSCRIPTIONS)
insignificant = ['genre','cohorte']
redundant_cols = ['id_programme','resultat_final','date_debut','date_annulation', ]
missing_cols = ['avg_note','avg_lateness_days']
cols_to_drop = insignificant + redundant_cols + missing_cols
data.drop(cols_to_drop,axis='columns', inplace=True)
data.to_csv(FINAL_INSCRIPTIONS, index=False)
import pandas as pd
from config import (
    CLEAN_INSCRIPTIONS,
    CLEAN_ACTIVITES,
    CLEAN_EVALUATIONS,
    PROCESSED_DATA,
    FEATURES,
)

def elapsed_days(
    events: pd.DataFrame, inscriptions: pd.DataFrame, date_col: str
) -> pd.Series:
    """Compute days elapsed between event date and enrollment start date."""
    merged = events[['id_inscription', date_col]].merge(
        inscriptions[['id_inscription', 'date_debut']],
        on='id_inscription',
        how='left',
    )
    return (
        pd.to_datetime(merged[date_col]) - pd.to_datetime(merged['date_debut'])
    ).dt.days


def build_activity_features(
    activities: pd.DataFrame, inscriptions: pd.DataFrame, T_obs: int
) -> pd.DataFrame:
    """
    Extract core engagement metrics within [0, T_obs].
    Features: total_volume, distinct_active_days, cv_volume, days_since_last_activity.
    """
    act = activities.copy()
    act['days_since_start'] = elapsed_days(act, inscriptions, 'horodatage')

    # Restrict to the valid observation window
    window = act[act['days_since_start'].between(0, T_obs)]

    # General activity aggregations
    features = (
        window.groupby('id_inscription')
        .agg(
            total_volume=('volume', 'sum'),
            distinct_active_days=('days_since_start', 'nunique'),
            last_activity_day=('days_since_start', 'max'),
        )
        .reset_index()
    )

    # Recency penalty: larger values mean longer inactivity prior to cutoff
    features['days_since_last_activity'] = T_obs - features['last_activity_day']
    features = features.drop(columns=['last_activity_day'])
    return features


def build_evaluation_features(
    evaluations: pd.DataFrame, inscriptions: pd.DataFrame, T_obs: int
) -> pd.DataFrame:
    """
    Extract academic assessment metrics submitted within [0, T_obs].
    Features: n_evaluations, avg_note, avg_lateness_days.
    """
    ev = evaluations.copy()
    ev['sub_days'] = elapsed_days(ev, inscriptions, 'date_soumission')
    ev['due_days'] = elapsed_days(ev, inscriptions, 'date_echeance')

    # Only include assignments actually submitted during the observation window
    window = ev[ev['sub_days'].between(0, T_obs)].copy()
    window['lateness_days'] = window['sub_days'] - window['due_days']

    features = (
        window.groupby('id_inscription')
        .agg(
            n_evaluations=('id_evaluation', 'count'),
            avg_note=('note', 'mean'),
            avg_lateness_days=('lateness_days', 'mean'),
        )
        .reset_index()
    )

    return features


def build_feature_matrix(
    inscriptions: pd.DataFrame,
    activities: pd.DataFrame,
    evaluations: pd.DataFrame,
    T_obs: int = 60,
) -> pd.DataFrame:
    """Assemble base demographic and enrollment data with windowed features."""
    df = inscriptions.copy()

    act_feats = build_activity_features(activities, inscriptions, T_obs)
    eval_feats = build_evaluation_features(evaluations, inscriptions, T_obs)

    df = df.merge(act_feats, on='id_inscription', how='left')
    df = df.merge(eval_feats, on='id_inscription', how='left')

    # defaults
    df['total_volume'] = df['total_volume'].fillna(0)
    df['distinct_active_days'] = df['distinct_active_days'].fillna(0)
    df['days_since_last_activity'] = df['days_since_last_activity'].fillna(
        T_obs
    )
    df['n_evaluations'] = df['n_evaluations'].fillna(0)

    # avg_note and avg_lateness_days remain NaN when n_evaluations == 0
    return df


def main():
    inscriptions = pd.read_csv(CLEAN_INSCRIPTIONS)
    activities = pd.read_csv(CLEAN_ACTIVITES)
    evaluations = pd.read_csv(CLEAN_EVALUATIONS)

    matrix = build_feature_matrix(inscriptions, activities, evaluations, T_obs=60)

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(FEATURES, index=False)
    print(f'Features generated with shape: {matrix.shape}')


if __name__ == '__main__':
    main()
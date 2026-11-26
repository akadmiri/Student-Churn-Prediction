"""Model comparison, evaluation and interpretation for the J30 experiment."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, log_loss,
    precision_recall_curve, precision_score, recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import FEATURES, SEED, TARGET, make_preprocessor


def pipe(
    estimator: Any, *, scale: bool = False, include_missingness: bool = True,
) -> Pipeline:
    """Fit all preprocessing inside the model's training pipeline."""
    return Pipeline([
        ("preprocess", make_preprocessor(
            scale_numeric=scale, include_missingness=include_missingness,
        )),
        ("model", estimator),
    ])


def describe(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize an enrollment population without exporting identifiers."""
    return {
        "n": len(data),
        "students": int(data.id_etudiant.nunique()),
        "positives": int(data[TARGET].sum()),
        "prevalence": float(data[TARGET].mean()) if len(data) else None,
    }


def forest(*, include_missingness: bool = True) -> Pipeline:
    """Build the selected Random Forest with its fixed hyperparameters."""
    return pipe(
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features="sqrt",
            random_state=SEED, n_jobs=-1,
        ),
        include_missingness=include_missingness,
    )


def threshold(y: pd.Series, p: np.ndarray) -> float | None:
    """Maximize recall at precision >=30%; break ties by precision, then threshold."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    candidates = pd.DataFrame({
        "threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1],
    })
    candidates = candidates.loc[candidates.precision.ge(.30)]
    if candidates.empty:
        return None
    return float(candidates.sort_values(
        ["recall", "precision", "threshold"], ascending=[False, False, False]
    ).iloc[0].threshold)


def scores(y: pd.Series, p: np.ndarray, tau: float | None = None) -> dict[str, Any]:
    """Calculate AP (not trapezoidal PR-AUC), probability and classification metrics."""
    result = {
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    if tau is not None:
        predicted = p >= tau
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        result.update(
            threshold=tau,
            precision=float(precision_score(y, predicted, zero_division=0)),
            recall=float(recall_score(y, predicted, zero_division=0)),
            f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.,
            accuracy=float((tn + tp) / len(y)),
            alerts=int(fp + tp),
            confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        )
    return result


def diagnostics(y: pd.Series, p: np.ndarray) -> dict[str, Any]:
    """Return binned calibration, threshold and gains statistics, not predictions."""
    labels = np.asarray(y, dtype=int)
    if not len(labels) or not labels.sum():
        raise ValueError("Gains require a nonempty population with positive cases.")
    calibration = []
    # Omit small bins rather than exposing individual scores.
    bin_ids = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        mask = bin_ids == i
        if mask.sum() >= 20:
            calibration.append({
                "lower": i / 10, "upper": (i + 1) / 10, "n": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "observed_rate": float(labels[mask].mean()),
            })
    grid = []
    for cutoff in np.linspace(0, 1, 101):
        mask = p >= cutoff
        if mask.sum() >= 20:
            positives = int(labels[mask].sum())
            grid.append({
                "threshold": float(cutoff), "alerts": int(mask.sum()),
                "precision": positives / int(mask.sum()),
                "recall": positives / int(labels.sum()),
            })
    gains = []
    # Fractional allocation of ties avoids dependence on row order.
    ranked = (
        pd.DataFrame({"score": p, "target": labels}).groupby("score")
        .target.agg(["size", "sum"]).sort_index(ascending=False)
    )
    for fraction in [.05, .10, .15, .20, .25, .30, .40, .50, .60, .70, .80, .90, 1.]:
        k = min(len(labels), max(1, int(np.ceil(fraction * len(labels)))))
        remaining, positives_selected = k, 0.
        for n, positives in ranked.itertuples(index=False, name=None):
            take = min(remaining, n)
            positives_selected += take * positives / n
            remaining -= take
            if not remaining:
                break
        gains.append({
            "fraction": k / len(labels),
            "recall": positives_selected / int(labels.sum()),
            "precision": positives_selected / k,
            "lift": (positives_selected / k) / float(labels.mean()),
        })
    return {"calibration_bins": calibration, "threshold_grid": grid, "gains": gains}


def evaluate_forest(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
) -> tuple[dict[str, Any], Pipeline]:
    """Train, select a validation threshold, refit, and evaluate the fixed forest."""
    if any(part[TARGET].nunique() != 2 for part in (train, validation, test)):
        raise ValueError("Each evaluation partition must contain both target classes.")
    model = forest().fit(train[FEATURES], train[TARGET])
    validation_probabilities = model.predict_proba(validation[FEATURES])[:, 1]
    cutoff = threshold(validation[TARGET], validation_probabilities)
    combined = pd.concat([train, validation])
    final = clone(model).fit(combined[FEATURES], combined[TARGET])
    test_probabilities = final.predict_proba(test[FEATURES])[:, 1]
    result = {
        "train": describe(train), "validation": describe(validation), "test": describe(test),
        "validation_metrics": scores(validation[TARGET], validation_probabilities, cutoff),
        "test_metrics": scores(test[TARGET], test_probabilities, cutoff),
        "baseline_brier": float(brier_score_loss(
            test[TARGET], np.full(len(test), combined[TARGET].mean())
        )),
        "validation_diagnostics": diagnostics(validation[TARGET], validation_probabilities),
        "test_diagnostics": diagnostics(test[TARGET], test_probabilities),
    }
    return result, final


def compare_models(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    """Compare model families and the decision-tree grid on validation only."""
    estimators = {
        "logistic": (LogisticRegression(max_iter=1000, random_state=SEED), True),
        "logistic_balanced": (LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED,
        ), True),
        "tree_default": (DecisionTreeClassifier(random_state=SEED), False),
        "tree_balanced": (DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=5, class_weight="balanced", random_state=SEED,
        ), False),
        "svm": (SVC(random_state=SEED), True),
        "svm_balanced": (SVC(class_weight="balanced", random_state=SEED), True),
        "gradient_boosting": (HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.05, max_leaf_nodes=15,
            l2_regularization=1., random_state=SEED,
        ), True),
    }
    results = {}
    for name, (estimator, scale) in estimators.items():
        model = pipe(estimator, scale=scale).fit(train[FEATURES], train[TARGET])
        if isinstance(estimator, SVC):
            metric = {"ap": float(average_precision_score(
                validation[TARGET], model.decision_function(validation[FEATURES])
            ))}
        else:
            metric = scores(validation[TARGET], model.predict_proba(validation[FEATURES])[:, 1])
        results[name] = {"metrics": metric, "parameters": estimator.get_params()}
    candidates = []
    for depth in [3, 5, 7, 10, None]:
        for leaf in [1, 5, 10, 20]:
            estimator = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=leaf, random_state=SEED,
            )
            model = pipe(estimator).fit(train[FEATURES], train[TARGET])
            candidates.append({
                "metrics": scores(validation[TARGET], model.predict_proba(validation[FEATURES])[:, 1]),
                "parameters": estimator.get_params(),
            })
    results["tree_tuned"] = max(candidates, key=lambda candidate: candidate["metrics"]["ap"])
    return results


def validation_importance(tr: pd.DataFrame, va: pd.DataFrame) -> list[dict[str, Any]]:
    """Measure AP decreases on validation without selecting new predictors."""
    model = forest().fit(tr[FEATURES], tr[TARGET])
    importance = permutation_importance(
        model, va[FEATURES], va[TARGET], scoring="average_precision",
        n_repeats=15, random_state=SEED, n_jobs=1,
    )
    return [
        {"feature": column, "mean_ap_decrease": float(mean), "sd": float(sd)}
        for column, mean, sd in zip(
            FEATURES, importance.importances_mean, importance.importances_std
        )
    ]


def temporal_partitions(
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Hold out 2024/2025 and remove its students from the earlier cohort."""
    early = eligible.loc[eligible.annee_scolaire.eq("2023/2024")].copy()
    later = eligible.loc[eligible.annee_scolaire.eq("2024/2025")].copy()
    shared = set(early.id_etudiant) & set(later.id_etudiant)
    early = early.loc[~early.id_etudiant.isin(shared)]
    if early.id_etudiant.nunique() < 2:
        return early.iloc[:0], early, later, len(shared)
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED).split(
            early, groups=early.id_etudiant
        )
    )
    return early.iloc[train_idx], early.iloc[val_idx], later, len(shared)


def evaluate_temporal(eligible: pd.DataFrame) -> dict[str, Any]:
    """Repeat the forest protocol with an independent temporal validation threshold."""
    train, validation, test, shared = temporal_partitions(eligible)
    if not all(part[TARGET].nunique() == 2 for part in (train, validation, test)):
        return {"note": "At least one temporal partition lacks both classes; no metrics calculated."}
    result, _ = evaluate_forest(train, validation, test)
    result["shared_students_removed_from_earlier_year"] = shared
    return result


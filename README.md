# Student Churn Prediction

Engineering internship project at **YOOL Education**, completed during the third year at **INSEA** in *Économie appliquée, Statistique et Big Data*.

The objective is to predict **subsequent recorded student abandonment** using information available at J30 (30 days after the programme start), and to explore how statistical learning can support educational follow-up.

## Dataset and confidentiality

The dataset covers the 2023/2024 and 2024/2025 academic years through five related tables:

| File | Contents |
|---|---|
| `etudiants.csv` | Student characteristics. |
| `programmes.csv` | School level, academic year and semester. |
| `inscriptions.csv` | Enrollments, dates and recorded outcomes. |
| `activites_virtuelles.csv` | Learning activities, timestamps and volume. |
| `evaluations.csv` | Grades, submission dates and due dates. |

Of **7,169** initial enrollments, **7,024** remain after cleaning and **6,626** are eligible at J30, including **1,204** subsequent abandonments.

These educational data are confidential and are not distributed. Full numerical reproduction requires authorized access to the dataset. Raw and processed data, individual predictions and trained models must remain private. Local outputs are excluded from Git; only approved aggregate results and figures may be shared.

## Methodological workflow

1. **Data inspection — Notebook 01:** examine keys, relationships, missing values and date consistency, and document cleaning decisions.
2. **Observation window — Notebook 02:** study activity coverage and abandonment timing. J30 provides an empirically motivated compromise between available information and early intervention.
3. **Longitudinal exploration — Notebook 03:** describe observations at J30, J60 and J90 using `target_next_30d`, abandonment during the following 30 days. This exploratory target is distinct from the final target.
4. **J30 modeling — Notebook 04:** predict `churn`, subsequent recorded abandonment. Split the J30 panel by student, then exclude enrollments with missing start dates, enrollments starting after J30, and those already cancelled by that date. Compare models and missing-value treatments, select a threshold on validation, and evaluate calibration, permutation importance, cumulative gains and performance across academic years.

The final partitions contain **3,952 training**, **1,005 validation** and **1,669 test** observations, with no students shared between them. Missing numerical values are imputed using training-set medians; two indicators identify unavailable average grades and delays. School level is imputed by the training mode and one-hot encoded.

## Final explanatory variables

| Variable | Meaning at J30 |
|---|---|
| `age` | Recorded student age. |
| `volume_per_active_day_30d` | Activity volume per active day on relative days 1–30. |
| `days_since_last_activity` | Days since the latest activity on days 0–30; 30 when absent. |
| `n_evaluations_30d` | Number of submitted assessments on days 1–30. |
| `avg_note_cum` | Mean available grade for submissions at or before J30. |
| `avg_lateness_cum` | Mean submission delay relative to the due date, in days. |
| `niveau_scolaire` | Programme school level. |
| `grade_missing` | Indicator of an unavailable average grade. |
| `lateness_missing` | Indicator of an unavailable average submission delay. |

Cumulative assessment means include submissions at or before the exact J30 timestamp, including eligible pre-programme submissions. Recent counts and activity windows use integer relative days. Grades and due dates are assumed to be available at submission; missing grades and delays are distinguished from genuine zeros.

## Model selection and results

The comparison includes logistic regression, decision trees, SVM, Random Forest and histogram gradient boosting, with class-weighted variants where applicable. Models are ranked by **Average Precision (AP)**, which differs from trapezoidal precision–recall AUC.

The weighted SVM achieves the highest validation AP (**0.3550**). The selected Random Forest performs similarly (**0.3536**) and has the lowest Brier score and log-loss among the compared models providing probabilities. It uses **300 trees**, `min_samples_leaf=10`, `max_features="sqrt"`, `random_state=10` and no class weighting.

The threshold **0.1976794** maximizes validation recall subject to precision of at least **30%**. It remains fixed after refitting on training and validation together.

| Metric | Validation | Test |
|---|---:|---:|
| Average Precision | 0.3535674 | 0.3415786 |
| Precision | 0.3010471 | 0.2966102 |
| Recall | 0.6117021 | 0.5833333 |
| F1-score | 0.4035088 | 0.3932584 |
| Brier score | 0.1394743 | 0.1357810 |
| Log-loss | 0.4417650 | 0.4376094 |

The test prevalence of **0.1797484** provides a random-ranking reference for AP. A constant probability equal to training-and-validation prevalence gives a test Brier score of **0.1474457**. Confusion counts are **TN=954, FP=415, FN=125, TP=175**.

Zero encoding of unavailable assessment means gives validation AP **0.3515450** and recall **0.5585106**. That comparison retains the longitudinal panel’s integer-day means, while the final treatment uses the exact J30 cutoff; it does not isolate imputation alone.

Retrospective evaluation on 2024/2025 yields AP **0.3383258**, precision **0.4039301** and recall **0.2530780**, using a threshold selected on the preceding year. On the main test set, examining approximately 10% of enrollments with the highest scores captures 75 of 300 recorded abandonments (25% recall, 44.91% precision).

## Repository structure

- `notebooks/`: the four analyses, from data inspection to final modeling.
- `src/cleaning.py` and `src/features.py`: data preparation and longitudinal features.
- `src/preprocessing.py` and `src/modeling.py`: J30 preparation, model comparison and evaluation.
- `src/config.py`: local data and output paths.
- `pyproject.toml` and `uv.lock`: environment definition and locked dependencies.

The local `data/raw/`, `data/processed/` and `output/` folders are excluded from Git.

## Installation and execution

Use Python **3.14** and `uv`. From the repository root:

```bash
uv sync --locked
```

Place the five authorized CSV files in `data/raw/`, then prepare the data and open JupyterLab:

```bash
uv run --locked python -m src.cleaning
uv run --locked python -m src.features
uv run --locked --with jupyterlab --with ipykernel jupyter lab
```

Execute `01_data_inspection.ipynb`, `02_observation_window.ipynb`, `03_data_exploration.ipynb` and `04_model_selection.ipynb` in that order, running all cells in a fresh Python kernel for each notebook. Clear all notebook outputs before saving files for publication. The original CSV files remain unchanged.

## Limitations

J30 is a practical checkpoint, not an experimentally proven optimal window. Recorded abandonment may lag actual disengagement. The dataset excludes enrollments with no activity over the entire recorded period, which can introduce selection bias at J30. Official programme end dates are unavailable, and activity traces alone do not establish complete follow-up. Grade availability at submission and due-date availability are assumptions. Repeated observations limit the interpretation of longitudinal associations.

Ranking performance is modest and false positives are frequent. The validation precision constraint does not guarantee the same precision on new data. Exploratory feature selection used the full panel, including students in the final test set. Neither evaluation is a fully independent external validation, and the retrospective year holdout does not establish prospective performance or intervention effectiveness.

## Author

**Abderrahmane KADMIRI** — third-year engineering student, **Institut National de Statistique et d’Économie Appliquée (INSEA)**, *Économie appliquée, Statistique et Big Data*.

Application internship at **YOOL Education**, 2 July–15 August 2026, academic year 2025–2026.

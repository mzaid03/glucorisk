from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes_2021_2023_screening.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "glucorisk_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as metadata_file:
    metadata = json.load(metadata_file)


features = metadata["features"]
threshold = metadata["decision_threshold"]

missing_features = [
    feature
    for feature in features
    if feature not in data.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )


X_new = data[features]
y_new = data["elevated_a1c"].astype(int)
survey_weights = data["sample_weight"]

probabilities = model.predict_proba(X_new)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


def calculate_metrics(
    actual,
    predicted,
    probability,
    sample_weight=None,
):
    return {
        "accuracy": accuracy_score(
            actual,
            predicted,
            sample_weight=sample_weight,
        ),
        "precision": precision_score(
            actual,
            predicted,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predicted,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            predicted,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            actual,
            probability,
            sample_weight=sample_weight,
        ),
        "brier_score": brier_score_loss(
            actual,
            probability,
            sample_weight=sample_weight,
        ),
    }


unweighted_metrics = calculate_metrics(
    y_new,
    predictions,
    probabilities,
)

weighted_metrics = calculate_metrics(
    y_new,
    predictions,
    probabilities,
    sample_weight=survey_weights,
)


historical_metrics = metadata["test_metrics"]

comparison = pd.DataFrame(
    [
        {
            "evaluation": "2017-2018 held-out test",
            **historical_metrics,
        },
        {
            "evaluation": "2021-2023 temporal test",
            **{
                key: unweighted_metrics[key]
                for key in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                ]
            },
        },
    ]
).set_index("evaluation")


tn, fp, fn, tp = confusion_matrix(
    y_new,
    predictions,
    labels=[0, 1],
).ravel()

unweighted_prevalence = y_new.mean()

weighted_prevalence = np.average(
    y_new,
    weights=survey_weights,
)

unweighted_prediction_rate = predictions.mean()

weighted_prediction_rate = np.average(
    predictions,
    weights=survey_weights,
)


print("Temporal validation cohort")
print(f"Participants: {len(data):,}")

print(
    "Observed elevated-A1C rate: "
    f"{unweighted_prevalence:.3f}"
)

print(
    "Survey-weighted elevated-A1C rate: "
    f"{weighted_prevalence:.3f}"
)

print(
    "Predicted higher-risk rate: "
    f"{unweighted_prediction_rate:.3f}"
)

print(
    "Survey-weighted predicted higher-risk rate: "
    f"{weighted_prediction_rate:.3f}"
)

print(f"\nFrozen decision threshold: {threshold:.3f}")

print("\nHistorical versus temporal performance:")
print(comparison.round(3))

print("\n2021-2023 unweighted metrics:")
print(
    pd.Series(unweighted_metrics).round(3)
)

print("\n2021-2023 survey-weighted metrics:")
print(
    pd.Series(weighted_metrics).round(3)
)

print("\n2021-2023 confusion matrix:")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")
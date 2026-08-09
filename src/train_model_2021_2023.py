from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


RANDOM_STATE = 42
TARGET_RECALL = 0.75

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes_2021_2023_screening.csv"
)

CANDIDATE_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "candidates"
)

CANDIDATE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


numeric_features = [
    "age",
    "bmi",
    "waist_cm",
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "recreation_met_minutes_week",
    "sedentary_minutes",
    "average_sleep_hours",
]

categorical_features = [
    "smoking_status",
]

features = (
    numeric_features
    + categorical_features
)


data = pd.read_csv(DATA_PATH)

X = data[features]
y = data["elevated_a1c"].astype(int)
weights = data["sample_weight"]


# 70% training, 15% validation, 15% final test
(
    X_train,
    X_temporary,
    y_train,
    y_temporary,
    weights_train,
    weights_temporary,
) = train_test_split(
    X,
    y,
    weights,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

(
    X_validation,
    X_test,
    y_validation,
    y_test,
    weights_validation,
    weights_test,
) = train_test_split(
    X_temporary,
    y_temporary,
    weights_temporary,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temporary,
)


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent",
        ),
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
        ),
    ),
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features,
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features,
    ),
])

model = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
])


# Normalize weights so their average is one.
normalized_training_weights = (
    weights_train / weights_train.mean()
)

model.fit(
    X_train,
    y_train,
    classifier__sample_weight=(
        normalized_training_weights
    ),
)


# Tune the threshold using validation data only.
validation_probabilities = (
    model.predict_proba(X_validation)[:, 1]
)

(
    validation_precision,
    validation_recall,
    thresholds,
) = precision_recall_curve(
    y_validation,
    validation_probabilities,
    sample_weight=weights_validation,
)

valid_indices = np.where(
    validation_recall[:-1] >= TARGET_RECALL
)[0]

if len(valid_indices) == 0:
    raise ValueError(
        "No threshold reached the target recall."
    )

best_index = valid_indices[
    np.argmax(
        validation_precision[:-1][valid_indices]
    )
]

decision_threshold = thresholds[best_index]


test_probabilities = (
    model.predict_proba(X_test)[:, 1]
)

test_predictions = (
    test_probabilities
    >= decision_threshold
).astype(int)


def calculate_metrics(
    actual,
    predictions,
    probabilities,
    sample_weight=None,
):
    return {
        "accuracy": accuracy_score(
            actual,
            predictions,
            sample_weight=sample_weight,
        ),
        "precision": precision_score(
            actual,
            predictions,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predictions,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            predictions,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            actual,
            probabilities,
            sample_weight=sample_weight,
        ),
        "brier_score": brier_score_loss(
            actual,
            probabilities,
            sample_weight=sample_weight,
        ),
    }


unweighted_metrics = calculate_metrics(
    y_test,
    test_predictions,
    test_probabilities,
)

weighted_metrics = calculate_metrics(
    y_test,
    test_predictions,
    test_probabilities,
    sample_weight=weights_test,
)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()


candidate_model_path = (
    CANDIDATE_DIRECTORY
    / "glucorisk_model_2021_2023.joblib"
)

candidate_metadata_path = (
    CANDIDATE_DIRECTORY
    / "model_metadata_2021_2023.json"
)

joblib.dump(
    model,
    candidate_model_path,
)

candidate_metadata = {
    "model_name": "GlucoRisk",
    "version": "2.0.0-candidate",
    "dataset": "NHANES August 2021-August 2023",
    "model_type": "Survey-weighted Logistic Regression",
    "decision_threshold": float(
        decision_threshold
    ),
    "target_recall": TARGET_RECALL,
    "features": features,
    "excluded_sensitive_features": [
        "sex_code",
        "race_ethnicity_code",
    ],
    "split_sizes": {
        "training": len(X_train),
        "validation": len(X_validation),
        "test": len(X_test),
    },
    "test_metrics_unweighted": {
        key: float(value)
        for key, value
        in unweighted_metrics.items()
    },
    "test_metrics_survey_weighted": {
        key: float(value)
        for key, value
        in weighted_metrics.items()
    },
}

with open(
    candidate_metadata_path,
    "w",
    encoding="utf-8",
) as metadata_file:
    json.dump(
        candidate_metadata,
        metadata_file,
        indent=4,
    )


print("Dataset split sizes:")
print(f"Training:   {len(X_train):,}")
print(f"Validation: {len(X_validation):,}")
print(f"Test:       {len(X_test):,}")

print(
    "\nSelected decision threshold: "
    f"{decision_threshold:.3f}"
)

print("\nUnweighted test metrics:")
print(
    pd.Series(unweighted_metrics).round(3)
)

print("\nSurvey-weighted test metrics:")
print(
    pd.Series(weighted_metrics).round(3)
)

print("\nTest confusion matrix:")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")

print(f"\nCandidate model saved to: {candidate_model_path}")
print(
    "Candidate metadata saved to: "
    f"{candidate_metadata_path}"
)
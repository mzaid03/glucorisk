from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

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
    / "candidates"
    / "glucorisk_model_2021_2023.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "model_metadata_2021_2023.json"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

FIGURE_DIRECTORY = (
    REPORT_DIRECTORY
    / "figures"
)

REPORT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
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

X = data[features]
y = data["elevated_a1c"].astype(int)
weights = data["sample_weight"]


# Recreate the exact split used during training.
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


test_probabilities = (
    model.predict_proba(X_test)[:, 1]
)

test_predictions = (
    test_probabilities >= threshold
).astype(int)


# --------------------------------------------------
# Calibration
# --------------------------------------------------

observed_rate, predicted_rate = calibration_curve(
    y_test,
    test_probabilities,
    n_bins=8,
    strategy="quantile",
)

model_brier = brier_score_loss(
    y_test,
    test_probabilities,
)

baseline_probabilities = np.full(
    len(y_test),
    y_train.mean(),
)

baseline_brier = brier_score_loss(
    y_test,
    baseline_probabilities,
)

plt.figure(figsize=(7, 6))

plt.plot(
    predicted_rate,
    observed_rate,
    marker="o",
    label="GlucoRisk 2.0 candidate",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Perfect calibration",
)

plt.xlabel("Average predicted score")
plt.ylabel("Observed elevated-A1C rate")
plt.title("2021–2023 Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

calibration_path = (
    FIGURE_DIRECTORY
    / "calibration_2021_2023.png"
)

plt.savefig(
    calibration_path,
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Subgroup auditing
# --------------------------------------------------

audit_data = data.loc[X_test.index].copy()

audit_data["actual"] = np.asarray(y_test)
audit_data["prediction"] = test_predictions
audit_data["probability"] = test_probabilities

audit_data["sex"] = audit_data["sex_code"].map({
    1.0: "Male",
    2.0: "Female",
})

audit_data["race_ethnicity"] = (
    audit_data["race_ethnicity_code"].map({
        1.0: "Mexican American",
        2.0: "Other Hispanic",
        3.0: "Non-Hispanic White",
        4.0: "Non-Hispanic Black",
        6.0: "Non-Hispanic Asian",
        7.0: "Other / Multiracial",
    })
)


def calculate_group_metrics(
    frame,
    group_column,
):
    rows = []

    for group_name, group in frame.groupby(
        group_column,
        dropna=False,
    ):
        actual = group["actual"]
        predictions = group["prediction"]
        probabilities = group["probability"]
        group_weights = group["sample_weight"]

        weighted_confusion = confusion_matrix(
            actual,
            predictions,
            labels=[0, 1],
            sample_weight=group_weights,
        )

        weighted_tn, weighted_fp, _, _ = (
            weighted_confusion.ravel()
        )

        weighted_auc = (
            roc_auc_score(
                actual,
                probabilities,
                sample_weight=group_weights,
            )
            if actual.nunique() == 2
            else np.nan
        )

        rows.append({
            "group": group_name,
            "sample_size": len(group),
            "elevated_cases": int(actual.sum()),
            "recall": recall_score(
                actual,
                predictions,
                zero_division=0,
            ),
            "weighted_recall": recall_score(
                actual,
                predictions,
                sample_weight=group_weights,
                zero_division=0,
            ),
            "weighted_precision": precision_score(
                actual,
                predictions,
                sample_weight=group_weights,
                zero_division=0,
            ),
            "weighted_false_positive_rate": (
                weighted_fp
                / (weighted_fp + weighted_tn)
                if (weighted_fp + weighted_tn) > 0
                else np.nan
            ),
            "weighted_roc_auc": weighted_auc,
        })

    return (
        pd.DataFrame(rows)
        .set_index("group")
        .round(3)
    )


sex_results = calculate_group_metrics(
    audit_data,
    "sex",
)

race_results = calculate_group_metrics(
    audit_data,
    "race_ethnicity",
)

sex_path = (
    REPORT_DIRECTORY
    / "audit_by_sex_2021_2023.csv"
)

race_path = (
    REPORT_DIRECTORY
    / "audit_by_race_2021_2023.csv"
)

sex_results.to_csv(sex_path)
race_results.to_csv(race_path)


# --------------------------------------------------
# Logistic-regression coefficients
# --------------------------------------------------

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model
    .named_steps["classifier"]
    .coef_[0]
)

coefficient_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
})

coefficient_table["absolute_coefficient"] = (
    coefficient_table["coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    "absolute_coefficient",
    ascending=False,
)

coefficient_path = (
    REPORT_DIRECTORY
    / "coefficients_2021_2023.csv"
)

coefficient_table.to_csv(
    coefficient_path,
    index=False,
)


print(f"Model Brier score:    {model_brier:.3f}")
print(f"Baseline Brier score: {baseline_brier:.3f}")

print("\nResults by sex:")
print(sex_results)

print("\nResults by race and ethnicity:")
print(race_results)

print("\nTop model coefficients:")
print(
    coefficient_table[
        ["feature", "coefficient"]
    ]
    .head(15)
    .round(3)
    .to_string(index=False)
)

print(f"\nCalibration graph saved to: {calibration_path}")
print(f"Sex audit saved to: {sex_path}")
print(f"Race audit saved to: {race_path}")
print(f"Coefficients saved to: {coefficient_path}")
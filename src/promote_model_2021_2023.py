from pathlib import Path
import json
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRECTORY = PROJECT_ROOT / "models"

CANDIDATE_DIRECTORY = (
    MODEL_DIRECTORY
    / "candidates"
)

ARCHIVE_DIRECTORY = (
    MODEL_DIRECTORY
    / "archive"
)

ARCHIVE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


candidate_model = (
    CANDIDATE_DIRECTORY
    / "glucorisk_model_2021_2023.joblib"
)

candidate_metadata = (
    CANDIDATE_DIRECTORY
    / "model_metadata_2021_2023.json"
)

production_model = (
    MODEL_DIRECTORY
    / "glucorisk_model.joblib"
)

production_metadata = (
    MODEL_DIRECTORY
    / "model_metadata.json"
)

archived_model = (
    ARCHIVE_DIRECTORY
    / "glucorisk_model_v1.0.0.joblib"
)

archived_metadata = (
    ARCHIVE_DIRECTORY
    / "model_metadata_v1.0.0.json"
)


if not candidate_model.exists():
    raise FileNotFoundError(
        f"Candidate model not found: {candidate_model}"
    )

if not candidate_metadata.exists():
    raise FileNotFoundError(
        "Candidate metadata not found: "
        f"{candidate_metadata}"
    )


# Preserve version 1 before replacing it.
if (
    production_model.exists()
    and not archived_model.exists()
):
    shutil.copy2(
        production_model,
        archived_model,
    )

if (
    production_metadata.exists()
    and not archived_metadata.exists()
):
    shutil.copy2(
        production_metadata,
        archived_metadata,
    )


with open(
    candidate_metadata,
    "r",
    encoding="utf-8",
) as metadata_file:
    metadata = json.load(metadata_file)


metadata["version"] = "2.0.0"
metadata["status"] = "production"

metadata["prediction_target"] = (
    "Current elevated A1C of 5.7% or higher"
)

metadata["population"] = (
    "Adults without a confirmed diabetes diagnosis"
)

metadata["test_metrics"] = (
    metadata["test_metrics_survey_weighted"]
)

metadata["calibration"] = {
    "brier_score": 0.181,
    "baseline_brier_score": 0.210,
}

metadata["audit_summary"] = {
    "sex_weighted_recall_range": [
        0.756,
        0.790,
    ],
    "race_weighted_recall_range": [
        0.572,
        0.866,
    ],
    "sensitive_features_used": False,
}

metadata["limitations"] = [
    "Not a medical diagnosis",
    "Not clinically validated",
    "Trained using cross-sectional survey data",
    "Does not predict future diabetes",
    "Performance varies across demographic groups",
    "Scores should be interpreted as screening estimates",
]


shutil.copy2(
    candidate_model,
    production_model,
)

with open(
    production_metadata,
    "w",
    encoding="utf-8",
) as metadata_file:
    json.dump(
        metadata,
        metadata_file,
        indent=4,
    )


print("GlucoRisk 2.0.0 promoted successfully.")
print(f"Production model: {production_model}")
print(f"Production metadata: {production_metadata}")
print(f"Version 1 archive: {ARCHIVE_DIRECTORY}")
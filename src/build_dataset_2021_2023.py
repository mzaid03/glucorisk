from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nhanes_2021_2023"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def read_xpt(filename):
    return pd.read_sas(
        RAW_DIRECTORY / filename,
        format="xport",
        encoding="utf-8",
    )


def replace_special_codes(data, columns, codes):
    cleaned = data.copy()

    for column in columns:
        cleaned[column] = cleaned[column].replace(
            codes
        )

    return cleaned


def weekly_activity_minutes(
    frequency,
    unit,
    minutes_per_session,
):
    unit_factors = unit.map({
        "D": 7.0,
        "W": 1.0,
        "M": 12.0 / 52.0,
        "Y": 1.0 / 52.0,
    })

    weekly_minutes = (
        frequency
        * unit_factors
        * minutes_per_session
    )

    # NHANES leaves the unit and duration empty when
    # the participant reports zero activity.
    weekly_minutes = weekly_minutes.mask(
        frequency == 0,
        0,
    )

    return weekly_minutes


# --------------------------------------------------
# Demographics
# --------------------------------------------------

demographics = read_xpt("DEMO_L.xpt")[
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ]
].rename(columns={
    "RIDAGEYR": "age",
    "RIAGENDR": "sex_code",
    "RIDRETH3": "race_ethnicity_code",
    "WTMEC2YR": "mec_sample_weight",
    "SDMVPSU": "survey_cluster",
    "SDMVSTRA": "survey_stratum",
})

demographics = demographics[
    demographics["age"] >= 18
].copy()


# --------------------------------------------------
# Body measurements
# --------------------------------------------------

body = read_xpt("BMX_L.xpt")[
    [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ]
].rename(columns={
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_cm",
})


# --------------------------------------------------
# A1C and phlebotomy weights
# --------------------------------------------------

a1c = read_xpt("GHB_L.xpt")[
    [
        "SEQN",
        "WTPH2YR",
        "LBXGH",
    ]
].rename(columns={
    "WTPH2YR": "sample_weight",
    "LBXGH": "a1c",
})

a1c = a1c.dropna(
    subset=["a1c"]
).copy()


# --------------------------------------------------
# Blood pressure
# --------------------------------------------------

blood_pressure = read_xpt("BPXO_L.xpt")

systolic_columns = [
    "BPXOSY1",
    "BPXOSY2",
    "BPXOSY3",
]

diastolic_columns = [
    "BPXODI1",
    "BPXODI2",
    "BPXODI3",
]

blood_pressure["avg_systolic_bp"] = (
    blood_pressure[systolic_columns].mean(axis=1)
)

blood_pressure["avg_diastolic_bp"] = (
    blood_pressure[diastolic_columns].mean(axis=1)
)

blood_pressure = blood_pressure[
    [
        "SEQN",
        "avg_systolic_bp",
        "avg_diastolic_bp",
    ]
]


# --------------------------------------------------
# Physical activity
# --------------------------------------------------

activity = read_xpt("PAQ_L.xpt")

activity = replace_special_codes(
    activity,
    [
        "PAD790Q",
        "PAD800",
        "PAD810Q",
        "PAD820",
        "PAD680",
    ],
    {
        7777: np.nan,
        9999: np.nan,
    },
)

# Convert extremely small decoded values into real zeros.
activity_numeric_columns = [
    "PAD790Q",
    "PAD800",
    "PAD810Q",
    "PAD820",
    "PAD680",
]

for column in activity_numeric_columns:
    near_zero = (
        activity[column].notna()
        & activity[column].abs().lt(1e-10)
    )

    activity.loc[near_zero, column] = 0.0


activity["PAD790U"] = (
    activity["PAD790U"]
    .astype("string")
    .str.upper()
    .str.extract(
        r"([DWMY])",
        expand=False,
    )
)

activity["PAD810U"] = (
    activity["PAD810U"]
    .astype("string")
    .str.upper()
    .str.extract(
        r"([DWMY])",
        expand=False,
    )
)

activity["moderate_minutes_week"] = (
    weekly_activity_minutes(
        activity["PAD790Q"],
        activity["PAD790U"],
        activity["PAD800"],
    )
)

activity["vigorous_minutes_week"] = (
    weekly_activity_minutes(
        activity["PAD810Q"],
        activity["PAD810U"],
        activity["PAD820"],
    )
)

activity["recreation_met_minutes_week"] = (
    4 * activity["moderate_minutes_week"]
    + 8 * activity["vigorous_minutes_week"]
)

activity = activity[
    [
        "SEQN",
        "recreation_met_minutes_week",
        "PAD680",
    ]
].rename(columns={
    "PAD680": "sedentary_minutes",
})


# --------------------------------------------------
# Sleep
# --------------------------------------------------

sleep = read_xpt("SLQ_L.xpt")

sleep = replace_special_codes(
    sleep,
    [
        "SLD012",
        "SLD013",
    ],
    {
        77: np.nan,
        99: np.nan,
    },
)

sleep["average_sleep_hours"] = (
    5 * sleep["SLD012"]
    + 2 * sleep["SLD013"]
) / 7

sleep = sleep[
    [
        "SEQN",
        "average_sleep_hours",
    ]
]


# --------------------------------------------------
# Smoking status
# --------------------------------------------------

smoking = read_xpt("SMQ_L.xpt")

smoking = replace_special_codes(
    smoking,
    [
        "SMQ020",
        "SMQ040",
    ],
    {
        7: np.nan,
        9: np.nan,
    },
)

smoking["smoking_status"] = np.select(
    [
        smoking["SMQ020"] == 2,
        (
            (smoking["SMQ020"] == 1)
            & (smoking["SMQ040"] == 3)
        ),
        (
            (smoking["SMQ020"] == 1)
            & smoking["SMQ040"].isin([1, 2])
        ),
    ],
    [
        0,  # Never smoker
        1,  # Former smoker
        2,  # Current smoker
    ],
    default=np.nan,
)

smoking = smoking[
    [
        "SEQN",
        "smoking_status",
    ]
]


# --------------------------------------------------
# Previous diabetes diagnosis
# --------------------------------------------------

diabetes = read_xpt("DIQ_L.xpt")[
    [
        "SEQN",
        "DIQ010",
    ]
].rename(columns={
    "DIQ010": "diabetes_diagnosis_code",
})


# --------------------------------------------------
# Merge components
# --------------------------------------------------

dataset = demographics.merge(
    a1c,
    on="SEQN",
    how="inner",
)

for component in [
    body,
    blood_pressure,
    activity,
    sleep,
    smoking,
    diabetes,
]:
    dataset = dataset.merge(
        component,
        on="SEQN",
        how="left",
    )

if not dataset["SEQN"].is_unique:
    raise ValueError(
        "Duplicate participant identifiers detected."
    )

dataset["elevated_a1c"] = (
    dataset["a1c"] >= 5.7
).astype(int)

column_order = [
    "SEQN",
    "age",
    "sex_code",
    "race_ethnicity_code",
    "sample_weight",
    "mec_sample_weight",
    "survey_cluster",
    "survey_stratum",
    "bmi",
    "waist_cm",
    "a1c",
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "recreation_met_minutes_week",
    "sedentary_minutes",
    "average_sleep_hours",
    "smoking_status",
    "diabetes_diagnosis_code",
    "elevated_a1c",
]

dataset = (
    dataset[column_order]
    .sort_values("SEQN")
    .reset_index(drop=True)
)

screening_dataset = dataset[
    dataset["diabetes_diagnosis_code"].isin([2, 3])
].copy()

enriched_output = (
    PROCESSED_DIRECTORY
    / "nhanes_2021_2023_enriched.csv"
)

screening_output = (
    PROCESSED_DIRECTORY
    / "nhanes_2021_2023_screening.csv"
)

dataset.to_csv(
    enriched_output,
    index=False,
)

screening_dataset.to_csv(
    screening_output,
    index=False,
)


# --------------------------------------------------
# Validation report
# --------------------------------------------------

model_columns = [
    "bmi",
    "waist_cm",
    "a1c",
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "recreation_met_minutes_week",
    "sedentary_minutes",
    "average_sleep_hours",
    "smoking_status",
    "diabetes_diagnosis_code",
    "elevated_a1c",
]

print(f"Enriched dataset shape: {dataset.shape}")
print(
    "Screening dataset shape: "
    f"{screening_dataset.shape}"
)

print("\nMissing values:")
print(dataset[model_columns].isna().sum())

print("\nDiabetes diagnosis codes:")
print(
    dataset["diabetes_diagnosis_code"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nScreening target counts:")
print(
    screening_dataset["elevated_a1c"]
    .value_counts()
    .sort_index()
)

print("\nScreening target percentages:")
print(
    screening_dataset["elevated_a1c"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print(f"\nSaved enriched dataset to: {enriched_output}")
print(f"Saved screening dataset to: {screening_output}")
from pathlib import Path

import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
raw_directory = project_root / "data" / "raw"
processed_directory = project_root / "data" / "processed"


def read_xpt(filename):
    return pd.read_sas(
        raw_directory / filename,
        format="xport"
    )


demo = read_xpt("DEMO_J.XPT")
body = read_xpt("BMX_J.XPT")
a1c = read_xpt("GHB_J.XPT")
blood_pressure = read_xpt("BPX_J.XPT")
activity = read_xpt("PAQ_J.XPT")
sleep = read_xpt("SLQ_J.XPT")
smoking = read_xpt("SMQ_J.XPT")
diabetes = read_xpt("DIQ_J.XPT")


# Calculate average blood pressure from available readings.
systolic_columns = [
    "BPXSY1",
    "BPXSY2",
    "BPXSY3",
    "BPXSY4"
]

diastolic_columns = [
    "BPXDI1",
    "BPXDI2",
    "BPXDI3",
    "BPXDI4"
]

blood_pressure["avg_systolic_bp"] = (
    blood_pressure[systolic_columns].mean(axis=1)
)

blood_pressure["avg_diastolic_bp"] = (
    blood_pressure[diastolic_columns].mean(axis=1)
)


# Clean physical-activity special codes.
vigorous_days = activity["PAQ655"].replace(
    {77: np.nan, 99: np.nan}
)

vigorous_minutes = activity["PAD660"].replace(
    {7777: np.nan, 9999: np.nan}
)

moderate_days = activity["PAQ670"].replace(
    {77: np.nan, 99: np.nan}
)

moderate_minutes = activity["PAD675"].replace(
    {7777: np.nan, 9999: np.nan}
)

activity["vigorous_minutes_week"] = (
    vigorous_days * vigorous_minutes
)

activity["moderate_minutes_week"] = (
    moderate_days * moderate_minutes
)

# Participants who reported no activity receive zero minutes.
activity.loc[
    activity["PAQ650"] == 2,
    "vigorous_minutes_week"
] = 0

activity.loc[
    activity["PAQ665"] == 2,
    "moderate_minutes_week"
] = 0

activity["recreation_met_minutes_week"] = (
    activity["vigorous_minutes_week"] * 8
    + activity["moderate_minutes_week"] * 4
)

activity["sedentary_minutes"] = activity["PAD680"].replace(
    {7777: np.nan, 9999: np.nan}
)


# Calculate average daily sleep.
sleep["average_sleep_hours"] = (
    sleep["SLD012"] * 5
    + sleep["SLD013"] * 2
) / 7


# Create smoking status:
# 0 = never, 1 = former, 2 = current.
smoking["smoking_status"] = np.nan

smoking.loc[
    smoking["SMQ020"] == 2,
    "smoking_status"
] = 0

smoking.loc[
    (smoking["SMQ020"] == 1)
    & (smoking["SMQ040"] == 3),
    "smoking_status"
] = 1

smoking.loc[
    (smoking["SMQ020"] == 1)
    & (smoking["SMQ040"].isin([1, 2])),
    "smoking_status"
] = 2


dataset = (
    demo[
        [
            "SEQN",
            "RIDAGEYR",
            "RIAGENDR",
            "RIDRETH3",
            "WTMEC2YR",
            "SDMVPSU",
            "SDMVSTRA"
        ]
    ]
    .merge(
        body[["SEQN", "BMXBMI", "BMXWAIST"]],
        on="SEQN",
        how="inner"
    )
    .merge(
        a1c[["SEQN", "LBXGH"]],
        on="SEQN",
        how="inner"
    )
    .merge(
        blood_pressure[
            [
                "SEQN",
                "avg_systolic_bp",
                "avg_diastolic_bp"
            ]
        ],
        on="SEQN",
        how="left"
    )
    .merge(
        activity[
            [
                "SEQN",
                "recreation_met_minutes_week",
                "sedentary_minutes"
            ]
        ],
        on="SEQN",
        how="left"
    )
    .merge(
        sleep[["SEQN", "average_sleep_hours"]],
        on="SEQN",
        how="left"
    )
    .merge(
        smoking[["SEQN", "smoking_status"]],
        on="SEQN",
        how="left"
    )
    .merge(
        diabetes[["SEQN", "DIQ010"]],
        on="SEQN",
        how="left"
    )
)

dataset = dataset[dataset["RIDAGEYR"] >= 18].copy()
dataset = dataset.dropna(subset=["LBXGH"])

dataset["elevated_a1c"] = (
    dataset["LBXGH"] >= 5.7
).astype(int)

dataset = dataset.rename(
    columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "sex_code",
        "RIDRETH3": "race_ethnicity_code",
        "WTMEC2YR": "sample_weight",
        "SDMVPSU": "survey_cluster",
        "SDMVSTRA": "survey_stratum",
        "BMXBMI": "bmi",
        "BMXWAIST": "waist_cm",
        "LBXGH": "a1c",
        "DIQ010": "diabetes_diagnosis_code"
    }
)

processed_directory.mkdir(parents=True, exist_ok=True)

output_path = (
    processed_directory
    / "nhanes_2017_2018_enriched.csv"
)

dataset.to_csv(output_path, index=False)

print("Enriched dataset shape:", dataset.shape)

print("\nMissing values:")
print(dataset.isna().sum())

print("\nDiabetes diagnosis codes:")
print(
    dataset["diabetes_diagnosis_code"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nTarget percentages:")
print(
    dataset["elevated_a1c"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(f"\nSaved dataset to: {output_path}")
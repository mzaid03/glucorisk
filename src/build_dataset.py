from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
raw_data_directory = project_root / "data" / "raw"
processed_data_directory = project_root / "data" / "processed"

demo = pd.read_sas(
    raw_data_directory / "DEMO_J.XPT",
    format="xport"
)

body = pd.read_sas(
    raw_data_directory / "BMX_J.XPT",
    format="xport"
)

a1c = pd.read_sas(
    raw_data_directory / "GHB_J.XPT",
    format="xport"
)

demographic_columns = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "WTMEC2YR",
    "SDMVPSU",
    "SDMVSTRA"
]

body_columns = [
    "SEQN",
    "BMXBMI",
    "BMXWAIST"
]

a1c_columns = [
    "SEQN",
    "LBXGH"
]

dataset = (
    demo[demographic_columns]
    .merge(body[body_columns], on="SEQN", how="inner")
    .merge(a1c[a1c_columns], on="SEQN", how="inner")
)

# Keep adult participants only.
dataset = dataset[dataset["RIDAGEYR"] >= 18].copy()

# The target cannot be missing.
dataset = dataset.dropna(subset=["LBXGH"])

# A1C of 5.7% or higher is considered elevated.
dataset["elevated_a1c"] = (dataset["LBXGH"] >= 5.7).astype(int)

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
        "LBXGH": "a1c"
    }
)

processed_data_directory.mkdir(parents=True, exist_ok=True)

output_path = processed_data_directory / "nhanes_2017_2018.csv"
dataset.to_csv(output_path, index=False)

print("Final dataset shape:", dataset.shape)

print("\nMissing values:")
print(dataset.isna().sum())

print("\nTarget counts:")
print(dataset["elevated_a1c"].value_counts())

print("\nTarget percentages:")
print(
    dataset["elevated_a1c"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(f"\nSaved dataset to: {output_path}")
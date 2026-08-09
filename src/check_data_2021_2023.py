from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nhanes_2021_2023"
)

FILES = {
    "Demographics": "DEMO_L.xpt",
    "Body measurements": "BMX_L.xpt",
    "A1C": "GHB_L.xpt",
    "Blood pressure": "BPXO_L.xpt",
    "Physical activity": "PAQ_L.xpt",
    "Sleep": "SLQ_L.xpt",
    "Smoking": "SMQ_L.xpt",
    "Diabetes questionnaire": "DIQ_L.xpt",
}

EXPECTED_COLUMNS = {
    "Demographics": [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ],
    "Body measurements": [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ],
    "A1C": [
        "SEQN",
        "LBXGH",
    ],
    "Blood pressure": [
        "SEQN",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3",
    ],
    "Physical activity": [
        "SEQN",
        "PAD790Q",
        "PAD790U",
        "PAD800",
        "PAD810Q",
        "PAD810U",
        "PAD820",
        "PAD680",
    ],
    "Sleep": [
        "SEQN",
        "SLD012",
        "SLD013",
    ],
    "Smoking": [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ],
    "Diabetes questionnaire": [
        "SEQN",
        "DIQ010",
    ],
}


for component_name, filename in FILES.items():
    file_path = DATA_DIRECTORY / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}"
        )

    data = pd.read_sas(
        file_path,
        format="xport",
    )

    expected = EXPECTED_COLUMNS[component_name]

    available_expected = [
        column
        for column in expected
        if column in data.columns
    ]

    missing_expected = [
        column
        for column in expected
        if column not in data.columns
    ]

    print(f"\n{'=' * 60}")
    print(component_name)
    print(f"File: {filename}")
    print(f"Shape: {data.shape}")

    print("\nExpected columns found:")
    print(available_expected)

    print("\nExpected columns missing:")
    print(
        missing_expected
        if missing_expected
        else "None"
    )

    print("\nAll columns:")
    print(data.columns.tolist())
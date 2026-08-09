from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nhanes_2021_2023"
    / "PAQ_L.xpt"
)

activity = pd.read_sas(
    FILE_PATH,
    format="xport",
    encoding="utf-8",
)

columns = [
    "PAD790Q",
    "PAD790U",
    "PAD800",
    "PAD810Q",
    "PAD810U",
    "PAD820",
    "PAD680",
]

print("Data types:")
print(activity[columns].dtypes)

print("\nRaw moderate units:")
print(
    [
        repr(value)
        for value in activity["PAD790U"]
        .dropna()
        .unique()
    ]
)

print("\nRaw vigorous units:")
print(
    [
        repr(value)
        for value in activity["PAD810U"]
        .dropna()
        .unique()
    ]
)

print("\nModerate frequency summary:")
print("Zero:", (activity["PAD790Q"] == 0).sum())
print("Missing:", activity["PAD790Q"].isna().sum())
print(
    "Special:",
    activity["PAD790Q"].isin([7777, 9999]).sum(),
)

print("\nVigorous frequency summary:")
print("Zero:", (activity["PAD810Q"] == 0).sum())
print("Missing:", activity["PAD810Q"].isna().sum())
print(
    "Special:",
    activity["PAD810Q"].isin([7777, 9999]).sum(),
)

moderate_unit = (
    activity["PAD790U"]
    .astype("string")
    .str.upper()
    .str.extract(r"([DWMY])", expand=False)
)

vigorous_unit = (
    activity["PAD810U"]
    .astype("string")
    .str.upper()
    .str.extract(r"([DWMY])", expand=False)
)

print("\nNormalized moderate units:")
print(moderate_unit.value_counts(dropna=False))

print("\nNormalized vigorous units:")
print(vigorous_unit.value_counts(dropna=False))

print("\nExample activity rows:")
print(
    activity[
        [
            "PAD790Q",
            "PAD790U",
            "PAD800",
            "PAD810Q",
            "PAD810U",
            "PAD820",
        ]
    ]
    .head(20)
    .to_string(index=False)
)
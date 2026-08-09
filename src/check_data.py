from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
data_directory = project_root / "data" / "raw"

demo = pd.read_sas(data_directory / "DEMO_J.XPT", format="xport")
body = pd.read_sas(data_directory / "BMX_J.XPT", format="xport")
a1c = pd.read_sas(data_directory / "GHB_J.XPT", format="xport")

print("Demographics:", demo.shape)
print("Body measurements:", body.shape)
print("A1C results:", a1c.shape)

print("\nDemographic columns:")
print(demo.columns.tolist())

print("\nBody measurement columns:")
print(body.columns.tolist())

print("\nA1C columns:")
print(a1c.columns.tolist())
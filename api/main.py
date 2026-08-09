import os

from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "glucorisk_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

model = joblib.load(MODEL_PATH)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as metadata_file:
    metadata = json.load(metadata_file)

decision_threshold = metadata["decision_threshold"]


app = FastAPI(
    title="GlucoRisk API",
    description=(
        "A machine-learning screening API that estimates "
        "the likelihood of elevated A1C."
    ),
    version=metadata["version"],
)

default_origins = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000"
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        default_origins,
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class RiskInput(BaseModel):
    age: int = Field(ge=18, le=80)
    bmi: float = Field(ge=10, le=80)
    waist_cm: float = Field(ge=40, le=200)

    avg_systolic_bp: float = Field(
        ge=60,
        le=250,
    )

    avg_diastolic_bp: float = Field(
        ge=30,
        le=150,
    )

    recreation_met_minutes_week: float = Field(
        ge=0,
        le=20000,
    )

    sedentary_minutes: float = Field(
        ge=0,
        le=1440,
    )

    average_sleep_hours: float = Field(
        ge=1,
        le=24,
    )

    # 0 = never, 1 = former, 2 = current
    smoking_status: int = Field(ge=0, le=2)


@app.get("/")
def root():
    return {
        "name": "GlucoRisk API",
        "version": metadata["version"],
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict_risk(user_input: RiskInput):
    input_data = pd.DataFrame(
        [user_input.model_dump()]
    )

    probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    elevated_screening_result = (
        probability >= decision_threshold
    )

    if elevated_screening_result:
        result = "higher_screening_risk"
        recommendation = (
            "Consider discussing A1C screening with "
            "a licensed healthcare professional."
        )
    else:
        result = "lower_screening_risk"
        recommendation = (
            "Continue routine health screenings and "
            "healthy lifestyle habits."
        )

    return {
        "risk_score": round(probability, 4),
        "risk_percentage": round(
            probability * 100,
            1,
        ),
        "screening_result": result,
        "decision_threshold": round(
            decision_threshold,
            4,
        ),
        "recommendation": recommendation,
        "disclaimer": (
            "This result is an educational screening "
            "estimate, not a diagnosis or medical advice."
        ),
    }
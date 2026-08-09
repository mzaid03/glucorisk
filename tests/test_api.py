from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

VALID_INPUT = {
    "age": 45,
    "bmi": 29.5,
    "waist_cm": 100,
    "avg_systolic_bp": 132,
    "avg_diastolic_bp": 84,
    "recreation_met_minutes_week": 600,
    "sedentary_minutes": 480,
    "average_sleep_hours": 7,
    "smoking_status": 0,
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["risk_score"] <= 1
    assert 0 <= result["risk_percentage"] <= 100

    assert result["screening_result"] in {
        "lower_screening_risk",
        "higher_screening_risk",
    }

    assert "decision_threshold" in result
    assert "recommendation" in result
    assert "disclaimer" in result


def test_invalid_age_is_rejected():
    invalid_input = VALID_INPUT.copy()
    invalid_input["age"] = 10

    response = client.post(
        "/predict",
        json=invalid_input,
    )

    assert response.status_code == 422


def test_invalid_smoking_status_is_rejected():
    invalid_input = VALID_INPUT.copy()
    invalid_input["smoking_status"] = 5

    response = client.post(
        "/predict",
        json=invalid_input,
    )

    assert response.status_code == 422


def test_missing_input_is_rejected():
    incomplete_input = VALID_INPUT.copy()
    del incomplete_input["bmi"]

    response = client.post(
        "/predict",
        json=incomplete_input,
    )

    assert response.status_code == 422
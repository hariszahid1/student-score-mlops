from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_score():
    response = client.post(
        "/predict", json={"study_hours": 6, "attendance_pct": 90}
    )
    assert response.status_code == 200
    assert response.json()["score"] > 0


def test_bad_input_is_rejected():
    response = client.post(
        "/predict", json={"study_hours": "six", "attendance_pct": 90}
    )
    assert response.status_code == 422

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


MODEL_PATH = Path("artifacts/models/message_classifier.joblib")


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Модель ещё не обучена. Запусти: uv run python -m project.src.train",
)
def test_health_endpoint():
    from project.src.service import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Модель ещё не обучена. Запусти: uv run python -m project.src.train",
)
def test_predict_endpoint():
    from project.src.service import app

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"text": "Купи рекламу и заработай деньги"},
    )

    data = response.json()

    assert response.status_code == 200
    assert data["label"] in {"normal", "spam", "profanity"}
    assert "confidence" in data
    assert "recommended_action" in data
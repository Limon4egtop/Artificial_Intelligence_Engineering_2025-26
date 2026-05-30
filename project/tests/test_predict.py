from pathlib import Path

import pytest

from project.src.predict import MessageClassifier


MODEL_PATH = Path("artifacts/models/message_classifier.joblib")


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Модель ещё не обучена. Запусти: uv run python -m project.src.train",
)
def test_classifier_predict_structure():
    classifier = MessageClassifier(MODEL_PATH)

    result = classifier.predict("Привет, как дела?")

    assert "label" in result
    assert "confidence" in result
    assert "is_violation" in result
    assert "recommended_action" in result
    assert "probabilities" in result


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Модель ещё не обучена. Запусти: uv run python -m project.src.train",
)
def test_classifier_predict_allowed_labels():
    classifier = MessageClassifier(MODEL_PATH)

    result = classifier.predict("Заработок без опыта, пиши в личку")

    assert result["label"] in {"normal", "spam", "profanity"}
    assert 0 <= result["confidence"] <= 1
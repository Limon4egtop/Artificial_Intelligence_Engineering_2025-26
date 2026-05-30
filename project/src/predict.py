from pathlib import Path

import joblib

from project.src.preprocessing import normalize_text


MODEL_PATH = Path("project/artifacts/models/message_classifier.joblib")


class MessageClassifier:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {self.model_path}. "
                f"Сначала запусти обучение: uv run python -m src.train"
            )

        self.model = joblib.load(self.model_path)

    def predict(self, text: str) -> dict:
        normalized_text = normalize_text(text)

        label = self.model.predict([normalized_text])[0]
        probabilities = self.model.predict_proba([normalized_text])[0]

        class_probabilities = dict(zip(self.model.classes_, probabilities))
        confidence = float(class_probabilities[label])

        return {
            "text": text,
            "label": label,
            "confidence": round(confidence, 4),
            "is_violation": label != "normal",
            "recommended_action": self._get_action(label, confidence),
            "probabilities": {
                key: round(float(value), 4)
                for key, value in class_probabilities.items()
            },
        }

    @staticmethod
    def _get_action(label: str, confidence: float) -> str:
        if label == "normal":
            return "allow"

        if confidence >= 0.85:
            return "mute_user"

        if confidence >= 0.65:
            return "delete_message"

        return "send_to_manual_review"
from fastapi import FastAPI
from pydantic import BaseModel, Field

from project.src.predict import MessageClassifier


app = FastAPI(
    title="AI Moderator Bot API",
    description="API для интеллектуальной модерации сообщений",
    version="0.1.0",
)

classifier = MessageClassifier()


class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict_message(request: MessageRequest) -> dict:
    return classifier.predict(request.text)
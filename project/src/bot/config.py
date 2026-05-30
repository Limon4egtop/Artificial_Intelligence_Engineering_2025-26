from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class BotConfig:
    bot_token: str
    moderation_chat_id: int
    target_chat_ids: set[int]
    model_path: Path


def _parse_chat_ids(value: str) -> set[int]:
    if not value:
        return set()

    return {
        int(chat_id.strip())
        for chat_id in value.split(",")
        if chat_id.strip()
    }


def load_config() -> BotConfig:
    bot_token = os.getenv("BOT_TOKEN")
    moderation_chat_id = os.getenv("MODERATION_CHAT_ID")
    target_chat_ids = os.getenv("TARGET_CHAT_IDS", "")
    model_path = os.getenv(
        "MODEL_PATH",
        "project/artifacts/models/message_classifier.joblib",
    )

    if not bot_token:
        raise ValueError("Не задан BOT_TOKEN в .env")

    if not moderation_chat_id:
        raise ValueError("Не задан MODERATION_CHAT_ID в .env")

    parsed_target_chat_ids = _parse_chat_ids(target_chat_ids)

    if not parsed_target_chat_ids:
        raise ValueError("Не задан TARGET_CHAT_IDS в .env")

    return BotConfig(
        bot_token=bot_token,
        moderation_chat_id=int(moderation_chat_id),
        target_chat_ids=parsed_target_chat_ids,
        model_path=Path(model_path),
    )
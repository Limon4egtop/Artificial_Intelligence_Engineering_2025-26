"""
Заготовка для Telegram-бота.

В первой версии проекта основная ценность находится в ML-модели и API.
Интеграционный слой бота можно доработать отдельно через Telegram Bot API
или заменить на API другого мессенджера.
"""

import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    api_url = os.getenv("MODERATION_API_URL", "http://127.0.0.1:8000/predict")

    if not token:
        print("TELEGRAM_BOT_TOKEN не задан. Создайте .env на основе configs/.env.example")
        return

    print("Bot integration placeholder")
    print("Moderation API:", api_url)
    print("Дальше сюда добавляется обработчик входящих сообщений Telegram.")


if __name__ == "__main__":
    main()

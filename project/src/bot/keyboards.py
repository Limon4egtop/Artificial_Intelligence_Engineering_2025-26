from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def moderation_keyboard(case_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Отменить наказание",
                    callback_data=f"moderation:cancel:{case_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Исключить из чата",
                    callback_data=f"moderation:kick_chat:{case_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Исключить из всех чатов",
                    callback_data=f"moderation:kick_all:{case_id}",
                ),
            ],
        ]
    )
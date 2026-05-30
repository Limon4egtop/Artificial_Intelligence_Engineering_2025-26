import asyncio
import html
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatMemberStatus, ChatType, ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    ChatPermissions,
    Message,
)

from project.src.bot.config import BotConfig, load_config
from project.src.bot.keyboards import moderation_keyboard
from project.src.bot.storage import (
    InMemoryModerationStorage,
    ModerationCase,
)
from project.src.predict import MessageClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = Router()
storage = InMemoryModerationStorage()

config: BotConfig
classifier: MessageClassifier


def make_user_link(user_id: int, username: str | None) -> str:
    if username:
        safe_username = html.escape(username)
        return f'<a href="https://t.me/{safe_username}">@{safe_username}</a>'

    return f'<a href="tg://user?id={user_id}">Пользователь</a>'


def get_case_title(case: ModerationCase) -> str:
    if case.status.value == "проверено":
        return "✅ <b>Проверено</b>"

    return "🚨 <b>Новое нарушение</b>"


def format_label(label: str) -> str:
    labels = {
        "normal": "✅ normal",
        "spam": "📢 spam",
        "profanity": "🤬 profanity",
    }

    return labels.get(label, f"❓ {label}")


def format_action(action: str) -> str:
    if "mute_10_minutes" in action:
        return "🔇 мут на 10 минут"

    if "mute_7_days" in action:
        return "🔕 ограничение на 7 дней"

    if "delete" in action:
        return f"🗑 {action}"

    if "failed" in action:
        return f"⚠️ {html.escape(action)}"

    return html.escape(action)


def format_checked_by(case: ModerationCase) -> str:
    if not case.checked_by:
        return "—"

    if case.checked_by_username:
        username = html.escape(case.checked_by_username)
        return f"<code>{case.checked_by}</code>, @{username}"

    return f"<code>{case.checked_by}</code>"


def build_moderation_text(case: ModerationCase) -> str:
    safe_text = html.escape(case.text)
    safe_username = html.escape(case.username) if case.username else "нет"
    safe_chat_title = (
        html.escape(case.chat_title)
        if case.chat_title
        else "без названия"
    )

    status_icon = "✅" if case.status.value == "проверено" else "❗️"

    return (
        f"{get_case_title(case)}\n\n"
        f"🆔 <b>ID кейса:</b> <code>{case.case_id}</code>\n"
        f"👤 <b>Пользователь:</b> {case.user_link}\n"
        f"🔗 <b>Username:</b> <code>{safe_username}</code>\n"
        f"🧾 <b>ID пользователя:</b> <code>{case.user_id}</code>\n"
        f"💬 <b>Чат:</b> <code>{safe_chat_title}</code>\n"
        f"🆔 <b>ID чата:</b> <code>{case.source_chat_id}</code>\n"
        f"✉️ <b>ID сообщения:</b> <code>{case.source_message_id}</code>\n"
        f"🕒 <b>Дата и время:</b> <code>{case.created_at.isoformat()}</code>\n\n"
        f"🏷 <b>Класс модели:</b> <code>{format_label(case.label)}</code>\n"
        f"📊 <b>Уверенность:</b> <code>{case.confidence}</code>\n"
        f"⚙️ <b>Действие:</b> <code>{format_action(case.action)}</code>\n"
        f"{status_icon} <b>Статус модерации:</b> <code>{case.status.value}</code>\n"
        f"👮 <b>Проверено модератором:</b> {format_checked_by(case)}\n\n"
        f"📝 <b>Текст сообщения:</b>\n"
        f"<blockquote>{safe_text}</blockquote>"
    )


async def restrict_user_for_10_minutes(
    bot: Bot,
    chat_id: int,
    user_id: int,
) -> None:
    until_date = datetime.now(timezone.utc) + timedelta(minutes=10)

    await bot.restrict_chat_member(
        chat_id=chat_id,
        user_id=user_id,
        permissions=ChatPermissions(
            can_send_messages=False,
            can_send_audios=False,
            can_send_documents=False,
            can_send_photos=False,
            can_send_videos=False,
            can_send_video_notes=False,
            can_send_voice_notes=False,
            can_send_polls=False,
            can_send_other_messages=False,
            can_add_web_page_previews=False,
            can_change_info=False,
            can_invite_users=True,
            can_pin_messages=False,
            can_manage_topics=False,
        ),
        until_date=until_date,
    )


async def restrict_user_for_7_days(
    bot: Bot,
    chat_id: int,
    user_id: int,
) -> None:
    until_date = datetime.now(timezone.utc) + timedelta(days=7)

    await bot.restrict_chat_member(
        chat_id=chat_id,
        user_id=user_id,
        permissions=ChatPermissions(
            can_send_messages=False,
            can_send_audios=False,
            can_send_documents=False,
            can_send_photos=False,
            can_send_videos=False,
            can_send_video_notes=False,
            can_send_voice_notes=False,
            can_send_polls=False,
            can_send_other_messages=False,
            can_add_web_page_previews=False,
            can_change_info=False,
            can_invite_users=True,
            can_pin_messages=False,
            can_manage_topics=False,
        ),
        until_date=until_date,
    )


async def unrestrict_user(
    bot: Bot,
    chat_id: int,
    user_id: int,
) -> None:
    await bot.restrict_chat_member(
        chat_id=chat_id,
        user_id=user_id,
        permissions=ChatPermissions(
            can_send_messages=True,
            can_send_audios=True,
            can_send_documents=True,
            can_send_photos=True,
            can_send_videos=True,
            can_send_video_notes=True,
            can_send_voice_notes=True,
            can_send_polls=True,
            can_send_other_messages=True,
            can_add_web_page_previews=True,
            can_change_info=False,
            can_invite_users=True,
            can_pin_messages=False,
            can_manage_topics=False,
        ),
    )


async def kick_user_from_chat(
    bot: Bot,
    chat_id: int,
    user_id: int,
) -> None:
    await bot.ban_chat_member(
        chat_id=chat_id,
        user_id=user_id,
        revoke_messages=False,
    )


async def send_to_moderation_chat(
    bot: Bot,
    case: ModerationCase,
) -> None:
    sent_message = await bot.send_message(
        chat_id=config.moderation_chat_id,
        text=build_moderation_text(case),
        reply_markup=moderation_keyboard(case.case_id),
    )

    case.moderator_message_id = sent_message.message_id


async def is_bot_admin_in_chat(bot: Bot, chat_id: int) -> bool:
    me = await bot.get_me()
    member = await bot.get_chat_member(chat_id=chat_id, user_id=me.id)

    return member.status in {
        ChatMemberStatus.ADMINISTRATOR,
        ChatMemberStatus.CREATOR,
    }


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "🤖 AI Moderator Bot запущен.\n"
        "Добавьте бота администратором и отключите privacy mode."
    )


@router.message()
async def moderate_text_message(message: Message, bot: Bot) -> None:
    message_text = message.text or message.caption

    if not message_text:
        return

    logger.info(
        "Получено сообщение: chat_id=%s chat_title=%r user_id=%s text=%r",
        message.chat.id,
        message.chat.title,
        message.from_user.id if message.from_user else None,
        message_text,
    )

    if message.chat.type not in {
        ChatType.GROUP,
        ChatType.SUPERGROUP,
    }:
        return

    if message.chat.id not in config.target_chat_ids:
        return

    if not message.from_user:
        return

    if message.from_user.is_bot:
        return

    prediction = classifier.predict(message_text)

    logger.info("Результат модели: %s", prediction)

    label = prediction["label"]
    confidence = prediction["confidence"]

    if label == "profanity" and confidence < 0.65:
        logger.info("LOW CONFIDENCE PROFANITY: %s", prediction)
        return

    if label == "spam" and confidence < 0.70:
        logger.info("LOW CONFIDENCE SPAM: %s", prediction)
        return

    if label == "normal":
        return

    user_id = message.from_user.id
    username = message.from_user.username
    user_link = make_user_link(user_id, username)

    created_at = message.date
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    if label == "profanity":
        action = "mute_10_minutes"

        try:
            await restrict_user_for_10_minutes(
                bot=bot,
                chat_id=message.chat.id,
                user_id=user_id,
            )
        except (TelegramBadRequest, TelegramForbiddenError) as error:
            action = f"mute_failed: {error}"
            logger.exception("Не удалось ограничить пользователя")

    elif label == "spam":
        action = "delete_message_and_mute_7_days"

        try:
            await message.delete()
        except (TelegramBadRequest, TelegramForbiddenError) as error:
            action = f"delete_failed: {error}"
            logger.exception("Не удалось удалить сообщение")

        try:
            await restrict_user_for_7_days(
                bot=bot,
                chat_id=message.chat.id,
                user_id=user_id,
            )
        except (TelegramBadRequest, TelegramForbiddenError) as error:
            action = f"{action}; mute_7_days_failed: {error}"
            logger.exception(
                "Не удалось ограничить пользователя на 7 дней"
            )

    else:
        action = "unknown_label_manual_review"

    case = ModerationCase(
        case_id=uuid4().hex[:12],
        source_chat_id=message.chat.id,
        chat_title=message.chat.title,
        source_message_id=message.message_id,
        user_id=user_id,
        username=username,
        user_link=user_link,
        text=message_text,
        label=label,
        confidence=confidence,
        action=action,
        created_at=created_at,
    )

    storage.add(case)

    await send_to_moderation_chat(bot, case)


@router.callback_query(F.data.startswith("moderation:"))
async def moderation_callback_handler(
    callback: CallbackQuery,
    bot: Bot,
) -> None:
    if callback.message is None:
        await callback.answer(
            "❌ Сообщение модерации не найдено",
            show_alert=True,
        )
        return

    if callback.message.chat.id != config.moderation_chat_id:
        await callback.answer(
            "⛔ Эта кнопка доступна только в чате модераторов",
            show_alert=True,
        )
        return

    if callback.from_user is None:
        await callback.answer(
            "❌ Не удалось определить модератора",
            show_alert=True,
        )
        return

    try:
        _, action, case_id = callback.data.split(":")
    except ValueError:
        await callback.answer(
            "❌ Некорректная команда",
            show_alert=True,
        )
        return

    case = storage.get(case_id)

    if case is None:
        await callback.answer(
            "❌ Кейс не найден",
            show_alert=True,
        )
        return

    try:
        if action == "cancel":
            await unrestrict_user(
                bot=bot,
                chat_id=case.source_chat_id,
                user_id=case.user_id,
            )

            await callback.answer("✅ Наказание отменено")

        elif action == "kick_chat":
            await kick_user_from_chat(
                bot=bot,
                chat_id=case.source_chat_id,
                user_id=case.user_id,
            )

            await callback.answer("🚫 Пользователь исключён из чата")

        elif action == "kick_all":
            for chat_id in config.target_chat_ids:
                try:
                    await kick_user_from_chat(
                        bot=bot,
                        chat_id=chat_id,
                        user_id=case.user_id,
                    )
                except (
                    TelegramBadRequest,
                    TelegramForbiddenError,
                ) as error:
                    logger.warning(
                        "Не удалось исключить пользователя %s "
                        "из чата %s: %s",
                        case.user_id,
                        chat_id,
                        error,
                    )

            await callback.answer(
                "🚫 Пользователь исключён из всех чатов"
            )

        else:
            await callback.answer(
                "❌ Неизвестное действие",
                show_alert=True,
            )
            return

    except (
        TelegramBadRequest,
        TelegramForbiddenError,
    ) as error:
        await callback.answer(
            f"⚠️ Ошибка Telegram API: {error}",
            show_alert=True,
        )
        return

    storage.mark_checked(
        case.case_id,
        callback.from_user.id,
        callback.from_user.username,
    )

    await callback.message.edit_text(
        text=build_moderation_text(case),
        reply_markup=moderation_keyboard(case.case_id),
    )


async def main() -> None:
    global config
    global classifier

    config = load_config()
    classifier = MessageClassifier(config.model_path)

    bot = Bot(
        token=config.bot_token,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
        ),
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    logger.info("🤖 AI Moderator Bot запущен")

    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
"""Telegram-бот: фото пробы зерна или растения -> полный разбор.

Голосовые сообщения пересылаются напрямую в Gemini Flash (он же распознаёт
речь и отвечает) — собственного распознавания речи в проекте нет.
"""

import asyncio
import logging
import os
from io import BytesIO

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import gemini
from formatting import format_disease, format_grain

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dnai-bot")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
REQUEST_TIMEOUT = float(os.environ.get("BOT_REQUEST_TIMEOUT", 90))

WELCOME = (
    "<b>Dän-AI — советник по вашей партии зерна</b>\n\n"
    "Пришлите фото — я сам пойму, что на нём, и отвечу на три вопроса:\n"
    "какой класс, сколько теряете в тенге и что сделать прямо сейчас.\n\n"
    "🌾 <b>Проба зерна</b> — предварительный класс, цена за тонну, потери "
    "против 3 класса и прибавка после очистки\n"
    "🌱 <b>Лист или растение</b> — болезнь, вредитель или сорняк и меры обработки\n\n"
    "Можно задать вопрос текстом или голосовым (казахский или русский) — отвечу подробно.\n\n"
    "<i>Как снимать зерно:</i> разложите пробу тонким слоем на контрастном фоне, "
    "снимайте сверху при ровном свете.\n"
    "<i>Как снимать растение:</i> поражённый лист крупным планом, при дневном свете."
)


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(WELCOME)


async def _analyze_file_id(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id: str) -> None:
    """Скачиваем фото, отправляем в /predict/auto (модуль определяется сам)
    и присылаем разбор."""
    status = await update.message.reply_text("Анализирую… это займёт несколько секунд")
    await context.bot.send_chat_action(update.message.chat_id, ChatAction.TYPING)

    try:
        tg_file = await context.bot.get_file(file_id)
        buffer = BytesIO()
        await tg_file.download_to_memory(buffer)

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{BACKEND_URL}/predict/auto",
                files={"file": ("photo.jpg", buffer.getvalue(), "image/jpeg")},
            )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:  # noqa: BLE001
            pass
        await status.edit_text(f"Сервер вернул ошибку. {detail}".strip())
        return
    except Exception:  # noqa: BLE001
        logger.exception("Ошибка запроса к backend")
        await status.edit_text(
            "Не удалось связаться с сервером анализа. Попробуйте ещё раз через минуту."
        )
        return

    module = data.get("detected_module")
    if module == "unknown" or data.get("ok") is False:
        await status.edit_text(
            data.get("message")
            or "На фото не удалось распознать пробу зерна или растение. "
            "Пришлите фото пробы зерна или поражённого листа."
        )
        return
    text = format_disease(data) if module == "disease" else format_grain(data)
    # запоминаем итог анализа — следующие вопросы (текстом/голосом) консультант
    # будет понимать в контексте этой партии/растения
    context.user_data["last_analysis"] = _analysis_summary(data, module)
    parts = _split_for_telegram(text)
    # первую часть кладём в статус-сообщение, остальные (если ответ длинный) —
    # отдельными сообщениями, чтобы не упереться в лимит Telegram
    await status.edit_text(parts[0], parse_mode="HTML")
    for extra in parts[1:]:
        await update.message.reply_text(extra, parse_mode="HTML")


def _analysis_summary(data: dict, module: str) -> str:
    """Короткая выжимка анализа для контекста консультанта."""
    if module == "disease":
        d = data.get("diagnosis") or {}
        return (
            f"фото растения, диагноз: {d.get('name_ru')} "
            f"(уверенность {round((d.get('confidence') or 0) * 100)}%)"
        )
    cats = ", ".join(
        f"{c['label']} {c['percent']}%" for c in data.get("categories") or [] if c.get("percent")
    )
    price = data.get("price_kzt_per_ton")
    return (
        f"проба зерна: {data.get('grade_label')}"
        + (", ориентировочная цена " + f"{price:,.0f}".replace(",", " ") + " ₸/т" if price else "")
        + (f"; состав на фото: {cats}" if cats else "")
    )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _analyze_file_id(update, context, update.message.photo[-1].file_id)


async def on_document_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not (doc.mime_type or "").startswith("image/"):
        await update.message.reply_text("Пришлите, пожалуйста, изображение.")
        return
    await _analyze_file_id(update, context, doc.file_id)


VOICE_TASK = (
    "Это голосовое сообщение фермера (на казахском или русском). Внимательно "
    "распознай речь, учитывая казахское произношение, и развёрнуто ответь на "
    "вопрос из него на том же языке."
)

# Telegram не принимает сообщения длиннее 4096 символов — длинный ответ иначе
# просто не отправляется («сообщение потерялось»). Режем на части по границам
# абзацев/строк с запасом и шлём несколькими сообщениями.
TG_MAX_CHARS = 3800


def _split_for_telegram(text: str) -> list[str]:
    chunks: list[str] = []
    rest = text.strip()
    while len(rest) > TG_MAX_CHARS:
        cut = rest.rfind("\n\n", 0, TG_MAX_CHARS)
        if cut <= 0:
            cut = rest.rfind("\n", 0, TG_MAX_CHARS)
        if cut <= 0:
            cut = rest.rfind(" ", 0, TG_MAX_CHARS)
        if cut <= 0:
            cut = TG_MAX_CHARS
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        chunks.append(rest)
    return chunks or [""]


async def _reply_long(message, text: str, parse_mode: str | None = None) -> None:
    """Отправляет ответ, разбивая на части, если он длиннее лимита Telegram."""
    for part in _split_for_telegram(text):
        await message.reply_text(part, parse_mode=parse_mode)


def _context_parts(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    last = context.user_data.get("last_analysis")
    if not last:
        return []
    return [gemini.text_part(
        f"Контекст: последний анализ фото этого фермера — {last}. "
        "Учитывай его, если вопрос относится к этой партии или растению."
    )]


async def _answer(update: Update, context: ContextTypes.DEFAULT_TYPE, parts: list[dict]) -> None:
    """Вопрос -> Gemini -> развёрнутый ответ несколькими сообщениями.
    Пока ждём ответ, держим индикатор «печатает…»."""
    if not GEMINI_API_KEY:
        await update.message.reply_text(
            "Консультант пока недоступен: не задан GEMINI_API_KEY на сервере."
        )
        return

    chat_id = update.message.chat_id
    stop = asyncio.Event()

    async def keep_typing() -> None:
        while not stop.is_set():
            try:
                await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.5)
            except asyncio.TimeoutError:
                pass

    typing_task = asyncio.create_task(keep_typing())
    reply, error = "", ""
    try:
        reply = await gemini.ask(parts)
    except gemini.GeminiError as exc:
        error = str(exc)
        logger.error("Gemini не ответил: %s", error)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Ошибка консультанта")
    finally:
        stop.set()
        await typing_task

    if reply:
        await _reply_long(update.message, reply)
    else:
        await update.message.reply_text(
            "Не получилось ответить, попробуйте ещё раз.\n\n"
            f"Техническая причина: {error[:600] or 'пустой ответ модели'}"
        )


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Голосовое -> Gemini (он и распознаёт речь, и отвечает)."""
    voice = update.message.voice or update.message.audio
    tg_file = await context.bot.get_file(voice.file_id)
    buffer = BytesIO()
    await tg_file.download_to_memory(buffer)
    parts = (
        [gemini.text_part(VOICE_TASK)]
        + _context_parts(context)
        + [gemini.audio_part(voice.mime_type or "audio/ogg", buffer.getvalue())]
    )
    await _answer(update, context, parts)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Текстовый вопрос -> консультант (с учётом последнего анализа фото)."""
    question = (update.message.text or "").strip()
    if not question:
        return
    await _answer(update, context, _context_parts(context) + [gemini.text_part(question)])


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN. Скопируйте .env.example в .env и заполните его."
        )

    # concurrent_updates: несколько сообщений/пользователей обрабатываются
    # параллельно, а не в очередь друг за другом
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, on_document_image))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Бот запущен, backend: %s", BACKEND_URL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

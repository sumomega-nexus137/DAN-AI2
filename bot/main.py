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
    "Можно надиктовать голосовое (казахский или русский) — отвечу текстом.\n\n"
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
    await status.edit_text(text, parse_mode="HTML")


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _analyze_file_id(update, context, update.message.photo[-1].file_id)


async def on_document_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not (doc.mime_type or "").startswith("image/"):
        await update.message.reply_text("Пришлите, пожалуйста, изображение.")
        return
    await _analyze_file_id(update, context, doc.file_id)


VOICE_PROMPT = (
    "Это аудио — вопрос фермера, на казахском ИЛИ на русском языке. "
    "Сначала внимательно распознай сказанное (учти казахскую речь), затем "
    "ответь на том же языке, на котором был вопрос: коротко и по делу, "
    "до 6 предложений, как агроном-консультант из Казахстана. "
    "Если вопрос про качество зерна или болезни растений — добавь, что можно "
    "прислать фото боту для точной оценки. Отвечай обычным текстом, без markdown."
)


def _extract_text(result) -> str:
    """Безопасно достаём текст ответа Gemini: result.text бросает исключение,
    если ответ пустой/заблокирован, поэтому пробуем и кандидатов."""
    try:
        text = (result.text or "").strip()
        if text:
            return text
    except Exception:  # noqa: BLE001
        pass
    try:
        for cand in getattr(result, "candidates", []) or []:
            parts = getattr(getattr(cand, "content", None), "parts", []) or []
            joined = " ".join(getattr(p, "text", "") for p in parts).strip()
            if joined:
                return joined
    except Exception:  # noqa: BLE001
        pass
    return ""


# Агрономия — это разговоры про протравители, фунгициды и дозы. Стандартные
# фильтры Gemini нередко режут такие ответы в пустоту, и пользователь видел
# «не расслышал». Поэтому блокируем только явно опасный контент.
SAFETY = [
    {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]
# 300 токенов хватает на ответ до 6 предложений, а генерация вдвое короче,
# чем при 600 — заметно быстрее отклик на голосовое.
GEN_CONFIG = {"temperature": 0.4, "max_output_tokens": 300}
GEMINI_TIMEOUT_S = 45


async def _voice_reply(mime_type: str, audio_bytes: bytes) -> str:
    """Аудио -> Gemini, с одним повтором при пустом ответе."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    payload = [VOICE_PROMPT, {"mime_type": mime_type, "data": audio_bytes}]

    for attempt in range(2):
        try:
            result = await asyncio.to_thread(
                model.generate_content,
                payload,
                safety_settings=SAFETY,
                generation_config=GEN_CONFIG,
                request_options={"timeout": GEMINI_TIMEOUT_S},
            )
            text = _extract_text(result)
            if text:
                return text
            logger.warning("Пустой ответ Gemini на голосовое (попытка %d)", attempt + 1)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка Gemini на голосовом (попытка %d)", attempt + 1)
        await asyncio.sleep(1.0)
    return ""


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Голосовое -> напрямую в Gemini Flash (он и распознаёт речь, и отвечает)."""
    if not GEMINI_API_KEY:
        await update.message.reply_text(
            "Голосовые пока недоступны: не задан GEMINI_API_KEY на сервере."
        )
        return

    await context.bot.send_chat_action(update.message.chat_id, ChatAction.TYPING)

    voice = update.message.voice or update.message.audio
    tg_file = await context.bot.get_file(voice.file_id)
    buffer = BytesIO()
    await tg_file.download_to_memory(buffer)
    audio_bytes = buffer.getvalue()

    reply = await _voice_reply(voice.mime_type or "audio/ogg", audio_bytes)
    if reply:
        await update.message.reply_text(reply)
    else:
        await update.message.reply_text(
            "Не расслышал вопрос. Запишите ещё раз чуть длиннее и ближе к "
            "микрофону, без фонового шума — или напишите вопрос текстом."
        )


async def on_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(
        "Пришлите фото пробы зерна или растения — разберу.\n"
        "Или надиктуйте голосовое с вопросом.\n\n/start — как это работает"
    )


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN. Скопируйте .env.example в .env и заполните его."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, on_document_image))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Бот запущен, backend: %s", BACKEND_URL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

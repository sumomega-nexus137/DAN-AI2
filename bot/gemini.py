"""Быстрый клиент Gemini для бота: прямой REST-вызов через httpx.

Почему не google-generativeai SDK:
* у gemini-2.5-flash по умолчанию включено «мышление» (thinking) — оно
  съедает часть лимита ответа (ответ обрывался на полуслове) и добавляет
  секунды задержки. Через REST его можно выключить (thinkingBudget=0);
* если ответ всё же упёрся в лимит (finishReason=MAX_TOKENS), просим модель
  продолжить с места остановки и склеиваем — ответ никогда не теряется;
* одно общее HTTP-соединение на весь бот — без лишних рукопожатий.
"""

import base64
import logging
import os

import httpx

logger = logging.getLogger("dnai-bot")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

MAX_OUTPUT_TOKENS = int(os.environ.get("BOT_MAX_OUTPUT_TOKENS", 8192))
MAX_CONTINUATIONS = 2  # сколько раз дописывать, если упёрлись в лимит
TIMEOUT_S = float(os.environ.get("BOT_GEMINI_TIMEOUT", 60))

# Агрономия — протравители, фунгициды, дозы: стандартные фильтры режут такие
# ответы в пустоту. Блокируем только явно опасное.
_SAFETY = [
    {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

SYSTEM_PROMPT = (
    "Ты — опытный агроном-консультант сервиса Dän-AI для фермеров Казахстана: "
    "зерно (класс, цена, очистка, хранение, продажа), болезни и вредители "
    "растений, сорняки, агротехника. Отвечай СТРОГО на том языке, на котором "
    "задан вопрос (казахский или русский). Отвечай развёрнуто и по делу: "
    "объясни причины, разбей ответ на понятные пункты, дай конкретные шаги, "
    "сроки и ориентиры по деньгам в тенге за тонну, где это уместно. Цены "
    "пшеницы (Акмолинская обл., осень 2026, с НДС): 1 кл 128–140 тыс., 2 кл "
    "118–128 тыс., 3 кл 98–118 тыс., 4 кл 84–98 тыс., 5 кл 72–84 тыс., фураж "
    "58–72 тыс. ₸/т. Не выдумывай точные дозировки препаратов — советуй "
    "сверить их с регламентом применения и агрономом. Пиши обычным текстом "
    "без markdown (без звёздочек и решёток); для списков используй «•» или "
    "нумерацию. Всегда заканчивай мысль полностью."
)

_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=TIMEOUT_S)
    return _client


def _gen_config(with_thinking_off: bool) -> dict:
    cfg = {"temperature": 0.5, "maxOutputTokens": MAX_OUTPUT_TOKENS}
    if with_thinking_off:
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    return cfg


def _extract(data: dict) -> tuple[str, str]:
    """(текст, finishReason) из ответа REST."""
    cands = data.get("candidates") or []
    if not cands:
        return "", (data.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return text, cand.get("finishReason", "")


async def _call(contents: list[dict]) -> tuple[str, str]:
    # ключ/модель читаем в момент вызова: .env может загрузиться позже импорта
    model = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
    key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    url = _URL.format(model=model)
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": _gen_config(with_thinking_off=True),
        "safetySettings": _SAFETY,
    }
    resp = await _http().post(url, params={"key": key}, json=body)
    if resp.status_code == 400 and "thinking" in resp.text.lower():
        # модель без поддержки thinkingConfig (напр. gemini-2.0-flash) — без него
        body["generationConfig"] = _gen_config(with_thinking_off=False)
        resp = await _http().post(url, params={"key": key}, json=body)
    resp.raise_for_status()
    return _extract(resp.json())


async def ask(user_parts: list[dict]) -> str:
    """Один вопрос (текст и/или аудио) -> полный ответ. Если ответ упёрся в
    лимит токенов — дописываем продолжение, чтобы ничего не обрывалось."""
    contents = [{"role": "user", "parts": user_parts}]
    text, reason = await _call(contents)
    full = text
    for _ in range(MAX_CONTINUATIONS):
        if reason != "MAX_TOKENS" or not text:
            break
        contents += [
            {"role": "model", "parts": [{"text": text}]},
            {"role": "user", "parts": [{"text": "Продолжи ответ ровно с того места, где остановился, без повторов."}]},
        ]
        text, reason = await _call(contents)
        full += text
    if reason not in ("STOP", "MAX_TOKENS", ""):
        logger.warning("Gemini finishReason=%s", reason)
    return full.strip()


def text_part(text: str) -> dict:
    return {"text": text}


def audio_part(mime_type: str, data: bytes) -> dict:
    return {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(data).decode()}}

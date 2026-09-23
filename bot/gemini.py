"""Клиент Gemini для бота: быстро, без обрывов, с запасными путями.

Порядок попыток (первая успешная — ответ):
1. google-genai (официальный новый SDK, асинхронный): «мышление» выключено
   (thinking_budget=0) — быстрее и не съедает лимит ответа;
2. прямой REST через httpx с тем же thinkingBudget=0;
3. старый google-generativeai SDK (через него бот отвечал раньше) — с большим
   лимитом токенов.
Если ответ упёрся в лимит (MAX_TOKENS), просим продолжение и склеиваем.
Если все пути упали — поднимаем GeminiError с короткой причиной, её бот
покажет пользователю, чтобы по скрину было понятно, что не так.
"""

import asyncio
import base64
import logging
import os

import httpx

logger = logging.getLogger("dnai-bot")

MAX_OUTPUT_TOKENS = int(os.environ.get("BOT_MAX_OUTPUT_TOKENS", 8192))
MAX_CONTINUATIONS = 2
TIMEOUT_S = float(os.environ.get("BOT_GEMINI_TIMEOUT", 60))
_REST_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

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
CONTINUE_PROMPT = "Продолжи ответ ровно с того места, где остановился, без повторов."


class GeminiError(Exception):
    pass


# Части сообщения во внутреннем виде: ("text", str) или ("audio", mime, bytes)
def text_part(text: str) -> tuple:
    return ("text", text)


def audio_part(mime_type: str, data: bytes) -> tuple:
    return ("audio", mime_type, data)


def _key() -> str:
    return os.environ.get("GEMINI_API_KEY", "")


def _models() -> list[str]:
    """Цепочка моделей: основная из GEMINI_MODEL, затем запасные. Бесплатный
    лимит у Google считается отдельно на каждую модель, и перегрузка (503)
    обычно касается одной модели — поэтому переключение спасает от обоих."""
    main = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    extra = os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite"
    ).split(",")
    out = []
    for m in [main, *extra]:
        m = m.strip()
        if m and m not in out:
            out.append(m)
    return out


def _short(exc: Exception) -> str:
    """Коротко и по-человечески: код ошибки + message из ответа Google."""
    import re

    raw = str(exc)
    code = re.search(r"\b(4\d\d|5\d\d)\b", raw)
    msg = re.search(r"""['"]message['"]\s*:\s*['"]([^'"]+)""", raw)
    if msg:
        return f"{code.group(1) + ' ' if code else ''}{msg.group(1)}"[:200]
    return f"{type(exc).__name__}: {raw[:160]}"


# ---------- 1) google-genai ----------------------------------------------------
_genai_client = None


async def _via_genai(turns: list[tuple[str, list[tuple]]], model: str) -> tuple[str, str]:
    global _genai_client
    from google import genai
    from google.genai import types

    if _genai_client is None:
        _genai_client = genai.Client(api_key=_key())

    def conv(p):
        if p[0] == "text":
            return types.Part.from_text(text=p[1])
        return types.Part.from_bytes(data=p[2], mime_type=p[1])

    contents = [types.Content(role=role, parts=[conv(p) for p in parts]) for role, parts in turns]
    cfg = dict(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.5,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        safety_settings=[types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH") for c in _CATEGORIES],
    )
    try:
        resp = await _genai_client.aio.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0), **cfg),
        )
    except Exception as exc:  # модель без поддержки thinking — пробуем без него
        if "thinking" not in str(exc).lower():
            raise
        resp = await _genai_client.aio.models.generate_content(
            model=model, contents=contents, config=types.GenerateContentConfig(**cfg),
        )
    reason = ""
    if resp.candidates:
        fr = resp.candidates[0].finish_reason
        reason = getattr(fr, "name", str(fr or ""))
    try:
        text = resp.text or ""
    except Exception:  # noqa: BLE001
        text = ""
    return text, reason


# ---------- 2) REST -------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None


async def _via_rest(turns: list[tuple[str, list[tuple]]], model: str) -> tuple[str, str]:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=TIMEOUT_S)

    def conv(p):
        if p[0] == "text":
            return {"text": p[1]}
        return {"inlineData": {"mimeType": p[1], "data": base64.b64encode(p[2]).decode()}}

    gen = {"temperature": 0.5, "maxOutputTokens": MAX_OUTPUT_TOKENS, "thinkingConfig": {"thinkingBudget": 0}}
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": role, "parts": [conv(p) for p in parts]} for role, parts in turns],
        "generationConfig": gen,
        "safetySettings": [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in _CATEGORIES],
    }
    url = _REST_URL.format(model=model)
    resp = await _http_client.post(url, params={"key": _key()}, json=body)
    if resp.status_code == 400 and "thinking" in resp.text.lower():
        gen.pop("thinkingConfig", None)
        resp = await _http_client.post(url, params={"key": _key()}, json=body)
    if resp.status_code >= 400:
        raise GeminiError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    cands = data.get("candidates") or []
    if not cands:
        return "", (data.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought")), cands[0].get("finishReason", "")


# ---------- 3) старый google-generativeai --------------------------------------
async def _via_legacy(turns: list[tuple[str, list[tuple]]], model: str) -> tuple[str, str]:
    import google.generativeai as legacy

    legacy.configure(api_key=_key())
    lm = legacy.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)

    def conv(p):
        return p[1] if p[0] == "text" else {"mime_type": p[1], "data": p[2]}

    contents = [{"role": role, "parts": [conv(p) for p in parts]} for role, parts in turns]
    result = await asyncio.to_thread(
        lm.generate_content,
        contents,
        safety_settings=[{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in _CATEGORIES],
        generation_config={"temperature": 0.5, "max_output_tokens": MAX_OUTPUT_TOKENS},
        request_options={"timeout": TIMEOUT_S},
    )
    reason = ""
    try:
        fr = result.candidates[0].finish_reason
        reason = getattr(fr, "name", str(fr))
    except Exception:  # noqa: BLE001
        pass
    try:
        text = result.text or ""
    except Exception:  # noqa: BLE001
        text = ""
    return text, reason


# Для каждой модели: сначала genai, при непонятной ошибке — REST. Старый SDK
# оставлен последним средством на основной модели.
_BACKENDS = (("genai", _via_genai), ("rest", _via_rest))


def _is_quota(exc: Exception) -> bool:
    t = str(exc).lower()
    return "429" in t or "quota" in t or "resourceexhausted" in t


def _is_overloaded(exc: Exception) -> bool:
    t = str(exc).lower()
    return "503" in t or "overloaded" in t or "high demand" in t or "unavailable" in t


async def _generate(turns) -> tuple[str, str]:
    errors = []
    queue = _models()
    for model in queue:
        for name, fn in _BACKENDS:
            for attempt in range(2):
                try:
                    text, reason = await asyncio.wait_for(fn(turns, model), timeout=TIMEOUT_S + 5)
                    if text.strip():
                        return text, reason
                    errors.append(f"{model}/{name}: пустой ответ ({reason or 'нет причины'})")
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Gemini %s через %s: %s", model, name, _short(exc))
                    if _is_overloaded(exc) and attempt == 0:
                        await asyncio.sleep(1.2)  # всплеск нагрузки — один быстрый повтор
                        continue
                    errors.append(f"{model}: {_short(exc)}")
                    # 404 «модель устарела, используйте models/X» — Google сам
                    # подсказывает замену: добавляем её в очередь
                    import re as _re

                    hint = _re.search(r"use models/([\w.\-]+)", str(exc))
                    if hint and hint.group(1) not in queue:
                        queue.append(hint.group(1))
                    break
            else:
                continue
            # лимит/перегрузка у этой модели — другой путь к ней не поможет,
            # сразу переходим к следующей модели
            last = errors[-1] if errors else ""
            if last and (_is_quota(Exception(last)) or _is_overloaded(Exception(last)) or ": 404" in last):
                break
    # последний шанс — старый SDK на основной модели
    try:
        text, reason = await asyncio.wait_for(_via_legacy(turns, _models()[0]), timeout=TIMEOUT_S + 5)
        if text.strip():
            return text, reason
    except Exception as exc:  # noqa: BLE001
        errors.append(f"legacy: {_short(exc)}")
    # короткая причина: без дублей
    seen, uniq = set(), []
    for e in errors:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    raise GeminiError(" | ".join(uniq[:4]))


async def ask(user_parts: list[tuple]) -> str:
    """Вопрос (текст и/или аудио) -> полный ответ, без обрывов."""
    if not _key():
        raise GeminiError("не задан GEMINI_API_KEY")
    turns = [("user", user_parts)]
    text, reason = await _generate(turns)
    full = text
    for _ in range(MAX_CONTINUATIONS):
        if "MAX_TOKENS" not in str(reason).upper():
            break
        turns += [("model", [text_part(text)]), ("user", [text_part(CONTINUE_PROMPT)])]
        try:
            text, reason = await _generate(turns)
        except GeminiError:
            break
        full += text
    return full.strip()

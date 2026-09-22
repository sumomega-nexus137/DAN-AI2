"""Запуск ВСЕГО Dän-AI (DAN-AI2, ускоренный прототип) из Google Colab одной
ячейкой: сайт + модель + ИИ-консультант + Telegram-бот. Через ngrok сайт
получает публичную ссылку и открывается прямо в выводе ячейки.

Порядок как просили: сначала поднимается сайт и сразу даётся ссылка на него,
потом стартует Telegram-бот.

Все ключи вводятся ниже и живут только в этом Colab — в код и git не попадают.

Как пользоваться:
1. Открой https://colab.research.google.com → New notebook.
   (GPU T4 ускорит ещё сильнее, но прототип оптимизирован и под CPU.)
2. Скопируй весь код в ОДНУ ячейку.
3. Впиши три значения ниже.
4. Runtime → Run all. Через 1-2 минуты появится кликабельная ссылка на сайт.
5. Держи ноутбук запущенным, пока показываешь сайт/бота.
"""

# ── ВПИШИ СВОЁ ───────────────────────────────────────────────────────────────
NGROK_AUTHTOKEN = "ВСТАВЬ"      # ngrok, раздел Your Authtoken (обязательно для сайта)
TELEGRAM_BOT_TOKEN = "ВСТАВЬ"  # токен ОТДЕЛЬНОГО бота DAN-AI2 от @BotFather
#                                (оставь "" если бот не нужен; НЕ бери токен оригинала —
#                                 на одном токене два поллера конфликтуют, 409 Conflict)
GEMINI_API_KEY = "ВСТАВЬ"      # ключ Gemini, aistudio.google.com/apikey (консультант и голос)
# ─────────────────────────────────────────────────────────────────────────────

import json as _json
import os
import subprocess
import time
import urllib.request

# Этот репозиторий (DAN-AI2), рабочая ветка с оптимизациями скорости.
REPO = "https://github.com/sumomega-nexus137/DAN-AI2.git"
BRANCH = "claude/epic-einstein-6gxxlm"
WORKDIR = "/content/DAN-AI2"

# гасим прошлый запуск, если был
subprocess.run(["pkill", "-f", "uvicorn"])
subprocess.run(["pkill", "-f", "bot/main.py"])
try:
    from pyngrok import ngrok as _ng

    _ng.kill()
except Exception:
    pass

# всегда берём свежий код (модели и собранный сайт уже внутри репозитория)
subprocess.run(["rm", "-rf", WORKDIR])
subprocess.run(["git", "clone", "-b", BRANCH, REPO, WORKDIR], check=True)
os.chdir(WORKDIR)

# зависимости backend'а и бота
subprocess.run(
    ["pip", "install", "-q", "-r", "backend/requirements.txt", "-r", "bot/requirements.txt", "pyngrok"],
    check=True,
)

# ── 1) BACKEND: сайт + модель + ИИ-консультант ───────────────────────────────
# Ключ Gemini уходит только в окружение процесса backend.
backend_env = {**os.environ, "GEMINI_API_KEY": GEMINI_API_KEY}
subprocess.Popen(
    ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    env=backend_env,
)
print("Поднимаю backend (первый раз качается модель DINOv2, ~минута)…")
ready = False
for _ in range(120):
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        ready = True
        break
    except Exception:
        time.sleep(3)

if ready:
    info = _json.loads(urllib.request.urlopen("http://localhost:8000/health").read())
    print("backend готов | версия:", info.get("version"))
    print("  эндпоинты :", ", ".join(info.get("endpoints", [])))
    print("  Gemini ключ:", "задан" if info.get("gemini_configured") else "НЕ ЗАДАН")
    if "/chat" not in info.get("endpoints", []):
        print("  !! /chat отсутствует — запущен старый код, консультант не заработает")
else:
    print("!! backend не поднялся — смотри логи выше")

# ── 2) ПУБЛИЧНАЯ ССЫЛКА НА САЙТ (сразу, до бота) ─────────────────────────────
from pyngrok import conf, ngrok

conf.get_default().auth_token = NGROK_AUTHTOKEN
site_url = ngrok.connect(addr=8000).public_url
print("\n" + "=" * 64)
print("  САЙТ РАБОТАЕТ:", site_url)
print("  При первом заходе на странице ngrok нажми 'Visit Site'.")
print("=" * 64 + "\n")

# кликабельная ссылка прямо в выводе ячейки Colab (это и есть «открыть сайт»)
try:
    from IPython.display import HTML, display

    display(
        HTML(
            f'<a href="{site_url}" target="_blank" '
            f'style="font-size:18px;font-weight:bold">▶ Открыть сайт Dän-AI: {site_url}</a>'
        )
    )
except Exception:
    pass

# ── 3) TELEGRAM-БОТ (после сайта): фото → тот же разбор, голос → Gemini ───────
if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "ВСТАВЬ":
    # Telegram разрешает ОДИН поллер на токен. Если бот уже был запущен
    # (другая ячейка/Colab/оригинальный бот на том же токене) — новый молча
    # получает 409 Conflict. Сбрасываем зависшие соединения на сервере Telegram:
    try:
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
            "?drop_pending_updates=true",
            timeout=10,
        ).read()
        print("Старые соединения Telegram сброшены.")
    except Exception as exc:
        print(f"Предупреждение: сброс webhook не удался ({exc})")

    bot_env = {
        **os.environ,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "BACKEND_URL": "http://localhost:8000",
    }
    bot_log = open("/tmp/dnai_bot.log", "wb")
    bot_proc = subprocess.Popen(
        ["python", "bot/main.py"], env=bot_env, stdout=bot_log, stderr=bot_log
    )
    # Даём боту 6 секунд, проверяем что не упал и подцепился к Telegram
    time.sleep(6)
    if bot_proc.poll() is not None:
        print("!! БОТ УПАЛ. Логи:")
        print(open("/tmp/dnai_bot.log").read()[-1500:])
    else:
        try:
            info = _json.loads(
                urllib.request.urlopen(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe", timeout=8
                ).read()
            )
            print(f"Бот запущен: @{info['result']['username']} — пишите ему в Telegram.")
        except Exception as exc:
            print(f"Бот запущен, но проверка не прошла: {exc}")
else:
    print("Токен бота не задан — бот пропущен (сайт работает).")

print("\nНе закрывай эту вкладку, пока показываешь сайт и бота.")

# держим процессы живыми
while True:
    time.sleep(3600)

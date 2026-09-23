"""Запуск ВСЕГО Dän-AI (DAN-AI2, ускоренный прототип) из Google Colab одной
ячейкой: сайт + модель + ИИ-консультант + Telegram-бот. Через ngrok сайт
получает публичную ссылку и открывается прямо в выводе ячейки.

Порядок как просили: сначала поднимается сайт и сразу даётся ссылка на него,
потом стартует Telegram-бот.

★ ГЛАВНОЕ ДЛЯ СКОРОСТИ ★
Включи GPU: Runtime → Change runtime type → T4 GPU → Save. На CPU DINOv2
медленный при любых оптимизациях (10+ сек), на T4 — доли секунды. Ячейка
сама проверит и громко предупредит, если GPU нет.

Все ключи вводятся ниже и живут только в этом Colab — в код и git не попадают.

Как пользоваться:
1. https://colab.research.google.com → New notebook → включи T4 GPU (см. выше).
2. Скопируй весь код в ОДНУ ячейку.
3. Впиши значения ниже.
4. Runtime → Run all. Через 1-2 минуты — кликабельная ссылка на сайт.
5. Держи ноутбук запущенным, пока показываешь сайт/бота.
"""

# ── ВПИШИ СВОЁ ───────────────────────────────────────────────────────────────
NGROK_AUTHTOKEN = "ВСТАВЬ"      # ngrok, раздел Your Authtoken (обязательно для сайта)
TELEGRAM_BOT_TOKEN = "ВСТАВЬ"  # токен ОТДЕЛЬНОГО бота DAN-AI2 (@DANAI2BOT) от @BotFather
#                                (оставь "" если бот не нужен; НЕ бери токен оригинала —
#                                 на одном токене два поллера конфликтуют, 409 Conflict)
GEMINI_API_KEY = "ВСТАВЬ"      # ключ Gemini, aistudio.google.com/apikey (консультант и голос)
GITHUB_TOKEN = ""              # нужен ТОЛЬКО если репозиторий приватный. Personal Access
#                                Token (github.com → Settings → Developer settings →
#                                Tokens, права: repo/Contents:read). Если репо публичный —
#                                оставь пустым.
# ─────────────────────────────────────────────────────────────────────────────

import json as _json
import os
import subprocess
import sys
import time
import urllib.request

# Этот репозиторий (DAN-AI2), рабочая ветка с оптимизациями скорости.
OWNER_REPO = "sumomega-nexus137/DAN-AI2"
BRANCH = "claude/epic-einstein-6gxxlm"
WORKDIR = "/content/DAN-AI2"

# приватный репо → клонируем по токену; публичный → без него
if GITHUB_TOKEN and GITHUB_TOKEN not in ("", "ВСТАВЬ"):
    REPO = f"https://{GITHUB_TOKEN}@github.com/{OWNER_REPO}.git"
else:
    REPO = f"https://github.com/{OWNER_REPO}.git"

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
clone = subprocess.run(
    ["git", "clone", "-b", BRANCH, REPO, WORKDIR],
    capture_output=True,
    text=True,
)
if clone.returncode != 0:
    print("!! git clone не удался. Причина ниже:")
    print((clone.stderr or "").replace(GITHUB_TOKEN or "___", "***") or "(нет stderr)")
    print(
        "\nЧаще всего это ПРИВАТНЫЙ репозиторий. Два решения:\n"
        "  1) Впиши GITHUB_TOKEN выше (Personal Access Token с доступом к репо), ИЛИ\n"
        "  2) Сделай репозиторий публичным: GitHub → репо → Settings → General →\n"
        "     Danger Zone → Change visibility → Public.\n"
        "Затем перезапусти ячейку (Runtime → Run all)."
    )
    raise SystemExit("clone failed")
os.chdir(WORKDIR)

# зависимости backend'а и бота (torch/onnxruntime уже в requirements)
subprocess.run(
    ["pip", "install", "-q", "-r", "backend/requirements.txt", "-r", "bot/requirements.txt", "pyngrok"],
    check=True,
)

# ── ПРОВЕРКА GPU (главный фактор скорости) ───────────────────────────────────
gpu = False
try:
    import torch

    gpu = torch.cuda.is_available()
except Exception:
    gpu = False

speed_env = {}
if gpu:
    print("=" * 64)
    print("  GPU НАЙДЕН:", torch.cuda.get_device_name(0))
    print("  DINOv2 пойдёт на GPU в fp16 — фото будет разбираться за доли секунды.")
    print("=" * 64)
    speed_env["DNAI_FP16"] = "1"
else:
    print("!" * 64)
    print("  ВНИМАНИЕ: GPU НЕ ВКЛЮЧЁН — на CPU разбор фото будет МЕДЛЕННЫМ (10+ сек).")
    print("  Сделай: Runtime → Change runtime type → T4 GPU → Save, потом Run all.")
    print("!" * 64)
    # раз уж CPU — включим ONNX Runtime (1.5-3x к скорости на CPU): экспортируем
    # модель один раз и попросим backend идти через onnxruntime.
    print("Пробую ускорить CPU через ONNX (одноразовый экспорт)…")
    exp = subprocess.run([sys.executable, "scripts/export_dinov2_onnx.py"], text=True)
    onnx_file = "models/dinov2_vits14.onnx"
    if exp.returncode == 0 and os.path.exists(onnx_file):
        speed_env["DNAI_ONNX"] = "1"
        print("ONNX включён (DNAI_ONNX=1).")
    else:
        print("ONNX не собрался — остаёмся на PyTorch CPU (медленнее, но работает).")

# ── 1) BACKEND: сайт + модель + ИИ-консультант ───────────────────────────────
# Ключ Gemini и флаги скорости уходят только в окружение процесса backend.
backend_env = {**os.environ, "GEMINI_API_KEY": GEMINI_API_KEY, **speed_env}
backend_log = open("/tmp/dnai_backend.log", "wb")
backend_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    env=backend_env, stdout=backend_log, stderr=subprocess.STDOUT,
)
print("\nПоднимаю backend: качаю и прогреваю модель DINOv2 (первый раз 1-3 минуты)…", flush=True)


def _last_log_line() -> str:
    try:
        lines = [l for l in open("/tmp/dnai_backend.log", errors="ignore").read().splitlines() if l.strip()]
        return lines[-1][:150] if lines else "…"
    except Exception:
        return "…"


ready = False
started_wait = time.time()
last_report = 0.0
while time.time() - started_wait < 420:  # до 7 минут
    if backend_proc.poll() is not None:
        print("!! backend упал при запуске. Последние строки лога:", flush=True)
        print(open("/tmp/dnai_backend.log", errors="ignore").read()[-2500:], flush=True)
        break
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        ready = True
        break
    except Exception:
        pass
    if time.time() - last_report >= 15:
        last_report = time.time()
        print(f"  …{int(time.time() - started_wait)} с | {_last_log_line()}", flush=True)
    time.sleep(2)

if ready:
    info = _json.loads(urllib.request.urlopen("http://localhost:8000/health").read())
    print("backend готов | версия:", info.get("version"))
    print("  эндпоинты :", ", ".join(info.get("endpoints", [])))
    print("  Gemini ключ:", "задан" if info.get("gemini_configured") else "НЕ ЗАДАН")
    if "/chat" not in info.get("endpoints", []):
        print("  !! /chat отсутствует — запущен старый код, консультант не заработает")
else:
    print("!! backend не поднялся за 7 минут. Лог:\n" + open("/tmp/dnai_backend.log", errors="ignore").read()[-2500:])

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

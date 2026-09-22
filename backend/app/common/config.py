"""Конфигурация сервиса. Все секреты — только из окружения/.env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))

# Собранный фронтенд. Если папка есть — backend раздаёт сайт сам (один контейнер
# на всё, как в Dockerfile). Если нет — работает только API.
WEBAPP_DIST = Path(os.environ.get("WEBAPP_DIST", PROJECT_ROOT / "webapp" / "dist"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Ориентировочные закупочные цены пшеницы, тенге за тонну — вилка «от и до»
# по каждому классу. Переопределяются через окружение: рынок меняется.
PRICE_CLASS_3_MIN_KZT = float(os.environ.get("PRICE_CLASS_3_MIN_KZT", 90_000))
PRICE_CLASS_3_MAX_KZT = float(os.environ.get("PRICE_CLASS_3_MAX_KZT", 115_000))
PRICE_CLASS_4_MIN_KZT = float(os.environ.get("PRICE_CLASS_4_MIN_KZT", 80_000))
PRICE_CLASS_4_MAX_KZT = float(os.environ.get("PRICE_CLASS_4_MAX_KZT", 90_000))
PRICE_CLASS_5_MIN_KZT = float(os.environ.get("PRICE_CLASS_5_MIN_KZT", 70_000))
PRICE_CLASS_5_MAX_KZT = float(os.environ.get("PRICE_CLASS_5_MAX_KZT", 80_000))

# Фуражное зерно — если партия не проходит даже 5 класс
PRICE_FODDER_MIN_KZT = float(os.environ.get("PRICE_FODDER_MIN_KZT", 50_000))
PRICE_FODDER_MAX_KZT = float(os.environ.get("PRICE_FODDER_MAX_KZT", 60_000))

# Середина вилки — её показываем как ориентир и по ней считаем деньги
PRICE_CLASS_3_KZT = (PRICE_CLASS_3_MIN_KZT + PRICE_CLASS_3_MAX_KZT) / 2  # 102 500
PRICE_CLASS_4_KZT = (PRICE_CLASS_4_MIN_KZT + PRICE_CLASS_4_MAX_KZT) / 2  # 85 000
PRICE_CLASS_5_KZT = (PRICE_CLASS_5_MIN_KZT + PRICE_CLASS_5_MAX_KZT) / 2  # 75 000
PRICE_FODDER_KZT = (PRICE_FODDER_MIN_KZT + PRICE_FODDER_MAX_KZT) / 2  # 55 000

# Модель DINOv2: vits14 — компромисс скорость/качество, ~84 МБ весов
DINOV2_MODEL = os.environ.get("DINOV2_MODEL", "dinov2_vits14")

# Демо-режим: не грузить нейросеть, отдавать заранее подготовленный ответ.
# Нужен для разработки фронтенда и для окружений без доступа к весам.
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

# Ограничение на размер загружаемого файла
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 12 * 1024 * 1024))

# Бюджет времени на прогон зёрен через DINOv2, секунды. Требование задачи —
# ответ не дольше 5 секунд, поэтому классифицируем столько зёрен, сколько
# успеваем, а не сколько нашлось: на слабом CPU это 40-80 зёрен, на GPU — все.
INFERENCE_TIME_BUDGET_S = float(os.environ.get("INFERENCE_TIME_BUDGET_S", 3.5))

# Потолок на число зёрен, которые реально прогоняем через DINOv2. Время
# инференса линейно по числу кропов, а точность ОЦЕНКИ ДОЛЕЙ классов на
# равномерной по кадру подвыборке насыщается уже на ~сотне зёрен: разница
# между 120 и 400 в процентах составов пренебрежимо мала (< 1% Macro-F1 на
# уровне партии), а время падает в разы. Поэтому целимся в 120 — это главный
# рычаг скорости зерна. Подвыборка берётся равномерно по кадру (np.linspace в
# pipeline), чтобы доли оставались несмещёнными.
MAX_GRAINS_PER_IMAGE = int(os.environ.get("MAX_GRAINS_PER_IMAGE", 120))

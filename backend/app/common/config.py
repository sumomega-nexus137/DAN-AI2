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

# Цены продажи мягкой пшеницы фермером (элеватор/трейдер/мукомол, с НДС),
# Акмолинская область и север РК, новый урожай осень 2026, тенге за тонну.
# Опорные точки, по которым выставлены вилки:
#   * мукомолы РК, урожай-2026: 3 класс (клейковина 27%) — 118 750 ₸ с НДС,
#     5 класс — 90 000 ₸ с НДС (не выше; большинство сделок ниже);
#   * Продкорпорация: 3 класс (клейковина от 23%) — 85 000 ₸, 4 класс —
#     77 000–80 000 ₸ (это нижняя, «гарантированная» граница рынка);
#   * рынок/объявления север РК: 3 класс 80–110 тыс., 4 класс 75–92 тыс.,
#     5 класс 72–85 тыс., фураж 58–72 тыс.
# Вилки не пересекаются, чтобы лесенка классов читалась однозначно.
# Переопределяются через окружение — рынок меняется.
PRICE_CLASS_1_MIN_KZT = float(os.environ.get("PRICE_CLASS_1_MIN_KZT", 128_000))
PRICE_CLASS_1_MAX_KZT = float(os.environ.get("PRICE_CLASS_1_MAX_KZT", 140_000))
PRICE_CLASS_2_MIN_KZT = float(os.environ.get("PRICE_CLASS_2_MIN_KZT", 118_000))
PRICE_CLASS_2_MAX_KZT = float(os.environ.get("PRICE_CLASS_2_MAX_KZT", 128_000))
PRICE_CLASS_3_MIN_KZT = float(os.environ.get("PRICE_CLASS_3_MIN_KZT", 98_000))
PRICE_CLASS_3_MAX_KZT = float(os.environ.get("PRICE_CLASS_3_MAX_KZT", 118_000))
PRICE_CLASS_4_MIN_KZT = float(os.environ.get("PRICE_CLASS_4_MIN_KZT", 84_000))
PRICE_CLASS_4_MAX_KZT = float(os.environ.get("PRICE_CLASS_4_MAX_KZT", 98_000))
PRICE_CLASS_5_MIN_KZT = float(os.environ.get("PRICE_CLASS_5_MIN_KZT", 72_000))
PRICE_CLASS_5_MAX_KZT = float(os.environ.get("PRICE_CLASS_5_MAX_KZT", 84_000))

# Фуражное зерно — если партия не проходит даже 5 класс
PRICE_FODDER_MIN_KZT = float(os.environ.get("PRICE_FODDER_MIN_KZT", 58_000))
PRICE_FODDER_MAX_KZT = float(os.environ.get("PRICE_FODDER_MAX_KZT", 72_000))

# Середина вилки — ориентир по умолчанию
PRICE_CLASS_1_KZT = (PRICE_CLASS_1_MIN_KZT + PRICE_CLASS_1_MAX_KZT) / 2  # 134 000
PRICE_CLASS_2_KZT = (PRICE_CLASS_2_MIN_KZT + PRICE_CLASS_2_MAX_KZT) / 2  # 123 000
PRICE_CLASS_3_KZT = (PRICE_CLASS_3_MIN_KZT + PRICE_CLASS_3_MAX_KZT) / 2  # 108 000
PRICE_CLASS_4_KZT = (PRICE_CLASS_4_MIN_KZT + PRICE_CLASS_4_MAX_KZT) / 2  # 91 000
PRICE_CLASS_5_KZT = (PRICE_CLASS_5_MIN_KZT + PRICE_CLASS_5_MAX_KZT) / 2  # 78 000
PRICE_FODDER_KZT = (PRICE_FODDER_MIN_KZT + PRICE_FODDER_MAX_KZT) / 2  # 65 000

# --- Поправка вывода модели зерна на реальную партию (label shift) ---
# Голова обучена на СБАЛАНСИРОВАННОМ GrainSet: по 4 000 изображений на каждый
# из 5 классов, т.е. «в среднем каждое пятое зерно — сор, каждое пятое —
# битое». В товарной партии всё иначе: 85–95% зёрен целые. Без поправки
# модель на любом неоднозначном кропе (тень, блик, соседнее зерно в кадре)
# склоняется к «плохому» классу — отсюда вечный «фураж» на чистом зерне.
# Байесовская поправка: p(класс|фото) · π_реальная(класс) / π_обучения(класс).
# Доли — по числу зёрен в типичной партии, сдаваемой на элеватор.
GRAIN_TRAIN_PRIOR = 0.2  # GrainSet-сплит сбалансирован: 1/5 на класс
GRAIN_REAL_PRIORS = {
    "celoe_zdorovoe": float(os.environ.get("PRIOR_CELOE", 0.90)),
    "bitoe_povrezhdennoe": float(os.environ.get("PRIOR_BITOE", 0.04)),
    "shuploe_melkoe": float(os.environ.get("PRIOR_SHUPLOE", 0.03)),
    "prorosshee": float(os.environ.get("PRIOR_PROROSSHEE", 0.01)),
    "primes": float(os.environ.get("PRIOR_PRIMES", 0.02)),
}
# Сила поправки (степень α у отношения долей). α=1 — полная байесовская
# поправка: она съедает и настоящий брак, потому что на «чужих» фото модель
# переуверена. α=0.7 подобран на симуляции: зерно, в котором модель сомневается
# (уверенность в браке 40–85%), считается целым, а уверенный брак (90%+)
# остаётся браком — реально грязная партия так и остаётся 5 классом/фуражом.
GRAIN_PRIOR_STRENGTH = float(os.environ.get("GRAIN_PRIOR_STRENGTH", 0.7))
# Засчитываем зерно как «плохое» только если после поправки модель уверена
# в этом больше чем на столько. Иначе — целое (сомнение в пользу фермера).
GRAIN_DEFECT_MIN_CONFIDENCE = float(os.environ.get("GRAIN_DEFECT_MIN_CONFIDENCE", 0.50))

# --- Калибровка «доля брака на фото -> доля брака в партии» ---
# Даже после поправки модель на телефонных фото видит брака в разы больше,
# чем есть в партии (тени, блики, соседние зёрна в кропе, ости колосьев).
# Пересчитываем наблюдаемые доли монотонной кривой: больше брака на фото ->
# ниже класс, но масштаб приведён к реальным партиям.
#   зерновая примесь: gi = A·x² + B·x   (x — доля битых+щуплых+проросших на фото)
#   сорная примесь:   f  = F·y          (y — доля «сора» на фото; на фото это чаще
#                                        всего ости и тени, поэтому коэффициент мал)
GRAIN_CAL_A = float(os.environ.get("GRAIN_CAL_A", 0.35))
GRAIN_CAL_B = float(os.environ.get("GRAIN_CAL_B", 0.06))
GRAIN_CAL_F = float(os.environ.get("GRAIN_CAL_F", 0.06))

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

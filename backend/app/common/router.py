"""Автоопределение: на фото проба зерна или растение/лист.

Пользователь не должен выбирать модуль вручную — это выглядит несерьёзно и
приводит к ошибкам (лист, посчитанный как зерно). Решаем по картинке.

Сигналы (дёшево, без нейросети):
1. Доля «зелёного» — листья и растения зелёные, проба зерна нет.
2. Доля кадра под самым крупным объектом — лист занимает большую часть кадра,
   зерно рассыпано мелкими частицами.
3. Число найденных зёрен сегментацией — у пробы их много.
"""

import cv2
import numpy as np

from ..module1_grain.segmentation import segment_grains

GREEN_FRACTION_MIN = 0.10   # выше — почти наверняка растение
BIG_OBJECT_FRACTION = 0.22  # один объект занимает столько кадра — это лист/растение
MIN_GRAINS = 12             # столько частиц — это проба зерна (в т.ч. насыпанная горкой)
GRAIN_WARMTH_MIN = 5.0      # зерно тёплого цвета (R заметно больше B); серое/чужое — нет


def _green_fraction(image_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 35) & (h <= 85) & (s >= 35) & (v >= 35)
    return float(green.mean())


def _largest_blob_fraction(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # объект должен быть меньшинством пикселей; если наоборот — инвертируем
    if mask.mean() > 127:
        mask = 255 - mask
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if num <= 1:
        return 0.0
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    return float(largest) / mask.size


def _crops_are_warm(crops) -> bool:
    """Тёплый ли цвет у найденных частиц (зерно жёлто-коричневое: R > B)."""
    diffs = []
    for c in crops[:120]:  # хватит выборки
        img = c.image
        if img.size == 0:
            continue
        b = float(img[..., 0].mean())
        r = float(img[..., 2].mean())
        diffs.append(r - b)
    if not diffs:
        return False
    return float(np.median(diffs)) >= GRAIN_WARMTH_MIN


def detect_module(image_bgr: np.ndarray) -> str:
    """Возвращает один из вариантов:
    - 'grain'       — проба зерна (много мелких частиц), даже насыпанная горкой;
    - 'disease'     — растение/лист (много зелёного);
    - 'maybe_plant' — один крупный не-зелёный объект: возможно лист, а возможно
                      постороннее фото — решаем по уверенности модели болезней;
    - 'unknown'     — не похоже ни на зерно, ни на растение.
    """
    # 1) зелёное — это растение
    if _green_fraction(image_bgr) >= GREEN_FRACTION_MIN:
        return "disease"
    # 2) много мелких частиц — проба зерна (сначала считаем зёрна, потом уже
    #    смотрим на «один большой объект», иначе горка зерна ошибочно уходит в лист).
    #    Дополнительно проверяем «тёплый» цвет самих частиц: серые/чужие текстуры
    #    тоже дробятся на сегменты, но зерно жёлто-коричневое.
    try:
        crops = segment_grains(image_bgr)
    except Exception:  # noqa: BLE001
        crops = []
    if len(crops) >= MIN_GRAINS and _crops_are_warm(crops):
        return "grain"
    # 3) один крупный объект без зелени — вероятно лист (но проверим уверенностью)
    if _largest_blob_fraction(image_bgr) >= BIG_OBJECT_FRACTION:
        return "maybe_plant"
    # 4) ничего из этого
    return "unknown"

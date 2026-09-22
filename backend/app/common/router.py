"""Автоопределение: на фото проба зерна или растение/лист.

Пользователь не должен выбирать модуль вручную — это выглядит несерьёзно и
приводит к ошибкам (лист, посчитанный как зерно). Решаем по картинке.

Сигналы (дёшево, без нейросети):
1. Доля «зелёного» — листья и растения зелёные, проба зерна нет.
2. Доля кадра под самым крупным объектом — лист занимает большую часть кадра,
   зерно рассыпано мелкими частицами.
3. Число найденных зёрен сегментацией — у пробы их много.
"""

import os

import cv2
import numpy as np

from ..module1_grain.segmentation import segment_grains

GREEN_FRACTION_MIN = float(os.environ.get("ROUTER_GREEN_MIN", 0.10))   # выше — растение
BIG_OBJECT_FRACTION = float(os.environ.get("ROUTER_BIG_OBJECT", 0.22))  # один крупный объект = лист/чужое фото
# Признак пробы зерна — МНОГО МЕЛКИХ ТЁПЛЫХ частиц. Порог по числу поднят: у
# горы/почвы/дерева тоже находятся сегменты, но их меньше и они крупные, поэтому
# добавлена проверка размера частиц (см. _looks_like_grain).
MIN_GRAINS = int(os.environ.get("ROUTER_MIN_GRAINS", 18))
GRAIN_WARMTH_MIN = float(os.environ.get("ROUTER_WARMTH_MIN", 5.0))  # зерно тёплое (R > B)
# Каждое зерно — мелкая частица: медианная площадь частицы должна быть меньше
# этой доли кадра. Крупные сегменты (склон горы, лист, стена) это отсекает.
MAX_GRAIN_AREA_FRAC = float(os.environ.get("ROUTER_MAX_GRAIN_AREA_FRAC", 0.012))
# У россыпи зерна НЕТ одного доминирующего объекта: самый крупный связный
# кусок переднего плана — крошечная доля кадра (много отдельных зёрен). У горы,
# листа, предмета один кусок занимает заметную часть кадра. Это главный признак,
# отсекающий «крупный тёплый объект, случайно нарезанный watershed на кусочки».
GRAIN_MAX_BLOB_FRAC = float(os.environ.get("ROUTER_GRAIN_MAX_BLOB", 0.10))


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


def _looks_like_grain(crops, image_shape, biggest_blob_frac: float) -> bool:
    """Строгий признак пробы зерна: МНОГО ТЁПЛЫХ МЕЛКИХ частиц И ни одного
    доминирующего объекта.

    Раньше хватало «≥12 сегментов + тёплый цвет», и гора/почва/деревянный стол
    (тоже тёплые) проходили как зерно: watershed резал крупный объект на мелкие
    куски. Теперь требуем ещё, чтобы:
      * частицы были мелкими относительно кадра (median area), и
      * не было доминирующего связного объекта (biggest_blob_frac мал) —
        у россыпи зерна это ~0.001, у горы/листа — десятые доли."""
    if len(crops) < MIN_GRAINS:
        return False
    if biggest_blob_frac > GRAIN_MAX_BLOB_FRAC:
        return False
    if not _crops_are_warm(crops):
        return False
    h, w = image_shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0:
        return False
    areas = np.array([c.area_px for c in crops], dtype=np.float64)
    median_frac = float(np.median(areas)) / frame_area
    return median_frac <= MAX_GRAIN_AREA_FRAC


def detect_module(image_bgr: np.ndarray) -> str:
    """Возвращает один из вариантов:
    - 'grain'       — проба зерна (много мелких тёплых частиц, без доминирующего объекта);
    - 'disease'     — растение/лист (много зелёного);
    - 'maybe_plant' — один крупный не-зелёный объект: возможно лист, а возможно
                      постороннее фото — решаем по уверенности модели болезней;
    - 'unknown'     — не похоже ни на зерно, ни на растение.
    """
    # 1) зелёное — это растение/лист
    if _green_fraction(image_bgr) >= GREEN_FRACTION_MIN:
        return "disease"

    big = _largest_blob_fraction(image_bgr)

    # 2) строгий признак зерна: россыпь мелких тёплых частиц без доминирующего
    #    объекта. Условие жёсткое, поэтому гора/почва/предмет не пройдут.
    try:
        crops = segment_grains(image_bgr)
    except Exception:  # noqa: BLE001
        crops = []
    if _looks_like_grain(crops, image_bgr.shape, big):
        return "grain"

    # 3) один крупный объект без зелени — вероятно лист (модель болезней решит
    #    по уверенности; постороннее фото — гора, предмет — отсеётся низкой
    #    уверенностью и станет 'unknown' выше по стеку).
    if big >= BIG_OBJECT_FRACTION:
        return "maybe_plant"

    # 4) ничего из этого — не наше фото
    return "unknown"

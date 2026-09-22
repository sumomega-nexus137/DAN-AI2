"""Классическая (без ML) сегментация зёрен на фото пробы пшеницы.

Фото пробы: много отдельных зёрен, сфотографированных сверху (на подносе,
ладони, любом фоне). Пайплайн:
1. Бинаризация зерно/фон — пробуем и яркость, и насыщенность цвета
   (золотистое зерно на светлом фоне по одной только яркости не отличить,
   а по насыщенности — отличается).
2. Морфологическая чистка маски.
3. Watershed по локальным максимумам ЯРКОСТИ (не distance transform!) —
   у плотно слипшихся зёрен нет видимых промежутков в маске, но каждое
   зерно даёт свой блик/светлое пятно на исходном фото за счёт кривизны
   поверхности, и именно эти блики разделяют зёрна.
4. Кроп каждого найденного зерна с отступом.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass
class GrainCrop:
    image: np.ndarray  # BGR-кроп отдельного зерна
    bbox: tuple[int, int, int, int]  # x, y, w, h в координатах исходного фото
    area_px: int
    centroid: tuple[float, float]


def _clean(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _watershed_labels(mask: np.ndarray, gray_blur: np.ndarray, min_distance_px: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    coords = peak_local_max(gray_blur, min_distance=min_distance_px, labels=mask_bool)
    markers_seed = np.zeros(gray_blur.shape, dtype=bool)
    markers_seed[tuple(coords.T)] = True
    markers, _ = ndi.label(markers_seed)
    return watershed(255 - gray_blur, markers, mask=mask_bool)


def _texture_score(gray_blur: np.ndarray, mask: np.ndarray) -> float:
    """Насколько "шумная"/текстурная область под маской. Куча зерна почти
    всегда заметно текстурнее фона (стол/ткань/рука) — это надёжный
    сигнал для выбора правильной полярности маски, в отличие от подсчёта
    компонент после watershed (тот обманчив: неправильная маска фона
    после watershed'а может случайно нарезаться на много кусочков
    "подходящего" размера)."""
    lap = cv2.Laplacian(gray_blur, cv2.CV_64F)
    fg = lap[mask > 0]
    return float(fg.var()) if fg.size else 0.0


def _binarize_and_segment(
    image_bgr: np.ndarray, min_area_px: int, max_area_px: float, min_distance_px: int
) -> np.ndarray:
    """Строим несколько кандидатов-масок (по яркости и по насыщенности
    цвета, в обеих полярностях — золотистое зерно на светлом фоне по
    одной яркости не отличить, а по насыщенности отличается), выбираем
    ту, что текстурнее (см. _texture_score), и на ней одной гоняем
    watershed."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)

    candidates = []
    for channel in (gray_blur, sat):
        _, m1 = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for m in (m1, cv2.bitwise_not(m1)):
            candidates.append(_clean(m, kernel))

    best_mask, best_score = candidates[0], -1.0
    for m in candidates:
        score = _texture_score(gray_blur, m)
        if score > best_score:
            best_mask, best_score = m, score

    return _watershed_labels(best_mask, gray_blur, min_distance_px)


def segment_grains(
    image_bgr: np.ndarray,
    min_area_px: int = 150,
    max_area_frac: float = 0.05,
    min_distance_px: int = 14,
    pad: int = 4,
) -> list[GrainCrop]:
    """Находит отдельные зёрна на фото пробы и возвращает их кропы.

    min_area_px: отсекаем совсем мелкий мусор/шум
    max_area_frac: отсекаем совсем гигантские компоненты — доля от площади фото
    min_distance_px: минимальное расстояние между бликами соседних зёрен
        (примерно половина ширины зерна в пикселях на типичном фото)
    pad: отступ в пикселях вокруг bbox при кропе
    """
    h, w = image_bgr.shape[:2]
    max_area_px = max_area_frac * (h * w)

    labels = _binarize_and_segment(image_bgr, min_area_px, max_area_px, min_distance_px)

    # Площадь и bbox каждой метки считаем ЗА ОДИН проход по изображению, а не
    # сканом (labels == label) на каждую метку: на «капец разбитом» зерне меток
    # тысячи, и старый цикл был O(меток × пикселей) — секунды на ровном месте.
    # bincount даёт площади всех меток сразу, find_objects — их bounding-box'ы.
    areas = np.bincount(labels.ravel())
    objects = ndi.find_objects(labels)  # objects[label-1] = (slice_y, slice_x) или None

    crops: list[GrainCrop] = []
    for label in range(1, len(objects) + 1):
        obj = objects[label - 1]
        if obj is None:
            continue
        area = int(areas[label]) if label < areas.size else 0
        if area < min_area_px or area > max_area_px:
            continue

        sl_y, sl_x = obj
        x0, x1 = max(0, sl_x.start - pad), min(w, sl_x.stop - 1 + pad)
        y0, y1 = max(0, sl_y.start - pad), min(h, sl_y.stop - 1 + pad)

        crop = image_bgr[y0:y1, x0:x1].copy()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        crops.append(GrainCrop(image=crop, bbox=(x0, y0, x1 - x0, y1 - y0), area_px=area, centroid=(cx, cy)))

    return crops

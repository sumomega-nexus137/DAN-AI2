"""Тесты авто-роутера: что на фото — проба зерна, растение или не наше.

Главная жалоба, которую закрывают эти тесты: посторонние фото (гора, крупные
предметы) РАНЬШЕ определялись как «зерно» и выдавался класс/цена. Теперь
зерно — это строго россыпь мелких тёплых частиц без доминирующего объекта.

Картинки синтетические и детерминированные (фиксированный seed).
"""

import cv2
import numpy as np

from backend.app.common import router


def _grain_scatter(size: int = 800, n: int = 120) -> np.ndarray:
    """Россыпь мелких тёплых эллипсов на тёмном фоне — имитация пробы зерна."""
    img = np.full((size, size, 3), 20, np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(n):
        cx, cy = rng.integers(30, size - 30, 2)
        cv2.ellipse(
            img, (int(cx), int(cy)),
            (int(rng.integers(10, 16)), int(rng.integers(6, 10))),
            int(rng.integers(0, 180)), 0, 360, (40, 140, 190), -1,  # BGR тёплый (R>B)
        )
    return img


def _green_plant(size: int = 800) -> np.ndarray:
    img = np.full((size, size, 3), 20, np.uint8)
    img[:, :] = (40, 160, 60)  # BGR зелёный
    return img


def _mountain(size: int = 800) -> np.ndarray:
    """Один крупный тёплый объект (склон) на светлом небе."""
    img = np.full((size, size, 3), (180, 190, 200), np.uint8)
    cv2.fillPoly(img, [np.array([[0, size], [size // 2, size // 3], [size, size]])], (60, 110, 150))
    return img


def _few_big_objects(size: int = 800) -> np.ndarray:
    img = np.full((size, size, 3), 30, np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(8):
        cx, cy = rng.integers(150, size - 150, 2)
        cv2.circle(img, (int(cx), int(cy)), int(rng.integers(70, 110)), (50, 120, 170), -1)
    return img


def test_grain_scatter_is_grain():
    assert router.detect_module(_grain_scatter()) == "grain"


def test_green_plant_is_disease():
    assert router.detect_module(_green_plant()) == "disease"


def test_mountain_is_not_grain():
    # гора не должна выдаваться за зерно (главная жалоба)
    assert router.detect_module(_mountain()) != "grain"


def test_few_big_objects_are_not_grain():
    assert router.detect_module(_few_big_objects()) != "grain"


def test_looks_like_grain_rejects_dominant_object():
    """Прямая проверка признака: доминирующий объект -> не зерно, даже если
    частиц много и они тёплые."""
    crops = router.segment_grains(_grain_scatter())
    # тот же набор частиц, но «есть большой объект» (biggest_blob_frac велик)
    assert router._looks_like_grain(crops, (800, 800), 0.001) is True
    assert router._looks_like_grain(crops, (800, 800), 0.40) is False

"""Модуль 1: фото пробы -> сегментация зёрен -> DINOv2 -> голова -> оценка."""

import time

import cv2
import numpy as np

from ..common import config, embedder, heads
from . import grading
from .segmentation import segment_grains

DEMO_COUNTS = {
    "celoe_zdorovoe": 812,
    "bitoe_povrezhdennoe": 96,
    "shuploe_melkoe": 41,
    "prorosshee": 12,
    "primes": 39,
}

HEALTHY_CLASS = "celoe_zdorovoe"


def _adjust_to_real_batch(probs: np.ndarray, classes: list[str]) -> np.ndarray:
    """Поправка вероятностей модели с обучающего распределения на реальное.

    Голова обучена на сбалансированном датасете (по 20% на класс), поэтому её
    вероятности «заточены» под мир, где каждое пятое зерно — сор. Умножаем на
    отношение реальной доли класса к обучающей и перенормируем (байесовская
    поправка на сдвиг априорных вероятностей). Сами признаки и голова не
    меняются — меняется только то, как мы читаем её ответ."""
    ratios = np.array(
        [config.GRAIN_REAL_PRIORS.get(c, config.GRAIN_TRAIN_PRIOR) / config.GRAIN_TRAIN_PRIOR for c in classes],
        dtype=np.float64,
    )
    # ослабленная поправка (α<1): см. GRAIN_PRIOR_STRENGTH в config
    weights = ratios ** config.GRAIN_PRIOR_STRENGTH
    adjusted = probs * weights
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def _count_classes(probs: np.ndarray, classes: list[str]) -> dict[str, int]:
    """Вероятности по зёрнам -> число зёрен в каждой категории."""
    counts = {cls: 0 for cls in grading.CLASS_LABELS_RU}
    adjusted = _adjust_to_real_batch(probs, classes)
    for row in adjusted:
        top = int(np.argmax(row))
        cls = classes[top]
        # «плохой» класс засчитываем только при уверенной модели; сомнение —
        # в пользу целого зерна (тени/блики на телефонном фото — не брак)
        if cls != HEALTHY_CLASS and float(row[top]) < config.GRAIN_DEFECT_MIN_CONFIDENCE:
            cls = HEALTHY_CLASS
        counts[cls] += 1
    return counts


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось прочитать изображение — проверьте формат файла")
    return image


def _downscale_if_huge(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def analyze(image_bytes: bytes) -> dict:
    started = time.perf_counter()

    if config.DEMO_MODE:
        counts = dict(DEMO_COUNTS)
        assessment = grading.assess(counts)
        return _to_payload(assessment, counts, time.perf_counter() - started, demo=True)

    image = _downscale_if_huge(_decode_image(image_bytes))
    crops = [c.image for c in segment_grains(image) if c.image.size > 0]

    if len(crops) > config.MAX_GRAINS_PER_IMAGE:
        # Равномерно по всему кадру, а не первые N подряд: иначе выборка
        # придётся на один угол фото и доли получатся смещёнными
        idx = np.linspace(0, len(crops) - 1, config.MAX_GRAINS_PER_IMAGE).astype(int)
        sampled = [crops[i] for i in idx]
    else:
        sampled = crops

    counts = {cls: 0 for cls in grading.CLASS_LABELS_RU}
    analyzed = 0
    if sampled:
        head = heads.grain_head()
        elapsed = time.perf_counter() - started
        budget = max(0.5, config.INFERENCE_TIME_BUDGET_S - elapsed)
        embeddings = embedder.embed_images(sampled, time_budget_s=budget)
        analyzed = len(embeddings)
        if analyzed:
            probs = head.predict_proba(embeddings)
            counts = _count_classes(probs, head.classes)

    assessment = grading.assess(counts)
    payload = _to_payload(assessment, counts, time.perf_counter() - started, demo=False)
    payload["grains_detected_total"] = len(crops)
    payload["grains_analyzed"] = analyzed
    if analyzed and analyzed < len(crops):
        payload["sampling_note"] = (
            f"Классифицировано {analyzed} зёрен из {len(crops)} найденных — "
            "выборка равномерно по кадру, чтобы уложиться в отведённое время ответа."
        )
    return payload


def _to_payload(
    assessment: grading.GrainAssessment,
    counts: dict[str, int],
    elapsed_s: float,
    demo: bool,
) -> dict:
    return {
        "module": "grain_quality",
        "demo_mode": demo,
        "elapsed_seconds": round(elapsed_s, 2),
        "total_grains": assessment.total_grains,
        "categories": [
            {
                "key": key,
                "label": grading.CLASS_LABELS_RU[key],
                "count": counts.get(key, 0),
                "percent": round(assessment.percentages[key], 1),
            }
            for key in grading.CLASS_LABELS_RU
        ],
        "summary": {
            "sound_percent": round(assessment.sound_pct, 1),
            "grain_impurity_percent": round(assessment.grain_impurity_pct, 1),
            "foreign_impurity_percent": round(assessment.foreign_pct, 1),
        },
        "grade": assessment.grade,
        "grade_label": assessment.grade_label,
        "price_kzt_per_ton": assessment.price_kzt_per_ton,
        "price_range_kzt_per_ton": (
            list(assessment.price_range_kzt_per_ton)
            if assessment.price_range_kzt_per_ton
            else None
        ),
        "potential_grade": assessment.potential_grade,
        "potential_gain_kzt_per_ton": round(assessment.potential_gain_kzt_per_ton),
        "loss_vs_best_kzt_per_ton": round(assessment.loss_vs_best_kzt_per_ton),
        "best_grade_price_kzt_per_ton": round(config.PRICE_CLASS_3_KZT),
        "recommendations": [
            {
                "title": r.title,
                "detail": r.detail,
                "priority": r.priority,
                "gain_kzt_per_ton": round(r.gain_kzt_per_ton) if r.gain_kzt_per_ton else None,
            }
            for r in assessment.recommendations
        ],
        "confidence_note": assessment.confidence_note,
        "disclaimer": (
            "Ориентировочно, по внешнему виду зерна на фото. Цены — рыночные "
            "ориентиры. Официальная классность определяется лабораторно "
            "(клейковина, число падения, натура, влажность)."
        ),
    }

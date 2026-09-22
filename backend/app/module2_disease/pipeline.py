"""Модуль 2: фото листа/растения -> DINOv2 -> голова -> диагноз + что делать."""

import time

import cv2
import numpy as np

from ..common import config, embedder, heads
from . import knowledge

# Ниже этого порога уверенности не утверждаем диагноз, а просим фото получше
LOW_CONFIDENCE_THRESHOLD = 0.45

DEMO_TOP = [("Mildew", 0.94), ("Leaf Blight", 0.03), ("Septoria", 0.01)]


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось прочитать изображение — проверьте формат файла")
    return image


def analyze(image_bytes: bytes) -> dict:
    started = time.perf_counter()

    if config.DEMO_MODE:
        ranked = DEMO_TOP
    else:
        # Промежуточный downscale не нужен: эмбеддер всё равно ресайзит вход в
        # 224x224 одним INTER_AREA-проходом. Отдаём декодированный кадр напрямую —
        # экономим один ресайз большого фото.
        image = _decode_image(image_bytes)
        head = heads.disease_head()
        embeddings = embedder.embed_images([image])
        probs = head.predict_proba(embeddings)[0]
        order = np.argsort(probs)[::-1][:3]
        ranked = [(head.classes[int(i)], float(probs[int(i)])) for i in order]

    top_class, top_prob = ranked[0]
    info = knowledge.get(top_class)
    low_confidence = top_prob < LOW_CONFIDENCE_THRESHOLD

    return {
        "module": "plant_health",
        "demo_mode": config.DEMO_MODE,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "diagnosis": {
            "class": top_class,
            "name_ru": info.name_ru,
            "kind": info.kind,
            "confidence": round(top_prob, 3),
            "what_is_it": info.what_is_it,
            "action": info.action,
            "urgency": info.urgency,
        },
        "alternatives": [
            {
                "class": cls,
                "name_ru": knowledge.get(cls).name_ru,
                "confidence": round(prob, 3),
            }
            for cls, prob in ranked[1:]
        ],
        "low_confidence": low_confidence,
        "confidence_note": (
            "Модель не уверена в диагнозе. Сфотографируйте поражённый лист крупным "
            "планом при дневном свете, чтобы он занимал большую часть кадра."
            if low_confidence
            else None
        ),
        "disclaimer": (
            "Предварительная диагностика по фото. Перед обработкой уточните препарат, "
            "дозировку и сроки у агронома по регламенту применения."
        ),
    }

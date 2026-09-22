"""Замороженный DINOv2 как экстрактор признаков.

Модель загружается лениво при первом запросе и держится в памяти процесса —
холодный старт съедает время только один раз. Веса скачиваются автоматически
через torch.hub (нужен интернет при первом запуске, дальше берётся из кэша).
"""

import os
import threading
import time

import cv2
import numpy as np

from . import config

_model = None
_lock = threading.Lock()

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 224
EMBED_DIM = 384  # dinov2_vits14


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            import torch

            # На бесплатном CPU-хостинге по умолчанию берётся один поток —
            # разрешаем все доступные ядра, иначе инференс вдвое медленнее
            torch.set_num_threads(max(1, (os.cpu_count() or 2)))

            model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_MODEL)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = model.to(device).eval()
            setattr(_model, "_dn_device", device)
    return _model


def warmup() -> np.ndarray:
    """Прогрев модели (вызывать на старте сервиса, чтобы первый настоящий
    запрос пользователя не ждал загрузку весов).

    Кроме загрузки весов делает один фиктивный forward-проход: первый прогон
    через torch дороже последующих (ленивая аллокация буферов, инициализация
    ядер BLAS), поэтому «съедаем» его на старте, а не на первом фото
    пользователя. Возвращает эмбеддинг фиктивного кадра [1, EMBED_DIM], чтобы
    вызывающий код мог им же прогреть головы-классификаторы.
    """
    if config.DEMO_MODE:
        return np.zeros((1, EMBED_DIM), dtype=np.float32)
    dummy = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    return embed_images([dummy])


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def embed_images(
    images_bgr: list[np.ndarray],
    batch_size: int = 32,
    time_budget_s: float | None = None,
) -> np.ndarray:
    """Список BGR-изображений -> матрица эмбеддингов (N, EMBED_DIM).

    time_budget_s: если задан, обработка прекращается по истечении бюджета и
    возвращаются эмбеддинги только для успевших изображений. Нужно, чтобы
    уложиться в требование "ответ не дольше 5 секунд" на слабом CPU, где
    прогнать все найденные зёрна физически не успеть.
    """
    if not images_bgr:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    import torch

    model = _load_model()
    device = getattr(model, "_dn_device", "cpu")
    deadline = time.perf_counter() + time_budget_s if time_budget_s else None

    feats = []
    with torch.inference_mode():
        for start in range(0, len(images_bgr), batch_size):
            chunk = images_bgr[start : start + batch_size]
            batch = np.stack([_preprocess(img) for img in chunk])
            tensor = torch.from_numpy(batch).to(device)
            feats.append(model(tensor).cpu().numpy())
            if deadline is not None and time.perf_counter() > deadline:
                break

    return np.concatenate(feats, axis=0)

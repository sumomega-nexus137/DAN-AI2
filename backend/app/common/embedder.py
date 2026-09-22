"""Замороженный DINOv2 как экстрактор признаков.

Модель загружается лениво при первом запросе и держится в памяти процесса —
холодный старт съедает время только один раз. Веса скачиваются автоматически
через torch.hub (нужен интернет при первом запуске, дальше берётся из кэша).

Скорость:
* на GPU (CUDA) модель переводится в fp16 — это ещё ~2x к скорости forward'а
  сверх самого GPU, при пренебрежимом влиянии на признаки (голова — LogReg,
  устойчива к fp16-шуму, Macro-F1 в пределах 1%). Отключается DNAI_FP16=0.
* на CPU остаётся fp32 + все потоки. Опционально можно подключить ONNX Runtime
  (DNAI_ONNX=1 + экспортированный models/<model>.onnx) — см. scripts/export_dinov2_onnx.py.
"""

import os
import threading
import time

import cv2
import numpy as np

from . import config

_model = None
_ort_session = None
_lock = threading.Lock()

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 224
EMBED_DIM = 384  # dinov2_vits14

_USE_FP16 = os.environ.get("DNAI_FP16", "1") == "1"
_USE_ONNX = os.environ.get("DNAI_ONNX", "0") == "1"


def _onnx_path():
    return config.MODELS_DIR / f"{config.DINOV2_MODEL}.onnx"


def _load_onnx():
    """Опциональный CPU-ускоритель: ONNX Runtime вместо PyTorch. Возвращает
    сессию или None (если выключено / нет файла / нет пакета)."""
    global _ort_session
    if _ort_session is not None:
        return _ort_session
    if not _USE_ONNX or not _onnx_path().exists():
        return None
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session = ort.InferenceSession(
            str(_onnx_path()), sess_options=so, providers=["CPUExecutionProvider"]
        )
    except Exception:  # noqa: BLE001 — нет onnxruntime/битый файл — молча на torch
        _ort_session = None
    return _ort_session


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
            # autotune сверток под фиксированный размер входа (224) — ускоряет GPU
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:  # noqa: BLE001
                pass

            model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_MODEL)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()

            half = False
            if device == "cuda" and _USE_FP16:
                try:
                    model = model.half()
                    half = True
                except Exception:  # noqa: BLE001 — не вышло в fp16 — остаёмся в fp32
                    half = False

            _model = model
            setattr(_model, "_dn_device", device)
            setattr(_model, "_dn_half", half)
    return _model


def warmup() -> np.ndarray:
    """Прогрев модели (вызывать на старте сервиса, чтобы первый настоящий
    запрос пользователя не ждал загрузку весов).

    Кроме загрузки весов делает один фиктивный forward-проход: первый прогон
    через torch/onnx дороже последующих (ленивая аллокация буферов, инициализация
    ядер BLAS/cudnn), поэтому «съедаем» его на старте, а не на первом фото
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
    batch_size: int = 64,
    time_budget_s: float | None = None,
) -> np.ndarray:
    """Список BGR-изображений -> матрица эмбеддингов (N, EMBED_DIM).

    time_budget_s: если задан, обработка прекращается по истечении бюджета и
    возвращаются эмбеддинги только для успевших изображений. Нужно, чтобы
    уложиться в требование по времени ответа на слабом CPU, где прогнать все
    найденные зёрна физически не успеть. На GPU бюджет обычно не срабатывает.
    """
    if not images_bgr:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    deadline = time.perf_counter() + time_budget_s if time_budget_s else None

    # --- быстрый CPU-путь через ONNX Runtime (если включён и доступен) ---
    session = _load_onnx()
    if session is not None:
        input_name = session.get_inputs()[0].name
        feats = []
        for start in range(0, len(images_bgr), batch_size):
            chunk = images_bgr[start : start + batch_size]
            batch = np.stack([_preprocess(img) for img in chunk]).astype(np.float32)
            out = session.run(None, {input_name: batch})[0]
            feats.append(np.asarray(out, dtype=np.float32))
            if deadline is not None and time.perf_counter() > deadline:
                break
        return np.concatenate(feats, axis=0)

    # --- обычный путь через PyTorch (GPU fp16 или CPU fp32) ---
    import torch

    model = _load_model()
    device = getattr(model, "_dn_device", "cpu")
    half = getattr(model, "_dn_half", False)

    feats = []
    with torch.inference_mode():
        for start in range(0, len(images_bgr), batch_size):
            chunk = images_bgr[start : start + batch_size]
            batch = np.stack([_preprocess(img) for img in chunk])
            tensor = torch.from_numpy(batch).to(device)
            if half:
                tensor = tensor.half()
            out = model(tensor)
            feats.append(out.float().cpu().numpy())
            if deadline is not None and time.perf_counter() > deadline:
                break

    return np.concatenate(feats, axis=0)

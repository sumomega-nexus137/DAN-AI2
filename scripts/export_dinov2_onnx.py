"""Одноразовый экспорт DINOv2 (тот же, что в config.DINOV2_MODEL) в ONNX (fp32).

Зачем: на CPU ONNX Runtime обычно в 1.5-3x быстрее PyTorch на этой модели.
После экспорта запускай backend с DNAI_ONNX=1 — embedder сам подхватит
models/<model>.onnx и пойдёт через onnxruntime (с откатом на torch, если
файла/пакета нет). На GPU это не нужно — там быстрее fp16-путь PyTorch.

Запуск (из корня репозитория, нужен torch):
    python scripts/export_dinov2_onnx.py
Результат: models/dinov2_vits14.onnx (имя — из config.DINOV2_MODEL).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.common import config  # noqa: E402


def main() -> int:
    import torch

    out_path = config.MODELS_DIR / f"{config.DINOV2_MODEL}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Гружу {config.DINOV2_MODEL} через torch.hub…")
    model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_MODEL).eval()

    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    print(f"Экспортирую в ONNX -> {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        # батч динамический: эмбеддим пачками разного размера
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # быстрая проверка, что файл читается onnxruntime и даёт ту же форму
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: np.zeros((2, 3, 224, 224), dtype=np.float32)})[0]
        print(f"OK: onnxruntime отдаёт эмбеддинги формы {out.shape} (ожидалось (2, 384)).")
        print("Теперь запускай backend с переменной окружения DNAI_ONNX=1.")
    except Exception as exc:  # noqa: BLE001
        print(f"Файл экспортирован, но проверка onnxruntime не прошла: {exc}")
        print("Установи onnxruntime (pip install onnxruntime) и повтори проверку.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

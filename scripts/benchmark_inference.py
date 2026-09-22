"""Замер времени отклика инференса на одном и том же фото.

Зачем: dev-окружение агента без torch/GPU не может измерить реальную скорость
DINOv2. Этот скрипт запускается там, где модель реально работает (Colab/VPS),
и даёт честные числа для коммитов и PR: медиану по N прогонам одного кадра
плюс отдельно стоимость холодного старта (первый запрос без прогрева).

Что меряет (для выбранного модуля):
  * cold  — первый вызов analyze() БЕЗ warmup (ленивая загрузка весов + первый
            forward + первый predict_proba): это то, что убивает шаг 2 warmup;
  * warm  — медиана/мин/макс по N прогонам ПОСЛЕ warmup: установившаяся
            латентность, которую видит пользователь на втором и далее фото.

Запуск (из корня репозитория):
    python scripts/benchmark_inference.py --image sample_grain.jpg --module grain
    python scripts/benchmark_inference.py --image sample_leaf.jpg  --module disease
    python scripts/benchmark_inference.py --image sample.jpg       --module all --runs 5

Замечания:
  * DEMO_MODE должен быть выключен (иначе меряется заглушка). Скрипт это
    проверяет и предупреждает.
  * Для честного cold-замера процесс должен быть свежим — поэтому cold меряется
    ДО warmup и до любого другого прогона в этом процессе.
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# запуск из корня репозитория без установки пакета
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.common import config, embedder  # noqa: E402
from backend.app.module1_grain import pipeline as grain_pipeline  # noqa: E402
from backend.app.module2_disease import pipeline as disease_pipeline  # noqa: E402

# main.py тут не импортируется (не нужен ASGI), поэтому воспроизводим его
# настройку потоков torch, чтобы замер соответствовал проду.
try:  # pragma: no cover
    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(1)
except Exception:  # noqa: BLE001
    pass


def _analyzer(module: str):
    if module == "grain":
        return grain_pipeline.analyze
    if module == "disease":
        return disease_pipeline.analyze
    if module == "auto":
        # /predict/auto роутит по картинке; для чистого замера инференса
        # достаточно грубо развести на grain/disease здесь не нужно — берём
        # обе ветки отдельными модулями. Для auto меряем grain как более
        # тяжёлую ветку (сегментация + N кропов).
        return grain_pipeline.analyze
    raise ValueError(f"неизвестный модуль: {module}")


def _bench_one(module: str, image_bytes: bytes, runs: int) -> None:
    analyze = _analyzer(module)

    # --- COLD: первый вызов без предварительного прогрева ---
    t0 = time.perf_counter()
    analyze(image_bytes)
    cold = time.perf_counter() - t0

    # --- WARMUP: как на старте FastAPI ---
    embedder.warmup()

    # --- WARM: N установившихся прогонов ---
    warm_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        analyze(image_bytes)
        warm_times.append(time.perf_counter() - t0)

    median = statistics.median(warm_times)
    print(f"\n=== модуль: {module} ===")
    print(f"  cold (1-й запрос, без warmup): {cold * 1000:8.0f} мс")
    print(
        f"  warm  x{runs}: медиана {median * 1000:7.0f} мс "
        f"(мин {min(warm_times) * 1000:.0f} / макс {max(warm_times) * 1000:.0f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="путь к фото для замера")
    ap.add_argument(
        "--module",
        default="all",
        choices=["grain", "disease", "auto", "all"],
        help="какой модуль мерить (all = grain и disease)",
    )
    ap.add_argument("--runs", type=int, default=5, help="число прогонов для медианы (по умолчанию 5)")
    args = ap.parse_args()

    if config.DEMO_MODE:
        print(
            "ВНИМАНИЕ: DEMO_MODE=1 — меряется заглушка, а не реальный инференс. "
            "Отключите DEMO_MODE и повторите.",
            file=sys.stderr,
        )

    image_bytes = Path(args.image).read_bytes()
    print(f"фото: {args.image} ({len(image_bytes)} байт), прогонов warm: {args.runs}")

    modules = ["grain", "disease"] if args.module == "all" else [args.module]
    if len(modules) > 1:
        print(
            "  (примечание: DINOv2 загружается один раз на процесс, поэтому честный "
            "cold-замер даёт только ПЕРВЫЙ модуль. Для точного cold по каждому "
            "запускайте модули отдельными процессами: --module grain / --module disease)"
        )
    for m in modules:
        _bench_one(m, image_bytes, args.runs)


if __name__ == "__main__":
    main()

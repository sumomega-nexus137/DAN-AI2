"""Тесты поправки вывода модели зерна на реальную партию (label shift).

Голова обучена на сбалансированном GrainSet (по 20% на класс) и на «живых»
фото склонна раскидывать сомнительные зёрна по «плохим» классам — отсюда был
вечный фураж. Поправка должна: (1) сомнительное зерно считать целым,
(2) уверенный брак оставлять браком — грязная партия не должна стать 1 классом.
"""

import numpy as np

from backend.app.module1_grain import grading
from backend.app.module1_grain import pipeline

CLASSES = ["bitoe_povrezhdennoe", "celoe_zdorovoe", "primes", "prorosshee", "shuploe_melkoe"]


def _row(**p) -> np.ndarray:
    v = np.array([p.get(c, 0.0) for c in CLASSES], dtype=np.float64)
    return v / v.sum()


def test_doubtful_defect_counts_as_healthy():
    # модель «склоняется» к сору на 70%, но это типичное сомнение на фото
    probs = np.array([_row(primes=0.70, celoe_zdorovoe=0.20, bitoe_povrezhdennoe=0.05,
                           shuploe_melkoe=0.03, prorosshee=0.02)])
    counts = pipeline._count_classes(probs, CLASSES)
    assert counts["celoe_zdorovoe"] == 1


def test_confident_defect_stays_defect():
    probs = np.array([_row(primes=0.94, celoe_zdorovoe=0.02, bitoe_povrezhdennoe=0.02,
                           shuploe_melkoe=0.01, prorosshee=0.01)])
    counts = pipeline._count_classes(probs, CLASSES)
    assert counts["primes"] == 1


def test_doubtful_clean_batch_is_not_fodder():
    """Картина со скриншотов: 70% зёрен модель сомнительно относит к сору."""
    doubtful = _row(primes=0.70, celoe_zdorovoe=0.20, bitoe_povrezhdennoe=0.05,
                    shuploe_melkoe=0.03, prorosshee=0.02)
    healthy = _row(celoe_zdorovoe=0.88, primes=0.03, bitoe_povrezhdennoe=0.03,
                   shuploe_melkoe=0.03, prorosshee=0.03)
    probs = np.array([doubtful] * 84 + [healthy] * 36)
    a = grading.assess(pipeline._count_classes(probs, CLASSES))
    assert a.grade is not None and a.grade <= 3


def test_really_dirty_batch_stays_low():
    """25% уверенного сора — не должно превратиться в хороший класс."""
    dirty = _row(primes=0.92, celoe_zdorovoe=0.02, bitoe_povrezhdennoe=0.02,
                 shuploe_melkoe=0.02, prorosshee=0.02)
    healthy = _row(celoe_zdorovoe=0.90, primes=0.04, bitoe_povrezhdennoe=0.02,
                   shuploe_melkoe=0.02, prorosshee=0.02)
    probs = np.array([dirty] * 30 + [healthy] * 90)
    a = grading.assess(pipeline._count_classes(probs, CLASSES))
    assert a.grade is None or a.grade == 5

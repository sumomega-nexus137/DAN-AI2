"""Оценка класса зерна, стоимости в тенге и рекомендаций фермеру.

ВАЖНО: это ОРИЕНТИРОВОЧНАЯ оценка по внешнему виду зерна на фото, а не
официальная лабораторная классификация. Настоящий ГОСТ учитывает ещё
клейковину, число падения, натуру и влажность — по фотографии их определить
невозможно. Пороги ниже — упрощённые, в духе ГОСТ по чистоте пробы, и
настраиваемые.

Как считается класс и цена (прозрачно, «математика внутри»):
1. Считаем доли трёх вещей: сорной примеси (не зерно), зерновой примеси
   (битое + щуплое + проросшее) и проростков отдельно.
2. Класс = лучший (наименьший номер), под лимиты которого проба проходит по
   ВСЕМ трём показателям. Не проходит даже под 5 класс -> фуражное.
3. Цена = не плоская «цена класса», а ПЛАВНАЯ величина внутри вилки класса:
   чем ближе проба к верхней границе лимитов (грязнее), тем ближе цена к
   нижней границе вилки; идеально чистая для своего класса -> верх вилки.
   Так цена реагирует на конкретный процентный состав, а не прыгает
   ступеньками.
"""

from dataclasses import dataclass, field

from ..common import config

# Наши 5 классов модели -> роль в оценке качества
CLASS_LABELS_RU = {
    "celoe_zdorovoe": "Целое здоровое",
    "bitoe_povrezhdennoe": "Битое / повреждённое",
    "shuploe_melkoe": "Щуплое / мелкое",
    "prorosshee": "Проросшее",
    "primes": "Сорная примесь",
}

# Зерновая примесь по смыслу ГОСТ — повреждённое зерно культуры
GRAIN_IMPURITY_CLASSES = ("bitoe_povrezhdennoe", "shuploe_melkoe", "prorosshee")
# Сорная примесь — всё, что зерном не является
FOREIGN_IMPURITY_CLASSES = ("primes",)

# Упрощённые пороги классности (в процентах от пробы). Лесенка 1..5:
# чем выше класс (меньше номер), тем строже по чистоте.
GRADE_LIMITS = {
    1: {"foreign_max": 0.5, "grain_impurity_max": 2.0, "sprouted_max": 0.3},
    2: {"foreign_max": 1.0, "grain_impurity_max": 4.0, "sprouted_max": 0.5},
    3: {"foreign_max": 2.0, "grain_impurity_max": 8.0, "sprouted_max": 1.5},
    4: {"foreign_max": 3.0, "grain_impurity_max": 12.0, "sprouted_max": 2.5},
    5: {"foreign_max": 5.0, "grain_impurity_max": 20.0, "sprouted_max": 5.0},
}
GRADES = (1, 2, 3, 4, 5)

GRADE_PRICES_KZT = {
    1: config.PRICE_CLASS_1_KZT,
    2: config.PRICE_CLASS_2_KZT,
    3: config.PRICE_CLASS_3_KZT,
    4: config.PRICE_CLASS_4_KZT,
    5: config.PRICE_CLASS_5_KZT,
}

# Вилка «от и до» по каждому классу — из неё берём плавную цену
GRADE_PRICE_RANGES_KZT = {
    1: (config.PRICE_CLASS_1_MIN_KZT, config.PRICE_CLASS_1_MAX_KZT),
    2: (config.PRICE_CLASS_2_MIN_KZT, config.PRICE_CLASS_2_MAX_KZT),
    3: (config.PRICE_CLASS_3_MIN_KZT, config.PRICE_CLASS_3_MAX_KZT),
    4: (config.PRICE_CLASS_4_MIN_KZT, config.PRICE_CLASS_4_MAX_KZT),
    5: (config.PRICE_CLASS_5_MIN_KZT, config.PRICE_CLASS_5_MAX_KZT),
}
FODDER_RANGE_KZT = (config.PRICE_FODDER_MIN_KZT, config.PRICE_FODDER_MAX_KZT)

# «Лучший» ориентир, против которого считаем потери — товарный 3 класс
# (реалистичная цель для продовольственной пшеницы).
BEST_REFERENCE_PRICE_KZT = config.PRICE_CLASS_3_KZT

# Эффективность механической очистки (решётная очистка / просеивание):
# сорную примесь убирает почти полностью, битое и щуплое — частично.
CLEANING_REMOVAL = {
    "primes": 0.90,
    "bitoe_povrezhdennoe": 0.50,
    "shuploe_melkoe": 0.60,
    "prorosshee": 0.0,  # проросшее очисткой не убрать
}

MIN_GRAINS_FOR_CONFIDENCE = 80


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _pct(value: float) -> str:
    """Процент в русском формате: запятая как разделитель дробной части."""
    return f"{value:.1f}".replace(".", ",") + "%"


def _kzt(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") + " ₸"


@dataclass
class Recommendation:
    title: str
    detail: str
    priority: str  # high | medium | low
    gain_kzt_per_ton: float | None = None


@dataclass
class GrainAssessment:
    total_grains: int
    percentages: dict[str, float]
    foreign_pct: float
    grain_impurity_pct: float
    sound_pct: float
    grade: int | None
    grade_label: str
    price_kzt_per_ton: float | None
    price_range_kzt_per_ton: tuple[float, float] | None
    potential_grade: int | None
    potential_gain_kzt_per_ton: float
    loss_vs_best_kzt_per_ton: float = 0.0
    recommendations: list[Recommendation] = field(default_factory=list)
    confidence_note: str | None = None


def _percentages(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in CLASS_LABELS_RU}
    return {k: 100.0 * counts.get(k, 0) / total for k in CLASS_LABELS_RU}


def _grade_for(foreign_pct: float, grain_impurity_pct: float, sprouted_pct: float) -> int | None:
    """Наименьший (то есть лучший) класс, под требования которого проба
    проходит по всем трём показателям. None — не проходит даже под 5 класс."""
    for grade in GRADES:
        limits = GRADE_LIMITS[grade]
        if (
            foreign_pct <= limits["foreign_max"]
            and grain_impurity_pct <= limits["grain_impurity_max"]
            and sprouted_pct <= limits["sprouted_max"]
        ):
            return grade
    return None


def _edge_ratio(grade: int, foreign_pct: float, grain_impurity_pct: float, sprouted_pct: float) -> float:
    """Насколько проба близка к худшему краю лимитов своего класса: 0.0 —
    идеально чисто, 1.0 — у самого порога (хуже). По самому «узкому» из трёх
    показателей — он и определяет реальное качество внутри класса."""
    limits = GRADE_LIMITS[grade]
    ratios = [
        foreign_pct / limits["foreign_max"] if limits["foreign_max"] else 0.0,
        grain_impurity_pct / limits["grain_impurity_max"] if limits["grain_impurity_max"] else 0.0,
        sprouted_pct / limits["sprouted_max"] if limits["sprouted_max"] else 0.0,
    ]
    return _clamp(max(ratios), 0.0, 1.0)


def _estimate_price(
    grade: int | None, foreign_pct: float, grain_impurity_pct: float, sprouted_pct: float
) -> float:
    """Плавная цена по составу пробы.

    Внутри класса цена линейно идёт от верхней границы вилки (идеально чисто)
    к нижней (у порога класса). Ниже 5 класса — по фуражной вилке, тем ниже,
    чем сильнее перебор над лимитами 5 класса."""
    if grade is None:
        lo, hi = FODDER_RANGE_KZT
        limits5 = GRADE_LIMITS[5]
        over = (
            max(0.0, foreign_pct - limits5["foreign_max"]) / 15.0
            + max(0.0, grain_impurity_pct - limits5["grain_impurity_max"]) / 22.0
            + max(0.0, sprouted_pct - limits5["sprouted_max"]) / 15.0
        )
        t = _clamp(1.0 - over, 0.0, 1.0)
        return lo + t * (hi - lo)

    lo, hi = GRADE_PRICE_RANGES_KZT[grade]
    r = _edge_ratio(grade, foreign_pct, grain_impurity_pct, sprouted_pct)
    return hi - r * (hi - lo)


def _simulate_cleaning(percentages: dict[str, float]) -> dict[str, float]:
    """Как изменится состав пробы после механической очистки."""
    remaining = {}
    for cls, pct in percentages.items():
        removed_share = CLEANING_REMOVAL.get(cls, 0.0)
        remaining[cls] = pct * (1.0 - removed_share)

    total = sum(remaining.values())
    if total == 0:
        return remaining
    return {cls: 100.0 * val / total for cls, val in remaining.items()}


def _grade_label(grade: int | None) -> str:
    if grade is None:
        return "Фуражное (ниже 5 класса)"
    return f"{grade} класс"


def _build_recommendations(
    pct: dict[str, float],
    grade: int | None,
    potential_grade: int | None,
    gain: float,
) -> list[Recommendation]:
    recs: list[Recommendation] = []
    foreign = pct["primes"]
    broken = pct["bitoe_povrezhdennoe"]
    thin = pct["shuploe_melkoe"]
    sprouted = pct["prorosshee"]

    # Очистка поднимает класс? (gain уже посчитан согласованно с этим условием)
    can_upgrade = (
        potential_grade is not None
        and gain > 0
        and (grade is None or potential_grade < grade)
    )
    if can_upgrade:
        recs.append(
            Recommendation(
                title=f"Очистить партию — поднимете до {potential_grade} класса",
                detail=f"Прибавка примерно {_kzt(gain)} с каждой тонны.",
                priority="high",
                gain_kzt_per_ton=gain,
            )
        )

    if foreign > 2.0:
        recs.append(
            Recommendation(
                title="Просеять: много сора",
                detail=f"Сорной примеси {_pct(foreign)} при норме 2%. Решётная очистка уберёт основное.",
                priority="high",
            )
        )

    if broken + thin > 5.0:
        recs.append(
            Recommendation(
                title="Дочистить на сепараторе",
                detail=f"Битого и щуплого {_pct(broken + thin)}. Калибровка по размеру отсеет мелочь.",
                priority="high" if broken + thin > 12.0 else "medium",
            )
        )

    if sprouted > 1.0:
        recs.append(
            Recommendation(
                title="Проверить склад — есть проростки",
                detail=(
                    f"Проросшего {_pct(sprouted)}. Очисткой не убрать: проверьте влажность "
                    "и вентиляцию, продавайте быстрее."
                ),
                priority="high" if sprouted > 3.0 else "medium",
            )
        )

    if grade == 5 or grade is None:
        recs.append(
            Recommendation(
                title="Не выводится выше — продавайте на корм",
                detail="Фуражные цели или сдача на подработку элеватору будут выгоднее.",
                priority="medium",
            )
        )

    # Хорошая партия (товарный класс, чисто, и очистка класс не поднимет) —
    # подтверждаем, чтобы фермер не чистил зря. НЕ показываем вместе с советом
    # «очистить и поднять класс», иначе совет противоречивый.
    if (
        not can_upgrade
        and grade is not None
        and grade <= 3
        and foreign <= 2.0
        and broken + thin <= 5.0
        and sprouted <= 1.0
    ):
        recs.append(
            Recommendation(
                title="Партия в хорошем состоянии",
                detail="Держите влажность в норме при хранении, чтобы не потерять класс до продажи.",
                priority="low",
            )
        )

    return recs


def assess(counts: dict[str, int]) -> GrainAssessment:
    """Полная оценка пробы по подсчёту зёрен в каждой категории."""
    total = sum(counts.values())
    pct = _percentages(counts)

    if total == 0:
        return GrainAssessment(
            total_grains=0,
            percentages=pct,
            foreign_pct=0.0,
            grain_impurity_pct=0.0,
            sound_pct=0.0,
            grade=None,
            grade_label="Не определено",
            price_kzt_per_ton=None,
            price_range_kzt_per_ton=None,
            potential_grade=None,
            potential_gain_kzt_per_ton=0.0,
            recommendations=[],
            confidence_note=(
                "На фото не удалось выделить отдельные зёрна. Разложите пробу тонким "
                "слоем на контрастном фоне и снимите сверху при ровном освещении."
            ),
        )

    foreign_pct = sum(pct[c] for c in FOREIGN_IMPURITY_CLASSES)
    grain_impurity_pct = sum(pct[c] for c in GRAIN_IMPURITY_CLASSES)
    sound_pct = pct["celoe_zdorovoe"]
    sprouted_pct = pct["prorosshee"]

    grade = _grade_for(foreign_pct, grain_impurity_pct, sprouted_pct)
    price = _estimate_price(grade, foreign_pct, grain_impurity_pct, sprouted_pct)

    # Цена показывается всегда: если партия не проходит даже 5 класс — это
    # фуражное зерно с фуражной ценой, а не «нет цены».
    grade_label = _grade_label(grade)
    price_range = GRADE_PRICE_RANGES_KZT[grade] if grade is not None else FODDER_RANGE_KZT

    cleaned = _simulate_cleaning(pct)
    potential_grade = _grade_for(
        sum(cleaned[c] for c in FOREIGN_IMPURITY_CLASSES),
        sum(cleaned[c] for c in GRAIN_IMPURITY_CLASSES),
        cleaned["prorosshee"],
    )
    potential_price = (
        _estimate_price(
            potential_grade,
            sum(cleaned[c] for c in FOREIGN_IMPURITY_CLASSES),
            sum(cleaned[c] for c in GRAIN_IMPURITY_CLASSES),
            cleaned["prorosshee"],
        )
        if potential_grade is not None
        else None
    )

    # «Прибавку от очистки» показываем ТОЛЬКО когда очистка реально поднимает
    # КЛАСС (иначе цена растёт лишь внутри той же вилки — это сбивает с толку:
    # число есть, а совета «очистить» нет). Так gain > 0 всегда совпадает с
    # рекомендацией поднять класс.
    class_improves = potential_grade is not None and (grade is None or potential_grade < grade)
    gain = 0.0
    if class_improves and potential_price is not None and potential_price > price:
        gain = potential_price - price

    # Сколько партия теряет против товарного 3 класса — ориентир по деньгам
    loss_vs_best = max(0.0, BEST_REFERENCE_PRICE_KZT - price)

    recommendations = _build_recommendations(pct, grade, potential_grade, gain)

    confidence_note = None
    if total < MIN_GRAINS_FOR_CONFIDENCE:
        confidence_note = (
            f"Распознано всего {total} зёрен — для устойчивой оценки желательно "
            f"не меньше {MIN_GRAINS_FOR_CONFIDENCE}. Снимите пробу крупнее или разложите шире."
        )

    return GrainAssessment(
        total_grains=int(round(total)),
        percentages=pct,
        foreign_pct=foreign_pct,
        grain_impurity_pct=grain_impurity_pct,
        sound_pct=sound_pct,
        grade=grade,
        grade_label=grade_label,
        price_kzt_per_ton=int(round(price, -2)),
        price_range_kzt_per_ton=price_range,
        potential_grade=potential_grade,
        potential_gain_kzt_per_ton=gain,
        loss_vs_best_kzt_per_ton=loss_vs_best,
        recommendations=recommendations,
        confidence_note=confidence_note,
    )

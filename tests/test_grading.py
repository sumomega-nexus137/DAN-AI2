"""Тесты логики классности и экономики — самой ответственной части проекта.

Здесь считается то, что фермер увидит как «3 класс, 113 000 ₸ за тонну,
после очистки +15 000». Ошибка тут стоит дороже, чем ошибка модели на
одном зерне, поэтому граничные случаи закрыты тестами.

Запуск из корня репозитория:  pytest -q
"""

from backend.app.common import config
from backend.app.module1_grain import grading


def counts(celoe=0, bitoe=0, shuploe=0, prorosshee=0, primes=0) -> dict[str, int]:
    return {
        "celoe_zdorovoe": celoe,
        "bitoe_povrezhdennoe": bitoe,
        "shuploe_melkoe": shuploe,
        "prorosshee": prorosshee,
        "primes": primes,
    }


def titles(assessment) -> list[str]:
    return [r.title for r in assessment.recommendations]


def in_range(price, rng) -> bool:
    return rng[0] <= price <= rng[1]


# --- пустая проба: главный опасный случай ---------------------------------


def test_empty_sample_has_no_grade_and_no_price():
    """Если зёрен не нашли — нельзя выдавать класс и цену с потолка."""
    a = grading.assess(counts())

    assert a.total_grains == 0
    assert a.grade is None
    assert a.grade_label == "Не определено"
    assert a.price_kzt_per_ton is None
    assert a.price_range_kzt_per_ton is None
    assert a.recommendations == []
    assert "не удалось выделить" in a.confidence_note


# --- чистые партии -> высокие классы (не «всегда ниже 5-го») ----------------


def test_perfect_sample_is_first_grade():
    a = grading.assess(counts(celoe=1000))
    assert a.grade == 1
    assert in_range(a.price_kzt_per_ton, grading.GRADE_PRICE_RANGES_KZT[1])


def test_clean_sample_reaches_top_grades():
    """98,5% целых -> 1 класс (не «вечный фураж»)."""
    a = grading.assess(counts(celoe=985, bitoe=10, shuploe=5))
    assert a.grade == 1
    assert in_range(a.price_kzt_per_ton, grading.GRADE_PRICE_RANGES_KZT[1])


def test_second_grade_reachable():
    a = grading.assess(counts(celoe=965, bitoe=35))  # 3,5% зерновой примеси
    assert a.grade == 2
    assert in_range(a.price_kzt_per_ton, grading.GRADE_PRICE_RANGES_KZT[2])


def test_foreign_impurity_within_norm_gives_no_sieving_advice():
    a = grading.assess(counts(celoe=990, primes=10))  # 1% сора
    assert a.grade == 2
    assert "Просеять: много сора" not in titles(a)


# --- цена реагирует на состав (плавно, «математика внутри») -----------------


def test_price_is_within_class_range():
    a = grading.assess(counts(celoe=880, bitoe=80, shuploe=40))  # ~3 класс
    assert a.grade is not None
    assert in_range(a.price_kzt_per_ton, grading.GRADE_PRICE_RANGES_KZT[a.grade])


def test_dirtier_sample_of_same_class_is_cheaper():
    """Внутри одного класса грязнее проба -> ниже цена (не ступенька)."""
    cleaner = grading.assess(counts(celoe=950, bitoe=30, shuploe=20))  # 5% зерн.прим
    dirtier = grading.assess(counts(celoe=930, bitoe=40, shuploe=30))  # 7% зерн.прим
    assert cleaner.grade == dirtier.grade  # оба один класс
    assert cleaner.price_kzt_per_ton > dirtier.price_kzt_per_ton


def test_prices_are_monotonic_by_quality():
    p1 = grading.assess(counts(celoe=1000)).price_kzt_per_ton
    p3 = grading.assess(counts(celoe=880, bitoe=80, shuploe=40)).price_kzt_per_ton
    p5 = grading.assess(counts(celoe=800, bitoe=100, shuploe=50, primes=50)).price_kzt_per_ton
    fodder = grading.assess(counts(celoe=900, primes=100)).price_kzt_per_ton
    assert p1 > p3 > p5 > fodder


# --- грязная партия: сор + битое -------------------------------------------


def test_dirty_sample_gets_cleaning_advice_and_upgrade():
    # 5% сора и 15% зерновой примеси -> проходит только под 5 класс
    a = grading.assess(counts(celoe=800, bitoe=100, shuploe=50, primes=50))

    assert a.grade == 5
    assert a.foreign_pct == 5.0
    assert a.grain_impurity_pct == 15.0
    assert in_range(a.price_kzt_per_ton, grading.GRADE_PRICE_RANGES_KZT[5])

    # очистка убирает сор и часть битого -> класс выше
    assert a.potential_grade is not None and a.potential_grade < a.grade
    assert a.potential_gain_kzt_per_ton > 0

    assert "Просеять: много сора" in titles(a)
    assert "Дочистить на сепараторе" in titles(a)
    assert any(t.startswith("Очистить партию — поднимете до") for t in titles(a))
    assert "Не выводится выше — продавайте на корм" in titles(a)


# --- проросшее: очисткой не лечится ----------------------------------------


def test_sprouted_grain_warns_about_storage_not_cleaning():
    a = grading.assess(counts(celoe=950, prorosshee=50))  # 5% проросшего
    assert "Проверить склад — есть проростки" in titles(a)
    # проросшее очистка не убирает, класс подняться не должен
    assert a.potential_grade == a.grade


def test_sprouted_above_three_percent_is_high_priority():
    a = grading.assess(counts(celoe=950, prorosshee=50))
    rec = next(r for r in a.recommendations if r.title == "Проверить склад — есть проростки")
    assert rec.priority == "high"


# --- партия ниже 5 класса ---------------------------------------------------


def test_below_fifth_grade_shows_fodder_price_and_gain():
    """Ниже 5 класса — показываем фуражную цену (не «нет цены») и прибавку."""
    a = grading.assess(counts(celoe=900, primes=100))  # 10% сора

    assert a.grade is None
    assert a.grade_label == "Фуражное (ниже 5 класса)"
    assert a.price_kzt_per_ton is not None
    assert in_range(a.price_kzt_per_ton, grading.FODDER_RANGE_KZT)
    assert a.price_range_kzt_per_ton == grading.FODDER_RANGE_KZT
    # сор убирается очисткой -> можно поднять до товарного класса
    assert a.potential_grade is not None
    assert a.potential_gain_kzt_per_ton > 0
    assert any(t.startswith("Очистить партию — поднимете до") for t in titles(a))


# --- доверие к оценке -------------------------------------------------------


def test_small_sample_gets_confidence_warning():
    a = grading.assess(counts(celoe=50))
    assert a.grade == 1
    assert a.confidence_note is not None
    assert str(grading.MIN_GRAINS_FOR_CONFIDENCE) in a.confidence_note


def test_large_sample_has_no_confidence_warning():
    a = grading.assess(counts(celoe=500))
    assert a.confidence_note is None


# --- проценты считаются от общего числа ------------------------------------


def test_percentages_sum_to_hundred():
    a = grading.assess(counts(celoe=317, bitoe=41, shuploe=23, prorosshee=7, primes=12))
    assert round(sum(a.percentages.values()), 6) == 100.0
    assert a.sound_pct + a.grain_impurity_pct + a.foreign_pct == a.percentages[
        "celoe_zdorovoe"
    ] + sum(
        a.percentages[c]
        for c in grading.GRAIN_IMPURITY_CLASSES + grading.FOREIGN_IMPURITY_CLASSES
    )


# --- формат чисел для фермера ----------------------------------------------


def test_numbers_are_formatted_in_russian():
    """Запятая в дробях и пробел в тысячах — иначе текст читается как чужой."""
    assert grading._pct(3.9) == "3,9%"
    assert grading._pct(15.0) == "15,0%"
    assert grading._kzt(15000) == "15 000 ₸"
    assert grading._kzt(102000) == "102 000 ₸"


# --- цены монотонны по лесенке классов -------------------------------------


def test_class_price_ladder_is_monotonic():
    mids = [grading.GRADE_PRICES_KZT[g] for g in grading.GRADES]
    assert mids == sorted(mids, reverse=True)  # 1 класс дороже 2, и т.д.
    assert mids[-1] > config.PRICE_FODDER_KZT  # 5 класс дороже фуража

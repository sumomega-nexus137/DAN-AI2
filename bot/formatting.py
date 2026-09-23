"""Форматирование ответов backend в сообщения Telegram (HTML-разметка)."""

URGENCY_RU = {
    "high": "🔴 Срочно",
    "medium": "🟡 В ближайшее время",
    "low": "⚪️ Под наблюдением",
    "none": "🟢 Норма",
}

PRIORITY_MARK = {"high": "🔴", "medium": "🟡", "low": "⚪️"}

KIND_RU = {
    "disease": "Болезнь",
    "pest": "Вредитель",
    "weed": "Сорняк",
    "healthy": "Здоровое растение",
}


def _kzt(value) -> str:
    if value is None:
        return "—"
    return f"{round(value):,}".replace(",", " ") + " ₸"


def _pct(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + "%"


def format_grain(data: dict) -> str:
    lines = ["<b>📊 Качество зерна</b>", ""]

    lines.append(f"<b>Класс:</b> {data['grade_label']} <i>(ориентировочно, по фото)</i>")
    if data.get("price_kzt_per_ton"):
        lines.append(f"<b>Цена:</b> ~{_kzt(data['price_kzt_per_ton'])} за тонну")
        rng = data.get("price_range_kzt_per_ton")
        if rng:
            lines.append(f"<i>Рынок: {_kzt(rng[0])} — {_kzt(rng[1])}</i>")

    loss = data.get("loss_vs_best_kzt_per_ton") or 0
    if loss > 0:
        lines.append(f"<b>Теряете против 3 класса:</b> −{_kzt(loss)} с тонны")

    gain = data.get("potential_gain_kzt_per_ton") or 0
    if gain > 0:
        lines.append(
            f"<b>После очистки:</b> {data['potential_grade']} класс, "
            f"+{_kzt(gain)} за тонну"
        )

    lines.append("")
    lines.append("<b>Состав пробы:</b>")
    for cat in data["categories"]:
        if cat["percent"] > 0:
            lines.append(f"• {cat['label']}: {_pct(cat['percent'])}")
    if data.get("composition_note"):
        lines.append(f"<i>{data['composition_note']}</i>")

    analyzed = data.get("grains_analyzed") or data.get("total_grains") or 0
    detected = data.get("grains_detected_total")
    if detected and analyzed and detected > analyzed:
        lines.append(f"<i>Разобрано {analyzed} зёрен из {detected} найденных</i>")
    elif analyzed:
        lines.append(f"<i>Разобрано {analyzed} зёрен</i>")

    recs = data.get("recommendations") or []
    if recs:
        lines.append("")
        lines.append("<b>Что делать:</b>")
        for rec in recs:
            mark = PRIORITY_MARK.get(rec["priority"], "•")
            gain_txt = ""
            if rec.get("gain_kzt_per_ton"):
                gain_txt = f" <b>(+{_kzt(rec['gain_kzt_per_ton'])}/т)</b>"
            lines.append(f"{mark} <b>{rec['title']}</b>{gain_txt}")
            lines.append(f"   {rec['detail']}")

    note = data.get("confidence_note") or data.get("sampling_note")
    if note:
        lines.append("")
        lines.append(f"ℹ️ <i>{note}</i>")

    lines.append("")
    lines.append(f"<i>{data['disclaimer']}</i>")
    return "\n".join(lines)


def format_disease(data: dict) -> str:
    d = data["diagnosis"]
    lines = [
        f"<b>🌱 {d['name_ru']}</b>",
        f"{KIND_RU.get(d['kind'], 'Диагноз')} · уверенность {round(d['confidence'] * 100)}%",
        f"{URGENCY_RU.get(d['urgency'], '')}",
        "",
        f"<b>Что это:</b> {d['what_is_it']}",
        "",
        f"<b>Что делать:</b> {d['action']}",
    ]

    alts = data.get("alternatives") or []
    if alts:
        lines.append("")
        lines.append("<b>Другие возможные варианты:</b>")
        for alt in alts:
            lines.append(f"• {alt['name_ru']} — {round(alt['confidence'] * 100)}%")

    if data.get("low_confidence") and data.get("confidence_note"):
        lines.append("")
        lines.append(f"⚠️ <i>{data['confidence_note']}</i>")

    lines.append("")
    lines.append(f"<i>{data['disclaimer']}</i>")
    return "\n".join(lines)

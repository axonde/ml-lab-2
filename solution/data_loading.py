import json
from pathlib import Path

import pandas as pd

# Примерные границы Амурской области: точки за ними считаем ошибочными.
AMUR_LAT = (48.8, 57.2)
AMUR_LON = (119.5, 135.2)

# В поле region у городов областного значения (Белогорск, Свободный, Тында, Зея,
# Шимановск) стоит название области, а parent_region пустой.
DISTRICT_NAMES = {
    "Амурская область": "Города обл. значения",
    "7 - Завитинский район": "Завитинский район",
    "пгт. Углегорск": "Углегорск",
}
OTHER_CITIES = ("Амурская область", "Райчихинск", "пгт. Углегорск")
LIGHT_GROUPS = {
    "Светлое время суток": "День",
    "Сумерки": "Сумерки",
    "В темное время суток, освещение включено": "Темно, освещение есть",
    "В темное время суток, освещение не включено": "Темно, освещения нет",
    "В темное время суток, освещение отсутствует": "Темно, освещения нет",
}
SEASONS = {
    12: "Зима", 1: "Зима", 2: "Зима", 3: "Весна", 4: "Весна", 5: "Весна",
    6: "Лето", 7: "Лето", 8: "Лето", 9: "Осень", 10: "Осень", 11: "Осень",
}
BAD_WEATHER = {"Дождь", "Снегопад", "Метель", "Туман", "Ураганный ветер"}


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        records = payload.get("features")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Ожидается GeoJSON FeatureCollection или список записей.")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("Выгрузка должна содержать список объектов JSON.")
    return records


def properties(record: dict) -> dict:
    if record.get("type") == "Feature":
        value = record.get("properties")
        return value if isinstance(value, dict) else {}
    return record


def nested_flags(props: dict) -> dict:
    """Признаки из вложенных списков, по одному значению на ДТП."""
    vehicles = props.get("vehicles") or []
    kinds = " ".join(str(v.get("category") or "") for v in vehicles).lower()
    nearby = [str(x).lower() for x in props.get("nearby") or []]
    return {
        "vehicles_count": len(vehicles),
        "has_moto": "мотоцикл" in kinds or "мопед" in kinds,
        "has_truck": "грузов" in kinds or "тягач" in kinds or "самосвал" in kinds,
        "stretch": any(x.startswith("перегон") for x in nearby),
        "intersection": any("перекр" in x for x in nearby),
        "crossing": any("пешеходный переход" in x for x in nearby),
        "bad_weather": bool(BAD_WEATHER & set(props.get("weather") or [])),
    }


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        props = properties(record)
        row = {
            key: props.get(key)
            for key in (
                "id", "datetime", "region", "category", "severity", "light",
                "participants_count", "injured_count", "dead_count", "address",
            )
        }
        row.update(nested_flags(props))
        row["source_row"] = index
        geometry = record.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        coords = geometry.get("coordinates")
        if geometry.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
            row["longitude"], row["latitude"] = coords[:2]
        else:
            point = props.get("point")
            point = point if isinstance(point, dict) else {}
            row["longitude"], row["latitude"] = point.get("long"), point.get("lat")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["datetime_raw"] = frame["datetime"]
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", format="mixed")
    for name in ("longitude", "latitude"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["map_valid"] = (
        frame["longitude"].between(-180, 180)
        & frame["latitude"].between(-90, 90)
        & ~((frame["longitude"] == 0) & (frame["latitude"] == 0))
    )
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["district"] = frame["region"].replace(DISTRICT_NAMES)
    frame["area_type"] = "Районы"
    frame.loc[frame["region"].isin(OTHER_CITIES), "area_type"] = "Другие города"
    frame.loc[frame["region"] == "Благовещенск", "area_type"] = "Благовещенск"
    frame["light_group"] = frame["light"].map(LIGHT_GROUPS)
    frame["hour"] = frame["datetime"].dt.hour
    frame["weekday"] = frame["datetime"].dt.dayofweek
    frame["month"] = frame["datetime"].dt.month
    frame["year"] = frame["datetime"].dt.year
    frame["season"] = frame["month"].map(SEASONS)
    frame["fatal"] = frame["dead_count"] > 0
    frame["victims"] = frame["injured_count"] + frame["dead_count"]
    frame["in_region"] = (
        frame["map_valid"]
        & frame["latitude"].between(*AMUR_LAT)
        & frame["longitude"].between(*AMUR_LON)
    )
    return frame

"""Small, source-backed online tools for factual gaps and current weather.

Only activated when the user enables online lookup. Queries are sent to Wikipedia or
Open-Meteo, never to a third-party LLM. Personal/biographical queries are not sent to Wikipedia.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

USER_AGENT = "EmotionalAI/0.4 (local educational assistant; MediaWiki API etiquette)"


def _get_json(url: str, timeout: float = 4.0) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wiki_query(text: str) -> str | None:
    query = text.strip().strip("?!. ")
    # Avoid sending personal or private questions to a public search service.
    if re.search(r"\b(?:я|мне|меня|мой|моя|мои|мое|у меня|обо мне|про меня)\b", query.lower()):
        return None
    if re.search(r"\b(?:погода|курс валют|стоимость|цена|сегодня|сейчас|последние новости|последний|актуальный)\b", query.lower()):
        return None
    prefixes = (
        r"^расскажи(?: мне)? (?:про|о|об)\s+", r"^объясни(?: мне)?\s+", r"^поясни\s+",
        r"^опиши\s+", r"^что такое\s+", r"^что значит\s+", r"^кто такой\s+",
        r"^кто такая\s+", r"^кто такое\s+", r"^почему\s+", r"^зачем\s+",
        r"^как работает\s+", r"^как устроен\s+", r"^как устроена\s+", r"^как устроено\s+",
        r"^где находится\s+", r"^когда появился\s+", r"^когда появилась\s+",
    )
    for pattern in prefixes:
        query = re.sub(pattern, "", query, flags=re.I)
    query = re.sub(r"\b(?:подробнее|кратко|простыми словами|пожалуйста)\b", " ", query, flags=re.I)
    query = re.sub(r"\s+", " ", query).strip(" .,!?—-\t")
    if len(query) < 2 or len(query) > 180:
        return None
    return query


def wikipedia_lookup(text: str, timeout: float = 4.0) -> str | None:
    """Search Wikipedia on demand and return a short credited extract, not a fake synthesis."""
    query = _wiki_query(text)
    if not query:
        return None
    # Latin technical names are often better indexed in English Wikipedia.
    cyrillic_count = len(re.findall(r"[а-яё]", query, re.I))
    language = "ru" if cyrillic_count >= 2 else "en"
    base = f"https://{language}.wikipedia.org/w/api.php?"
    search_params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "srnamespace": 0, "srlimit": 1, "format": "json", "formatversion": 2,
    })
    try:
        search = _get_json(base + search_params, timeout)
        rows = search.get("query", {}).get("search", [])
        if not rows:
            return None
        row = rows[0]
        title = str(row.get("title", "")).strip()
        pageid = row.get("pageid")
        if not title or pageid is None:
            return None
        extract_params = urllib.parse.urlencode({
            "action": "query", "prop": "extracts|info", "pageids": pageid,
            "exintro": 1, "explaintext": 1, "exchars": 1200, "inprop": "url",
            "redirects": 1, "format": "json", "formatversion": 2,
        })
        result = _get_json(base + extract_params, timeout)
        pages = result.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        extract = re.sub(r"\s+", " ", str(page.get("extract", "")).strip())
        url = str(page.get("fullurl", "")).strip()
        if len(extract) < 80 or not url:
            return None
        return f"По статье Wikipedia «{title}»:\n\n{extract}\n\nИсточник: {url}"
    except Exception:
        # Network trouble should not prevent the rest of the chat from working.
        return None


_CITY_CASES = {
    "амстердаме": "Амстердам", "москве": "Москва", "петербурге": "Санкт-Петербург",
    "санкт-петербурге": "Санкт-Петербург", "берлине": "Берлин", "лондоне": "Лондон",
    "париже": "Париж", "роттердаме": "Роттердам", "гааге": "Гаага", "утрехте": "Утрехт",
    "дронтене": "Dronten", "ню-йорке": "New York", "нью-йорке": "New York",
}
_WEATHER_CODES = {
    0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "туман с изморозью", 51: "слабая морось", 53: "морось", 55: "сильная морось",
    56: "слабая ледяная морось", 57: "сильная ледяная морось", 61: "небольшой дождь", 63: "дождь",
    65: "сильный дождь", 66: "слабый ледяной дождь", 67: "сильный ледяной дождь", 71: "небольшой снег",
    73: "снег", 75: "сильный снег", 77: "снежная крупа", 80: "небольшие дождевые ливни",
    81: "дождевые ливни", 82: "сильные дождевые ливни", 85: "снеговые ливни", 86: "сильные снеговые ливни",
    95: "гроза", 96: "гроза с небольшим градом", 99: "гроза с крупным градом",
}


def weather_answer(text: str, timeout: float = 4.0) -> str:
    """Fetch current conditions via Open-Meteo. Requires an explicit city; never guesses location."""
    city_match = re.search(r"\b(?:погода|погодой|прогноз)\b.*?\b(?:в|для|около)\s+([^?!.;,]+)", text, re.I)
    if not city_match:
        return "Для актуальной погоды напиши город, например: «Какая погода в Амстердаме?»"
    entered_city = city_match.group(1).strip(" \t\n,.")
    city = _CITY_CASES.get(entered_city.lower(), entered_city)
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
            "name": city, "count": 1, "language": "ru", "format": "json",
        })
        geo = _get_json(geo_url, timeout)
        results = geo.get("results", [])
        if not results:
            return f"Не нашла город «{entered_city}» в погодном сервисе. Проверь написание или добавь страну."
        place = results[0]
        weather_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto", "temperature_unit": "celsius", "wind_speed_unit": "kmh",
        })
        response = _get_json(weather_url, timeout)
        current = response.get("current", {})
        if not current or "temperature_2m" not in current:
            raise ValueError("погодный сервис не вернул текущие условия")
        code = int(current.get("weather_code", -1))
        description = _WEATHER_CODES.get(code, "условия не уточнены")
        name = str(place.get("name", entered_city))
        country = str(place.get("country", "")).strip()
        location = f"{name}, {country}" if country else name
        time = str(current.get("time", "")).replace("T", " ")
        return (
            f"Сейчас в {location}: {description}, {current['temperature_2m']} °C; "
            f"ощущается как {current.get('apparent_temperature', current['temperature_2m'])} °C. "
            f"Ветер {current.get('wind_speed_10m', '—')} км/ч, влажность {current.get('relative_humidity_2m', '—')}%. "
            f"Данные на {time}. Источник: Open-Meteo."
        )
    except Exception:
        return "Не удалось получить актуальную погоду. Проверь подключение к интернету и попробуй ещё раз."

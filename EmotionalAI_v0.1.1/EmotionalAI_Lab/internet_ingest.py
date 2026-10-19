"""Respectful loader for Russian/English Wikipedia articles into Emotional's local search corpus."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from knowledge_store import WebKnowledgeStore

ProgressCallback = Callable[[str], None]

DEFAULT_TOPICS_RU = [
    "искусственный интеллект", "нейронная сеть", "машинное обучение", "языковая модель", "информатика",
    "программирование", "Python", "C++", "Java", "Lua", "операционная система", "процессор",
    "оперативная память", "SSD", "жёсткий диск", "видеокарта", "компьютерная архитектура", "компьютерная сеть",
    "интернет", "криптография", "базы данных", "алгоритм", "структуры данных", "робототехника", "электроника",
    "физика", "математика", "алгебра", "геометрия", "статистика", "биология", "химия", "астрономия",
    "космология", "геология", "экология", "медицина", "анатомия", "эволюция", "история", "археология",
    "древний мир", "средневековье", "история России", "история Европы", "география", "страны мира", "экономика",
    "финансы", "психология", "социология", "философия", "логика", "этика", "образование", "языкознание",
    "русский язык", "английский язык", "литература", "писатель", "поэзия", "музыка", "акустика", "звук",
    "изобразительное искусство", "анимация", "кинематограф", "видеоигра", "разработка видеоигр", "Roblox",
    "Minecraft", "Sonic the Hedgehog", "история видеоигр", "графический процессор", "рендеринг", "3D-графика",
    "электричество", "магнетизм", "термодинамика", "механика", "квантовая механика", "теория относительности",
    "климат", "океан", "растение", "животное", "кошка", "собака", "человеческий мозг", "память человека",
    "сон", "физическая активность", "питание", "право", "международные отношения", "государство", "демократия",
    "архитектура", "строительство", "транспорт", "железная дорога", "автомобиль", "авиация", "космонавтика",
    "энергетика", "солнечная энергия", "ветроэнергетика", "сельское хозяйство", "производство", "материаловедение",
    "информационная безопасность", "Linux", "Windows", "Git", "компилятор", "веб-разработка", "API",
    "информационный поиск", "обработка естественного языка", "компьютерное зрение", "обучение с подкреплением",
    "история науки", "научный метод", "критическое мышление", "теория вероятностей", "линейная алгебра",
]


def _request_json(language: str, params: dict, timeout: int = 25) -> dict:
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    url = f"https://{language}.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "EmotionalAI/0.1.1 (educational local assistant; https://www.mediawiki.org/wiki/API:Etiquette)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
            if "error" in data:
                error = data["error"]
                code = str(error.get("code", ""))
                if code in {"ratelimited", "maxlag"}:
                    time.sleep(min(30, 2 ** (attempt + 1)))
                    continue
                raise RuntimeError(f"Wikipedia API: {error.get('info', code)}")
            return data
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (429, 500, 502, 503, 504):
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                time.sleep(min(60, float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)))
                continue
            raise RuntimeError(f"Wikipedia API вернул HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(f"Не удалось получить данные с Wikipedia после повторных попыток: {last_error}")


def load_topics(path: str | Path, language: str = "ru") -> list[str]:
    file_path = Path(path)
    if file_path.exists():
        lines = [x.strip() for x in file_path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]
        if lines:
            return list(dict.fromkeys(lines))
    return DEFAULT_TOPICS_RU if language == "ru" else [
        "artificial intelligence", "machine learning", "neural network", "programming", "computer science",
        "computer hardware", "operating system", "physics", "mathematics", "biology", "chemistry", "astronomy",
        "history", "geography", "economics", "psychology", "philosophy", "linguistics", "music", "video game",
        "robotics", "electronics", "internet", "cybersecurity", "data science", "natural language processing",
        "computer vision", "climate", "medicine", "transport", "energy", "architecture", "literature",
    ]


def download_wikipedia(
    db_path: str | Path,
    language: str = "ru",
    max_pages: int = 500,
    topics: list[str] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    if language not in {"ru", "en"}:
        raise ValueError("Язык Wikipedia должен быть ru или en")
    max_pages = max(10, min(5000, int(max_pages)))
    topics = list(dict.fromkeys(x.strip() for x in (topics or load_topics("", language)) if x.strip()))
    if not topics:
        topics = load_topics("", language)
    store = WebKnowledgeStore(db_path)
    page_titles: list[str] = []
    title_seen: set[str] = set()
    request_count = 0
    for i, topic in enumerate(topics):
        if len(page_titles) >= max_pages:
            break
        if progress:
            progress(f"Поиск статей: запрос {i + 1}/{len(topics)} — {topic} (найдено {len(page_titles)}/{max_pages})")
        data = _request_json(language, {
            "action": "query", "list": "search", "srsearch": topic,
            "srnamespace": "0", "srlimit": "10",
        })
        request_count += 1
        for item in data.get("query", {}).get("search", []):
            title = str(item.get("title", "")).strip()
            if title and title not in title_seen:
                title_seen.add(title)
                page_titles.append(title)
                if len(page_titles) >= max_pages:
                    break
        # Serialized low-rate requests; obey the host's public API etiquette.
        time.sleep(0.4)
    added = skipped = chars = 0
    for offset in range(0, len(page_titles), 20):
        batch = page_titles[offset:offset + 20]
        if progress:
            progress(f"Скачивание статей: {min(offset + len(batch), len(page_titles))}/{len(page_titles)}")
        data = _request_json(language, {
            "action": "query", "prop": "extracts|info", "inprop": "url",
            "explaintext": "1", "exsectionformat": "plain",
            "titles": "|".join(batch),
        })
        request_count += 1
        pages = data.get("query", {}).get("pages", [])
        for page in pages:
            title = str(page.get("title", "")).strip()
            extract = str(page.get("extract", "")).strip()
            extract = re.sub(r"\[\d+\]", "", extract)
            extract = re.sub(r"\s+", " ", extract).strip()[:10000]
            if len(extract) < 250:
                skipped += 1
                continue
            url = str(page.get("fullurl", "")).strip()
            if not url:
                url = f"https://{language}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            if store.add_document(title, url, extract, f"Wikipedia ({language})", "CC BY-SA 4.0; GNU FDL where applicable"):
                added += 1
                chars += len(extract)
            else:
                skipped += 1
        time.sleep(0.4)
    stats = store.stats()
    if progress:
        progress(f"Готово: добавлено {added} новых статей. Всего локально: {stats['documents']} статей, {stats['chunks']} фрагментов.")
    return {
        "added_pages": added,
        "skipped_pages": skipped,
        "characters_downloaded": chars,
        "total_pages": stats["documents"],
        "total_chunks": stats["chunks"],
        "total_characters": stats["characters"],
        "requests": request_count,
    }



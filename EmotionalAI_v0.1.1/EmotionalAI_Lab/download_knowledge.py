"""Download and index an openly licensed Wikipedia knowledge subset for Emotional."""
from __future__ import annotations

import argparse
from pathlib import Path
from internet_ingest import download_wikipedia, load_topics
from knowledge_store import WebKnowledgeStore


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Загрузить статьи Wikipedia в локальную базу Emotional")
    parser.add_argument("--language", choices=("ru", "en"), default="ru", help="язык статей: ru или en")
    parser.add_argument("--pages", type=int, default=1000, help="максимум статей за запуск (10–5000)")
    parser.add_argument("--topics", type=Path, default=None, help="свой список тем, по одной теме на строку")
    args = parser.parse_args()
    if not 10 <= args.pages <= 5000:
        parser.error("--pages должен быть от 10 до 5000")
    topic_path = args.topics or (root / "data" / ("internet_topics_ru.txt" if args.language == "ru" else "internet_topics_en.txt"))
    topics = load_topics(topic_path, args.language)
    db_path = root / "data" / "internet_knowledge.sqlite3"

    try:
        result = download_wikipedia(
            db_path=db_path,
            language=args.language,
            max_pages=args.pages,
            topics=topics,
            progress=print,
        )
    except KeyboardInterrupt:
        print("Прервано. Уже загруженные страницы остаются в базе.")
        return 130
    except Exception as exc:
        print(f"Ошибка загрузки: {exc}")
        return 1
    print("\nРезультат:")
    print(f"  Новых статей: {result['added_pages']}")
    print(f"  Всего статей: {result['total_pages']}")
    print(f"  Фрагментов для поиска: {result['total_chunks']}")
    print(f"  Символов в корпусе: {result['total_characters']:,}".replace(",", " "))
    print(f"  База: {db_path}")
    print("  Источник: Wikipedia; см. data/INTERNET_DATA_LICENSES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

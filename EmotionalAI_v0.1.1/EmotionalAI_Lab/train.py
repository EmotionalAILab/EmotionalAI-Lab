"""Небольшое офлайн-обучение статистике слов из обычного текстового корпуса.

Каждая непустая строка считается одним контекстом. Часто встречающиеся рядом слова
получают более сильную связь; затем эти связи немного расширяют поиск в памяти.
Это не обучение нейросети и не понимание смысла на человеческом уровне.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from memory import MemoryStore
from semantic import tokenize


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучить ассоциативную память на текстовом корпусе")
    parser.add_argument("--corpus", default="data/seed_corpus.txt", help="UTF-8 текст, одна фраза/абзац на строку")
    parser.add_argument("--reset", action="store_true", help="удалить прежнюю БД перед обучением (ОСТОРОЖНО: сотрёт память)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = root / "config.json"
    if not config_path.exists():
        config_path = root / "config.template.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    db_path = root / config.get("database_path", "data/agent_memory.sqlite3")
    if args.reset and db_path.exists():
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(db_path) + suffix)
            if candidate.exists():
                candidate.unlink()
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = root / corpus
    if not corpus.exists():
        print(f"Не найден корпус: {corpus}")
        return 2
    memory = MemoryStore(db_path)
    seen = 0
    token_total = 0
    try:
        for line in corpus.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            memory.learn_associations(line)
            seen += 1
            token_total += len(tokenize(line))
    finally:
        memory.close()
    print(f"Готово. Контекстов обработано: {seen}; слов после фильтрации: {token_total}.")
    print("Ассоциативные связи сохранены в локальной базе данных.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

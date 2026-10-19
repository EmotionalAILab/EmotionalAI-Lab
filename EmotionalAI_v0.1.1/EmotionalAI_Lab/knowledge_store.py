"""Локальный поисковый корпус для текстов из Интернета и файлов пользователя.

Тексты сохраняются с источником и лицензией. Поиск работает локально через SQLite FTS5;
веса языковой модели не меняются, зато модель получает релевантные фрагменты по запросу.
"""
from __future__ import annotations

import math
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{2,}")
STOPWORDS = {
    "это", "как", "что", "когда", "где", "почему", "зачем", "кто", "кого", "какой", "какая", "какие",
    "можно", "нужно", "надо", "если", "или", "для", "при", "про", "мне", "меня", "мной", "мой", "моя",
    "мои", "мое", "могу", "будет", "быть", "есть", "нет", "так", "вот", "там", "тут", "уже", "ещё",
    "еще", "всё", "все", "очень", "просто", "ли", "же", "бы", "то", "на", "по", "из", "за", "от",
    "до", "не", "ни", "и", "а", "но", "в", "во", "с", "со", "к", "ко", "о", "об", "обо", "й",
    "the", "and", "are", "was", "were", "for", "with", "that", "this", "what", "why", "how", "who", "from",
}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 2 and t.lower() not in STOPWORDS]


def split_text(text: str, chunk_chars: int = 1500, overlap: int = 180) -> list[str]:
    """Split text into readable chunks while retaining overlap for context."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= chunk_chars:
        return [clean]
    out: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_chars)
        if end < len(clean):
            boundary = clean.rfind(". ", start + chunk_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        piece = clean[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return out


class WebKnowledgeStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS documents (
                url TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
                license TEXT NOT NULL, char_count INTEGER NOT NULL DEFAULT 0,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            try:
                conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    title, body, url UNINDEXED, source UNINDEXED, license UNINDEXED,
                    tokenize='unicode61'
                )""")
                self.fts5_enabled = True
            except sqlite3.OperationalError:
                # Compatibility fallback for custom SQLite builds without FTS5.
                self.fts5_enabled = False
                conn.execute("""CREATE TABLE IF NOT EXISTS chunks_plain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, body TEXT NOT NULL,
                    url TEXT NOT NULL, source TEXT NOT NULL, license TEXT NOT NULL
                )""")

    def add_document(self, title: str, url: str, text: str, source: str = "Wikipedia", license_name: str = "CC BY-SA 4.0") -> bool:
        title, url, text = title.strip(), url.strip(), text.strip()
        chunks = split_text(text)
        if not title or not url or len(text) < 120 or not chunks:
            return False
        with self._connection() as conn:
            if conn.execute("SELECT 1 FROM documents WHERE url=?", (url,)).fetchone():
                return False
            conn.execute("INSERT INTO documents(url,title,source,license,char_count) VALUES(?,?,?,?,?)",
                         (url, title, source, license_name, len(text)))
            for chunk in chunks:
                table = "chunks" if self.fts5_enabled else "chunks_plain"
                conn.execute(f"INSERT INTO {table}(title,body,url,source,license) VALUES(?,?,?,?,?)",
                             (title, chunk, url, source, license_name))
        return True

    def search(self, query: str, limit: int = 3, min_score: float = 0.12) -> list[dict[str, Any]]:
        query_tokens = list(dict.fromkeys(tokens(query)))
        if not query_tokens:
            return []
        # FTS OR-search makes broad questions work; ranking is then adjusted for actual token overlap.
        match_query = " OR ".join('"' + t.replace('"', '""') + '"' for t in query_tokens[:16])
        try:
            with self._connection() as conn:
                if self.fts5_enabled:
                    rows = conn.execute(
                        "SELECT title,body,url,source,license,bm25(chunks) AS rank "
                        "FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
                        (match_query, max(20, limit * 8)),
                    ).fetchall()
                else:
                    where = " OR ".join("LOWER(title || ' ' || body) LIKE ?" for _ in query_tokens[:8])
                    params = tuple("%" + token.lower() + "%" for token in query_tokens[:8])
                    rows = conn.execute(
                        "SELECT title,body,url,source,license,0.0 AS rank FROM chunks_plain WHERE " + where + " LIMIT ?",
                        (*params, max(200, limit * 20)),
                    ).fetchall()
        except sqlite3.OperationalError:
            return []
        ranked: list[dict[str, Any]] = []
        for row in rows:
            title = str(row["title"])
            body = str(row["body"])
            title_set = set(tokens(title))
            body_set = set(tokens(body))
            title_hits = len(set(query_tokens) & title_set)
            body_hits = len(set(query_tokens) & body_set)
            overlap = title_hits + body_hits
            unique_hits = len(set(query_tokens) & (title_set | body_set))
            score = (unique_hits / max(1, min(len(set(query_tokens)), 6))) + (title_hits * 0.28)
            # One-term searches are allowed; longer queries need more than a stopword-level hit.
            needed = 1 if len(set(query_tokens)) <= 2 else 2
            if unique_hits < needed or score < min_score:
                continue
            ranked.append({
                "title": title,
                "text": body,
                "url": str(row["url"]),
                "source": str(row["source"]),
                "license": str(row["license"]),
                "score": score,
                "matched_terms": unique_hits,
            })
        ranked.sort(key=lambda item: (item["score"], item["matched_terms"]), reverse=True)
        # One source shouldn't swamp all context with adjacent chunks from the same article.
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        per_url: dict[str, int] = {}
        for item in ranked:
            key = (item["url"], item["text"][:120])
            if key in seen or per_url.get(item["url"], 0) >= 2:
                continue
            seen.add(key)
            per_url[item["url"]] = per_url.get(item["url"], 0) + 1
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def documents_for_training(self, limit: int = 500, source_prefix: str = "Wikipedia", offset: int = 0) -> list[dict[str, str]]:
        """Return a bounded page of article texts for incremental local training."""
        limit = max(1, min(1000, int(limit)))
        offset = max(0, int(offset))
        with self._connection() as conn:
            if self.fts5_enabled:
                rows = conn.execute(
                    "SELECT d.url,d.title,d.source,d.license,COALESCE(GROUP_CONCAT(c.body,' '),'') AS content "
                    "FROM documents d LEFT JOIN chunks c ON c.url=d.url "
                    "WHERE d.source LIKE ? GROUP BY d.url,d.title,d.source,d.license "
                    "ORDER BY d.added_at DESC LIMIT ? OFFSET ?",
                    (source_prefix + "%", limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT d.url,d.title,d.source,d.license,COALESCE(GROUP_CONCAT(c.body,' '),'') AS content "
                    "FROM documents d LEFT JOIN chunks_plain c ON c.url=d.url "
                    "WHERE d.source LIKE ? GROUP BY d.url,d.title,d.source,d.license "
                    "ORDER BY d.added_at DESC LIMIT ? OFFSET ?",
                    (source_prefix + "%", limit, offset),
                ).fetchall()
        return [{key: str(row[key] or "") for key in ("url", "title", "source", "license", "content")} for row in rows]

    def document_count(self, source_prefix: str = "") -> int:
        with self._connection() as conn:
            if source_prefix:
                row = conn.execute("SELECT COUNT(*) AS n FROM documents WHERE source LIKE ?", (source_prefix + "%",)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])

    def stats(self) -> dict[str, int]:
        with self._connection() as conn:
            docs = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            table = "chunks" if self.fts5_enabled else "chunks_plain"
            chunks = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            chars = int(conn.execute("SELECT COALESCE(SUM(char_count),0) FROM documents").fetchone()[0])
        return {"documents": docs, "chunks": chunks, "characters": chars}

    def import_text_file(self, path: str | Path) -> int:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        # Split long files into a number of documents so individual parts can be retrieved separately.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 120]
        if not paragraphs:
            paragraphs = split_text(text, chunk_chars=6000, overlap=250)
        added = 0
        buffer: list[str] = []
        size = 0
        part = 1
        for paragraph in paragraphs:
            if size + len(paragraph) > 6000 and buffer:
                url = f"{file_path.resolve().as_uri()}#part-{part}"
                if self.add_document(f"{file_path.name} — часть {part}", url, "\n\n".join(buffer), "Локальный файл", "Лицензия определяется владельцем файла"):
                    added += 1
                part += 1
                buffer, size = [], 0
            buffer.append(paragraph)
            size += len(paragraph)
        if buffer:
            url = f"{file_path.resolve().as_uri()}#part-{part}"
            if self.add_document(f"{file_path.name} — часть {part}", url, "\n\n".join(buffer), "Локальный файл", "Лицензия определяется владельцем файла"):
                added += 1
        return added

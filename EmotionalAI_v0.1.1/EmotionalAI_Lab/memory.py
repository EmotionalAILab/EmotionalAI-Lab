"""Долговременная память: эпизоды диалога, заметки, факты и связи слов (SQLite)."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from semantic import similarity, tokenize, normalize


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class MemoryStore:
    """Хранит всё общение и умеет доставать похожие фрагменты спустя время.

    Старые базы автоматически мигрируют при открытии: новые таблицы/колонки
    добавляются, прежние факты и эпизоды не удаляются.
    """

    CONTROL_PREFIXES = (
        "/", "что ты помнишь", "что помнишь", "покажи память", "/память",
        "как ты себя чувствуешь", "/состояние", "/цели", "почему ты ответил",
        "почему ты так ответил", "объясни свой ответ", "запомни", "сохрани",
        "правильно", "исправление", "забудь", "удали из памяти", "не запоминай",
        "/история", "история диалога",
    )

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_text TEXT NOT NULL,
            agent_text TEXT NOT NULL,
            emotion_json TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.3,
            tags TEXT NOT NULL DEFAULT '[]',
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.75,
            source TEXT NOT NULL DEFAULT 'conversation',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.55,
            source TEXT NOT NULL DEFAULT 'conversation',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS associations (
            word_a TEXT NOT NULL,
            word_b TEXT NOT NULL,
            strength REAL NOT NULL DEFAULT 0.1,
            count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(word_a, word_b)
        );
        CREATE TABLE IF NOT EXISTS agent_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS behavior_rules (
            rule_key TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.85,
            source TEXT NOT NULL DEFAULT 'user_feedback',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS learning_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_text TEXT NOT NULL,
            previous_answer TEXT NOT NULL DEFAULT '',
            correction TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'correction',
            confidence REAL NOT NULL DEFAULT 0.85,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS trained_sources (
            url TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject, active);
        CREATE INDEX IF NOT EXISTS idx_notes_active ON notes(active, updated_at);
        CREATE INDEX IF NOT EXISTS idx_learning_examples_active ON learning_examples(active, created_at);
        """)
        # Migrate databases created by the previous hotfixes without deleting data.
        episode_columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(episodes)").fetchall()}
        if "active" not in episode_columns:
            self.conn.execute("ALTER TABLE episodes ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_active ON episodes(active, timestamp)")
        self.conn.commit()

    @classmethod
    def _is_control_text(cls, text: str) -> bool:
        low = normalize(text)
        return any(low.startswith(normalize(prefix)) for prefix in cls.CONTROL_PREFIXES)

    def add_episode(self, user_text: str, agent_text: str, emotion: dict,
                    importance: float = 0.3, tags: list[str] | None = None) -> None:
        stamp = now_iso()
        self.conn.execute(
            "INSERT INTO episodes(timestamp,user_text,agent_text,emotion_json,importance,tags) VALUES(?,?,?,?,?,?)",
            (stamp, user_text, agent_text,
             json.dumps(emotion, ensure_ascii=False), max(0, min(1, importance)),
             json.dumps(tags or [], ensure_ascii=False)),
        )
        # Keep a searchable record of ordinary user messages as well as a full episode.
        # Explicit memory/control commands are stored in episodes, but not as conversational facts.
        # A message tagged as direct criticism remains in the episode history for conversational
        # context, but it should not become a lasting profile note or distort word associations.
        if user_text.strip() and not self._is_control_text(user_text) and "criticism" not in (tags or []):
            self.add_note(user_text, confidence=0.55, source="conversation")
            self.learn_associations(user_text)
        self.conn.commit()

    def add_note(self, content: str, confidence: float = 0.55, source: str = "conversation") -> None:
        content = " ".join(content.strip().split())
        if not content:
            return
        # Avoid accumulating exact duplicate notes, but retain separate recollections about a topic.
        recent = self.conn.execute(
            "SELECT id,content,confidence FROM notes WHERE active=1 ORDER BY id DESC LIMIT 800"
        ).fetchall()
        key = normalize(content)
        for row in recent:
            if normalize(row["content"]) == key:
                self.conn.execute(
                    "UPDATE notes SET confidence=MAX(confidence,?), source=?, updated_at=? WHERE id=?",
                    (max(0.0, min(1.0, confidence)), source, now_iso(), row["id"]),
                )
                self.conn.commit()
                return
        stamp = now_iso()
        self.conn.execute(
            "INSERT INTO notes(content,confidence,source,created_at,updated_at,active) VALUES(?,?,?,?,?,1)",
            (content, max(0.0, min(1.0, confidence)), source, stamp, stamp),
        )
        self.conn.commit()

    def all_notes(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE active=1 ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 0.8,
                 source: str = "conversation") -> None:
        subject, predicate, obj = subject.strip(), predicate.strip(), obj.strip()
        if not subject or not predicate or not obj:
            return
        stamp = now_iso()
        bounded_confidence = max(0.0, min(1.0, confidence))
        # Exact repeats reinforce an existing memory instead of creating endless duplicates.
        existing = self.conn.execute(
            "SELECT id FROM facts WHERE subject=? AND predicate=? AND object=? AND active=1 ORDER BY id DESC LIMIT 1",
            (subject, predicate, obj),
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE facts SET confidence=MAX(confidence,?), source=?, updated_at=? WHERE id=?",
                (bounded_confidence, source, stamp, existing["id"]),
            )
            self.conn.commit()
            return
        # Profile-like properties normally have one current value; preferences can be plural.
        multi_value_predicates = {"любит", "не любит", "нравится", "не нравится", "предпочитает", "имеет", "играет в", "использует", "планирует", "хочет"}
        if predicate not in multi_value_predicates:
            self.conn.execute(
                "UPDATE facts SET active=0 WHERE subject=? AND predicate=? AND active=1",
                (subject, predicate),
            )
        self.conn.execute(
            "INSERT INTO facts(subject,predicate,object,confidence,source,created_at,updated_at,active) VALUES(?,?,?,?,?,?,?,1)",
            (subject, predicate, obj, bounded_confidence, source, stamp, stamp),
        )
        self.conn.commit()

    def facts(self, subject: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if subject:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE active=1 AND subject=? ORDER BY updated_at DESC LIMIT ?",
                (subject, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE active=1 ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _expanded_query(self, query: str) -> str:
        words = tokenize(query)
        related = [word for word, strength in self.associated_words(words, 12) if strength >= 0.25]
        return query + " " + " ".join(related) if related else query

    def search_facts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM facts WHERE active=1").fetchall()
        ranked = []
        expanded_query = self._expanded_query(query)
        has_related = expanded_query != query
        for row in rows:
            content = f"{row['subject']} {row['predicate']} {row['object']}"
            direct = similarity(query, content)
            expanded = similarity(expanded_query, content) if has_related else direct
            score = (0.78 * direct + 0.22 * expanded) * float(row["confidence"])
            if score > 0.025:
                ranked.append((score, dict(row)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [dict(item, relevance=score) for score, item in ranked[:limit]]

    def search_notes(self, query: str, limit: int = 5, min_score: float = 0.0) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM notes WHERE active=1").fetchall()
        ranked = []
        expanded_query = self._expanded_query(query)
        has_related = expanded_query != query
        for index, row in enumerate(rows):
            direct = similarity(query, row["content"])
            expanded = similarity(expanded_query, row["content"]) if has_related else direct
            score = (0.80 * direct + 0.20 * expanded) * float(row["confidence"])
            # Very gentle recency preference; relevance remains the main signal.
            score *= 1.0 / (1.0 + index * 0.0002)
            if score >= min_score:
                ranked.append((score, dict(row)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [dict(item, relevance=score) for score, item in ranked[:limit]]

    def forget_matching(self, query: str) -> int:
        query = query.strip()
        if not query:
            return 0
        total = 0
        rows = self.conn.execute("SELECT id,subject,predicate,object FROM facts WHERE active=1").fetchall()
        fact_ids = [r["id"] for r in rows if similarity(query, f"{r['subject']} {r['predicate']} {r['object']}") >= 0.12 or normalize(query) in normalize(str(r["object"]))]
        if fact_ids:
            self.conn.executemany("UPDATE facts SET active=0 WHERE id=?", [(i,) for i in fact_ids])
            total += len(fact_ids)
        notes = self.conn.execute("SELECT id,content FROM notes WHERE active=1").fetchall()
        note_ids = [r["id"] for r in notes if similarity(query, r["content"]) >= 0.16 or normalize(query) in normalize(r["content"])]
        if note_ids:
            self.conn.executemany("UPDATE notes SET active=0 WHERE id=?", [(i,) for i in note_ids])
            total += len(note_ids)
        eps = self.conn.execute("SELECT id,user_text,agent_text FROM episodes WHERE active=1").fetchall()
        episode_ids = [r["id"] for r in eps if similarity(query, r["user_text"]) >= 0.20 or normalize(query) in normalize(r["user_text"])]
        if episode_ids:
            self.conn.executemany("UPDATE episodes SET active=0 WHERE id=?", [(i,) for i in episode_ids])
            total += len(episode_ids)
        self.conn.commit()
        return total

    def search_episodes(self, query: str, limit: int = 5, min_score: float = 0.0) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM episodes WHERE active=1 ORDER BY id DESC").fetchall()
        ranked = []
        expanded_query = self._expanded_query(query)
        has_related = expanded_query != query
        for index, row in enumerate(rows):
            content = row["user_text"] + " " + row["agent_text"]
            direct = similarity(query, content)
            expanded = similarity(expanded_query, content) if has_related else direct
            score = (0.80 * direct + 0.20 * expanded)
            score *= 1.0 + min(0.25, float(row["importance"]) * 0.25)
            score *= 1.0 / (1.0 + index * 0.0002)
            if score >= min_score:
                ranked.append((score, dict(row)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [dict(item, relevance=score) for score, item in ranked[:limit]]

    def learn_associations(self, text: str, commit: bool = True) -> None:
        words = list(dict.fromkeys(tokenize(text)))[:18]
        for i, a in enumerate(words):
            for b in words[i + 1:]:
                x, y = sorted((a, b))
                row = self.conn.execute(
                    "SELECT strength,count FROM associations WHERE word_a=? AND word_b=?", (x, y)
                ).fetchone()
                if row:
                    new_count = int(row["count"]) + 1
                    strength = min(1.0, float(row["strength"]) + 0.06 * (1 - float(row["strength"])) )
                    self.conn.execute(
                        "UPDATE associations SET strength=?,count=? WHERE word_a=? AND word_b=?",
                        (strength, new_count, x, y),
                    )
                else:
                    self.conn.execute(
                        "INSERT INTO associations(word_a,word_b,strength,count) VALUES(?,?,0.16,1)", (x, y)
                    )
        if commit:
            self.conn.commit()

    def learn_associations_bulk(self, texts: list[str] | tuple[str, ...], max_words_per_text: int = 12) -> int:
        """Learn co-occurrence strengths from many contexts in a single SQLite transaction.

        This is lightweight associative learning, not neural-network weight training.
        Frequent word pairs become easier to retrieve later. It is designed for the
        bundled seed corpus and selected Wikipedia extracts.
        """
        from collections import Counter
        pair_counts: Counter[tuple[str, str]] = Counter()
        max_words = max(3, min(18, int(max_words_per_text)))
        for text in texts:
            words = list(dict.fromkeys(tokenize(str(text))))[:max_words]
            for i, word_a in enumerate(words):
                for word_b in words[i + 1:]:
                    if word_a == word_b:
                        continue
                    pair_counts[tuple(sorted((word_a, word_b)))] += 1
        if not pair_counts:
            return 0
        with self.conn:
            self.conn.executemany(
                """INSERT INTO associations(word_a,word_b,strength,count) VALUES(?,?,?,?)
                   ON CONFLICT(word_a,word_b) DO UPDATE SET
                     strength=MIN(1.0, associations.strength + (1.0-associations.strength)*
                                  MIN(0.30, 0.035*excluded.count)),
                     count=associations.count+excluded.count""",
                [(a, b, min(0.95, 0.16 + 0.025 * (count - 1)), count)
                 for (a, b), count in pair_counts.items()],
            )
        return sum(pair_counts.values())

    def association_strength(self, a: str, b: str) -> float:
        x, y = sorted((normalize(a), normalize(b)))
        row = self.conn.execute(
            "SELECT strength FROM associations WHERE word_a=? AND word_b=?", (x, y)
        ).fetchone()
        return float(row["strength"]) if row else 0.0

    def associated_words(self, words: list[str], limit: int = 12) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for word in list(dict.fromkeys(normalize(w) for w in words))[:15]:
            rows = self.conn.execute(
                "SELECT word_b AS other,strength FROM associations WHERE word_a=? "
                "UNION ALL SELECT word_a AS other,strength FROM associations WHERE word_b=?",
                (word, word),
            ).fetchall()
            for row in rows:
                other = str(row["other"])
                if other != word:
                    scores[other] = scores.get(other, 0.0) + float(row["strength"])
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

    def recent_episodes(self, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM episodes WHERE active=1 ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def load_agent_state(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM agent_state WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def save_agent_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO agent_state(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (str(key), str(value), now_iso()),
        )
        self.conn.commit()

    def add_behavior_rule(self, rule_key: str, content: str, confidence: float = 0.88,
                          source: str = "user_feedback") -> None:
        rule_key = normalize(rule_key).strip()
        content = " ".join(str(content).split()).strip()[:1000]
        if not rule_key or not content:
            return
        stamp = now_iso()
        self.conn.execute(
            """INSERT INTO behavior_rules(rule_key,content,confidence,source,created_at,updated_at,active)
               VALUES(?,?,?,?,?,?,1)
               ON CONFLICT(rule_key) DO UPDATE SET content=excluded.content,
                 confidence=MAX(behavior_rules.confidence,excluded.confidence), source=excluded.source,
                 updated_at=excluded.updated_at, active=1""",
            (rule_key, content, max(0.0, min(1.0, confidence)), source, stamp, stamp),
        )
        self.conn.commit()

    def behavior_rules(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM behavior_rules WHERE active=1 ORDER BY confidence DESC, updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def add_learning_example(self, trigger_text: str, previous_answer: str, correction: str,
                             category: str = "correction", confidence: float = 0.86) -> None:
        trigger_text = " ".join(str(trigger_text).split()).strip()[:1600]
        previous_answer = " ".join(str(previous_answer).split()).strip()[:2200]
        correction = " ".join(str(correction).split()).strip()[:2200]
        if len(trigger_text) < 5 or len(correction) < 12:
            return
        # Avoid duplicating the same correction when the user repeats it.
        rows = self.conn.execute(
            "SELECT id,trigger_text,correction FROM learning_examples WHERE active=1 ORDER BY id DESC LIMIT 120"
        ).fetchall()
        if any(normalize(row["correction"]) == normalize(correction)
               and similarity(trigger_text, str(row["trigger_text"])) > 0.92
               for row in rows):
            return
        self.conn.execute(
            "INSERT INTO learning_examples(trigger_text,previous_answer,correction,category,confidence,created_at,active) "
            "VALUES(?,?,?,?,?,?,1)",
            (trigger_text, previous_answer, correction, category,
             max(0.0, min(1.0, confidence)), now_iso()),
        )
        self.conn.commit()

    def search_learning_examples(self, query: str, limit: int = 3, min_score: float = 0.0) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM learning_examples WHERE active=1 ORDER BY id DESC"
        ).fetchall()
        ranked = []
        for index, row in enumerate(rows):
            score = similarity(query, str(row["trigger_text"]))
            if score >= min_score:
                item = dict(row)
                item["relevance"] = score * float(row["confidence"]) * (1.0 / (1.0 + index * 0.0005))
                ranked.append(item)
        ranked.sort(key=lambda item: item["relevance"], reverse=True)
        return ranked[:max(1, limit)]

    def source_was_trained(self, url: str) -> bool:
        return self.conn.execute("SELECT 1 FROM trained_sources WHERE url=?", (url,)).fetchone() is not None

    def mark_sources_trained(self, sources: list[tuple[str, str]]) -> None:
        stamp = now_iso()
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO trained_sources(url,title,trained_at) VALUES(?,?,?)",
                [(url, title, stamp) for url, title in sources if url],
            )

    def stats(self) -> dict[str, int]:
        """Counts persistent memories without loading the full database into memory."""
        result = {}
        for table, name in (("episodes", "episodes"), ("notes", "notes"), ("facts", "facts"), ("associations", "associations")):
            if table == "associations":
                row = self.conn.execute("SELECT COUNT(*) AS n FROM associations").fetchone()
            else:
                row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE active=1").fetchone()
            result[name] = int(row["n"])
        result["behavior_rules"] = int(self.conn.execute("SELECT COUNT(*) FROM behavior_rules WHERE active=1").fetchone()[0])
        result["learning_examples"] = int(self.conn.execute("SELECT COUNT(*) FROM learning_examples WHERE active=1").fetchone()[0])
        result["trained_sources"] = int(self.conn.execute("SELECT COUNT(*) FROM trained_sources").fetchone()[0])
        return result

    def close(self) -> None:
        self.conn.close()

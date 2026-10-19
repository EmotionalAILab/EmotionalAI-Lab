"""Обучение на явных указаниях, биографических фактах и языковых ассоциациях."""
from __future__ import annotations
import re
from memory import MemoryStore


class LearningEngine:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    @staticmethod
    def _clean(value: str) -> str:
        value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,!?;:\"'«»"))
        return value[:500]

    def learn_from(self, text: str, associate: bool = True) -> dict:
        raw = text.strip()
        low = raw.lower().replace("ё", "е")
        if not raw:
            return {"learned": False, "kind": "empty"}

        # Explicit instruction has the highest confidence.
        explicit = re.match(
            r"^(?:запомни(?:,? что)?|сохрани(?:,? что)?|правильно|исправление)\s*[:,-]?\s*(.+)$",
            raw, flags=re.IGNORECASE,
        )
        if explicit:
            statement = self._clean(explicit.group(1))
            if statement:
                source = "explicit_user_correction" if low.startswith(("правильно", "исправление")) else "explicit_user_memory"
                self._store_statement(statement, confidence=0.97, source=source)
                if source == "explicit_user_correction":
                    prior = self.memory.recent_episodes(1)
                    if prior:
                        previous = prior[0]
                        trigger = str(previous.get("user_text", "")).strip()
                        previous_answer = str(previous.get("agent_text", "")).strip()
                        if trigger and not self.memory._is_control_text(trigger):
                            self.memory.add_learning_example(
                                trigger_text=trigger,
                                previous_answer=previous_answer,
                                correction=statement,
                                category="correction",
                                confidence=0.94,
                            )
                if associate:
                    self.memory.learn_associations(statement)
                return {"learned": True, "kind": "explicit", "statement": statement}

        # Automatically learn simple profile statements even when the user didn't say "remember".
        # Ordered with negative/preference phrases before the more general variants.
        patterns = [
            (r"\bя не люблю\s+(.+)", "не любит", 0.86),
            (r"\bмне не нравится\s+(.+)", "не нравится", 0.86),
            (r"\bя предпочитаю\s+(.+)", "предпочитает", 0.84),
            (r"\bмне нравится\s+(.+)", "нравится", 0.83),
            (r"\bя люблю\s+(.+)", "любит", 0.83),
            (r"\bменя зовут\s+(.+)", "имя", 0.96),
            (r"\bя живу в\s+(.+)", "живёт в", 0.80),
            (r"\bя учусь в\s+(.+)", "учится в", 0.80),
            (r"\bя играю в\s+(.+)", "играет в", 0.76),
            (r"\bя использую\s+(.+)", "использует", 0.74),
            (r"\bу меня есть\s+(.+)", "имеет", 0.76),
            (r"\bя планирую\s+(.+)", "планирует", 0.66),
            (r"\bя хочу\s+(.+)", "хочет", 0.62),
        ]
        for pattern, predicate, confidence in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if not match:
                continue
            obj = self._clean(re.split(r"[.!?]\s", match.group(1), maxsplit=1)[0])
            if obj:
                self.memory.add_fact("пользователь", predicate, obj, confidence, "auto_extracted_profile")
                if associate:
                    self.memory.learn_associations(raw)
                return {"learned": True, "kind": "auto_fact", "predicate": predicate, "object": obj}

        # Natural-language format like "мой любимый цвет — синий".
        property_match = re.search(
            r"\bмои?\s+(.{2,70}?)\s*(?:—|–|-|это|:)\s*(.{1,160})$", raw,
            flags=re.IGNORECASE,
        )
        if property_match and "?" not in raw:
            prop = self._clean(property_match.group(1))
            obj = self._clean(property_match.group(2))
            if prop and obj:
                self.memory.add_fact("пользователь", prop, obj, 0.78, "auto_extracted_profile")
                if associate:
                    self.memory.learn_associations(raw)
                return {"learned": True, "kind": "auto_fact", "predicate": prop, "object": obj}

        # Even if no clean fact can be extracted, add_episode() records the whole message
        # and updates associations. This keeps information retrievable without inventing a fact.
        if associate:
            self.memory.learn_associations(raw)
        return {"learned": False, "kind": "associations"}

    def _store_statement(self, statement: str, confidence: float, source: str) -> None:
        # Notes are append-only historical memories. Do not put all statements into a single
        # (knowledge, user_statement) fact slot: that would overwrite the previous statement.
        self.memory.add_note(statement, confidence=confidence, source=source)

        patterns = [
            (r"(?:мой|моя|мое|мои)\s+(.+?)\s*(?:—|–|-|это|:)\s*(.+)", "property"),
            (r"\bя не люблю\s+(.+)", "не любит"),
            (r"\bмне не нравится\s+(.+)", "не нравится"),
            (r"\bя люблю\s+(.+)", "любит"),
            (r"\bмне нравится\s+(.+)", "нравится"),
            (r"\bя предпочитаю\s+(.+)", "предпочитает"),
            (r"\bя живу в\s+(.+)", "живёт в"),
            (r"\bя учусь в\s+(.+)", "учится в"),
            (r"\bу меня есть\s+(.+)", "имеет"),
        ]
        for pattern, predicate in patterns:
            match = re.search(pattern, statement, flags=re.IGNORECASE)
            if not match:
                continue
            if predicate == "property":
                subject = self._clean(match.group(1))
                obj = self._clean(match.group(2))
                if subject and obj:
                    self.memory.add_fact("пользователь", subject, obj, confidence, source)
            else:
                obj = self._clean(match.group(1))
                if obj:
                    self.memory.add_fact("пользователь", predicate, obj, confidence, source)
            break

    def forget(self, text: str) -> int:
        query = re.sub(
            r"^(?:забудь|удали из памяти|не запоминай)\s*", "", text.strip(), flags=re.IGNORECASE
        )
        return self.memory.forget_matching(query)

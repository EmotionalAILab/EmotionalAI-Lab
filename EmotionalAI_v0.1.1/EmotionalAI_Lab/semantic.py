"""Упрощённая языковая обработка для русского текста (без внешних библиотек)."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
STOP_WORDS = set("а без более бы был была были быть в во вот все всё вы где да даже для до же за и из или им их к как ко когда кто ли либо мне меня мне мой моя моё мои мы на над не него неё нет ни них но ну о об от очень по под при про раз с со так такой там те тем то того тоже той только том ты у уже что чтобы эта это этот эти я меня наш наша нечто оно они она он каков какая какое какой это ведь если потому поэтому почему зачем чей чья чье который которая которое которые который самого себе себе свой своя свои своими".split())


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = [normalize(t) for t in WORD_RE.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def similarity(a: str, b: str) -> float:
    """Jaccard similarity with a small substring bonus for Russian word forms."""
    aa, bb = token_set(a), token_set(b)
    if not aa or not bb:
        return 0.0
    exact = len(aa & bb) / len(aa | bb)
    near = 0
    pairs = 0
    for x in aa:
        for y in bb:
            if x != y and min(len(x), len(y)) >= 4 and (x.startswith(y[:4]) or y.startswith(x[:4])):
                near += 1
            pairs += 1
    bonus = (near / pairs) * 0.12 if pairs else 0.0
    return min(1.0, exact + bonus)


class KnowledgeBase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict = {}
        self.reload()

    def reload(self) -> None:
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.entries = {}
        else:
            self.entries = {}

    def find(self, text: str) -> list[tuple[str, dict, float]]:
        """Rank exact and specific concept matches above broad keyword overlap."""
        text_n = normalize(text)
        words = set(tokenize(text))
        results = []
        for concept, item in self.entries.items():
            concept_n = normalize(concept)
            score = 0.0
            exact_concept = bool(concept_n and re.search(r"(?<!\w)" + re.escape(concept_n) + r"(?!\w)", text_n))
            if exact_concept:
                score = 1.0
            for kw in item.get("keywords", []):
                kw_n = normalize(kw)
                if not kw_n:
                    continue
                if re.search(r"(?<!\w)" + re.escape(kw_n) + r"(?!\w)", text_n):
                    direct_score = 0.94 if " " in kw_n else 0.88
                    score = max(score, direct_score)
                    continue
                overlap = words & set(tokenize(kw))
                if overlap:
                    score = max(score, 0.34 + 0.06 * min(2, len(overlap)))
            if score > 0:
                # Long, explicit concepts such as "оперативная память" should beat
                # broad concepts such as "память" when both occur in the same phrase.
                specificity = len(concept_n) if exact_concept else 0
                results.append((score, specificity, concept, item))
        results.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [(concept, item, score) for score, _specificity, concept, item in results]

    def find_explicit(self, text: str) -> list[tuple[str, dict, float]]:
        """Return only concepts whose full name/alias is literally present in the text.

        Unlike ``find`` this method does not use broad token overlap. It is intended for
        operations where an accidental keyword match is harmful, especially comparisons.
        A generic alias contained inside a longer matched phrase ("память" inside
        "оперативная память") is suppressed when the longer concept is itself matched.
        """
        text_n = normalize(text)
        candidates: list[tuple[int, int, int, str, dict, float]] = []
        for concept, item in self.entries.items():
            phrases = [concept, *item.get("keywords", [])]
            seen_phrases = set()
            for phrase in phrases:
                phrase_n = normalize(str(phrase)).strip()
                if not phrase_n or phrase_n in seen_phrases:
                    continue
                seen_phrases.add(phrase_n)
                pattern = re.compile(r"(?<!\w)" + re.escape(phrase_n) + r"(?!\w)")
                for match in pattern.finditer(text_n):
                    candidates.append((match.start(), match.end(), len(phrase_n), concept, item, 0.96 if " " in phrase_n else 0.88))
        # Keep the longest phrase for a given span; remove shorter concepts fully embedded
        # in longer explicit concepts to avoid false entities in comparisons.
        filtered = []
        for cand in candidates:
            start, end, length, concept, item, score = cand
            contained = any(
                other is not cand
                and other[0] <= start and other[1] >= end
                and (other[1] - other[0]) > (end - start)
                for other in candidates
            )
            if not contained:
                filtered.append(cand)
        best_by_concept: dict[str, tuple[int, int, int, str, dict, float]] = {}
        for cand in filtered:
            concept = cand[3]
            existing = best_by_concept.get(concept)
            if existing is None or (cand[2], -(cand[0])) > (existing[2], -(existing[0])):
                best_by_concept[concept] = cand
        ordered = sorted(best_by_concept.values(), key=lambda x: (x[0], -x[2]))
        return [(concept, item, score) for _start, _end, _length, concept, item, score in ordered]

    def related_words(self, word: str) -> list[str]:
        target = normalize(word)
        out: list[str] = []
        for concept, item in self.entries.items():
            kws = [normalize(k) for k in item.get("keywords", [])]
            if target == normalize(concept) or any(target in k or k in target for k in kws):
                out.extend(item.get("related", []))
        return list(dict.fromkeys(out))

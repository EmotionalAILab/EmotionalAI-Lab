"""Моделируемое эмоциональное состояние Emotional.

Это программная модель реакции на ход диалога, а не утверждение о сознании.
Состояние влияет на выбор формулировок и сохраняется между запусками.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import re

POSITIVE = (
    "спасибо", "классно", "отлично", "здорово", "круто", "молодец", "полезно",
    "нравится", "люблю", "рад", "рада", "смешно", "угар", "получилось", "ура",
    "помогло", "супер", "великолепно", "обожаю", "прикольно", "интересно",
)
NEGATIVE = (
    "неправильно", "неверно", "ошибка", "бесит", "злюсь", "плохо", "ужасно",
    "ненавижу", "тупой", "тупо", "провал", "не работает", "сломалось", "разочарован",
    "раздражает", "задолбал", "заебал", "хрень", "чушь", "не то",
)
SADNESS_MARKERS = (
    "грустно", "печально", "одиноко", "мне плохо", "устал", "устала", "тоскливо",
    "расстроен", "расстроена", "обидно", "потерял", "потеряла", "скучаю",
)
ANGER_MARKERS = (
    "бесит", "злюсь", "раздражает", "ненавижу", "тупой", "тупая", "идиот",
    "дебил", "заебал", "задолбал", "в ярости", "пиздец",
)
SURPRISE_MARKERS = (
    "ничего себе", "ого", "вау", "офигеть", "не может быть", "серьезно?!", "серьёзно?!",
)
CORRECTION_MARKERS = (
    "правильно:", "исправление:", "я имел в виду", "я имела в виду", "я спрашивал про",
    "я спрашивала про", "не то", "неправильно", "ответ должен быть", "я просил",
)


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


@dataclass
class EmotionState:
    valence: float = 0.10       # -1 .. +1, общий знак реакции
    arousal: float = 0.25       # 0 .. 1, интенсивность
    curiosity: float = 0.55     # 0 .. 1
    trust: float = 0.35         # 0 .. 1, рабочее доверие в разговоре
    frustration: float = 0.05   # 0 .. 1
    fatigue: float = 0.05       # 0 .. 1
    joy: float = 0.16           # 0 .. 1
    sadness: float = 0.04       # 0 .. 1
    anger: float = 0.02         # 0 .. 1
    surprise: float = 0.10      # 0 .. 1
    empathy: float = 0.48       # 0 .. 1

    def clamp(self) -> None:
        self.valence = max(-1.0, min(1.0, self.valence))
        for key in (
            "arousal", "curiosity", "trust", "frustration", "fatigue",
            "joy", "sadness", "anger", "surprise", "empathy",
        ):
            setattr(self, key, max(0.0, min(1.0, float(getattr(self, key)))))

    def appraise(self, text: str, event: str = "user_message") -> None:
        """Evaluate message cues; negated positive wording isn't treated as praise."""
        s = text.lower().replace("ё", "е")
        correction = _has_marker(s, CORRECTION_MARKERS)
        negative_preference = bool(re.search(r"\bне\s+(?:очень\s+)?(?:нравится|люблю|обожаю|доволен|довольна)\b", s))
        negated_praise = bool(re.search(r"\bне\s+(?:круто|классно|хорошо|отлично)\b", s))
        positive_hits = sum(1 for marker in POSITIVE if marker in s)
        if negative_preference or negated_praise:
            positive_hits = max(0, positive_hits - 1)
        negative_hits = sum(1 for marker in NEGATIVE if marker in s)
        sad_hits = sum(1 for marker in SADNESS_MARKERS if marker in s)
        anger_hits = sum(1 for marker in ANGER_MARKERS if marker in s)
        surprise_hits = sum(1 for marker in SURPRISE_MARKERS if marker in s)
        question = "?" in s or s.startswith(("почему", "как", "что", "зачем", "кто", "где", "когда"))
        exclamations = min(3, s.count("!"))

        if positive_hits:
            amount = min(positive_hits, 3)
            self.valence += 0.07 * amount
            self.joy += 0.09 * amount
            self.trust += 0.025 * amount
            self.frustration -= 0.045 * amount
            self.anger -= 0.035 * amount
        if negative_hits:
            amount = min(negative_hits, 3)
            self.valence -= 0.07 * amount
            self.frustration += (0.035 if correction else 0.065) * amount
            if not correction:
                self.anger += 0.055 * min(anger_hits or negative_hits, 2)
        if sad_hits:
            self.valence -= 0.06 * min(sad_hits, 2)
            self.sadness += 0.10 * min(sad_hits, 2)
            self.empathy += 0.045
        if anger_hits:
            self.arousal += 0.08 * min(anger_hits, 2)
        if surprise_hits:
            self.surprise += 0.13 * min(surprise_hits, 2)
            self.curiosity += 0.04
        if question:
            self.curiosity += 0.055
        if exclamations:
            self.arousal += 0.04 * exclamations
            if positive_hits:
                self.joy += 0.035 * exclamations
            elif negative_hits:
                self.frustration += 0.025 * exclamations

        if correction:
            # Feedback is useful information; it should not feel like an attack by default.
            self.curiosity += 0.10
            self.empathy += 0.02
            self.frustration -= 0.025
        if event == "learned_fact":
            self.curiosity += 0.04
            self.joy += 0.035
            self.valence += 0.025
        elif event == "successful_recall":
            self.valence += 0.035
            self.joy += 0.045
            self.trust += 0.02
        elif event == "uncertain_answer":
            self.curiosity += 0.05
            self.surprise += 0.025
        elif event == "feedback_received":
            self.curiosity += 0.08
            self.frustration += 0.015
        self.clamp()

    def decay(self, amount: float = 0.06) -> None:
        """Move state gradually toward a neutral baseline, not an instant reset."""
        amount = max(0.0, min(1.0, amount))
        self.valence *= (1.0 - amount)
        for key, baseline in (
            ("arousal", 0.22), ("curiosity", 0.48), ("trust", 0.34),
            ("joy", 0.16), ("sadness", 0.04), ("anger", 0.02),
            ("surprise", 0.10), ("empathy", 0.48),
        ):
            current = getattr(self, key)
            setattr(self, key, current + (baseline - current) * amount)
        self.frustration = max(0.0, self.frustration * (1.0 - amount * 1.35))
        self.fatigue = min(1.0, self.fatigue + 0.0015)
        self.clamp()

    def tone(self, latest_message: str = "") -> str:
        """Choose a response style from persistent state and the current message.

        A strong signal in the current message is allowed to affect the next response
        immediately; stored state then decays gradually across later turns.
        """
        s = latest_message.lower().replace("ё", "е")
        negative_preference = bool(re.search(r"\bне\s+(?:очень\s+)?(?:нравится|люблю|обожаю|доволен|довольна)\b", s))
        negated_praise = bool(re.search(r"\bне\s+(?:круто|классно|хорошо|отлично)\b", s))
        if _has_marker(s, SADNESS_MARKERS):
            return "gentle"
        if _has_marker(s, ANGER_MARKERS) or (_has_marker(s, CORRECTION_MARKERS) and _has_marker(s, NEGATIVE)):
            return "careful"
        if not negative_preference and not negated_praise and _has_marker(s, POSITIVE):
            return "excited" if "!" in s or self.joy >= 0.26 else "warm"
        if self.frustration > 0.09 or self.anger > 0.06:
            return "careful"
        if self.sadness > 0.10:
            return "gentle"
        if self.joy > 0.22 and self.arousal > 0.27:
            return "excited"
        if self.curiosity > 0.65 or (s and ("?" in s or s.startswith(("почему", "как", "что", "зачем", "кто", "где")))):
            return "curious"
        if self.valence > 0.30 or self.trust > 0.55:
            return "warm"
        return "neutral"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, text: str | None) -> "EmotionState":
        """Load old or new state snapshots; unknown fields are ignored."""
        if not text:
            return cls()
        try:
            data = json.loads(text)
            obj = cls(**{k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__})
            obj.clamp()
            return obj
        except (ValueError, TypeError, json.JSONDecodeError):
            return cls()

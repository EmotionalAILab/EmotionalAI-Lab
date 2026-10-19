"""Простая динамика целей — это приоритеты агента, а не человеческие желания."""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass
class Goal:
    name: str
    priority: float
    reason: str

class GoalManager:
    def choose(self, emotion: dict, uncertainty: float = 0.5, recalled: bool = False) -> list[Goal]:
        goals = [
            Goal("понять собеседника", 0.55 + 0.30 * float(emotion.get("curiosity", 0.5)), "любопытство и неполный контекст"),
            Goal("сохранить полезный опыт", 0.42 + (0.20 if recalled else 0.0), "связать текущий диалог с памятью"),
            Goal("снизить неопределённость", 0.30 + 0.50 * max(0.0, min(1.0, uncertainty)), "не хватает уверенных сведений"),
            Goal("поддерживать спокойный диалог", 0.45 + 0.25 * float(emotion.get("frustration", 0.0)), "важно не отвечать резко при напряжении"),
        ]
        return sorted(goals, key=lambda g: g.priority, reverse=True)

    def format(self, emotion: dict, uncertainty: float = 0.5, recalled: bool = False) -> str:
        lines = [f"• {g.name} — приоритет {g.priority:.2f} ({g.reason})" for g in self.choose(emotion, uncertainty, recalled)]
        return "Текущие приоритеты агента:\n" + "\n".join(lines)

from __future__ import annotations
import sys
from pathlib import Path
from agent import EmotionalAgent

BANNER = r"""
╔══════════════════════════════════════════════╗
║          EmotionalAI Lab · v0.1.1          ║
║  Память • эмоции • знания • обучение         ║
╚══════════════════════════════════════════════╝
Команды: /состояние, /память, /история, /цели, /ядро, /выход
"""

def main() -> int:
    print(BANNER)
    agent = EmotionalAgent(Path(__file__).resolve().parent)
    print(f"{agent.name}: Привет! О чём поговорим?\n")
    try:
        while True:
            try:
                user_text = input("Ты: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{agent.name}: До встречи!")
                break
            if not user_text:
                continue
            answer = agent.respond(user_text)
            print(f"{agent.name}: {answer}\n")
            if user_text.lower().strip() in ("/выход", "/exit", "пока", "до свидания", "бай"):
                break
    finally:
        agent.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

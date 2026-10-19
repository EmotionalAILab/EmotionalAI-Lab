"""Встроенные процедуры автономного агента для простых проверяемых задач.

Это встроенные процедуры автономного ядра: здесь только детерминированные операции, которые лучше не
поручать генерации текста (например, арифметика и типовые примеры синтаксиса).
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any


_NUMBER_WORDS = {
    "ноль": 0, "нуль": 0, "один": 1, "одна": 1, "одно": 1, "одну": 1,
    "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
    "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100,
}

_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_UNARY_OPS = {ast.UAdd: lambda x: x, ast.USub: lambda x: -x}


def _eval_math_node(node: ast.AST, depth: int = 0) -> float | int:
    if depth > 24:
        raise ValueError("Слишком сложное выражение")
    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if abs(node.value) > 1e12:
            raise ValueError("Слишком большое число")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_math_node(node.left, depth + 1)
        right = _eval_math_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
            raise ValueError("Слишком большая степень")
        result = _BINARY_OPS[type(node.op)](left, right)
        if not isinstance(result, (int, float)) or not math.isfinite(float(result)) or abs(result) > 1e15:
            raise ValueError("Результат вышел за допустимые пределы")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_math_node(node.operand, depth + 1))
    raise ValueError("В выражении есть недопустимая операция")


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-10:
        return str(rounded)
    return f"{value:.10f}".rstrip("0").rstrip(".")


def calculate_message(text: str) -> str | None:
    """Calculate an arithmetic expression if one is clearly present in the message."""
    original = text.strip()
    low = original.lower().replace("ё", "е")
    if not re.search(r"\d", low) and not any(re.search(rf"\b{re.escape(w)}\b", low) for w in _NUMBER_WORDS):
        return None

    # Common Russian verbal operators and number words.
    low = re.sub(r"\b(?:сколько будет|посчитай|вычисли|реши|чему равно|найди значение)\b", " ", low)
    low = re.sub(r"\b(?:умножить|умножь|умножаем|помножить|помножь)\s+на\b", " * ", low)
    low = re.sub(r"\b(?:разделить|раздели|делить|поделить)\s+на\b", " / ", low)
    low = re.sub(r"\b(?:плюс|прибавить)\b", " + ", low)
    low = re.sub(r"\b(?:минус|вычесть)\b", " - ", low)
    low = re.sub(r"\b(?:умножить|умножь|помножить|помножь)\b", " * ", low)
    low = re.sub(r"\b(?:разделить|раздели|делить)\b", " / ", low)
    for word, number in sorted(_NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
        low = re.sub(rf"\b{re.escape(word)}\b", str(number), low)
    low = low.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    low = re.sub(r"(?<=\d)\s*[xх]\s*(?=\d)", " * ", low)
    low = re.sub(r"\^", "**", low)
    low = re.sub(r"(\d),(\d)", r"\1.\2", low)

    # Percent-of is common natural language and is less ambiguous as a separate pattern.
    percent = re.search(r"(-?\d+(?:\.\d+)?)\s*%\s*(?:от|of)\s*(-?\d+(?:\.\d+)?)", low)
    if percent:
        try:
            result = float(percent.group(1)) * float(percent.group(2)) / 100
            if math.isfinite(result):
                return f"{percent.group(1)}% от {percent.group(2)} = {_format_number(result)}."
        except Exception:
            return "Не получилось вычислить процент: проверь числа."

    # Extract a numeric expression, not arbitrary natural-language text.
    candidates = re.findall(r"[0-9.\s()+*/%\-]+", low)
    candidates = [re.sub(r"\s+", " ", item).strip(" .\t\n") for item in candidates]
    candidates = [c for c in candidates if re.search(r"\d", c) and re.search(r"[+*/%\-]", c)]
    if not candidates:
        return None
    # The longest expression is usually the one after the user's question prefix.
    expression = max(candidates, key=len).strip()
    if len(expression) > 100 or not re.fullmatch(r"[0-9.\s()+*/%\-]+", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_math_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return "Не получилось вычислить выражение. Проверь скобки, операторы и деление на ноль."
    display_expression = expression.replace("*", " × ").replace("/", " ÷ ")
    display_expression = re.sub(r"\s+", " ", display_expression).strip()
    return f"{display_expression} = {_format_number(value)}."


def code_help_answer(text: str) -> str:
    """Return a small, correct starter example for common code-help requests."""
    s = text.lower().replace("ё", "е")
    lang_python = any(x in s for x in ("python", "питон", "пайтон"))
    lang_lua = any(x in s for x in ("lua", "luau", "роблокс", "roblox"))
    lang_js = any(x in s for x in ("javascript", "js", "джavascript", "джаваскрипт"))
    loop = any(x in s for x in ("цикл", "for ", "while", "перебор", "повтор"))
    function = any(x in s for x in ("функци", "def ", "function"))
    condition = any(x in s for x in ("услови", "if ", "если"))

    if lang_python:
        if loop:
            return (
                "В Python цикл `for` перебирает элементы последовательности. Например, чтобы выполнить действие 5 раз:\n\n"
                "```python\nfor i in range(5):\n    print(i)\n```\n\n"
                "`range(5)` даёт числа 0, 1, 2, 3, 4; двоеточие начинает блок, а отступ показывает, какие строки входят в цикл. "
                "Если нужно перебрать список, можно написать `for item in items:`."
            )
        if function:
            return (
                "В Python функцию объявляют через `def`:\n\n```python\ndef greet(name):\n    return f\"Привет, {name}!\"\n\nprint(greet(\"Мир\"))\n```\n\n"
                "Параметр `name` принимает аргумент, `return` возвращает результат, а вызов функции запускает её код."
            )
        if condition:
            return (
                "Условие в Python записывается через `if`:\n\n```python\ntemperature = 28\nif temperature > 25:\n    print(\"Жарко\")\nelse:\n    print(\"Не жарко\")\n```\n\n"
                "После `if` идёт выражение, а двоеточие и отступы обозначают блок кода."
            )
        return "Напиши, что именно нужно сделать и на каком примере: цикл, функция, условие, файл или интерфейс. Я смогу дать готовый код и объяснить его."

    if lang_lua:
        if loop:
            return (
                "В Lua/Luau числовой цикл `for` выглядит так:\n\n```lua\nfor i = 1, 5 do\n    print(i)\nend\n```\n\n"
                "Он выведет числа от 1 до 5 включительно. В Roblox-скриптах Luau использует такой же синтаксис."
            )
        if function:
            return (
                "Функция в Lua/Luau объявляется так:\n\n```lua\nlocal function greet(name)\n    return \"Привет, \" .. name\nend\n\nprint(greet(\"Мир\"))\n```\n\n"
                "`..` соединяет строки, а `return` возвращает значение."
            )
        return "Уточни, что именно нужно сделать в Lua/Luau или Roblox Studio, и я приведу пример кода."

    if lang_js:
        if loop:
            return (
                "Пример цикла `for` в JavaScript:\n\n```javascript\nfor (let i = 0; i < 5; i++) {\n  console.log(i);\n}\n```\n\n"
                "Он выведет числа от 0 до 4. `i++` увеличивает счётчик после каждой итерации."
            )
        if function:
            return "```javascript\nfunction greet(name) {\n  return `Привет, ${name}!`;\n}\n\nconsole.log(greet('Мир'));\n```"
        return "Уточни задачу для JavaScript: цикл, функция, условие, работа с DOM или запрос к API."

    return "Для примера кода уточни язык — Python, Lua/Luau для Roblox, JavaScript или другой — и что именно код должен делать."


_TRANSLATIONS_EN_RU = {
    "hello": "привет", "hi": "привет", "hey": "привет", "good morning": "доброе утро",
    "good evening": "добрый вечер", "good night": "спокойной ночи", "goodbye": "до свидания",
    "bye": "пока", "thank you": "спасибо", "thanks": "спасибо", "please": "пожалуйста",
    "yes": "да", "no": "нет", "sorry": "извини", "computer": "компьютер",
    "processor": "процессор", "memory": "память", "random access memory": "оперативная память",
    "storage": "хранилище", "solid state drive": "твердотельный накопитель (SSD)",
    "game": "игра", "old games": "старые игры", "I love old games": "Я люблю старые игры",
    "blue": "синий", "red": "красный", "green": "зелёный", "cat": "кот", "dog": "собака",
    "robot": "робот", "artificial intelligence": "искусственный интеллект", "neural network": "нейросеть",
    "how are you": "как дела", "what is your name": "как тебя зовут", "my name is": "меня зовут",
    "I don't know": "я не знаю", "I understand": "я понимаю", "see you later": "до скорого",
}
_TRANSLATIONS_RU_EN = {
    "привет": "hello", "здравствуй": "hello", "доброе утро": "good morning", "добрый вечер": "good evening",
    "спокойной ночи": "good night", "пока": "bye", "до свидания": "goodbye", "спасибо": "thank you",
    "пожалуйста": "please", "да": "yes", "нет": "no", "извини": "sorry", "компьютер": "computer",
    "процессор": "processor", "память": "memory", "оперативная память": "random access memory",
    "игра": "game", "старые игры": "old games", "синий": "blue", "красный": "red", "зелёный": "green",
    "кот": "cat", "собака": "dog", "робот": "robot", "искусственный интеллект": "artificial intelligence",
    "нейросеть": "neural network", "как дела": "how are you", "как тебя зовут": "what is your name",
    "я не знаю": "I don't know", "я понимаю": "I understand", "до скорого": "see you later",
}


def translate_message(text: str) -> str | None:
    """Small offline phrasebook for basic translation requests."""
    original = text.strip()
    low = original.lower().replace("ё", "е")
    target = "ru" if any(x in low for x in ("на русский", "по-русски", "на русском")) else "en" if any(
        x in low for x in ("на английский", "по-английски", "на английском")
    ) else None
    match = re.search(r"^(?:переведи|перевести|перевод(?: слова)?|как переводится|что значит)\s+(.+?)\s*(?:на русский|на английский|по-русски|по-английски|на русском|на английском)?[?!. ]*$", original, re.I)
    if not match:
        return None
    phrase = match.group(1).strip(" \t\n\"'«»?.!")
    # Strip any target-language suffix captured as part of the phrase.
    phrase = re.sub(r"\s+(?:на русский|на английский|по-русски|по-английски|на русском|на английском)$", "", phrase, flags=re.I).strip()
    source = normalize_phrase(phrase)
    if not source:
        return "Напиши слово или фразу, которую нужно перевести."
    if target is None:
        target = "ru" if re.search(r"[a-z]", phrase, re.I) else "en"
    dictionary = _TRANSLATIONS_EN_RU if target == "ru" else _TRANSLATIONS_RU_EN
    if source in dictionary:
        translated = dictionary[source]
        language = "по-русски" if target == "ru" else "по-английски"
        return f"{phrase} — {translated} ({language})."
    return f"Встроенный словарик пока не знает перевод «{phrase}». Можно добавить нужную пару слов в локальные данные или переформулировать фразу."


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))


def creative_answer(text: str) -> str | None:
    """Generate a simple creative text using the built-in standalone engine."""
    s = text.strip()
    low = s.lower().replace("ё", "е")
    if any(p in low for p in ("анекдот", "расскажи шутку", "придумай шутку")):
        return (
            "Программист попросил компьютер сказать что-нибудь ободряющее.\n"
            "Компьютер ответил: «Не переживай — даже у меня иногда всё работает только после перезапуска»."
        )
    if any(p in low for p in ("стих", "стихотворение", "напиши песню", "сочини песню")):
        topic_match = re.search(r"(?:про|о|об)\s+(.+?)[.!?]*$", s, re.I)
        topic = topic_match.group(1).strip(" .!?,") if topic_match else "мечту"
        if len(topic) > 60:
            topic = topic[:60].rstrip()
        return (
            f"И стих этот — про {topic},\n"
            "Про путь сквозь тишину и свет.\n"
            "Пускай не всё легко даётся —\n"
            "У любопытства есть ответ."
        )
    if any(p in low for p in ("историю", "сказку", "рассказ")) and any(p in low for p in ("напиши", "сочини", "придумай", "расскажи")):
        topic_match = re.search(r"(?:про|о|об)\s+(.+?)[.!?]*$", s, re.I)
        topic = topic_match.group(1).strip(" .!?,") if topic_match else "маленького робота"
        return (
            f"Это история про {topic}. Однажды герой столкнулся с задачей, на которую никто не мог сразу ответить. "
            "Вместо того чтобы гадать, он проверил несколько идей и заметил важную мелочь, которую все пропускали. "
            "Решение оказалось не волшебным, а результатом терпения и любопытства."
        )
    return None

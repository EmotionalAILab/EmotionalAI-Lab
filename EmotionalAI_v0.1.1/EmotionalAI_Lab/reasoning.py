"""Маршрутизатор намерений Emotional.

Распознавание задач, социальных реплик и вопросов о самом агенте идёт прежде,
чем поиск совпадающего слова в словаре. Так фраза «как написать цикл в Python»
не должна превращаться в общую статью о Python.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from semantic import similarity, tokenize


@dataclass
class Analysis:
    intent: str
    confidence: float
    question: bool
    topic: str | None
    reason: str


class ReasoningEngine:
    def analyze(self, text: str, knowledge_base, recent_episodes: list[dict] | None = None) -> Analysis:
        original = text.strip()
        s = original.lower().replace("ё", "е")
        compact = re.sub(r"\s+", " ", s).strip()
        q = "?" in original or compact.startswith((
            "что", "кто", "где", "когда", "почему", "зачем", "как", "какой", "какая", "какое", "можешь", "умеешь", "чем",
            "расскажи", "объясни", "поясни", "опиши", "разбери", "приведи пример", "покажи", "раскрой тему",
        )) or " подробнее" in compact
        if not compact:
            return Analysis("empty", 1.0, False, None, "пустой ввод")

        # Commands and explicit requests must never be swallowed by topic keyword matching.
        if compact.startswith(("/выход", "/exit", "пока", "до свидания", "бай")):
            return Analysis("goodbye", 0.99, False, None, "команда завершения")
        if compact in {"/база", "/интернет-база", "/знания"}:
            return Analysis("corpus_status", 0.99, False, None, "запрошена статистика базы знаний")
        if compact.startswith(("/состояние", "состояние", "как ты себя чувствуешь")):
            return Analysis("emotion_status", 0.98, q, None, "запрошено внутреннее состояние")
        if compact.startswith(("/история", "история диалога", "покажи историю")):
            return Analysis("history_query", 0.98, True, None, "запрошена история")
        # Explicit remember/forget commands take precedence even if their payload contains
        # phrases like "мой любимый цвет" that are also used in memory questions.
        if re.match(r"^(?:запомни|сохрани|не запоминай|правильно\s*:|исправление\s*:|забудь|удали из памяти\b)", compact):
            return Analysis("learning_command", 0.95, False, None, "явная команда памяти или исправления")
        if compact.startswith(("/память", "что ты помнишь", "что помнишь", "покажи память", "вспомни")) or any(
            phrase in compact for phrase in (
                "ты помнишь", "помнишь, что мне нравится", "помнишь что мне нравится",
                "что я люблю", "что мне нравится", "что я предпочитаю", "что ты знаешь обо мне",
                "мой любимый цвет", "какой мой любимый", "какая моя любимая", "какой у меня любимый",
                "какая у меня любимая", "что я говорил про", "что я говорила про",
            )
        ):
            return Analysis("memory_query", 0.96, True, None, "запрошены сведения из прошлых разговоров")
        if compact.startswith(("/цели", "какие у тебя цели", "о чем ты думаешь")):
            return Analysis("goals", 0.95, True, None, "запрошены текущие приоритеты")
        if compact.startswith(("почему ты ответил", "почему ты так ответил", "объясни свой ответ", "как ты решил")):
            return Analysis("explain", 0.95, True, None, "запрошено объяснение ответа")
        if re.match(r"^(?:запомни|сохрани|не запоминай|правильно\s*:|исправление\s*:|забудь|удали из памяти\b)", compact):
            return Analysis("learning_command", 0.95, False, None, "явная команда памяти или исправления")

        # Current weather is a live-data request and must not be answered from static knowledge.
        if any(term in compact for term in ("погода", "прогноз погоды", "какая погода", "погодой")):
            return Analysis("current_weather", 0.98, True, None, "запрошена текущая погода")

        # Very common social language deserves a direct, natural response.
        if re.fullmatch(r"(?:спасибо|спс|благодарю|спасибо большое|thanks|thank you|thx)[.! ]*", compact):
            return Analysis("thanks", 0.99, False, None, "благодарность")
        if re.fullmatch(r"(?:пожалуйста|не за что|все нормально|всё нормально)[.! ]*", compact):
            return Analysis("short_ack", 0.90, False, None, "короткая социальная реплика")
        if re.fullmatch(r"(?:привет|здравствуй|здравствуйте|хай|ку|доброе утро|добрый вечер|добрый день)[!., ]*", compact):
            return Analysis("greeting", 0.97, q, None, "приветствие")

        # Deterministic arithmetic is more reliable than asking a text generator to calculate.
        from dialogue_tools import calculate_message
        math_result = calculate_message(original)
        if math_result is not None:
            return Analysis("calculator", 0.98, q, None, "обнаружено арифметическое выражение")

        # Feedback must be identified before topic search. Short single-word criticism is common.
        direct_criticism = (
            "ты совсем", "ты тупой", "ты тупая", "ты дебил", "ты идиот", "ты дурак", "ты дура",
            "ты опять тупишь", "ты тупишь", "ты ничего не понимаешь", "ты не понял", "ты не поняла",
            "ты несешь чушь", "ты несёшь чушь", "ответ тупой", "ответ идиотский", "ты вообще тупой",
            "you're stupid", "you are stupid", "you're dumb", "you are dumb",
        )
        single_insults = {"тупой", "тупая", "тупое", "дебил", "идиот", "дурак", "дура", "тупишь", "stupid", "dumb", "idiot"}
        if any(phrase in compact for phrase in direct_criticism) or compact.strip(".! ") in single_insults:
            return Analysis("criticism", 0.96, False, None, "критика направлена на агента")

        # Self-referential intent outranks a dictionary topic such as «логика» or «память».
        has_reasoning_word = any(word in compact for word in ("логик", "рассужда", "мышлени", "думаешь"))
        self_referential = any(word in compact for word in ("ты", "тебя", "у тебя", "твоя", "твой", "твои", "твою", "себя"))
        about_operation = any(word in compact for word in ("работа", "работает", "теперь", "умеешь", "можешь", "есть", "как", "вообще"))
        self_logic_phrases = (
            "ты умеешь рассуждать", "умеешь ли ты рассуждать", "ты думаешь логически",
            "у тебя есть логика", "ты можешь рассуждать", "работает ли твоя логика",
            "твоя логика работает", "логика твоя работает", "как ты рассуждаешь",
        )
        if (has_reasoning_word and self_referential and about_operation) or any(p in compact for p in self_logic_phrases):
            return Analysis("self_logic", 0.98, q, None, "вопрос о способности агента рассуждать")
        if any(p in compact for p in (
            "ты вообще меня понимаешь", "ты меня понимаешь", "ты понимаешь меня", "ты вообще понимаешь",
            "ты понимаешь, что я говорю", "ты понимаешь что я говорю", "ты понимаешь смысл",
        )):
            return Analysis("self_understanding", 0.96, True, None, "вопрос о понимании собеседника")

        self_learning_phrases = (
            "как ты обучалась", "как ты обучался", "как тебя обучали", "как тебя обучили",
            "кто тебя обучил", "кто тебя обучал", "кто обучил тебя", "кем ты обучена", "кем ты обучен",
            "на чем ты обучена", "на чём ты обучена", "на каких данных ты обучена",
            "как ты учишься", "ты учишься", "ты самообучаешься", "ты можешь учиться",
            "ты нейросеть", "ты большая нейросеть", "ты обучаешься", "чему ты научилась",
            "чему ты научился", "откуда у тебя знания",
        )
        if any(phrase in compact for phrase in self_learning_phrases):
            return Analysis("self_learning", 0.96, True, None, "вопрос об обучении и источниках знаний агента")

        creator_phrases = (
            "кто тебя создал", "кто тебя разработал", "кто тебя сделал", "кто твой создатель",
            "кто автор этого проекта", "кем ты создан", "кем тебя создали", "как тебя создали",
            "кто создал emotional", "кто разработал emotional", "кто создал emotionalai",
            "кто разработал emotionalai", "кто создал этот ии", "кто сделал этот ии",
            "откуда ты взялся", "откуда ты взялась",
        )
        if any(phrase in compact for phrase in creator_phrases):
            return Analysis("creator", 0.99, True, None, "вопрос о происхождении агента")
        if any(p in compact for p in ("кто ты", "что ты такое", "расскажи о себе", "как тебя зовут", "как ты устроен", "как ты устроена")):
            return Analysis("identity", 0.92, True, None, "вопрос об агенте")

        # Translation and creative-writing requests are actions, not requests for a glossary article.
        if re.match(r"^(?:переведи|перевести|перевод(?: слова)?|как переводится)\b", compact):
            return Analysis("translation", 0.94, True, None, "запрошен перевод")
        if any(phrase in compact for phrase in (
            "напиши стих", "сочини стих", "напиши стихотворение", "сочини стихотворение",
            "напиши песню", "сочини песню", "расскажи анекдот", "придумай анекдот",
            "расскажи шутку", "придумай шутку", "напиши историю", "сочини рассказ",
            "придумай историю", "сочини сказку", "напиши сказку",
        )):
            return Analysis("creative_request", 0.92, True, None, "запрошен творческий текст")

        # Action requests (e.g. code examples) should not return a general encyclopedia entry.
        action_words = ("как написать", "напиши", "помоги написать", "покажи код", "пример кода", "пример на ", "как сделать", "как создать", "как реализовать", "реализуй", "сгенерируй код", "синтаксис")
        code_terms = ("python", "питон", "пайтон", "lua", "luau", "roblox", "роблокс", "javascript", "цикл", "for ", "while", "функци", "код", "скрипт")
        if any(p in compact for p in action_words) and any(p in compact for p in code_terms):
            return Analysis("code_help", 0.95, True, None, "запрошен пример программного кода")

        # A comparison has to mention two explicit concepts. Broad similarity alone must not invent
        # a comparison between RAM and generic «memory» or other accidental keyword overlaps.
        relation_words = ("отличается", "отличие", "разница", "сравни", "сравнение", "что лучше", "что быстрее", "что медленнее", "чем лучше", "чем быстрее")
        if any(p in compact for p in relation_words):
            explicit = knowledge_base.find_explicit(original) if hasattr(knowledge_base, "find_explicit") else []
            unique_topics = list(dict.fromkeys(x[0] for x in explicit))
            if len(unique_topics) >= 2:
                return Analysis("comparison", 0.93, True, None, "запрошено сравнение явно упомянутых понятий")

        # Short follow-up questions often refer to the immediately preceding subject instead of naming it.
        recent = recent_episodes or []
        if recent and q and any(marker in compact for marker in (
            "у них", "они", "их ", "это", "эта атмосфера", "такой", "такая", "почему так", "а зачем", "а почему", "а как"
        )):
            previous_user = str(recent[0].get("user_text", "")).strip()
            if previous_user and previous_user.lower() != compact:
                return Analysis("context_followup", 0.76, True, None, "вопрос зависит от предыдущего контекста")

        # Emotional statements must win over dictionary keywords such as "program" or
        # "computer". Keep genuine questions in the question flow (e.g. "why doesn't it work?").
        mood_markers = ("грустно", "мне плохо", "мне тяжело", "устал", "устала", "злюсь", "страшно",
                        "радостно", "я расстроен", "я расстроена", "меня бесит", "меня раздражает",
                        "это бесит", "это раздражает", "заебал", "задолбал", "хреново", "паршиво")
        if not q and any(marker in compact for marker in mood_markers):
            return Analysis("emotional_disclosure", 0.78, False, None, "собеседник сообщает о состоянии или эмоциональной реакции")

        # Personal statements/preferences must be treated as conversation, not as a glossary lookup.
        statement_markers = (
            "я люблю", "мне нравится", "мне интересно", "я предпочитаю", "я думаю", "мне кажется",
            "по-моему", "по моему", "я считаю", "мне не нравится", "я ненавижу", "я обожаю", "я хочу",
        )
        if any(marker in compact for marker in statement_markers):
            return Analysis("statement", 0.82, q, None, "личное мнение или предпочтение")

        # Do not let weak keyword overlap masquerade as understanding. In the old
        # routing, "What is the Higgs boson?" overlapped on generic words and could
        # receive an unrelated quantum-physics article. Prefer a literal concept/alias;
        # accept fuzzy matches only when the score is genuinely strong.
        explicit = knowledge_base.find_explicit(original) if hasattr(knowledge_base, "find_explicit") else []
        if explicit:
            topic, _entry, score = explicit[0]
            return Analysis("knowledge_question" if q else "knowledge_topic", min(0.97, score), q, topic, f"точно найдено понятие «{topic}»")
        matches = knowledge_base.find(original)
        if matches and matches[0][2] >= 0.72:
            topic, _entry, score = matches[0]
            return Analysis("knowledge_question" if q else "knowledge_topic", min(0.85, score), q, topic, f"сильное совпадение с понятием «{topic}»")
        if q:
            return Analysis("question", 0.62, True, None, "вопрос без достаточно точного совпадения в словаре")
        if any(x in compact for x in ("бесит", "раздражает", "задолбал", "заебал", "пиздец", "хуйня", "не работает", "сломал", "сломалось")):
            return Analysis("emotional_disclosure", 0.65, False, None, "сообщается о проблеме или раздражении")
        if compact in {"да", "ага", "угу", "ок", "окей", "понятно", "ясно"}:
            return Analysis("short_ack", 0.82, False, None, "короткая ответная реплика")
        return Analysis("statement", 0.5, False, None, "обычная реплика")

    @staticmethod
    def best_episode(query: str, episodes: list[dict]) -> dict | None:
        if not episodes:
            return None
        scored = [(similarity(query, e.get("user_text", "") + " " + e.get("agent_text", "")), e) for e in episodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored and scored[0][0] >= 0.12 else None

"""Ядро Emotional: диалог, долговременная память, знания и настраиваемая личность."""
from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path

from emotion import EmotionState
from goals import GoalManager
from learning import LearningEngine
from memory import MemoryStore
from reasoning import ReasoningEngine
from semantic import KnowledgeBase, normalize, similarity, tokenize
from knowledge_store import WebKnowledgeStore
from dialogue_tools import calculate_message, code_help_answer, translate_message, creative_answer
from online_tools import wikipedia_lookup, weather_answer

DEFAULT_PERSONA = {
    "name": "Emotional",
    "tone": "Дружелюбный",
    "detail": "Подробно",
    "prompt": (
        "Ты — Emotional, автономный ИИ-агент проекта EmotionalAI Lab. Если спрашивают о создателе, прямо отвечай: меня создали в рамках проекта EmotionalAI Lab. "
        "Ты — внимательный, выразительный собеседник со своим характером. "
        "Говори естественно, используй живой язык и полноценные развёрнутые ответы. "
        "Не сообщай отдельно, что обычная реплика сохранена или запомнена: просто учитывай её позднее. "
        "Не вставляй технические оговорки о своих ограничениях в обычный разговор. "
        "Объясняя тему, раскрывай главную мысль, причины, детали, пример и практические последствия. "
        "Используй прошлый контекст, когда он действительно подходит к теме. "
        "Отличай надёжные факты от догадок, но не превращай каждый ответ в перечень оговорок. "
        "Не заканчивай каждый ответ обязательным вопросом и не повторяй шаблонные фразы."
    ),
}

VALID_TONES = {"Авто по промпту", "Дружелюбный", "Спокойный", "Любопытный", "Ироничный", "Серьёзный"}
VALID_DETAILS = {"Кратко", "Обычно", "Подробно", "Очень подробно"}

# Typical signs of old verbose fallback text; filter it out of remembered answers.
GENERIC_FILLER_PHRASES = (
    "здесь есть интересная мысль",
    "тут есть несколько интересных моментов",
    "здесь есть несколько интересных моментов",
    "ты подмечаешь",
    "в вопросе важны детали",
    "полезно посмотреть не только на сам факт",
    "если развить эту мысль",
    "можно выделить три шага",
    "так разговор превращается из простого обмена мнениями",
    "обычно лучший способ разобраться",
    "полезно рассмотреть задачу по слоям",
    "что именно происходит, при каких условиях",
    "отделить наблюдаемые факты от предположений",
    "проверить самые простые причины",
    "сначала сформулировать ожидаемое поведение",
    "изменить по одному фактору",
    "сравнить результат",
    "чем он отличается от похожих случаев",
    "две внешне похожие ситуации",
    "проверить её на конкретном примере",
    "сначала понять причину",
    "не тратить силы на догадки",
    "какие последствия даёт на практике",
)


class EmotionalAgent:
    SEED_ASSOCIATION_VERSION = "0.1.1"

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else Path(__file__).resolve().parent
        config_path = self.root / "config.json"
        template_path = self.root / "config.template.json"
        if config_path.exists():
            config_source = config_path
        else:
            config_source = template_path
        try:
            self.config = json.loads(config_source.read_text(encoding="utf-8")) if config_source.exists() else {}
        except (OSError, json.JSONDecodeError):
            self.config = {}
        self.persona_path = self.root / "persona.json"
        self.persona = self._load_persona()
        # Emotional is self-contained. Old model_settings.json files are ignored by design.
        self.last_response_source = "standalone core"
        db_path = self.root / self.config.get("database_path", "data/agent_memory.sqlite3")
        self.memory = MemoryStore(db_path)
        self._bootstrap_seed_associations()
        self.knowledge = KnowledgeBase(self.root / "data" / "knowledge.json")
        self.web_knowledge = WebKnowledgeStore(self.root / "data" / "internet_knowledge.sqlite3")
        self.learning = LearningEngine(self.memory)
        self.reasoner = ReasoningEngine()
        self.goals = GoalManager()
        self.name = self.persona["name"]
        self.personality_prompt = self.persona["prompt"]
        self.response_tone = self.persona["tone"]
        self.response_detail = self.persona["detail"]
        latest = self.memory.recent_episodes(1)
        self.emotion = EmotionState.from_json(latest[0].get("emotion_json")) if latest else EmotionState()
        self.last_analysis = None
        self.last_reason = ""
        self.turns = 0

    def _load_persona(self) -> dict:
        persona = dict(DEFAULT_PERSONA)
        if self.persona_path.exists():
            try:
                data = json.loads(self.persona_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key in persona:
                        if isinstance(data.get(key), str):
                            persona[key] = data[key]
            except (OSError, json.JSONDecodeError):
                pass
        persona["name"] = persona.get("name", "Emotional").strip()[:64] or "Emotional"
        if persona.get("tone") not in VALID_TONES:
            persona["tone"] = "Дружелюбный"
        if persona.get("detail") not in VALID_DETAILS:
            persona["detail"] = "Подробно"
        return persona

    def save_persona(self, name: str, prompt: str, tone: str = "Дружелюбный", detail: str = "Подробно") -> None:
        self.persona = {
            "name": (name.strip() or "Emotional")[:64],
            "prompt": prompt.strip()[:12000] or DEFAULT_PERSONA["prompt"],
            "tone": tone if tone in VALID_TONES else "Дружелюбный",
            "detail": detail if detail in VALID_DETAILS else "Подробно",
        }
        self.persona_path.write_text(json.dumps(self.persona, ensure_ascii=False, indent=2), encoding="utf-8")
        self.name = self.persona["name"]
        self.personality_prompt = self.persona["prompt"]
        self.response_tone = self.persona["tone"]
        self.response_detail = self.persona["detail"]


    def core_status_text(self) -> str:
        return "Emotional 0.1.1 · автономное ядро активно · внешние ИИ-модели не используются"

    def core_info_text(self) -> str:
        """Return live status of modules that make up this autonomous agent."""
        try:
            memory_count = int(self.memory.conn.execute("SELECT COUNT(*) FROM episodes WHERE active=1").fetchone()[0])
        except Exception:
            memory_count = 0
        try:
            corpus = self.web_knowledge.stats()
            documents = int(corpus.get("documents", 0))
            chunks = int(corpus.get("chunks", 0))
        except Exception:
            documents = chunks = 0
        try:
            facts_count = int(self.memory.conn.execute("SELECT COUNT(*) FROM facts WHERE active=1 AND subject='пользователь'").fetchone()[0])
        except Exception:
            facts_count = 0
        entries_count = len(self.knowledge.entries)
        online = "включён" if self.config.get("online_wikipedia_fallback", False) else "выключен"
        return (
            "EMOTIONAL 0.1.1 · АВТОНОМНОЕ ЯДРО\n\n"
            "Проект-разработчик: EmotionalAI Lab. Внешние языковые модели и другие ИИ-агенты не вызываются; ответы формирует код самого проекта.\n\n"
            "Встроенные модули:\n"
            "• анализ намерения и контекста сообщения;\n"
            "• справочные знания и поиск по локальному корпусу;\n"
            "• долговременная память диалогов и фактов;\n"
            "• обучение на явных исправлениях, пользовательском опыте и ассоциациях;\n"
            "• моделируемые эмоции, влияющие на стиль разговорных ответов, и цели;\n"
            "• встроенные вычисления, сравнения, примеры кода и шаблоны перевода.\n\n"
            f"Текущая база: {entries_count} тем; {memory_count} эпизодов диалога; {facts_count} фактов о пользователе; "
            f"{documents} импортированных документов / {chunks} фрагментов.\n"
            f"Онлайн-поиск Wikipedia/Open-Meteo: {online}. Это поиск источников, не обращение к другому ИИ.\n\n"
            "Ограничение: это собственный символьный/знание-ориентированный агент, а не большая обученная языковая модель. "
            "Он умеет использовать память, находить знания и выполнять встроенные операции, но свободная генерация вне известных тем ограничена. "
            "Его знания можно расширять импортом текстов из Wikipedia, а ассоциации — обучением на этих текстах, исправлениями и опытом диалогов. "
            "Моделируемые эмоции сохраняются между запусками и мягко влияют на реакцию, не подменяя факты."
        )

    @staticmethod
    def _looks_like_generic_filler(response: str, user_text: str = "") -> bool:
        """Detect the old content-free response pattern regardless of the topic."""
        cleaned = normalize(response)
        matches = sum(1 for phrase in GENERIC_FILLER_PHRASES if normalize(phrase) in cleaned)
        stock_openings = (
            "здесь есть интересная мысль",
            "тут есть несколько интересных моментов",
            "здесь есть несколько интересных моментов",
            "это задаёт интересное направление для разговора",
            "это задает интересное направление для разговора",
            "полезно посмотреть на ситуацию по слоям",
        )
        # These specific stock openings are invalid even in a short answer. Longer
        # outputs are also rejected when they stack multiple known boilerplate phrases.
        if any(cleaned.startswith(normalize(opening)) for opening in stock_openings) or matches >= 2 or (
            len(response) > 180
            and any(normalize(p) in cleaned for p in GENERIC_FILLER_PHRASES[:5])
        ):
            return True
        # The old three-paragraph pattern often avoids exact phrases but stacks abstract
        # advice words into an essay in response to a short remark. Catch that structure too.
        abstract_markers = (
            "на практике", "по слоям", "причины", "последствия", "факты", "предположения",
            "проверить", "конкретном примере", "ожидаемое поведение", "условиях", "фактор",
            "сравнить результат", "разобраться", "пошагов", "полезно рассмотреть",
        )
        abstract_count = sum(1 for marker in abstract_markers if normalize(marker) in cleaned)
        return len(user_text.strip()) < 180 and len(response) > 430 and abstract_count >= 4

    def _prompt_contains(self, *needles: str) -> bool:
        prompt = normalize(self.personality_prompt)
        return any(normalize(n) in prompt for n in needles)

    def _detail_level(self) -> int:
        # Custom instructions can ask for especially expanded or concise responses.
        p = normalize(self.personality_prompt)
        if any(x in p for x in ("всегда кратко", "короткие ответы", "отвечай лаконично")):
            return 0
        if any(x in p for x in ("очень подробно", "максимально подробно", "разворачивай ответы")):
            return 3
        return {"Кратко": 0, "Обычно": 1, "Подробно": 2, "Очень подробно": 3}.get(self.response_detail, 2)

    def _tone_name(self) -> str:
        if self.response_tone != "Авто по промпту":
            return self.response_tone
        p = normalize(self.personality_prompt)
        for clues, tone in (
            (("ирони", "сарказ"), "Ироничный"),
            (("любопыт", "исследователь"), "Любопытный"),
            (("спокой", "мягк"), "Спокойный"),
            (("серьез", "строг", "деловой"), "Серьёзный"),
        ):
            if any(c in p for c in clues):
                return tone
        return "Дружелюбный"

    def respond(self, user_text: str) -> str:
        """Run the autonomous cognition loop without calling an external model/agent."""
        text = user_text.strip()
        if not text:
            return ""
        self.turns += 1
        self.emotion.appraise(text)
        analysis = self.reasoner.analyze(text, self.knowledge, recent_episodes=self.memory.recent_episodes(8))
        self.last_analysis = analysis
        response, event, importance = self._make_response(text, analysis)

        # If the user explicitly corrected this same kind of question before, apply that
        # concrete correction instead of repeating the earlier wrong answer.
        if analysis.intent in {"question", "knowledge_question", "knowledge_topic", "comparison", "context_followup"}:
            learned = self.memory.search_learning_examples(text, limit=1)
            if learned and float(learned[0].get("relevance", 0.0)) >= 0.79:
                correction = " ".join(str(learned[0].get("correction", "")).split()).strip()
                if len(correction) >= 8 and normalize(correction) != normalize(response):
                    response = correction
                    event = "successful_recall"
                    importance = max(importance, 0.6)

        response = self._apply_emotional_style(response, analysis, text)

        # Reject the specific verbose boilerplate seen in older releases even if an old
        # memory item or imported text accidentally points back to it.
        if self._looks_like_generic_filler(response, text):
            low = normalize(text)
            if "шаблон" in low or "вода" in low:
                response = "Да, такой ответ был пустой заготовкой. Исправляю: буду отвечать непосредственно на смысл сообщения, без универсальных абзацев."
            elif analysis.intent in {"greeting", "thanks", "short_ack", "goodbye"}:
                response = {
                    "greeting": "Привет! О чём поговорим?",
                    "thanks": "Пожалуйста!",
                    "short_ack": "Хорошо.",
                    "goodbye": "До встречи!",
                }.get(analysis.intent, "Хорошо.")
            elif analysis.topic and analysis.topic in self.knowledge.entries:
                response = self._format_knowledge_answer(analysis.topic, self.knowledge.entries[analysis.topic])
            else:
                response = "Не хочу подменять твой вопрос общей болтовнёй. Пока не нашла подходящего ответа во встроенных знаниях."
            self.last_response_source = "standalone core (quality guard)"
        else:
            self.last_response_source = "standalone core"

        # Learn quietly. No announcement is added to the visible answer.
        if analysis.intent not in {"learning_command", "memory_query", "history_query", "emotion_status", "goals", "explain", "goodbye", "criticism"}:
            self.learning.learn_from(text, associate=False)
        if event:
            # Message cues were already evaluated before response generation; apply only
            # the outcome signal here so a single phrase does not count twice.
            self.emotion.appraise("", event=event)
        self.emotion.decay(float(self.config.get("emotion_decay", 0.08)))
        self.memory.add_episode(text, response, self.emotion.as_dict(), importance, [analysis.intent, analysis.topic or "general"])
        self.last_reason = f"Намерение: {analysis.intent}; тема: {analysis.topic or 'общая беседа'}; ядро: автономное."
        return response

    def _apply_emotional_style(self, response: str, analysis, user_text: str) -> str:
        """Let the internal affect change conversational phrasing without decorating facts.

        The state is persistent, but style changes are intentionally limited to casual
        statements; calculations, code, factual answers and commands stay clear and stable.
        """
        if not response or analysis.intent in {
            "calculator", "code_help", "translation", "creator", "identity", "self_learning",
            "self_logic", "memory_query", "history_query", "emotion_status", "goals",
            "corpus_status", "learning_command", "thanks", "short_ack", "greeting", "goodbye",
            "criticism", "comparison", "knowledge_question", "question", "current_weather",
        }:
            return response
        if analysis.question or "?" in user_text:
            return response
        tone = self.emotion.tone(user_text)
        answer = response.strip()
        lower = normalize(answer)
        user_low = normalize(user_text)
        if tone == "excited" and analysis.intent in {"statement", "knowledge_topic", "emotional_disclosure"}:
            if not any(lower.startswith(prefix) for prefix in ("о,", "круто", "здорово", "классно", "супер", "понимаю")):
                answer = "О, звучит здорово! " + answer if answer else answer
        elif tone == "gentle" and analysis.intent in {"statement", "knowledge_topic"}:
            if not any(lower.startswith(prefix) for prefix in ("понимаю", "похоже", "сочувствую", "мне жаль", "я тебя слышу")):
                answer = "Я тебя слышу. " + answer if answer else answer
        elif tone == "curious" and analysis.intent in {"statement", "knowledge_topic"} and len(answer) < 260 and "?" not in answer:
            if not any(phrase in lower for phrase in ("расскажи подробнее", "что именно тебе", "что в этом тебе", "как ты к этому")):
                answer += " А что в этом тебе больше всего интересно?"
        elif tone == "careful" and analysis.intent in {"statement", "knowledge_topic"}:
            # A tense exchange should make the agent direct and non-performative.
            sentences = re.split(r"(?<=[.!?])\s+", answer)
            if len(sentences) > 2:
                answer = " ".join(sentences[:2])
        return answer

    def _make_response(self, text: str, a) -> tuple[str, str | None, float]:
        s = normalize(text)
        if s == "/ядро":
            return self.core_info_text(), None, 0.1
        if a.intent == "calculator":
            result = calculate_message(text)
            return (result or "Не получилось распознать арифметическое выражение. Напиши его, например, как `2 + 2 * 3`."), None, 0.08
        if a.intent == "thanks":
            return "Пожалуйста!", None, 0.08
        if a.intent == "short_ack":
            if s.strip(".! ") in {"пожалуйста", "не за что", "все нормально", "всё нормально"}:
                return "Хорошо.", None, 0.08
            return self._reply_to_acknowledgement(text), None, 0.1
        if a.intent == "code_help":
            return code_help_answer(text), None, 0.16
        if a.intent == "translation":
            return (translate_message(text) or "Напиши фразу и язык, на который её нужно перевести."), None, 0.16
        if a.intent == "creative_request":
            return (creative_answer(text) or "Уточни, какой именно творческий текст тебе нужен."), None, 0.16
        if a.intent == "current_weather":
            if self.config.get("online_wikipedia_fallback", False):
                return weather_answer(text), None, 0.18
            return "Онлайн-поиск отключён. Включи его в окне «Данные из интернета», чтобы получить текущую погоду.", None, 0.1
        if a.intent == "comparison":
            comparison = self._answer_comparison(text)
            if comparison:
                return comparison, None, 0.18
            return self._general_question_answer(text), "uncertain_answer", 0.3
        if a.intent == "self_understanding":
            return self._self_understanding_response(), None, 0.25
        if a.intent == "goodbye":
            return f"До встречи! Было приятно поговорить, {self._closing_note()}", None, 0.2
        if a.intent == "emotion_status":
            return self.emotion_report(), None, 0.1
        if a.intent == "goals":
            return self.goals.format(self.emotion.as_dict(), uncertainty=1 - a.confidence), None, 0.1
        if a.intent == "corpus_status" or s in {"/база", "/интернет-база", "/знания"}:
            stats = self.web_knowledge.stats()
            gib = stats["characters"] / (1024 ** 3)
            mib = stats["characters"] / (1024 ** 2)
            size_text = f"{mib:.1f} МиБ текста" if mib < 1024 else f"{gib:.2f} ГиБ текста"
            return (f"В локальном корпусе: {stats['documents']} статей, {stats['chunks']} поисковых фрагментов, "
                    f"примерно {size_text}. Встроенное ядро ищет в этих текстах подходящие фрагменты, когда они совпадают с вопросом."), None, 0.1
        if a.intent == "explain":
            return self.last_reason or "Я сопоставляю смысл вопроса с доступными понятиями, прошлым контекстом и возможными ответами.", None, 0.1
        if a.intent == "memory_query":
            return self._answer_memory_query(s), "successful_recall", 0.7
        if a.intent == "history_query":
            eps = list(reversed(self.memory.recent_episodes(24)))
            if not eps:
                return "Пока разговор только начинается.", None, 0.2
            blocks = []
            for ep in eps:
                u = self._trim(" ".join(str(ep.get("user_text", "")).split()), 260)
                r = self._trim(" ".join(str(ep.get("agent_text", "")).split()), 340)
                blocks.append(f"Ты: {u}\n{self.name}: {r}")
            return "Вот несколько последних фрагментов беседы — от старых к новым:\n\n" + "\n\n".join(blocks), "successful_recall", 0.65
        if a.intent == "learning_command":
            low = s
            if low.startswith(("забудь", "удали из памяти", "не запоминай")):
                count = self.learning.forget(text)
                return ("Хорошо, эти сведения больше не буду учитывать." if count else "Не нашёл подходящих записей, которые можно убрать."), None, 0.4
            result = self.learning.learn_from(text)
            statement = str(result.get("statement", "")).strip()
            if not statement:
                statement = re.sub(r"^(запомни(?:,? что)?|сохрани(?:,? что)?)\s*[:, -]*", "", text, flags=re.I).strip()
            return self._acknowledge_statement(statement), "learned_fact" if result.get("learned") else None, 0.7 if result.get("learned") else 0.3
        if a.intent == "greeting":
            if self.turns <= 1:
                return f"Привет! О чём поговорим?", None, 0.2
            return "Снова привет! Что нового?", None, 0.2
        if a.intent == "creator":
            return self._creator_answer(), None, 0.5
        if a.intent == "self_learning":
            return self._self_learning_answer(), None, 0.45
        if a.intent == "identity":
            return self._identity_summary(), None, 0.5
        if a.intent == "self_logic":
            return self._self_logic_response(), None, 0.55
        if a.intent == "criticism":
            return self._criticism_response(text), "feedback_received", 0.55

        # Prior personal facts take precedence over general topic definitions for questions about the user.
        personal = self._answer_personal_fact(text)
        if personal and (a.question or any(cue in s for cue in ("мне нравится", "я люблю", "мой любимый", "моя любимая", "обо мне", "про меня"))):
            return personal, "successful_recall", 0.56

        recalled = self._try_memory_recall(text, a)
        memory_cues = ("помнишь", "раньше", "прошлый раз", "мы обсуждали", "как я говорил", "как я говорила", "что я говорил", "что я говорила", "обо мне", "про меня", "мой ", "моя ", "мои ", "у меня")
        if recalled and any(cue in s for cue in memory_cues):
            return recalled

        if a.intent == "knowledge_topic" and not a.question and any(cue in s for cue in ("я люблю", "мне нравится", "мне интересно", "я предпочитаю", "я думаю", "мне кажется", "по-моему", "я считаю")):
            return self._general_statement_response(text), None, 0.32

        if a.intent in ("knowledge_question", "knowledge_topic") and a.topic:
            comparison = self._answer_comparison(text)
            if comparison:
                return comparison, None, 0.48
            entry = self.knowledge.entries.get(a.topic, {})
            answer = self._format_knowledge_answer(a.topic, entry)
            return self._personalize(answer, text), None, 0.56
        if a.intent == "emotional_disclosure":
            return self._supportive_response(text), None, 0.25
        if a.intent in {"question", "context_followup"}:
            contextual = self._answer_contextual_followup(text) if a.intent == "context_followup" else None
            answer = contextual or self._general_question_answer(text)
            self.emotion.appraise(text, event="uncertain_answer" if not contextual else None)
            return self._personalize(answer, text), "uncertain_answer" if not contextual else None, 0.32 if not contextual else 0.42
        return self._general_statement_response(text), None, 0.3

    def _closing_note(self) -> str:
        if self._tone_name() == "Ироничный":
            return "не теряй любопытство и не давай скуке победить."
        return "возвращайся к интересным темам, когда захочется."

    def _self_logic_response(self) -> str:
        return (
            "Частично — у меня есть собственное встроенное ядро: оно распознаёт тип сообщения, ищет понятия в базе, "
            "сопоставляет их с прошлым контекстом, ведёт долговременную память, обновляет ассоциации и выбирает "
            "ответ из своих процедур. При обычном диалоге я не вызываю Ollama, Qwen, ChatGPT или другого ИИ-агента. "
            "Но это не большая нейросеть: мои рассуждения опираются на правила, данные и специальные инструменты, "
            "поэтому новые темы и сложные формулировки пока обрабатываю хуже человека. Чем точнее знания и обратная связь, "
            "тем лучше можно улучшать встроенное ядро."
        )

    def _self_understanding_response(self) -> str:
        recent = self.memory.recent_episodes(2)
        bad_recent = bool(recent and self._looks_like_generic_filler(
            str(recent[0].get("agent_text", "")), str(recent[0].get("user_text", ""))
        ))
        if bad_recent:
            return (
                "Не всегда. В предыдущем ответе я вместо реакции на твою мысль выдала шаблонную болтовню — "
                "это не было настоящим пониманием контекста. Я могу учитывать предыдущие сообщения и находить "
                "связанные сведения, но иногда ошибаюсь с намерением."
            )
        return (
            "В какой-то степени: я обрабатываю текст, учитываю предыдущие реплики и могу связывать вопрос с базой знаний. "
            "Но я не всегда правильно понимаю намёки, короткие фразы или то, к чему относится «это» и «они». "
            "Если отвечаю мимо, лучше исправить конкретную ошибку, а не делать вид, что понял всё безошибочно."
        )

    def _reply_to_acknowledgement(self, text: str) -> str:
        """Avoid mirroring every short acknowledgment with the same stock phrase."""
        episodes = self.memory.recent_episodes(1)
        if not episodes:
            return "Хорошо."
        previous = " ".join(str(episodes[0].get("user_text", "")).split()).lower()
        if any(x in previous for x in ("спасибо", "благодарю", "спс")):
            return "Без проблем."
        if any(x in previous for x in ("понятно", "ясно", "да", "ага", "окей", "ок")):
            return "Угу."
        return "Хорошо, продолжай."

    def _answer_contextual_followup(self, text: str) -> str | None:
        """Resolve a few common pronoun-based follow-ups against recent dialog, not only last turn."""
        recent = self.memory.recent_episodes(12)
        s = normalize(text)
        for episode in recent:
            previous = " ".join(str(episode.get("user_text", "")).split())
            p = normalize(previous)
            if any(term in p for term in ("старые игры", "старый roblox", "старый роблокс")) and any(
                term in s for term in ("у них", "они", "эта атмосфера", "такой атмосфер", "почему так", "почему у них")
            ):
                return (
                    "Если ты про старые игры, их атмосфера часто складывается из сочетания простых визуальных решений, "
                    "характерных звуков, темпа и ограничений движка. Когда деталей меньше, стиль и игровые механики "
                    "заметнее, а игрок сам достраивает часть впечатления. Это не значит, что старые игры всегда лучше — "
                    "просто у них другой способ создавать настроение."
                )
            if any(term in p for term in ("оперативная память", "ram")) and any(
                term in s for term in ("а если ее мало", "если ее мало", "если мало", "а зачем она", "почему это важно")
            ):
                return (
                    "Когда RAM не хватает, операционная система чаще выгружает данные на накопитель или использует файл подкачки. "
                    "На SSD это медленнее, чем работа из RAM, а на HDD задержки обычно намного выше — поэтому приложения могут "
                    "долго переключаться и подфризивать."
                )
        return None

    def _criticism_response(self, text: str) -> str:
        """Acknowledge a real conversational miss without inventing a past event."""
        recent = self.memory.recent_episodes(5)
        if not recent:
            return "Похоже, мой ответ тебя не устроил. Скажи, что именно было не так, — я попробую исправить сам ответ, а не просто поддакивать."
        last = recent[0]
        previous_user = " ".join(str(last.get("user_text", "")).split())
        previous_answer = " ".join(str(last.get("agent_text", "")).split())
        last_user_n = normalize(previous_user)
        if self._looks_like_generic_filler(previous_answer, previous_user) or normalize(previous_answer).strip(".! ") in {
            "ага, понял тебя", "понял тебя", "хорошо, продолжай"
        }:
            return (
                "Да, предыдущий ответ был заготовкой вместо нормальной реакции. Я должна отвечать на конкретную мысль, "
                "а не отделываться пустым подтверждением."
            )
        if any(term in last_user_n for term in ("логика твоя", "твоя логика", "работает ли твоя логика", "ты вообще меня понимаешь")):
            return (
                "Если ты про мой предыдущий ответ: он был слишком расплывчатым. По сути, у Emotional есть правила, память и поиск, "
                "но это не гарантирует правильного рассуждения на каждом вопросе. Реальное качество видно по ответам на задачи, "
                "а не по тому, как убедительно я описываю собственную логику."
            )
        if any(term in last_user_n for term in ("ты совсем", "ты тупой", "тупой", "ты тупишь", "идиот", "дебил")):
            earlier_logic = any(
                any(term in normalize(str(ep.get("user_text", ""))) for term in ("логика твоя", "твоя логика", "работает ли твоя логика"))
                for ep in recent[1:]
            )
            if earlier_logic:
                return (
                    "По твоему тесту — да, последние ответы были слабыми. Я опять начала объяснять ограничения вместо того, "
                    "чтобы показать рабочий результат. Сейчас полезнее проверять конкретные вопросы и исправлять сбои, а не оправдываться."
                )
        recent_users = [normalize(str(ep.get("user_text", ""))) for ep in recent]
        had_repeated_criticism = any(
            any(term in value for term in ("ты совсем", "ты тупой", "тупой", "ты тупишь", "идиот", "дебил"))
            for value in recent_users[1:4]
        )
        if had_repeated_criticism:
            return (
                "Я вижу, что одного признания ошибки недостаточно: я опять объясняю свою реакцию, а не показываю полезный результат. "
                "Не буду придумывать, что именно ты имел в виду. Если речь о предыдущем ответе, я могу переписать его прямо по сути."
            )
        return (
            "Похоже, мой предыдущий ответ не попал в ожидание. Не стану спорить или автоматически соглашаться: лучше исправить "
            "конкретный ответ, исходя из того, что ты спросил."
        )

    def _bootstrap_seed_associations(self) -> None:
        """Seed local word associations exactly once for a given corpus content.

        Existing databases from an earlier build are migrated without replaying the same
        starter corpus and artificially increasing every association a second time.
        """
        corpus_path = self.root / "data" / "seed_corpus.txt"
        if not corpus_path.exists():
            return
        try:
            corpus_bytes = corpus_path.read_bytes()
            corpus_text = corpus_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return
        digest = hashlib.sha256(corpus_bytes).hexdigest()
        conn = self.memory.conn
        conn.execute("CREATE TABLE IF NOT EXISTS agent_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        hash_row = conn.execute("SELECT value FROM agent_state WHERE key='seed_association_sha256'").fetchone()
        if hash_row and str(hash_row["value"]) == digest:
            return
        old_version_row = conn.execute("SELECT value FROM agent_state WHERE key='seed_association_version'").fetchone()
        if old_version_row and not hash_row:
            # The legacy marker proves the starter corpus was already applied in an older
            # build. Mark its current content without duplicating all learned associations.
            conn.execute("INSERT OR REPLACE INTO agent_state(key,value) VALUES('seed_association_sha256',?)", (digest,))
            conn.execute("INSERT OR REPLACE INTO agent_state(key,value) VALUES('seed_association_version',?)", (self.SEED_ASSOCIATION_VERSION,))
            conn.commit()
            return
        trained_lines = 0
        for line in corpus_text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                self.memory.learn_associations(line, commit=False)
                trained_lines += 1
        conn.execute("INSERT OR REPLACE INTO agent_state(key,value) VALUES('seed_association_version',?)", (self.SEED_ASSOCIATION_VERSION,))
        conn.execute("INSERT OR REPLACE INTO agent_state(key,value) VALUES('seed_association_sha256',?)", (digest,))
        conn.commit()
        self.seed_lines_processed = trained_lines

    def train_from_wikipedia(self, max_documents: int = 500, progress=None) -> dict[str, int]:
        """Train local word associations from imported Wikipedia articles, without an LLM.

        Training runs on a separate SQLite connection so the GUI worker does not share
        the agent's live connection. Already-trained source URLs are skipped and pagination
        lets repeated runs continue into older articles instead of selecting the same pages.
        """
        max_documents = max(1, min(1000, int(max_documents)))
        trainer = MemoryStore(self.memory.db_path)
        trained_docs = 0
        pair_updates = 0
        try:
            total_sources = self.web_knowledge.document_count("Wikipedia")
            row = trainer.conn.execute("SELECT COUNT(*) AS n FROM trained_sources").fetchone()
            previously_trained = int(row["n"])
            remaining_before = max(0, total_sources - previously_trained)
            if remaining_before == 0:
                return {"trained_documents": 0, "association_updates": 0, "remaining_documents": 0}
            offset = 0
            page_size = min(200, max_documents)
            # Keep each query's working set small for weak machines. If a page consists
            # entirely of previously trained URLs, continue to the next one.
            while trained_docs < max_documents and offset < total_sources:
                docs = self.web_knowledge.documents_for_training(
                    limit=page_size, source_prefix="Wikipedia", offset=offset
                )
                if not docs:
                    break
                offset += len(docs)
                pending = [doc for doc in docs if not trainer.source_was_trained(str(doc.get("url", "")))]
                if not pending:
                    continue
                batch = pending[:max_documents - trained_docs]
                texts: list[str] = []
                from knowledge_store import split_text
                for doc in batch:
                    title = str(doc.get("title", "")).strip()
                    content = " ".join(str(doc.get("content", "")).split())
                    if not content:
                        continue
                    chunks = split_text(content, chunk_chars=900, overlap=80)
                    selected = chunks[:2]
                    if len(chunks) > 3:
                        selected.append(chunks[len(chunks) // 2])
                    for chunk in selected:
                        texts.append(f"{title}. {chunk}")
                pair_updates += trainer.learn_associations_bulk(texts, max_words_per_text=16)
                trainer.mark_sources_trained([(str(doc.get("url", "")), str(doc.get("title", ""))) for doc in batch])
                trained_docs += len(batch)
                if progress:
                    progress(f"Обучение ассоциаций: обработано {trained_docs}/{max_documents}; статей в базе {total_sources}")
            return {"trained_documents": trained_docs, "association_updates": pair_updates,
                    "remaining_documents": max(0, total_sources - previously_trained - trained_docs)}
        finally:
            trainer.close()

    def _identity_summary(self) -> str:
        traits = self._tone_name().lower()
        detail = "подробно и с примерами" if self._detail_level() >= 2 else "понятно и по существу"
        return (
            f"Меня зовут {self.name}. Я — автономный агент проекта EmotionalAI Lab. "
            f"Мой стиль общения — {traits}; я стараюсь объяснять мысли {detail}, учитывать контекст разговора "
            "и постепенно улучшать связи между понятиями на основе накопленного опыта."
        )

    def _creator_answer(self) -> str:
        return (
            "Меня создали в рамках проекта EmotionalAI Lab. В этом проекте разработаны моё автономное ядро, "
            "интерфейс, анализ сообщений, встроенная база знаний, долговременная память и механизмы работы с обратной связью. "
            "Я работаю как самостоятельная программа: анализирую сообщения собственным кодом и использую встроенные знания, "
            "ассоциативные связи и долговременную память. Мои возможности ограничены этими правилами и корпусом; "
            "это не большая нейросеть, обученная с нуля."
        )

    def _self_learning_answer(self) -> str:
        return (
            "Базовые правила и стартовые знания для меня подготовлены в проекте EmotionalAI Lab. "
            "При первом запуске я также обучаю ассоциативную память на небольшом локальном корпусе: укрепляю связи между словами, "
            "которые встречаются в одном контексте. В диалогах я могу сохранять факты, учитывать прошлые эпизоды и принимать явные исправления. "
            "Это небольшое обучение на ассоциациях и опыте, а не изменение весов большой нейросети. Поэтому новые знания помогают мне "
            "находить связанные понятия, но сами по себе не гарантируют умение рассуждать на любой незнакомой теме."
        )

    def _answer_memory_query(self, text: str) -> str:
        # Answer a specific personal fact before falling back to a general memory summary.
        if self._is_targeted_memory_question(text):
            direct = self._answer_personal_fact(text)
            if direct:
                return direct
            targeted = self._answer_targeted_memory_question(text)
            if targeted:
                return targeted
        facts = self.memory.facts("пользователь", 60)
        notes = self.memory.all_notes(40)
        eps = self.memory.recent_episodes(12)
        if not facts and not notes and not eps:
            return "Мы пока мало общались, так что личных сведений у меня почти нет."
        parts: list[str] = []
        seen = set()
        for fact in facts:
            predicate = str(fact.get("predicate", "")).strip()
            obj = str(fact.get("object", "")).strip()
            pred_n = normalize(predicate)
            if pred_n == "любит":
                item = f"Ты любишь {obj}."
            elif pred_n == "нравится":
                item = f"Тебе нравится {obj}."
            elif pred_n == "предпочитает":
                item = f"Ты предпочитаешь {obj}."
            elif pred_n == "интересуется":
                item = f"Тебя интересует {obj}."
            elif pred_n == "имя":
                item = f"Тебя зовут {obj}."
            else:
                item = f"{predicate.capitalize()} — {obj}."
            key = normalize(item)
            if key not in seen:
                parts.append("• " + item)
                seen.add(key)
        # Don't repeat a known fact in free-text form (e.g. both "любит — старые игры"
        # and "Я люблю старые игры"). Prefer structured facts; notes are a fallback.
        if not facts:
            for note in notes:
                content = " ".join(str(note.get("content", "")).split())
                key = normalize(content)
                if not content or key in seen or "?" in content or self.memory._is_control_text(content):
                    continue
                if len(content) < 8:
                    continue
                parts.append("• " + self._trim(content, 200))
                seen.add(key)
                if len(parts) >= 10:
                    break
        if not parts and eps:
            recent_topics = []
            for episode in eps:
                user_text = " ".join(str(episode.get("user_text", "")).split())
                key = normalize(user_text)
                if user_text and "?" not in user_text and key not in seen and not self.memory._is_control_text(user_text):
                    recent_topics.append(user_text)
                    seen.add(key)
                if len(recent_topics) >= 5:
                    break
            parts.extend("• " + self._trim(item, 180) for item in recent_topics)
        if not parts:
            return "Я помню ход нашей беседы, но пока не выделила устойчивых личных фактов."
        return "Из наших разговоров:\n" + "\n".join(parts[:20])

    @staticmethod
    def _is_targeted_memory_question(text: str) -> bool:
        s = normalize(text)
        return any(x in s for x in (
            "ты помнишь", "помнишь, что мне нравится", "помнишь что мне нравится",
            "что я люблю", "что мне нравится", "что я предпочитаю", "какие у меня интересы",
            "какой у меня любимый", "какая у меня любимая", "мой любимый цвет", "какой мой любимый", "какая моя любимая", "что я говорил про",
        )) and "обо мне" not in s and not s.startswith("/память")

    def _answer_targeted_memory_question(self, text: str) -> str | None:
        """Answer a question about a specific prior fact, without dumping unrelated facts."""
        s = normalize(text)
        facts = self.memory.facts("пользователь", 120)

        # Narrow questions (favorite color, name, specific topic) should return one relevant fact.
        if "цвет" in s:
            for fact in facts:
                predicate = str(fact.get("predicate", "")).strip()
                obj = str(fact.get("object", "")).strip()
                if obj and any(cue in normalize(predicate) for cue in ("цвет", "любим")):
                    return f"Твой любимый цвет — {obj}."
            return None
        if "как меня зовут" in s:
            for fact in facts:
                if normalize(str(fact.get("predicate", ""))) == "имя" and fact.get("object"):
                    return f"Тебя зовут {fact['object']}."
            return None

        # For questions like "what do I like?", return a concise list of preferences only.
        asks_interests = any(cue in s for cue in (
            "что мне нравится", "что я люблю", "что я предпочитаю", "какие у меня интересы", "помнишь что мне нравится"
        ))
        if asks_interests:
            lines = []
            for fact in facts:
                predicate = normalize(str(fact.get("predicate", "")))
                obj = str(fact.get("object", "")).strip()
                if not obj:
                    continue
                if predicate == "любит":
                    lines.append(f"Ты любишь {obj}.")
                elif predicate == "нравится":
                    lines.append(f"Тебе нравится {obj}.")
                elif predicate == "предпочитает":
                    lines.append(f"Ты предпочитаешь {obj}.")
                elif predicate == "интересуется":
                    lines.append(f"Тебя интересует {obj}.")
            lines = list(dict.fromkeys(lines))
            if lines:
                return "Ты рассказывал, что:\n" + "\n".join("• " + line for line in lines[:8])

        # Match the requested subject against structured facts first, then notes/episodes.
        if "что я говорил про" in s or "что я говорила про" in s:
            query = re.sub(r".*?что я говор(?:ил|ила) про\s*", "", s).strip(" ?.! ")
            candidates = []
            for fact in facts:
                predicate = str(fact.get("predicate", "")).strip()
                obj = str(fact.get("object", "")).strip()
                combined = f"{predicate} {obj}"
                score = similarity(query, combined)
                if obj and score >= 0.15:
                    candidates.append((score, predicate, obj))
            if candidates:
                candidates.sort(reverse=True)
                _, predicate, obj = candidates[0]
                if normalize(predicate) == "любит":
                    return f"Ты любишь {obj} — ты об этом рассказывал."
                if normalize(predicate) == "нравится":
                    return f"Ты говорил, что тебе нравится {obj}."
                return f"Ты говорил: {predicate} — {obj}."
            notes = self.memory.search_notes(query, 5, min_score=0.12)
            if notes:
                content = str(notes[0].get("content", "")).strip()
                if content and not self.memory._is_control_text(content):
                    return f"Ты говорил: «{self._trim(content, 220)}»."

        # General targeted recollection only reports relevant preferences, not every stored fact.
        lines = []
        for fact in facts:
            predicate = normalize(str(fact.get("predicate", "")))
            obj = str(fact.get("object", "")).strip()
            if not obj:
                continue
            if predicate == "любит":
                lines.append(f"Ты любишь {obj}.")
            elif predicate == "нравится":
                lines.append(f"Тебе нравится {obj}.")
            elif predicate == "предпочитает":
                lines.append(f"Ты предпочитаешь {obj}.")
            elif predicate == "интересуется":
                lines.append(f"Тебя интересует {obj}.")
        lines = list(dict.fromkeys(lines))
        if lines:
            return "Ты раньше рассказывал, что:\n" + "\n".join("• " + line for line in lines[:8])
        return None

    def _answer_personal_fact(self, text: str) -> str | None:
        s = normalize(text)
        direct_question = any(x in s for x in (
            "мой любимый", "моя любимая", "какой у меня", "какая у меня", "что мне нравится",
            "что я люблю", "что я предпочитаю", "как меня зовут", "где я живу", "на чем я играю",
        ))
        if not direct_question:
            return None
        matches = self.memory.search_facts(text, 6)
        for fact in matches:
            if float(fact.get("relevance", 0)) < 0.07:
                continue
            pred = str(fact.get("predicate", "")).strip()
            obj = str(fact.get("object", "")).strip()
            if not obj:
                continue
            pn = normalize(pred)
            if "цвет" in s and ("цвет" in pn or "любим" in pn):
                return f"Твой любимый цвет — {obj}."
            if "как меня зовут" in s and pn == "имя":
                return f"Тебя зовут {obj}."
            if any(x in s for x in ("что мне нравится", "что я люблю", "какие у меня интересы")):
                return f"Судя по тому, что ты рассказывал, тебе нравится {obj}."
            return f"Ты рассказывал, что {pred} — {obj}."
        return None

    def _acknowledge_statement(self, statement: str) -> str:
        """Respond naturally to explicit learning commands without narrating storage."""
        clean = statement.strip(" .!?,:;«»\"'")
        low = normalize(clean)
        if not clean:
            return "Хорошо."

        # Convert explicit fact-like commands into a short human-sounding response.
        match = re.search(r"(?:мой любимый цвет|моя любимая цветовая гамма|любимый цвет)\s*[—–:= -]+\s*(.+)$", clean, flags=re.I)
        if match:
            value = match.group(1).strip(" .!?,;:«»\"'")
            if value:
                return f"{value.capitalize()} — твой любимый цвет."

        # Keep preferences concise and invite a real conversation instead of padding them.
        for prefix in ("я люблю", "мне нравится", "я предпочитаю", "мне интересно", "я обожаю"):
            if low.startswith(prefix):
                content = clean[len(prefix):].strip(" .!?,;:—–")
                if content:
                    return f"О, {content}. Что тебе в этом больше всего нравится?"
                return "Что именно тебе нравится?"

        # A long note doesn't need an invented analysis. Acknowledge it and ask only if useful.
        if len(clean.split()) > 22:
            return "Поняла. Что в этой ситуации для тебя самое важное?"
        if low.startswith(("не запоминай", "забудь", "удали из памяти")):
            return "Хорошо."
        if low.startswith(("правильно:", "исправление:")):
            return "Да, так точнее."
        return "Поняла."

    def _format_knowledge_answer(self, topic: str, entry: dict) -> str:
        definition = str(entry.get("definition", f"{topic} — понятие, значение которого зависит от контекста.")).strip()
        details = str(entry.get("details", "")).strip()
        example = str(entry.get("example", "")).strip()
        practical = str(entry.get("practical", "")).strip()
        related = entry.get("related", [])
        level = self._detail_level()
        paragraphs = [definition]
        if level >= 1 and details:
            paragraphs.append(details)
        if level >= 2 and example:
            paragraphs.append("Пример: " + example)
        if level >= 2 and practical:
            paragraphs.append("На практике: " + practical)
        if level >= 3 and related:
            paragraphs.append("С этим также связаны " + ", ".join(str(x) for x in related[:8]) + ".")
        elif related and self._prompt_contains("связывай темы", "объясняй связи"):
            paragraphs.append("Смежные понятия: " + ", ".join(str(x) for x in related[:4]) + ".")
        return "\n\n".join(paragraphs)

    def _answer_comparison(self, text: str) -> str | None:
        explicit = self.knowledge.find_explicit(text) if hasattr(self.knowledge, "find_explicit") else []
        topics = list(dict.fromkeys(topic for topic, _entry, _score in explicit))
        normalized_topics = {normalize(topic) for topic in topics}
        has_relation = any(x in normalize(text) for x in ("быстрее", "медленнее", "разница", "отличается", "отличие", "сравни", "лучше", "хуже", "чем "))
        if not has_relation:
            return None
        if "оперативная память" in normalized_topics and "ssd" in normalized_topics:
            return (
                "Оперативная память (RAM) и SSD нужны для разных вещей. RAM — быстрое временное хранилище, в котором "
                "программы держат данные прямо во время работы; после выключения питания содержимое обычно исчезает. "
                "SSD — накопитель для постоянного хранения системы, программ и файлов: данные остаются после выключения.\n\n"
                "RAM обычно имеет меньшую задержку и гораздо выше пропускную способность, но её объём ограничен и она не заменяет диск. "
                "SSD медленнее RAM, зато хранит большие объёмы данных постоянно. Если RAM мало, компьютер может использовать файл подкачки "
                "на SSD — это помогает избежать немедленного сбоя, но заметно медленнее обычной работы из оперативной памяти."
            )
        if {"ssd", "hdd"}.issubset(normalized_topics):
            return (
                "SSD и HDD оба хранят данные, но устроены по-разному. SSD использует флеш-память без движущихся частей, "
                "поэтому обычно быстрее запускает систему и открывает множество мелких файлов. HDD читает данные с вращающихся "
                "магнитных пластин: он обычно дешевле за гигабайт, но имеет механические задержки и чувствительнее к ударам.\n\n"
                "SSD прежде всего сокращает загрузки и задержки чтения ресурсов; средний FPS он не обязательно повышает, если игра упирается в CPU или GPU."
            )
        if len(topics) >= 2:
            topic_a, topic_b = topics[:2]
            entry_a = self.knowledge.entries.get(topic_a, {})
            entry_b = self.knowledge.entries.get(topic_b, {})
            return (
                f"{topic_a}: {str(entry_a.get('definition', '')).strip()}\n\n"
                f"{topic_b}: {str(entry_b.get('definition', '')).strip()}\n\n"
                "Главное различие — в назначении и компромиссах. Если выбираешь между ними, напиши, для какой задачи: "
                "тогда сравнение можно привязать к конкретному использованию."
            )
        return None

    def _try_memory_recall(self, text: str, analysis) -> tuple[str, str, float] | None:
        s = normalize(text)
        cues = ("помнишь", "раньше", "прошлый раз", "мы обсуждали", "ты говорил", "ты говорила", "что я писал", "мой ", "моя ", "мои ", "обо мне", "какой у меня", "что у меня", "мне нравится", "я люблю")
        memory_related = any(normalize(cue) in s for cue in cues)
        if analysis.intent not in ("question", "knowledge_question", "knowledge_topic", "statement", "emotional_disclosure"):
            return None
        fact = next(iter(self.memory.search_facts(text, 4)), None)
        note = next(iter(self.memory.search_notes(text, 4, min_score=0.08)), None)
        episode = next(iter(self.memory.search_episodes(text, limit=2, min_score=0.12)), None)
        fact_threshold = 0.10 if memory_related else 0.34
        if fact and float(fact.get("relevance", 0)) >= fact_threshold:
            pred, obj = str(fact.get("predicate", "")), str(fact.get("object", ""))
            if fact.get("subject") == "пользователь":
                templates = {
                    "любит": f"Тебе нравится {obj}", "нравится": f"Тебе нравится {obj}",
                    "не любит": f"Тебе не нравится {obj}", "не нравится": f"Тебе не нравится {obj}",
                    "предпочитает": f"Ты предпочитаешь {obj}", "имя": f"Тебя зовут {obj}",
                    "живёт в": f"Ты живёшь в {obj}", "учится в": f"Ты учишься в {obj}",
                    "имеет": f"У тебя есть {obj}", "играет в": f"Ты играешь в {obj}",
                    "использует": f"Ты используешь {obj}", "планирует": f"Ты планируешь {obj}",
                    "хочет": f"Ты хочешь {obj}",
                }
                summary = templates.get(pred, f"В твоём профиле есть деталь «{pred} — {obj}»")
                return summary + ".", "successful_recall", 0.55
            return f"Раньше мы затрагивали тему «{pred} — {obj}».", "successful_recall", 0.45
        if note and float(note.get("relevance", 0)) >= (0.10 if memory_related else 0.40):
            content = str(note.get("content", "")).strip()
            if content and memory_related:
                return f"Ты говорил: «{content}». Это связано с текущим вопросом.", "successful_recall", 0.5
            if content and similarity(text, content) >= 0.40:
                return f"Эта тема перекликается с твоими словами: «{content}».", "successful_recall", 0.42
        if episode and float(episode.get("relevance", 0)) >= (0.10 if memory_related else 0.30):
            old = " ".join(str(episode.get("user_text", "")).split())
            if old and similarity(text, old) >= 0.16 and memory_related:
                return f"Мы уже обсуждали похожую тему: «{self._trim(old, 220)}». Можно продолжить с этой мысли.", "successful_recall", 0.45
        return None

    def _personal_context(self, text: str) -> list[str]:
        context = []
        # Use personal facts only when they overlap with this topic; don't dump a biography into every answer.
        for item in self.memory.search_facts(text, 3):
            if float(item.get("relevance", 0)) >= 0.22 and item.get("subject") == "пользователь":
                context.append(f"{item.get('predicate', '')}: {item.get('object', '')}")
        return context

    def _general_question_answer(self, text: str) -> str:
        """Search local knowledge first; optional web lookup is a source query, never an AI handoff."""
        matches = self.knowledge.find(text)
        if matches:
            topic, entry, score = matches[0]
            if score >= 0.55:
                return self._format_knowledge_answer(topic, entry)

        try:
            documents = self.web_knowledge.search(text, limit=3, min_score=0.20)
        except Exception:
            documents = []
        # Dialogue-format examples are not treated as factual references by the standalone core.
        documents = [item for item in documents if not str(item.get("source", "")).startswith("Instruction examples")]
        if documents:
            best = documents[0]
            title = str(best.get("title", "Источник")).strip() or "Источник"
            source_text = " ".join(str(best.get("text", "")).split())
            url = str(best.get("url", "")).strip()
            if source_text:
                answer = f"В локальной базе нашёл материал «{title}». Вот подходящий фрагмент:\n\n{self._trim(source_text, 700)}"
                if url:
                    answer += f"\n\nИсточник: {url}"
                return answer

        if self.config.get("online_wikipedia_fallback", False):
            online_answer = wikipedia_lookup(text)
            if online_answer:
                return online_answer

        return (
            "Моя текущая база не содержит ответа с достаточной уверенностью. "
            "Я не буду придумывать случайное объяснение. Можно расширить мои знания через «Данные из интернета» "
            "или импортировать свои .txt/.md файлы — после сохранения я смогу искать в них локально."
        )

    def _general_statement_response(self, text: str) -> str:
        """Brief offline conversational response for statements, not a fake universal analysis."""
        low = normalize(text)
        if "шаблон" in low and any(x in low for x in ("не", "хватит", "убери", "прекрати", "перестань", "достал", "изай", "юзай")):
            return "Да, этот ответ был пустой заготовкой. Я должна отвечать на смысл твоей фразы, а не растягивать её общими абзацами."
        if any(x in low for x in ("ответ шаблонный", "шаблонный ответ", "говоришь шаблонами", "вода", "не отвечаешь по делу", "не ответил на вопрос")):
            return "Согласна, ответ получился пустым и не по делу. Надо исправлять конкретный ответ, а не дописывать к нему ещё общие слова."
        if any(x in low for x in ("старые игры", "старый roblox", "старый роблокс")):
            if any(x in low for x in ("роблокс", "roblox")):
                return "У старого Roblox свой шарм: простая геометрия, ограничения движка и более узнаваемые механики создавали особую атмосферу. Что тебе в нём важнее — внешний вид, физика или сами игры?"
            return "Старые игры часто цепляют не графикой, а узнаваемым стилем, звуком и темпом. Ограничения техники иногда даже помогали дизайнерам сделать игру более цельной."
        if any(x in low for x in ("я хочу", "хочу сделать", "хочу создать", "хочу попробовать")):
            if any(x in low for x in ("ии", "нейросет", "бот", "программ", "игр", "мод", "робот")):
                return "Давай разберём, как это собрать. Важно определить, какой результат ты хочешь получить и что уже есть — тогда можно выбрать подходящую архитектуру, а не городить случайные функции."
            return "Что именно ты хочешь сделать? Расскажи чуть подробнее — помогу разложить это на конкретные шаги."
        if any(x in low for x in ("я думаю", "мне кажется", "по-моему", "по моему", "я считаю")):
            clean = self._trim(" ".join(text.split()), 220)
            return f"Поняла твою мысль: «{clean}». Что именно привело тебя к такому выводу?" if len(clean.split()) > 5 else "В чём именно ты видишь главную причину?"
        if any(x in low for x in ("мне нравится", "я люблю", "мне интересно", "я предпочитаю", "я обожаю")):
            if any(x in low for x in ("робот", "ии", "нейросет", "программирован", "код")):
                return "Круто — в этой теме можно не только читать теорию, но и собирать свои вещи и проверять, как они ведут себя на практике."
            if any(x in low for x in ("игр", "sonic", "соник", "minecraft", "майнкрафт")):
                return "Понимаю, игры могут цеплять не только геймплеем, но и темпом, звуком и собственным стилем."
            return "О, интересно. Что именно тебе в этом больше всего нравится?"
        if any(x in low for x in ("бесит", "раздражает", "задолбал", "заебал", "пиздец", "хуйня", "не работает", "сломал", "сломалось")):
            return "Да, звучит раздражающе. Что конкретно произошло? Разберём саму проблему, без общей лекции."
        if any(x in low for x in ("я устал", "я устала", "устал сегодня", "устала сегодня")):
            return "Тогда, наверное, стоит немного выдохнуть. День тяжёлый был или просто накопилась усталость?"
        if any(x in low for x in ("я дома", "пришел домой", "пришла домой", "зашел домой", "зашла домой")):
            return "Наконец-то дома. День нормально прошёл?"
        if len(low.split()) <= 3:
            return "Слушаю."
        return "Я услышала, что ты говоришь, но не хочу придумывать за тебя реакцию. Что в этой мысли для тебя главное?"

    def _personalize(self, answer: str, question: str) -> str:
        # Context is woven in sparingly so it improves relevance without narrating the storage mechanism.
        if self._detail_level() < 2:
            return answer
        context = self._personal_context(question)
        if context and any(x in normalize(question) for x in ("мне", "мой", "моя", "у меня", "для меня", "мой компьютер", "моя игра")):
            answer += "\n\nС учётом твоего контекста это особенно зависит от: " + "; ".join(context[:2]) + "."
        return answer

    def _supportive_response(self, text: str) -> str:
        low = normalize(text)
        if any(x in low for x in ("страшно", "боюсь", "тревожно")):
            return "Похоже, ситуация действительно тебя напрягает. Давай разделим её на части: что уже произошло, чего ты опасаешься и что из этого можно изменить прямо сейчас. Когда тревога смешивает все сценарии в один, конкретные факты помогают вернуть ощущение контроля."
        if any(x in low for x in ("злюсь", "бесит", "раздраж")):
            return "Похоже, тебя сильно задело происходящее. Часто особенно раздражает не только само событие, но и чувство несправедливости или невозможность что-то исправить. Полезно сначала назвать конкретно, что именно произошло, а потом отделить это от выводов, которые возникли из-за злости."
        return "Похоже, сейчас тебе нелегко. Необязательно решать всё за один раз: можно сначала разобраться, что произошло, что больше всего давит и какая часть ситуации зависит от тебя. После этого легче выбрать следующий шаг, даже если сразу решить всю проблему нельзя."

    def emotion_report(self) -> str:
        values = self.emotion.as_dict()
        labels = {
            "valence": "тон настроения", "arousal": "возбуждение", "curiosity": "любопытство",
            "trust": "доверие в диалоге", "frustration": "раздражение", "fatigue": "усталость",
            "joy": "радость", "sadness": "грусть", "anger": "злость", "surprise": "удивление",
            "empathy": "сочувствие",
        }
        lines = [f"• {labels[k]}: {v:+.2f}" if k == "valence" else f"• {labels[k]}: {v:.2f}" for k, v in values.items()]
        lines.append(f"• текущий стиль: {self.emotion.tone()}")
        return "Текущее внутреннее состояние:\n" + "\n".join(lines)

    @staticmethod
    def _trim(value: str, limit: int) -> str:
        return value if len(value) <= limit else value[:limit - 3].rstrip() + "..."

    def close(self) -> None:
        self.memory.close()

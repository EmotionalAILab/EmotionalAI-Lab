"""Regression tests using realistic user turns with the autonomous EmotionalAgent.

The suite exercises the built-in engine and requires no external model server.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from agent import EmotionalAgent
from dialogue_tools import calculate_message
from reasoning import ReasoningEngine
from semantic import KnowledgeBase


class RealDialogueRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copy(PROJECT / ("config.json" if (PROJECT / "config.json").exists() else "config.template.json"), self.root / "config.json")
        shutil.copytree(PROJECT / "data", self.root / "data")
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        config["database_path"] = "data/test_memory.sqlite3"
        config["online_wikipedia_fallback"] = False
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.agent = EmotionalAgent(self.root)

    def tearDown(self):
        self.agent.close()
        self.tmp.cleanup()

    def test_greeting_is_short_and_natural(self):
        self.assertEqual(self.agent.respond("Привет!"), "Привет! О чём поговорим?")

    def test_creator_question_is_direct(self):
        answer = self.agent.respond("Кто тебя создал?").lower()
        self.assertIn("emotionalai lab", answer)
        self.assertIn("меня создали", answer)
        self.assertNotIn("chatgpt", answer)
        self.assertNotIn("интересные моменты", answer)

    def test_self_identity_mentions_emotionalai_lab(self):
        answer = self.agent.respond("Расскажи о себе").lower()
        self.assertIn("emotionalai lab", answer)
        self.assertIn("автономный агент", answer)

    def test_training_question_is_not_routed_to_generic_training_article(self):
        answer = self.agent.respond("Как ты обучалась?").lower()
        self.assertEqual(self.agent.last_analysis.intent, "self_learning")
        self.assertIn("emotionalai lab", answer)
        self.assertIn("ассоциатив", answer)

    def test_self_logic_does_not_invent_a_recent_failure(self):
        answer = self.agent.respond("И так теперь логика твоя работает?").lower()
        self.assertEqual(self.agent.last_analysis.intent, "self_logic")
        self.assertIn("частично", answer)
        self.assertNotIn("только что", answer)
        self.assertNotIn("зацепилась за слово", answer)

    def test_self_understanding_question_gets_honest_reply(self):
        answer = self.agent.respond("Ты вообще меня понимаешь?").lower()
        self.assertEqual(self.agent.last_analysis.intent, "self_understanding")
        self.assertTrue("не всегда" in answer or "в какой-то степени" in answer)
        self.assertNotIn("небо", answer)

    def test_thanks_is_not_acknowledged_with_wrong_phrase(self):
        self.assertEqual(self.agent.respond("Спасибо"), "Пожалуйста!")

    def test_positive_mood_changes_casual_response_style(self):
        response = self.agent.respond("Я люблю старые игры!")
        self.assertTrue(response.startswith("О, звучит здорово!") or response.startswith("У старого Roblox"))
        self.assertIn("старые игры", response.lower())

    def test_emotional_statement_is_not_routed_to_dictionary_topic(self):
        response = self.agent.respond("Меня бесит, когда программа тупит")
        self.assertEqual(self.agent.last_analysis.intent, "emotional_disclosure")
        self.assertNotIn("программа — набор инструкций", response.lower())
        self.assertTrue(any(word in response.lower() for word in ("раздраж", "задел", "произошло", "неприят")))

    def test_explicit_correction_is_reused_for_same_question(self):
        self.agent.respond("Что такое оперативная память?")
        self.agent.respond("Правильно: оперативная память хранит данные активных программ и очищается после отключения питания.")
        self.assertGreaterEqual(self.agent.memory.stats()["learning_examples"], 1)
        corrected = self.agent.respond("Что такое оперативная память?")
        self.assertIn("данные активных программ", corrected.lower())

    def test_every_normal_turn_is_saved_and_associated(self):
        before = self.agent.memory.stats()
        self.agent.respond("Мне интересны роботы и старые игровые движки")
        after = self.agent.memory.stats()
        self.assertGreater(after["episodes"], before["episodes"])
        self.assertGreater(after["notes"], before["notes"])
        self.assertGreater(after["associations"], before["associations"])

    def test_wikipedia_training_updates_associations_only_once(self):
        text = ("Искусственный интеллект изучает методы обработки данных. Машинное обучение помогает алгоритмам "
                "находить закономерности, а нейронные сети применяются для анализа языка, изображений и звука. "
                "Обучение использует примеры и обратную связь, чтобы уменьшать ошибки и повышать качество результатов. ") * 5
        self.agent.web_knowledge.add_document(
            "Искусственный интеллект", "https://ru.wikipedia.org/wiki/Artificial_intelligence",
            text, "Wikipedia (ru)", "CC BY-SA 4.0"
        )
        first = self.agent.train_from_wikipedia(max_documents=10)
        self.assertEqual(first["trained_documents"], 1)
        self.assertGreater(first["association_updates"], 0)
        before = self.agent.memory.stats()["associations"]
        second = self.agent.train_from_wikipedia(max_documents=10)
        after = self.agent.memory.stats()["associations"]
        self.assertEqual(second["trained_documents"], 0)
        self.assertEqual(second["association_updates"], 0)
        self.assertEqual(before, after)

    def test_arithmetic_precedence(self):
        self.assertEqual(self.agent.respond("2 + 2 * 3"), "2 + 2 × 3 = 8.")
        self.assertEqual(self.agent.respond("Сколько будет 7 × 8?"), "7 × 8 = 56.")

    def test_arithmetic_spoken_operators(self):
        self.assertEqual(calculate_message("Сколько будет семь умножить на восемь?"), "7 × 8 = 56.")
        self.assertEqual(calculate_message("посчитай 10% от 200"), "10% от 200 = 20.")

    def test_zero_division_does_not_crash(self):
        result = self.agent.respond("10 / 0")
        self.assertIn("деление на ноль", result.lower())

    def test_ram_question_is_not_prefixed_by_duplicate_topic(self):
        answer = self.agent.respond("Что такое оперативная память?")
        self.assertTrue(answer.startswith("Оперативная память (RAM)"))
        self.assertIn("временное хранилище", answer.lower())
        self.assertIn("Пример:", answer)

    def test_ram_and_ssd_comparison_uses_two_explicit_concepts(self):
        answer = self.agent.respond("Чем оперативная память отличается от SSD?").lower()
        self.assertEqual(self.agent.last_analysis.intent, "comparison")
        self.assertIn("ram", answer)
        self.assertIn("ssd", answer)
        self.assertIn("временное хранилище", answer)
        self.assertIn("постоянного хранения", answer)
        self.assertNotIn("сравнивать назначение, сильные стороны", answer)

    def test_ssd_and_hdd_comparison(self):
        answer = self.agent.respond("Что быстрее, SSD или HDD?").lower()
        self.assertIn("ssd", answer)
        self.assertIn("hdd", answer)
        self.assertIn("механическ", answer)

    def test_code_request_is_not_an_encyclopedia_definition(self):
        answer = self.agent.respond("Как написать цикл for в Python?")
        self.assertEqual(self.agent.last_analysis.intent, "code_help")
        self.assertIn("for i in range(5):", answer)
        self.assertIn("0, 1, 2, 3, 4", answer)
        self.assertNotIn("высокоуровневый язык", answer)

    def test_luau_loop_request_uses_roblox_syntax(self):
        answer = self.agent.respond("Как сделать цикл for в Roblox Luau?")
        self.assertIn("for i = 1, 5 do", answer)
        self.assertIn("end", answer)

    def test_sky_color_question_has_an_actual_explanation(self):
        answer = self.agent.respond("Почему небо голубое?").lower()
        self.assertIn("рассеива", answer)
        self.assertIn("коротковолнов", answer)
        self.assertNotIn("база пока не содержит ответа", answer)

    def test_photosynthesis_question_has_an_actual_explanation(self):
        answer = self.agent.respond("Что такое фотосинтез?").lower()
        self.assertIn("энергию света", answer)
        self.assertIn("углекислый газ", answer)
        self.assertIn("кислород", answer)

    def test_explicit_memory_command_is_not_misrouted_to_memory_query(self):
        first = self.agent.respond("Запомни: мой любимый цвет — синий")
        self.assertEqual(self.agent.last_analysis.intent, "learning_command")
        self.assertNotIn("вспоминается следующее", first.lower())
        self.assertNotIn("запомнила", first.lower())
        second = self.agent.respond("Какой мой любимый цвет?")
        self.assertEqual(self.agent.last_analysis.intent, "memory_query")
        self.assertIn("синий", second.lower())

    def test_about_me_memory_avoids_duplicate_raw_notes(self):
        self.agent.respond("Я люблю старые игры")
        self.agent.respond("Запомни: мой любимый цвет — синий")
        result = self.agent.respond("Что ты помнишь обо мне?").lower()
        self.assertIn("старые игры", result)
        self.assertIn("синий", result)
        self.assertEqual(result.count("старые игры"), 1)
        self.assertEqual(result.count("любимый цвет"), 1)
        self.assertNotIn("что такое фотосинтез", result)

    def test_follow_up_uses_earlier_context_even_after_other_turns(self):
        self.agent.respond("Я люблю старые игры")
        self.agent.respond("Как тебя зовут?")
        self.agent.respond("Спасибо")
        result = self.agent.respond("А почему у них такая атмосфера?")
        self.assertIn("старые игры", result.lower())
        self.assertIn("звук", result.lower())

    def test_criticism_chain_does_not_repeat_stock_acknowledgment(self):
        self.agent.respond("И так теперь логика твоя работает?")
        first = self.agent.respond("Ты совсем")
        second = self.agent.respond("тупой")
        self.assertNotEqual(first, "Ага, понял тебя.")
        self.assertNotEqual(second, "Ага, понял тебя.")
        self.assertIn("проверять конкретные вопросы", second.lower())

    def test_unknown_entity_does_not_receive_unrelated_quantum_article(self):
        result = self.agent.respond("Что такое бозон Хиггса?").lower()
        self.assertEqual(self.agent.last_analysis.intent, "question")
        self.assertNotIn("квантовая физика изучает", result)
        self.assertIn("не содержит ответа", result)

    def test_known_topic_still_uses_specific_knowledge(self):
        result = self.agent.respond("Что такое квантовая физика?").lower()
        self.assertEqual(self.agent.last_analysis.intent, "knowledge_question")
        self.assertIn("атомном и субатомном", result)

    def test_unknown_question_does_not_generate_fake_analysis(self):
        result = self.agent.respond("Почему норбитальные фланцы светятся в пятый четверг?").lower()
        self.assertNotIn("интересная мысль", result)
        self.assertNotIn("три шага", result)
        self.assertNotIn("полезно посмотреть по слоям", result)
        self.assertLess(len(result), 350)

    def test_explicit_entity_match_does_not_treat_ram_as_generic_memory(self):
        kb = KnowledgeBase(PROJECT / "data" / "knowledge.json")
        names = [item[0].lower() for item in kb.find_explicit("Чем оперативная память отличается от SSD?")]
        self.assertIn("оперативная память", names)
        self.assertIn("ssd", names)
        self.assertNotIn("память", names)

    def test_specific_memory_question_does_not_dump_unrelated_preferences(self):
        self.agent.respond("Я люблю старые игры")
        self.agent.respond("Запомни: мой любимый цвет — синий")
        result = self.agent.respond("Какой мой любимый цвет?").lower()
        self.assertIn("синий", result)
        self.assertIn("любимый цвет", result)
        self.assertNotIn("старые игры", result)

    def test_translation_request_is_not_routed_to_language_article(self):
        result = self.agent.respond("Переведи hello на русский")
        self.assertEqual(self.agent.last_analysis.intent, "translation")
        self.assertIn("привет", result.lower())
        self.assertNotIn("система знаков", result.lower())

    def test_creative_request_returns_requested_text(self):
        result = self.agent.respond("Напиши стих про робота")
        self.assertEqual(self.agent.last_analysis.intent, "creative_request")
        self.assertIn("робота", result.lower())
        self.assertGreaterEqual(len(result.splitlines()), 3)

    def test_quantum_physics_question_has_relevant_explanation(self):
        result = self.agent.respond("Что такое квантовая физика?").lower()
        self.assertIn("атомном", result)
        self.assertIn("вероятност", result)

    def test_phone_battery_question_gets_useful_diagnostics(self):
        result = self.agent.respond("Почему телефон быстро разряжается?").lower()
        self.assertIn("аккумулятор", result)
        self.assertIn("сигнал", result)
        self.assertIn("настройки", result)

    def test_followup_explanation_request_uses_topic_knowledge(self):
        result = self.agent.respond("Расскажи про квантовую механику подробнее").lower()
        self.assertEqual(self.agent.last_analysis.intent, "knowledge_question")
        self.assertIn("квантов", result)
        self.assertIn("вероятност", result)

    def test_love_definition_does_not_become_generic_reaction(self):
        result = self.agent.respond("Что значит любовь?").lower()
        self.assertIn("эмоциональн", result)
        self.assertIn("привязан", result)

    def test_roblox_studio_question_is_not_generic_chat(self):
        result = self.agent.respond("Расскажи про Roblox Studio").lower()
        self.assertIn("luau", result)
        self.assertIn("скрипт", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

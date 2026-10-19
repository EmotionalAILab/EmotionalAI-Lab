import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion import EmotionState
from memory import MemoryStore
from semantic import KnowledgeBase, similarity, tokenize
from learning import LearningEngine

class ProjectTests(unittest.TestCase):
    def test_tokenize_russian(self):
        self.assertIn("память", tokenize("Память и обучение"))

    def test_similarity(self):
        self.assertGreater(similarity("мне нравится синий цвет", "нравится синий цвет"), 0.2)

    def test_emotion_changes_and_clamps(self):
        state = EmotionState()
        state.appraise("Это круто!!!")
        self.assertGreater(state.valence, 0.1)
        for _ in range(100):
            state.appraise("это ужасно бесит")
        self.assertLessEqual(state.frustration, 1.0)

    def test_emotion_tone_responds_to_latest_message_and_saved_state(self):
        state = EmotionState()
        self.assertEqual(state.tone("Я люблю старые игры!"), "excited")
        self.assertEqual(state.tone("Мне грустно"), "gentle")
        self.assertEqual(state.tone("Меня бесит эта ошибка"), "careful")
        self.assertEqual(state.tone("Почему это работает?"), "curious")
        persisted = EmotionState()
        persisted.appraise("Мне грустно")
        persisted.decay(0.08)
        self.assertEqual(persisted.tone(), "gentle")
        cheerful = EmotionState()
        cheerful.appraise("Круто!")
        cheerful.decay(0.08)
        self.assertEqual(cheerful.tone(), "excited")

    def test_fact_persists_in_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "m.sqlite3")
            learning = LearningEngine(memory)
            result = learning.learn_from("Запомни: мой любимый цвет — синий")
            self.assertTrue(result["learned"])
            self.assertTrue(memory.search_facts("любимый цвет синий"))
            memory.close()


    def test_episode_and_note_persist_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.sqlite3"
            memory = MemoryStore(path)
            memory.add_episode("Я люблю старые игры", "Понял", {"curiosity": 0.5})
            memory.close()
            reopened = MemoryStore(path)
            self.assertEqual(reopened.stats()["episodes"], 1)
            self.assertTrue(reopened.search_notes("старые игры", min_score=0.1))
            reopened.close()

    def test_multiple_preferences_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "m.sqlite3")
            memory.add_fact("пользователь", "любит", "старые игры")
            memory.add_fact("пользователь", "любит", "робототехнику")
            likes = [f["object"] for f in memory.facts("пользователь") if f["predicate"] == "любит"]
            self.assertIn("старые игры", likes)
            self.assertIn("робототехнику", likes)
            memory.close()

    def test_agent_does_not_claim_first_message_is_old_memory(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            first = agent.respond("Я люблю старые игры и старый Roblox.")
            second = agent.respond("Ты помнишь, что я говорил про старые игры?")
            self.assertNotIn("прошлый разговор", first.lower())
            self.assertIn("ты любишь старые игры", second.lower())
            acknowledgement = agent.respond("Запомни: мой любимый цвет — синий")
            self.assertNotIn("запомнил", acknowledgement.lower())
            self.assertNotIn("сохранил", acknowledgement.lower())
            color = agent.respond("Какой у меня любимый цвет?")
            self.assertIn("синий", color.lower())
            normal_question = agent.respond("Что такое оперативная память?")
            self.assertIn("оперативная память", normal_question.lower())
            self.assertIn("пример", normal_question.lower())
            agent.close()


    def test_knowledge_base_has_expanded_topics_and_specific_phrase_wins(self):
        root = Path(__file__).resolve().parents[1]
        kb = KnowledgeBase(root / "data" / "knowledge.json")
        self.assertGreaterEqual(len(kb.entries), 80)
        self.assertEqual(kb.find("Что такое оперативная память?")[0][0], "оперативная память")

    def test_persona_settings_persist(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            agent.save_persona("Raven", "Спокойный и остроумный собеседник. Отвечай очень подробно.", "Спокойный", "Очень подробно")
            agent.close()
            reopened = EmotionalAgent(root)
            self.assertEqual(reopened.name, "Raven")
            self.assertEqual(reopened.response_detail, "Очень подробно")
            self.assertIn("остроумный", reopened.personality_prompt.lower())
            reopened.close()


    def test_standalone_mode_ignores_legacy_model_settings_and_never_calls_llm(self):
        import json, shutil
        from unittest.mock import patch
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / "config.template.json", root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            # A stale legacy file that points at an LLM must not re-enable it.
            (root / "model_settings.json").write_text(json.dumps({"enabled": True, "base_url": "http://127.0.0.1:1", "model": "some-model"}), encoding="utf-8")
            agent = EmotionalAgent(root)
            self.assertFalse(hasattr(agent, "model_settings"))
            self.assertFalse(hasattr(agent, "llm"))
            with patch("urllib.request.urlopen", side_effect=AssertionError("network/LLM access should not occur")):
                self.assertIn("О чём поговорим", agent.respond("Привет"))
                answer = agent.respond("Сколько будет 7 × 8?")
                self.assertIn("56", answer)
            self.assertIn("автономное ядро", agent.core_status_text().lower())
            self.assertIn("не используются", agent.core_status_text().lower())
            agent.close()

    def test_knowledge_base_loads(self):
        root = Path(__file__).resolve().parents[1]
        kb = KnowledgeBase(root / "data" / "knowledge.json")
        self.assertTrue(kb.find("что такое память?"))

    def test_internet_corpus_search_and_dedup(self):
        from knowledge_store import WebKnowledgeStore
        with tempfile.TemporaryDirectory() as tmp:
            store = WebKnowledgeStore(Path(tmp) / "internet.sqlite3")
            article = (
                "Оперативная память (RAM) — энергозависимая память компьютера, в которой временно "
                "хранятся данные и команды программ, с которыми сейчас работает процессор. "
                "После отключения питания содержимое оперативной памяти обычно исчезает. "
                "Объём RAM влияет на число одновременно открытых приложений и на переключение между ними. "
                "Оперативную память нужно отличать от SSD, на котором файлы хранятся постоянно."
            )
            self.assertTrue(store.add_document("Оперативная память", "https://example.org/ram", article, "Test", "CC0"))
            self.assertFalse(store.add_document("Оперативная память", "https://example.org/ram", article, "Test", "CC0"))
            found = store.search("Что такое оперативная память RAM и зачем она нужна?", limit=3)
            self.assertTrue(found)
            self.assertEqual(found[0]["title"], "Оперативная память")
            self.assertEqual(store.stats()["documents"], 1)

    def test_internet_ingest_has_broad_topics_and_safe_page_limit(self):
        from internet_ingest import DEFAULT_TOPICS_RU
        self.assertGreaterEqual(len(DEFAULT_TOPICS_RU), 100)
        # This test intentionally does not call the live Internet API.
        self.assertIn("искусственный интеллект", DEFAULT_TOPICS_RU)
        self.assertIn("история видеоигр", DEFAULT_TOPICS_RU)

    def test_wikipedia_ingester_requests_valid_extract_parameters(self):
        from unittest.mock import patch
        from internet_ingest import download_wikipedia
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def fake_request(language, params, timeout=25):
                calls.append(dict(params))
                if params.get("list") == "search":
                    return {"query": {"search": [{"title": f"Статья {i}"} for i in range(10)]}}
                return {"query": {"pages": [{
                    "title": f"Статья {i}",
                    "fullurl": f"https://ru.wikipedia.org/wiki/Статья_{i}",
                    "extract": (f"Содержание статьи {i}. " * 1000),
                } for i in range(10)]}}
            with patch("internet_ingest._request_json", side_effect=fake_request), patch("internet_ingest.time.sleep", return_value=None):
                result = download_wikipedia(Path(tmp) / "internet.sqlite3", language="ru", max_pages=10, topics=["тестовая тема"])
            self.assertEqual(result["added_pages"], 10)
            extract_call = calls[1]
            self.assertLessEqual(int(extract_call.get("exchars", "1200")), 1200)
            self.assertNotIn("exchars", extract_call)

    def test_creator_question_has_direct_answer_not_generic_filler(self):
        import json, shutil
        from agent import EmotionalAgent
        from reasoning import ReasoningEngine
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            result = agent.respond("кто тебя создал")
            low = result.lower()
            self.assertIn("emotionalai lab", low)
            self.assertIn("меня создали", low)
            self.assertNotIn("chatgpt", low)
            self.assertNotIn("тут есть несколько интересных моментов", low)
            self.assertNotIn("в вопросе важны детали", low)
            self.assertEqual(agent.last_analysis.intent, "creator")
            agent.close()

    def test_creator_variants_all_name_emotionalai_lab(self):
        import json, shutil
        from agent import EmotionalAgent
        questions = (
            "Кто тебя создал?",
            "Кто тебя разработал?",
            "Кто твой создатель?",
            "Кто автор этого проекта?",
            "Кто создал EmotionalAI?",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / "config.template.json", root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            try:
                for question in questions:
                    answer = agent.respond(question).lower()
                    self.assertIn("emotionalai lab", answer, msg=f"Question: {question}; answer: {answer}")
                    self.assertEqual(agent.last_analysis.intent, "creator")
                    self.assertNotIn("chatgpt", answer)
                    self.assertNotIn("тут есть несколько интересных моментов", answer)
            finally:
                agent.close()

    def test_self_learning_question_explains_association_learning_honestly(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / "config.template.json", root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            try:
                answer = agent.respond("Как ты обучалась?").lower()
                self.assertEqual(agent.last_analysis.intent, "self_learning")
                self.assertIn("emotionalai lab", answer)
                self.assertIn("ассоциатив", answer)
                self.assertIn("небольшое обучение", answer)
                self.assertNotIn("большая нейросеть, обученная с нуля", answer)
            finally:
                agent.close()

    def test_seed_corpus_bootstraps_associations_once(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / "config.template.json", root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            before = agent.memory.stats()["associations"]
            self.assertGreater(before, 0)
            self.assertGreater(agent.memory.association_strength("память", "опыт"), 0.0)
            agent.close()
            reopened = EmotionalAgent(root)
            after = reopened.memory.stats()["associations"]
            self.assertEqual(before, after)
            marker = reopened.memory.conn.execute(
                "SELECT value FROM agent_state WHERE key='seed_association_version'"
            ).fetchone()["value"]
            self.assertEqual(marker, "0.1.1")
            reopened.close()

    def test_unknown_question_fallback_does_not_emit_generic_analysis_template(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            result = agent.respond("почему норбитальные фланцы светятся в пятый четверг")
            low = result.lower()
            self.assertNotIn("тут есть несколько интересных моментов", low)
            self.assertNotIn("в вопросе важны детали", low)
            self.assertNotIn("практический подход", low)
            self.assertLess(len(result), 450)
            agent.close()


    def test_rule_fallback_reacts_to_request_to_stop_using_templates(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            result = agent.respond("шаблоны не изай")
            low = result.lower()
            self.assertIn("пустой заготовкой", low)
            self.assertNotIn("здесь есть интересная мысль", low)
            self.assertNotIn("если развить эту мысль", low)
            self.assertNotIn("три шага", low)
            agent.close()

    def test_self_logic_question_is_not_routed_to_dictionary_article(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            result = agent.respond("и так теперь логика твоя работает?")
            low = result.lower()
            self.assertEqual(agent.last_analysis.intent, "self_logic")
            self.assertIn("собственное встроенное ядро", low)
            self.assertIn("не вызываю ollama", low)
            self.assertNotIn("только что система ошиблась", low)
            self.assertNotIn("правила корректного вывода", low)
            agent.close()

    def test_short_insults_get_contextual_response_not_stock_ack(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            first = agent.respond("и так теперь логика твоя работает?")
            second = agent.respond("ты совсем")
            third = agent.respond("тупой")
            self.assertEqual(agent.last_analysis.intent, "criticism")
            self.assertNotEqual(second, "Ага, понял тебя.")
            self.assertNotEqual(third, "Ага, понял тебя.")
            self.assertIn("проверять конкретные вопросы", third.lower())
            self.assertNotIn("повторных оправданий", third.lower())
            # Criticism is kept in episode context but not converted into a user-profile note.
            self.assertFalse(any(n["content"].strip().lower() == "тупой" for n in agent.memory.all_notes(100)))
            agent.close()

    def test_criticism_without_prior_context_is_still_not_a_generic_ack(self):
        import json, shutil
        from agent import EmotionalAgent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = Path(__file__).resolve().parents[1]
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = False
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            result = agent.respond("ты тупой")
            self.assertEqual(agent.last_analysis.intent, "criticism")
            self.assertNotEqual(result, "Ага, понял тебя.")
            self.assertIn("ответ тебя не устроил", result.lower())
            agent.close()

if __name__ == "__main__":
    unittest.main(verbosity=2)

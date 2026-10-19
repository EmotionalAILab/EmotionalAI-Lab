"""Network tools are tested with fake API responses; the suite never calls the live Internet."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import json
import shutil
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import online_tools
from agent import EmotionalAgent


class OnlineToolTests(unittest.TestCase):
    def test_wikipedia_lookup_returns_excerpt_and_citation(self):
        search_payload = {"query": {"search": [{"title": "Квантовая химия", "pageid": 123}]}}
        page_payload = {"query": {"pages": [{
            "title": "Квантовая химия",
            "extract": "Квантовая химия применяет квантовую механику для описания структуры и свойств молекул. " * 4,
            "fullurl": "https://ru.wikipedia.org/wiki/Квантовая_химия",
        }]}}
        with patch("online_tools._get_json", side_effect=[search_payload, page_payload]) as request:
            result = online_tools.wikipedia_lookup("Что такое квантовая химия?")
        self.assertIn("Квантовая химия", result)
        self.assertIn("Источник: https://ru.wikipedia.org/wiki/Квантовая_химия", result)
        self.assertEqual(request.call_count, 2)

    def test_wikipedia_lookup_does_not_send_personal_query(self):
        with patch("online_tools._get_json") as request:
            result = online_tools.wikipedia_lookup("Какой мой любимый цвет?")
        self.assertIsNone(result)
        request.assert_not_called()

    def test_wikipedia_lookup_ignores_live_weather_questions(self):
        with patch("online_tools._get_json") as request:
            result = online_tools.wikipedia_lookup("Какая погода в Амстердаме?")
        self.assertIsNone(result)
        request.assert_not_called()

    def test_weather_answer_formats_current_conditions(self):
        geocoding = {"results": [{
            "name": "Амстердам", "country": "Нидерланды", "latitude": 52.37, "longitude": 4.89,
        }]}
        forecast = {"current": {
            "temperature_2m": 13.2, "relative_humidity_2m": 78, "apparent_temperature": 11.5,
            "weather_code": 3, "wind_speed_10m": 14.4, "time": "2026-10-19T09:00",
        }}
        with patch("online_tools._get_json", side_effect=[geocoding, forecast]) as request:
            result = online_tools.weather_answer("Какая погода в Амстердаме?")
        self.assertIn("Амстердам", result)
        self.assertIn("пасмурно", result)
        self.assertIn("13.2 °C", result)
        self.assertIn("Open-Meteo", result)
        self.assertEqual(request.call_count, 2)

    def test_agent_uses_wikipedia_fallback_for_unknown_factual_question(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = True
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            try:
                with patch("agent.wikipedia_lookup", return_value="По статье Wikipedia «Экзополис»: тестовый источник.\n\nИсточник: https://ru.wikipedia.org/wiki/Test") as lookup:
                    answer = agent.respond("Что такое экзополис?")
                self.assertIn("Экзополис", answer)
                self.assertIn("Источник:", answer)
                lookup.assert_called_once()
            finally:
                agent.close()

    def test_agent_routes_current_weather_to_weather_source(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(project / ("config.json" if (project / "config.json").exists() else "config.template.json"), root / "config.json")
            shutil.copytree(project / "data", root / "data")
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["database_path"] = "data/test.sqlite3"
            config["online_wikipedia_fallback"] = True
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            agent = EmotionalAgent(root)
            try:
                with patch("agent.weather_answer", return_value="Сейчас в Амстердаме 13 °C.") as weather:
                    answer = agent.respond("Какая погода в Амстердаме?")
                self.assertIn("13 °C", answer)
                self.assertEqual(agent.last_analysis.intent, "current_weather")
                weather.assert_called_once()
            finally:
                agent.close()

    def test_weather_requires_a_city(self):
        self.assertIn("напиши город", online_tools.weather_answer("Какая погода сейчас?").lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Графический интерфейс Emotional: чат и отдельный редактор личности/промпта."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
from pathlib import Path
from agent import EmotionalAgent, DEFAULT_PERSONA

ROOT_DIR = Path(__file__).resolve().parent


class EmotionalAIWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Emotional 0.1.1 · EmotionalAI Lab")
        self.root.geometry("900x720")
        self.root.minsize(650, 500)
        self.root.configure(bg="#f2f4f7")
        self.agent = EmotionalAgent(ROOT_DIR)
        self._busy = False
        self._build_ui()
        self._append(self.agent.name, "Привет! О чём поговорим?")
        self.entry.focus_set()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        heading = ttk.Frame(outer)
        heading.pack(fill="x", pady=(0, 10))
        self.title_label = ttk.Label(heading, text=self.agent.name, font=("Segoe UI", 19, "bold"))
        self.title_label.pack(anchor="w")
        ttk.Label(heading, text="Чат", foreground="#555555").pack(anchor="w", pady=(2, 0))

        chat_frame = ttk.Frame(outer)
        chat_frame.pack(fill="both", expand=True)
        self.chat = tk.Text(
            chat_frame, wrap="word", state="disabled", font=("Segoe UI", 11),
            padx=14, pady=12, borderwidth=0, background="white", foreground="#202124",
            insertbackground="#202124", spacing1=3, spacing3=8,
        )
        scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=scrollbar.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.chat.tag_configure("user", foreground="#155eef", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.chat.tag_configure("agent", foreground="#16803c", font=("Segoe UI", 10, "bold"), spacing1=8)
        self.chat.tag_configure("body", foreground="#202124")

        quick = ttk.Frame(outer)
        quick.pack(fill="x", pady=(10, 8))
        for label, command in (
            ("Состояние", "/состояние"),
            ("История", "/история"),
            ("Память", "/память"),
            ("Цели", "/цели"),
        ):
            ttk.Button(quick, text=label, command=lambda c=command: self._send_text(c)).pack(side="left", padx=(0, 6))
        ttk.Button(quick, text="Личность / промпт", command=self._open_persona_editor).pack(side="right", padx=(6, 0))
        ttk.Button(quick, text="Данные из интернета", command=self._open_knowledge_editor).pack(side="right", padx=(6, 0))
        ttk.Button(quick, text="О ядре", command=self._open_core_info).pack(side="right")

        self.model_status = ttk.Label(outer, text=self.agent.core_status_text(), foreground="#666666", wraplength=840)
        self.model_status.pack(anchor="w", pady=(0, 5))
        ttk.Label(outer, text="Emotional 0.1.1", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        entry_row = ttk.Frame(outer)
        entry_row.pack(fill="x")
        self.entry = ttk.Entry(entry_row, font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        self.entry.bind("<Return>", self._on_enter)
        self.send_button = ttk.Button(entry_row, text="Отправить", command=self._send)
        self.send_button.pack(side="right", ipady=5)
        ttk.Label(outer, text="Emotional работает самостоятельно: внешний ИИ не требуется. Имя и характер настраиваются отдельно; тексты можно добавить в локальную базу знаний.", foreground="#666666", wraplength=840).pack(anchor="w", pady=(8, 0))

    def _append(self, speaker: str, text: str):
        self.chat.configure(state="normal")
        tag = "user" if speaker == "Ты" else "agent"
        self.chat.insert("end", f"{speaker}:\n", tag)
        self.chat.insert("end", text + "\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _on_enter(self, _event):
        self._send()
        return "break"

    def _send_text(self, text: str):
        if self._busy:
            return
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self._send()

    def _send(self):
        if self._busy:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._append("Ты", text)
        self._busy = True
        self.send_button.configure(state="disabled", text="Думаю…")
        self.entry.configure(state="disabled")
        worker = threading.Thread(target=self._process_in_background, args=(text,), daemon=True)
        worker.start()

    def _process_in_background(self, text: str):
        try:
            answer = self.agent.respond(text)
            error = None
        except Exception as exc:
            answer = ""
            error = str(exc)
        try:
            self.root.after(0, lambda: self._finish_response(text, answer, error))
        except tk.TclError:
            pass

    def _finish_response(self, user_text: str, answer: str, error: str | None):
        self._busy = False
        try:
            self.entry.configure(state="normal")
            self.send_button.configure(state="normal", text="Отправить")
        except tk.TclError:
            return
        if error:
            self._append("Ошибка", f"Не получилось обработать сообщение: {error}")
            messagebox.showerror("Emotional", error, parent=self.root)
        elif answer:
            self._append(self.agent.name, answer)
        self.model_status.configure(text=self.agent.core_status_text())
        if user_text.lower().strip() in ("/выход", "/exit", "пока", "до свидания", "бай"):
            self.root.after(250, self.close)
        else:
            self.entry.focus_set()

    def _open_core_info(self):
        win = tk.Toplevel(self.root)
        win.title("Автономное ядро — Emotional")
        win.geometry("700x600")
        win.minsize(580, 460)
        win.transient(self.root)
        frame = ttk.Frame(win, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Emotional — самостоятельный агент", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text=("Обычные сообщения обрабатываются встроенным кодом проекта. Программа не подключается к Ollama, "
                  "Qwen, ChatGPT API или другому ИИ-агенту. После загрузки или импорта тексты сохраняются локально "
                  "и ищутся без внешнего сервиса."),
            wraplength=650, justify="left"
        ).pack(anchor="w", pady=(8, 12))
        status = tk.Text(frame, wrap="word", height=19, font=("Segoe UI", 10), padx=10, pady=10)
        status.pack(fill="both", expand=True)
        status.insert("1.0", self.agent.core_info_text())
        status.configure(state="disabled")
        ttk.Button(frame, text="Закрыть", command=win.destroy).pack(anchor="e", pady=(12, 0))

    def _open_knowledge_editor(self):
        win = tk.Toplevel(self.root)
        win.title("Интернет-знания — Emotional")
        win.geometry("820x700")
        win.minsize(680, 560)
        win.transient(self.root)

        frame = ttk.Frame(win, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Пополнить базу знаний из интернета", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text=("Загрузчик скачивает статьи Wikipedia и сохраняет их на этом компьютере. Автономное ядро Emotional "
                  "ищет подходящие фрагменты в этой локальной базе и использует найденные сведения непосредственно "
                  "в ответе. Никакая внешняя языковая модель для этого не нужна. Можно также импортировать свои "
                  "текстовые документы; после импорта поиск работает локально."),
            wraplength=760, justify="left"
        ).pack(anchor="w", pady=(5, 10))

        options = ttk.Frame(frame)
        options.pack(fill="x", pady=(0, 8))
        ttk.Label(options, text="Язык Wikipedia").grid(row=0, column=0, sticky="w", padx=(0, 8))
        language_var = tk.StringVar(value="Русский (ru)")
        language_box = ttk.Combobox(options, textvariable=language_var, state="readonly", values=["Русский (ru)", "English (en)"], width=18)
        language_box.grid(row=0, column=1, sticky="w", padx=(0, 20))
        ttk.Label(options, text="Сколько статей максимум").grid(row=0, column=2, sticky="w", padx=(0, 8))
        pages_var = tk.StringVar(value="1000")
        ttk.Entry(options, textvariable=pages_var, width=10).grid(row=0, column=3, sticky="w")
        online_lookup_var = tk.BooleanVar(value=bool(self.agent.config.get("online_wikipedia_fallback", True)))

        def save_online_lookup():
            self.agent.config["online_wikipedia_fallback"] = bool(online_lookup_var.get())
            try:
                (ROOT_DIR / "config.json").write_text(
                    json.dumps(self.agent.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            except OSError as exc:
                messagebox.showerror("Emotional", f"Не удалось сохранить настройку: {exc}", parent=win)

        ttk.Checkbutton(
            frame,
            text="Разрешить поиск источников в интернете для неизвестных вопросов и погоды (без обращения к ИИ)",
            variable=online_lookup_var,
            command=save_online_lookup,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(frame, text="Темы для поиска (по одной на строку; можно редактировать)").pack(anchor="w")
        topic_box_frame = ttk.Frame(frame)
        topic_box_frame.pack(fill="both", expand=True, pady=(3, 8))
        topics_text = tk.Text(topic_box_frame, height=14, wrap="word", font=("Segoe UI", 9), padx=8, pady=6, undo=True)
        topic_scroll = ttk.Scrollbar(topic_box_frame, orient="vertical", command=topics_text.yview)
        topics_text.configure(yscrollcommand=topic_scroll.set)
        topics_text.pack(side="left", fill="both", expand=True)
        topic_scroll.pack(side="right", fill="y")
        topics_path = ROOT_DIR / "data" / "internet_topics_ru.txt"
        try:
            topics_text.insert("1.0", topics_path.read_text(encoding="utf-8"))
        except OSError:
            topics_text.insert("1.0", "искусственный интеллект\nпрограммирование\nистория\nфизика\nматематика\n")
        last_language_default = {"value": topics_text.get("1.0", "end").strip()}

        def update_default_topics(_event=None):
            current_text = topics_text.get("1.0", "end").strip()
            if current_text != last_language_default["value"]:
                return  # don't erase edits the user has already made
            new_language = "ru" if language_var.get().startswith("Русский") else "en"
            default_path = ROOT_DIR / "data" / ("internet_topics_ru.txt" if new_language == "ru" else "internet_topics_en.txt")
            try:
                new_text = default_path.read_text(encoding="utf-8").strip()
            except OSError:
                return
            topics_text.delete("1.0", "end")
            topics_text.insert("1.0", new_text)
            last_language_default["value"] = new_text

        language_box.bind("<<ComboboxSelected>>", update_default_topics)

        current_stats = self.agent.web_knowledge.stats()
        initial_stats = f"В базе: {current_stats['documents']} статей; {current_stats['chunks']} фрагментов; {current_stats['characters']:,} символов текста.".replace(",", " ")
        status_var = tk.StringVar(value=initial_stats)
        status = ttk.Label(frame, textvariable=status_var, wraplength=760, foreground="#555555", justify="left")
        status.pack(anchor="w", pady=(0, 8))

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        download_button = ttk.Button(buttons, text="Загрузить статьи из Wikipedia")
        download_button.pack(side="left")
        train_button = ttk.Button(buttons, text="Обучить на загруженных статьях")
        train_button.pack(side="left", padx=(8, 0))
        import_button = ttk.Button(buttons, text="Добавить .txt/.md файлы")
        import_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Закрыть", command=win.destroy).pack(side="right")

        def set_status(message: str):
            try:
                self.root.after(0, lambda msg=message: status_var.set(msg))
            except tk.TclError:
                pass

        def finish_buttons(message: str):
            def apply():
                status_var.set(message)
                download_button.configure(state="normal", text="Загрузить статьи из Wikipedia")
                train_button.configure(state="normal", text="Обучить на загруженных статьях")
                import_button.configure(state="normal", text="Добавить .txt/.md файлы")
            try:
                self.root.after(0, apply)
            except tk.TclError:
                pass

        def start_download():
            try:
                page_count = int(pages_var.get().strip())
                if not 10 <= page_count <= 5000:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Неверное количество", "Укажи число от 10 до 5000 статей.", parent=win)
                return
            language = "ru" if language_var.get().startswith("Русский") else "en"
            topic_list = [line.strip() for line in topics_text.get("1.0", "end").splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if not topic_list:
                messagebox.showerror("Нет тем", "Добавь хотя бы одну тему для поиска.", parent=win)
                return
            download_button.configure(state="disabled", text="Загрузка…")
            train_button.configure(state="disabled")
            import_button.configure(state="disabled")
            status_var.set("Подключаюсь к Wikipedia. Загрузка идёт в фоне; окно чата не должно зависать…")

            def worker():
                try:
                    from internet_ingest import download_wikipedia
                    db_path = ROOT_DIR / "data" / "internet_knowledge.sqlite3"
                    result = download_wikipedia(db_path, language, page_count, topic_list, progress=set_status)
                    set_status("Статьи загружены. Теперь обновляю ассоциации Emotional на основе новых текстов…")
                    trained = self.agent.train_from_wikipedia(max_documents=min(1000, max(page_count, result['added_pages'])), progress=set_status)
                    summary = (
                        f"Готово: добавлено {result['added_pages']} новых статей; пропущено {result['skipped_pages']}. "
                        f"Обучено ассоциаций по {trained['trained_documents']} статьям; обработано {trained['association_updates']} пар слов. "
                        f"Всего в базе {result['total_pages']} статей / {result['total_chunks']} фрагментов, "
                        f"{result['total_characters']:,} символов текста. Команда /база покажет статистику."
                    ).replace(",", " ")
                    finish_buttons(summary)
                except Exception as exc:
                    finish_buttons(f"Загрузка не завершена: {exc}. Уже полученные статьи сохранены; можно повторить позже.")
            threading.Thread(target=worker, daemon=True).start()

        def start_training():
            download_button.configure(state="disabled")
            train_button.configure(state="disabled", text="Обучение…")
            import_button.configure(state="disabled")
            status_var.set("Обучаю ассоциативную память на уже загруженных статьях Wikipedia…")

            def worker():
                try:
                    result = self.agent.train_from_wikipedia(max_documents=1000, progress=set_status)
                    summary = (
                        f"Обучение завершено: новые ассоциации обновлены по {result['trained_documents']} статьям; "
                        f"обработано {result['association_updates']} пар слов. "
                        f"Необученных статей осталось примерно: {result['remaining_documents']}. "
                        "Факты и история диалогов не удалялись."
                    )
                    finish_buttons(summary)
                except Exception as exc:
                    finish_buttons(f"Обучение не завершено: {exc}. Уже сохранённые ассоциации остаются в базе.")
            threading.Thread(target=worker, daemon=True).start()

        def start_import():
            selected = filedialog.askopenfilenames(
                parent=win,
                title="Выбери текстовые файлы для локальной базы знаний",
                filetypes=[("Текстовые файлы", "*.txt *.md"), ("Все файлы", "*.*")],
            )
            if not selected:
                return
            download_button.configure(state="disabled")
            train_button.configure(state="disabled")
            import_button.configure(state="disabled", text="Импорт…")
            status_var.set(f"Импортирую файлов: {len(selected)}…")

            def worker():
                added = 0
                errors = []
                try:
                    from knowledge_store import WebKnowledgeStore
                    store = WebKnowledgeStore(ROOT_DIR / "data" / "internet_knowledge.sqlite3")
                    for file_name in selected:
                        try:
                            added += store.import_text_file(file_name)
                        except Exception as exc:
                            errors.append(f"{Path(file_name).name}: {exc}")
                    stats = store.stats()
                    summary = f"Импорт завершён: добавлено {added} частей. Всего {stats['documents']} документов / {stats['chunks']} фрагментов."
                    if errors:
                        summary += " Ошибки: " + "; ".join(errors[:3])
                    finish_buttons(summary)
                except Exception as exc:
                    finish_buttons(f"Ошибка импорта: {exc}")
            threading.Thread(target=worker, daemon=True).start()

        download_button.configure(command=start_download)
        train_button.configure(command=start_training)
        import_button.configure(command=start_import)

    def _open_persona_editor(self):
        if self._busy:
            messagebox.showinfo("Emotional", "Дождись окончания текущего ответа, прежде чем менять личность.", parent=self.root)
            return
        win = tk.Toplevel(self.root)
        win.title("Личность и промпт — Emotional")
        win.geometry("760x690")
        win.minsize(600, 520)
        win.transient(self.root)
        win.grab_set()

        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Настрой личность", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Имя, манера общения и базовые правила поведения сохраняются в persona.json.").pack(anchor="w", pady=(3, 12))

        ttk.Label(frame, text="Имя агента").pack(anchor="w")
        name_var = tk.StringVar(value=self.agent.name)
        name_entry = ttk.Entry(frame, textvariable=name_var, font=("Segoe UI", 10))
        name_entry.pack(fill="x", pady=(3, 10), ipady=4)

        choices = ttk.Frame(frame)
        choices.pack(fill="x", pady=(0, 10))
        tone_frame = ttk.Frame(choices)
        tone_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Label(tone_frame, text="Тон общения").pack(anchor="w")
        tone_var = tk.StringVar(value=self.agent.response_tone)
        tone = ttk.Combobox(tone_frame, textvariable=tone_var, state="readonly", values=["Авто по промпту", "Дружелюбный", "Спокойный", "Любопытный", "Ироничный", "Серьёзный"])
        tone.pack(fill="x", pady=(3, 0))

        detail_frame = ttk.Frame(choices)
        detail_frame.pack(side="left", fill="x", expand=True)
        ttk.Label(detail_frame, text="Подробность ответов").pack(anchor="w")
        detail_var = tk.StringVar(value=self.agent.response_detail)
        detail = ttk.Combobox(detail_frame, textvariable=detail_var, state="readonly", values=["Кратко", "Обычно", "Подробно", "Очень подробно"])
        detail.pack(fill="x", pady=(3, 0))

        ttk.Label(frame, text="Промпт / описание личности").pack(anchor="w")
        prompt_frame = ttk.Frame(frame)
        prompt_frame.pack(fill="both", expand=True, pady=(3, 6))
        prompt_text = tk.Text(prompt_frame, wrap="word", font=("Segoe UI", 10), padx=9, pady=8, undo=True)
        prompt_scroll = ttk.Scrollbar(prompt_frame, orient="vertical", command=prompt_text.yview)
        prompt_text.configure(yscrollcommand=prompt_scroll.set)
        prompt_text.pack(side="left", fill="both", expand=True)
        prompt_scroll.pack(side="right", fill="y")
        prompt_text.insert("1.0", self.agent.personality_prompt)

        ttk.Label(
            frame,
            text="Опиши характер, манеру речи, интересы, стиль юмора, отношение к собеседнику и правила ответов. Имя, тон, подробность и текст промпта сохраняются между запусками.",
            foreground="#666666", wraplength=700, justify="left"
        ).pack(anchor="w", pady=(0, 12))

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")

        def save():
            try:
                self.agent.save_persona(name_var.get(), prompt_text.get("1.0", "end-1c"), tone_var.get(), detail_var.get())
            except OSError as exc:
                messagebox.showerror("Не удалось сохранить", str(exc), parent=win)
                return
            self.root.title("Emotional 0.1.1 · EmotionalAI Lab")
            self.title_label.configure(text=self.agent.name)
            win.destroy()

        def reset():
            name_var.set(DEFAULT_PERSONA["name"])
            tone_var.set("Дружелюбный")
            detail_var.set("Подробно")
            prompt_text.delete("1.0", "end")
            prompt_text.insert("1.0", DEFAULT_PERSONA["prompt"])

        ttk.Button(buttons, text="Сбросить настройки", command=reset).pack(side="left")
        ttk.Button(buttons, text="Отмена", command=win.destroy).pack(side="right", padx=(0, 8))
        ttk.Button(buttons, text="Сохранить", command=save).pack(side="right")
        name_entry.focus_set()

    def close(self):
        if self._busy:
            messagebox.showinfo("Emotional", "Ответ ещё генерируется. Закрой программу после его завершения.", parent=self.root)
            return
        try:
            self.agent.close()
        finally:
            self.root.destroy()


def main() -> int:
    root = tk.Tk()
    EmotionalAIWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()

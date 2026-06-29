import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

# Import should not fail regardless of Pillow availability
from pyopl import pyopl_ide_bootstrap
from pyopl.pyopl_ide_bootstrap import OPLIDE


class TestPyOPLIDETyping(unittest.TestCase):
    def test_schedule_highlight_skips_large_model_text(self):
        class DummyText:
            def __init__(self):
                self.removed = []

            def count(self, *_args):
                return (pyopl_ide_bootstrap.MAX_HIGHLIGHT_CHARS + 1,)

            def tag_remove(self, tag, start, end):
                self.removed.append((tag, start, end))

        class DummyStatus:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        dummy = SimpleNamespace(
            _shutting_down=False,
            _highlight_after_ids={},
            _last_syntax_error_by_widget={},
            _last_syntax_error="old error",
            status_syntax_var=DummyStatus(),
            after=lambda *args, **kwargs: "after-id",
            after_cancel=lambda _after_id: None,
        )
        dummy._cancel_scheduled_highlight = lambda text_widget, kind: OPLIDE._cancel_scheduled_highlight(
            dummy, text_widget, kind
        )
        dummy._text_too_large_for_highlight = lambda text_widget: OPLIDE._text_too_large_for_highlight(dummy, text_widget)
        dummy._clear_highlight_tags = lambda text_widget: OPLIDE._clear_highlight_tags(dummy, text_widget)
        dummy._disable_highlight_for_large_text = lambda text_widget: OPLIDE._disable_highlight_for_large_text(
            dummy, text_widget
        )
        text = DummyText()

        OPLIDE._schedule_highlight(dummy, text, is_data=False)

        self.assertEqual(dummy._highlight_after_ids, {})
        self.assertIsNone(dummy._last_syntax_error_by_widget[id(text)])
        self.assertIsNone(dummy._last_syntax_error)
        self.assertEqual(dummy.status_syntax_var.value, "Syntax validation disabled for large text")
        self.assertTrue(any(tag == "ERROR" for tag, _start, _end in text.removed))

    def test_update_caret_preserves_large_text_status(self):
        class DummyText:
            def count(self, *_args):
                return (pyopl_ide_bootstrap.MAX_HIGHLIGHT_CHARS + 1,)

            def winfo_exists(self):
                return True

            def index(self, _index):
                return "7.3"

        class DummyVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        dummy = SimpleNamespace(
            status_caret_var=DummyVar(),
            status_syntax_var=DummyVar(),
            status_solver_var=DummyVar(),
            status_genai_var=DummyVar(),
            solver=SimpleNamespace(get=lambda: "gurobi"),
            genai_provider=None,
            genai_model=None,
            genai_method_var=SimpleNamespace(get=lambda: "pyopl_generative"),
            _genai_methods=[("SyntAGM", "pyopl_generative")],
        )
        dummy._text_too_large_for_highlight = lambda text_widget: OPLIDE._text_too_large_for_highlight(dummy, text_widget)
        dummy._refresh_status_context = lambda: OPLIDE._refresh_status_context(dummy)
        dummy._label_for_method = lambda key: OPLIDE._label_for_method(dummy, key)

        OPLIDE._update_caret_position(dummy, DummyText())

        self.assertEqual(dummy.status_caret_var.value, "Ln 7, Col 3")
        self.assertEqual(dummy.status_syntax_var.value, "Syntax validation disabled for large text")

    def test_update_caret_shows_stored_eof_syntax_error_without_error_tag(self):
        class DummyText:
            def __init__(self):
                self.widget_id = id(self)

            def count(self, *_args):
                return (100,)

            def winfo_exists(self):
                return True

            def index(self, _index):
                return "13.0"

            def tag_ranges(self, _tag):
                return []

        class DummyVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        text = DummyText()
        dummy = SimpleNamespace(
            status_caret_var=DummyVar(),
            status_syntax_var=DummyVar(),
            status_solver_var=DummyVar(),
            status_genai_var=DummyVar(),
            solver=SimpleNamespace(get=lambda: "gurobi"),
            genai_provider=None,
            genai_model=None,
            genai_method_var=SimpleNamespace(get=lambda: "pyopl_generative"),
            _genai_methods=[("SyntAGM", "pyopl_generative")],
            _last_syntax_error_by_widget={
                id(text): "Parser Error on line 13: Semantic Error (Line 13): Syntax error in .dat file at end of file (EOF)."
            },
            _last_syntax_error=None,
        )
        dummy._text_too_large_for_highlight = lambda text_widget: OPLIDE._text_too_large_for_highlight(dummy, text_widget)
        dummy._refresh_status_context = lambda: OPLIDE._refresh_status_context(dummy)
        dummy._label_for_method = lambda key: OPLIDE._label_for_method(dummy, key)

        OPLIDE._update_caret_position(dummy, text)

        self.assertIn("Syntax error in .dat file at end of file", dummy.status_syntax_var.value)

    def test_autohide_scrollbar_idle_refresh_passes_string_fractions(self):
        class DummyWidget:
            def __init__(self):
                self.yscrollcommand = None

            def configure(self, **kwargs):
                self.yscrollcommand = kwargs["yscrollcommand"]

            def yview(self):
                return (0.0, 1.0)

        class DummyScrollbar:
            def __init__(self):
                self.set_calls = []
                self.removed = False

            def winfo_manager(self):
                return "grid"

            def set(self, first, last):
                self.set_calls.append((first, last))

            def grid(self):
                self.removed = False

            def grid_remove(self):
                self.removed = True

        widget = DummyWidget()
        scrollbar = DummyScrollbar()
        dummy = SimpleNamespace(after_idle=lambda callback: callback())

        OPLIDE._bind_autohide_vertical_scrollbar(dummy, widget, scrollbar)

        self.assertEqual(scrollbar.set_calls, [("0.0", "1.0")])
        self.assertTrue(scrollbar.removed)

    def test_apply_theme_colors_keeps_genai_button_font_independent_of_editor_font_size(self):
        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class DummyStyle:
            def __init__(self):
                self.configured = {}
                self.mapped = {}
                self.layouts = {}

            def configure(self, style_name, **kwargs):
                self.configured[style_name] = kwargs

            def map(self, style_name, **kwargs):
                self.mapped[style_name] = kwargs

            def layout(self, style_name, layout_spec=None):
                if layout_spec is None:
                    return self.layouts.get(style_name, [])
                self.layouts[style_name] = layout_spec

        class DummyWidget:
            def __init__(self):
                self.calls = []

            def config(self, **kwargs):
                self.calls.append(kwargs)

            def tag_configure(self, *_args, **_kwargs):
                pass

        dummy = SimpleNamespace(
            theme_var=DummyVar("flatly"),
            style=DummyStyle(),
            interface_font_family="TkDefaultFont",
            interface_button_font="TkDefaultFont",
            current_font_size=20,
            configure=lambda **_kwargs: None,
            _apply_macos_theme_appearance=lambda _theme: None,
            _configure_tk_scrollbar=lambda *_args, **_kwargs: None,
            _strip_focus_from_ttk_layout=lambda layout: layout,
            model_text=DummyWidget(),
            data_text=DummyWidget(),
            output_text=DummyWidget(),
        )

        OPLIDE._apply_theme_colors(dummy)

        self.assertEqual(dummy.style.configured["GenaiMode.TButton"]["font"], "TkDefaultFont")
        self.assertEqual(dummy.style.configured["GenaiModeActive.TButton"]["font"], "TkDefaultFont")

    def test_populate_genai_model_menus_preserves_current_selection(self):
        class DummyVar:
            def __init__(self, value=None):
                self.value = value

            def set(self, value):
                self.value = value

            def get(self):
                return self.value

        class DummyMenu:
            def __init__(self, *args, **kwargs):
                self.items = []

            def delete(self, *_args):
                self.items.clear()

            def add_cascade(self, **kwargs):
                self.items.append(("cascade", kwargs))

            def add_radiobutton(self, **kwargs):
                self.items.append(("radiobutton", kwargs))

            def add_separator(self):
                self.items.append(("separator", {}))

            def add_checkbutton(self, **kwargs):
                self.items.append(("checkbutton", kwargs))

            def add_command(self, **kwargs):
                self.items.append(("command", kwargs))

        dummy = SimpleNamespace(
            _shutting_down=False,
            genai_menu=DummyMenu(),
            menubar=SimpleNamespace(entryconfig=lambda *args, **kwargs: None),
            genai_selection_var=DummyVar("openai|custom-model"),
            genai_method_var=DummyVar("pyopl_generative"),
            show_genai_panel_var=DummyVar(True),
            verbose_llm_var=DummyVar(False),
            _genai_methods=[("SyntAGM", "pyopl_generative")],
            _desired_genai_provider="openai",
            _desired_genai_model="gpt-5.4",
            genai_provider="openai",
            genai_model="custom-model",
            debug=False,
            _active_operation=None,
            _refresh_genai_panel_state=lambda: None,
            _save_settings=lambda: None,
            _toggle_genai_panel_visibility=lambda: None,
            _genai_solve_and_explain=lambda: None,
            interrupt_active_operation=lambda: None,
            _accel=lambda key: f"Ctrl+{key}",
        )
        dummy._make_select_model_cmd = lambda provider_key, model_name: (lambda: None)
        dummy._make_select_genai_method_cmd = lambda key: (lambda: None)
        dummy._on_select_genai_model = lambda provider_key, model_name: OPLIDE._on_select_genai_model(
            dummy, provider_key, model_name
        )

        provider_models = {"openai": ["gpt-5.4", "custom-model"], "google": [], "ollama": []}

        with mock.patch.object(pyopl_ide_bootstrap.tk, "Menu", DummyMenu):
            OPLIDE._populate_genai_model_menus(dummy, provider_models)

        self.assertEqual(dummy.genai_provider, "openai")
        self.assertEqual(dummy.genai_model, "custom-model")
        self.assertEqual(dummy.genai_selection_var.get(), "openai|custom-model")

    def test_pillow_optional_imports_exist(self):
        # Module should define these attributes
        self.assertTrue(hasattr(pyopl_ide_bootstrap, "PILImage"))
        self.assertTrue(hasattr(pyopl_ide_bootstrap, "PILImageTk"))

    def test_index_from_pos_mapping(self):
        s = "hello\nworld"
        # pos 0 => line 1, col 0
        self.assertEqual(OPLIDE._index_from_pos(None, s, 0), "1.0")
        # after 'hello' (5), at newline index 5 => still line 1, col 5
        self.assertEqual(OPLIDE._index_from_pos(None, s, 5), "1.5")
        # index 6 is start of second line 'w' => line 2, col 0
        self.assertEqual(OPLIDE._index_from_pos(None, s, 6), "2.0")
        # end of string len=11 => line 2, col len('world')=5
        self.assertEqual(OPLIDE._index_from_pos(None, s, len(s)), "2.5")

    def test_append_output_can_target_non_current_session(self):
        dummy = SimpleNamespace(
            _output_sessions={},
            _output_session_ids=[],
            _output_session_display={},
            _current_output_session_id=None,
            _viewing_output_session_id=None,
            _save_session=lambda: None,
            _show_output_session=lambda session_id: None,
        )

        first = OPLIDE._begin_new_output_session(dummy, "First")
        second = OPLIDE._begin_new_output_session(dummy, "Second")

        OPLIDE._append_output(dummy, "\nfirst-session-update\n", first)

        self.assertIn("first-session-update", dummy._output_sessions[first])
        self.assertNotIn("first-session-update", dummy._output_sessions[second])

    def test_begin_new_output_session_initializes_missing_artifact_metadata(self):
        dummy = SimpleNamespace(
            _output_sessions={},
            _output_session_ids=[],
            _output_session_display={},
            _current_output_session_id=None,
            _viewing_output_session_id=None,
            _save_session=lambda: None,
            _show_output_session=lambda session_id: None,
        )

        session_id = OPLIDE._begin_new_output_session(dummy, "Solve: Solving model...")

        self.assertIn(session_id, dummy._output_session_timestamp)
        self.assertIn(session_id, dummy._output_session_artifacts)
        self.assertEqual(dummy._output_session_artifacts[session_id], {})

    def test_begin_new_output_session_snapshots_current_editor_state(self):
        class DummyText:
            def __init__(self, content):
                self.content = content

            def get(self, *_args):
                return self.content

        dummy = SimpleNamespace(
            _output_sessions={},
            _output_session_ids=[],
            _output_session_display={},
            _current_output_session_id=None,
            _viewing_output_session_id=None,
            _save_session=lambda: None,
            _show_output_session=lambda session_id: None,
            model_text=DummyText("dvar int x;\n"),
            data_text=DummyText("x = 3;\n"),
        )

        session_id = OPLIDE._begin_new_output_session(dummy, "Solve: Solving model...")
        artifacts = dummy._output_session_artifacts[session_id]

        self.assertEqual(artifacts["model_text"], "dvar int x;")
        self.assertEqual(artifacts["data_text"], "x = 3;")

    def test_begin_new_output_session_disambiguates_same_second_display_labels(self):
        dummy = SimpleNamespace(
            _output_sessions={},
            _output_session_ids=[],
            _output_session_display={},
            _current_output_session_id=None,
            _viewing_output_session_id=None,
            _save_session=lambda: None,
            _show_output_session=lambda session_id: None,
        )

        first = OPLIDE._make_output_session_display(dummy, "2026-05-03 12:00:00", "Ask")
        dummy._output_session_display["session-1"] = first
        second = OPLIDE._make_output_session_display(dummy, "2026-05-03 12:00:00", "Ask")

        self.assertEqual(first, "2026-05-03 12:00:00 • Ask")
        self.assertEqual(second, "2026-05-03 12:00:00 • Ask (2)")

    def test_apply_pending_genai_revisions_persists_inline_session_snapshot(self):
        class DummyText:
            def __init__(self, content):
                self.content = content

            def get(self, *_args):
                return self.content

            def delete(self, *_args):
                self.content = ""

            def insert(self, *_args):
                self.content = _args[-1]

        class DummyNotebook:
            def tab(self, *_args, **_kwargs):
                pass

        with TemporaryDirectory() as tmpdir:
            original_data = Path(tmpdir) / "data.dat"
            original_data.write_text("old = 1;\n", encoding="utf-8")
            original_model = Path(tmpdir) / "model.mod"
            original_model.write_text("dvar int x;\n", encoding="utf-8")

            log = []
            artifact_calls = []
            dummy = SimpleNamespace(
                _genai_pending_revisions={
                    "revised_model": "dvar int y;\n",
                    "revised_data": "y = 2;\n",
                    "model_path": str(original_model),
                    "data_path": str(original_data),
                    "safe_ts": "2026-05-03_01-20-41",
                    "had_data_file": True,
                    "session_id": "session-1",
                },
                model_file=str(original_model),
                data_file=str(original_data),
                model_text=DummyText("dvar int x;\n"),
                data_text=DummyText("old = 1;\n"),
                editor_notebook=DummyNotebook(),
                model_frame=object(),
                data_frame=object(),
                status_var=SimpleNamespace(set=lambda value: None),
                highlight=lambda *args, **kwargs: None,
                _append_output=lambda text, session_id=None: log.append((text, session_id)),
                _clear_pending_genai_revisions=lambda: None,
                _save_session=lambda: None,
            )
            dummy._record_output_session_artifacts = lambda *args, **kwargs: artifact_calls.append((args, kwargs))

            with mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir):
                OPLIDE._apply_pending_genai_revisions(dummy)

            self.assertEqual(Path(dummy.model_file).name, "model_2026-05-03_01-20-41.mod")
            self.assertEqual(Path(dummy.data_file).name, "data_2026-05-03_01-20-41.dat")
            self.assertEqual(
                artifact_calls,
                [
                    (
                        ("session-1",),
                        {"model_text": "dvar int y;", "data_text": "y = 2;"},
                    )
                ],
            )

    def test_start_foreground_operation_blocks_overlap(self):
        class DummyVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        session_ids = iter(["session-1", "session-2"])
        dummy = SimpleNamespace(
            _active_operation=None,
            status_var=DummyVar(),
            _clear_output=lambda header: next(session_ids),
        )
        dummy._ensure_no_active_operation = lambda label: OPLIDE._ensure_no_active_operation(dummy, label)
        dummy._refresh_foreground_operation_ui = lambda: None

        first = OPLIDE._start_foreground_operation(
            dummy,
            kind="solve",
            label="Solve Model",
            header="Solve: Solving model...",
            status="Solving model...",
        )

        self.assertIsNotNone(first)
        self.assertIs(dummy._active_operation, first)

        with mock.patch.object(pyopl_ide_bootstrap.messagebox, "showinfo") as showinfo:
            second = OPLIDE._start_foreground_operation(
                dummy,
                kind="generate",
                label="Generate Model & Data",
                header="GenAI: Generating model and data...",
                status="GenAI: generating...",
            )

        self.assertIsNone(second)
        showinfo.assert_called_once()
        self.assertIn("already running", dummy.status_var.value)

        OPLIDE._finish_foreground_operation(dummy, first)
        self.assertIsNone(dummy._active_operation)

    def test_start_and_finish_foreground_operation_toggle_editor_lock(self):
        class DummyVar:
            def set(self, value):
                self.value = value

        class DummyText:
            def __init__(self):
                self.state = "normal"

            def cget(self, name):
                self.assert_name = name
                return self.state

            def config(self, **kwargs):
                self.state = kwargs["state"]

        dummy = SimpleNamespace(
            _active_operation=None,
            status_var=DummyVar(),
            model_text=DummyText(),
            data_text=DummyText(),
            _genai_loading=False,
            _genai_provider_models={},
            _clear_output=lambda header: "session-1",
        )
        dummy._ensure_no_active_operation = lambda label: OPLIDE._ensure_no_active_operation(dummy, label)
        dummy._set_editors_locked = lambda locked: OPLIDE._set_editors_locked(dummy, locked)
        dummy._refresh_foreground_operation_ui = lambda: OPLIDE._refresh_foreground_operation_ui(dummy)

        op = OPLIDE._start_foreground_operation(
            dummy,
            kind="generate",
            label="Generate Model & Data",
            header="GenAI: Generating model and data...",
            status="GenAI: generating...",
        )

        self.assertEqual(dummy.model_text.state, "disabled")
        self.assertEqual(dummy.data_text.state, "disabled")

        OPLIDE._finish_foreground_operation(dummy, op)

        self.assertEqual(dummy.model_text.state, "normal")
        self.assertEqual(dummy.data_text.state, "normal")

    def test_cleanup_solver_ipc_closes_queue_and_joins_thread(self):
        events = []

        class DummyQueue:
            def close(self):
                events.append("close")

            def join_thread(self):
                events.append("join_thread")

            def cancel_join_thread(self):
                events.append("cancel_join_thread")

        dummy = SimpleNamespace(
            _solver_process=object(),
            _solver_queue=DummyQueue(),
        )

        OPLIDE._cleanup_solver_ipc(dummy, cancel_queue_thread=False)

        self.assertEqual(events, ["close", "join_thread"])
        self.assertIsNone(dummy._solver_process)
        self.assertIsNone(dummy._solver_queue)

    def test_cleanup_solver_ipc_can_cancel_queue_thread(self):
        events = []

        class DummyQueue:
            def close(self):
                events.append("close")

            def join_thread(self):
                events.append("join_thread")

            def cancel_join_thread(self):
                events.append("cancel_join_thread")

        dummy = SimpleNamespace(
            _solver_process=object(),
            _solver_queue=DummyQueue(),
        )

        OPLIDE._cleanup_solver_ipc(dummy, cancel_queue_thread=True)

        self.assertEqual(events, ["close", "cancel_join_thread"])
        self.assertIsNone(dummy._solver_process)
        self.assertIsNone(dummy._solver_queue)

    def test_interrupt_active_non_solve_operation_marks_cancelled(self):
        class DummyVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        log = []
        op = pyopl_ide_bootstrap._ForegroundOperation(
            kind="genai-generate",
            label="Generate Model & Data",
            session_id="session-1",
        )
        dummy = SimpleNamespace(
            _active_operation=op,
            _solver_process=None,
            status_var=DummyVar(),
        )
        dummy._append_output = lambda text, session_id=None: log.append((text, session_id))
        dummy._finish_foreground_operation = lambda operation: OPLIDE._finish_foreground_operation(dummy, operation)
        dummy._refresh_foreground_operation_ui = lambda: None

        OPLIDE.interrupt_active_operation(dummy)

        self.assertTrue(op.cancel_requested)
        self.assertIsNone(dummy._active_operation)
        self.assertEqual(log, [("\nOperation interrupted by user.\n", "session-1")])
        self.assertIn("interrupted", dummy.status_var.value)

    def test_format_prompt_for_output_text_only(self):
        formatted = OPLIDE._format_prompt_for_output(None, "Prompt", "Maximize profit subject to capacity.")

        self.assertEqual(formatted, "\nPrompt:\nMaximize profit subject to capacity.\n\n")

    def test_format_prompt_for_output_with_attachments(self):
        formatted = OPLIDE._format_prompt_for_output(
            None,
            "Question",
            {
                "text": "Please explain the bottleneck.",
                "images": [{"path": "/tmp/chart.png"}, {"path": "/tmp/table.png"}],
            },
        )

        self.assertIn("\nQuestion:\n", formatted)
        self.assertIn("Please explain the bottleneck.\n", formatted)
        self.assertIn("Attachments:\n", formatted)
        self.assertIn("- /tmp/chart.png\n", formatted)
        self.assertIn("- /tmp/table.png\n", formatted)

    def test_append_output_to_prompt_input_text_only(self):
        merged = OPLIDE._append_output_to_prompt_input(
            None,
            "What changed in the model?",
            "Solve: Solving model...\nStatus: OPTIMAL\nObjective: 42",
        )

        self.assertEqual(
            merged,
            "What changed in the model?\n\n<session_output>\nSolve: Solving model...\nStatus: OPTIMAL\nObjective: 42\n</session_output>",
        )

    def test_append_output_to_prompt_input_preserves_attachments(self):
        merged = OPLIDE._append_output_to_prompt_input(
            None,
            {
                "text": "Please explain this result.",
                "images": [{"path": "/tmp/chart.png"}],
            },
            "Solve: Solving model...\nStatus: OPTIMAL",
        )

        self.assertEqual(
            merged,
            {
                "text": "Please explain this result.\n\n<session_output>\nSolve: Solving model...\nStatus: OPTIMAL\n</session_output>",
                "images": [{"path": "/tmp/chart.png"}],
            },
        )

    def test_append_output_to_prompt_input_wraps_output_when_prompt_is_empty(self):
        merged = OPLIDE._append_output_to_prompt_input(
            None,
            {"text": "", "images": [{"path": "/tmp/chart.png"}]},
            "Solve: Solving model...\nStatus: OPTIMAL",
        )

        self.assertEqual(
            merged,
            {
                "text": "<session_output>\nSolve: Solving model...\nStatus: OPTIMAL\n</session_output>",
                "images": [{"path": "/tmp/chart.png"}],
            },
        )


if __name__ == "__main__":
    unittest.main()

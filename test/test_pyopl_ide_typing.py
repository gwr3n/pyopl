import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# Import should not fail regardless of Pillow availability
from pyopl import pyopl_ide_bootstrap
from pyopl.pyopl_ide_bootstrap import OPLIDE


class TestPyOPLIDETyping(unittest.TestCase):
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

        with TemporaryDirectory() as tmpdir:
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
                model_file=None,
                data_file=None,
            )

            with mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir):
                session_id = OPLIDE._begin_new_output_session(dummy, "Solve: Solving model...")

            artifacts = dummy._output_session_artifacts[session_id]
            model_path = Path(artifacts["model_path"])
            data_path = Path(artifacts["data_path"])

            self.assertTrue(model_path.exists())
            self.assertTrue(data_path.exists())
            self.assertEqual(model_path.read_text(encoding="utf-8"), "dvar int x;")
            self.assertEqual(data_path.read_text(encoding="utf-8"), "x = 3;")

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


if __name__ == "__main__":
    unittest.main()

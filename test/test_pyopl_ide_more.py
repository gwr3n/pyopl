import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pyopl import pyopl_ide_bootstrap
from pyopl.pyopl_ide_bootstrap import OPLIDE, _FdLogRedirector, _ForegroundOperation, _QueueTextWriter


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyText:
    def __init__(self, value=""):
        self.value = value
        self.deleted = False
        self.inserted = []
        self.config_calls = []

    def get(self, *_args):
        return self.value

    def delete(self, *_args):
        self.deleted = True
        self.value = ""

    def insert(self, *_args):
        text = _args[-1]
        self.inserted.append(text)
        self.value += text

    def configure(self, **kwargs):
        self.config_calls.append(kwargs)

    def config(self, **kwargs):
        self.configure(**kwargs)


class TestIDEUtilitiesMore(unittest.TestCase):
    def test_queue_text_writer_buffers_lines_and_ignores_queue_errors(self):
        """Solver-process text is forwarded line-by-line and queue failures are non-fatal."""
        class DummyQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        q = DummyQueue()
        writer = _QueueTextWriter(q)

        self.assertEqual(writer.write("alpha"), 5)
        self.assertEqual(q.items, [])
        writer.write(" beta\nnext")
        self.assertEqual(q.items, [("log", "alpha beta\n")])
        writer.flush()
        self.assertEqual(q.items[-1], ("log", "next"))

        writer._queue = SimpleNamespace(put=mock.Mock(side_effect=RuntimeError("closed")))
        writer.write("lost\n")
        writer.flush()

    def test_fd_log_redirector_restore_and_read_pipe_error_paths(self):
        """File-descriptor log redirection restores saved descriptors and tolerates read failures."""
        writer = SimpleNamespace(flush=mock.Mock(), write=mock.Mock())
        redirector = _FdLogRedirector(writer)

        redirector._restore_fds()
        self.assertFalse(redirector._active)

        with mock.patch.object(pyopl_ide_bootstrap.os, "dup2") as dup2_mock, mock.patch.object(
            pyopl_ide_bootstrap.os, "close"
        ) as close_mock:
            redirector._saved_fds = {1: 10, 2: 11}
            redirector._active = True
            redirector._restore_fds()

        self.assertEqual(dup2_mock.call_count, 2)
        self.assertEqual(close_mock.call_count, 2)
        self.assertFalse(redirector._active)

        redirector._pipe_read = 3
        with mock.patch.object(pyopl_ide_bootstrap.select, "select", side_effect=RuntimeError("select failed")):
            redirector._read_pipe()
        writer.write.assert_not_called()

    def test_find_and_toggle_run_menu_entry(self):
        """The Run menu item is found by label and toggled between solve and stop states."""
        class Menu:
            def __init__(self):
                self.labels = ["Open", "Solve Model", "Other"]
                self.configured = []

            def index(self, value):
                return len(self.labels) - 1 if value == "end" else None

            def entrycget(self, index, option):
                if index == 0:
                    raise RuntimeError("skip")
                return self.labels[index]

            def entryconfigure(self, index, **kwargs):
                self.configured.append((index, kwargs))
                if "label" in kwargs:
                    self.labels[index] = kwargs["label"]

        dummy = SimpleNamespace(run_menu=Menu(), stop_model=mock.Mock(), run_model=mock.Mock())
        dummy._find_run_stop_menu_index = lambda: OPLIDE._find_run_stop_menu_index(dummy)
        dummy._accel = lambda key: f"Cmd+{key}"

        self.assertEqual(OPLIDE._find_run_stop_menu_index(dummy), 1)
        OPLIDE._set_run_menu_running(dummy, True)
        self.assertEqual(dummy.run_menu.labels[1], "Stop Model")
        self.assertEqual(dummy.run_menu.configured[-1][1]["accelerator"], "")
        OPLIDE._set_run_menu_running(dummy, False)
        self.assertEqual(dummy.run_menu.labels[1], "Solve Model")
        self.assertEqual(dummy.run_menu.configured[-1][1]["accelerator"], "Cmd+R")

        self.assertIsNone(OPLIDE._find_run_stop_menu_index(SimpleNamespace()))

    def test_foreground_operation_lifecycle_and_editor_locking(self):
        """Foreground operations create an output session, lock editors, and block competing actions."""
        class Widget:
            def __init__(self):
                self.state = "normal"

            def cget(self, key):
                return self.state

            def config(self, **kwargs):
                self.state = kwargs["state"]

        dummy = SimpleNamespace(
            _active_operation=None,
            status_var=DummyVar(),
            model_text=Widget(),
            data_text=Widget(),
            _clear_output=mock.Mock(return_value="session-1"),
            _refresh_foreground_operation_ui=mock.Mock(),
        )
        dummy._ensure_no_active_operation = lambda label: OPLIDE._ensure_no_active_operation(dummy, label)

        operation = OPLIDE._start_foreground_operation(
            dummy,
            kind="solve",
            label="Solve Model",
            header="Header",
            status="Running",
            solver_choice="scipy",
        )

        self.assertIsInstance(operation, _ForegroundOperation)
        self.assertEqual(dummy.status_var.value, "Running")
        self.assertIs(dummy._active_operation, operation)
        dummy._refresh_foreground_operation_ui.assert_called_once()

        OPLIDE._set_editors_locked(dummy, True)
        self.assertEqual(dummy.model_text.state, "disabled")
        self.assertEqual(dummy.data_text.state, "disabled")
        OPLIDE._set_editors_locked(dummy, False)
        self.assertEqual(dummy.model_text.state, "normal")

        with mock.patch.object(pyopl_ide_bootstrap.messagebox, "showinfo"):
            blocked = OPLIDE._ensure_no_active_operation(dummy, "Export")
        self.assertFalse(blocked)
        self.assertIn("already running", dummy.status_var.value)
        OPLIDE._finish_foreground_operation(dummy, operation)
        self.assertIsNone(dummy._active_operation)

    def test_selected_output_session_text_and_prompt_append(self):
        """GenAI prompt helpers use the selected output session and preserve attachment payloads."""
        class Listbox:
            def __init__(self, selection):
                self.selection = selection

            def curselection(self):
                return self.selection

        dummy = SimpleNamespace(
            request_listbox=Listbox((1,)),
            _output_session_ids=["s1", "s2"],
            _output_sessions={"s1": "first\n", "s2": "second\n"},
            _viewing_output_session_id=None,
            _current_output_session_id="s1",
        )

        self.assertEqual(OPLIDE._get_selected_output_session_text(dummy), "second")
        dummy.request_listbox = Listbox(())
        dummy._viewing_output_session_id = "s1"
        self.assertEqual(OPLIDE._get_selected_output_session_text(dummy), "first")

        self.assertEqual(OPLIDE._append_output_to_prompt_input(dummy, "prompt", "out"), "prompt\n\n<session_output>\nout\n</session_output>")
        merged = OPLIDE._append_output_to_prompt_input(dummy, {"text": "prompt", "images": [1]}, "out")
        self.assertEqual(merged["images"], [1])
        self.assertIn("<session_output>", merged["text"])
        self.assertEqual(OPLIDE._append_output_to_prompt_input(dummy, "prompt", ""), "prompt")

    def test_session_restore_no_artifacts_and_success(self):
        """Session model restore handles missing artifacts and restores saved model/data snapshots."""
        dummy = SimpleNamespace(
            _get_selected_request_session_id=mock.Mock(return_value="s1"),
            _get_output_session_artifacts=mock.Mock(return_value={}),
        )
        with mock.patch.object(pyopl_ide_bootstrap.messagebox, "showinfo") as showinfo:
            OPLIDE._show_session_model_preview(dummy)
            OPLIDE._show_session_model_diff(dummy)
            OPLIDE._restore_session_model(dummy)
        self.assertEqual(showinfo.call_count, 3)

        model = DummyText()
        data = DummyText()
        dummy = SimpleNamespace(
            _get_selected_request_session_id=mock.Mock(return_value="s1"),
            _get_output_session_artifacts=mock.Mock(return_value={"model_text": "m", "data_text": "d"}),
            model_text=model,
            data_text=data,
            model_file="old.mod",
            data_file="old.dat",
            editor_notebook=SimpleNamespace(tab=mock.Mock()),
            model_frame=object(),
            data_frame=object(),
            highlight=mock.Mock(),
            status_var=DummyVar(),
        )
        with mock.patch.object(pyopl_ide_bootstrap.messagebox, "askyesno", return_value=True):
            OPLIDE._restore_session_model(dummy)

        self.assertEqual(model.value, "m")
        self.assertEqual(data.value, "d")
        self.assertIsNone(dummy.model_file)
        self.assertIsNone(dummy.data_file)
        self.assertEqual(dummy.status_var.value, "Model restored from session")
        self.assertEqual(dummy.highlight.call_count, 2)

    def test_solver_progress_recording_trimming_and_stats(self):
        """Solver progress samples compute gap, trim rolling history, and update status values."""
        dummy = SimpleNamespace(
            _solver_progress_pending_sample=None,
            _solver_progress_update_after_id=None,
            _solver_progress_samples=[{"runtime": i} for i in range(200)],
            _solver_progress_rolling_seconds=10,
            _solver_progress_stat_vars={
                "LB": DummyVar(),
                "UB": DummyVar(),
                "Gap": DummyVar(),
                "Nodes": DummyVar(),
                "Solutions": DummyVar(),
                "Runtime": DummyVar(),
            },
            _solver_progress_stats_frame=object(),
            interface_font_family="TkDefaultFont",
            after=mock.Mock(return_value="after-id"),
            _flush_solver_progress_update=lambda: OPLIDE._flush_solver_progress_update(dummy),
            _display_solver_progress_enabled=lambda: True,
            _solver_tracks_progress=lambda solver_choice=None: True,
            _update_solver_progress_stats=lambda sample: OPLIDE._update_solver_progress_stats(dummy, sample),
            _append_solver_progress_sample=lambda sample: OPLIDE._append_solver_progress_sample(dummy, sample),
            _trim_solver_progress_samples=lambda: OPLIDE._trim_solver_progress_samples(dummy),
            _progress_sample_time=lambda sample: OPLIDE._progress_sample_time(dummy, sample),
            _solver_progress_stats=lambda sample: OPLIDE._solver_progress_stats(dummy, sample),
            _format_progress_value=lambda value: OPLIDE._format_progress_value(dummy, value),
            _redraw_solver_progress_chart=mock.Mock(),
        )

        OPLIDE._record_solver_progress(dummy, {"lower_bound": 90, "upper_bound": 100, "runtime": 201})
        self.assertAlmostEqual(dummy._solver_progress_pending_sample["gap"], 0.1)
        self.assertEqual(dummy._solver_progress_update_after_id, "after-id")
        OPLIDE._flush_solver_progress_update(dummy)
        self.assertIsNone(dummy._solver_progress_pending_sample)
        self.assertLessEqual(len(dummy._solver_progress_samples), 180)
        self.assertEqual(dummy._solver_progress_stat_vars["Gap"].value, "10%")
        self.assertEqual(dummy._solver_progress_stat_vars["Runtime"].value, "201.00s")
        self.assertEqual(dummy._solver_progress_stat_vars["LB"].value, "90")

        self.assertEqual(OPLIDE._progress_sample_time(dummy, {"time": "2.5"}), 2.5)
        self.assertIsNone(OPLIDE._progress_sample_time(dummy, {"runtime": "nan"}))
        self.assertEqual(OPLIDE._format_progress_value(dummy, None), "-")
        self.assertEqual(OPLIDE._format_progress_value(dummy, 1_200_000), "1.2e+06")

    def test_redraw_solver_progress_chart_waiting_line_and_points(self):
        """Progress chart rendering covers the waiting state and LB/UB line drawing path."""
        class Canvas:
            def __init__(self):
                self.calls = []

            def winfo_exists(self):
                return True

            def winfo_width(self):
                return 300

            def winfo_height(self):
                return 220

            def delete(self, *args):
                self.calls.append(("delete", args))

            def create_line(self, *args, **kwargs):
                self.calls.append(("line", args, kwargs))

            def create_text(self, *args, **kwargs):
                self.calls.append(("text", args, kwargs))

            def create_oval(self, *args, **kwargs):
                self.calls.append(("oval", args, kwargs))

        canvas = Canvas()
        dummy = SimpleNamespace(
            _solver_progress_canvas=canvas,
            _solver_progress_rolling_seconds=60,
            _solver_progress_samples=[],
            _display_solver_progress_enabled=lambda: True,
            _solver_tracks_progress=lambda: True,
        )
        OPLIDE._redraw_solver_progress_chart(dummy)
        self.assertTrue(any(call[0] == "text" and "Waiting" in call[2].get("text", "") for call in canvas.calls))

        canvas.calls.clear()
        dummy._solver_progress_samples = [{"lower_bound": 1, "upper_bound": 3}, {"lower_bound": 2, "upper_bound": 4}]
        OPLIDE._redraw_solver_progress_chart(dummy)
        self.assertTrue(any(call[0] == "line" and call[2].get("width") == 2 for call in canvas.calls))

    def test_ensure_saved_unsaved_changes_and_close(self):
        """Close-time helpers save editor buffers, detect unsaved changes, and shut down cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.mod"
            data_path = Path(tmpdir) / "data.dat"
            dummy = SimpleNamespace(
                model_text=DummyText("model\n"),
                data_text=DummyText("data\n"),
                model_file=None,
                data_file=None,
            )
            with mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir):
                result = OPLIDE._ensure_model_data_saved(dummy)

            self.assertTrue(Path(result[0]).read_text(encoding="utf-8"), "model")
            self.assertTrue(Path(result[1]).read_text(encoding="utf-8"), "data")

            dummy = SimpleNamespace(model_text=DummyText("model"), data_text=DummyText("data"))
            result = OPLIDE._ensure_model_data_saved(dummy, str(model_path), str(data_path))
            self.assertEqual(result, (str(model_path), str(data_path)))

        dummy = SimpleNamespace(
            model_text=DummyText("new"),
            data_text=DummyText("same"),
            _model_saved_text="old",
            _data_saved_text="same",
        )
        dummy._get_editor_text = lambda widget: widget.get()
        self.assertTrue(OPLIDE._has_unsaved_editor_changes(dummy))

        dummy = SimpleNamespace(
            _has_unsaved_editor_changes=mock.Mock(return_value=False),
            _confirm_quit_with_unsaved_changes=mock.Mock(return_value=False),
            stop_model=mock.Mock(),
            _save_settings=mock.Mock(),
            _save_session=mock.Mock(),
            _cleanup_genai_pdf_temp_dir=mock.Mock(),
            destroy=mock.Mock(),
            quit=mock.Mock(),
        )
        OPLIDE._on_close(dummy)
        self.assertTrue(dummy._shutting_down)
        dummy.stop_model.assert_called_once()
        dummy.destroy.assert_called_once()
        dummy.quit.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
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
        text = _args[1] if len(_args) > 1 else _args[-1]
        self.inserted.append(text)
        self.value += text

    def configure(self, **kwargs):
        self.config_calls.append(kwargs)

    def config(self, **kwargs):
        self.configure(**kwargs)


class ExistingDummyText(DummyText):
    def __init__(self, value=""):
        super().__init__(value)
        self.seen = False
        self.destroyed = False
        self.tags = []

    def winfo_exists(self):
        return True

    def see(self, *_args):
        self.seen = True

    def destroy(self):
        self.destroyed = True

    def tag_configure(self, *args, **kwargs):
        self.tags.append((args, kwargs))


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

        with (
            mock.patch.object(pyopl_ide_bootstrap.os, "dup2") as dup2_mock,
            mock.patch.object(pyopl_ide_bootstrap.os, "close") as close_mock,
        ):
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

        self.assertEqual(
            OPLIDE._append_output_to_prompt_input(dummy, "prompt", "out"), "prompt\n\n<session_output>\nout\n</session_output>"
        )
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

    def test_load_session_restores_history_artifacts_files_and_tabs(self):
        """Session loading restores output history, normalized artifacts, editor files, and tab labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.mod"
            data_path = Path(tmpdir) / "data.dat"
            session_path = Path(tmpdir) / ".pyopl_session"
            model_path.write_text("model text", encoding="utf-8")
            data_path.write_text("data text", encoding="utf-8")
            session_path.write_text(
                pyopl_ide_bootstrap.json.dumps(
                    {
                        "output_sessions": {"s1": "output"},
                        "output_session_ids": ["s1"],
                        "output_session_display": {"s1": "2026-01-01 10:00:00 • Solve"},
                        "output_session_artifacts": {"s1": {"model_text": 123, "data_text": None}, "bad": "skip"},
                        "current_output_session_id": "s1",
                        "viewing_output_session_id": "s1",
                        "model_file": str(model_path),
                        "data_file": str(data_path),
                    }
                ),
                encoding="utf-8",
            )

            class Listbox:
                def __init__(self):
                    self.items = []
                    self.selected = None
                    self.active = None

                def delete(self, *_args):
                    self.items.clear()

                def insert(self, _index, value):
                    self.items.append(value)

                def selection_clear(self, *_args):
                    self.selected = None

                def selection_set(self, index):
                    self.selected = index

                def activate(self, index):
                    self.active = index

            class Notebook:
                def __init__(self):
                    self.tabs = []

                def tab(self, frame, **kwargs):
                    self.tabs.append((frame, kwargs))

            output_text = ExistingDummyText()
            model_text = DummyText()
            data_text = DummyText()
            listbox = Listbox()
            notebook = Notebook()
            dummy = SimpleNamespace(
                _session_file_path=lambda: str(session_path),
                _output_sessions={},
                _output_session_ids=[],
                _output_session_display={},
                _output_session_label={},
                _output_session_timestamp={},
                _output_session_artifacts={},
                _current_output_session_id=None,
                _viewing_output_session_id=None,
                request_listbox=listbox,
                output_text=output_text,
                model_file=None,
                data_file=None,
                model_text=model_text,
                data_text=data_text,
                editor_notebook=notebook,
                model_frame="model-frame",
                data_frame="data-frame",
            )

            OPLIDE._load_session(dummy)

        self.assertEqual(dummy._output_sessions, {"s1": "output"})
        self.assertEqual(dummy._output_session_timestamp["s1"], "2026-01-01 10:00:00")
        self.assertEqual(dummy._output_session_label["s1"], "Solve")
        self.assertEqual(dummy._output_session_artifacts, {"s1": {"model_text": "123", "data_text": ""}})
        self.assertEqual(listbox.items, ["2026-01-01 10:00:00 • Solve"])
        self.assertEqual(listbox.selected, 0)
        self.assertEqual(output_text.value, "output")
        self.assertEqual(model_text.value, "model text")
        self.assertEqual(data_text.value, "data text")
        self.assertEqual(
            notebook.tabs[-2:], [("model-frame", {"text": "Model: model.mod"}), ("data-frame", {"text": "Data: data.dat"})]
        )

    def test_populate_diff_preview_text_tags_headers_and_changes(self):
        """Diff preview rendering writes headers and tags additions/removals/context."""
        text = ExistingDummyText()
        dummy = SimpleNamespace(theme_var=DummyVar("flatly"))
        dummy._configure_diff_preview_tags = lambda text_widget: OPLIDE._configure_diff_preview_tags(dummy, text_widget)

        OPLIDE._populate_diff_preview_text(dummy, text, "same\nold\n", "same\nnew\n")

        self.assertIn("--- Historical", text.value)
        self.assertIn("+++ Current", text.value)
        self.assertIn("- old", text.value)
        self.assertIn("+ new", text.value)
        self.assertIn({"state": "normal"}, text.config_calls)
        self.assertEqual(text.config_calls[-1], {"state": "disabled"})
        configured_tags = {args[0] for args, _kwargs in text.tags}
        self.assertTrue({"diff_header", "diff_add", "diff_remove", "diff_context"}.issubset(configured_tags))

    def test_append_solver_log_text_and_stop_timer_cover_red_paths(self):
        """Solver log append and run-timer stop handle visible widgets and cancellation."""
        log_text = ExistingDummyText()
        dummy = SimpleNamespace(_solver_log_text=log_text)

        OPLIDE._append_solver_log_text(dummy, "solver line")

        self.assertEqual(log_text.value, "solver line")
        self.assertTrue(log_text.seen)
        self.assertEqual(log_text.config_calls[0], {"state": "normal"})
        self.assertEqual(log_text.config_calls[-1], {"state": "disabled"})

        status_runtime_var = DummyVar("old")
        dummy = SimpleNamespace(
            _run_timer_after_id="after-id",
            _run_started_at=123.0,
            after_cancel=mock.Mock(),
            status_runtime_var=status_runtime_var,
        )
        OPLIDE._stop_run_timer(dummy)
        dummy.after_cancel.assert_called_once_with("after-id")
        self.assertIsNone(dummy._run_timer_after_id)
        self.assertIsNone(dummy._run_started_at)
        self.assertEqual(status_runtime_var.value, "")

    def test_save_as_helpers_write_files_and_update_session(self):
        """Save-as helpers write editor contents, update baselines, tab labels, and save session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "saved.mod"
            data_path = Path(tmpdir) / "saved.dat"

            class Notebook:
                def __init__(self):
                    self.calls = []

                def tab(self, frame, **kwargs):
                    self.calls.append((frame, kwargs))

            notebook = Notebook()
            dummy = SimpleNamespace(
                model_file=None,
                data_file=None,
                model_text=DummyText("model body\n"),
                data_text=DummyText("data body\n"),
                editor_notebook=notebook,
                model_frame="model-frame",
                data_frame="data-frame",
                _get_editor_text=lambda widget: widget.get(),
                _save_session=mock.Mock(),
            )

            with mock.patch.object(pyopl_ide_bootstrap.filedialog, "asksaveasfilename", return_value=str(model_path)):
                OPLIDE.save_model_as(dummy)
            with mock.patch.object(pyopl_ide_bootstrap.filedialog, "asksaveasfilename", return_value=str(data_path)):
                OPLIDE.save_data_as(dummy)

            self.assertEqual(model_path.read_text(encoding="utf-8"), "model body")
            self.assertEqual(data_path.read_text(encoding="utf-8"), "data body")
            self.assertEqual(dummy._model_saved_text, "model body\n")
            self.assertEqual(dummy._data_saved_text, "data body\n")
            self.assertEqual(
                notebook.calls[-2:],
                [("model-frame", {"text": "Model: saved.mod"}), ("data-frame", {"text": "Data: saved.dat"})],
            )
            self.assertEqual(dummy._save_session.call_count, 2)

    def test_poll_solver_empty_queue_handles_running_and_unexpected_exit(self):
        """Polling reschedules live solvers and reports dead solvers with no terminal message."""

        class EmptyQueue:
            def get_nowait(self):
                raise pyopl_ide_bootstrap.queue.Empty

            def close(self):
                pass

            def cancel_join_thread(self):
                pass

        class Process:
            def __init__(self, alive):
                self.alive = alive

            def is_alive(self):
                return self.alive

        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve", "s1")
        running = SimpleNamespace(
            _solver_process=Process(True),
            _solver_queue=EmptyQueue(),
            after=mock.Mock(),
            _poll_solver=mock.Mock(),
        )
        OPLIDE._poll_solver(running, operation)
        running.after.assert_called_once()

        dead = SimpleNamespace(
            _solver_process=Process(False),
            _solver_queue=EmptyQueue(),
            _set_run_menu_running=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _append_output=mock.Mock(),
            _stop_run_timer=mock.Mock(),
            status_var=DummyVar(),
            _finish_solver_progress=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
        )
        dead._cleanup_solver_ipc = lambda cancel_queue_thread: OPLIDE._cleanup_solver_ipc(
            dead, cancel_queue_thread=cancel_queue_thread
        )

        OPLIDE._poll_solver(dead, operation)

        dead._append_output.assert_called_once_with("\nError: Solver process terminated unexpectedly.\n", "s1")
        self.assertEqual(dead.status_var.value, "Error: Solver process terminated.")
        dead._finish_solver_progress.assert_called_once_with(status="ended unexpectedly")
        dead._finish_foreground_operation.assert_called_once_with(operation)

    def test_poll_solver_routes_progress_filters_non_dict_and_handles_error(self):
        """Polling consumes progress/log messages and handles non-success terminal messages."""

        class Process:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        class Queue:
            def __init__(self):
                self.items = [("progress", {"runtime": 1}), ("progress", "ignore"), ("log", "hello"), ("error", "boom")]

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                raise pyopl_ide_bootstrap.queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve", "s1")
        dummy = SimpleNamespace(
            _solver_process=Process(),
            _solver_queue=Queue(),
            _record_solver_progress=mock.Mock(),
            _append_solver_log_text=mock.Mock(),
            _set_run_menu_running=mock.Mock(),
            _stop_run_timer=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _finish_solver_progress=mock.Mock(),
            _append_output=mock.Mock(),
            status_var=DummyVar(),
            _finish_foreground_operation=mock.Mock(),
        )
        dummy._cleanup_solver_ipc = lambda cancel_queue_thread: OPLIDE._cleanup_solver_ipc(
            dummy, cancel_queue_thread=cancel_queue_thread
        )

        OPLIDE._poll_solver(dummy, operation)

        dummy._record_solver_progress.assert_called_once_with({"runtime": 1})
        dummy._append_solver_log_text.assert_called_once_with("hello")
        dummy._finish_solver_progress.assert_called_once_with(status="failed")
        dummy._append_output.assert_called_once_with("\nError:\nboom\n", "s1")
        self.assertEqual(dummy.status_var.value, "Error running model")
        dummy._finish_foreground_operation.assert_called_once_with(operation)

    def test_poll_solver_solve_and_explain_skips_failed_solution(self):
        """Solve-and-explain success path skips GenAI explanation when solver output is unsuccessful."""

        class Process:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        class Queue:
            def __init__(self):
                self.items = [("success", {"status": "INFEASIBLE"})]

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                raise pyopl_ide_bootstrap.queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        operation = pyopl_ide_bootstrap._ForegroundOperation(
            "solve", "Solve", "s1", solver_choice="gurobi", explain_after_solve=True
        )
        dummy = SimpleNamespace(
            _solver_process=Process(),
            _solver_queue=Queue(),
            _set_run_menu_running=mock.Mock(),
            _stop_run_timer=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _finish_solver_progress=mock.Mock(),
            _display_solve_results=mock.Mock(),
            _append_output=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
        )
        dummy._cleanup_solver_ipc = lambda cancel_queue_thread: OPLIDE._cleanup_solver_ipc(
            dummy, cancel_queue_thread=cancel_queue_thread
        )

        OPLIDE._poll_solver(dummy, operation)

        dummy._display_solve_results.assert_called_once_with({"status": "INFEASIBLE"}, session_id="s1", solver_choice="gurobi")
        dummy._append_output.assert_called_once_with(
            "\n[GenAI] Skipping explanation because solve did not produce a successful solution.\n",
            "s1",
        )
        self.assertEqual(dummy._finish_foreground_operation.call_args_list, [mock.call(operation), mock.call(operation)])

    def test_poll_solver_solve_and_explain_success_starts_feedback_thread(self):
        """Successful solve-and-explain starts the asynchronous feedback worker and defers operation finish."""

        class Process:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        class Queue:
            def __init__(self):
                self.items = [("success", {"status": "OPTIMAL", "solution": {"x": 1}})]

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                raise pyopl_ide_bootstrap.queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        started = []

        class ThreadFactory:
            def __init__(self, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self):
                started.append(self)

        operation = pyopl_ide_bootstrap._ForegroundOperation(
            "solve",
            "Solve",
            "s1",
            solver_choice="gurobi",
            model_file="m.mod",
            data_file="d.dat",
            explain_after_solve=True,
        )
        dummy = SimpleNamespace(
            _solver_process=Process(),
            _solver_queue=Queue(),
            _set_run_menu_running=mock.Mock(),
            _stop_run_timer=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _finish_solver_progress=mock.Mock(),
            _display_solve_results=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
            genai_provider="openai",
            genai_model="model",
        )
        dummy._cleanup_solver_ipc = lambda cancel_queue_thread: OPLIDE._cleanup_solver_ipc(
            dummy, cancel_queue_thread=cancel_queue_thread
        )

        with mock.patch.object(
            pyopl_ide_bootstrap.threading, "Thread", side_effect=lambda target, daemon: ThreadFactory(target, daemon)
        ):
            OPLIDE._poll_solver(dummy, operation)

        dummy._display_solve_results.assert_called_once()
        dummy._finish_foreground_operation.assert_not_called()
        self.assertEqual(len(started), 1)
        self.assertTrue(started[0].daemon)

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

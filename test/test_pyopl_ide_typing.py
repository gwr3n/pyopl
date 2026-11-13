import os
import queue
import unittest
from builtins import open as builtin_open
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

# Import should not fail regardless of Pillow availability
from pyopl import pyopl_ide_bootstrap
from pyopl.pyopl_ide_bootstrap import OPLIDE
from pyopl.semantic_error import SemanticError


class _ImmediateThread:
    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class TestPyOPLIDETyping(unittest.TestCase):
    def test_data_paths_must_be_paired(self):
        self.assertTrue(pyopl_ide_bootstrap._data_paths_are_paired("", ""))
        self.assertTrue(pyopl_ide_bootstrap._data_paths_are_paired("left.dat", "right.dat"))
        self.assertFalse(pyopl_ide_bootstrap._data_paths_are_paired("left.dat", ""))
        self.assertFalse(pyopl_ide_bootstrap._data_paths_are_paired("", "right.dat"))

    def test_comparison_request_validates_paths_for_compare_and_mapping_generation(self):
        with TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.mod"
            right = Path(tmpdir) / "right.mod"
            left_data = Path(tmpdir) / "left.dat"
            right_data = Path(tmpdir) / "right.dat"
            for path in (left, right, left_data, right_data):
                path.write_text("", encoding="utf-8")
            missing_mapping = str(Path(tmpdir) / "missing.json")
            request = pyopl_ide_bootstrap._ComparisonRequest(
                str(left), str(left_data), str(right), str(right_data), "abstract", missing_mapping
            )

            request.validate_paths()
            with self.assertRaises(pyopl_ide_bootstrap._ComparisonInputError) as missing:
                request.validate_paths(include_mapping=True)
            self.assertFalse(missing.exception.warning)
            self.assertEqual(str(missing.exception), f"Observable mapping file does not exist:\n{missing_mapping}")

            invalid = (
                ("", "", str(right), "", "Choose both a left model and a right model.", True),
                (str(left), str(left_data), str(right), "", "Choose both data files or leave both data fields empty.", True),
                (str(left), "", str(Path(tmpdir) / "missing.mod"), "", "Right model file does not exist:", False),
            )
            for left_model, left_dat, right_model, right_dat, expected, warning in invalid:
                with self.subTest(expected=expected):
                    candidate = pyopl_ide_bootstrap._ComparisonRequest(
                        left_model, left_dat, right_model, right_dat, "abstract"
                    )
                    with self.assertRaises(pyopl_ide_bootstrap._ComparisonInputError) as error:
                        candidate.validate_paths(include_mapping=True)
                    self.assertEqual(error.exception.warning, warning)
                    self.assertIn(expected, str(error.exception))

    def test_comparison_process_owner_starts_and_releases_worker(self):
        request = pyopl_ide_bootstrap._ComparisonRequest("left.mod", "", "right.mod", "", "abstract")
        result_queue = mock.Mock()
        process = mock.Mock()
        owner = pyopl_ide_bootstrap._ComparisonProcessState()

        with (
            mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Queue", return_value=result_queue),
            mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Process", return_value=process) as process_cls,
        ):
            owner.start(request)

        process_cls.assert_called_once_with(
            target=pyopl_ide_bootstrap._compare_models_wrapper,
            args=("left.mod", "", "right.mod", "", "abstract", result_queue, ""),
        )
        process.start.assert_called_once_with()
        result_queue.get_nowait.return_value = ("success", "result")
        self.assertEqual(owner.read_nowait(), ("success", "result"))

        owner.finish()

        process.join.assert_called_once_with(timeout=0.1)
        result_queue.close.assert_called_once_with()
        result_queue.join_thread.assert_called_once_with()
        self.assertIsNone(owner.process)
        self.assertIsNone(owner.result_queue)

    def test_comparison_process_owner_cleans_up_failed_start_and_interrupt(self):
        request = pyopl_ide_bootstrap._ComparisonRequest("left.mod", "", "right.mod", "", "abstract")
        result_queue = mock.Mock()
        process = mock.Mock()
        process.start.side_effect = OSError("cannot spawn")
        owner = pyopl_ide_bootstrap._ComparisonProcessState()

        with (
            mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Queue", return_value=result_queue),
            mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Process", return_value=process),
        ):
            with self.assertRaises(OSError):
                owner.start(request)

        result_queue.cancel_join_thread.assert_called_once_with()
        self.assertIsNone(owner.process)

        process.start.side_effect = None
        process.is_alive.side_effect = [True, True]
        owner.process = process
        interrupt_queue = mock.Mock()
        owner.result_queue = interrupt_queue
        owner.interrupt()

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        interrupt_queue.cancel_join_thread.assert_called_once_with()
        self.assertIsNone(owner.process)
        self.assertIsNone(owner.result_queue)

    def test_comparison_process_owner_rejects_invalid_messages(self):
        owner = pyopl_ide_bootstrap._ComparisonProcessState(result_queue=mock.Mock())
        owner.result_queue.get_nowait.return_value = ("unexpected", "payload")

        with self.assertRaisesRegex(ValueError, "Invalid comparison worker message"):
            owner.read_nowait()

    def test_comparison_dialog_validates_completes_and_cancels_without_tk(self):
        class DummyVar:
            instances = []

            def __init__(self, value=""):
                self.value = value
                self.instances.append(self)

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        buttons = {}

        def make_button(*_args, **kwargs):
            button = mock.Mock()
            button.command = kwargs.get("command")

            def config(**updates):
                if "command" in updates:
                    button.command = updates["command"]

            button.config.side_effect = config
            buttons[kwargs.get("text")] = button
            return button

        dialog = mock.Mock()
        dialog.after.return_value = "poll-id"
        dialog.winfo_exists.return_value = True
        result_text = mock.Mock()
        host = SimpleNamespace(
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            genai_provider=None,
            genai_model=None,
            _active_operation=None,
            _ensure_no_active_operation=mock.Mock(return_value=True),
            _start_foreground_operation=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
            interrupt_active_operation=mock.Mock(),
            _append_output=mock.Mock(),
            status_var=mock.Mock(),
            after=mock.Mock(),
        )
        result = SimpleNamespace(
            status="unknown", level="inconclusive", reason="no proof", proof_steps=(), counterexample=None
        )
        first_queue, second_queue, third_queue = mock.Mock(), mock.Mock(), mock.Mock()
        first_queue.get_nowait.return_value = ("success", result)
        third_queue.get_nowait.side_effect = pyopl_ide_bootstrap.queue.Empty
        first_process, second_process, third_process = mock.Mock(), mock.Mock(), mock.Mock()
        third_process.is_alive.return_value = False

        with (
            TemporaryDirectory() as tmpdir,
            mock.patch.object(pyopl_ide_bootstrap.tk, "Toplevel", return_value=dialog),
            mock.patch.object(pyopl_ide_bootstrap.tk, "StringVar", DummyVar),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Frame", side_effect=lambda *_a, **_kw: mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "LabelFrame", side_effect=lambda *_a, **_kw: mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Label", side_effect=lambda *_a, **_kw: mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Entry", side_effect=lambda *_a, **_kw: mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Button", side_effect=make_button),
            mock.patch.object(pyopl_ide_bootstrap.scrolledtext, "ScrolledText", return_value=result_text),
            mock.patch.object(pyopl_ide_bootstrap.messagebox, "showwarning") as showwarning,
            mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
            mock.patch.object(
                pyopl_ide_bootstrap.multiprocessing, "Queue", side_effect=[first_queue, second_queue, third_queue]
            ),
            mock.patch.object(
                pyopl_ide_bootstrap.multiprocessing, "Process", side_effect=[first_process, second_process, third_process]
            ),
            mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", return_value=mock.Mock()) as thread_cls,
        ):
            OPLIDE.compare_models(host)
            compare_button = buttons["Compare"]
            compare_button.command()
            showwarning.assert_called_once()

            left_path = Path(tmpdir) / "left.mod"
            right_path = Path(tmpdir) / "right.mod"
            left_path.write_text("minimize 0;", encoding="utf-8")
            right_path.write_text("minimize 0;", encoding="utf-8")
            DummyVar.instances[1].set(str(left_path))
            DummyVar.instances[3].set(str(right_path))
            compare_button.command()
            poll = dialog.after.call_args.args[1:]
            poll[0](*poll[1:])
            self.assertIn("Status: unknown", result_text.insert.call_args.args[1])
            first_queue.join_thread.assert_called_once()

            escape = next(call.args[1] for call in dialog.bind.call_args_list if call.args[0] == "<Escape>")
            mapping_operation = pyopl_ide_bootstrap._ForegroundOperation("genai-observable-mapping", "Mapping", "s1")
            host.genai_provider = "openai"
            host.genai_model = "selected-model"
            host._start_foreground_operation.return_value = mapping_operation
            host._active_operation = mapping_operation
            host.interrupt_active_operation.side_effect = lambda: setattr(host, "_active_operation", None)
            buttons["Generate"].command()
            thread_cls.return_value.start.assert_called_once_with()
            escape(None)
            host.interrupt_active_operation.assert_called_once_with()

            compare_button.command()
            escape(None)
            second_queue.cancel_join_thread.assert_called_once()
            dialog.after_cancel.assert_called_once_with("poll-id")
            self.assertFalse(dialog.destroy.called)

            compare_button.command()
            poll = dialog.after.call_args.args[1:]
            poll[0](*poll[1:])
            third_queue.cancel_join_thread.assert_called_once()
            showerror.assert_called_once_with("Compare models", "Comparison process terminated unexpectedly.", parent=dialog)
            escape(None)
            dialog.destroy.assert_called_once()

    def test_compare_wrapper_reports_semantic_error_without_traceback(self):
        result_queue = mock.Mock()
        error = SemanticError("index 2 of 'shipment' can exceed its declared range")

        with TemporaryDirectory() as tmpdir:
            left_model = Path(tmpdir) / "left.mod"
            right_model = Path(tmpdir) / "right.mod"
            left_model.write_text("minimize 0; subject to {}", encoding="utf-8")
            right_model.write_text("minimize 0; subject to {}", encoding="utf-8")
            with mock.patch.object(pyopl_ide_bootstrap, "compare_models", side_effect=error):
                pyopl_ide_bootstrap._compare_models_wrapper(
                    str(left_model), "", str(right_model), "", "abstract", result_queue
                )

        kind, detail = result_queue.put.call_args.args[0]
        self.assertEqual(kind, "validation_error")
        self.assertEqual(detail, "index 2 of 'shipment' can exceed its declared range")
        self.assertNotIn("Traceback", detail)

    def test_compare_wrapper_accepts_observable_mapping_file(self):
        result_queue = mock.Mock()
        fixture_root = Path(__file__).parent / "fixtures" / "affine_observable" / "rs"
        left_model = fixture_root / "bm" / "bm.mod"
        right_model = fixture_root / "pm" / "pm.mod"
        data_file = fixture_root / "common" / "sample.dat"
        mapping_file = fixture_root / "observable_mapping.json"

        pyopl_ide_bootstrap._compare_models_wrapper(
            str(left_model),
            str(data_file),
            str(right_model),
            str(data_file),
            "concrete",
            result_queue,
            str(mapping_file),
        )

        kind, result = result_queue.put.call_args.args[0]
        self.assertEqual(kind, "success")
        self.assertTrue(result.equivalent)
        self.assertEqual(result.projected_assignment_counts, (16, 16))

    def test_compare_wrapper_accepts_uniform_indexed_mapping_file(self):
        from pyopl.equivalence import (
            IndexedAffineExpression,
            IndexedAffineFamily,
            IndexedAffineTerm,
            IndexedObservablePair,
            ParameterContract,
            SymbolicAffineExpression,
            SymbolicDomain,
            serialize_indexed_observable_mapping,
        )

        result_queue = mock.Mock()
        result = SimpleNamespace(status="unknown", proof_steps=())
        one = SymbolicAffineExpression.constant_value(1)
        domain = SymbolicDomain("t", one, SymbolicAffineExpression.parameter("N"))
        family = IndexedAffineFamily(
            ("t",),
            (domain,),
            IndexedAffineExpression((IndexedAffineTerm("x", ("t",)),)),
        )

        with TemporaryDirectory() as tmpdir:
            left_model = Path(tmpdir) / "left.mod"
            right_model = Path(tmpdir) / "right.mod"
            mapping_file = Path(tmpdir) / "mapping.json"
            left_model.write_text("minimize 0; subject to {}", encoding="utf-8")
            right_model.write_text("minimize 0; subject to {}", encoding="utf-8")
            mapping_file.write_text(
                serialize_indexed_observable_mapping(
                    (IndexedObservablePair("x[{t}]", family, family),),
                    ParameterContract(),
                ),
                encoding="utf-8",
            )
            with mock.patch.object(pyopl_ide_bootstrap, "compare_models", return_value=result) as compare:
                pyopl_ide_bootstrap._compare_models_wrapper(
                    str(left_model), "", str(right_model), "", "abstract", result_queue, str(mapping_file)
                )

        kwargs = compare.call_args.kwargs
        self.assertIsNotNone(kwargs["indexed_observable_mapping"])
        self.assertIsNotNone(kwargs["parameter_contract"])
        self.assertIsNone(kwargs["observable_mapping"])

    def test_comparison_validation_error_has_index_repair_hint(self):
        message = pyopl_ide_bootstrap._format_comparison_validation_error(
            "index 2 of 'shipment' can exceed its declared range"
        )

        self.assertIn("Model validation issue", message)
        self.assertIn("stopped before checking equivalence", message)
        self.assertIn("j < N", message)
        self.assertNotIn("Traceback", message)

    def test_concise_genai_error_hides_context_traceback(self):
        context_error = RuntimeError(
            "Request exceeds context length: 131000 input tokens\n\nTraceback (most recent call last):\n"
            + ("details\n" * 1000)
        )

        message = pyopl_ide_bootstrap._concise_genai_error(context_error)

        self.assertEqual(
            message,
            "The request exceeds the model's context window. Reduce the model/data size or choose a model with a larger context window.",
        )

    def test_save_exemplar_menu_item_follows_export_model(self):
        class FakeMenu:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.entries = []
                self.instances.append(self)

            def add_command(self, **kwargs):
                self.entries.append(("command", kwargs))

            def add_separator(self):
                self.entries.append(("separator", {}))

            def add_cascade(self, **kwargs):
                self.entries.append(("cascade", kwargs))

            def add_radiobutton(self, **kwargs):
                self.entries.append(("radiobutton", kwargs))

            def add_checkbutton(self, **kwargs):
                self.entries.append(("checkbutton", kwargs))

        class Dummy:
            solver = mock.Mock()
            display_solver_progress_var = mock.Mock()
            font_size_var = mock.Mock()
            theme_var = mock.Mock()

            def __getattr__(self, name):
                if name == "_accel":
                    return lambda key: key
                value = mock.Mock()
                setattr(self, name, value)
                return value

        dummy = Dummy()
        with mock.patch.object(pyopl_ide_bootstrap.tk, "Menu", FakeMenu):
            OPLIDE._setup_menu(dummy)

        file_labels = [entry[1]["label"] for entry in FakeMenu.instances[1].entries if entry[0] == "command"]
        open_index = file_labels.index("Open...")
        self.assertEqual(file_labels[open_index + 1], "Retrieve Exemplar...")
        export_index = file_labels.index("Export Model...")
        self.assertEqual(file_labels[export_index + 1], "Save Exemplar...")

    def test_classify_exemplar_search_result_handles_worker_messages(self):
        low = {"name": "low", "description_path": Path("/tmp/low.md")}
        high = {"name": "high", "description_path": Path("/tmp/high.md")}
        exemplars = [low, high]

        cases = [
            (("ready", None, None), 2, ("ready", None)),
            (("startup_error", None, "load failed"), 2, ("startup_error", "load failed")),
            (("success", 1, []), 2, ("ignore", None)),
            (("error", 2, "ranking failed"), 2, ("error", "ranking failed")),
            (
                (
                    "success",
                    2,
                    [{"path": "/tmp/high.md"}, "invalid", {"missing": "path"}, {"path": "/tmp/low.md"}],
                ),
                2,
                ("success", [high, low]),
            ),
        ]

        for message, current_generation, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    pyopl_ide_bootstrap._ExemplarSearchState.classify_result(message, current_generation, exemplars),
                    expected,
                )

    def test_exemplar_search_owns_timers_generations_and_worker_shutdown(self):
        exemplar = {"name": "fleet", "description_path": Path("/tmp/fleet.md")}
        worker = mock.Mock()
        worker.is_alive.return_value = True
        search = pyopl_ide_bootstrap._ExemplarSearchState(worker, [exemplar])
        dialog = mock.Mock()
        dialog.after.side_effect = ["search-1", "search-2", "poll-1"]

        search.schedule_search(dialog, mock.Mock())
        search.schedule_search(dialog, mock.Mock())
        self.assertFalse(search.start_search("  "))
        self.assertTrue(search.start_search("fleet"))
        worker.submit.assert_called_once_with(2, "fleet")
        worker.get_nowait.side_effect = [
            ("ready", None, None),
            ("success", 1, [{"path": "/tmp/fleet.md"}]),
            ("success", 2, [{"path": "/tmp/fleet.md"}]),
            queue.Empty(),
        ]
        self.assertEqual(search.drain(), [("ready", None), ("success", [exemplar])])
        self.assertTrue(search.worker_ready)
        search.schedule_poll(dialog, mock.Mock())
        search.close(dialog)
        search.close(dialog)

        self.assertEqual(dialog.after_cancel.call_args_list, [mock.call("search-1"), mock.call("poll-1")])
        worker.close.assert_called_once_with(wait=False)
        self.assertEqual(search.drain(), [])
        search.schedule_search(dialog, mock.Mock())
        self.assertEqual(dialog.after.call_count, 3)

    def test_exemplar_search_reports_unavailable_worker(self):
        worker = mock.Mock()
        worker.is_alive.return_value = False
        search = pyopl_ide_bootstrap._ExemplarSearchState(worker, [])

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            search.start_search("query")
        worker.submit.assert_not_called()

    def test_retrieve_copy_failure_keeps_dialog_and_editors_unchanged(self):
        exemplar = {
            "name": "fleet",
            "description_path": Path("/tmp/fleet.md"),
            "model_path": Path("/tmp/fleet.mod"),
            "data_path": Path("/tmp/fleet.dat"),
        }
        previous_cleanup = mock.Mock()
        host = SimpleNamespace(
            _ensure_no_active_operation=lambda action: True,
            _exemplar_search_cleanup=previous_cleanup,
            _available_exemplars=lambda roots: [exemplar],
            _copy_exemplar_to_tmp=mock.Mock(side_effect=OSError("copy failed")),
            _open_model_path=mock.Mock(),
            _open_data_path=mock.Mock(),
            status_var=mock.Mock(),
        )
        dialog = mock.Mock()
        dialog.after.return_value = "poll-id"
        exemplar_list = mock.Mock()
        exemplar_list.curselection.return_value = (0,)
        worker = mock.Mock()
        with (
            mock.patch.object(pyopl_ide_bootstrap.GenAIStrategyBase, "default_models_dirs", return_value=[]),
            mock.patch.object(pyopl_ide_bootstrap, "ExemplarRankingWorker", return_value=worker),
            mock.patch.object(pyopl_ide_bootstrap.tk, "Toplevel", return_value=dialog),
            mock.patch.object(pyopl_ide_bootstrap.tk, "StringVar", return_value=mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.tk, "Listbox", return_value=exemplar_list),
            mock.patch.object(pyopl_ide_bootstrap.tk, "Text", return_value=mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Label", return_value=mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Entry", return_value=mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Frame", return_value=mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Scrollbar", return_value=mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Button", return_value=mock.Mock()) as button,
            mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
        ):
            OPLIDE.retrieve_exemplar(host)
            button.call_args_list[0].kwargs["command"]()

        previous_cleanup.assert_called_once_with(None)
        self.assertIn("copy failed", showerror.call_args.args[1])
        host._open_model_path.assert_not_called()
        host._open_data_path.assert_not_called()
        dialog.destroy.assert_not_called()
        host._exemplar_search_cleanup(None)
        worker.close.assert_called_once_with(wait=False)

    def test_exemplar_catalog_only_lists_complete_triplets_once(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            complete = root / "complete"
            complete.mkdir()
            (complete / "complete.md").write_text("description", encoding="utf-8")
            (complete / "complete.mod").write_text("model", encoding="utf-8")
            (complete / "complete.dat").write_text("data", encoding="utf-8")
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "incomplete.md").write_text("description", encoding="utf-8")

            entries = OPLIDE._available_exemplars([root, root])

            self.assertEqual([entry["name"] for entry in entries], ["complete"])
            self.assertEqual(entries[0]["model_path"], complete / "complete.mod")

    def test_file_menu_quit_entry_is_platform_appropriate(self):
        class FakeMenu:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.entries = []
                self.instances.append(self)

            def add_command(self, **kwargs):
                self.entries.append(("command", kwargs))

            def add_separator(self):
                self.entries.append(("separator", {}))

            def add_cascade(self, **kwargs):
                self.entries.append(("cascade", kwargs))

            def add_radiobutton(self, **kwargs):
                self.entries.append(("radiobutton", kwargs))

            def add_checkbutton(self, **kwargs):
                self.entries.append(("checkbutton", kwargs))

        class Dummy:
            solver = mock.Mock()
            display_solver_progress_var = mock.Mock()
            font_size_var = mock.Mock()
            theme_var = mock.Mock()

            def __getattr__(self, name):
                if name == "_accel":
                    return lambda key: f"Ctrl+{key}"
                value = mock.Mock()
                setattr(self, name, value)
                return value

        for platform, expected_label in (("darwin", None), ("win32", "Exit"), ("linux", "Quit")):
            FakeMenu.instances = []
            with (
                mock.patch.object(pyopl_ide_bootstrap.tk, "Menu", FakeMenu),
                mock.patch.object(pyopl_ide_bootstrap.sys, "platform", platform),
            ):
                OPLIDE._setup_menu(Dummy())

            file_entries = FakeMenu.instances[1].entries
            quit_entries = [
                entry[1] for entry in file_entries if entry[0] == "command" and entry[1]["label"] in {"Exit", "Quit"}
            ]
            if expected_label is None:
                self.assertEqual(quit_entries, [])
            else:
                self.assertEqual(len(quit_entries), 1)
                self.assertEqual(quit_entries[0]["label"], expected_label)
                self.assertEqual(quit_entries[0]["accelerator"], "Ctrl+Q")

    def test_start_solver_process_loads_settings_from_working_directory(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve", "session-1")
        dummy = SimpleNamespace(
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(),
            status_var=mock.Mock(),
            _append_output=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
        )
        process = mock.Mock()

        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "highs.json").write_text('{"time_limit": 4.0}', encoding="utf-8")
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                settings_path = Path(os.getcwd(), "highs.json")
                with mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Queue", return_value=mock.Mock()):
                    with mock.patch.object(
                        pyopl_ide_bootstrap.multiprocessing, "Process", return_value=process
                    ) as process_cls:
                        started = OPLIDE._start_solver_process(dummy, "model.mod", "data.dat", "scipy", operation)
            finally:
                os.chdir(old_cwd)

        self.assertTrue(started)
        self.assertEqual(dummy._solver_ipc.session_id, operation.session_id)
        process_cls.assert_called_once_with(
            target=pyopl_ide_bootstrap._solve_wrapper,
            args=("model.mod", "data.dat", "scipy", {"time_limit": 4.0}, dummy._solver_ipc.result_queue),
        )
        process.start.assert_called_once_with()
        dummy._append_output.assert_called_once_with(
            f"\nSolver settings: loaded {settings_path}.\n",
            "session-1",
        )

    def test_start_solver_process_failure_releases_queue_and_ignores_late_poll(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve", "session-1")
        result_queue = mock.Mock()
        process = mock.Mock()
        process.start.side_effect = OSError("cannot spawn")
        dummy = SimpleNamespace(
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(),
            status_var=mock.Mock(),
            _append_output=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
        )

        with (
            TemporaryDirectory() as tmpdir,
            mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir),
            mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Queue", return_value=result_queue),
            mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Process", return_value=process),
        ):
            started = OPLIDE._start_solver_process(dummy, "model.mod", "data.dat", "scipy", operation)

        self.assertFalse(started)
        self.assertIsNone(dummy._solver_ipc.handles())
        result_queue.close.assert_called_once_with()
        result_queue.cancel_join_thread.assert_called_once_with()
        dummy._finish_foreground_operation.assert_called_once_with(operation)
        OPLIDE._poll_solver(dummy, operation)
        result_queue.get_nowait.assert_not_called()

    def test_old_solver_poll_does_not_consume_new_solver_queue(self):
        result_queue = mock.Mock()
        dummy = SimpleNamespace(_solver_ipc=pyopl_ide_bootstrap._SolverProcessState(mock.Mock(), result_queue, "new-session"))
        old_operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve", "old-session")

        OPLIDE._poll_solver(dummy, old_operation)

        result_queue.get_nowait.assert_not_called()

    def test_solver_process_stop_terminates_and_kills_unresponsive_worker(self):
        process = mock.Mock()
        process.is_alive.side_effect = [True, True]
        owner = pyopl_ide_bootstrap._SolverProcessState(process=process)

        owner.stop()

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.join.call_args_list, [mock.call(timeout=1.0), mock.call(timeout=1.0)])
        owner.join(timeout=0.1)
        self.assertEqual(process.join.call_args_list[-1], mock.call(timeout=0.1))

    def test_export_model_dialog_supports_lp_and_mps(self):
        class DummyText:
            def __init__(self, value):
                self.value = value

            def get(self, *_args):
                return self.value

        class DummyVar:
            def __init__(self, value="gurobi"):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        dummy = SimpleNamespace(
            model_text=DummyText("dvar float+ x; minimize x; subject to { x >= 1; }"),
            data_text=DummyText(""),
            solver=DummyVar("gurobi"),
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(path="example.mod"),
            status_var=DummyVar(),
            _append_output=mock.Mock(),
            _clear_output=mock.Mock(),
            _ensure_no_active_operation=lambda _label: True,
        )

        captured = {}

        def fake_saveas(**kwargs):
            captured.update(kwargs)
            return "/tmp/example.lp"

        with (
            mock.patch.object(pyopl_ide_bootstrap.filedialog, "asksaveasfilename", side_effect=fake_saveas),
            mock.patch.object(pyopl_ide_bootstrap, "export_opl_model") as export_opl_model,
            mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror"),
            mock.patch.object(pyopl_ide_bootstrap.messagebox, "showwarning"),
        ):
            stdout = StringIO()
            with redirect_stdout(stdout):
                OPLIDE.export_model(dummy)

        self.assertIn(("LP files", "*.lp"), captured.get("filetypes", []))
        self.assertIn(("MPS files", "*.mps"), captured.get("filetypes", []))
        export_opl_model.assert_called_once_with(
            "dvar float+ x; minimize x; subject to { x >= 1; }",
            "",
            "gurobi",
            "/tmp/example.lp",
        )
        dummy._clear_output.assert_not_called()
        dummy._append_output.assert_not_called()
        self.assertIn("Model exported to LP", stdout.getvalue())

    def test_save_exemplar_writes_reviewed_triplet_without_session_transcript(self):
        class DummyText:
            def __init__(self, value):
                self.value = value

            def get(self, *_args):
                return self.value

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_path = root / ".pyopl_session"

            def save_session():
                session_path.write_text('{"complete": true}\n', encoding="utf-8")

            dummy = SimpleNamespace(
                model_text=DummyText("model content\n"),
                data_text=DummyText("data content\n"),
                status_var=mock.Mock(),
                _save_session=save_session,
                _session_file_path=lambda: str(session_path),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _validate_exemplar_draft=OPLIDE._validate_exemplar_draft,
                _write_exemplar=OPLIDE._write_exemplar,
                _ask_short_text=mock.Mock(return_value="fleet"),
                _review_exemplar_draft=mock.Mock(
                    side_effect=lambda draft, _root: OPLIDE._with_reviewed_exemplar_description(
                        draft,
                        "Reviewed fleet assignment problem.",
                    )
                ),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider="openai",
                genai_model="test-model",
                after=lambda _delay, callback, *args: callback(*args),
            )

            with (
                mock.patch.object(Path, "cwd", return_value=root),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
                mock.patch.object(
                    pyopl_ide_bootstrap,
                    "distill_exemplar_description",
                    return_value="Generated fleet assignment problem.",
                ) as distill,
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", _ImmediateThread),
            ):
                OPLIDE.save_exemplar(dummy)

            exemplar = root / "opl_models" / "fleet"
            self.assertEqual((exemplar / "fleet.mod").read_text(encoding="utf-8"), "model content\n")
            self.assertEqual((exemplar / "fleet.dat").read_text(encoding="utf-8"), "data content\n")
            self.assertEqual((exemplar / "fleet.md").read_text(encoding="utf-8"), "Reviewed fleet assignment problem.")
            distill.assert_called_once_with(
                model="model content\n",
                data="data content\n",
                source_session='{"complete": true}\n',
                provider="openai",
                model_name="test-model",
            )
            showerror.assert_not_called()
            dummy._ask_short_text.assert_called_once_with("Save Exemplar", "Exemplar name:")
            self.assertEqual(dummy.status_var.set.call_args_list[-1], mock.call(f"Saved exemplar to {exemplar.resolve()}"))

    def test_save_exemplar_reprompts_for_existing_folder(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "opl_models" / "existing"
            existing.mkdir(parents=True)
            session_path = root / ".pyopl_session"
            session_path.write_text("session", encoding="utf-8")
            dummy = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "model"),
                data_text=SimpleNamespace(get=lambda *_args: "data"),
                status_var=mock.Mock(),
                _save_session=mock.Mock(),
                _session_file_path=lambda: str(session_path),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _validate_exemplar_draft=OPLIDE._validate_exemplar_draft,
                _write_exemplar=OPLIDE._write_exemplar,
                _ask_short_text=mock.Mock(side_effect=["existing", "replacement"]),
                _review_exemplar_draft=mock.Mock(side_effect=lambda draft, _root: draft),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider="openai",
                genai_model="test-model",
                after=lambda _delay, callback, *args: callback(*args),
            )

            with (
                mock.patch.object(Path, "cwd", return_value=root),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
                mock.patch.object(
                    pyopl_ide_bootstrap,
                    "distill_exemplar_description",
                    return_value="description",
                ),
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", _ImmediateThread),
            ):
                OPLIDE.save_exemplar(dummy)

            self.assertEqual(dummy._ask_short_text.call_count, 2)
            showerror.assert_called_once()
            self.assertTrue((root / "opl_models" / "replacement" / "replacement.mod").is_file())

    def test_save_exemplar_cancelled_review_writes_nothing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_path = root / ".pyopl_session"
            session_path.write_text("session", encoding="utf-8")
            dummy = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "model"),
                data_text=SimpleNamespace(get=lambda *_args: ""),
                status_var=mock.Mock(),
                _save_session=mock.Mock(),
                _session_file_path=lambda: str(session_path),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _validate_exemplar_draft=OPLIDE._validate_exemplar_draft,
                _write_exemplar=mock.Mock(),
                _ask_short_text=mock.Mock(return_value="cancelled"),
                _review_exemplar_draft=mock.Mock(return_value=None),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider="openai",
                genai_model="test-model",
                after=lambda _delay, callback, *args: callback(*args),
            )

            with (
                mock.patch.object(Path, "cwd", return_value=root),
                mock.patch.object(pyopl_ide_bootstrap, "distill_exemplar_description", return_value="description"),
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", _ImmediateThread),
            ):
                OPLIDE.save_exemplar(dummy)

            dummy._write_exemplar.assert_not_called()
            self.assertFalse((root / "opl_models").exists())

    def test_save_exemplar_requires_genai_selection_before_session_save(self):
        with TemporaryDirectory() as tmpdir:
            dummy = SimpleNamespace(
                _ask_short_text=mock.Mock(return_value="fleet"),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _save_session=mock.Mock(),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider=None,
                genai_model=None,
            )
            with (
                mock.patch.object(Path, "cwd", return_value=Path(tmpdir)),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
            ):
                OPLIDE.save_exemplar(dummy)

            dummy._save_session.assert_not_called()
            self.assertIn("Select a GenAI provider", showerror.call_args.args[1])

    def test_save_exemplar_distillation_failure_writes_nothing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_path = root / ".pyopl_session"
            session_path.write_text("session", encoding="utf-8")
            dummy = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "model"),
                data_text=SimpleNamespace(get=lambda *_args: "data"),
                status_var=mock.Mock(),
                _save_session=mock.Mock(),
                _session_file_path=lambda: str(session_path),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _write_exemplar=mock.Mock(),
                _ask_short_text=mock.Mock(return_value="failed"),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider="openai",
                genai_model="test-model",
                after=lambda _delay, callback, *args: callback(*args),
            )
            with (
                mock.patch.object(Path, "cwd", return_value=root),
                mock.patch.object(
                    pyopl_ide_bootstrap,
                    "distill_exemplar_description",
                    side_effect=RuntimeError("generation failed"),
                ),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror"),
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", _ImmediateThread),
            ):
                OPLIDE.save_exemplar(dummy)

            dummy._write_exemplar.assert_not_called()
            self.assertFalse((root / "opl_models").exists())

    def test_save_exemplar_revalidates_destination_after_review(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_path = root / ".pyopl_session"
            session_path.write_text("session", encoding="utf-8")

            def create_competing_destination(draft, models_root):
                (models_root / draft.name).mkdir(parents=True)
                return draft

            dummy = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "model"),
                data_text=SimpleNamespace(get=lambda *_args: "data"),
                status_var=mock.Mock(),
                _save_session=mock.Mock(),
                _session_file_path=lambda: str(session_path),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _validate_exemplar_draft=OPLIDE._validate_exemplar_draft,
                _write_exemplar=mock.Mock(),
                _ask_short_text=mock.Mock(return_value="race"),
                _review_exemplar_draft=mock.Mock(side_effect=create_competing_destination),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider="openai",
                genai_model="test-model",
                after=lambda _delay, callback, *args: callback(*args),
            )
            with (
                mock.patch.object(Path, "cwd", return_value=root),
                mock.patch.object(pyopl_ide_bootstrap, "distill_exemplar_description", return_value="description"),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", _ImmediateThread),
            ):
                OPLIDE.save_exemplar(dummy)

            dummy._write_exemplar.assert_not_called()
            self.assertIn("already exists", showerror.call_args.args[1])

    def test_save_exemplar_returns_before_distillation_runs(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_path = root / ".pyopl_session"
            session_path.write_text("session", encoding="utf-8")
            worker = mock.Mock()
            dummy = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "model"),
                data_text=SimpleNamespace(get=lambda *_args: "data"),
                status_var=mock.Mock(),
                _save_session=mock.Mock(),
                _session_file_path=lambda: str(session_path),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _ask_short_text=mock.Mock(return_value="async"),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider="openai",
                genai_model="test-model",
            )
            thread = mock.Mock()
            thread.start = worker

            with (
                mock.patch.object(Path, "cwd", return_value=root),
                mock.patch.object(pyopl_ide_bootstrap, "distill_exemplar_description") as distill,
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", return_value=thread) as thread_cls,
            ):
                OPLIDE.save_exemplar(dummy)

            worker.assert_called_once_with()
            distill.assert_not_called()
            self.assertTrue(dummy._exemplar_save_state.pending)
            self.assertTrue(thread_cls.call_args.kwargs["daemon"])

    def test_save_exemplar_snapshots_inputs_and_blocks_duplicate_preparation(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_path = root / ".pyopl_session"
            session_path.write_text("original session", encoding="utf-8")
            host = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "original model"),
                data_text=SimpleNamespace(get=lambda *_args: "original data"),
                status_var=mock.Mock(),
                _save_session=mock.Mock(),
                _session_file_path=lambda: str(session_path),
                _resolve_exemplar_destination=OPLIDE._resolve_exemplar_destination,
                _ask_short_text=mock.Mock(return_value="fleet"),
                _review_exemplar_draft=mock.Mock(return_value=None),
                _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
                genai_provider="openai",
                genai_model="first-model",
                after=lambda _delay, callback, *args: callback(*args),
            )
            thread = mock.Mock()
            with (
                mock.patch.object(Path, "cwd", return_value=root),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showinfo") as showinfo,
                mock.patch.object(pyopl_ide_bootstrap, "distill_exemplar_description", return_value="description") as distill,
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", return_value=thread) as thread_cls,
            ):
                OPLIDE.save_exemplar(host)
                OPLIDE.save_exemplar(host)
                host.genai_provider, host.genai_model = "other-provider", "other-model"
                host.model_text.get = lambda *_args: "changed model"
                session_path.write_text("changed session", encoding="utf-8")
                thread_cls.call_args.kwargs["target"]()

            showinfo.assert_called_once()
            host._ask_short_text.assert_called_once()
            self.assertEqual(thread_cls.call_count, 1)
            distill.assert_called_once_with(
                model="original model",
                data="original data",
                source_session="original session",
                provider="openai",
                model_name="first-model",
            )
            self.assertFalse(host._exemplar_save_state.pending)
            self.assertFalse((root / "opl_models").exists())

    def test_exemplar_save_state_releases_pending_when_worker_cannot_start(self):
        state = pyopl_ide_bootstrap._ExemplarSaveState()
        with mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", side_effect=RuntimeError("cannot start")):
            with self.assertRaisesRegex(RuntimeError, "cannot start"):
                state.start(mock.Mock())
        self.assertFalse(state.pending)

    def test_exemplar_preparation_completion_after_shutdown_skips_review(self):
        state = pyopl_ide_bootstrap._ExemplarSaveState(pending=True, closed=True)
        host = SimpleNamespace(_exemplar_save_state=state, _review_exemplar_draft=mock.Mock())
        draft = pyopl_ide_bootstrap.ExemplarDraft("name", "model", "data", "description", "session")

        OPLIDE._finish_exemplar_preparation(host, draft, Path("/tmp/opl_models"))

        self.assertFalse(state.pending)
        host._review_exemplar_draft.assert_not_called()

    def test_reviewed_exemplar_description_preserves_model_and_data(self):
        draft = pyopl_ide_bootstrap.ExemplarDraft(
            name="fleet",
            model="original model",
            data="original data",
            description="generated description",
            source_session="session",
        )

        reviewed = OPLIDE._with_reviewed_exemplar_description(draft, "edited description")

        self.assertEqual(reviewed.description, "edited description")
        self.assertEqual(reviewed.model, "original model")
        self.assertEqual(reviewed.data, "original data")

    def test_save_exemplar_name_validation_rejects_folders_and_cancel_does_nothing(self):
        root = Path("/tmp/workspace/opl_models")
        for invalid_name in ("", "../outside", "nested/fleet", r"nested\fleet", "/absolute", "bad:name"):
            with self.subTest(name=invalid_name):
                with self.assertRaises(ValueError):
                    OPLIDE._resolve_exemplar_destination(root, invalid_name)

        dummy = SimpleNamespace(
            _save_session=mock.Mock(),
            _ask_short_text=mock.Mock(return_value=None),
            _exemplar_save_state=pyopl_ide_bootstrap._ExemplarSaveState(),
        )
        OPLIDE.save_exemplar(dummy)
        dummy._save_session.assert_not_called()

    def test_write_exemplar_removes_temporary_folder_after_failure(self):
        with TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "opl_models" / "broken"
            original_write_text = Path.write_text

            def fail_data_write(path, content, *args, **kwargs):
                if path.suffix == ".dat":
                    raise OSError("disk failure")
                return original_write_text(path, content, *args, **kwargs)

            with mock.patch.object(Path, "write_text", autospec=True, side_effect=fail_data_write):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    OPLIDE._write_exemplar(destination, "broken", "model", "data", "session")

            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.iterdir()), [])

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
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(syntax_error="old error"),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(syntax_error="data error"),
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
        dummy.model_text = text

        OPLIDE._schedule_highlight(dummy, text, is_data=False)

        self.assertEqual(dummy._model_editor_state.highlight_after_ids, {})
        self.assertIsNone(dummy._model_editor_state.syntax_error)
        self.assertEqual(dummy._data_editor_state.syntax_error, "data error")
        self.assertEqual(dummy.status_syntax_var.value, "Syntax validation disabled for large text")
        self.assertTrue(any(tag == "ERROR" for tag, _start, _end in text.removed))

    def test_model_and_data_highlight_callbacks_are_independent(self):
        model_text = mock.Mock()
        data_text = mock.Mock()
        dummy = SimpleNamespace(
            model_text=model_text,
            data_text=data_text,
            _shutting_down=False,
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            _highlight_debounce_ms=150,
            _highlight_validate_idle_ms=800,
            after=mock.Mock(side_effect=["model-first", "data-fast", "data-validate", "model-second"]),
            after_cancel=mock.Mock(),
            _run_scheduled_highlight=mock.Mock(),
            _text_too_large_for_highlight=lambda _widget: False,
            highlight=mock.Mock(),
            editor_notebook=SimpleNamespace(select=lambda: "model", index=lambda _tab: 0),
        )
        dummy._cancel_scheduled_highlight = lambda widget, kind: OPLIDE._cancel_scheduled_highlight(dummy, widget, kind)

        OPLIDE._schedule_highlight(dummy, model_text, is_data=False)
        OPLIDE._schedule_highlight(dummy, data_text, is_data=True)
        OPLIDE._schedule_highlight(dummy, model_text, is_data=False)

        self.assertEqual(dummy._model_editor_state.highlight_after_ids, {"validate": "model-second"})
        self.assertEqual(dummy._data_editor_state.highlight_after_ids, {"fast": "data-fast", "validate": "data-validate"})
        dummy.after_cancel.assert_called_once_with("model-first")

        old_version = dummy.after.call_args_list[0].args[-1]
        OPLIDE._run_scheduled_highlight(dummy, model_text, is_data=False, validate=True, version=old_version)
        self.assertEqual(dummy._model_editor_state.highlight_after_ids, {"validate": "model-second"})
        dummy.highlight.assert_not_called()

        OPLIDE._run_scheduled_highlight(dummy, data_text, is_data=True, validate=False)
        self.assertEqual(dummy._data_editor_state.highlight_after_ids, {"validate": "data-validate"})
        dummy.highlight.assert_called_once_with(data_text, is_data=True, validate=False)

    def test_editor_paths_and_saved_baselines_have_independent_state(self):
        class DummyText:
            def __init__(self, value):
                self.value = value

            def get(self, *_args):
                return self.value

        ide = OPLIDE.__new__(OPLIDE)
        ide._model_editor_state = pyopl_ide_bootstrap._EditorBufferState()
        ide._data_editor_state = pyopl_ide_bootstrap._EditorBufferState()
        ide.model_text = DummyText("model")
        ide.data_text = DummyText("data")
        ide._model_editor_state.path = "model.mod"
        ide._data_editor_state.path = "data.dat"
        OPLIDE._mark_editor_baselines_saved(ide)

        self.assertEqual(ide._model_editor_state.path, "model.mod")
        self.assertEqual(ide._data_editor_state.path, "data.dat")
        self.assertFalse(OPLIDE._has_unsaved_editor_changes(ide))

        ide.data_text.value = "changed data"
        self.assertTrue(OPLIDE._has_unsaved_editor_changes(ide))
        self.assertEqual(ide._model_editor_state.saved_text, "model")
        self.assertEqual(ide._data_editor_state.saved_text, "data")

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
        error_message = "Parser Error on line 13: Semantic Error (Line 13): Syntax error in .dat file at end of file (EOF)."
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
            model_text=text,
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(syntax_error=error_message),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
        )
        dummy._text_too_large_for_highlight = lambda text_widget: OPLIDE._text_too_large_for_highlight(dummy, text_widget)
        dummy._refresh_status_context = lambda: OPLIDE._refresh_status_context(dummy)
        dummy._label_for_method = lambda key: OPLIDE._label_for_method(dummy, key)

        OPLIDE._update_caret_position(dummy, text)

        self.assertIn("Syntax error in .dat file at end of file", dummy.status_syntax_var.value)

    def test_run_model_does_not_open_live_progress_window_for_scipy(self):
        class DummyText:
            def __init__(self, value):
                self.value = value

            def get(self, *_args):
                return self.value

        class DummyVar:
            def __init__(self, value=""):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        class DummyProcess:
            def __init__(self, *args, **kwargs):
                self.started = False

            def start(self):
                self.started = True

            def is_alive(self):
                return False

        with TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model.mod")
            data_path = str(Path(tmpdir) / "data.dat")
            operation = pyopl_ide_bootstrap._ForegroundOperation(
                kind="solve",
                label="Solve Model",
                session_id="session-1",
                solver_choice="scipy",
            )
            dummy = SimpleNamespace(
                _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(),
                model_text=DummyText("dvar float+ x;\nminimize x;\nsubject to { x >= 1; }"),
                data_text=DummyText(""),
                solver=DummyVar("scipy"),
                _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
                _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
                status_var=DummyVar(),
                _start_foreground_operation=mock.Mock(return_value=operation),
                _finish_foreground_operation=mock.Mock(),
                _append_output=mock.Mock(),
                _set_run_menu_running=mock.Mock(),
                _show_solver_log_textbox=mock.Mock(),
                _reset_solver_progress_window=mock.Mock(),
                _start_run_timer=mock.Mock(),
                _poll_solver=mock.Mock(),
                after=mock.Mock(),
            )
            dummy._solver_tracks_progress = lambda solver_choice=None: OPLIDE._solver_tracks_progress(dummy, solver_choice)
            dummy._display_solver_progress_enabled = lambda: OPLIDE._display_solver_progress_enabled(dummy)

            with mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Queue", mock.Mock(return_value=mock.Mock())):
                with mock.patch.object(pyopl_ide_bootstrap.multiprocessing, "Process", DummyProcess):
                    OPLIDE.run_model(dummy, model_file_override=model_path, data_file_override=data_path)

        dummy._reset_solver_progress_window.assert_not_called()
        dummy._show_solver_log_textbox.assert_called_once()
        dummy._start_run_timer.assert_called_once_with("Solving model...")
        dummy.after.assert_called_once()

    def test_run_model_validation_failure_does_not_start_solver_ui(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve", "session-1")
        dummy = SimpleNamespace(
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(),
            model_text=SimpleNamespace(get=lambda *_args: "int required = ...;"),
            data_text=SimpleNamespace(get=lambda *_args: ""),
            solver=SimpleNamespace(get=lambda: "scipy"),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            status_var=mock.Mock(),
            _start_foreground_operation=mock.Mock(return_value=operation),
            _finish_foreground_operation=mock.Mock(),
            _append_output=mock.Mock(),
            _set_run_menu_running=mock.Mock(),
            _show_solver_log_textbox=mock.Mock(),
        )

        OPLIDE.run_model(dummy)

        dummy._finish_foreground_operation.assert_called_once_with(operation)
        dummy._append_output.assert_called_once_with("\nError: Data missing for: required\n", "session-1")
        dummy._set_run_menu_running.assert_not_called()
        dummy._show_solver_log_textbox.assert_not_called()

    def test_save_run_model_files_uses_shared_solve_timestamp_for_unsaved_files(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve Model", "session-1")
        dummy = SimpleNamespace(
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            status_var=mock.Mock(),
            _append_output=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
        )

        with TemporaryDirectory() as tmpdir:
            with mock.patch.object(pyopl_ide_bootstrap, "datetime") as datetime_mock:
                datetime_mock.now.return_value = datetime(2026, 8, 24, 0, 47, 51)
                result = OPLIDE._save_run_model_files(
                    dummy,
                    "dvar int x;",
                    "limit = 1;",
                    tmpdir,
                    None,
                    None,
                    operation,
                )

            self.assertEqual(
                result,
                (
                    str(Path(tmpdir) / "tmp_2026-08-24_00-47-51.mod"),
                    str(Path(tmpdir) / "tmp_2026-08-24_00-47-51.dat"),
                ),
            )
            self.assertEqual(Path(result[0]).read_text(encoding="utf-8"), "dvar int x;")
            self.assertEqual(Path(result[1]).read_text(encoding="utf-8"), "limit = 1;")

    def test_poll_solver_streams_logs_to_temporary_textbox(self):
        class DummyProcess:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        class DummyQueue:
            def __init__(self):
                self.items = [("log", "first line\n"), ("success", {"status": "OPTIMAL"})]

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                raise pyopl_ide_bootstrap.queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        operation = pyopl_ide_bootstrap._ForegroundOperation(
            kind="solve",
            label="Solve Model",
            session_id="session-1",
            solver_choice="scipy",
        )
        dummy = SimpleNamespace(
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(DummyProcess(), DummyQueue()),
            _set_run_menu_running=mock.Mock(),
            _append_solver_log_text=mock.Mock(),
            _record_solver_progress=mock.Mock(),
            _stop_run_timer=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _finish_solver_progress=mock.Mock(),
            _display_solve_results=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
            _cleanup_solver_ipc=lambda cancel_queue_thread: OPLIDE._cleanup_solver_ipc(
                dummy, cancel_queue_thread=cancel_queue_thread
            ),
        )

        OPLIDE._poll_solver(dummy, operation)

        dummy._append_solver_log_text.assert_called_once_with("first line\n")
        dummy._display_solve_results.assert_called_once_with(
            {"status": "OPTIMAL"}, session_id="session-1", solver_choice="scipy"
        )
        dummy._restore_output_textbox.assert_called_once()

    def test_malformed_solver_worker_messages_use_error_path(self):
        cases = (
            (("success", "not a result"), "success payload must be a dictionary"),
            (("unknown", "payload"), "unknown solver message kind"),
            (("log",), "expected a (kind, payload) solver message"),
        )
        for message, detail in cases:
            with self.subTest(message=message):
                result_queue = mock.Mock()
                result_queue.get_nowait.return_value = message
                dummy = SimpleNamespace(_solver_ipc=pyopl_ide_bootstrap._SolverProcessState(mock.Mock(), result_queue))

                kind, payload = OPLIDE._drain_solver_queue(dummy, dummy._solver_ipc.process, None)

                self.assertEqual(kind, "error")
                self.assertIn(detail, payload)

    def test_model_error_marks_solver_progress_failed_without_explanation(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation(
            kind="solve",
            label="Solve Model",
            session_id="session-1",
            solver_choice="gurobi",
            explain_after_solve=True,
        )
        payload = {"status": "MODEL_ERROR", "message": "index 2 exceeds its declared range"}
        dummy = SimpleNamespace(
            _finish_solver_progress=mock.Mock(),
            _display_solve_results=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
        )

        with mock.patch.object(OPLIDE, "_handle_solver_explanation") as explain:
            OPLIDE._handle_successful_solver_poll(dummy, payload, operation)

        dummy._finish_solver_progress.assert_called_once_with(payload, status="failed")
        dummy._display_solve_results.assert_called_once_with(
            payload,
            session_id="session-1",
            solver_choice="gurobi",
        )
        dummy._finish_foreground_operation.assert_called_once_with(operation)
        explain.assert_not_called()

    def test_poll_solver_persists_traceback_before_failed_result_message(self):
        class DummyProcess:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        class DummyQueue:
            def __init__(self):
                self.items = [
                    ("log", "Traceback (most recent call last):\nSyntaxError: bad token\n"),
                    ("success", {"status": "FAILED", "message": "Model compilation failed."}),
                ]

            def get_nowait(self):
                if self.items:
                    return self.items.pop(0)
                raise pyopl_ide_bootstrap.queue.Empty

            def close(self):
                pass

            def join_thread(self):
                pass

        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve Model", "session-1")
        dummy = SimpleNamespace(
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(DummyProcess(), DummyQueue()),
            _solver_log_text=SimpleNamespace(
                get=lambda *_args: "Traceback (most recent call last):\nSyntaxError: bad token\n"
            ),
            _append_solver_log_text=mock.Mock(),
            _record_solver_progress=mock.Mock(),
            _set_run_menu_running=mock.Mock(),
            _stop_run_timer=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _finish_solver_progress=mock.Mock(),
            _append_output=mock.Mock(),
            status_var=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
            _cleanup_solver_ipc=lambda cancel_queue_thread: OPLIDE._cleanup_solver_ipc(
                dummy, cancel_queue_thread=cancel_queue_thread
            ),
        )

        OPLIDE._poll_solver(dummy, operation)

        self.assertEqual(
            dummy._append_output.call_args_list,
            [
                mock.call("Traceback (most recent call last):\nSyntaxError: bad token\n", "session-1"),
                mock.call("\nError:\nModel compilation failed.\n", "session-1"),
            ],
        )
        dummy._finish_solver_progress.assert_called_once_with(status="failed")
        dummy._finish_foreground_operation.assert_called_once_with(operation)

    def test_restore_output_textbox_refreshes_hidden_session_history(self):
        class DummyFrame:
            def __init__(self):
                self.destroyed = False

            def winfo_exists(self):
                return True

            def destroy(self):
                self.destroyed = True

        class DummyLogText:
            def __init__(self):
                self.frame = DummyFrame()
                self.destroyed = False

            def winfo_exists(self):
                return True

            def destroy(self):
                self.destroyed = True

        class DummyOutputText:
            def __init__(self):
                self.content = "old visible content"
                self.state = "disabled"
                self.packed = False
                self.seen = False

            def winfo_exists(self):
                return True

            def winfo_manager(self):
                return "pack"

            def pack(self, **_kwargs):
                self.packed = True

            def config(self, **kwargs):
                if "state" in kwargs:
                    self.state = kwargs["state"]

            def delete(self, *_args):
                self.content = ""

            def insert(self, *_args):
                self.content += _args[-1]

            def see(self, *_args):
                self.seen = True

        log_text = DummyLogText()
        output_text = DummyOutputText()
        dummy = SimpleNamespace(
            _solver_log_text=log_text,
            output_text=output_text,
            _session_state=pyopl_ide_bootstrap._OutputSessionState(
                sessions={"session-1": "Solve header\nFinal result\n"},
                current_id="session-1",
                viewing_id="session-1",
            ),
        )
        dummy._destroy_scrolled_text = lambda text_widget: OPLIDE._destroy_scrolled_text(dummy, text_widget)

        OPLIDE._restore_output_textbox(dummy)

        self.assertTrue(log_text.frame.destroyed)
        self.assertFalse(log_text.destroyed)
        self.assertIsNone(dummy._solver_log_text)
        self.assertTrue(output_text.packed)
        self.assertEqual(output_text.content, "Solve header\nFinal result\n")
        self.assertEqual(output_text.state, "disabled")
        self.assertTrue(output_text.seen)

    def test_finish_solver_progress_for_scipy_opens_summary_without_chart(self):
        class DummyWindow:
            def __init__(self):
                self.geometry_values = []
                self.rowconfigure_calls = []
                self.deiconified = False

            def winfo_exists(self):
                return True

            def geometry(self, value):
                self.geometry_values.append(value)

            def rowconfigure(self, *args, **kwargs):
                self.rowconfigure_calls.append((args, kwargs))

            def deiconify(self):
                self.deiconified = True

            def lift(self):
                pass

        class DummyCanvas:
            def __init__(self):
                self.removed = False
                self.deleted = False

            def winfo_exists(self):
                return True

            def grid(self, *args, **kwargs):
                self.removed = False

            def grid_remove(self):
                self.removed = True

            def delete(self, *_args):
                self.deleted = True

        class DummyStatsFrame:
            def winfo_children(self):
                return []

        class DummyVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        window = DummyWindow()
        canvas = DummyCanvas()
        status_var = DummyVar()
        dummy = SimpleNamespace(
            _current_solver_choice="scipy",
            _solver_progress_update_after_id=None,
            _solver_progress_samples=[],
            _solver_progress_pending_sample=None,
            _solver_progress_window=window,
            _solver_progress_canvas=canvas,
            _solver_progress_stats_frame=DummyStatsFrame(),
            _solver_progress_status_var=status_var,
            _solver_progress_stat_vars={},
            _solver_progress_rolling_seconds=120.0,
            after_cancel=mock.Mock(),
            _update_solver_progress_stats=mock.Mock(),
            _append_solver_progress_sample=mock.Mock(),
        )
        dummy._solver_tracks_progress = lambda solver_choice=None: OPLIDE._solver_tracks_progress(dummy, solver_choice)
        dummy._display_solver_progress_enabled = lambda: OPLIDE._display_solver_progress_enabled(dummy)
        dummy._set_solver_progress_status = lambda text: OPLIDE._set_solver_progress_status(dummy, text)
        dummy._reset_solver_progress_window = lambda solver_choice: OPLIDE._reset_solver_progress_window(dummy, solver_choice)

        scipy_stats = OPLIDE._solver_progress_stats(
            dummy,
            {
                "objective_value": 3.0,
                "dual_bound": 2.5,
                "gap": 0.1,
                "runtime": 1.2,
                "iterations": 4,
                "nodes": 9,
                "solutions": 2,
            },
        )
        self.assertEqual(
            [label for label, _value in scipy_stats],
            ["Objective", "Dual Bound", "Gap", "Nodes", "Runtime", "Iterations"],
        )

        OPLIDE._finish_solver_progress(
            dummy,
            {
                "objective_value": 3.0,
                "stats": {
                    "mip_dual_bound": 2.5,
                    "mip_gap": 0.1,
                    "mip_node_count": 9,
                    "time": 1.2,
                    "nit": 4,
                },
            },
            status="complete",
        )

        summary = dummy._update_solver_progress_stats.call_args.args[0]
        self.assertEqual(summary["dual_bound"], 2.5)
        self.assertEqual(summary["gap"], 0.1)
        self.assertEqual(summary["nodes"], 9)

        self.assertEqual(window.geometry_values[-1], "520x220")
        self.assertTrue(canvas.removed)
        self.assertFalse(canvas.deleted)
        self.assertTrue(window.deiconified)
        self.assertEqual(status_var.value, "Solve complete.")

        OPLIDE._redraw_solver_progress_chart(dummy)
        self.assertFalse(canvas.deleted)

    def test_solver_progress_display_can_be_disabled(self):
        class DummyWindow:
            def __init__(self):
                self.withdrawn = False

            def winfo_exists(self):
                return True

            def withdraw(self):
                self.withdrawn = True

        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        window = DummyWindow()
        dummy = SimpleNamespace(
            display_solver_progress_var=DummyVar(False),
            _solver_progress_update_after_id="after-id",
            _solver_progress_pending_sample={"runtime": 1.0},
            _solver_progress_window=window,
            after_cancel=mock.Mock(),
            _solver_tracks_progress=lambda *_args: True,
            _reset_solver_progress_window=mock.Mock(),
            _update_solver_progress_stats=mock.Mock(),
            _append_solver_progress_sample=mock.Mock(),
            _set_solver_progress_status=mock.Mock(),
        )
        dummy._display_solver_progress_enabled = lambda: OPLIDE._display_solver_progress_enabled(dummy)
        dummy._hide_solver_progress_window = lambda: OPLIDE._hide_solver_progress_window(dummy)

        OPLIDE._finish_solver_progress(dummy, {"objective_value": 3.0}, status="complete")

        dummy.after_cancel.assert_called_once_with("after-id")
        self.assertIsNone(dummy._solver_progress_update_after_id)
        self.assertIsNone(dummy._solver_progress_pending_sample)
        self.assertTrue(window.withdrawn)
        dummy._reset_solver_progress_window.assert_not_called()
        dummy._update_solver_progress_stats.assert_not_called()
        dummy._append_solver_progress_sample.assert_not_called()
        dummy._set_solver_progress_status.assert_not_called()

    def test_stop_model_when_idle_does_not_finish_solver_progress(self):
        dummy = SimpleNamespace(
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(),
            _active_operation=None,
            _cleanup_solver_ipc=mock.Mock(),
            _stop_run_timer=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _set_run_menu_running=mock.Mock(),
            _finish_solver_progress=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
            _append_output=mock.Mock(),
        )

        OPLIDE.stop_model(dummy)

        dummy._cleanup_solver_ipc.assert_called_once_with(cancel_queue_thread=True)
        dummy._stop_run_timer.assert_called_once()
        dummy._restore_output_textbox.assert_called_once()
        dummy._set_run_menu_running.assert_called_once_with(False)
        dummy._finish_solver_progress.assert_not_called()
        dummy._append_output.assert_not_called()

    def test_stop_model_releases_queue_and_ignores_late_poll(self):
        process = mock.Mock()
        process.is_alive.side_effect = [True, True, False]
        result_queue = mock.Mock()
        operation = pyopl_ide_bootstrap._ForegroundOperation("solve", "Solve", "session-1")
        dummy = SimpleNamespace(
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(process, result_queue),
            _active_operation=operation,
            _stop_run_timer=mock.Mock(),
            _restore_output_textbox=mock.Mock(),
            _append_output=mock.Mock(),
            status_var=mock.Mock(),
            _finish_solver_progress=mock.Mock(),
            _set_run_menu_running=mock.Mock(),
            _finish_foreground_operation=mock.Mock(),
        )
        dummy._cleanup_solver_ipc = lambda cancel_queue_thread: OPLIDE._cleanup_solver_ipc(
            dummy, cancel_queue_thread=cancel_queue_thread
        )

        OPLIDE.stop_model(dummy)
        OPLIDE._poll_solver(dummy, operation)

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        result_queue.cancel_join_thread.assert_called_once_with()
        result_queue.get_nowait.assert_not_called()
        dummy._append_output.assert_called_once_with("\nExecution stopped by user.\n", "session-1")
        dummy._finish_solver_progress.assert_called_once_with(status="stopped")
        dummy._finish_foreground_operation.assert_called_once_with(operation)

    def test_save_settings_persists_display_solver_progress(self):
        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "settings.json"
            dummy = SimpleNamespace(
                _config_path=config_path,
                theme_var=DummyVar("flatly"),
                solver=DummyVar("scipy"),
                current_font_size=12,
                verbose_llm_var=DummyVar(False),
                display_solver_progress_var=DummyVar(False),
                genai_provider=None,
                genai_model=None,
                genai_method_var=DummyVar("pyopl_generative"),
            )

            OPLIDE._save_settings(dummy)

            payload = pyopl_ide_bootstrap.json.loads(config_path.read_text(encoding="utf-8"))
            loaded = OPLIDE._load_settings(dummy)

        self.assertFalse(payload["display-solver-progress"])
        self.assertEqual(payload["solver"], "scipy")
        self.assertEqual(loaded, payload)

    def test_initialize_preferences_keeps_valid_values_with_invalid_font_size(self):
        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        saved = {
            "font-size": "not a number",
            "theme": "darkly",
            "solver": "scipy",
            "verbose-llm-logs": True,
            "display-solver-progress": False,
            "genai-method": "pyopl_standard",
            "genai-selection": {"provider": "elm", "model": "selected-model"},
        }
        dummy = SimpleNamespace(
            _init_settings_storage=lambda: None,
            _load_settings=lambda: saved,
            current_font_size=12,
            debug=False,
            solver=DummyVar("gurobi"),
            genai_selection_var=DummyVar(""),
        )
        dummy._restore_genai_selection = lambda value: OPLIDE._restore_genai_selection(dummy, value)

        with (
            mock.patch.object(pyopl_ide_bootstrap.tk, "BooleanVar", DummyVar),
            mock.patch.object(pyopl_ide_bootstrap.tk, "IntVar", DummyVar),
            mock.patch.object(pyopl_ide_bootstrap.tk, "StringVar", DummyVar),
        ):
            theme, loaded = OPLIDE._initialize_preferences(dummy)

        self.assertIs(loaded, saved)
        self.assertEqual(theme, "darkly")
        self.assertEqual(dummy.current_font_size, 12)
        self.assertEqual(dummy.solver.get(), "scipy")
        self.assertFalse(dummy.display_solver_progress_var.get())
        self.assertTrue(dummy.verbose_llm_var.get())
        self.assertEqual(dummy.genai_method_var.get(), "pyopl_standard")
        self.assertEqual(dummy.genai_selection_var.get(), "elm|selected-model")

    def test_invalid_saved_genai_method_does_not_import_composer_mode(self):
        for saved_method in ("ask", "generate", "pyopl.genai.ask", "", None, []):
            with self.subTest(saved_method=saved_method):
                preferences = pyopl_ide_bootstrap._parse_ide_preferences({"genai-method": saved_method}, 12, False)
                self.assertEqual(preferences.genai_method, "pyopl_generative")

                selection = mock.Mock()
                selection.get.return_value = preferences.genai_method
                generator = OPLIDE._import_selected_genai_module(SimpleNamespace(genai_method_var=selection))
                self.assertEqual(generator.__name__, "pyopl.genai.pyopl_generative")
                selection.set.assert_not_called()

    def test_invalid_runtime_genai_method_uses_default_without_import_warning(self):
        selection = mock.Mock()
        selection.get.return_value = "ask"
        with mock.patch.object(pyopl_ide_bootstrap.logging.getLogger(pyopl_ide_bootstrap.__name__), "warning") as warning:
            generator = OPLIDE._import_selected_genai_module(SimpleNamespace(genai_method_var=selection))
        warning.assert_not_called()
        self.assertEqual(generator.__name__, "pyopl.genai.pyopl_generative")
        selection.set.assert_called_once_with("pyopl_generative")

    def test_selected_genai_import_propagates_missing_dependency(self):
        selection = mock.Mock()
        selection.get.return_value = "pyopl_standard"
        with mock.patch(
            "importlib.import_module", side_effect=ModuleNotFoundError("missing dependency", name="optional_dependency")
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, "missing dependency"):
                OPLIDE._import_selected_genai_module(SimpleNamespace(genai_method_var=selection))
        selection.set.assert_not_called()

    def test_missing_selected_genai_module_falls_back(self):
        selection = mock.Mock()
        selection.get.return_value = "pyopl_standard"
        generator = mock.sentinel.generator
        with (
            mock.patch(
                "importlib.import_module",
                side_effect=[ModuleNotFoundError("missing method", name="pyopl.genai.pyopl_standard"), generator],
            ) as import_module,
            mock.patch.object(pyopl_ide_bootstrap.logging.getLogger(pyopl_ide_bootstrap.__name__), "warning"),
        ):
            self.assertIs(OPLIDE._import_selected_genai_module(SimpleNamespace(genai_method_var=selection)), generator)
        import_module.assert_any_call("pyopl.genai.pyopl_standard")
        import_module.assert_any_call("pyopl.genai.pyopl_generative")
        selection.set.assert_called_once_with("pyopl_generative")

    def test_load_settings_rejects_non_object_json_with_warning(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("[]", encoding="utf-8")
            dummy = SimpleNamespace(_config_path=path)
            output = StringIO()

            with redirect_stdout(output):
                loaded = OPLIDE._load_settings(dummy)

        self.assertEqual(loaded, {})
        self.assertIn("settings must contain a JSON object", output.getvalue())

    def test_solver_selection_persists_and_refreshes_status(self):
        class DummyVar:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        dummy = SimpleNamespace(
            solver=DummyVar("scipy"),
            _current_solver_choice="gurobi",
            _refresh_status_context=mock.Mock(),
            _save_settings=mock.Mock(),
        )

        OPLIDE._on_solver_selected(dummy)

        self.assertEqual(dummy._current_solver_choice, "scipy")
        dummy._refresh_status_context.assert_called_once()
        dummy._save_settings.assert_called_once()

    def test_highlight_suppresses_model_eof_while_typing(self):
        class DummyText:
            def __init__(self, value):
                self.value = value
                self.tags = []

            def get(self, *_args):
                return self.value

            def count(self, *_args):
                return (len(self.value),)

            def tag_add(self, tag, *ranges):
                for start, end in zip(ranges[0::2], ranges[1::2]):
                    self.tags.append((tag, start, end))

            def tag_remove(self, *_args):
                pass

        text = DummyText("dvar float+ x;\nminimize x;\nsubject to {\n")
        dummy = SimpleNamespace(
            model_text=text,
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(syntax_error="old error"),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
        )
        dummy._text_too_large_for_highlight = lambda text_widget: OPLIDE._text_too_large_for_highlight(dummy, text_widget)
        dummy._disable_highlight_for_large_text = lambda text_widget: OPLIDE._disable_highlight_for_large_text(
            dummy, text_widget
        )
        dummy._clear_highlight_tags = lambda text_widget: OPLIDE._clear_highlight_tags(dummy, text_widget)
        dummy._is_transient_live_parse_error = OPLIDE._is_transient_live_parse_error

        OPLIDE.highlight(dummy, text, is_data=False, validate=True)

        self.assertFalse(any(tag == "ERROR" for tag, _start, _end in text.tags))
        self.assertIsNone(dummy._model_editor_state.syntax_error)

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

    def test_poll_genai_model_discovery_delivers_elm_models(self):
        discovery_queue = queue.Queue()
        provider_models = {"openai": [], "elm": ["gpt-5.4"], "google": [], "ollama": []}
        discovery_queue.put(provider_models)
        dummy = SimpleNamespace(
            _shutting_down=False,
            _genai_loading=True,
            after=mock.Mock(),
            _populate_genai_model_menus=mock.Mock(),
        )

        OPLIDE._poll_genai_model_discovery(dummy, discovery_queue)

        self.assertFalse(dummy._genai_loading)
        dummy._populate_genai_model_menus.assert_called_once_with(provider_models)
        dummy.after.assert_not_called()

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

    def test_highlight_ranges_map_model_tokens_and_data_keywords_to_lines(self):
        code = "param limit = 12;\nset S = {true, 3.5};"
        index = pyopl_ide_bootstrap._make_text_indexer(code)
        self.assertEqual(index(-1), "1.0")
        self.assertEqual(index(len(code) + 5), index(len(code)))

        tokens = [
            SimpleNamespace(type="PARAM", index=0, value="param"),
            SimpleNamespace(type="NAME", index=6, value="limit"),
            SimpleNamespace(type="UNKNOWN", index=12, value="="),
        ]
        self.assertEqual(
            pyopl_ide_bootstrap._token_highlight_ranges(tokens, index),
            {"PARAM": ["1.0", "1.5"], "NAME": ["1.6", "1.11"]},
        )
        ranges = pyopl_ide_bootstrap._data_highlight_ranges(code, index)
        self.assertEqual(ranges["PARAM"], ["1.0", "1.5"])
        self.assertEqual(ranges["SET"], ["2.0", "2.3"])
        self.assertEqual(ranges["BOOLEAN"], ["2.9", "2.13"])
        self.assertEqual(ranges["NUMBER"], ["1.14", "1.16", "2.15", "2.18"])

    def test_find_previous_match_wraps_and_handles_case_and_regex(self):
        find = pyopl_ide_bootstrap._find_previous_match
        self.assertEqual(find("Alpha beta alpha", "alpha", 7, False, False), (0, 5))
        self.assertEqual(find("Alpha beta alpha", "alpha", 0, False, False), (11, 16))
        self.assertEqual(find("Alpha beta alpha", "alpha", 7, False, True), (11, 16))
        self.assertEqual(find("abc 123 abc", r"\d+", 8, True, True), (4, 7))

    def test_find_replace_dialog_owns_shortcuts_and_replace_all_without_tk(self):
        class DummyVar:
            instances = []

            def __init__(self, value=""):
                self.value = value
                self.instances.append(self)

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        buttons = {}
        entries = []

        def make_button(*_args, **kwargs):
            button = mock.Mock()
            button.command = kwargs["command"]
            buttons[kwargs["text"]] = button
            return button

        def make_entry(*_args, **kwargs):
            entry = mock.Mock()
            entry.get.side_effect = kwargs["textvariable"].get
            entries.append(entry)
            return entry

        dialog = mock.Mock()
        dialog.winfo_exists.return_value = True
        model_text, data_text = mock.Mock(), mock.Mock()
        model_text.get.return_value = "alpha beta Alpha"
        host = SimpleNamespace(
            data_text=data_text,
            _ensure_no_active_operation=mock.Mock(return_value=True),
            _undo=mock.Mock(),
            _redo=mock.Mock(),
            _get_active_editor=mock.Mock(return_value=model_text),
            _on_text_change=mock.Mock(),
        )
        with (
            mock.patch.object(pyopl_ide_bootstrap.tk, "Toplevel", return_value=dialog),
            mock.patch.object(pyopl_ide_bootstrap.tk, "StringVar", DummyVar),
            mock.patch.object(pyopl_ide_bootstrap.tk, "BooleanVar", DummyVar),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Frame", side_effect=lambda *_a, **_kw: mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Label", side_effect=lambda *_a, **_kw: mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Checkbutton", side_effect=lambda *_a, **_kw: mock.Mock()),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Entry", side_effect=make_entry),
            mock.patch.object(pyopl_ide_bootstrap.ttk, "Button", side_effect=make_button),
        ):
            OPLIDE._open_find_replace_dialog(host, replace=True)
            DummyVar.instances[0].set("alpha")
            DummyVar.instances[1].set("omega")
            model_text.tag_ranges.return_value = ()
            model_text.search.return_value = "1.0"
            buttons["Next"].command()
            buttons["Replace All"].command()
            data_text.get.return_value = "alpha"
            host._get_active_editor.return_value = data_text
            buttons["Replace All"].command()
            undo = next(call.args[1] for call in dialog.bind.call_args_list if call.args[0] == "<Control-z>")
            undo(None)
            buttons["Close"].command()

        model_text.replace.assert_called_once_with("1.0", "end-1c", "omega beta omega")
        model_text.tag_add.assert_any_call("find_match", "1.0", "1.0+5c")
        data_text.replace.assert_called_once_with("1.0", "end-1c", "omega")
        self.assertEqual(
            host._on_text_change.call_args_list,
            [mock.call(model_text, is_data=False), mock.call(data_text, is_data=True)],
        )
        host._undo.assert_called_once_with()
        entries[0].focus_set.assert_called_once_with()
        dialog.destroy.assert_called_once_with()

    def test_append_output_can_target_non_current_session(self):
        ide = OPLIDE.__new__(OPLIDE)
        ide.tk = SimpleNamespace()
        ide._session_state = pyopl_ide_bootstrap._OutputSessionState()
        ide.request_listbox = mock.Mock()
        ide._save_session = lambda: None
        ide._show_output_session = lambda session_id: None

        first = OPLIDE._begin_new_output_session(ide, "First")
        second = OPLIDE._begin_new_output_session(ide, "Second")

        OPLIDE._append_output(ide, "\nfirst-session-update\n", first)

        self.assertIn("first-session-update", ide._session_state.sessions[first])
        self.assertNotIn("first-session-update", ide._session_state.sessions[second])

    def test_real_ide_session_fields_share_one_state_through_create_append_remove(self):
        class DummyText:
            def __init__(self, value):
                self.value = value

            def get(self, *_args):
                return self.value

        ide = OPLIDE.__new__(OPLIDE)
        ide._session_state = pyopl_ide_bootstrap._OutputSessionState()
        ide.model_text = DummyText("model\n")
        ide.data_text = DummyText("data\n")

        session_id, _display = ide._session_state.begin("Solve:", datetime(2026, 1, 1))
        OPLIDE._snapshot_output_session_artifacts(ide, session_id)
        ide._session_state.select(session_id, make_current=True)
        ide._session_state.append("finished")

        self.assertEqual(ide._session_state.sessions[session_id], "Solve:\nfinished")
        self.assertEqual(set(ide._session_state.model_snapshots.values()), {"model"})
        self.assertEqual(set(ide._session_state.data_snapshots.values()), {"data"})

        ide._session_state.remove(session_id, 0)

        self.assertEqual(ide._session_state.ids, [])
        self.assertEqual(ide._session_state.sessions, {})
        self.assertEqual(ide._session_state.model_snapshots, {})
        self.assertIsNone(ide._session_state.current_id)
        self.assertIsNone(ide._session_state.viewing_id)

        ide._session_state.fold_view_states = {"model.mod": {"folded_lines": [1], "edited_at": ""}}
        OPLIDE._clear_output_session_state(ide)
        self.assertEqual(ide._session_state.fold_view_states, {})

    def test_save_session_preserves_payload_shape_from_grouped_state(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".pyopl_session"
            ide = OPLIDE.__new__(OPLIDE)
            ide._session_state = pyopl_ide_bootstrap._OutputSessionState(
                sessions={"s1": "Solve:\nresult"},
                ids=["s1"],
                display={"s1": "2026-01-01 10:00:00 • Solve"},
                labels={"s1": "Solve"},
                timestamps={"s1": "2026-01-01 10:00:00"},
                fold_view_states={"model.mod": {"folded_lines": [2], "edited_at": "now"}},
                current_id="s1",
                viewing_id="s1",
            )
            ide._model_editor_state = pyopl_ide_bootstrap._EditorBufferState()
            ide._data_editor_state = pyopl_ide_bootstrap._EditorBufferState()
            ide._model_editor_state.path = "model.mod"
            ide._data_editor_state.path = None
            ide.model_gutter = None
            ide.data_gutter = None
            ide._session_file_path = lambda: str(path)
            OPLIDE._record_output_session_artifacts(ide, "s1", model_text="model", data_text="data")

            OPLIDE._save_session(ide)

            saved = pyopl_ide_bootstrap.json.loads(path.read_text(encoding="utf-8"))
            restored = OPLIDE.__new__(OPLIDE)
            restored._session_state = pyopl_ide_bootstrap._OutputSessionState()
            restored.request_listbox = mock.Mock()
            restored.output_text = mock.Mock()
            restored.output_text.winfo_exists.return_value = False
            OPLIDE._restore_session_history(restored, saved)
            OPLIDE._restore_fold_view_states(restored, saved)

        self.assertEqual(
            set(saved),
            {
                "output_sessions",
                "output_session_ids",
                "output_session_display",
                "output_session_label",
                "output_session_timestamp",
                "output_session_artifacts",
                "model_snapshots",
                "data_snapshots",
                "current_output_session_id",
                "viewing_output_session_id",
                "model_file",
                "data_file",
                "fold_view_states",
                "saved_at",
            },
        )
        self.assertEqual(saved["output_sessions"], {"s1": "Solve:\nresult"})
        self.assertEqual(saved["model_file"], "model.mod")
        self.assertIsNone(saved["data_file"])
        self.assertEqual(saved["fold_view_states"]["model.mod"]["folded_lines"], [2])
        self.assertEqual(set(saved["model_snapshots"].values()), {"model"})
        self.assertEqual(set(saved["data_snapshots"].values()), {"data"})
        self.assertEqual(restored._session_state.sessions, saved["output_sessions"])
        self.assertEqual(restored._session_state.artifacts, saved["output_session_artifacts"])
        self.assertEqual(restored._session_state.current_id, "s1")
        self.assertEqual(restored._session_state.viewing_id, "s1")
        self.assertEqual(restored._session_state.fold_view_states[OPLIDE._fold_state_key("model.mod")]["folded_lines"], [2])

    def test_selecting_another_session_does_not_replace_current_request(self):
        ide = OPLIDE.__new__(OPLIDE)
        ide._session_state = pyopl_ide_bootstrap._OutputSessionState(current_id="s1", viewing_id="s1")

        ide._session_state.select("s2")
        self.assertEqual(ide._session_state.current_id, "s1")
        self.assertEqual(ide._session_state.viewing_id, "s2")

        ide._session_state.select(None)
        self.assertEqual(ide._session_state.current_id, "s1")
        self.assertIsNone(ide._session_state.viewing_id)

        ide._session_state.select("s2", make_current=True)
        self.assertEqual(ide._session_state.current_id, "s2")

    def test_begin_new_output_session_initializes_artifact_metadata(self):
        ide = OPLIDE.__new__(OPLIDE)
        ide.tk = SimpleNamespace()
        ide._session_state = pyopl_ide_bootstrap._OutputSessionState()
        ide.request_listbox = mock.Mock()
        ide._save_session = lambda: None
        ide._show_output_session = lambda session_id: None

        session_id = OPLIDE._begin_new_output_session(ide, "Solve: Solving model...")

        self.assertIn(session_id, ide._session_state.timestamps)
        self.assertIn(session_id, ide._session_state.artifacts)
        self.assertEqual(ide._session_state.artifacts[session_id], {})

    def test_begin_new_output_session_snapshots_current_editor_state(self):
        class DummyText:
            def __init__(self, content):
                self.content = content

            def get(self, *_args):
                return self.content

        ide = OPLIDE.__new__(OPLIDE)
        ide._session_state = pyopl_ide_bootstrap._OutputSessionState()
        ide.request_listbox = mock.Mock()
        ide._save_session = lambda: None
        ide._show_output_session = lambda session_id: None
        ide.model_text = DummyText("dvar int x;\n")
        ide.data_text = DummyText("x = 3;\n")

        session_id = OPLIDE._begin_new_output_session(ide, "Solve: Solving model...")
        artifacts = ide._session_state.artifacts[session_id]

        self.assertEqual(artifacts["model_hash"], OPLIDE._session_snapshot_hash("dvar int x;"))
        self.assertEqual(artifacts["data_hash"], OPLIDE._session_snapshot_hash("x = 3;"))
        self.assertEqual(ide._session_state.model_snapshots, {artifacts["model_hash"]: "dvar int x;"})
        self.assertEqual(ide._session_state.data_snapshots, {artifacts["data_hash"]: "x = 3;"})

    def test_output_session_snapshots_are_deduplicated(self):
        dummy = SimpleNamespace(_session_state=pyopl_ide_bootstrap._OutputSessionState())

        OPLIDE._record_output_session_artifacts(dummy, "s1", model_text="same model", data_text="same data")
        OPLIDE._record_output_session_artifacts(dummy, "s2", model_text="same model", data_text="same data")

        self.assertEqual(dummy._session_state.artifacts["s1"], dummy._session_state.artifacts["s2"])
        self.assertEqual(len(dummy._session_state.model_snapshots), 1)
        self.assertEqual(len(dummy._session_state.data_snapshots), 1)

    def test_prune_session_snapshots_keeps_only_referenced_content(self):
        dummy = SimpleNamespace(_session_state=pyopl_ide_bootstrap._OutputSessionState())
        OPLIDE._record_output_session_artifacts(dummy, "s1", model_text="old model", data_text="shared data")
        OPLIDE._record_output_session_artifacts(dummy, "s2", model_text="new model", data_text="shared data")
        old_model_hash = OPLIDE._session_snapshot_hash("old model")

        dummy._session_state.artifacts.pop("s1")
        OPLIDE._prune_session_snapshots(dummy)

        self.assertNotIn(old_model_hash, dummy._session_state.model_snapshots)
        self.assertEqual(set(dummy._session_state.model_snapshots.values()), {"new model"})
        self.assertEqual(set(dummy._session_state.data_snapshots.values()), {"shared data"})

    def test_remove_output_session_prunes_only_unreferenced_snapshots(self):
        state = pyopl_ide_bootstrap._OutputSessionState(
            ids=["s1", "s2"],
            sessions={"s1": "first", "s2": "second"},
            display={"s1": "first", "s2": "second"},
            labels={"s1": "Solve", "s2": "Solve"},
            timestamps={"s1": "earlier", "s2": "later"},
            current_id="s1",
            viewing_id="s2",
        )
        state.record_artifacts(
            state.artifacts, state.model_snapshots, state.data_snapshots, "s1", "first model", "shared data"
        )
        state.record_artifacts(
            state.artifacts, state.model_snapshots, state.data_snapshots, "s2", "second model", "shared data"
        )

        state.remove("s1", 0)

        self.assertEqual(state.ids, ["s2"])
        self.assertEqual(state.sessions, {"s2": "second"})
        self.assertEqual(state.artifacts.keys(), {"s2"})
        self.assertEqual(set(state.model_snapshots.values()), {"second model"})
        self.assertEqual(set(state.data_snapshots.values()), {"shared data"})
        self.assertIsNone(state.current_id)
        self.assertEqual(state.viewing_id, "s2")

    def test_begin_new_output_session_disambiguates_same_second_display_labels(self):
        state = pyopl_ide_bootstrap._OutputSessionState()

        first = state.make_display("2026-05-03 12:00:00", "Ask")
        state.display["session-1"] = first
        second = state.make_display("2026-05-03 12:00:00", "Ask")

        self.assertEqual(first, "2026-05-03 12:00:00 • Ask")
        self.assertEqual(second, "2026-05-03 12:00:00 • Ask (2)")

    def test_rename_session_updates_record_and_visible_label_without_collision(self):
        timestamp = "2026-05-03 12:00:00"
        listbox = mock.Mock()
        ide = OPLIDE.__new__(OPLIDE)
        ide._session_state = pyopl_ide_bootstrap._OutputSessionState(
            ids=["s1", "s2"],
            timestamps={"s1": timestamp, "s2": timestamp},
            labels={"s1": "First", "s2": "Taken"},
            display={"s1": f"{timestamp} • First", "s2": f"{timestamp} • Taken"},
        )
        ide._get_selected_request_session_id = lambda: "s1"
        ide._ask_short_text = lambda **_kwargs: "Taken"
        ide._save_session = mock.Mock()
        ide.request_listbox = listbox

        OPLIDE._rename_selected_request(ide)

        self.assertEqual(ide._session_state.labels, {"s1": "Taken", "s2": "Taken"})
        self.assertEqual(ide._session_state.display["s1"], f"{timestamp} • Taken (2)")
        self.assertEqual(ide._session_state.display["s2"], f"{timestamp} • Taken")
        listbox.insert.assert_called_once_with(0, f"{timestamp} • Taken (2)")
        ide._save_session.assert_called_once()

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
                _genai_revisions=pyopl_ide_bootstrap._GenAIRevisionState(
                    pyopl_ide_bootstrap._PendingGenAIRevision(
                        revised_model="dvar int y;\n",
                        revised_data="y = 2;\n",
                        current_model="dvar int x;\n",
                        current_data="old = 1;\n",
                        model_path=str(original_model),
                        data_path=str(original_data),
                        safe_ts="2026-05-03_01-20-41",
                        had_data_file=True,
                        session_id="session-1",
                    )
                ),
                _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(path=str(original_model)),
                _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(path=str(original_data)),
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
                _get_editor_text=lambda widget: widget.get(),
            )
            dummy._record_output_session_artifacts = lambda *args, **kwargs: artifact_calls.append((args, kwargs))

            with mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir):
                OPLIDE._apply_pending_genai_revisions(dummy)

            self.assertEqual(Path(dummy._model_editor_state.path).name, "model_2026-05-03_01-20-41.mod")
            self.assertEqual(Path(dummy._data_editor_state.path).name, "data_2026-05-03_01-20-41.dat")
            self.assertIsNone(dummy._genai_revisions.pending)
            self.assertEqual(
                artifact_calls,
                [
                    (
                        ("session-1",),
                        {"model_text": "dvar int y;", "data_text": "y = 2;"},
                    )
                ],
            )

    def test_apply_genai_revisions_preserves_unsaved_unchanged_data(self):
        class DummyText:
            def __init__(self, content):
                self.content = content

            def get(self, *_args):
                return self.content

            def delete(self, *_args):
                self.content = ""

            def insert(self, *_args):
                self.content = _args[-1]

        cases = (
            ("new model", "", "edited data", "old data", True),
            ("new model", "", "old data", "old data", False),
            ("", "new data", "edited data", "old data", False),
            ("new model", "new data", "edited data", "old data", False),
        )
        for revised_model, revised_data, current_data, saved_data, dirty in cases:
            with self.subTest(revised_model=revised_model, revised_data=revised_data, current_data=current_data):
                with TemporaryDirectory() as tmpdir:
                    model_path = Path(tmpdir) / "model.mod"
                    data_path = Path(tmpdir) / "data.dat"
                    model_path.write_text("old model", encoding="utf-8")
                    data_path.write_text(saved_data, encoding="utf-8")
                    pending = pyopl_ide_bootstrap._PendingGenAIRevision(
                        revised_model,
                        revised_data,
                        "old model",
                        saved_data,
                        str(model_path),
                        str(data_path),
                        "2026-05-03_01-20-41",
                        True,
                        "session-1",
                    )
                    host = SimpleNamespace(
                        _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(
                            path=str(model_path), saved_text="old model"
                        ),
                        _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(path=str(data_path), saved_text=saved_data),
                        model_text=DummyText("edited model"),
                        data_text=DummyText(current_data),
                        editor_notebook=mock.Mock(),
                        model_frame=object(),
                        data_frame=object(),
                        highlight=mock.Mock(),
                        _record_output_session_artifacts=mock.Mock(),
                        _append_output=mock.Mock(),
                        _get_editor_text=lambda widget: widget.get(),
                    )
                    host._mark_editor_baselines_saved = lambda: OPLIDE._mark_editor_baselines_saved(host)

                    with mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir):
                        OPLIDE._apply_pending_revision(host, pending)

                    self.assertEqual(OPLIDE._has_unsaved_editor_changes(host), dirty)
                    self.assertEqual(Path(host._model_editor_state.path).read_text(), host.model_text.content)
                    self.assertEqual(Path(host._data_editor_state.path).read_text(), revised_data or saved_data)

    def test_pending_unsaved_genai_revisions_use_tmp_timestamp_basename(self):
        with TemporaryDirectory() as tmpdir:
            pending = pyopl_ide_bootstrap._PendingGenAIRevision(
                revised_model="revised model",
                revised_data="revised data",
                current_model="model",
                current_data="",
                model_path=str(Path(tmpdir) / "temp_model.mod"),
                data_path=str(Path(tmpdir) / "temp_data.dat"),
                safe_ts="2026-08-24_00-47-51",
                had_data_file=False,
                session_id=None,
            )
            model_target, data_target = pyopl_ide_bootstrap._write_genai_revision_files(
                pending,
                tmpdir,
                pending.model_path,
                pending.data_path,
                pending.safe_ts,
                pending.current_model,
            )

            self.assertEqual(Path(model_target).name, "tmp_2026-08-24_00-47-51.mod")
            self.assertEqual(Path(data_target).name, "tmp_2026-08-24_00-47-51.dat")
            second_model, second_data = pyopl_ide_bootstrap._write_genai_revision_files(
                pending, tmpdir, pending.model_path, pending.data_path, pending.safe_ts, pending.current_model
            )
            self.assertEqual(Path(second_model).name, "tmp_2026-08-24_00-47-51_1.mod")
            self.assertEqual(Path(second_data).name, "tmp_2026-08-24_00-47-51_1.dat")

    def test_pending_genai_revisions_preview_before_apply_and_survive_failure(self):
        pending = pyopl_ide_bootstrap._PendingGenAIRevision(
            revised_model="new model",
            revised_data="",
            current_model="old model",
            current_data="",
            model_path="original.mod",
            data_path="",
            safe_ts="2026-11-12_16-00-00",
            had_data_file=False,
            session_id="session-1",
        )
        dummy = SimpleNamespace(
            _genai_revisions=pyopl_ide_bootstrap._GenAIRevisionState(),
            genai_pending_var=mock.Mock(),
            genai_pending_frame=mock.Mock(),
            _show_pending_genai_diff_preview=mock.Mock(),
            _close_pending_genai_diff_preview=mock.Mock(),
        )

        OPLIDE._set_pending_genai_revisions(dummy, pending)

        self.assertIs(dummy._genai_revisions.pending, pending)
        dummy._show_pending_genai_diff_preview.assert_called_once_with()

        def fail_apply(_pending):
            raise OSError("disk failure")

        with self.assertRaisesRegex(OSError, "disk failure"):
            dummy._genai_revisions.apply(fail_apply)
        self.assertIs(dummy._genai_revisions.pending, pending)
        self.assertTrue(dummy._genai_revisions.apply(lambda _pending: None))
        OPLIDE._clear_pending_genai_revisions(dummy)
        self.assertIsNone(dummy._genai_revisions.pending)
        dummy._close_pending_genai_diff_preview.assert_called_once_with()

    def test_apply_revision_reports_write_failure_and_keeps_pending_for_retry(self):
        pending = pyopl_ide_bootstrap._PendingGenAIRevision(
            "new model", "", "old model", "", "original.mod", "", "2026-11-12_16-00-00", False, "session-1"
        )
        host = SimpleNamespace(
            _genai_revisions=pyopl_ide_bootstrap._GenAIRevisionState(pending),
            _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(path="original.mod"),
            _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
            model_text=mock.Mock(),
            status_var=mock.Mock(),
            _clear_pending_genai_revisions=mock.Mock(),
            _save_session=mock.Mock(),
        )
        with (
            TemporaryDirectory() as tmpdir,
            mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir),
            mock.patch.object(pyopl_ide_bootstrap, "_write_genai_revision_files", side_effect=OSError("disk full")),
            mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
            self.assertLogs(pyopl_ide_bootstrap.__name__, level="ERROR"),
        ):
            OPLIDE._apply_pending_genai_revisions(host)

        self.assertIs(host._genai_revisions.pending, pending)
        host._clear_pending_genai_revisions.assert_not_called()
        host._save_session.assert_not_called()
        host.status_var.set.assert_called_once_with("GenAI: failed to apply revisions")
        showerror.assert_called_once_with("GenAI Error", "Failed to apply revisions:\nOSError: disk full", parent=host)

        with mock.patch.object(OPLIDE, "_apply_pending_revision"):
            OPLIDE._apply_pending_genai_revisions(host)
        self.assertIsNone(host._genai_revisions.pending)
        host._clear_pending_genai_revisions.assert_called_once()

    def test_revision_file_failure_removes_partial_model_file(self):
        pending = pyopl_ide_bootstrap._PendingGenAIRevision(
            revised_model="new model",
            revised_data="new data",
            current_model="old model",
            current_data="old data",
            model_path="source.mod",
            data_path="source.dat",
            safe_ts="2026-11-12_16-00-00",
            had_data_file=True,
            session_id="session-1",
        )

        def failing_open(path, *args, **kwargs):
            if str(path).endswith(".dat"):
                raise OSError("cannot write data")
            return builtin_open(path, *args, **kwargs)

        with TemporaryDirectory() as tmpdir:
            with mock.patch.object(pyopl_ide_bootstrap, "open", failing_open, create=True):
                with self.assertRaisesRegex(OSError, "cannot write data"):
                    pyopl_ide_bootstrap._write_genai_revision_files(
                        pending, tmpdir, pending.model_path, pending.data_path, pending.safe_ts, pending.current_model
                    )
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

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
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(object(), DummyQueue(), "session-1"),
        )

        OPLIDE._cleanup_solver_ipc(dummy, cancel_queue_thread=False)

        self.assertEqual(events, ["close", "join_thread"])
        self.assertIsNone(dummy._solver_ipc.process)
        self.assertIsNone(dummy._solver_ipc.result_queue)
        self.assertIsNone(dummy._solver_ipc.session_id)

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
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(object(), DummyQueue()),
        )

        OPLIDE._cleanup_solver_ipc(dummy, cancel_queue_thread=True)

        self.assertEqual(events, ["close", "cancel_join_thread"])
        self.assertIsNone(dummy._solver_ipc.process)
        self.assertIsNone(dummy._solver_ipc.result_queue)

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
            _solver_ipc=pyopl_ide_bootstrap._SolverProcessState(),
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
        dummy = SimpleNamespace(_genai_attachments=pyopl_ide_bootstrap._GenAIAttachments())
        formatted = OPLIDE._format_prompt_for_output(
            dummy,
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

    def test_format_prompt_for_output_uses_friendly_attachment_labels(self):
        dummy = SimpleNamespace(
            _genai_attachments=pyopl_ide_bootstrap._GenAIAttachments(
                labels={"/tmp/rendered/page_1.png": "/Users/gwren/Downloads/diet-problem.pdf page 1"}
            )
        )

        formatted = OPLIDE._format_prompt_for_output(
            dummy,
            "Prompt",
            {"text": "Build this model.", "images": [{"path": "/tmp/rendered/page_1.png"}]},
        )

        self.assertIn("- /Users/gwren/Downloads/diet-problem.pdf page 1\n", formatted)
        self.assertNotIn("/tmp/rendered/page_1.png", formatted)

    def test_attachment_list_label_compacts_rendered_pdf_page(self):
        attachments = pyopl_ide_bootstrap._GenAIAttachments(
            labels={"/tmp/rendered/page_1.png": "/Users/gwren/Downloads/diet-problem.pdf page 1"}
        )

        self.assertEqual(
            attachments.list_label("/tmp/rendered/page_1.png"),
            "diet-problem.pdf page 1",
        )

    def test_attachment_list_label_keeps_image_basename(self):
        attachments = pyopl_ide_bootstrap._GenAIAttachments()

        self.assertEqual(
            attachments.list_label("/Users/gwren/Downloads/Screenshot 2026-06-30 at 21.37.52.png"),
            "Screenshot 2026-06-30 at 21.37.52.png",
        )

    def test_genai_add_images_converts_pdf_to_rendered_paths(self):
        attachments = pyopl_ide_bootstrap._GenAIAttachments(paths=["/tmp/existing.png"])
        attachments.render_pdf = mock.Mock(return_value=["/tmp/rendered/page_1.png", "/tmp/rendered/page_2.png"])
        dummy = SimpleNamespace(
            _genai_attachments=attachments,
            _refresh_genai_attachment_list=mock.Mock(),
        )

        with mock.patch.object(
            pyopl_ide_bootstrap.filedialog,
            "askopenfilenames",
            return_value=("/tmp/sketch.pdf", "/tmp/photo.png"),
        ):
            OPLIDE._genai_add_images(dummy)

        self.assertEqual(
            attachments.paths,
            ["/tmp/existing.png", "/tmp/rendered/page_1.png", "/tmp/rendered/page_2.png", "/tmp/photo.png"],
        )
        self.assertEqual(attachments.render_pdf.call_args.args[0], "/tmp/sketch.pdf")
        dummy._refresh_genai_attachment_list.assert_called_once()

    def test_genai_prompt_uses_attachment_snapshot_after_list_clears(self):
        attachments = pyopl_ide_bootstrap._GenAIAttachments(paths=["/tmp/sketch.png"])
        dummy = SimpleNamespace(
            genai_prompt_text=SimpleNamespace(get=lambda *_args: ""),
            genai_panel_mode_var=SimpleNamespace(get=lambda: "generate"),
            _genai_attachments=attachments,
        )

        prompt_input = OPLIDE._collect_genai_prompt_input(dummy)
        attachments.clear()

        self.assertEqual(prompt_input, {"text": "", "images": [{"path": "/tmp/sketch.png"}]})
        self.assertIsNone(OPLIDE._collect_genai_prompt_input(dummy))

    def test_clearing_attachments_removes_rendered_pdf_temp_dir(self):
        with TemporaryDirectory() as tmpdir:
            render_dir = Path(tmpdir) / "rendered"
            render_dir.mkdir()
            page = render_dir / "page_1.png"
            page.write_text("png", encoding="utf-8")
            attachments = pyopl_ide_bootstrap._GenAIAttachments(
                paths=[str(page)], labels={str(page): "paper.pdf page 1"}, pdf_temp_dir=str(render_dir)
            )

            attachments.clear()

            self.assertFalse(render_dir.exists())
            self.assertEqual(attachments.snapshot(), ())
            self.assertEqual(attachments.labels, {})
            self.assertIsNone(attachments.pdf_temp_dir)

    def test_render_genai_pdf_attachment_rejects_large_pdf(self):
        class FakeDocument:
            page_count = pyopl_ide_bootstrap.GENAI_MAX_PDF_PAGES + 1

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        fake_document = FakeDocument()
        fake_fitz = SimpleNamespace(open=mock.Mock(return_value=fake_document))
        attachments = pyopl_ide_bootstrap._GenAIAttachments()

        with (mock.patch.dict("sys.modules", {"fitz": fake_fitz}),):
            showwarning = mock.Mock()
            rendered = attachments.render_pdf("/tmp/too-large.pdf", mock.Mock(), showwarning)

        self.assertEqual(rendered, [])
        showwarning.assert_called_once()
        self.assertTrue(fake_document.closed)

    def test_render_genai_pdf_attachment_writes_page_images(self):
        class FakePixmap:
            def save(self, path):
                Path(path).write_text("png", encoding="utf-8")

        class FakePage:
            def get_pixmap(self, matrix, alpha):
                return FakePixmap()

        class FakeDocument:
            page_count = 2

            def load_page(self, page_index):
                return FakePage()

            def close(self):
                pass

        class FakeMatrix:
            def __init__(self, x_scale, y_scale):
                self.x_scale = x_scale
                self.y_scale = y_scale

        with TemporaryDirectory() as tmpdir:
            attachments = pyopl_ide_bootstrap._GenAIAttachments(pdf_temp_dir=tmpdir)
            fake_fitz = SimpleNamespace(open=mock.Mock(return_value=FakeDocument()), Matrix=FakeMatrix)

            with mock.patch.dict("sys.modules", {"fitz": fake_fitz}):
                rendered = attachments.render_pdf("/tmp/sketch.pdf", mock.Mock(), mock.Mock())

            self.assertEqual(len(rendered), 2)
            self.assertTrue(rendered[0].endswith("page_1.png"))
            self.assertTrue(rendered[1].endswith("page_2.png"))
            self.assertTrue(Path(rendered[0]).exists())
            self.assertTrue(Path(rendered[1]).exists())
            self.assertEqual(attachments.labels[rendered[0]], "/tmp/sketch.pdf page 1")
            self.assertEqual(attachments.labels[rendered[1]], "/tmp/sketch.pdf page 2")

    def test_failed_pdf_render_removes_partial_pages_and_labels(self):
        class FakePage:
            def __init__(self, index):
                self.index = index

            def get_pixmap(self, **_kwargs):
                return self

            def save(self, path):
                if self.index:
                    raise OSError("render failed")
                Path(path).write_text("page one", encoding="utf-8")

        class FakeDocument:
            page_count = 2

            def load_page(self, index):
                return FakePage(index)

            def close(self):
                pass

        with TemporaryDirectory() as tmpdir:
            attachments = pyopl_ide_bootstrap._GenAIAttachments(pdf_temp_dir=tmpdir)
            fake_fitz = SimpleNamespace(open=lambda _path: FakeDocument(), Matrix=lambda *_args: object())
            on_error = mock.Mock()

            with mock.patch.dict("sys.modules", {"fitz": fake_fitz}):
                rendered = attachments.render_pdf("/tmp/sketch.pdf", on_error, mock.Mock())

            self.assertEqual(rendered, [])
            self.assertEqual(attachments.labels, {})
            self.assertEqual(list(Path(tmpdir).iterdir()), [])
            self.assertIn("render failed", on_error.call_args.args[0])

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


class TestGenAIRequestRunner(unittest.TestCase):
    def test_generate_unsaved_named_files_save_discard_or_cancel(self):
        class DummyText:
            def __init__(self, content):
                self.content = content

            def get(self, *_args):
                return self.content

            def delete(self, *_args):
                self.content = ""

            def insert(self, *_args):
                self.content = _args[-1]

        for decision in (True, False, None):
            with self.subTest(decision=decision), TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.mod"
                data_path = Path(tmpdir) / "data.dat"
                model_path.write_text("saved model", encoding="utf-8")
                data_path.write_text("saved data", encoding="utf-8")
                host = SimpleNamespace(
                    model_text=DummyText("edited model\n"),
                    data_text=DummyText("edited data\n"),
                    _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(model_path), "saved model"),
                    _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(data_path), "saved data"),
                    _get_editor_text=lambda widget: widget.get(),
                    _replace_editor_buffer=lambda widget, text, **_kwargs: setattr(widget, "content", text),
                    editor_notebook=mock.Mock(),
                    model_frame=object(),
                    data_frame=object(),
                    _save_session=mock.Mock(),
                )
                host._has_unsaved_editor_changes = lambda: OPLIDE._has_unsaved_editor_changes(host)

                with mock.patch.object(pyopl_ide_bootstrap.messagebox, "askyesnocancel", return_value=decision) as prompt:
                    proceed = OPLIDE._prepare_genai_generation(host)

                prompt.assert_called_once()
                self.assertEqual(proceed, decision is not None)
                if decision is None:
                    self.assertEqual(host.model_text.content, "edited model\n")
                    self.assertEqual(host.data_text.content, "edited data\n")
                    expected_model, expected_data = "saved model", "saved data"
                elif decision is False:
                    self.assertEqual(host.model_text.content, "saved model")
                    self.assertEqual(host.data_text.content, "saved data")
                    expected_model, expected_data = "saved model", "saved data"
                else:
                    expected_model, expected_data = "edited model\n", "edited data\n"
                self.assertEqual(model_path.read_text(encoding="utf-8"), expected_model)
                self.assertEqual(data_path.read_text(encoding="utf-8"), expected_data)

    def test_generate_save_as_preserves_untitled_editor_or_cancels(self):
        with TemporaryDirectory() as tmpdir:
            for save_path in (str(Path(tmpdir) / "saved.mod"), ""):
                with self.subTest(save_path=save_path):
                    host = SimpleNamespace(
                        model_text=SimpleNamespace(get=lambda *_args: "untitled model\n"),
                        data_text=SimpleNamespace(get=lambda *_args: ""),
                        _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
                        _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
                        _get_editor_text=lambda widget: widget.get(),
                        editor_notebook=mock.Mock(),
                        model_frame=object(),
                        data_frame=object(),
                        _save_session=mock.Mock(),
                    )
                    host._has_unsaved_editor_changes = lambda: OPLIDE._has_unsaved_editor_changes(host)
                    with (
                        mock.patch.object(pyopl_ide_bootstrap.messagebox, "askyesnocancel", return_value=True),
                        mock.patch.object(pyopl_ide_bootstrap.filedialog, "asksaveasfilename", return_value=save_path),
                    ):
                        self.assertEqual(OPLIDE._prepare_genai_generation(host), bool(save_path))
                    self.assertEqual(host._model_editor_state.path, save_path or None)
                    if save_path:
                        self.assertEqual(Path(save_path).read_text(encoding="utf-8"), "untitled model\n")
                        self.assertFalse(OPLIDE._has_unsaved_editor_changes(host))
                    else:
                        self.assertTrue(OPLIDE._has_unsaved_editor_changes(host))

    def test_generate_does_not_proceed_if_saving_changes_fails(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.mod"
            model_path.write_text("saved model", encoding="utf-8")
            host = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "edited model"),
                data_text=SimpleNamespace(get=lambda *_args: ""),
                _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(model_path), "saved model"),
                _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
                _get_editor_text=lambda widget: widget.get(),
                editor_notebook=mock.Mock(),
                model_frame=object(),
                data_frame=object(),
                _save_session=mock.Mock(),
            )
            host._has_unsaved_editor_changes = lambda: OPLIDE._has_unsaved_editor_changes(host)
            with (
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "askyesnocancel", return_value=True),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
                mock.patch.object(pyopl_ide_bootstrap, "open", side_effect=OSError("disk full"), create=True),
                self.assertLogs(pyopl_ide_bootstrap.__name__, level="ERROR"),
            ):
                self.assertFalse(OPLIDE._prepare_genai_generation(host))

            self.assertEqual(model_path.read_text(encoding="utf-8"), "saved model")
            self.assertTrue(OPLIDE._has_unsaved_editor_changes(host))
            host._save_session.assert_not_called()
            showerror.assert_called_once_with("Generate", "Could not save model changes:\ndisk full", parent=host)

    def test_ask_unsaved_changes_keep_discard_or_cancel(self):
        class DummyText:
            def __init__(self, content):
                self.content = content

            def get(self, *_args):
                return self.content

            def delete(self, *_args):
                self.content = ""

            def insert(self, *_args):
                self.content = _args[-1]

        for decision in (True, False, None):
            with self.subTest(decision=decision), TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.mod"
                data_path = Path(tmpdir) / "data.dat"
                model_path.write_text("saved model", encoding="utf-8")
                data_path.write_text("saved data", encoding="utf-8")
                host = SimpleNamespace(
                    model_text=DummyText("edited model"),
                    data_text=DummyText("edited data"),
                    _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(model_path), "saved model"),
                    _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(data_path), "saved data"),
                    _get_editor_text=lambda widget: widget.get(),
                    _replace_editor_buffer=lambda widget, text, **_kwargs: setattr(widget, "content", text),
                )
                host._has_unsaved_editor_changes = lambda: OPLIDE._has_unsaved_editor_changes(host)
                host._ensure_model_data_saved = lambda: OPLIDE._ensure_model_data_saved(host)

                with mock.patch.object(pyopl_ide_bootstrap.messagebox, "askyesnocancel", return_value=decision) as prompt:
                    result = OPLIDE._prepare_genai_feedback_files(host)

                prompt.assert_called_once()
                expected = "edited" if decision is True else "saved"
                if decision is None:
                    self.assertEqual(result, (None, None))
                    self.assertEqual(model_path.read_text(encoding="utf-8"), "saved model")
                    self.assertEqual(data_path.read_text(encoding="utf-8"), "saved data")
                    self.assertEqual(host.model_text.content, "edited model")
                    self.assertEqual(host.data_text.content, "edited data")
                    self.assertTrue(OPLIDE._has_unsaved_editor_changes(host))
                else:
                    self.assertEqual(result, (str(model_path), str(data_path)))
                    self.assertEqual(model_path.read_text(encoding="utf-8"), f"{expected} model")
                    self.assertEqual(data_path.read_text(encoding="utf-8"), f"{expected} data")
                    self.assertEqual(host.model_text.content, f"{expected} model")
                    self.assertEqual(host.data_text.content, f"{expected} data")
                    self.assertFalse(OPLIDE._has_unsaved_editor_changes(host))

    def test_ask_discard_aborts_without_changes_if_saved_file_cannot_be_read(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "missing.mod"
            data_path = Path(tmpdir) / "data.dat"
            data_path.write_text("saved data", encoding="utf-8")
            host = SimpleNamespace(
                model_text=mock.Mock(),
                data_text=mock.Mock(),
                _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(model_path), "saved model"),
                _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(data_path), "saved data"),
                _get_editor_text=lambda widget: "edited model" if widget is host.model_text else "edited data",
                _replace_editor_buffer=mock.Mock(),
            )
            host._has_unsaved_editor_changes = lambda: OPLIDE._has_unsaved_editor_changes(host)
            host._ensure_model_data_saved = mock.Mock()
            with (
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "askyesnocancel", return_value=False),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
            ):
                self.assertEqual(OPLIDE._prepare_genai_feedback_files(host), (None, None))
            host._replace_editor_buffer.assert_not_called()
            host._ensure_model_data_saved.assert_not_called()
            showerror.assert_called_once()
            self.assertFalse(model_path.exists())
            self.assertEqual(data_path.read_text(encoding="utf-8"), "saved data")

    def test_ask_keep_preserves_trailing_newlines_and_untitled_stays_unsaved(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.mod"
            model_path.write_text("saved", encoding="utf-8")
            host = SimpleNamespace(
                model_text=SimpleNamespace(get=lambda *_args: "edited\n\n"),
                data_text=SimpleNamespace(get=lambda *_args: "untitled data\n"),
                _model_editor_state=pyopl_ide_bootstrap._EditorBufferState(str(model_path), "saved"),
                _data_editor_state=pyopl_ide_bootstrap._EditorBufferState(),
                _get_editor_text=lambda widget: widget.get(),
            )
            host._has_unsaved_editor_changes = lambda: OPLIDE._has_unsaved_editor_changes(host)
            host._ensure_model_data_saved = lambda: OPLIDE._ensure_model_data_saved(host)
            with (
                mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir),
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "askyesnocancel", return_value=True),
            ):
                model, data = OPLIDE._prepare_genai_feedback_files(host)

            self.assertEqual(model, str(model_path))
            self.assertEqual(Path(model).read_text(encoding="utf-8"), "edited\n\n")
            self.assertEqual(Path(data).read_text(encoding="utf-8"), "untitled data\n")
            self.assertIsNone(host._data_editor_state.path)
            self.assertTrue(OPLIDE._has_unsaved_editor_changes(host))

    def _request_host(self, generator, operation, scheduled):
        return SimpleNamespace(
            genai_provider="selected-provider",
            genai_model="selected-model",
            verbose_llm_var=SimpleNamespace(get=lambda: False),
            genai_method_var=SimpleNamespace(get=lambda: "default"),
            _label_for_method=lambda value: value,
            _import_selected_genai_module=lambda: generator,
            _start_foreground_operation=lambda **kwargs: operation,
            _finish_foreground_operation=mock.Mock(),
            _append_output=mock.Mock(),
            _format_prompt_for_output=lambda label, prompt: label,
            _clear_pending_genai_revisions=mock.Mock(),
            _session_state=SimpleNamespace(timestamps={operation.session_id: "2026-01-01 12:00:00"}),
            after=lambda delay, callback, *args: scheduled.append((callback, args)),
            _clear_genai_composer=mock.Mock(),
            _save_session=mock.Mock(),
            status_var=mock.Mock(),
            model_text=mock.Mock(),
            data_text=mock.Mock(),
            _prepare_genai_feedback_files=mock.Mock(return_value=("model.mod", None)),
            _prepare_genai_generation=mock.Mock(return_value=True),
        )

    def test_import_failure_is_reported_for_generate_and_ask(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("genai", "GenAI", "session-1")
        for request in (OPLIDE._run_genai_generate, OPLIDE._run_genai_feedback):
            with self.subTest(request=request.__name__):
                host = self._request_host(None, operation, [])
                host.genai_method_var = mock.Mock()
                host.genai_method_var.get.return_value = "pyopl_standard"
                host._import_selected_genai_module = lambda: OPLIDE._import_selected_genai_module(host)
                host._start_foreground_operation = mock.Mock()
                host._data_editor_state = SimpleNamespace(path=None)
                host._prepare_genai_feedback_files.return_value = ("model.mod", None)
                error = ModuleNotFoundError("No module named 'optional_dependency'", name="optional_dependency")
                with (
                    mock.patch("importlib.import_module", side_effect=error),
                    mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
                    self.assertLogs(pyopl_ide_bootstrap.__name__, level="ERROR"),
                ):
                    request(host, "prompt")

                showerror.assert_called_once_with("GenAI Error", "ModuleNotFoundError: No module named 'optional_dependency'")
                host.status_var.set.assert_called_once_with("GenAI: error")
                host._start_foreground_operation.assert_not_called()
                host._clear_pending_genai_revisions.assert_not_called()

    def test_cancelled_ask_does_not_start_request(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("genai-feedback", "Ask", "session-1")
        host = self._request_host(mock.Mock(), operation, [])
        host._data_editor_state = pyopl_ide_bootstrap._EditorBufferState()
        host._prepare_genai_feedback_files.return_value = (None, None)
        host._import_selected_genai_module = mock.Mock()
        host._start_foreground_operation = mock.Mock()

        OPLIDE._run_genai_feedback(host, "question")

        host._import_selected_genai_module.assert_not_called()
        host._start_foreground_operation.assert_not_called()
        host._append_output.assert_not_called()

    def test_cancelled_generate_does_not_start_request(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("genai-generate", "Generate", "session-1")
        host = self._request_host(mock.Mock(), operation, [])
        host._prepare_genai_generation.return_value = False
        host._start_foreground_operation = mock.Mock()

        OPLIDE._run_genai_generate(host, "prompt")

        host._start_foreground_operation.assert_not_called()
        host._append_output.assert_not_called()

    def test_generate_snapshots_selection_and_discards_cancelled_completion(self):
        with TemporaryDirectory() as tmpdir:
            operation = pyopl_ide_bootstrap._ForegroundOperation("genai-generate", "Generate", "session-1")
            scheduled = []
            generator = mock.Mock()
            generator.__name__ = "pyopl.test_genai_generate"

            def generate(prompt, model_path, data_path, **kwargs):
                Path(model_path).write_text("generated model")
                Path(data_path).write_text("generated data")
                kwargs["progress"]("working")
                return "assessment"

            generator.generative_solve.side_effect = generate
            host = self._request_host(generator, operation, scheduled)
            thread = mock.Mock()
            with (
                mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir),
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", return_value=thread) as thread_cls,
            ):
                OPLIDE._run_genai_generate(host, "prompt")
                host.genai_provider, host.genai_model = "changed-provider", "changed-model"
                thread_cls.call_args.kwargs["target"]()

            self.assertEqual(generator.generative_solve.call_args.kwargs["llm_provider"], "selected-provider")
            self.assertEqual(generator.generative_solve.call_args.kwargs["model_name"], "selected-model")
            operation.cancel_requested = True
            for callback, args in scheduled:
                callback(*args)
            host._append_output.assert_called_once_with("Prompt", "session-1")
            host._finish_foreground_operation.assert_not_called()
            host.model_text.delete.assert_not_called()

    def test_feedback_snapshots_selection_and_previews_revision(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "original.mod"
            model_path.write_text("original model")
            operation = pyopl_ide_bootstrap._ForegroundOperation("genai-feedback", "Ask", "session-1")
            scheduled = []
            generator = mock.Mock()
            generator.__name__ = "pyopl.test_genai_feedback"
            generator.generative_feedback.return_value = {"feedback": "suggestion", "revised_model": "new model"}
            host = self._request_host(generator, operation, scheduled)
            host._data_editor_state = SimpleNamespace(path=None)
            host._prepare_genai_feedback_files.return_value = (str(model_path), None)
            host._set_pending_genai_revisions = mock.Mock()
            host.model_text.get.return_value = "original model"
            host.data_text.get.return_value = "original data"
            thread = mock.Mock()
            with (
                mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir),
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", return_value=thread) as thread_cls,
            ):
                OPLIDE._run_genai_feedback(host, "question")
                host.genai_provider, host.genai_model = "changed-provider", "changed-model"
                thread_cls.call_args.kwargs["target"]()

            self.assertEqual(generator.generative_feedback.call_args.kwargs["llm_provider"], "selected-provider")
            self.assertEqual(generator.generative_feedback.call_args.kwargs["model_name"], "selected-model")
            for callback, args in scheduled:
                callback(*args)
            pending = host._set_pending_genai_revisions.call_args.args[0]
            self.assertEqual(pending.revised_model, "new model")
            self.assertEqual(pending.current_model, "original model")
            self.assertEqual(pending.session_id, "session-1")
            host._append_output.assert_any_call("\nFeedback:\nsuggestion\n", "session-1")
            host.model_text.delete.assert_not_called()

    def test_feedback_provider_failure_is_reported_in_request_session(self):
        with TemporaryDirectory() as tmpdir:
            operation = pyopl_ide_bootstrap._ForegroundOperation("genai-feedback", "Ask", "session-1")
            scheduled = []
            generator = mock.Mock()
            generator.__name__ = "pyopl.test_genai_feedback_error"
            generator.generative_feedback.side_effect = ValueError("provider failed")
            host = self._request_host(generator, operation, scheduled)
            host._data_editor_state = SimpleNamespace(path=None)
            host._prepare_genai_feedback_files.return_value = (str(Path(tmpdir) / "model.mod"), None)
            host._set_pending_genai_revisions = mock.Mock()
            thread = mock.Mock()
            with (
                mock.patch.object(pyopl_ide_bootstrap.os, "getcwd", return_value=tmpdir),
                mock.patch.object(pyopl_ide_bootstrap.threading, "Thread", return_value=thread) as thread_cls,
                mock.patch.object(pyopl_ide_bootstrap.messagebox, "showerror") as showerror,
            ):
                OPLIDE._run_genai_feedback(host, "question")
                thread_cls.call_args.kwargs["target"]()
                for callback, args in scheduled:
                    callback(*args)

            showerror.assert_called_once()
            host._append_output.assert_any_call("\nGenAI Error: provider failed\n", "session-1")
            host.status_var.set.assert_called_with("GenAI: error")
            host._set_pending_genai_revisions.assert_not_called()

    def test_generate_and_feedback_use_selected_provider_and_session_progress(self):
        operation = pyopl_ide_bootstrap._ForegroundOperation("genai", "GenAI", "session-1")
        scheduled = []
        append_output = mock.Mock()
        gen_module = mock.Mock()
        gen_module.__name__ = "pyopl.test_genai_runner"
        gen_module.generative_solve.side_effect = lambda *args, **kwargs: kwargs["progress"]("generating")
        gen_module.generative_feedback.side_effect = lambda *args, **kwargs: kwargs["progress"]("asking")
        runner = pyopl_ide_bootstrap._GenAIRequestRunner(
            operation,
            "selected-provider",
            "selected-model",
            gen_module,
            False,
            lambda delay, callback, *args: scheduled.append((callback, args)),
            append_output,
        )

        runner.generate("prompt", "model.mod", "data.dat")
        runner.feedback("prompt", "model.mod", "data.dat")
        self.assertEqual(gen_module.generative_solve.call_args.kwargs["llm_provider"], "selected-provider")
        self.assertEqual(gen_module.generative_solve.call_args.kwargs["model_name"], "selected-model")
        self.assertEqual(gen_module.generative_feedback.call_args.kwargs["llm_provider"], "selected-provider")
        self.assertEqual(gen_module.generative_feedback.call_args.kwargs["model_name"], "selected-model")

        for callback, args in scheduled:
            callback(*args)
        append_output.assert_has_calls([mock.call("generating\n", "session-1"), mock.call("asking\n", "session-1")])

        operation.cancel_requested = True
        gen_module.generative_feedback.call_args.kwargs["progress"]("late")
        for callback, args in scheduled:
            callback(*args)
        self.assertEqual(append_output.call_count, 2)

    def test_verbose_logger_restored_after_provider_failure(self):
        import logging

        logger = logging.getLogger("pyopl.test_genai_runner_failure")
        previous_level, previous_handlers = logger.level, list(logger.handlers)
        gen_module = mock.Mock()
        gen_module.__name__ = logger.name
        gen_module.generative_feedback.side_effect = ValueError("provider failed")
        runner = pyopl_ide_bootstrap._GenAIRequestRunner(
            pyopl_ide_bootstrap._ForegroundOperation("genai", "GenAI", "session-1"),
            "provider",
            "model",
            gen_module,
            True,
            mock.Mock(),
            mock.Mock(),
        )

        with self.assertRaisesRegex(ValueError, "provider failed"):
            runner.feedback("prompt", "model.mod", "data.dat")
        self.assertEqual(logger.level, previous_level)
        self.assertEqual(logger.handlers, previous_handlers)


if __name__ == "__main__":
    unittest.main()

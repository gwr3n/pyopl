import json
import queue
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from threading import Event
from unittest.mock import patch

from pyopl.batch_solve import (
    _batch_solve_worker,
    _format_duration,
    _json_safe,
    _load_partial_records,
    _markdown_value,
    batch_solve,
    batch_solve_with_progress,
)


class TestBatchSolve(unittest.TestCase):
    def test_rejects_missing_archive_and_unsupported_solver(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_archive = Path(temporary_directory) / "missing.zip"
            with self.assertRaisesRegex(FileNotFoundError, "Batch archive not found"):
                batch_solve(missing_archive)

            archive = Path(temporary_directory) / "batch.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("data.dat", "data")

            with self.assertRaisesRegex(ValueError, "Unsupported solver"):
                batch_solve(archive, solver="unknown")
            with self.assertRaisesRegex(ValueError, "Unsupported solver"):
                batch_solve(archive, solver=None)

    def test_rejects_archive_without_a_model_data_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "empty.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("first.mod", "model")
                batch_archive.writestr("second.mod", "model")
                batch_archive.writestr("data.dat", "data")

            with self.assertRaisesRegex(ValueError, "exactly one .mod"):
                batch_solve(archive)

    def test_rejects_invalid_solver_configuration(self):
        for contents, expected_message in (("not json", "Invalid solver configuration"), ("[]", "must contain a JSON object")):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as temporary_directory:
                archive = Path(temporary_directory) / "configured.zip"
                with zipfile.ZipFile(archive, "w") as batch_archive:
                    batch_archive.writestr("model.mod", "model")
                    batch_archive.writestr("data.dat", "data")
                    batch_archive.writestr("highs.json", contents)

                with self.assertRaisesRegex(ValueError, expected_message):
                    batch_solve(archive)

    def test_load_partial_records_ignores_invalid_reports_and_other_solvers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "batch.json"
            self.assertEqual(_load_partial_records(report_path, "highs"), [])

            report_path.write_text("invalid", encoding="utf-8")
            self.assertEqual(_load_partial_records(report_path, "highs"), [])

            report_path.write_text(json.dumps({"instances": "invalid"}), encoding="utf-8")
            self.assertEqual(_load_partial_records(report_path, "highs"), [])

            report_path.write_text(
                json.dumps({"instances": [None, {"solver": "gurobi"}, {"solver": "highs", "data": "a.dat"}]}),
                encoding="utf-8",
            )
            self.assertEqual(_load_partial_records(report_path, "highs"), [{"solver": "highs", "data": "a.dat"}])

    def test_serializes_solver_values_and_escapes_markdown(self):
        class Scalar:
            def item(self):
                return 7

        class Array:
            def tolist(self):
                return (1, 2)

        class Unsupported:
            def __str__(self):
                return "unsupported"

        self.assertEqual(
            _json_safe({1: (Scalar(), Array(), Unsupported())}),
            {"1": [7, [1, 2], "unsupported"]},
        )
        self.assertEqual(_markdown_value(None), "")
        self.assertEqual(_markdown_value("a|b\nc"), "a\\|b c")
        self.assertEqual(_markdown_value({"b": 2, "a": 1}), '{"a":1,"b":2}')

    def test_formats_durations(self):
        self.assertEqual(_format_duration(-1), "less than 1 minute")
        self.assertEqual(_format_duration(60), "1 minute")
        self.assertEqual(_format_duration(2 * 60 * 60 + 3 * 60), "2 hours 3 minutes")
        self.assertEqual(_format_duration(24 * 60 * 60), "1 day")
        self.assertEqual(_format_duration(2 * 24 * 60 * 60 + 60 * 60), "2 days 1 hour")

    def test_worker_always_emits_finished_event(self):
        class Events:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        events = Events()
        with patch("pyopl.batch_solve.batch_solve", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                _batch_solve_worker("batch.zip", "highs", events)

        self.assertEqual(events.items, [{"event": "finished"}])

    def test_progress_window_handles_events_and_returns_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "batch.zip"
            report = {"instances": [{"status": "OPTIMAL"}]}
            archive.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
            events = queue.Queue()
            for event in (
                {"event": "started", "total": 2, "completed": 0},
                {
                    "event": "instance_started",
                    "model": "model.mod",
                    "data": "a.dat",
                    "total": 2,
                    "completed": 0,
                    "remaining": 2,
                    "average_solution_time": 0.0,
                },
                {
                    "event": "instance_started",
                    "model": "model.mod",
                    "data": "b.dat",
                    "total": 2,
                    "completed": 1,
                    "remaining": 1,
                    "average_solution_time": 1.5,
                },
                {
                    "event": "progress",
                    "model": "model.mod",
                    "data": "b.dat",
                    "total": 2,
                    "completed": 2,
                    "remaining": 0,
                    "average_solution_time": 1.25,
                },
                {
                    "event": "stopped",
                    "total": 2,
                    "completed": 2,
                    "remaining": 0,
                    "average_solution_time": 1.25,
                },
                {"event": "finished"},
            ):
                events.put(event)

            class Widget:
                def __init__(self, *args, **kwargs):
                    self.options = kwargs

                def grid(self, *args, **kwargs):
                    pass

                def configure(self, **kwargs):
                    self.options.update(kwargs)

            class Root:
                def __init__(self):
                    self.callback = None
                    self.destroyed = False

                def title(self, title):
                    pass

                def resizable(self, width, height):
                    pass

                def update_idletasks(self):
                    pass

                def winfo_width(self):
                    return 400

                def winfo_height(self):
                    return 200

                def minsize(self, width, height):
                    pass

                def after(self, delay, callback):
                    self.callback = callback

                def mainloop(self):
                    self.callback()

                def destroy(self):
                    self.destroyed = True

            class Process:
                exitcode = 0

                def __init__(self, *args, **kwargs):
                    self.running = False

                def start(self):
                    self.running = False

                def is_alive(self):
                    return self.running

                def terminate(self):
                    self.running = False

                def join(self):
                    pass

            root = Root()
            widgets = []

            def make_widget(*args, **kwargs):
                widget = Widget(*args, **kwargs)
                widgets.append(widget)
                return widget

            with (
                patch("tkinter.Tk", return_value=root),
                patch("tkinter.ttk.Frame", side_effect=make_widget),
                patch("tkinter.ttk.Label", side_effect=make_widget),
                patch("tkinter.ttk.Progressbar", side_effect=make_widget),
                patch("tkinter.ttk.Button", side_effect=make_widget),
                patch("pyopl.batch_solve.multiprocessing.Queue", return_value=events),
                patch("pyopl.batch_solve.multiprocessing.Process", Process),
            ):
                result = batch_solve_with_progress(archive)

            self.assertEqual(result, report)
            self.assertEqual(widgets[-1].options["text"], "Close")
            widgets[-1].options["command"]()
            self.assertTrue(root.destroyed)

    def test_progress_window_reports_process_and_report_failures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "batch.zip"

            class Root:
                def title(self, title):
                    pass

                def resizable(self, width, height):
                    pass

                def update_idletasks(self):
                    pass

                def winfo_width(self):
                    return 1

                def winfo_height(self):
                    return 1

                def minsize(self, width, height):
                    pass

                def after(self, delay, callback):
                    pass

                def mainloop(self):
                    pass

            class Widget:
                def __init__(self, *args, **kwargs):
                    pass

                def grid(self, *args, **kwargs):
                    pass

                def configure(self, **kwargs):
                    pass

            class FailedProcess:
                exitcode = 1

                def __init__(self, *args, **kwargs):
                    pass

                def start(self):
                    pass

                def is_alive(self):
                    return False

                def join(self):
                    pass

            patches = (
                patch("tkinter.Tk", return_value=Root()),
                patch("tkinter.ttk.Frame", Widget),
                patch("tkinter.ttk.Label", Widget),
                patch("tkinter.ttk.Progressbar", Widget),
                patch("tkinter.ttk.Button", Widget),
                patch("pyopl.batch_solve.multiprocessing.Queue", return_value=queue.Queue()),
                patch("pyopl.batch_solve.multiprocessing.Process", FailedProcess),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    batch_solve_with_progress(archive)

            archive.with_suffix(".json").write_text(json.dumps({"partial": True}), encoding="utf-8")
            patches = (
                patch("tkinter.Tk", return_value=Root()),
                patch("tkinter.ttk.Frame", Widget),
                patch("tkinter.ttk.Label", Widget),
                patch("tkinter.ttk.Progressbar", Widget),
                patch("tkinter.ttk.Button", Widget),
                patch("pyopl.batch_solve.multiprocessing.Queue", return_value=queue.Queue()),
                patch("pyopl.batch_solve.multiprocessing.Process", FailedProcess),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                self.assertEqual(batch_solve_with_progress(archive), {"partial": True})

    def test_rejects_non_zip_extension_without_overwriting_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "batch.json"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("data.dat", "data")
            original_contents = archive.read_bytes()

            with self.assertRaisesRegex(ValueError, r"\.zip extension"):
                batch_solve(archive)

            self.assertEqual(archive.read_bytes(), original_contents)
            self.assertTrue(zipfile.is_zipfile(archive))

    def test_accepts_single_top_level_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "wrapped.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("knapsack/model.mod", "model")
                batch_archive.writestr("knapsack/data.dat", "data")
                batch_archive.writestr("knapsack/highs.json", "{}")

            with patch("pyopl.batch_solve.solve", return_value={"status": "OPTIMAL"}):
                report = batch_solve(archive)

            self.assertEqual(report["model"], "knapsack/model.mod")
            self.assertEqual(report["instances"][0]["data"], "knapsack/data.dat")

    def test_recursively_solves_every_model_folder_and_continues_after_errors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "nested.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("first/model.mod", "invalid model")
                batch_archive.writestr("first/data.dat", "data")
                batch_archive.writestr("collection/second/model.mod", "valid model")
                batch_archive.writestr("collection/second/a.dat", "data")
                batch_archive.writestr("collection/second/b.dat", "data")
                batch_archive.writestr("collection/ignored.dat", "no model in this folder")

            def fake_solve(model, data, solver, solver_settings):
                if model.endswith("first/model.mod"):
                    raise ValueError("model failed to compile")
                return {"status": "OPTIMAL"}

            with patch("pyopl.batch_solve.solve", side_effect=fake_solve) as solve_mock:
                report = batch_solve(archive)

            self.assertEqual(solve_mock.call_count, 3)
            self.assertEqual(
                [(record["model"], record["data"], record["status"]) for record in report["instances"]],
                [
                    ("collection/second/model.mod", "collection/second/a.dat", "OPTIMAL"),
                    ("collection/second/model.mod", "collection/second/b.dat", "OPTIMAL"),
                    ("first/model.mod", "first/data.dat", "ERROR"),
                ],
            )
            self.assertEqual(report["models"], ["collection/second/model.mod", "first/model.mod"])
            self.assertIn("model failed to compile", report["instances"][2]["message"])
            self.assertTrue(archive.with_suffix(".json").exists())
            self.assertIn("| model | data | solver |", archive.with_suffix(".md").read_text(encoding="utf-8"))

    def test_ignores_macos_archive_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "metadata.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("data.dat", "data")
                batch_archive.writestr("__MACOSX/batch/._model.mod", "metadata")
                batch_archive.writestr("__MACOSX/batch/._data.dat", "metadata")
                batch_archive.writestr("__MACOSX/batch/._highs.json", "metadata")

            with patch("pyopl.batch_solve.solve", return_value={"status": "OPTIMAL"}):
                report = batch_solve(archive)

            self.assertEqual(len(report["instances"]), 1)

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("data.dat", "data")
                batch_archive.writestr("../outside.txt", "unsafe")

            with self.assertRaises(ValueError):
                batch_solve(archive)

    def test_defaults_to_highs_and_formats_stats(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "knapsack.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("a.dat", "valid")
                batch_archive.writestr("b.dat", "invalid")
                batch_archive.writestr("highs.json", json.dumps({"time_limit": 2}))
                batch_archive.writestr("gurobi.json", json.dumps({"TimeLimit": 2}))

            def fake_solve(model, data, solver, solver_settings):
                if data.endswith("b.dat"):
                    raise ValueError("invalid data")
                return {
                    "status": "OPTIMAL",
                    "objective_value": 4,
                    "stats": {"runtime": 0.1, "message": "solver message", "status": 0},
                }

            with patch("pyopl.batch_solve.solve", side_effect=fake_solve) as solve_mock:
                report = batch_solve(archive)

            self.assertEqual(len(report["instances"]), 2)
            self.assertEqual(solve_mock.call_count, 2)
            self.assertTrue((root / "knapsack.json").exists())
            self.assertTrue((root / "knapsack.md").exists())
            payload = json.loads((root / "knapsack.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["instances"][1]["status"], "ERROR")
            markdown = (root / "knapsack.md").read_text(encoding="utf-8")
            self.assertIn("| model | data | solver | status |", markdown)
            self.assertIn("| model | data | solver | status | objective_value | message | runtime |", markdown)
            self.assertIn("| model.mod | a.dat | highs | OPTIMAL | 4 | solver message | 0.1 |", markdown)
            self.assertEqual(markdown.splitlines()[2].count("message"), 1)
            self.assertNotIn('{"runtime":0.1}', markdown)

    def test_selects_gurobi_solver_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "gurobi.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("data.dat", "data")
                batch_archive.writestr("highs.json", "invalid JSON that must be ignored")
                batch_archive.writestr("gurobi.json", json.dumps({"TimeLimit": 2}))

            with patch("pyopl.batch_solve.solve", return_value={"status": "OPTIMAL"}) as solve_mock:
                report = batch_solve(archive, solver="gurobi")

            self.assertEqual(report["instances"][0]["solver"], "gurobi")
            markdown = (archive.with_suffix(".md")).read_text(encoding="utf-8")
            self.assertNotIn("| message |", markdown.splitlines()[2])
            solve_mock.assert_called_once_with(
                unittest.mock.ANY,
                unittest.mock.ANY,
                solver="gurobi",
                solver_settings={"TimeLimit": 2},
            )

    def test_writes_partial_results_and_resumes_from_existing_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "resume.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("a.dat", "data")
                batch_archive.writestr("b.dat", "data")

            solve_results = iter([{"status": "OPTIMAL"}, KeyboardInterrupt(), {"status": "OPTIMAL"}])
            with patch("pyopl.batch_solve.solve", side_effect=solve_results) as solve_mock:
                with self.assertRaises(KeyboardInterrupt):
                    batch_solve(archive)

                partial = json.loads(archive.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual([record["data"] for record in partial["instances"]], ["a.dat"])

                report = batch_solve(archive)

            self.assertEqual([record["data"] for record in report["instances"]], ["a.dat", "b.dat"])
            self.assertEqual(solve_mock.call_count, 3)

    def test_reports_progress_metrics_and_stops_before_next_instance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "progress.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("a.dat", "data")
                batch_archive.writestr("b.dat", "data")

            stop_event = Event()
            events = []

            def fake_solve(model, data, solver, solver_settings):
                stop_event.set()
                time.sleep(0.001)
                return {"status": "OPTIMAL"}

            with patch("pyopl.batch_solve.solve", side_effect=fake_solve) as solve_mock:
                report = batch_solve(archive, progress_callback=events.append, stop_event=stop_event)

            self.assertEqual(solve_mock.call_count, 1)
            self.assertEqual(len(report["instances"]), 1)
            self.assertEqual(events[0]["event"], "started")
            self.assertEqual(events[0]["total"], 2)
            self.assertEqual(events[-1]["event"], "stopped")
            self.assertEqual(events[-1]["completed"], 1)
            self.assertEqual(events[-1]["remaining"], 1)
            self.assertGreater(events[-1]["average_solution_time"], 0)


if __name__ == "__main__":
    unittest.main()

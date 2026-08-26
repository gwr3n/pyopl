import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pyopl.batch_solve import batch_solve


class TestBatchSolve(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

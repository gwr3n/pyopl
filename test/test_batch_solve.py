import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pyopl.batch_solve import batch_solve


class TestBatchSolve(unittest.TestCase):
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
                return {"status": "OPTIMAL", "objective_value": 4, "stats": {"runtime": 0.1}}

            with patch("pyopl.batch_solve.solve", side_effect=fake_solve) as solve_mock:
                report = batch_solve(archive)

            self.assertEqual(len(report["instances"]), 2)
            self.assertEqual(solve_mock.call_count, 2)
            self.assertTrue((root / "knapsack.json").exists())
            self.assertTrue((root / "knapsack.md").exists())
            payload = json.loads((root / "knapsack.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["instances"][1]["status"], "ERROR")
            markdown = (root / "knapsack.md").read_text(encoding="utf-8")
            self.assertIn("| data | solver | status |", markdown)
            self.assertIn("| data | solver | status | objective_value | message | runtime |", markdown)
            self.assertIn("| a.dat | highs | OPTIMAL | 4 |  | 0.1 |", markdown)
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
            solve_mock.assert_called_once_with(
                unittest.mock.ANY,
                unittest.mock.ANY,
                solver="gurobi",
                solver_settings={"TimeLimit": 2},
            )


if __name__ == "__main__":
    unittest.main()

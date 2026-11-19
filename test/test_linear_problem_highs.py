import tempfile
import threading
import unittest
from pathlib import Path

import highspy

from pyopl.highs_codegen_csc import HighsCSCCodeGenerator
from pyopl.import_model import convert_file
from pyopl.linear_problem_highs import build_highs_model, export_linear_problem
from pyopl.pyopl_core import linear_problem_from_opl, parse_model


class TestLinearProblemHighs(unittest.TestCase):
    def test_build_highs_model_and_export_lp_mps(self):
        ast = parse_model("dvar float+ x; minimize 2 * x + 3; subject to { x >= 1; x <= 4; }")
        problem = HighsCSCCodeGenerator(ast).build_problem()

        highs = build_highs_model(problem)
        self.assertEqual(highs.getNumCol(), 1)
        self.assertEqual(highs.getNumRow(), 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            lp_path = export_linear_problem(problem, Path(tmp_dir) / "model.lp")
            mps_path = export_linear_problem(problem, Path(tmp_dir) / "model.mps")

            self.assertGreater(lp_path.stat().st_size, 0)
            self.assertGreater(mps_path.stat().st_size, 0)
            lp_text = lp_path.read_text(encoding="utf-8")
            self.assertIn("obj:", lp_text)
            self.assertIn("x", lp_text)

    def test_import_lp_and_mps_preserves_mixed_integer_ranged_model(self):
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.addCols(3, [2.0, -1.0, 3.0], [0.0, 0.0, -highspy.kHighsInf], [1.0, 10.0, 5.0], 0, [0, 0, 0, 0], [], [])
        highs.changeColIntegrality(0, highspy.HighsVarType.kInteger)
        highs.changeColIntegrality(1, highspy.HighsVarType.kInteger)
        highs.addRow(2.0, 4.0, 3, [0, 1, 2], [1.0, 2.0, -1.0])
        highs.changeObjectiveSense(highspy.ObjSense.kMaximize)
        highs.changeObjectiveOffset(7.0)

        with tempfile.TemporaryDirectory() as directory:
            for extension in ("lp", "mps"):
                with self.subTest(extension=extension):
                    source = Path(directory) / f"source.{extension}"
                    model = Path(directory) / f"imported_{extension}.mod"
                    data = Path(directory) / f"imported_{extension}.dat"
                    self.assertIn(highs.writeModel(str(source)), (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning))
                    convert_file(source, model, data)
                    self.assertIn("maximize objOffset", model.read_text())
                    self.assertIn("isBinary = [1, 0, 0]", data.read_text())
                    self.assertIn("isInteger = [0, 1, 0]", data.read_text())
                    self.assertIn("numConstraints = 2", data.read_text())
                    problem = linear_problem_from_opl(model.read_text(), data.read_text())
                    self.assertEqual(problem.sense, "maximize")
                    self.assertEqual(problem.c, [-2.0, 1.0, -3.0])

    def test_import_rejects_quadratic_models_without_generating_files(self):
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.addVar(0, 10)
        highs.passHessian(1, 1, highspy.HessianFormat.kTriangular, [0, 1], [0], [2.0])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quadratic.lp"
            model = Path(directory) / "quadratic.mod"
            data = Path(directory) / "quadratic.dat"
            self.assertIn(highs.writeModel(str(source)), (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning))
            with self.assertRaisesRegex(ValueError, "Quadratic objectives"):
                convert_file(source, model, data)
            self.assertFalse(model.exists())
            self.assertFalse(data.exists())

    def test_import_unconstrained_model_preserves_infinite_bound(self):
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.addCol(1.0, 0.0, highspy.kHighsInf, 0, [], [])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unconstrained.mps"
            model = Path(directory) / "unconstrained.mod"
            data = Path(directory) / "unconstrained.dat"
            self.assertIn(highs.writeModel(str(source)), (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning))
            convert_file(source, model, data)
            problem = linear_problem_from_opl(model.read_text(), data.read_text())
            self.assertEqual(problem.bounds, [[0.0, None]])
            self.assertNotIn("numConstraints", model.read_text())

    def test_import_wraps_wide_data_without_changing_model(self):
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        count = 120
        highs.addCols(count, [1.0] * count, [0.0] * count, [1.0] * count, 0, [0] * (count + 1), [], [])
        highs.addRow(1.0, highspy.kHighsInf, count, list(range(count)), [1.0] * count)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wide.mps"
            model = Path(directory) / "wide.mod"
            data = Path(directory) / "wide.dat"
            self.assertIn(highs.writeModel(str(source)), (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning))
            convert_file(source, model, data)
            self.assertLess(max(map(len, data.read_text().splitlines())), 1500)
            problem = linear_problem_from_opl(model.read_text(), data.read_text())
            self.assertEqual(len(problem.c), count)

    def test_import_stop_removes_incomplete_model_and_data(self):
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.addCol(1.0, 0.0, 2.0, 0, [], [])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mps"
            model = Path(directory) / "imported.mod"
            data = Path(directory) / "imported.dat"
            self.assertIn(highs.writeModel(str(source)), (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning))
            stop = threading.Event()
            stages = []

            def progress(stage, percent):
                stages.append((stage, percent))
                if percent == 90:
                    stop.set()

            with self.assertRaisesRegex(InterruptedError, "stopped"):
                convert_file(source, model, data, progress=progress, stop_requested=stop.is_set)
            self.assertIn(("Writing data...", 90), stages)
            self.assertFalse(model.exists())
            self.assertFalse(data.exists())


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from pyopl.linear_problem_highs import build_highs_model, export_linear_problem
from pyopl.pyopl_core import parse_model
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


class TestLinearProblemHighs(unittest.TestCase):
    def test_build_highs_model_and_export_lp_mps(self):
        ast = parse_model("dvar float+ x; minimize 2 * x + 3; subject to { x >= 1; x <= 4; }")
        problem = SciPyCSCCodeGenerator(ast).build_problem()

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


if __name__ == "__main__":
    unittest.main()

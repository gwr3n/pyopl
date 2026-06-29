import unittest

from pyopl.linear_problem import LinearProblem
from pyopl.pyopl_core import parse_model
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


class TestLinearProblem(unittest.TestCase):
    def test_scipy_generator_builds_linear_problem_snapshot(self):
        ast = parse_model("dvar float+ x; minimize 2 * x + 3; subject to { x >= 1; x <= 4; }")
        gen = SciPyCSCCodeGenerator(ast)

        problem = gen.build_problem()

        self.assertIsInstance(problem, LinearProblem)
        self.assertEqual(problem.sense, "minimize")
        self.assertEqual(problem.var_names, ["x"])
        self.assertEqual(problem.bounds, [[1.0, 4.0]])
        self.assertEqual(problem.integrality, [0])
        self.assertEqual(problem.c, [2.0])
        self.assertEqual(problem.A_ub, [[-1.0], [1.0]])
        self.assertEqual(problem.b_ub, [-1.0, 4.0])
        self.assertEqual(problem.A_eq, [])
        self.assertEqual(problem.b_eq, [])
        self.assertEqual(problem.objective_offset, 3.0)


if __name__ == "__main__":
    unittest.main()
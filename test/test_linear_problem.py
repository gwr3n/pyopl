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
        self.assertTrue(problem.objective_is_minimization_form)

    def test_maximization_snapshot_uses_minimization_form_for_full_objective(self):
        ast = parse_model("dvar float+ x; maximize 2*x+5; subject to { x <= 4; }")

        problem = SciPyCSCCodeGenerator(ast).build_problem()

        self.assertEqual(problem.sense, "maximize")
        self.assertEqual(problem.c, [-2.0])
        self.assertEqual(problem.objective_offset, -5.0)
        self.assertTrue(problem.objective_is_minimization_form)


if __name__ == "__main__":
    unittest.main()

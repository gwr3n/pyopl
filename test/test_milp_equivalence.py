import unittest

from pyopl.linear_problem import LinearProblem
from pyopl.milp_equivalence import EquivalenceResult, compare, prove_equivalent
from pyopl.pyopl_core import OPLLexer, OPLParser
from pyopl.scipy_codegen import SciPyCodeGenerator


class CompareTests(unittest.TestCase):
    def linear_problem_from_opl(self, model: str) -> LinearProblem:
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(model))
        generator = SciPyCodeGenerator(ast)
        return generator.build_problem()

    def test_compare_accepts_equivalent_opl_models_compiled_by_scipy_codegen(self):
        first = self.linear_problem_from_opl(
            """
            dvar float+ x;

            minimize 2 * x;

            subject to {
                x <= 3;
            }
            """
        )
        second = self.linear_problem_from_opl(
            """
            dvar float+ y;

            minimize 2 * y;

            subject to {
                2 * y <= 6;
            }
            """
        )

        self.assertNotEqual(first.var_names, second.var_names)
        self.assertNotEqual(first.A_ub, second.A_ub)
        self.assertTrue(compare(first, second))

    def test_prove_equivalent_returns_equivalent_result(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x"],
            bounds=[[0, 1]],
            integrality=[1],
            c=[1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )
        right = LinearProblem(
            sense="minimize",
            var_names=["renamed_x"],
            bounds=[[0, 1]],
            integrality=[1],
            c=[1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )

        result = prove_equivalent(left, right)

        self.assertIsInstance(result, EquivalenceResult)
        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "solver_implied")
        self.assertTrue(result.equivalent)
        self.assertIn("normalized both models", result.proof_steps)
        self.assertIsNone(result.counterexample)

    def test_prove_equivalent_returns_different_result(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )
        right = LinearProblem(
            sense="minimize",
            var_names=["x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[2.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )

        result = prove_equivalent(left, right)

        self.assertEqual(result.status, "different")
        self.assertIn("isomorphic", result.reason)
        self.assertFalse(result.equivalent)
        self.assertIn("tested labelled graph isomorphism", result.proof_steps)
        self.assertEqual(result.counterexample, "normalized graphs are not isomorphic")

    def test_prove_equivalent_returns_unknown_for_unimplemented_projection_mode(self):
        problem = LinearProblem(
            sense="minimize",
            var_names=["x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )

        result = prove_equivalent(problem, problem, mode="projection")

        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.equivalent)
        self.assertEqual(result.proof_steps, ())
        self.assertIsNone(result.counterexample)

    def test_prove_equivalent_honors_user_variable_mapping(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, None], [0, None]],
            integrality=[0, 1],
            c=[1.0, 2.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[4.0],
        )
        right = LinearProblem(
            sense="minimize",
            var_names=["b", "a"],
            bounds=[[0, None], [0, None]],
            integrality=[1, 0],
            c=[2.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[4.0],
        )

        correct = prove_equivalent(left, right, variable_mapping={"x": "a", "y": "b"})
        incorrect = prove_equivalent(left, right, variable_mapping={"x": "b", "y": "a"})

        self.assertEqual(correct.status, "equivalent")
        self.assertEqual(incorrect.status, "different")
        self.assertIn("variable mapping", incorrect.reason)

    def test_projection_mode_ignores_independent_auxiliary_variables(self):
        base = LinearProblem(
            sense="minimize",
            var_names=["x"],
            bounds=[[0, 3]],
            integrality=[0],
            c=[2.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )
        extended = LinearProblem(
            sense="minimize",
            var_names=["renamed_x", "aux"],
            bounds=[[0, 3], [0, 1]],
            integrality=[0, 0],
            c=[2.0, 0.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )

        result = prove_equivalent(
            base,
            extended,
            mode="projection",
            variable_mapping={"x": "renamed_x"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertIn("projected unmapped auxiliary variables", result.proof_steps)

    def test_compare_accepts_permuted_names_rows_columns_and_row_scaling(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x", "y", "z"],
            bounds=[[0, None], [0, 10], [1, None]],
            integrality=[0, 1, 0],
            c=[3.0, 2.0, -1.0],
            A_eq=[[1.0, 1.0, 0.0]],
            b_eq=[5.0],
            A_ub=[[2.0, 0.0, 1.0], [0.0, 4.0, 4.0]],
            b_ub=[8.0, 12.0],
            objective_offset=7.0,
        )

        right = LinearProblem(
            sense="minimize",
            var_names=["renamed_z", "renamed_x", "renamed_y"],
            bounds=[[1, None], [0, None], [0, 10]],
            integrality=[0, 0, 1],
            c=[-1.0, 3.0, 2.0],
            A_eq=[[0.0, 3.0, 3.0]],
            b_eq=[15.0],
            A_ub=[[12.0, 0.0, 12.0], [5.0, 10.0, 0.0]],
            b_ub=[36.0, 40.0],
            objective_offset=7.0,
        )

        self.assertTrue(compare(left, right))

    def test_compare_rejects_changed_objective_coefficient(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, None], [0, None]],
            integrality=[0, 1],
            c=[1.0, 2.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[4.0],
        )
        right = LinearProblem(
            sense="minimize",
            var_names=["a", "b"],
            bounds=[[0, None], [0, None]],
            integrality=[0, 1],
            c=[1.0, 3.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[4.0],
        )

        self.assertFalse(compare(left, right))

    def test_compare_handles_maximize_by_normalizing_objective_sign(self):
        minimize = LinearProblem(
            sense="minimize",
            var_names=["x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[-2.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0]],
            b_ub=[3.0],
        )
        maximize = LinearProblem(
            sense="maximize",
            var_names=["renamed_x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[2.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[2.0]],
            b_ub=[6.0],
        )

        self.assertTrue(compare(minimize, maximize))

    def test_compare_handles_mixed_finite_and_infinite_bounds(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, None], [0, 5]],
            integrality=[0, 0],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[3.0],
        )
        right = LinearProblem(
            sense="minimize",
            var_names=["renamed_y", "renamed_x"],
            bounds=[[0, 5], [0, None]],
            integrality=[0, 0],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[2.0, 2.0]],
            b_ub=[6.0],
        )

        self.assertTrue(compare(left, right))

    def test_compare_accepts_finite_bounds_as_explicit_rows(self):
        bounded = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, 5], [-2, None]],
            integrality=[0, 1],
            c=[1.0, -3.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[4.0],
        )
        row_bounded = LinearProblem(
            sense="minimize",
            var_names=["renamed_y", "renamed_x"],
            bounds=[[None, None], [None, None]],
            integrality=[1, 0],
            c=[-3.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[
                [2.0, 2.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            b_ub=[8.0, 0.0, 5.0, 2.0],
        )

        self.assertTrue(compare(bounded, row_bounded))

    def test_compare_ignores_duplicate_scaled_rows(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, None], [0, None]],
            integrality=[0, 0],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 2.0]],
            b_ub=[6.0],
        )
        right = LinearProblem(
            sense="minimize",
            var_names=["a", "b"],
            bounds=[[0, None], [0, None]],
            integrality=[0, 0],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[
                [1.0, 2.0],
                [2.0, 4.0],
                [0.5, 1.0],
            ],
            b_ub=[6.0, 12.0, 3.0],
        )

        self.assertTrue(compare(left, right))

    def test_compare_ignores_lp_redundant_inequality(self):
        minimal = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, 4], [0, 5]],
            integrality=[0, 0],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )
        redundant = LinearProblem(
            sense="minimize",
            var_names=["a", "b"],
            bounds=[[0, 4], [0, 5]],
            integrality=[0, 0],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[10.0],
        )

        self.assertTrue(compare(minimal, redundant))

    def test_compare_ignores_milp_redundant_inequality(self):
        minimal = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, 1], [0, 1]],
            integrality=[1, 1],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[],
            b_ub=[],
        )
        redundant = LinearProblem(
            sense="minimize",
            var_names=["a", "b"],
            bounds=[[0, 1], [0, 1]],
            integrality=[1, 1],
            c=[1.0, 1.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 1.0]],
            b_ub=[2.0],
        )

        self.assertTrue(compare(minimal, redundant))

    def test_compare_accepts_fixed_variable_substitution(self):
        with_fixed = LinearProblem(
            sense="minimize",
            var_names=["x", "z"],
            bounds=[[0, None], [2, 2]],
            integrality=[0, 0],
            c=[3.0, 4.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0, 5.0]],
            b_ub=[17.0],
            objective_offset=1.0,
        )
        substituted = LinearProblem(
            sense="minimize",
            var_names=["renamed_x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[3.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[1.0]],
            b_ub=[7.0],
            objective_offset=9.0,
        )

        self.assertTrue(compare(with_fixed, substituted))

    def test_compare_accepts_explicit_nonnegative_slack_variable(self):
        with_slack = LinearProblem(
            sense="minimize",
            var_names=["x", "s"],
            bounds=[[0, None], [0, None]],
            integrality=[0, 0],
            c=[2.0, 0.0],
            A_eq=[[3.0, 1.0]],
            b_eq=[12.0],
            A_ub=[],
            b_ub=[],
        )
        inequality = LinearProblem(
            sense="minimize",
            var_names=["renamed_x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[2.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[3.0]],
            b_ub=[12.0],
        )

        self.assertTrue(compare(with_slack, inequality))

    def test_compare_accepts_affine_alias_substitution(self):
        with_alias = LinearProblem(
            sense="minimize",
            var_names=["x", "y"],
            bounds=[[0, None], [None, None]],
            integrality=[0, 0],
            c=[3.0, 5.0],
            A_eq=[[-2.0, 1.0]],
            b_eq=[1.0],
            A_ub=[[1.0, 3.0]],
            b_ub=[20.0],
            objective_offset=7.0,
        )
        substituted = LinearProblem(
            sense="minimize",
            var_names=["renamed_x"],
            bounds=[[0, None]],
            integrality=[0],
            c=[13.0],
            A_eq=[],
            b_eq=[],
            A_ub=[[7.0]],
            b_ub=[17.0],
            objective_offset=12.0,
        )

        self.assertTrue(compare(with_alias, substituted))

    def test_compare_accepts_isomorphic_symmetric_constraint_graphs(self):
        left = LinearProblem(
            sense="minimize",
            var_names=["x0", "x1", "x2"],
            bounds=[[0, None], [0, None], [0, None]],
            integrality=[0, 0, 0],
            c=[0.0, 0.0, 0.0],
            A_eq=[],
            b_eq=[],
            A_ub=[
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            b_ub=[1.0, 1.0, 1.0],
        )
        right = LinearProblem(
            sense="minimize",
            var_names=["y0", "y1", "y2"],
            bounds=[[0, None], [0, None], [0, None]],
            integrality=[0, 0, 0],
            c=[0.0, 0.0, 0.0],
            A_eq=[],
            b_eq=[],
            A_ub=[
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            b_ub=[1.0, 1.0, 1.0],
        )

        self.assertTrue(compare(left, right))


if __name__ == "__main__":
    unittest.main()
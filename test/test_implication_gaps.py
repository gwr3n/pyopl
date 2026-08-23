import io
import unittest
from contextlib import redirect_stdout
from typing import Any

from pyopl.pyopl_core import OPLCompiler
from pyopl.scipy_codegen_csc import BOOL_EPS, SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError

try:
    import gurobipy  # noqa: F401

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


def _solve_generated_model(model: str, solver: str) -> dict:
    _ast, code, _data = OPLCompiler().compile_model(model, solver=solver)
    namespace: dict[str, Any] = {"results_container": {}}
    with redirect_stdout(io.StringIO()):
        exec(code, namespace)
    return namespace["results_container"][f"{solver}_output"]


class TestGurobiImplicationGaps(unittest.TestCase):
    def test_unsupported_continuous_comparison_truth_is_rejected(self):
        model = """
        dvar float x;
        dvar float y;
        minimize 0;
        subject to {
            (x >= 0) => (y >= 1);
        }
        """

        with self.assertRaisesRegex(SemanticError, "cannot be represented exactly"):
            OPLCompiler().compile_model(model, solver="gurobi")

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is not available")
    def test_true_integer_equality_antecedent_enforces_consequent(self):
        result = _solve_generated_model(
            """
            dvar int+ x;
            dvar int+ y;
            minimize y;
            subject to {
                x <= 3;
                y <= 3;
                (x == 2) => (y >= 2);
                x == 2;
            }
            """,
            "gurobi",
        )

        self.assertEqual(result["status"], "OPTIMAL")
        self.assertGreaterEqual(result["solution"]["y"], 2.0 - 1e-9)

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is not available")
    def test_true_composite_antecedent_enforces_consequent(self):
        result = _solve_generated_model(
            """
            dvar boolean a;
            dvar boolean b;
            dvar boolean c;
            minimize c;
            subject to {
                ((a == 1) && (b == 1)) => (c == 1);
                a == 1;
                b == 1;
            }
            """,
            "gurobi",
        )

        self.assertEqual(result["status"], "OPTIMAL")
        self.assertGreaterEqual(result["solution"]["c"], 1.0 - 1e-9)

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is not available")
    def test_strict_positive_antecedent_allows_small_value_when_binary_is_one(self):
        result = _solve_generated_model(
            f"""
            dvar float x;
            dvar boolean y;
            minimize y;
            subject to {{
                x >= 0;
                x <= 1;
                x == {BOOL_EPS / 2};
                y == 1;
                (x > 0) => (y == 1);
            }}
            """,
            "gurobi",
        )

        self.assertEqual(result["status"], "OPTIMAL")

    def test_strict_positive_antecedent_uses_exact_contrapositive(self):
        model = """
        dvar float x;
        dvar boolean y;
        minimize y;
        subject to {
            x >= 0;
            x <= 7;
            (x > 0) => (y == 1);
        }
        """

        _ast, code, _data = OPLCompiler().compile_model(model, solver="gurobi")

        self.assertIn("addGenConstrIndicator(y, 0, x <= 0", code)
        self.assertNotIn("implication_flag_", code)


class TestSciPyImplicationGaps(unittest.TestCase):
    def test_unsupported_continuous_comparison_truth_is_rejected(self):
        model = """
        dvar float x;
        dvar float y;
        minimize 0;
        subject to {
            (x >= 0) => (y >= 1);
        }
        """

        with self.assertRaisesRegex(SemanticError, "cannot be represented exactly"):
            OPLCompiler().compile_model(model, solver="scipy")

    def test_strict_positive_antecedent_allows_small_value_when_binary_is_one(self):
        result = _solve_generated_model(
            f"""
            dvar float x;
            dvar boolean y;
            minimize y;
            subject to {{
                x >= 0;
                x <= 1;
                x == {BOOL_EPS / 2};
                y == 1;
                (x > 0) => (y == 1);
            }}
            """,
            "scipy",
        )

        self.assertEqual(result["status"], "OPTIMAL")

    def test_strict_positive_antecedent_uses_finite_upper_bound(self):
        model = """
        dvar float x;
        dvar boolean y;
        minimize y;
        subject to {
            x >= 0;
            x <= 7;
            (x > 0) => (y == 1);
        }
        """

        ast, _code, _data = OPLCompiler().compile_model(model, solver="scipy")
        generator = SciPyCSCCodeGenerator(ast)
        generator.generate_code()
        x_index = generator.var_indices["x"]
        y_index = generator.var_indices["y"]

        self.assertTrue(
            any(
                abs(row[x_index] - 1.0) < 1e-9
                and abs(row[y_index] + 7.0) < 1e-9
                and abs(rhs) < 1e-9
                for row, rhs in zip(generator.A_ub, generator.b_ub)
            )
        )


class TestImplicationParserGaps(unittest.TestCase):
    def test_nested_implication_is_right_associative_and_supported(self):
        model = """
        dvar boolean a;
        dvar boolean b;
        dvar boolean c;
        minimize 0;
        subject to {
            (a == 1) => ((b == 1) => (c == 1));
        }
        """

        ast, _code, _data = OPLCompiler().compile_model(model, solver="scipy")
        outer = ast["constraints"][0]

        self.assertEqual(outer["type"], "implication_constraint")
        nested = outer["consequent"]
        self.assertEqual(nested["type"], "implies")

        for solver in ("scipy", "gurobi"):
            _ast, code, _data = OPLCompiler().compile_model(model, solver=solver)
            self.assertTrue(code)


@unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is not available")
class TestImplicationBackendParity(unittest.TestCase):
    def test_strict_affine_to_binary_patterns_match_exact_truth_table(self):
        cases = (
            ("(x > 0) => (y == 1);", BOOL_EPS / 2, 1, "OPTIMAL"),
            ("(x > 0) => (y == 1);", BOOL_EPS / 2, 0, "INFEASIBLE"),
            ("(x < 0) => (y == 0);", -BOOL_EPS / 2, 0, "OPTIMAL"),
            ("(x < 0) => (y == 0);", -BOOL_EPS / 2, 1, "INFEASIBLE"),
        )

        for implication, x_value, y_value, expected_status in cases:
            for solver in ("scipy", "gurobi"):
                with self.subTest(implication=implication, x=x_value, y=y_value, solver=solver):
                    result = _solve_generated_model(
                        f"""
                        dvar float x;
                        dvar boolean y;
                        minimize 0;
                        subject to {{
                            x >= -1;
                            x <= 1;
                            x == {x_value};
                            y == {y_value};
                            {implication}
                        }}
                        """,
                        solver,
                    )

                    self.assertEqual(result["status"], expected_status)

    def test_boolean_implication_forms_have_matching_projected_solutions(self):
        cases = {
            "simple": "(a == 1) => (c == 1);",
            "negated": "(!(a == 1)) => (c == 1);",
            "composite": "((a == 1) && (b == 1)) => (c == 1);",
            "nested": "(a == 1) => ((b == 1) => (c == 1));",
        }
        assignments = ((0, 0), (0, 1), (1, 0), (1, 1))

        for case_name, implication in cases.items():
            for a_value, b_value in assignments:
                with self.subTest(case=case_name, a=a_value, b=b_value):
                    model = f"""
                    dvar boolean a;
                    dvar boolean b;
                    dvar boolean c;
                    minimize c;
                    subject to {{
                        {implication}
                        a == {a_value};
                        b == {b_value};
                    }}
                    """
                    scipy_result = _solve_generated_model(model, "scipy")
                    gurobi_result = _solve_generated_model(model, "gurobi")

                    self.assertEqual(scipy_result["status"], gurobi_result["status"])
                    self.assertAlmostEqual(
                        scipy_result["solution"]["c"],
                        gurobi_result["solution"]["c"],
                        places=7,
                    )

    def test_integer_comparison_antecedents_match_for_all_operators(self):
        expected = {
            "==": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
            "!=": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
            "<": {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 0},
            "<=": {(0, 0): 1, (0, 1): 1, (1, 0): 0, (1, 1): 1},
            ">": {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0},
            ">=": {(0, 0): 1, (0, 1): 0, (1, 0): 1, (1, 1): 1},
        }

        for operator, outcomes in expected.items():
            for (x_value, y_value), expected_c in outcomes.items():
                with self.subTest(operator=operator, x=x_value, y=y_value):
                    model = f"""
                    dvar int x;
                    dvar int y;
                    dvar boolean c;
                    minimize c;
                    subject to {{
                        (x {operator} y) => (c == 1);
                        x == {x_value};
                        y == {y_value};
                    }}
                    """
                    scipy_result = _solve_generated_model(model, "scipy")
                    gurobi_result = _solve_generated_model(model, "gurobi")

                    self.assertEqual(scipy_result["status"], "OPTIMAL")
                    self.assertEqual(gurobi_result["status"], "OPTIMAL")
                    self.assertAlmostEqual(scipy_result["solution"]["c"], expected_c, places=7)
                    self.assertAlmostEqual(gurobi_result["solution"]["c"], expected_c, places=7)


if __name__ == "__main__":
    unittest.main()

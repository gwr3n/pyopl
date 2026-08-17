import unittest

from pyopl.milp_abstract_equivalence import (
    AbstractEquivalenceResult,
    compare_abstract,
    parse_abstract_model,
    prove_abstract_equivalent,
)


LEFT_MODEL = """
    int N = ...;
    range Items = 1..N;
    float cost[Items] = ...;
    float capacity = ...;
    dvar float+ x[Items];

    minimize sum(i in Items) cost[i] * x[i];

    subject to {
        sum(i in Items) x[i] <= capacity;
        forall(i in Items) x[i] >= 0;
    }
"""


RENAMED_MODEL = """
    int itemCount = ...;
    range ProductIds = 1..itemCount;
    float unitCost[ProductIds] = ...;
    float limit = ...;
    dvar float+ quantity[ProductIds];

    minimize sum(product in ProductIds) quantity[product] * unitCost[product];

    subject to {
        forall(product in ProductIds) 0 <= quantity[product];
        limit >= sum(product in ProductIds) quantity[product];
    }
"""


class AbstractEquivalenceTests(unittest.TestCase):
    def test_compare_abstract_accepts_renamed_and_reordered_model_schema(self):
        self.assertTrue(compare_abstract(LEFT_MODEL, RENAMED_MODEL))

    def test_prove_abstract_equivalent_accepts_parser_asts(self):
        left_ast = parse_abstract_model(LEFT_MODEL)
        right_ast = parse_abstract_model(RENAMED_MODEL)

        result = prove_abstract_equivalent(left_ast, right_ast)

        self.assertIsInstance(result, AbstractEquivalenceResult)
        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "schema_isomorphic")
        self.assertTrue(result.equivalent)
        self.assertIn("linked declarations and bound iterators to their references", result.proof_steps)
        self.assertIsNone(result.counterexample)

    def test_compare_abstract_rejects_changed_symbolic_objective(self):
        changed = RENAMED_MODEL.replace(
            "quantity[product] * unitCost[product]",
            "2 * quantity[product] * unitCost[product]",
        )

        result = prove_abstract_equivalent(LEFT_MODEL, changed)

        self.assertEqual(result.status, "different")
        self.assertFalse(result.equivalent)
        self.assertIn("not isomorphic", result.reason)
        self.assertIsNotNone(result.counterexample)

    def test_compare_abstract_preserves_index_dimension_order(self):
        left = """
            int N = ...;
            int M = ...;
            range I = 1..N;
            range J = 1..M;
            float c[I][J] = ...;
            dvar float+ x[I][J];
            minimize sum(i in I, j in J) c[i][j] * x[i][j];
            subject to { forall(i in I, j in J) x[i][j] <= 1; }
        """
        right = """
            int rows = ...;
            int columns = ...;
            range R = 1..rows;
            range C = 1..columns;
            float cost[R][C] = ...;
            dvar float+ value[R][C];
            minimize sum(r in R, c in C) cost[r][c] * value[r][c];
            subject to { forall(r in R, c in C) value[c][r] <= 1; }
        """

        self.assertFalse(compare_abstract(left, right))

    def test_prove_abstract_equivalent_returns_unknown_for_malformed_ast(self):
        result = prove_abstract_equivalent(
            {"declarations": [], "constraints": []},
            parse_abstract_model(LEFT_MODEL),
        )

        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.equivalent)
        self.assertIn("objective", result.reason)
        self.assertEqual(result.proof_steps, ())

    def test_parse_abstract_model_keeps_symbolic_external_parameters(self):
        ast = parse_abstract_model(LEFT_MODEL)
        declarations = {declaration["name"]: declaration for declaration in ast["declarations"]}

        self.assertEqual(declarations["N"]["type"], "parameter_external")
        self.assertEqual(declarations["cost"]["type"], "parameter_external_explicit_indexed")
        self.assertNotIn("value", declarations["cost"])

    def test_algebraic_mode_normalizes_expanded_affine_expressions(self):
        left = """
            dvar float x;
            minimize 2 * (x + 1);
            subject to { x >= 0; }
        """
        right = """
            dvar float y;
            minimize 2 * y + 2;
            subject to { 0 <= y; }
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "symbolically_normalized")

    def test_algebraic_mode_eliminates_arbitrary_affine_alias(self):
        left = """
            dvar float x;
            minimize 3 * x + 4;
            subject to { x >= 0; x <= 5; }
        """
        right = """
            dvar float y;
            dvar float alias;
            minimize alias;
            subject to {
                2 * alias == 6 * y + 8;
                y >= 0;
                y <= 5;
            }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"alias"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "rewrite_certified")
        self.assertIn("affine aliases", " ".join(result.proof_steps))

    def test_algebraic_mode_projects_continuous_auxiliary(self):
        left = """
            dvar float x;
            minimize x;
            subject to { x >= 0; x <= 1; }
        """
        right = """
            dvar float y;
            dvar float slack;
            minimize y;
            subject to {
                y >= 0;
                slack >= y;
                slack <= 1;
            }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"slack"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "polyhedrally_proven")
        self.assertIn("Fourier-Motzkin", " ".join(result.proof_steps))

    def test_algebraic_mode_verifies_farkas_redundancy_certificates(self):
        left = """
            dvar float x;
            dvar float y;
            minimize x + y;
            subject to { x >= 0; y >= 0; x <= 1; y <= 1; }
        """
        right = """
            dvar float a;
            dvar float b;
            minimize a + b;
            subject to { a >= 0; b >= 0; a <= 1; b <= 1; a + b <= 2; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "a", "y": "b"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "polyhedrally_proven")
        self.assertIn("Farkas", " ".join(result.proof_steps))

    def test_algebraic_mode_honors_parameter_mapping(self):
        left = """
            float p = ...;
            dvar float x;
            minimize p * x + p;
            subject to { x >= 0; }
        """
        right = """
            float coefficient = ...;
            dvar float y;
            minimize coefficient * (y + 1);
            subject to { 0 <= y; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            parameter_mapping={"p": "coefficient"},
            variable_mapping={"x": "y"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "symbolically_normalized")

    def test_algebraic_mode_eliminates_bounded_integer_auxiliary(self):
        left = """
            dvar int x;
            minimize x;
            subject to { x >= 0; x <= 1; }
        """
        right = """
            dvar int y;
            dvar int auxiliary;
            minimize y;
            subject to { y >= 0; y <= 1; auxiliary >= 0; auxiliary <= 1; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"auxiliary"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "presburger_proven")

    def test_algebraic_mode_saturates_chained_alias_rewrites(self):
        left = """
            dvar float x;
            minimize x + 2;
            subject to { x >= 0; }
        """
        right = """
            dvar float y;
            dvar float first;
            dvar float second;
            minimize second;
            subject to { first == y + 1; second == first + 1; y >= 0; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"first", "second"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "rewrite_certified")

    def test_algebraic_mode_returns_unknown_for_indexed_schema(self):
        result = prove_abstract_equivalent(LEFT_MODEL, RENAMED_MODEL, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertIn("scalar declarations", result.reason)


if __name__ == "__main__":
    unittest.main()
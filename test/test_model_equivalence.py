import unittest
from typing import Any

from pyopl import compare_models
from pyopl.milp_abstract_equivalence import AbstractEquivalenceResult
from pyopl.milp_concrete_equivalence import EquivalenceResult
from pyopl.model_equivalence import comparison_result_to_dict


class ModelEquivalenceApiTests(unittest.TestCase):
    def test_public_auxiliary_examples(self):
        cases = (
            (
                "dvar float x; minimize 3*x+4; subject to {x>=0; x<=5;}",
                "dvar float y; dvar float auxiliary; minimize auxiliary; subject to {2*auxiliary==6*y+8; y>=0; y<=5;}",
                "rewrite_certified",
            ),
            (
                "dvar float x; minimize x; subject to {x>=0; x<=1;}",
                "dvar float y; dvar float auxiliary; minimize y; subject to {y>=0; auxiliary>=y; auxiliary<=1;}",
                "polyhedrally_proven",
            ),
        )
        for left, right, level in cases:
            with self.subTest(level=level):
                result = compare_models(left, right, variable_mapping={"x": "y"}, right_auxiliaries={"auxiliary"})
                self.assertTrue(result.equivalent)
                self.assertEqual(result.level, level)
                payload = comparison_result_to_dict(result, strategy="abstract")
                self.assertEqual(payload["relation"], "projected_value")
                self.assertEqual(payload["variable_mapping"], {"x": "y"})
                self.assertEqual(payload["scope"], "parameterless_instance")
                self.assertEqual(payload["arithmetic"], "exact_on_parsed_values")

    def test_supplied_data_cannot_be_bypassed_by_schema_match(self):
        model = "float bound = ...; dvar float x; minimize x; subject to {x>=0; x<=bound;}"
        result = compare_models(model, model, left_data_text="bound=1;", right_data_text="bound=2;")
        self.assertFalse(result.equivalent)
        self.assertEqual(result.scope, "supplied_instances")
        self.assertEqual(result.arithmetic, "exact_on_embedded_matrix_values")

    def test_public_invalid_options_raise(self) -> None:
        model = "dvar float x; minimize x; subject to {}"
        invalid_options: tuple[dict[str, Any], ...] = (
            {"right_auxiliaries": {"missing"}},
            {"variable_mapping": {"x": "x"}, "left_auxiliaries": {"x"}},
            {"strategy": "concrete", "assumptions": {"theta": "positive"}},
            {"strategy": "concrete", "right_auxiliaries": {"x"}},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare_models(model, model, **options)

    def test_public_concrete_partition(self):
        left = "dvar boolean x; minimize x; subject to {}"
        right = "dvar boolean y; dvar float auxiliary; minimize y; subject to {auxiliary>=y;}"
        result = compare_models(left, right, strategy="concrete", variable_mapping={"x": "y"}, right_auxiliaries={"auxiliary"})
        self.assertTrue(result.equivalent)
        self.assertEqual(result.relation, "projected_value")
        self.assertEqual(result.right_auxiliaries, ("auxiliary",))

    def test_schema_assumptions_and_parameter_map(self):
        left = "float theta = ...; dvar float x; minimize x; subject to {}"
        right = "float theta = ...; dvar float y; dvar float auxiliary; minimize y; subject to {theta*auxiliary==-y;}"
        result = compare_models(
            left,
            right,
            variable_mapping={"x": "y"},
            parameter_mapping={"theta": "theta"},
            right_auxiliaries={"auxiliary"},
            assumptions={"theta": "positive"},
        )
        self.assertTrue(result.equivalent)
        self.assertEqual(result.scope, "uniform_schema")
        self.assertEqual(dict(result.assumptions), {"theta": "positive"})

    def test_structural_and_numerical_provenance(self):
        model = "dvar float+ x; minimize x; subject to {x<=1;}"
        symbolic = compare_models(model, model)
        numeric = compare_models(model, model, strategy="concrete")
        self.assertEqual(symbolic.relation, "schema_structural")
        self.assertEqual(dict(symbolic.variable_mapping), {"x": "x"})
        self.assertEqual(numeric.relation, "normalized_structural")
        self.assertEqual(numeric.arithmetic, "quantized_with_numerical_reductions")

    def test_compare_models_defaults_to_abstract_strategy(self):
        left = "dvar float+ x; minimize x; subject to { x <= 3; }"
        right = "dvar float+ y; minimize y; subject to { 2 * y <= 6; }"

        result = compare_models(left, right)

        self.assertIsInstance(result, AbstractEquivalenceResult)
        self.assertTrue(result.equivalent)

    def test_compare_models_selects_concrete_strategy(self):
        left = "dvar float+ x; minimize x; subject to { x <= 3; }"
        right = "dvar float+ y; minimize y; subject to { 2 * y <= 6; }"

        result = compare_models(left, right, strategy="concrete")

        self.assertIsInstance(result, EquivalenceResult)
        self.assertTrue(result.equivalent)

    def test_concrete_strategy_normalizes_compiled_maximization_once(self):
        minimize = "dvar float+ x; minimize -2*x-5; subject to { x <= 3; }"
        maximize = "dvar float+ y; maximize 2*y+5; subject to { y <= 3; }"

        result = compare_models(minimize, maximize, strategy="concrete")

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "solver_implied")

    def test_concrete_strategy_uses_projected_milp_fallback(self):
        left = """
            dvar boolean x; dvar float+ load;
            minimize x; subject to { load <= x; }
        """
        right = """
            dvar boolean x; dvar float+ flowA; dvar float+ flowB;
            minimize x; subject to { flowA + flowB <= x; }
        """

        result = compare_models(left, right, strategy="concrete")

        self.assertTrue(result.equivalent)
        self.assertEqual(result.level, "projected_milp_proven")

    def test_compare_models_selects_abstract_strategy_without_data(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float+ x[I];
            minimize sum(i in I) x[i];
            subject to { forall(i in I) x[i] <= N; }
        """
        right = """
            int count = ...;
            range J = 1..count;
            dvar float+ y[J];
            minimize sum(j in J) y[j];
            subject to { forall(j in J) y[j] <= count; }
        """

        result = compare_models(left, right, strategy="abstract")

        self.assertIsInstance(result, AbstractEquivalenceResult)
        self.assertTrue(result.equivalent)
        self.assertEqual(result.level, "schema_isomorphic")

    def test_comparison_result_dictionary_records_strategy(self):
        result = compare_models(
            "dvar float x; minimize 2 * (x + 1); subject to { x >= 0; }",
            "dvar float y; minimize 2 * y + 2; subject to { y >= 0; }",
            strategy="abstract",
        )

        payload = comparison_result_to_dict(result, strategy="abstract")

        self.assertEqual(payload["strategy"], "abstract")
        self.assertTrue(payload["equivalent"])

    def test_abstract_strategy_passes_data_for_indexed_algebra(self):
        left = """
            int N = ...; range I = 1..N; float c[I] = ...; dvar float+ x[I];
            minimize sum(i in I) c[i] * x[i];
            subject to { forall(i in I) x[i] <= 2; }
        """
        right = """
            int N = ...; range I = 1..N; float c[I] = ...; dvar float+ x[I];
            minimize sum(i in I) x[i] * c[i];
            subject to { forall(i in I) 2 >= x[i]; }
        """
        data = "N = 2; c = [4, 5];"

        result = compare_models(
            left,
            right,
            strategy="abstract",
            left_data_text=data,
            right_data_text=data,
        )

        self.assertTrue(result.equivalent)

    def test_compare_models_rejects_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported model comparison strategy"):
            compare_models("left", "right", strategy="unsupported")


if __name__ == "__main__":
    unittest.main()

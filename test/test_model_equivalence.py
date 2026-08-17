import unittest

from pyopl import compare_models
from pyopl.milp_abstract_equivalence import AbstractEquivalenceResult
from pyopl.milp_concrete_equivalence import EquivalenceResult
from pyopl.model_equivalence import comparison_result_to_dict


class ModelEquivalenceApiTests(unittest.TestCase):
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

    def test_compare_models_rejects_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported model comparison strategy"):
            compare_models("left", "right", strategy="unsupported")


if __name__ == "__main__":
    unittest.main()
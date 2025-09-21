import os
import tempfile
import unittest

from pyopl.pyopl_core import solve

try:
    import pyopl  # noqa: F401

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class TestSumOfComparisonsCardinality(unittest.TestCase):
    """Stage 1 TDD tests for sum-of-comparisons linearization parity.

    These tests exercise cardinality-style constraints that sum freshly formed
    linear comparison predicates. They are currently expected to fail for both
    backends (no generic reification of comparison terms inside sums yet).

    Remove @expectedFailure decorators once reification + linearization is implemented.
    """

    def test_sum_of_comparisons_ge(self):
        model_code = """
        range I = 1..3;
        dvar int x[I];
        dvar int y[I];
        minimize 0;
        subject to {
            forall(i in I) x[i] >= 0;
            forall(i in I) y[i] >= 0;
            forall(i in I) x[i] <= 5;
            forall(i in I) y[i] <= 5;
            sum(i in I) (x[i] >= y[i]) >= 2;
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                # Desired future behavior: model solves optimally.
                # Assert success (will currently fail, satisfying expectedFailure).
                self.assertEqual(
                    res.get("status"),
                    "OPTIMAL",
                    f"{solver} failed to solve sum-of-comparisons >= model",
                )
                self.assertIn("objective_value", res)
            finally:
                os.remove(path)

    def test_sum_of_comparisons_eq(self):
        model_code = """
        range I = 1..3;
        dvar int x[I];
        dvar int y[I];
        minimize 0;
        subject to {
            forall(i in I) x[i] >= 0;
            forall(i in I) y[i] >= 0;
            forall(i in I) x[i] <= 4;
            forall(i in I) y[i] <= 4;
            sum(i in I) (x[i] >= y[i]) == 1;
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertEqual(
                    res.get("status"),
                    "OPTIMAL",
                    f"{solver} failed to solve sum-of-comparisons == model",
                )
                self.assertIn("objective_value", res)
            finally:
                os.remove(path)

    def test_reified_cardinality_equality(self):
        model_code = """
        range I = 1..4;
        dvar int x[I];
        dvar int y[I];
        dvar boolean b;
        minimize b;
        subject to {
            forall(i in I) x[i] >= 0;
            forall(i in I) y[i] >= 0;
            forall(i in I) x[i] <= 3;
            forall(i in I) y[i] <= 3;
            b == (sum(i in I) (x[i] >= y[i]) >= 3);
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertEqual(
                    res.get("status"),
                    "OPTIMAL",
                    f"{solver} failed to solve reified cardinality model",
                )
                self.assertIn("objective_value", res)
            finally:
                os.remove(path)

    # --- Edge cases and strict inequalities ---
    def test_sum_of_comparisons_ge_zero(self):
        model_code = """
        range I = 1..3;
        dvar int x[I];
        dvar int y[I];
        minimize 0;
        subject to {
            forall(i in I) x[i] >= 0;
            forall(i in I) y[i] >= 0;
            sum(i in I) (x[i] >= y[i]) >= 0;
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertEqual(res.get("status"), "OPTIMAL", f"{solver} failed for k=0 edge case")
            finally:
                os.remove(path)

    def test_sum_of_comparisons_ge_full(self):
        model_code = """
        range I = 1..3;
        dvar int x[I];
        dvar int y[I];
        minimize 0;
        subject to {
            forall(i in I) x[i] >= 0;
            forall(i in I) y[i] >= 0;
            forall(i in I) x[i] <= 5;
            forall(i in I) y[i] <= 5;
            sum(i in I) (x[i] >= y[i]) >= 3; // |I|
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertEqual(
                    res.get("status"),
                    "OPTIMAL",
                    f"{solver} failed for full cardinality >= |I|",
                )
            finally:
                os.remove(path)

    def test_sum_of_comparisons_empty_iterator(self):
        model_code = """
        range I = 1..3;
        dvar int x[I];
        dvar int y[I];
        minimize 0;
        subject to {
            // index constraint false => empty sum
            sum(i in I : 1==0) (x[i] >= y[i]) == 0;
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertEqual(
                    res.get("status"),
                    "OPTIMAL",
                    f"{solver} failed for empty iterator sum",
                )
            finally:
                os.remove(path)

    def test_sum_of_comparisons_strict_gt(self):
        model_code = """
        range I = 1..4;
        dvar int x[I];
        dvar int y[I];
        minimize 0;
        subject to {
            forall(i in I) x[i] >= 0;
            forall(i in I) y[i] >= 0;
            forall(i in I) x[i] <= 5;
            forall(i in I) y[i] <= 5;
            sum(i in I) (x[i] >= y[i]) > 2; // requires at least 3 true
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertEqual(
                    res.get("status"),
                    "OPTIMAL",
                    f"{solver} failed for strict > cardinality",
                )
            finally:
                os.remove(path)

    def test_reified_cardinality_strict_gt(self):
        model_code = """
        range I = 1..4;
        dvar int x[I];
        dvar int y[I];
        dvar boolean b;
        minimize b;
        subject to {
            forall(i in I) x[i] >= 0;
            forall(i in I) y[i] >= 0;
            forall(i in I) x[i] <= 4;
            forall(i in I) y[i] <= 4;
            b == (sum(i in I) (x[i] >= y[i]) > 2); // reified strict >
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertEqual(
                    res.get("status"),
                    "OPTIMAL",
                    f"{solver} failed for reified strict > cardinality",
                )
            finally:
                os.remove(path)

    def test_infeasible_cardinality_conflict(self):
        """Infeasible: sum >= 3 and sum <= 1 when |I|=3."""
        model_code = """
        range I = 1..3;
        dvar int x[I]; dvar int y[I];
        minimize 0;
        subject to {
            forall(i in I) x[i] >= 0; forall(i in I) y[i] >= 0;
            forall(i in I) x[i] <= 5; forall(i in I) y[i] <= 5;
            sum(i in I) (x[i] >= y[i]) >= 3;
            sum(i in I) (x[i] >= y[i]) <= 1;
        }
        """
        for solver in ("gurobi", "scipy"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertIn(
                    res.get("status"),
                    ("INFEASIBLE", "FAILED"),
                    f"{solver} should report infeasible for conflicting cardinality",
                )
            finally:
                os.remove(path)


if __name__ == "__main__":
    unittest.main()

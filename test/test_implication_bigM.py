import re
import unittest

from pyopl.gurobi_codegen import GurobiCodeGenerator
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


class TestImplicationBigMTightness(unittest.TestCase):
    """Validate that implication big-M is tightened using boolean sum bounds (span small, not 1e6).

    We use boolean sums so _linear_bounds_safe can infer bounds:
      a+b in [0,2]; c+d in [0,2]; (a+b) - (c+d) in [-2,2] so span=2.
    Implication: (a + b >= c + d) => (a + b <= c + d + 1)
    Expected bigM around 2 (<= 10) and definitely << 1e6 in both backends.
    """

    def _ast(self):
        return {
            "declarations": [
                {"type": "dvar", "name": "a", "var_type": "boolean"},
                {"type": "dvar", "name": "b", "var_type": "boolean"},
                {"type": "dvar", "name": "c", "var_type": "boolean"},
                {"type": "dvar", "name": "d", "var_type": "boolean"},
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
            "constraints": [
                {
                    "type": "implication_constraint",
                    "antecedent": {
                        "type": "constraint",
                        "op": ">=",
                        "left": {
                            "type": "binop",
                            "op": "+",
                            "left": {"type": "name", "value": "a"},
                            "right": {"type": "name", "value": "b"},
                        },
                        "right": {
                            "type": "binop",
                            "op": "+",
                            "left": {"type": "name", "value": "c"},
                            "right": {"type": "name", "value": "d"},
                        },
                    },
                    "consequent": {
                        "type": "constraint",
                        "op": "<=",
                        "left": {
                            "type": "binop",
                            "op": "+",
                            "left": {"type": "name", "value": "a"},
                            "right": {"type": "name", "value": "b"},
                        },
                        "right": {
                            "type": "binop",
                            "op": "+",
                            "left": {
                                "type": "binop",
                                "op": "+",
                                "left": {"type": "name", "value": "c"},
                                "right": {"type": "name", "value": "d"},
                            },
                            "right": {"type": "number", "value": 1},
                        },
                    },
                }
            ],
        }

    def test_gurobi_bigM_tight(self):
        ast = self._ast()
        code = GurobiCodeGenerator(ast).generate_code()
        # Antecedent now uses indicator constraints; ensure they are present
        self.assertIn(
            "addGenConstrIndicator(implication_flag_c0, 1",
            code,
            f"Expected indicator constraint for antecedent. Code:\n{code}",
        )
        self.assertIn(
            "addGenConstrIndicator(implication_flag_c0, 0",
            code,
            f"Expected indicator constraint for negated antecedent. Code:\n{code}",
        )
        # Big-M now appears on the consequent side; extract it from the (1 - implication_flag_c0) term
        m = re.search(r"[<>]=\s*([0-9]+(?:\.[0-9]+)?)\s*\*\s*\(1 - implication_flag_c0\)", code)
        self.assertIsNotNone(m, f"Could not find big-M consequent line. Code:\n{code}")
        bigM = float(m.group(1))
        self.assertLess(bigM, 1000, f"Expected tightened M < 1000, got {bigM}")
        self.assertLessEqual(bigM, 10, f"Expected M <= 10 (span-based), got {bigM}")

    def test_scipy_bigM_tight(self):
        ast = self._ast()
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        flag_idx = gen.var_indices.get("implication_flag_c0")
        self.assertIsNotNone(flag_idx, f"Flag var missing; vars={gen.var_names}")
        max_coef = 0
        for row in gen.A_ub:
            coef = row[flag_idx]
            if abs(coef) > max_coef:
                max_coef = abs(coef)
        self.assertGreater(max_coef, 0, "Expected non-zero big-M coefficient in SciPy inequalities")
        self.assertLess(max_coef, 1000, f"Expected tightened M < 1000, got {max_coef}")
        self.assertLessEqual(max_coef, 10, f"Expected M <= 10 (span-based), got {max_coef}")


if __name__ == "__main__":
    unittest.main()

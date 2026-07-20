import unittest

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


class TestImplicationEqualityAntecedent(unittest.TestCase):
    def _decl_bool(self, name):
        return {"type": "dvar", "name": name, "var_type": "boolean"}

    def _name(self, v):
        return {"type": "name", "value": v}

    def _num(self, v):
        return {"type": "number", "value": v}

    def _bin(self, op, left, right):
        return {"type": "binop", "op": op, "left": left, "right": right}

    def _constraint(self, left, op, right):
        return {"type": "constraint", "left": left, "op": op, "right": right}

    def test_bigM_tight_for_equality_antecedent_and_consequent(self):
        # (a == b) => (a + b <= 2)  diff antecedent in [-1,1] so bigM should be 1; consequent diff a+b-2 in [-2,0]; M_c should be 1.
        ast = {
            "declarations": [self._decl_bool("a"), self._decl_bool("b")],
            "objective": {"type": "minimize", "expression": self._num(0)},
            "constraints": [
                {
                    "type": "implication_constraint",
                    "antecedent": self._constraint(self._name("a"), "==", self._name("b")),
                    "consequent": self._constraint(
                        self._bin("+", self._name("a"), self._name("b")),
                        "<=",
                        self._num(2),
                    ),
                }
            ],
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        flag = "implication_flag_c0"
        self.assertIn(flag, gen.var_indices, f"Expected flag var {flag}; vars={gen.var_names}")
        flag_idx = gen.var_indices[flag]
        max_flag_coef = 0.0
        for row in gen.A_ub:
            if flag_idx < len(row):
                coef = row[flag_idx]
                max_flag_coef = max(max_flag_coef, abs(coef))
        self.assertGreater(max_flag_coef, 0, "Flag should appear in inequalities")
        self.assertLessEqual(max_flag_coef, 10, f"Expected tight bigM <=10, got {max_flag_coef}")
        self.assertLess(max_flag_coef, 1000, f"bigM seems too large {max_flag_coef}")

    def test_unsupported_equality_consequent_error(self):
        # (a == b) => (a == 1) should raise semantic error until supported
        ast = {
            "declarations": [self._decl_bool("a"), self._decl_bool("b")],
            "objective": {"type": "minimize", "expression": self._num(0)},
            "constraints": [
                {
                    "type": "implication_constraint",
                    "antecedent": self._constraint(self._name("a"), "==", self._name("b")),
                    "consequent": self._constraint(self._name("a"), "==", self._num(1)),
                }
            ],
        }
        with self.assertRaises(SemanticError):
            gen = SciPyCSCCodeGenerator(ast)
            gen._build_variables()
            gen._build_objective()
            gen._build_constraints()

    def test_false_equality_antecedent_does_not_force_consequent(self):
        ast = {
            "declarations": [
                {"type": "dvar", "name": "x", "var_type": "int+"},
                {"type": "dvar", "name": "y", "var_type": "int+"},
            ],
            "objective": {"type": "minimize", "expression": self._name("y")},
            "constraints": [
                self._constraint(self._name("x"), ">=", self._num(1)),
                self._constraint(self._name("x"), "<=", self._num(3)),
                self._constraint(self._name("y"), "<=", self._num(3)),
                {
                    "type": "implication_constraint",
                    "antecedent": self._constraint(self._name("x"), "==", self._num(2)),
                    "consequent": self._constraint(self._name("y"), ">=", self._num(2)),
                },
            ],
        }
        gen = SciPyCSCCodeGenerator(ast)
        problem = gen.build_problem()
        constraints = []
        if problem.A_ub is not None:
            constraints.append(LinearConstraint(problem.A_ub, -float("inf"), problem.b_ub))
        if problem.A_eq:
            constraints.append(LinearConstraint(problem.A_eq, problem.b_eq, problem.b_eq))
        lower_bounds = [bound[0] if bound[0] is not None else -np.inf for bound in problem.bounds]
        upper_bounds = [bound[1] if bound[1] is not None else np.inf for bound in problem.bounds]
        result = milp(
            problem.c,
            integrality=problem.integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=constraints,
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.fun, 0.0)
        self.assertNotEqual(result.x[gen.var_indices["x"]], 2.0)


if __name__ == "__main__":
    unittest.main()

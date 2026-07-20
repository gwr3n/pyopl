import itertools
import unittest

import numpy as np

from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


class TestBooleanSemanticSafety(unittest.TestCase):
    def _decl(self, name, var_type):
        return {"type": "dvar", "name": name, "var_type": var_type}

    def _name(self, name, sem_type=None):
        node = {"type": "name", "value": name}
        if sem_type is not None:
            node["sem_type"] = sem_type
        return node

    def _number(self, value):
        return {"type": "number", "value": value}

    def _constraint(self, left, op, right):
        return {"type": "constraint", "left": left, "op": op, "right": right}

    def _comparison(self, left, op, right):
        return {"type": "binop", "left": left, "op": op, "right": right, "sem_type": "boolean"}

    def _ast(self, declarations, constraints, objective=None):
        return {
            "declarations": declarations,
            "constraints": constraints,
            "objective": objective or {"type": "minimize", "expression": self._number(0)},
        }

    def _build(self, ast):
        return SciPyCSCCodeGenerator(ast).build_problem()

    def _assignment_has_auxiliary_extension(self, problem, fixed_values):
        auxiliary_names = [name for name in problem.var_names if name not in fixed_values]
        for auxiliary_values in itertools.product((0, 1), repeat=len(auxiliary_names)):
            values = dict(fixed_values)
            values.update(zip(auxiliary_names, auxiliary_values))
            vector = np.array([values[name] for name in problem.var_names], dtype=float)
            if all(np.dot(row, vector) <= rhs + 1e-9 for row, rhs in zip(problem.A_ub, problem.b_ub)) and all(
                abs(np.dot(row, vector) - rhs) <= 1e-9 for row, rhs in zip(problem.A_eq, problem.b_eq)
            ):
                return True
        return False

    def test_asserted_boolean_tree_cannot_be_false(self):
        a_is_one = self._constraint(self._name("a"), "==", self._number(1))
        b_is_one = self._constraint(self._name("b"), "==", self._number(1))
        tree = {"type": "and", "left": a_is_one, "right": b_is_one, "sem_type": "boolean"}
        ast = self._ast(
            [self._decl("a", "boolean"), self._decl("b", "boolean")],
            [self._constraint(tree, "==", {"type": "boolean_literal", "value": True})],
        )

        problem = self._build(ast)

        self.assertFalse(
            self._assignment_has_auxiliary_extension(problem, {"a": 0, "b": 0}),
            "Asserting a boolean tree true must reject assignments where the tree is false",
        )

    def test_equality_truth_variable_is_true_when_equality_holds(self):
        equality = self._comparison(self._name("x", "int"), "==", self._number(0))
        ast = self._ast(
            [self._decl("x", "int"), self._decl("q", "boolean")],
            [
                self._constraint(self._name("x"), ">=", self._number(-1)),
                self._constraint(self._name("x"), "<=", self._number(1)),
                self._constraint(self._name("q", "boolean"), "==", equality),
            ],
        )

        problem = self._build(ast)

        self.assertFalse(
            self._assignment_has_auxiliary_extension(problem, {"x": 0, "q": 0}),
            "A comparison truth variable must equal one whenever its equality predicate holds",
        )

    def test_not_equal_truth_variable_accepts_negative_difference(self):
        not_equal = self._comparison(self._name("x", "int"), "!=", self._name("y", "int"))
        ast = self._ast(
            [self._decl("x", "int"), self._decl("y", "int"), self._decl("q", "boolean")],
            [
                self._constraint(self._name("x"), ">=", self._number(0)),
                self._constraint(self._name("x"), "<=", self._number(1)),
                self._constraint(self._name("y"), ">=", self._number(0)),
                self._constraint(self._name("y"), "<=", self._number(1)),
                self._constraint(self._name("q", "boolean"), "==", not_equal),
            ],
        )

        problem = self._build(ast)

        self.assertTrue(
            self._assignment_has_auxiliary_extension(problem, {"x": 0, "y": 1, "q": 1}),
            "A not-equal truth variable must accept either sign of a nonzero integer difference",
        )

    def test_unbounded_comparison_truth_variable_is_rejected(self):
        comparison = self._comparison(self._name("x", "float"), "<=", self._number(0))
        ast = self._ast(
            [self._decl("x", "float"), self._decl("q", "boolean")],
            [self._constraint(self._name("q", "boolean"), "==", comparison)],
        )

        with self.assertRaisesRegex(SemanticError, "finite.*bounds|big-M"):
            self._build(ast)

    def test_zero_antecedent_equality_enforces_both_sides(self):
        ast = self._ast(
            [self._decl("b", "boolean"), self._decl("x", "int")],
            [
                self._constraint(self._name("x"), ">=", self._number(-10)),
                self._constraint(self._name("x"), "<=", self._number(10)),
                {
                    "type": "implication_constraint",
                    "antecedent": self._constraint(self._name("b"), "==", self._number(0)),
                    "consequent": self._constraint(self._name("x"), "==", self._number(0)),
                },
            ],
        )

        problem = self._build(ast)

        self.assertFalse(
            self._assignment_has_auxiliary_extension(problem, {"b": 0, "x": -10}),
            "An equality consequent must enforce both its upper and lower sides",
        )

    def test_boolean_antecedent_with_unbounded_consequent_is_rejected(self):
        ast = self._ast(
            [self._decl("b", "boolean"), self._decl("x", "float")],
            [
                {
                    "type": "implication_constraint",
                    "antecedent": self._constraint(self._name("b"), "==", self._number(1)),
                    "consequent": self._constraint(self._name("x"), "<=", self._number(0)),
                }
            ],
        )

        with self.assertRaisesRegex(SemanticError, "finite.*bounds|big-M"):
            self._build(ast)


if __name__ == "__main__":
    unittest.main()

import itertools
import unittest

import numpy as np

import pyopl.scipy_codegen_csc as scipy_codegen_csc
from pyopl.pyopl_core import OPLLexer, OPLParser
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


class TestResidualSemanticShortcuts(unittest.TestCase):
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

    def _atom(self, name, value):
        return self._constraint(self._name(name, "boolean"), "==", self._number(value))

    def _ast(self, declarations, constraints, objective=None):
        return {
            "declarations": declarations,
            "constraints": constraints,
            "objective": objective or {"type": "minimize", "expression": self._number(0)},
        }

    def _build(self, ast, data=None):
        return SciPyCSCCodeGenerator(ast, data).build_problem()

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

    def test_composite_implication_does_not_assert_antecedent_leaves(self):
        source = """
        dvar boolean a;
        dvar boolean b;
        dvar boolean c;
        minimize 0;
        subject to {
            ((a == 1) && (b == 1)) => (c >= 1);
        }
        """
        ast = OPLParser().parse(OPLLexer().tokenize(source))

        problem = self._build(ast)

        self.assertTrue(
            self._assignment_has_auxiliary_extension(problem, {"a": 0, "b": 0, "c": 0}),
            "A false composite antecedent must leave its leaves and consequent unrestricted",
        )

    def test_reified_boolean_not_equal_has_xor_truth_table(self):
        expression = {
            "type": "constraint",
            "op": "!=",
            "left": self._atom("a", 1),
            "right": self._atom("b", 1),
        }
        ast = self._ast(
            [self._decl("a", "boolean"), self._decl("b", "boolean"), self._decl("q", "boolean")],
            [self._constraint(self._name("q", "boolean"), "==", expression)],
        )

        problem = self._build(ast)

        expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        for (a_value, b_value), q_value in expected.items():
            with self.subTest(a=a_value, b=b_value):
                self.assertTrue(
                    self._assignment_has_auxiliary_extension(
                        problem,
                        {"a": a_value, "b": b_value, "q": q_value},
                    )
                )
                self.assertFalse(
                    self._assignment_has_auxiliary_extension(
                        problem,
                        {"a": a_value, "b": b_value, "q": 1 - q_value},
                    )
                )

    def test_missing_range_bound_parameter_is_rejected(self):
        ast = self._ast(
            [
                {"type": "parameter_external", "name": "N", "base_type": "int"},
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [
                        {
                            "type": "range_index",
                            "start": self._number(1),
                            "end": self._name("N"),
                        }
                    ],
                },
            ],
            [],
        )

        with self.assertRaisesRegex(SemanticError, "N|bound|parameter"):
            self._build(ast)

    def test_missing_external_set_data_is_rejected(self):
        ast = self._ast(
            [
                {"type": "typed_set_external", "name": "Items", "base_type": "string", "value": None},
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [{"type": "named_set_dimension", "name": "Items"}],
                },
            ],
            [],
        )

        with self.assertRaisesRegex(SemanticError, "Items|set.*data"):
            self._build(ast)

    def test_reified_cardinality_preserves_duplicate_terms(self):
        duplicated_sum = {
            "type": "binop",
            "op": "+",
            "left": self._name("a", "boolean"),
            "right": self._name("a", "boolean"),
            "sem_type": "int",
        }
        predicate = self._constraint(duplicated_sum, ">=", self._number(2))
        ast = self._ast(
            [self._decl("a", "boolean"), self._decl("q", "boolean")],
            [self._constraint(self._name("q", "boolean"), "==", predicate)],
        )

        problem = self._build(ast)

        self.assertTrue(self._assignment_has_auxiliary_extension(problem, {"a": 1, "q": 1}))
        self.assertFalse(self._assignment_has_auxiliary_extension(problem, {"a": 1, "q": 0}))

    def test_unresolvable_cardinality_iterator_is_rejected(self):
        sum_node = {
            "type": "sum",
            "iterators": [{"iterator": "i", "range": {"type": "named_set", "name": "MissingSet"}}],
            "index_constraint": None,
            "expression": {
                "type": "binop",
                "op": "<=",
                "left": self._name("x", "int"),
                "right": self._number(0),
                "sem_type": "boolean",
            },
            "sem_type": "int",
        }
        ast = self._ast(
            [self._decl("x", "int")],
            [
                self._constraint(self._name("x"), ">=", self._number(-1)),
                self._constraint(self._name("x"), "<=", self._number(1)),
                self._constraint(sum_node, "<=", self._number(2)),
            ],
        )

        with self.assertRaisesRegex(SemanticError, "MissingSet|range or set|iterator"):
            self._build(ast)

    def test_unresolved_coefficient_is_never_dropped(self):
        generator = SciPyCSCCodeGenerator(self._ast([self._decl("x", "float")], []))
        generator._build_variables()
        vector = [0.0]

        with self.assertRaisesRegex(SemanticError, "missing|coefficient|variable"):
            generator._update_vector_from_coef_dict({"missing": 7.0}, vector)

        self.assertEqual(vector, [0.0])

    def test_strict_comparison_uses_solver_feasibility_tolerance(self):
        self.assertTrue(
            hasattr(scipy_codegen_csc, "SCIPY_FEASIBILITY_TOLERANCE"),
            "Strict inequalities need one named tolerance shared with the HiGHS solver options",
        )
        tolerance = scipy_codegen_csc.SCIPY_FEASIBILITY_TOLERANCE
        self.assertEqual(scipy_codegen_csc.BOOL_EPS, tolerance)
        self.assertEqual(scipy_codegen_csc.LINEAR_ZERO_TOLERANCE, 1e-12)

        ast = self._ast([self._decl("x", "float")], [self._constraint(self._name("x"), "<", self._number(1))])
        code = SciPyCSCCodeGenerator(ast).generate_code()
        self.assertIn(f"'primal_feasibility_tolerance': {tolerance!r}", code)
        self.assertIn(f"'dual_feasibility_tolerance': {tolerance!r}", code)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from pyopl.pyopl_core import OPLLexer, OPLParser
from pyopl.scipy_codegen_csc import BOOL_EPS, SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


class TestBooleanMixedTrees(unittest.TestCase):
    def test_mixed_precedence_with_parentheses_around_neq(self):
        """Test mixed precedence with parentheses around != (e.g., (a==1) != (b==0 OR c==1))."""
        # Expression: (a==1) != ((b==0) OR (c==1))
        expr = {
            "type": "constraint",
            "left": self._atom("a", 1),
            "op": "!=",
            "right": {
                "type": "or",
                "left": self._atom("b", 0),
                "right": self._atom("c", 1),
            },
        }
        ast = {
            "declarations": [
                self._decl_bool("a"),
                self._decl_bool("b"),
                self._decl_bool("c"),
            ],
            "constraints": [expr],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        aux_vars = [v for v in gen.var_names if v not in ("a", "b", "c")]
        # We expect auxiliaries for (b==0), (b==0 OR c==1), and for the != itself
        # (a==1) is atomic, but (b==0 OR c==1) should introduce an aux, and the != should introduce an aux
        # So expect at least 2 auxiliaries: one for (b==0 OR c==1), one for the !=
        self.assertGreaterEqual(
            len(aux_vars),
            2,
            f"Expected at least 2 auxiliary vars for (a==1) != (b==0 OR c==1); got {len(aux_vars)}: {aux_vars}",
        )
        # Optionally, check that one of the auxiliaries is used for the OR subtree
        # Should produce at least one equality or inequality constraint
        self.assertTrue(
            len(gen.A_eq) + len(gen.A_ub) > 0,
            "Expected at least one constraint row for != with parentheses.",
        )

    def test_nested_implications(self):
        """Test nested implications: (a==1 => (b==0 => c==1))."""
        # Expression: (a==1) => ((b==0) => (c==1))
        inner_imp = {
            "type": "implies",
            "left": self._atom("b", 0),
            "right": self._atom("c", 1),
        }
        outer_imp = {"type": "implies", "left": self._atom("a", 1), "right": inner_imp}
        expr = {
            "type": "constraint",
            "left": outer_imp,
            "op": "==",
            "right": {"type": "boolean_literal", "value": True},
        }
        ast = {
            "declarations": [
                self._decl_bool("a"),
                self._decl_bool("b"),
                self._decl_bool("c"),
            ],
            "constraints": [expr],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        aux_vars = [v for v in gen.var_names if v.startswith("_baux")]
        self.assertTrue(
            aux_vars,
            f"Expected auxiliary variable(s) for nested implications; var_names={gen.var_names}",
        )
        self.assertTrue(
            len(gen.A_eq) > 0 or len(gen.A_ub) > 0,
            "Expected at least one constraint row for nested implications.",
        )

    def _decl_bool(self, name):
        return {"type": "dvar", "name": name, "var_type": "boolean"}

    def _atom(self, var, val):
        # (var == val)
        return {
            "type": "constraint",
            "left": {"type": "name", "value": var},
            "op": "==",
            "right": {"type": "number", "value": val},
        }

    def _gen_from_opl(self, src):
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(src))
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        return gen

    def test_mixed_and_or_equivalent_comparisons(self):
        """Mixed AND/OR tree enforced via ==1, >=1, !=0 should each fix expression to 1 (three equality rows).
        Builds AST manually because parser currently restricts >=/!= on boolean trees.
        """
        # Boolean expression: (a==1) AND ((b==0) OR (c==1))
        and_or_tree = {
            "type": "and",
            "left": self._atom("a", 1),
            "right": {
                "type": "or",
                "left": self._atom("b", 0),
                "right": self._atom("c", 1),
            },
        }
        constraints = [
            {
                "type": "constraint",
                "left": and_or_tree,
                "op": "==",
                "right": {"type": "boolean_literal", "value": True},
            },
            {
                "type": "constraint",
                "left": and_or_tree,
                "op": ">=",
                "right": {"type": "number", "value": 1},
            },
            {
                "type": "constraint",
                "left": and_or_tree,
                "op": "!=",
                "right": {"type": "number", "value": 0},
            },
        ]
        ast = {
            "declarations": [
                self._decl_bool("a"),
                self._decl_bool("b"),
                self._decl_bool("c"),
            ],
            "constraints": constraints,
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        aux_vars = [v for v in gen.var_names if v.startswith("_baux")]
        self.assertTrue(
            aux_vars,
            f"Expected auxiliary variable for mixed tree; var_names={gen.var_names}",
        )
        ones = [rhs for rhs in gen.b_eq if abs(rhs - 1.0) < 1e-9]
        self.assertGreaterEqual(
            len(ones),
            3,
            f"Expected >=3 equality rows with RHS=1 enforcing expression true, got {len(ones)}; b_eq={gen.b_eq}",
        )

    def test_right_hand_boolean_tree_literal_comparisons(self):
        not_tree = {
            "type": "not",
            "value": self._atom("a", 1),
        }
        constraints = [
            {
                "type": "constraint",
                "left": {"type": "boolean_literal", "value": False},
                "op": "==",
                "right": not_tree,
            },
            {
                "type": "constraint",
                "left": not_tree,
                "op": ">=",
                "right": {"type": "number", "value": 1},
            },
            {
                "type": "constraint",
                "left": {"type": "boolean_literal", "value": False},
                "op": "<=",
                "right": not_tree,
            },
            {
                "type": "constraint",
                "left": {"type": "number", "value": 0},
                "op": ">=",
                "right": not_tree,
            },
            {
                "type": "constraint",
                "left": {"type": "number", "value": 1},
                "op": "<=",
                "right": not_tree,
            },
        ]
        ast = {
            "declarations": [self._decl_bool("a")],
            "constraints": constraints,
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
        }

        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()

        aux_vars = [v for v in gen.var_names if v.startswith("_baux")]
        self.assertEqual(len(aux_vars), 1, f"Expected one shared NOT auxiliary; var_names={gen.var_names}")
        self.assertEqual(gen.b_eq.count(1.0), 2, f"Expected NOT definition plus one enforced true row; b_eq={gen.b_eq}")
        self.assertEqual(gen.b_eq.count(0.0), 2, f"Expected NOT definition plus one enforced false row; b_eq={gen.b_eq}")
        self.assertEqual(gen.A_ub, [], f"Expected tautologies to add no inequality rows; A_ub={gen.A_ub}")

    def test_or_with_nested_and_linear_comparisons(self):
        def comparison(var, op, value):
            return {
                "type": "binop",
                "left": {"type": "name", "value": var},
                "op": op,
                "right": {"type": "number", "value": value},
                "sem_type": "boolean",
            }

        expression = {
            "type": "or",
            "left": {
                "type": "and",
                "left": comparison("a", "<=", 0),
                "right": {
                    "type": "and",
                    "left": comparison("b", ">=", 1),
                    "right": comparison("d", "==", 1),
                    "sem_type": "boolean",
                },
                "sem_type": "boolean",
            },
            "right": comparison("c", ">=", 1),
            "sem_type": "boolean",
        }
        ast = {
            "declarations": [self._decl_bool(name) for name in ("a", "b", "c", "d")],
            "constraints": [
                {
                    "type": "constraint",
                    "left": expression,
                    "op": "==",
                    "right": {"type": "boolean_literal", "value": True},
                }
            ],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
        }

        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        with patch.object(gen, "_bool_expr_var", side_effect=SemanticError("use specialized OR fallback")):
            gen._build_constraints()

        self.assertIn("or_flag_0", gen.var_indices)
        self.assertIn("or_flag_1", gen.var_indices)
        self.assertEqual(len(gen.A_ub), 6)
        selector_row = gen.A_ub[-1]
        self.assertEqual(selector_row[gen.var_indices["or_flag_0"]], -1.0)
        self.assertEqual(selector_row[gen.var_indices["or_flag_1"]], -1.0)
        self.assertEqual(gen.b_ub[-1], -1.0)

        equality_rows = [row for row in gen.A_ub if row[gen.var_indices["d"]] != 0]
        self.assertEqual(len(equality_rows), 2)
        self.assertEqual({row[gen.var_indices["d"]] for row in equality_rows}, {-1.0, 1.0})

    def test_and_or_literal_fast_path_resolves_variable_polarity(self):
        and_tree = {
            "type": "and",
            "left": self._atom("a", 1),
            "right": {
                "type": "constraint",
                "left": {"type": "number", "value": 0},
                "op": "==",
                "right": {"type": "name", "value": "b"},
            },
        }
        or_tree = {
            "type": "or",
            "left": self._atom("a", 1),
            "right": self._atom("b", 0),
        }
        ast = {
            "declarations": [self._decl_bool("a"), self._decl_bool("b")],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {"type": "boolean_literal", "value": True},
                    "op": "==",
                    "right": and_tree,
                },
                {
                    "type": "constraint",
                    "left": {"type": "boolean_literal", "value": False},
                    "op": "==",
                    "right": and_tree,
                },
                {
                    "type": "constraint",
                    "left": {"type": "boolean_literal", "value": True},
                    "op": "==",
                    "right": or_tree,
                },
                {
                    "type": "constraint",
                    "left": {"type": "boolean_literal", "value": False},
                    "op": "==",
                    "right": or_tree,
                },
            ],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
        }

        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()

        a_idx = gen.var_indices["a"]
        b_idx = gen.var_indices["b"]
        self.assertIn([1.0, 0.0], [[row[a_idx], row[b_idx]] for row in gen.A_eq])
        self.assertIn([0.0, 1.0], [[row[a_idx], row[b_idx]] for row in gen.A_eq])
        self.assertIn([1.0, -1.0], [[row[a_idx], row[b_idx]] for row in gen.A_ub])
        self.assertIn([-1.0, 1.0], [[row[a_idx], row[b_idx]] for row in gen.A_ub])
        self.assertIn([1.0, -1.0], [[row[a_idx], row[b_idx]] for row in gen.A_eq])
        self.assertIn(([1.0, -1.0], 0.0), [([row[a_idx], row[b_idx]], rhs) for row, rhs in zip(gen.A_ub, gen.b_ub)])
        self.assertIn(([-1.0, 1.0], 0.0), [([row[a_idx], row[b_idx]], rhs) for row, rhs in zip(gen.A_ub, gen.b_ub)])
        self.assertIn(([1.0, -1.0], -1.0), [([row[a_idx], row[b_idx]], rhs) for row, rhs in zip(gen.A_eq, gen.b_eq)])

    def test_weighted_boolean_sum_composite_comparison_rows(self):
        def comparison(var, op, value):
            return {
                "type": "binop",
                "left": {"type": "name", "value": var},
                "op": op,
                "right": {"type": "number", "value": value},
                "sem_type": "boolean",
            }

        weighted_sum = {
            "type": "sum",
            "iterators": [],
            "expression": {
                "type": "binop",
                "left": {"type": "number", "value": 2},
                "op": "*",
                "right": comparison("a", ">=", 1),
            },
        }

        cases = [
            (">=", "ub", -2.0, -2.0),
            (">", "ub", -2.0, -(2.0 + BOOL_EPS)),
            ("<=", "ub", 2.0, 2.0),
            ("<", "ub", 2.0, 2.0 - BOOL_EPS),
            ("==", "eq", 2.0, 2.0),
        ]

        for op, row_kind, expected_coef, expected_rhs in cases:
            with self.subTest(op=op):
                ast = {
                    "declarations": [self._decl_bool("a")],
                    "constraints": [
                        {
                            "type": "constraint",
                            "left": weighted_sum,
                            "op": op,
                            "right": {"type": "number", "value": 2},
                        }
                    ],
                    "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
                }

                gen = SciPyCSCCodeGenerator(ast)
                gen._build_variables()
                gen._build_objective()
                gen._build_constraints()

                cmp_flags = [name for name in gen.var_names if name.startswith("cmp_flag_")]
                self.assertEqual(len(cmp_flags), 1, f"Expected one comparison flag; var_names={gen.var_names}")
                flag_idx = gen.var_indices[cmp_flags[0]]
                rows = gen.A_eq if row_kind == "eq" else gen.A_ub
                rhs_values = gen.b_eq if row_kind == "eq" else gen.b_ub

                self.assertTrue(rows, f"Expected {row_kind} rows for weighted boolean sum")
                self.assertAlmostEqual(rows[-1][flag_idx], expected_coef)
                self.assertAlmostEqual(rhs_values[-1], expected_rhs)

    def test_weighted_boolean_sum_composite_filtered_iterator_rows(self):
        gen = self._gen_from_opl("""
            range I = 1..3;
            dvar boolean x[I];
            dvar boolean y[I];
            minimize 0;
            subject to {
                sum(i in I : i != 2) 2 * ((x[i] == 1) && (y[i] == 1)) >= 4;
            }
        """)

        baux_vars = [name for name in gen.var_names if name.startswith("_baux")]
        weighted_rows = [
            (row, rhs) for row, rhs in zip(gen.A_ub, gen.b_ub) if rhs == -4.0 and sum(1 for coef in row if coef == -2.0) == 2
        ]

        self.assertGreaterEqual(len(baux_vars), 2, f"Expected iterator-expanded AND auxiliaries; var_names={gen.var_names}")
        self.assertEqual(len(weighted_rows), 1, f"Expected one weighted >= row; A_ub={gen.A_ub}, b_ub={gen.b_ub}")

    def test_weighted_boolean_sum_rejects_symbolic_weight(self):
        ast = {
            "declarations": [self._decl_bool("a")],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {
                        "type": "sum",
                        "iterators": [],
                        "expression": {
                            "type": "binop",
                            "left": {"type": "string_literal", "value": "bad"},
                            "op": "*",
                            "right": {
                                "type": "and",
                                "left": self._atom("a", 1),
                                "right": self._atom("a", 1),
                                "sem_type": "boolean",
                            },
                        },
                    },
                    "op": ">=",
                    "right": {"type": "number", "value": 1},
                }
            ],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
        }

        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()

        with self.assertRaisesRegex(SemanticError, "numeric weights"):
            gen._build_constraints()

    # def test_boolean_tautologies_eliminated(self):
    #   """Tautological comparisons (expr >= 0, expr <= 1) should not add constraint rows (built manually)."""
    #   or_tree = {'type': 'or', 'left': self._atom('a', 1), 'right': self._atom('b', 0)}
    #   and_tree = {'type': 'and', 'left': self._atom('a', 1), 'right': self._atom('b', 1)}
    #   constraints = [
    #     {'type': 'constraint', 'left': or_tree, 'op': '<=', 'right': {'type': 'number', 'value': 1}},  # tautology
    #     {'type': 'constraint', 'left': and_tree, 'op': '>=', 'right': {'type': 'number', 'value': 0}},  # tautology
    #   ]
    #   ast = {
    #     'declarations': [self._decl_bool('a'), self._decl_bool('b')],
    #     'constraints': constraints,
    #     'objective': {'type': 'minimize', 'expression': {'type': 'number', 'value': 0}}
    #   }
    #   gen = SciPyCSCCodeGenerator(ast)
    #   gen._build_variables()
    #   gen._build_objective()
    #   gen._build_constraints()
    #   self.assertEqual(len(gen.A_eq), 0, f"Expected no equality constraints; got {gen.A_eq}")
    #   self.assertEqual(len(gen.A_ub), 0, f"Expected no inequality constraints; got {gen.A_ub}")

    def test_structural_aux_sharing(self):
        """Identical subtree structures in separate constraints should share same auxiliary variable."""
        # Expression: (a==1 AND b==0) OR (c==1)
        left_subtree1 = {
            "type": "and",
            "left": self._atom("a", 1),
            "right": self._atom("b", 0),
        }
        # Create a structurally identical but different object
        left_subtree2 = {
            "type": "and",
            "left": self._atom("a", 1),
            "right": self._atom("b", 0),
        }
        expr1 = {"type": "or", "left": left_subtree1, "right": self._atom("c", 1)}
        expr2 = {"type": "or", "left": left_subtree2, "right": self._atom("c", 1)}
        constraints = [
            {
                "type": "constraint",
                "left": expr1,
                "op": "==",
                "right": {"type": "boolean_literal", "value": True},
            },
            {
                "type": "constraint",
                "left": expr2,
                "op": "==",
                "right": {"type": "boolean_literal", "value": True},
            },
        ]
        ast = {
            "declarations": [
                self._decl_bool("a"),
                self._decl_bool("b"),
                self._decl_bool("c"),
            ],
            "constraints": constraints,
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        aux_vars = [v for v in gen.var_names if v.startswith("_baux")]
        # Expected auxiliaries:
        #  - One negation variable for (b == 0)
        #  - One AND auxiliary for (a==1 AND b==0)
        #  - One OR auxiliary for ( (a==1 AND b==0) OR (c==1) ) reused across constraints
        # Total expected with structural sharing across constraints: 3
        self.assertEqual(
            len(aux_vars),
            3,
            f"Expected exactly 3 auxiliary vars (neg(b), AND, OR) with sharing; got {len(aux_vars)}: {aux_vars}",
        )


if __name__ == "__main__":
    unittest.main()

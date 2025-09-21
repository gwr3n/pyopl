import unittest

from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


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
        aux_vars = [v for v in gen.var_names if v.startswith("_baux")]
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

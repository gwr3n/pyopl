import unittest

from pyopl.pyopl_core import OPLCompiler, SemanticError


def number(value):
    return {"type": "number", "value": value}


def name(value):
    return {"type": "name", "value": value}


class TestPyOPLCoreCoverageGaps(unittest.TestCase):
    def setUp(self):
        self.compiler = OPLCompiler()

    def test_boolean_literal_folding_handles_both_operands(self):
        true = self.compiler._boolean_literal(True)
        false = self.compiler._boolean_literal(False)
        variable = name("x")
        cases = (
            ({"type": "and", "left": true, "right": variable}, variable),
            ({"type": "and", "left": false, "right": variable}, false),
            ({"type": "and", "left": variable, "right": true}, variable),
            ({"type": "and", "left": variable, "right": false}, false),
            ({"type": "or", "left": true, "right": variable}, true),
            ({"type": "or", "left": false, "right": variable}, variable),
            ({"type": "or", "left": variable, "right": true}, true),
            ({"type": "or", "left": variable, "right": false}, variable),
            ({"type": "not", "value": true}, false),
            ({"type": "not", "value": false}, true),
        )

        for expression, expected in cases:
            with self.subTest(expression=expression):
                self.assertEqual(self.compiler._fold_boolean_literals(expression), expected)

    def test_condition_conversion_normalizes_supported_node_shapes(self):
        expression = {"type": "boolean_literal", "value": True}
        self.assertIs(
            self.compiler._condition_to_boolean_expr(
                {"type": "parenthesized_expression", "expression": expression}
            ),
            expression,
        )

        comparison = {"type": "constraint", "op": "<", "left": name("x"), "right": number(2)}
        self.assertEqual(
            self.compiler._condition_to_boolean_expr(comparison),
            {"type": "binop", "op": "<", "left": name("x"), "right": number(2), "sem_type": "boolean"},
        )
        self.assertEqual(
            self.compiler._condition_to_boolean_expr(
                {"type": "constraint", "op": "==", "left": expression, "right": self.compiler._boolean_literal(True)}
            ),
            expression,
        )
        self.assertEqual(
            self.compiler._condition_to_boolean_expr(
                {"type": "constraint", "op": "==", "left": expression, "right": self.compiler._boolean_literal(False)}
            ),
            {"type": "not", "value": expression, "sem_type": "boolean"},
        )
        self.assertEqual(
            self.compiler._condition_to_boolean_expr(name("x")),
            {"type": "binop", "op": "==", "left": name("x"), "right": self.compiler._boolean_literal(True), "sem_type": "boolean"},
        )
        self.assertEqual(self.compiler._condition_to_boolean_expr(True), self.compiler._boolean_literal(True))
        self.assertEqual(
            self.compiler._condition_to_boolean_expr(3)["left"],
            {"type": "number", "value": 3, "sem_type": "int"},
        )
        self.assertEqual(
            self.compiler._condition_to_boolean_expr(3.5)["left"],
            {"type": "number", "value": 3.5, "sem_type": "float"},
        )

    def test_forall_implication_rewrites_ground_antecedents(self):
        iterators = [{"iterator": "i", "range": {"type": "range_specifier", "start": number(1), "end": number(3)}}]
        base_condition = {"type": "boolean_literal", "value": True}
        consequent = {"type": "constraint", "op": ">=", "left": name("x"), "right": number(1)}
        implication = {
            "type": "implication_constraint",
            "antecedent": {"type": "constraint", "op": ">", "left": name("i"), "right": number(1)},
            "consequent": consequent,
        }

        rewritten = self.compiler._rewrite_forall_implication(implication, iterators, base_condition, {"x"})

        self.assertEqual(len(rewritten), 1)
        self.assertEqual(rewritten[0]["type"], "forall_constraint")
        self.assertEqual(rewritten[0]["iterators"], iterators)
        self.assertEqual(rewritten[0]["index_constraint"]["type"], "and")
        self.assertEqual(rewritten[0]["constraint"], consequent)

        variable_implication = dict(implication, antecedent=name("x"))
        self.assertEqual(
            self.compiler._rewrite_forall_implication(variable_implication, iterators, None, {"x"}),
            [variable_implication],
        )
        self.assertEqual(
            self.compiler._rewrite_forall_implication(
                dict(implication, consequent=None),
                iterators,
                None,
                set(),
            ),
            [],
        )

    def test_top_level_implications_only_keep_true_ground_antecedents(self):
        consequent = {"type": "constraint", "op": ">=", "left": name("x"), "right": number(1)}
        true_implication = {
            "type": "implication_constraint",
            "antecedent": self.compiler._boolean_literal(True),
            "consequent": consequent,
        }
        false_implication = dict(true_implication, antecedent=self.compiler._boolean_literal(False))

        self.assertIs(self.compiler._rewrite_top_level_implication(true_implication, {}, {"x"}), consequent)
        self.assertIsNone(self.compiler._rewrite_top_level_implication(false_implication, {}, {"x"}))
        variable_implication = dict(true_implication, antecedent=name("x"))
        self.assertIs(
            self.compiler._rewrite_top_level_implication(variable_implication, {}, {"x"}),
            variable_implication,
        )

    def test_maxmin_forall_expansion_preserves_multiple_constraints(self):
        aggregate = {"type": "maxl", "args": [name("x"), name("y")]}
        aggregate_constraint = {
            "type": "constraint",
            "op": "<=",
            "left": aggregate,
            "right": number(5),
        }
        forall = {"type": "forall_constraint", "iterators": [], "constraint": aggregate_constraint}

        rewritten = self.compiler._expand_maxmin_forall(forall)

        self.assertEqual(len(rewritten), 1)
        self.assertEqual(len(rewritten[0]["constraints"]), 2)
        self.assertNotIn("constraint", rewritten[0])

        grouped = self.compiler._expand_maxmin_forall(
            {"type": "forall_constraint", "iterators": [], "constraints": [aggregate_constraint, "skip"]}
        )
        self.assertEqual(len(grouped[0]["constraints"]), 3)
        untouched = {"type": "forall_constraint", "iterators": []}
        self.assertEqual(self.compiler._expand_maxmin_forall(untouched), [untouched])

    def test_minmax_aggregate_constraint_rewrites_all_convex_orientations(self):
        iterators = [{"iterator": "i", "range": {"type": "range_specifier", "start": number(1), "end": number(2)}}]
        cases = (
            ("max_agg", "<=", True),
            ("max_agg", ">=", False),
            ("min_agg", ">=", True),
            ("min_agg", "<=", False),
        )

        for aggregate_type, operator, aggregate_on_left in cases:
            aggregate = {
                "type": aggregate_type,
                "expression": name("x"),
                "iterators": iterators,
            }
            other = number(5)
            constraint = {
                "type": "constraint",
                "op": operator,
                "left": aggregate if aggregate_on_left else other,
                "right": other if aggregate_on_left else aggregate,
            }

            with self.subTest(aggregate=aggregate_type, operator=operator, left=aggregate_on_left):
                result = self.compiler._rewrite_minmax_constraint_body(constraint)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["type"], "forall_constraint")
                self.assertEqual(result[0]["constraint"]["op"], operator)
                self.assertEqual(result[0]["iterators"], iterators)

        with self.assertRaisesRegex(SemanticError, "Unsupported non-convex aggregate placement"):
            self.compiler._rewrite_minmax_constraint_body(
                {
                    "type": "constraint",
                    "op": "==",
                    "left": {"type": "max_agg", "expression": name("x"), "iterators": []},
                    "right": number(5),
                }
            )

    def test_minmax_objective_lowering_adds_auxiliary_variable_and_constraints(self):
        ast = {"declarations": [], "constraints": []}
        objective = {
            "type": "minimize",
            "expression": {
                "type": "max_agg",
                "expression": name("x"),
                "iterators": [{"iterator": "i", "range": {"type": "range_specifier", "start": number(1), "end": number(2)}}],
            },
        }

        self.compiler._lower_minmax_objective(ast, objective)

        auxiliary = objective["expression"]
        self.assertEqual(auxiliary["type"], "name")
        self.assertEqual(ast["declarations"][0]["name"], auxiliary["value"])
        self.assertEqual(ast["constraints"][0]["type"], "forall_constraint")
        self.assertEqual(ast["constraints"][0]["constraint"]["op"], "<=")

        with self.assertRaisesRegex(SemanticError, "Non-convex objective"):
            self.compiler._lower_minmax_objective(
                {"declarations": [], "constraints": []},
                {
                    "type": "minimize",
                    "expression": {"type": "min_agg", "expression": name("x"), "iterators": []},
                },
            )

    def test_minmax_objective_lowering_and_ground_conditional_evaluation(self):
        ast = {"declarations": [{"type": "dvar", "name": "x"}], "constraints": []}
        objective = {
            "type": "minimize",
            "expression": {"type": "maxl", "args": [name("x"), name("y")]},
        }

        self.compiler._lower_maxmin_objective(ast, objective)

        self.assertEqual(ast["declarations"][-1]["name"], objective["expression"]["value"])
        self.assertEqual(len(ast["constraints"]), 2)
        self.assertTrue(all(constraint["op"] == ">=" for constraint in ast["constraints"]))

        conditional = {
            "type": "conditional",
            "condition": {"type": "boolean_literal", "value": True},
            "then": number(7),
            "else": number(3),
        }
        self.assertEqual(self.compiler._eval_ground_expr(conditional, {}), 7)
        conditional["condition"]["value"] = False
        self.assertEqual(self.compiler._eval_ground_expr(conditional, {}), 3)


if __name__ == "__main__":
    unittest.main()

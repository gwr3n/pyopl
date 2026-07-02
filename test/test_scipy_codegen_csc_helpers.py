import unittest

from pyopl.pyopl_core import OPLLexer, OPLParser
from pyopl.scipy_codegen_csc import ExpressionEvaluator, SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


def make_generator(src: str = "dvar float x; minimize 0; subject to { }") -> SciPyCSCCodeGenerator:
    lexer = OPLLexer()
    parser = OPLParser()
    ast = parser.parse(lexer.tokenize(src))
    return SciPyCSCCodeGenerator(ast)


class TestScipyCSCExpressionEvaluatorHelpers(unittest.TestCase):
    def test_minl_maxl_tuple_and_conditional_literals(self) -> None:
        evaluator = ExpressionEvaluator(make_generator())

        self.assertEqual(
            evaluator.eval(
                {
                    "type": "minl",
                    "args": [
                        {"type": "number", "value": 3},
                        {"type": "boolean_literal", "value": True},
                        {"type": "string_literal", "value": "2.5"},
                    ],
                }
            ),
            ({}, 1.0),
        )
        self.assertEqual(
            evaluator.eval(
                {
                    "type": "maxl",
                    "args": [
                        {"type": "number", "value": 3},
                        {"type": "boolean_literal", "value": False},
                        {"type": "string_literal", "value": "2.5"},
                    ],
                }
            ),
            ({}, 3.0),
        )
        self.assertEqual(
            evaluator.eval(
                {
                    "type": "conditional",
                    "condition": {"type": "boolean_literal", "value": False},
                    "then": {"type": "number", "value": 1},
                    "else": {
                        "type": "tuple_literal",
                        "elements": [{"type": "number", "value": 2}, {"type": "string_literal", "value": "a"}],
                    },
                }
            ),
            ({}, (2.0, "a")),
        )

    def test_minl_maxl_reject_empty_nonnumeric_and_nonground(self) -> None:
        evaluator = ExpressionEvaluator(make_generator("dvar float x; minimize 0; subject to { }"))
        gen = evaluator.parent
        gen._build_variables()

        with self.assertRaisesRegex(SemanticError, "minl\(\) requires"):
            evaluator.eval({"type": "minl", "args": []})
        with self.assertRaisesRegex(SemanticError, "Non-numeric argument"):
            evaluator.eval({"type": "maxl", "args": [{"type": "string_literal", "value": "abc"}]})
        with self.assertRaisesRegex(SemanticError, "Non-ground argument"):
            evaluator.eval({"type": "minl", "args": [{"type": "name", "value": "x"}]})

    def test_binop_arithmetic_symbolic_and_error_paths(self) -> None:
        evaluator = ExpressionEvaluator(make_generator("dvar float x; dvar float y; minimize 0; subject to { }"))
        gen = evaluator.parent
        gen._build_variables()

        self.assertEqual(
            evaluator.eval(
                {
                    "type": "binop",
                    "op": "+",
                    "left": {"type": "string_literal", "value": "a"},
                    "right": {"type": "number", "value": 2},
                }
            ),
            ({}, "(a) + (2.0)"),
        )
        self.assertEqual(
            evaluator.eval(
                {"type": "binop", "op": "/", "left": {"type": "name", "value": "x"}, "right": {"type": "number", "value": 2}}
            ),
            ({"x": 0.5}, 0.0),
        )
        with self.assertRaisesRegex(SemanticError, "division by variable"):
            evaluator.eval(
                {"type": "binop", "op": "/", "left": {"type": "number", "value": 1}, "right": {"type": "name", "value": "x"}}
            )
        with self.assertRaisesRegex(SemanticError, "Division by zero"):
            evaluator.eval(
                {"type": "binop", "op": "/", "left": {"type": "number", "value": 1}, "right": {"type": "number", "value": 0}}
            )
        with self.assertRaisesRegex(SemanticError, "variable \* variable"):
            evaluator.eval(
                {"type": "binop", "op": "*", "left": {"type": "name", "value": "x"}, "right": {"type": "name", "value": "y"}}
            )

    def test_boolean_logic_ground_and_symbolic_gating(self) -> None:
        evaluator = ExpressionEvaluator(make_generator("dvar float x; minimize 0; subject to { }"))
        gen = evaluator.parent
        gen._build_variables()

        self.assertEqual(evaluator.eval({"type": "not", "value": {"type": "boolean_literal", "value": False}}), ({}, 1.0))
        self.assertEqual(
            evaluator.eval(
                {
                    "type": "and",
                    "left": {"type": "boolean_literal", "value": True},
                    "right": {"type": "boolean_literal", "value": False},
                }
            ),
            ({}, 0.0),
        )
        with self.assertRaisesRegex(SemanticError, "Non-ground boolean"):
            evaluator.eval({"type": "not", "value": {"type": "name", "value": "x"}})

        gen._allow_symbolic_bool = True
        self.assertEqual(evaluator.eval({"type": "not", "value": {"type": "name", "value": "x"}}), ({}, "!(0.0)"))

    def test_index_expr_and_tuple_field_helpers(self) -> None:
        gen = make_generator()
        gen.data_dict["N"] = "3"
        gen.tuple_types = {"Arc": [{"name": "i"}, {"name": "j"}]}
        gen.ast["declarations"].append({"type": "set_of_tuples", "name": "Arcs", "tuple_type": "Arc"})
        evaluator = ExpressionEvaluator(gen)

        self.assertEqual(evaluator._eval_index_expr({"type": "name_reference_index", "name": "N"}, {}), ({}, 3))
        self.assertEqual(
            evaluator._eval_index_expr(
                {
                    "type": "binop",
                    "op": "+",
                    "left": {"type": "number_literal_index", "value": 1},
                    "right": {"type": "number_literal_index", "value": 2},
                },
                {},
            ),
            ({}, 3),
        )
        self.assertEqual(evaluator._resolve_tuple_field_access_by_index({"value": "a"}, "j", (10, 20)), ({}, 20))


class TestScipyCSCGeneratorHelpers(unittest.TestCase):
    def test_aux_binary_and_flat_kv_helpers(self) -> None:
        gen = make_generator()

        first = gen._ensure_aux_binary("flag")
        second = gen._ensure_aux_binary("flag")

        self.assertEqual(first, "flag")
        self.assertEqual(second, "flag_1")
        self.assertEqual(gen._convert_flat_kv_to_dict(["a", 1, "b", 2.5]), {"a": 1, "b": 2.5})
        self.assertIsNone(gen._convert_flat_kv_to_dict(["a", "not-number"]))
        self.assertEqual(gen._make_constraint_row({"flag": 2.0, "missing": 9.0}), [2.0, 0.0])

    def test_bool_flatten_and_sum_inclusion_helpers(self) -> None:
        gen = make_generator()
        comp1 = {
            "type": "binop",
            "sem_type": "boolean",
            "op": "<=",
            "left": {"type": "number", "value": 1},
            "right": {"type": "number", "value": 2},
        }
        comp2 = {
            "type": "binop",
            "sem_type": "boolean",
            "op": "!=",
            "left": {"type": "number", "value": 1},
            "right": {"type": "number", "value": 3},
        }
        tree = {
            "type": "and",
            "left": comp1,
            "right": {"type": "and", "left": comp2, "right": {"type": "boolean_literal", "value": True}},
        }

        flat = gen._flatten_bool(tree, "and")
        env2, include = gen._should_include_sum_term(["i"], (1,), set(), {}, {"type": "boolean_literal", "value": False}, {})

        self.assertEqual(flat[:2], [comp1, comp2])
        self.assertEqual(env2, {"i": 1})
        self.assertFalse(include)

    def test_metadata_refresh_snapshot_and_zero_variable_codegen(self) -> None:
        gen = make_generator("minimize 5; subject to { }")

        code = gen.generate_code()
        problem = gen._snapshot_linear_problem()

        self.assertEqual(problem.var_names, [])
        self.assertIn("# No decision variables: short-circuit without linprog", code)
        self.assertIn("results['objective_value'] = 5.0", code)

        gen2 = make_generator()
        gen2.var_names = ["x", "y"]
        gen2.bounds = [[0, 1]]
        gen2.integrality = []
        gen2.c = [3.0, 4.0, 5.0]
        gen2.scipy_code_lines = ["var_names = []", "bounds = []", "integrality = []", "c = []"]

        gen2._reconcile_problem_metadata()
        gen2._refresh_problem_metadata_code_lines()

        self.assertEqual(gen2.bounds, [[0, 1], [0, 1]])
        self.assertEqual(gen2.integrality, [1, 1])
        self.assertEqual(gen2.c, [3.0, 4.0])
        self.assertIn("var_names = ['x', 'y']", gen2.scipy_code_lines)
        self.assertIn("integrality = [1, 1]", gen2.scipy_code_lines)

    def test_build_problem_and_top_level_binary_assignment(self) -> None:
        gen = make_generator("dvar boolean b; minimize 0; subject to { b == 1; }")

        problem = gen.build_problem()

        self.assertEqual(problem.var_names, ["b"])
        self.assertEqual(problem.bounds, [[1.0, 1.0]])
        self.assertEqual(problem.integrality, [1])
        self.assertTrue(any(row[gen.var_indices["b"]] == 1.0 and rhs == 1.0 for row, rhs in zip(gen.A_eq, gen.b_eq)))

    def test_resolve_variable_parameter_and_index_helpers(self) -> None:
        ast = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "T",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 3},
                },
                {"type": "dvar_indexed", "name": "x", "dimensions": [{"type": "named_range_dimension", "name": "T"}]},
                {"type": "parameter_inline", "name": "p", "value": 7},
                {
                    "type": "parameter_inline_indexed",
                    "name": "q",
                    "dimensions": [{"type": "named_range_dimension", "name": "T"}],
                    "value": [10, 20, 30],
                },
            ],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
            "constraints": [],
        }
        gen = SciPyCSCCodeGenerator(ast, {"r": ["a", 1, "b", 2]})
        gen.var_indices = {"x_1": 0, "x_2": 1, "x_3": 2}

        self.assertEqual(gen._lookup_var_or_param("x", [1]), (True, "x_1", False))
        self.assertEqual(gen._lookup_var_or_param("r", ["a"]), (False, 1.0, False))
        self.assertEqual(gen._lookup_var_or_param("p"), (False, 7.0, False))
        self.assertEqual(gen._lookup_var_or_param("q", [2]), (False, 20.0, False))
        self.assertEqual(gen._lookup_var_or_param("missing", ["i"], default_zero_if_missing=True), (False, 0.0, False))
        self.assertEqual(gen._eval_index("N + 2", {"N": 3}), 5)
        self.assertEqual(SciPyCSCCodeGenerator.normalize_index([["a", 1], 2]), (("a", 1), 2))

        with self.assertRaisesRegex(SemanticError, "out of declared domain"):
            gen._lookup_var_or_param("x", [9])

    def test_data_declaration_set_range_and_set_set_normalization(self) -> None:
        ast = {
            "declarations": [
                {"type": "typed_set", "base_type": "string", "name": "Products", "value": ["P1", "P2"]},
                {
                    "type": "range_declaration_inline",
                    "name": "T",
                    "start": {"type": "number", "value": 2},
                    "end": {"type": "number", "value": 3},
                },
                {
                    "type": "parameter_external_indexed",
                    "name": "demand",
                    "dimensions": [
                        {"type": "named_set_dimension", "name": "Products"},
                        {
                            "type": "named_range_dimension",
                            "name": "T",
                            "start": {"type": "number", "value": 2},
                            "end": {"type": "number", "value": 3},
                        },
                    ],
                },
                {
                    "type": "parameter_external_indexed",
                    "name": "grid",
                    "dimensions": [
                        {"type": "named_set_dimension", "name": "Products"},
                        {"type": "named_set_dimension", "name": "Products"},
                    ],
                },
            ]
        }
        data = {"demand": {"P1": [1, 2], "P2": [3, 4]}, "grid": {"P1": [5, 6], "P2": [7, 8]}}
        gen = SciPyCSCCodeGenerator(ast, data)

        gen._generate_data_declarations(data)

        self.assertEqual(data["demand"], {"P1": {2: 1.0, 3: 2.0}, "P2": {2: 3.0, 3: 4.0}})
        self.assertEqual(data["grid"], {"P1": {"P1": 5.0, "P2": 6.0}, "P2": {"P1": 7.0, "P2": 8.0}})
        self.assertIn("Products = ['P1', 'P2']", gen.scipy_code_lines)

    def test_tuple_data_declarations_and_inline_tuple_parameter(self) -> None:
        ast = {
            "declarations": [
                {"type": "tuple_type", "name": "Arc", "fields": [{"name": "i"}, {"name": "j"}]},
                {
                    "type": "set_of_tuples",
                    "name": "Arcs",
                    "tuple_type": "Arc",
                    "value": [{"elements": ["A", "B"]}, {"elements": ["B", "C"]}],
                },
                {
                    "type": "range_declaration_inline",
                    "name": "T",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 2},
                },
                {
                    "type": "parameter_inline_indexed",
                    "name": "cost",
                    "dimensions": [{"type": "named_set_dimension", "name": "Arcs"}],
                    "value": [5, 7],
                },
                {
                    "type": "parameter_external_indexed",
                    "name": "cap",
                    "dimensions": [
                        {"type": "named_set_dimension", "name": "Arcs"},
                        {"type": "named_range_dimension", "name": "T"},
                    ],
                },
            ]
        }
        data = {"cap": [[1, 2], [3, 4]]}
        gen = SciPyCSCCodeGenerator(ast, data)

        gen._generate_data_declarations(data)
        code = "\n".join(gen.scipy_code_lines)

        self.assertIn("cost = {('A', 'B'): 5", code)
        self.assertEqual(data["cap"], {("A", "B"): {1: 1.0, 2: 2.0}, ("B", "C"): {1: 3.0, 2: 4.0}})

    def test_parameter_resolution_tuple_keys_pairs_and_non_numeric_indexes(self) -> None:
        ast = {
            "declarations": [
                {"type": "parameter_inline_indexed", "name": "inline", "dimensions": [], "value": {"A": "label"}},
            ],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
            "constraints": [],
        }
        gen = SciPyCSCCodeGenerator(
            ast,
            {
                "tuple_param": {("RoleA", "CoreSite"): "chosen"},
                "pair_param": [[("A", "B"), 9], ["C", 4]],
                "nested": [[10, 20], [30, 40]],
            },
        )

        self.assertEqual(gen._lookup_var_or_param("tuple_param", ["RoleA", "CoreSite"]), (False, "chosen", False))
        self.assertEqual(gen._lookup_var_or_param("pair_param", [("A", "B")]), (False, 9.0, False))
        with self.assertRaisesRegex(SemanticError, "Parameter or variable 'nested'"):
            gen._resolve_parameter("nested", ["bad"], {}, default_zero_if_missing=False)
        self.assertEqual(gen._eval_index("(1, 2)", {}), (1, 2))
        self.assertEqual(gen._eval_index("N // 2", {"N": 5}), 2)
        self.assertEqual(gen._eval_index("unknown_label", {}), "unknown_label")

        with self.assertRaisesRegex(SemanticError, "AST parameter 'missing'"):
            gen._resolve_ast_parameter("missing", None)

    def test_linear_bounds_and_tuple_index_varname_helpers(self) -> None:
        gen = make_generator("dvar float x; minimize 0; subject to { }")
        gen._build_variables()
        gen._collected_lbs = {"x": 2.0}
        gen._collected_ubs = {"x": 5.0}

        self.assertEqual(gen._linear_bounds_safe({"type": "name", "value": "x"}), (2.0, 5.0))
        gen.bounds[gen.var_indices["x"]] = [1.0, 4.0]
        self.assertEqual(
            gen._linear_bounds_safe(
                {"type": "binop", "op": "*", "left": {"type": "number", "value": -2}, "right": {"type": "name", "value": "x"}}
            ),
            (None, None),
        )
        self.assertEqual(
            gen._linear_bounds_safe(
                {"type": "binop", "op": "+", "left": {"type": "name", "value": "x"}, "right": {"type": "number", "value": 3}}
            ),
            (5.0, 8.0),
        )

        gen.var_indices["arc[('A', 'B')]"] = 3
        self.assertEqual(gen._resolve_tuple_index_varname("arc[('A', 'B')]"), 3)
        with self.assertRaisesRegex(SemanticError, "Variable 'arc\[missing\]'"):
            gen._resolve_tuple_index_varname("arc[missing]")


if __name__ == "__main__":
    unittest.main()

import unittest

from pyopl.gurobi_codegen import GurobiCodeGenerator
from pyopl.pyopl_core import OPLCompiler, SemanticError, parse_model
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


def num(value):
    return {"type": "number", "value": value}


def name(value):
    return {"type": "name", "value": value}


def op(left, operator, right):
    return {"type": "binop", "left": left, "op": operator, "right": right}


def rng(start, end):
    return {"type": "range_specifier", "start": num(start), "end": num(end)}


class TestGurobiBoundsAndRanges(unittest.TestCase):
    def setUp(self):
        declarations = [
            {"type": "range_declaration_inline", "name": "I", "start": num(2), "end": num(4)},
            {"type": "typed_set", "name": "S", "value": [1, 2]},
            {
                "type": "parameter_external_indexed",
                "name": "p",
                "dimensions": [{"type": "range_index", "start": num(2), "end": num(4)}],
            },
            {"type": "dvar", "name": "x", "var_type": "float+"},
            {"type": "dvar", "name": "b", "var_type": "boolean"},
        ]
        self.gen = GurobiCodeGenerator({"declarations": declarations, "constraints": []}, {"p": [7, 9, 11], "S": [1, 2]})

    def test_indexed_and_arithmetic_declaration_bounds(self):
        indexed = {"type": "indexed_name", "name": "p", "dimensions": [name("i")]}
        unary = {"type": "uminus", "value": indexed}
        self.assertEqual(self.gen._evaluate_declaration_bound(indexed, {"i": 3}), 9)
        self.assertEqual(self.gen._evaluate_declaration_bound(unary, {"i": 3}), -9)
        self.assertEqual(self.gen._evaluate_declaration_bound(op(indexed, "+", num(2)), {"i": 4}), 13)
        self.assertIsNone(self.gen._evaluate_declaration_bound(indexed, {"i": 10}))
        self.assertIsNone(self.gen._evaluate_declaration_bound(indexed, {}))
        self.assertIsNone(self.gen._evaluate_declaration_bound(op(num(1), "/", num(0)), {}))
        self.assertIsNone(self.gen._evaluate_declaration_bound(op(num(1), "^", num(2)), {}))

    def test_bound_envelope_from_iterator_and_unresolved_data(self):
        indexed = {"type": "indexed_name", "name": "p", "dimensions": [name("i")]}
        declaration = {
            "lower_bound": indexed,
            "upper_bound": op(indexed, "+", num(3)),
            "iterators": [{"iterator": "i", "range": {"type": "named_range", "name": "I"}}],
        }
        self.assertEqual(self.gen._explicit_declaration_bounds(declaration), (7, 14))
        declaration["iterators"][0]["range"]["name"] = "missing"
        self.assertEqual(self.gen._explicit_declaration_bounds(declaration), (None, None))
        self.assertEqual(self.gen._var_bounds_safe(name("x")), (0.0, None))
        self.assertEqual(self.gen._var_bounds_safe(name("b")), (0.0, 1.0))
        self.assertEqual(self.gen._var_bounds_safe(name("missing")), (None, None))

    def test_bound_collection_handles_reversed_and_nested_constraints(self):
        def constraint(left, operator, right):
            return {"type": "constraint", "left": left, "op": operator, "right": right}

        self.gen._collect_variable_bounds(
            [
                constraint(num(5), ">=", name("x")),
                {"type": "forall_constraint", "constraint": constraint(name("x"), ">=", num(1))},
            ]
        )
        self.assertEqual(self.gen._linear_bounds_safe(name("x")), (1, 5))
        self.assertEqual(self.gen._linear_bounds_safe(op(name("x"), "*", num(-2))), (-10, -2))
        self.assertEqual(self.gen._linear_bounds_safe(op(num(8), "-", name("x"))), (3, 7))
        self.assertIsNone(self.gen._extract_variable_bound(constraint(name("x"), ">=", num("invalid"))))

    def test_range_expressions_resolve_named_sets_and_reject_missing_domains(self):
        for emit in (self.gen._forall_range_expr, self.gen._sum_iterator_range):
            with self.subTest(method=emit.__name__):
                self.assertEqual(emit(rng(1, 3), {}), "range(1, 3 + 1)")
                self.assertEqual(emit({"type": "named_range", "name": "I"}, {}), "range(2, 4 + 1)")
                self.assertEqual(emit({"type": "named_set", "name": "S"}, {}), "S")
                with self.assertRaisesRegex(ValueError, "not found"):
                    emit({"type": "named_range", "name": "missing"}, {})
                with self.assertRaisesRegex(ValueError, "Unsupported range type"):
                    emit({"type": "unsupported"}, {})

    def test_index_expression_preserves_tuple_and_unary_structure(self):
        emit = self.gen._emit_index_expr
        index = {"type": "name_reference_index", "name": "i"}
        self.assertEqual(emit(op(index, "+", num(2)), {}, True), "(i + 2)")
        self.assertEqual(emit({"type": "uminus", "value": index}, {}, True), "-(i)")
        self.assertEqual(emit({"type": "parenthesized_expression", "expression": index}, {}, True), "(i)")
        self.assertEqual(emit({"type": "tuple_literal", "elements": [name("i")]}, {}, True), "(i,)")
        with self.assertRaisesRegex(ValueError, "Unsupported index"):
            emit({"type": "unrecognized"}, {}, True)

    def test_integer_comparison_indicators_reify_each_order_relation(self):
        expected = {
            ">=": (">= 0", "<= -1"),
            ">": (">= 1", "<= 0"),
            "<=": ("<= 0", ">= 1"),
            "<": ("<= -1", ">= 0"),
        }
        for operator, fragments in expected.items():
            with self.subTest(operator=operator):
                self.gen.gurobi_code_lines.clear()
                comparison = op(num(2), operator, num(3))
                self.gen._bind_implication_comparison_to_binary("flag", comparison, {}, "ct")
                code = "\n".join(self.gen.gurobi_code_lines)
                self.assertEqual(code.count("model.addGenConstrIndicator"), 2)
                for fragment in fragments:
                    self.assertIn(fragment, code)
        with self.assertRaisesRegex(SemanticError, "Continuous comparison truth"):
            self.gen._bind_implication_comparison_to_binary("flag", op(num(1.5), ">=", num(3)), {}, "ct")

    def test_strict_antecedent_uses_contrapositive_binary_indicator(self):
        emit = self.gen._emit_true_consequent_indicator
        args = (name("x"), num(3), ">", "x", "3", name("b"), "b", num(1), "1", "==", "ct")
        self.assertTrue(emit(*args))
        self.assertIn("b, 0, x <= 3", "\n".join(self.gen.gurobi_code_lines))
        self.gen.gurobi_code_lines.clear()
        args = (name("x"), num(3), "<", "x", "3", num(0), "0", name("b"), "b", "==", "ct")
        self.assertTrue(emit(*args))
        self.assertIn("b, 1, x >= 3", "\n".join(self.gen.gurobi_code_lines))
        invalid = (name("x"), num(3), ">", "x", "3", name("b"), "b", num(2), "2", "==", "ct")
        self.assertFalse(emit(*invalid))

    def test_data_shape_checks_reject_scalar_and_wrong_lengths(self):
        dim = [{"type": "named_range_dimension", "name": "I"}]
        self.gen._check_parameter_shape([1, 2, 3], dim, {}, "p")
        with self.assertRaisesRegex(SemanticError, "got scalar"):
            self.gen._check_parameter_shape(1, dim, {}, "p")
        with self.assertRaisesRegex(SemanticError, "data length 2"):
            self.gen._check_parameter_shape([1, 2], dim, {}, "p")

    def test_boolean_composites_emit_linking_constraints(self):
        true = {"type": "boolean_literal", "value": True}
        false = {"type": "boolean_literal", "value": False}
        cases = [
            ({"type": "not", "value": true}, 1),
            ({"type": "and", "left": true, "right": false}, 3),
            ({"type": "or", "left": true, "right": false}, 3),
            ({"type": "implies", "left": true, "right": false}, 4),
        ]
        for expression, expected_constraints in cases:
            with self.subTest(expression=expression):
                self.gen.gurobi_code_lines.clear()
                variable = self.gen._boolean_expr_to_binary_expr(expression, {}, "ct")
                code = "\n".join(self.gen.gurobi_code_lines)
                self.assertIn(f"{variable} = model.addVar", code)
                self.assertEqual(code.count("model.addConstr"), expected_constraints)
                compile(code, "<generated boolean constraints>", "exec")
        with self.assertRaisesRegex(ValueError, "Unsupported boolean expression"):
            self.gen._boolean_expr_to_binary_expr({"type": "unknown"}, {}, "ct")

    def test_parameter_data_maps_range_and_set_indices(self):
        keys = self.gen._dimension_keys(
            [
                {"type": "range_index", "start": num(2), "end": num(3)},
                {"type": "named_set_dimension", "name": "S"},
            ],
            {"S": {"elements": ["A", "B"]}},
        )
        self.assertEqual(keys, [[2, 3], ["A", "B"]])
        self.assertEqual(
            self.gen._flatten_positional_parameter("p", [[1, 2], [3, 4]], keys),
            {(2, "A"): 1, (2, "B"): 2, (3, "A"): 3, (3, "B"): 4},
        )
        self.assertIsNone(self.gen._dimension_keys([{"type": "unknown"}], {}))
        declaration = {"dimensions": [{"type": "named_set_dimension", "name": "S"}]}
        self.assertTrue(self.gen._emit_1d_set_parameter("weights", [10, 20], declaration, {"S": ["A", "B"]}))
        self.assertIn("weights = {'A': 10, 'B': 20}", "\n".join(self.gen.gurobi_code_lines))
        self.assertFalse(self.gen._emit_1d_set_parameter("weights", [10], declaration, {"S": ["A", "B"]}))

    def test_parameter_indexing_renders_named_range_and_set_offsets(self):
        self.assertEqual(
            self.gen._emit_parameter_list_index({"type": "named_range_dimension", "name": "I"}, "i"),
            "((i) - (2))",
        )
        declaration = {"dimensions": [{"type": "named_set_dimension", "name": "S"}]}
        index = {"type": "name_reference_index", "name": "i"}
        expression = {"type": "indexed_name", "name": "weights", "dimensions": [index]}
        self.assertEqual(
            self.gen._emit_parameter_indexed_name("weights", expression, declaration, {}, True), "weights[S_index[i]]"
        )

    def test_comparison_reification_generates_rows_for_each_operator(self):
        expected = {">=": 2, ">": 2, "<=": 2, "<": 2, "==": 4, "!=": 4}
        for operator, row_count in expected.items():
            with self.subTest(operator=operator):
                code = self.gen._emit_reify_comparison(num(2), num(3), "2", "3", operator, "flag")
                self.assertIn(f"Reify (2 {operator} 3)", code)
                self.assertEqual(code.count("model.addConstr("), row_count)
                if operator == "!=":
                    self.assertIn("flag_neq_side = model.addVar", code)
                compile(code, "<generated reification>", "exec")

    def test_matching_parameter_set_elements_and_missing_set(self):
        self.assertEqual(self.gen._matching_1d_set_elements("S", [2, 4], {"S": {"elements": ["A", "B"]}}), ["A", "B"])
        self.assertIsNone(self.gen._matching_1d_set_elements("missing", [2], {}))
        self.assertIsNone(self.gen._matching_1d_set_elements("missing", [2], {"missing": ["A", "B"]}))

    def test_set_declarations_require_supplied_data(self):
        self.gen._decl_set_declaration({"name": "S"})
        self.assertIn("S = [1, 2]", "\n".join(self.gen.gurobi_code_lines))
        with self.assertRaisesRegex(SemanticError, "not found in .dat"):
            self.gen._decl_set_declaration({"name": "missing"})
        with self.assertRaisesRegex(SemanticError, "declared as external"):
            self.gen._decl_range_declaration_external({"name": "I"})

    def test_range_data_bounds_calculate_and_reject_unknown_operators(self):
        self.assertEqual(self.gen._eval_data_bound(op(name("n"), "/", num(2)), {"n": 5}), 2)
        with self.assertRaisesRegex(Exception, "Unsupported binop"):
            self.gen._eval_data_bound(op(num(1), "%", num(2)), {})
        with self.assertRaisesRegex(Exception, "Unsupported range bound"):
            self.gen._eval_data_bound({"type": "unknown"}, {})


class TestCompilerDataValidation(unittest.TestCase):
    def setUp(self):
        self.compiler = OPLCompiler()

    def test_tuple_validation_handles_nested_records_and_errors(self):
        types = {
            "Point": [{"name": "x", "type": "int+"}, {"name": "y", "type": "float"}],
            "Record": [{"name": "point", "type": "Point"}, {"name": "label", "type": "string"}],
        }
        self.compiler._validate_tuple_value({"point": [2, 3.5], "label": "a"}, "Record", types, "r")
        for value, expected in [
            ({"point": [2, 3], "extra": 1}, "missing fields"),
            ({"point": [-1, 3], "label": "a"}, "nonnegative"),
            ({"point": [True, 3], "label": "a"}, "expects type"),
            ({"point": [1], "label": "a"}, "fields; expected"),
            (1, "must be a tuple or record"),
        ]:
            with self.subTest(value=value), self.assertRaisesRegex(SemanticError, expected):
                self.compiler._validate_tuple_value(value, "Record", types, "r")
        with self.assertRaisesRegex(SemanticError, "Unknown tuple type"):
            self.compiler._validate_tuple_value([], "missing", types, "r")

    def test_tuple_values_evaluate_numeric_expressions(self):
        evaluate = self.compiler._tuple_value_elements
        self.assertEqual(evaluate(op(num(9), "-", num(4))), 5)
        self.assertEqual(evaluate(op(num(9), "/", num(3))), 3)
        self.assertEqual(evaluate({"type": "uminus", "value": num(2)}), -2)
        self.assertEqual(evaluate({"type": "parenthesized_expression", "expression": num(3)}), 3)
        unsupported = op(num(1), "^", num(2))
        self.assertEqual(evaluate(unsupported), unsupported)

    def test_parameter_indices_respect_ranges_sets_and_invalid_accesses(self):
        ast = {
            "declarations": [
                {"name": "a", "dimensions": [{"type": "range_index", "start": num(2), "end": num(4)}]},
                {"name": "b", "dimensions": [{"type": "named_set_dimension", "name": "S"}]},
                {"name": "c", "dimensions": []},
            ]
        }
        position = self.compiler._parameter_list_position
        self.assertEqual(position(ast, {}, "a", 0, 3), 1)
        self.assertEqual(position(ast, {"S": {"elements": ["A", "B"]}}, "b", 0, "B"), 1)
        self.assertEqual(position(ast, {}, "c", 0, 2, 3), 1)
        for parameter, index, dimension in [("a", 1, 0), ("b", "C", 0), ("c", 0, 0), ("a", 1, 1)]:
            with self.subTest(parameter=parameter, index=index), self.assertRaises(SemanticError):
                position(ast, {"S": ["A", "B"]}, parameter, dimension, index, 3)

    def test_comprehension_expression_resolves_index_and_boolean_operations(self):
        evaluate = self.compiler._eval_comprehension_expr
        ast = {"declarations": [{"name": "a", "dimensions": []}]}
        data = {"a": [10, 20], "limit": 15}
        indexed = {"type": "indexed_name", "name": "a", "dimensions": [name("i")]}
        condition = {"type": "and", "left": op(indexed, ">", name("limit")), "right": {"type": "not", "value": num(0)}}
        self.assertFalse(evaluate(condition, {"i": 1}, data, ast))
        self.assertTrue(evaluate(condition, {"i": 2}, data, ast))
        with self.assertRaisesRegex(SemanticError, "Non-integer list index"):
            evaluate(indexed, {"i": "bad"}, data, ast)
        with self.assertRaisesRegex(SemanticError, "Unknown name"):
            evaluate(name("missing"), {}, data, ast)

    def test_computed_parameter_indices_validate_operators_and_operands(self):
        evaluate = self.compiler._eval_computed_parameter_index
        self.assertEqual(evaluate(op(name("i"), "*", num(2)), {"i": 3}), 6)
        self.assertEqual(evaluate({"type": "uminus", "value": name("i")}, {"i": 3}), -3)
        for expression, message in [
            (op(num(1), "/", num(2)), "Unsupported index binop"),
            (op(None, "+", num(2)), "Unsupported index binop operands"),
            ({"type": "uminus"}, "missing unary operand"),
            ({"type": "field_access"}, "Cannot resolve tuple field"),
        ]:
            with self.subTest(expression=expression), self.assertRaisesRegex(SemanticError, message):
                evaluate(expression, {})

    def test_tuple_schema_checks_nested_arrays_and_invalid_types(self):
        point = {"type": "tuple_type", "name": "Point", "fields": [{"name": "x", "type": "int"}]}
        array = {
            "type": "tuple_array_external",
            "name": "a",
            "tuple_type": "Point",
            "dimensions": [{"type": "named_set_dimension", "name": "S"}],
        }
        ast = {"declarations": [point, array]}
        self.compiler._validate_tuple_schemas(ast, {"a": [(1,), (2,)], "S": [1, 2]})
        with self.assertRaisesRegex(SemanticError, "expected 2"):
            self.compiler._validate_tuple_schemas(ast, {"a": [(1,)], "S": [1, 2]})
        with self.assertRaisesRegex(SemanticError, "sequence or mapping"):
            self.compiler._validate_tuple_schemas(ast, {"a": 5})
        with self.assertRaisesRegex(SemanticError, "unknown tuple type"):
            self.compiler._validate_tuple_schemas(
                {"declarations": [{"type": "tuple_type", "name": "Bad", "fields": [{"name": "x", "type": "Missing"}]}]},
                {},
            )
        with self.assertRaisesRegex(SemanticError, "duplicate fields"):
            self.compiler._validate_tuple_schemas(
                {"declarations": [{"type": "tuple_type", "name": "Bad", "fields": point["fields"] * 2}]}, {}
            )

    def test_computed_parameter_maps_keep_declared_index_domains(self):
        ast = {
            "declarations": [
                {"type": "range_declaration_inline", "name": "I", "start": num(2), "end": num(3)},
                {
                    "type": "parameter_inline_indexed",
                    "name": "p",
                    "dimensions": [
                        {"type": "named_range_dimension", "name": "I"},
                        {"type": "named_set_dimension", "name": "S"},
                    ],
                },
            ]
        }
        data = {"p": [[10, 11], [20, 21]], "S": ["A", "B"]}
        self.compiler._build_computed_parameter_maps(ast, data)
        self.assertEqual(data["p__map"], {(2, "A"): 10, (2, "B"): 11, (3, "A"): 20, (3, "B"): 21})
        with self.assertRaisesRegex(SemanticError, "shorter than domain"):
            self.compiler._flatten_computed_parameter_map("p", [[1, 2]], [5], {}, 0, [])

    def test_comprehensions_materialize_filtered_tuple_and_scalar_sets(self):
        iterator = {"iterator": "i", "range": rng(1, 3)}
        guard = op(name("i"), ">", num(1))
        tuple_declaration = {
            "type": "set_of_tuples_comprehension",
            "name": "pairs",
            "tuple_type": "Pair",
            "comprehension": {
                "iterators": [iterator],
                "index_constraint": guard,
                "tuple_expr": {"type": "tuple_literal", "elements": [name("i"), op(name("i"), "*", num(2))]},
            },
        }
        scalar_declaration = {
            "type": "typed_set_comprehension",
            "name": "values",
            "base_type": "int",
            "comprehension": {"iterators": [iterator], "index_constraint": guard, "expression": op(name("i"), "+", num(1))},
        }
        ast = {"declarations": [tuple_declaration, scalar_declaration]}
        data = {}
        self.compiler._materialize_set_of_tuples_comprehensions(ast, data)
        self.compiler._materialize_typed_set_comprehensions(ast, data)
        self.assertEqual(data["pairs"], [(2, 4), (3, 6)])
        self.assertEqual(data["values"], [3, 4])
        self.assertEqual([decl["type"] for decl in ast["declarations"]], ["set_of_tuples", "typed_set"])

    def test_typed_sets_validate_values_and_normalize_floats(self):
        declarations = [
            {"type": "typed_set_external", "name": "weights", "base_type": "float"},
            {"type": "typed_set_external", "name": "flags", "base_type": "boolean"},
        ]
        data = {"weights": {"elements": [1, 2.5]}, "flags": [True, False]}
        self.compiler._validate_typed_sets({"declarations": declarations}, data)
        self.assertEqual(data["weights"], [1.0, 2.5])
        for base, values, message in [
            ("int", [True], "integers"),
            ("float", [False], "numeric"),
            ("boolean", [1], "booleans"),
            ("string", [2], "strings"),
            ("int", (1, 2), "list of values"),
        ]:
            with self.subTest(base=base), self.assertRaisesRegex(SemanticError, message):
                self.compiler._validate_typed_set_values("s", base, values)

    def test_index_ranges_must_be_declared_in_model(self):
        ast = {
            "declarations": [
                {"type": "dvar_indexed", "name": "x", "dimensions": [{"type": "named_range_dimension", "name": "I"}]}
            ],
            "objective": None,
            "constraints": [],
        }
        with self.assertRaisesRegex(SemanticError, "not declared"):
            self.compiler._validate_named_ranges(ast, {})
        with self.assertRaisesRegex(SemanticError, "supplied in the data file"):
            self.compiler._validate_named_ranges(ast, {"I": {"type": "range_data"}})
        ast["declarations"].append({"type": "range_declaration_inline", "name": "I", "start": num(1), "end": num(2)})
        self.compiler._validate_named_ranges(ast, {})

    def test_boolean_equality_simplification_preserves_comparison_truth(self):
        comparison = {**op(name("x"), "<=", num(3)), "sem_type": "boolean"}
        constraint = {
            "type": "constraint",
            "op": "==",
            "left": comparison,
            "right": {"type": "boolean_literal", "value": False},
        }
        simplified = self.compiler._simplify_boolean_equality(constraint, {}, {"x"})
        self.assertEqual(simplified[0]["op"], ">")
        constraint["right"]["value"] = True
        self.assertEqual(self.compiler._simplify_boolean_equality(constraint, {}, {"x"})[0]["op"], "<=")
        constraint["left"] = {"type": "boolean_literal", "value": True}
        self.assertEqual(self.compiler._simplify_boolean_equality(constraint, {}, {"x"}), [])
        constraint["right"]["value"] = False
        contradiction = self.compiler._simplify_boolean_equality(constraint, {}, {"x"})
        self.assertEqual((contradiction[0]["left"]["value"], contradiction[0]["right"]["value"]), (0, 1))

    def test_parameter_shape_rejects_non_scalar_keyed_values(self):
        dimension = {"type": "named_set_dimension", "name": "S"}
        self.compiler._validate_parameter_shape({"A": 1, "B": 2}, [dimension], "p", {"declarations": []}, {})
        with self.assertRaisesRegex(SemanticError, "expected a scalar"):
            self.compiler._validate_parameter_shape({"A": [1]}, [dimension], "p", {"declarations": []}, {})

    def test_ground_condition_evaluates_nested_index_and_rejects_bad_inputs(self):
        condition = {
            "type": "and",
            "left": op({"type": "indexed_name", "name": "a", "dimensions": [num(1)]}, "==", num(4)),
            "right": {"type": "not", "value": op(name("n"), "<", num(2))},
        }
        self.assertTrue(self.compiler._eval_ground_condition(condition, {"a": [3, 4], "n": 2}))
        self.assertFalse(self.compiler._eval_ground_condition(condition, {"a": [3, 4], "n": 1}))
        for expression, message in [
            (name("missing"), "Unknown symbol"),
            ({"type": "indexed_name", "name": "a", "dimensions": [num(9)]}, "Index error"),
            (op(num(1), "/", num(0)), "Error evaluating ground binop"),
            (op(num(1), "unknown", num(2)), "Unsupported operator"),
            ({"type": "string_literal", "value": "a"}, "does not evaluate to boolean"),
        ]:
            with self.subTest(expression=expression), self.assertRaisesRegex(SemanticError, message):
                self.compiler._eval_ground_condition(expression, {"a": [1]})

    def test_if_inside_forall_rewrites_to_filtered_then_and_else(self):
        then = {"type": "constraint", "op": ">=", "left": name("x"), "right": num(1)}
        otherwise = {"type": "constraint", "op": "<=", "left": name("x"), "right": num(2)}
        forall = {
            "type": "forall_constraint",
            "iterators": [{"iterator": "i", "range": rng(1, 3)}],
            "constraint": {
                "type": "if_constraint",
                "condition": op(name("i"), ">", num(1)),
                "then_constraints": [then],
                "else_constraints": [otherwise],
            },
        }
        result = self.compiler._rewrite_forall_conditions(forall, {"x"})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["constraint"], then)
        self.assertEqual(result[1]["constraint"], otherwise)
        self.assertEqual(result[1]["index_constraint"]["type"], "not")
        forall["constraint"]["condition"] = op(name("x"), ">", num(1))
        with self.assertRaisesRegex(SemanticError, "must not reference decision variables"):
            self.compiler._rewrite_forall_conditions(forall, {"x"})

    def test_convex_maxmin_constraints_expand_and_nonconvex_forms_fail(self):
        maximum = {"type": "maxl", "args": [name("x"), name("y")]}
        minimum = {"type": "minl", "args": [name("x"), name("y")]}
        for aggregate, relation, aggregate_on_left in [
            (maximum, "<=", True),
            (maximum, ">=", False),
            (minimum, ">=", True),
            (minimum, "<=", False),
        ]:
            with self.subTest(aggregate=aggregate["type"], relation=relation, left=aggregate_on_left):
                left, right = (aggregate, num(5)) if aggregate_on_left else (num(5), aggregate)
                result = self.compiler._expand_maxmin_constraint(
                    {"type": "constraint", "op": relation, "left": left, "right": right, "label": "limit"}
                )
                self.assertEqual(len(result), 2)
                self.assertTrue(all(item["label"] == "limit" and item["op"] == relation for item in result))
        with self.assertRaisesRegex(SemanticError, "Non-convex"):
            self.compiler._expand_maxmin_constraint({"type": "constraint", "op": ">=", "left": maximum, "right": num(5)})


class TestScipyDomainAndShapeValidation(unittest.TestCase):
    def setUp(self):
        ast = {
            "declarations": [
                {"type": "range_declaration_inline", "name": "I", "start": num(2), "end": num(4)},
                {"type": "typed_set", "name": "S", "value": ["A", "B"]},
                {"type": "typed_set_external", "name": "Missing", "value": None},
            ],
            "objective": {"type": "minimize", "expression": num(0)},
            "constraints": [],
        }
        self.gen = SciPyCSCCodeGenerator(ast, {"groups": {"A": [2, 3]}, "bound": 4})

    def test_static_and_dynamic_bounds_reject_unsupported_values(self):
        self.assertEqual(self.gen._eval_bound(op(name("bound"), "*", num(2))), 8)
        self.assertEqual(self.gen._eval_bound_dynamic(op(name("i"), "-", num(1)), {"i": 4}), 3)
        for expression in [name("missing"), {"type": "minl", "args": []}, op(num(1), "/", num(2))]:
            with self.subTest(expression=expression), self.assertRaises(SemanticError):
                self.gen._eval_bound(expression)

    def test_iterator_domains_support_sets_ranges_and_indexed_sets(self):
        domain = self.gen._uncached_iterator_domain_dynamic
        self.assertEqual(domain({"range": rng(3, 2)}, {}), [])
        self.assertEqual(domain({"range": {"type": "named_range", "name": "I"}}, {}), [2, 3, 4])
        self.assertEqual(domain({"range": {"type": "named_set", "name": "S"}}, {}), ["A", "B"])
        indexed = {"type": "indexed_set", "name": "groups", "dimensions": [name("i")]}
        self.assertEqual(domain({"range": indexed}, {"i": "A"}), [2, 3])
        for domain_node in [
            {"type": "named_set", "name": "Missing"},
            {"type": "named_set", "name": "Unknown"},
            {"type": "named_range", "name": "Unknown"},
            {"type": "unknown"},
        ]:
            with self.subTest(domain=domain_node), self.assertRaises(SemanticError):
                domain({"range": domain_node}, {})

    def test_parameter_shapes_check_ranges_sets_and_rectangular_rows(self):
        validate = self.gen._validate_parameter_shape
        range_dim = [{"type": "named_range_dimension", "name": "I"}]
        set_dim = [{"type": "named_set_dimension", "name": "S"}]
        validate([1, 2, 3], range_dim, {}, "a")
        validate([1, 2], set_dim, {}, "b")
        with self.assertRaisesRegex(SemanticError, "expects 3"):
            validate([1, 2], range_dim, {}, "a")
        with self.assertRaisesRegex(SemanticError, "has 2 elements"):
            validate([1], set_dim, {}, "b")
        with self.assertRaisesRegex(SemanticError, "rectangular"):
            validate([[1, 2], [3]], [range_dim[0], set_dim[0]], {}, "c")

    def test_index_expressions_evaluate_nested_tuples_and_unary_values(self):
        evaluate = self.gen._eval_index_expr
        expression = {
            "type": "tuple_literal",
            "elements": [op(name("i"), "+", num(2)), {"type": "uminus", "value": name("i")}, "A"],
        }
        self.assertEqual(evaluate(expression, {"i": 3}), ({}, (5, -3, "A")))
        self.assertEqual(evaluate({"type": "uminus", "value": name("i")}, {"i": "symbol"}), ({}, "-(symbol)"))
        with self.assertRaises(SemanticError):
            evaluate(op(num(1), "/", num(2)), {})

    def test_sum_accumulation_combines_constants_and_variable_coefficients(self):
        generator = SciPyCSCCodeGenerator(parse_model("dvar float x; minimize 0; subject to {}"))
        generator._build_variables()
        sum_node = {"type": "sum", "iterators": [{"iterator": "i", "range": rng(1, 3)}], "expression": name("x")}
        coefficients, constant = generator._accumulate_sum_to_dict(op(sum_node, "-", num(2)), {})
        self.assertEqual(coefficients, {"x": 3})
        self.assertEqual(constant, -2)
        coefficients, constant = generator._accumulate_sum_to_dict(op(num(2), "+", sum_node), {})
        self.assertEqual(coefficients, {"x": 3})
        self.assertEqual(constant, 2)

    def test_indexed_variable_dimensions_resolve_or_report_missing_domains(self):
        resolve = self.gen._resolve_indexed_variable_dimension
        self.assertEqual(resolve({"type": "named_range_dimension", "name": "I"})[0], [2, 3, 4])
        self.assertEqual(resolve({"type": "named_set_dimension", "name": "S"})[0], ["A", "B"])
        for dimension in [
            {"type": "named_range_dimension", "name": "unknown"},
            {"type": "named_set_dimension", "name": "Missing"},
            {"type": "named_set_dimension", "name": "unknown"},
        ]:
            with self.subTest(dimension=dimension), self.assertRaises(SemanticError):
                resolve(dimension)

    def test_parameter_resolution_uses_declared_range_start_and_tuple_keys(self):
        self.gen.ast["declarations"].append(
            {"type": "parameter_external_indexed", "name": "p", "dimensions": [{"type": "named_range_dimension", "name": "I"}]}
        )
        self.gen.data_dict.update({"p": [10, 20, 30], "cost": {("A", "B"): 7}})
        self.assertEqual(self.gen._resolve_parameter("p", [2], {}), (False, 10.0, False))
        self.assertEqual(self.gen._resolve_parameter("p", [4], {}), (False, 30.0, False))
        self.assertEqual(self.gen._resolve_parameter("cost", ["A", "B"], {}), (False, 7.0, False))
        with self.assertRaisesRegex(SemanticError, "not found"):
            self.gen._resolve_parameter("p", ["wrong"], {})

    def test_sum_accumulation_of_two_sums_and_unsupported_operator(self):
        generator = SciPyCSCCodeGenerator(parse_model("dvar float x; minimize 0; subject to {}"))
        generator._build_variables()
        left = {"type": "sum", "iterators": [{"iterator": "i", "range": rng(1, 3)}], "expression": name("x")}
        right = {"type": "sum", "iterators": [{"iterator": "j", "range": rng(1, 2)}], "expression": num(2)}
        coefficients, constant = generator._accumulate_sum_to_dict(op(left, "-", right), {})
        self.assertEqual(coefficients, {"x": 3})
        self.assertEqual(constant, -4)
        with self.assertRaises(SemanticError):
            generator._accumulate_sum_to_dict(op(left, "*", right), {})

    def test_linear_expression_evaluation_rejects_nonlinear_and_zero_division(self):
        generator = SciPyCSCCodeGenerator(parse_model("dvar float x; minimize 0; subject to {}"))
        generator._build_variables()
        self.assertEqual(generator._eval_expr(op(name("x"), "*", num(3)), {}), ({"x": 3.0}, 0.0))
        self.assertEqual(generator._eval_expr(op(name("x"), "/", num(2)), {}), ({"x": 0.5}, 0.0))
        for expression, message in [
            (op(name("x"), "*", name("x")), "nonlinear"),
            (op(num(1), "/", name("x")), "nonlinear"),
            (op(num(1), "/", num(0)), "Division by zero"),
        ]:
            with self.subTest(expression=expression), self.assertRaisesRegex(SemanticError, message):
                generator._eval_expr(expression, {})


if __name__ == "__main__":
    unittest.main()

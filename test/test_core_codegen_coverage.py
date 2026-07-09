import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from pyopl.gurobi_codegen import EPS, GurobiCodeGenerator
from pyopl.pyopl_core import (
    OPLCompiler,
    SemanticError,
    _append_list_item,
    _coerce_float_set_element,
    _coerce_int_set_element,
    _dat_tuple_literal,
    _empty_dat_tuple_literal,
    _empty_model_tuple_literal,
    _execution_error_with_hint,
    _list_with_item,
    _load_failure_message,
    _model_boolean_literal_to_bool,
    _model_tuple_literal,
    _parser_error_with_hint,
    _prepend_list_item,
    _string_label_value_pair,
    _unquote_string_literal,
    export_model,
)
from pyopl.scipy_codegen import SciPyCodeGenerator
from pyopl.scipy_codegen_base import SciPyCodeGeneratorBase
from pyopl.scipy_codegen_csc import BOOL_EPS, SciPyCSCCodeGenerator


def _num(value):
    return {"type": "number", "value": value}


def _name(value, sem_type="float"):
    return {"type": "name", "value": value, "sem_type": sem_type}


def _cmp(left, op, right):
    return {"type": "binop", "op": op, "left": left, "right": right, "sem_type": "boolean"}


class TestCoreHelperCoverage(unittest.TestCase):
    def test_parser_error_hint_specializes_unexpected_in(self):
        msg = _parser_error_with_hint("IN", "in")

        self.assertIn("unexpected 'in'", msg)
        self.assertIn("unsupported filtered declaration", msg)

    def test_parser_error_hint_uses_generic_rewrite_hint(self):
        msg = _parser_error_with_hint("NAME", "foo")

        self.assertIn("Syntax error at or near token NAME", msg)
        self.assertIn("rewrite the construct", msg)

    def test_execution_error_hint_specializes_string_arithmetic(self):
        exc = TypeError("unsupported operand type(s) for -: 'str' and 'str'")

        msg = _execution_error_with_hint(exc, "scipy")

        self.assertIn("string comparison", msg)
        self.assertIn("inside an algebraic expression", msg)

    def test_execution_error_hint_specializes_temp_constraint_arithmetic(self):
        exc = TypeError("unsupported operand type(s) for -: 'gurobipy._core.LinExpr' and 'TempConstr'")

        msg = _execution_error_with_hint(exc, "gurobi")

        self.assertIn("boolean comparison", msg)
        self.assertIn("separate linear constraints", msg)

    def test_execution_error_hint_falls_back_to_backend_construct_hint(self):
        msg = _execution_error_with_hint(RuntimeError("boom"), "scipy")

        self.assertIn("construct accepted by parsing", msg)
        self.assertIn("Simplify boolean logic", msg)

    def test_load_failure_message_mentions_common_fixes(self):
        msg = _load_failure_message()

        self.assertIn("Failed to load or parse", msg)
        self.assertIn("unsupported declaration filters", msg)

    def test_small_parser_helpers_build_expected_values(self):
        items = _list_with_item("a")

        self.assertEqual(items, ["a"])
        self.assertIs(_append_list_item(items, "b"), items)
        self.assertEqual(items, ["a", "b"])
        self.assertEqual(_prepend_list_item("z", items), ["z", "a", "b"])
        self.assertEqual(_unquote_string_literal('"label"'), "label")
        self.assertTrue(_model_boolean_literal_to_bool("true"))
        self.assertFalse(_model_boolean_literal_to_bool("false"))
        self.assertEqual(_string_label_value_pair('"k"', 3), ("k", 3))
        self.assertEqual(_model_tuple_literal([1, "x"]), {"type": "tuple_literal", "elements": [1, "x"]})
        self.assertEqual(_empty_model_tuple_literal(), {"type": "tuple_literal", "elements": []})
        self.assertEqual(_dat_tuple_literal([1, "x"]), (1, "x"))
        self.assertEqual(_empty_dat_tuple_literal(), ())

    def test_set_element_coercion_rejects_booleans(self):
        self.assertEqual(_coerce_int_set_element(3), 3)
        self.assertEqual(_coerce_float_set_element(3), 3.0)

        with self.assertRaises(SemanticError):
            _coerce_int_set_element(True)
        with self.assertRaises(SemanticError):
            _coerce_int_set_element(3.5)
        with self.assertRaises(SemanticError):
            _coerce_float_set_element(False)

    def test_export_model_writes_python_file_from_strings(self):
        model_code = "dvar float+ x; minimize x; subject to { x >= 1; }"

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.py"
            written_path = export_model(model_code, "", "scipy", output_path)

            self.assertEqual(written_path, output_path)
            exported = output_path.read_text(encoding="utf-8")

        self.assertIn("linprog", exported)
        self.assertIn("var_names = ['x']", exported)


class TestCodeGeneratorCoverage(unittest.TestCase):
    def test_scipy_codegen_factory_rejects_unknown_mode(self):
        ast = {
            "declarations": [],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
            "constraints": [],
        }

        with self.assertRaisesRegex(ValueError, "Unknown mode: dense"):
            SciPyCodeGenerator(ast, mode="dense")

    def test_scipy_codegen_base_initializes_shared_state_and_is_abstract(self):
        ast = {"declarations": []}
        base = SciPyCodeGeneratorBase(ast, data_dict={"p": 1})

        self.assertIs(base.ast, ast)
        self.assertEqual(base.data_dict, {"p": 1})
        self.assertEqual(base.var_names, [])
        self.assertEqual(base.results_varname, "results")
        with self.assertRaisesRegex(NotImplementedError, "Subclasses must implement"):
            base.generate_code()

    def test_gurobi_name_and_comparison_helpers(self):
        gen = GurobiCodeGenerator({"declarations": [], "objective": {}, "constraints": []})

        self.assertEqual(gen._format_name_expr("c0"), "'c0'")
        self.assertEqual(gen._format_name_expr("c0", "_rhs"), "'c0_rhs'")

        gen._active_label_name_expr = "label_expr"
        self.assertEqual(gen._format_name_expr("c0"), "label_expr")
        self.assertEqual(gen._format_name_expr("c0", "_rhs"), "(label_expr + '_rhs')")

        self.assertEqual(gen._gurobi_comparison_expr("x", ">", "1"), f"x >= (1) + {EPS}")
        self.assertEqual(gen._gurobi_comparison_expr("x", "<", "5"), f"x <= (5) - {EPS}")
        self.assertEqual(gen._gurobi_comparison_expr("x", "==", "y"), "x == y")

    def test_gurobi_label_expr_and_simple_expressions(self):
        gen = GurobiCodeGenerator({"declarations": [], "objective": {}, "constraints": []}, data_dict={"p": 9})

        self.assertEqual(gen._compute_label_expr({"name": "Cap"}), "'Cap'")
        self.assertEqual(
            gen._compute_label_expr({"name": "Cap", "iterators": ["i", "j"]}),
            "('Cap' + '[' + ','.join(str(v) for v in [i, j]) + ']')",
        )
        self.assertEqual(gen._traverse_expression({"type": "number", "value": 7}, {}), "7")
        self.assertEqual(gen._traverse_expression({"type": "boolean_literal", "value": True}, {}), "1")
        self.assertEqual(gen._traverse_expression({"type": "boolean_literal", "value": False}, {}), "0")
        self.assertEqual(gen._traverse_expression({"type": "string_literal", "value": "A"}, {}), "'A'")
        self.assertEqual(gen._traverse_expression({"type": "name", "value": "i"}, {"i": 1}), "i")
        self.assertEqual(gen._traverse_expression({"type": "name", "value": "p"}, {}), "p")

    def test_gurobi_function_expression_helpers(self):
        gen = GurobiCodeGenerator({"declarations": [], "objective": {}, "constraints": []})

        sqrt_expr = {"type": "funcall", "name": "sqrt", "args": [{"type": "number", "value": 16}]}
        abs_expr = {"type": "funcall", "name": "abs", "args": [{"type": "number", "value": -3}]}
        log_expr = {"type": "funcall", "name": "log", "args": [{"type": "number", "value": 1}]}
        min_expr = {"type": "minl", "args": [{"type": "number", "value": 2}, {"type": "number", "value": 3}]}
        max_expr = {"type": "maxl", "args": [{"type": "number", "value": 2}, {"type": "number", "value": 3}]}

        self.assertEqual(gen._traverse_expression(sqrt_expr, {}), "math.sqrt(16)")
        self.assertEqual(gen._traverse_expression(abs_expr, {}), "abs(-3)")
        self.assertEqual(gen._traverse_expression(log_expr, {}), "math.log(1)")
        self.assertEqual(gen._traverse_expression(min_expr, {}), "min(2, 3)")
        self.assertEqual(gen._traverse_expression(max_expr, {}), "max(2, 3)")
        with self.assertRaisesRegex(NotImplementedError, "Unsupported function call 'unknown'"):
            gen._traverse_expression({"type": "funcall", "name": "unknown", "args": [{"type": "number", "value": 1}]}, {})

    def test_scipy_csc_strict_rhs_and_vector_updates(self):
        gen = SciPyCSCCodeGenerator({"declarations": [], "constraints": []})
        gen.var_indices = {"x": 0, "y": 1}
        vector = [0.0, 0.0]

        gen._update_vector_from_coef_dict({"x": 2.0, "missing": 9.0}, vector)
        self.assertEqual(vector, [2.0, 0.0])
        gen._update_vector_from_coef_dict({"x": 3.0, "y": 4.0}, vector, op="+")
        self.assertEqual(vector, [5.0, 4.0])
        gen._update_vector_from_coef_dict({"x": 1.5, "y": 2.0}, vector, op="-")
        self.assertEqual(vector, [3.5, 2.0])

        self.assertEqual(gen._strict_adjusted_rhs(">", 1.0), (">=", 1.0 + BOOL_EPS))
        self.assertEqual(gen._strict_adjusted_rhs("<", 5.0), ("<=", 5.0 - BOOL_EPS))
        self.assertEqual(gen._strict_adjusted_rhs("==", 2.0), ("==", 2.0))

    def test_scipy_csc_variable_and_tuple_helpers(self):
        gen = SciPyCSCCodeGenerator({"declarations": [], "constraints": []})
        gen.tuple_types = {"Arc": [{"name": "from"}, {"name": "to"}]}

        first_name, first_idx = gen._add_variable("aux", lower=-1.0, upper=2.0)
        second_name, second_idx = gen._add_variable("aux")

        self.assertEqual((first_name, first_idx), ("aux", 0))
        self.assertEqual((second_name, second_idx), ("aux_1", 1))
        self.assertEqual(gen.lower_bounds[:2], [-1.0, 0.0])
        self.assertEqual(gen.upper_bounds[:2], [2.0, 1.0])
        self.assertEqual(gen._resolve_tuple_field("Arc", "from", ("A", "B")), "A")
        self.assertEqual(gen._resolve_tuple_field("Arc", "to", {"to": "B"}), "B")
        self.assertIsNone(gen._resolve_tuple_field("Arc", "cost", ("A", "B")))
        self.assertIsNone(gen._resolve_tuple_field("Missing", "from", ("A", "B")))

    def test_scipy_csc_error_factories_return_semantic_errors(self):
        gen = SciPyCSCCodeGenerator({"declarations": [], "constraints": []})

        self.assertIsInstance(gen._not_found_error("parameter", "p"), SemanticError)
        self.assertEqual(str(gen._not_found_error("parameter", "p")), "Semantic Error: Not found: parameter 'p'")
        self.assertEqual(
            str(gen._unsupported_type_error("declaration", "tuple")),
            "Semantic Error: Semantic Error: Unsupported declaration type: tuple",
        )
        self.assertEqual(
            str(gen._unsupported_operator_error("constraint", "~")),
            "Semantic Error: Semantic Error: Unsupported operator in constraint: ~",
        )

    def test_scipy_csc_indentation_and_symbolic_traversal(self):
        gen = SciPyCSCCodeGenerator({"declarations": [], "constraints": []})
        gen.indent_level = 2
        gen._add_code_line("x = 1")
        self.assertEqual(gen.scipy_code_lines[-1], "        x = 1")

        gen.tuple_types = {"Arc": [{"name": "from"}, {"name": "to"}]}
        expr = {
            "type": "conditional",
            "condition": {
                "type": "binop",
                "op": ">=",
                "left": {"type": "name", "value": "x"},
                "right": {"type": "number", "value": 0},
            },
            "then": {"type": "indexed_name", "name": "cost", "dimensions": [{"type": "name_reference_index", "name": "i"}]},
            "else": {"type": "uminus", "value": {"type": "number", "value": 1}},
        }
        field_expr = {
            "type": "field_access",
            "base": {"type": "name", "value": "arc", "sem_type": "Arc"},
            "field": "to",
        }
        tuple_expr = {
            "type": "tuple_literal",
            "elements": [{"type": "string_literal", "value": "A"}, {"type": "number_literal_index", "value": 2}],
        }

        self.assertEqual(gen._traverse_expression(expr), "(cost[i] if ((x >= 0)) else -(1))")
        self.assertEqual(gen._traverse_expression(field_expr), "arc[1]")
        self.assertEqual(gen._traverse_expression(tuple_expr), "('A', 2)")
        self.assertEqual(gen._traverse_expression({"type": "unknown", "value": 1}), "")

    def test_scipy_csc_flat_kv_rows_aux_and_tuple_set_helpers(self):
        ast = {"declarations": [{"type": "set_of_tuples", "name": "Arcs"}]}
        gen = SciPyCSCCodeGenerator(ast)
        gen._find_decl = lambda name, decl_type=None: ast["declarations"][0] if name == "Arcs" else None

        self.assertEqual(gen._convert_flat_kv_to_dict(["A", 1, "B", 2.5]), {"A": 1, "B": 2.5})
        self.assertIsNone(gen._convert_flat_kv_to_dict(["A", 1, "B"]))

        gen.var_names = ["x", "y"]
        gen.var_indices = {"x": 0, "y": 1}
        self.assertEqual(gen._make_constraint_row({"x": 2.0, "missing": 9.0, "y": -1.0}), [2.0, -1.0])
        self.assertEqual(gen._get_tuple_set_names([{"iterator": "a", "range": {"type": "named_set", "name": "Arcs"}}]), {"a"})
        self.assertEqual(gen._get_tuple_set_names([{"iterator": "i", "range": {"type": "range_specifier"}}]), set())

        gen.aux_created = []
        aux = gen._ensure_aux_binary("flag")
        aux2 = gen._ensure_aux_binary("flag")
        self.assertEqual((aux, aux2), ("flag", "flag_1"))
        self.assertEqual(gen.bounds[-2:], [[0, 1], [0, 1]])
        self.assertEqual(gen.integrality[-2:], [1, 1])
        self.assertEqual(gen.aux_created[-2:], ["flag", "flag_1"])

    def test_scipy_csc_bool_flatten_and_sum_term_filtering(self):
        gen = SciPyCSCCodeGenerator({"declarations": [], "constraints": []})
        leaf1 = _cmp(_name("x"), "<=", _num(1))
        leaf2 = _cmp(_name("y"), ">=", _num(0))
        tree = {"type": "and", "left": leaf1, "right": {"type": "and", "left": leaf2, "right": "raw"}}

        self.assertTrue(gen._is_linear_comparison(leaf1))
        self.assertFalse(gen._is_linear_comparison({"type": "binop", "op": "+", "sem_type": "int"}))
        self.assertEqual(gen._flatten_bool(tree, "and"), [leaf1, leaf2, "raw"])
        self.assertEqual(gen._flatten_bool("raw", "and"), ["raw"])

        tuple_env, include = gen._should_include_sum_term(
            ["a"],
            (["A", "B"],),
            {"a"},
            {"outer": 1},
            None,
            {},
        )
        self.assertEqual(tuple_env, {"outer": 1, "a": ("A", "B")})
        self.assertTrue(include)

        gen._eval_expr = lambda node, env: ({}, 0)
        _env, include = gen._should_include_sum_term(["i"], (1,), set(), {}, _cmp(_name("i"), "==", _num(2)), {})
        self.assertFalse(include)
        gen._eval_expr = lambda node, env: (_ for _ in ()).throw(RuntimeError("ignore"))
        _env, include = gen._should_include_sum_term(["i"], (1,), set(), {}, _cmp(_name("i"), "==", _num(2)), {})
        self.assertTrue(include)

    def test_scipy_csc_big_m_and_bounds_helpers(self):
        ast = {
            "declarations": [
                {"type": "dvar", "var_type": "boolean", "name": "b"},
                {"type": "dvar", "var_type": "float+", "name": "x"},
                {"type": "dvar", "var_type": "float", "name": "y"},
            ]
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen.var_names = ["x", "y", "b"]
        gen.var_indices = {"x": 0, "y": 1, "b": 2}
        gen._collected_lbs = {"x": 2.0, "y": -3.0}
        gen._collected_ubs = {"x": 5.0, "y": 4.0}

        self.assertEqual(gen._var_bounds_safe(_name("b", "boolean")), (0.0, 1.0))
        self.assertEqual(gen._var_bounds_safe(_name("x")), (0.0, None))
        self.assertEqual(gen._var_bounds_safe(_num(7)), (7.0, 7.0))

        expr = {
            "type": "binop",
            "op": "+",
            "left": _name("x"),
            "right": {"type": "binop", "op": "*", "left": _num(-2), "right": _name("b", "boolean")},
        }
        self.assertEqual(gen._linear_bounds_safe(expr), (0.0, 5.0))
        self.assertEqual(
            gen._linear_bounds_safe({"type": "binop", "op": "-", "left": _name("x"), "right": _name("y")}), (-2.0, 8.0)
        )

        comp = _cmp({"type": "binop", "op": "-", "left": _name("x"), "right": _name("y")}, "<=", _num(1))
        self.assertEqual(gen._big_m_for_comparison(comp), 11.0)
        gen._eval_expr = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fallback"))
        self.assertGreaterEqual(gen._big_m_for_comparison(comp), 1_000_000.0)

    def test_scipy_csc_linearize_or_and_expand_and_helpers(self):
        gen = SciPyCSCCodeGenerator({"declarations": []})
        gen.var_names = ["x", "y"]
        gen.var_indices = {"x": 0, "y": 1}
        gen.bounds = [[0, 10], [0, 10]]
        gen.integrality = [0, 0]
        gen.c = [0.0, 0.0]
        gen._collected_lbs = {"x": 0.0, "y": 0.0}
        gen._collected_ubs = {"x": 10.0, "y": 10.0}
        le = _cmp(_name("x"), "<=", _num(3))
        ge = _cmp(_name("y"), ">=", _num(2))
        eq = _cmp(_name("x"), "==", _name("y"))

        gen._linearize_or([le, ge, eq])
        self.assertEqual(len([name for name in gen.var_names if name.startswith("or_flag")]), 3)
        self.assertGreaterEqual(len(gen.A_ub), 5)
        self.assertEqual(gen.b_ub[-1], -1.0)

        gen2 = SciPyCSCCodeGenerator({"declarations": []})
        gen2.var_names = ["x", "y"]
        gen2.var_indices = {"x": 0, "y": 1}
        gen2._eval_expr = lambda node, env=None: (
            ({node["value"]: 1.0}, 0.0) if node.get("type") == "name" else ({}, float(node["value"]))
        )
        gen2._accumulate_sum_to_dict = lambda node, env=None, sign=1: gen2._eval_expr(node, env)
        gen2._expand_and([_cmp(_name("x"), "==", _num(1)), _cmp(_name("x"), "<=", _num(3)), _cmp(_name("y"), ">=", _num(2))])
        self.assertEqual(len(gen2.A_eq), 1)
        self.assertEqual(len(gen2.A_ub), 2)

    def test_scipy_csc_accumulate_sum_helpers(self):
        gen = SciPyCSCCodeGenerator({"declarations": []})
        gen.var_indices = {"x_1": 0, "x_2": 1, "z": 2}
        gen._iterate_iterators_dynamic = lambda iterators, env: [({"i": 1}, (1,)), ({"i": 2}, (2,))]
        gen._eval_expr = lambda expr, env=None: (
            ({f"x_{env['i']}": 1.5}, env["i"]) if expr.get("type") == "indexed_name" else ({"z": 2.0}, 4.0)
        )
        sum_expr = {"type": "sum", "iterators": [{"iterator": "i"}], "expression": {"type": "indexed_name", "name": "x"}}
        coef = defaultdict(float)
        const = [0.0]

        gen._accumulate_sum_expr(sum_expr, {}, coef, 1.0, const)
        self.assertEqual(dict(coef), {"x_1": 1.5, "x_2": 1.5})
        self.assertEqual(const[0], 3.0)

        target = defaultdict(float)
        const_box = [0.0]
        expr = {"type": "binop", "op": "-", "left": sum_expr, "right": {"type": "name", "value": "z"}}
        gen._accumulate_binop_with_sum(expr, {}, target, 1.0, const_box)
        self.assertEqual(target["x_1"], 1.5)
        self.assertEqual(target["x_2"], 1.5)
        self.assertEqual(target[2], -2.0)
        self.assertEqual(const_box[0], -1.0)

    def test_model_codegen_weighted_boolean_sum_paths(self):
        model = """
        range I = 1..2;
        float w[I] = [1, 2];
        dvar float+ x[I];
        dvar float+ y[I];
        minimize sum(i in I) x[i];
        subject to {
            sum(i in I) w[i] * ((x[i] >= 1) && (y[i] <= 2)) >= 1;
        }
        """

        scipy_ast, scipy_code, scipy_data = OPLCompiler().compile_model(model, solver="scipy")
        gurobi_ast, gurobi_code, gurobi_data = OPLCompiler().compile_model(model, solver="gurobi")

        scipy_gen = SciPyCSCCodeGenerator(scipy_ast, scipy_data)
        scipy_gen.generate_code()
        self.assertTrue(any(name.startswith("bool_") or name.startswith("cmp_") for name in scipy_gen.var_names))
        self.assertIn("quicksum", gurobi_code)
        self.assertEqual(gurobi_ast["objective"]["type"], "minimize")
        self.assertEqual(gurobi_data["w"], [1, 2])

    def test_model_codegen_reified_cardinality_paths(self):
        model = """
        range I = 1..3;
        dvar boolean b;
        dvar float+ x[I];
        minimize b;
        subject to {
            b == (sum(i in I) (x[i] >= i) >= 2);
        }
        """

        scipy_ast, _scipy_code, scipy_data = OPLCompiler().compile_model(model, solver="scipy")
        gurobi_ast, gurobi_code, _gurobi_data = OPLCompiler().compile_model(model, solver="gurobi")

        scipy_gen = SciPyCSCCodeGenerator(scipy_ast, scipy_data)
        scipy_gen.generate_code()
        self.assertIn("_cmp_sum_list", gurobi_code)
        self.assertIn("model.addConstr(b == _cmp_expr", gurobi_code)
        self.assertTrue(any(name.startswith("cmp_") for name in scipy_gen.var_names))
        self.assertEqual(gurobi_ast["constraints"][0]["op"], "==")

    def test_model_codegen_tuple_range_parameter_flattening(self):
        model = """
        tuple Store { string id; }
        {Store} Stores = { <"A">, <"B"> };
        range T = 1..3;
        float demand[Stores][T] = ...;
        dvar float+ x[Stores][T];
        minimize sum(s in Stores, t in T) x[s][t];
        subject to {
            forall(s in Stores, t in T) x[s][t] >= demand[s][t];
        }
        """
        data = """
        demand = [ <"A"> [1, 2, 3], <"B"> [4, 5, 6] ];
        """

        ast, _code, data_dict = OPLCompiler().compile_model(model, data, solver="gurobi")
        gen = GurobiCodeGenerator(ast, data_dict)
        gen._generate_data_declarations(data_dict)
        gurobi_data_code = "\n".join(gen.gurobi_code_lines)

        scipy_gen = SciPyCSCCodeGenerator(ast, data_dict)
        scipy_gen._generate_data_declarations(data_dict)
        scipy_data_code = "\n".join(scipy_gen.scipy_code_lines)

        self.assertIn("demand =", gurobi_data_code)
        self.assertIn("(('A',), 1)", gurobi_data_code)
        self.assertIn("demand =", scipy_data_code)
        self.assertIn("('A',): {1: 1.0", scipy_data_code)

    def test_gurobi_data_declarations_cover_flattening_and_shape_errors(self):
        ast = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "T",
                    "start": _num(2),
                    "end": {"type": "binop", "op": "+", "left": _num(3), "right": _num(1)},
                },
                {"type": "typed_set", "base_type": "string", "name": "Products", "value": ["P1", "P2"]},
                {
                    "type": "parameter_external_indexed",
                    "name": "by_range",
                    "dimensions": [
                        {
                            "type": "named_range_dimension",
                            "name": "T",
                            "start": _num(2),
                            "end": {"type": "binop", "op": "+", "left": _num(3), "right": _num(1)},
                        }
                    ],
                },
                {
                    "type": "parameter_external_indexed",
                    "name": "by_set",
                    "dimensions": [{"type": "named_set_dimension", "name": "Products"}],
                },
                {
                    "type": "parameter_external_indexed",
                    "name": "matrix",
                    "dimensions": [
                        {"type": "named_set_dimension", "name": "Products"},
                        {
                            "type": "named_range_dimension",
                            "name": "T",
                            "start": _num(2),
                            "end": {"type": "binop", "op": "+", "left": _num(3), "right": _num(1)},
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
                {
                    "type": "parameter_external_indexed",
                    "name": "flat",
                    "dimensions": [{"type": "named_set_dimension", "name": "Products"}],
                },
            ]
        }
        data = {
            "by_range": [10, 20, 30],
            "by_set": [1, 2],
            "matrix": {"P1": [1, 2, 3], "P2": [4, 5, 6]},
            "grid": [[1, 2], [3, 4]],
            "flat": ["P1", 9, "P2", 8],
        }
        gen = GurobiCodeGenerator(ast, data)
        gen._generate_data_declarations(data)
        code = "\n".join(gen.gurobi_code_lines)

        self.assertEqual(data["flat"], {"P1": 9, "P2": 8})
        self.assertIn("by_range = {2: 10, 3: 20, 4: 30}", code)
        self.assertIn("by_set = {'P1': 1, 'P2': 2}", code)
        self.assertIn("matrix = {('P1', 2): 1", code)
        self.assertIn("grid = {('P1', 'P1'): 1", code)

        bad = GurobiCodeGenerator(ast, {"by_range": [1, 2]})
        with self.assertRaises(SemanticError):
            bad._generate_data_declarations(bad.data_dict)

    def test_scipy_data_declarations_cover_nested_and_structured_forms(self):
        ast = {
            "declarations": [
                {"type": "tuple_type", "name": "Node", "fields": [{"name": "id"}, {"name": "cost"}]},
                {"type": "typed_set", "base_type": "string", "name": "Products", "value": ["P1", "P2"]},
                {"type": "range_declaration_inline", "name": "T", "start": _num(1), "end": _num(2)},
                {"type": "tuple_array_external", "name": "nodes", "tuple_type": "Node", "index_set": "Products"},
                {
                    "type": "parameter_external_indexed",
                    "name": "cube",
                    "dimensions": [
                        {"type": "named_set_dimension", "name": "Products"},
                        {"type": "named_range_dimension", "name": "T"},
                        {"type": "named_set_dimension", "name": "Products"},
                    ],
                },
                {
                    "type": "parameter_external_indexed",
                    "name": "by_range",
                    "dimensions": [{"type": "named_range_dimension", "name": "T"}],
                },
            ]
        }
        data = {
            "nodes": [["A", 1], {"id": "B", "cost": 2}],
            "cube": {"P1": [[1, 2], [3, 4]], "P2": [[5, 6], [7, 8]]},
            "by_range": [9, 10],
        }
        gen = SciPyCSCCodeGenerator(ast, data)
        gen._generate_data_declarations(data)
        code = "\n".join(gen.scipy_code_lines)

        self.assertIn("nodes = {1: {'id': 'A', 'cost': 1}", code)
        self.assertIn("cube = {'P1': [[1, 2], [3, 4]]", code)
        self.assertIn("by_range = [9, 10]", code)

        bad = SciPyCSCCodeGenerator(ast, {"by_range": [1]})
        with self.assertRaises(SemanticError):
            bad._generate_data_declarations(bad.data_dict)

    def test_core_materializes_computed_parameter_expression_variants(self):
        compiler = OPLCompiler()
        ast = {
            "declarations": [
                {"type": "tuple_type", "name": "Pair", "fields": [{"name": "id"}, {"name": "weight"}]},
                {
                    "type": "set_of_tuples",
                    "name": "Pairs",
                    "tuple_type": "Pair",
                    "value": [{"elements": ["A", 2]}, {"elements": ["B", 3]}],
                },
                {"type": "range_declaration_inline", "name": "R", "start": _num(1), "end": _num(2)},
                {"type": "parameter_inline", "var_type": "float", "name": "base", "value": 4},
                {"type": "parameter_inline", "var_type": "float", "name": "arr", "value": [10, 20, 30]},
                {"type": "parameter_inline", "var_type": "float", "name": "lookup", "value": {"A": 7, "B": 9}},
                {
                    "type": "parameter_inline_expr",
                    "var_type": "float",
                    "name": "expr_value",
                    "expression": {
                        "type": "binop",
                        "op": "+",
                        "left": {
                            "type": "binop",
                            "op": "+",
                            "left": {"type": "funcall", "name": "sqrt", "args": [_name("base")]},
                            "right": {"type": "funcall", "name": "floor", "args": [_num(2.9)]},
                        },
                        "right": {"type": "maxl", "args": [_num(3), {"type": "minl", "args": [_num(8), _num(5)]}]},
                    },
                },
                {
                    "type": "parameter_inline_expr",
                    "var_type": "boolean",
                    "name": "logic_value",
                    "expression": {
                        "type": "or",
                        "left": {
                            "type": "and",
                            "left": {"type": "boolean_literal", "value": True},
                            "right": {"type": "not", "value": {"type": "boolean_literal", "value": False}},
                        },
                        "right": _cmp(_num(1), "!=", _num(1)),
                    },
                },
                {
                    "type": "parameter_inline_indexed_expr",
                    "var_type": "float",
                    "name": "computed",
                    "dimensions": [
                        {"type": "named_set_dimension", "name": "Pairs"},
                        {"type": "named_range_dimension", "name": "R"},
                    ],
                    "iterators": [
                        {"iterator": "p", "range": {"type": "named_set", "name": "Pairs"}},
                        {"iterator": "i", "range": {"type": "named_range", "name": "R"}},
                    ],
                    "expression": {
                        "type": "conditional",
                        "condition": {
                            "type": "binop",
                            "op": ">=",
                            "left": {"type": "field_access", "base": _name("p", "Pair"), "field": "weight"},
                            "right": _num(3),
                        },
                        "then": {
                            "type": "binop",
                            "op": "+",
                            "left": {
                                "type": "indexed_name",
                                "name": "arr",
                                "dimensions": [
                                    {
                                        "type": "parenthesized_expression",
                                        "expression": {
                                            "type": "binop",
                                            "op": "+",
                                            "left": {"type": "name_reference_index", "name": "i"},
                                            "right": {"type": "number_literal_index", "value": 1},
                                        },
                                    }
                                ],
                            },
                            "right": {
                                "type": "sum",
                                "iterators": [
                                    {"iterator": "j", "range": {"type": "range_specifier", "start": _num(1), "end": _num(2)}}
                                ],
                                "index_constraint": _cmp(_name("j"), "<=", _name("i")),
                                "expression": _name("j"),
                            },
                        },
                        "else": {
                            "type": "max_agg",
                            "iterators": [{"iterator": "q", "range": {"type": "named_set", "name": "Pairs"}}],
                            "expression": {"type": "field_access", "base": _name("q", "Pair"), "field": "weight"},
                        },
                    },
                },
            ]
        }
        working_data = {
            "Pairs": {"elements": [["A", 2], ["B", 3]], "tuple_type": "Pair"},
            "base": 4,
            "arr": [10, 20, 30],
            "lookup": {"A": 7, "B": 9},
        }

        compiler._materialize_computed_parameters(ast, working_data)

        self.assertEqual(working_data["expr_value"], 9.0)
        self.assertTrue(working_data["logic_value"])
        self.assertEqual(working_data["computed"][0][0], 3.0)
        self.assertEqual(working_data["computed"][1][1], 33.0)
        self.assertEqual(working_data["computed__map"][(("B", 3), 2)], 33.0)

    def test_core_computed_parameter_error_branches(self):
        compiler = OPLCompiler()
        ast = {
            "declarations": [
                {
                    "type": "parameter_inline_expr",
                    "var_type": "float",
                    "name": "bad",
                    "expression": {"type": "funcall", "name": "unknown", "args": [_num(1)]},
                }
            ]
        }

        with self.assertRaisesRegex(SemanticError, "Unsupported function 'unknown'"):
            compiler._materialize_computed_parameters(ast, {})

        ast = {
            "declarations": [
                {"type": "range_declaration_inline", "name": "R", "start": _num(1), "end": _num(1)},
                {
                    "type": "parameter_inline_indexed_expr",
                    "var_type": "float",
                    "name": "bad_idx",
                    "dimensions": [{"type": "named_range_dimension", "name": "R"}],
                    "iterators": [{"iterator": "i", "range": {"type": "named_range", "name": "R"}}],
                    "expression": {
                        "type": "indexed_name",
                        "name": "arr",
                        "dimensions": [{"type": "string_literal", "value": "x"}],
                    },
                },
            ]
        }
        with self.assertRaisesRegex(SemanticError, "requires integer indices"):
            compiler._materialize_computed_parameters(ast, {"arr": [1]})

    def test_gurobi_bounds_and_composite_implication_helpers(self):
        ast = {
            "declarations": [
                {"type": "dvar", "var_type": "boolean", "name": "b"},
                {"type": "dvar", "var_type": "float+", "name": "x"},
                {"type": "dvar", "var_type": "float", "name": "y"},
            ],
            "constraints": [],
        }
        gen = GurobiCodeGenerator(ast)
        gen.gurobi_var_map = {"b": "b", "x": "x", "y": "y"}
        gen._collected_lbs = {"x": 1.0, "y": -2.0}
        gen._collected_ubs = {"x": 4.0, "y": 3.0}

        self.assertEqual(
            gen._linear_bounds_safe({"type": "binop", "op": "+", "left": _name("x"), "right": _num(2)}), (3.0, 6.0)
        )
        self.assertEqual(
            gen._linear_bounds_safe({"type": "binop", "op": "*", "left": _num(-2), "right": _name("b", "boolean")}),
            (-2.0, -0.0),
        )
        self.assertIsNone(gen._linear_bounds_safe({"type": "binop", "op": "*", "left": _name("x"), "right": _name("y")}))

        implication = {
            "type": "implication_constraint",
            "antecedent": {
                "type": "and",
                "left": _cmp(_name("x"), ">=", _num(1)),
                "right": {"type": "not", "value": _cmp(_name("y"), "<=", _num(0))},
            },
            "consequent": {
                "type": "or",
                "left": _cmp(_name("x"), "<=", _num(4)),
                "right": {"type": "boolean_literal", "value": True},
            },
        }
        gen._constraint_implication_constraint(implication, "c_imp", {})
        code = "\n".join(gen.gurobi_code_lines)

        self.assertIn("and_b", code)
        self.assertIn("not_b", code)
        self.assertIn("or_b", code)
        self.assertIn("model.addConstr", code)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

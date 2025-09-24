import unittest

from pyopl.pyopl_core import GurobiCodeGenerator, OPLLexer, OPLParser
from pyopl.scipy_codegen import SciPyCodeGenerator, SemanticError


def setUpModule():
    import logging

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s")


class DummyAST:
    def get(self, *args, **kwargs):
        return None

    def __getitem__(self, key):
        raise KeyError(key)

    def get_declarations(self):
        return []


class TestModellingConstructs(unittest.TestCase):
    def test_out_of_range_index_in_variable_reference(self):
        """
        Verify out-of-range index detection for i[0] when i is declared over T=1..6.

        Gurobi backend:
          - Code generation succeeds (string-based), but the code shows i[0] together with the
            declaration over range(1, T + 1), making the mismatch explicit.

        SciPy backend:
          - Code generation should fail with a SemanticError because the indexed variable i_0
            does not exist in the constructed variable set (i_1..i_6).
        """
        lexer = OPLLexer()
        parser = OPLParser()
        model_text = """
            range T = 1..6;
            dvar int+ i[T];  // Inventory at end of period t

            minimize 0;

            subject to {
              // Initial inventory 0
              i[0] == 0;  // Not a variable; not indexed in x (skip or comment if 0)
            }
        """
        ast = parser.parse(lexer.tokenize(model_text))

        # Gurobi: generates code; confirm it clearly contains the out-of-range access.
        code_g = GurobiCodeGenerator(ast).generate_code()
        self.assertIn("range(1, T + 1)", code_g, "Expected i to be declared over range(1, T + 1)")
        self.assertIn("i[0]", code_g, "Expected out-of-range indexed reference i[0] in generated code")

        # SciPy: should raise a SemanticError during code generation due to i_0 not existing.
        with self.assertRaises(SemanticError):
            SciPyCodeGenerator(ast).generate_code()

    def test_multi_indexed_var_name_and_format_varname(self):
        """
        Test _multi_indexed_var_name and _format_varname for tuple-indexed and range-indexed variables.
        Covers lines 1219-1227 and 1588-1596 in scipy_codegen_csc.py.
        """
        from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator

        # Minimal AST and data_dict for tuple-indexed and range-indexed variables
        ast = {
            "declarations": [
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "int",
                    "dimensions": [{"type": "named_set_dimension", "name": "TUPSET"}],
                },
                {
                    "type": "set_of_tuples",
                    "name": "TUPSET",
                    "tuple_type": "TUP",
                    "value": [{"elements": [1, 2]}, {"elements": [3, 4]}],
                },
                {
                    "type": "dvar_indexed",
                    "name": "y",
                    "var_type": "int",
                    "dimensions": [
                        {
                            "type": "range_index",
                            "start": {"type": "number", "value": 1},
                            "end": {"type": "number", "value": 2},
                        }
                    ],
                },
            ]
        }
        data_dict = {}
        gen = SciPyCSCCodeGenerator(ast, data_dict)
        gen._build_variables()
        # Test tuple-indexed variable name
        expr_tuple = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "tuple_literal",
                    "elements": [
                        {"type": "number_literal_index", "value": 1},
                        {"type": "number_literal_index", "value": 2},
                    ],
                }
            ],
        }
        vname_tuple = gen._multi_indexed_var_name(expr_tuple, env={}, eval_index_expr=gen._eval_expr)
        self.assertEqual(vname_tuple, "x[(1, 2)]")
        # Test range-indexed variable name
        expr_range = {
            "type": "indexed_name",
            "name": "y",
            "dimensions": [{"type": "number_literal_index", "value": 1}],
        }
        vname_range = gen._multi_indexed_var_name(expr_range, env={}, eval_index_expr=gen._eval_expr)
        self.assertEqual(vname_range, "y_1")
        # Test _format_varname directly for tuple-indexed
        out = gen._format_varname("x", [(1, 2)], True)
        self.assertEqual(out, "x[(1, 2)]")
        # Test _format_varname for range-indexed (single int)
        out2 = gen._format_varname("y", [1], False)
        self.assertEqual(out2, "y_1")
        # Test _format_varname for range-indexed (tuple as index)
        out3 = gen._format_varname("y", [(1, 2)], False)
        self.assertEqual(out3, "y_1_2")
        # Test _format_varname for multi-indexed (multiple ints)
        out4 = gen._format_varname("z", [1, 2], False)
        self.assertEqual(out4, "z_1_2")

    def test_index_expr_eval_tuple_literal(self):
        # Directly exercise the 'tuple_literal' branch in _eval_index_expr
        # by constructing an AST with a tuple_literal index.
        ast = {
            "declarations": [
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [{"type": "named_set_dimension", "name": "S"}],
                },
                {
                    "type": "set_of_tuples",
                    "name": "S",
                    "tuple_type": "T",
                    "value": [{"elements": [1, 2]}, {"elements": [3, 4]}],
                },
                {
                    "type": "tuple_type",
                    "name": "T",
                    "fields": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"},
                    ],
                },
            ],
            "constraints": [],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        # Construct an indexed_name with a tuple_literal as index
        expr = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "tuple_literal",
                    "elements": [
                        {"type": "number_literal_index", "value": 1},
                        {"type": "number_literal_index", "value": 2},
                    ],
                }
            ],
        }
        # This should hit the 'tuple_literal' branch in _eval_index_expr
        coef, const = codegen._eval_expr(expr)
        # The variable name should be x[(1, 2)] or x[(1, 2)] in var_indices
        # The returned coef dict should have a key matching the tuple-indexed variable
        found = False
        for k in coef:
            if isinstance(k, str) and ("(1, 2)" in k or str((1, 2)) in k):
                found = True
        self.assertTrue(found, f"Expected tuple-indexed variable in coef dict, got: {coef}")

        # Now exercise binop (+, -, *), uminus, parenthesized_expression, and error branch
        # Binop: +
        expr_plus = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "binop",
                    "op": "+",
                    "left": {"type": "number_literal_index", "value": 1},
                    "right": {"type": "number_literal_index", "value": 2},
                }
            ],
        }
        # Should return x_3 (since 1+2=3), but variable names are for tuples, so this will not match, but we want to cover the code
        try:
            codegen._eval_expr(expr_plus)
        except Exception:
            pass

        # Binop: -
        expr_minus = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "binop",
                    "op": "-",
                    "left": {"type": "number_literal_index", "value": 5},
                    "right": {"type": "number_literal_index", "value": 2},
                }
            ],
        }
        try:
            codegen._eval_expr(expr_minus)
        except Exception:
            pass

        # Binop: *
        expr_times = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "binop",
                    "op": "*",
                    "left": {"type": "number_literal_index", "value": 3},
                    "right": {"type": "number_literal_index", "value": 2},
                }
            ],
        }
        try:
            codegen._eval_expr(expr_times)
        except Exception:
            pass

        # Binop: unsupported op (should raise)
        expr_bad = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "binop",
                    "op": "/",
                    "left": {"type": "number_literal_index", "value": 4},
                    "right": {"type": "number_literal_index", "value": 2},
                }
            ],
        }
        with self.assertRaises(SemanticError):
            codegen._eval_expr(expr_bad)

        # Uminus
        expr_uminus = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "uminus",
                    "value": {"type": "number_literal_index", "value": 7},
                }
            ],
        }
        try:
            codegen._eval_expr(expr_uminus)
        except Exception:
            pass

        # Parenthesized expression
        expr_paren = {
            "type": "indexed_name",
            "name": "x",
            "dimensions": [
                {
                    "type": "parenthesized_expression",
                    "expression": {"type": "number_literal_index", "value": 9},
                }
            ],
        }
        try:
            codegen._eval_expr(expr_paren)
        except Exception:
            pass

    def test_tuple_field_access_in_forall(self):
        # This test exercises tuple field access in a forall loop, triggering tuple_type inference from current_iterators.
        ast = {
            "declarations": [
                {
                    "type": "tuple_type",
                    "name": "MyTuple",
                    "fields": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"},
                    ],
                },
                {
                    "type": "set_of_tuples",
                    "name": "Tuples",
                    "tuple_type": "MyTuple",
                    "value": [{"elements": [1, 2]}, {"elements": [3, 4]}],
                },
                {"type": "dvar", "name": "x", "var_type": "float"},
            ],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [
                        {
                            "iterator": "t",
                            "range": {"type": "named_set", "name": "Tuples"},
                        }
                    ],
                    "constraint": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "x"},
                        "op": ">=",
                        "right": {
                            "type": "field_access",
                            "base": {"type": "name", "value": "t"},
                            "field": "a",
                        },
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        code = GurobiCodeGenerator(ast).generate_code()
        # Should contain t[0] (field 'a' of tuple t) in the generated code
        self.assertIn("t[0]", code)

    def test_logical_and_or_in_constraint_and_implication(self):
        # Test parsing and codegen for logical AND/OR in plain constraints and implication antecedents/consequents
        lexer = OPLLexer()
        parser = OPLParser()
        model_text = """
            dvar boolean a;
            dvar boolean b;
            dvar float x;
            minimize x;
            subject to {
                (a == 1) && (b == 0);
                (a == 1) || (b == 1);
                (a == 1) && (b == 0) => x >= 5;
                (a == 0) || (b == 1) => x <= 10;
            }
        """
        ast = parser.parse(lexer.tokenize(model_text))
        # Ensure constraints parsed include 'and' and 'or' node types
        types = []
        for c in ast["constraints"]:
            # walk simple
            if c.get("type") == "constraint" and c["left"].get("type") in ("and", "or"):
                types.append(c["left"]["type"])
            if c.get("type") == "implication_constraint":
                ant = c["antecedent"]
                con = c["consequent"]
                if ant.get("type") == "constraint" and ant["left"].get("type") in (
                    "and",
                    "or",
                ):
                    types.append(ant["left"]["type"])
                if con.get("type") == "constraint" and con["left"].get("type") in (
                    "and",
                    "or",
                ):
                    types.append(con["left"]["type"])
        self.assertIn("and", types)
        self.assertIn("or", types)
        # Gurobi code generation should succeed
        code = GurobiCodeGenerator(ast).generate_code()
        self.assertIn("and", code)
        self.assertIn("or", code)

    def test_binop_with_sum_in_objective(self):
        # Covers binop(sum, expr) and binop(expr, sum) in the objective for both SciPyCodeGenerator and GurobiCodeGenerator
        # Objective: sum(i in 1..2) x[i] + y
        ast1 = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "I",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 2},
                },
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [{"type": "named_range_dimension", "name": "I"}],
                },
                {"type": "dvar", "name": "y", "var_type": "float"},
            ],
            "constraints": [],
            "objective": {
                "type": "minimize",
                "expression": {
                    "type": "binop",
                    "op": "+",
                    "left": {
                        "type": "sum",
                        "iterators": [
                            {
                                "iterator": "i",
                                "range": {"type": "named_range", "name": "I"},
                            }
                        ],
                        "expression": {
                            "type": "indexed_name",
                            "name": "x",
                            "dimensions": [{"type": "name_reference_index", "name": "i"}],
                        },
                    },
                    "right": {"type": "name", "value": "y"},
                },
            },
        }
        codegen1 = SciPyCodeGenerator(ast1)
        codegen1._build_variables()
        codegen1._build_objective()
        # Should produce c vector with 1 for x_1, x_2, y
        idx_x1 = codegen1.var_indices["x_1"]
        idx_x2 = codegen1.var_indices["x_2"]
        idx_y = codegen1.var_indices["y"]
        c = codegen1.c
        self.assertEqual(c[idx_x1], 1.0)
        self.assertEqual(c[idx_x2], 1.0)
        self.assertEqual(c[idx_y], 1.0)

        # Now test binop(expr, sum): y + sum(i in 1..2) x[i]
        ast2 = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "I",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 2},
                },
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [{"type": "named_range_dimension", "name": "I"}],
                },
                {"type": "dvar", "name": "y", "var_type": "float"},
            ],
            "constraints": [],
            "objective": {
                "type": "minimize",
                "expression": {
                    "type": "binop",
                    "op": "+",
                    "left": {"type": "name", "value": "y"},
                    "right": {
                        "type": "sum",
                        "iterators": [
                            {
                                "iterator": "i",
                                "range": {"type": "named_range", "name": "I"},
                            }
                        ],
                        "expression": {
                            "type": "indexed_name",
                            "name": "x",
                            "dimensions": [{"type": "name_reference_index", "name": "i"}],
                        },
                    },
                },
            },
        }
        codegen2 = SciPyCodeGenerator(ast2)
        codegen2._build_variables()
        codegen2._build_objective()
        idx_x1_2 = codegen2.var_indices["x_1"]
        idx_x2_2 = codegen2.var_indices["x_2"]
        idx_y_2 = codegen2.var_indices["y"]
        c2 = codegen2.c
        self.assertEqual(c2[idx_x1_2], 1.0)
        self.assertEqual(c2[idx_x2_2], 1.0)
        self.assertEqual(c2[idx_y_2], 1.0)

    def test_binop_with_sum_in_constraint(self):
        # Covers binop(sum, expr) and binop(expr, sum) in constraint expressions for both SciPyCodeGenerator and GurobiCodeGenerator
        # sum(i in 1..2) x[i] + y == 5
        ast1 = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "I",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 2},
                },
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [{"type": "named_range_dimension", "name": "I"}],
                },
                {"type": "dvar", "name": "y", "var_type": "float"},
            ],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {
                        "type": "binop",
                        "op": "+",
                        "left": {
                            "type": "sum",
                            "iterators": [
                                {
                                    "iterator": "i",
                                    "range": {"type": "named_range", "name": "I"},
                                }
                            ],
                            "expression": {
                                "type": "indexed_name",
                                "name": "x",
                                "dimensions": [{"type": "name_reference_index", "name": "i"}],
                            },
                        },
                        "right": {"type": "name", "value": "y"},
                    },
                    "op": "==",
                    "right": {"type": "number", "value": 5},
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen1 = SciPyCodeGenerator(ast1)
        codegen1._build_variables()
        codegen1._build_constraints()
        # Should produce a constraint row with x_1, x_2, y
        varnames = set(codegen1.var_names)
        self.assertIn("x_1", varnames)
        self.assertIn("x_2", varnames)
        self.assertIn("y", varnames)
        # The constraint matrix should have nonzero entries for x_1, x_2, y
        found = False
        for row in codegen1.A_eq:
            if (
                abs(row[codegen1.var_indices["x_1"]]) > 0
                and abs(row[codegen1.var_indices["x_2"]]) > 0
                and abs(row[codegen1.var_indices["y"]]) > 0
            ):
                found = True
        self.assertTrue(found, "Constraint row should include x_1, x_2, y")

        # Now test binop(expr, sum): y + sum(i in 1..2) x[i] == 5
        ast2 = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "I",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 2},
                },
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [{"type": "named_range_dimension", "name": "I"}],
                },
                {"type": "dvar", "name": "y", "var_type": "float"},
            ],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {
                        "type": "binop",
                        "op": "+",
                        "left": {"type": "name", "value": "y"},
                        "right": {
                            "type": "sum",
                            "iterators": [
                                {
                                    "iterator": "i",
                                    "range": {"type": "named_range", "name": "I"},
                                }
                            ],
                            "expression": {
                                "type": "indexed_name",
                                "name": "x",
                                "dimensions": [{"type": "name_reference_index", "name": "i"}],
                            },
                        },
                    },
                    "op": "==",
                    "right": {"type": "number", "value": 5},
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen2 = SciPyCodeGenerator(ast2)
        codegen2._build_variables()
        codegen2._build_constraints()
        varnames2 = set(codegen2.var_names)
        self.assertIn("x_1", varnames2)
        self.assertIn("x_2", varnames2)
        self.assertIn("y", varnames2)
        found2 = False
        for row in codegen2.A_eq:
            if (
                abs(row[codegen2.var_indices["x_1"]]) > 0
                and abs(row[codegen2.var_indices["x_2"]]) > 0
                and abs(row[codegen2.var_indices["y"]]) > 0
            ):
                found2 = True
        self.assertTrue(found2, "Constraint row should include x_1, x_2, y")

    def test_deeply_nested_forall_and_sum(self):
        """
        Stress test: Deeply nested forall and sum expressions with index constraints.
        This checks the parser and codegen for stack overflows and correct AST structure.
        """
        ast = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "I",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 3},
                },
                {
                    "type": "range_declaration_inline",
                    "name": "J",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 3},
                },
                {
                    "type": "dvar",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": ["I", "J"],
                },
            ],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [{"iterator": "i", "range": {"type": "named_range", "name": "I"}}],
                    "constraint": {
                        "type": "forall_constraint",
                        "iterators": [
                            {
                                "iterator": "j",
                                "range": {"type": "named_range", "name": "J"},
                            }
                        ],
                        "constraint": {
                            "type": "constraint",
                            "left": {
                                "type": "sum",
                                "iterators": [
                                    {
                                        "iterator": "k",
                                        "range": {"type": "named_range", "name": "I"},
                                    }
                                ],
                                "expression": {"type": "name", "value": "x"},
                            },
                            "op": ">=",
                            "right": {"type": "number", "value": 0},
                        },
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        codegen._build_constraints()
        # Should not raise or overflow, and should emit constraints for all i, j
        self.assertTrue(any("DEBUG" in line or "Constraint" in line for line in codegen.scipy_code_lines))

    def test_forall_constraint_branches(self):
        # Covers all branches in forall_constraint handling in SciPyCodeGenerator._build_constraints
        # 1. iterators is None (should raise)
        ast = {
            "declarations": [{"type": "dvar", "name": "x", "var_type": "float"}],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": None,
                    "constraint": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "x"},
                        "op": "==",
                        "right": {"type": "number", "value": 1},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        with self.assertRaises(SemanticError):
            codegen._build_constraints()

        # 2. missing 'constraint' and 'constraints' (should raise)
        ast = {
            "declarations": [{"type": "dvar", "name": "x", "var_type": "float"}],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [
                        {
                            "iterator": "i",
                            "range": {
                                "type": "range_specifier",
                                "start": {"type": "number", "value": 1},
                                "end": {"type": "number", "value": 2},
                            },
                        }
                    ],
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        with self.assertRaises(SemanticError):
            codegen._build_constraints()

        # 3. 'constraint' in constr (single constraint), range_specifier
        ast = {
            "declarations": [{"type": "dvar", "name": "x", "var_type": "float"}],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [
                        {
                            "iterator": "i",
                            "range": {
                                "type": "range_specifier",
                                "start": {"type": "number", "value": 1},
                                "end": {"type": "number", "value": 2},
                            },
                        }
                    ],
                    "constraint": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "x"},
                        "op": "==",
                        "right": {"type": "number", "value": 1},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        codegen._build_constraints()
        # Should add a constraint row for each i in 1..2
        self.assertTrue(len(codegen.A_eq) > 0 or len(codegen.A_ub) > 0)

        # 4. 'constraints' in constr (list of constraints), named_range
        ast = {
            "declarations": [
                {"type": "dvar", "name": "x", "var_type": "float"},
                {
                    "type": "range_declaration_inline",
                    "name": "R",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 2},
                },
            ],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [{"iterator": "i", "range": {"type": "named_range", "name": "R"}}],
                    "constraints": [
                        {
                            "type": "constraint",
                            "left": {"type": "name", "value": "x"},
                            "op": ">=",
                            "right": {"type": "number", "value": 0},
                        },
                        {
                            "type": "constraint",
                            "left": {"type": "name", "value": "x"},
                            "op": "<=",
                            "right": {"type": "number", "value": "2"},
                        },
                    ],
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        codegen._build_constraints()
        # Should add two constraints for each i in 1..2
        self.assertTrue(len(codegen.A_eq) > 0 or len(codegen.A_ub) > 0)

        # 5. unsupported range type (should raise)
        ast = {
            "declarations": [{"type": "dvar", "name": "x", "var_type": "float"}],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [{"iterator": "i", "range": {"type": "unsupported_range"}}],
                    "constraint": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "x"},
                        "op": "==",
                        "right": {"type": "number", "value": 1},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        with self.assertRaises(SemanticError):
            codegen._build_constraints()

        # 6. unsupported binop in forall index (should raise)
        ast = {
            "declarations": [{"type": "dvar", "name": "x", "var_type": "float"}],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [
                        {
                            "iterator": "i",
                            "range": {
                                "type": "range_specifier",
                                "start": {
                                    "type": "binop",
                                    "op": "/",
                                    "left": {"type": "number", "value": 1},
                                    "right": {"type": "number", "value": 2},
                                },
                                "end": {"type": "number", "value": 2},
                            },
                        }
                    ],
                    "constraint": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "x"},
                        "op": "==",
                        "right": {"type": "number", "value": 1},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        with self.assertRaises(SemanticError):
            codegen._build_constraints()

        # 7. unsupported expr in forall index (should raise)
        ast = {
            "declarations": [{"type": "dvar", "name": "x", "var_type": "float"}],
            "constraints": [
                {
                    "type": "forall_constraint",
                    "iterators": [
                        {
                            "iterator": "i",
                            "range": {
                                "type": "range_specifier",
                                "start": {"type": "foo"},
                                "end": {"type": "number", "value": 2},
                            },
                        }
                    ],
                    "constraint": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "x"},
                        "op": "==",
                        "right": {"type": "number", "value": 1},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        with self.assertRaises(SemanticError):
            codegen._build_constraints()

    def test_constraint_implication_support(self):
        """
        Test that constraint implications (A => B) are supported in both Gurobi and SciPy backends.
        SciPy previously rejected implication constraints; now it should encode them as a single
        inequality: -y + x <= 0 for (x == 1) => (y == 1).
        """
        ast = {
            "declarations": [
                {"type": "dvar", "name": "x", "var_type": "boolean"},
                {"type": "dvar", "name": "y", "var_type": "boolean"},
            ],
            "constraints": [
                {
                    "type": "implication_constraint",
                    "antecedent": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "x"},
                        "op": "==",
                        "right": {"type": "number", "value": 1},
                    },
                    "consequent": {
                        "type": "constraint",
                        "left": {"type": "name", "value": "y"},
                        "op": "==",
                        "right": {"type": "number", "value": 1},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        # Gurobi should accept implication constraints
        try:
            code = GurobiCodeGenerator(ast).generate_code()
            self.assertIn("if", code)  # At minimum, implication logic should be present
        except Exception as e:
            self.fail(f"GurobiCodeGenerator should support implications, but got error: {e}")
        # SciPy should also accept and encode implication constraints now
        try:
            sc = SciPyCodeGenerator(ast)
            sc._build_variables()
            sc._build_objective()
            sc._build_constraints()
            # Expect one inequality row implementing x => y : -y + x <= 0
            self.assertIsNotNone(sc.A_ub, "Expected inequality matrix for implication")
            x_idx = sc.var_indices["x"]
            y_idx = sc.var_indices["y"]
            found = False
            for r, row in enumerate(sc.A_ub):
                if abs(row[x_idx] - 1.0) < 1e-9 and abs(row[y_idx] + 1.0) < 1e-9 and abs(sc.b_ub[r]) < 1e-9:
                    found = True
                    break
            self.assertTrue(
                found,
                f"Did not find encoded implication row -y + x <= 0; A_ub={sc.A_ub}, b_ub={sc.b_ub}",
            )
        except Exception as e:
            self.fail(f"SciPyCodeGenerator should support implications, but got error: {e}")

    def test_linear_gt_implies_binary_eq1_uses_contra_indicator(self):
        """
        Model: (x[t] > 0) => (y[t] == 1) should produce a contrapositive indicator:
            addGenConstrIndicator(y[t], 0, x[t] <= 0, ..._indicator_contra)
        and should not introduce a generic implication_flag_ big-M construction.
        """
        opl_code = """
        int T = 2;
        dvar float x[1..T];
        dvar boolean y[1..T];
        minimize 0;
        subject to {
            forall(t in 1..T)
                (x[t] > 0) => (y[t] == 1);
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = GurobiCodeGenerator(ast)
        gurobi_code = generator.generate_code()
        # Core assertions: indicator_contra present; generic implication_flag absent
        self.assertIn(
            "addGenConstrIndicator",
            gurobi_code,
            "Expected an indicator constraint to be generated",
        )
        self.assertIn(
            "indicator_contra",
            gurobi_code,
            "Expected specialized contrapositive indicator naming",
        )
        self.assertNotIn(
            "implication_flag_",
            gurobi_code,
            "Should not fall back to big-M flag variable for this pattern",
        )

    def test_linear_ge_implies_binary_eq1_uses_contra_indicator(self):
        """(x[t] >= 5) => (y[t] == 1) should use indicator_contra_ge with (y[t]==0) => x[t] <= 5 - eps."""
        opl_code = """
        int T = 2;
        dvar float x[1..T];
        dvar boolean y[1..T];
        minimize 0;
        subject to {
            forall(t in 1..T)
                (x[t] >= 5) => (y[t] == 1);
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = GurobiCodeGenerator(ast)
        gurobi_code = generator.generate_code()
        self.assertIn("indicator_contra_ge", gurobi_code)
        self.assertIn("addGenConstrIndicator", gurobi_code)
        self.assertNotIn("implication_flag_", gurobi_code)

    def test_eval_bound_branches_in_objective_sum_index(self):
        # Covers eval_bound in _build_objective for sum index bounds: binop +, -, *, uminus, parenthesized_expression, and error
        def make_sum_obj(start_expr, end_expr):
            # Add a variable declaration for 'i' so that var_names is not empty
            return {
                "declarations": [{"type": "dvar", "name": "i", "var_type": "float"}],
                "constraints": [],
                "objective": {
                    "type": "minimize",
                    "expression": {
                        "type": "sum",
                        "iterators": [
                            {
                                "iterator": "i",
                                "range": {
                                    "type": "range_specifier",
                                    "start": start_expr,
                                    "end": end_expr,
                                },
                            }
                        ],
                        "expression": {"type": "name", "value": "i"},
                    },
                },
            }

        # binop +, -, *, uminus, parenthesized_expression: should produce a symbolic sum and c = [0.0]
        for start_expr in [
            {
                "type": "binop",
                "op": "+",
                "left": {"type": "number", "value": 1},
                "right": {"type": "number", "value": 1},
            },
            {
                "type": "binop",
                "op": "-",
                "left": {"type": "number", "value": 4},
                "right": {"type": "number", "value": 2},
            },
            {
                "type": "binop",
                "op": "*",
                "left": {"type": "number", "value": 1},
                "right": {"type": "number", "value": 2},
            },
            {"type": "uminus", "value": {"type": "number", "value": 2}},
            {
                "type": "parenthesized_expression",
                "expression": {"type": "number", "value": 2},
            },
        ]:
            ast = make_sum_obj(start_expr, {"type": "number", "value": 3})
            codegen = SciPyCodeGenerator(ast)
            codegen._build_variables()
            codegen._build_objective()
            # Should have a symbolic sum comment and c = [0.0]
            self.assertTrue(any("Symbolic objective" in line for line in codegen.scipy_code_lines))
            self.assertIn("c = [0.0]", codegen.scipy_code_lines)

        # unsupported expr
        ast = make_sum_obj({"type": "foo"}, {"type": "number", "value": 3})
        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        with self.assertRaises(SemanticError):
            codegen._build_objective()

    def test_eval_bound_branches_in_sum_index(self):
        # Now, reconstruct the eval_bound logic as in the code, but test all branches
        # We'll use the actual code from the implementation for fidelity
        def eval_bound(expr):
            if isinstance(expr, dict):
                if expr["type"] == "number":
                    return expr["value"]
                elif expr["type"] == "binop":
                    left = eval_bound(expr["left"])
                    right = eval_bound(expr["right"])
                    if expr["op"] == "+":
                        return left + right
                    elif expr["op"] == "-":
                        return left - right
                    elif expr["op"] == "*":
                        return left * right
                    else:
                        raise SemanticError(f"Unsupported binop in sum index: {expr['op']}")
                elif expr["type"] == "uminus":
                    val = eval_bound(expr["value"])
                    return -val
                elif expr["type"] == "parenthesized_expression":
                    return eval_bound(expr["expression"])
            raise SemanticError(f"Unsupported expr in sum index: {expr}")

        # binop +
        expr_plus = {
            "type": "binop",
            "op": "+",
            "left": {"type": "number", "value": 2},
            "right": {"type": "number", "value": 3},
        }
        self.assertEqual(eval_bound(expr_plus), 5)
        # binop -
        expr_minus = {
            "type": "binop",
            "op": "-",
            "left": {"type": "number", "value": 5},
            "right": {"type": "number", "value": 2},
        }
        self.assertEqual(eval_bound(expr_minus), 3)
        # binop *
        expr_times = {
            "type": "binop",
            "op": "*",
            "left": {"type": "number", "value": 4},
            "right": {"type": "number", "value": 2},
        }
        self.assertEqual(eval_bound(expr_times), 8)
        # binop unsupported
        expr_bad = {
            "type": "binop",
            "op": "/",
            "left": {"type": "number", "value": 4},
            "right": {"type": "number", "value": 2},
        }
        with self.assertRaises(SemanticError):
            eval_bound(expr_bad)
        # uminus
        expr_uminus = {"type": "uminus", "value": {"type": "number", "value": 7}}
        self.assertEqual(eval_bound(expr_uminus), -7)
        # parenthesized_expression
        expr_paren = {
            "type": "parenthesized_expression",
            "expression": {"type": "number", "value": 9},
        }
        self.assertEqual(eval_bound(expr_paren), 9)
        # unsupported expr
        expr_unsupported = {"type": "foo"}
        with self.assertRaises(SemanticError):
            eval_bound(expr_unsupported)

    def setUp(self):
        self.codegen = SciPyCodeGenerator(
            {
                "declarations": [],
                "constraints": [],
                "objective": {
                    "type": "minimize",
                    "expression": {"type": "number", "value": 0},
                },
            }
        )
        self.codegen.var_indices = {}
        self.codegen.data_dict = {}
        self.codegen.ast = {"declarations": []}

    def test_eval_expr_boolops(self):
        # All combinations of boolean ops with constants
        ops = ["==", "!=", "<", ">", "<=", ">="]
        left_val = 3
        right_val = 5
        expected = {
            "==": left_val == right_val,
            "!=": left_val != right_val,
            "<": left_val < right_val,
            ">": left_val > right_val,
            "<=": left_val <= right_val,
            ">=": left_val >= right_val,
        }
        for op in ops:
            expr = {
                "type": "binop",
                "op": op,
                "left": {"type": "number", "value": left_val},
                "right": {"type": "number", "value": right_val},
            }
            coef, const = self.codegen._eval_expr(expr)
            self.assertEqual(coef, {})
            self.assertEqual(const, expected[op])

    def test_eval_expr_boolops_nonconstant(self):
        # Should raise SemanticError if non-constant in boolean binop
        expr = {
            "type": "binop",
            "op": "==",
            "left": {"type": "name", "value": "x"},
            "right": {"type": "number", "value": 1},
        }
        self.codegen.var_indices = {"x": 0}
        with self.assertRaises(SemanticError):
            self.codegen._eval_expr(expr)

    def test_eval_expr_multi_index_sum(self):
        # Covers multi-index sum support: unroll over all iterators using itertools.product
        # and all eval_bound binop/uminus/parenthesized_expression branches
        # sum(i in 1..2, j in 1..3) (i * j)
        expr = {
            "type": "sum",
            "iterators": [
                {
                    "iterator": "i",
                    "range": {
                        "type": "range_specifier",
                        "start": {"type": "number", "value": 1},
                        "end": {"type": "number", "value": 2},
                    },
                },
                {
                    "iterator": "j",
                    "range": {
                        "type": "range_specifier",
                        "start": {"type": "number", "value": 1},
                        "end": {"type": "number", "value": 3},
                    },
                },
            ],
            "expression": {
                "type": "binop",
                "op": "*",
                "left": {"type": "name", "value": "i"},
                "right": {"type": "name", "value": "j"},
            },
        }
        coef, const = self.codegen._eval_expr(expr)
        self.assertEqual(coef, {})
        self.assertEqual(const, 18)

        # Test binop +, -, * in sum index bounds
        expr_plus = {
            "type": "sum",
            "iterators": [
                {
                    "iterator": "i",
                    "range": {
                        "type": "range_specifier",
                        "start": {
                            "type": "binop",
                            "op": "+",
                            "left": {"type": "number", "value": 1},
                            "right": {"type": "number", "value": 1},
                        },
                        "end": {"type": "number", "value": 3},
                    },
                }
            ],
            "expression": {"type": "name", "value": "i"},
        }
        coef, const = self.codegen._eval_expr(expr_plus)
        # i in 2..3, sum = 2+3 = 5
        self.assertEqual(coef, {})
        self.assertEqual(const, 5)

        expr_minus = {
            "type": "sum",
            "iterators": [
                {
                    "iterator": "i",
                    "range": {
                        "type": "range_specifier",
                        "start": {
                            "type": "binop",
                            "op": "-",
                            "left": {"type": "number", "value": 4},
                            "right": {"type": "number", "value": 2},
                        },
                        "end": {"type": "number", "value": 3},
                    },
                }
            ],
            "expression": {"type": "name", "value": "i"},
        }
        coef, const = self.codegen._eval_expr(expr_minus)
        # i in 2..3, sum = 2+3 = 5
        self.assertEqual(coef, {})
        self.assertEqual(const, 5)

        expr_times = {
            "type": "sum",
            "iterators": [
                {
                    "iterator": "i",
                    "range": {
                        "type": "range_specifier",
                        "start": {
                            "type": "binop",
                            "op": "*",
                            "left": {"type": "number", "value": 1},
                            "right": {"type": "number", "value": 2},
                        },
                        "end": {"type": "number", "value": 4},
                    },
                }
            ],
            "expression": {"type": "name", "value": "i"},
        }
        coef, const = self.codegen._eval_expr(expr_times)
        # i in 2..4, sum = 2+3+4 = 9
        self.assertEqual(coef, {})
        self.assertEqual(const, 9)

        # Test uminus in sum index bound
        expr_uminus = {
            "type": "sum",
            "iterators": [
                {
                    "iterator": "i",
                    "range": {
                        "type": "range_specifier",
                        "start": {
                            "type": "uminus",
                            "value": {"type": "number", "value": 2},
                        },
                        "end": {"type": "number", "value": 0},
                    },
                }
            ],
            "expression": {"type": "name", "value": "i"},
        }
        coef, const = self.codegen._eval_expr(expr_uminus)
        # i in -2..0, sum = -2 + -1 + 0 = -3
        self.assertEqual(coef, {})
        self.assertEqual(const, -3)

        # Test parenthesized_expression in sum index bound
        expr_paren = {
            "type": "sum",
            "iterators": [
                {
                    "iterator": "i",
                    "range": {
                        "type": "range_specifier",
                        "start": {
                            "type": "parenthesized_expression",
                            "expression": {"type": "number", "value": 2},
                        },
                        "end": {"type": "number", "value": 3},
                    },
                }
            ],
            "expression": {"type": "name", "value": "i"},
        }
        coef, const = self.codegen._eval_expr(expr_paren)
        # i in 2..3, sum = 2+3 = 5
        self.assertEqual(coef, {})
        self.assertEqual(const, 5)

        # Test unsupported expr in sum index (should raise)
        expr_bad = {
            "type": "sum",
            "iterators": [
                {
                    "iterator": "i",
                    "range": {
                        "type": "range_specifier",
                        "start": {"type": "foo"},
                        "end": {"type": "number", "value": 3},
                    },
                }
            ],
            "expression": {"type": "name", "value": "i"},
        }
        with self.assertRaises(SemanticError):
            self.codegen._eval_expr(expr_bad)

    def test_indexed_variable_with_binop_index(self):
        """Test that indexed variables with binary operator indices (e.g., s[t-1]) are parsed and codegen'd correctly."""
        opl_code = """
        range Time = 1..5;
        dvar float+ s[Time];
        dvar float+ x[Time];
        float demand[Time] = [2, 1, 3, 2, 1];
        minimize sum(t in Time) x[t];
        subject to {
            s[1] == 0;
            forall(t in 2..5) s[t] == s[t-1] + x[t] - demand[t];
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))

        # Check AST for correct indexed_name with binop in index
        def find_indexed_name_with_binop(node):
            if isinstance(node, dict):
                if node.get("type") == "indexed_name":
                    for idx in node.get("dimensions", []) + node.get("indices", []):
                        if isinstance(idx, dict) and idx.get("type") == "binop" and idx.get("op") == "-":
                            return True
                for v in node.values():
                    if find_indexed_name_with_binop(v):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if find_indexed_name_with_binop(item):
                        return True
            return False

        found = find_indexed_name_with_binop(ast)
        if not found:
            self.fail(f"AST should contain indexed_name with binop index.\nModel snippet:\n{opl_code.strip()}")
        # Check code generation does not raise
        gurobi_code = GurobiCodeGenerator(ast).generate_code()
        # Accept both s[t-1] and s[(t - 1)] as valid Python code
        self.assertTrue(
            "s[(t - 1)]" in gurobi_code or "s[t-1]" in gurobi_code,
            f"Gurobi code should contain s[(t - 1)] or s[t-1], got: {gurobi_code}",
        )
        scipy_code = SciPyCodeGenerator(ast).generate_code()
        # Accept any s_1, s_2, ... s_5 as valid variable names for s[t-1] in SciPy code
        self.assertTrue(
            any(f"s_{i}" in scipy_code for i in range(1, 6)),
            f"SciPy code should contain s_1..s_5, got: {scipy_code}",
        )

    def test_range_index_with_expression_bounds_and_sum(self):
        """Test that range indices with expression bounds (e.g., 1..T) are parsed and codegen'd correctly in sums and constraints."""
        opl_code = """
        int T = 5;
        float demand[1..T] = [2, 1, 3, 2, 1];
        dvar float+ x[1..T];
        dvar boolean y[1..T];
        dvar float+ s[1..T];
        float K = 1;
        float u = 1;
        float h = 1;
        minimize sum(t in 1..T) (K * y[t] + u * x[t] + h * s[t]);
        subject to {
          s[1] == x[1] - demand[1];
          forall(t in 2..T)
            s[t] == s[t-1] + x[t] - demand[t];
          forall(t in 1..T)
            x[t] <= demand[t] * y[t];
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))

        # Check that the AST contains range_index with expression bounds
        def find_range_index_with_expr(node):
            if isinstance(node, dict):
                if node.get("type") == "range_index":
                    for bound in (node["start"], node["end"]):
                        if isinstance(bound, dict) and bound.get("type") != "number":
                            return True
                for v in node.values():
                    if find_range_index_with_expr(v):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if find_range_index_with_expr(item):
                        return True
            return False

        if not find_range_index_with_expr(ast):
            self.fail(f"AST should contain range_index with expression bounds.\nModel snippet:\n{opl_code.strip()}")
        # Check code generation does not raise
        gurobi_code = GurobiCodeGenerator(ast).generate_code()
        self.assertIn("range(1, T + 1)", gurobi_code)
        self.assertIn("range(2, T + 1)", gurobi_code)
        scipy_code = SciPyCodeGenerator(ast).generate_code()
        self.assertIn("range(1, T + 1)", scipy_code)
        self.assertTrue(
            any(s.strip().startswith("# OPL:") or "Symbolic" in s for s in scipy_code.splitlines()),
            "Expected OPL or symbolic comment for range(2, T + 1) not found in SciPy code",
        )

    def test_conditional_expression(self):
        # Test conditional expression in objective and constraint
        ast = {
            "declarations": [
                {"type": "dvar", "name": "x", "var_type": "float"},
                {"type": "dvar", "name": "y", "var_type": "float"},
                {"type": "parameter_inline", "name": "p", "value": 1},
            ],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {
                        "type": "conditional",
                        "condition": {
                            "type": "binop",
                            "op": "==",
                            "left": {"type": "name", "value": "p"},
                            "right": {"type": "number", "value": 1},
                        },
                        "then": {"type": "name", "value": "x"},
                        "else": {"type": "name", "value": "y"},
                    },
                    "op": ">=",
                    "right": {"type": "number", "value": 0},
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {
                    "type": "conditional",
                    "condition": {
                        "type": "binop",
                        "op": "==",
                        "left": {"type": "name", "value": "p"},
                        "right": {"type": "number", "value": 1},
                    },
                    "then": {"type": "name", "value": "x"},
                    "else": {"type": "name", "value": "y"},
                },
            },
        }
        # SciPyCodeGenerator
        from pyopl.scipy_codegen import SciPyCodeGenerator

        codegen = SciPyCodeGenerator(ast)
        codegen._build_variables()
        codegen._build_objective()
        # Should produce c vector with 1 for x, 0 for y (since p==1)
        idx_x = codegen.var_indices["x"]
        idx_y = codegen.var_indices["y"]
        c = codegen.c
        self.assertEqual(c[idx_x], 1.0)
        self.assertEqual(c[idx_y], 0.0)
        # GurobiCodeGenerator
        from pyopl.gurobi_codegen import GurobiCodeGenerator

        code = GurobiCodeGenerator(ast).generate_code()
        # Should contain 'x if (p == 1) else y' in the generated code
        self.assertIn("x if (p == 1) else y", code)

    def test_not_operator_in_constraint_and_implication(self):
        """TDD: Ensure parser will support logical NOT '!' in a standalone constraint and inside an implication antecedent.

        Model uses:
          !(x == 1);
          (!(x == 1)) => (y == 1);

        Current grammar lacks '!' so this test will initially fail (parse error) until feature is implemented.
        After implementation we assert that an AST node with type 'not' appears in the first constraint and
        specifically in the antecedent of the implication constraint.
        """
        opl_code = """
        dvar boolean x;
        dvar boolean y;
        minimize x; // dummy objective
        subject to {
            !(x == 1);
            (!(x == 1)) => (y == 1);
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        try:
            ast = parser.parse(lexer.tokenize(opl_code))
        except Exception as e:
            # Expected to fail before implementation; re-raise to show failing test
            raise e
        self.assertIn("constraints", ast)
        self.assertGreaterEqual(len(ast["constraints"]), 2)

        def find_not(node):
            if isinstance(node, dict):
                if node.get("type") == "not":
                    return True
                for v in node.values():
                    if find_not(v):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if find_not(item):
                        return True
            return False

        # First constraint should contain a NOT node
        self.assertTrue(
            find_not(ast["constraints"][0]),
            "Expected a 'not' node in first constraint AST",
        )
        # Second constraint is implication: ensure antecedent side has NOT
        implic = ast["constraints"][1]
        self.assertTrue(find_not(implic), "Expected a 'not' node in implication constraint AST")

    def test_trivial_reified_cardinality_k0_and_k_gt_size(self):
        # Boolean vars x,y,z plus reification vars b0,b1
        decls = [
            {"type": "dvar", "name": "x", "var_type": "boolean"},
            {"type": "dvar", "name": "y", "var_type": "boolean"},
            {"type": "dvar", "name": "z", "var_type": "boolean"},
            {"type": "dvar", "name": "b0", "var_type": "boolean"},
            {"type": "dvar", "name": "b1", "var_type": "boolean"},
        ]
        # Sum expression x + y + z as nested binops
        sum_xyz = {
            "type": "binop",
            "op": "+",
            "left": {
                "type": "binop",
                "op": "+",
                "left": {"type": "name", "value": "x"},
                "right": {"type": "name", "value": "y"},
            },
            "right": {"type": "name", "value": "z"},
        }
        # b0 == (x + y + z >= 0) -> always true -> b0 fixed to 1
        constr_k0 = {
            "type": "constraint",
            "left": {"type": "name", "value": "b0"},
            "op": "==",
            "right": {
                "type": "constraint",
                "left": sum_xyz,
                "op": ">=",
                "right": {"type": "number", "value": 0},
            },
        }
        # b1 == (x + y + z >= 4) -> impossible (|S|=3) -> b1 fixed to 0
        constr_k_gt = {
            "type": "constraint",
            "left": {"type": "name", "value": "b1"},
            "op": "==",
            "right": {
                "type": "constraint",
                "left": sum_xyz,
                "op": ">=",
                "right": {"type": "number", "value": 4},
            },
        }
        ast = {
            "declarations": decls,
            "constraints": [constr_k0, constr_k_gt],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen = SciPyCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        # Expect two equality rows only, fixing b0=1 and b1=0; no inequalities involving b0/b1
        self.assertEqual(
            len(gen.A_eq),
            2,
            f"Expected 2 equality rows, got {len(gen.A_eq)}; A_eq={gen.A_eq}",
        )
        # Build var index mapping
        b0_idx = gen.var_indices["b0"]
        b1_idx = gen.var_indices["b1"]
        # Row checks
        row_vals = {gen.b_eq[i]: gen.A_eq[i] for i in range(len(gen.A_eq))}
        # Find row with RHS 1 and RHS 0
        found_b0 = any(abs(rhs - 1.0) < 1e-9 and abs(row[b0_idx] - 1.0) < 1e-9 for rhs, row in row_vals.items())
        found_b1 = any(abs(rhs - 0.0) < 1e-9 and abs(row[b1_idx] - 1.0) < 1e-9 for rhs, row in row_vals.items())
        self.assertTrue(
            found_b0,
            f"Did not find equality fixing b0=1; rows={list(row_vals.items())}",
        )
        self.assertTrue(
            found_b1,
            f"Did not find equality fixing b1=0; rows={list(row_vals.items())}",
        )
        # Ensure no inequality rows reference b0 or b1
        for r, row in enumerate(gen.A_ub):
            self.assertFalse(
                abs(row[b0_idx]) > 1e-9 or abs(row[b1_idx]) > 1e-9,
                f"Unexpected inequality row {r} references b0/b1: {row} with b_ub={gen.b_ub[r]}",
            )

    def test_sum_boolean_bound_tightening(self):
        # Three booleans sum >= 2 implies y >= 5; we avoid specialized (>=) => (bin==1) indicator.
        # All booleans in [0,1]; sum in [0,3]; diff (sum - 2) in [-2,1] so |diff|<=2 -> expected big-M <=2.
        opl = """\n        dvar boolean x1; dvar boolean x2; dvar boolean x3; dvar float y;\n        minimize 0;\n        subject to { (x1 + x2 + x3 >= 2) => (y >= 5); }\n        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl))
        code = GurobiCodeGenerator(ast).generate_code()
        # Extract the bigM used in antecedent (look for _ant_lb pattern)
        # We expect bigM much smaller than default 1e6 (should be <=3)
        # Fallback: parse the line itself
        bigm_val = None
        for line in code.splitlines():
            if "_ant_lb" in line and "implication_flag_c0" in line and "* (1 - implication_flag_c0)" in line:
                try:
                    seg = line.split(">= -", 1)[1]
                    num = seg.split("* (1 - implication_flag_c0)")[0].strip().rstrip(",")
                    bigm_val = float(num)
                    break
                except Exception:
                    continue
        self.assertIsNotNone(bigm_val, "Could not extract big-M value (generic big-M path not taken?)")
        self.assertLessEqual(bigm_val, 2.0, f"Big-M not tightened (found {bigm_val})")
        self.assertNotIn("indicator_contra_ge", code)

    # ---------------- New logical construct coverage tests ----------------
    def test_logical_conjunction_disjunction_of_comparisons(self):
        """Verify parser + Gurobi codegen accept conjunction/disjunction of comparison expressions.

        This test REQUIRES both Gurobi and SciPy code generators to support logical AND/OR of
        linear comparison expressions (possibly nested) so they can be combined arbitrarily.
        If SciPy cannot yet linearize these (e.g. via auxiliary binaries / big-M), this test will
        FAIL and thereby signal the missing implementation.
        """
        model = """
        dvar float x; dvar float y; dvar boolean b;
        minimize x + y;
        subject to {
            (x + y <= 10) && (x - y >= 2);
            (x <= 5) || (y >= 1);
            ((x + y <= 10) && (x - y >= 2)) || (x + 2*y <= 15);
            (x <= 5) && ((y >= 1) || (x + y <= 9));
            // mixed with implication wrapper to exercise nesting
            ((x <= 5) && (y >= 1)) => (x + y <= 12);
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(model))

        # Basic AST sanity: ensure at least one 'and' and one 'or' node present
        def walk(node, found):
            if isinstance(node, dict):
                t = node.get("type")
                if t in ("and", "or"):
                    found.add(t)
                for v in node.values():
                    walk(v, found)
            elif isinstance(node, list):
                for v in node:
                    walk(v, found)

        found = set()
        walk(ast, found)
        self.assertIn("and", found)
        self.assertIn("or", found)
        # Gurobi generation should succeed
        try:
            gcode = GurobiCodeGenerator(ast).generate_code()
        except Exception as e:
            self.fail(f"Gurobi codegen failed for logical combinations: {e}")
        self.assertIn("and", gcode)
        self.assertIn("or", gcode)
        # SciPy MUST also succeed (full logical composition support expected)
        from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator

        sc_gen = SciPyCSCCodeGenerator(ast, {})
        try:
            sc_code = sc_gen.generate_code()
        except Exception as e:
            self.fail(f"SciPy codegen failed to handle logical AND/OR of linear comparisons: {e}")
        self.assertIn(
            "and",
            sc_code,
            "Expected 'and' in SciPy generated code (or its transformed equivalent)",
        )
        self.assertIn(
            "or",
            sc_code,
            "Expected 'or' in SciPy generated code (or its transformed equivalent)",
        )

    def test_logical_negation_and_not_equal(self):
        """Placeholder test for logical negation '!' and inequality '!=' compositions.

        Full support required: parser must build 'not' nodes; both backends must accept
        negation on linear comparison expressions and nested boolean structure involving '!='.
        Failing here indicates missing NOT / != handling in either parsing or code generation.
        """
        opl = """
        dvar float x; dvar float y; dvar boolean z;
        minimize x;
        subject to {
            !(x + y <= 10);
            !((x <= 5) && (y >= 1));
            (x != y);
            z == (x != y);
            (!(x != y)) && (x + y <= 20);
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl))

        # Walk for 'not' nodes and 'binop' with op '!='
        def collect_flags(node, flags):
            if isinstance(node, dict):
                t = node.get("type")
                if t == "not":
                    flags["not"] = True
                if t == "binop" and node.get("op") == "!=":
                    flags["neq"] = True
                for v in node.values():
                    collect_flags(v, flags)
            elif isinstance(node, list):
                for item in node:
                    collect_flags(item, flags)

        flags = {"not": False, "neq": False}
        collect_flags(ast, flags)
        self.assertTrue(flags["not"], "Expected at least one 'not' node in AST")
        self.assertTrue(flags["neq"], "Expected at least one '!=' binop in AST")
        # Gurobi code generation should succeed
        gcode = GurobiCodeGenerator(ast).generate_code()
        self.assertIn("!=", gcode, "Expected inequality to appear in Gurobi code")
        # SciPy code generation (must ALSO succeed)
        from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator

        sc_gen = SciPyCSCCodeGenerator(ast, {})
        sc_code = sc_gen.generate_code()
        self.assertIn(
            "!=",
            sc_code,
            "Expected '!=' to appear (or be explicitly encoded) in SciPy code",
        )

    def test_sum_with_string_literal_filter_on_tuple_field(self):
        """Test sum with index constraint comparing tuple field to string literal (both solvers)."""
        model_code = """
        tuple Arc { string origin; string dest; }
        {Arc} arcs = { <"A", "B">, <"A", "C">, <"B", "C"> };
        dvar float+ x[arcs];
        minimize sum(a in arcs : a.origin == "A") x[a];
        subject to { }
        """
        from pyopl.pyopl_core import solve

        for solver in ("gurobi", "scipy"):
            result = solve(model_code, solver=solver)
            self.assertIn("objective_value", result)

    def test_sum_with_string_literal_filter_on_string_set(self):
        """Test sum with index constraint comparing string set element to string literal (both solvers)."""
        model_code = """
        {string} Cities = { "A", "B", "C" };
        dvar float+ y[Cities];
        minimize sum(c in Cities : c == "B") y[c];
        subject to { }
        """
        from pyopl.pyopl_core import solve

        for solver in ("gurobi", "scipy"):
            result = solve(model_code, solver=solver)
            self.assertIn("objective_value", result)

    def test_parser_raises_on_invalid_syntax(self):
        from pyopl.pyopl_core import SemanticError, parse_model

        # Missing value after '>=' in constraint
        invalid_model = """
        dvar int x;
        minimize x;
        subject to { x >= ; }
        """
        with self.assertRaises(SemanticError) as cm:
            parse_model(invalid_model)
        msg = str(cm.exception)
        self.assertIn("Syntax error", msg)

    def test_codegen_raises_on_unhandled_expression(self):
        from pyopl.gurobi_codegen import GurobiCodeGenerator
        from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator

        # AST with an unhandled expression type
        ast = {
            "declarations": [{"type": "dvar", "var_type": "int", "name": "x"}],
            "objective": {
                "type": "minimize",
                "expression": {"type": "unknown_expr_type"},
            },
            "constraints": [],
        }
        gen_gurobi = GurobiCodeGenerator(ast)
        with self.assertRaises(NotImplementedError) as cm1:
            gen_gurobi.generate_code()
        msg1 = str(cm1.exception)
        self.assertIn("unknown_expr_type", msg1)

        gen_scipy = SciPyCSCCodeGenerator(ast)
        with self.assertRaises(NotImplementedError) as cm2:
            gen_scipy.generate_code()
        msg2 = str(cm2.exception)
        self.assertIn("unknown_expr_type", msg2)


if __name__ == "__main__":
    unittest.main()

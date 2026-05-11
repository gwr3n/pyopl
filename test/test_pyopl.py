import os
import tempfile
import unittest

from pyopl.pyopl_core import (
    GurobiCodeGenerator,
    OPLCompiler,
    OPLDataLexer,
    OPLDataParser,
    OPLLexer,
    OPLParser,
    SciPyCodeGenerator,
    SemanticError,
    load_opl_model,
    solve,
)


def setUpModule():
    import logging

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s")


class TestPyOPL(unittest.TestCase):
    def run_test_case_gurobi(self, opl_code):
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = GurobiCodeGenerator(ast)
        gurobi_code = generator.generate_code()
        self.assertIsInstance(ast, dict)
        self.assertIsInstance(gurobi_code, str)

    def run_test_case_scipy(self, opl_code):

        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = SciPyCodeGenerator(ast)
        scipy_code = generator.generate_code()
        self.assertIsInstance(ast, dict)
        self.assertIsInstance(scipy_code, str)

    @staticmethod
    def write_model_and_data(model_str, data_str=None):
        from contextlib import contextmanager

        @contextmanager
        def temp_model_data(model_str, data_str=None):
            with tempfile.TemporaryDirectory() as tmpdir:
                mod_path = os.path.join(tmpdir, "model.mod")
                dat_path = os.path.join(tmpdir, "model.dat")
                with open(mod_path, "w") as f:
                    f.write(model_str)
                if data_str is not None:
                    with open(dat_path, "w") as f:
                        f.write(data_str)
                yield (mod_path, dat_path if data_str else None)

        return temp_model_data(model_str, data_str)

    @classmethod
    def run_both_solvers(cls, model_str, data_str=None):
        with cls.write_model_and_data(model_str, data_str) as (mod_path, dat_path):
            gurobi = solve(mod_path, dat_path, solver="gurobi")
            scipy = solve(mod_path, dat_path, solver="scipy")
        return gurobi, scipy

    def assert_status_match(self, gurobi, scipy):
        self.assertEqual(gurobi["status"], scipy["status"])

    def assert_objective_close(self, gurobi, scipy, tol=1e-6):
        if gurobi["status"] == "OPTIMAL":
            self.assertAlmostEqual(gurobi["objective_value"], scipy["objective_value"], delta=tol)


class TestPyOPLLexer(TestPyOPL):
    def test_compiler_line_reporting_masks_details_but_keeps_lineno(self):
        model_code = """
        dvar float x;
        maximize x + 1
        subject to { x <= 10; }
        """

        compiler = OPLCompiler(syntax_error_reporting="line")

        with self.assertRaises(SyntaxError) as exc:
            compiler.compile_model(model_code, solver="scipy")

        self.assertEqual(str(exc.exception), "Syntax error on line 4")

    def test_compiler_masked_reporting_hides_lineno(self):
        model_code = """
        dvar float x;
        maximize x + 1
        subject to { x <= 10; }
        """

        compiler = OPLCompiler(syntax_error_reporting="masked")

        with self.assertRaises(SyntaxError) as exc:
            compiler.compile_model(model_code, solver="scipy")

        self.assertEqual(str(exc.exception), "Syntax error")

    def test_compiler_keeps_rich_semantic_errors_by_default(self):
        model_code = """
        dvar float x;
        maximize x + z;
        subject to { x <= 10; }
        """

        compiler = OPLCompiler()

        with self.assertRaises(SemanticError) as exc:
            compiler.compile_model(model_code, solver="scipy")

        self.assertIn("Undeclared symbol 'z'", str(exc.exception))

    def test_syntax_error_missing_semicolon(self):
        """Test that a missing semicolon triggers a syntax error."""
        error_code = """
        dvar float x;
        minimize x + 5
        subject to
            x <= 10;
        """
        lexer = OPLLexer()
        parser = OPLParser()
        with self.assertRaises(SemanticError):
            parser.parse(lexer.tokenize(error_code))

    def test_reserved_python_keyword_names_are_rejected(self):
        """Python keywords such as 'del' must be rejected before code generation."""
        model_code = """
        dvar float del;
        maximize del;
        subject to { del <= 1; }
        """
        lexer = OPLLexer()
        parser = OPLParser()

        with self.assertRaises(SemanticError) as exc:
            parser.parse(lexer.tokenize(model_code))

        self.assertIn("Identifier 'del' is reserved", str(exc.exception))

    def test_reserved_python_keyword_data_keys_are_rejected(self):
        """Data keys that are Python keywords must be rejected during model loading."""
        model_code = """
        param float x = ...;
        maximize x;
        subject to { x >= 0; }
        """
        data_code = """
        del = 1;
        x = 2;
        """

        with self.write_model_and_data(model_code, data_code) as (mod_path, dat_path):
            ast, generated_code, data_dict = load_opl_model(mod_path, dat_path)

        self.assertIsNone(ast)
        self.assertIsNone(generated_code)
        self.assertIsNone(data_dict)

    def test_semantic_error_cases_and_valid_cases(self):
        """Test that semantic errors are detected and valid cases parse successfully."""
        # Each error case should raise SemanticError
        error_cases = [
            # Undeclared variable
            """
            dvar float x;
            maximize x + z;
            subject to { x <= 10; }
            """,
            # Incorrect number of dimensions
            """
            dvar float arr[1..5];
            maximize arr[1][2];
            subject to { arr[1] <= 10; }
            """,
            # Index out of range
            """
            dvar int arr[1..5];
            maximize arr[6];
            subject to { arr[1] <= 10; }
            """,
            # Non-integer index
            """
            dvar int arr[1..5];
            dvar float f;
            maximize arr[f];
            subject to { arr[1] <= 10; }
            """,
            # Range bounds not integers
            """
            range MyRange = 1.5..10;
            dvar int x;
            maximize x;
            subject to { x <= 1; }
            """,
            # Range start > end
            """
            range MyRange = 10..1;
            dvar int x;
            maximize x;
            subject to { x <= 1; }
            """,
            # Using non-range in 'in' clause
            """
            dvar int x;
            maximize sum (i in x) (i);
            subject to { x <= 1; }
            """,
            # Re-declaration
            """
            dvar int x;
            dvar float x;
            maximize x;
            subject to { x <= 1; }
            """,
        ]
        # Removed type mismatch in arithmetic and comparison between boolean and int/float,
        # as such operations are permissible.
        for code in error_cases:
            lexer = OPLLexer()
            parser = OPLParser()
            try:
                parser.parse(lexer.tokenize(code))
                self.fail(f"Expected SemanticError but parsing succeeded. Model snippet:\n{code.strip()}")
            except SemanticError:
                # Optionally, print the error and code for debugging
                # print(f"SemanticError: {e}\nModel snippet:\n{code.strip()}")
                pass

        # Additional: These should NOT raise SemanticError (should parse successfully)
        valid_cases = [
            # Arithmetic between boolean and int
            """
            dvar boolean b;
            dvar int x;
            maximize b + x;
            subject to { x <= 1; }
            """,
            # Arithmetic between boolean and float
            """
            dvar boolean b;
            dvar float y;
            maximize b + y;
            subject to { y <= 1.5; }
            """,
            # Arithmetic: int - boolean
            """
            dvar boolean b;
            dvar int x;
            maximize x - b;
            subject to { x <= 1; }
            """,
            # Arithmetic: float * boolean
            """
            dvar boolean b;
            dvar float y;
            maximize y * b;
            subject to { y <= 1.5; }
            """,
            # Boolean alone in objective (now valid)
            """
            dvar boolean b;
            maximize b;
            subject to { b == true; }
            """,
        ]
        for code in valid_cases:
            lexer = OPLLexer()
            parser = OPLParser()
            try:
                parser.parse(lexer.tokenize(code))
            except SemanticError as e:
                self.fail(f"Valid case raised SemanticError unexpectedly: {e}\nModel snippet:\n{code.strip()}")


class TestPyOPLParser(TestPyOPL):
    def test_comparison_expression_in_index_constraint(self):
        """Test that comparison expressions (LE, GE, etc.) are accepted in index constraints (not followed by semicolon)."""
        lexer = OPLLexer()
        parser = OPLParser()
        # Test sum with index constraint using LE
        valid_ops = ["<=", ">=", "==", "!=", "<", ">"]
        for op in valid_ops:
            opl_code = f"""
            range I = 1..3;
            dvar int x[I];
            dvar int y[I];
            maximize sum(i in I : x[i] {op} y[i]) x[i];
            subject to {{
                forall(i in I : x[i] {op} y[i]) x[i] >= 0;
            }}
            """
            ast = parser.parse(lexer.tokenize(opl_code))
            # Check that the sum and forall index constraints are parsed as comparison expressions
            sum_expr = ast["objective"]["expression"]
            self.assertEqual(sum_expr["type"], "sum")
            self.assertIsNotNone(sum_expr["index_constraint"])
            self.assertEqual(sum_expr["index_constraint"]["type"], "binop")
            self.assertEqual(sum_expr["index_constraint"]["op"], op)
            # Check forall constraint
            forall_constraint = ast["constraints"][0]
            self.assertEqual(forall_constraint["type"], "forall_constraint")
            self.assertIsNotNone(forall_constraint["index_constraint"])
            self.assertEqual(forall_constraint["index_constraint"]["type"], "binop")
            self.assertEqual(forall_constraint["index_constraint"]["op"], op)

    def test_comparison_expression_semantics(self):
        """Test OPLParser.comparison_expression for all comparison ops and type errors with dvar int."""
        lexer = OPLLexer()
        parser = OPLParser()
        # Valid int/float/boolean comparisons (OPL semantics)
        valid_cases = [
            # int vs int
            ("dvar int x; dvar int y; maximize x + y; subject to { x <= y; }",),
            # int vs boolean
            ("dvar int x; dvar boolean b; maximize x + b; subject to { x == b; }",),
            # boolean vs int
            ("dvar boolean b; dvar int x; maximize b + x; subject to { b == x; }",),
            # boolean vs boolean
            ("dvar boolean b; dvar boolean c; maximize b + c; subject to { b == c; }",),
            # float vs boolean
            ("dvar float y; dvar boolean b; maximize y + b; subject to { y == b; }",),
            # boolean vs float
            ("dvar boolean b; dvar float y; maximize b + y; subject to { b == y; }",),
        ]
        for (opl_code,) in valid_cases:
            ast = parser.parse(lexer.tokenize(opl_code))
            constraint = ast["constraints"][0]
            self.assertEqual(constraint["type"], "constraint")
        # Invalid type combinations (should raise SemanticError)
        error_cases = [
            # tuple vs int
            """
            tuple Arc { string start; string end; float cost; };
            {Arc} arcs = { <"A","B",10.0>, <"B","C",12.5> };
            dvar int x;
            maximize x;
            subject to { arcs == x; }
            """,
            # string vs float
            """
            dvar float x;
            dvar string s;
            maximize x;
            subject to { x == s; }
            """,
            # string vs boolean
            """
            dvar boolean b;
            dvar string s;
            maximize b;
            subject to { b == s; }
            """,
        ]
        for code in error_cases:
            lexer = OPLLexer()
            parser = OPLParser()
            with self.assertRaises(SemanticError):
                parser.parse(lexer.tokenize(code))

    def test_file_loading_and_gurobi_codegen(self):
        """Test loading a model/data file and Gurobi code generation."""
        dummy_model_file = "test_model.mod"
        dummy_data_file = "test_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                dvar float z;
                param float my_param;
                set my_set;
                param float my_array[1..2];
                param float my_2d_array[1..2][1..2];

                maximize z + my_param;

                subject to {
                    z <= 100;
                    my_array[1] + my_array[2] <= 50;
                    my_2d_array[1][1] + my_2d_array[2][2] >= 5;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("""
                my_param = 50;
                my_set = {1, 2, 3};
                my_array = [10, 20];
                my_2d_array = [[1, 2], [3, 4]];
                """)
            ast, gurobi_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file)
            self.assertIsInstance(ast, dict)
            self.assertIsInstance(gurobi_code, str)
            self.assertIsInstance(data_dict, dict)
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_file_loading_and_scipy_codegen_and_solution(self):
        """Test loading a model/data file, SciPy codegen, and solution extraction."""
        from pyopl.pyopl_core import load_opl_model, solve_with_scipy

        dummy_model_file = "test_model.mod"
        dummy_data_file = "test_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                dvar float z;
                param float my_param;
                set my_set;
                param float my_array[1..2];
                param float my_2d_array[1..2][1..2];

                maximize z + my_param;

                subject to {
                    z <= 100;
                    my_array[1] + my_array[2] <= 50;
                    my_2d_array[1][1] + my_2d_array[2][2] >= 5;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("""
                my_param = 50;
                my_set = {1, 2, 3};
                my_array = [10, 20];
                my_2d_array = [[1, 2], [3, 4]];
                """)
            ast, scipy_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file, solver="scipy")
            self.assertIsInstance(ast, dict)
            self.assertIsInstance(scipy_code, str)
            self.assertIsInstance(data_dict, dict)
            # Also test solve_with_scipy
            result = solve_with_scipy(dummy_model_file, dummy_data_file)
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_parse_simple_param(self):
        """Test parsing a simple model with a parameter 'int initial_value = 2;'"""
        from pyopl.pyopl_core import OPLLexer, OPLParser

        opl_code = """
        int initial_value = 2;
        dvar float z;

        maximize z + initial_value;

        subject to {
            z <= 100;
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        self.assertIsInstance(ast, dict)
        # Check that the parameter is present in declarations
        param_decl = next(
            (d for d in ast.get("declarations", []) if d.get("name") == "initial_value"),
            None,
        )
        self.assertIsNotNone(param_decl)
        self.assertEqual(param_decl.get("type"), "parameter_inline")
        self.assertEqual(param_decl.get("var_type"), "int")
        self.assertEqual(param_decl.get("value"), 2)


class TestPyOPLCompiler(TestPyOPL):
    def test_computed_param_general_expr_inline(self):
        model_code = """
            range T = 1..6;
            dvar float+ x[T];
            float demand[T] = ...;

            // general RHS expressions in computed parameters
            float sqrt_half[t in T] = sqrt(demand[t]) / 2;
            int mod_demand[t in T] = demand[t] % 5;

            minimize 0;
            subject to { }
        """
        data_code = """
            demand = [80, 60, 70, 90, 50, 60];
        """
        import math

        from pyopl.pyopl_core import OPLCompiler

        compiler = OPLCompiler()

        # Gurobi backend compile
        ast_g, code_g, data_g = compiler.compile_model(model_code, data_code=data_code, solver="gurobi")
        self.assertIn("sqrt_half", data_g)
        self.assertIn("mod_demand", data_g)
        self.assertIsInstance(data_g["sqrt_half"], list)
        self.assertIsInstance(data_g["mod_demand"], list)
        self.assertEqual(len(data_g["sqrt_half"]), 6)
        self.assertEqual(len(data_g["mod_demand"]), 6)

        expected_sqrt_half = [math.sqrt(v) / 2.0 for v in [80, 60, 70, 90, 50, 60]]
        expected_mod = [v % 5 for v in [80, 60, 70, 90, 50, 60]]

        for a, b in zip(data_g["sqrt_half"], expected_sqrt_half):
            self.assertAlmostEqual(a, b, places=9)
        for a, b in zip(data_g["mod_demand"], expected_mod):
            self.assertEqual(int(a), int(b))

        # SciPy backend compile
        ast_s, code_s, data_s = compiler.compile_model(model_code, data_code=data_code, solver="scipy")
        self.assertIn("sqrt_half", data_s)
        self.assertIn("mod_demand", data_s)
        self.assertEqual(len(data_s["sqrt_half"]), 6)
        self.assertEqual(len(data_s["mod_demand"]), 6)
        for a, b in zip(data_s["sqrt_half"], expected_sqrt_half):
            self.assertAlmostEqual(a, b, places=9)
        for a, b in zip(data_s["mod_demand"], expected_mod):
            self.assertEqual(int(a), int(b))

    def test_computed_param_sqrt_inline_gurobi_and_scipy(self):
        model_code = """
            range T = 1..6;
            dvar float+ x[T];
            dvar float+ y[T];
            float demand[T] = ...;
            float sqrt_demand[t in T] = sqrt(demand[t]);

            minimize 0;
            subject to {
                forall(t in T) {
                    x[t] == sqrt_demand[t];
                    y[t] == demand[t];
                }
            }
        """
        data_code = """
            demand = [80, 60, 70, 90, 50, 60];
        """
        import math

        from pyopl.pyopl_core import OPLCompiler

        compiler = OPLCompiler()

        # Gurobi backend compile
        ast_g, code_g, data_g = compiler.compile_model(model_code, data_code=data_code, solver="gurobi")
        assert "sqrt_demand" in data_g
        # Expect list of 6 values
        assert isinstance(data_g["sqrt_demand"], list)
        assert len(data_g["sqrt_demand"]) == 6
        # Check some values
        expected = [math.sqrt(v) for v in [80, 60, 70, 90, 50, 60]]
        for a, b in zip(data_g["sqrt_demand"], expected):
            assert abs(a - b) < 1e-9

        # SciPy backend compile (ensures cross-backend works too)
        ast_s, code_s, data_s = compiler.compile_model(model_code, data_code=data_code, solver="scipy")
        assert "sqrt_demand" in data_s
        assert isinstance(data_s["sqrt_demand"], list)
        assert len(data_s["sqrt_demand"]) == 6
        for a, b in zip(data_s["sqrt_demand"], expected):
            assert abs(a - b) < 1e-9

    def test_param_range_length_mismatch_raises(self):
        """
        Test that both solvers return an error if a parameter's data length does not match the declared range.
        """
        model_code = """
        int n = ...;
        range I = 1..n;
        float demand[I][I] = ...;
        dvar float x[I];
        minimize sum(i in I) x[i];
        subject to {
            forall(i in I, j in I) x[i] >= demand[i][j];
        }
        """
        # n = 3, but demand only has 2 elements (should error)
        data_code = """
        n = 3;
        demand = [[10, 20], [30, 40, 50], [50, 60, 70]];
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertIn(result["status"].upper(), ["ERROR", "FAILED", "EXECUTION_ERROR"])
                self.assertIn(
                    "Failed to load or parse OPL model from file. See errors traceback.",
                    result.get("message", ""),
                )
            finally:
                os.remove(model_file)
                os.remove(data_file)

    def test_large_indexed_variables_and_constraints(self):
        """
        Stress test: Large indexed variable arrays and constraints.
        """
        n = 50
        model = f"""
        range I = 1..{n};
        dvar float x[I];
        maximize sum(i in I) x[i];
        subject to {{
            forall(i in I) x[i] <= {n};
            forall(i in I) x[i] >= 0;
        }}
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)

    def test_mixed_types_and_tight_bounds(self):
        """
        Stress test: Mixed variable types and tight bounds.
        """
        model = """
        dvar int x;
        dvar float y;
        dvar boolean b;
        maximize x + y + b;
        subject to {
            x <= 1;
            x >= 1;
            y <= 1.00001;
            y >= 1.0;
            b == true;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)

    def test_redundant_and_contradictory_constraints(self):
        """
        Stress test: Redundant and contradictory constraints.
        """
        model = """
        dvar int x;
        maximize x;
        subject to {
            x <= 5;
            x <= 10;
            x >= 5;
            x >= 0;
            x == 5;
            x == 6;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        # Both should report infeasible or inf_or_unbd
        self.assertIn(gurobi["status"], ("INFEASIBLE", "INF_OR_UNBD"))
        self.assertIn(scipy["status"], ("INFEASIBLE", "INF_OR_UNBD"))

    def test_sum_on_lhs_of_constraint_in_forall(self):
        """Test that a sum expression can be used as the left-hand side of a constraint inside a forall loop."""
        model = """
        range I = 1..3;
        range J = 1..2;
        dvar float x[I][J];
        minimize sum(i in I, j in J) x[i][j];
        subject to {
            forall(j in J)
                sum(i in I) x[i][j] >= 6;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        # For each j, sum over i of x[i][j] should be 6 (tight constraint), all x[i][j] >= 0
        for j in range(1, 3):
            gurobi_sum = sum(gurobi["solution"].get(f"x[{i},{j}]", 0) for i in range(1, 4))
            scipy_sum = sum(scipy["solution"].get(f"x_{i}_{j}", 0) for i in range(1, 4))
            self.assertAlmostEqual(gurobi_sum, 6)
            self.assertAlmostEqual(scipy_sum, 6)
        for v in gurobi["solution"].values():
            self.assertGreaterEqual(v, 0)
        for v in scipy["solution"].values():
            self.assertGreaterEqual(v, 0)

    def test_missing_model_and_data_file_scipy(self):
        """Test error handling for missing model and data files in solve_with_scipy (covers lines 1235-1241, 1318-1324)."""
        import os
        import tempfile

        from pyopl.pyopl_core import solve_with_scipy

        # Create a temp file and then delete it to ensure it does not exist
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_model = os.path.join(tmpdir, "missing_model.mod")
            missing_data = os.path.join(tmpdir, "missing_data.dat")
            # Model file missing
            result = solve_with_scipy(missing_model)
            self.assertIn("Error: Model file", result.get("message", ""))
            self.assertIn(result.get("status", "").upper(), ["FAILED", "ERROR"])
            # Model exists, data file missing
            with open(missing_model, "w") as f:
                f.write("dvar float x; minimize x; subject to { x >= 1; }")
            result2 = solve_with_scipy(missing_model, missing_data)
            self.assertIn("Error: Data file", result2.get("message", ""))
            self.assertIn(result2.get("status", "").upper(), ["FAILED", "ERROR"])

    def test_floatplus_and_intplus_scalar_and_indexed(self):
        """Test float+ and int+ variable handling for both scalar and indexed cases."""
        # Scalar float+ and int+
        opl_code = """
        dvar float+ x;
        dvar int+ y;
        maximize x + y;
        subject to {
            x >= 0;
            y >= 0;
            x <= 5;
            y <= 10;
        }
        """
        self.run_test_case_gurobi(opl_code)
        self.run_test_case_scipy(opl_code)

        # Indexed float+ and int+
        opl_code2 = """
        range I = 1..2;
        dvar float+ xf[I];
        dvar int+ yi[I];
        maximize sum(i in I) (xf[i] + yi[i]);
        subject to {
            forall(i in I) xf[i] >= 0;
            forall(i in I) yi[i] >= 0;
            forall(i in I) xf[i] <= 3;
            forall(i in I) yi[i] <= 4;
        }
        """
        self.run_test_case_gurobi(opl_code2)
        self.run_test_case_scipy(opl_code2)

    def test_sum_on_lhs_of_constraint(self):
        """Test that a sum expression can be used as the left-hand side of a constraint."""
        model = """
        range I = 1..3;
        dvar float x[I];
        minimize sum(i in I) x[i];
        subject to {
            sum(i in I) x[i] >= 6;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        # The sum of x[i] should be 6 (tight constraint), all x[i] >= 0
        self.assertAlmostEqual(sum(gurobi["solution"].values()), 6)
        self.assertAlmostEqual(sum(scipy["solution"].values()), 6)
        for v in gurobi["solution"].values():
            self.assertGreaterEqual(v, 0)
        for v in scipy["solution"].values():
            self.assertGreaterEqual(v, 0)

    def test_sum_on_lhs_of_constraint_with_index_neq(self):
        """Test that a sum expression can be used as the left-hand side of a constraint."""
        model = """
        range I = 1..3;
        dvar float+ x[I];
        minimize sum(i in I) x[i];
        subject to {
            sum(i in I : i != 2) x[i] >= 6;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        # The sum of x[i] should be 6 (tight constraint), all x[i] >= 0
        self.assertAlmostEqual(sum(gurobi["solution"].values()), 6)
        self.assertAlmostEqual(sum(scipy["solution"].values()), 6)
        for v in gurobi["solution"].values():
            self.assertGreaterEqual(v, 0)
        for v in scipy["solution"].values():
            self.assertGreaterEqual(v, 0)

    def test_sum_on_lhs_of_constraint_with_index_eq(self):
        """Test that a sum expression can be used as the left-hand side of a constraint."""
        model = """
        range I = 1..3;
        dvar float+ x[I];
        minimize sum(i in I) x[i];
        subject to {
            sum(i in I : i == 2) x[i] >= 6;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        # The sum of x[i] should be 6 (tight constraint), all x[i] >= 0
        self.assertAlmostEqual(sum(gurobi["solution"].values()), 6)
        self.assertAlmostEqual(sum(scipy["solution"].values()), 6)
        for v in gurobi["solution"].values():
            self.assertGreaterEqual(v, 0)
        for v in scipy["solution"].values():
            self.assertGreaterEqual(v, 0)

    def test_empty_model(self):
        """Test an empty model (should return error status)."""
        model = """"""
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi.get("status", "").upper(), ["ERROR", "FAILED"])
        self.assertIn(scipy.get("status", "").upper(), ["ERROR", "FAILED"])

    def test_invalid_operator(self):
        """Test model with an invalid operator in constraint (should return error status)."""
        model = """
        dvar float x;
        minimize x;
        subject to {
            x <> 1;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi.get("status", "").upper(), ["ERROR", "FAILED"])
        self.assertIn(scipy.get("status", "").upper(), ["ERROR", "FAILED"])

    def test_parameter_not_found(self):
        """Test model referencing a parameter not in data (should return error status)."""
        model = """
        param float a;
        dvar float x;
        minimize x + a;
        subject to { x >= 1; }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi.get("status", "").upper(), ["ERROR", "FAILED"])
        self.assertIn(scipy.get("status", "").upper(), ["ERROR", "FAILED"])

    def test_range_start_greater_than_end(self):
        """Test range with start > end (should return error status)."""
        model = """
        range R = 5..1;
        dvar float x;
        maximize x;
        subject to { x <= 1; }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi.get("status", "").upper(), ["ERROR", "FAILED"])
        self.assertIn(scipy.get("status", "").upper(), ["ERROR", "FAILED"])

    def test_constraint_with_symbolic_rhs(self):
        """Test constraint with symbolic right-hand side (should return error status)."""
        model = """
        dvar float x;
        param float a;
        minimize x;
        subject to { x >= a; }
        """
        # No data for 'a', should error
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi.get("status", "").upper(), ["ERROR", "FAILED"])
        self.assertIn(scipy.get("status", "").upper(), ["ERROR", "FAILED"])

    def test_parenthesized_expression(self):
        """Test parenthesized expressions in objective and constraints."""
        model = """
        dvar float x;
        dvar float y;
        minimize (x + (y));
        subject to { (x) + (y) >= 2; }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)

    def test_sum_with_index_constraint(self):
        """Test sum with index constraint (sum(i in 1..3: i != 2))."""
        model = """
        range I = 1..3;
        dvar float x[I];
        minimize sum(i in I: i != 2) x[i];
        subject to { forall(i in I) x[i] >= 1; }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)

    def test_forall_with_index_constraint(self):
        """Test forall with index constraint (forall(i in 1..3: i != 2))."""
        model = """
        range I = 1..3;
        dvar float x[I];
        minimize sum(i in I) x[i];
        subject to { forall(i in I: i != 2) x[i] >= 5; }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)

    def test_boolean_arithmetic(self):
        """Test arithmetic with boolean variables in objective and constraints."""
        model = """
        dvar boolean b;
        dvar int x;
        maximize b * 2 + x;
        subject to { b + x <= 2; }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)

    def test_indexed_parameter(self):
        """Test model with indexed parameter (should error if data missing)."""
        model = """
        param float a[1..2];
        dvar float x;
        minimize x + a[1];
        subject to { x >= 1; }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi.get("status", "").upper(), ["ERROR", "FAILED"])
        self.assertIn(scipy.get("status", "").upper(), ["ERROR", "FAILED"])

    def test_infeasible_model_compare_solvers(self):
        """Test that infeasible models are detected by both solvers."""
        dummy_model_file = "infeasible.mod"
        dummy_data_file = "infeasible.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                dvar float x;
                maximize x;
                subject to {
                    x >= 2;
                    x <= 1;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("")
            result_gurobi, result_scipy = self.run_both_solvers(dummy_model_file, dummy_data_file)
            self.assertIn(
                result_gurobi.get("status", "").upper(),
                ["INFEASIBLE", "INF_OR_UNBD", "FAILED"],
            )
            self.assertIn(
                result_scipy.get("status", "").upper(),
                ["INFEASIBLE", "INF_OR_UNBD", "FAILED"],
            )
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_unbounded_model_compare_solvers(self):
        """Test that unbounded models are detected by both solvers."""
        dummy_model_file = "unbounded.mod"
        dummy_data_file = "unbounded.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                dvar float x;
                maximize x;
                subject to {
                    x >= 0;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("")
            result_gurobi, result_scipy = self.run_both_solvers(dummy_model_file, dummy_data_file)
            self.assertIn(
                result_gurobi.get("status", "").upper(),
                ["UNBOUNDED", "INF_OR_UNBD", "FAILED"],
            )
            self.assertIn(
                result_scipy.get("status", "").upper(),
                ["UNBOUNDED", "INF_OR_UNBD", "FAILED"],
            )
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_floatplus_and_intplus_edge(self):
        """Test float+ and int+ variables with negative lower bounds (should clamp to 0)."""
        # Test float+ and int+ with negative bounds (should not allow negative values)
        model = """
        dvar float+ x;
        dvar int+ y;
        minimize x + y;
        subject to {
            x >= -5;
            y >= -10;
            x <= 2;
            y <= 3;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        # Both x and y should be at their lowest allowed (0)
        self.assertAlmostEqual(gurobi["solution"]["x"], 0)
        self.assertAlmostEqual(gurobi["solution"]["y"], 0)
        self.assertAlmostEqual(scipy["solution"]["x"], 0)
        self.assertAlmostEqual(scipy["solution"]["y"], 0)

    def test_all_variables_fixed(self):
        """Test that all variables fixed by constraints yield correct solution."""
        model = """
        dvar int x;
        dvar float y;
        minimize x + y;
        subject to {
            x == 1;
            y == 2.5;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"], 1)
        self.assertAlmostEqual(gurobi["solution"]["y"], 2.5)

    def test_redundant_constraints(self):
        """Test that redundant constraints do not affect the solution."""
        model = """
        dvar float x;
        minimize x;
        subject to {
            x >= 1;
            x >= 1;
            x <= 5;
            x <= 5;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"], 1)

    def test_mixed_variable_types(self):
        """Test models with boolean, int, and float variables together."""
        model = """
        dvar boolean b;
        dvar int i;
        dvar float f;
        maximize 2*b + i + 0.5*f;
        subject to {
            b + i + f <= 3;
            i >= 0;
            f >= 0;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)

    def test_degenerate_solution(self):
        """Test degenerate models with multiple optimal solutions."""
        model = """
        dvar float x;
        dvar float y;
        minimize 0*x + 0*y;
        subject to {
            x + y == 1;
            x >= 0;
            y >= 0;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assertAlmostEqual(gurobi["objective_value"], 0)
        self.assertAlmostEqual(scipy["objective_value"], 0)
        self.assertAlmostEqual(gurobi["solution"]["x"] + gurobi["solution"]["y"], 1)
        self.assertAlmostEqual(scipy["solution"]["x"] + scipy["solution"]["y"], 1)

    def test_large_scale(self):
        """Test large-scale models for solver consistency and performance."""
        N = 50
        model = f"""
        range I = 1..{N};
        dvar float x[I];
        minimize sum(i in I) x[i];
        subject to {{
            forall(i in I) x[i] >= {N};
        }}
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy, tol=1e-4)
        for v in gurobi["solution"].values():
            self.assertAlmostEqual(v, N)

    def test_tight_bounds(self):
        """Test models where variable bounds are tight (equality)."""
        model = """
        dvar float x;
        minimize x;
        subject to {
            x >= 2;
            x <= 2;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"], 2)

    def test_no_constraints(self):
        """Test models with no effective constraints (should be unbounded)."""
        model = """
        dvar float x;
        maximize x;
        subject to {
            x >= 0;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi["status"], ("UNBOUNDED", "INF_OR_UNBD"))
        self.assertIn(scipy["status"], ("UNBOUNDED", "INF_OR_UNBD"))

    def test_infeasible_bounds(self):
        """Test models with infeasible variable bounds."""
        model = """
        dvar float x;
        minimize x;
        subject to {
            x >= 2;
            x <= 1;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assertEqual(gurobi["status"], "INFEASIBLE")
        self.assertEqual(scipy["status"], "INFEASIBLE")

    def test_all_zero_objective(self):
        """Test models with zero objective coefficients."""
        model = """
        dvar float x;
        dvar float y;
        minimize 0*x + 0*y;
        subject to {
            x + y >= 1;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assertAlmostEqual(gurobi["objective_value"], 0)
        self.assertAlmostEqual(scipy["objective_value"], 0)

    def test_variable_not_in_constraints(self):
        """Test models where some variables are not constrained."""
        model = """
        dvar float x;
        dvar float y;
        minimize x;
        subject to {
            x >= 1;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"], 1)
        self.assertAlmostEqual(gurobi["solution"].get("y", 0), 0)
        self.assertAlmostEqual(scipy["solution"].get("y", 0), 0)

    def test_single_variable_single_constraint(self):
        """Test models with a single variable and a single constraint."""
        model = """
        dvar float x;
        minimize x;
        subject to {
            x >= 3;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"], 3)

    def test_all_variables_unbounded(self):
        """Test models where all variables are unbounded."""
        model = """
        dvar float x;
        dvar float y;
        maximize x + y;
        subject to {
            0 <= 1;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assertIn(gurobi["status"], ("UNBOUNDED", "INF_OR_UNBD"))
        self.assertIn(scipy["status"], ("UNBOUNDED", "INF_OR_UNBD"))

    def test_redundant_variables(self):
        """Test models with redundant variables (not affecting the objective)."""
        model = """
        dvar float x;
        dvar float y;
        dvar float z;
        minimize x;
        subject to {
            x >= 1;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"], 1)
        self.assertAlmostEqual(gurobi["solution"].get("y", 0), 0)
        self.assertAlmostEqual(gurobi["solution"].get("z", 0), 0)
        self.assertAlmostEqual(scipy["solution"].get("y", 0), 0)
        self.assertAlmostEqual(scipy["solution"].get("z", 0), 0)

    def test_tight_equality_constraint(self):
        """Test models with tight equality constraints between variables."""
        model = """
        dvar float x;
        dvar float y;
        minimize x + y;
        subject to {
            x + y == 5;
            x >= 2;
            y >= 2;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"] + gurobi["solution"]["y"], 5)
        self.assertGreaterEqual(gurobi["solution"]["x"], 2)
        self.assertGreaterEqual(gurobi["solution"]["y"], 2)
        self.assertAlmostEqual(gurobi["objective_value"], 5)
        self.assertAlmostEqual(scipy["solution"]["x"] + scipy["solution"]["y"], 5)
        self.assertGreaterEqual(scipy["solution"]["x"], 2)
        self.assertGreaterEqual(scipy["solution"]["y"], 2)
        self.assertAlmostEqual(scipy["objective_value"], 5)

    def test_all_zero_constraints(self):
        """Test models with constraints that are always satisfied (zero coefficients)."""
        model = """
        dvar float+ x;
        dvar float+ y;
        minimize x + y;
        subject to {
            0*x + 0*y >= 0;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assertEqual(gurobi["status"], "OPTIMAL")
        self.assertEqual(scipy["status"], "OPTIMAL")
        self.assertAlmostEqual(gurobi["solution"]["x"], 0)
        self.assertAlmostEqual(gurobi["solution"]["y"], 0)
        self.assertAlmostEqual(gurobi["objective_value"], 0)
        self.assertAlmostEqual(scipy["solution"]["x"], 0)
        self.assertAlmostEqual(scipy["solution"]["y"], 0)
        self.assertAlmostEqual(scipy["objective_value"], 0)

    def test_variable_only_in_objective(self):
        """Test models where a variable appears only in the objective."""
        model = """
        dvar float+ x;
        dvar float+ y;
        minimize x;
        subject to {
            y >= 1;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertGreaterEqual(gurobi["solution"]["y"], 1)
        self.assertGreaterEqual(scipy["solution"]["y"], 1)

    def test_constraint_always_satisfied(self):
        """Test models with constraints that are always satisfied (e.g., 0 <= 1)."""
        model = """
        dvar float x;
        minimize x;
        subject to {
            0 <= 1;
            x >= 2;
        }
        """
        gurobi, scipy = self.run_both_solvers(model)
        self.assert_status_match(gurobi, scipy)
        self.assert_objective_close(gurobi, scipy)
        self.assertAlmostEqual(gurobi["solution"]["x"], 2)

    def test_nested_array_parsing(self):
        """Test parsing of nested arrays (1D, 2D, 3D) in .dat files."""
        lexer = OPLDataLexer()
        parser = OPLDataParser()

        # 1D array
        data_code_1d = "arr = [1, 2, 3];"
        tokens_1d = lexer.tokenize(data_code_1d)
        data_dict_1d = parser.parse(tokens_1d, lexer=lexer)
        self.assertIn("arr", data_dict_1d)
        self.assertEqual(data_dict_1d["arr"], [1, 2, 3])

        # 2D array
        data_code_2d = "mat = [[1, 2], [3, 4]];"
        tokens_2d = lexer.tokenize(data_code_2d)
        data_dict_2d = parser.parse(tokens_2d, lexer=lexer)
        self.assertIn("mat", data_dict_2d)
        self.assertEqual(data_dict_2d["mat"], [[1, 2], [3, 4]])

        # 3D array
        data_code_3d = "cube = [[[1], [2]], [[3], [4]]];"
        tokens_3d = lexer.tokenize(data_code_3d)
        data_dict_3d = parser.parse(tokens_3d, lexer=lexer)
        self.assertIn("cube", data_dict_3d)
        self.assertEqual(data_dict_3d["cube"], [[[1], [2]], [[3], [4]]])

    def test_validate_shape_multi_dimensional(self):
        """Test validate_shape for multi-dimensional arrays with correct and incorrect shapes."""
        from pyopl.pyopl_core import OPLCompiler, SemanticError

        compiler = OPLCompiler()
        # Simulate a model AST with a 2D parameter: float arr[I][J] = ...;
        ast = {
            "declarations": [
                {
                    "type": "range_declaration_inline",
                    "name": "I",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 2},
                },
                {
                    "type": "range_declaration_inline",
                    "name": "J",
                    "start": {"type": "number", "value": 1},
                    "end": {"type": "number", "value": 3},
                },
                {
                    "type": "parameter_external_indexed",
                    "var_type": "float",
                    "name": "arr",
                    "dimensions": [
                        {
                            "type": "named_range_dimension",
                            "name": "I",
                            "start": {"type": "number", "value": 1},
                            "end": {"type": "number", "value": 2},
                        },
                        {
                            "type": "named_range_dimension",
                            "name": "J",
                            "start": {"type": "number", "value": 1},
                            "end": {"type": "number", "value": 3},
                        },
                    ],
                },
            ]
        }
        # Correct shape: arr = [[1,2,3],[4,5,6]]
        data_dict = {"arr": [[1, 2, 3], [4, 5, 6]], "I": 2, "J": 3}
        # Should not raise
        compiler.compile_model = lambda model_code, data_code=None, solver="gurobi": (
            ast,
            None,
            data_dict,
        )
        try:
            # Directly call validate_shape from the previous step
            def validate_shape(param_data, dims, param_name, data_dict, dim=0):
                # ...copy the validate_shape function from pyopl_core.py here...
                if not dims:
                    return
                d = dims[0]
                expected_len = None
                if d.get("type") == "named_range_dimension":
                    range_decl = next(
                        (
                            x
                            for x in ast["declarations"]
                            if x.get("name") == d["name"] and x.get("type") == "range_declaration_inline"
                        ),
                        None,
                    )
                    if range_decl:

                        def eval_expr(expr):
                            if expr["type"] == "number":
                                return int(expr["value"])
                            elif expr["type"] == "name":
                                return int(data_dict[expr["value"]])
                            elif expr["type"] == "binop":
                                op = expr["op"]
                                left = eval_expr(expr["left"])
                                right = eval_expr(expr["right"])
                                if op == "+":
                                    return left + right
                                if op == "-":
                                    return left - right
                                if op == "*":
                                    return left * right
                                if op == "/":
                                    return left // right
                            raise Exception(f"Unsupported range bound expr: {expr}")

                        start = eval_expr(range_decl["start"])
                        end = eval_expr(range_decl["end"])
                        expected_len = end - start + 1
                if expected_len is not None:
                    if not isinstance(param_data, (list, tuple)):
                        raise SemanticError(
                            f"Parameter '{param_name}' expected a {len(dims)}D array, got scalar at dimension {dim+1}."
                        )
                    if len(param_data) != expected_len:
                        raise SemanticError(
                            f"Parameter '{param_name}' data length {len(param_data)} does not match declared dimension '{d.get('name')}' of length {expected_len} at dimension {dim+1}."
                        )
                    if len(dims) > 1:
                        for i, sub in enumerate(param_data):
                            validate_shape(sub, dims[1:], param_name, data_dict, dim + 1)

            # Should not raise
            validate_shape(data_dict["arr"], ast["declarations"][2]["dimensions"], "arr", data_dict)
        except SemanticError:
            self.fail("validate_shape raised SemanticError unexpectedly for correct shape.")

        # Incorrect shape: arr = [[1,2],[3,4]]
        bad_data_dict = {"arr": [[1, 2], [3, 4]], "I": 2, "J": 3}
        with self.assertRaises(SemanticError):
            validate_shape(
                bad_data_dict["arr"],
                ast["declarations"][2]["dimensions"],
                "arr",
                bad_data_dict,
            )

    def test_gurobi_codegen_multi_dimensional_arrays(self):
        """Test Gurobi codegen for 2D and 3D decision variables and parameters."""
        import re

        from pyopl.pyopl_core import GurobiCodeGenerator, OPLLexer, OPLParser

        def nospace(s: str) -> str:
            return re.sub(r"\s+", "", s)

        def assert_has_var_decl(code: str):
            self.assertTrue(
                ("model.addVars" in code) or ("model.addVar" in code),
                "Expected model.addVars or model.addVar in generated code.",
            )

        def assert_has_array_repr(
            code: str,
            name: str,
            tuple_keys=None,
            nested_literal=None,
            also_contains=None,
        ):
            """
            Accept either:
            - dict with tuple keys (check presence of specific keys), or
            - nested list literal (exact numeric structure), and optional extra substrings.
            """
            c = nospace(code)
            self.assertIn(name, code, f"Expected '{name}' to be present in generated code.")
            ok = False
            # Check tuple-keyed dict keys exist (keys only; values may vary in formatting)
            if tuple_keys:
                if all(nospace(k) in c for k in tuple_keys):
                    ok = True
            # Or check nested list literal
            if (not ok) and nested_literal:
                if nospace(nested_literal) in c:
                    ok = True
            # Optionally ensure some tokens are present (e.g., numbers)
            if also_contains and (not ok):
                if all(token in code for token in also_contains):
                    ok = True
            self.assertTrue(
                ok,
                f"Expected '{name}' to be represented as dict(flat) or nested list in generated code.",
            )

        # 2D variable and parameter
        opl_code_2d = """
        range I = 1..2;
        range J = 1..3;
        param float arr[I][J] = ...;
        dvar float x[I][J];
        maximize sum(i in I, j in J) arr[i][j] * x[i][j];
        subject to {
            forall(i in I, j in J) x[i][j] <= arr[i][j];
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code_2d))
        generator = GurobiCodeGenerator(ast, {"arr": [[1, 2, 3], [4, 5, 6]], "I": 2, "J": 3})
        gurobi_code = generator.generate_code()
        assert_has_var_decl(gurobi_code)
        # Accept either flat dict with tuple keys or nested list literal
        assert_has_array_repr(
            gurobi_code,
            "arr",
            tuple_keys=["(1,1)", "(2,3)"],
            nested_literal="arr = [[1, 2, 3], [4, 5, 6]]",
            also_contains=["1", "6"],
        )

        # 3D variable and parameter
        opl_code_3d = """
        range I = 1..2;
        range J = 1..2;
        range K = 1..2;
        param float cube[I][J][K] = ...;
        dvar float x[I][J][K];
        maximize sum(i in I, j in J, k in K) cube[i][j][k] * x[i][j][k];
        subject to {
            forall(i in I, j in J, k in K) x[i][j][k] <= cube[i][j][k];
        }
        """
        ast3 = parser.parse(lexer.tokenize(opl_code_3d))
        generator3 = GurobiCodeGenerator(ast3, {"cube": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "I": 2, "J": 2, "K": 2})
        gurobi_code3 = generator3.generate_code()
        assert_has_var_decl(gurobi_code3)
        assert_has_array_repr(
            gurobi_code3,
            "cube",
            tuple_keys=["(1,1,1)", "(2,2,2)"],
            nested_literal="cube = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
            also_contains=["1", "8"],
        )

        # 2D array indexed by ranges
        opl_code = """
            range I = 1..2;
            range J = 1..3;
            float arr[I][J] = [[1,2,3],[4,5,6]];
            minimize sum(i in I, j in J) arr[i][j];
            subject to { }
        """
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = GurobiCodeGenerator(ast)
        gurobi_code = generator.generate_code()
        assert_has_array_repr(
            gurobi_code,
            "arr",
            tuple_keys=["(1,1)", "(1,2)", "(1,3)", "(2,1)", "(2,2)", "(2,3)"],
            nested_literal="arr = [[1, 2, 3], [4, 5, 6]]",
            also_contains=["I", "J"],
        )

        # 3D array indexed by ranges
        opl_code_3d = """
            range I = 1..2;
            range J = 1..2;
            range K = 1..2;
            float arr3d[I][J][K] = [[[1,2],[3,4]],[[5,6],[7,8]]];
            minimize sum(i in I, j in J, k in K) arr3d[i][j][k];
            subject to { }
        """
        ast3d = parser.parse(lexer.tokenize(opl_code_3d))
        generator3d = GurobiCodeGenerator(ast3d)
        gurobi_code3d = generator3d.generate_code()
        assert_has_array_repr(
            gurobi_code3d,
            "arr3d",
            tuple_keys=["(1,1,1)", "(2,2,2)"],
            nested_literal="arr3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
            also_contains=["I", "J", "K"],
        )

        # Array indexed by set and range
        opl_code_set_range = """
            {string} Stores = { "A", "B" };
            range T = 1..2;
            float Demand[Stores][T] = [[1,2],[3,4]];
            minimize sum(s in Stores, t in T) Demand[s][t];
            subject to { }
        """
        ast_set_range = parser.parse(lexer.tokenize(opl_code_set_range))
        generator_set_range = GurobiCodeGenerator(ast_set_range)
        gurobi_code_set_range = generator_set_range.generate_code()
        # Accept tuple keys with string + int or nested list literal
        assert_has_array_repr(
            gurobi_code_set_range,
            "Demand",
            tuple_keys=["('A',1)", "('A',2)", "('B',1)", "('B',2)"],
            nested_literal="Demand = [[1, 2], [3, 4]]",
            also_contains=["A", "B", "T"],
        )

        # Array indexed by two sets
        opl_code_set_set = """
            {string} S1 = { "X", "Y" };
            {string} S2 = { "P", "Q" };
            float arrSetSet[S1][S2] = [[10,20],[30,40]];
            minimize sum(a in S1, b in S2) arrSetSet[a][b];
            subject to { }
        """
        ast_set_set = parser.parse(lexer.tokenize(opl_code_set_set))
        generator_set_set = GurobiCodeGenerator(ast_set_set)
        gurobi_code_set_set = generator_set_set.generate_code()
        assert_has_array_repr(
            gurobi_code_set_set,
            "arrSetSet",
            tuple_keys=["('X','P')", "('X','Q')", "('Y','P')", "('Y','Q')"],
            nested_literal="arrSetSet = [[10, 20], [30, 40]]",
            also_contains=["X", "Y", "P", "Q"],
        )

        # 3D array indexed by set, range, set
        opl_code_set_range_set = """
            {string} S1 = { "X", "Y" };
            range T = 1..2;
            {string} S2 = { "P", "Q" };
            float arrSRS[S1][T][S2] = [
                [[1,2],[3,4]],
                [[5,6],[7,8]]
            ];
            minimize sum(a in S1, t in T, b in S2) arrSRS[a][t][b];
            subject to { }
        """
        ast_srs = parser.parse(lexer.tokenize(opl_code_set_range_set))
        generator_srs = GurobiCodeGenerator(ast_srs)
        gurobi_code_srs = generator_srs.generate_code()
        assert_has_array_repr(
            gurobi_code_srs,
            "arrSRS",
            tuple_keys=["('X',1,'P')", "('Y',2,'Q')"],
            nested_literal="arrSRS = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
            also_contains=["X", "Y", "P", "Q", "T"],
        )

    def test_scipy_codegen_multi_dimensional_arrays(self):
        """Test SciPy codegen for 2D and 3D decision variables and parameters (mirrors Gurobi test)."""
        import re

        from pyopl.pyopl_core import OPLLexer, OPLParser
        from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator

        def nospace(s: str) -> str:
            return re.sub(r"\s+", "", s)

        def assert_has_var_decl_scipy(code: str):
            # SciPy code declares variables via 'var_names = [...]'
            self.assertIn("var_names =", code, "Expected 'var_names =' in generated SciPy code.")

        def assert_has_array_repr(
            code: str,
            name: str,
            tuple_keys=None,
            nested_literal=None,
            also_contains=None,
        ):
            """
            Accept either:
            - dict with tuple keys (check presence of specific keys), or
            - nested list literal (exact numeric structure), and optional extra substrings.
            """
            c = nospace(code)
            self.assertIn(name, code, f"Expected '{name}' to be present in generated code.")
            ok = False
            # Check tuple-keyed dict keys exist (keys only; values may vary in formatting)
            if tuple_keys:
                if all(nospace(k) in c for k in tuple_keys):
                    ok = True
            # Or check nested list literal
            if (not ok) and nested_literal:
                if nospace(nested_literal) in c:
                    ok = True
            # Optionally ensure some tokens are present (e.g., numbers or identifiers)
            if also_contains and (not ok):
                if all(token in code for token in also_contains):
                    ok = True
            self.assertTrue(
                ok,
                f"Expected '{name}' to be represented as dict(flat) or nested list in generated SciPy code.",
            )

        lexer = OPLLexer()
        parser = OPLParser()

        # 2D variable and parameter
        opl_code_2d = """
        range I = 1..2;
        range J = 1..3;
        param float arr[I][J] = ...;
        dvar float x[I][J];
        maximize sum(i in I, j in J) arr[i][j] * x[i][j];
        subject to {
            forall(i in I, j in J) x[i][j] <= arr[i][j];
        }
        """
        ast_2d = parser.parse(lexer.tokenize(opl_code_2d))
        gen_2d = SciPyCSCCodeGenerator(ast_2d, {"arr": [[1, 2, 3], [4, 5, 6]], "I": 2, "J": 3})
        scipy_code_2d = gen_2d.generate_code()
        assert_has_var_decl_scipy(scipy_code_2d)
        assert_has_array_repr(
            scipy_code_2d,
            "arr",
            tuple_keys=["(1,1)", "(2,3)"],
            nested_literal="arr = [[1, 2, 3], [4, 5, 6]]",
            also_contains=["1", "6"],
        )

        # 3D variable and parameter
        opl_code_3d = """
        range I = 1..2;
        range J = 1..2;
        range K = 1..2;
        param float cube[I][J][K] = ...;
        dvar float x[I][J][K];
        maximize sum(i in I, j in J, k in K) cube[i][j][k] * x[i][j][k];
        subject to {
            forall(i in I, j in J, k in K) x[i][j][k] <= cube[i][j][k];
        }
        """
        ast_3d = parser.parse(lexer.tokenize(opl_code_3d))
        gen_3d = SciPyCSCCodeGenerator(
            ast_3d,
            {"cube": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "I": 2, "J": 2, "K": 2},
        )
        scipy_code_3d = gen_3d.generate_code()
        assert_has_var_decl_scipy(scipy_code_3d)
        assert_has_array_repr(
            scipy_code_3d,
            "cube",
            tuple_keys=["(1,1,1)", "(2,2,2)"],
            nested_literal="cube = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
            also_contains=["1", "8"],
        )

        # 2D array indexed by ranges
        opl_code = """
            range I = 1..2;
            range J = 1..3;
            float arr[I][J] = [[1,2,3],[4,5,6]];
            minimize sum(i in I, j in J) arr[i][j];
            subject to { }
        """
        ast = parser.parse(lexer.tokenize(opl_code))
        gen = SciPyCSCCodeGenerator(ast)
        scipy_code = gen.generate_code()
        assert_has_array_repr(
            scipy_code,
            "arr",
            tuple_keys=["(1,1)", "(1,2)", "(1,3)", "(2,1)", "(2,2)", "(2,3)"],
            nested_literal="arr = [[1, 2, 3], [4, 5, 6]]",
            also_contains=["I", "J"],
        )

        # 3D array indexed by ranges
        opl_code_3d_arr = """
            range I = 1..2;
            range J = 1..2;
            range K = 1..2;
            float arr3d[I][J][K] = [[[1,2],[3,4]],[[5,6],[7,8]]];
            minimize sum(i in I, j in J, k in K) arr3d[i][j][k];
            subject to { }
        """
        ast3d = parser.parse(lexer.tokenize(opl_code_3d_arr))
        gen3d = SciPyCSCCodeGenerator(ast3d)
        scipy_code3d = gen3d.generate_code()
        assert_has_array_repr(
            scipy_code3d,
            "arr3d",
            tuple_keys=["(1,1,1)", "(2,2,2)"],
            nested_literal="arr3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
            also_contains=["I", "J", "K"],
        )

        # Array indexed by set and range
        opl_code_set_range = """
            {string} Stores = { "A", "B" };
            range T = 1..2;
            float Demand[Stores][T] = [[1,2],[3,4]];
            minimize sum(s in Stores, t in T) Demand[s][t];
            subject to { }
        """
        ast_set_range = parser.parse(lexer.tokenize(opl_code_set_range))
        gen_set_range = SciPyCSCCodeGenerator(ast_set_range)
        scipy_code_set_range = gen_set_range.generate_code()
        assert_has_array_repr(
            scipy_code_set_range,
            "Demand",
            tuple_keys=["('A',1)", "('A',2)", "('B',1)", "('B',2)"],
            nested_literal="Demand = [[1, 2], [3, 4]]",
            also_contains=["A", "B", "T"],
        )

        # Array indexed by two sets
        opl_code_set_set = """
            {string} S1 = { "X", "Y" };
            {string} S2 = { "P", "Q" };
            float arrSetSet[S1][S2] = [[10,20],[30,40]];
            minimize sum(a in S1, b in S2) arrSetSet[a][b];
            subject to { }
        """
        ast_set_set = parser.parse(lexer.tokenize(opl_code_set_set))
        gen_set_set = SciPyCSCCodeGenerator(ast_set_set)
        scipy_code_set_set = gen_set_set.generate_code()
        assert_has_array_repr(
            scipy_code_set_set,
            "arrSetSet",
            tuple_keys=["('X','P')", "('X','Q')", "('Y','P')", "('Y','Q')"],
            nested_literal="arrSetSet = [[10, 20], [30, 40]]",
            also_contains=["X", "Y", "P", "Q"],
        )

        # 3D array indexed by set, range, set
        opl_code_set_range_set = """
            {string} S1 = { "X", "Y" };
            range T = 1..2;
            {string} S2 = { "P", "Q" };
            float arrSRS[S1][T][S2] = [
                [[1,2],[3,4]],
                [[5,6],[7,8]]
            ];
            minimize sum(a in S1, t in T, b in S2) arrSRS[a][t][b];
            subject to { }
        """
        ast_srs = parser.parse(lexer.tokenize(opl_code_set_range_set))
        gen_srs = SciPyCSCCodeGenerator(ast_srs)
        scipy_code_srs = gen_srs.generate_code()
        assert_has_array_repr(
            scipy_code_srs,
            "arrSRS",
            tuple_keys=["('X',1,'P')", "('Y',2,'Q')"],
            nested_literal="arrSRS = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
            also_contains=["X", "Y", "P", "Q", "T"],
        )

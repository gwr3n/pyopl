import os
import tempfile
import unittest

from pyopl.pyopl_core import (
    GurobiCodeGenerator,
    OPLCompiler,
    OPLLexer,
    OPLParser,
    load_opl_model,
    solve,
)


def setUpModule():
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("pyopl.scipy_codegen_csc").setLevel(logging.DEBUG)
    logging.getLogger("pyopl.gurobi_codegen").setLevel(logging.DEBUG)


# Import pyopl interface
try:
    import pyopl

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class TestPyOPLProblems(unittest.TestCase):

    def test_production_planning_compare_solvers(self):
        """
        Test production planning model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            // Production Planning
            int nbProducts = ...;
            range Products = 1..nbProducts;
            int nbPeriods = ...;
            range Periods = 1..nbPeriods;
            float cost[Products][Periods] = ...;
            float demand[Periods] = ...;
            float capacity[Periods] = ...;

            dvar float+ x[Products][Periods];

            minimize sum(p in Products, t in Periods) cost[p][t] * x[p][t];

            subject to {
                forall(p in Products)
                    sum(t in Periods) x[p][t] >= demand[p];
                forall(t in Periods)
                    sum(p in Products) x[p][t] <= capacity[t];
            }
            """
        data_code = """
            nbProducts = 2;
            nbPeriods = 3;
            cost = [ [3, 2, 4], [2, 3, 5] ];
            demand = [40, 50, 0];
            capacity = [30, 40, 20];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
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
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_vehicle_routing_with_nested_tuples_dat(self):
        """
        Test vehicle routing problem with nested tuples, where arcs and nodes are read from a .dat file.
        Checks that both solvers return the same objective value.
        """
        model_code = """
        tuple Node {
            int id;
            float x;
            float y;
        };
        tuple Arc {
            Node from;
            Node to;
            float cost;
        };
        {Node} nodes = ...;
        {Arc} arcs = ...;
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            // Each node must have exactly one outgoing arc
            forall(n in nodes)
                sum(a in arcs : a.from.id == n.id) (x[a]) == 1;
            // Each node must have exactly one incoming arc
            forall(n in nodes)
                sum(a in arcs : a.to.id == n.id) (x[a]) == 1;
        }
        """
        data_code = """
        nodes = { <1,0.0,0.0>, <2,1.0,0.0>, <3,0.0,1.0> };
        arcs = { < <1,0.0,0.0>, <2,1.0,0.0>, 10.0 >, < <2,1.0,0.0>, <3,0.0,1.0>, 12.5 >, < <3,0.0,1.0>, <1,0.0,0.0>, 8.0 > };
        """
        obj_values = {}
        for solver in ("scipy", "gurobi"):
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
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_not_operator_in_forall_and_constraint(self):
        """TDD: Parser/codegen support for logical NOT '!' inside forall index constraint and implication.

        Uses: forall(i in 1..3 : !(i == 2)) x[i] >= 0; and (!(x[1] == 0)) => (x[1] >= 0);
        Should parse to AST containing 'not' nodes. Initially fails before implementation.
        """
        model_code = """
        range I = 1..3;
        dvar float x[I];
        dvar boolean y;
        minimize x[1];
        subject to {
            forall(i in I : !(i == 2)) x[i] >= 0;
            (!(x[1] == 0)) => (x[1] >= 0);
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        try:
            ast = parser.parse(lexer.tokenize(model_code))
        except Exception as e:
            raise e  # Ensure failing test prior to implementation

        def has_not(node):
            if isinstance(node, dict):
                if node.get("type") == "not":
                    return True
                return any(has_not(v) for v in node.values())
            if isinstance(node, list):
                return any(has_not(x) for x in node)
            return False

        self.assertTrue(has_not(ast), "Expected at least one 'not' node in AST for ! operator usage")

    def test_and_or_operators_in_constraint_and_implication(self):
        """TDD: Parser/codegen support for logical AND '&&' and OR '||' in constraints and implications.

        Model uses:
          (a == 1) && (b == 0);
          (a == 1) || (b == 1);
          (a == 1) && (b == 0) => y == 1;
          (a == 0) || (b == 1) => y == 0;
        Ensures AST contains 'and' and 'or' nodes. Gurobi codegen should succeed.
        """
        model_code = """
        dvar boolean a;
        dvar boolean b;
        dvar boolean y;
        minimize a;
        subject to {
            (a == 1) && (b == 0);
            (a == 1) || (b == 1);
            (a == 1) && (b == 0) => y == 1;
            (a == 0) || (b == 1) => y == 0;
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(model_code))
        found_and = False
        found_or = False

        def walk(node):
            nonlocal found_and, found_or
            if isinstance(node, dict):
                t = node.get("type")
                if t == "and":
                    found_and = True
                elif t == "or":
                    found_or = True
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)

        walk(ast)
        self.assertTrue(found_and, "Expected at least one 'and' node in AST for && operator usage")
        self.assertTrue(found_or, "Expected at least one 'or' node in AST for || operator usage")
        # Gurobi code generation should include 'and'/'or' text (string form of Python boolean ops)
        code = GurobiCodeGenerator(ast).generate_code()
        self.assertIn(" and ", code)
        self.assertIn(" or ", code)

    def test_composite_boolean_implication(self):
        """Composite antecedent (a && b) => (c || !d) linearization with auxiliaries (Gurobi) and fallback (SciPy).
        Gurobi should build model; SciPy currently lacks composite boolean linearization and should raise.
        """
        model_code = """
        dvar boolean a;
        dvar boolean b;
        dvar boolean c;
        dvar boolean d;
        minimize a;
        subject to {
            (a == 1) && (b == 1) => (c == 1) || !(d == 1);
        }
        """
        # Parse once
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(model_code))
        # Ensure 'and' and 'or' and 'not' nodes present
        found = {k: False for k in ["and", "or", "not"]}

        def walk(n):
            if isinstance(n, dict):
                t = n.get("type")
                if t in found:
                    found[t] = True
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for x in n:
                    walk(x)

        walk(ast)
        for k, v in found.items():
            self.assertTrue(v, f"Missing {k} node in composite implication AST")
        # Gurobi codegen should succeed and contain implication aux constructs
        code = GurobiCodeGenerator(ast).generate_code()
        # We no longer rely on specific 'impl_bin' name; ensure auxiliary binary variables were introduced
        self.assertRegex(code, r"_b\d+_c0")
        # SciPy solve should raise (unsupported) for now
        import os
        import tempfile

        from pyopl.pyopl_core import solve_with_scipy

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
            tmp.write(model_code)
            tmp.flush()
            path = tmp.name
        try:
            res = solve_with_scipy(path)
            # SciPy currently unsupported for implication => expect FAILED status, message may be generic
            self.assertEqual(res["status"], "FAILED")
            msg = res.get("message", "")
            self.assertTrue("Implication constraints are not supported" in msg or "Failed to load or parse OPL model" in msg)
        finally:
            os.remove(path)

    def test_vehicle_routing_with_nested_tuples(self):
        """
        This test extends the vehicle routing problem with tuples by including nested tuples.
        It checks tuple type, set of nested tuples, dvar indexed by nested tuples, and constraints using nested tuple fields.
        It also checks that both solvers return the same objective value.
        """
        code = """
        tuple Node {
            int id;
            float x;
            float y;
        };
        tuple Arc {
            Node from;
            Node to;
            float cost;
        };
        {Node} nodes = { <1,0.0,0.0>, <2,1.0,0.0>, <3,0.0,1.0> };
        {Arc} arcs = { < <1,0.0,0.0>, <2,1.0,0.0>, 10.0 >, < <2,1.0,0.0>, <3,0.0,1.0>, 12.5 >, < <3,0.0,1.0>, <1,0.0,0.0>, 8.0 > };
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            // Each node must have exactly one outgoing arc
            forall(n in nodes)
                sum(a in arcs : a.from.id == n.id) (x[a]) == 1;
            // Each node must have exactly one incoming arc
            forall(n in nodes)
                sum(a in arcs : a.to.id == n.id) (x[a]) == 1;
        }
        """
        obj_values = {}
        for solver in ("scipy", "gurobi"):
            print(f"\n[DEBUG] Parsing with solver: {solver}")
            print("[DEBUG] OPL code being parsed:")
            print(code)
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            # Check tuple type declarations
            tuple_types = [d for d in ast["declarations"] if d["type"] == "tuple_type"]
            self.assertEqual(len(tuple_types), 2)
            self.assertEqual(tuple_types[0]["name"], "Node")
            self.assertEqual(tuple_types[1]["name"], "Arc")
            # Check set of tuples declaration
            set_of_nodes_decl = next((d for d in ast["declarations"] if d.get("name") == "nodes"), None)
            set_of_arcs_decl = next((d for d in ast["declarations"] if d.get("name") == "arcs"), None)
            self.assertIsNotNone(set_of_nodes_decl)
            self.assertIsNotNone(set_of_arcs_decl)
            # Check dvar indexed by arcs
            dvar_decl = next((d for d in ast["declarations"] if d.get("name") == "x"), None)
            self.assertIsNotNone(dvar_decl)
            self.assertEqual(dvar_decl["var_type"], "boolean")
            # Check objective is sum over arcs of a.cost * x[a]
            obj = ast["objective"]
            self.assertEqual(obj["type"], "minimize")
            expr = obj["expression"]
            self.assertEqual(expr["type"], "sum")
            sum_expr = expr["expression"]
            self.assertEqual(sum_expr["type"], "binop")
            # Check left side is a field_access (a.cost)
            left = sum_expr["left"]
            self.assertEqual(left["type"], "field_access")
            self.assertEqual(left["field"], "cost")
            # Check right side is indexed_name (x[a])
            right = sum_expr["right"]
            self.assertEqual(right["type"], "indexed_name")
            self.assertEqual(right["name"], "x")
            # Check constraints: two forall constraints
            constraints = ast["constraints"]
            forall_constrs = [c for c in constraints if c["type"] == "forall_constraint"]
            self.assertEqual(len(forall_constrs), 2)
            # Check the first forall constraint structure (outgoing arcs)
            fc1 = forall_constrs[0]
            self.assertEqual(fc1["iterators"][0]["iterator"], "n")
            inner1 = fc1["constraint"]
            self.assertEqual(inner1["type"], "constraint")
            self.assertEqual(inner1["op"], "==")
            # The left side should be a sum with index constraint a.from.id == n.id
            left1 = inner1["left"]
            self.assertEqual(left1["type"], "sum")
            self.assertEqual(left1["index_constraint"]["type"], "binop")
            self.assertEqual(left1["index_constraint"]["op"], "==")
            # The right side should be 1
            self.assertEqual(inner1["right"]["type"], "number")
            self.assertEqual(inner1["right"]["value"], 1)
            # --- Solve the model and store the objective value ---
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                model_file = tmp.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        # Check that both solvers return the same objective value (within tolerance)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_wagner_whitin_linear(self):
        """
        Test Wagner-Whitin 5-period lot-sizing model with both solvers.
        Checks that both solvers produce the expected objective and solution for the Wagner-Whitin model with provided data.
        """
        model_code = """
        // Wagner-Whitin 5-period lot-sizing model (PyOPL syntax)

        int T = 5; // Number of periods
        float demand[1..T] = [20, 40, 30, 10, 50]; // Demand per period
        float unit_cost = 2;   // Unit production cost per period
        float setup_cost = 100; // Setup cost per period
        float holding_cost = 1; // Holding cost per period

        dvar float x[1..T]; // Amount produced in period t
        dvar float s[0..T]; // Inventory at end of period t
        dvar boolean y[1..T]; // 1 if setup/order occurs in period t

        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);

        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T)
                x[t] <= y[t] * sum(tt in t..T) demand[tt] ;
            forall(t in 1..T) {
                x[t] >= 0;
                s[t] >= 0;
            }
            y[1] == 1;
            y[2] == 0;
            y[3] == 0;
            y[4] == 0;
            y[5] == 1;
        }
        """
        expected_obj = 630.0
        expected_solution = {
            "x[1]": 100.0,
            "x[2]": 0.0,
            "x[3]": 0.0,
            "x[4]": 0.0,
            "x[5]": 50.0,
            "s[0]": 0.0,
            "s[1]": 80.0,
            "s[2]": 40.0,
            "s[3]": 10.0,
            "s[4]": 0.0,
            "s[5]": 0.0,
            "y[1]": 1.0,
            "y[2]": 0.0,
            "y[3]": 0.0,
            "y[4]": 0.0,
            "y[5]": 1.0,
        }
        import os
        import tempfile

        from pyopl.pyopl_core import solve, solve_with_scipy

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
            tmp_mod.write(model_code)
            tmp_mod.flush()
            model_file = tmp_mod.name
            try:
                for solver, solve_fn in [
                    ("gurobi", solve),
                    ("scipy", solve_with_scipy),
                ]:
                    result = solve_fn(model_file)
                    self.assertNotEqual(result["status"], "FAILED")
                    self.assertIn("objective_value", result)
                    self.assertAlmostEqual(result["objective_value"], expected_obj, places=4)
                    sol = result.get("solution", {})
                    # Normalize variable names for comparison
                    norm_sol = {self.normalize_varname(k): v for k, v in sol.items()}
                    for k, v in expected_solution.items():
                        self.assertIn(k, norm_sol, f"Missing variable {k} in {solver} solution")
                        self.assertAlmostEqual(
                            norm_sol[k],
                            v,
                            places=4,
                            msg=f"{solver}: {k}={norm_sol[k]}, expected {v}",
                        )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)

    def test_wagner_whitin_model_data(self):
        """
        Test Wagner-Whitin 5-period lot-sizing model with both solvers.
        Checks that both solvers produce the expected objective and solution for the Wagner-Whitin model with provided data
        """
        model_code = """
        // Wagner-Whitin 5-period lot-sizing model (PyOPL syntax)

        int T = ...; // Number of periods
        float demand[1..T] = ...; // Demand per period
        float unit_cost = ...;    // Unit production cost per period
        float setup_cost = ...;   // Setup cost per period
        float holding_cost = ...; // Holding cost per period

        dvar float x[1..T]; // Amount produced in period t
        dvar float s[0..T]; // Inventory at end of period t
        dvar boolean y[1..T]; // 1 if setup/order occurs in period t

        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);

        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T)
                x[t] <= (sum(tt in t..T) demand[t]) * y[t];
            forall(t in 1..T) {
                x[t] >= 0;
                s[t] >= 0;
            }
        }
        """
        data_code = """
        // Wagner-Whitin 5-period lot-sizing model data (PyOPL syntax)
        T = 5; // Number of periods
        demand = [20, 40, 30, 10, 50]; // Demand per period
        unit_cost = 2;   // Unit production cost per period
        setup_cost = 100; // Setup cost per period
        holding_cost = 1; // Holding cost per period
        """
        expected_obj = 630.0
        expected_solution = {
            "x[1]": 100.0,
            "x[2]": 0.0,
            "x[3]": 0.0,
            "x[4]": 0.0,
            "x[5]": 50.0,
            "s[0]": 0.0,
            "s[1]": 80.0,
            "s[2]": 40.0,
            "s[3]": 10.0,
            "s[4]": 0.0,
            "s[5]": 0.0,
            "y[1]": 1.0,
            "y[2]": 0.0,
            "y[3]": 0.0,
            "y[4]": 0.0,
            "y[5]": 1.0,
        }
        import os
        import tempfile

        from pyopl.pyopl_core import solve, solve_with_scipy

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
        ):
            tmp_mod.write(model_code)
            tmp_mod.flush()
            model_file = tmp_mod.name
            tmp_dat.write(data_code)
            tmp_dat.flush()
            data_file = tmp_dat.name
            try:
                for solver, solve_fn in [
                    ("gurobi", solve),
                    ("scipy", solve_with_scipy),
                ]:
                    result = solve_fn(model_file, data_file)
                    self.assertNotEqual(result["status"], "FAILED")
                    self.assertIn("objective_value", result)
                    self.assertAlmostEqual(result["objective_value"], expected_obj, places=4)
                    sol = result.get("solution", {})
                    # Normalize variable names for comparison
                    norm_sol = {self.normalize_varname(k): v for k, v in sol.items()}
                    for k, v in expected_solution.items():
                        self.assertIn(k, norm_sol, f"Missing variable {k} in {solver} solution")
                        self.assertAlmostEqual(
                            norm_sol[k],
                            v,
                            places=4,
                            msg=f"{solver}: {k}={norm_sol[k]}, expected {v}",
                        )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)

    def test_wagner_whitin_implication(self):
        """
        Variant of Wagner-Whitin 5-period lot-sizing model using implication constraints:
        x[t] > 0 => y[t] == 1
        Should solve with Gurobi, and raise error with SciPy.
        """
        model_code = """
        // Wagner-Whitin 5-period lot-sizing model with implication constraint
        int T = 5;
        float demand[1..T] = [20, 40, 30, 10, 50];
        float unit_cost = 2;
        float setup_cost = 100;
        float holding_cost = 1;

        dvar float x[1..T];
        dvar float s[0..T];
        dvar boolean y[1..T];

        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);

        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            //forall(t in 1..T)
            //    x[t] <= (sum(tt in t..T) demand[t]) * y[t];
            forall(t in 1..T) {
                x[t] >= 0;
                s[t] >= 0;
            }
            // Implication: if x[t] > 0 then y[t] == 1
            forall(t in 1..T)
                (x[t] > 0) => (y[t] == 1);
        }
        """
        expected_obj = 630.0
        expected_solution = {
            "x[1]": 100.0,
            "x[2]": 0.0,
            "x[3]": 0.0,
            "x[4]": 0.0,
            "x[5]": 50.0,
            "s[0]": 0.0,
            "s[1]": 80.0,
            "s[2]": 40.0,
            "s[3]": 10.0,
            "s[4]": 0.0,
            "s[5]": 0.0,
            "y[1]": 1.0,
            "y[2]": 0.0,
            "y[3]": 0.0,
            "y[4]": 0.0,
            "y[5]": 1.0,
        }
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
            tmp_mod.write(model_code)
            tmp_mod.flush()
            model_file = tmp_mod.name
        try:
            # Gurobi: should solve
            result = solve(model_file, solver="gurobi")
            self.assertNotEqual(result["status"], "FAILED")
            self.assertIn("objective_value", result)
            self.assertAlmostEqual(result["objective_value"], expected_obj, places=4)
            sol = result.get("solution", {})
            norm_sol = {self.normalize_varname(k): v for k, v in sol.items()}
            for k, v in expected_solution.items():
                self.assertIn(k, norm_sol, f"Missing variable {k} in gurobi solution")
                self.assertAlmostEqual(
                    norm_sol[k],
                    v,
                    places=4,
                    msg=f"gurobi: {k}={norm_sol[k]}, expected {v}",
                )
            # SciPy: now supports this implication via big-M gating (x <= M*y)
            result_scipy = solve(model_file, solver="scipy")
            self.assertNotEqual(result_scipy["status"], "FAILED")
            self.assertIn("objective_value", result_scipy)
            self.assertAlmostEqual(result_scipy["objective_value"], expected_obj, places=4)
            sol_scipy = result_scipy.get("solution", {})
            norm_sol_scipy = {self.normalize_varname(k): v for k, v in sol_scipy.items()}
            for k, v in expected_solution.items():
                self.assertIn(k, norm_sol_scipy, f"Missing variable {k} in scipy solution")
                self.assertAlmostEqual(
                    norm_sol_scipy[k],
                    v,
                    places=4,
                    msg=f"scipy: {k}={norm_sol_scipy[k]}, expected {v}",
                )
        finally:
            if os.path.exists(model_file):
                os.remove(model_file)

    def run_test_case_gurobi(self, opl_code):
        """Helper: Check Gurobi code generation for a given OPL model string."""
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = GurobiCodeGenerator(ast)
        gurobi_code = generator.generate_code()
        # Just check that code generation does not raise
        self.assertIsInstance(ast, dict)
        self.assertIsInstance(gurobi_code, str)

    def test_job_shop(self):
        """
        Warehouse Location Problem
        """
        model_code = """
        // Job Shop Scheduling Problem
        int nbJobs = ...;
        int nbMachines = ...;
        range Jobs = 1..nbJobs;
        range Machines = 1..nbMachines;
        int duration[Jobs][Machines] = ...;
        int M = 1000;

        dvar int+ start[Jobs][Machines];
        dvar boolean z[Jobs][Jobs][Machines];
        dvar int+ makespan;

        minimize makespan;

        subject to {
        // Each job must be processed on each machine in order
        forall(j in Jobs, m in Machines)
            start[j][m] >= 0;
        // No overlap on machines (simplified)
        forall(m in Machines)
            forall(j1 in Jobs, j2 in Jobs: j1 != j2){
            start[j1][m] + duration[j1][m] <=  start[j2][m] - 1 + M * z[j1][j2][m];
            start[j2][m] + duration[j2][m] <=  start[j1][m] - 1 + M * (1 - z[j1][j2][m]);
            }
        // Each job must be processed on each machine in order
        forall(j in Jobs, m in 1..nbMachines-1)
            start[j][m+1] >= start[j][m] + duration[j][m];
        // Makespan constraint
        forall(j in Jobs)
            makespan >= start[j][nbMachines] + duration[j][nbMachines];
        }
        """
        data_code = """
        nbJobs = 3;
        nbMachines = 2;
        duration = [
        [3, 2],   // Job 1: Machine 1 = 3, Machine 2 = 2
        [2, 4],   // Job 2: Machine 1 = 2, Machine 2 = 4
        [5, 1]    // Job 3: Machine 1 = 5, Machine 2 = 1
        ];
        """
        obj_values = {}
        import tempfile

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
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_warehouse_location(self):
        """
        Warehouse Location Problem
        """
        model_code = """
        // Warehouse Location Problem
        int nbWarehouses = ...;
        int nbCustomers = ...;

        range Warehouses = 1..nbWarehouses;
        range Customers = 1..nbCustomers;

        float fixed_cost[Warehouses] = ...;
        float trans_cost[Warehouses][Customers] = ...;
        float demand[Customers] = ...;
        float capacity[Warehouses] = ...;

        dvar boolean y[Warehouses];
        dvar float+ x[Warehouses][Customers];

        minimize sum(i in Warehouses) fixed_cost[i] * y[i] + sum(i in Warehouses, j in Customers) trans_cost[i][j] * x[i][j];

        subject to {
        forall(j in Customers)
            sum(i in Warehouses) x[i][j] == demand[j];
        forall(i in Warehouses, j in Customers)
            x[i][j] <= capacity[i] * y[i];
        }
        """
        data_code = """
        nbWarehouses = 2;
        nbCustomers = 3;
        fixed_cost = [80, 90];
        trans_cost = [ [3, 5, 8],
                       [4, 3, 6] ];
        demand = [15, 20, 10];
        capacity = [25, 30];
        """
        obj_values = {}
        import tempfile

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
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_graph_coloring_tuples(self):
        """
        Graph Coloring Problem using tuples and sets.
        """
        model_code = """
        // Proper Graph Coloring Problem (no !=, uses big-M encoding)
        int nbNodes = ...;
        range Nodes = 1..nbNodes;

        tuple Edge {
            int source;
            int dest;
        };

        {Edge} Edges = ...;

        dvar int+ color[Nodes];
        dvar int+ maxColor;
        dvar boolean z[Edges]; // auxiliary binary for big-M encoding

        minimize maxColor;

        subject to {
            // Each node's color is at least 1 and at most nbNodes
            forall(i in Nodes) color[i] >= 1;
            forall(i in Nodes) color[i] <= nbNodes;
            // Adjacent nodes must have different colors (big-M encoding)
            forall(e in Edges)
                color[e.source] >= color[e.dest] + 1 - nbNodes * z[e];
            forall(e in Edges)
                color[e.dest] >= color[e.source] + 1 - nbNodes * (1 - z[e]);
            // maxColor is at least as large as any color used
            forall(i in Nodes) maxColor >= color[i];
        }
        """
        data_code = """
        nbNodes = 4;
        Edges = { <1,2>, <2,3>, <3,4>, <4,1> };
        """
        obj_values = {}
        import tempfile

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
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_graph_coloring_matrix(self):
        """
        Graph Coloring Problem using an adjacency matrix (no tuples).
        """
        model_code = """
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        int adj[Nodes][Nodes] = ...; // adjacency matrix: 1 if edge, 0 otherwise

        dvar int+ color[Nodes];
        dvar int+ maxColor;
        dvar boolean z[Nodes][Nodes]; // auxiliary binary for big-M encoding

        minimize maxColor;

        subject to {
            // Each node's color is at least 1 and at most nbNodes
            forall(i in Nodes) color[i] >= 1;
            forall(i in Nodes) color[i] <= nbNodes;
            // Adjacent nodes must have different colors
            forall(i in Nodes, j in Nodes : adj[i][j] == 1)
                color[i] >= color[j] + 1 - nbNodes * z[i][j];
            forall(i in Nodes, j in Nodes : adj[i][j] == 1)
                color[j] >= color[i] + 1 - nbNodes * (1-z[i][j]);
            // maxColor is at least as large as any color used
            forall(i in Nodes) maxColor >= color[i];
        }
        """
        data_code = """
        nbNodes = 4;
        adj = [
            [0,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
            [1,0,1,0]
        ];
        """
        obj_values = {}
        import tempfile

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
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_vehicle_routing_matrix_dat(self):
        """
        Test vehicle routing problem with matrix, where arcs are read from a .dat file.
        """

        model_code = """
        // Matrix-based vehicle routing problem
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        float cost[Nodes][Nodes] = ...;
        dvar boolean x[Nodes][Nodes];
        minimize sum(i in Nodes, j in Nodes) cost[i][j] * x[i][j];
        subject to {
        forall(i in Nodes)
            sum(j in Nodes) (x[i][j]) == 1;
        forall(j in Nodes)
            sum(i in Nodes) (x[i][j]) == 1;
        }
        """
        data_code = """
        nbNodes = 3;
        cost = [
            [1000, 10.0, 1000],
            [1000, 1000, 12.5],
            [8.0, 1000, 1000]
        ];
        """
        obj_values = {}
        import tempfile

        for solver in ("scipy", "gurobi"):

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
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_vehicle_routing_with_tuples_dat(self):
        """
        Test vehicle routing problem with tuples, where arcs are read from a .dat file.
        """

        model_code = """
        tuple Arc {
            int from;
            int to;
            float cost;
        };
        {Arc} arcs = ...;
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            forall(i in 1..3)
                sum(a in arcs : a.from == i) (x[a]) == 1;
            forall(j in 1..3)
                sum(a in arcs : a.to == j) (x[a]) == 1;
        }
        """
        data_code = """
        arcs = { <1,2,10.0>, <2,3,12.5>, <3,1,8.0> };
        """
        obj_values = {}
        import tempfile

        for solver in ("scipy", "gurobi"):

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
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_vehicle_routing_with_tuples(self):
        """
        This test embeds a small vehicle routing problem using tuples, similar to classical OPL models.
        It checks tuple type, set of tuples, dvar indexed by tuples, and constraints using tuple fields.
        It also checks that both solvers return the same objective value.
        """
        code = """
        tuple Arc {
            int from;
            int to;
            float cost;
        };
        {Arc} arcs = { <1,2,10.0>, <2,3,12.5>, <3,1,8.0> };
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            // Each node must have exactly one outgoing arc
            forall(i in 1..3)
                sum(a in arcs : a.from == i) (x[a]) == 1;
            // Each node must have exactly one incoming arc
            forall(j in 1..3)
                sum(a in arcs : a.to == j) (x[a]) == 1;
        }
        """
        import os
        import tempfile

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            print(f"\n[DEBUG] Parsing with solver: {solver}")
            print("[DEBUG] OPL code being parsed:")
            print(code)
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            # Check tuple type declaration
            tuple_type_decl = next((d for d in ast["declarations"] if d["type"] == "tuple_type"), None)
            self.assertIsNotNone(tuple_type_decl)
            self.assertEqual(tuple_type_decl["name"], "Arc")
            # Check set of tuples declaration
            set_of_tuples_decl = next((d for d in ast["declarations"] if d.get("name") == "arcs"), None)
            self.assertIsNotNone(set_of_tuples_decl)
            # Check dvar indexed by arcs
            dvar_decl = next((d for d in ast["declarations"] if d.get("name") == "x"), None)
            self.assertIsNotNone(dvar_decl)
            self.assertEqual(dvar_decl["var_type"], "boolean")
            # Check objective is sum over arcs of a.cost * x[a]
            obj = ast["objective"]
            self.assertEqual(obj["type"], "minimize")
            expr = obj["expression"]
            self.assertEqual(expr["type"], "sum")
            sum_expr = expr["expression"]
            self.assertEqual(sum_expr["type"], "binop")
            # Check left side is a field_access (a.cost)
            left = sum_expr["left"]
            self.assertEqual(left["type"], "field_access")
            self.assertEqual(left["field"], "cost")
            # Check right side is indexed_name (x[a])
            right = sum_expr["right"]
            self.assertEqual(right["type"], "indexed_name")
            self.assertEqual(right["name"], "x")
            # Check constraints: two forall constraints
            constraints = ast["constraints"]
            forall_constrs = [c for c in constraints if c["type"] == "forall_constraint"]
            self.assertEqual(len(forall_constrs), 2)
            # Check the first forall constraint structure (outgoing arcs)
            fc1 = forall_constrs[0]
            self.assertEqual(fc1["iterators"][0]["iterator"], "i")
            inner1 = fc1["constraint"]
            self.assertEqual(inner1["type"], "constraint")
            self.assertEqual(inner1["op"], "==")
            # The left side should be a sum with index constraint a.from == i
            left1 = inner1["left"]
            self.assertEqual(left1["type"], "sum")
            self.assertEqual(left1["index_constraint"]["type"], "binop")
            self.assertEqual(left1["index_constraint"]["op"], "==")
            # The right side should be 1
            self.assertEqual(inner1["right"]["type"], "number")
            self.assertEqual(inner1["right"]["value"], 1)
            # --- Solve the model and store the objective value ---
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                model_file = tmp.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        # Check that both solvers return the same objective value (within tolerance)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_basic_production_planning_gurobi(self):
        """Test Gurobi codegen for a basic production planning model."""
        opl_code = """
        dvar float x;
        dvar float y;

        maximize x + y;

        subject to {
            x <= 10;
            y <= 15;
            x + y <= 20;
        }
        """
        self.run_test_case_gurobi(opl_code)

    def run_test_case_scipy(self, opl_code, data_dict=None):
        """Helper: Check SciPy code generation for a given OPL model string."""
        from pyopl.pyopl_core import OPLLexer, OPLParser, SciPyCodeGenerator

        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = SciPyCodeGenerator(ast, data_dict or {})
        scipy_code = generator.generate_code()
        self.assertIsInstance(ast, dict)
        self.assertIsInstance(scipy_code, str)

    def test_basic_production_planning_scipy(self):
        """Test SciPy codegen for a basic production planning model."""
        opl_code = """
        dvar float x;
        dvar float y;

        maximize x + y;

        subject to {
            x <= 10;
            y <= 15;
            x + y <= 20;
        }
        """
        self.run_test_case_scipy(opl_code)

    def pyopl_vs_cplex_output(self, model, data, cplex_obj=None):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsack.mod/dat."""

        # Solve with pyopl (default solver is gurobi)
        result = pyopl.solve(model, data)
        self.assertNotIn(result["status"], ["ERROR", "FAILED", "EXECUTION_ERROR"])
        if isinstance(result, dict) and "objective_value" in result:
            gurobi_obj = result["objective_value"]
        else:
            gurobi_obj = result
        if cplex_obj is not None:
            self.assertAlmostEqual(
                gurobi_obj,
                cplex_obj,
                places=4,
                msg=f"CPLEX: {cplex_obj}, pyopl+gurobi: {gurobi_obj}",
            )

        # Solve with pyopl using scipy
        result_scipy = pyopl.solve(model, data, solver="scipy")
        self.assertNotIn(result_scipy["status"], ["ERROR", "FAILED", "EXECUTION_ERROR"])
        if isinstance(result_scipy, dict) and "objective_value" in result_scipy:
            scipy_obj = result_scipy["objective_value"]
        else:
            scipy_obj = result_scipy
        if cplex_obj is not None:
            self.assertAlmostEqual(
                scipy_obj,
                cplex_obj,
                places=4,
                msg=f"CPLEX: {cplex_obj}, pyopl+scipy: {scipy_obj}",
            )

        if cplex_obj is None:
            self.assertAlmostEqual(
                scipy_obj,
                gurobi_obj,
                places=4,
                msg=f"CPLEX: {cplex_obj}, pyopl+scipy: {scipy_obj}",
            )

    @unittest.skipUnless(GUROBI_AVAILABLE, "pyopl or gurobi not available")
    def test_knapsack_pyopl_vs_cplex_output(self):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsack.mod/dat."""
        # CPLEX reference solution
        cplex_obj = 10.0

        KNAPSACK_MOD = os.path.join(os.path.dirname(__file__), "../opl_models/knapsack/knapsack.mod")
        KNAPSACK_DAT = os.path.join(os.path.dirname(__file__), "../opl_models/knapsack/knapsack.dat")

        self.pyopl_vs_cplex_output(KNAPSACK_MOD, KNAPSACK_DAT, cplex_obj)

    @unittest.skipUnless(GUROBI_AVAILABLE, "pyopl or gurobi not available")
    def test_knapsackp_pyopl_vs_cplex_output(self):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsackp.mod/dat."""
        # CPLEX reference solution
        cplex_obj = 498.0

        KNAPSACKP_MOD = os.path.join(os.path.dirname(__file__), "../opl_models/knapsack/knapsackp.mod")
        KNAPSACKP_DAT = os.path.join(os.path.dirname(__file__), "../opl_models/knapsack/knapsackp.dat")

        self.pyopl_vs_cplex_output(KNAPSACKP_MOD, KNAPSACKP_DAT, cplex_obj)

    @unittest.skipUnless(GUROBI_AVAILABLE, "pyopl or gurobi not available")
    def test_inventory_routing_pyopl_vs_cplex_output(self):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsackp.mod/dat."""
        # CPLEX reference solution
        cplex_obj = 103.0

        INVENTORY_ROUTING_MOD = os.path.join(
            os.path.dirname(__file__),
            "../opl_models/inventory_routing/inventory_routing.mod",
        )
        INVENTORY_ROUTING_DAT = os.path.join(
            os.path.dirname(__file__),
            "../opl_models/inventory_routing/inventory_routing.dat",
        )

        self.pyopl_vs_cplex_output(INVENTORY_ROUTING_MOD, INVENTORY_ROUTING_DAT, cplex_obj)

    def test_tsp_model_parsing_and_codegen_gurobi(self):
        """Test parsing and codegen for the TSP model (Gurobi)."""
        # Paths to the TSP model and data
        model_path = os.path.join(os.path.dirname(__file__), "../opl_models/tsp/tsp.mod")
        data_path = os.path.join(os.path.dirname(__file__), "../opl_models/tsp/tsp.dat")
        with open(model_path) as f:
            model_code = f.read()
        with open(data_path) as f:
            data_code = f.read()
        compiler = OPLCompiler()
        ast, gurobi_code, data_dict = compiler.compile_model(model_code, data_code, solver="gurobi")
        print("\n==== DEBUG: Generated Gurobi Code ====")
        print(gurobi_code)
        print("==== END DEBUG ====")

        def find_node_with_index_constraint(node, node_type):
            if isinstance(node, dict):
                if node.get("type") == node_type and node.get("index_constraint") is not None:
                    return True
                # Recursively search all dict/list children
                for v in node.values():
                    if find_node_with_index_constraint(v, node_type):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if find_node_with_index_constraint(item, node_type):
                        return True
            return False

        found_sum = find_node_with_index_constraint(ast, "sum")
        found_forall = find_node_with_index_constraint(ast, "forall_constraint")
        self.assertTrue(found_sum, "Sum with index constraint not found in AST")
        self.assertTrue(found_forall, "Forall with index constraint not found in AST")
        # Check that the generated code uses itertools.product and 'if' for index constraint
        self.assertIn("itertools.product", gurobi_code)
        self.assertIn("if ", gurobi_code)
        self.assertIn("gp.quicksum", gurobi_code)
        # Optionally, check that the code compiles
        compile(gurobi_code, "<string>", "exec")

    def test_tsp_model_parsing_and_codegen_scipy(self):
        """Test parsing and codegen for the TSP model (SciPy)."""
        # Paths to the TSP model and data
        model_path = os.path.join(os.path.dirname(__file__), "../opl_models/tsp/tsp.mod")
        data_path = os.path.join(os.path.dirname(__file__), "../opl_models/tsp/tsp.dat")
        with open(model_path) as f:
            model_code = f.read()
        with open(data_path) as f:
            data_code = f.read()
        compiler = OPLCompiler()
        ast, scipy_code, data_dict = compiler.compile_model(model_code, data_code, solver="scipy")
        print("\n==== DEBUG: Generated SciPy Code ====")
        print(scipy_code)
        print("==== END DEBUG ====")

        def find_node_with_index_constraint(node, node_type):
            if isinstance(node, dict):
                if node.get("type") == node_type and node.get("index_constraint") is not None:
                    return True
                for v in node.values():
                    if find_node_with_index_constraint(v, node_type):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if find_node_with_index_constraint(item, node_type):
                        return True
            return False

        found_sum = find_node_with_index_constraint(ast, "sum")
        found_forall = find_node_with_index_constraint(ast, "forall_constraint")
        self.assertTrue(found_sum, "Sum with index constraint not found in AST (scipy)")
        self.assertTrue(found_forall, "Forall with index constraint not found in AST (scipy)")
        self.assertIn("linprog", scipy_code)
        self.assertIn("if ", scipy_code)
        # Optionally, check that the code compiles
        compile(scipy_code, "<string>", "exec")

    def test_knapsack_problem_compare_solvers(self):
        """Compare Gurobi and SciPy solutions for a generated knapsack problem."""
        from pyopl.pyopl_core import solve, solve_with_scipy

        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write(
                    """
                range Items = 1..5;
                param float weight[1..5];
                param float value[1..5];
                param float C;

                dvar boolean x[1..5];

                maximize sum (i in Items) (value[i] * x[i]);

                subject to {
                    sum (i in Items) (weight[i] * x[i]) <= C;
                }
                """
                )
            with open(dummy_data_file, "w") as f:
                f.write(
                    """
                weight = [2,3,4,5,5];
                value = [2,3,4,5,5];
                C = 10;
                """
                )
            # Gurobi
            result_gurobi = solve(dummy_model_file, dummy_data_file)
            # SciPy
            result_scipy = solve_with_scipy(dummy_model_file, dummy_data_file)
            # Print diagnostic output
            print("Gurobi solution:", result_gurobi.get("solution", {}))
            print("Gurobi objective:", result_gurobi.get("objective_value"))
            print("SciPy solution:", result_scipy.get("solution", {}))
            print("SciPy objective:", result_scipy.get("objective_value"))
            # Only compare objectives, since multiple optima are possible
            try:
                self.compare_objectives(
                    result_gurobi.get("objective_value"),
                    result_scipy.get("objective_value"),
                )
            except AssertionError as e:
                msg = (
                    f"Objective mismatch in knapsack_problem_compare_solvers.\n"
                    f"Gurobi objective: {result_gurobi.get('objective_value')}\n"
                    f"SciPy objective: {result_scipy.get('objective_value')}\n"
                    f"Gurobi solution: {result_gurobi.get('solution', {})}\n"
                    f"SciPy solution: {result_scipy.get('solution', {})}\n"
                )
                raise AssertionError(msg) from e
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_knapsack_problem_scipy(self):
        """Test SciPy codegen and solution for a generated knapsack problem."""
        from pyopl.pyopl_core import load_opl_model, solve_with_scipy

        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write(
                    """
                range Items = 1..5;
                param float weight[1..5];
                param float value[1..5];
                param float C;

                dvar boolean x[1..5];

                maximize sum (i in Items) (value[i] * x[i]);

                subject to {
                    sum (i in Items) (weight[i] * x[i]) <= C;
                }
                """
                )
            with open(dummy_data_file, "w") as f:
                f.write(
                    """
                weight = [2,3,4,5,5];
                value = [2,3,4,5,5];
                C = 10;
                """
                )
            ast, scipy_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file)
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

    def normalize_varname(self, name):
        import re

        # Accept x_1, x_1_2, x[1], x[1,2] and map all to canonical form x[1] or x[1,2]
        # Match var_1_2_3 -> var[1,2,3]
        m = re.match(r"([a-zA-Z][a-zA-Z0-9]*)((?:_[0-9]+)+)$", name)
        if m:
            indices = m.group(2).lstrip("_").split("_")
            return f"{m.group(1)}[{','.join(indices)}]"
        # Match var[1,2,3] or var[1]
        m = re.match(r"([a-zA-Z][a-zA-Z0-9]*)\[([0-9,]+)\]$", name)
        if m:
            return f"{m.group(1)}[{m.group(2)}]"
        # Match var_1 -> var[1]
        m = re.match(r"([a-zA-Z][a-zA-Z0-9]*)_([0-9]+)$", name)
        if m:
            return f"{m.group(1)}[{m.group(2)}]"
        return name

    def compare_solutions(self, sol1, sol2, tol=1e-5):
        # Normalize variable names in both solutions
        norm1 = {self.normalize_varname(k): v for k, v in sol1.items()}
        norm2 = {self.normalize_varname(k): v for k, v in sol2.items()}
        self.assertEqual(set(norm1.keys()), set(norm2.keys()))
        for k in norm1:
            self.assertAlmostEqual(norm1[k], norm2[k], delta=tol)

    def compare_objectives(self, obj1, obj2, tol=1e-5):
        self.assertAlmostEqual(obj1, obj2, delta=tol)

    def test_assignment_problem_compare_solvers(self):
        """Compare Gurobi and SciPy solutions for a generated assignment problem."""
        from pyopl.pyopl_core import solve, solve_with_scipy

        dummy_model_file = "assign_model.mod"
        dummy_data_file = "assign_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write(
                    """
                dvar boolean assign[1..2][1..2];
                range Persons = 1..2;
                range Tasks = 1..2;

                minimize sum (p in Persons) (sum (t in Tasks) (5 * assign[p][t]));

                subject to {
                    forall (p in Persons)
                        sum (t in Tasks) (assign[p][t]) == 1;
                    forall (t in Tasks)
                        sum (p in Persons) (assign[p][t]) == 1;
                }
                """
                )
            with open(dummy_data_file, "w") as f:
                f.write("")  # No data needed
            # Gurobi
            result_gurobi = solve(dummy_model_file, dummy_data_file)
            # SciPy
            result_scipy = solve_with_scipy(dummy_model_file, dummy_data_file)
            # Compare solutions
            try:
                self.compare_solutions(result_gurobi.get("solution", {}), result_scipy.get("solution", {}))
            except AssertionError as e:
                msg = (
                    f"Solution mismatch in assignment_problem_compare_solvers.\n"
                    f"Gurobi solution: {result_gurobi.get('solution', {})}\n"
                    f"SciPy solution: {result_scipy.get('solution', {})}\n"
                )
                raise AssertionError(msg) from e
            try:
                self.compare_objectives(
                    result_gurobi.get("objective_value"),
                    result_scipy.get("objective_value"),
                )
            except AssertionError as e:
                msg = (
                    f"Objective mismatch in assignment_problem_compare_solvers.\n"
                    f"Gurobi objective: {result_gurobi.get('objective_value')}\n"
                    f"SciPy objective: {result_scipy.get('objective_value')}\n"
                )
                raise AssertionError(msg) from e
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_knapsack_problem(self):
        """Test Gurobi codegen and parsing for a generated knapsack problem."""
        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write(
                    """
                range Items = 1..5;
                param float weight[1..5];
                param float value[1..5];
                param float C;

                dvar boolean x[1..5];

                maximize sum (i in Items) (value[i] * x[i]);

                subject to {
                    sum (i in Items) (weight[i] * x[i]) <= C;
                }
                """
                )
            with open(dummy_data_file, "w") as f:
                f.write(
                    """
                weight = [2,3,4,5,5];
                value = [2,3,4,5,5];
                C = 10;
                """
                )
            ast, gurobi_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file)
            self.assertIsInstance(ast, dict)
            self.assertIsInstance(gurobi_code, str)
            self.assertIsInstance(data_dict, dict)
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_multi_resource_knapsack_problem(self):
        """Test Gurobi codegen and parsing for a multi-resource knapsack problem."""
        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write(
                    """
                        range Items = 1..12;
                        range Resources = 1..7;
                        float Capacity[Items] = ...;
                        float Value[Items];
                        float Use[Resources][Items];

                        dvar boolean Take[Items];

                        maximize sum(i in Items) Value[i] * Take[i];

                        subject to {
                        forall( r in Resources )
                            sum( i in Items )
                                Use[r][i] * Take[i] <= Capacity[r];
                        }
                """
                )
            with open(dummy_data_file, "w") as f:
                f.write(
                    """
                        Capacity = [ 18209, 7692, 1333, 924, 26638, 61188, 13360,
                                     18209, 7692, 1333, 924, 26638 ];
                        Value = [ 96, 76, 56, 11, 86, 10, 66, 86, 83, 12, 9, 81 ];
                        Use = [ [ 19,   1,  10,  1,   1,  14, 152, 11,  1,   1, 1, 1 ],
                            [  0,   4,  53,  0,   0,  80,   0,  4,  5,   0, 0, 0 ],
                            [  4, 660,   3,  0,  30,   0,   3,  0,  4,  90, 0, 0],
                            [  7,   0,  18,  6, 770, 330,   7,  0,  0,   6, 0, 0],
                            [  0,  20,   0,  4,  52,   3,   0,  0,  0,   5, 4, 0],
                            [  0,   0,  40, 70,   4,  63,   0,  0, 60,   0, 4, 0],
                            [  0,  32,   0,  0,   0,   5,   0,  3,  0, 660, 0, 9]];
                """
                )
            ast, gurobi_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file)
            self.assertIsInstance(ast, dict)
            self.assertIsInstance(gurobi_code, str)
            self.assertIsInstance(data_dict, dict)
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_transportation_problem(self):
        """Test Gurobi codegen for a transportation problem."""
        opl_code = """
        dvar float flow[1..2][1..3];
        range Origins = 1..2;
        range Destinations = 1..3;

        minimize sum (i in Origins) (sum (j in Destinations) (10 * flow[i][j]));

        subject to {
            forall (i in Origins)
                sum (j in Destinations) (flow[i][j]) <= 100;
            forall (j in Destinations)
                sum (i in Origins) (flow[i][j]) >= 50;
        }
        """
        self.run_test_case_gurobi(opl_code)

    def test_simple_assignment_problem(self):
        """Test Gurobi codegen for a simple assignment problem."""
        opl_code = """
        dvar boolean assign[1..2][1..2];
        range Persons = 1..2;
        range Tasks = 1..2;

        minimize sum (p in Persons) (sum (t in Tasks) (5 * assign[p][t]));

        subject to {
            forall (p in Persons)
                sum (t in Tasks) (assign[p][t]) == 1;
            forall (t in Tasks)
                sum (p in Persons) (assign[p][t]) == 1;
        }
        """
        self.run_test_case_gurobi(opl_code)

    def test_multi_indexed_variable_and_constraint(self):
        """Test 3D indexed variables and constraints (multi-indexed arrays) with both solvers."""
        opl_code = """
        dvar float x[1..2][1..3][1..2];
        range I = 1..2;
        range J = 1..3;
        range K = 1..2;
        minimize sum(i in I, j in J, k in K) x[i][j][k];
        subject to {
            forall(i in I, j in J)
                sum(k in K) x[i][j][k] <= 5;
        }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_tuple_field_access_and_nested_tuple_set(self):
        """Test tuple field access and nested tuple sets with both solvers."""
        opl_code = """
        tuple Inner { int id; float val; };
        tuple Outer { Inner inner; float weight; };
        {Outer} outers = { < <1, 2.5>, 3.0 >, < <2, 4.0>, 1.5 > };
        dvar float x[outers];
        minimize sum(o in outers) o.inner.val * x[o];
        subject to {
            forall(o in outers) x[o] <= o.weight;
        }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_inline_and_external_data_mix(self):
        """Test model with both inline and .dat data, including parameter arrays."""
        model_code = """
        int N = ...;
        range I = 1..N;
        float cost[I] = ...;
        dvar float x[I];
        minimize sum(i in 1..N) cost[i] * x[i];
        subject to {
            forall(i in I) x[i] >= 0;
            sum(i in I) x[i] == 10;
        }
        """
        data_code = """
        N = 3;
        cost = [2.0, 3.0, 1.5];
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
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
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_filtered_sum_and_nested_forall(self):
        """Test constraints using filtered sums and nested forall with both solvers."""
        opl_code = """
        range I = 1..3;
        range J = 1..3;
        dvar boolean x[I][J];
        minimize sum(i in I, j in J) x[i][j];
        subject to {
            forall(i in I)
                sum(j in J : j != i) x[i][j] == 1;
            forall(j in J)
                sum(i in I : i != j) x[i][j] == 1;
        }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_simple_blending_problem(self):
        """Test codegen & solve for a simple blending problem with both Gurobi and SciPy."""
        opl_code = """
        dvar float blendA;
        dvar float blendB;

        minimize 2.5 * blendA + 3.0 * blendB;

        subject to {
            blendA + blendB == 100;
            0.3 * blendA + 0.6 * blendB >=  45;
            0.1 * blendA + 0.2 * blendB <= 20;
        }
        """
        # Expected optimal solution: solve small LP analytically.
        # Binding constraints: blendA + blendB == 100 and 0.3A + 0.6B == 45 -> A + 2B = 150 -> A = 150 - 2B.
        # Substitute into A+B=100 -> (150 - 2B) + B = 100 -> 150 - B = 100 -> B = 50, A = 50.
        # Check third: 0.1*50 + 0.2*50 = 15 <= 20 OK. Objective = 2.5*50 + 3.0*50 = 125 + 150 = 275.
        expected_obj = 275.0
        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=5,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_blending_string_sets_list_index_error(self):
        """Blending with string-indexed scalar sets and list data (bug regression test).

        Original bug: parameters stored as Python lists were indexed by string labels, raising
        'list indices must be integers or slices, not str'. The codegen now emits <Set>_index
        maps and remaps string labels to integer positions for both Gurobi and SciPy backends.

        This test verifies both solvers produce the same optimal objective (342.5) and thus
        guards against regressions in typed string set indexing for 1D/2D list parameters.
        """
        model_code = """
            {string} Products = ...;
            {string} Resources = ...;

            float Consumption[Products][Resources] = ...;
            float Capacity[Resources] = ...;
            float Demand[Products] = ...;
            float InsideCost[Products] = ...;
            float OutsideCost[Products]  = ...;

            dvar float+ Inside[Products];
            dvar float+ Outside[Products];

            minimize
                sum( p in Products )
                    ( InsideCost[p] * Inside[p] + OutsideCost[p] * Outside[p] );

            subject to {
                forall( r in Resources )
                    ctCapacity:
                        sum( p in Products )
                            Consumption[p][r] * Inside[p] <= Capacity[r];

                forall(p in Products)
                    ctDemand:
                        Inside[p] + Outside[p] >= Demand[p];
            }
            """
        data_code = """
            Products = { "ProdA", "ProdB" };
            Resources = { "Res1", "Res2" };

            Consumption = [
                [ 1.0, 2.0 ],
                [ 0.5, 1.5 ]
            ];
            Capacity = [ 100.0, 80.0 ];
            Demand = [ 40.0, 50.0 ];
            InsideCost = [ 2.0, 3.0 ];
            OutsideCost = [ 5.0, 6.0 ];
            """
        expected_obj = 342.5
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
                # Objective close to expected
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=3,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        # Cross-solver objective agreement
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=3)

    def test_workforce_planning_conditional_vs_explicit(self):
        """
        Test that explicit and conditional-expression workforce planning models produce the same solution and objective.
        """
        # Explicit model (no conditional expressions)
        explicit_model = """
        // ----------------------
        // SETS AND PARAMETERS
        // ----------------------

        int T = ...; // Number of periods
        int S = ...; // Number of skill levels
        int K = ...; // Number of tasks (job types)

        range Periods = 1..T;
        range Skills = 1..S;
        range SkillTrans = 1..S-1; // Transitions for possible training from s to s+1
        range Tasks = 1..K;

        float hiringCost[Skills];
        float firingCost[Skills];
        float wage[Skills];
        float otWage[Skills];
        float productivity[Skills];
        float maxOvertime[Skills];
        float trainingCost[SkillTrans];

        int initialWorkforce[Skills];
        int demand[Tasks][Periods];
        int skillsRequired[Tasks][Skills];
        float budget[Periods];
        int maxHire[Skills][Periods];
        int maxFire[Skills][Periods];
        int spanControl;
        int nManagers;

        dvar int+ hire[Skills][Periods];
        dvar int+ fire[Skills][Periods];
        dvar int+ train[SkillTrans][Periods];
        dvar int+ assign[Skills][Tasks][Periods];
        dvar int+ overtime[Skills][Periods];
        dvar int+ workforce[Skills][Periods];

        minimize
        sum(s in Skills, p in Periods) (hiringCost[s] * hire[s][p] + firingCost[s] * fire[s][p])
        + sum(s in SkillTrans, p in Periods) trainingCost[s] * train[s][p]
        + sum(s in Skills, p in Periods) wage[s] * sum(t in Tasks) assign[s][t][p]
        + sum(s in Skills, p in Periods) otWage[s] * overtime[s][p];

        subject to {
            workforce[1][1] == initialWorkforce[1] + hire[1][1] - fire[1][1];
            forall(s in 2..S)
            workforce[s][1] == initialWorkforce[s] + hire[s][1] - fire[s][1];

            forall(p in 2..T)
            workforce[1][p] == workforce[1][p-1] + hire[1][p] - fire[1][p] - train[1][p-1];

            forall(s in 2..S-1, p in 2..T)
            workforce[s][p] == workforce[s][p-1] + hire[s][p] - fire[s][p] + train[s-1][p-1] - train[s][p-1];

            forall(p in 2..T)
            workforce[S][p] == workforce[S][p-1] + hire[S][p] - fire[S][p] + train[S-1][p-1];

            forall(s in Skills, p in Periods)
            sum(t in Tasks) assign[s][t][p] <= workforce[s][p]*productivity[s] + overtime[s][p];

            forall(s in Skills, p in Periods)
            overtime[s][p] <= workforce[s][p]*maxOvertime[s];

            forall(s in Skills, p in Periods)
            hire[s][p] <= maxHire[s][p];
            forall(s in Skills, p in Periods)
            fire[s][p] <= maxFire[s][p];

            forall(s in Skills)
            fire[s][1] <= initialWorkforce[s];
            forall(s in Skills, p in 2..T)
            fire[s][p] <= workforce[s][p-1];

            forall(s in SkillTrans)
            train[s][1] <= initialWorkforce[s];
            forall(s in SkillTrans, p in 2..T)
            train[s][p] <= workforce[s][p-1];

            forall(t in Tasks, p in Periods)
            sum(s in Skills : skillsRequired[t][s]==1) assign[s][t][p] >= demand[t][p];

            forall(p in Periods)
            sum(s in Skills) workforce[s][p] <= nManagers * spanControl;

            forall(p in Periods)
                sum(s in Skills)
                (hiringCost[s]*hire[s][p] + firingCost[s]*fire[s][p] + wage[s]*sum(t in Tasks) assign[s][t][p] + otWage[s]*overtime[s][p])
                + sum(s in SkillTrans)
                trainingCost[s]*train[s][p]
                <= budget[p];
        }
        """

        # Conditional-expression model
        conditional_model = """
        // ASSUMPTIONS:
        // * Time is discretized into periods.
        // * There is a finite and known set of skill levels and tasks.
        // * Productivity is normalized per worker per period.
        // * Overtime is allowed only up to a specified maximum per worker.
        // * All monetary values (costs, wages) and worker-hours are known input data.

        int T = ...;
        int S = ...;
        int K = ...;

        range Periods = 1..T;
        range Skills = 1..S;
        range SkillTrans = 1..S-1;
        range Tasks = 1..K;

        float hiringCost[Skills];
        float firingCost[Skills];
        float trainingCost[SkillTrans];
        float wage[Skills];
        float otWage[Skills];
        float productivity[Skills];
        float maxOvertime[Skills];
        int initialWorkforce[Skills];
        int demand[Tasks][Periods];
        int skillsRequired[Tasks][Skills];
        float budget[Periods];
        int maxHire[Skills][Periods];
        int maxFire[Skills][Periods];
        int spanControl;
        int nManagers;

        dvar int+ hire[Skills][Periods];
        dvar int+ fire[Skills][Periods];
        dvar int+ train[SkillTrans][Periods];
        dvar int+ assign[Skills][Tasks][Periods];
        dvar int+ overtime[Skills][Periods];
        dvar int+ workforce[Skills][Periods];

        minimize
        sum(s in Skills, p in Periods) (hiringCost[s] * hire[s][p] + firingCost[s] * fire[s][p])
        + sum(s in SkillTrans, p in Periods) trainingCost[s] * train[s][p]
        + sum(s in Skills, p in Periods) wage[s] * sum(t in Tasks) assign[s][t][p]
        + sum(s in Skills, p in Periods) otWage[s] * overtime[s][p];

        subject to {
            workforce[1][1] == initialWorkforce[1] + hire[1][1] - fire[1][1];
            forall(s in 2..S)
                workforce[s][1] == initialWorkforce[s] + hire[s][1] - fire[s][1];

            forall(p in 2..T)
                workforce[1][p] == workforce[1][p-1] + hire[1][p] - fire[1][p] - train[1][p-1];

            forall(s in 2..S-1, p in 2..T)
                workforce[s][p] == workforce[s][p-1] + hire[s][p] - fire[s][p] + train[s-1][p-1] - train[s][p-1];

            forall(p in 2..T)
                workforce[S][p] == workforce[S][p-1] + hire[S][p] - fire[S][p] + train[S-1][p-1];

            forall(s in Skills, p in Periods)
                sum(t in Tasks) assign[s][t][p] <= workforce[s][p]*productivity[s] + overtime[s][p];

            forall(s in Skills, p in Periods)
                overtime[s][p] <= workforce[s][p]*maxOvertime[s];

            forall(s in Skills, p in Periods)
                hire[s][p] <= maxHire[s][p];
            forall(s in Skills, p in Periods)
                fire[s][p] <= maxFire[s][p];

            forall(s in Skills)
                fire[s][1] <= initialWorkforce[s];
            forall(s in Skills, p in 2..T)
                fire[s][p] <= workforce[s][p-1];

            forall(s in SkillTrans)
                train[s][1] <= initialWorkforce[s];
            forall(s in SkillTrans, p in 2..T)
                train[s][p] <= workforce[s][p-1];

            forall(t in Tasks, p in Periods)
                sum(s in Skills : skillsRequired[t][s]==1) assign[s][t][p] >= demand[t][p];

            forall(p in Periods)
                sum(s in Skills) workforce[s][p] <= nManagers * spanControl;

            forall(p in Periods)
                sum(s in Skills)
                (hiringCost[s]*hire[s][p] + firingCost[s]*fire[s][p] + wage[s]*sum(t in Tasks) assign[s][t][p] + otWage[s]*overtime[s][p])
                + sum(s in SkillTrans)
                trainingCost[s]*train[s][p]
                <= budget[p];
        }
        """

        # Data file as provided
        data_code = """
        T = 3;    // number of periods
        S = 2;    // number of skill levels
        K = 2;    // number of tasks/job types

        hiringCost = [ 1000, 1500 ];
        firingCost = [ 500, 800 ];
        trainingCost = [ 700 ];   // only S-1, i.e., training from level 1 to level 2

        wage = [ 25, 35 ];
        otWage = [ 40, 55 ];

        productivity = [ 40, 50 ];

        maxOvertime = [ 10, 15 ];

        initialWorkforce = [ 15, 10 ];

        demand = [
        [ 400, 530, 460 ],
        [ 250, 220, 300 ]
        ];

        skillsRequired = [
        [1, 1],
        [0, 1]
        ];

        budget = [ 25000, 25000, 25000 ];

        maxHire = [
        [ 5, 5, 5 ],
        [ 3, 3, 3 ]
        ];
        maxFire = [
        [ 5, 5, 5 ],
        [ 3, 3, 3 ]
        ];

        spanControl = 10;
        nManagers = 3;
        """

        import os
        import tempfile

        from pyopl.pyopl_core import solve

        # Write models and data to temp files
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod1,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod2,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
        ):
            tmp_mod1.write(explicit_model)
            tmp_mod1.flush()
            tmp_mod2.write(conditional_model)
            tmp_mod2.flush()
            tmp_dat.write(data_code)
            tmp_dat.flush()
            model_file1 = tmp_mod1.name
            model_file2 = tmp_mod2.name
            data_file = tmp_dat.name

        try:
            # Test both solvers for both models
            for solver in ("gurobi", "scipy"):
                result_explicit = solve(model_file1, data_file, solver=solver)
                result_conditional = solve(model_file2, data_file, solver=solver)
                self.assertNotEqual(
                    result_explicit["status"],
                    "FAILED",
                    f"Explicit model failed for {solver}",
                )
                self.assertNotEqual(
                    result_conditional["status"],
                    "FAILED",
                    f"Conditional model failed for {solver}",
                )
                self.assertIn("objective_value", result_explicit)
                self.assertIn("objective_value", result_conditional)
                # Compare objective values
                self.assertAlmostEqual(
                    result_explicit["objective_value"],
                    result_conditional["objective_value"],
                    places=4,
                    msg=f"Objective mismatch for {solver}: explicit={result_explicit['objective_value']}, conditional={result_conditional['objective_value']}",
                )
                # Compare solutions (variable values)
                sol_explicit = result_explicit.get("solution", {})
                sol_conditional = result_conditional.get("solution", {})
                norm_explicit = {self.normalize_varname(k): v for k, v in sol_explicit.items()}
                norm_conditional = {self.normalize_varname(k): v for k, v in sol_conditional.items()}
                self.assertEqual(
                    set(norm_explicit.keys()),
                    set(norm_conditional.keys()),
                    msg=f"Variable set mismatch for {solver}: explicit={set(norm_explicit.keys())}, conditional={set(norm_conditional.keys())}",
                )
                for k in norm_explicit:
                    self.assertAlmostEqual(
                        norm_explicit[k],
                        norm_conditional[k],
                        places=4,
                        msg=f"Variable {k} mismatch for {solver}: explicit={norm_explicit[k]}, conditional={norm_conditional[k]}",
                    )
        finally:
            if os.path.exists(model_file1):
                os.remove(model_file1)
            if os.path.exists(model_file2):
                os.remove(model_file2)
            if os.path.exists(data_file):
                os.remove(data_file)

    def test_rich_opl_model(self):
        """
        Test a rich OPL model with ranges, tuples, sets, dvars, constraints, and data.
        Logical constraints are omitted due to lack of implementation.
        """
        model_code = """
        int N = ...;
        range Items = 1..N;

        tuple Product {
            int id;
            float profit;
            float weight;
        };

        {Product} products = ...;

        float capacity = ...;

        dvar boolean take[products];

        maximize sum(p in products) p.profit * take[p];

        subject to {
            sum(p in products) p.weight * take[p] <= capacity;
            forall(p in products){
                 //(take[p] == 0) || (take[p] == 1); //no general logical OR over linear constraints
                 (take[p]) + (1 - (take[p])) == 1;
            }
        }
        """
        data_code = """
        N = 4;
        products = { <1, 10.0, 2.0>, <2, 15.0, 3.0>, <3, 7.0, 1.5>, <4, 8.0, 2.5> };
        capacity = 5.0;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
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
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)

    def test_mini_graph_coloring_with_neq_and_implication(self):
        """4-cycle coloring using native != plus an extra linear implication for span-based big-M.

                Constructs used:
                    - color[i] != color[j] (numeric != with tightening)
                    - Implication: (color[1] == 2) => (color[2] >= 2)  (uses a simple equality antecedent supported by SciPy)
        Both backends should solve with same objective (minimize maxColor).
        """
        base_model = """
        int N = 4;
        range V = 1..N;
        tuple Edge { int u; int v; };
        {Edge} arcs = { <1,2>, <2,3>, <3,4>, <4,1> };
        dvar int+ color[V];
        dvar int+ maxColor;
        minimize maxColor;
        subject to {
            forall(i in V) {
                color[i] >= 1;
                color[i] <= 4;
                maxColor >= color[i];
            }
            // Edge coloring constraints via tuples
            forall(e in arcs)
                color[e.u] != color[e.v];
            // IMPL_LINE
        }
        """
        results = {}
        # Use implication with equality antecedent so SciPy can encode (pattern b==1 style).
        for solver in ("gurobi", "scipy"):
            model_code = base_model.replace("// IMPL_LINE", "(color[1] == 2) => (color[2] >= 2);")
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertNotEqual(res.get("status"), "FAILED")
                if res.get("objective_value") is not None:
                    results[solver] = res.get("objective_value")
            finally:
                if os.path.exists(path):
                    os.remove(path)
        self.assertAlmostEqual(results["gurobi"], results["scipy"], places=6)

    def test_food_blending_problem(self):
        """Food blending problem: mixes ingredients into foods meeting nutrient demands.

        Model uses:
            - {string} sets Foods, Ingredients
            - tuple types foodType / ingredientType with demand, price, protein, fat fields
            - Arrays Food[Foods], Ingredient[Ingredients]
            - Decision vars: slack[Foods] (over-production), Mix[Ingredients][Foods]
            Objective maximizes margin minus slack penalties; optimal slack expected zero.
        """
        model_code = """
        {string} Foods = ...;
        {string} Ingredients = ...;
        tuple foodType { float demand; float price; float protein; float fat; };
        tuple ingredientType { float capacity; float price; float protein; float fat; };
        foodType Food[Foods] = ...;
        ingredientType Ingredient[Ingredients] = ...;
        float MaxProduction = ...;
        float ProcCost = ...; // processing cost per unit

        dvar float+ slack[Foods];
        dvar float+ Mix[Ingredients][Foods];

        maximize
            sum( f in Foods , ing in Ingredients )
                (Food[f].price - Ingredient[ing].price - ProcCost) * Mix[ing][f]
                - sum(f in Foods) slack[f];
        subject to {
            forall( f in Foods )
                sum( ing in Ingredients ) Mix[ing][f] == Food[f].demand + 10*slack[f];
            // Ingredient capacity
            forall( ing in Ingredients )
                sum( f in Foods ) Mix[ing][f] <= Ingredient[ing].capacity;
            // Global production limit
            sum( ing in Ingredients , f in Foods ) Mix[ing][f] <= MaxProduction;
            // Protein quality: blended protein must not fall below required (weighted diff >= 0)
            forall( f in Foods )
                sum( ing in Ingredients ) (Ingredient[ing].protein - Food[f].protein) * Mix[ing][f] >= 0;
            // Fat limit: blended fat must not exceed target (weighted diff <= 0)
            forall( f in Foods )
                sum( ing in Ingredients ) (Ingredient[ing].fat - Food[f].fat) * Mix[ing][f] <= 0;
        }
        """
        data_code = """
        Foods = { "Meal1", "Meal2", "Meal3" };
        Ingredients = { "Chicken", "Beef", "Soy" };

        Food = [ <3000, 9, 30, 10>,
                 <2000, 7, 25, 15>,
                 <1000, 6, 20, 12> ];

        Ingredient = [ <5000, 4, 35, 6>,
                        <5000, 5, 28, 18>,
                        <5000, 3, 22, 14> ];

        MaxProduction = 14000;
        ProcCost = 1.5;
        """
        expected_obj = 35766.66666666666
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
                # Objective close to expected
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=3,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        # Cross-solver objective agreement
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=3)

    def test_transportation_problem_with_tuples_and_string_sets(self):
        """
        Transportation Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Transportation Problem with advanced OPL constructs

        tuple Arc {
            string origin;
            string dest;
        }

        // Sets
        {string} Origins = ...;
        {string} Destinations = ...;
        {string} SpecialOrigins = { "Seattle" };
        {Arc} arcs = ...;

        // Parameters
        param float supply[Origins];
        param float demand[Destinations];
        param float cost[arcs];
        param float capacity[arcs];         // NEW: arc capacities
        param float min_shipment[arcs];     // NEW: minimum shipment per arc
        param float total_shipment_limit;   // NEW: global shipment limit

        // Decision variables
        dvar float+ x[arcs];

        // Objective
        minimize sum(a in arcs) cost[a] * x[a];

        subject to {
            // Supply constraints
            forall(o in Origins)
                sum(a in arcs : a.origin == o) x[a] <= supply[o];

            // Demand constraints
            forall(d in Destinations)
                sum(a in arcs : a.dest == d) x[a] >= demand[d];

            // Arc capacity constraints
            forall(a in arcs)
                x[a] <= capacity[a];

            // Minimum shipment on certain arcs
            forall(o in SpecialOrigins)
                forall(a in arcs : a.origin == o)
                    x[a] >= 10;

            // Total shipment limit
            sum(a in arcs) x[a] <= total_shipment_limit;
        }

        """
        data_code = """
        // Data for the transportation problem

        Origins = { "Seattle", "San-Diego" };
        Destinations = { "New-York", "Chicago", "Topeka" };

        arcs = {
            <"Seattle", "New-York">,
            <"Seattle", "Chicago">,
            <"Seattle", "Topeka">,
            <"San-Diego", "New-York">,
            <"San-Diego", "Chicago">,
            <"San-Diego", "Topeka">
        };

        supply = [
            "Seattle"   350,
            "San-Diego" 600
        ];

        demand = [
            "New-York" 325,
            "Chicago"  300,
            "Topeka"   275
        ];

        cost = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      2.5,
            <"Seattle", "Topeka">       1.7,
            <"San-Diego", "New-York">   2.5,
            <"San-Diego", "Chicago">    1.8,
            <"San-Diego", "Topeka">     1.4
        ];

        // Arc capacities
        capacity = [
            <"Seattle", "New-York">     200,
            <"Seattle", "Chicago">      250,
            <"Seattle", "Topeka">       200,
            <"San-Diego", "New-York">   300,
            <"San-Diego", "Chicago">    300,
            <"San-Diego", "Topeka">     400
        ];

        // Minimum shipment per arc (0 for most, but you can set >0 for some)
        min_shipment = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      0,
            <"Seattle", "Topeka">       0,
            <"San-Diego", "New-York">   0,
            <"San-Diego", "Chicago">    0,
            <"San-Diego", "Topeka">     50
        ];

        // Total shipment limit
        total_shipment_limit = 900;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_transportation_problem_with_tuples_and_string_sets_and_string_filtering(
        self,
    ):
        """
        Transportation Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Transportation Problem with advanced OPL constructs

        tuple Arc {
            string origin;
            string dest;
        }

        // Sets
        {string} Origins = ...;
        {string} Destinations = ...;
        {Arc} arcs = ...;

        // Parameters
        param float supply[Origins];
        param float demand[Destinations];
        param float cost[arcs];
        param float capacity[arcs];         // NEW: arc capacities
        param float min_shipment[arcs];     // NEW: minimum shipment per arc
        param float total_shipment_limit;   // NEW: global shipment limit

        // Decision variables
        dvar float+ x[arcs];

        // Objective
        minimize sum(a in arcs) cost[a] * x[a];

        subject to {
            // Supply constraints
            forall(o in Origins)
                sum(a in arcs : a.origin == o) x[a] <= supply[o];

            // Demand constraints
            forall(d in Destinations)
                sum(a in arcs : a.dest == d) x[a] >= demand[d];

            // Arc capacity constraints
            forall(a in arcs)
                x[a] <= capacity[a];

            // Minimum shipment on certain arcs
            forall(a in arcs : a.origin == "Seattle")
                x[a] >= 10;

            // Total shipment limit
            sum(a in arcs) x[a] <= total_shipment_limit;
        }

        """
        data_code = """
        // Data for the transportation problem

        Origins = { "Seattle", "San-Diego" };
        Destinations = { "New-York", "Chicago", "Topeka" };

        arcs = {
            <"Seattle", "New-York">,
            <"Seattle", "Chicago">,
            <"Seattle", "Topeka">,
            <"San-Diego", "New-York">,
            <"San-Diego", "Chicago">,
            <"San-Diego", "Topeka">
        };

        supply = [
            "Seattle"   350,
            "San-Diego" 600
        ];

        demand = [
            "New-York" 325,
            "Chicago"  300,
            "Topeka"   275
        ];

        cost = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      2.5,
            <"Seattle", "Topeka">       1.7,
            <"San-Diego", "New-York">   2.5,
            <"San-Diego", "Chicago">    1.8,
            <"San-Diego", "Topeka">     1.4
        ];

        // Arc capacities
        capacity = [
            <"Seattle", "New-York">     200,
            <"Seattle", "Chicago">      250,
            <"Seattle", "Topeka">       200,
            <"San-Diego", "New-York">   300,
            <"San-Diego", "Chicago">    300,
            <"San-Diego", "Topeka">     400
        ];

        // Minimum shipment per arc (0 for most, but you can set >0 for some)
        min_shipment = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      0,
            <"Seattle", "Topeka">       0,
            <"San-Diego", "New-York">   0,
            <"San-Diego", "Chicago">    0,
            <"San-Diego", "Topeka">     50
        ];

        // Total shipment limit
        total_shipment_limit = 900;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_inventory_problem_with_tuples(self):
        """
        Inventory Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Inventory Problem with tuple arcs and string-indexed sets.

        tuple Store { string name; }
        {Store} Stores;

        range Periods = 1..3;

        int Capacity[Stores];
        int Demand[Periods];
        int OrderingCost[Periods];
        int HoldingCost;

        dvar int I[Stores][Periods];
        dvar int Q[Stores][Periods];

        minimize sum(s in Stores, p in Periods) OrderingCost[p] * Q[s][p] + HoldingCost * I[s][p];

        subject to {
            forall(s in Stores)
                I[s][1] == Q[s][1] - Demand[1];
            forall(s in Stores, p in 2..3)
                I[s][p] == I[s][p-1] + Q[s][p] - Demand[p];

            forall(s in Stores, p in Periods)
                I[s][p] <= Capacity[s];

            forall(s in Stores, p in Periods)
                I[s][p] >= 0;

            forall(s in Stores, p in Periods)
                Q[s][p] >= 0;
        }


        """
        data_code = """
        // Data for the inventory problem

        Stores = { <"S1">, <"S2"> };
        Capacity = [<"S1"> 100, <"S2"> 100];
        Demand = [
            1, 2, 3
        ];
        OrderingCost = [10, 13 , 15];
        HoldingCost = 1;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        expected_obj = 136
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                print(f"{solver} objective: {obj}")
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=3,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        # Cross-solver objective agreement
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=3)

    def test_complex_inventory_problem_with_tuples(self):
        """
        Inventory Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Inventory Problem with tuple arcs and string-indexed sets.

        range Periods = 1..3;

        tuple Store {
        string name;
        }

        {Store} Stores;

        int Capacity[Stores];
        int Demand[Stores][Periods];
        int TransportCost[Stores][Periods];
        int HoldingCost;

        dvar int Inventory[Stores][Periods];
        dvar int Shipments[Stores][Periods];

        minimize sum(s in Stores, p in Periods) TransportCost[s][p] * Shipments[s][p] + HoldingCost * Inventory[s][p];

        subject to {
        forall(s in Stores)
            Inventory[s][1] == 0 + Shipments[s][1] - Demand[s][1];
        forall(s in Stores, p in 2..3)
            Inventory[s][p] == Inventory[s][p-1] + Shipments[s][p] - Demand[s][p];

        forall(s in Stores, p in Periods) {
            Inventory[s][p] <= Capacity[s];
        }

        forall(s in Stores, p in Periods) {
            Inventory[s][p] >= 0;
        }

        forall(s in Stores, p in Periods)
            Shipments[s][p] >= 0;
        }


        """
        data_code = """
        // Data for the inventory problem

        Stores = { <"StoreA">, <"StoreB"> };

        // Capacity per store (keys must match Stores tuple elements)
        Capacity = [
            <"StoreA"> 100,
            <"StoreB"> 100
        ];

        // Demand per store and period, provided as a 2D array (rows aligned with Stores order)
        Demand = [
            [1, 2, 3],
            [4, 5, 6]
        ];

        // TransportCost per store and period, provided as a 2D array (rows aligned with Stores order)
        TransportCost = [
            [10, 12, 15],
            [8, 11, 13]
        ];

        HoldingCost = 1;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_shortest_path_with_tuples(self):
        """
        Shortest Path with tuple arcs.
        """
        model_code = """
        // Shortest Path with tuple arcs.

        tuple Arc { int from; int to; float cost; }

        int N = ...;
        range Nodes = 1..N;
        {Arc} arcs = ...;
        int source = ...;
        int dest = ...;
        dvar int x[arcs];

        minimize sum(a in arcs) a.cost * x[a];

        subject to {
            forall(i in Nodes) (
                sum(a in arcs: a.from == i) x[a] - sum(a in arcs: a.to == i) x[a] == ((i == source) ? 1 : ((i == dest) ? -1 : 0))
            );
        }
        """
        data_code = """
        // Data for the inshortest pathventory problem

        N = 5;
        source = 1;
        dest = 5;
        arcs = {
        <1, 2, 2.0>,
        <1, 3, 3.0>,
        <2, 3, 1.0>,
        <2, 4, 1.0>,
        <3, 4, 1.0>,
        <4, 5, 2.0>,
        <3, 5, 5.0>
        };
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_shortest_path_with_tuples_and_strings(self):
        """
        Shortest Path with tuple arcs and strings.
        """
        model_code = """
        // Shortest Path with tuple arcs and strings.

        tuple Arc { string from; string to; float cost; }

        {string} Cities = ...;

        {Arc} arcs = ...;
        string source = ...;
        string dest = ...;
        dvar int x[arcs];

        minimize sum(a in arcs) a.cost * x[a];

        subject to {
        forall(i in Cities) (
            sum(a in arcs: a.from == i) x[a] - sum(a in arcs: a.to == i) x[a] == ((i == source) ? 1 : ((i == dest) ? -1 : 0))
        );
        }
        """
        data_code = """
        // Data for the shortest path problem

        Cities = { "London", "Oxford", "Cambridge",
           "Norwich", "Birmingham", "Manchester" };
        source = "London";
        dest = "Birmingham";
        arcs = {
        <"London", "Oxford", 90.0>,
        <"London", "Cambridge", 100.0>,
        <"London", "Norwich", 180.0>,
        <"London", "Birmingham", 205.0>,
        <"London", "Manchester", 335.0>,
        <"Oxford", "London", 90.0>,
        <"Oxford", "Cambridge", 140.0>,
        <"Oxford", "Norwich", 220.0>,
        <"Oxford", "Birmingham", 110.0>,
        <"Oxford", "Manchester", 260.0>,
        <"Cambridge", "London", 100.0>,
        <"Cambridge", "Oxford", 140.0>,
        <"Cambridge", "Norwich", 100.0>,
        <"Cambridge", "Birmingham", 160.0>,
        <"Cambridge", "Manchester", 250.0>,
        <"Norwich", "London", 180.0>,
        <"Norwich", "Oxford", 220.0>,
        <"Norwich", "Cambridge", 100.0>,
        <"Norwich", "Birmingham", 240.0>,
        <"Norwich", "Manchester", 350.0>,
        <"Birmingham", "London", 205.0>,
        <"Birmingham", "Oxford", 110.0>,
        <"Birmingham", "Cambridge", 160.0>,
        <"Birmingham", "Norwich", 240.0>,
        <"Birmingham", "Manchester", 140.0>,
        <"Manchester", "London", 335.0>,
        <"Manchester", "Oxford", 260.0>,
        <"Manchester", "Cambridge", 250.0>,
        <"Manchester", "Norwich", 350.0>,
        <"Manchester", "Birmingham", 140.0>
        };
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_logistics_with_tuples_and_strings(self):
        """
        Logistics with tuple arcs and strings.
        """
        model_code = """
        // Logistics with tuple arcs and strings.

        {string} Factories = { "F1", "F2", "F3", "F4" };
        {string} Warehouses = { "W1", "W2", "W3", "W4" };

        float cost[Factories][Warehouses];
        int supply[Factories];
        int demand[Warehouses];

        dvar int+ x[Factories][Warehouses];

        minimize sum(f in Factories, w in Warehouses) cost[f][w] * x[f][w];

        subject to {
            forall(f in Factories)
                sum(w in Warehouses) x[f][w] == supply[f];

            forall(w in Warehouses)
                sum(f in Factories) x[f][w] == demand[w];

            // Restrict F1 to only supply W1 and W2
            forall(f in Factories, w in Warehouses : f == "F1" && w != "W1" && w != "W2")
                x[f][w] == 0;

            // Shut down F1
            forall(w in Warehouses) x["F1"][w] <= 30;
        }
        """
        data_code = """
        // Data for the logistics problem

        cost = [ [0, 100, 400, 200],
                [100, 0, 300, 400],
                [400, 300, 0, 700],
                [200, 400, 700, 0] ];
        supply = [50, 20, 90, 30];
        demand = [40, 40, 60, 50];
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

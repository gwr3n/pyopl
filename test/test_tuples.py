import os
import unittest

from pyopl.pyopl_core import OPLCompiler, OPLLexer, OPLParser, solve


def setUpModule():
    import logging

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s")


class TestTupleParsing(unittest.TestCase):

    def test_large_tuple_set_and_field_access(self):
        """
        Stress test: Large set of tuples and field access in constraints and objective.
        """
        # Generate a large set of tuples
        tuple_literals = ", ".join(f"<{i},{i+1},{float(i)*1.5}>" for i in range(1, 51))
        code = f"""
        tuple Arc {{ int from; int to; float cost; }};
        {{Arc}} arcs = {{ {tuple_literals} }};
        dvar float x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {{ forall(a in arcs) x[a] >= a.from; }}
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            # Check that the set of tuples is parsed correctly
            set_decl = next((d for d in ast["declarations"] if d["type"] == "set_of_tuples"), None)
            self.assertIsNotNone(set_decl)
            self.assertEqual(len(set_decl["value"]), 50)
            # Check that field access is present in the objective
            obj = ast["objective"]
            self.assertIn("sum", obj["expression"]["type"])
            sum_expr = obj["expression"]["expression"]
            self.assertEqual(sum_expr["type"], "binop")
            self.assertEqual(sum_expr["left"]["type"], "field_access")

    def test_tuple_field_access_in_nested_forall(self):
        """
        This test covers:
        - Tuple type declaration and set of tuples.
        - Nested forall with two tuple iterators.
        - Field access in binop expressions (a.from + b.to).
        - Ensures correct AST structure for nested tuple field access in constraints.
        """
        code = """
        tuple Arc { int from; int to; };
        {Arc} arcs = { <1,2>, <2,3> };
        {Arc} arcs2 = { <2,1>, <3,2> };
        dvar int x;
        minimize x;
        subject to { forall(a in arcs, b in arcs2) a.from + b.to >= 0; }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)

            # Find the nested forall constraint
            constraints = ast["constraints"]
            forall_constr = next((c for c in constraints if c["type"] == "forall_constraint"), None)
            self.assertIsNotNone(forall_constr)
            # The inner constraint should be a binop '+' of two field_access nodes
            inner = forall_constr["constraint"]
            self.assertEqual(inner["type"], "constraint")
            self.assertEqual(inner["op"], ">=")
            left = inner["left"]
            self.assertEqual(left["type"], "binop")
            self.assertEqual(left["op"], "+")
            # Check left side of binop: a.from
            left_a = left["left"]
            self.assertEqual(left_a["type"], "field_access")
            self.assertEqual(left_a["field"], "from")
            self.assertEqual(left_a["base"]["type"], "name")
            self.assertEqual(left_a["base"]["value"], "a")
            # Check right side of binop: b.to
            right_b = left["right"]
            self.assertEqual(right_b["type"], "field_access")
            self.assertEqual(right_b["field"], "to")
            self.assertEqual(right_b["base"]["type"], "name")
            self.assertEqual(right_b["base"]["value"], "b")

    def test_tuple_field_access_in_constraint(self):
        """
        This test covers:
        - Tuple type declaration and set of tuples.
        - Forall with a tuple iterator.
        - Field access in constraint expressions (a.from >= 1).
        - Ensures correct AST structure for tuple field access in constraints.
        """
        code = """
        tuple Arc { int from; int to; };
        {Arc} arcs = { <1,2>, <2,3> };
        dvar int x;
        minimize x;
        subject to { forall(a in arcs) a.from >= 1; }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)

            # Find the forall constraint
            constraints = ast["constraints"]
            forall_constr = next((c for c in constraints if c["type"] == "forall_constraint"), None)
            self.assertIsNotNone(forall_constr)
            # The inner constraint should be a comparison with field_access on the left
            inner = forall_constr["constraint"]
            self.assertEqual(inner["type"], "constraint")
            self.assertEqual(inner["op"], ">=")
            left = inner["left"]
            self.assertEqual(left["type"], "field_access")
            self.assertEqual(left["field"], "from")
            self.assertEqual(left["base"]["type"], "name")
            self.assertEqual(left["base"]["value"], "a")

    def test_tuple_field_access_in_sum_expression(self):
        """
        This test covers:
        - Tuple type declaration and set of tuples.
        - Sum expression over a tuple iterator.
        - Field access in sum expressions (sum(a in arcs) a.from).
        - Ensures correct AST structure for tuple field access in sum expressions.
        """
        code = """
        tuple Arc { int from; int to; };
        {Arc} arcs = { <1,2>, <2,3> };
        dvar int x;
        minimize sum(a in arcs) a.from;
        subject to { x >= 0; }
        """
        # Try compiling with both code generators
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)

            # Find the objective node
            obj = ast["objective"]
            self.assertEqual(obj["type"], "minimize")
            expr = obj["expression"]
            self.assertEqual(expr["type"], "sum")
            # The sum's expression should be a field_access node
            sum_expr = expr["expression"]
            self.assertEqual(sum_expr["type"], "field_access")
            self.assertEqual(sum_expr["field"], "from")
            # The base should be a name node for the iterator variable 'a'
            self.assertEqual(sum_expr["base"]["type"], "name")
            self.assertEqual(sum_expr["base"]["value"], "a")

    def test_tuple_field_access_in_objective(self):
        code = """
        tuple Arc {
            string start;
            string end;
            float cost;
        };
        {Arc} arcs = { <"A","B",10.0>, <"B","C",12.5> };
        dvar int x;
        minimize sum(a in arcs) a.cost * x;
        subject to { x >= 0; }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            # Find the sum node in the objective
            obj = ast["objective"]
            self.assertEqual(obj["type"], "minimize")
            expr = obj["expression"]
            self.assertEqual(expr["type"], "sum")
            # The sum expression should be a binop with field_access
            sum_expr = expr["expression"]
            self.assertEqual(sum_expr["type"], "binop")
            left = sum_expr["left"]
            self.assertEqual(left["type"], "field_access")
            self.assertEqual(left["field"], "cost")
            self.assertEqual(left["base"]["type"], "name")
            self.assertEqual(left["base"]["value"], "a")
            self.assertEqual(left["sem_type"], "float")

    def test_set_of_tuples_declaration(self):
        """
        This test covers:
        - Tuple type declaration with multiple fields.
        - Set of tuples with tuple literals.
        - Ensures correct AST structure for set of tuples declarations and tuple literal parsing.
        """
        code = """
        tuple Arc {
            string start;
            string end;
            float cost;
        };
        {Arc} arcs = { <"A","B",10.0>, <"B","C",12.5> };
        dvar int x;
        minimize x;
        subject to { x >= 0; }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            found = False
            decls = []
            if isinstance(ast, dict) and "declarations" in ast:
                decls = ast["declarations"]
            elif isinstance(ast, list):
                decls = ast
            for decl in decls:
                if decl.get("type") == "set_of_tuples" and decl.get("name") == "arcs":
                    found = True
                    # Check tuple elements structure
                    self.assertEqual(decl.get("tuple_type"), "Arc")
                    self.assertEqual(len(decl.get("value", [])), 2)
                    self.assertEqual(decl["value"][0]["elements"], ["A", "B", 10.0])
            self.assertTrue(found, f"Set of tuples declaration not found in AST for solver {solver}")

    def test_tuple_type_and_set_of_tuples_declaration(self):
        """
        This test covers:
        - Tuple type declaration and set of tuples.
        - Ensures correct AST structure for both tuple type and set of tuples declarations.
        - Checks tuple field types and tuple literal values in the AST.
        """
        model_code = """
            tuple Arc { int from; int to; };
            {Arc} arcs = { <1,2>, <2,3> };
            dvar int x;
            minimize x;
            subject to { x >= 0; }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        tokens = list(lexer.tokenize(model_code))
        ast = parser.parse(tokens)

        # Check tuple type in AST
        tuple_type_decl = next((d for d in ast["declarations"] if d["type"] == "tuple_type"), None)
        self.assertIsNotNone(tuple_type_decl)
        self.assertEqual(tuple_type_decl["name"], "Arc")
        self.assertTrue(any(f["name"] == "from" and f["type"] == "int" for f in tuple_type_decl["fields"]))
        self.assertTrue(any(f["name"] == "to" and f["type"] == "int" for f in tuple_type_decl["fields"]))

        # Check set of tuples in AST
        set_of_tuples_decl = next((d for d in ast["declarations"] if d["type"] == "set_of_tuples"), None)
        self.assertIsNotNone(set_of_tuples_decl)
        self.assertEqual(set_of_tuples_decl["name"], "arcs")
        self.assertEqual(set_of_tuples_decl["tuple_type"], "Arc")
        self.assertEqual(
            set_of_tuples_decl["value"],
            [
                {"type": "tuple_literal", "elements": [1, 2]},
                {"type": "tuple_literal", "elements": [2, 3]},
            ],
        )

    def test_tuple_type_declaration(self):
        """
        This test covers:
        - Tuple type declaration with multiple fields.
        - Ensures tuple type is present in the AST for both code generators.
        """
        code = """
        tuple Arc {
            string start;
            string end;
            float cost;
        };
        dvar int x;
        minimize x;
        subject to { x >= 0; }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            found = False
            decls = []
            if isinstance(ast, dict) and "declarations" in ast:
                decls = ast["declarations"]
            elif isinstance(ast, list):
                decls = ast
            for decl in decls:
                if decl.get("type") == "tuple_type" and decl.get("name") == "Arc":
                    found = True
            self.assertTrue(found, f"Tuple type declaration not found in AST for solver {solver}")

    def test_tuple_indexed_variable_and_param_over_tuple_set(self):
        """
        This test covers:
        - Tuple type declaration with multiple fields.
        - Set of tuples with tuple literals.
        - Indexed decision variable and parameter over the tuple set.
        - Sum objective over the indexed variable.
        - Forall constraint comparing two indexed variables.
        This test closely matches the provided AST and exercises richer tuple features in parsing and code generation.
        """
        code = """
        tuple Arc {
            string start;
            string end;
            float cost;
        };
        {Arc} arcs = { <"A","B",10.0>, <"B","C",12.5> };
        dvar float x[arcs];
        float w[arcs] = [1.5, 2.5];
        minimize sum(a in arcs) x[a];
        subject to {
            forall(a in arcs)
                x[a] >= w[a];
        }
        """
        import os
        import tempfile

        obj_values = {}
        solvers = ("gurobi", "scipy")
        for solver in solvers:
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
            self.assertEqual(dvar_decl["var_type"], "float")
            # Check parameter indexed by arcs
            param_decl = next((d for d in ast["declarations"] if d.get("name") == "w"), None)
            self.assertIsNotNone(param_decl)
            self.assertEqual(param_decl["var_type"], "float")
            # Check objective is sum over arcs of x[a]
            obj = ast["objective"]
            self.assertEqual(obj["type"], "minimize")
            expr = obj["expression"]
            self.assertEqual(expr["type"], "sum")
            self.assertEqual(expr["expression"]["type"], "indexed_name")
            self.assertEqual(expr["expression"]["name"], "x")
            # Check constraint forall(a in arcs) x[a] >= w[a]
            constraints = ast["constraints"]
            forall_constr = next((c for c in constraints if c["type"] == "forall_constraint"), None)
            self.assertIsNotNone(forall_constr)
            inner = forall_constr["constraint"]
            self.assertEqual(inner["type"], "constraint")
            self.assertEqual(inner["op"], ">=")
            self.assertEqual(inner["left"]["type"], "indexed_name")
            self.assertEqual(inner["left"]["name"], "x")
            self.assertEqual(inner["right"]["type"], "indexed_name")
            self.assertEqual(inner["right"]["name"], "w")
            # --- Solve the model and store the objective value ---
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                model_file = tmp.name
            try:
                result = solve(model_file, solver=solver)
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        for solver in solvers:
            self.assertAlmostEqual(obj_values[solver], 4, places=6)

    def test_tuple_arrays_with_decisions(self):
        """
        Test tuple arrays and field access.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            tuple Arc { int from; int to; float cost; };
            {int} Nodes = ...;
            Arc arcs[Nodes];  // or: Arc arcs[Nodes] = ...;

            dvar float+ x[Nodes];

            minimize sum(i in Nodes) arcs[i].cost * x[i];
            subject to {
                // you can access fields like this
                forall(i in Nodes) (arcs[i].from >= 1);
                // normalize flow/selection
                sum(i in Nodes) x[i] == 1;
            }
            """
        data_code = """
            Nodes = {1,2,3};
            arcs = [
            <1,2,10.0>,
            <2,3,12.5>,
            <3,1,8.0>
            ];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
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
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_tuple_arrays(self):
        """
        Test tuple arrays and field access.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            tuple Arc { int from; int to; float cost; };
            {int} Nodes = ...;
            Arc arcs[Nodes];  // or: Arc arcs[Nodes] = ...;
            minimize sum(i in Nodes) arcs[i].cost;
            subject to {
            // you can access fields like this
            forall(i in Nodes) (arcs[i].from >= 1);
            }
            """
        data_code = """
            Nodes = {1,2,3};
            arcs = [
            <1,2,10.0>,
            <2,3,12.5>,
            <3,1,8.0>
            ];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
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
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )


class TestNestedTupleParsing(unittest.TestCase):

    def test_nested_tuples(self):
        """
        Nested tuples
        """
        model_code = """
        tuple Inner {
        int a;
        int b;
        }

        tuple Outer {
        Inner inner;
        int c;
        }

        {Outer} Tuples = { < <1, 2>, 3 >, < <4, 5>, 6 > };

        dvar float x;

        minimize x;

        subject to {
        // Accessing nested field: t.inner.a
        forall(t in Tuples)
            x >= t.inner.a + t.c;
        }
        """
        obj_values = {}
        for solver in ("gurobi", "scipy"):
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_nested_tuple_set_and_field_access(self):
        """
        Test parsing and field access for deeper nested tuples.
        """
        code = """
        tuple Level1 { int x; };
        tuple Level2 { Level1 l1; int y; };
        tuple Level3 { Level2 l2; int z; };
        {Level3} S = { < < <1>, 2 >, 3 >, < < <4>, 5 >, 6 > };
        dvar float x;
        minimize x;
        subject to {
            forall(t in S)
                x >= t.l2.l1.x + t.l2.y + t.z;
        }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            # Find the set declaration
            set_decl = next((d for d in ast["declarations"] if d.get("name") == "S"), None)
            self.assertIsNotNone(set_decl)
            # Check tuple structure
            elems = set_decl["value"]
            self.assertEqual(len(elems), 2)
            # Check nested elements
            e0 = elems[0]["elements"]
            self.assertIsInstance(e0[0]["elements"][0]["elements"][0], int)
            self.assertEqual(e0[0]["elements"][0]["elements"][0], 1)
            self.assertEqual(e0[0]["elements"][1], 2)
            self.assertEqual(e0[1], 3)

    def test_nested_tuple_in_dat_file(self):
        """
        Test nested tuple parsing from a .dat file.
        """
        model_code = """
        tuple Inner { int a; int b; };
        tuple Outer { Inner inner; int c; };
        {Outer} Tuples;
        dvar float x;
        minimize x;
        subject to { forall(t in Tuples) x >= t.inner.a + t.c; }
        """
        dat_code = """
        Tuples = { < <1,2>, 3 >, < <4,5>, 6 > };
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(model_code, dat_code, solver=solver)
            self.assertIn("Tuples", data_dict)
            tuples = data_dict["Tuples"]
            print(f"[DEBUG] Parsed Tuples from .dat: {tuples}")

            # Normalize to a list of native tuples like [((1,2),3), ((4,5),6)]
            def to_native(x):
                if isinstance(x, dict) and x.get("type") == "tuple_literal":
                    return tuple(to_native(e) for e in x.get("elements", []))
                if isinstance(x, list):
                    return [to_native(e) for e in x]
                return x

            if isinstance(tuples, dict) and "elements" in tuples:
                elems = tuples["elements"]
            else:
                elems = tuples
            native = [to_native(e) for e in elems]

            self.assertEqual(len(native), 2)
            self.assertEqual(native[0][0][0], 1)
            self.assertEqual(native[0][0][1], 2)
            self.assertEqual(native[0][1], 3)

    def test_empty_and_singleton_nested_tuples(self):
        """
        Test edge cases: empty and singleton nested tuples.
        """
        code = """
        tuple A { };
        tuple B { A a; };
        {B} S = { < < > >, < < > > };
        dvar int x;
        minimize x;
        subject to { x >= 0; }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            set_decl = next((d for d in ast["declarations"] if d.get("name") == "S"), None)
            self.assertIsNotNone(set_decl)
            elems = set_decl["value"]
            self.assertEqual(len(elems), 2)
            self.assertEqual(elems[0]["elements"][0]["elements"], [])

    def test_tuple_of_tuples_and_mixed_types(self):
        """
        Test tuples containing tuples and mixed types.
        """
        code = """
        tuple A { int a; };
        tuple B { float b; };
        tuple C { A a; B b; string s; };
        {C} S = { < <1>, <2.5>, "foo" >, < <3>, <4.5>, "bar" > };
        dvar int x;
        minimize x;
        subject to { x >= 0; }
        """
        for solver in ("gurobi", "scipy"):
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            set_decl = next((d for d in ast["declarations"] if d.get("name") == "S"), None)
            self.assertIsNotNone(set_decl)
            elems = set_decl["value"]
            self.assertEqual(len(elems), 2)
            e0 = elems[0]["elements"]
            self.assertIsInstance(e0[0]["elements"][0], int)
            self.assertIsInstance(e0[1]["elements"][0], float)
            self.assertIsInstance(e0[2], str)


if __name__ == "__main__":
    unittest.main()

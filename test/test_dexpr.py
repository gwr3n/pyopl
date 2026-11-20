import io
import unittest
from contextlib import redirect_stdout

from pyopl.pyopl_core import OPLCompiler

try:
    import gurobipy  # noqa: F401

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class TestDexpr(unittest.TestCase):
    def test_scalar_and_indexed_dexpr_expand_on_use(self):
        model = r"""
            range I = 1..3;
            dvar float+ x[I];

            // Indexed dexpr using iterator in RHS
            dexpr float y[i in I] = 2 * x[i];

            // Scalar dexpr
            dexpr float z = x[1] + 5;

            minimize sum(i in I) y[i] + z;
            subject to { }
        """
        # Compile with Gurobi backend (codegen only) to ensure no errors and inlining works
        compiler = OPLCompiler()
        ast, code, data = compiler.compile_model(model, solver="gurobi")

        # AST should contain dexpr declarations but no usage of y[...] in objective (it should be expanded)
        self.assertIsNotNone(ast)
        self.assertIn("declarations", ast)
        # Ensure dexpr declarations are present
        dexprs = [d for d in ast["declarations"] if d.get("type") in ("dexpr", "dexpr_indexed")]
        self.assertTrue(len(dexprs) >= 2)

        # Walk objective expression to ensure no 'indexed_name' with name 'y' remains
        def contains_y(node):
            if isinstance(node, dict):
                if node.get("type") == "indexed_name" and node.get("name") == "y":
                    return True
                for v in node.values():
                    if contains_y(v):
                        return True
            elif isinstance(node, list):
                return any(contains_y(x) for x in node)
            return False

        self.assertFalse(contains_y(ast["objective"]))

        # Code string should be generated
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)

    def test_dexpr_dimension_count_check(self):
        model = r"""
            range I = 1..2;
            dvar float+ x[I];
            dexpr float y[i in I] = x[i];
            minimize y[1] + y[2];
            subject to { }
        """
        compiler = OPLCompiler()
        ast, code, _ = compiler.compile_model(model, solver="gurobi")
        self.assertIsNotNone(ast)
        self.assertGreater(len(code), 0)

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is not available")
    def test_filtered_variable_sums_preserve_matches_and_empty_results(self):
        model = """
            range V = 1..4;
            {int} B = {1, 3};
            {int} Empty = {v | v in V: v > 4};
            tuple Entry { int index; };
            {Entry} entries = {<1>, <2>, <3>, <4>};
            dvar boolean x[B];
            dvar float unused[Empty];
            dexpr float value[v in V] =
                sum(b in B: b == v) x[b] + sum(k in Empty: k == v) unused[k];
            maximize sum(e in entries) e.index * value[e.index];
            subject to {
                x[1] == 1;
                x[3] == 1;
                sum(b in B: b == 2) x[b] == 0;
                sum(b in B: 1 == b) x[b] == 1;
            }
        """
        _, code, _ = OPLCompiler().compile_model(model, solver="gurobi")
        self.assertNotIn("for b in B if", code)
        self.assertNotIn("for k in Empty if", code)
        namespace = {"results_container": {}}
        with redirect_stdout(io.StringIO()):
            exec(code, namespace)
        result = namespace["results_container"]["gurobi_output"]
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["objective_value"], 4.0)

    @unittest.skipUnless(GUROBI_AVAILABLE, "gurobipy is not available")
    def test_filtered_sum_does_not_lookup_variables_from_a_different_domain(self):
        model = """
            range V = 1..3;
            {int} B = {1, 3};
            dvar boolean x[V];
            dvar boolean y[B];
            maximize sum(b in B: b == 2) x[b] + sum(b in B: b == b) y[b];
            subject to { x[2] == 1; y[1] == 1; y[3] == 1; }
        """
        _, code, _ = OPLCompiler().compile_model(model, solver="gurobi")
        namespace = {"results_container": {}}
        with redirect_stdout(io.StringIO()):
            exec(code, namespace)
        result = namespace["results_container"]["gurobi_output"]
        self.assertEqual(result["status"], "OPTIMAL")
        self.assertEqual(result["objective_value"], 2.0)


if __name__ == "__main__":
    unittest.main()

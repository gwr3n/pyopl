import unittest

from pyopl.pyopl_core import OPLCompiler


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


if __name__ == "__main__":
    unittest.main()

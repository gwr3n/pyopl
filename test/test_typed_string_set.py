import unittest

from pyopl.gurobi_codegen import GurobiCodeGenerator
from pyopl.pyopl_core import parse_model
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator

MODEL = '{string} Gasolines = { "R92", "R95" };\n minimize 0;\n subject to { }'


class TestTypedStringSet(unittest.TestCase):
    def test_parse_and_codegen(self):
        ast = parse_model(MODEL)
        # Find declaration
        decl = next(
            (d for d in ast["declarations"] if d.get("type") == "typed_set" and d.get("name") == "Gasolines"),
            None,
        )
        self.assertIsNotNone(decl, f"typed_set Gasolines not parsed: {ast['declarations']}")
        self.assertEqual(decl["value"], ["R92", "R95"])
        # Gurobi code generation should emit Python list Gasolines = ['R92','R95']
        code = GurobiCodeGenerator(ast).generate_code()
        self.assertIn("Gasolines = ['R92', 'R95']", code.replace('"', "'").replace('"', ""))
        # SciPy generator should ignore during variable build without error
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        # No decision variables expected
        self.assertEqual(len(gen.var_names), 0)


if __name__ == "__main__":
    unittest.main()

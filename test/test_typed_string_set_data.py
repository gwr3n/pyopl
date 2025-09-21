import unittest

from pyopl.pyopl_core import OPLCompiler

MODEL = "{string} Fuels;\n minimize 0;\n subject to { }"
DATA = 'Fuels = { "G1", "G2", "G3" };'


class TestTypedStringSetData(unittest.TestCase):
    def test_data_file_population_gurobi(self):
        self._run_test("gurobi")

    def test_data_file_population_scipy(self):
        self._run_test("scipy")

    def _run_test(self, solver):
        compiler = OPLCompiler()
        ast, code, data = compiler.compile_model(MODEL, DATA, solver=solver)
        decl = next(d for d in ast["declarations"] if d.get("type") == "typed_set" and d.get("name") == "Fuels")
        self.assertIsNone(decl["value"])
        # Code should contain Fuels list from data
        self.assertIn("Fuels = ['G1', 'G2', 'G3']", code.replace('"', "'"))


if __name__ == "__main__":
    unittest.main()

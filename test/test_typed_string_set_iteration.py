import unittest

from pyopl.pyopl_core import OPLCompiler

MODEL = '{string} Fuels = { "G1","G2","G3" };\n dvar boolean y[Fuels];\n minimize sum(f in Fuels) y[f];\n subject to { forall(f in Fuels) y[f] >= 0; }'


class TestTypedStringSetIteration(unittest.TestCase):
    def test_sum_and_forall_gurobi(self):
        self._run_test("gurobi")

    def test_sum_and_forall_scipy(self):
        self._run_test("scipy")

    def _run_test(self, solver):
        compiler = OPLCompiler()
        ast, code, data = compiler.compile_model(MODEL, solver=solver)
        # Expect Python list for Fuels (string set emission)
        self.assertIn("Fuels = ['G1', 'G2', 'G3']", code.replace('"', "'"))
        if solver == "gurobi":
            # Gurobi emits explicit loop and addVars with Fuels
            self.assertIn("for f in Fuels:", code)
            self.assertRegex(code, r"y = model.addVars\(Fuels")
        else:  # scipy
            # SciPy currently emits symbolic comment and expanded var_names list
            self.assertIn("# Symbolic objective: sum(y[f] for f in Fuels)", code)
            # Variables expanded into var_names list as y['G1'], etc.
            self.assertIn("y['G1']", code)


if __name__ == "__main__":
    unittest.main()

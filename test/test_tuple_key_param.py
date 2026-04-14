import os
import tempfile
import unittest

from pyopl.pyopl_core import solve

try:
    import pyopl  # noqa: F401

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class TestTupleKeyParam(unittest.TestCase):
    def test_dict_of_lists_keyed_by_tuple_set(self):
        model_code = """
                    {string} Assets = ...;
                    tuple Arc { int parent; int child; };
                    {Arc} Arcs = ...;
                    float ret[Arcs][Assets] = ...;
                    dvar float+ x[Arcs][Assets];
                    minimize Obj: sum(ar in Arcs, a in Assets) ret[ar][a] * x[ar][a];
                    subject to { }
                    """

        data_code = """
                    Assets = { "StockA", "StockB" };
                    Arcs = { <1,2>, <2,3> };
                    ret = [
                    <1,2> [1.2, 0.9],
                    <2,3> [1.0, 1.1]
                    ];
                    """

        for solver in ("scipy", "gurobi"):
            if solver == "gurobi" and not GUROBI_AVAILABLE:
                continue
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
                res = solve(model_file, data_file, solver=solver)
                self.assertEqual(res.get("status"), "OPTIMAL")
                self.assertIn("objective_value", res)
            finally:
                os.remove(model_file)
                os.remove(data_file)


if __name__ == "__main__":
    unittest.main()

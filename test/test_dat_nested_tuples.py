import unittest

from pyopl.pyopl_core import OPLCompiler


class TestDatNestedTuples(unittest.TestCase):
    def test_nested_tuple_literals_in_dat(self):
        model_code = r"""
            tuple Inner { int i; int j; }
            tuple Outer { Inner pair; float value; }
            tuple Empty { }

            {Outer} items = ...;
            {Empty} empties = ...;
            float v[items] = ...;
            float w[empties] = ...;

            dvar float+ x[items];
            dvar float+ y[empties];

            minimize sum(o in items) v[o] * x[o] + sum(e in empties) w[e] * y[e];
            subject to {
                sum(o in items) x[o] + sum(e in empties) y[e] == 1;
            }
        """

        data_code = r"""
            items = { < <1,2>, 3.5 >, < <2,3>, 4.0 > };
            empties = { <> };
            v = [
                < <1,2>, 3.5 >  10.0,
                < <2,3>, 4.0 >  20.0
            ];
            w = [
                <>  30.0
            ];
        """

        compiler = OPLCompiler(syntax_error_reporting="full")
        # SciPy backend is sufficient to validate parsing/semantic merge/codegen.
        # (No need for Gurobi license in tests.)
        ast, code_str, data_dict = compiler.compile_model(model_code=model_code, data_code=data_code, solver="scipy")
        self.assertIn("items", data_dict)
        self.assertIn("empties", data_dict)
        self.assertIn("v", data_dict)
        self.assertIn("w", data_dict)

        self.assertIn(((1, 2), 3.5), data_dict["v"])
        self.assertEqual(data_dict["v"][((1, 2), 3.5)], 10.0)
        self.assertIn((), data_dict["w"])
        self.assertEqual(data_dict["w"][()], 30.0)

import unittest

from pyopl.scipy_codegen_csc import ExpressionEvaluator, SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


class TestHandleBinopCmp(unittest.TestCase):
    def setUp(self):
        # Minimal generator with empty AST/data and a usable evaluator
        self.gen = SciPyCSCCodeGenerator(
            ast={
                "declarations": [],
                "constraints": [],
                "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
            },
            data_dict={},
        )
        self.ev = ExpressionEvaluator(self.gen)

    def num(self, v):
        return {"type": "number", "value": v}

    def name(self, s):
        return {"type": "name", "value": s}

    def tuple_lit(self, elements):
        return {"type": "tuple_literal", "elements": [{"type": "number", "value": e} for e in elements]}

    def test_numeric_gt_ground(self):
        d, v = self.ev._handle_binop_cmp(self.num(5), self.num(2), ">", {})
        self.assertEqual(d, {})
        self.assertEqual(v, 1.0)
        d, v = self.ev._handle_binop_cmp(self.num(2), self.num(5), ">", {})
        self.assertEqual(v, 0.0)

    def test_symbolic_with_coeffs_raises_when_not_allowed(self):
        # Declare a boolean/integer variable 'x' so name lookup yields a coefficient dict
        self.gen.var_names = ["x"]
        self.gen.var_indices = {"x": 0}
        self.gen._allow_symbolic_bool = False
        with self.assertRaises(SemanticError):
            self.ev._handle_binop_cmp(self.name("x"), self.num(0), ">", {})

    def test_symbolic_with_coeffs_returns_str_when_allowed(self):
        self.gen.var_names = ["x"]
        self.gen.var_indices = {"x": 0}
        self.gen._allow_symbolic_bool = True
        d, v = self.ev._handle_binop_cmp(self.name("x"), self.num(0), ">", {})
        self.assertEqual(d, {})
        self.assertIsInstance(v, str)
        self.assertIn(">", v)

    def test_non_numeric_ground_values_respect_flag(self):
        # tuple vs number -> symbolic
        left = self.tuple_lit([1, 2])
        right = self.num(2)
        self.gen._allow_symbolic_bool = False
        with self.assertRaises(SemanticError):
            self.ev._handle_binop_cmp(left, right, ">", {})
        self.gen._allow_symbolic_bool = True
        d, v = self.ev._handle_binop_cmp(left, right, ">", {})
        self.assertEqual(d, {})
        self.assertIsInstance(v, str)
        self.assertIn(">", v)


if __name__ == "__main__":
    unittest.main()

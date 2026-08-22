import itertools
import math
import unittest
from unittest import mock

from pyopl.affine_bounds import affine_interval, combine_intervals, scale_interval
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


class TestSciPyAffineBounds(unittest.TestCase):
    def test_shared_interval_operations(self):
        self.assertEqual(combine_intervals((2.0, 5.0), (-3.0, 4.0), "+"), (-1.0, 9.0))
        self.assertEqual(combine_intervals((2.0, 5.0), (-3.0, 4.0), "-"), (-2.0, 8.0))
        self.assertEqual(scale_interval((2.0, 5.0), -3.0), (-15.0, -6.0))
        self.assertEqual(
            affine_interval({"x": 2.0, "y": -3.0}, 4.0, lambda name: {"x": (2.0, 5.0), "y": (-3.0, 4.0)}[name]),
            (-4.0, 23.0),
        )

    def test_collected_bounds_work_before_bounds_vector_is_populated(self):
        generator = self._generator()
        generator.bounds = []
        generator._collected_lbs = {"x": 2.0}
        generator._collected_ubs = {"x": 5.0}

        self.assertEqual(generator._infer_var_bounds("x"), (2.0, 5.0))

    def _generator(self) -> SciPyCSCCodeGenerator:
        ast = {
            "declarations": [
                {"type": "dvar", "name": "x", "var_type": "float"},
                {"type": "dvar", "name": "y", "var_type": "float"},
            ],
            "constraints": [],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
        }
        generator = SciPyCSCCodeGenerator(ast)
        generator.var_names = ["x", "y"]
        generator.var_indices = {"x": 0, "y": 1}
        generator.bounds = [[2.0, 5.0], [-3.0, 4.0]]
        generator.integrality = [0, 0]
        return generator

    def test_parser_unary_minus_preserves_interval(self):
        generator = self._generator()
        expression = {"type": "uminus", "value": {"type": "name", "value": "x"}}

        self.assertEqual(generator._linear_bounds_safe(expression), (-5.0, -2.0))

    def test_parenthesized_expression_preserves_interval(self):
        generator = self._generator()
        expression = {
            "type": "parenthesized_expression",
            "expression": {
                "type": "binop",
                "op": "-",
                "left": {"type": "name", "value": "x"},
                "right": {"type": "name", "value": "y"},
            },
        }

        self.assertEqual(generator._linear_bounds_safe(expression), (-2.0, 8.0))

    def test_finite_affine_bounds_reject_non_finite_inputs(self):
        cases = (
            ({"x": 1.0}, math.nan),
            ({"x": math.inf}, 0.0),
        )
        for coefficients, constant in cases:
            with self.subTest(coefficients=coefficients, constant=constant):
                with self.assertRaisesRegex(SemanticError, "finite"):
                    self._generator()._finite_affine_bounds(coefficients, constant, "test expression")

        generator = self._generator()
        generator.bounds[0][1] = math.inf
        with self.assertRaisesRegex(SemanticError, "finite"):
            generator._finite_affine_bounds({"x": 1.0}, 0.0, "test expression")

    def test_affine_interval_contains_all_box_corners(self):
        generator = self._generator()
        for coefficients, constant in (
            ({"x": 2.0, "y": -3.0}, 4.0),
            ({"x": -0.5, "y": 1.25}, -7.0),
            ({"x": 0.0, "y": -2.0}, 1.0),
        ):
            with self.subTest(coefficients=coefficients, constant=constant):
                lower, upper = generator._finite_affine_bounds(coefficients, constant, "test expression")
                values = [
                    constant + coefficients["x"] * x_value + coefficients["y"] * y_value
                    for x_value, y_value in itertools.product((2.0, 5.0), (-3.0, 4.0))
                ]
                self.assertLessEqual(lower, min(values))
                self.assertGreaterEqual(upper, max(values))

    def test_indexed_base_bounds_aggregate_conservatively_in_any_order(self):
        ast = {
            "declarations": [
                {
                    "type": "dvar_indexed",
                    "name": "x",
                    "var_type": "float",
                    "dimensions": [{"type": "range_index", "start": 1, "end": 2}],
                }
            ],
            "constraints": [],
            "objective": {"type": "minimize", "expression": {"type": "number", "value": 0}},
        }
        generator = SciPyCSCCodeGenerator(ast)
        generator.var_names = ["x_1", "x_2"]
        generator.var_indices = {"x_1": 0, "x_2": 1}
        generator.bounds = [[None, None], [None, None]]
        generator.integrality = [0, 0]
        generator._collected_lbs = {}
        generator._collected_ubs = {}

        def indexed(index):
            return {
                "type": "indexed_name",
                "name": "x",
                "dimensions": [{"type": "number_literal_index", "value": index}],
            }

        def number(value):
            return {"type": "number", "value": value}

        for index, lower, upper in ((1, 5.0, 6.0), (2, 2.0, 10.0)):
            generator._collect_passive_constraint_bounds(
                {"type": "constraint", "op": ">=", "left": indexed(index), "right": number(lower)},
                {},
                lambda *_args: None,
            )
            generator._collect_passive_constraint_bounds(
                {"type": "constraint", "op": "<=", "left": indexed(index), "right": number(upper)},
                {},
                lambda *_args: None,
            )

        self.assertEqual(generator._collected_lbs["x"], 2.0)
        self.assertEqual(generator._collected_ubs["x"], 10.0)

    def test_passive_bound_collection_does_not_swallow_unexpected_failures(self):
        generator = self._generator()
        generator._collected_lbs = {}
        generator._collected_ubs = {}
        constraint = {
            "type": "constraint",
            "op": ">=",
            "left": {"type": "name", "value": "x"},
            "right": {"type": "number", "value": 1},
        }

        with mock.patch.object(generator, "_collect_passive_bound_for_pair", side_effect=RuntimeError("defect")):
            with self.assertRaisesRegex(RuntimeError, "defect"):
                generator._collect_passive_constraint_bounds(constraint, {}, lambda *_args: None)

    def test_bound_tightening_does_not_swallow_unexpected_failures(self):
        generator = self._generator()
        constraint = {
            "type": "constraint",
            "op": ">=",
            "left": {"type": "name", "value": "x"},
            "right": {"type": "number", "value": 1},
        }

        with mock.patch.object(generator, "_eval_expr", side_effect=RuntimeError("defect")):
            with self.assertRaisesRegex(RuntimeError, "defect"):
                generator._tighten_bounds_from_constraints(
                    generator.bounds,
                    generator.var_names,
                    generator.var_indices,
                    [constraint],
                )


if __name__ == "__main__":
    unittest.main()

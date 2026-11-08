import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyopl._index_safety import find_index_safety_issue
from pyopl.gurobi_codegen import GurobiCodeGenerator
from pyopl.linear_problem import LinearProblem
from pyopl.linear_problem_highs import _import_highspy, build_highs_model, export_linear_problem
from pyopl.pyopl_core import OPLDataLexer, OPLDataParser, SemanticError
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


def number(value):
    return {"type": "number", "value": value}


def name(value):
    return {"type": "name", "value": value}


def binary(left, op, right):
    return {"type": "binop", "op": op, "left": left, "right": right}


def interval(start, end):
    return {"type": "range_specifier", "start": number(start), "end": number(end)}


def indexed_model(index, *, domain=None, guard=None):
    return {
        "declarations": [
            {"name": "x", "dimensions": [interval(1, 3) if domain is None else domain]},
        ],
        "objective": {
            "type": "sum",
            "iterators": [{"iterator": "i", "range": interval(1, 3)}],
            "index_constraint": guard,
            "expression": {"type": "indexed_name", "name": "x", "dimensions": [index]},
        },
        "constraints": [],
    }


class TestIndexSafetyCoverage(unittest.TestCase):
    def test_reverse_filter_and_negated_endpoint_protect_shifted_accesses(self):
        i = name("i")
        cases = [
            (binary(i, "-", number(1)), binary(number(1), "<", i)),
            (binary(i, "+", number(1)), {"type": "not", "value": binary(number(3), "==", i)}),
        ]
        for index, guard in cases:
            with self.subTest(index=index):
                self.assertIsNone(find_index_safety_issue(indexed_model(index, guard=guard)))
                self.assertEqual(find_index_safety_issue(indexed_model(index)).status, "unsafe")

    def test_affine_negative_coefficients_and_parentheses(self):
        index = binary(number(4), "-", {"type": "parenthesized_expression", "expression": name("i")})
        self.assertIsNone(find_index_safety_issue(indexed_model(index)))
        issue = find_index_safety_issue(indexed_model(binary(number(3), "-", name("i"))))
        self.assertEqual(issue.status, "unsafe")
        self.assertIn("fall below", issue.reason)

    def test_uninterpretable_or_unbounded_indices_are_unknown(self):
        cases = [
            (name("i"), {"type": "named_range_dimension", "name": "Missing"}, "cannot interpret declared range"),
            (name("missing"), None, "cannot prove index"),
            (binary(name("i"), "*", number(2)), None, "cannot prove index"),
        ]
        for index, domain, reason in cases:
            with self.subTest(index=index, domain=domain):
                issue = find_index_safety_issue(indexed_model(index, domain=domain))
                self.assertEqual(issue.status, "unknown")
                self.assertIn(reason, issue.reason)

    def test_unsafe_access_takes_precedence_over_unknown_access(self):
        ast = indexed_model(name("missing"))
        ast["constraints"] = [{"type": "indexed_name", "name": "x", "dimensions": [number(4)]}]
        issue = find_index_safety_issue(ast)
        self.assertEqual(issue.status, "unsafe")
        self.assertIn("exceed", issue.reason)


class TestHighsCoverage(unittest.TestCase):
    def test_maximization_equality_integrality_and_duplicate_names(self):
        problem = LinearProblem(
            sense="maximize",
            var_names=["x", "x", ""],
            bounds=[[None, None], [0, 3], [0, 2]],
            integrality=[1, 0, 0],
            c=[-2.0, -1.0, 0.0],
            A_eq=[[1.0, 1.0, 0.0]],
            b_eq=[2.0],
            A_ub=[[0.0, 1.0, 1.0]],
            b_ub=[3.0],
        )
        highs = build_highs_model(problem)
        lp = highs.getLp()
        self.assertEqual(list(lp.col_cost_), [2.0, 1.0, 0.0])
        self.assertEqual(list(lp.col_names_), ["x", "x_1", "c2"])
        self.assertEqual(list(lp.row_names_), ["ub_0", "eq_0"])
        self.assertEqual(lp.col_lower_[0], -highs.getInfinity())
        self.assertEqual(lp.col_upper_[0], highs.getInfinity())
        self.assertEqual(str(highs.getColIntegrality(0)[1]), "HighsVarType.kInteger")
        self.assertEqual(str(lp.sense_), "ObjSense.kMaximize")
        self.assertEqual(list(build_highs_model(problem, objective_is_minimization_form=False).getLp().col_cost_), problem.c)
        with tempfile.TemporaryDirectory() as directory:
            path = export_linear_problem(problem, Path(directory) / "nested" / "model.lp")
            self.assertTrue(path.is_file())

    def test_export_reports_highs_errors(self):
        problem = LinearProblem("minimize", [], [], [], [], [], [], [], [])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "write model"):
                export_linear_problem(problem, Path(directory) / "model.unsupported")

    def test_missing_highspy_reports_optional_dependency(self):
        original_import = builtins.__import__

        def import_without_highspy(module, *args, **kwargs):
            if module == "highspy":
                raise ImportError("not installed")
            return original_import(module, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_highspy):
            with self.assertRaisesRegex(ImportError, "optional 'highspy' package"):
                _import_highspy()


class TestParserAndGeneratorCoverage(unittest.TestCase):
    def test_data_parser_reports_actionable_syntax_errors(self):
        for source, message in [
            ("int n = 3;", "plain data assignments"),
            ('A = [ "k" [ "sub" [1] ] ];', "nested keyed-array"),
            ("cost = [1 0.25];", "bare numeric keys"),
            ("cost = [1,", "end of file"),
            ("A = 1, 2;", "token ,"),
        ]:
            with self.subTest(source=source):
                lexer = OPLDataLexer()
                with self.assertRaisesRegex(SemanticError, message):
                    OPLDataParser().parse(lexer.tokenize(source), lexer=lexer)

    def test_gurobi_sum_bounds_cover_cardinality_and_invalid_domains(self):
        generator = GurobiCodeGenerator({"declarations": [], "constraints": []})
        expression = {"type": "sum", "expression": number(-2), "iterators": [{"range": interval(1, 4)}]}
        self.assertEqual(generator._linear_bounds_safe(expression), (-8.0, -8.0))
        expression["iterators"] = [{"range": interval(3, 1)}]
        self.assertEqual(generator._linear_bounds_safe(expression), (None, None))
        expression["iterators"] = [{"range": {"type": "named_range", "name": "I"}}]
        self.assertEqual(generator._linear_bounds_safe(expression), (None, None))

    def test_gurobi_binary_indicator_comparison_values(self):
        indicator = GurobiCodeGenerator._indicator_value_for_binary_comparison
        for operator, value, expected in [
            ("==", 1, 1),
            ("==", 0, 0),
            (">=", 1, 1),
            ("<=", 0, 0),
            (">=", 0, None),
            ("==", 2, None),
            ("==", "bad", None),
        ]:
            with self.subTest(operator=operator, value=value):
                self.assertEqual(indicator(operator, number(value)), expected)

    def test_scipy_bounds_for_negation_subtraction_and_unbounded_product(self):
        generator = SciPyCSCCodeGenerator(
            {
                "declarations": [],
                "objective": {"type": "minimize", "expression": number(0)},
                "constraints": [],
            }
        )
        generator._collected_lbs = {"x": 2.0}
        generator._collected_ubs = {"x": 5.0}
        self.assertEqual(generator._linear_bounds_safe({"type": "uminus", "value": name("x")}), (-5.0, -2.0))
        self.assertEqual(generator._linear_bounds_safe(binary(number(10), "-", name("x"))), (5.0, 8.0))
        self.assertEqual(generator._linear_bounds_safe(binary(name("x"), "*", name("x"))), (None, None))
        self.assertEqual(generator._linear_bounds_safe({"type": "uminus", "value": None}), (None, None))

    def test_scipy_boolean_disjuncts_flatten_and_reject_noncomparisons(self):
        generator = SciPyCSCCodeGenerator(
            {
                "declarations": [],
                "objective": {"type": "minimize", "expression": number(0)},
                "constraints": [],
            }
        )
        first = {**binary(name("x"), "<=", number(1)), "sem_type": "boolean"}
        second = {**binary(name("x"), ">=", number(3)), "sem_type": "boolean"}
        conjunction = {"type": "and", "left": first, "right": second}
        disjunction = {"type": "or", "left": conjunction, "right": first}
        result = generator._boolean_or_disjuncts(disjunction)
        self.assertEqual(len(result), 2)
        self.assertCountEqual(result[0], [first, second])
        self.assertEqual(result[1], [first])
        with self.assertRaises(SemanticError):
            generator._boolean_or_disjuncts({"type": "and", "left": first, "right": number(1)})


if __name__ == "__main__":
    unittest.main()

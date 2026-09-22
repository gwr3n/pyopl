import unittest
from unittest.mock import patch

from pyopl.milp_abstract_equivalence import (
    AbstractEquivalenceResult,
    compare_abstract,
    parse_abstract_model,
    prove_abstract_equivalent,
)
from pyopl.pyopl_core import linear_problem_from_opl
from pyopl.semantic_error import SemanticError

LEFT_MODEL = """
    int N = ...;
    range Items = 1..N;
    float cost[Items] = ...;
    float capacity = ...;
    dvar float+ x[Items];

    minimize sum(i in Items) cost[i] * x[i];

    subject to {
        sum(i in Items) x[i] <= capacity;
        forall(i in Items) x[i] >= 0;
    }
"""


RENAMED_MODEL = """
    int itemCount = ...;
    range ProductIds = 1..itemCount;
    float unitCost[ProductIds] = ...;
    float limit = ...;
    dvar float+ quantity[ProductIds];

    minimize sum(product in ProductIds) quantity[product] * unitCost[product];

    subject to {
        forall(product in ProductIds) 0 <= quantity[product];
        limit >= sum(product in ProductIds) quantity[product];
    }
"""


PORTFOLIO_17_MODELS = {
    "left": """
        float theta = ...;
        dvar float x;

        minimize x;

        subject to {
            x >= 0;
            x <= theta;
        }
    """,
    "right": """
        float theta = ...;
        dvar float x;

        minimize x;

        subject to {
            x >= 0;
            x <= 1;
        }
    """,
}

PORTFOLIO_17_DATA = {
    "theta_1.dat": "theta = 1;",
    "theta_2.dat": "theta = 2;",
}


class AbstractEquivalenceTests(unittest.TestCase):
    def test_abstract_comparison_rejects_provably_out_of_range_index(self):
        model = """
            int N = ...;
            range I = 1..N;
            dvar float+ x[I];
            minimize sum(i in I) x[i + 1];
            subject to {}
        """

        with self.assertRaisesRegex(SemanticError, "can exceed"):
            prove_abstract_equivalent(model, model, mode="auto")

    def test_definite_index_error_precedes_unresolved_parameter_index(self):
        model = """
            int N = ...;
            range I = 1..N;
            int next[I] = ...;
            dvar float+ x[I][I];
            minimize sum(i in I, j in I) x[next[i]][j + 1];
            subject to {}
        """

        with self.assertRaisesRegex(SemanticError, "index 2.*can exceed"):
            prove_abstract_equivalent(model, model, mode="auto")

    def test_abstract_comparison_accepts_guarded_affine_index(self):
        model = """
            int N = ...;
            range I = 1..N;
            dvar float+ x[I];
            minimize sum(i in I : i < N) x[i + 1];
            subject to {}
        """

        self.assertTrue(prove_abstract_equivalent(model, model, mode="auto").equivalent)

    def test_conjoined_filter_guards_affine_index_in_multi_iterator_sum(self):
        model = """
            int N = ...;
            range I = 1..N;
            int next[I] = ...;
            dvar float+ x[I][I];
            minimize sum(i in I, j in I : i >= 1 && j < N) x[next[i]][j + 1];
            subject to {}
        """

        result = prove_abstract_equivalent(model, model, mode="auto")

        self.assertEqual(result.status, "unknown")
        self.assertIn("index 1", result.reason)

    def test_conditional_guard_protects_affine_index(self):
        model = """
            range I = 1..8;
            dvar float+ x[I];
            minimize sum(i in I) ((i > 1) ? x[i - 1] : x[i]);
            subject to {}
        """

        self.assertTrue(prove_abstract_equivalent(model, model, mode="auto").equivalent)

    def test_conditional_without_guard_rejects_unsafe_index(self):
        model = """
            range I = 1..8;
            dvar float+ x[I];
            minimize sum(i in I) ((i > 1) ? x[i - 2] : x[i]);
            subject to {}
        """

        with self.assertRaisesRegex(SemanticError, "can fall below"):
            prove_abstract_equivalent(model, model, mode="auto")

    def test_forall_rejects_unsafe_index(self):
        model = """
            range I = 1..3;
            dvar float+ x[I];
            minimize 0;
            subject to { forall(i in I) x[i + 1] >= 0; }
        """

        with self.assertRaisesRegex(SemanticError, "can exceed"):
            prove_abstract_equivalent(model, model, mode="auto")

    def test_loose_filter_does_not_widen_iterator_domain(self):
        model = """
            int N = ...;
            range I = 1..N;
            dvar float+ x[I];
            minimize sum(i in I : i <= N + 1) x[i];
            subject to {}
        """

        self.assertTrue(prove_abstract_equivalent(model, model, mode="auto").equivalent)

    def test_incomparable_symbolic_ranges_are_unknown(self):
        model = """
            int N = ...;
            int M = ...;
            range I = 1..N;
            range J = 1..M;
            dvar float+ x[J];
            minimize sum(i in I) x[i];
            subject to {}
        """

        result = prove_abstract_equivalent(model, model, mode="auto")

        self.assertEqual(result.status, "unknown")
        self.assertIn("cannot prove index", result.reason)

    def test_abstract_comparison_is_unknown_for_data_selected_index(self):
        model = """
            int N = ...;
            range I = 1..N;
            int next[I] = ...;
            dvar float+ x[I];
            minimize sum(i in I) x[next[i]];
            subject to {}
        """

        result = prove_abstract_equivalent(model, model, mode="auto")

        self.assertEqual(result.status, "unknown")
        self.assertIn("cannot prove index", result.reason)

    def test_portfolio_instance_counterexample(self):
        witness = 1.5
        for data_name, data in PORTFOLIO_17_DATA.items():
            for side, model in PORTFOLIO_17_MODELS.items():
                with self.subTest(data=data_name, side=side):
                    problem = linear_problem_from_opl(model, data)
                    self.assertEqual(problem.var_names, ["x"])
                    self.assertEqual(problem.integrality, [0])
                    lower, upper = problem.bounds[0]
                    feasible = (
                        (lower is None or lower <= witness)
                        and (upper is None or witness <= upper)
                        and all(row[0] * witness <= bound for row, bound in zip(problem.A_ub, problem.b_ub))
                        and all(row[0] * witness == bound for row, bound in zip(problem.A_eq, problem.b_eq))
                    )
                    self.assertEqual(feasible, data_name == "theta_2.dat" and side == "left")

    def test_soundness_boundary_refusals(self):
        cases = (
            (
                "nonnegative_parameter_denominator",
                "float+ theta = ...; dvar float x; minimize x; subject to {}",
                "float+ theta = ...; dvar float x; dvar float auxiliary; minimize x; " "subject to {theta*auxiliary == x;}",
                {"right_auxiliaries": {"auxiliary"}, "assumptions": {"theta": "nonnegative"}},
                "uniform_schema",
            ),
            (
                "unbounded_integer_divisibility",
                "dvar int x; minimize x; subject to {x>=0;}",
                "dvar int x; dvar int auxiliary; minimize x; subject to {x>=0; 2*auxiliary==x;}",
                {"right_auxiliaries": {"auxiliary"}},
                "parameterless_instance",
            ),
            (
                "narrow_interval_is_not_fixed",
                "dvar float x; minimize x; subject to {x>=0; x<=0.0000000005;}",
                "dvar float x; minimize x; subject to {x==0;}",
                {},
                "parameterless_instance",
            ),
            (
                "sub_tolerance_margin_is_not_redundant",
                "dvar float x; minimize x; subject to {x>=0; x<=1;}",
                "dvar float x; minimize x; subject to {x>=0; x<=1; x<=0.9999999995;}",
                {},
                "parameterless_instance",
            ),
            (
                "leading_negative_row_preserves_orientation",
                "float theta = ...; dvar float x; minimize x; subject to {-2*x<=-2*theta;}",
                "float theta = ...; dvar float x; minimize x; subject to {x<=theta;}",
                {},
                "uniform_schema",
            ),
        )
        for name, left, right, options, scope in cases:
            with self.subTest(case=name):
                result = prove_abstract_equivalent(left, right, mode="algebraic", variable_mapping={"x": "x"}, **options)
                self.assertEqual(result.status, "unknown")
                self.assertEqual(result.scope, scope)
                self.assertIsNone(result.counterexample)

    def test_rewrite_limit_reports_budget(self):
        model = "dvar float x; minimize x; subject to {x>=0;}"
        result = prove_abstract_equivalent(model, model, mode="algebraic", max_rewrite_iterations=0)
        self.assertEqual(result.status, "unknown")
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.termination, "budget_exhausted")

    def test_integer_enumeration_limit_reports_budget(self):
        left = "dvar int x; minimize x; subject to {x>=0; x<=100001;}"
        right = left.replace("minimize x", "minimize 2*x")
        result = prove_abstract_equivalent(left, right, mode="algebraic")
        self.assertEqual(result.status, "unknown")
        self.assertTrue(result.budget_exhausted)

    def test_objective_certificate_failure_is_unknown(self):
        left = "dvar float x; dvar float y; minimize x; subject to {x+y==1;}"
        right = left.replace("minimize x", "minimize 1-y")
        with patch("pyopl._abstract_algebra._objectives_equal_on_polyhedron", return_value=False):
            result = prove_abstract_equivalent(left, right, mode="algebraic", variable_mapping={"x": "x", "y": "y"})
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.counterexample)

    def test_equality_certificate_retains_both_directions(self):
        import sympy as symbolic

        from pyopl._abstract_algebra import AffineConstraint, _farkas_certificate, _verify_farkas

        decision = symbolic.Symbol("decision", real=True)
        equality = AffineConstraint(decision - 1, "=")
        certificates = _farkas_certificate((equality,), equality, ["decision"])
        self.assertIsInstance(certificates, tuple)
        for sign, certificate in zip((1, -1), certificates):
            self.assertTrue(
                _verify_farkas((equality,), AffineConstraint(sign * equality.expression, "<="), ["decision"], certificate)
            )

    def test_parameterized_rational_row_scaling(self):
        left = "float theta = ...; dvar float x; minimize x; subject to {2*x<=2*theta;}"
        right = "float theta = ...; dvar float x; minimize x; subject to {x<=theta;}"
        result = prove_abstract_equivalent(left, right, mode="algebraic")
        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "symbolically_normalized")
        reversed_row = right.replace("x<=theta", "-x<=-theta")
        self.assertFalse(prove_abstract_equivalent(left, reversed_row, mode="algebraic").equivalent)

    def test_equivalent_retained_equality_bases(self):
        left = "dvar float x; dvar float y; minimize x; subject to {x==0; y==0;}"
        right = "dvar float x; dvar float y; minimize x; subject to {x+y==0; x-y==0;}"
        result = prove_abstract_equivalent(left, right, mode="algebraic", variable_mapping={"x": "x", "y": "y"})
        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "polyhedrally_proven")

    def test_objectives_equal_on_retained_equality(self):
        left = "dvar float x; dvar float y; minimize x; subject to {x+y==1;}"
        right = left.replace("minimize x", "minimize 1-y")
        self.assertTrue(
            prove_abstract_equivalent(left, right, mode="algebraic", variable_mapping={"x": "x", "y": "y"}).equivalent
        )

    def test_projection_handles_one_sided_and_empty_fibers(self):
        left = "dvar float x; minimize x; subject to {}"
        right = "dvar float x; dvar float auxiliary; minimize x; subject to {auxiliary>=x;}"
        options = dict(mode="algebraic", variable_mapping={"x": "x"}, right_auxiliaries={"auxiliary"})
        self.assertTrue(prove_abstract_equivalent(left, right, **options).equivalent)
        empty = right.replace("auxiliary>=x;", "auxiliary>=1; auxiliary<=0;")
        self.assertFalse(prove_abstract_equivalent(left, empty, **options).equivalent)

    def test_failed_implication_discovery_is_unknown(self):
        left = "dvar float x; minimize x; subject to { x>=0; x<=1; }"
        right = "dvar float x; minimize x; subject to { x>=0; x<=1; x<=2; }"
        for target in ("_solve_farkas_multipliers", "_verify_farkas"):
            with self.subTest(target=target), patch("pyopl._abstract_algebra." + target, return_value=None):
                result = prove_abstract_equivalent(left, right, mode="algebraic", variable_mapping={"x": "x"})
                self.assertEqual(result.status, "unknown")
                self.assertIsNone(result.counterexample)

    def test_missing_mapping_is_unknown(self):
        left = "dvar float x; minimize x; subject to {}"
        right = "dvar float x; dvar float auxiliary; minimize x; subject to {}"
        self.assertEqual(prove_abstract_equivalent(left, right, mode="auto").status, "unknown")

    def test_mapping_search_limit_is_unknown(self):
        from pyopl._abstract_algebra import AlgebraicProof

        model = " ".join(f"dvar int decision{index};" for index in range(6)) + " minimize decision0; subject to {}"
        rejection = AlgebraicProof("different", "presburger_proven", "candidate rejected")
        with patch("pyopl._abstract_algebra._prove_candidate_mapping", return_value=rejection) as attempt:
            result = prove_abstract_equivalent(model, model, mode="algebraic")
        self.assertEqual(result.status, "unknown")
        self.assertIn("limit", result.reason)
        self.assertEqual(attempt.call_count, 256)
        self.assertTrue(result.budget_exhausted)

    def test_mapping_search_continues_after_candidate_disproof(self):
        left = "dvar int x; dvar int y; minimize x+2*y; subject to {x>=0; x<=1; y>=0; y<=1;}"
        right = "dvar int a; dvar int b; minimize 2*a+b; subject to {a>=0; a<=1; b>=0; b<=1;}"
        self.assertTrue(prove_abstract_equivalent(left, right, mode="algebraic").equivalent)

    def test_symbolic_alias_requires_nonzero_parameter(self):
        left = "float+ theta = ...; dvar float x; minimize x; subject to {}"
        for equality in ("theta*auxiliary == x", "theta*auxiliary == -x"):
            right = "float+ theta = ...; dvar float x; dvar float auxiliary; minimize x; subject to {" + equality + ";}"
            for condition in (None, "nonnegative", "positive", "nonzero"):
                with self.subTest(equality=equality, condition=condition):
                    result = prove_abstract_equivalent(
                        left,
                        right,
                        mode="algebraic",
                        variable_mapping={"x": "x"},
                        right_auxiliaries={"auxiliary"},
                        assumptions=None if condition is None else {"theta": condition},
                    )
                    self.assertEqual(result.status, "equivalent" if condition in {"positive", "nonzero"} else "unknown")

    def test_structural_and_auto_modes_honor_explicit_variable_mapping(self):
        left = "dvar float+ x; dvar float+ y; minimize x + 2*y; subject to { x <= 1; y <= 1; }"
        right = "dvar float+ a; dvar float+ b; minimize a + 2*b; subject to { a <= 1; b <= 1; }"
        for mode in ("structural", "auto"):
            with self.subTest(mode=mode):
                correct = prove_abstract_equivalent(left, right, mode=mode, variable_mapping={"x": "a", "y": "b"})
                incorrect = prove_abstract_equivalent(left, right, mode=mode, variable_mapping={"x": "b", "y": "a"})

                self.assertTrue(correct.equivalent)
                self.assertFalse(incorrect.equivalent)

    def test_auto_mode_honors_explicit_parameter_mapping(self):
        left = "float p = ...; float q = ...; dvar float+ x; minimize p*x + q; subject to { x <= 1; }"
        right = "float a = ...; float b = ...; dvar float+ y; minimize a*y + b; subject to { y <= 1; }"
        correct = prove_abstract_equivalent(left, right, mode="auto", parameter_mapping={"p": "a", "q": "b"})
        incorrect = prove_abstract_equivalent(left, right, mode="auto", parameter_mapping={"p": "b", "q": "a"})

        self.assertTrue(correct.equivalent)
        self.assertFalse(incorrect.equivalent)

    def test_indexed_schema_isomorphism_honors_explicit_mapping(self):
        for mapping, expected in (({"x": "quantity"}, True),):
            with self.subTest(mapping=mapping):
                result = prove_abstract_equivalent(LEFT_MODEL, RENAMED_MODEL, mode="auto", variable_mapping=mapping)

                self.assertEqual(result.equivalent, expected)
        for mapping in ({"x": "unitCost"}, {"missing": "quantity"}):
            with self.assertRaises(ValueError):
                prove_abstract_equivalent(LEFT_MODEL, RENAMED_MODEL, mode="auto", variable_mapping=mapping)

    def test_structural_mapping_must_be_injective(self):
        model = "dvar float+ x; dvar float+ y; minimize x+y; subject to { x <= 1; y <= 1; }"
        with self.assertRaises(ValueError):
            prove_abstract_equivalent(model, model, variable_mapping={"x": "x", "y": "x"})

    def test_compare_abstract_accepts_renamed_and_reordered_model_schema(self):
        self.assertTrue(compare_abstract(LEFT_MODEL, RENAMED_MODEL))

    def test_prove_abstract_equivalent_accepts_parser_asts(self):
        left_ast = parse_abstract_model(LEFT_MODEL)
        right_ast = parse_abstract_model(RENAMED_MODEL)

        result = prove_abstract_equivalent(left_ast, right_ast)

        self.assertIsInstance(result, AbstractEquivalenceResult)
        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "schema_isomorphic")
        self.assertTrue(result.equivalent)
        self.assertIn("linked declarations and bound iterators to their references", result.proof_steps)
        self.assertIsNone(result.counterexample)

    def test_compare_abstract_rejects_changed_symbolic_objective(self):
        changed = RENAMED_MODEL.replace(
            "quantity[product] * unitCost[product]",
            "2 * quantity[product] * unitCost[product]",
        )

        result = prove_abstract_equivalent(LEFT_MODEL, changed)

        self.assertEqual(result.status, "different")
        self.assertFalse(result.equivalent)
        self.assertIn("not isomorphic", result.reason)
        self.assertIsNotNone(result.counterexample)

    def test_compare_abstract_preserves_index_dimension_order(self):
        left = """
            int N = ...;
            int M = ...;
            range I = 1..N;
            range J = 1..M;
            float c[I][J] = ...;
            dvar float+ x[I][J];
            minimize sum(i in I, j in J) c[i][j] * x[i][j];
            subject to { forall(i in I, j in J) x[i][j] <= 1; }
        """
        right = """
            int rows = ...;
            int columns = ...;
            range R = 1..rows;
            range C = 1..columns;
            float cost[R][C] = ...;
            dvar float+ value[R][C];
            minimize sum(r in R, c in C) cost[r][c] * value[r][c];
            subject to { forall(r in R, c in C) value[c][r] <= 1; }
        """

        self.assertFalse(compare_abstract(left, right))

    def test_prove_abstract_equivalent_returns_unknown_for_malformed_ast(self):
        result = prove_abstract_equivalent(
            {"declarations": [], "constraints": []},
            parse_abstract_model(LEFT_MODEL),
        )

        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.equivalent)
        self.assertIn("objective", result.reason)
        self.assertEqual(result.proof_steps, ())

    def test_parse_abstract_model_keeps_symbolic_external_parameters(self):
        ast = parse_abstract_model(LEFT_MODEL)
        declarations = {declaration["name"]: declaration for declaration in ast["declarations"]}

        self.assertEqual(declarations["N"]["type"], "parameter_external")
        self.assertEqual(declarations["cost"]["type"], "parameter_external_explicit_indexed")
        self.assertNotIn("value", declarations["cost"])

    def test_algebraic_mode_normalizes_expanded_affine_expressions(self):
        left = """
            dvar float x;
            minimize 2 * (x + 1);
            subject to { x >= 0; }
        """
        right = """
            dvar float y;
            minimize 2 * y + 2;
            subject to { 0 <= y; }
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "symbolically_normalized")

    def test_algebraic_mode_normalizes_indexed_distributive_sum(self):
        left = """
            int N = ...;
            range Products = 1..N;
            float unitCost[Products] = ...;
            dvar float quantity[Products];
            minimize sum(p in Products) unitCost[p] * (quantity[p] + 1);
            subject to {}
        """
        right = """
            int count = ...;
            range Items = 1..count;
            float cost[Items] = ...;
            dvar float amount[Items];
            minimize sum(item in Items) (cost[item] * amount[item] + cost[item]);
            subject to {}
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "symbolically_normalized")
        self.assertEqual(result.scope, "uniform_schema")
        self.assertIn("normalized indexed affine expressions", result.proof_steps)
        self.assertIn("lifted equality through alpha-normalized quantifiers", result.proof_steps)

    def test_indexed_algebra_rejects_different_inline_parameter_definitions(self):
        left = """
            int N = 2;
            range I = 1..N;
            dvar float x[I];
            minimize sum(i in I) x[i];
            subject to {}
        """
        right = left.replace("int N = 2;", "int N = 3;")

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.equivalent)
        self.assertIn("no compatible indexed declaration mapping", result.reason)

    def test_index_safety_requires_interpretable_named_set_domain(self):
        model = """
            {string} Products = {"A", "B"};
            dvar float x[Products];
            minimize sum(product in Products) x[product];
            subject to {}
        """

        result = prove_abstract_equivalent(model, model, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertFalse(result.equivalent)
        self.assertIn("cannot interpret declared range", result.reason)

    def test_algebraic_mode_normalizes_pointwise_indexed_constraint(self):
        left = """
            int N = ...;
            range I = 1..N;
            float a[I] = ...;
            dvar float x[I];
            minimize sum(i in I) x[i];
            subject to { forall(i in I) a[i] * (x[i] + 1) <= 0; }
        """
        right = """
            int size = ...;
            range J = 1..size;
            float coefficient[J] = ...;
            dvar float value[J];
            minimize sum(j in J) value[j];
            subject to { forall(j in J) 0 >= coefficient[j] * value[j] + coefficient[j]; }
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("normalized pointwise affine constraints", result.proof_steps)

    def test_indexed_algebraic_near_misses_remain_unknown(self):
        base = """
            int N = ...;
            range I = 1..N;
            range J = 1..N;
            float a[I] = ...;
            dvar float x[I];
            minimize sum(i in I) a[i] * x[i];
            subject to {}
        """
        variants = {
            "different domain": base.replace("sum(i in I)", "sum(i in J)"),
            "filtered sum": base.replace("sum(i in I)", "sum(i in I : i >= 1)"),
            "symbolic division": base.replace("a[i] * x[i]", "x[i] / a[i]"),
            "decision product": base.replace("a[i] * x[i]", "x[i] * x[i]"),
            "rank mismatch": base.replace("dvar float x[I];", "dvar float x[I][I];").replace("x[i];", "x[i][i];"),
        }
        for name, variant in variants.items():
            with self.subTest(case=name):
                result = prove_abstract_equivalent(base, variant, mode="algebraic")
                self.assertEqual(result.status, "unknown")
                self.assertIsNone(result.counterexample)

    def test_algebraic_mode_normalizes_nested_indexed_sums(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I) sum(j in I) 2 * (x[i][j] + 1);
            subject to {}
        """
        right = """
            int size = ...;
            range Items = 1..size;
            dvar float value[Items][Items];
            minimize sum(row in Items) sum(column in Items) (2 * value[row][column] + 2);
            subject to {}
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("alpha-normalized nested indexed binders", result.proof_steps)

    def test_nested_indexed_shadowing_does_not_capture_outer_binder(self):
        shadowed = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I) sum(i in I) x[i][i];
            subject to {}
        """
        distinct = shadowed.replace("sum(i in I) x[i][i]", "sum(j in I) x[i][j]")

        result = prove_abstract_equivalent(shadowed, distinct, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.counterexample)

    def test_multiple_shadowed_binders_receive_fresh_identities(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I, j in I) sum(i in I, j in I) x[i][j];
            subject to {}
        """
        right = """
            int size = ...;
            range Items = 1..size;
            dvar float value[Items][Items];
            minimize sum(outerRow in Items, outerColumn in Items)
                sum(innerRow in Items, innerColumn in Items) value[innerRow][innerColumn];
            subject to {}
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")

    def test_algebraic_mode_normalizes_filtered_indexed_sums(self):
        left = """
            int N = ...;
            range I = 1..N;
            float a[I] = ...;
            dvar float x[I];
            minimize sum(i in I : i >= 1 && i <= N) a[i] * (x[i] + 1);
            subject to {}
        """
        right = """
            int size = ...;
            range Items = 1..size;
            float coefficient[Items] = ...;
            dvar float value[Items];
            minimize sum(item in Items : item <= size && item >= 1)
                (coefficient[item] * value[item] + coefficient[item]);
            subject to {}
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("canonicalized indexed quantifier filters", result.proof_steps)

    def test_different_indexed_filters_remain_unknown(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I];
            minimize sum(i in I : i >= 1) x[i];
            subject to {}
        """
        right = left.replace("i >= 1", "i >= 2")

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.counterexample)

    def test_algebraic_mode_preserves_nested_index_expressions(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I, j in I : j < N) 2 * (x[i][j + 1] + 1);
            subject to {}
        """
        right = """
            int size = ...;
            range Items = 1..size;
            dvar float value[Items][Items];
            minimize sum(row in Items, column in Items : column < size)
                (2 * value[row][column + 1] + 2);
            subject to {}
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("preserved complete indexed access expressions", result.proof_steps)

    def test_index_position_permutation_remains_unknown(self):
        left = """
            int N = ...;
            int M = ...;
            range I = 1..N;
            range J = 1..M;
            dvar float x[I][J];
            minimize sum(i in I, j in J) x[i][j];
            subject to {}
        """
        right = left.replace("x[i][j];", "x[j][i];")

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.counterexample)
        self.assertIn("cannot prove index", result.reason)

    def test_indexed_normalization_limit_reports_budget(self):
        model = """
            int N = ...;
            range I = 1..N;
            dvar float x[I];
            minimize sum(i in I) 2 * (x[i] + 1);
            subject to {}
        """

        result = prove_abstract_equivalent(model, model, mode="algebraic", max_rewrite_iterations=0)

        self.assertEqual(result.status, "unknown")
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.termination, "budget_exhausted")

    def test_algebraic_mode_normalizes_nested_forall_constraints(self):
        left = """
            int N = ...;
            range I = 1..N;
            float a[I] = ...;
            dvar float x[I][I];
            minimize 0;
            subject to {
                forall(i in I) {
                    forall(j in I) a[j] * (x[i][j] + 1) <= 0;
                }
            }
        """
        right = """
            int size = ...;
            range Items = 1..size;
            float coefficient[Items] = ...;
            dvar float value[Items][Items];
            minimize 0;
            subject to {
                forall(row in Items) {
                    forall(column in Items)
                        0 >= coefficient[column] * value[row][column] + coefficient[column];
                }
            }
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("alpha-normalized nested forall constraints", result.proof_steps)

    def test_nested_forall_shadowing_does_not_capture_outer_binder(self):
        shadowed = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize 0;
            subject to { forall(i in I) { forall(i in I) x[i][i] <= 1; } }
        """
        distinct = shadowed.replace("forall(i in I) x[i][i]", "forall(j in I) x[i][j]")

        result = prove_abstract_equivalent(shadowed, distinct, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.counterexample)

    def test_dependent_domains_preserve_binder_dependencies(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I, j in 1..i) 2 * (x[i][j] + 1);
            subject to {}
        """
        right = """
            int size = ...;
            range Items = 1..size;
            dvar float value[Items][Items];
            minimize sum(row in Items) sum(column in 1..row) (2 * value[row][column] + 2);
            subject to {}
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("preserved dependent quantifier domains", result.proof_steps)
        self.assertIn("flattened ordered indexed binders", result.proof_steps)

    def test_different_dependent_domain_bound_remains_unknown(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I, j in 1..i) x[i][j];
            subject to {}
        """
        right = left.replace("j in 1..i", "j in 1..N")

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.counterexample)

    def test_parameter_selected_indices_require_data_for_safety(self):
        left = """
            int N = ...;
            range I = 1..N;
            int next[I] = ...;
            dvar float x[I][I];
            minimize sum(i in I, j in I : j < N) 2 * (x[next[i]][j + 1] + 1);
            subject to {}
        """
        right = """
            int size = ...;
            range Items = 1..size;
            int successor[Items] = ...;
            dvar float value[Items][Items];
            minimize sum(row in Items, column in Items : column < size)
                (2 * value[successor[row]][column + 1] + 2);
            subject to {}
        """

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "unknown")
        self.assertIn("cannot prove index 1", result.reason)

    def test_different_parameter_selected_index_remains_unknown(self):
        left = """
            int N = ...;
            range I = 1..N;
            int next[I] = ...;
            int previous[I] = ...;
            dvar float x[I];
            minimize sum(i in I) x[next[i]];
            subject to {}
        """
        right = left.replace("x[next[i]]", "x[previous[i]]")

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            parameter_mapping={"N": "N", "next": "next", "previous": "previous"},
        )

        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.counterexample)

    def test_independent_binders_can_be_reordered(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I, j in I) x[i][j];
            subject to {}
        """
        right = left.replace("sum(i in I, j in I)", "sum(j in I, i in I)")

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("reordered independent indexed binders", result.proof_steps)

    def test_dependent_binder_order_is_not_reordered(self):
        left = """
            int N = ...;
            range I = 1..N;
            dvar float x[I][I];
            minimize sum(i in I, j in 1..i) x[i][j];
            subject to {}
        """
        right = left.replace("j in 1..i", "j in 1..N")

        result = prove_abstract_equivalent(left, right, mode="algebraic")

        self.assertEqual(result.status, "unknown")

    def test_sums_split_and_fuse_only_over_identical_domains_and_filters(self):
        fused = """
            int N = ...;
            range I = 1..N;
            float a[I] = ...;
            dvar float x[I];
            minimize sum(i in I : i >= 1) (a[i] * x[i] + a[i]);
            subject to {}
        """
        split = """
            int size = ...;
            range Items = 1..size;
            float coefficient[Items] = ...;
            dvar float value[Items];
            minimize sum(item in Items : item >= 1) coefficient[item] * value[item]
                + sum(item in Items : item >= 1) coefficient[item];
            subject to {}
        """

        result = prove_abstract_equivalent(fused, split, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertIn("fused indexed sums with identical domains and filters", result.proof_steps)

        changed_filter = split.replace(
            "sum(item in Items : item >= 1) coefficient[item];",
            "sum(item in Items : item >= 2) coefficient[item];",
        )
        refused = prove_abstract_equivalent(fused, changed_filter, mode="algebraic")
        self.assertEqual(refused.status, "unknown")
        self.assertIsNone(refused.counterexample)

    def test_nested_filtered_identity_has_grounded_cross_check(self):
        left = """
            int N = ...;
            range I = 1..N;
            float a[I] = ...;
            dvar float x[I][I];
            minimize sum(i in I) sum(j in 1..i : j >= 1) a[j] * (x[i][j] + 1);
            subject to {}
        """
        right = """
            int N = ...;
            range I = 1..N;
            float a[I] = ...;
            dvar float x[I][I];
            minimize sum(i in I) sum(j in 1..i : j >= 1) (a[j] * x[i][j] + a[j]);
            subject to {}
        """
        data = "N = 3; a = [2, 3, 5];"

        symbolic = prove_abstract_equivalent(left, right, mode="algebraic")
        grounded = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            left_data_text=data,
            right_data_text=data,
        )

        self.assertEqual(symbolic.status, "equivalent")
        self.assertEqual(symbolic.scope, "uniform_schema")
        self.assertEqual(grounded.status, "equivalent")
        self.assertEqual(grounded.scope, "supplied_instances")

    def test_algebraic_mode_eliminates_arbitrary_affine_alias(self):
        left = """
            dvar float x;
            minimize 3 * x + 4;
            subject to { x >= 0; x <= 5; }
        """
        right = """
            dvar float y;
            dvar float alias;
            minimize alias;
            subject to {
                2 * alias == 6 * y + 8;
                y >= 0;
                y <= 5;
            }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"alias"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "rewrite_certified")
        self.assertIn("affine aliases", " ".join(result.proof_steps))

    def test_algebraic_mode_projects_continuous_auxiliary(self):
        left = """
            dvar float x;
            minimize x;
            subject to { x >= 0; x <= 1; }
        """
        right = """
            dvar float y;
            dvar float slack;
            minimize y;
            subject to {
                y >= 0;
                slack >= y;
                slack <= 1;
            }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"slack"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "polyhedrally_proven")
        self.assertIn("Fourier-Motzkin", " ".join(result.proof_steps))

    def test_algebraic_mode_verifies_farkas_redundancy_certificates(self):
        left = """
            dvar float x;
            dvar float y;
            minimize x + y;
            subject to { x >= 0; y >= 0; x <= 1; y <= 1; }
        """
        right = """
            dvar float a;
            dvar float b;
            minimize a + b;
            subject to { a >= 0; b >= 0; a <= 1; b <= 1; a + b <= 2; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "a", "y": "b"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "polyhedrally_proven")
        self.assertIn("Farkas", " ".join(result.proof_steps))

    def test_algebraic_mode_honors_parameter_mapping(self):
        left = """
            float p = ...;
            dvar float x;
            minimize p * x + p;
            subject to { x >= 0; }
        """
        right = """
            float coefficient = ...;
            dvar float y;
            minimize coefficient * (y + 1);
            subject to { 0 <= y; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            parameter_mapping={"p": "coefficient"},
            variable_mapping={"x": "y"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "symbolically_normalized")

    def test_algebraic_mode_eliminates_bounded_integer_auxiliary(self):
        left = """
            dvar int x;
            minimize x;
            subject to { x >= 0; x <= 1; }
        """
        right = """
            dvar int y;
            dvar int auxiliary;
            minimize y;
            subject to { y >= 0; y <= 1; auxiliary >= 0; auxiliary <= 1; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"auxiliary"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "presburger_proven")

    def test_algebraic_mode_preserves_integer_alias_divisibility(self):
        left = """
            dvar int x;
            minimize x;
            subject to { x >= 0; x <= 3; }
        """
        right = """
            dvar int y;
            dvar int half;
            minimize y;
            subject to { y >= 0; y <= 3; half >= 0; half <= 2; 2 * half == y; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"half"},
        )

        self.assertEqual(result.status, "different")
        self.assertEqual(result.level, "presburger_proven")
        self.assertIn("x=1", result.counterexample or "")

    def test_algebraic_mode_saturates_chained_alias_rewrites(self):
        left = """
            dvar float x;
            minimize x + 2;
            subject to { x >= 0; }
        """
        right = """
            dvar float y;
            dvar float first;
            dvar float second;
            minimize second;
            subject to { first == y + 1; second == first + 1; y >= 0; }
        """

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            variable_mapping={"x": "y"},
            right_auxiliaries={"first", "second"},
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "rewrite_certified")

    def test_algebraic_mode_normalizes_indexed_schema(self):
        result = prove_abstract_equivalent(LEFT_MODEL, RENAMED_MODEL, mode="algebraic")

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.scope, "uniform_schema")

    def test_algebraic_mode_grounds_indexed_schema_with_data(self):
        left = """
            int N = ...;
            range I = 1..N;
            float c[I] = ...;
            dvar float+ x[I];
            minimize sum(i in I) c[i] * x[i];
            subject to { forall(i in I) x[i] <= 2; }
        """
        right = """
            int N = ...;
            range I = 1..N;
            float c[I] = ...;
            dvar float+ x[I];
            minimize sum(i in I) x[i] * c[i];
            subject to { forall(i in I) 2 >= x[i]; }
        """
        data = "N = 3; c = [1, 2, 3];"

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            left_data_text=data,
            right_data_text=data,
        )

        self.assertEqual(result.status, "equivalent")
        self.assertEqual(result.level, "symbolically_normalized")
        self.assertIn("grounded finite models", " ".join(result.proof_steps))

    def test_grounded_indexed_formulations_require_auxiliary_partition(self):
        left = """
            int N = ...; range I = 1..N;
            dvar boolean x[I]; dvar float+ load[I];
            minimize sum(i in I) x[i];
            subject to { forall(i in I) load[i] >= x[i]; }
        """
        right = """
            int N = ...; range I = 1..N;
            dvar boolean x[I]; dvar float+ flow[I][I];
            minimize sum(i in I) x[i];
            subject to { forall(i in I) sum(j in I) flow[i][j] >= x[i]; }
        """
        data = "N = 2;"

        result = prove_abstract_equivalent(
            left,
            right,
            mode="algebraic",
            left_data_text=data,
            right_data_text=data,
        )

        self.assertEqual(result.status, "unknown")
        self.assertIn("no compatible parameter and variable mapping", result.reason)


if __name__ == "__main__":
    unittest.main()

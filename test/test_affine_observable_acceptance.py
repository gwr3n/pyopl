import itertools
import unittest
from dataclasses import replace
from math import factorial
from pathlib import Path
from typing import Sequence

from pyopl.equivalence import (
    AffineObservable,
    IndexDomain,
    IndexedAffineExpression,
    IndexedAffineFamily,
    IndexedAffineTerm,
    IndexedObservablePair,
    ObservablePair,
    ObservableTemplate,
    ParameterContract,
    ParameterFact,
    SymbolicAffineExpression,
    SymbolicDomain,
    SymbolicPredicate,
    parse_observable_mapping,
)
from pyopl.equivalence.abstract import prove_abstract_equivalent
from pyopl.equivalence.concrete import prove_equivalent
from pyopl.pyopl_core import linear_problem_from_opl

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "affine_observable"


def _identity_arc_template(size: int) -> ObservableTemplate:
    return ObservableTemplate(
        "x_{origin}_{destination}",
        (IndexDomain("origin", tuple(range(1, size + 1))), IndexDomain("destination", tuple(range(1, size + 1)))),
        lambda indices: AffineObservable(((f"x_{indices['origin']}_{indices['destination']}", 1.0),)),
        lambda indices: AffineObservable(((f"x_{indices['origin']}_{indices['destination']}", 1.0),)),
        guard=lambda indices: indices["origin"] != indices["destination"],
    )


def _identity_arc_mapping(size: int) -> tuple[ObservablePair, ...]:
    return _identity_arc_template(size).expand()


def _rs_mapping(size: int) -> tuple[ObservablePair, ...]:
    return tuple(
        ObservablePair(
            f"order_{period}",
            AffineObservable(((f"order_{period}", 1.0),)),
            AffineObservable(tuple((f"x_{period}_{destination}", 1.0) for destination in range(period + 1, size + 2))),
        )
        for period in range(1, size + 1)
    )


def _rs_indexed_mapping() -> tuple[IndexedObservablePair, ...]:
    one = SymbolicAffineExpression.constant_value(1)
    period = SymbolicDomain("t", one, SymbolicAffineExpression.parameter("N"))
    arc_end = SymbolicDomain(
        "j",
        SymbolicAffineExpression.parameter("t").add(one),
        SymbolicAffineExpression.parameter("N").add(one),
    )
    return (
        IndexedObservablePair(
            "order[{t}]",
            IndexedAffineFamily(
                ("t",),
                (period,),
                IndexedAffineExpression((IndexedAffineTerm("order", ("t",)),)),
            ),
            IndexedAffineFamily(
                ("t",),
                (period,),
                IndexedAffineExpression(
                    (IndexedAffineTerm("x", ("t", "j")),),
                    sum_binders=("j",),
                    sum_domains=(arc_end,),
                ),
            ),
        ),
    )


def _tsp_indexed_mapping() -> tuple[IndexedObservablePair, ...]:
    one = SymbolicAffineExpression.constant_value(1)
    city = SymbolicDomain("i", one, SymbolicAffineExpression.parameter("N"))
    destination = SymbolicDomain("j", one, SymbolicAffineExpression.parameter("N"))
    guard = SymbolicPredicate(
        SymbolicAffineExpression.parameter("i"),
        "!=",
        SymbolicAffineExpression.parameter("j"),
    )
    left = IndexedAffineFamily(
        ("i", "j"),
        (city, destination),
        IndexedAffineExpression((IndexedAffineTerm("x", ("i", "j")),)),
        guard,
    )
    right = IndexedAffineFamily(
        ("u", "v"),
        (
            SymbolicDomain("u", one, SymbolicAffineExpression.parameter("N")),
            SymbolicDomain("v", one, SymbolicAffineExpression.parameter("N")),
        ),
        IndexedAffineExpression((IndexedAffineTerm("x", ("u", "v")),)),
        SymbolicPredicate(
            SymbolicAffineExpression.parameter("u"),
            "!=",
            SymbolicAffineExpression.parameter("v"),
        ),
    )
    return (IndexedObservablePair("x[{i}][{j}]", left, right),)


def _tsp_data(size: int) -> str:
    cities = range(1, size + 1)
    costs = [[0 if origin == destination else 3 * origin + destination for destination in cities] for origin in cities]
    subsets = [
        set(subset) for cardinality in range(2, size) for subset in itertools.combinations(range(2, size + 1), cardinality)
    ]
    incidence = [[int(city in subset) for city in cities] for subset in subsets]

    def vector(values: Sequence[object]) -> str:
        return "[" + ", ".join(str(value) for value in values) + "]"

    def matrix(values: Sequence[Sequence[object]]) -> str:
        return "[\n  " + ",\n  ".join(vector(row) for row in values) + "\n]"

    return "\n".join(
        (
            f"N = {size};",
            f"cost = {matrix(costs)};",
            f"subsetCount = {len(subsets)};",
            f"subsetMember = {matrix(incidence)};",
            f"subsetSize = {vector([len(subset) for subset in subsets])};",
        )
    )


def _rs_data(size: int) -> str:
    means = [10] * size
    prefix = [0] + [10 * index for index in range(1, size + 1)]
    cycle_mean = [
        [10 * (end - start + 1) if end >= start else 0 for end in range(1, size + 1)] for start in range(1, size + 1)
    ]
    cycle_std = [
        [(end - start + 1) ** 0.5 if end >= start else 0 for end in range(1, size + 1)] for start in range(1, size + 1)
    ]

    def vector(values: Sequence[object]) -> str:
        return "[" + ", ".join(str(value) for value in values) + "]"

    def matrix(values: Sequence[Sequence[object]]) -> str:
        return "[\n  " + ",\n  ".join(vector(row) for row in values) + "\n]"

    return "\n".join(
        (
            f"N = {size};",
            "setupCost = 20;",
            "holdingCost = 1;",
            "backorderCost = 5;",
            "bigM = 100;",
            "lossBigM = 100;",
            f"meanDemand = {vector(means)};",
            f"stdDemand = {vector([1] * size)};",
            f"prefixMean = {vector(prefix)};",
            f"cycleMean = {matrix(cycle_mean)};",
            f"cycleStd = {matrix(cycle_std)};",
            "pieces = 3;",
            "normalTail = [0.8, 0.5, 0.2];",
            "normalDensity = [0.2, 0.4, 0.2];",
        )
    )


class AffineObservableAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rs_root = FIXTURE_ROOT / "rs"
        cls.rs_left_source = (rs_root / "bm" / "bm.mod").read_text()
        cls.rs_right_source = (rs_root / "pm" / "pm.mod").read_text()
        cls.rs_data = (rs_root / "common" / "sample.dat").read_text()
        cls.rs_mapping = _rs_mapping(5)
        cls.rs_left = linear_problem_from_opl(cls.rs_left_source, cls.rs_data)
        cls.rs_right = linear_problem_from_opl(cls.rs_right_source, cls.rs_data)

        tsp_root = FIXTURE_ROOT / "tsp"
        cls.tsp_data = (tsp_root / "common.dat").read_text()
        cls.tsp_sources = {path.stem: path.read_text() for path in sorted(tsp_root.glob("*.mod"))}
        cls.tsp_problems = {name: linear_problem_from_opl(source, cls.tsp_data) for name, source in cls.tsp_sources.items()}
        cls.tsp_template = _identity_arc_template(6)
        cls.tsp_mapping = cls.tsp_template.expand()

        service_route_root = FIXTURE_ROOT / "service_route"
        cls.service_route_dfj_source = (service_route_root / "service_route_dfj.mod").read_text()
        cls.service_route_f1_source = (service_route_root / "service_route_f1.mod").read_text()
        cls.service_route_data = (service_route_root / "common.dat").read_text()
        cls.service_route_mapping = parse_observable_mapping((service_route_root / "observable_mapping.json").read_text())

    def test_rs_bm_pm_concrete_projected_value_equivalence(self):
        result = prove_equivalent(
            self.rs_left,
            self.rs_right,
            mode="projected_milp",
            observable_mapping=self.rs_mapping,
            max_projected_assignments=32,
        )

        self.assertTrue(result.equivalent)
        self.assertIn("all 32 directed", result.reason)
        self.assertEqual(result.projected_assignment_counts, (16, 16))
        self.assertEqual(len(result.observable_mapping), 5)

    def test_rs_bm_pm_grounded_abstract_projected_value_equivalence(self):
        result = prove_abstract_equivalent(
            self.rs_left_source,
            self.rs_right_source,
            mode="algebraic",
            left_data_text=self.rs_data,
            right_data_text=self.rs_data,
            observable_mapping=self.rs_mapping,
            max_projected_assignments=32,
        )

        self.assertTrue(result.equivalent)
        self.assertEqual(result.scope, "supplied_instances")
        self.assertEqual(len(result.observable_mapping), 5)

    def test_service_route_dfj_and_f1_return_grounded_feasibility_counterexample(self):
        result = prove_abstract_equivalent(
            self.service_route_dfj_source,
            self.service_route_f1_source,
            mode="algebraic",
            left_data_text=self.service_route_data,
            right_data_text=self.service_route_data,
            observable_mapping=self.service_route_mapping,
        )

        self.assertEqual(result.status, "different")
        self.assertEqual(result.scope, "supplied_instances")
        self.assertEqual(result.relation, "projected_value")
        self.assertIn("feasible in only one formulation", result.reason)
        self.assertIn("feasible only in left", result.counterexample or "")
        self.assertIn("trip_", result.counterexample or "")

    def test_rs_bm_pm_data_free_route_is_unsupported(self):
        one = SymbolicAffineExpression.constant_value(1)
        contract = ParameterContract(
            (
                ParameterFact(
                    SymbolicPredicate(SymbolicAffineExpression.parameter("N"), ">=", one),
                    "declared",
                ),
            )
        )
        result = prove_abstract_equivalent(
            self.rs_left_source,
            self.rs_right_source,
            mode="auto",
            indexed_observable_mapping=_rs_indexed_mapping(),
            parameter_contract=contract,
        )

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.unresolved_obligations, ("indexed_observable_mapping",))

    def test_rs_grounded_projection_differentially_covers_small_horizons(self):
        for size in (2, 3):
            with self.subTest(size=size):
                data = _rs_data(size)
                left = linear_problem_from_opl(self.rs_left_source, data)
                right = linear_problem_from_opl(self.rs_right_source, data)
                result = prove_equivalent(
                    left,
                    right,
                    mode="projected_milp",
                    observable_mapping=_rs_mapping(size),
                    max_projected_assignments=64,
                )

                self.assertTrue(result.equivalent, result.reason)
                self.assertEqual(result.projected_assignment_counts, (2 ** (size - 1), 2 ** (size - 1)))

    def test_rs_uniform_route_does_not_certify_removed_flow_conservation(self):
        weakened_source = self.rs_right_source.replace(
            """  forall(t in 2..N)
    CycleFlow:
      sum(i in 1..t-1) x[i][t] == sum(j in t+1..N+1) x[t][j];

""",
            "",
        )
        result = prove_abstract_equivalent(
            self.rs_left_source,
            weakened_source,
            mode="auto",
            indexed_observable_mapping=_rs_indexed_mapping(),
            parameter_contract=ParameterContract(),
        )

        self.assertEqual(result.status, "unknown")

    def test_rs_uniform_route_does_not_certify_a_perturbed_loss_coefficient(self):
        perturbed_source = self.rs_right_source.replace(
            "- normalTail[k] * (q[i][j] - prefixMean[i-1] * x[i][j]);",
            "- 2 * normalTail[k] * (q[i][j] - prefixMean[i-1] * x[i][j]);",
        )
        result = prove_abstract_equivalent(
            self.rs_left_source,
            perturbed_source,
            mode="auto",
            indexed_observable_mapping=_rs_indexed_mapping(),
            parameter_contract=ParameterContract(),
        )

        self.assertEqual(result.status, "unknown")

    def test_rs_uniform_route_does_not_certify_a_perturbed_objective(self):
        perturbed_source = self.rs_right_source.replace(
            "setupCost * x[i][j]",
            "2 * setupCost * x[i][j]",
        )
        result = prove_abstract_equivalent(
            self.rs_left_source,
            perturbed_source,
            mode="auto",
            indexed_observable_mapping=_rs_indexed_mapping(),
            parameter_contract=ParameterContract(),
        )

        self.assertEqual(result.status, "unknown")

    @unittest.skip("expensive acceptance matrix; run explicitly for full 28-pair coverage")
    def test_all_tsp_formulations_share_off_diagonal_projected_value(self):
        for left_name, right_name in itertools.combinations(sorted(self.tsp_problems), 2):
            with self.subTest(left=left_name, right=right_name):
                result = prove_equivalent(
                    self.tsp_problems[left_name],
                    self.tsp_problems[right_name],
                    mode="projected_milp",
                    observable_mapping=self.tsp_mapping,
                    max_projected_assignments=256,
                )

                self.assertTrue(result.equivalent, result.reason)
                self.assertEqual(result.projected_assignment_counts, (120, 120))
                self.assertEqual(len(result.observable_mapping), 30)
                self.assertNotIn("x_1_1", " ".join(result.observable_mapping))

    @unittest.skip("expensive acceptance matrix; run explicitly for full 28-pair coverage")
    def test_all_tsp_formulations_share_off_diagonal_grounded_abstract_value(self):
        for left_name, right_name in itertools.combinations(sorted(self.tsp_sources), 2):
            with self.subTest(left=left_name, right=right_name):
                result = prove_abstract_equivalent(
                    self.tsp_sources[left_name],
                    self.tsp_sources[right_name],
                    mode="algebraic",
                    left_data_text=self.tsp_data,
                    right_data_text=self.tsp_data,
                    observable_templates=(self.tsp_template,),
                    max_projected_assignments=256,
                )

                self.assertTrue(result.equivalent, result.reason)
                self.assertEqual(result.projected_assignment_counts, (120, 120))
                self.assertEqual(len(result.observable_mapping), 30)
                self.assertNotIn("x_1_1", " ".join(result.observable_mapping))

    def test_tsp_template_expands_to_off_diagonal_arcs_only(self):
        mapping = self.tsp_template.expand()

        self.assertEqual(len(mapping), 30)
        self.assertEqual(
            {pair.name for pair in mapping},
            {f"x_{origin}_{destination}" for origin in range(1, 7) for destination in range(1, 7) if origin != destination},
        )
        self.assertNotIn("x_1_1", {pair.name for pair in mapping})

    def test_tsp_data_free_route_does_not_prove_formulation_projection(self):
        contract = ParameterContract(
            (
                ParameterFact(
                    SymbolicPredicate(
                        SymbolicAffineExpression.parameter("N"),
                        ">=",
                        SymbolicAffineExpression.constant_value(2),
                    ),
                    "declared",
                ),
            )
        )
        result = prove_abstract_equivalent(
            self.tsp_sources["C_conventional"],
            self.tsp_sources["F1_single_commodity"],
            mode="auto",
            indexed_observable_mapping=_tsp_indexed_mapping(),
            parameter_contract=contract,
        )

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.termination, "unsupported_input")
        self.assertEqual(result.scope, "source_schemas")
        self.assertIn("data-free", result.reason)

    def test_tsp_data_free_identity_mapping_is_unsupported(self):
        contract = ParameterContract(
            (
                ParameterFact(
                    SymbolicPredicate(
                        SymbolicAffineExpression.parameter("N"),
                        ">=",
                        SymbolicAffineExpression.constant_value(2),
                    ),
                    "declared",
                ),
            )
        )
        result = prove_abstract_equivalent(
            self.tsp_sources["C_conventional"],
            self.tsp_sources["C_conventional"],
            mode="auto",
            indexed_observable_mapping=_tsp_indexed_mapping(),
            parameter_contract=contract,
        )

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.unresolved_obligations, ("indexed_observable_mapping",))

    def test_tsp_data_free_certificate_route_is_unsupported(self):
        contract = ParameterContract(
            (
                ParameterFact(
                    SymbolicPredicate(
                        SymbolicAffineExpression.parameter("N"),
                        ">=",
                        SymbolicAffineExpression.constant_value(2),
                    ),
                    "declared",
                ),
            )
        )
        result = prove_abstract_equivalent(
            self.tsp_sources["C_conventional"],
            self.tsp_sources["T3_third_stage_dependent"],
            mode="auto",
            indexed_observable_mapping=_tsp_indexed_mapping(),
            parameter_contract=contract,
        )

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.unresolved_obligations, ("indexed_observable_mapping",))

    def test_tsp_data_free_mapping_is_unsupported_for_formulation_pairs(self):
        contract = ParameterContract(
            (
                ParameterFact(
                    SymbolicPredicate(
                        SymbolicAffineExpression.parameter("N"),
                        ">=",
                        SymbolicAffineExpression.constant_value(2),
                    ),
                    "declared",
                ),
            )
        )
        for left_name, right_name in itertools.combinations(sorted(self.tsp_sources), 2):
            with self.subTest(left=left_name, right=right_name):
                result = prove_abstract_equivalent(
                    self.tsp_sources[left_name],
                    self.tsp_sources[right_name],
                    mode="auto",
                    indexed_observable_mapping=_tsp_indexed_mapping(),
                    parameter_contract=contract,
                )

                self.assertEqual(result.status, "unknown")
                self.assertEqual(result.unresolved_obligations, ("indexed_observable_mapping",))

    def test_tsp_data_free_mapping_does_not_check_objective_perturbation(self):
        contract = ParameterContract(
            (
                ParameterFact(
                    SymbolicPredicate(
                        SymbolicAffineExpression.parameter("N"),
                        ">=",
                        SymbolicAffineExpression.constant_value(2),
                    ),
                    "declared",
                ),
            )
        )
        mutated = self.tsp_sources["C_conventional"].replace(
            "cost[i][j] * x[i][j]",
            "2 * cost[i][j] * x[i][j]",
        )
        self.assertNotEqual(mutated, self.tsp_sources["C_conventional"])
        result = prove_abstract_equivalent(
            self.tsp_sources["C_conventional"],
            mutated,
            mode="auto",
            indexed_observable_mapping=_tsp_indexed_mapping(),
            parameter_contract=contract,
        )

        self.assertEqual(result.status, "unknown")

    def test_tsp_grounded_projection_differentially_covers_small_horizons(self):
        for size in (3, 4):
            data = _tsp_data(size)
            problems = {name: linear_problem_from_opl(source, data) for name, source in self.tsp_sources.items()}
            pairs = (
                list(itertools.combinations(sorted(problems), 2)) if size == 3 else [("C_conventional", "F1_single_commodity")]
            )
            mapping = _identity_arc_mapping(size)
            for left_name, right_name in pairs:
                with self.subTest(size=size, left=left_name, right=right_name):
                    result = prove_equivalent(
                        problems[left_name],
                        problems[right_name],
                        mode="projected_milp",
                        observable_mapping=mapping,
                        max_projected_assignments=256,
                    )

                    self.assertTrue(result.equivalent, result.reason)
                    self.assertEqual(result.projected_assignment_counts, (factorial(size - 1), factorial(size - 1)))

    def test_tsp_representative_concrete_pair_remains_equivalent(self):
        result = prove_equivalent(
            self.tsp_problems["C_conventional"],
            self.tsp_problems["F1_single_commodity"],
            mode="projected_milp",
            observable_mapping=self.tsp_mapping,
            max_projected_assignments=256,
        )

        self.assertTrue(result.equivalent, result.reason)
        self.assertEqual(result.projected_assignment_counts, (120, 120))

    def test_tsp_representative_grounded_abstract_pair_remains_equivalent(self):
        result = prove_abstract_equivalent(
            self.tsp_sources["C_conventional"],
            self.tsp_sources["F1_single_commodity"],
            mode="algebraic",
            left_data_text=self.tsp_data,
            right_data_text=self.tsp_data,
            observable_templates=(self.tsp_template,),
            max_projected_assignments=256,
        )

        self.assertTrue(result.equivalent, result.reason)
        self.assertEqual(result.projected_assignment_counts, (120, 120))

    def test_tsp_staged_auxiliary_domains_remain_inside_complete_models(self):
        for name in ("T1_first_stage_dependent", "T2_second_stage_dependent", "T3_third_stage_dependent"):
            stage_integrality = [
                integrality
                for variable, integrality in zip(self.tsp_problems[name].var_names, self.tsp_problems[name].integrality)
                if variable.startswith("stageArc_")
            ]
            self.assertTrue(stage_integrality)
            if name == "T1_first_stage_dependent":
                self.assertTrue(all(value != 0 for value in stage_integrality))
            else:
                self.assertTrue(all(value == 0 for value in stage_integrality))

    def test_rs_objective_perturbation_returns_counterexample(self):
        perturbed = replace(
            self.rs_right,
            c=[
                coefficient + (1.0 if name == "x_1_2" else 0.0)
                for name, coefficient in zip(self.rs_right.var_names, self.rs_right.c)
            ],
        )

        result = prove_equivalent(
            self.rs_left,
            perturbed,
            mode="projected_milp",
            observable_mapping=self.rs_mapping,
            max_projected_assignments=32,
        )

        self.assertEqual(result.status, "different")
        self.assertIn("objective", result.reason)
        self.assertIn("order_", result.counterexample or "")

    def test_weakened_tsp_connectivity_returns_feasibility_counterexample(self):
        weakened = replace(self.tsp_problems["C_conventional"], A_ub=[], b_ub=[])

        result = prove_equivalent(
            weakened,
            self.tsp_problems["F1_single_commodity"],
            mode="projected_milp",
            observable_mapping=self.tsp_mapping,
            max_projected_assignments=256,
        )

        self.assertEqual(result.status, "different")
        self.assertIn("feasible in only one", result.reason)
        self.assertIn("x_", result.counterexample or "")

    def test_tsp_objective_perturbation_returns_counterexample(self):
        original = self.tsp_problems["C_conventional"]
        perturbed = replace(
            original,
            c=[coefficient + (1.0 if name == "x_1_2" else 0.0) for name, coefficient in zip(original.var_names, original.c)],
        )

        result = prove_equivalent(
            perturbed,
            self.tsp_problems["F1_single_commodity"],
            mode="projected_milp",
            observable_mapping=self.tsp_mapping,
            max_projected_assignments=256,
        )

        self.assertEqual(result.status, "different")
        self.assertIn("objective", result.reason)
        self.assertIn("x_1_2", result.counterexample or "")

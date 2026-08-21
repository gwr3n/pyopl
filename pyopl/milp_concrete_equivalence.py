"""MILP/LP equivalence checks inspired by the OptiBench evaluator.

OptiBench proposes evaluating generated optimization models by converting each
LP/MILP instance into an attributed bipartite graph and then checking whether
the generated model graph is equivalent to the reference model graph.  In that
representation, variable vertices carry variable attributes such as bounds,
objective coefficients, and type; constraint vertices carry constraint
attributes such as sense and right-hand side; and weighted edges carry matrix
coefficients.  Model equivalence is therefore invariant to variable names and to
row or column ordering, and can be studied as labelled graph isomorphism.

This module implements that core graph-equivalence idea for already parsed
``LinearProblem`` objects.  It first normalizes matrix data into a stable
internal representation, converts that representation into a labelled bipartite
row-column graph, and uses NetworkX labelled graph isomorphism to decide exact
structural equivalence.  The implementation also includes a Weisfeiler-Lehman-
style color-refinement pass, but uses it as a canonical ordering aid before the
exact isomorphism check, not as the final decision rule.

The code intentionally differs from the OptiBench algorithm in two ways.  First,
it does not implement OptiBench's benchmark-specific sufficient-condition test
based on WL-determinable and decomposable-symmetric instances; exact labelled
isomorphism is used instead.  Second, it adds presolve-aware normalization that
goes beyond plain graph relabelling, including finite-bound row normalization,
row scaling, duplicate-row removal, fixed-variable substitution,
slack-variable elimination, affine alias substitution, and solver-proven
redundant-row removal.

Reference:
    Zhuohan Wang, Ziwei Zhu, Yizhou Han, Yufeng Lin, Zhihang Lin, Ruoyu Sun,
    and Tian Ding. "OptiBench: Benchmarking Large Language Models in
    Optimization Modeling with Equivalence-Detection Evaluation." 2024.
    https://openreview.net/forum?id=KD9F5Ap878
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from pyopl.linear_problem import BoundValue, LinearProblem, Number, ObjectiveSense

EquivalenceStatus = Literal["equivalent", "different", "unknown"]
EquivalenceLevel = Literal["structural", "normalized", "solver_implied", "projection", "projected_milp_proven"]


@dataclass(frozen=True)
class EquivalenceResult:
    status: EquivalenceStatus
    level: EquivalenceLevel
    reason: str
    proof_steps: tuple[str, ...] = ()
    counterexample: str | None = None

    @property
    def equivalent(self) -> bool:
        return self.status == "equivalent"


@dataclass(frozen=True)
class _Row:
    sense: Literal["<=", "="]
    rhs: int
    entries: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Column:
    name: str
    objective: int
    integrality: int


@dataclass(frozen=True)
class _NormalizedProblem:
    objective_offset: int
    columns: tuple[_Column, ...]
    rows: tuple[_Row, ...]


class _LPRedundancyInputs(TypedDict):
    c: list[float]
    A_ub: list[list[float]] | None
    b_ub: list[float] | None
    A_eq: list[list[float]] | None
    b_eq: list[float] | None
    bounds: list[tuple[None, None]]
    method: Literal["highs"]


def compare(
    left: LinearProblem,
    right: LinearProblem,
    *,
    tolerance: float = 1e-9,
    max_iterations: int | None = None,
) -> bool:
    """Convenience wrapper for ``prove_equivalent`` that returns a boolean result."""

    return prove_equivalent(
        left,
        right,
        mode="auto",
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).equivalent


def prove_equivalent(
    left: LinearProblem,
    right: LinearProblem,
    *,
    mode: Literal["structural", "normalized", "solver", "projection", "projected_milp", "auto"] = "solver",
    variable_mapping: dict[str, str] | None = None,
    tolerance: float = 1e-9,
    max_iterations: int | None = None,
    max_projected_assignments: int = 100_000,
) -> EquivalenceResult:
    """Return a status-bearing equivalence result for two MILP matrix models.

    In ``"solver"`` mode, both models are validated, then canonicalized by
    eliminating simple affine aliases, explicit nonnegative slack variables, and
    fixed variables; normalizing objective sense, finite bounds, scaled rows,
    duplicate rows, and objective offsets; and removing inequality rows proven
    redundant by the SciPy LP/MILP solver.  The resulting matrix forms are
    converted into labelled bipartite row-column graphs and tested for labelled
    isomorphism.

    Variable names are ignored unless ``variable_mapping`` is supplied.  Column
    labels contain the normalized objective coefficient and integrality flag,
    while row labels contain the constraint sense and right-hand side.  The
    comparison is invariant to row order, column order, positive scaling of
    ``<=`` rows, nonzero scaling of equality rows, explicit finite bounds versus
    equivalent bound rows, fixed-variable substitution, simple affine-alias
    substitution, explicit slack-variable reformulations, solver-proven
    redundant inequalities, and the usual conversion from maximization to
    minimization.

    Numeric values are quantized according to ``tolerance`` before graph
    construction.  Values whose absolute value is at most ``tolerance`` are
    treated as zero; all other values are rounded to integer multiples of the
    tolerance.  ``max_iterations`` limits the color-refinement prepass used to
    order rows and columns before graph construction.  If omitted, the limit is
    based on the total number of rows and columns.

    ``mode="projection"`` first drops unmapped continuous auxiliary variables
    that have zero objective coefficient and do not appear in any row, then runs
    the solver-mode comparison under the supplied ``variable_mapping``.  The
    current implementation accepts ``"structural"`` and ``"normalized"`` modes
    as aliases of the same canonicalized graph comparison used by ``"solver"``.

    The returned ``EquivalenceResult`` has status ``"equivalent"`` only when
    the selected proof path establishes equivalence.  It has status
    ``"different"`` when the implemented normalization, mapping, or graph
    comparison finds a mismatch under that proof procedure; this should be read
    as different in the normalized comparison, not necessarily as a complete
    proof of mathematical non-equivalence for every possible reformulation.  It
    has status ``"unknown"`` when the requested mode cannot be applied, such as
    projection mode without a usable variable mapping.  The ``level``,
    ``reason``, ``proof_steps``, and ``counterexample`` fields describe how that
    status was reached.

    Raises:
        ValueError: If either model has inconsistent dimensions, invalid bounds,
            an unsupported objective sense, or a nonpositive tolerance.
    """

    if mode == "projected_milp":
        return _prove_projected_milp_equivalent(
            left,
            right,
            variable_mapping,
            tolerance,
            max_projected_assignments,
        )
    if mode == "projection":
        return _prove_projection_equivalent(
            left,
            right,
            variable_mapping,
            tolerance,
            max_iterations,
        )
    if mode not in {"structural", "normalized", "solver", "auto"}:
        return EquivalenceResult(
            status="unknown",
            level="projection",
            reason=f"unsupported equivalence mode: {mode}",
        )

    return _prove_normalized_equivalent(
        left,
        right,
        mode,
        variable_mapping,
        tolerance,
        max_iterations,
        max_projected_assignments,
    )


def _prove_projection_equivalent(
    left: LinearProblem,
    right: LinearProblem,
    variable_mapping: dict[str, str] | None,
    tolerance: float,
    max_iterations: int | None,
) -> EquivalenceResult:
    if variable_mapping is None:
        return EquivalenceResult(
            status="unknown",
            level="projection",
            reason="projection mode requires a user variable mapping",
        )
    projection = _project_to_mapping(left, right, variable_mapping, tolerance)
    if projection is None:
        return EquivalenceResult(
            status="unknown",
            level="projection",
            reason="projection mode only supports independent zero-objective auxiliary variables",
        )
    projected_left, projected_right, projection_steps = projection
    result = prove_equivalent(
        projected_left,
        projected_right,
        mode="solver",
        variable_mapping=variable_mapping,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return EquivalenceResult(
        status=result.status,
        level="projection",
        reason=result.reason,
        proof_steps=projection_steps + result.proof_steps,
        counterexample=result.counterexample,
    )


def _prove_normalized_equivalent(
    left: LinearProblem,
    right: LinearProblem,
    mode: Literal["structural", "normalized", "solver", "auto"],
    variable_mapping: dict[str, str] | None,
    tolerance: float,
    max_iterations: int | None,
    max_projected_assignments: int,
) -> EquivalenceResult:
    left_normalized = _canonicalize(left, tolerance, max_iterations)
    right_normalized = _canonicalize(right, tolerance, max_iterations)
    mapping_issue = _mapping_issue(left_normalized, right_normalized, variable_mapping)
    if mapping_issue is not None:
        return EquivalenceResult(
            status="different",
            level="solver_implied",
            reason=mapping_issue,
            proof_steps=("normalized both models", "checked user variable mapping"),
            counterexample=mapping_issue,
        )
    proof_steps: tuple[str, ...] = (
        "normalized both models",
        "removed duplicate and solver-proven redundant rows",
    )
    if left_normalized.objective_offset != right_normalized.objective_offset:
        if mode == "auto":
            return _prove_projected_milp_equivalent(left, right, variable_mapping, tolerance, max_projected_assignments)
        return EquivalenceResult(
            status="different",
            level="solver_implied",
            reason="normalized objective offsets differ",
            proof_steps=proof_steps,
            counterexample="normalized objective offsets differ",
        )
    if len(left_normalized.columns) != len(right_normalized.columns):
        if mode == "auto":
            return _prove_projected_milp_equivalent(left, right, variable_mapping, tolerance, max_projected_assignments)
        return EquivalenceResult(
            status="different",
            level="solver_implied",
            reason="normalized column counts differ",
            proof_steps=proof_steps,
            counterexample="normalized column counts differ",
        )
    if len(left_normalized.rows) != len(right_normalized.rows):
        if mode == "auto":
            return _prove_projected_milp_equivalent(left, right, variable_mapping, tolerance, max_projected_assignments)
        return EquivalenceResult(
            status="different",
            level="solver_implied",
            reason="normalized row counts differ",
            proof_steps=proof_steps,
            counterexample="normalized row counts differ",
        )

    left_graph = _to_graph(left_normalized)
    right_graph = _to_graph(right_normalized)
    matcher = isomorphism.GraphMatcher(
        left_graph,
        right_graph,
        node_match=isomorphism.categorical_node_match(["kind", "attributes"], [None, None]),
        edge_match=isomorphism.categorical_edge_match("coefficient", None),
    )
    proof_steps = proof_steps + ("tested labelled graph isomorphism",)
    if matcher.is_isomorphic():
        if variable_mapping is not None and not _has_mapped_isomorphism(
            matcher, left_normalized, right_normalized, variable_mapping
        ):
            return EquivalenceResult(
                status="different",
                level="solver_implied",
                reason="normalized graphs are isomorphic, but not under the user variable mapping",
                proof_steps=proof_steps,
                counterexample="user variable mapping is incompatible with normalized graph isomorphism",
            )
        return EquivalenceResult(
            status="equivalent",
            level="solver_implied",
            reason="normalized graphs are isomorphic",
            proof_steps=proof_steps,
        )
    if mode == "auto":
        return _prove_projected_milp_equivalent(left, right, variable_mapping, tolerance, max_projected_assignments)
    return EquivalenceResult(
        status="different",
        level="solver_implied",
        reason="normalized graphs are not isomorphic",
        proof_steps=proof_steps,
        counterexample="normalized graphs are not isomorphic",
    )


def _prove_projected_milp_equivalent(
    left: LinearProblem,
    right: LinearProblem,
    variable_mapping: dict[str, str] | None,
    tolerance: float,
    max_assignments: int,
) -> EquivalenceResult:
    _validate(left, tolerance)
    _validate(right, tolerance)
    if max_assignments <= 0:
        return EquivalenceResult(
            status="unknown",
            level="projected_milp_proven",
            reason="max_projected_assignments must be positive",
        )

    mapping = variable_mapping or _infer_shared_binary_mapping(left, right, tolerance)
    issue = _projected_mapping_issue(left, right, mapping, tolerance)
    if issue is not None:
        return EquivalenceResult(
            status="unknown",
            level="projected_milp_proven",
            reason=issue,
        )
    objective_issue = _projected_objective_issue(left, right, mapping, tolerance)
    if objective_issue is not None:
        return EquivalenceResult(
            status="different",
            level="projected_milp_proven",
            reason=objective_issue,
            counterexample=objective_issue,
        )

    checked = 0
    for source, target, source_to_target in (
        (left, right, mapping),
        (right, left, {right_name: left_name for left_name, right_name in mapping.items()}),
    ):
        direction = _find_projected_counterexample(
            source,
            target,
            source_to_target,
            tolerance,
            max_assignments - checked,
        )
        checked += direction[1]
        if direction[0] == "limit":
            return EquivalenceResult(
                status="unknown",
                level="projected_milp_proven",
                reason=f"projected MILP enumeration exceeded {max_assignments} shared assignments",
                proof_steps=("enumerated feasible shared binary assignments with no-good cuts",),
            )
        if direction[0] == "solver":
            return EquivalenceResult(
                status="unknown",
                level="projected_milp_proven",
                reason="projected MILP solver did not return an optimal or infeasible status",
                proof_steps=("enumerated feasible shared binary assignments with no-good cuts",),
            )
        if direction[0] == "counterexample":
            assignment = direction[2] or {}
            rendered = ", ".join(f"{name}={value}" for name, value in sorted(assignment.items()))
            return EquivalenceResult(
                status="different",
                level="projected_milp_proven",
                reason="a shared binary assignment is feasible in only one formulation",
                proof_steps=(
                    "enumerated feasible shared binary assignments with no-good cuts",
                    "checked continuous auxiliary feasibility in the other formulation",
                ),
                counterexample=rendered,
            )

    return EquivalenceResult(
        status="equivalent",
        level="projected_milp_proven",
        reason=f"all {checked} directed feasible shared assignments agree after auxiliary projection",
        proof_steps=(
            "validated shared binary variables and objective coefficients",
            "enumerated feasible shared binary assignments with no-good cuts in both directions",
            "checked continuous auxiliary feasibility in the other formulation",
        ),
    )


def _infer_shared_binary_mapping(
    left: LinearProblem,
    right: LinearProblem,
    tolerance: float,
) -> dict[str, str]:
    right_index = {name: index for index, name in enumerate(right.var_names)}
    mapping: dict[str, str] = {}
    for left_index, name in enumerate(left.var_names):
        candidate = right_index.get(name)
        if candidate is None:
            continue
        if _is_binary_column(left, left_index, tolerance) and _is_binary_column(right, candidate, tolerance):
            mapping[name] = name
    return mapping


def _is_binary_column(problem: LinearProblem, index: int, tolerance: float) -> bool:
    lower, upper = problem.bounds[index]
    return (
        problem.integrality[index] != 0
        and lower is not None
        and upper is not None
        and float(lower) >= -tolerance
        and float(upper) <= 1.0 + tolerance
        and abs(float(lower) - round(float(lower))) <= tolerance
        and abs(float(upper) - round(float(upper))) <= tolerance
    )


def _projected_mapping_issue(
    left: LinearProblem,
    right: LinearProblem,
    mapping: dict[str, str],
    tolerance: float,
) -> str | None:
    if not mapping:
        return "projected MILP comparison requires at least one shared binary variable"
    left_index = {name: index for index, name in enumerate(left.var_names)}
    right_index = {name: index for index, name in enumerate(right.var_names)}
    if len(set(mapping.values())) != len(mapping):
        return "projected MILP variable mapping must be injective"
    for left_name, right_name in mapping.items():
        if left_name not in left_index or right_name not in right_index:
            return "projected MILP variable mapping references an unknown variable"
        if not _is_binary_column(left, left_index[left_name], tolerance) or not _is_binary_column(
            right, right_index[right_name], tolerance
        ):
            return "projected MILP comparison currently supports shared binary variables only"
    mapped_left = set(mapping)
    mapped_right = set(mapping.values())
    for problem, mapped in ((left, mapped_left), (right, mapped_right)):
        for index, name in enumerate(problem.var_names):
            if name in mapped:
                continue
            if problem.integrality[index] != 0:
                return "projected MILP comparison requires all unmapped auxiliaries to be continuous"
            if abs(float(problem.c[index])) > tolerance:
                return "projected MILP comparison requires zero-objective auxiliary variables"
    return None


def _projected_objective_issue(
    left: LinearProblem,
    right: LinearProblem,
    mapping: dict[str, str],
    tolerance: float,
) -> str | None:
    left_sign = -1.0 if left.sense == "maximize" else 1.0
    right_sign = -1.0 if right.sense == "maximize" else 1.0
    if abs(left_sign * float(left.objective_offset) - right_sign * float(right.objective_offset)) > tolerance:
        return "projected objective offsets differ"
    left_index = {name: index for index, name in enumerate(left.var_names)}
    right_index = {name: index for index, name in enumerate(right.var_names)}
    for left_name, right_name in mapping.items():
        left_coefficient = left_sign * float(left.c[left_index[left_name]])
        right_coefficient = right_sign * float(right.c[right_index[right_name]])
        if abs(left_coefficient - right_coefficient) > tolerance:
            return f"projected objective coefficients differ for {left_name} and {right_name}"
    return None


def _find_projected_counterexample(
    source: LinearProblem,
    target: LinearProblem,
    mapping: dict[str, str],
    tolerance: float,
    remaining: int,
) -> tuple[Literal["complete", "counterexample", "limit", "solver"], int, dict[str, int] | None]:
    source_indices = {name: source.var_names.index(name) for name in mapping}
    cuts: list[LinearConstraint] = []
    checked = 0
    while True:
        if checked >= remaining:
            return "limit", checked, None
        result = milp(
            c=np.zeros(len(source.var_names)),
            integrality=np.asarray(source.integrality, dtype=np.int8),
            bounds=_problem_bounds(source),
            constraints=[*_problem_constraints(source), *cuts],
        )
        if result.status == 2:
            return "complete", checked, None
        if result.status != 0 or result.x is None:
            return "solver", checked, None
        assignment = {name: int(round(float(result.x[index]))) for name, index in source_indices.items()}
        checked += 1
        target_assignment = {mapping[name]: value for name, value in assignment.items()}
        feasibility = _fixed_assignment_feasibility(target, target_assignment)
        if feasibility == "infeasible":
            return "counterexample", checked, assignment
        if feasibility != "feasible":
            return "solver", checked, None
        cuts.append(_no_good_cut(source, assignment, source_indices))


def _problem_bounds(problem: LinearProblem) -> Bounds:
    return Bounds(
        lb=np.asarray([-float("inf") if bounds[0] is None else float(bounds[0]) for bounds in problem.bounds]),
        ub=np.asarray([float("inf") if bounds[1] is None else float(bounds[1]) for bounds in problem.bounds]),
    )


def _problem_constraints(problem: LinearProblem) -> list[LinearConstraint]:
    constraints: list[LinearConstraint] = []
    if problem.A_eq:
        rhs = np.asarray(problem.b_eq, dtype=float)
        constraints.append(LinearConstraint(np.asarray(problem.A_eq, dtype=float), rhs, rhs))
    if problem.A_ub:
        constraints.append(
            LinearConstraint(
                np.asarray(problem.A_ub, dtype=float),
                np.full(len(problem.A_ub), -float("inf")),
                np.asarray(problem.b_ub, dtype=float),
            )
        )
    return constraints


def _fixed_assignment_feasibility(
    problem: LinearProblem,
    assignment: dict[str, int],
) -> Literal["feasible", "infeasible", "unknown"]:
    bounds = _problem_bounds(problem)
    lower = np.array(bounds.lb, copy=True)
    upper = np.array(bounds.ub, copy=True)
    index_by_name = {name: index for index, name in enumerate(problem.var_names)}
    for name, value in assignment.items():
        index = index_by_name[name]
        lower[index] = value
        upper[index] = value
    result = milp(
        c=np.zeros(len(problem.var_names)),
        integrality=np.asarray(problem.integrality, dtype=np.int8),
        bounds=Bounds(lower, upper),
        constraints=_problem_constraints(problem),
    )
    if result.status == 0:
        return "feasible"
    if result.status == 2:
        return "infeasible"
    return "unknown"


def _no_good_cut(
    problem: LinearProblem,
    assignment: dict[str, int],
    indices: dict[str, int],
) -> LinearConstraint:
    coefficients = np.zeros(len(problem.var_names))
    one_count = 0
    for name, value in assignment.items():
        if value == 1:
            coefficients[indices[name]] = 1.0
            one_count += 1
        else:
            coefficients[indices[name]] = -1.0
    return LinearConstraint(coefficients, -float("inf"), float(one_count - 1))


def _to_graph(problem: _NormalizedProblem) -> nx.Graph:
    graph = nx.Graph()
    for column_index, column in enumerate(problem.columns):
        graph.add_node(
            ("column", column_index),
            kind="column",
            attributes=_column_local_key(column),
        )
    for row_index, row in enumerate(problem.rows):
        row_node = ("row", row_index)
        graph.add_node(row_node, kind="row", attributes=_row_local_key(row))
        for column_index, coefficient in row.entries:
            graph.add_edge(
                row_node,
                ("column", column_index),
                coefficient=coefficient,
            )
    return graph


def _project_to_mapping(
    left: LinearProblem,
    right: LinearProblem,
    variable_mapping: dict[str, str],
    tolerance: float,
) -> tuple[LinearProblem, LinearProblem, tuple[str, ...]] | None:
    left_projected = _drop_independent_auxiliaries(
        left,
        set(variable_mapping.keys()),
        tolerance,
    )
    right_projected = _drop_independent_auxiliaries(
        right,
        set(variable_mapping.values()),
        tolerance,
    )
    if left_projected is None or right_projected is None:
        return None
    return (
        left_projected,
        right_projected,
        ("projected unmapped auxiliary variables",),
    )


def _drop_independent_auxiliaries(problem: LinearProblem, kept_names: set[str], tolerance: float) -> LinearProblem | None:
    kept_indices = [index for index, name in enumerate(problem.var_names) if name in kept_names]
    if len(kept_indices) != len(kept_names):
        return None

    auxiliary_indices = [index for index, name in enumerate(problem.var_names) if name not in kept_names]
    if not _are_independent_auxiliaries(problem, auxiliary_indices, tolerance):
        return None

    return LinearProblem(
        sense=problem.sense,
        var_names=[problem.var_names[index] for index in kept_indices],
        bounds=[problem.bounds[index] for index in kept_indices],
        integrality=[problem.integrality[index] for index in kept_indices],
        c=[problem.c[index] for index in kept_indices],
        A_eq=[_remove_columns(row, kept_indices) for row in problem.A_eq],
        b_eq=problem.b_eq[:],
        A_ub=[_remove_columns(row, kept_indices) for row in problem.A_ub],
        b_ub=problem.b_ub[:],
        objective_offset=problem.objective_offset,
    )


def _are_independent_auxiliaries(
    problem: LinearProblem,
    auxiliary_indices: list[int],
    tolerance: float,
) -> bool:
    for index in auxiliary_indices:
        if abs(float(problem.c[index])) > tolerance or problem.integrality[index] != 0:
            return False
        if any(abs(float(row[index])) > tolerance for row in problem.A_eq + problem.A_ub):
            return False
    return True


def _mapping_issue(
    left: _NormalizedProblem,
    right: _NormalizedProblem,
    variable_mapping: dict[str, str] | None,
) -> str | None:
    if variable_mapping is None:
        return None
    left_names = {column.name for column in left.columns}
    right_names = {column.name for column in right.columns}
    for left_name, right_name in variable_mapping.items():
        if left_name not in left_names:
            return f"user variable mapping references unknown left variable: {left_name}"
        if right_name not in right_names:
            return f"user variable mapping references unknown right variable: {right_name}"
    if len(set(variable_mapping.values())) != len(variable_mapping):
        return "user variable mapping maps multiple left variables to the same right variable"
    return None


def _has_mapped_isomorphism(
    matcher: isomorphism.GraphMatcher,
    left: _NormalizedProblem,
    right: _NormalizedProblem,
    variable_mapping: dict[str, str],
) -> bool:
    left_name_by_index = {column_index: column.name for column_index, column in enumerate(left.columns)}
    right_index_by_name = {column.name: column_index for column_index, column in enumerate(right.columns)}
    required_node_mapping = {
        ("column", left_index): ("column", right_index_by_name[right_name])
        for left_index, left_name in left_name_by_index.items()
        if (right_name := variable_mapping.get(left_name)) is not None
    }
    return any(
        all(mapping[left_node] == right_node for left_node, right_node in required_node_mapping.items())
        for mapping in matcher.isomorphisms_iter()
    )


def _canonicalize(problem: LinearProblem, tolerance: float, max_iterations: int | None) -> _NormalizedProblem:
    _validate(problem, tolerance)
    problem = _eliminate_affine_aliases(problem, tolerance)
    problem = _eliminate_slack_variables(problem, tolerance)
    problem = _eliminate_fixed_variables(problem, tolerance)
    objective_sign = -1.0 if problem.sense == "maximize" else 1.0
    columns = tuple(
        _Column(
            name=var_name,
            objective=_quantize(objective_sign * objective, tolerance),
            integrality=int(integrality),
        )
        for var_name, objective, integrality in zip(problem.var_names, problem.c, problem.integrality, strict=True)
    )
    rows = (
        tuple(_normalize_row("=", row, rhs, tolerance) for row, rhs in zip(problem.A_eq, problem.b_eq, strict=True))
        + tuple(_normalize_row("<=", row, rhs, tolerance) for row, rhs in zip(problem.A_ub, problem.b_ub, strict=True))
        + _bound_rows(problem.bounds, len(columns), tolerance)
    )
    rows = _deduplicate_rows(rows)
    rows = _remove_redundant_rows(rows, columns, tolerance)

    row_colors, column_colors = _refine_colors(rows, columns, max_iterations)
    row_order = _row_order(rows, row_colors, column_colors)
    column_order = _column_order(rows, columns, row_colors, column_colors)
    remapped_column_index = {original_index: canonical_index for canonical_index, original_index in enumerate(column_order)}

    canonical_columns = tuple(columns[index] for index in column_order)
    canonical_rows = tuple(
        _Row(
            sense=rows[row_index].sense,
            rhs=rows[row_index].rhs,
            entries=tuple(
                sorted(
                    (remapped_column_index[column_index], coefficient) for column_index, coefficient in rows[row_index].entries
                )
            ),
        )
        for row_index in row_order
    )
    return _NormalizedProblem(
        objective_offset=_quantize(objective_sign * problem.objective_offset, tolerance),
        columns=canonical_columns,
        rows=canonical_rows,
    )


def _eliminate_affine_aliases(problem: LinearProblem, tolerance: float) -> LinearProblem:
    current = problem
    while True:
        alias = _find_affine_alias(current, tolerance)
        if alias is None:
            return current
        current = _substitute_affine_alias(current, alias)


def _find_affine_alias(problem: LinearProblem, tolerance: float) -> tuple[int, int] | None:
    for column_index, (bounds, integrality) in enumerate(zip(problem.bounds, problem.integrality, strict=True)):
        if integrality != 0 or bounds != [None, None]:
            continue
        equality_rows = [row_index for row_index, row in enumerate(problem.A_eq) if abs(float(row[column_index])) > tolerance]
        if len(equality_rows) != 1:
            continue
        row_index = equality_rows[0]
        if abs(float(problem.A_eq[row_index][column_index]) - 1.0) <= tolerance:
            return column_index, row_index
    return None


def _substitute_affine_alias(problem: LinearProblem, alias: tuple[int, int]) -> LinearProblem:
    alias_index, alias_row_index = alias
    alias_row = problem.A_eq[alias_row_index]
    alias_rhs = problem.b_eq[alias_row_index]
    kept_indices = [index for index in range(len(problem.var_names)) if index != alias_index]
    alias_coefficients = {index: -alias_row[index] for index in kept_indices}
    alias_constant = alias_rhs
    alias_objective = problem.c[alias_index]

    return LinearProblem(
        sense=problem.sense,
        var_names=[problem.var_names[index] for index in kept_indices],
        bounds=[problem.bounds[index] for index in kept_indices],
        integrality=[problem.integrality[index] for index in kept_indices],
        c=[problem.c[index] + alias_objective * alias_coefficients[index] for index in kept_indices],
        A_eq=[
            _substitute_alias_row(row, alias_index, alias_coefficients, kept_indices)
            for row_index, row in enumerate(problem.A_eq)
            if row_index != alias_row_index
        ],
        b_eq=[
            _substitute_alias_rhs(row, rhs, alias_index, alias_constant)
            for row_index, (row, rhs) in enumerate(zip(problem.A_eq, problem.b_eq, strict=True))
            if row_index != alias_row_index
        ],
        A_ub=[_substitute_alias_row(row, alias_index, alias_coefficients, kept_indices) for row in problem.A_ub],
        b_ub=[
            _substitute_alias_rhs(row, rhs, alias_index, alias_constant)
            for row, rhs in zip(problem.A_ub, problem.b_ub, strict=True)
        ],
        objective_offset=problem.objective_offset + alias_objective * alias_constant,
    )


def _substitute_alias_row(
    row: list[float],
    alias_index: int,
    alias_coefficients: dict[int, float],
    kept_indices: list[int],
) -> list[float]:
    alias_multiplier = row[alias_index]
    return [row[index] + alias_multiplier * alias_coefficients[index] for index in kept_indices]


def _substitute_alias_rhs(row: list[float], rhs: float, alias_index: int, alias_constant: float) -> float:
    return rhs - row[alias_index] * alias_constant


def _eliminate_slack_variables(problem: LinearProblem, tolerance: float) -> LinearProblem:
    slack_columns = _find_slack_columns(problem, tolerance)
    if not slack_columns:
        return problem

    removed_columns = set(slack_columns)
    removed_equalities = set(slack_columns.values())
    kept_indices = [column_index for column_index in range(len(problem.var_names)) if column_index not in removed_columns]
    added_ub_rows = [_remove_columns(problem.A_eq[row_index], kept_indices) for row_index in removed_equalities]
    added_ub_rhs = [problem.b_eq[row_index] for row_index in removed_equalities]
    return LinearProblem(
        sense=problem.sense,
        var_names=[problem.var_names[index] for index in kept_indices],
        bounds=[problem.bounds[index] for index in kept_indices],
        integrality=[problem.integrality[index] for index in kept_indices],
        c=[problem.c[index] for index in kept_indices],
        A_eq=[
            _remove_columns(row, kept_indices)
            for row_index, row in enumerate(problem.A_eq)
            if row_index not in removed_equalities
        ],
        b_eq=[rhs for row_index, rhs in enumerate(problem.b_eq) if row_index not in removed_equalities],
        A_ub=[_remove_columns(row, kept_indices) for row in problem.A_ub] + added_ub_rows,
        b_ub=[*problem.b_ub, *added_ub_rhs],
        objective_offset=problem.objective_offset,
    )


def _find_slack_columns(problem: LinearProblem, tolerance: float) -> dict[int, int]:
    slack_columns: dict[int, int] = {}
    for column_index, (objective, bounds, integrality) in enumerate(
        zip(problem.c, problem.bounds, problem.integrality, strict=True)
    ):
        lower_bound, upper_bound = bounds
        if (
            abs(float(objective)) > tolerance
            or integrality != 0
            or lower_bound is None
            or abs(float(lower_bound)) > tolerance
            or upper_bound is not None
        ):
            continue
        equality_rows = [row_index for row_index, row in enumerate(problem.A_eq) if abs(float(row[column_index])) > tolerance]
        if len(equality_rows) != 1:
            continue
        row_index = equality_rows[0]
        if abs(float(problem.A_eq[row_index][column_index]) - 1.0) > tolerance:
            continue
        if any(abs(float(row[column_index])) > tolerance for row in problem.A_ub):
            continue
        slack_columns[column_index] = row_index
    return _unique_value_items(slack_columns)


def _unique_value_items(items: dict[int, int]) -> dict[int, int]:
    value_counts: dict[int, int] = {}
    for value in items.values():
        value_counts[value] = value_counts.get(value, 0) + 1
    return {key: value for key, value in items.items() if value_counts[value] == 1}


def _remove_columns(row: list[float], kept_indices: list[int]) -> list[float]:
    return [row[index] for index in kept_indices]


def _eliminate_fixed_variables(problem: LinearProblem, tolerance: float) -> LinearProblem:
    fixed_values: dict[int, float] = {}
    kept_indices: list[int] = []
    for column_index, (lower_bound, upper_bound) in enumerate(problem.bounds):
        if lower_bound is not None and upper_bound is not None and abs(float(lower_bound) - float(upper_bound)) <= tolerance:
            fixed_values[column_index] = float(lower_bound)
        else:
            kept_indices.append(column_index)

    if not fixed_values:
        return problem

    objective_offset = problem.objective_offset + sum(
        problem.c[column_index] * value for column_index, value in fixed_values.items()
    )
    return LinearProblem(
        sense=problem.sense,
        var_names=[problem.var_names[index] for index in kept_indices],
        bounds=[problem.bounds[index] for index in kept_indices],
        integrality=[problem.integrality[index] for index in kept_indices],
        c=[problem.c[index] for index in kept_indices],
        A_eq=[_substitute_fixed_row(row, fixed_values, kept_indices) for row in problem.A_eq],
        b_eq=[_substitute_fixed_rhs(row, rhs, fixed_values) for row, rhs in zip(problem.A_eq, problem.b_eq, strict=True)],
        A_ub=[_substitute_fixed_row(row, fixed_values, kept_indices) for row in problem.A_ub],
        b_ub=[_substitute_fixed_rhs(row, rhs, fixed_values) for row, rhs in zip(problem.A_ub, problem.b_ub, strict=True)],
        objective_offset=objective_offset,
    )


def _substitute_fixed_row(row: list[float], fixed_values: dict[int, float], kept_indices: list[int]) -> list[float]:
    return [row[index] for index in kept_indices]


def _substitute_fixed_rhs(row: list[float], rhs: float, fixed_values: dict[int, float]) -> float:
    return rhs - sum(row[column_index] * value for column_index, value in fixed_values.items())


def _validate(problem: LinearProblem, tolerance: float) -> None:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if problem.sense not in {"minimize", "maximize"}:
        raise ValueError("sense must be 'minimize' or 'maximize'")

    variable_count = len(problem.var_names)
    if len(problem.bounds) != variable_count:
        raise ValueError("bounds length must match var_names length")
    if len(problem.integrality) != variable_count:
        raise ValueError("integrality length must match var_names length")
    if len(problem.c) != variable_count:
        raise ValueError("c length must match var_names length")

    for index, bounds in enumerate(problem.bounds):
        if len(bounds) != 2:
            raise ValueError(f"bounds[{index}] must contain lower and upper bounds")
        lower_bound, upper_bound = bounds
        if lower_bound is not None and upper_bound is not None and float(lower_bound) > float(upper_bound) + tolerance:
            raise ValueError(f"bounds[{index}] lower bound exceeds upper bound")

    _validate_matrix("A_eq", problem.A_eq, variable_count)
    _validate_matrix("A_ub", problem.A_ub, variable_count)
    if len(problem.A_eq) != len(problem.b_eq):
        raise ValueError("A_eq row count must match b_eq length")
    if len(problem.A_ub) != len(problem.b_ub):
        raise ValueError("A_ub row count must match b_ub length")


def _validate_matrix(name: str, matrix: list[list[float]], variable_count: int) -> None:
    for row_index, row in enumerate(matrix):
        if len(row) != variable_count:
            raise ValueError(f"{name}[{row_index}] length must match var_names length")


def _normalize_row(sense: Literal["<=", "="], row: list[float], rhs: float, tolerance: float) -> _Row:
    scale = max((abs(float(value)) for value in row), default=0.0)
    if scale <= tolerance:
        scale = abs(float(rhs)) or 1.0

    scaled_values = [float(value) / scale for value in row]
    scaled_rhs = float(rhs) / scale
    if sense == "=" and _first_nonzero([*scaled_values, scaled_rhs], tolerance) < 0:
        scaled_values = [-value for value in scaled_values]
        scaled_rhs = -scaled_rhs

    entries = tuple(
        (column_index, coefficient)
        for column_index, value in enumerate(scaled_values)
        if (coefficient := _quantize(value, tolerance)) != 0
    )
    return _Row(sense=sense, rhs=_quantize(scaled_rhs, tolerance), entries=entries)


def _bound_rows(bounds: list[list[BoundValue]], variable_count: int, tolerance: float) -> tuple[_Row, ...]:
    rows: list[_Row] = []
    for column_index, (lower_bound, upper_bound) in enumerate(bounds):
        row = [0.0] * variable_count
        if lower_bound is not None:
            row[column_index] = -1.0
            rows.append(_normalize_row("<=", row, -float(lower_bound), tolerance))
            row[column_index] = 0.0
        if upper_bound is not None:
            row[column_index] = 1.0
            rows.append(_normalize_row("<=", row, float(upper_bound), tolerance))
    return tuple(rows)


def _deduplicate_rows(rows: tuple[_Row, ...]) -> tuple[_Row, ...]:
    return tuple(sorted(set(rows), key=_row_sort_key))


def _remove_redundant_rows(rows: tuple[_Row, ...], columns: tuple[_Column, ...], tolerance: float) -> tuple[_Row, ...]:
    kept_rows = list(rows)
    index = 0
    while index < len(kept_rows):
        row = kept_rows[index]
        if row.sense != "<=":
            index += 1
            continue
        other_rows = tuple(kept_rows[:index] + kept_rows[index + 1 :])
        if _is_redundant_row(row, other_rows, columns, tolerance):
            del kept_rows[index]
        else:
            index += 1
    return tuple(kept_rows)


def _is_redundant_row(
    row: _Row,
    other_rows: tuple[_Row, ...],
    columns: tuple[_Column, ...],
    tolerance: float,
) -> bool:
    if any(column.integrality != 0 for column in columns):
        return _is_milp_redundant_row(row, other_rows, columns, tolerance)
    return _is_lp_redundant_row(row, other_rows, len(columns), tolerance)


def _is_lp_redundant_row(
    row: _Row,
    other_rows: tuple[_Row, ...],
    variable_count: int,
    tolerance: float,
) -> bool:
    result = linprog(**_lp_redundancy_inputs(row, other_rows, variable_count, tolerance))
    if result.status != 0:
        return False
    if result.fun is None:
        return False
    return -float(result.fun) <= _row_rhs(row, tolerance) + tolerance


def _lp_redundancy_inputs(
    row: _Row,
    other_rows: tuple[_Row, ...],
    variable_count: int,
    tolerance: float,
) -> _LPRedundancyInputs:
    inequality_rows = [other_row for other_row in other_rows if other_row.sense == "<="]
    equality_rows = [other_row for other_row in other_rows if other_row.sense == "="]
    return {
        "c": [-coefficient for coefficient in _row_coefficients(row, variable_count, tolerance)],
        "A_ub": [_row_coefficients(other_row, variable_count, tolerance) for other_row in inequality_rows] or None,
        "b_ub": [_row_rhs(other_row, tolerance) for other_row in inequality_rows] or None,
        "A_eq": [_row_coefficients(other_row, variable_count, tolerance) for other_row in equality_rows] or None,
        "b_eq": [_row_rhs(other_row, tolerance) for other_row in equality_rows] or None,
        "bounds": [(None, None)] * variable_count,
        "method": "highs",
    }


def _is_milp_redundant_row(
    row: _Row,
    other_rows: tuple[_Row, ...],
    columns: tuple[_Column, ...],
    tolerance: float,
) -> bool:
    variable_count = len(columns)
    constraints = _linear_constraints(other_rows, variable_count, tolerance)
    objective = [-coefficient for coefficient in _row_coefficients(row, variable_count, tolerance)]
    result = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.asarray([column.integrality for column in columns], dtype=np.int8),
        bounds=Bounds(
            lb=np.full(variable_count, -float("inf")),
            ub=np.full(variable_count, float("inf")),
        ),
        constraints=constraints,
    )
    if result.status != 0:
        return False
    if result.fun is None:
        return False
    return -float(result.fun) <= _row_rhs(row, tolerance) + tolerance


def _linear_constraints(rows: tuple[_Row, ...], variable_count: int, tolerance: float) -> list[LinearConstraint]:
    constraints: list[LinearConstraint] = []
    for row in rows:
        coefficients = np.asarray(_row_coefficients(row, variable_count, tolerance), dtype=float)
        rhs = _row_rhs(row, tolerance)
        if row.sense == "<=":
            constraints.append(LinearConstraint(coefficients, -float("inf"), rhs))
        else:
            constraints.append(LinearConstraint(coefficients, rhs, rhs))
    return constraints


def _row_coefficients(row: _Row, variable_count: int, tolerance: float) -> list[float]:
    coefficients = [0.0] * variable_count
    for column_index, coefficient in row.entries:
        coefficients[column_index] = coefficient * tolerance
    return coefficients


def _row_rhs(row: _Row, tolerance: float) -> float:
    return row.rhs * tolerance


def _row_sort_key(row: _Row) -> tuple[str, int, tuple[tuple[int, int], ...]]:
    return row.sense, row.rhs, row.entries


def _first_nonzero(values: list[float], tolerance: float) -> float:
    for value in values:
        if abs(value) > tolerance:
            return value
    return 0.0


def _refine_colors(
    rows: tuple[_Row, ...], columns: tuple[_Column, ...], max_iterations: int | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    row_colors = _compact_colors(_row_local_key(row) for row in rows)
    column_colors = _compact_colors(_column_local_key(column) for column in columns)
    row_neighbors, column_neighbors = _neighbors(rows, len(columns))
    iteration_limit = max_iterations or (len(rows) + len(columns) + 2)

    for _ in range(iteration_limit):
        next_column_colors = _compact_colors(
            (
                _column_local_key(columns[column_index]),
                tuple(
                    sorted((coefficient, row_colors[row_index]) for row_index, coefficient in column_neighbors[column_index])
                ),
            )
            for column_index in range(len(columns))
        )
        next_row_colors = _compact_colors(
            (
                _row_local_key(rows[row_index]),
                tuple(
                    sorted(
                        (coefficient, next_column_colors[column_index])
                        for column_index, coefficient in row_neighbors[row_index]
                    )
                ),
            )
            for row_index in range(len(rows))
        )
        if next_row_colors == row_colors and next_column_colors == column_colors:
            break
        row_colors = next_row_colors
        column_colors = next_column_colors

    return row_colors, column_colors


def _neighbors(
    rows: tuple[_Row, ...], column_count: int
) -> tuple[list[tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]]:
    row_neighbors = [row.entries for row in rows]
    column_neighbors: list[list[tuple[int, int]]] = [[] for _ in range(column_count)]
    for row_index, row in enumerate(rows):
        for column_index, coefficient in row.entries:
            column_neighbors[column_index].append((row_index, coefficient))
    return row_neighbors, column_neighbors


def _row_order(rows: tuple[_Row, ...], row_colors: tuple[int, ...], column_colors: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(rows)),
            key=lambda row_index: (
                row_colors[row_index],
                _row_local_key(rows[row_index]),
                tuple(
                    sorted((coefficient, column_colors[column_index]) for column_index, coefficient in rows[row_index].entries)
                ),
            ),
        )
    )


def _column_order(
    rows: tuple[_Row, ...],
    columns: tuple[_Column, ...],
    row_colors: tuple[int, ...],
    column_colors: tuple[int, ...],
) -> tuple[int, ...]:
    _, column_neighbors = _neighbors(rows, len(columns))
    return tuple(
        sorted(
            range(len(columns)),
            key=lambda column_index: (
                column_colors[column_index],
                _column_local_key(columns[column_index]),
                tuple(
                    sorted((coefficient, row_colors[row_index]) for row_index, coefficient in column_neighbors[column_index])
                ),
            ),
        )
    )


def _compact_colors(keys) -> tuple[int, ...]:
    keys = tuple(keys)
    color_by_key = {key: color for color, key in enumerate(sorted(set(keys)))}
    return tuple(color_by_key[key] for key in keys)


def _row_local_key(row: _Row) -> tuple[str, int]:
    return row.sense, row.rhs


def _column_local_key(
    column: _Column,
) -> tuple[int, int]:
    return (
        column.objective,
        column.integrality,
    )


def _quantize(value: float, tolerance: float) -> int:
    if abs(value) <= tolerance:
        return 0
    return int(round(value / tolerance))


__all__ = [
    "BoundValue",
    "EquivalenceResult",
    "LinearProblem",
    "Number",
    "ObjectiveSense",
    "compare",
    "prove_equivalent",
]

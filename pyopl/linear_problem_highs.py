from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .linear_problem import LinearProblem

if TYPE_CHECKING:
    import highspy  # type: ignore[import-untyped]


def _import_highspy():
    try:
        import highspy
    except ImportError as exc:
        raise ImportError("Exporting LinearProblem through HiGHS requires the optional 'highspy' package.") from exc
    return highspy


def _is_ok(status: object) -> bool:
    return str(status) in {"HighsStatus.kOk", "HighsStatus.kWarning"}


def _require_ok(status: object, action: str) -> None:
    if not _is_ok(status):
        raise RuntimeError(f"HiGHS failed to {action}: {status}")


def _bound_value(value: float | int | None, infinity: float, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _unique_name(raw_name: str, used: set[str], fallback_prefix: str, index: int) -> str:
    name = str(raw_name).strip() or f"{fallback_prefix}{index}"
    candidate = name
    suffix = 1
    while candidate in used:
        candidate = f"{name}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def build_highs_model(
    problem: LinearProblem,
    *,
    objective_is_minimization_form: bool = True,
) -> "highspy.Highs":
    """Build a highspy.Highs model from a LinearProblem.

    Current SciPy-derived LinearProblem instances store ``c`` in minimization form,
    so maximization objectives have already been negated. Leave
    ``objective_is_minimization_form`` as True for those instances.
    """

    highspy = _import_highspy()
    highs = highspy.Highs()
    infinity = highs.getInfinity()

    costs = [float(value) for value in problem.c]
    if problem.sense == "maximize" and objective_is_minimization_form:
        costs = [-value for value in costs]

    lower_bounds = []
    upper_bounds = []
    for lower, upper in problem.bounds:
        lower_bounds.append(_bound_value(lower, infinity, default=-infinity))
        upper_bounds.append(_bound_value(upper, infinity, default=infinity))

    num_cols = len(problem.var_names)
    _require_ok(
        highs.addCols(
            num_cols,
            np.array(costs, dtype=np.float64),
            np.array(lower_bounds, dtype=np.float64),
            np.array(upper_bounds, dtype=np.float64),
            0,
            np.array([0] * (num_cols + 1), dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        ),
        "add columns",
    )

    used_col_names: set[str] = set()
    for col_idx, var_name in enumerate(problem.var_names):
        highs.passColName(col_idx, _unique_name(var_name, used_col_names, "c", col_idx))
        if col_idx < len(problem.integrality) and problem.integrality[col_idx]:
            _require_ok(
                highs.changeColIntegrality(col_idx, highspy.HighsVarType.kInteger),
                f"set integrality for column {col_idx}",
            )

    row_names: list[str] = []

    def add_row(row: list[float], lower: float, upper: float, row_name: str) -> None:
        indices = [idx for idx, coef in enumerate(row) if abs(coef) > 1e-12]
        values = [float(row[idx]) for idx in indices]
        _require_ok(
            highs.addRow(
                lower,
                upper,
                len(indices),
                np.array(indices, dtype=np.int32),
                np.array(values, dtype=np.float64),
            ),
            f"add row {row_name}",
        )
        row_names.append(row_name)

    for row_idx, (row, rhs) in enumerate(zip(problem.A_ub, problem.b_ub)):
        add_row(row, -infinity, float(rhs), f"ub_{row_idx}")
    for row_idx, (row, rhs) in enumerate(zip(problem.A_eq, problem.b_eq)):
        rhs_float = float(rhs)
        add_row(row, rhs_float, rhs_float, f"eq_{row_idx}")

    used_row_names: set[str] = set()
    for row_idx, row_name in enumerate(row_names):
        highs.passRowName(row_idx, _unique_name(row_name, used_row_names, "r", row_idx))

    if problem.sense == "maximize":
        _require_ok(highs.changeObjectiveSense(highspy.ObjSense.kMaximize), "set maximization sense")
    else:
        _require_ok(highs.changeObjectiveSense(highspy.ObjSense.kMinimize), "set minimization sense")

    return highs


def export_linear_problem(
    problem: LinearProblem,
    output_path: str | Path,
    *,
    objective_is_minimization_form: bool = True,
) -> Path:
    highs = build_highs_model(problem, objective_is_minimization_form=objective_is_minimization_form)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_ok(highs.writeModel(str(path)), f"write model to {path}")
    return path
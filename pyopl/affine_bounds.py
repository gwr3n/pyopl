import math
from collections.abc import Callable, Mapping

Interval = tuple[float, float]


def _finite(value: float, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def combine_intervals(left: Interval, right: Interval, operator: str) -> Interval:
    left_lower, left_upper = (_finite(value, "interval bound") for value in left)
    right_lower, right_upper = (_finite(value, "interval bound") for value in right)
    if operator == "+":
        return left_lower + right_lower, left_upper + right_upper
    if operator == "-":
        return left_lower - right_upper, left_upper - right_lower
    raise ValueError(f"Unsupported interval operator '{operator}'")


def scale_interval(interval: Interval, coefficient: float) -> Interval:
    lower, upper = (_finite(value, "interval bound") for value in interval)
    coefficient = _finite(coefficient, "coefficient")
    if coefficient >= 0:
        return coefficient * lower, coefficient * upper
    return coefficient * upper, coefficient * lower


def affine_interval(
    coefficients: Mapping[str, float],
    constant: float,
    bound_for: Callable[[str], Interval],
) -> Interval:
    interval = (_finite(constant, "constant"),) * 2
    for variable_name, coefficient in coefficients.items():
        interval = combine_intervals(interval, scale_interval(bound_for(variable_name), coefficient), "+")
    return interval

from dataclasses import dataclass

from .numerical_policy import EQUALITY_COMPARISON_TOLERANCE, STRICT_COMPARISON_EPSILON


@dataclass(frozen=True)
class ComparisonPolicy:
    strict_separation: float
    equality_tolerance: float = EQUALITY_COMPARISON_TOLERANCE


def comparison_policy(*, integer_valued: bool) -> ComparisonPolicy:
    return ComparisonPolicy(strict_separation=1.0 if integer_valued else STRICT_COMPARISON_EPSILON)

"""Public strategy-dispatch API for PyOPL model equivalence."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pyopl.milp_abstract_equivalence import AbstractEquivalenceResult, prove_abstract_equivalent
from pyopl.milp_concrete_equivalence import EquivalenceResult, prove_equivalent
from pyopl.pyopl_core import linear_problem_from_opl

ComparisonStrategy = Literal["concrete", "abstract"]
ModelComparisonResult: TypeAlias = EquivalenceResult | AbstractEquivalenceResult


def compare_models(
    left_model_text: str,
    right_model_text: str,
    *,
    strategy: str = "abstract",
    left_data_text: str | None = None,
    right_data_text: str | None = None,
) -> ModelComparisonResult:
    """Compare two PyOPL models using the selected equivalence strategy.

    ``strategy="concrete"`` instantiates each model with its optional data and
    compares the resulting matrix models. ``strategy="abstract"`` compares the
    model families before data materialization, using structural comparison
    followed by the supported algebraic proof backend. When data is supplied,
    it may also ground finite indexed schemas for instance-level algebraic proof.
    """

    if strategy == "concrete":
        left_problem = linear_problem_from_opl(left_model_text, left_data_text)
        right_problem = linear_problem_from_opl(right_model_text, right_data_text)
        return prove_equivalent(left_problem, right_problem, mode="auto")
    if strategy == "abstract":
        return prove_abstract_equivalent(
            left_model_text,
            right_model_text,
            mode="auto",
            left_data_text=left_data_text,
            right_data_text=right_data_text,
        )
    raise ValueError(f"unsupported model comparison strategy: {strategy}")


def comparison_result_to_dict(
    result: ModelComparisonResult,
    *,
    strategy: str,
) -> dict[str, object]:
    """Convert a comparison result to a JSON-friendly dictionary."""

    return {
        "strategy": strategy,
        "status": result.status,
        "equivalent": result.equivalent,
        "level": result.level,
        "reason": result.reason,
        "proof_steps": list(result.proof_steps),
        "counterexample": result.counterexample,
    }


__all__ = [
    "ComparisonStrategy",
    "ModelComparisonResult",
    "compare_models",
    "comparison_result_to_dict",
]

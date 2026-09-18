"""Public strategy-dispatch API for the paper's two-route proof procedure.

``strategy="abstract"`` enters the schema route of Section 6;
``strategy="concrete"`` enters the instantiated matrix route of Section 5.
Their common fallback methods and outcome semantics are assembled by Theorem
9.1 (Soundness of the layered exact procedure), with implementation result
labels defined in Section 9.2 of the attached paper.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Collection, Literal, Mapping, TypeAlias

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
    variable_mapping: Mapping[str, str] | None = None,
    parameter_mapping: Mapping[str, str] | None = None,
    left_auxiliaries: Collection[str] = (),
    right_auxiliaries: Collection[str] = (),
    assumptions: Mapping[str, str] | None = None,
) -> ModelComparisonResult:
    """Compare two PyOPL models using the selected equivalence strategy.

    ``strategy="concrete"`` instantiates each model with its optional data and
    compares the resulting matrix models. ``strategy="abstract"`` compares the
    model families before data materialization, using structural comparison
    followed by the supported algebraic proof backend. Supplying both data texts
    requests instance comparison, even when the source schemas are isomorphic.

    Variable mappings identify retained coordinates; auxiliary sets identify
    variables to hide. The abstract route also accepts parameter mappings and
    positive/nonnegative/nonzero assumptions on parameter names shared by both
    schemas. Those schema options cannot be combined with supplied data.
    The concrete route rejects schema options and requires explicit auxiliary
    sets, when provided, to complement the retained map exactly. Invalid maps
    and overlapping partitions raise ValueError. Indexed declaration maps are
    not expanded automatically during grounding.

    Results describe their relation, scope, arithmetic provenance, correspondence,
    and termination. Exact parsed-value or embedded-matrix arithmetic does not
    imply lossless source decimal ingestion. Concrete results are numerical and
    may be quantized. Evidence is an internal-check summary, not an exported
    independently replayable certificate.

    The abstract structural result is the sufficient condition of Theorem 6.4
    (Schema isomorphism is uniformly sound).  Concrete reduction and graph
    matching follow Theorem 5.14 (Exact concrete-path soundness).  If abstract
    lowering uses supplied data, Proposition 6.8 (Correct finite grounding)
    restricts that proof to those data rather than all admissible schema
    valuations.  The returned status preserves the ``equivalent``,
    ``different``, and ``unknown`` outcomes required by Theorem 9.1.
    """

    if strategy == "concrete":
        if parameter_mapping is not None or assumptions is not None:
            raise ValueError("concrete strategy does not accept parameter mappings or assumptions")
        left_problem = linear_problem_from_opl(left_model_text, left_data_text)
        right_problem = linear_problem_from_opl(right_model_text, right_data_text)
        if left_auxiliaries or right_auxiliaries:
            if variable_mapping is None:
                raise ValueError("concrete auxiliary partitions require a retained variable mapping")
            if set(left_auxiliaries) != set(left_problem.var_names) - set(variable_mapping) or set(right_auxiliaries) != set(
                right_problem.var_names
            ) - set(variable_mapping.values()):
                raise ValueError("concrete auxiliary partitions must exactly complement the retained mapping")
        result = prove_equivalent(
            left_problem,
            right_problem,
            mode="auto",
            variable_mapping=None if variable_mapping is None else dict(variable_mapping),
        )
        return replace(
            result,
            scope="supplied_instances" if left_data_text is not None or right_data_text is not None else "compiled_instances",
        )
    if strategy == "abstract":
        return prove_abstract_equivalent(
            left_model_text,
            right_model_text,
            mode="auto",
            left_data_text=left_data_text,
            right_data_text=right_data_text,
            variable_mapping=variable_mapping,
            parameter_mapping=parameter_mapping,
            left_auxiliaries=left_auxiliaries,
            right_auxiliaries=right_auxiliaries,
            assumptions=assumptions,
        )
    raise ValueError(f"unsupported model comparison strategy: {strategy}")


def comparison_result_to_dict(
    result: ModelComparisonResult,
    *,
    strategy: str,
) -> dict[str, object]:
    """Serialize the evidence summary prescribed by Theorem 9.1.

    The method level, proof steps, and counterexample keep a boolean answer
    tied to its proof route and evidence, as required by the reporting step in
    Section 9.
    """

    return {
        **asdict(result),
        "strategy": strategy,
        "status": result.status,
        "equivalent": result.equivalent,
        "level": result.level,
        "reason": result.reason,
        "proof_steps": list(result.proof_steps),
        "counterexample": result.counterexample,
        "variable_mapping": dict(result.variable_mapping),
        "parameter_mapping": dict(result.parameter_mapping),
        "assumptions": dict(result.assumptions),
        "left_auxiliaries": list(result.left_auxiliaries),
        "right_auxiliaries": list(result.right_auxiliaries),
    }


__all__ = [
    "ComparisonStrategy",
    "ModelComparisonResult",
    "compare_models",
    "comparison_result_to_dict",
]

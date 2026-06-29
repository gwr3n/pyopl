from dataclasses import dataclass
from typing import Literal

Number = int | float
BoundValue = Number | None
ObjectiveSense = Literal["minimize", "maximize"]


@dataclass
class LinearProblem:
    sense: ObjectiveSense
    var_names: list[str]
    bounds: list[list[BoundValue]]
    integrality: list[int]
    c: list[float]
    A_eq: list[list[float]]
    b_eq: list[float]
    A_ub: list[list[float]]
    b_ub: list[float]
    objective_offset: float = 0.0

from typing import Any


def boolean_not(node: dict[str, Any]) -> dict[str, Any]:
    return {"type": "not", "value": node, "sem_type": "boolean"}


def boolean_or(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"type": "or", "left": left, "right": right, "sem_type": "boolean"}


def lower_implication(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return boolean_or(boolean_not(left), right)

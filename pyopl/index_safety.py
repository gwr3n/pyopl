"""Conservative index-domain validation for abstract model schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class IndexSafetyIssue:
    status: str
    reason: str


@dataclass(frozen=True)
class _Affine:
    coefficients: tuple[tuple[str, int], ...]
    constant: int = 0

    def add(self, other: _Affine, scale: int = 1) -> _Affine:
        coefficients = dict(self.coefficients)
        for name, value in other.coefficients:
            coefficients[name] = coefficients.get(name, 0) + scale * value
        return _Affine(
            tuple(sorted((name, value) for name, value in coefficients.items() if value)),
            self.constant + scale * other.constant,
        )

    def scale(self, factor: int) -> _Affine:
        return _Affine(tuple((name, factor * value) for name, value in self.coefficients), factor * self.constant)


@dataclass(frozen=True)
class _Interval:
    lower: _Affine
    upper: _Affine


def find_index_safety_issue(ast: Mapping[str, Any]) -> IndexSafetyIssue | None:
    """Return a definite index error before any unresolved safety obligation."""
    declarations: dict[str, Mapping[str, Any]] = {}
    for declaration in ast.get("declarations", ()):
        if isinstance(declaration, Mapping) and isinstance(declaration.get("name"), str):
            declarations[str(declaration["name"])] = declaration
    issues: list[IndexSafetyIssue] = []
    _visit(ast.get("objective"), declarations, {}, issues)
    _visit(ast.get("constraints"), declarations, {}, issues)
    for status in ("unsafe", "unknown"):
        for issue in issues:
            if issue.status == status:
                return issue
    return None


def _visit(
    node: Any,
    declarations: Mapping[str, Mapping[str, Any]],
    binders: Mapping[str, _Interval],
    issues: list[IndexSafetyIssue],
) -> None:
    if isinstance(node, list):
        for item in node:
            _visit(item, declarations, binders, issues)
        return
    if not isinstance(node, Mapping):
        return
    node_type = node.get("type")
    if node_type in {"sum", "forall_constraint"}:
        nested_binders = dict(binders)
        for iterator in node.get("iterators", ()):
            if not isinstance(iterator, Mapping) or not isinstance(iterator.get("iterator"), str):
                continue
            interval = _iterator_interval(iterator.get("range"), declarations, nested_binders)
            if interval is not None:
                nested_binders[str(iterator["iterator"])] = interval
        _apply_filter(node.get("index_constraint"), nested_binders)
        _visit(node.get("expression"), declarations, nested_binders, issues)
        return
    if node_type == "indexed_name":
        _validate_access(node, declarations, binders, issues)
    for value in node.values():
        _visit(value, declarations, binders, issues)


def _iterator_interval(
    domain: Any,
    declarations: Mapping[str, Mapping[str, Any]],
    binders: Mapping[str, _Interval],
) -> _Interval | None:
    if not isinstance(domain, Mapping):
        return None
    if domain.get("type") == "named_range":
        name = domain.get("name")
        if not isinstance(name, str):
            return None
        declaration = declarations.get(name)
        if declaration is None:
            return None
        return _range_interval(declaration, binders)
    if domain.get("type") == "range_specifier":
        return _range_interval(domain, binders)
    return None


def _range_interval(node: Mapping[str, Any], binders: Mapping[str, _Interval]) -> _Interval | None:
    lower = _affine(node.get("start"), binders)
    upper = _affine(node.get("end"), binders)
    if lower is None or upper is None:
        return None
    return _Interval(_expand_binders(lower, binders, upper=False), _expand_binders(upper, binders, upper=True))


def _expand_binders(expression: _Affine, binders: Mapping[str, _Interval], *, upper: bool) -> _Affine:
    result = _Affine((), expression.constant)
    for name, coefficient in expression.coefficients:
        interval = binders.get(name)
        if interval is None:
            result = result.add(_Affine(((name, coefficient),)))
            continue
        use_upper = upper if coefficient >= 0 else not upper
        bound = interval.upper if use_upper else interval.lower
        result = result.add(bound.scale(coefficient))
    return result


def _apply_filter(node: Any, binders: dict[str, _Interval]) -> None:
    if not isinstance(node, Mapping):
        return
    if node.get("type") == "binop" and node.get("op") in {"&&", "and"}:
        _apply_filter(node.get("left"), binders)
        _apply_filter(node.get("right"), binders)
        return
    if node.get("type") != "binop" or node.get("op") not in {"<", "<=", ">", ">="}:
        return
    left_name = _name(node.get("left"))
    right_name = _name(node.get("right"))
    if left_name in binders:
        bound = _affine(node.get("right"), binders)
        if bound is not None:
            interval = binders[left_name]
            if node["op"] in {"<", "<="}:
                candidate = bound.add(_Affine((), -1)) if node["op"] == "<" else bound
                binders[left_name] = _Interval(interval.lower, _tighter_upper(interval.upper, candidate))
            else:
                candidate = bound.add(_Affine((), 1)) if node["op"] == ">" else bound
                binders[left_name] = _Interval(_tighter_lower(interval.lower, candidate), interval.upper)
    elif right_name in binders:
        reversed_op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[str(node["op"])]
        _apply_filter(
            {
                "type": "binop",
                "op": reversed_op,
                "left": node.get("right"),
                "right": node.get("left"),
            },
            binders,
        )


def _validate_access(
    node: Mapping[str, Any],
    declarations: Mapping[str, Mapping[str, Any]],
    binders: Mapping[str, _Interval],
    issues: list[IndexSafetyIssue],
) -> None:
    name = node.get("name")
    if not isinstance(name, str):
        return
    declaration = declarations.get(name)
    if declaration is None:
        return
    dimensions = zip(node.get("dimensions", ()), declaration.get("dimensions", ()))
    for dimension_number, (index, dimension) in enumerate(dimensions, 1):
        declared = _range_interval(dimension, binders) if isinstance(dimension, Mapping) else None
        actual = _expression_interval(index, binders)
        if declared is None:
            continue
        if actual is None:
            issues.append(
                IndexSafetyIssue(
                    "unknown",
                    f"cannot prove index {dimension_number} of '{name}' stays in its declared range",
                )
            )
            continue
        lower_delta = actual.lower.add(declared.lower, -1)
        upper_delta = actual.upper.add(declared.upper, -1)
        if lower_delta.coefficients or upper_delta.coefficients:
            issues.append(
                IndexSafetyIssue(
                    "unknown",
                    f"cannot prove index {dimension_number} of '{name}' stays in its declared range",
                )
            )
            continue
        if not lower_delta.coefficients and lower_delta.constant < 0:
            issues.append(
                IndexSafetyIssue(
                    "unsafe",
                    f"index {dimension_number} of '{name}' can fall below its declared range",
                )
            )
        if not upper_delta.coefficients and upper_delta.constant > 0:
            issues.append(
                IndexSafetyIssue(
                    "unsafe",
                    f"index {dimension_number} of '{name}' can exceed its declared range",
                )
            )


def _tighter_upper(current: _Affine, candidate: _Affine) -> _Affine:
    difference = candidate.add(current, -1)
    return candidate if not difference.coefficients and difference.constant < 0 else current


def _tighter_lower(current: _Affine, candidate: _Affine) -> _Affine:
    difference = candidate.add(current, -1)
    return candidate if not difference.coefficients and difference.constant > 0 else current


def _expression_interval(node: Any, binders: Mapping[str, _Interval]) -> _Interval | None:
    expression = _affine(node, binders)
    if expression is None:
        return None
    lower = _Affine((), expression.constant)
    upper = _Affine((), expression.constant)
    for name, coefficient in expression.coefficients:
        interval = binders.get(name)
        if interval is None:
            return None
        if coefficient >= 0:
            lower = lower.add(interval.lower.scale(coefficient))
            upper = upper.add(interval.upper.scale(coefficient))
        else:
            lower = lower.add(interval.upper.scale(coefficient))
            upper = upper.add(interval.lower.scale(coefficient))
    return _Interval(lower, upper)


def _affine(node: Any, binders: Mapping[str, _Interval]) -> _Affine | None:
    if not isinstance(node, Mapping):
        return None
    node_type = node.get("type")
    if node_type in {"number", "number_literal_index"} and isinstance(node.get("value"), int):
        return _Affine((), int(node["value"]))
    name = _name(node)
    if name is not None:
        return _Affine(((name, 1),))
    if node_type == "parenthesized_expression":
        return _affine(node.get("expression"), binders)
    if node_type == "uminus":
        operand = _affine(node.get("expression") or node.get("operand"), binders)
        return None if operand is None else operand.scale(-1)
    if node_type != "binop" or node.get("op") not in {"+", "-"}:
        return None
    left = _affine(node.get("left"), binders)
    right = _affine(node.get("right"), binders)
    if left is None or right is None:
        return None
    return left.add(right, -1 if node.get("op") == "-" else 1)


def _name(node: Any) -> str | None:
    if not isinstance(node, Mapping) or node.get("type") not in {"name", "name_reference_index"}:
        return None
    value = node.get("value", node.get("name"))
    return value if isinstance(value, str) else None

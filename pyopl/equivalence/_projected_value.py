"""Representations shared by projected-value equivalence proofs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from pyopl.equivalence._indexed_projection import IndexedObservablePair, ParameterContract


@dataclass(frozen=True)
class AffineObservable:
    """A sparse affine expression over grounded ``LinearProblem`` columns."""

    coefficients: tuple[tuple[str, float], ...]
    constant: float = 0.0

    def __post_init__(self) -> None:
        terms: dict[str, float] = {}
        for name, coefficient in self.coefficients:
            if not isinstance(name, str) or not name:
                raise ValueError("observable column names must be nonempty strings")
            try:
                numeric_coefficient = float(coefficient)
            except (TypeError, ValueError) as exc:
                raise ValueError("observable coefficients must be finite numbers") from exc
            if not isfinite(numeric_coefficient):
                raise ValueError("observable coefficients must be finite")
            terms[name] = terms.get(name, 0.0) + numeric_coefficient
        try:
            numeric_constant = float(self.constant)
        except (TypeError, ValueError) as exc:
            raise ValueError("observable constants must be finite numbers") from exc
        if not isfinite(numeric_constant):
            raise ValueError("observable constants must be finite")
        canonical = tuple(
            (name, 0.0 if abs(coefficient) == 0.0 else coefficient)
            for name, coefficient in sorted(terms.items())
            if abs(coefficient) > 1e-12
        )
        object.__setattr__(self, "coefficients", canonical)
        object.__setattr__(self, "constant", 0.0 if abs(numeric_constant) <= 1e-12 else numeric_constant)


@dataclass(frozen=True)
class ObservablePair:
    """A named correspondence between affine observables in two models."""

    name: str
    left: AffineObservable
    right: AffineObservable

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observable names must be nonempty")


@dataclass(frozen=True)
class IndexDomain:
    """A finite grounding domain for one observable-template index."""

    name: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or len(set(self.values)) != len(self.values):
            raise ValueError("index domains require a unique nonempty name and unique values")

    @classmethod
    def from_model(cls, name: str, model_code: str, data_code: str | None = None) -> IndexDomain:
        """Build a finite domain from a named range in model and data text."""
        from pyopl.pyopl_core import resolve_named_range

        start, end = resolve_named_range(model_code, data_code, name)
        if start > end:
            raise ValueError(f"model range {name!r} has descending bounds")
        return cls(name, tuple(range(start, end + 1)))


@dataclass(frozen=True)
class ObservableTemplate:
    """A typed finite family of affine observable correspondences.

    Callbacks receive the current grounded index environment. They keep source
    expression syntax out of the proof engine while supporting indexed sums and
    grounding-time guards.
    """

    name: str
    indices: tuple[IndexDomain, ...]
    left: Callable[[Mapping[str, int]], AffineObservable]
    right: Callable[[Mapping[str, int]], AffineObservable]
    guard: Callable[[Mapping[str, int]], bool] | None = None

    def expand(self) -> tuple[ObservablePair, ...]:
        index_names = tuple(index.name for index in self.indices)
        pairs: list[ObservablePair] = []
        domains = tuple(index.values for index in self.indices)
        for values in product(*domains) if domains else ((),):
            environment = dict(zip(index_names, values))
            if self.guard is not None and not self.guard(environment):
                continue
            pairs.append(
                ObservablePair(
                    self.name.format(**environment),
                    self.left(environment),
                    self.right(environment),
                )
            )
        return tuple(pairs)


def expand_observable_templates(templates: tuple[ObservableTemplate, ...]) -> tuple[ObservablePair, ...]:
    """Expand finite templates in declaration and index order."""

    return tuple(pair for template in templates for pair in template.expand())


def parse_observable_mapping(text: str) -> tuple[ObservablePair, ...] | tuple[Any, ...]:
    """Parse grounded or tagged indexed observable correspondences.

    Grounded entries retain the original sparse-column format. Entries tagged
    with ``kind: "indexed"`` are decoded into ``IndexedObservablePair`` values
    through the exact symbolic record representation.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid observable mapping JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise ValueError("observable mapping JSON must contain a list")

    def expression(value: Any) -> AffineObservable:
        if not isinstance(value, dict):
            raise ValueError("observable expressions must be JSON objects")
        coefficients = value.get("coefficients", value)
        if not isinstance(coefficients, dict):
            raise ValueError("observable coefficients must be a column-to-coefficient object")
        constant = value.get("constant", 0.0)
        if not isinstance(constant, (int, float)):
            raise ValueError("observable constants must be numbers")
        return AffineObservable(tuple(coefficients.items()), constant)

    pairs: list[Any] = []
    names: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("each observable mapping entry requires a string name")
        if entry["name"] in names:
            raise ValueError(f"observable mapping names must be unique: {entry['name']}")
        names.add(entry["name"])
        if entry.get("kind") == "indexed":
            from pyopl.equivalence._indexed_projection import indexed_observable_pair_from_record

            pairs.append(indexed_observable_pair_from_record(entry))
            continue
        if "left" not in entry or "right" not in entry:
            raise ValueError(f"observable {entry['name']} requires left and right expressions")
        pairs.append(ObservablePair(entry["name"], expression(entry["left"]), expression(entry["right"])))
    return tuple(pairs)


def serialize_observable_mapping(mapping: tuple[Any, ...]) -> str:
    """Serialize grounded and tagged indexed mappings through one JSON envelope."""
    from pyopl.equivalence._indexed_projection import IndexedObservablePair, indexed_observable_pair_to_record

    names: set[str] = set()
    records: list[dict[str, Any]] = []
    for pair in mapping:
        if isinstance(pair, IndexedObservablePair):
            name = pair.name_template
            record = indexed_observable_pair_to_record(pair)
        elif isinstance(pair, ObservablePair):
            name = pair.name
            record = {
                "name": name,
                "left": {"coefficients": dict(pair.left.coefficients), "constant": pair.left.constant},
                "right": {"coefficients": dict(pair.right.coefficients), "constant": pair.right.constant},
            }
        else:
            raise ValueError("observable mapping contains an unsupported entry type")
        if name in names:
            raise ValueError(f"observable mapping names must be unique: {name}")
        names.add(name)
        records.append(record)
    return json.dumps(records, sort_keys=True)


def serialize_indexed_observable_mapping(
    mapping: tuple["IndexedObservablePair", ...],
    parameter_contract: "ParameterContract",
) -> str:
    """Serialize a uniform indexed mapping and its typed contract."""
    from pyopl.equivalence._indexed_projection import (
        IndexedObservablePair,
        indexed_observable_pair_to_record,
        parameter_contract_to_record,
    )

    if not mapping or not all(isinstance(pair, IndexedObservablePair) for pair in mapping):
        raise ValueError("uniform indexed mappings require at least one indexed observable pair")
    names = [pair.name_template for pair in mapping]
    if len(set(names)) != len(names):
        raise ValueError("uniform indexed mapping names must be unique")
    payload = {
        "schema": "pyopl.uniform_indexed_observable_mapping",
        "version": 1,
        "observables": [indexed_observable_pair_to_record(pair) for pair in mapping],
        "parameter_contract": parameter_contract_to_record(parameter_contract),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def parse_indexed_observable_mapping(
    text: str,
) -> tuple[tuple["IndexedObservablePair", ...], "ParameterContract"]:
    """Parse the versioned uniform indexed mapping JSON envelope."""
    from pyopl.equivalence._indexed_projection import indexed_observable_pair_from_record, parameter_contract_from_record

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid indexed observable mapping JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("uniform indexed mapping JSON must contain an object")
    if payload.get("schema") != "pyopl.uniform_indexed_observable_mapping":
        raise ValueError("unsupported uniform indexed mapping schema")
    if payload.get("version") != 1:
        raise ValueError("unsupported uniform indexed mapping schema version")
    observables = payload.get("observables")
    if not isinstance(observables, list) or not observables:
        raise ValueError("uniform indexed mapping requires a nonempty observables array")
    pairs = tuple(indexed_observable_pair_from_record(record) for record in observables)
    names = tuple(pair.name_template for pair in pairs)
    if len(set(names)) != len(names):
        raise ValueError("uniform indexed mapping names must be unique")
    contract_record = payload.get("parameter_contract")
    if not isinstance(contract_record, dict):
        raise ValueError("uniform indexed mapping requires a parameter_contract object")
    contract = parameter_contract_from_record(contract_record)
    return pairs, contract


def ground_indexed_observable_mapping(
    mapping: tuple[Any, ...],
    left_model: str,
    right_model: str,
    left_data: str | None = None,
    right_data: str | None = None,
) -> tuple[ObservablePair, ...]:
    """Expand source-level indexed families using model/data parameter values."""
    from pyopl.equivalence._indexed_projection import IndexedObservablePair

    left_values = _model_data_values(left_model, left_data)
    right_values = _model_data_values(right_model, right_data)
    pairs: list[ObservablePair] = []
    for pair in mapping:
        if not isinstance(pair, IndexedObservablePair):
            raise ValueError("finite indexed grounding requires indexed observable pairs")
        left_domains = tuple(_ground_domain_values(domain, left_values, {}) for domain in pair.left.domains)
        right_domains = tuple(_ground_domain_values(domain, right_values, {}) for domain in pair.right.domains)
        if left_domains != right_domains:
            raise ValueError(f"indexed observable {pair.name_template} has mismatched grounded domains")
        index_names = tuple(domain.name for domain in pair.left.domains)
        for values in product(*left_domains) if left_domains else ((),):
            environment = dict(zip(index_names, values))
            name = pair.name_template.format(**environment)
            pairs.append(
                ObservablePair(
                    name,
                    _ground_indexed_expression(pair.left.expression, environment, left_values),
                    _ground_indexed_expression(pair.right.expression, environment, right_values),
                )
            )
    return tuple(pairs)


def _model_data_values(model_code: str, data_code: str | None) -> dict[str, Any]:
    from pyopl.pyopl_core import OPLCompiler

    compiler = OPLCompiler()
    model_ast, data = compiler._parse_model_inputs(model_code, data_code)
    return compiler._merge_declaration_data(model_ast.get("declarations", []), data)


def _evaluate_symbolic_value(expression: Any, environment: Mapping[str, Any], data: Mapping[str, Any]) -> int:
    value = float(expression.constant)
    for name, coefficient in expression.coefficients:
        source = environment.get(name, data.get(name))
        if source is None or isinstance(source, (list, tuple, dict)):
            raise ValueError(f"cannot evaluate symbolic index value {name!r}")
        value += float(coefficient) * float(source)
    if not value.is_integer():
        raise ValueError(f"symbolic index value is not integral: {value}")
    return int(value)


def _ground_domain_values(domain: Any, data: Mapping[str, Any], environment: Mapping[str, Any]) -> tuple[int, ...]:
    lower = _evaluate_symbolic_value(domain.lower, environment, data)
    upper = _evaluate_symbolic_value(domain.upper, environment, data)
    if lower > upper:
        return ()
    return tuple(range(lower, upper + 1))


def _ground_indexed_expression(
    expression: Any, outer_environment: Mapping[str, int], data: Mapping[str, Any]
) -> AffineObservable:
    terms: list[tuple[str, float]] = []
    constant = 0.0
    components = getattr(expression, "components", None)
    if components is None:
        components = (expression,)
        component_constants_are_external = True
    else:
        component_constants_are_external = False

    for component in components:
        environments: list[dict[str, int]] = []

        def visit(position: int, environment: dict[str, int]) -> None:
            if position == len(component.sum_binders):
                environments.append(environment)
                return
            binder = component.sum_binders[position]
            domain = component.sum_domains[position]
            for value in _ground_domain_values(domain, data, environment):
                visit(position + 1, {**environment, binder: value})

        visit(0, dict(outer_environment))
        for environment in environments:
            for term in component.terms:
                indices = tuple(_ground_index(index, environment, data) for index in term.indices)
                name = term.declaration if not indices else f"{term.declaration}_{'_'.join(str(index) for index in indices)}"
                terms.append((name, float(term.coefficient)))
            if not component_constants_are_external:
                constant += float(component.constant)
        if component_constants_are_external:
            constant += float(component.constant)
    return AffineObservable(tuple(terms), constant)


def _ground_index(index: str, environment: Mapping[str, int], data: Mapping[str, Any]) -> int:
    match = re.fullmatch(r"(-?)([A-Za-z_]\w*)(?:\s*([+-])\s*([0-9]+))?", index)
    if match is None:
        raise ValueError(f"unsupported indexed term: {index}")
    sign, name, operator, offset = match.groups()
    source = environment.get(name, data.get(name))
    if source is None or isinstance(source, (list, tuple, dict)):
        raise ValueError(f"cannot evaluate indexed term {index}")
    value = int(source)
    if sign == "-":
        value = -value
    if operator == "+":
        value += int(offset)
    elif operator == "-":
        value -= int(offset)
    return value


def legacy_observables(variable_mapping: dict[str, str]) -> tuple[ObservablePair, ...]:
    """Adapt the legacy one-column mapping to affine observables."""

    return tuple(
        ObservablePair(name, AffineObservable(((name, 1.0),)), AffineObservable(((target, 1.0),)))
        for name, target in sorted(variable_mapping.items())
    )


def render_observable_mapping(mapping: tuple[ObservablePair, ...]) -> tuple[str, ...]:
    """Return deterministic human-readable correspondence evidence."""

    def render(expression: AffineObservable) -> str:
        terms = [f"{coefficient:g}*{name}" for name, coefficient in expression.coefficients]
        if expression.constant:
            terms.append(f"{expression.constant:g}")
        return " + ".join(terms) or "0"

    return tuple(f"{pair.name}: {render(pair.left)} <-> {render(pair.right)}" for pair in mapping)

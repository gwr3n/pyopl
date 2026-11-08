"""Typed data structures and codecs for indexed observable mappings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, Mapping


def _fraction(value: int | float | str | Fraction) -> Fraction:
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("symbolic affine values must be finite rational numbers") from exc


@dataclass(frozen=True)
class SymbolicAffineExpression:
    """An exact affine expression over named scalar parameters."""

    coefficients: tuple[tuple[str, Fraction | int | float | str], ...] = ()
    constant: Fraction | int | float | str = 0

    def __post_init__(self) -> None:
        terms: dict[str, Fraction] = {}
        for name, coefficient in self.coefficients:
            if not isinstance(name, str) or not name:
                raise ValueError("symbolic parameter names must be nonempty strings")
            terms[name] = terms.get(name, Fraction(0)) + _fraction(coefficient)
        canonical = tuple((name, coefficient) for name, coefficient in sorted(terms.items()) if coefficient)
        object.__setattr__(self, "coefficients", canonical)
        object.__setattr__(self, "constant", _fraction(self.constant))

    @classmethod
    def constant_value(cls, value: int | float | str | Fraction) -> SymbolicAffineExpression:
        return cls(constant=value)

    @classmethod
    def parameter(cls, name: str) -> SymbolicAffineExpression:
        return cls(((name, 1),))

    def add(self, other: SymbolicAffineExpression) -> SymbolicAffineExpression:
        return SymbolicAffineExpression(
            self.coefficients + other.coefficients,
            _fraction(self.constant) + _fraction(other.constant),
        )

    def subtract(self, other: SymbolicAffineExpression) -> SymbolicAffineExpression:
        return self.add(other.scale(-1))

    def scale(self, factor: int | float | str | Fraction) -> SymbolicAffineExpression:
        numeric_factor = _fraction(factor)
        return SymbolicAffineExpression(
            tuple((name, _fraction(coefficient) * numeric_factor) for name, coefficient in self.coefficients),
            _fraction(self.constant) * numeric_factor,
        )

    def rename_symbols(self, symbol_mapping: dict[str, str]) -> SymbolicAffineExpression:
        return SymbolicAffineExpression(
            tuple((symbol_mapping.get(name, name), coefficient) for name, coefficient in self.coefficients),
            self.constant,
        )

    def substitute(self, substitutions: dict[str, SymbolicAffineExpression]) -> SymbolicAffineExpression:
        result = SymbolicAffineExpression.constant_value(self.constant)
        for name, coefficient in self.coefficients:
            replacement = substitutions.get(name, SymbolicAffineExpression.parameter(name))
            result = result.add(replacement.scale(coefficient))
        return result

    def render(self) -> str:
        terms = [f"{coefficient}*{name}" for name, coefficient in self.coefficients]
        if self.constant:
            terms.append(str(self.constant))
        return " + ".join(terms) or "0"


PredicateRelation = Literal["==", "!=", "<=", "<", ">=", ">"]


@dataclass(frozen=True)
class SymbolicPredicate:
    """A comparison between two exact affine expressions."""

    left: SymbolicAffineExpression
    relation: PredicateRelation
    right: SymbolicAffineExpression

    def __post_init__(self) -> None:
        if self.relation not in {"==", "!=", "<=", "<", ">=", ">"}:
            raise ValueError(f"unsupported symbolic relation: {self.relation}")

    def render(self) -> str:
        return f"{self.left.render()} {self.relation} {self.right.render()}"

    def rename_symbols(self, symbol_mapping: dict[str, str]) -> SymbolicPredicate:
        return SymbolicPredicate(
            self.left.rename_symbols(symbol_mapping),
            self.relation,
            self.right.rename_symbols(symbol_mapping),
        )


@dataclass(frozen=True)
class SymbolicDomain:
    """A closed integer interval with affine symbolic endpoints."""

    name: str
    lower: SymbolicAffineExpression
    upper: SymbolicAffineExpression
    nonempty: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("symbolic domains require a nonempty name")

    def render(self) -> str:
        suffix = " (nonempty)" if self.nonempty else ""
        return f"{self.name} in {self.lower.render()}..{self.upper.render()}{suffix}"


@dataclass(frozen=True)
class IndexedAffineTerm:
    """One numeric coefficient times an indexed decision atom."""

    declaration: str
    indices: tuple[str, ...] = ()
    coefficient: Fraction | int | float | str = 1

    def __post_init__(self) -> None:
        if not self.declaration:
            raise ValueError("indexed affine terms require a declaration name")
        if any(not re.fullmatch(r"-?(?:[0-9]+|[A-Za-z_]\w*(?:\s*[+-]\s*[0-9]+)?)", index) for index in self.indices):
            raise ValueError("indexed terms require binder names with optional integer offsets")
        object.__setattr__(self, "coefficient", _fraction(self.coefficient))


@dataclass(frozen=True)
class IndexedAffineComponent:
    """One affine component with its own optional summation scope."""

    terms: tuple[IndexedAffineTerm, ...] = ()
    constant: Fraction | int | float | str = 0
    sum_binders: tuple[str, ...] = ()
    sum_domains: tuple[SymbolicDomain, ...] = ()

    def __post_init__(self) -> None:
        if len(self.sum_binders) != len(self.sum_domains) or len(set(self.sum_binders)) != len(self.sum_binders):
            raise ValueError("indexed components require one unique binder per domain")
        if any(not binder for binder in self.sum_binders):
            raise ValueError("indexed component sum binders must be nonempty")
        if any(domain.name != binder for binder, domain in zip(self.sum_binders, self.sum_domains)):
            raise ValueError("indexed component sum domains must name their binders")
        merged: dict[tuple[str, tuple[str, ...]], Fraction] = {}
        for term in self.terms:
            key = (term.declaration, term.indices)
            merged[key] = merged.get(key, Fraction(0)) + _fraction(term.coefficient)
        canonical = tuple(
            IndexedAffineTerm(declaration, indices, coefficient)
            for (declaration, indices), coefficient in sorted(merged.items())
            if coefficient
        )
        object.__setattr__(self, "terms", canonical)
        object.__setattr__(self, "constant", _fraction(self.constant))


@dataclass(frozen=True)
class IndexedScopedAffineExpression:
    """An affine expression made of independently scoped components."""

    components: tuple[IndexedAffineComponent, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("scoped indexed expressions require at least one component")

    def alpha_normalized(self) -> IndexedScopedAffineExpression:
        components = tuple(sorted((_alpha_normalize_component(component) for component in self.components), key=repr))
        return IndexedScopedAffineExpression(components)


@dataclass(frozen=True)
class IndexedAffineExpression:
    """An affine expression whose decision atoms retain symbolic indices."""

    terms: tuple[IndexedAffineTerm, ...] = ()
    constant: Fraction | int | float | str = 0
    sum_binders: tuple[str, ...] = ()
    sum_domains: tuple[SymbolicDomain, ...] = ()

    def __post_init__(self) -> None:
        if len(self.sum_binders) != len(self.sum_domains) or len(set(self.sum_binders)) != len(self.sum_binders):
            raise ValueError("indexed sums require one unique binder per domain")
        if any(not binder for binder in self.sum_binders):
            raise ValueError("indexed sum binders must be nonempty")
        if any(domain.name != binder for binder, domain in zip(self.sum_binders, self.sum_domains)):
            raise ValueError("indexed sum domains must name their corresponding binders")
        merged: dict[tuple[str, tuple[str, ...]], Fraction] = {}
        for term in self.terms:
            key = (term.declaration, term.indices)
            merged[key] = merged.get(key, Fraction(0)) + _fraction(term.coefficient)
        canonical = tuple(
            IndexedAffineTerm(declaration, indices, coefficient)
            for (declaration, indices), coefficient in sorted(merged.items())
            if coefficient
        )
        object.__setattr__(self, "terms", canonical)
        object.__setattr__(self, "constant", _fraction(self.constant))

    def alpha_normalized(self) -> IndexedAffineExpression:
        """Return the expression with its sum binders renamed deterministically."""
        sum_mapping = {binder: f"s{position}" for position, binder in enumerate(self.sum_binders)}
        domains = tuple(
            SymbolicDomain(
                sum_mapping[binder],
                domain.lower.rename_symbols(sum_mapping),
                domain.upper.rename_symbols(sum_mapping),
                domain.nonempty,
            )
            for binder, domain in zip(self.sum_binders, self.sum_domains)
        )
        terms = tuple(
            IndexedAffineTerm(
                term.declaration,
                tuple(_rename_index_term(index, sum_mapping) for index in term.indices),
                term.coefficient,
            )
            for term in self.terms
        )
        return IndexedAffineExpression(terms, self.constant, tuple(sum_mapping.values()), domains)

    def add(self, other: IndexedAffineExpression) -> IndexedAffineExpression:
        if self.sum_binders != other.sum_binders or self.sum_domains != other.sum_domains:
            raise ValueError("indexed affine expressions require matching sum schemas")
        return IndexedAffineExpression(
            self.terms + other.terms,
            _fraction(self.constant) + _fraction(other.constant),
            self.sum_binders,
            self.sum_domains,
        )

    def scale(self, factor: int | float | str | Fraction) -> IndexedAffineExpression:
        numeric_factor = _fraction(factor)
        return IndexedAffineExpression(
            tuple(
                IndexedAffineTerm(term.declaration, term.indices, _fraction(term.coefficient) * numeric_factor)
                for term in self.terms
            ),
            _fraction(self.constant) * numeric_factor,
            self.sum_binders,
            self.sum_domains,
        )

    def subtract(self, other: IndexedAffineExpression) -> IndexedAffineExpression:
        return self.add(other.scale(-1))


def _alpha_normalize_component(component: IndexedAffineComponent) -> IndexedAffineComponent:
    mapping = {binder: f"s{position}" for position, binder in enumerate(component.sum_binders)}
    domains = tuple(
        SymbolicDomain(
            mapping[binder],
            domain.lower.rename_symbols(mapping),
            domain.upper.rename_symbols(mapping),
            domain.nonempty,
        )
        for binder, domain in zip(component.sum_binders, component.sum_domains)
    )
    terms = tuple(
        IndexedAffineTerm(
            term.declaration,
            tuple(_rename_index_term(index, mapping) for index in term.indices),
            term.coefficient,
        )
        for term in component.terms
    )
    return IndexedAffineComponent(terms, component.constant, tuple(mapping.values()), domains)


def _expression_components(
    expression: IndexedAffineExpression | IndexedScopedAffineExpression,
) -> tuple[IndexedAffineComponent, ...]:
    if isinstance(expression, IndexedScopedAffineExpression):
        return expression.components
    return (IndexedAffineComponent(expression.terms, expression.constant, expression.sum_binders, expression.sum_domains),)


@dataclass(frozen=True)
class IndexedAffineFamily:
    """An affine expression family with explicit binders and domains."""

    binders: tuple[str, ...]
    domains: tuple[SymbolicDomain, ...]
    expression: IndexedAffineExpression | IndexedScopedAffineExpression
    guard: SymbolicPredicate | None = None

    def __post_init__(self) -> None:
        if len(self.binders) != len(self.domains) or len(set(self.binders)) != len(self.binders):
            raise ValueError("indexed families require one unique binder per domain")
        if any(not binder for binder in self.binders):
            raise ValueError("indexed family binders must be nonempty")
        for component in _expression_components(self.expression):
            if set(self.binders) & set(component.sum_binders):
                raise ValueError("indexed sum binders cannot shadow family binders")
            bound_names = set(self.binders) | set(component.sum_binders)
            for term in component.terms:
                for index in term.indices:
                    match = re.fullmatch(r"-?([A-Za-z_]\w*)(?:\s*[+-]\s*[0-9]+)?", index)
                    if match is None or match.group(1) not in bound_names:
                        raise ValueError(f"indexed term uses unbound index: {index}")
        if self.guard is not None and not isinstance(self.guard, SymbolicPredicate):
            raise ValueError("indexed family guards require a symbolic predicate")

    def alpha_normalized(self) -> IndexedAffineFamily:
        """Return the family with binders renamed to deterministic names."""
        binder_mapping = {binder: f"i{position}" for position, binder in enumerate(self.binders)}
        domains = tuple(
            SymbolicDomain(
                binder_mapping[binder],
                domain.lower.rename_symbols(binder_mapping),
                domain.upper.rename_symbols(binder_mapping),
                domain.nonempty,
            )
            for binder, domain in zip(self.binders, self.domains)
        )
        components = []
        for component in _expression_components(self.expression):
            sum_mapping = {binder: f"s{position}" for position, binder in enumerate(component.sum_binders)}
            symbol_mapping = {**binder_mapping, **sum_mapping}
            sum_domains = tuple(
                SymbolicDomain(
                    sum_mapping[binder],
                    domain.lower.rename_symbols(symbol_mapping),
                    domain.upper.rename_symbols(symbol_mapping),
                    domain.nonempty,
                )
                for binder, domain in zip(component.sum_binders, component.sum_domains)
            )
            terms = tuple(
                IndexedAffineTerm(
                    term.declaration,
                    tuple(_rename_index_term(index, symbol_mapping) for index in term.indices),
                    term.coefficient,
                )
                for term in component.terms
            )
            components.append(IndexedAffineComponent(terms, component.constant, tuple(sum_mapping.values()), sum_domains))
        expression: IndexedAffineExpression | IndexedScopedAffineExpression
        if isinstance(self.expression, IndexedScopedAffineExpression):
            expression = IndexedScopedAffineExpression(tuple(components))
        else:
            component = components[0]
            expression = IndexedAffineExpression(
                component.terms, component.constant, component.sum_binders, component.sum_domains
            )
        guard = self.guard.rename_symbols(binder_mapping) if self.guard is not None else None
        return IndexedAffineFamily(tuple(binder_mapping.values()), domains, expression, guard)


@dataclass(frozen=True)
class IndexedObservablePair:
    """A named correspondence between two symbolic observable families."""

    name_template: str
    left: IndexedAffineFamily
    right: IndexedAffineFamily

    def __post_init__(self) -> None:
        if not self.name_template:
            raise ValueError("indexed observable names must be nonempty")


def indexed_observable_pair_to_record(pair: IndexedObservablePair) -> dict[str, Any]:
    """Serialize an indexed observable pair using the shared mapping envelope."""
    return {
        "kind": "indexed",
        "name": pair.name_template,
        "left": _indexed_family_to_record(pair.left),
        "right": _indexed_family_to_record(pair.right),
    }


def indexed_observable_pair_from_record(record: Mapping[str, Any]) -> IndexedObservablePair:
    """Deserialize one tagged indexed observable mapping entry."""
    if record.get("kind") != "indexed":
        raise ValueError("indexed observable entries require kind='indexed'")
    name = record.get("name")
    if not isinstance(name, str):
        raise ValueError("indexed observable entries require a string name")
    try:
        left = _indexed_family_from_record(record["left"])
        right = _indexed_family_from_record(record["right"])
    except KeyError as exc:
        raise ValueError(f"indexed observable {name} is missing {exc.args[0]}") from exc
    return IndexedObservablePair(name, left, right)


def _rename_index_term(index: str, binder_mapping: dict[str, str]) -> str:
    match = re.fullmatch(r"(-?)([A-Za-z_]\w*)(\s*[+-]\s*[0-9]+)?", index)
    if match is None or match.group(2) not in binder_mapping:
        return index
    return match.group(1) + binder_mapping[match.group(2)] + (match.group(3) or "")


FactProvenance = Literal["declared", "inferred", "derived", "assumed"]


@dataclass(frozen=True)
class ParameterFact:
    """A symbolic fact and the route by which it entered a contract."""

    predicate: SymbolicPredicate
    provenance: FactProvenance

    def __post_init__(self) -> None:
        if self.provenance not in {"declared", "inferred", "derived", "assumed"}:
            raise ValueError(f"unsupported parameter fact provenance: {self.provenance}")


@dataclass(frozen=True)
class IndexedLinearPredicate:
    left: IndexedAffineExpression
    relation: PredicateRelation
    right: IndexedAffineExpression
    binders: tuple[str, ...] = ()
    domains: tuple[SymbolicDomain, ...] = ()

    def __post_init__(self) -> None:
        if len(self.binders) != len(self.domains) or len(set(self.binders)) != len(self.binders):
            raise ValueError("indexed row predicates require one unique binder per domain")
        if any(not binder for binder in self.binders):
            raise ValueError("indexed row binders must be nonempty")
        if any(domain.name != binder for binder, domain in zip(self.binders, self.domains)):
            raise ValueError("indexed row domains must name their corresponding binders")

    def alpha_normalized(self) -> IndexedLinearPredicate:
        outer_mapping = {binder: f"r{position}" for position, binder in enumerate(self.binders)}
        domains = tuple(
            SymbolicDomain(
                outer_mapping[binder],
                domain.lower.rename_symbols(outer_mapping),
                domain.upper.rename_symbols(outer_mapping),
                domain.nonempty,
            )
            for binder, domain in zip(self.binders, self.domains)
        )
        left = _rename_indexed_expression_symbols(self.left, outer_mapping).alpha_normalized()
        right = _rename_indexed_expression_symbols(self.right, outer_mapping).alpha_normalized()
        return IndexedLinearPredicate(left, self.relation, right, tuple(outer_mapping.values()), domains)


@dataclass(frozen=True)
class IndexedParameterFact:
    """A quantified indexed fact and the route by which it entered a contract."""

    predicate: IndexedLinearPredicate
    provenance: FactProvenance

    def __post_init__(self) -> None:
        if self.provenance not in {"declared", "inferred", "derived"}:
            raise ValueError(f"unsupported indexed parameter fact provenance: {self.provenance}")


@dataclass(frozen=True)
class ParameterContract:
    """Caller-visible assumptions for a uniform indexed comparison."""

    facts: tuple[ParameterFact, ...] = ()
    indexed_facts: tuple[IndexedParameterFact, ...] = ()

    def render(self) -> tuple[str, ...]:
        scalar_facts = tuple((f"{fact.provenance}: {fact.predicate.render()}" for fact in self.facts))
        indexed_facts = tuple((f"{fact.provenance}: {fact.predicate.alpha_normalized()}" for fact in self.indexed_facts))
        return scalar_facts + indexed_facts


def parameter_contract_to_record(contract: ParameterContract) -> dict[str, Any]:
    """Serialize a typed parameter contract for the uniform JSON envelope."""
    return {
        "facts": [
            {"predicate": _predicate_to_record(fact.predicate), "provenance": fact.provenance} for fact in contract.facts
        ],
        "indexed_facts": [
            {"predicate": _indexed_predicate_to_record(fact.predicate), "provenance": fact.provenance}
            for fact in contract.indexed_facts
        ],
    }


def parameter_contract_from_record(record: Mapping[str, Any]) -> ParameterContract:
    """Deserialize and validate a typed parameter contract record."""
    if not isinstance(record, Mapping):
        raise ValueError("parameter contracts must be JSON objects")
    facts = record.get("facts", ())
    indexed_facts = record.get("indexed_facts", ())
    if not isinstance(facts, (list, tuple)) or not isinstance(indexed_facts, (list, tuple)):
        raise ValueError("parameter contract facts must be JSON arrays")
    try:
        scalar = tuple(ParameterFact(_predicate_from_record(item["predicate"]), item["provenance"]) for item in facts)
        indexed = tuple(
            IndexedParameterFact(_indexed_predicate_from_record(item["predicate"]), item["provenance"])
            for item in indexed_facts
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("parameter contract facts require predicate and provenance") from exc
    return ParameterContract(scalar, indexed)


def _indexed_predicate_to_record(predicate: IndexedLinearPredicate) -> dict[str, Any]:
    return {
        "left": _indexed_expression_to_record(predicate.left),
        "relation": predicate.relation,
        "right": _indexed_expression_to_record(predicate.right),
        "binders": list(predicate.binders),
        "domains": [_domain_to_record(domain) for domain in predicate.domains],
    }


def _predicate_to_record(predicate: SymbolicPredicate) -> dict[str, Any]:
    return {
        "left": _affine_to_record(predicate.left),
        "relation": predicate.relation,
        "right": _affine_to_record(predicate.right),
    }


def _predicate_from_record(record: Mapping[str, Any]) -> SymbolicPredicate:
    return SymbolicPredicate(
        _affine_from_record(record["left"]),
        record["relation"],
        _affine_from_record(record["right"]),
    )


def _indexed_predicate_from_record(record: Mapping[str, Any]) -> IndexedLinearPredicate:
    return IndexedLinearPredicate(
        _legacy_indexed_expression_from_record(record["left"]),
        record["relation"],
        _legacy_indexed_expression_from_record(record["right"]),
        tuple(record["binders"]),
        tuple(_domain_from_record(domain) for domain in record["domains"]),
    )


def _indexed_expression_to_record(
    expression: IndexedAffineExpression | IndexedScopedAffineExpression,
) -> dict[str, Any]:
    if isinstance(expression, IndexedScopedAffineExpression):
        return {
            "components": [_indexed_component_to_record(component) for component in expression.components],
        }
    return {
        "terms": [
            {
                "declaration": term.declaration,
                "indices": list(term.indices),
                "coefficient": str(term.coefficient),
            }
            for term in expression.terms
        ],
        "constant": str(expression.constant),
        "sum_binders": list(expression.sum_binders),
        "sum_domains": [_domain_to_record(domain) for domain in expression.sum_domains],
    }


def _indexed_expression_from_record(
    record: Mapping[str, Any],
) -> IndexedAffineExpression | IndexedScopedAffineExpression:
    components = record.get("components")
    if isinstance(components, list):
        return IndexedScopedAffineExpression(tuple(_indexed_component_from_record(component) for component in components))
    return IndexedAffineExpression(
        tuple(IndexedAffineTerm(term["declaration"], tuple(term["indices"]), term["coefficient"]) for term in record["terms"]),
        record["constant"],
        tuple(record["sum_binders"]),
        tuple(_domain_from_record(domain) for domain in record["sum_domains"]),
    )


def _legacy_indexed_expression_from_record(record: Mapping[str, Any]) -> IndexedAffineExpression:
    expression = _indexed_expression_from_record(record)
    if isinstance(expression, IndexedScopedAffineExpression):
        raise ValueError("this certificate requires a single-scope indexed expression")
    return expression


def _indexed_component_to_record(component: IndexedAffineComponent) -> dict[str, Any]:
    return {
        "terms": [
            {
                "declaration": term.declaration,
                "indices": list(term.indices),
                "coefficient": str(term.coefficient),
            }
            for term in component.terms
        ],
        "constant": str(component.constant),
        "sum_binders": list(component.sum_binders),
        "sum_domains": [_domain_to_record(domain) for domain in component.sum_domains],
    }


def _indexed_component_from_record(record: Mapping[str, Any]) -> IndexedAffineComponent:
    return IndexedAffineComponent(
        tuple(IndexedAffineTerm(term["declaration"], tuple(term["indices"]), term["coefficient"]) for term in record["terms"]),
        record["constant"],
        tuple(record["sum_binders"]),
        tuple(_domain_from_record(domain) for domain in record["sum_domains"]),
    )


def _indexed_family_to_record(family: IndexedAffineFamily) -> dict[str, Any]:
    return {
        "binders": list(family.binders),
        "domains": [_domain_to_record(domain) for domain in family.domains],
        "expression": _indexed_expression_to_record(family.expression),
        "guard": _predicate_to_record(family.guard) if family.guard is not None else None,
    }


def _indexed_family_from_record(record: Mapping[str, Any]) -> IndexedAffineFamily:
    if isinstance(record.get("expression"), str):
        return _indexed_family_from_source_record(record)
    return IndexedAffineFamily(
        tuple(record["binders"]),
        tuple(_domain_from_record(domain) for domain in record["domains"]),
        _indexed_expression_from_record(record["expression"]),
        _predicate_from_record(record["guard"]) if isinstance(record.get("guard"), Mapping) else None,
    )


def _indexed_family_from_source_record(record: Mapping[str, Any]) -> IndexedAffineFamily:
    binders = tuple(domain["name"] for domain in record["domains"])
    domains = tuple(_source_domain_from_record(domain) for domain in record["domains"])
    expression = _parse_source_indexed_expression(record["expression"], binders)
    return IndexedAffineFamily(binders, domains, expression)


def _source_domain_from_record(record: Mapping[str, Any]) -> SymbolicDomain:
    name = record.get("name")
    source_range = record.get("range")
    if not isinstance(name, str) or not isinstance(source_range, str):
        raise ValueError("source indexed domains require a name and a range such as '1..N'")
    bounds = source_range.split("..")
    if len(bounds) != 2:
        raise ValueError(f"source indexed domain {name} requires one '..' range separator")
    return SymbolicDomain(
        name,
        _parse_source_affine(bounds[0]),
        _parse_source_affine(bounds[1]),
        bool(record.get("nonempty", False)),
    )


def _parse_source_affine(source: str) -> SymbolicAffineExpression:
    from pyopl.pyopl_core import parse_expression_ast

    return _source_affine_from_ast(parse_expression_ast(source))


def _parse_source_indexed_expression(
    source: str,
    outer_binders: tuple[str, ...],
) -> IndexedAffineExpression | IndexedScopedAffineExpression:
    from pyopl.pyopl_core import parse_expression_ast

    expression_ast = parse_expression_ast(source, outer_binders)
    components = _source_components_from_ast(expression_ast, outer_binders)
    if len(components) == 1:
        component = components[0]
        return IndexedAffineExpression(component.terms, component.constant, component.sum_binders, component.sum_domains)
    return IndexedScopedAffineExpression(components)


def _source_affine_from_ast(node: Any) -> SymbolicAffineExpression:
    if not isinstance(node, Mapping):
        raise ValueError(f"unsupported source affine expression: {node!r}")
    node_type = node.get("type")
    if node_type == "number":
        return SymbolicAffineExpression.constant_value(str(node.get("value")))
    if node_type == "name" and isinstance(node.get("value"), str):
        return SymbolicAffineExpression.parameter(node["value"])
    if node_type == "parenthesized_expression":
        return _source_affine_from_ast(node.get("expression"))
    if node_type == "uminus":
        return _source_affine_from_ast(node.get("value")).scale(-1)
    if node_type == "binop":
        operator = node.get("op")
        left = _source_affine_from_ast(node.get("left"))
        right = _source_affine_from_ast(node.get("right"))
        if operator == "+":
            return left.add(right)
        if operator == "-":
            return left.subtract(right)
        factor = _source_numeric_value(node.get("left"))
        if operator == "*" and factor is not None:
            return right.scale(factor)
        factor = _source_numeric_value(node.get("right"))
        if operator == "*" and factor is not None:
            return left.scale(factor)
    raise ValueError(f"unsupported source affine expression node: {node_type}")


def _source_numeric_value(node: Any) -> Fraction | None:
    if not isinstance(node, Mapping):
        return None
    node_type = node.get("type")
    if node_type == "number":
        return _fraction(str(node.get("value")))
    if node_type == "parenthesized_expression":
        return _source_numeric_value(node.get("expression"))
    if node_type == "uminus":
        value = _source_numeric_value(node.get("value"))
        return -value if value is not None else None
    if node_type != "binop":
        return None
    left = _source_numeric_value(node.get("left"))
    right = _source_numeric_value(node.get("right"))
    if left is None or right is None:
        return None
    operator = node.get("op")
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/" and right:
        return left / right
    return None


def _source_index_from_ast(node: Any) -> str:
    if not isinstance(node, Mapping):
        raise ValueError(f"unsupported indexed expression index: {node!r}")
    if node.get("type") == "name_reference_index" and isinstance(node.get("name"), str):
        return node["name"]
    affine = _source_affine_from_ast(node)
    if len(affine.coefficients) != 1:
        raise ValueError("indexed expression indices must be one bound name with an optional integer offset")
    name, coefficient = affine.coefficients[0]
    constant = _fraction(affine.constant)
    if coefficient not in (1, -1) or constant.denominator != 1:
        raise ValueError("indexed expression indices must be one bound name with an optional integer offset")
    sign = "-" if coefficient == -1 else ""
    offset = int(constant)
    if offset > 0:
        return f"{sign}{name} + {offset}"
    if offset < 0:
        return f"{sign}{name} - {-offset}"
    return f"{sign}{name}"


def _source_components_from_ast(
    node: Any,
    bound_names: tuple[str, ...],
) -> tuple[IndexedAffineComponent, ...]:
    if not isinstance(node, Mapping):
        raise ValueError(f"unsupported source indexed expression: {node!r}")
    node_type = node.get("type")
    if node_type == "parenthesized_expression":
        return _source_components_from_ast(node.get("expression"), bound_names)
    if node_type == "uminus":
        return _scale_source_components(_source_components_from_ast(node.get("value"), bound_names), -1)
    if node_type == "binop":
        return _source_binary_components(node, bound_names)
    if node_type == "sum":
        return _source_sum_components(node, bound_names)
    if node_type == "number":
        return (IndexedAffineComponent(constant=str(node.get("value"))),)
    if node_type == "indexed_name":
        declaration = node.get("name")
        dimensions = node.get("dimensions")
        if not isinstance(declaration, str) or not isinstance(dimensions, list):
            raise ValueError("invalid indexed declaration reference")
        indices = tuple(_source_index_from_ast(index) for index in dimensions)
        return (IndexedAffineComponent((IndexedAffineTerm(declaration, indices),)),)
    raise ValueError(f"unsupported source indexed expression node: {node_type}")


def _source_binary_components(
    node: Mapping[str, Any],
    bound_names: tuple[str, ...],
) -> tuple[IndexedAffineComponent, ...]:
    operator = node.get("op")
    left_node = node.get("left")
    right_node = node.get("right")
    if operator in ("+", "-"):
        left = _source_components_from_ast(left_node, bound_names)
        right = _source_components_from_ast(right_node, bound_names)
        if operator == "-":
            right = _scale_source_components(right, -1)
        return left + right
    left_factor = _source_numeric_value(left_node)
    right_factor = _source_numeric_value(right_node)
    if operator == "*" and left_factor is not None:
        return _scale_source_components(_source_components_from_ast(right_node, bound_names), left_factor)
    if operator == "*" and right_factor is not None:
        return _scale_source_components(_source_components_from_ast(left_node, bound_names), right_factor)
    if operator == "/" and right_factor:
        return _scale_source_components(_source_components_from_ast(left_node, bound_names), 1 / right_factor)
    if operator in ("*", "/"):
        raise ValueError("indexed affine expressions require numeric coefficients; symbolic products are unsupported")
    raise ValueError(f"unsupported operator in indexed affine expression: {operator}")


def _source_sum_components(
    node: Mapping[str, Any],
    bound_names: tuple[str, ...],
) -> tuple[IndexedAffineComponent, ...]:
    if node.get("index_constraint") is not None:
        raise ValueError("filtered indexed sums are unsupported; use explicit interval domains")
    iterators = node.get("iterators")
    if not isinstance(iterators, list) or not iterators:
        raise ValueError("indexed sums require at least one interval binder")
    binders: list[str] = []
    domains: list[SymbolicDomain] = []
    for iterator in iterators:
        binder, domain = _source_sum_domain(iterator, bound_names, tuple(binders))
        binders.append(binder)
        domains.append(domain)
    scoped_binders = bound_names + tuple(binders)
    components = _source_components_from_ast(node.get("expression"), scoped_binders)
    return tuple(
        IndexedAffineComponent(
            component.terms,
            component.constant,
            tuple(binders) + component.sum_binders,
            tuple(domains) + component.sum_domains,
        )
        for component in components
    )


def _source_sum_domain(
    iterator: Any,
    bound_names: tuple[str, ...],
    prior_binders: tuple[str, ...],
) -> tuple[str, SymbolicDomain]:
    if not isinstance(iterator, Mapping):
        raise ValueError("invalid indexed sum binder")
    binder = iterator.get("iterator")
    domain_record = iterator.get("range")
    if not isinstance(binder, str) or binder in bound_names or binder in prior_binders:
        raise ValueError(f"indexed sum binder shadows an existing binder: {binder}")
    if not isinstance(domain_record, Mapping) or domain_record.get("type") != "range_specifier":
        raise ValueError("indexed sums currently require closed interval domains")
    domain = SymbolicDomain(
        binder,
        _source_affine_from_ast(domain_record.get("start")),
        _source_affine_from_ast(domain_record.get("end")),
    )
    return binder, domain


def _scale_source_components(
    components: tuple[IndexedAffineComponent, ...], factor: int | Fraction
) -> tuple[IndexedAffineComponent, ...]:
    return tuple(
        IndexedAffineComponent(
            tuple(
                IndexedAffineTerm(term.declaration, term.indices, _fraction(term.coefficient) * factor)
                for term in component.terms
            ),
            _fraction(component.constant) * factor,
            component.sum_binders,
            component.sum_domains,
        )
        for component in components
    )


def _domain_to_record(domain: SymbolicDomain) -> dict[str, Any]:
    return {
        "name": domain.name,
        "lower": _affine_to_record(domain.lower),
        "upper": _affine_to_record(domain.upper),
        "nonempty": domain.nonempty,
    }


def _domain_from_record(record: Mapping[str, Any]) -> SymbolicDomain:
    return SymbolicDomain(
        record["name"],
        _affine_from_record(record["lower"]),
        _affine_from_record(record["upper"]),
        bool(record.get("nonempty", False)),
    )


def _affine_to_record(expression: SymbolicAffineExpression) -> dict[str, Any]:
    return {
        "coefficients": [[name, str(coefficient)] for name, coefficient in expression.coefficients],
        "constant": str(expression.constant),
    }


def _affine_from_record(record: Mapping[str, Any]) -> SymbolicAffineExpression:
    return SymbolicAffineExpression(
        tuple((name, coefficient) for name, coefficient in record["coefficients"]), record["constant"]
    )


def _rename_indexed_expression_symbols(
    expression: IndexedAffineExpression,
    symbol_mapping: dict[str, str],
) -> IndexedAffineExpression:
    terms = tuple(
        IndexedAffineTerm(
            term.declaration,
            tuple(_rename_index_term_mapping(index, symbol_mapping) for index in term.indices),
            term.coefficient,
        )
        for term in expression.terms
    )
    sum_domains = tuple(
        SymbolicDomain(
            domain.name,
            domain.lower.rename_symbols(symbol_mapping),
            domain.upper.rename_symbols(symbol_mapping),
        )
        for domain in expression.sum_domains
    )
    return IndexedAffineExpression(terms, expression.constant, expression.sum_binders, sum_domains)


def _rename_index_term_mapping(index: str, symbol_mapping: dict[str, str]) -> str:
    match = re.fullmatch(r"(-?)([A-Za-z_]\w*)(\s*[+-]\s*[0-9]+)?", index)
    if match is None or match.group(2) not in symbol_mapping:
        return index
    return match.group(1) + symbol_mapping[match.group(2)] + (match.group(3) or "")


__all__ = [
    "FactProvenance",
    "IndexedAffineComponent",
    "IndexedAffineExpression",
    "IndexedAffineFamily",
    "IndexedAffineTerm",
    "IndexedLinearPredicate",
    "IndexedObservablePair",
    "IndexedParameterFact",
    "IndexedScopedAffineExpression",
    "ParameterContract",
    "ParameterFact",
    "PredicateRelation",
    "SymbolicAffineExpression",
    "SymbolicDomain",
    "SymbolicPredicate",
    "indexed_observable_pair_from_record",
    "indexed_observable_pair_to_record",
    "parameter_contract_from_record",
    "parameter_contract_to_record",
]

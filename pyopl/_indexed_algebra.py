"""Binder-aware affine normalization for a deliberately small indexed fragment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Collection, Literal, Mapping

import sympy as sp

from pyopl._abstract_algebra import AlgebraicProof, UnsupportedAlgebra


@dataclass(frozen=True)
class Declaration:
    name: str
    kind: Literal["parameter", "variable", "domain"]
    value_type: str
    dimensions: tuple[str, ...]
    node: Mapping[str, Any]


@dataclass(frozen=True)
class IndexTerm:
    kind: Literal["binder", "number", "declaration", "arithmetic", "negate"]
    value: Any


@dataclass(frozen=True)
class DecisionAtom:
    declaration: str
    indices: tuple[IndexTerm, ...]


@dataclass
class IndexedAffineExpression:
    constant: sp.Expr
    terms: dict[DecisionAtom, sp.Expr]


@dataclass(frozen=True)
class QuantifiedExpression:
    domains: tuple[str, ...]
    filter: Any
    body: tuple[sp.Expr, tuple[tuple[DecisionAtom, sp.Expr], ...]] | QuantifiedExpression


def prove_indexed_equivalence(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    *,
    parameter_mapping: Mapping[str, str] | None = None,
    variable_mapping: Mapping[str, str] | None = None,
    left_auxiliaries: Collection[str] = (),
    right_auxiliaries: Collection[str] = (),
    max_rewrite_iterations: int = 12,
) -> AlgebraicProof | None:
    """Prove the supported indexed affine schemas equal, or decline the route.

    ``None`` means neither model uses indexed algebra and lets scalar lowering
    proceed. Unsupported indexed syntax raises ``UnsupportedAlgebra`` so the
    public API preserves its inconclusive outcome.
    """

    left = _declarations(left_ast)
    right = _declarations(right_ast)
    if not any(declaration.dimensions for declaration in left.values()) and not any(
        declaration.dimensions for declaration in right.values()
    ):
        return None
    if left_auxiliaries or right_auxiliaries:
        raise UnsupportedAlgebra("indexed algebra does not support auxiliary partitions")
    if max_rewrite_iterations <= 0:
        return AlgebraicProof(
            "unknown",
            "symbolically_normalized",
            "indexed normalization rewrite limit reached",
            budget_exhausted=True,
        )

    candidates = _declaration_mappings(left, right, parameter_mapping or {}, variable_mapping or {})
    if not candidates:
        raise UnsupportedAlgebra("no compatible indexed declaration mapping found")
    for mapping in candidates[:256]:
        try:
            budget = [max_rewrite_iterations * 100]
            if _canonical_model(left_ast, left, {}, budget) == _canonical_model(right_ast, right, _invert(mapping), budget):
                steps = [
                    "alpha-normalized indexed declarations and binders",
                    "normalized indexed affine expressions",
                    "normalized pointwise affine constraints",
                    "lifted equality through alpha-normalized quantifiers",
                ]
                if _contains_nested_sum(left_ast) or _contains_nested_sum(right_ast):
                    steps.append("alpha-normalized nested indexed binders")
                if _contains_filter(left_ast) or _contains_filter(right_ast):
                    steps.append("canonicalized indexed quantifier filters")
                if _contains_complex_index(left_ast) or _contains_complex_index(right_ast):
                    steps.append("preserved complete indexed access expressions")
                return AlgebraicProof(
                    "equivalent",
                    "symbolically_normalized",
                    "indexed affine schemas have equal canonical forms",
                    steps=tuple(steps),
                    variable_mapping=tuple(
                        sorted((name, target) for name, target in mapping.items() if left[name].kind == "variable")
                    ),
                    parameter_mapping=tuple(
                        sorted((name, target) for name, target in mapping.items() if left[name].kind == "parameter")
                    ),
                )
        except UnsupportedAlgebra as exc:
            if "rewrite limit" in str(exc):
                return AlgebraicProof(
                    "unknown",
                    "symbolically_normalized",
                    str(exc),
                    budget_exhausted=True,
                )
            continue
    if len(candidates) > 256:
        raise UnsupportedAlgebra("indexed declaration mapping search limit reached")
    raise UnsupportedAlgebra("indexed affine schemas could not be normalized to the same supported form")


def _declarations(ast: Mapping[str, Any]) -> dict[str, Declaration]:
    declarations: dict[str, Declaration] = {}
    for node in ast.get("declarations", []):
        if not isinstance(node, Mapping) or not isinstance(node.get("name"), str):
            raise UnsupportedAlgebra("indexed declaration must be a named object")
        node_type = str(node.get("type"))
        if node_type == "tuple_type" or node_type in {"set_of_tuples", "tuple_array"}:
            raise UnsupportedAlgebra("indexed algebra does not support tuple declarations")
        if node_type in {"parameter_inline_indexed", "parameter_computed_indexed"}:
            raise UnsupportedAlgebra("indexed algebra does not support computed indexed parameters")
        if node.get("lower_bound") is not None or node.get("upper_bound") is not None:
            raise UnsupportedAlgebra("indexed algebra does not support explicit indexed decision bounds")
        if node_type.startswith("range_declaration") or node_type in {"set_declaration", "typed_set"}:
            kind: Literal["parameter", "variable", "domain"] = "domain"
        elif node_type == "dvar" or node_type == "dvar_indexed":
            kind = "variable"
        elif node_type.startswith("parameter"):
            kind = "parameter"
        else:
            raise UnsupportedAlgebra(f"unsupported indexed declaration: {node_type}")
        dimensions = tuple(_dimension_name(dimension) for dimension in node.get("dimensions", ()))
        declarations[str(node["name"])] = Declaration(
            str(node["name"]), kind, str(node.get("var_type", node_type)), dimensions, node
        )
    return declarations


def _dimension_name(node: Any) -> str:
    if not isinstance(node, Mapping) or not isinstance(node.get("name"), str):
        raise UnsupportedAlgebra("indexed algebra requires named declaration dimensions")
    return str(node["name"])


def _declaration_mappings(
    left: Mapping[str, Declaration],
    right: Mapping[str, Declaration],
    parameter_mapping: Mapping[str, str],
    variable_mapping: Mapping[str, str],
) -> list[dict[str, str]]:
    if len(left) != len(right):
        return []
    prescribed = dict(parameter_mapping) | dict(variable_mapping)
    groups: list[tuple[list[str], list[str]]] = []
    signatures = sorted({_signature(declaration) for declaration in left.values()})
    for signature in signatures:
        left_names = sorted(name for name, declaration in left.items() if _signature(declaration) == signature)
        right_names = sorted(name for name, declaration in right.items() if _signature(declaration) == signature)
        if len(left_names) != len(right_names):
            return []
        groups.append((left_names, right_names))

    choices: list[list[dict[str, str]]] = []
    for left_names, right_names in groups:
        choices.append(
            [
                dict(zip(left_names, ordering, strict=True))
                for ordering in permutations(right_names)
                if all(name not in prescribed or prescribed[name] == target for name, target in zip(left_names, ordering))
            ]
        )
    mappings: list[dict[str, str]] = []
    for parts in product(*choices):
        mapping = {name: target for part in parts for name, target in part.items()}
        if _mapped_declarations_match(left, right, mapping):
            mappings.append(mapping)
            if len(mappings) > 256:
                break
    return mappings


def _signature(declaration: Declaration) -> tuple[str, str, int]:
    return declaration.kind, declaration.value_type, len(declaration.dimensions)


def _mapped_declarations_match(
    left: Mapping[str, Declaration], right: Mapping[str, Declaration], mapping: Mapping[str, str]
) -> bool:
    inverse = _invert(mapping)
    for name, declaration in left.items():
        other = right[mapping[name]]
        if tuple(mapping.get(domain, domain) for domain in declaration.dimensions) != other.dimensions:
            return False
        if declaration.kind == "domain":
            try:
                if _canonical_domain(declaration.node, {}) != _canonical_domain(other.node, inverse):
                    return False
            except UnsupportedAlgebra:
                return False
    return True


def _invert(mapping: Mapping[str, str]) -> dict[str, str]:
    return {target: source for source, target in mapping.items()}


def _canonical_model(
    ast: Mapping[str, Any],
    declarations: Mapping[str, Declaration],
    rename: Mapping[str, str],
    budget: list[int],
) -> tuple[Any, ...]:
    _consume_budget(budget)
    objective = ast.get("objective")
    if not isinstance(objective, Mapping) or objective.get("type") not in {"minimize", "maximize"}:
        raise UnsupportedAlgebra("unsupported indexed objective")
    objective_form = (
        objective.get("type"),
        _canonical_expression(objective.get("expression"), declarations, rename, {}, budget),
    )
    constraints = tuple(
        sorted(
            (_canonical_constraint(node, declarations, rename, {}, budget) for node in ast.get("constraints", [])),
            key=repr,
        )
    )
    return objective_form, constraints


def _canonical_expression(
    node: Any,
    declarations: Mapping[str, Declaration],
    rename: Mapping[str, str],
    binders: Mapping[str, int],
    budget: list[int],
    next_binder_id: int = 0,
) -> Any:
    _consume_budget(budget)
    if isinstance(node, Mapping) and node.get("type") == "sum":
        return _quantified(node, declarations, rename, binders, budget, next_binder_id)
    return _freeze_affine(_affine(node, declarations, rename, binders, budget))


def _quantified(
    node: Mapping[str, Any],
    declarations: Mapping[str, Declaration],
    rename: Mapping[str, str],
    outer_binders: Mapping[str, int],
    budget: list[int],
    next_binder_id: int,
) -> QuantifiedExpression:
    _consume_budget(budget)
    iterators = node.get("iterators")
    if not isinstance(iterators, list) or not iterators:
        raise UnsupportedAlgebra("indexed quantifier requires binders")
    binders = dict(outer_binders)
    domains: list[str] = []
    for iterator in iterators:
        if not isinstance(iterator, Mapping) or not isinstance(iterator.get("iterator"), str):
            raise UnsupportedAlgebra("malformed indexed binder")
        domain = iterator.get("range")
        if not isinstance(domain, Mapping) or domain.get("type") not in {"named_range", "named_set"}:
            raise UnsupportedAlgebra("indexed algebra requires named binder domains")
        domain_name = domain.get("name")
        if not isinstance(domain_name, str):
            raise UnsupportedAlgebra("indexed binder domain must be named")
        domains.append(rename.get(domain_name, domain_name))
        binders[str(iterator["iterator"])] = next_binder_id
        next_binder_id += 1
    predicate = _canonical_predicate(node.get("index_constraint"), rename, binders)
    body = node.get("expression")
    if isinstance(body, Mapping) and body.get("type") == "sum":
        return QuantifiedExpression(
            tuple(domains), predicate, _quantified(body, declarations, rename, binders, budget, next_binder_id)
        )
    if isinstance(body, Mapping) and body.get("type") == "forall_constraint":
        raise UnsupportedAlgebra("indexed algebra does not support nested forall constraints")
    return QuantifiedExpression(
        tuple(domains), predicate, _freeze_affine(_affine(body, declarations, rename, binders, budget))
    )


def _contains_nested_sum(node: Any, inside_sum: bool = False) -> bool:
    if isinstance(node, Mapping):
        is_sum = node.get("type") == "sum"
        if inside_sum and is_sum:
            return True
        return any(_contains_nested_sum(value, inside_sum or is_sum) for value in node.values())
    if isinstance(node, list):
        return any(_contains_nested_sum(value, inside_sum) for value in node)
    return False


def _contains_filter(node: Any) -> bool:
    if isinstance(node, Mapping):
        if node.get("type") in {"sum", "forall_constraint"} and node.get("index_constraint") is not None:
            return True
        return any(_contains_filter(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_filter(value) for value in node)
    return False


def _canonical_predicate(node: Any, rename: Mapping[str, str], binders: Mapping[str, int]) -> Any:
    if node is None:
        return True
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra("malformed indexed quantifier filter")
    if node.get("type") == "parenthesized_expression":
        return _canonical_predicate(node.get("expression"), rename, binders)
    if node.get("type") == "and":
        operands = _predicate_conjuncts(node, rename, binders)
        return "and", tuple(sorted(operands, key=repr))
    if node.get("type") == "binop" and node.get("op") in {"<", "<=", "==", "!=", ">=", ">"}:
        return (
            "compare",
            node.get("op"),
            _canonical_predicate_term(node.get("left"), rename, binders),
            _canonical_predicate_term(node.get("right"), rename, binders),
        )
    raise UnsupportedAlgebra("indexed algebra supports comparison and conjunction filters only")


def _predicate_conjuncts(node: Mapping[str, Any], rename: Mapping[str, str], binders: Mapping[str, int]) -> list[Any]:
    operands: list[Any] = []
    for child in (node.get("left"), node.get("right")):
        if isinstance(child, Mapping) and child.get("type") == "and":
            operands.extend(_predicate_conjuncts(child, rename, binders))
        else:
            operands.append(_canonical_predicate(child, rename, binders))
    return operands


def _canonical_predicate_term(node: Any, rename: Mapping[str, str], binders: Mapping[str, int]) -> Any:
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra("malformed indexed filter term")
    node_type = node.get("type")
    if node_type == "parenthesized_expression":
        return _canonical_predicate_term(node.get("expression"), rename, binders)
    if node_type == "number":
        return "number", str(node.get("value"))
    if node_type == "name" and isinstance(node.get("value"), str):
        name = str(node["value"])
        if name in binders:
            return "binder", binders[name]
        return "declaration", rename.get(name, name)
    if node_type == "unary" and node.get("op") == "-":
        return "negate", _canonical_predicate_term(node.get("operand"), rename, binders)
    if node_type == "binop" and node.get("op") in {"+", "-", "*"}:
        return (
            "arithmetic",
            node.get("op"),
            _canonical_predicate_term(node.get("left"), rename, binders),
            _canonical_predicate_term(node.get("right"), rename, binders),
        )
    raise UnsupportedAlgebra("unsupported indexed filter term")


def _canonical_constraint(
    node: Any,
    declarations: Mapping[str, Declaration],
    rename: Mapping[str, str],
    binders: Mapping[str, int],
    budget: list[int],
) -> Any:
    _consume_budget(budget)
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra("malformed indexed constraint")
    if node.get("type") == "forall_constraint":
        body = node.get("constraint")
        if body is None:
            constraints = node.get("constraints")
            if not isinstance(constraints, list) or len(constraints) != 1:
                raise UnsupportedAlgebra("indexed algebra supports one constraint per forall")
            body = constraints[0]
        quantifier = _quantified(
            {**node, "expression": _constraint_residual(body)},
            declarations,
            rename,
            binders,
            budget,
            max(binders.values(), default=-1) + 1,
        )
        return "forall", _constraint_sense(body), quantifier
    _constraint_sense(node)
    left = node.get("left")
    right = node.get("right")
    if node.get("op") == ">=":
        left, right = right, left
    if _is_quantified_expression(left) or _is_quantified_expression(right):
        return (
            "constraint",
            "=" if node.get("op") == "==" else "<=",
            _canonical_expression(left, declarations, rename, binders, budget),
            _canonical_expression(right, declarations, rename, binders, budget),
        )
    return (
        "constraint",
        _constraint_sense(node),
        _freeze_affine(_affine(_constraint_residual(node), declarations, rename, binders, budget)),
    )


def _is_quantified_expression(node: Any) -> bool:
    return isinstance(node, Mapping) and node.get("type") == "sum"


def _constraint_sense(node: Any) -> str:
    if not isinstance(node, Mapping) or node.get("type") != "constraint" or node.get("op") not in {"<=", ">=", "=="}:
        raise UnsupportedAlgebra("indexed algebra supports affine comparison constraints only")
    return "=" if node.get("op") == "==" else "<="


def _constraint_residual(node: Any) -> Mapping[str, Any]:
    sense = _constraint_sense(node)
    assert isinstance(node, Mapping)
    left, right = node.get("left"), node.get("right")
    if node.get("op") == ">=":
        left, right = right, left
    return {"type": "binop", "op": "-", "left": left, "right": right, "sem_type": sense}


def _affine(
    node: Any,
    declarations: Mapping[str, Declaration],
    rename: Mapping[str, str],
    binders: Mapping[str, int],
    budget: list[int],
) -> IndexedAffineExpression:
    _consume_budget(budget)
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra("malformed indexed expression")
    node_type = node.get("type")
    if node_type == "parenthesized_expression":
        return _affine(node.get("expression"), declarations, rename, binders, budget)
    if node_type == "number":
        return IndexedAffineExpression(sp.Rational(str(node.get("value"))), {})
    if node_type in {"name", "indexed_name"}:
        name = node.get("value") if node_type == "name" else node.get("name")
        if not isinstance(name, str) or name not in declarations:
            raise UnsupportedAlgebra(f"unknown indexed symbol: {name}")
        declaration = declarations[name]
        indices = tuple(_index_term(index, binders, rename) for index in node.get("dimensions", ()))
        if len(indices) != len(declaration.dimensions):
            raise UnsupportedAlgebra(f"indexed rank mismatch for {name}")
        canonical_name = rename.get(name, name)
        if declaration.kind == "variable":
            return IndexedAffineExpression(sp.S.Zero, {DecisionAtom(canonical_name, indices): sp.S.One})
        if declaration.kind != "parameter":
            raise UnsupportedAlgebra("domain declarations cannot appear as arithmetic values")
        return IndexedAffineExpression(_coefficient_atom(canonical_name, indices), {})
    if node_type == "unary" and node.get("op") == "-":
        return _scale(_affine(node.get("operand"), declarations, rename, binders, budget), -1)
    if node_type != "binop":
        raise UnsupportedAlgebra(f"unsupported indexed expression: {node_type}")
    left = _affine(node.get("left"), declarations, rename, binders, budget)
    right = _affine(node.get("right"), declarations, rename, binders, budget)
    operator = node.get("op")
    if operator == "+":
        return _add(left, right)
    if operator == "-":
        return _add(left, _scale(right, -1))
    if operator == "*":
        if left.terms and right.terms:
            raise UnsupportedAlgebra("indexed algebra supports affine decision expressions only")
        if left.terms:
            return _multiply_by_coefficient(left, right.constant)
        if right.terms:
            return _multiply_by_coefficient(right, left.constant)
        return IndexedAffineExpression(sp.expand(left.constant * right.constant), {})
    if operator == "/":
        raise UnsupportedAlgebra("division in indexed expressions requires side-condition proving")
    raise UnsupportedAlgebra(f"unsupported indexed operator: {operator}")


def _index_term(node: Any, binders: Mapping[str, int], rename: Mapping[str, str]) -> IndexTerm:
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra("malformed indexed access expression")
    node_type = node.get("type")
    if node_type == "parenthesized_expression":
        return _index_term(node.get("expression"), binders, rename)
    if node_type == "name_reference_index":
        name = node.get("name")
        if not isinstance(name, str) or name not in binders:
            raise UnsupportedAlgebra(f"unbound indexed iterator: {name}")
        return IndexTerm("binder", binders[name])
    if node_type == "name" and isinstance(node.get("value"), str):
        name = str(node["value"])
        if name in binders:
            return IndexTerm("binder", binders[name])
        return IndexTerm("declaration", rename.get(name, name))
    if node_type == "number":
        return IndexTerm("number", str(node.get("value")))
    if node_type == "unary" and node.get("op") == "-":
        return IndexTerm("negate", _index_term(node.get("operand"), binders, rename))
    if node_type == "binop" and node.get("op") in {"+", "-", "*"}:
        return IndexTerm(
            "arithmetic",
            (
                node.get("op"),
                _index_term(node.get("left"), binders, rename),
                _index_term(node.get("right"), binders, rename),
            ),
        )
    raise UnsupportedAlgebra("unsupported indexed access expression")


def _contains_complex_index(node: Any) -> bool:
    if isinstance(node, Mapping):
        if node.get("type") == "indexed_name":
            dimensions = node.get("dimensions", ())
            if any(
                not isinstance(dimension, Mapping) or dimension.get("type") != "name_reference_index"
                for dimension in dimensions
            ):
                return True
        return any(_contains_complex_index(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_complex_index(value) for value in node)
    return False


def _coefficient_atom(name: str, indices: tuple[IndexTerm, ...]) -> sp.Symbol:
    suffix = "".join(f"[{index!r}]" for index in indices)
    return sp.Symbol(f"parameter:{name}{suffix}", real=True)


def _add(left: IndexedAffineExpression, right: IndexedAffineExpression) -> IndexedAffineExpression:
    terms = dict(left.terms)
    for atom, coefficient in right.terms.items():
        terms[atom] = sp.expand(terms.get(atom, sp.S.Zero) + coefficient)
        if terms[atom] == 0:
            del terms[atom]
    return IndexedAffineExpression(sp.expand(left.constant + right.constant), terms)


def _scale(expression: IndexedAffineExpression, coefficient: Any) -> IndexedAffineExpression:
    return IndexedAffineExpression(
        sp.expand(expression.constant * coefficient),
        {atom: sp.expand(value * coefficient) for atom, value in expression.terms.items()},
    )


def _multiply_by_coefficient(expression: IndexedAffineExpression, coefficient: sp.Expr) -> IndexedAffineExpression:
    return _scale(expression, coefficient)


def _freeze_affine(expression: IndexedAffineExpression) -> tuple[sp.Expr, tuple[tuple[DecisionAtom, sp.Expr], ...]]:
    return sp.expand(expression.constant), tuple(sorted(expression.terms.items(), key=lambda item: repr(item[0])))


def _canonical_domain(node: Mapping[str, Any], rename: Mapping[str, str]) -> Any:
    node_type = node.get("type")
    if node_type == "range_declaration_external":
        return "external_range"
    if node_type == "range_declaration_inline":
        return "range", _canonical_scalar(node.get("start"), rename), _canonical_scalar(node.get("end"), rename)
    if node_type in {"set_declaration", "typed_set"}:
        return "set", str(node.get("var_type", "unknown"))
    raise UnsupportedAlgebra(f"unsupported indexed domain: {node_type}")


def _canonical_scalar(node: Any, rename: Mapping[str, str]) -> Any:
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra("malformed domain expression")
    if node.get("type") == "number":
        return "number", str(node.get("value"))
    if node.get("type") == "name" and isinstance(node.get("value"), str):
        return "name", rename.get(str(node["value"]), str(node["value"]))
    raise UnsupportedAlgebra("indexed domain bounds must be scalar names or numbers")


def _consume_budget(budget: list[int]) -> None:
    budget[0] -= 1
    if budget[0] < 0:
        raise UnsupportedAlgebra("indexed normalization rewrite limit reached")


__all__ = [
    "DecisionAtom",
    "IndexTerm",
    "IndexedAffineExpression",
    "QuantifiedExpression",
    "prove_indexed_equivalence",
]

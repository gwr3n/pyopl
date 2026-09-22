"""Binder-aware affine normalization for a deliberately small indexed fragment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Collection, Literal, Mapping, Sequence

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
    kind: Literal["binder", "number", "declaration", "application", "arithmetic", "negate"]
    value: Any


@dataclass(frozen=True)
class DecisionAtom:
    declaration: str
    indices: tuple[IndexTerm, ...]


@dataclass(frozen=True)
class DomainTerm:
    kind: Literal["named", "range"]
    value: Any


@dataclass
class IndexedAffineExpression:
    constant: sp.Expr
    terms: dict[DecisionAtom, sp.Expr]


@dataclass(frozen=True)
class QuantifiedExpression:
    domains: tuple[DomainTerm, ...]
    filter: Any
    body: tuple[sp.Expr, tuple[tuple[DecisionAtom, sp.Expr], ...]] | QuantifiedExpression


@dataclass(frozen=True)
class QuantifiedConstraint:
    domains: tuple[DomainTerm, ...]
    filter: Any
    body: Any


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
    return _prove_mapped_indexed_equivalence(
        left_ast,
        right_ast,
        left,
        right,
        candidates,
        max_rewrite_iterations,
    )


def _prove_mapped_indexed_equivalence(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    left: Mapping[str, Declaration],
    right: Mapping[str, Declaration],
    candidates: Sequence[Mapping[str, str]],
    max_rewrite_iterations: int,
) -> AlgebraicProof:
    first_mismatch: tuple[Any, Any] | None = None
    for mapping in candidates[:256]:
        try:
            left_canonical, right_canonical = _canonical_model_pair(
                left_ast,
                right_ast,
                left,
                right,
                mapping,
                max_rewrite_iterations,
            )
            if left_canonical == right_canonical:
                return _equivalent_indexed_proof(left_ast, right_ast, left, mapping)
            if first_mismatch is None:
                first_mismatch = left_canonical, right_canonical
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
    if first_mismatch is not None:
        left_canonical, right_canonical = first_mismatch
        raise UnsupportedAlgebra(
            "indexed affine schemas differ after supported normalization; "
            f"left canonical: {render_indexed_ir(left_canonical)}; "
            f"right canonical: {render_indexed_ir(right_canonical)}"
        )
    raise UnsupportedAlgebra("indexed affine schemas could not be normalized to the same supported form")


def _canonical_model_pair(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    left: Mapping[str, Declaration],
    right: Mapping[str, Declaration],
    mapping: Mapping[str, str],
    max_rewrite_iterations: int,
) -> tuple[Any, Any]:
    budget = [max_rewrite_iterations * 100]
    return (
        _canonical_model(left_ast, left, {}, budget),
        _canonical_model(right_ast, right, _invert(mapping), budget),
    )


def _equivalent_indexed_proof(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    left: Mapping[str, Declaration],
    mapping: Mapping[str, str],
) -> AlgebraicProof:
    return AlgebraicProof(
        "equivalent",
        "symbolically_normalized",
        "indexed affine schemas have equal canonical forms",
        steps=tuple(_indexed_proof_steps(left_ast, right_ast)),
        variable_mapping=_mapping_for_kind(mapping, left, "variable"),
        parameter_mapping=_mapping_for_kind(mapping, left, "parameter"),
    )


def _indexed_proof_steps(left_ast: Mapping[str, Any], right_ast: Mapping[str, Any]) -> list[str]:
    steps = [
        "alpha-normalized indexed declarations and binders",
        "normalized indexed affine expressions",
        "normalized pointwise affine constraints",
        "lifted equality through alpha-normalized quantifiers",
    ]
    optional_steps = (
        (_contains_nested_sum, "alpha-normalized nested indexed binders"),
        (_contains_nested_forall, "alpha-normalized nested forall constraints"),
        (_contains_filter, "canonicalized indexed quantifier filters"),
        (_contains_complex_index, "preserved complete indexed access expressions"),
        (_contains_parameter_selected_index, "preserved parameter-selected index applications"),
        (_contains_dependent_domain, "preserved dependent quantifier domains"),
        (_contains_multi_or_nested_binders, "flattened ordered indexed binders"),
        (_contains_reorderable_binders, "reordered independent indexed binders"),
        (_contains_split_sums, "fused indexed sums with identical domains and filters"),
    )
    for predicate, description in optional_steps:
        if predicate(left_ast) or predicate(right_ast):
            steps.append(description)
    return steps


def _mapping_for_kind(
    mapping: Mapping[str, str],
    declarations: Mapping[str, Declaration],
    kind: Literal["parameter", "variable"],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, target) for name, target in mapping.items() if declarations[name].kind == kind))


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
    groups = _declaration_groups(left, right)
    if groups is None:
        return []
    choices = [_mapping_choices(left_names, right_names, prescribed) for left_names, right_names in groups]
    mappings: list[dict[str, str]] = []
    for parts in product(*choices):
        mapping = {name: target for part in parts for name, target in part.items()}
        if _mapped_declarations_match(left, right, mapping):
            mappings.append(mapping)
            if len(mappings) > 256:
                break
    return mappings


def _declaration_groups(
    left: Mapping[str, Declaration], right: Mapping[str, Declaration]
) -> list[tuple[list[str], list[str]]] | None:
    groups: list[tuple[list[str], list[str]]] = []
    for signature in sorted({_signature(declaration) for declaration in left.values()}):
        left_names = sorted(name for name, declaration in left.items() if _signature(declaration) == signature)
        right_names = sorted(name for name, declaration in right.items() if _signature(declaration) == signature)
        if len(left_names) != len(right_names):
            return None
        groups.append((left_names, right_names))
    return groups


def _mapping_choices(
    left_names: Sequence[str], right_names: Sequence[str], prescribed: Mapping[str, str]
) -> list[dict[str, str]]:
    return [
        dict(zip(left_names, ordering, strict=True))
        for ordering in permutations(right_names)
        if all(name not in prescribed or prescribed[name] == target for name, target in zip(left_names, ordering))
    ]


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
        if declaration.kind == "parameter" and not _mapped_parameter_definitions_match(
            declaration.node, other.node, inverse
        ):
            return False
    return True


def _mapped_parameter_definitions_match(
    left: Mapping[str, Any], right: Mapping[str, Any], rename: Mapping[str, str]
) -> bool:
    left_has_value = "value" in left and left.get("value") is not None
    right_has_value = "value" in right and right.get("value") is not None
    if left_has_value != right_has_value:
        return False
    if not left_has_value:
        return True
    left_value = left.get("value")
    right_value = right.get("value")
    if isinstance(left_value, Mapping) or isinstance(right_value, Mapping):
        try:
            return _canonical_index(left_value, {}, {}, [100]) == _canonical_index(
                right_value, rename, {}, [100]
            )
        except UnsupportedAlgebra:
            return False
    return left_value == right_value


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
    if isinstance(node, Mapping) and node.get("type") == "binop" and node.get("op") in {"+", "-"}:
        left_node = node.get("left")
        right_node = node.get("right")
        if _contains_sum(left_node) or _contains_sum(right_node):
            left = _canonical_expression(left_node, declarations, rename, binders, budget, next_binder_id)
            right = _canonical_expression(right_node, declarations, rename, binders, budget, next_binder_id)
            return _combine_quantified_expressions(left, right, -1 if node.get("op") == "-" else 1)
    return _freeze_affine(_affine(node, declarations, rename, binders, budget))


def _combine_quantified_expressions(left: Any, right: Any, right_sign: int) -> QuantifiedExpression:
    if not isinstance(left, QuantifiedExpression) or not isinstance(right, QuantifiedExpression):
        raise UnsupportedAlgebra("indexed sum fusion requires quantified expressions on both sides")
    if left.domains != right.domains or left.filter != right.filter:
        raise UnsupportedAlgebra("indexed sum fusion requires identical domains and filters")
    if isinstance(left.body, QuantifiedExpression) or isinstance(right.body, QuantifiedExpression):
        body = _combine_quantified_expressions(left.body, right.body, right_sign)
    elif _is_frozen_affine(left.body) and _is_frozen_affine(right.body):
        body = _combine_frozen_affine(left.body, right.body, right_sign)
    else:
        raise UnsupportedAlgebra("indexed sum fusion requires matching quantifier depth")
    return QuantifiedExpression(left.domains, left.filter, body)


def _combine_frozen_affine(left: Any, right: Any, right_sign: int) -> Any:
    left_constant, left_terms = left
    right_constant, right_terms = right
    terms = dict(left_terms)
    for atom, coefficient in right_terms:
        terms[atom] = sp.expand(terms.get(atom, sp.S.Zero) + right_sign * coefficient)
        if terms[atom] == 0:
            del terms[atom]
    return (
        sp.expand(left_constant + right_sign * right_constant),
        tuple(sorted(terms.items(), key=lambda item: repr(item[0]))),
    )


def _contains_sum(node: Any) -> bool:
    if isinstance(node, Mapping):
        return node.get("type") == "sum" or any(_contains_sum(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_sum(value) for value in node)
    return False


def _contains_split_sums(node: Any) -> bool:
    if isinstance(node, Mapping):
        if node.get("type") == "binop" and node.get("op") in {"+", "-"}:
            if _contains_sum(node.get("left")) and _contains_sum(node.get("right")):
                return True
        return any(_contains_split_sums(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_split_sums(value) for value in node)
    return False


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
    iterators = _canonical_iterator_order(iterators, node.get("expression"), node.get("index_constraint"))
    binders = dict(outer_binders)
    layers: list[tuple[DomainTerm, int]] = []
    for iterator in iterators:
        if not isinstance(iterator, Mapping) or not isinstance(iterator.get("iterator"), str):
            raise UnsupportedAlgebra("malformed indexed binder")
        domain = _canonical_iterator_domain(iterator.get("range"), binders, rename)
        layers.append((domain, next_binder_id))
        binders[str(iterator["iterator"])] = next_binder_id
        next_binder_id += 1
    predicate = _canonical_predicate(node.get("index_constraint"), rename, binders)
    body = node.get("expression")
    if isinstance(body, Mapping) and body.get("type") == "sum":
        canonical_body: Any = _quantified(body, declarations, rename, binders, budget, next_binder_id)
    elif isinstance(body, Mapping) and body.get("type") == "forall_constraint":
        raise UnsupportedAlgebra("indexed algebra does not support nested forall constraints")
    else:
        canonical_body = _freeze_affine(_affine(body, declarations, rename, binders, budget))
    for position, (domain, _) in enumerate(reversed(layers)):
        canonical_body = QuantifiedExpression((domain,), predicate if position == 0 else True, canonical_body)
    return canonical_body


def _canonical_iterator_domain(node: Any, binders: Mapping[str, int], rename: Mapping[str, str]) -> DomainTerm:
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra("malformed indexed binder domain")
    node_type = node.get("type")
    if node_type in {"named_range", "named_set"}:
        name = node.get("name")
        if not isinstance(name, str):
            raise UnsupportedAlgebra("indexed binder domain must be named")
        return DomainTerm("named", rename.get(name, name))
    if node_type == "range_specifier":
        return DomainTerm(
            "range",
            (
                _index_term(node.get("start"), binders, rename),
                _index_term(node.get("end"), binders, rename),
            ),
        )
    raise UnsupportedAlgebra("indexed algebra supports named and scalar range binder domains only")


def _canonical_iterator_order(iterators: list[Any], body: Any, predicate: Any) -> list[Any]:
    if len(iterators) < 2 or not _iterators_are_independent(iterators):
        return iterators
    names = [iterator.get("iterator") for iterator in iterators if isinstance(iterator, Mapping)]
    if len(names) != len(iterators) or not all(isinstance(name, str) for name in names):
        return iterators
    return sorted(
        iterators,
        key=lambda iterator: (
            _binder_occurrence_paths(body, str(iterator["iterator"])),
            _binder_occurrence_paths(predicate, str(iterator["iterator"])),
        ),
    )


def _iterators_are_independent(iterators: list[Any]) -> bool:
    names = {
        str(iterator["iterator"])
        for iterator in iterators
        if isinstance(iterator, Mapping) and isinstance(iterator.get("iterator"), str)
    }
    return all(not (_source_names(iterator.get("range")) & names) for iterator in iterators if isinstance(iterator, Mapping))


def _source_names(node: Any) -> frozenset[str]:
    if isinstance(node, Mapping):
        names: set[str] = set()
        if node.get("type") == "name" and isinstance(node.get("value"), str):
            names.add(str(node["value"]))
        if node.get("type") == "name_reference_index" and isinstance(node.get("name"), str):
            names.add(str(node["name"]))
        for value in node.values():
            names.update(_source_names(value))
        return frozenset(names)
    if isinstance(node, list):
        return frozenset().union(*(_source_names(value) for value in node))
    return frozenset()


def _binder_occurrence_paths(node: Any, name: str, path: tuple[str, ...] = ()) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    if isinstance(node, Mapping):
        if node.get("type") == "name" and node.get("value") == name:
            paths.append(path)
        if node.get("type") == "name_reference_index" and node.get("name") == name:
            paths.append(path)
        for key, value in node.items():
            if key not in {"sem_type", "iterator"}:
                paths.extend(_binder_occurrence_paths(value, name, path + (str(key),)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            paths.extend(_binder_occurrence_paths(value, name, path + (str(index),)))
    return tuple(paths)


def _contains_nested_sum(node: Any, inside_sum: bool = False) -> bool:
    if isinstance(node, Mapping):
        is_sum = node.get("type") == "sum"
        if inside_sum and is_sum:
            return True
        return any(_contains_nested_sum(value, inside_sum or is_sum) for value in node.values())
    if isinstance(node, list):
        return any(_contains_nested_sum(value, inside_sum) for value in node)
    return False


def _contains_nested_forall(node: Any, inside_forall: bool = False) -> bool:
    if isinstance(node, Mapping):
        is_forall = node.get("type") == "forall_constraint"
        if inside_forall and is_forall:
            return True
        return any(_contains_nested_forall(value, inside_forall or is_forall) for value in node.values())
    if isinstance(node, list):
        return any(_contains_nested_forall(value, inside_forall) for value in node)
    return False


def _contains_filter(node: Any) -> bool:
    if isinstance(node, Mapping):
        if node.get("type") in {"sum", "forall_constraint"} and node.get("index_constraint") is not None:
            return True
        return any(_contains_filter(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_filter(value) for value in node)
    return False


def _contains_dependent_domain(node: Any) -> bool:
    if isinstance(node, Mapping):
        if node.get("type") == "range_specifier":
            return True
        return any(_contains_dependent_domain(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_dependent_domain(value) for value in node)
    return False


def _contains_multi_or_nested_binders(node: Any) -> bool:
    if isinstance(node, Mapping):
        if node.get("type") in {"sum", "forall_constraint"}:
            iterators = node.get("iterators")
            if isinstance(iterators, list) and len(iterators) > 1:
                return True
        return any(_contains_multi_or_nested_binders(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_multi_or_nested_binders(value) for value in node)
    return False


def _contains_reorderable_binders(node: Any) -> bool:
    if isinstance(node, Mapping):
        if node.get("type") in {"sum", "forall_constraint"}:
            iterators = node.get("iterators")
            if isinstance(iterators, list) and len(iterators) > 1 and _iterators_are_independent(iterators):
                return True
        return any(_contains_reorderable_binders(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_reorderable_binders(value) for value in node)
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
        return _canonical_forall_constraint(
            node,
            declarations,
            rename,
            binders,
            budget,
            max(binders.values(), default=-1) + 1,
        )
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


def _canonical_forall_constraint(
    node: Mapping[str, Any],
    declarations: Mapping[str, Declaration],
    rename: Mapping[str, str],
    outer_binders: Mapping[str, int],
    budget: list[int],
    next_binder_id: int,
) -> QuantifiedConstraint:
    _consume_budget(budget)
    iterators = node.get("iterators")
    if not isinstance(iterators, list) or not iterators:
        raise UnsupportedAlgebra("indexed forall requires binders")
    body_node = node.get("constraint") if node.get("constraint") is not None else node.get("constraints")
    iterators = _canonical_iterator_order(iterators, body_node, node.get("index_constraint"))
    binders = dict(outer_binders)
    layers: list[DomainTerm] = []
    for iterator in iterators:
        if not isinstance(iterator, Mapping) or not isinstance(iterator.get("iterator"), str):
            raise UnsupportedAlgebra("malformed indexed forall binder")
        domain = _canonical_iterator_domain(iterator.get("range"), binders, rename)
        layers.append(domain)
        binders[str(iterator["iterator"])] = next_binder_id
        next_binder_id += 1

    body = node.get("constraint")
    if body is None:
        constraints = node.get("constraints")
        if not isinstance(constraints, list) or len(constraints) != 1:
            raise UnsupportedAlgebra("indexed algebra supports one constraint per forall")
        body = constraints[0]
    if not isinstance(body, Mapping):
        raise UnsupportedAlgebra("malformed indexed forall body")
    if body.get("type") == "forall_constraint":
        canonical_body: Any = _canonical_forall_constraint(body, declarations, rename, binders, budget, next_binder_id)
    else:
        canonical_body = _canonical_constraint(body, declarations, rename, binders, budget)
    predicate = _canonical_predicate(node.get("index_constraint"), rename, binders)
    for position, domain in enumerate(reversed(layers)):
        canonical_body = QuantifiedConstraint((domain,), predicate if position == 0 else True, canonical_body)
    return canonical_body


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
        return _affine_symbol(node, declarations, rename, binders)
    if node_type == "unary" and node.get("op") == "-":
        return _scale(_affine(node.get("operand"), declarations, rename, binders, budget), -1)
    if node_type != "binop":
        raise UnsupportedAlgebra(f"unsupported indexed expression: {node_type}")
    left = _affine(node.get("left"), declarations, rename, binders, budget)
    right = _affine(node.get("right"), declarations, rename, binders, budget)
    return _combine_affine(left, right, node.get("op"))


def _affine_symbol(
    node: Mapping[str, Any],
    declarations: Mapping[str, Declaration],
    rename: Mapping[str, str],
    binders: Mapping[str, int],
) -> IndexedAffineExpression:
    name = node.get("value") if node.get("type") == "name" else node.get("name")
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


def _combine_affine(left: IndexedAffineExpression, right: IndexedAffineExpression, operator: Any) -> IndexedAffineExpression:
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
    if node_type in {"name_reference_index", "name", "indexed_name"}:
        return _named_index_term(node, binders, rename)
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


def _named_index_term(node: Mapping[str, Any], binders: Mapping[str, int], rename: Mapping[str, str]) -> IndexTerm:
    node_type = node.get("type")
    if node_type == "name_reference_index":
        name = node.get("name")
        if not isinstance(name, str) or name not in binders:
            raise UnsupportedAlgebra(f"unbound indexed iterator: {name}")
        return IndexTerm("binder", binders[name])
    name = node.get("value") if node_type == "name" else node.get("name")
    if not isinstance(name, str):
        raise UnsupportedAlgebra("unsupported indexed access expression")
    if node_type == "name":
        if name in binders:
            return IndexTerm("binder", binders[name])
        return IndexTerm("declaration", rename.get(name, name))
    return IndexTerm(
        "application",
        (
            rename.get(name, name),
            tuple(_index_term(dimension, binders, rename) for dimension in node.get("dimensions", ())),
        ),
    )


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


def _contains_parameter_selected_index(node: Any, inside_dimension: bool = False) -> bool:
    if isinstance(node, Mapping):
        if inside_dimension and node.get("type") == "indexed_name":
            return True
        if node.get("type") == "indexed_name":
            return any(_contains_parameter_selected_index(dimension, True) for dimension in node.get("dimensions", ()))
        return any(_contains_parameter_selected_index(value, inside_dimension) for value in node.values())
    if isinstance(node, list):
        return any(_contains_parameter_selected_index(value, inside_dimension) for value in node)
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


def render_indexed_ir(value: Any) -> str:
    """Render canonical indexed IR with explicit binder identities."""

    return _render_indexed_ir(value, 0)


def _render_indexed_ir(value: Any, next_binder_id: int) -> str:
    if isinstance(value, IndexTerm):
        return _render_index_term(value)
    if isinstance(value, DecisionAtom):
        indices = ", ".join(_render_index_term(index) for index in value.indices)
        return f"{value.declaration}[{indices}]" if value.indices else value.declaration
    if isinstance(value, DomainTerm):
        return _render_domain(value)
    if isinstance(value, QuantifiedExpression):
        binders = tuple(range(next_binder_id, next_binder_id + len(value.domains)))
        header = ", ".join(
            f"${binder} in {_render_domain(domain)}" for binder, domain in zip(binders, value.domains, strict=True)
        )
        filter_text = "" if value.filter is True else f" : {_render_predicate(value.filter)}"
        body = _render_indexed_ir(value.body, next_binder_id + len(value.domains))
        return f"sum({header}{filter_text}) {body}"
    if isinstance(value, QuantifiedConstraint):
        binders = tuple(range(next_binder_id, next_binder_id + len(value.domains)))
        header = ", ".join(
            f"${binder} in {_render_domain(domain)}" for binder, domain in zip(binders, value.domains, strict=True)
        )
        filter_text = "" if value.filter is True else f" : {_render_predicate(value.filter)}"
        body = _render_indexed_ir(value.body, next_binder_id + len(value.domains))
        return f"forall({header}{filter_text}) {body}"
    if _is_frozen_affine(value):
        return _render_affine(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(_render_indexed_ir(item, next_binder_id) for item in value) + ")"
    return str(value)


def _render_index_term(term: IndexTerm) -> str:
    if term.kind == "binder":
        return f"${term.value}"
    if term.kind in {"number", "declaration"}:
        return str(term.value)
    if term.kind == "application":
        declaration, indices = term.value
        return f"{declaration}[{', '.join(_render_index_term(index) for index in indices)}]"
    if term.kind == "negate":
        return f"(-{_render_index_term(term.value)})"
    if term.kind == "arithmetic":
        operator, left, right = term.value
        return f"({_render_index_term(left)} {operator} {_render_index_term(right)})"
    return repr(term)


def _render_domain(domain: Any) -> str:
    if isinstance(domain, str):
        return domain
    if not isinstance(domain, DomainTerm):
        return repr(domain)
    if domain.kind == "named":
        return str(domain.value)
    start, end = domain.value
    return f"{_render_index_term(start)}..{_render_index_term(end)}"


def free_binders(value: Any, first_binder_id: int = 0) -> frozenset[int]:
    """Return binder identities referenced by an indexed IR value."""

    if isinstance(value, IndexTerm):
        return _index_term_free_binders(value, first_binder_id)
    if isinstance(value, DecisionAtom):
        return frozenset().union(*(free_binders(index, first_binder_id) for index in value.indices))
    if isinstance(value, DomainTerm):
        if value.kind == "range":
            start, end = value.value
            return free_binders(start, first_binder_id) | free_binders(end, first_binder_id)
        return frozenset()
    if isinstance(value, (QuantifiedExpression, QuantifiedConstraint)):
        return _quantified_free_binders(value, first_binder_id)
    if isinstance(value, tuple):
        if len(value) == 2 and value[0] == "binder" and isinstance(value[1], int):
            return frozenset({value[1]})
        return frozenset().union(*(free_binders(item, first_binder_id) for item in value))
    if isinstance(value, list):
        return frozenset().union(*(free_binders(item, first_binder_id) for item in value))
    return frozenset()


def _index_term_free_binders(value: IndexTerm, first_binder_id: int) -> frozenset[int]:
    if value.kind == "binder":
        return frozenset({int(value.value)})
    if value.kind == "negate":
        return free_binders(value.value, first_binder_id)
    if value.kind == "arithmetic":
        _, left, right = value.value
        return free_binders(left, first_binder_id) | free_binders(right, first_binder_id)
    if value.kind == "application":
        _, indices = value.value
        return frozenset().union(*(free_binders(index, first_binder_id) for index in indices))
    return frozenset()


def _quantified_free_binders(value: QuantifiedExpression | QuantifiedConstraint, first_binder_id: int) -> frozenset[int]:
    local_ids = frozenset(range(first_binder_id, first_binder_id + len(value.domains)))
    domains = frozenset().union(*(free_binders(domain, first_binder_id) for domain in value.domains))
    nested_start = first_binder_id + len(value.domains)
    references = domains | free_binders(value.filter, nested_start) | free_binders(value.body, nested_start)
    return references - local_ids


def domain_dependency_graph(domains: Sequence[DomainTerm], first_binder_id: int = 0) -> dict[int, frozenset[int]]:
    """Return prior-binder dependencies for an ordered domain sequence."""

    graph: dict[int, frozenset[int]] = {}
    for offset, domain in enumerate(domains):
        binder_id = first_binder_id + offset
        dependencies = free_binders(domain, first_binder_id)
        if any(dependency >= binder_id for dependency in dependencies):
            raise UnsupportedAlgebra("quantifier domain references its own or a later binder")
        graph[binder_id] = dependencies
    return graph


def _is_frozen_affine(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[1], tuple)
        and all(isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], DecisionAtom) for term in value[1])
    )


def _render_affine(value: tuple[sp.Expr, tuple[tuple[DecisionAtom, sp.Expr], ...]]) -> str:
    constant, terms = value
    parts: list[str] = []
    for atom, coefficient in terms:
        atom_text = _render_indexed_ir(atom, 0)
        if coefficient == 1:
            parts.append(atom_text)
        elif coefficient == -1:
            parts.append(f"-{atom_text}")
        else:
            parts.append(f"({coefficient}) * {atom_text}")
    if constant != 0 or not parts:
        parts.append(str(constant))
    return " + ".join(parts)


def _render_predicate(predicate: Any) -> str:
    if isinstance(predicate, tuple):
        return "(" + ", ".join(_render_predicate(item) for item in predicate) + ")"
    return str(predicate)


__all__ = [
    "DecisionAtom",
    "DomainTerm",
    "IndexTerm",
    "IndexedAffineExpression",
    "QuantifiedConstraint",
    "QuantifiedExpression",
    "prove_indexed_equivalence",
    "free_binders",
    "domain_dependency_graph",
    "render_indexed_ir",
]

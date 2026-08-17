"""Structural equivalence checks for abstract PyOPL MILP model schemas.

Unlike :mod:`pyopl.milp_concrete_equivalence`, this module compares models before data
values expand indexed declarations and constraints into a concrete matrix.  It
uses the PyOPL parser AST as its source IR and converts that AST into a
symbol-linked graph.  Exact labelled graph isomorphism then recognizes
declaration renaming, bound-iterator renaming, declaration and constraint
reordering, and common expression-ordering differences.

An equivalent result is a proof that the supported abstract schemas are
isomorphic.  A different result only means that this structural procedure did
not establish equivalence; it is not a complete decision procedure for all
parameterized MILP reformulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Literal, Mapping

import networkx as nx
from networkx.algorithms import isomorphism

from pyopl._abstract_algebra import UnsupportedAlgebra, lower_symbolic_model, prove_algebraic_equivalence
from pyopl.pyopl_core import OPLLexer, OPLParser

AbstractEquivalenceStatus = Literal["equivalent", "different", "unknown"]
AbstractEquivalenceLevel = Literal[
    "schema_isomorphic",
    "symbolically_normalized",
    "rewrite_certified",
    "polyhedrally_proven",
    "presburger_proven",
]
AbstractModelInput = str | Mapping[str, Any]

_ASSOCIATIVE_COMMUTATIVE_OPERATORS = {"+", "*"}
_COMMUTATIVE_OPERATORS = {"==", "!="}
_IGNORED_KEYS = {"label", "label_template", "lineno"}
_SYMBOL_NAME_NODE_TYPES = {
    "indexed_name",
    "name_reference_index",
    "named_range",
    "named_range_dimension",
    "named_set",
    "named_set_dimension",
}


@dataclass(frozen=True)
class AbstractEquivalenceResult:
    """Status-bearing result for an abstract model comparison."""

    status: AbstractEquivalenceStatus
    level: AbstractEquivalenceLevel
    reason: str
    proof_steps: tuple[str, ...] = ()
    counterexample: str | None = None

    @property
    def equivalent(self) -> bool:
        return self.status == "equivalent"


def parse_abstract_model(model_code: str) -> dict[str, Any]:
    """Parse PyOPL source without data-dependent compiler materialization."""

    ast = OPLParser().parse(OPLLexer().tokenize(model_code))
    if not isinstance(ast, dict):
        raise ValueError("PyOPL parser did not return a model AST")
    return ast


def compare_abstract(
    left: AbstractModelInput,
    right: AbstractModelInput,
    *,
    mode: Literal["structural", "algebraic", "auto"] = "structural",
    **proof_options: Any,
) -> bool:
    """Return whether two PyOPL sources or parser ASTs are schema-isomorphic."""

    return prove_abstract_equivalent(left, right, mode=mode, **proof_options).equivalent


def prove_abstract_equivalent(
    left: AbstractModelInput,
    right: AbstractModelInput,
    *,
    mode: Literal["structural", "algebraic", "auto"] = "structural",
    parameter_mapping: Mapping[str, str] | None = None,
    variable_mapping: Mapping[str, str] | None = None,
    left_auxiliaries: Collection[str] = (),
    right_auxiliaries: Collection[str] = (),
    assumptions: Mapping[str, str] | None = None,
    max_rewrite_iterations: int = 12,
) -> AbstractEquivalenceResult:
    """Compare two abstract PyOPL models using exact labelled graph isomorphism.

    The accepted inputs are model source strings or AST mappings returned
    directly by :class:`pyopl.pyopl_core.OPLParser`.  The comparison ignores
    declaration names, bound-iterator names, declaration order, constraint
    order, labels, parentheses, and operand order for associative-commutative
    arithmetic and logical operators.  It also canonicalizes comparison
    direction, so ``a >= b`` and ``b <= a`` have the same representation.

    ``mode="structural"`` performs only schema isomorphism and preserves the
    original API behavior.  ``mode="algebraic"`` lowers scalar affine schemas
    to a typed symbolic IR and applies exact normalization, certified affine
    substitutions, Fourier--Motzkin projection, checked Farkas certificates,
    and exhaustive bounded-integer elimination.  ``mode="auto"`` accepts a
    schema isomorphism immediately and otherwise tries the algebraic backend.

    Algebraic mode returns ``unknown`` for unsupported indexed, nonlinear,
    parameterized-implication, or unbounded-integer fragments.  Parameter
    values are deliberately not accepted here: concrete instances remain the
    responsibility of :func:`pyopl.milp_concrete_equivalence.prove_equivalent`.
    """

    left_ast = _coerce_ast(left)
    right_ast = _coerce_ast(right)
    left_issue = _model_ast_issue(left_ast)
    right_issue = _model_ast_issue(right_ast)
    if left_issue is not None or right_issue is not None:
        issue = left_issue or right_issue
        return AbstractEquivalenceResult(
            status="unknown",
            level="schema_isomorphic",
            reason=issue or "invalid abstract model AST",
        )

    structural_result = _prove_schema_isomorphism(left_ast, right_ast)
    if mode == "structural" or (mode == "auto" and structural_result.equivalent):
        return structural_result
    if mode not in {"algebraic", "auto"}:
        return AbstractEquivalenceResult(
            status="unknown",
            level="schema_isomorphic",
            reason=f"unsupported abstract equivalence mode: {mode}",
        )

    try:
        left_model = lower_symbolic_model(left_ast, assumptions)
        right_model = lower_symbolic_model(right_ast, assumptions)
        proof = prove_algebraic_equivalence(
            left_model,
            right_model,
            parameter_mapping=parameter_mapping,
            variable_mapping=variable_mapping,
            left_auxiliaries=left_auxiliaries,
            right_auxiliaries=right_auxiliaries,
            max_rewrite_iterations=max_rewrite_iterations,
        )
    except UnsupportedAlgebra as exc:
        return AbstractEquivalenceResult(
            status="unknown",
            level="symbolically_normalized",
            reason=str(exc),
            proof_steps=structural_result.proof_steps if mode == "auto" else (),
        )
    proof_steps = proof.steps
    if mode == "auto":
        proof_steps = structural_result.proof_steps + proof_steps
    return AbstractEquivalenceResult(
        status=proof.status,
        level=proof.level,
        reason=proof.reason,
        proof_steps=tuple(dict.fromkeys(proof_steps)),
        counterexample=proof.counterexample,
    )


def _prove_schema_isomorphism(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
) -> AbstractEquivalenceResult:
    try:
        left_graph = _AbstractGraphBuilder(left_ast).build()
        right_graph = _AbstractGraphBuilder(right_ast).build()
    except _UnsupportedAbstractNode as exc:
        return AbstractEquivalenceResult(
            status="unknown",
            level="schema_isomorphic",
            reason=str(exc),
        )

    proof_steps = (
        "parsed abstract model schemas",
        "linked declarations and bound iterators to their references",
        "tested labelled abstract-syntax graph isomorphism",
    )
    matcher = isomorphism.DiGraphMatcher(
        left_graph,
        right_graph,
        node_match=isomorphism.categorical_node_match("label", None),
        edge_match=isomorphism.categorical_edge_match("role", None),
    )
    if matcher.is_isomorphic():
        return AbstractEquivalenceResult(
            status="equivalent",
            level="schema_isomorphic",
            reason="abstract model schemas are isomorphic",
            proof_steps=proof_steps,
        )
    return AbstractEquivalenceResult(
        status="different",
        level="schema_isomorphic",
        reason="abstract model schemas are not isomorphic",
        proof_steps=proof_steps,
        counterexample="no label-preserving abstract-syntax graph isomorphism exists",
    )


def _coerce_ast(model: AbstractModelInput) -> Mapping[str, Any]:
    if isinstance(model, str):
        return parse_abstract_model(model)
    if isinstance(model, Mapping):
        return model
    raise TypeError("abstract model input must be PyOPL source or an AST mapping")


def _model_ast_issue(ast: Mapping[str, Any]) -> str | None:
    for key in ("declarations", "objective", "constraints"):
        if key not in ast:
            return f"abstract model AST is missing '{key}'"
    if not isinstance(ast["declarations"], list):
        return "abstract model AST declarations must be a list"
    if not isinstance(ast["objective"], Mapping):
        return "abstract model AST objective must be an object"
    if not isinstance(ast["constraints"], list):
        return "abstract model AST constraints must be a list"
    return None


class _UnsupportedAbstractNode(ValueError):
    pass


class _AbstractGraphBuilder:
    def __init__(self, ast: Mapping[str, Any]) -> None:
        self.ast = ast
        self.graph = nx.DiGraph()
        self._next_node = 0
        self._global_symbols: dict[str, int] = {}
        self._declaration_nodes: list[tuple[Mapping[str, Any], int]] = []

    def build(self) -> nx.DiGraph:
        root = self._new_node(("model",))
        declarations = self.ast["declarations"]
        for declaration in declarations:
            if not isinstance(declaration, Mapping):
                raise _UnsupportedAbstractNode("abstract declaration must be an object")
            node = self._new_node(("ast", declaration.get("type", "declaration")))
            self._edge(root, node, "declaration")
            self._declaration_nodes.append((declaration, node))
            name = declaration.get("name")
            if isinstance(name, str):
                if name in self._global_symbols:
                    raise _UnsupportedAbstractNode(f"duplicate abstract declaration: {name}")
                self._global_symbols[name] = node

        for declaration, node in self._declaration_nodes:
            self._populate_mapping(node, declaration, self._global_symbols, declaration=True)

        objective = self._add_value(self.ast["objective"], self._global_symbols)
        self._edge(root, objective, "objective")
        for constraint in self.ast["constraints"]:
            constraint_node = self._add_value(constraint, self._global_symbols)
            self._edge(root, constraint_node, "constraint")
        return self.graph

    def _new_node(self, label: tuple[Any, ...]) -> int:
        node = self._next_node
        self._next_node += 1
        self.graph.add_node(node, label=label)
        return node

    def _edge(self, parent: int, child: int, role: str) -> None:
        self.graph.add_edge(parent, child, role=role)

    def _add_value(self, value: Any, scope: Mapping[str, int]) -> int:
        if isinstance(value, Mapping):
            return self._add_mapping(value, scope)
        if isinstance(value, list):
            node = self._new_node(("list",))
            for index, item in enumerate(value):
                child = self._add_value(item, scope)
                self._edge(node, child, f"item:{index}")
            return node
        if isinstance(value, (str, int, float, bool)) or value is None:
            return self._new_node(("literal", type(value).__name__, value))
        raise _UnsupportedAbstractNode(f"unsupported abstract AST value: {type(value).__name__}")

    def _add_mapping(self, value: Mapping[str, Any], scope: Mapping[str, int]) -> int:
        node_type = value.get("type", "mapping")
        if node_type == "parenthesized_expression":
            expression = value.get("expression")
            return self._add_value(expression, scope)
        if node_type in {"binop", "constraint", "and", "or"}:
            return self._add_operator(value, scope)

        node = self._new_node(("ast", node_type))
        self._populate_mapping(node, value, scope)
        return node

    def _add_operator(self, value: Mapping[str, Any], scope: Mapping[str, int]) -> int:
        node_type = value.get("type")
        operator = value.get("op") if node_type in {"binop", "constraint"} else node_type
        left = value.get("left")
        right = value.get("right")
        if operator == ">":
            operator, left, right = "<", right, left
        elif operator == ">=":
            operator, left, right = "<=", right, left

        node = self._new_node(("operator", node_type, operator, value.get("sem_type")))
        if operator in _ASSOCIATIVE_COMMUTATIVE_OPERATORS or operator in {"and", "or"}:
            for operand in self._flatten_operator(value, operator):
                child = self._add_value(operand, scope)
                self._edge(node, child, "operand")
        elif operator in _COMMUTATIVE_OPERATORS:
            for operand in (left, right):
                child = self._add_value(operand, scope)
                self._edge(node, child, "operand")
        else:
            left_node = self._add_value(left, scope)
            right_node = self._add_value(right, scope)
            self._edge(node, left_node, "left")
            self._edge(node, right_node, "right")
        return node

    def _flatten_operator(self, value: Any, operator: str) -> list[Any]:
        if not isinstance(value, Mapping):
            return [value]
        node_type = value.get("type")
        value_operator = value.get("op") if node_type == "binop" else node_type
        if value_operator != operator:
            return [value]
        return self._flatten_operator(value.get("left"), operator) + self._flatten_operator(
            value.get("right"), operator
        )

    def _populate_mapping(
        self,
        node: int,
        value: Mapping[str, Any],
        scope: Mapping[str, int],
        *,
        declaration: bool = False,
    ) -> None:
        node_type = value.get("type")
        if "iterators" in value:
            scope = self._add_iterators(node, value.get("iterators"), scope)

        for key in sorted(value):
            if key == "type" or key == "iterators" or key in _IGNORED_KEYS or key.startswith("_"):
                continue
            child_value = value[key]
            if declaration and key == "name":
                continue
            symbol_name = self._referenced_symbol(node_type, key, child_value, scope)
            if symbol_name is not None:
                reference = self._new_node(("symbol_reference",))
                self._edge(node, reference, key)
                self._edge(reference, scope[symbol_name], "refers_to")
                continue
            child = self._add_value(child_value, scope)
            self._edge(node, child, key)

    def _add_iterators(
        self,
        parent: int,
        iterators: Any,
        outer_scope: Mapping[str, int],
    ) -> Mapping[str, int]:
        if not isinstance(iterators, list):
            raise _UnsupportedAbstractNode("abstract model iterators must be a list")
        scope = dict(outer_scope)
        iterator_nodes: list[tuple[Mapping[str, Any], int]] = []
        for iterator in iterators:
            if not isinstance(iterator, Mapping) or not isinstance(iterator.get("iterator"), str):
                raise _UnsupportedAbstractNode("abstract iterator must contain an iterator name")
            iterator_node = self._new_node(("iterator",))
            self._edge(parent, iterator_node, "iterator")
            scope[iterator["iterator"]] = iterator_node
            iterator_nodes.append((iterator, iterator_node))
        for iterator, iterator_node in iterator_nodes:
            range_node = self._add_value(iterator.get("range"), scope)
            self._edge(iterator_node, range_node, "range")
        return scope

    def _referenced_symbol(
        self,
        node_type: Any,
        key: str,
        value: Any,
        scope: Mapping[str, int],
    ) -> str | None:
        if not isinstance(value, str) or value not in scope:
            return None
        if node_type == "name" and key == "value":
            return value
        if key == "name" and node_type in _SYMBOL_NAME_NODE_TYPES:
            return value
        if key in {"tuple_type", "index_set"}:
            return value
        return None


__all__ = [
    "AbstractEquivalenceResult",
    "compare_abstract",
    "parse_abstract_model",
    "prove_abstract_equivalent",
]
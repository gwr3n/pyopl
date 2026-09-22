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

Paper correspondence
--------------------
The binding-aware graph construction implements Lemma 6.3 (Binding-aware
renaming), and a successful graph match is the computational case of Theorem
6.4 (Schema isomorphism is uniformly sound).  Algebraic fallback follows
Proposition 6.6 (Sound symbolic normalization), Theorem 6.7 (Certified rewrite
chains), and the projection results in Theorems 7.6 and 7.9.  Supplying data
crosses from a schema claim to the instance claim of Proposition 6.9 (Correct
finite grounding).  The result levels are interpreted in Section 9.2 (Result
labels) of the attached paper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Collection, Literal, Mapping

import networkx as nx
from networkx.algorithms import isomorphism

from pyopl._abstract_algebra import (
    AlgebraicProof,
    SymbolicModel,
    UnsupportedAlgebra,
    lower_linear_problem,
    lower_symbolic_model,
    prove_algebraic_equivalence,
)
from pyopl._index_safety import find_index_safety_issue
from pyopl._indexed_algebra import prove_indexed_equivalence
from pyopl.pyopl_core import OPLLexer, OPLParser, linear_problem_from_opl
from pyopl.semantic_error import SemanticError

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
    """Status-bearing abstract result from the layered procedure.

    ``level`` identifies the supporting result as described in
    Section 9.2 (Result labels); ``unknown`` has the inconclusive meaning
    specified after Theorem 9.1, rather than non-equivalence.
    """

    status: AbstractEquivalenceStatus
    level: AbstractEquivalenceLevel
    reason: str
    proof_steps: tuple[str, ...] = ()
    counterexample: str | None = None
    relation: str = "not_recorded"
    scope: str = "not_recorded"
    arithmetic: str = "not_recorded"
    variable_mapping: tuple[tuple[str, str], ...] = ()
    parameter_mapping: tuple[tuple[str, str], ...] = ()
    left_auxiliaries: tuple[str, ...] = ()
    right_auxiliaries: tuple[str, ...] = ()
    assumptions: tuple[tuple[str, str], ...] = ()
    termination: str = "not_recorded"
    budget_exhausted: bool = False
    evidence_kind: str = "internal_checks_only"

    @property
    def equivalent(self) -> bool:
        return self.status == "equivalent"


def parse_abstract_model(model_code: str) -> dict[str, Any]:
    """Parse source into the schema representation of Section 6.

    No valuation is supplied here, so indexed declarations and binders remain
    available for the uniform comparison in Theorem 6.4.
    """

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
    """Return whether the selected paper-backed proof path establishes equivalence.

    This boolean projection intentionally maps both ``different`` and
    ``unknown`` to ``False``; use :func:`prove_abstract_equivalent` to retain
    the three outcomes required by Theorem 9.1.
    """

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
    left_data_text: str | None = None,
    right_data_text: str | None = None,
) -> AbstractEquivalenceResult:
    """Compare two abstract PyOPL models using exact labelled graph isomorphism.

    The accepted inputs are model source strings or AST mappings returned
    directly by :class:`pyopl.pyopl_core.OPLParser`.  The comparison ignores
    declaration names, bound-iterator names, declaration order, constraint
    order, labels, parentheses, and operand order for associative-commutative
    arithmetic and logical operators.  It also canonicalizes comparison
    direction, so ``a >= b`` and ``b <= a`` have the same representation.
    Explicit parameter and variable mappings constrain schema isomorphism as
    well as the algebraic proof stages.

    ``mode="structural"`` performs only schema isomorphism and preserves the
    original API behavior.  ``mode="algebraic"`` lowers scalar affine schemas
    to a typed symbolic IR and applies exact normalization, certified affine
    substitutions, Fourier--Motzkin projection, checked Farkas certificates,
    and exhaustive bounded-integer elimination.  ``mode="auto"`` accepts a
    schema isomorphism immediately and otherwise tries the algebraic backend.

    When both data texts are supplied, models are finitely grounded by
    PyOPL's matrix lowering before the
    algebraic proof stages. Such a result proves equivalence for those supplied
    data instances, not universally for every parameter assignment. Algebraic
    mode otherwise returns ``unknown`` for unsupported indexed, nonlinear,
    parameterized-implication, or unbounded-integer fragments.
    """

    left_ast = _coerce_ast(left)
    right_ast = _coerce_ast(right)
    preflight_result = _abstract_model_preflight(left_ast, right_ast, left_data_text, right_data_text)
    if preflight_result is not None:
        return preflight_result

    _validate_comparison_options(
        left_ast, right_ast, parameter_mapping, variable_mapping, left_auxiliaries, right_auxiliaries, assumptions
    )
    requires_algebra = _validate_proof_route(
        mode, left_data_text, right_data_text, parameter_mapping, assumptions, left_auxiliaries, right_auxiliaries
    )
    structural_result = _prove_schema_isomorphism(left_ast, right_ast, parameter_mapping, variable_mapping)
    if mode == "structural" or (mode == "auto" and structural_result.equivalent and not requires_algebra):
        return structural_result
    if mode not in {"algebraic", "auto"}:
        return AbstractEquivalenceResult(
            status="unknown",
            level="schema_isomorphic",
            reason=f"unsupported abstract equivalence mode: {mode}",
            termination="unsupported_input",
        )

    context = _algebraic_result_context(
        structural_result.proof_steps if mode == "auto" else (),
        left_data_text is not None,
        variable_mapping,
        parameter_mapping,
        left_auxiliaries,
        right_auxiliaries,
        assumptions,
    )
    return _prove_algebraic_models(
        left_ast,
        right_ast,
        left,
        right,
        assumptions,
        left_data_text,
        right_data_text,
        parameter_mapping,
        variable_mapping,
        left_auxiliaries,
        right_auxiliaries,
        max_rewrite_iterations,
        context,
    )


def _abstract_model_preflight(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    left_data_text: str | None,
    right_data_text: str | None,
) -> AbstractEquivalenceResult | None:
    issue = _model_ast_issue(left_ast) or _model_ast_issue(right_ast)
    if issue is not None:
        return AbstractEquivalenceResult(
            status="unknown",
            level="schema_isomorphic",
            reason=issue,
            termination="unsupported_input",
        )
    return _index_safety_preflight(left_ast, right_ast, left_data_text, right_data_text)


def _index_safety_preflight(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    left_data_text: str | None,
    right_data_text: str | None,
) -> AbstractEquivalenceResult | None:
    safety_issues = (find_index_safety_issue(left_ast), find_index_safety_issue(right_ast))
    unsafe_issue = next((issue for issue in safety_issues if issue is not None and issue.status == "unsafe"), None)
    if unsafe_issue is not None:
        raise SemanticError(unsafe_issue.reason)
    unresolved_issue = next((issue for issue in safety_issues if issue is not None), None)
    if unresolved_issue is None or left_data_text is not None or right_data_text is not None:
        return None
    return AbstractEquivalenceResult(
        status="unknown",
        level="schema_isomorphic",
        reason=unresolved_issue.reason,
        scope="source_schemas",
        termination="unsupported_fragment",
    )


def _validate_proof_route(
    mode: str,
    left_data_text: str | None,
    right_data_text: str | None,
    parameter_mapping: Mapping[str, str] | None,
    assumptions: Mapping[str, str] | None,
    left_auxiliaries: Collection[str],
    right_auxiliaries: Collection[str],
) -> bool:
    """Validate schema/instance options before accepting a structural shortcut."""
    supplied_data = left_data_text is not None or right_data_text is not None
    if supplied_data and (left_data_text is None or right_data_text is None):
        raise ValueError("abstract instance comparison requires both data texts")
    if supplied_data and (parameter_mapping or assumptions):
        raise ValueError("parameter mappings and assumptions apply to schemas, not supplied instances")
    requires_algebra = bool(supplied_data or assumptions or left_auxiliaries or right_auxiliaries)
    if mode == "structural" and requires_algebra:
        raise ValueError("structural mode does not accept data, assumptions, or auxiliary partitions; use auto or algebraic")
    return requires_algebra


def _algebraic_result_context(
    proof_steps: tuple[str, ...],
    supplied_data: bool,
    variable_mapping: Mapping[str, str] | None,
    parameter_mapping: Mapping[str, str] | None,
    left_auxiliaries: Collection[str],
    right_auxiliaries: Collection[str],
    assumptions: Mapping[str, str] | None,
) -> AbstractEquivalenceResult:
    """Record the requested correspondence and scope before attempting lowering."""
    return AbstractEquivalenceResult(
        status="unknown",
        level="symbolically_normalized",
        reason="",
        proof_steps=proof_steps,
        relation="projected_value",
        scope="supplied_instances" if supplied_data else "source_schemas",
        termination="unsupported_fragment",
        variable_mapping=tuple(sorted((variable_mapping or {}).items())),
        parameter_mapping=tuple(sorted((parameter_mapping or {}).items())),
        left_auxiliaries=tuple(sorted(left_auxiliaries)),
        right_auxiliaries=tuple(sorted(right_auxiliaries)),
        assumptions=tuple(sorted((assumptions or {}).items())),
    )


def _prove_algebraic_models(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    left: AbstractModelInput,
    right: AbstractModelInput,
    assumptions: Mapping[str, str] | None,
    left_data_text: str | None,
    right_data_text: str | None,
    parameter_mapping: Mapping[str, str] | None,
    variable_mapping: Mapping[str, str] | None,
    left_auxiliaries: Collection[str],
    right_auxiliaries: Collection[str],
    max_rewrite_iterations: int,
    context: AbstractEquivalenceResult,
) -> AbstractEquivalenceResult:
    """Run lowering and algebraic proof while preserving inconclusive outcomes."""
    try:
        if left_data_text is None and right_data_text is None:
            if assumptions:
                indexed_proof = None
            else:
                indexed_proof = prove_indexed_equivalence(
                    left_ast,
                    right_ast,
                    parameter_mapping=parameter_mapping,
                    variable_mapping=variable_mapping,
                    left_auxiliaries=left_auxiliaries,
                    right_auxiliaries=right_auxiliaries,
                    max_rewrite_iterations=max_rewrite_iterations,
                )
            if indexed_proof is not None:
                return _indexed_algebraic_public_result(indexed_proof, context)
        left_model, right_model, grounded_indexed_schema = _lower_comparison_models(
            left_ast,
            right_ast,
            left,
            right,
            assumptions,
            left_data_text,
            right_data_text,
        )
        effective_left_auxiliaries = set(left_auxiliaries)
        effective_right_auxiliaries = set(right_auxiliaries)
        if grounded_indexed_schema:
            _validate_grounded_correspondence(left_model, right_model, variable_mapping, left_auxiliaries, right_auxiliaries)
        proof = prove_algebraic_equivalence(
            left_model,
            right_model,
            parameter_mapping=parameter_mapping,
            variable_mapping=variable_mapping,
            left_auxiliaries=effective_left_auxiliaries,
            right_auxiliaries=effective_right_auxiliaries,
            max_rewrite_iterations=max_rewrite_iterations,
        )
    except UnsupportedAlgebra as exc:
        return replace(context, reason=str(exc))
    context = replace(
        context,
        left_auxiliaries=tuple(sorted(effective_left_auxiliaries)),
        right_auxiliaries=tuple(sorted(effective_right_auxiliaries)),
    )
    return _algebraic_public_result(proof, context, left_model, right_model, grounded_indexed_schema)


def _indexed_algebraic_public_result(
    proof: AlgebraicProof,
    context: AbstractEquivalenceResult,
) -> AbstractEquivalenceResult:
    """Expose a uniform indexed-schema proof without scalarizing its families."""

    return replace(
        context,
        status=proof.status,
        level=proof.level,
        reason=proof.reason,
        proof_steps=tuple(dict.fromkeys(context.proof_steps + proof.steps)),
        scope="uniform_schema",
        arithmetic="exact_on_parsed_values",
        variable_mapping=proof.variable_mapping or context.variable_mapping,
        parameter_mapping=proof.parameter_mapping or context.parameter_mapping,
        termination="budget_exhausted" if proof.budget_exhausted else "completed",
        budget_exhausted=proof.budget_exhausted,
    )


def _validate_grounded_correspondence(
    left: SymbolicModel,
    right: SymbolicModel,
    variable_mapping: Mapping[str, str] | None,
    left_auxiliaries: Collection[str],
    right_auxiliaries: Collection[str],
) -> None:
    """Require named scalar columns after Proposition 6.9 finite grounding."""
    left_names = {variable.name for variable in left.variables}
    right_names = {variable.name for variable in right.variables}
    if (
        not set(variable_mapping or {}) <= left_names
        or not set((variable_mapping or {}).values()) <= right_names
        or not set(left_auxiliaries) <= left_names
        or not set(right_auxiliaries) <= right_names
    ):
        raise UnsupportedAlgebra(
            "grounded indexed correspondences require scalar column names; indexed declaration maps are not expanded"
        )


def _algebraic_public_result(
    proof: AlgebraicProof,
    context: AbstractEquivalenceResult,
    left: SymbolicModel,
    right: SymbolicModel,
    grounded: bool,
) -> AbstractEquivalenceResult:
    """Attach scope and arithmetic to the backend outcome, as in Section 9.2."""
    proof_steps = proof.steps
    if grounded:
        proof_steps = ("grounded finite models with supplied data",) + proof_steps
    scope = "uniform_schema" if left.parameters or right.parameters else "parameterless_instance"
    termination = "completed" if proof.status != "unknown" else "inconclusive"
    return replace(
        context,
        status=proof.status,
        level=proof.level,
        reason=proof.reason,
        proof_steps=tuple(dict.fromkeys(context.proof_steps + proof_steps)),
        counterexample=proof.counterexample,
        scope="supplied_instances" if grounded else scope,
        arithmetic="exact_on_embedded_matrix_values" if grounded else "exact_on_parsed_values",
        variable_mapping=proof.variable_mapping or context.variable_mapping,
        parameter_mapping=proof.parameter_mapping or context.parameter_mapping,
        termination="budget_exhausted" if proof.budget_exhausted else termination,
        budget_exhausted=proof.budget_exhausted,
    )


def _validate_comparison_options(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    parameter_mapping: Mapping[str, str] | None,
    variable_mapping: Mapping[str, str] | None,
    left_auxiliaries: Collection[str],
    right_auxiliaries: Collection[str],
    assumptions: Mapping[str, str] | None,
) -> None:
    """Validate declaration maps, auxiliary partitions, and parameter assumptions."""
    left_declarations = {item.get("name"): item for item in left["declarations"] if isinstance(item, Mapping)}
    right_declarations = {item.get("name"): item for item in right["declarations"] if isinstance(item, Mapping)}
    _validate_declaration_mapping(left_declarations, right_declarations, variable_mapping, "dvar")
    _validate_declaration_mapping(left_declarations, right_declarations, parameter_mapping, "parameter")
    for declarations, auxiliaries, retained in (
        (left_declarations, left_auxiliaries, set(variable_mapping or {})),
        (right_declarations, right_auxiliaries, set((variable_mapping or {}).values())),
    ):
        _validate_auxiliary_partition(declarations, auxiliaries, retained)
    _validate_parameter_assumptions(left_declarations, right_declarations, assumptions)


def _validate_declaration_mapping(
    left: Mapping[Any, Any],
    right: Mapping[Any, Any],
    mapping: Mapping[str, str] | None,
    declaration_prefix: str,
) -> None:
    """Check injectivity and declaration kind before any proof search."""
    if len(set((mapping or {}).values())) != len(mapping or {}):
        raise ValueError("declaration mappings must be injective")
    for name, target in (mapping or {}).items():
        for declarations, candidate in ((left, name), (right, target)):
            if not str(declarations.get(candidate, {}).get("type", "")).startswith(declaration_prefix):
                raise ValueError(f"mapping references unknown or wrong-kind declaration: {candidate}")


def _validate_auxiliary_partition(declarations: Mapping[Any, Any], auxiliaries: Collection[str], retained: set[str]) -> None:
    """Require auxiliaries to be declared decisions outside the retained map."""
    for name in auxiliaries:
        if not str(declarations.get(name, {}).get("type", "")).startswith("dvar"):
            raise ValueError(f"unknown auxiliary variable: {name}")
        if name in retained:
            raise ValueError(f"retained and auxiliary variables must be disjoint: {name}")


def _validate_parameter_assumptions(
    left: Mapping[Any, Any], right: Mapping[Any, Any], assumptions: Mapping[str, str] | None
) -> None:
    """Require supported assumption facts on parameter names shared by both inputs."""
    for name, condition in (assumptions or {}).items():
        if condition not in {"positive", "nonnegative", "nonzero"}:
            raise ValueError(f"unsupported assumption condition: {condition}")
        if any(not str(declarations.get(name, {}).get("type", "")).startswith("parameter") for declarations in (left, right)):
            raise ValueError(f"assumptions require a parameter with this name on both sides: {name}")


def _lower_comparison_models(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    left: AbstractModelInput,
    right: AbstractModelInput,
    assumptions: Mapping[str, str] | None,
    left_data_text: str | None,
    right_data_text: str | None,
) -> tuple[SymbolicModel, SymbolicModel, bool]:
    """Lower schemas symbolically, or ground supplied finite instances.

    Grounding is justified only at the supplied valuations, as stated by
    Proposition 6.9 (Correct finite grounding); the returned flag preserves that scope in
    the reported proof steps.
    """

    if left_data_text is not None and right_data_text is not None:
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("grounding requires model source strings")
        return (
            lower_linear_problem(linear_problem_from_opl(left, left_data_text)),
            lower_linear_problem(linear_problem_from_opl(right, right_data_text)),
            True,
        )
    return lower_symbolic_model(left_ast, assumptions), lower_symbolic_model(right_ast, assumptions), False


def _prove_schema_isomorphism(
    left_ast: Mapping[str, Any],
    right_ast: Mapping[str, Any],
    parameter_mapping: Mapping[str, str] | None = None,
    variable_mapping: Mapping[str, str] | None = None,
) -> AbstractEquivalenceResult:
    """Test the sufficient schema-equivalence condition of Theorem 6.4.

    Failure rejects this graph correspondence only; Theorem 9.1
    permits algebraic and projected stages to establish semantic equivalence.
    """

    try:
        left_labels, right_labels = _schema_mapping_labels(parameter_mapping, variable_mapping)
        left_builder = _AbstractGraphBuilder(left_ast)
        right_builder = _AbstractGraphBuilder(right_ast)
        left_graph = left_builder.build(left_labels)
        right_graph = right_builder.build(right_labels)
    except _UnsupportedAbstractNode as exc:
        return AbstractEquivalenceResult(
            status="unknown",
            level="schema_isomorphic",
            reason=str(exc),
            relation="schema_structural",
            scope="source_schemas",
            arithmetic="parsed_ast_labels",
            termination="unsupported_fragment",
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
        right_names = {node: name for name, node in right_builder._global_symbols.items()}
        declaration_map = {name: right_names[matcher.mapping[node]] for name, node in left_builder._global_symbols.items()}
        variable_names = {item["name"] for item in left_ast["declarations"] if str(item.get("type", "")).startswith("dvar")}
        parameter_names = {
            item["name"] for item in left_ast["declarations"] if str(item.get("type", "")).startswith("parameter")
        }
        return AbstractEquivalenceResult(
            status="equivalent",
            level="schema_isomorphic",
            reason="abstract model schemas are isomorphic",
            proof_steps=proof_steps,
            relation="schema_structural",
            scope="uniform_schema" if parameter_names else "parameterless_instance",
            arithmetic="parsed_ast_labels",
            variable_mapping=tuple(
                sorted((name, target) for name, target in declaration_map.items() if name in variable_names)
            ),
            parameter_mapping=tuple(
                sorted((name, target) for name, target in declaration_map.items() if name in parameter_names)
            ),
            termination="completed",
        )
    return AbstractEquivalenceResult(
        status="different",
        level="schema_isomorphic",
        reason="abstract model schemas are not isomorphic",
        proof_steps=proof_steps,
        counterexample="no label-preserving abstract-syntax graph isomorphism exists",
        relation="schema_structural",
        scope="source_schemas",
        arithmetic="parsed_ast_labels",
        termination="completed",
    )


def _schema_mapping_labels(
    parameter_mapping: Mapping[str, str] | None,
    variable_mapping: Mapping[str, str] | None,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """Encode prescribed declaration pairs as isomorphism constraints.

    This is the abstract counterpart of the mapping restriction in
    Corollary 5.10 (Mapping-constrained comparison).
    """

    left_labels: dict[str, tuple[str, str]] = {}
    right_labels: dict[str, tuple[str, str]] = {}
    for kind, mapping in (("parameter", parameter_mapping), ("variable", variable_mapping)):
        for left_name, right_name in (mapping or {}).items():
            if left_name in left_labels or right_name in right_labels:
                raise _UnsupportedAbstractNode("abstract declaration mappings must be disjoint and injective")
            left_labels[left_name] = (kind, left_name)
            right_labels[right_name] = (kind, left_name)
    return left_labels, right_labels


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
    """Build the faithful syntax-and-binding graph of Section 6.

    Declaration and iterator reference edges carry the binding evidence needed
    by Lemma 6.3 (Binding-aware renaming); edge roles preserve ordered operands while
    repeated operand vertices preserve multiplicity.
    """

    def __init__(self, ast: Mapping[str, Any]) -> None:
        self.ast = ast
        self.graph = nx.DiGraph()
        self._next_node = 0
        self._global_symbols: dict[str, int] = {}
        self._declaration_nodes: list[tuple[Mapping[str, Any], int]] = []

    def build(self, mapping_labels: Mapping[str, tuple[str, str]] | None = None) -> nx.DiGraph:
        """Construct a complete model graph with declarations linked before uses."""

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

        for name, label in (mapping_labels or {}).items():
            if name not in self._global_symbols:
                raise _UnsupportedAbstractNode(f"abstract mapping references unknown declaration: {name}")
            mapping_node = self._new_node(("user_mapping", *label))
            self._edge(mapping_node, self._global_symbols[name], "refers_to")

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
        """Apply the semantics-preserving AST standardization preceding Theorem 6.4."""

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
        """Flatten associative-commutative operands while retaining occurrences."""

        if not isinstance(value, Mapping):
            return [value]
        node_type = value.get("type")
        value_operator = value.get("op") if node_type == "binop" else node_type
        if value_operator != operator:
            return [value]
        return self._flatten_operator(value.get("left"), operator) + self._flatten_operator(value.get("right"), operator)

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
        """Create binder vertices and extend lexical scope as in Lemma 6.3."""

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

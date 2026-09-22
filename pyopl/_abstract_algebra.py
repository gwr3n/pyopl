"""Algebraic comparison companion to the paper's abstract and projected routes.

Read this module alongside *Abstract and Concrete MILP Model Equivalence:
A Layered Proof Framework* (September 2026). The numbered results below refer to
that paper; their titles identify them if subsequent revisions change numbering.
Binding-aware syntax matching belongs to ``milp_abstract_equivalence``;
numerically qualified matrix matching belongs to ``milp_concrete_equivalence``.
This backend supplies symbolic normalization and the supported exact-arithmetic
comparisons reached when syntax alone does not settle the question.

Reading the proof path
---------------------
``lower_symbolic_model`` translates scalar declarations and affine expressions
into rows of the form ``expression <= 0`` or ``expression = 0``. Decision
domains become explicit rows, while parameter assumptions remain side
conditions. ``lower_linear_problem`` provides the alternative entry for an
already grounded matrix. The latter covers only that instance, not every
valuation of its originating schema.

``prove_algebraic_equivalence`` searches typed coordinate correspondences.
For each correspondence, the right model is renamed into the left namespace
before normalization and continuous-alias substitution. Equal terminal forms
settle the comparison. If parameters remain, further implication is outside
this backend. Otherwise, wholly continuous models use Fourier--Motzkin
projection followed by checked Farkas implications, and wholly integer models
use exhaustive finite-domain value tables. Unresolved mixed domains abstain.

Paper correspondence
--------------------
* Section 4.4 defines retained decisions and their best completion costs.
    Lemma 4.4 (Objective-neutral auxiliaries) explains the continuous projection
    objective condition; Theorem 4.6 (Composition of witnesses) transfers a
    terminal comparison through justified reductions.
* Lemma 5.1 (Affine elimination), Proposition 5.5 (Elementary row
    transformations), Proposition 6.6 (Sound symbolic normalization), and
    Theorem 6.7 (Certified rewrite chains) justify normalization and substitution.
    Proposition 6.9 (Correct finite grounding) delimits the matrix adapter.
* Lemma 7.1 (Fourier--Motzkin elimination) and Corollary 7.2 (Exact rational
    projection) justify real elimination. Theorem 7.4 (Farkas certificates for
    row implication), Proposition 7.5 (Equality of affine objectives on a feasible
    set), and Theorem 7.6 (Projected polyhedral equivalence) justify comparison.
* Theorem 7.9 (Complete finite-domain comparison) justifies the integer value
    tables. The legacy level ``presburger_proven`` names this bounded method,
    not a general Presburger decision procedure.
* Section 8 separates numerical discovery from exact checking. Theorem 9.1
    (Soundness of the layered exact procedure) and Sections 9.1--9.2 specify
    outcome semantics and the evidence required to interpret result labels.

Arithmetic, evidence, and limits
-------------------------------
Rationals here embed the decimal strings of values already parsed or lowered;
they cannot recover source digits lost upstream. HiGHS proposes implication
multipliers, but acceptance requires an exact rational identity. Failure to
find such multipliers is inconclusive. Mapping, rewrite, and enumeration
budgets likewise leave an unfinished comparison ``unknown``. Fourier--Motzkin
row growth has no dedicated budget in this implementation.

The complete exact procedures in the paper do not imply completeness of this
budgeted implementation: mapping types, bound recognition, symbolic side
conditions, and rational multiplier reconstruction are deliberately limited.
The parser, lowering, symbolic transformations, and arithmetic libraries remain
trusted. Results expose summaries of internal checks, not an independently
replayable certificate or an exported projection reconstruction map.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from typing import Any, Collection, Literal, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import linprog

from pyopl.linear_problem import LinearProblem


class UnsupportedAlgebra(ValueError):
    """Signal an unsupported proof obligation or exhausted budget, not disproof."""

    pass


@dataclass(frozen=True)
class Symbol:
    """A declaration whose kind and original type constrain candidate renaming.

    Integrality is recorded here rather than in SymPy's real-valued symbols;
    dispatch must therefore inspect ``value_type`` before real projection.
    """

    name: str
    kind: Literal["parameter", "variable"]
    value_type: str


@dataclass(frozen=True)
class AffineConstraint:
    """Store the paper's row predicate with zero on the right-hand side.

    For example, ``2*x <= 3`` is represented by ``expression = 2*x - 3``
    and ``sense = '<='``. This convention fixes the signs in projection and
    in the residual identity checked by ``_verify_farkas``.
    """

    expression: sp.Expr
    sense: Literal["<=", "="]


@dataclass(frozen=True)
class SymbolicModel:
    """Scalar affine representation shared by schema and grounded comparisons.

    Nonnegativity and Boolean bounds on decisions are constraint rows so
    substitution preserves them. Assumptions describe admissible parameters.
    An empty parameter tuple permits instance-level projection or enumeration;
    it does not itself establish that the model is continuous.
    """

    parameters: tuple[Symbol, ...]
    variables: tuple[Symbol, ...]
    constraints: tuple[AffineConstraint, ...]
    objective: sp.Expr
    objective_sense: Literal["minimize", "maximize"]
    assumptions: tuple[sp.Expr, ...] = ()


@dataclass(frozen=True)
class AlgebraicProof:
    """Internal outcome summary following Sections 9.1--9.2 of the paper.

    ``level`` identifies a method, including an unsuccessful attempted method.
    A finite-table counterexample is relative to the recorded correspondence.
    The public wrapper adds input scope and arithmetic provenance; ``steps``
    contains descriptions rather than exported transformation certificates.
    """

    status: Literal["equivalent", "different", "unknown"]
    level: Literal[
        "symbolically_normalized",
        "rewrite_certified",
        "polyhedrally_proven",
        "presburger_proven",
    ]
    reason: str
    steps: tuple[str, ...] = ()
    counterexample: str | None = None
    variable_mapping: tuple[tuple[str, str], ...] = ()
    parameter_mapping: tuple[tuple[str, str], ...] = ()
    budget_exhausted: bool = False


@dataclass(frozen=True)
class FarkasCertificate:
    """Exact multipliers for the implication identity of Theorem 7.4.

    Entries follow the premises filtered by sense, retaining their original
    order. Inequality multipliers must be nonnegative; equality multipliers
    may have either sign. An equality conclusion requires two such certificates.
    """

    inequality_multipliers: tuple[Fraction, ...]
    equality_multipliers: tuple[Fraction, ...]


def _register_declaration(
    declaration: Mapping[str, Any],
    parameter_declarations: list[Symbol],
    variable_declarations: list[Symbol],
    symbols: dict[str, sp.Symbol],
    inferred_assumptions: list[sp.Expr],
) -> None:
    """Register scalar types and parameter-domain facts for Section 6.3.

    Range and tuple metadata are skipped, not algebraically compared here;
    indexed and set-valued uses remain outside scalar lowering. A nonnegative
    parameter declaration contributes a weak inequality, never nonzeroness.
    """

    node_type = declaration.get("type")
    name = declaration.get("name")
    if node_type in {"range_declaration_inline", "range_declaration_external", "tuple_type"}:
        return
    if not isinstance(name, str):
        raise UnsupportedAlgebra(f"unsupported declaration in algebraic backend: {node_type}")
    if declaration.get("dimensions") or declaration.get("iterators") or "indexed" in str(node_type):
        raise UnsupportedAlgebra("algebraic backend currently supports scalar declarations only")
    value_type = str(declaration.get("var_type", "unknown"))
    if node_type == "dvar":
        variable_declarations.append(Symbol(name, "variable", value_type))
        symbols[name] = sp.Symbol(name, real=True)
    elif str(node_type).startswith("parameter"):
        parameter_declarations.append(Symbol(name, "parameter", value_type))
        symbols[name] = sp.Symbol(name, real=True)
        if value_type in {"int+", "float+"}:
            inferred_assumptions.append(symbols[name] >= 0)
    elif node_type in {"typed_set", "set_declaration", "set_of_tuples", "tuple_array"}:
        raise UnsupportedAlgebra("algebraic backend does not lower set-valued declarations")
    else:
        raise UnsupportedAlgebra(f"unsupported declaration in algebraic backend: {node_type}")


def _lower_inline_values(
    declarations: list[Any],
    symbols: Mapping[str, sp.Symbol],
) -> dict[str, sp.Expr]:
    """Evaluate supported inline parameter expressions in declaration order.

    The resulting substitutions simplify decision expressions; parameter
    declarations are retained, so this is not the general grounding operation
    of Proposition 6.9 (Correct finite grounding).
    """

    inline_values: dict[str, sp.Expr] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            continue
        name = declaration.get("name")
        if not isinstance(name, str) or name not in symbols:
            continue
        node_type = str(declaration.get("type"))
        value = declaration.get("expression", declaration.get("value"))
        if node_type in {"parameter_inline", "parameter_inline_expr"} and value is not None:
            inline_values[name] = _expression(value, symbols, inline_values, set())
    return inline_values


def _lower_declarations(
    declarations: list[Any],
) -> tuple[list[Symbol], list[Symbol], dict[str, sp.Symbol], dict[str, sp.Expr], list[sp.Expr]]:
    """Collect declaration identities before lowering their inline expressions."""

    parameter_declarations: list[Symbol] = []
    variable_declarations: list[Symbol] = []
    symbols: dict[str, sp.Symbol] = {}
    inferred_assumptions: list[sp.Expr] = []

    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise UnsupportedAlgebra("abstract declaration must be an object")
        _register_declaration(
            declaration,
            parameter_declarations,
            variable_declarations,
            symbols,
            inferred_assumptions,
        )

    inline_values = _lower_inline_values(declarations, symbols)
    return parameter_declarations, variable_declarations, symbols, inline_values, inferred_assumptions


def _lower_explicit_assumptions(
    assumptions: Mapping[str, str] | None,
    symbols: Mapping[str, sp.Symbol],
    inferred_assumptions: list[sp.Expr],
) -> list[sp.Expr]:
    """Translate the finite assumption vocabulary described in Section 6.3.

    This constructs facts, not their logical closure. The public entry point
    validates assumption names; the alias recognizer later checks only its
    supported nonzeroness patterns.
    """

    explicit_assumptions = list(inferred_assumptions)
    builders = {
        "positive": lambda symbol: symbol > 0,
        "nonnegative": lambda symbol: symbol >= 0,
        "nonzero": lambda symbol: sp.Ne(symbol, 0),
    }
    for name, condition in (assumptions or {}).items():
        symbol = symbols.get(name)
        if symbol is None:
            raise UnsupportedAlgebra(f"assumption references unknown symbol: {name}")
        builder = builders.get(condition)
        if builder is None:
            raise UnsupportedAlgebra(f"unsupported assumption condition: {condition}")
        explicit_assumptions.append(builder(symbol))
    return explicit_assumptions


def lower_symbolic_model(
    ast: Mapping[str, Any],
    assumptions: Mapping[str, str] | None = None,
) -> SymbolicModel:
    """Prepare scalar schemas for Proposition 6.6 (Sound symbolic normalization).

    Type-derived decision bounds become rows before any rewrite, ensuring
    that eliminating a variable also substitutes through its domain. Parameters
    remain symbolic and admissibility facts are stored separately. Affineness
    is checked in decisions, allowing parameter-dependent coefficients.
    Unsupported syntax raises ``UnsupportedAlgebra`` rather than being grounded
    implicitly. Exact arithmetic here starts from parsed values.
    """

    declarations = ast.get("declarations")
    if not isinstance(declarations, list):
        raise UnsupportedAlgebra("abstract declarations must be a list")
    parameter_declarations, variable_declarations, symbols, inline_values, inferred_assumptions = _lower_declarations(
        declarations
    )

    constraints: list[AffineConstraint] = []
    for node in ast.get("constraints", []):
        constraints.extend(_lower_constraint(node, symbols, inline_values))

    for variable in variable_declarations:
        variable_symbol = symbols[variable.name]
        if variable.value_type in {"int+", "float+", "boolean"}:
            constraints.append(AffineConstraint(-variable_symbol, "<="))
        if variable.value_type == "boolean":
            constraints.append(AffineConstraint(variable_symbol - 1, "<="))

    objective_node = ast.get("objective")
    if not isinstance(objective_node, Mapping):
        raise UnsupportedAlgebra("abstract objective must be an object")
    objective_sense = objective_node.get("type")
    if objective_sense not in {"minimize", "maximize"}:
        raise UnsupportedAlgebra("unsupported objective sense")
    objective = _expression(objective_node.get("expression"), symbols, inline_values, set())

    model = SymbolicModel(
        parameters=tuple(parameter_declarations),
        variables=tuple(variable_declarations),
        constraints=tuple(constraints),
        objective=sp.expand(objective),
        objective_sense=objective_sense,
        assumptions=tuple(_lower_explicit_assumptions(assumptions, symbols, inferred_assumptions)),
    )
    _validate_affine(model)
    return model


def lower_linear_problem(problem: LinearProblem) -> SymbolicModel:
    """Lower a finite grounded PyOPL matrix model to the symbolic proof IR.

    This adapter lets indexed declarations, sums, and ``forall`` constraints
    use PyOPL's established finite-domain expansion before certified algebraic
    comparison. Proposition 6.9 (Correct finite grounding) restricts the
    conclusion to that supplied instance. Finite bounds are expanded into rows
    as in Proposition 5.5, and an already normalized objective is not negated
    again. Numeric values are embedded through their decimal strings; subsequent
    arithmetic is rational, but this cannot undo prior floating-point rounding.
    """

    variables: list[Symbol] = []
    symbols: dict[str, sp.Symbol] = {}
    constraints: list[AffineConstraint] = []
    for name, integrality in zip(problem.var_names, problem.integrality, strict=True):
        value_type = "int" if integrality else "float"
        variables.append(Symbol(name, "variable", value_type))
        symbols[name] = sp.Symbol(name, real=True)

    def row_expression(row: Sequence[float], rhs: float) -> sp.Expr:
        return sp.expand(
            sum(_rational(coefficient) * symbols[name] for name, coefficient in zip(problem.var_names, row, strict=True))
            - _rational(rhs)
        )

    constraints.extend(
        AffineConstraint(row_expression(row, rhs), "=") for row, rhs in zip(problem.A_eq, problem.b_eq, strict=True)
    )
    constraints.extend(
        AffineConstraint(row_expression(row, rhs), "<=") for row, rhs in zip(problem.A_ub, problem.b_ub, strict=True)
    )
    for name, bounds in zip(problem.var_names, problem.bounds, strict=True):
        lower, upper = bounds
        if lower is not None:
            constraints.append(AffineConstraint(_rational(lower) - symbols[name], "<="))
        if upper is not None:
            constraints.append(AffineConstraint(symbols[name] - _rational(upper), "<="))

    objective = sum(
        _rational(coefficient) * symbols[name] for name, coefficient in zip(problem.var_names, problem.c, strict=True)
    ) + _rational(problem.objective_offset)
    return SymbolicModel(
        parameters=(),
        variables=tuple(variables),
        constraints=tuple(constraints),
        objective=sp.expand(objective),
        objective_sense="minimize" if problem.objective_is_minimization_form else problem.sense,
    )


def _rational(value: object) -> sp.Rational:
    """Embed a stored value's decimal string, not its original source token."""

    return sp.Rational(str(value))


def prove_algebraic_equivalence(
    left: SymbolicModel,
    right: SymbolicModel,
    *,
    parameter_mapping: Mapping[str, str] | None = None,
    variable_mapping: Mapping[str, str] | None = None,
    left_auxiliaries: Collection[str] = (),
    right_auxiliaries: Collection[str] = (),
    max_rewrite_iterations: int = 12,
) -> AlgebraicProof:
    """Search the supported proof routes with the outcome discipline of Section 9.

    One accepted correspondence suffices for a positive result. A disproof
    returned by one candidate cannot reject another untried correspondence;
    a negative result requires exhausting the compatible typed candidate class
    with no inconclusive attempts. At most 256 candidates are tried, with a
    257th candidate used to distinguish truncation from complete exhaustion.
    Empty or truncated searches return ``unknown``. Public callers validate
    declared maps and auxiliary partitions before entering this backend.
    """

    mappings = _candidate_mappings(left, right, parameter_mapping, variable_mapping, left_auxiliaries, right_auxiliaries)
    truncated = len(mappings) > 256
    if not mappings:
        return AlgebraicProof("unknown", "symbolically_normalized", "no compatible parameter and variable mapping found")

    unknown_proof: AlgebraicProof | None = None
    different_proof: AlgebraicProof | None = None
    for parameter_map, variable_map in mappings[:256]:
        try:
            proof = _prove_candidate_mapping(
                left,
                right,
                parameter_map,
                variable_map,
                left_auxiliaries,
                right_auxiliaries,
                max_rewrite_iterations,
            )
        except UnsupportedAlgebra as exc:
            proof = AlgebraicProof("unknown", "rewrite_certified", str(exc), budget_exhausted="limit" in str(exc))
        if proof is None:
            unknown_proof = AlgebraicProof("unknown", "rewrite_certified", "retained coordinates could not be aligned")
            continue
        proof = replace(
            proof,
            variable_mapping=tuple(
                sorted((name, target) for name, target in variable_map.items() if target not in right_auxiliaries)
            ),
            parameter_mapping=tuple(sorted(parameter_map.items())),
        )
        if proof.status == "equivalent":
            return proof
        if proof.status == "unknown":
            if unknown_proof is None or proof.budget_exhausted:
                unknown_proof = proof
        elif different_proof is None:
            different_proof = proof

    if truncated:
        return AlgebraicProof("unknown", "rewrite_certified", "candidate mapping search limit reached", budget_exhausted=True)
    return _unproved_algebraic_result(unknown_proof, different_proof, len(mappings))


def _prove_candidate_mapping(
    left: SymbolicModel,
    right: SymbolicModel,
    parameter_map: Mapping[str, str],
    variable_map: Mapping[str, str],
    left_auxiliaries: Collection[str],
    right_auxiliaries: Collection[str],
    max_rewrite_iterations: int,
) -> AlgebraicProof | None:
    """Align retained decisions before applying the composition argument.

    The right model, including assumptions, is renamed into the left namespace.
    Fresh right-auxiliary names remain distinct from retained coordinates;
    only the retained sets must coincide for the projected-value question.
    """

    renamed_right = _rename_model(right, parameter_map, variable_map)
    left_kept = {variable.name for variable in left.variables if variable.name not in left_auxiliaries}
    right_kept = {left_name for left_name, right_name in variable_map.items() if right_name not in right_auxiliaries}
    if left_kept != right_kept:
        return None
    mapped_right_auxiliaries = {left_name for left_name, right_name in variable_map.items() if right_name in right_auxiliaries}
    return _prove_mapped_models(
        left,
        renamed_right,
        left_kept,
        set(left_auxiliaries),
        mapped_right_auxiliaries,
        max_rewrite_iterations,
    )


def _unproved_algebraic_result(
    unknown_proof: AlgebraicProof | None,
    different_proof: AlgebraicProof | None,
    mapping_count: int,
) -> AlgebraicProof:
    """Aggregate an exhausted search without turning an unknown into disproof.

    When all candidates were disproved, the retained witness illustrates one
    recorded map; it is not a single witness against every correspondence.
    """

    if unknown_proof is not None:
        return unknown_proof
    if different_proof is not None:
        return replace(
            different_proof,
            reason=f"all {mapping_count} compatible typed coordinate mappings were disproved; witness shown for the recorded mapping",
        )
    return AlgebraicProof(
        "unknown",
        "symbolically_normalized",
        "no compatible mapping produced equivalent normalized models",
    )


def _prove_mapped_models(
    left: SymbolicModel,
    right: SymbolicModel,
    kept_variables: set[str],
    left_auxiliaries: set[str],
    right_auxiliaries: set[str],
    max_iterations: int,
) -> AlgebraicProof:
    """Compare aligned models by Theorems 6.7, 7.6, or 7.9 as applicable.

    First require matching assumption sets and perform certified local
    rewrites. Matching terminal expressions can establish a uniform schema
    result. Remaining parameterized implication is unsupported. Without
    parameters, wholly integer models use finite value tables and wholly
    continuous models use projection plus objective implication. This dispatch
    does not apply real elimination to an unresolved integer coordinate.
    """

    if set(left.assumptions) != set(right.assumptions):
        raise UnsupportedAlgebra("mapped admissibility assumptions do not match")
    left_normalized, left_steps = _saturate(left, left_auxiliaries, max_iterations)
    right_normalized, right_steps = _saturate(right, right_auxiliaries, max_iterations)
    steps = (
        "lowered parser ASTs to typed symbolic models",
        "normalized exact affine expressions",
        *left_steps,
        *right_steps,
    )

    if _symbolic_models_equal(left_normalized, right_normalized):
        return AlgebraicProof(
            "equivalent",
            "rewrite_certified" if left_steps or right_steps else "symbolically_normalized",
            "certified symbolic normal forms are equal",
            tuple(dict.fromkeys(steps)),
        )

    all_parameters = left_normalized.parameters + right_normalized.parameters
    if all_parameters:
        return AlgebraicProof(
            "unknown",
            "rewrite_certified",
            "remaining parameterized implication requires an SMT or quantified algebra backend",
            tuple(dict.fromkeys(steps)),
        )

    integer_variables = {
        variable.name
        for variable in left_normalized.variables + right_normalized.variables
        if variable.value_type in {"int", "int+", "boolean"}
    }
    if integer_variables:
        if any(
            variable.value_type not in {"int", "int+", "boolean"}
            for variable in left_normalized.variables + right_normalized.variables
        ):
            return AlgebraicProof(
                "unknown",
                "presburger_proven",
                "mixed integer/continuous projection requires a quantified MILP backend",
                tuple(dict.fromkeys(steps)),
            )
        return _prove_finite_integer_models(left_normalized, right_normalized, kept_variables, steps)

    left_projected = _project_continuous(left_normalized, left_auxiliaries)
    right_projected = _project_continuous(right_normalized, right_auxiliaries)
    implication_steps = steps + ("eliminated continuous auxiliaries by exact Fourier-Motzkin projection",)
    equivalent, certificates = _polyhedra_equal(left_projected, right_projected, sorted(kept_variables))
    if not equivalent:
        return AlgebraicProof(
            "unknown",
            "polyhedrally_proven",
            "no verified certificate established equality of projected affine feasible sets",
            tuple(dict.fromkeys(implication_steps)),
        )
    objective_equal = _objectives_equal_on_polyhedron(left_projected, right_projected, sorted(kept_variables))
    if not objective_equal:
        return AlgebraicProof(
            "unknown",
            "polyhedrally_proven",
            "no verified certificate established objective equality on the projected feasible set",
            tuple(dict.fromkeys(implication_steps)),
        )
    return AlgebraicProof(
        "equivalent",
        "polyhedrally_proven",
        f"projected polyhedra and objectives are equal ({certificates} checked Farkas certificates)",
        tuple(dict.fromkeys(implication_steps + ("verified exact rational Farkas certificates",))),
    )


def _expression(
    node: Any,
    symbols: Mapping[str, sp.Symbol],
    inline_values: Mapping[str, sp.Expr],
    resolving: set[str],
) -> sp.Expr:
    """Translate supported scalar syntax into the expressions of Section 6.3.

    Parsed numeric values become rational constants. Translation alone does
    not prove affineness; ``_validate_affine`` checks the collected model.
    """

    if isinstance(node, bool):
        return sp.Integer(int(node))
    if isinstance(node, int):
        return sp.Integer(node)
    if isinstance(node, float):
        return sp.Rational(str(node))
    if not isinstance(node, Mapping):
        raise UnsupportedAlgebra(f"unsupported symbolic expression: {node!r}")
    node_type = node.get("type")
    if node_type == "number":
        return sp.Rational(str(node.get("value")))
    if node_type == "boolean_literal":
        return sp.Integer(int(bool(node.get("value"))))
    if node_type == "parenthesized_expression":
        return _expression(node.get("expression"), symbols, inline_values, resolving)
    if node_type == "uminus":
        return -_expression(node.get("value"), symbols, inline_values, resolving)
    if node_type == "name":
        return _expression_name(node, symbols, inline_values, resolving)
    if node_type == "binop":
        result = _expression_binop(node, symbols, inline_values, resolving)
        if result is not None:
            return result
    raise UnsupportedAlgebra(f"unsupported symbolic expression node: {node_type}")


def _expression_name(
    node: Mapping[str, Any],
    symbols: Mapping[str, sp.Symbol],
    inline_values: Mapping[str, sp.Expr],
    resolving: set[str],
) -> sp.Expr:
    """Resolve a scalar declaration, substituting an available inline value."""

    name = node.get("value")
    if not isinstance(name, str) or name not in symbols:
        raise UnsupportedAlgebra(f"unknown symbolic name: {name}")
    if name in inline_values:
        if name in resolving:
            raise UnsupportedAlgebra(f"cyclic computed parameter: {name}")
        return inline_values[name]
    return symbols[name]


def _expression_binop(
    node: Mapping[str, Any],
    symbols: Mapping[str, sp.Symbol],
    inline_values: Mapping[str, sp.Expr],
    resolving: set[str],
) -> sp.Expr | None:
    """Lower arithmetic while declining general symbolic source division.

    Alias substitution has a separate, restricted nonzeroness check; its
    supported divisions do not imply support for arbitrary input denominators.
    """

    left = _expression(node.get("left"), symbols, inline_values, resolving)
    right = _expression(node.get("right"), symbols, inline_values, resolving)
    operator = node.get("op")
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator != "/":
        return None
    if right.free_symbols & set(symbols.values()):
        raise UnsupportedAlgebra("division by symbolic expressions requires side-condition proving")
    return left / right


def _lower_constraint(
    node: Any,
    symbols: Mapping[str, sp.Symbol],
    inline_values: Mapping[str, sp.Expr],
) -> list[AffineConstraint]:
    """Move a scalar row to zero and reverse >= before exact normalization."""

    if not isinstance(node, Mapping) or node.get("type") != "constraint":
        raise UnsupportedAlgebra("algebraic backend currently supports scalar affine constraints only")
    operator = node.get("op")
    if operator not in {"<=", ">=", "=="}:
        raise UnsupportedAlgebra(f"unsupported algebraic constraint operator: {operator}")
    left = _expression(node.get("left"), symbols, inline_values, set())
    right = _expression(node.get("right"), symbols, inline_values, set())
    expression = sp.expand(left - right)
    if operator == ">=":
        expression = -expression
    return [AffineConstraint(expression, "=" if operator == "==" else "<=")]


def _validate_affine(model: SymbolicModel) -> None:
    """Require degree at most one in decisions, treating parameters as coefficients.

    Thus ``theta*x`` can be affine for each parameter valuation without being
    linear jointly in all quantified symbols; see the limitation in Section 6.3.
    """

    variable_symbols = [sp.Symbol(variable.name, real=True) for variable in model.variables]
    if not variable_symbols:
        return
    for expression in [model.objective, *(constraint.expression for constraint in model.constraints)]:
        try:
            polynomial = sp.Poly(sp.expand(expression), *variable_symbols)
        except sp.PolynomialError as exc:
            raise UnsupportedAlgebra("model is not polynomial in its decision variables") from exc
        if polynomial.total_degree() > 1:
            raise UnsupportedAlgebra("algebraic backend supports affine decision expressions only")


def _candidate_mappings(
    left: SymbolicModel,
    right: SymbolicModel,
    parameter_mapping: Mapping[str, str] | None,
    variable_mapping: Mapping[str, str] | None,
    left_auxiliaries: Collection[str],
    right_auxiliaries: Collection[str],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Combine typed parameter and retained-variable maps, with an overflow probe.

    Auxiliary spaces need not have equal dimensions. Right auxiliaries receive
    collision-free names for substitution, not matched retained counterparts.
    Returning 257 candidates tells the dispatcher that its 256-attempt budget
    cannot exhaust the search.
    """

    parameter_maps = _typed_bijections(left.parameters, right.parameters, parameter_mapping)
    left_primary = tuple(symbol for symbol in left.variables if symbol.name not in left_auxiliaries)
    right_primary = tuple(symbol for symbol in right.variables if symbol.name not in right_auxiliaries)
    variable_maps = _typed_bijections(left_primary, right_primary, variable_mapping)
    results: list[tuple[dict[str, str], dict[str, str]]] = []
    for parameter_map in parameter_maps:
        for variable_map in variable_maps:
            full_variable_map = dict(variable_map)
            occupied = {symbol.name for symbol in left.variables + left.parameters}
            for auxiliary in right_auxiliaries:
                auxiliary_name = f"__right_aux_{auxiliary}"
                while auxiliary_name in occupied:
                    auxiliary_name = "_" + auxiliary_name
                occupied.add(auxiliary_name)
                full_variable_map[auxiliary_name] = auxiliary
            results.append((parameter_map, full_variable_map))
            if len(results) >= 257:
                return results
    return results


def _typed_bijections(
    left: Sequence[Symbol],
    right: Sequence[Symbol],
    required: Mapping[str, str] | None,
) -> list[dict[str, str]]:
    """Enumerate up to 257 renamings extending the caller's prescribed pairs.

    Matching uses literal declaration types, not semantic equivalence of domain
    predicates. Failure to find a candidate therefore does not disprove the
    broader equivalence relations of Section 4.4.
    """

    if len(left) != len(right):
        return []
    required = dict(required or {})
    left_by_name = {symbol.name: symbol for symbol in left}
    right_by_name = {symbol.name: symbol for symbol in right}
    if any(name not in left_by_name for name in required) or any(name not in right_by_name for name in required.values()):
        return []
    if len(set(required.values())) != len(required):
        return []
    candidates: list[dict[str, str]] = []

    def visit(index: int, current: dict[str, str], used: set[str]) -> None:
        if len(candidates) >= 257:
            return
        if index == len(left):
            candidates.append(dict(current))
            return
        left_symbol = left[index]
        choices = [required[left_symbol.name]] if left_symbol.name in required else [symbol.name for symbol in right]
        for right_name in choices:
            right_symbol = right_by_name.get(right_name)
            if right_symbol is None or right_name in used or right_symbol.value_type != left_symbol.value_type:
                continue
            current[left_symbol.name] = right_name
            used.add(right_name)
            visit(index + 1, current, used)
            used.remove(right_name)
            del current[left_symbol.name]

    visit(0, {}, set())
    return candidates


def _rename_model(
    model: SymbolicModel,
    parameter_mapping: Mapping[str, str],
    variable_mapping: Mapping[str, str],
) -> SymbolicModel:
    """Apply simultaneous right-to-left renaming to rows, costs, and assumptions.

    Simultaneous replacement preserves swaps without cascading substitutions.
    Normalization then uses one shared namespace for equality orientation.
    """

    reverse_names = {right: left for left, right in parameter_mapping.items()}
    reverse_names.update({right: left for left, right in variable_mapping.items()})
    replacements = {sp.Symbol(old, real=True): sp.Symbol(new, real=True) for old, new in reverse_names.items()}
    return SymbolicModel(
        parameters=tuple(replace(symbol, name=reverse_names.get(symbol.name, symbol.name)) for symbol in model.parameters),
        variables=tuple(replace(symbol, name=reverse_names.get(symbol.name, symbol.name)) for symbol in model.variables),
        constraints=tuple(
            AffineConstraint(sp.expand(constraint.expression.xreplace(replacements)), constraint.sense)
            for constraint in model.constraints
        ),
        objective=sp.expand(model.objective.xreplace(replacements)),
        objective_sense=model.objective_sense,
        assumptions=tuple(assumption.xreplace(replacements) for assumption in model.assumptions),
    )


def _saturate(
    model: SymbolicModel,
    auxiliaries: set[str],
    max_iterations: int,
) -> tuple[SymbolicModel, tuple[str, ...]]:
    """Build the bounded rewrite chain justified by Theorem 6.7 (Certified rewrite chains).

    Each accepted alias is substituted throughout the model and the result is
    normalized again. ``auxiliaries`` is a caller-owned working set: eliminated
    names are removed in place so projection sees only surviving auxiliaries.
    Exhausting iterations before observing a fixed point remains inconclusive.
    """

    current = _normalize_model(model)
    steps: list[str] = []
    for _ in range(max_iterations):
        rewritten = _eliminate_affine_alias(current, auxiliaries)
        if rewritten is None:
            break
        current, eliminated = rewritten
        auxiliaries.discard(eliminated)
        steps.append("eliminated affine aliases with exact substitution certificates")
        current = _normalize_model(current)
    else:
        raise UnsupportedAlgebra("rewrite saturation iteration limit reached")
    return current, tuple(steps)


def _normalize_model(model: SymbolicModel) -> SymbolicModel:
    """Normalize minimization costs and rows using Propositions 5.5 and 6.6.

    Exact duplicate removal and sorting provide a comparison form, not a
    canonical representation of every semantically equivalent polyhedron.
    """

    objective = sp.expand(-model.objective if model.objective_sense == "maximize" else model.objective)
    constraints = tuple(sorted({_normalize_constraint(constraint) for constraint in model.constraints}, key=_constraint_key))
    return replace(model, constraints=constraints, objective=objective, objective_sense="minimize")


def _normalize_constraint(constraint: AffineConstraint) -> AffineConstraint:
    """Remove positive rational content, preserving the row predicate.

    For example, ``2*x + 4*y - 6 <= 0`` becomes ``x + 2*y - 3 <= 0``.
    Only equalities may additionally change sign. No symbolic factor is
    cancelled without a side-condition proof; the shared renamed namespace
    makes the equality ordering comparable across the two models.
    """

    expression = sp.expand(constraint.expression)
    coefficient, _ = expression.as_content_primitive()
    if coefficient != 0:
        expression = sp.expand(expression / coefficient)
    if constraint.sense == "=":
        ordered = sorted(expression.as_ordered_terms(), key=sp.default_sort_key)
        if ordered and ordered[0].could_extract_minus_sign():
            expression = -expression
    return AffineConstraint(sp.expand(expression), constraint.sense)


def _constraint_key(constraint: AffineConstraint) -> tuple[str, str]:
    """Order normalized predicates syntactically, without an implication test."""

    return constraint.sense, sp.srepr(constraint.expression)


def _symbolic_models_equal(left: SymbolicModel, right: SymbolicModel) -> bool:
    """Test the sufficient normal-form agreement of Proposition 6.6.

    The caller has already aligned types and assumptions. A failed expression
    comparison may reflect different but equivalent row systems, so subsequent
    supported methods may still prove equivalence.
    """

    return (
        sp.simplify(left.objective - right.objective) == 0
        and left.objective_sense == right.objective_sense
        and len(left.constraints) == len(right.constraints)
        and all(
            left_constraint.sense == right_constraint.sense
            and sp.simplify(left_constraint.expression - right_constraint.expression) == 0
            for left_constraint, right_constraint in zip(left.constraints, right.constraints, strict=True)
        )
    )


def _eliminate_affine_alias(
    model: SymbolicModel,
    auxiliaries: set[str],
) -> tuple[SymbolicModel, str] | None:
    """Recognize one continuous auxiliary elimination under Lemma 5.1.

    Requiring a unique defining equality is a conservative recognition rule,
    not a necessary condition of the lemma. Integer auxiliaries are excluded:
    substituting ``half = x/2`` must not erase the requirement that x be even.
    """

    for auxiliary in sorted(auxiliaries):
        if not _is_continuous_auxiliary(model, auxiliary):
            continue
        symbol = sp.Symbol(auxiliary, real=True)
        defining = _unique_defining_constraint(model, symbol)
        if defining is None:
            continue
        replacement = _alias_replacement(model, defining, symbol)
        if replacement is None:
            continue
        return _substitute_affine_alias(model, auxiliary, defining, symbol, replacement), auxiliary
    return None


def _is_continuous_auxiliary(model: SymbolicModel, auxiliary: str) -> bool:
    """Keep integer reconstruction obligations outside continuous alias removal."""

    variable = next((variable for variable in model.variables if variable.name == auxiliary), None)
    return variable is not None and variable.value_type not in {"int", "int+", "boolean"}


def _unique_defining_constraint(model: SymbolicModel, symbol: sp.Symbol) -> AffineConstraint | None:
    """Select a single equality containing the auxiliary, when unambiguous."""

    defining = [
        constraint
        for constraint in model.constraints
        if constraint.sense == "=" and symbol in constraint.expression.free_symbols
    ]
    return defining[0] if len(defining) == 1 else None


def _alias_replacement(
    model: SymbolicModel,
    defining: AffineConstraint,
    symbol: sp.Symbol,
) -> sp.Expr | None:
    """Obtain the reconstruction formula only after justifying division.

    For ``alpha*a + q(x) = 0``, Lemma 5.1 reconstructs ``a = -q(x)/alpha``.
    A symbolic alpha must be nonzero on every admitted parameter valuation;
    solving the equation symbolically without that condition would be unsound.
    """

    coefficient = sp.expand(defining.expression).coeff(symbol)
    if coefficient == 0 or symbol in coefficient.free_symbols:
        return None
    if coefficient.free_symbols and not _assumptions_prove_nonzero(coefficient, model.assumptions):
        return None
    replacements = sp.solve(defining.expression, symbol, dict=False)
    return replacements[0] if len(replacements) == 1 else None


def _substitute_affine_alias(
    model: SymbolicModel,
    auxiliary: str,
    defining: AffineConstraint,
    symbol: sp.Symbol,
    replacement: sp.Expr,
) -> SymbolicModel:
    """Carry Lemma 5.1's reconstruction through every remaining row and cost.

    Domain bounds already occur among these rows. Thus substituting ``a=2-x``
    through ``0<=a<=1`` preserves ``1<=x<=2`` rather than discarding a's bounds.
    The defining equality and auxiliary declaration can then be removed.
    """

    substituted_constraints = tuple(
        AffineConstraint(sp.expand(constraint.expression.subs(symbol, replacement)), constraint.sense)
        for constraint in model.constraints
        if constraint is not defining
    )
    return replace(
        model,
        variables=tuple(variable for variable in model.variables if variable.name != auxiliary),
        constraints=substituted_constraints,
        objective=sp.expand(model.objective.subs(symbol, replacement)),
    )


def _assumptions_prove_nonzero(expression: sp.Expr, assumptions: Sequence[sp.Expr]) -> bool:
    """Recognize the restricted uniform nonzeroness facts of Section 6.3.

    Numeric nonzero values, SymPy's established nonzeroness, and matching
    strict inequalities or disequalities suffice after removing a rational
    multiplicative factor. A fact such as ``theta >= 0`` permits theta=0
    and must never license division. This is not general assumption solving.
    """

    if expression.is_number:
        return expression != 0
    _, expression = expression.as_coeff_Mul()
    if expression.is_nonzero is True:
        return True
    for assumption in assumptions:
        if assumption == sp.Ne(expression, 0) or assumption == (expression > 0) or assumption == (expression < 0):
            return True
    return False


def _project_continuous(model: SymbolicModel, auxiliaries: set[str]) -> SymbolicModel:
    """Eliminate real auxiliaries by Lemma 7.1 and Corollary 7.2.

    An equality contributes opposite inequalities. For ``alpha*a + r <= 0``,
    positive alpha gives the upper bound ``a <= -r/alpha``; negative alpha
    gives the lower bound ``a >= r/(-alpha)``. Every lower/upper pair yields
    ``r_upper/alpha_upper + r_lower/(-alpha_lower) <= 0``. Zero-coefficient
    rows survive unchanged. With only one bound direction, a real completion
    always exists and those bounds add no restriction on retained variables.

    This is feasibility projection. Lemma 4.4 makes it sufficient for values
    only when the eliminated coordinates are objective-neutral; unresolved
    objective-bearing auxiliaries raise ``UnsupportedAlgebra``. The caller
    guarantees continuous domains and resolved coefficients. The returned
    rows are computed exactly, but no elimination trace or reconstruction
    function is exported, and pairwise row growth can be large (Section 10.3).
    """

    constraints = [
        AffineConstraint(sign * constraint.expression, "<=")
        for constraint in model.constraints
        for sign in ((1, -1) if constraint.sense == "=" else (1,))
    ]
    objective = model.objective
    for auxiliary in sorted(auxiliaries):
        symbol = sp.Symbol(auxiliary, real=True)
        if symbol in objective.free_symbols:
            raise UnsupportedAlgebra("projected auxiliary remains in the objective")
        positive: list[sp.Expr] = []
        negative: list[sp.Expr] = []
        zero: list[AffineConstraint] = []
        for constraint in constraints:
            coefficient = sp.expand(constraint.expression).coeff(symbol)
            remainder = sp.expand(constraint.expression - coefficient * symbol)
            if not coefficient.is_Rational:
                raise UnsupportedAlgebra("Fourier-Motzkin projection requires rational auxiliary coefficients")
            if coefficient > 0:
                positive.append(sp.expand(remainder / coefficient))
            elif coefficient < 0:
                negative.append(sp.expand(remainder / -coefficient))
            else:
                zero.append(constraint)
        combined = [AffineConstraint(sp.expand(upper + lower), "<=") for upper in positive for lower in negative]
        constraints = zero + combined
    return _normalize_model(
        replace(
            model, constraints=tuple(constraints), variables=tuple(v for v in model.variables if v.name not in auxiliaries)
        )
    )


def _polyhedra_equal(left: SymbolicModel, right: SymbolicModel, variables: list[str]) -> tuple[bool, int]:
    """Attempt both inclusions required by Theorem 7.6.

    Every target row must follow from the source in each direction. ``False``
    means a certificate was not obtained, not that a violating point was found.
    The count records accepted target-row implications, not all multipliers
    or the separate objective checks.
    """

    certificates = 0
    for source, target in ((left, right), (right, left)):
        for constraint in target.constraints:
            certificate = _farkas_certificate(source.constraints, constraint, variables)
            if certificate is None:
                return False, certificates
            certificates += 1
    return True, certificates


def _farkas_numeric_premises(
    premises: Sequence[AffineConstraint],
    variables: list[str],
) -> tuple[list[tuple[list[sp.Expr], sp.Expr]], list[tuple[list[sp.Expr], sp.Expr]]]:
    """Separate rational inequality and equality vectors for Theorem 7.4."""

    inequalities = [constraint for constraint in premises if constraint.sense == "<="]
    equalities = [constraint for constraint in premises if constraint.sense == "="]
    premise_vectors = [_numeric_row(constraint.expression, variables) for constraint in inequalities]
    equality_vectors = [_numeric_row(constraint.expression, variables) for constraint in equalities]
    return premise_vectors, equality_vectors


def _solve_farkas_multipliers(
    premise_vectors: list[tuple[list[sp.Expr], sp.Expr]],
    equality_vectors: list[tuple[list[sp.Expr], sp.Expr]],
    conclusion_coefficients: list[sp.Expr],
    conclusion_constant: sp.Expr,
    variables: list[str],
) -> np.ndarray | None:
    """Ask HiGHS for candidate multipliers, not an accepted exact proof.

    The LP variables are nonnegative inequality multipliers and positive/
    negative parts of unrestricted equality multipliers. A zero objective
    asks only for feasibility. Non-optimal status or absent values mean
    discovery failed; Section 8 requires exact checking before acceptance.
    """

    variable_count = len(premise_vectors) + 2 * len(equality_vectors)
    if variable_count == 0:
        return _trivial_farkas_solution(conclusion_coefficients, conclusion_constant)

    equality_matrix, equality_rhs, upper_matrix = _farkas_lp_matrices(
        premise_vectors,
        equality_vectors,
        conclusion_coefficients,
        variables,
    )
    result = linprog(
        c=np.zeros(variable_count),
        A_ub=np.asarray(upper_matrix),
        b_ub=np.asarray([-float(conclusion_constant)]),
        A_eq=np.asarray(equality_matrix) if equality_matrix else None,
        b_eq=np.asarray(equality_rhs) if equality_rhs else None,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if result.status != 0 or result.x is None:
        return None
    return result.x


def _trivial_farkas_solution(conclusion_coefficients: list[sp.Expr], conclusion_constant: sp.Expr) -> np.ndarray | None:
    """Recognize a constant true inequality when there are no premises."""

    if conclusion_constant <= 0 and all(value == 0 for value in conclusion_coefficients):
        return np.asarray([])
    return None


def _farkas_lp_matrices(
    premise_vectors: list[tuple[list[sp.Expr], sp.Expr]],
    equality_vectors: list[tuple[list[sp.Expr], sp.Expr]],
    conclusion_coefficients: list[sp.Expr],
    variables: list[str],
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Encode coefficient matching and the constant inequality for discovery.

    Rows are stored as ``a*x + k <= 0``. The weighted variable coefficients
    must equal the conclusion's coefficients, while the weighted constants
    must be at least its constant. Negating this last comparison gives the
    LP's upper-bound row. Equality multipliers are split into nonnegative
    positive and negative parts, hence the paired columns with opposite signs.
    """

    equality_matrix = [
        [float(row[0][index]) for row in premise_vectors]
        + [float(row[0][index]) for row in equality_vectors]
        + [-float(row[0][index]) for row in equality_vectors]
        for index in range(len(variables))
    ]
    equality_rhs = [float(coefficient) for coefficient in conclusion_coefficients]
    upper_matrix = [
        [-float(row[1]) for row in premise_vectors]
        + [-float(row[1]) for row in equality_vectors]
        + [float(row[1]) for row in equality_vectors]
    ]
    return equality_matrix, equality_rhs, upper_matrix


def _farkas_certificate(
    premises: Sequence[AffineConstraint],
    conclusion: AffineConstraint,
    variables: list[str],
) -> FarkasCertificate | tuple[FarkasCertificate, FarkasCertificate] | None:
    """Discover, rationalize, and verify a Theorem 7.4 implication certificate.

    Float proposals are reconstructed with denominator at most one million;
    this heuristic can miss an existing rational certificate. Only exact
    verification permits acceptance. An equality conclusion needs both signs,
    returned as a pair. ``None`` always means this search was inconclusive,
    including when the premises are infeasible and this particular search
    does not find the required implication multipliers.
    """

    if conclusion.sense == "=":
        positive = _farkas_certificate(premises, AffineConstraint(conclusion.expression, "<="), variables)
        negative = _farkas_certificate(premises, AffineConstraint(-conclusion.expression, "<="), variables)
        return (
            (positive, negative)
            if isinstance(positive, FarkasCertificate) and isinstance(negative, FarkasCertificate)
            else None
        )
    premise_vectors, equality_vectors = _farkas_numeric_premises(premises, variables)
    conclusion_coefficients, conclusion_constant = _numeric_row(conclusion.expression, variables)
    multipliers = _solve_farkas_multipliers(
        premise_vectors,
        equality_vectors,
        conclusion_coefficients,
        conclusion_constant,
        variables,
    )
    if multipliers is None:
        return None
    if not premise_vectors and not equality_vectors:
        return FarkasCertificate((), ())
    inequalities = [constraint for constraint in premises if constraint.sense == "<="]
    equalities = [constraint for constraint in premises if constraint.sense == "="]
    fractions = [Fraction(float(value)).limit_denominator(1_000_000) for value in multipliers]
    inequality_multipliers = tuple(fractions[: len(inequalities)])
    positive_equalities = fractions[len(inequalities) : len(inequalities) + len(equalities)]
    negative_equalities = fractions[len(inequalities) + len(equalities) :]
    equality_multipliers = tuple(pos - neg for pos, neg in zip(positive_equalities, negative_equalities, strict=True))
    certificate = FarkasCertificate(inequality_multipliers, equality_multipliers)
    return certificate if _verify_farkas(premises, conclusion, variables, certificate) else None


def _verify_farkas(
    premises: Sequence[AffineConstraint],
    conclusion: AffineConstraint,
    variables: list[str],
    certificate: FarkasCertificate,
) -> bool:
    """Check the exact implication identity in the zero-right-hand-side convention.

    For inequality premises e_i<=0 and equality premises h_j=0, construct
    ``weighted = sum(lambda_i*e_i) + sum(mu_j*h_j)`` with lambda_i>=0.
    If ``weighted - conclusion = delta`` is constant and delta>=0, then
    ``conclusion = weighted - delta <= 0`` at every feasible point. Vanishing
    variable coefficients and the residual sign are checked rationally.
    This verifies one inequality direction; the caller handles equalities
    with two certificates. See Theorem 7.4 and Appendix A.3.
    """

    inequalities = [constraint for constraint in premises if constraint.sense == "<="]
    equalities = [constraint for constraint in premises if constraint.sense == "="]
    weighted = sp.Integer(0)
    for multiplier, constraint in zip(certificate.inequality_multipliers, inequalities, strict=True):
        if multiplier < 0:
            return False
        weighted += sp.Rational(multiplier.numerator, multiplier.denominator) * constraint.expression
    for multiplier, constraint in zip(certificate.equality_multipliers, equalities, strict=True):
        weighted += sp.Rational(multiplier.numerator, multiplier.denominator) * constraint.expression
    difference = sp.expand(weighted - conclusion.expression)
    coefficients, constant = _numeric_row(difference, variables)
    return all(coefficient == 0 for coefficient in coefficients) and constant >= 0


def _numeric_row(expression: sp.Expr, variables: list[str]) -> tuple[list[Fraction], Fraction]:
    """Extract rational coefficients and the residual constant in a fixed order.

    Any remaining symbolic or non-rational residual is unsupported rather
    than silently coerced to a numerical approximation.
    """

    symbols = [sp.Symbol(name, real=True) for name in variables]
    expanded = sp.expand(expression)
    coefficients = [_as_fraction(expanded.coeff(symbol)) for symbol in symbols]
    constant = _as_fraction(
        expanded
        - sum(
            sp.Rational(value.numerator, value.denominator) * symbol
            for value, symbol in zip(coefficients, symbols, strict=True)
        )
    )
    return coefficients, constant


def _as_fraction(value: sp.Expr) -> Fraction:
    """Require an exact rational after simplification; do not approximate symbols."""

    simplified = sp.simplify(value)
    if not simplified.is_Rational:
        raise UnsupportedAlgebra("polyhedral proof requires rational coefficients")
    return Fraction(int(simplified.p), int(simplified.q))


def _objectives_equal_on_polyhedron(left: SymbolicModel, right: SymbolicModel, variables: list[str]) -> bool:
    """Prove objective equality on the common domain via Proposition 7.5.

    The caller must already have established equal projected feasible sets.
    If the objective difference is not identically zero, prove both its
    nonpositivity and nonnegativity on the left domain. This recognizes, for
    example, x and 1-y under x+y=1 without requiring equal coefficient vectors.
    A failed certificate attempt is inconclusive.
    """

    if left.objective_sense != right.objective_sense:
        return False
    difference = sp.expand(left.objective - right.objective)
    if difference == 0:
        return True
    return (
        _farkas_certificate(left.constraints, AffineConstraint(difference, "<="), variables) is not None
        and _farkas_certificate(left.constraints, AffineConstraint(-difference, "<="), variables) is not None
    )


def _prove_finite_integer_models(
    left: SymbolicModel,
    right: SymbolicModel,
    kept_variables: set[str],
    steps: tuple[str, ...],
) -> AlgebraicProof:
    """Compare finite retained-value tables as in Theorem 7.9.

    A table records the best completion cost at each feasible retained
    assignment. A missing key or a different minimum disproves projected-value
    equivalence under this candidate map. The counterexample identifies which
    table contains the displayed assignment/value pair; a cost mismatch need
    not mean that the assignment itself is feasible on only one side.
    """

    left_points = _enumerate_integer_projection(left, kept_variables)
    right_points = _enumerate_integer_projection(right, kept_variables)
    if left_points is None or right_points is None:
        return AlgebraicProof(
            "unknown",
            "presburger_proven",
            "integer projection requires finite constant bounds with at most 100000 assignments",
            tuple(dict.fromkeys(steps)),
        )
    if left_points == right_points:
        return AlgebraicProof(
            "equivalent",
            "presburger_proven",
            "bounded integer feasible assignments and objective values are identical",
            tuple(dict.fromkeys(steps + ("exhaustively eliminated bounded integer auxiliaries",))),
        )
    assignment, objective = min(left_points.symmetric_difference(right_points))
    rendered_assignment = ", ".join(f"{name}={value}" for name, value in assignment)
    return AlgebraicProof(
        "different",
        "presburger_proven",
        "bounded integer projections differ",
        tuple(dict.fromkeys(steps + ("exhaustively eliminated bounded integer auxiliaries",))),
        counterexample=f"projected assignment/objective occurs only in {'left' if (assignment, objective) in left_points else 'right'} value table: {rendered_assignment}; objective={objective}",
    )


def _enumerate_integer_projection(
    model: SymbolicModel,
    kept_variables: set[str],
) -> set[tuple[tuple[tuple[str, int], ...], Fraction]] | None:
    """Construct the finite fiber-value function in equation (14).

    Enumerate all coordinates, including auxiliaries, inside a proved finite
    box and test every row exactly. Feasible completions are grouped by the
    retained assignment, keeping only the minimum normalized cost. Empty
    fibers have no entry; finite nonempty fibers attain their minima. Unlike
    continuous feasibility projection, this permits objective-bearing integer
    auxiliaries. Missing recognized bounds return None, not an empty table.
    """

    variables = [variable.name for variable in model.variables]
    bounds = _bounded_integer_domains(model.constraints, variables)
    if bounds is None:
        return None

    points: dict[tuple[tuple[str, int], ...], Fraction] = {}
    ranges = [range(lower, upper + 1) for lower, upper in bounds.values()]
    symbols = {name: sp.Symbol(name, real=True) for name in variables}
    for values in product(*ranges):
        assignment = dict(zip(variables, values, strict=True))
        substitutions = {symbols[name]: value for name, value in assignment.items()}
        if not all(_constraint_holds(constraint, substitutions) for constraint in model.constraints):
            continue
        _record_integer_projection_point(model, assignment, substitutions, kept_variables, points)
    return {(key, objective) for key, objective in points.items()}


def _bounded_integer_domains(
    constraints: Sequence[AffineConstraint],
    variables: Sequence[str],
) -> dict[str, tuple[int, int]] | None:
    """Recognize finite boxes for Theorem 7.9 within the enumeration budget.

    Contradictory recognized bounds give an empty box. Missing bounds leave
    the procedure unsupported, while a Cartesian product above 100000
    assignments raises a budget exception. These are different from proving
    the model unbounded or infeasible by a general integer solver.
    """

    bounds: dict[str, tuple[int, int]] = {}
    assignment_count = 1
    for variable in variables:
        lower, upper = _constant_integer_bounds(constraints, variable, variables)
        if lower is None or upper is None:
            return None
        bounds[variable] = (lower, upper)
        if lower > upper:
            return {name: (1, 0) for name in variables}
        assignment_count *= upper - lower + 1
        if assignment_count > 100_000:
            raise UnsupportedAlgebra("integer enumeration limit of 100000 assignments reached")
    return bounds


def _record_integer_projection_point(
    model: SymbolicModel,
    assignment: dict[str, int],
    substitutions: Mapping[sp.Symbol, int],
    kept_variables: set[str],
    points: dict[tuple[tuple[str, int], ...], Fraction],
) -> None:
    """Minimize over completions of one retained decision, as in Section 7.2.1."""

    key = tuple(sorted((name, assignment[name]) for name in kept_variables))
    objective = _as_fraction(sp.expand(model.objective.subs(substitutions)))
    incumbent = points.get(key)
    if incumbent is None or objective < incumbent:
        points[key] = objective


def _constant_integer_bounds(
    constraints: Sequence[AffineConstraint],
    variable: str,
    all_variables: Sequence[str],
) -> tuple[int | None, int | None]:
    """Read single-coordinate rational inequalities as inclusive integer bounds.

    Positive coefficients yield floored upper bounds and negative coefficients
    yield ceiled lower bounds. Coupled rows and equalities are not used for
    bound inference, even if together they imply a finite domain. The resulting
    box contains every feasible integer assignment when both endpoints exist.
    """

    symbol = sp.Symbol(variable, real=True)
    other_symbols = {sp.Symbol(name, real=True) for name in all_variables if name != variable}
    lower: int | None = None
    upper: int | None = None
    for constraint in constraints:
        if constraint.sense != "<=":
            continue
        expression = sp.expand(constraint.expression)
        coefficient = expression.coeff(symbol)
        remainder = sp.expand(expression - coefficient * symbol)
        if coefficient == 0:
            continue
        if remainder.free_symbols & other_symbols or not coefficient.is_Rational or not remainder.is_Rational:
            continue
        bound = -sp.Rational(remainder) / sp.Rational(coefficient)
        if coefficient > 0:
            candidate = int(sp.floor(bound))
            upper = candidate if upper is None else min(upper, candidate)
        elif coefficient < 0:
            candidate = int(sp.ceiling(bound))
            lower = candidate if lower is None else max(lower, candidate)
    return lower, upper


def _constraint_holds(constraint: AffineConstraint, substitutions: Mapping[sp.Symbol, int]) -> bool:
    """Test a row exactly at a complete assignment during finite enumeration."""

    value = sp.simplify(constraint.expression.subs(substitutions))
    return bool(value == 0) if constraint.sense == "=" else bool(value <= 0)


__all__ = [
    "AlgebraicProof",
    "FarkasCertificate",
    "Symbol",
    "SymbolicModel",
    "UnsupportedAlgebra",
    "lower_linear_problem",
    "lower_symbolic_model",
    "prove_algebraic_equivalence",
]

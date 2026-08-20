"""Certified algebraic proof backend for abstract PyOPL model comparison.

The backend intentionally targets a sound, useful fragment rather than
claiming completeness: scalar affine LP/MILP schemas with exact rational
constants and symbolic scalar parameters.  Unsupported constructs are reported
to the caller so it can return ``unknown``.
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
    pass


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: Literal["parameter", "variable"]
    value_type: str


@dataclass(frozen=True)
class AffineConstraint:
    expression: sp.Expr
    sense: Literal["<=", "="]


@dataclass(frozen=True)
class SymbolicModel:
    parameters: tuple[Symbol, ...]
    variables: tuple[Symbol, ...]
    constraints: tuple[AffineConstraint, ...]
    objective: sp.Expr
    objective_sense: Literal["minimize", "maximize"]
    assumptions: tuple[sp.Expr, ...] = ()


@dataclass(frozen=True)
class AlgebraicProof:
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


@dataclass(frozen=True)
class FarkasCertificate:
    inequality_multipliers: tuple[Fraction, ...]
    equality_multipliers: tuple[Fraction, ...]


def _register_declaration(
    declaration: Mapping[str, Any],
    parameter_declarations: list[Symbol],
    variable_declarations: list[Symbol],
    symbols: dict[str, sp.Symbol],
    inferred_assumptions: list[sp.Expr],
) -> None:
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
        if value_type in {"int+", "float+", "boolean"}:
            inferred_assumptions.append(symbols[name] >= 0)
        if value_type == "boolean":
            inferred_assumptions.append(symbols[name] <= 1)
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
    """Lower a parser AST to the typed scalar affine proof IR."""

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
    comparison. Numeric values are converted through their decimal string so
    the resulting SymPy expressions use exact rationals rather than binary
    floating-point approximations.
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
        objective_sense=problem.sense,
    )


def _rational(value: object) -> sp.Rational:
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
    """Run normalization, certified rewrites, projection, and finite MILP proofs."""

    mappings = _candidate_mappings(left, right, parameter_mapping, variable_mapping, left_auxiliaries, right_auxiliaries)
    if not mappings:
        return AlgebraicProof("different", "symbolically_normalized", "no compatible parameter and variable mapping")

    unknown_reasons: list[str] = []
    for parameter_map, variable_map in mappings:
        renamed_right = _rename_model(right, parameter_map, variable_map)
        left_kept = {variable.name for variable in left.variables if variable.name not in left_auxiliaries}
        right_kept = {left_name for left_name, right_name in variable_map.items() if right_name not in right_auxiliaries}
        if left_kept != right_kept:
            continue
        try:
            proof = _prove_mapped_models(
                left,
                renamed_right,
                left_kept,
                set(left_auxiliaries),
                {left_name for left_name, right_name in variable_map.items() if right_name in right_auxiliaries},
                max_rewrite_iterations,
            )
        except UnsupportedAlgebra as exc:
            unknown_reasons.append(str(exc))
            continue
        if proof.status == "equivalent":
            return proof
        if proof.status == "unknown":
            unknown_reasons.append(proof.reason)

    if unknown_reasons:
        return AlgebraicProof("unknown", "rewrite_certified", unknown_reasons[0])
    return AlgebraicProof(
        "different",
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
            "different",
            "polyhedrally_proven",
            "projected affine feasible sets or objectives differ",
            tuple(dict.fromkeys(implication_steps)),
        )
    objective_equal = _objectives_equal_on_polyhedron(left_projected, right_projected, sorted(kept_variables))
    if not objective_equal:
        return AlgebraicProof(
            "different",
            "polyhedrally_proven",
            "objectives differ on the common projected feasible set",
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
    variable_symbols = [sp.Symbol(variable.name, real=True) for variable in model.variables]
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
    parameter_maps = _typed_bijections(left.parameters, right.parameters, parameter_mapping)
    left_primary = tuple(symbol for symbol in left.variables if symbol.name not in left_auxiliaries)
    right_primary = tuple(symbol for symbol in right.variables if symbol.name not in right_auxiliaries)
    variable_maps = _typed_bijections(left_primary, right_primary, variable_mapping)
    results: list[tuple[dict[str, str], dict[str, str]]] = []
    for parameter_map in parameter_maps:
        for variable_map in variable_maps:
            full_variable_map = dict(variable_map)
            for auxiliary in right_auxiliaries:
                full_variable_map[f"__right_aux_{auxiliary}"] = auxiliary
            results.append((parameter_map, full_variable_map))
            if len(results) >= 256:
                return results
    return results


def _typed_bijections(
    left: Sequence[Symbol],
    right: Sequence[Symbol],
    required: Mapping[str, str] | None,
) -> list[dict[str, str]]:
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
        if len(candidates) >= 256:
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
    objective = sp.expand(-model.objective if model.objective_sense == "maximize" else model.objective)
    constraints = tuple(sorted({_normalize_constraint(constraint) for constraint in model.constraints}, key=_constraint_key))
    return replace(model, constraints=constraints, objective=objective, objective_sense="minimize")


def _normalize_constraint(constraint: AffineConstraint) -> AffineConstraint:
    expression = sp.expand(constraint.expression)
    coefficient, _ = expression.as_coeff_Mul()
    if coefficient.is_Rational and coefficient != 0:
        if constraint.sense == "=" or coefficient > 0:
            expression = sp.expand(expression / abs(coefficient))
    if constraint.sense == "=":
        ordered = sorted(expression.as_ordered_terms(), key=sp.default_sort_key)
        if ordered and ordered[0].could_extract_minus_sign():
            expression = -expression
    return AffineConstraint(sp.factor_terms(expression), constraint.sense)


def _constraint_key(constraint: AffineConstraint) -> tuple[str, str]:
    return constraint.sense, sp.srepr(constraint.expression)


def _symbolic_models_equal(left: SymbolicModel, right: SymbolicModel) -> bool:
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
    for auxiliary in sorted(auxiliaries):
        symbol = sp.Symbol(auxiliary, real=True)
        defining = [
            constraint
            for constraint in model.constraints
            if constraint.sense == "=" and symbol in constraint.expression.free_symbols
        ]
        if len(defining) != 1:
            continue
        coefficient = sp.expand(defining[0].expression).coeff(symbol)
        if coefficient == 0 or symbol in coefficient.free_symbols:
            continue
        if coefficient.free_symbols and not _assumptions_prove_nonzero(coefficient, model.assumptions):
            continue
        replacement = sp.solve(defining[0].expression, symbol, dict=False)
        if len(replacement) != 1:
            continue
        substituted_constraints = tuple(
            AffineConstraint(sp.expand(constraint.expression.subs(symbol, replacement[0])), constraint.sense)
            for constraint in model.constraints
            if constraint is not defining[0]
        )
        return (
            replace(
                model,
                variables=tuple(variable for variable in model.variables if variable.name != auxiliary),
                constraints=substituted_constraints,
                objective=sp.expand(model.objective.subs(symbol, replacement[0])),
            ),
            auxiliary,
        )
    return None


def _assumptions_prove_nonzero(expression: sp.Expr, assumptions: Sequence[sp.Expr]) -> bool:
    if expression.is_number:
        return expression != 0
    if expression.is_nonzero is True:
        return True
    for assumption in assumptions:
        if assumption == sp.Ne(expression, 0) or assumption == (expression > 0) or assumption == (expression < 0):
            return True
        if assumption.func in {sp.GreaterThan, sp.LessThan} and assumption.lhs == expression and assumption.rhs == 0:
            return True
    return False


def _project_continuous(model: SymbolicModel, auxiliaries: set[str]) -> SymbolicModel:
    if any(constraint.sense == "=" for constraint in model.constraints):
        raise UnsupportedAlgebra("remaining equalities could not be eliminated before projection")
    constraints = list(model.constraints)
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
    if conclusion_constant <= 0 and all(value == 0 for value in conclusion_coefficients):
        return np.asarray([])
    return None


def _farkas_lp_matrices(
    premise_vectors: list[tuple[list[sp.Expr], sp.Expr]],
    equality_vectors: list[tuple[list[sp.Expr], sp.Expr]],
    conclusion_coefficients: list[sp.Expr],
    variables: list[str],
) -> tuple[list[list[float]], list[float], list[list[float]]]:
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
) -> FarkasCertificate | None:
    if conclusion.sense == "=":
        positive = _farkas_certificate(premises, AffineConstraint(conclusion.expression, "<="), variables)
        negative = _farkas_certificate(premises, AffineConstraint(-conclusion.expression, "<="), variables)
        return positive if positive is not None and negative is not None else None
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
    simplified = sp.simplify(value)
    if not simplified.is_Rational:
        raise UnsupportedAlgebra("polyhedral proof requires rational coefficients")
    return Fraction(int(simplified.p), int(simplified.q))


def _objectives_equal_on_polyhedron(left: SymbolicModel, right: SymbolicModel, variables: list[str]) -> bool:
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
    witness = next(iter(left_points.symmetric_difference(right_points)))
    return AlgebraicProof(
        "different",
        "presburger_proven",
        "bounded integer projections differ",
        tuple(dict.fromkeys(steps + ("exhaustively eliminated bounded integer auxiliaries",))),
        counterexample=f"projected assignment/objective differs: {witness}",
    )


def _enumerate_integer_projection(
    model: SymbolicModel,
    kept_variables: set[str],
) -> set[tuple[tuple[tuple[str, int], ...], Fraction]] | None:
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
    bounds: dict[str, tuple[int, int]] = {}
    assignment_count = 1
    for variable in variables:
        lower, upper = _constant_integer_bounds(constraints, variable, variables)
        if lower is None or upper is None:
            return None
        bounds[variable] = (lower, upper)
        assignment_count *= upper - lower + 1
        if assignment_count > 100_000:
            return None
    return bounds


def _record_integer_projection_point(
    model: SymbolicModel,
    assignment: dict[str, int],
    substitutions: Mapping[sp.Symbol, int],
    kept_variables: set[str],
    points: dict[tuple[tuple[str, int], ...], Fraction],
) -> None:
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

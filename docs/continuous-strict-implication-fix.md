# Continuous Strict Implication Fix

## Problem

PyOPL previously reified continuous strict comparisons with a fixed separation
of `1e-4`. For an implication such as:

```opl
(x > 0) => (y == 1);
```

the generated model partitioned the antecedent into `x >= 1e-4` and `x <= 0`.
This incorrectly excluded every value in `0 < x < 1e-4`, even when `y == 1`.
SciPy/HiGHS and Gurobi agreed because both used the same approximation.

CPLEX Studio 22.1.1 rejects the source continuous strict comparison. The
DOcplex cross-checks establish the intended semantics through the equivalent
bounded formulations `Q[t] <= 410 * order[t]` and `x[t] <= 150 * y[t]`.

## Implemented Lowering

PyOPL now recognizes strict affine antecedents whose consequent fixes a binary
decision variable:

```opl
(affine > rhs) => (binary == value);
(affine < rhs) => (binary == value);
```

Gurobi receives the exact contrapositive as a native indicator constraint. For
example, `(x > 0) => (y == 1)` becomes `y == 0 => x <= 0`.

SciPy/HiGHS receives an exact bounded row. If `d = affine - rhs` has finite
upper bound `U`, `(d > 0) => (y == 1)` becomes `d <= U * y`.

The lowering supports binary values `0` and `1`, either equality operand order,
and both strict inequality orientations. It does not use the strict-comparison
epsilon.

## Unsupported Cases

Arbitrary truth reification of a continuous comparison is not generally
representable as a closed MILP feasible set. PyOPL now raises `SemanticError`
for those cases instead of silently introducing an epsilon dead zone. Integer
and Boolean comparison reification remains exact and supported.

## Regression Evidence

Focused tests verify that values below the former epsilon threshold follow the
mathematical truth table, that Gurobi emits no implication truth auxiliary for
the specialized pattern, that SciPy uses the inferred finite upper bound, and
that unsupported continuous truth reification is rejected explicitly.

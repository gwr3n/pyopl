# pyopl - Python Optimisation Programming Language

Core package badges:

![Codecov (with branch)](https://img.shields.io/codecov/c/gh/gwr3n/pyopl/main)
 ![Python package](https://img.shields.io/github/actions/workflow/status/gwr3n/pyopl/.github%2Fworkflows%2Fpython-package.yml) ![Lint and type-check](https://img.shields.io/github/actions/workflow/status/gwr3n/pyopl/.github%2Fworkflows%2Flint-type.yml?branch=main&label=lint%20%2B%20type-check) [![PyPI](https://img.shields.io/pypi/v/pyopl)](https://pypi.org/project/pyopl/) [![Python versions](https://img.shields.io/pypi/pyversions/pyopl)](https://pypi.org/project/pyopl/) [![License](https://img.shields.io/github/license/gwr3n/pyopl)](LICENSE) [![Downloads](https://static.pepy.tech/badge/pyopl)](https://pepy.tech/project/pyopl) [![Release](https://img.shields.io/github/v/release/gwr3n/pyopl)](https://github.com/gwr3n/pyopl/releases) [![Wheel](https://img.shields.io/pypi/wheel/pyopl)](https://pypi.org/project/pyopl/)

Quality and tooling:

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?logo=python)](https://github.com/psf/black) [![Ruff](https://img.shields.io/badge/lint-ruff-1f79ff?logo=python)](https://github.com/astral-sh/ruff) [![mypy](https://img.shields.io/badge/type--checked-mypy-blue?logo=python)](https://github.com/python/mypy)

Project/community:

[![Issues](https://img.shields.io/github/issues/gwr3n/pyopl)](https://github.com/gwr3n/pyopl/issues) [![PRs](https://img.shields.io/github/issues-pr/gwr3n/pyopl)](https://github.com/gwr3n/pyopl/pulls) [![Stars](https://img.shields.io/github/stars/gwr3n/pyopl?style=social)](https://github.com/gwr3n/pyopl/stargazers)

Docs:

[![Docs](https://img.shields.io/badge/docs-site-blue)](https://github.com/gwr3n/pyopl)

`pyopl` is a Python library for parsing and solving OPL-like [1] mathematical programming models using either Gurobi or the open-source SciPy (HiGHS) solver. PyOPL supports a rich subset of Optimisation Programming Language (OPL) syntax for linear and mixed-integer programming.

[1] Van Hentenryck, P. (1999). The OPL optimization programming language. London, England: MIT Press.

## Installation

Install via pip (recommended):

```
pip install rhetor
```

Or clone the repository and install locally:

```
git clone https://github.com/gwr3n/pyopl.git
cd pyopl
pip install .
```

Dependencies are managed via `pyproject.toml` and are listed in [`requirements.txt`](./requirements.txt)

PyOPL requires Python 3.10+

You can use either Gurobi or SciPy/HiGHS as the solver. Both solvers are selectable in the API and the IDE. PyOPL provides robust support for tuple/nested tuple data, advanced boolean logic, implication, and field access in both models and data files.

## Usage

### Solving an OPL Model

You can use the `solve` function to load and solve an OPL model (and optional data file). The function parses the model, performs semantic validation, generates backend-specific code (Gurobi Python or SciPy/HiGHS matrices), applies logical encodings (including implication and `!=` big-M or indicator formulations), and executes it. Choose the solver with the `solver` argument:

```python
from pyopl import solve
results = solve('model.mod', 'data.dat', solver='gurobi')  # Use Gurobi (default)
results = solve('model.mod', 'data.dat', solver='scipy')   # Use SciPy/HiGHS (LP or MIP, if supported)
```

#### Example

Suppose you have the following files:

- `knapsack.mod` (your OPL model)
- `knapsack.dat` (your data file)

You can solve the model as follows:


```python
from pyopl import solve

model = "knapsack.mod"
data = "knapsack.dat"

results = solve(model, data)
print(results)
```


This will print the parsed AST, the generated solver code, and the solution output from the selected solver. The `solve` function returns a dictionary with the following keys:
- `status`: The status of the optimization (e.g., 'OPTIMAL', 'INFEASIBLE', etc.)
- `solution`: A dictionary of variable names and their values (if optimal)
- `objective_value`: The value of the objective function (if optimal)
- `stats`: Additional statistics (e.g., MIPGap, Runtime, NodeCount, IterCount)
- `message`: Any error or status messages

Below, we provide model and data file for our knapsack example.

Example contents for `knapsack.mod`:
```opl
// knapsack.mod
int+ n = ...; // non-negative integer parameter
float+ c = ...; // non-negative float parameter
float+ w[1..n] = ...; // indexed parameter
float+ v[1..n] = ...;

dvar boolean x[1..n];

maximize sum(i in 1..n) v[i] * x[i];
subject to {
    sum(i in 1..n) w[i] * x[i] <= c;
    forall(i in 1..n) x[i] >= 0;
}
```

Example contents for `knapsack.dat`:
```opl
// knapsack.dat
n = 4;
c = 10;
w = [2, 3, 4, 5];
v = [3, 4, 5, 6];
```

See `examples.py` for a repository of examples. 


### Function Reference

#### `solve(model_file, data_file=None, solver='gurobi')`

- `model_file`: Path to your `.mod` or `.opl` OPL model file
- `data_file`: (Optional) Path to a `.dat` data file
- `solver`: `'gurobi'` (default) or `'scipy'`

The function returns a dictionary with solver results and prints:
- Parsed AST (Abstract Syntax Tree)
- Loaded data dictionary (if any)
- Generated solver code
- Output from the selected solver

**Return value:**
- Dictionary with keys:
  - `status`: Optimization status (e.g., 'OPTIMAL', 'INFEASIBLE')
  - `solution`: Variable values (if optimal)
  - `objective_value`: Objective value (if optimal)
  - `stats`: Solver statistics (MIPGap, Runtime, etc.)
  - `message`: Error or status messages

### Notes

- You must have a valid Gurobi license to solve models with Gurobi.
- SciPy/HiGHS is open-source.
- The library is designed for educational and prototyping purposes and supports a rich subset of OPL syntax, including advanced tuple, boolean, and logical constructs.



## PyOPL IDE

PyOPL includes the Rhetor graphical IDE for editing, running, and debugging OPL models and data files. The IDE features:

- Syntax highlighting for OPL models and data files
- Side-by-side model and data editors
- Output panel for solver results and messages
- File tree for easy switching between model and data
- Solver selection (Gurobi or SciPy/HiGHS) — choose your preferred solver from the menu
- Font size adjustment and modern UI

### Launching the IDE

You can launch the IDE from the command line or if installed as a package:

```sh
python -m pyopl
```

This will open the PyOPL IDE window. You can open `.mod` (model) and `.dat` (data) files, edit them, and run your model directly from the interface. You can select either Gurobi or SciPy/HiGHS as the solver from the IDE's menu bar.


## OPL Constructs Supported by PyOPL

For a complete language reference and advanced examples, see the [PyOPL user guide](./docs/PyOPL%20user%20guide.md).

PyOPL supports a rich subset of OPL constructs for linear and mixed-integer programming, including logical and boolean extensions:

- **Variable Types:**
  - `int`, `int+` (non-negative integer), `float`, `float+` (non-negative float), `boolean`
- **Decision Variables:**
  - Scalar, indexed, and tuple-indexed variables (including multi-dimensional and nested tuple indices):
    ```opl
    dvar float x;
    dvar int+ y[1..N][1..M];
    dvar boolean z[i in Items, j in Cities];
    dvar float x[arcs]; // tuple-indexed (set of tuples)
    ```
- **Ranges:**
  - Inline ranges with general integer expressions: `range T = 1..(N+M-1);`
  - Named ranges used for indexing must be declared in the model with explicit bounds. .dat-supplied ranges are not accepted for indexing.
- **Sets:**
  - Typed scalar sets declared in the model, values provided in `.dat`:
    ```opl
    {string} Cities = { "A", "B" };
    {int}    Periods = { 1, 2, 3 };
    {float}  Weights = { 1.5, 2.0 };
    {boolean} Flags = { true, false };
    ```
  - Sets of tuples (including nested), declared in the model; values can be provided inline or in `.dat`:
    ```opl
    tuple Arc { string u; string v; float cost; }
    {Arc} arcs = { <"A","B",10.0>, <"B","C",12.5> };
    ```
- **Tuple Types and Sets of Tuples:**
  - Define tuple types and use them as indices; field access supported (e.g., `a.cost`):
    ```opl
    tuple Inner { int i; int j; }
    tuple Outer { Inner pair; float val; }
    {Outer} nested = { <<1,2>, 3.5>, <<2,3>, 4.0> };
    dvar float x[arcs];
    dvar float y[nested];
    ```
  - Tuple arrays indexed by a set (data records) are supported and accessible via field access in expressions.
- **Parameters:**
  - Scalar and indexed parameters with inline values or external values in `.dat`:
    ```opl
    param float C;
    float alpha = 5.0;
    param float d[i in Items, j in Cities];
    param float w[arcs]; // tuple-indexed parameter
    ```
  - Computed parameters from expressions and iterator headers are supported; they are evaluated at compile time into concrete arrays.
- **Indexing:**
  - Use named ranges/sets or expressions as indices (including tuple indices over sets of tuples).
- **Constraints:**
  - Linear comparisons: `<=`, `>=`, `==`, `<`, `>`
  - Not-equal: `!=`
    - For boolean vars: XOR linearization
    - For numeric: disjunctive big-M with automatic tightening where possible
  - Implication: `(antecedent) => (consequent)`
    - Gurobi: uses indicator constraints when possible, or tightened big-M otherwise
    - SciPy: supports linear antecedent/consequent via big-M; boolean combinations are linearized to auxiliaries
  - Conditional (ternary) expressions: `(cond) ? thenExpr : elseExpr` (condition must be ground)
  - Boolean-valued constraints and boolean expression trees (and/or/not) are supported and linearized in constraints
  - Conditional constraints: `if (ground_condition) { ... } else { ... }` (compile-time rewrite)
- **forall:**
  - Multi-indexed, with optional index constraints, over ranges, sets, or sets of tuples:
    ```opl
    forall (i in Items, j in Items: i != j)
        x[i] + x[j] <= 1;
    forall (a in arcs)
        x[a] >= a.cost;
    ```
- **sum:**
  - Multi-indexed summation, with optional index constraints, over ranges, sets, or sets of tuples:
    ```opl
    minimize sum (i in Items, j in Items: i != j) (cost[i][j] * x[i][j]);
    minimize sum (a in arcs) (a.cost * x[a]);
    ```
  - Sum of comparisons (cardinality constraints) and reified forms (e.g., `b == (sum(...) >= k)`) are recognized and linearized.
- **Tuple Field Access:**
  - Use dot notation or positional access in expressions, e.g., `a.cost * x[a]`, `o.pair.i`.
- **Logical / Boolean Expressions:**
  - Gurobi: boolean expression trees with `and`, `or`, `not` over comparisons are linearized to auxiliary binaries and can appear inside implications.
  - SciPy: boolean trees over linear comparisons are supported via auxiliaries; combined with big-M for gating. Implication antecedents must be a single comparison or a boolean variable (no composite trees in implications).
- **Automatic Big-M Tightening:**
  - For `!=`, implication, and reification, PyOPL infers bounds from variable types, simple linear combinations, and finite sums to compute tighter M values; falls back to conservative constants if unknown.
- **Functions and Aggregates:**
  - Functions (ground-only): `sqrt(...)`
  - Aggregates: `maxl(arg1, arg2, ...)`, `minl(arg1, arg2, ...)`
  - Convex lowering:
    - Objective: supported in convex forms only (minimize maxl(...), maximize minl(...)); introduces auxiliary epigraph/hypograph constraints
    - Constraints: supported in monotone convex forms (e.g., `maxl(...) <= rhs`, `lhs >= maxl(...)`, `minl(...) >= rhs`, `lhs <= minl(...)`)
- **Comments:**
  - Single-line: `// comment` or `# comment`
  - Multi-line: `/* ... */`
- **Data Files:**
  - Support numbers, lists, nested lists, sets, ranges, and nested tuple literals:
    ```opl
    n = 4;
    w = [2, 3, 4, 5];
    Items = 1..5;
    arcs = { <"A","B",10.0>, <"B","C",12.5> };
    nested = { <<1,2>, 3.5>, <<2,3>, 4.0> };
    ```
  - 2D parameters accept row-major lists or keyed-row dict-of-lists for common shapes (set×range, set×set), e.g.:
    ```
    Demand = [ "StoreA" [1,2,3], "StoreB" [4,5,6] ];
    ```
  - Typed prefixes for sets must be in the model; data files use untyped assignments (e.g., `Cities = { "A", "B" };`).

## License

MIT License.


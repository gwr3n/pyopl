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

`pyopl` is a Python library for parsing and solving OPL-like [1] mathematical programming models using either Gurobi or the open-source SciPy (HiGHS) solver. You can choose which solver to use. PyOPL supports a rich subset of Optimisation Programming Language (OPL) syntax for linear and mixed-integer programming.

[1] Van Hentenryck, P. (1999). The OPL optimization programming language. London, England: MIT Press.

## Installation


You need Python 3.7+ and the following packages:

- `sly` (for parsing OPL)
- `gurobipy` (for Gurobi solver, requires a Gurobi license)
- `scipy` and `highspy` (for open-source HiGHS solver via SciPy)
- `numpy`, `Pillow`

Install all dependencies with:

```sh
pip install sly gurobipy scipy numpy Pillow highspy
```



You can use either Gurobi or SciPy/HiGHS as the solver. Gurobi is required for mixed-integer models; SciPy/HiGHS supports linear programs and, in recent versions, can also handle integer and boolean variables (MIP) if the solver and SciPy version support it. Integrality is passed to `linprog` if present, but full MIP support depends on your SciPy installation. Both solvers are selectable in the API and the IDE. PyOPL provides robust support for tuple/nested tuple data, advanced boolean logic, implication, and field access in both models and data files.

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
dvar boolean x[1..n];
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
- SciPy/HiGHS is open-source and can be used for linear programs and, if supported by your SciPy version, mixed-integer programs (MIP). Integrality is passed to `linprog` if present, but full MIP support depends on your SciPy installation.
- The library is designed for educational and prototyping purposes and supports a rich subset of OPL syntax, including advanced tuple, boolean, and logical constructs.



## PyOPL IDE

PyOPL includes a graphical IDE for editing, running, and debugging OPL models and data files. The IDE features:

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
    dvar float x[arcs]; // tuple-indexed
    dvar float y[nested]; // nested tuple-indexed
    ```
- **Ranges:**
  - Inline ranges: `range T = 1..N;`
  - Ranges with general integer expressions: `range MyRange = 10..(N-1);`
  - External ranges (value from `.dat`): `range T;`
- **Sets:**
  - Declared in the model, values provided in `.dat`:
    ```opl
    set Cities;
    set MySet;
    {string} Gasolines = { "R92", "R95" };
    {Arc} arcs = { <"A", "B", 10.0>, <"B", "C", 12.5> };
    {Outer} nested = { <<1,2>, 3.5>, <<2,3>, 4.0> };
    ```
- **Tuple Types and Sets of Tuples:**
  - Define tuple types and sets of tuples for use as indices:
    ```opl
    tuple Arc { string start; string end; float cost; }
    tuple Inner { int i; int j; }
    tuple Outer { Inner pair; float value; }
    {Arc} arcs = { <"A", "B", 10.0>, <"B", "C", 12.5> };
    {Outer} nested = { <<1,2>, 3.5>, <<2,3>, 4.0> };
    dvar float x[arcs];
    dvar float y[nested];
    param float w[arcs] = [1.5, 2.5];
    ```
  - Tuple field access: `a.cost`, `a[2]`, including nested fields (`o.pair.i`).
  - Empty and singleton tuples: `< >`, `<1,>`
  - All tuple literal forms (including nested and empty) are supported in both model and data files.
- **Parameters:**
  - Scalar and indexed, with inline values, implicit external, or explicit external declaration:
    ```opl
    param float C;           // external, value from .dat
    param int num_items = ...; // explicit external
    float alpha = 5.0;       // inline value
    int+ n = ...;            // non-negative, external
    param float d[i in Items, j in Cities] = ...; // explicit external indexed
    param float w[arcs] = [1.5, 2.5]; // tuple-indexed
    ```
- **Indexing:**
  - Use named ranges/sets or integer expressions as indices:
    ```opl
    dvar float x[i in Items, j in Cities];
    dvar float y[1..N][1..M];
    dvar float x[arcs]; // tuple-indexed
    dvar float y[nested]; // nested tuple-indexed
    ```
- **Constraints:**
  - Comparison: `<=`, `>=`, `==`, `<`, `>`
  - Not-equal: `!=` (boolean XOR or numeric disjunctive big-M with automatic span-based tightening)
  - Implication: `(antecedent) => (consequent)`; Gurobi uses indicator constraints when possible, otherwise a tightened big-M; SciPy uses tightened big-M for linear comparison antecedent/consequent pairs.
  - Conditional (ternary) expressions: `(cond) ? thenExpr : elseExpr` (ground conditions only)
  - Boolean-valued objectives and constraints (interpreted as 1/0)
  - Boolean expression trees (and/or/not) in constraints (Gurobi: full support, SciPy: limited)
- **forall:**
  - Multi-indexed, with optional index constraints, over ranges, sets, or sets of tuples:
    ```opl
    forall (i in Items, j in Items: i != j)
        x[i] + x[j] <= 1;
    forall (a in arcs)
        x[a] >= w[a];
    forall (o in nested)
        y[o] >= o.value;
    ```
- **sum:**
  - Multi-indexed summation, with optional index constraints, over ranges, sets, or sets of tuples:
    ```opl
    minimize sum (i in Items, j in Items: i != j) (cost[i][j] * x[i][j]);
    minimize sum (a in arcs) (a.cost * x[a]);
    minimize sum (o in nested) (o.value * y[o]);
    ```
  - Tuple field access is supported in the sum/forall body, objectives, and constraints.
- **Tuple Field Access:**
  - Use dot notation or integer index to access tuple fields in expressions, e.g., `a.cost * x[a]`, `o.pair.i`.
- **Logical / Boolean Expressions (Gurobi advanced):**
  - Boolean expression trees with `and`, `or`, `not` over comparison atoms are linearized to auxiliary binaries and can appear inside implications.
  - SciPy currently restricts implication antecedents to a single comparison (or `b == 1`).
- **Automatic Big-M Tightening:**
  - For `!=` and general implication encodings, PyOPL infers bounds from variable types (boolean in [0,1], non-negative domains, simple linear combinations, finite sums) to compute the smallest safe M (span of possible difference). Falls back to a large constant only if bounds are unknown.
- **Comments:**
  - Single-line: `// comment` or `# comment`
  - Multi-line: `/* ... */`
- **Data Files:**
  - Support for numbers, lists, nested lists, sets, ranges, and nested tuple literals in `.dat` files:
    ```opl
    n = 4;
    w = [2, 3, 4, 5];
    my_set = {1, 2, 3};
    Items = 1..5;
    arcs = { <"A", "B", 10.0>, <"B", "C", 12.5> };
    nested = { <<1,2>, 3.5>, <<2,3>, 4.0> };
    singletons = { <1,>, <2,> };
    empties = { < > };
    ```

## License

MIT License.

## Limitations

- PyOPL robustly supports tuple types, sets of tuples (including nested), tuple field access, `sum` / `forall` over tuple sets, `!=`, and implication constraints.
- Composite boolean implication antecedents are currently only available in the Gurobi backend.
- Features not yet supported: piecewise linear expressions, SOS constraints, user-defined functions, nonlinear arithmetic, bi-implication `<=>`, global constraints (alldiff, etc.).
- SciPy backend implication support is limited to linear comparisons (no multi-operator boolean trees yet).
- Automatic big-M tightening applies where bounds are inferable; otherwise a conservative fallback is used.
- Gurobi must be installed and licensed for `solver='gurobi'`.
- SciPy/HiGHS is open-source and can be used for linear programs and, if supported by your SciPy version, mixed-integer programs (MIP, i.e., integer and boolean variables). Integrality is passed to `linprog` if present, but full MIP support depends on your SciPy installation.
- The library is designed for educational and prototyping purposes and supports a subset of OPL syntax.


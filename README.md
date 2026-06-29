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

## User Guide

A comprehensive [User Guide](docs/PyOPL%20user%20guide.md) is available in the `docs` folder of the repository.

## License

MIT License.


# pyopl - Python Optimisation Programming Language

Core package badges:

[![Codecov](https://img.shields.io/codecov/c/gh/gwr3n/pyopl/main)](https://codecov.io/gh/gwr3n/pyopl)
[![Python package](https://img.shields.io/github/actions/workflow/status/gwr3n/pyopl/python-package.yml?branch=main&label=python%20package)](https://github.com/gwr3n/pyopl/actions/workflows/python-package.yml) 
[![Lint and type-check](https://img.shields.io/github/actions/workflow/status/gwr3n/pyopl/lint-type.yml?branch=main&label=lint%20%2B%20type-check)](https://github.com/gwr3n/pyopl/actions/workflows/lint-type.yml) 
[![Rhetor on PyPI](https://img.shields.io/pypi/v/rhetor)](https://pypi.org/project/rhetor/)
[![Python versions](https://img.shields.io/pypi/pyversions/rhetor)](https://pypi.org/project/rhetor/)
[![Downloads](https://img.shields.io/pypi/dm/rhetor)](https://pypistats.org/packages/rhetor) 
[![Release](https://img.shields.io/github/v/release/gwr3n/pyopl)](https://github.com/gwr3n/pyopl/releases) 
[![Wheel](https://img.shields.io/pypi/wheel/rhetor)](https://pypi.org/project/rhetor/)

Quality and tooling:

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?logo=python)](https://github.com/psf/black) [![Ruff](https://img.shields.io/badge/lint-ruff-1f79ff?logo=python)](https://github.com/astral-sh/ruff) [![mypy](https://img.shields.io/badge/type--checked-mypy-blue?logo=python)](https://github.com/python/mypy) [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) [![security: pip-audit](https://img.shields.io/badge/security-pip--audit-008080.svg)](https://github.com/pypa/pip-audit) [![secrets: Betterleaks](https://img.shields.io/badge/secrets-Betterleaks-24292e.svg?logo=github)](https://github.com/betterleaks/betterleaks)

Project/community:

[![Issues](https://img.shields.io/github/issues/gwr3n/pyopl)](https://github.com/gwr3n/pyopl/issues) [![PRs](https://img.shields.io/github/issues-pr/gwr3n/pyopl)](https://github.com/gwr3n/pyopl/pulls) [![Stars](https://img.shields.io/github/stars/gwr3n/pyopl?style=social)](https://github.com/gwr3n/pyopl/stargazers)

Docs:

[![Docs](https://img.shields.io/badge/docs-user%20guide-blue)](docs/PyOPL%20user%20guide.md)

License: 

![GitHub License](https://img.shields.io/github/license/gwr3n/pyopl)


`pyopl` is a Python library for parsing and solving OPL-like [1] mathematical programming models using either Gurobi or the open-source SciPy (HiGHS) solver. PyOPL supports a rich subset of Optimisation Programming Language (OPL) syntax for linear and mixed-integer programming.

[1] Van Hentenryck, P. (1999). The OPL optimization programming language. London, England: MIT Press.

## Installation

The GitHub project and importable compiler package are named `pyopl`; the published PyPI distribution is named `rhetor`.

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

You can use either Gurobi or SciPy/HiGHS as the solver. Both solvers are selectable in the API and the Rhetor IME. PyOPL provides robust support for tuple/nested tuple data, advanced boolean logic, implication, and field access in both models and data files.

## Usage

### Solving an OPL Model

You can use the `solve` function to load and solve an OPL model (and optional data file). Choose the solver with the `solver` argument:

```python
from pyopl import solve
results = solve('model.mod', 'data.dat', solver='gurobi')  # Use Gurobi (default)
results = solve('model.mod', 'data.dat', solver='scipy')   # Use SciPy/HiGHS
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

See `tools/examples.py` for a repository of examples. 


## PyOPL IME

PyOPL includes [Rhetor](https://gwr3n.github.io/rhetor), a GenAI-first integrated modelling environment (IME) for creating, revising, solving, exporting, and versioning OPL models and data files.

![Demo](https://github.com/gwr3n/rhetor/raw/main/sudoku.gif)

The IME features:

- GenAI-first modelling workflows for generating models, revising existing model/data pairs, asking questions about a formulation, and explaining solutions with OpenAI, Google/Gemini, or Ollama models when configured
- Optional visual prompt attachments for supported GenAI workflows, including images and short PDFs
- Session-based model version tracking: each run/request can keep a timestamped snapshot that can be previewed, diffed against the current editors, restored, renamed, or deleted
- Output and session panels for reviewing recent runs, generated artifacts, model/data snapshots, and GenAI interactions
- Syntax-highlighted model and data editors with open, save, save-as, undo, redo, find, and replace
- Integrated solve workflow with Gurobi or SciPy/HiGHS selection, solver logs, elapsed-time status, and optional solver-progress display
- Export support for compiled Python, LP, and MPS artifacts where supported by the selected backend
- Light/dark themes and configurable editor font sizes, with settings saved between sessions

### Launching the IME

Running PyOPL with no subcommand launches the IME:

```sh
python -m pyopl
```

If installed as a package, the `pyopl` console command can be used in place of `python -m pyopl`.

This opens the Rhetor IME window. You can open `.mod` model files and optional `.dat` data files, generate or revise formulations with GenAI assistance, edit them directly, solve them from the interface, export generated artifacts, and choose either Gurobi or SciPy/HiGHS from the Solve menu.

### Sessions and Model Versions

Rhetor keeps an IME session history in a `.pyopl_session` file in the current working directory. Session entries preserve output history and associated model/data snapshots, so you can track how a model changes over an interactive, GenAI-assisted modelling session. From the session list, use the context menu to preview a saved snapshot, diff it against the current editors, restore it into the editors, rename the session entry, or delete it.



## PyOPL CLI

PyOPL also includes a command-line interface for solving models, exporting generated artifacts, and using GenAI helpers from scripts or terminals.

Basic usage:

```sh
python -m pyopl solve model.mod data.dat --solver highs --out json
```

The CLI supports:

- `python -m pyopl solve <model.mod> [data.dat]` to compile and solve a model
- `--solver highs|gurobi` to choose the backend solver
- `--out json|py|lp|mps` to print results, export generated Python, or write LP/MPS solver files
- `--out-file <path>` to write output to a file

Examples:

```sh
python -m pyopl solve knapsack.mod knapsack.dat --solver highs --out json
python -m pyopl solve knapsack.mod knapsack.dat --solver highs --out lp --out-file knapsack.lp
python -m pyopl solve knapsack.mod knapsack.dat --solver highs --out mps --out-file knapsack.mps
```



## PyOPL MCP

PyOPL MCP exposes core PyOPL compiler/solver functionality as MCP tools so external clients (IDEs, editors, automation) can call compile, solve, export, and compare operations over stdio-based MCP servers.

- **Purpose**: Provide programmatic access to compilation, solving, Python code export, and model equivalence comparison for OPL models using in-memory model/data strings.
- **Exposed tools**:
  - **read_pyopl_grammar_tool**: Return the bundled grammar text.
  - **solve_strings_tool**: Compile and solve a model from `model_text` and optional `data_text`.
  - **export_py_strings_tool**: Compile `model_text` and optional `data_text` to generated Python source and return it as a string.
  - **compare_model_strings_tool**: Compare two OPL models from strings using the same MILP equivalence engine as the IDE's Compare models workflow. Returns `status`, `equivalent`, `level`, `reason`, `proof_steps`, and `counterexample`.
- **Not exposed as MCP tools**: File-path helpers such as `solve_files_tool` and `export_py_files_tool` are retained in the Python module for trusted local/internal use, but are not registered as MCP tools by default for security reasons.
- **Solver mapping**: Default solver alias `highs` → SciPy/HiGHS; `gurobi` → Gurobi. See the tool `solver` parameter for selection.
- **Quick start (VS Code MCP example - .vscode/mcp.json)**:
```json
{
  "servers": {
    "PyOPL MCP": {
      "type": "stdio",
      "command": "${workspaceFolder}/venv/bin/python",
      "args": ["-m", "pyopl.pyopl_mcp"]
    }
  }
}
```
- **Run locally**:
```bash
python -m pyopl.pyopl_mcp
```
- **Files**: Implementation and tool list in pyopl_mcp.py.
- **Notes**: Errors from the compiler/solver/comparison engine are propagated to the caller; generated Python code is returned (not written) so the client can persist it as desired.



## User Guide

A comprehensive [User Guide](docs/PyOPL%20user%20guide.md) is available in the `docs` folder of the repository.
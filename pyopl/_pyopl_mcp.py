"""FastMCP server exposing PyOPL solver functionality as MCP tools.

This module can be run as a standalone server or integrated into an existing
MCP setup. It exposes tools for:

- solving OPL models from files or strings
- exporting compiled Python code from OPL inputs

Example VS Code MCP config (.vscode/mcp.json):

{
  "servers": {
    "PyOPL MCP": {
      "type": "stdio",
      "command": "${workspaceFolder}/venv/bin/python",
      "args": ["-m", "pyopl.pyopl_mcp"]
    }
  },
  "inputs": []
}
"""

from __future__ import annotations

import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Optional, TypeVar, Union

from mcp.server.fastmcp import FastMCP

from . import solve
from .milp_equivalence import EquivalenceResult, prove_equivalent
from .pyopl_core import OPLCompiler, linear_problem_from_opl

PathLike = Union[str, Path]
T = TypeVar("T")

DEFAULT_SOLVER = "highs"
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5"

METHODS: list[tuple[str, str]] = [
    ("SyntAGM", "pyopl_generative"),
    ("Standard", "pyopl_standard"),
    ("Chain of Thought", "pyopl_chain_of_thought"),
    ("Tree of Thoughts", "pyopl_tree_of_thoughts"),
    ("CAFA", "pyopl_cafa"),
    ("Chain of Experts", "pyopl_chain_of_experts"),
    ("Reflexion", "pyopl_reflexion"),
]

mcp = FastMCP("PyOPL MCP")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_solver(solver: Optional[str]) -> str:
    """Normalize solver names for compiler/backend compatibility."""
    if not solver:
        return "scipy"

    normalized = solver.strip().lower()

    solver_aliases = {
        "highs": "scipy",
        "scipy": "scipy",
        "gurobi": "gurobi",
    }
    return solver_aliases.get(normalized, normalized)


def _solve_backend(solver: Optional[str]) -> str:
    """Normalize solver names for solve() backend selection."""
    return "gurobi" if _normalize_solver(solver) == "gurobi" else "scipy"


def _compile_to_python(
    model_text: str,
    data_text: Optional[str] = None,
    solver: str = DEFAULT_SOLVER,
) -> str:
    """Compile OPL model/data text to Python code."""
    compiler = OPLCompiler()
    _, code_str, _ = compiler.compile_model(
        model_text,
        data_text,
        solver=_normalize_solver(solver),
    )
    return code_str


def solve_from_files(
    model_path: PathLike,
    data_path: Optional[PathLike] = None,
    solver: str = DEFAULT_SOLVER,
) -> dict:
    """Solve a model from filesystem paths."""
    model_p = Path(model_path)
    data_p = Path(data_path) if data_path else None
    return solve(
        str(model_p),
        str(data_p) if data_p else None,
        solver=_solve_backend(solver),
    )


def solve_from_strings(
    model_text: str,
    data_text: Optional[str] = None,
    solver: str = DEFAULT_SOLVER,
) -> dict:
    """Solve a model provided as strings."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "model.mod"
        model_file.write_text(model_text, encoding="utf-8")

        data_file: Optional[Path] = None
        if data_text is not None:
            data_file = Path(tmp_dir) / "data.dat"
            data_file.write_text(data_text, encoding="utf-8")

        return solve_from_files(model_file, data_file, solver=solver)


def export_py_from_files(
    model_path: PathLike,
    data_path: Optional[PathLike] = None,
    solver: str = DEFAULT_SOLVER,
) -> str:
    """Compile a model from files into Python code."""
    model_code = _read_text(Path(model_path))
    data_code = _read_text(Path(data_path)) if data_path else None
    return _compile_to_python(model_code, data_code, solver=solver)


def export_py_from_strings(
    model_text: str,
    data_text: Optional[str] = None,
    solver: str = DEFAULT_SOLVER,
) -> str:
    """Compile model/data text into Python code."""
    return _compile_to_python(model_text, data_text, solver=solver)


def _equivalence_result_to_dict(result: EquivalenceResult) -> dict:
    """Convert an equivalence result to an MCP-friendly JSON object."""
    return {
        "status": result.status,
        "equivalent": result.equivalent,
        "level": result.level,
        "reason": result.reason,
        "proof_steps": list(result.proof_steps),
        "counterexample": result.counterexample,
    }


def compare_model_strings(
    left_model_text: str,
    right_model_text: str,
    left_data_text: Optional[str] = None,
    right_data_text: Optional[str] = None,
) -> dict:
    """Compare two OPL models provided as strings for MILP equivalence."""
    left_problem = linear_problem_from_opl(left_model_text, left_data_text)
    right_problem = linear_problem_from_opl(right_model_text, right_data_text)
    return _equivalence_result_to_dict(prove_equivalent(left_problem, right_problem))


def read_pyopl_grammar() -> str:
    """Return the PyOPL grammar file contents as a UTF-8 string."""
    return (files(__package__) / "grammars" / "PyOPL grammar.md").read_text(encoding="utf-8")


@mcp.tool()
def read_pyopl_grammar_tool() -> str:
    """MCP tool returning the PyOPL grammar as a string."""
    return read_pyopl_grammar()


# @mcp.tool()
def solve_files_tool(
    model_path: str,
    data_path: Optional[str] = None,
    solver: str = DEFAULT_SOLVER,
) -> dict:
    """Solve an optimization model from `.mod` and optional `.dat` files.

    Args:
        model_path: Path to the OPL model file.
        data_path: Optional path to the OPL data file.
        solver: Solver name or alias. Supported values currently map to SciPy/HiGHS
            or Gurobi depending on configuration.

    Returns:
        A dictionary containing solver outputs (status, objective value, variable
        assignments, and any solver metadata) as produced by :func:`solve`.
    """
    return solve_from_files(model_path, data_path, solver)


# @mcp.tool()
def export_py_files_tool(
    model_path: str,
    data_path: Optional[str] = None,
    solver: str = DEFAULT_SOLVER,
) -> str:
    """Compile an OPL model (and optional data) to Python source.

    This tool reads the given `.mod` and optional `.dat` file paths,
    compiles them to Python using the internal compiler, and returns the
    generated Python code as a string. It does not write the generated
    code to disk (the caller may do so).

    Args:
        model_path: Filesystem path to an OPL model file (.mod).
        data_path: Optional path to an OPL data file (.dat).
        solver: Solver name or alias influencing backend codegen.

    Returns:
        A string containing the compiled Python source.

    Raises:
        Exceptions from the compiler (syntax/semantic errors) or I/O
        errors reading the input files are propagated.
    """
    return export_py_from_files(model_path, data_path, solver)


@mcp.tool()
def solve_strings_tool(
    model_text: str,
    data_text: Optional[str] = None,
    solver: str = DEFAULT_SOLVER,
) -> dict:
    """Solve an OPL model and optional data provided as strings.

    Args:
        model_text: OPL model source code as a string.
        data_text: Optional OPL data file contents as a string.
        solver: Solver name or alias. This maps to the internal solver
            backends (for example, ``highs`` -> SciPy/HiGHS or ``gurobi``).

    Returns:
        A dictionary containing solver outputs (status, objective value,
        variable assignments, and any solver metadata) as produced by
        :func:`solve`.
    """
    return solve_from_strings(model_text, data_text, solver)


@mcp.tool()
def export_py_strings_tool(
    model_text: str,
    data_text: Optional[str] = None,
    solver: str = DEFAULT_SOLVER,
) -> str:
    """Compile OPL model and optional data provided as strings to Python.

    Useful for programmatic use where model/data are held in memory. The
    function returns the compiled Python source as a string.

    Args:
        model_text: OPL model source as a string.
        data_text: Optional OPL data contents as a string.
        solver: Solver name or alias affecting code generation.

    Returns:
        A string with the compiled Python source.

    Raises:
        Compilation errors or other exceptions raised by the internal
        compiler are propagated to the caller.
    """
    return export_py_from_strings(model_text, data_text, solver)


@mcp.tool()
def compare_model_strings_tool(
    left_model_text: str,
    right_model_text: str,
    left_data_text: Optional[str] = None,
    right_data_text: Optional[str] = None,
) -> dict:
    """Compare two OPL models provided as strings for MILP equivalence.

    This exposes the same comparison engine used by the GUI's Compare models
    workflow, but avoids filesystem paths by accepting model/data contents
    directly.

    Args:
        left_model_text: OPL source for the left model.
        right_model_text: OPL source for the right model.
        left_data_text: Optional OPL data contents for the left model.
        right_data_text: Optional OPL data contents for the right model.

    Returns:
        A dictionary containing status, equivalent, level, reason, proof_steps,
        and counterexample fields.
    """
    return compare_model_strings(left_model_text, right_model_text, left_data_text, right_data_text)


if __name__ == "__main__":
    mcp.run()

"""Command-line interface for PyOPL.

Behavior:
- Running with no CLI flags launches the IDE (preserves current behavior).
- Use `--solve model.mod [data.dat]` to run a model from the command-line.
- Solver selection: default is HiGHS (`--highs`) which maps to the `scipy` backend.
  Use `--gurobi` to select Gurobi.
- Output: `--out json` (default) prints JSON result to stdout (or file with `--out-file`).
  Use `--out py` to export the compiled model code as a Python module.

This module intentionally avoids extra dependencies and uses `argparse`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import solve
from .pyopl_core import OPLCompiler
from .pyopl_ide_bootstrap import OPLIDE


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _run_solve(model_path: Path, data_path: Optional[Path], solver_key: str):
    # solver_key is 'scipy' or 'gurobi'
    try:
        results = solve(str(model_path), str(data_path) if data_path else None, solver=solver_key)
        return results
    except Exception as e:
        raise


def _export_py(model_path: Path, data_path: Optional[Path], solver_key: str) -> str:
    # compile_model expects code strings
    model_code = _read_text(model_path)
    data_code = _read_text(data_path) if data_path else None
    compiler = OPLCompiler()
    ast, code_str, data_dict = compiler.compile_model(model_code, data_code, solver=solver_key)
    return code_str


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pyopl", description="PyOPL command-line interface")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode / verbose logging")

    parser.add_argument(
        "--solve",
        nargs="+",
        metavar=("MODEL", "DATA"),
        help="Solve a model: provide Model.mod and optional Data.dat",
    )

    solver_group = parser.add_mutually_exclusive_group()
    solver_group.add_argument("--highs", action="store_true", help="Use HiGHS (scipy) solver (default)")
    solver_group.add_argument("--gurobi", action="store_true", help="Use Gurobi solver")

    parser.add_argument(
        "--out",
        choices=["json", "py"],
        default="json",
        help="Output format for --solve: json (results) or py (export compiled code)",
    )
    parser.add_argument("--out-file", help="Write output to file instead of stdout")

    args = parser.parse_args(argv)

    # If no --solve provided, launch the IDE (preserve existing behaviour)
    if not args.solve:
        ide = OPLIDE(debug=args.debug)
        ide.mainloop()
        return 0

    # Solve path
    files = args.solve
    if len(files) < 1:
        print("Error: --solve requires at least a model file", file=sys.stderr)
        return 2

    model_path = Path(files[0])
    data_path = Path(files[1]) if len(files) > 1 else None

    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        return 2
    if data_path and not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        return 2

    # Map flags to solver key
    if args.gurobi:
        solver_key = "gurobi"
    else:
        # default to HiGHS/scipy
        solver_key = "scipy"

    try:
        if args.out == "json":
            results = _run_solve(model_path, data_path, solver_key)
            out_text = json.dumps(results, indent=2, sort_keys=True, default=str)
            if args.out_file:
                _write_text(Path(args.out_file), out_text)
            else:
                print(out_text)
            return 0

        if args.out == "py":
            code = _export_py(model_path, data_path, solver_key)
            if args.out_file:
                _write_text(Path(args.out_file), code)
            else:
                print(code)
            return 0

    except Exception as e:
        print(f"Error during solve/export: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

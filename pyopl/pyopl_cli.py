"""Command-line interface for PyOPL.

Behavior:
- Running with no CLI flags launches the IDE (preserves current behavior).
- Use `solve model.mod [data.dat]` to run a model from the command-line.
- Solver selection: `--solver highs` (default) or `--solver gurobi`.
- Output: `--out json` (default) prints JSON result to stdout (or file with `--out-file`).
    Use `--out py` to export the compiled model code as a Python module.
    Use `--out lp` or `--out mps` with `--out-file` to export a solver model file.

This module intentionally avoids extra dependencies and uses `argparse`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import generative_feedback, generative_solve, solve
from .batch_solve import batch_solve
from .genai._strategy_base import (
    list_gemini_models,
    list_ollama_models,
    list_openai_models,
)
from .model_equivalence import compare_models, comparison_result_to_dict
from .pyopl_core import OPLCompiler, export_model
from .pyopl_ide_bootstrap import OPLIDE


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _run_solve(
    model_path: Path,
    data_path: Optional[Path],
    solver_key: str,
    solver_settings: Optional[dict] = None,
):
    try:
        results = solve(
            str(model_path),
            str(data_path) if data_path else None,
            solver=solver_key,
            solver_settings=solver_settings,
        )
        return results
    except Exception:
        raise


def _export_py(model_path: Path, data_path: Optional[Path], solver_key: str) -> str:
    model_code = _read_text(model_path)
    data_code = _read_text(data_path) if data_path else None
    compiler = OPLCompiler()
    ast, code_str, data_dict = compiler.compile_model(model_code, data_code, solver=solver_key)
    return code_str


def _export_lp_mps(model_path: Path, data_path: Optional[Path], out_file: Path) -> Path:
    model_code = _read_text(model_path)
    data_code = _read_text(data_path) if data_path else None
    return export_model(model_code, data_code, "scipy", out_file)


def _compare_models(
    left_model_path: Path,
    right_model_path: Path,
    left_data_path: Optional[Path],
    right_data_path: Optional[Path],
    strategy: str = "abstract",
) -> dict:
    left_model_code = _read_text(left_model_path)
    right_model_code = _read_text(right_model_path)
    left_data_code = _read_text(left_data_path) if left_data_path else None
    right_data_code = _read_text(right_data_path) if right_data_path else None
    result = compare_models(
        left_model_code,
        right_model_code,
        strategy=strategy,
        left_data_text=left_data_code,
        right_data_text=right_data_code,
    )
    return comparison_result_to_dict(result, strategy=strategy)


def _validate_input_file(path: Path, label: str) -> bool:
    if path.exists():
        return True
    print(f"Error: {label} file not found: {path}", file=sys.stderr)
    return False


def _load_solver_settings(path_text: Optional[str]) -> Optional[dict]:
    if not path_text:
        return None
    settings_path = Path(path_text)
    if not _validate_input_file(settings_path, "solver settings"):
        raise FileNotFoundError(settings_path)
    settings = json.loads(_read_text(settings_path))
    if not isinstance(settings, dict):
        raise ValueError("solver settings JSON must contain an object at the top level")
    return settings


def _write_json_result(results: object, out_file: Optional[str]) -> None:
    out_text = json.dumps(results, indent=2, sort_keys=True, default=str)
    if out_file:
        _write_text(Path(out_file), out_text)
    else:
        print(out_text)


def _validate_export_path(out_format: str, out_file: Optional[str]) -> Path:
    if not out_file:
        raise ValueError(f"--out {out_format} requires --out-file")
    out_path = Path(out_file)
    if out_path.suffix.lower() != f".{out_format}":
        raise ValueError(f"--out {out_format} requires an output file ending in .{out_format}")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyopl", description="PyOPL command-line interface")

    subparsers = parser.add_subparsers(dest="command")

    # ide subcommand (explicit debug only available here)
    p_ide = subparsers.add_parser("ide", help="Launch the PyOPL IDE")
    p_ide.add_argument("--debug", action="store_true", help="Enable debug mode / verbose logging")

    # solve subcommand
    p_solve = subparsers.add_parser("solve", help="Solve a model")
    p_solve.add_argument("model", help="Path to model (.mod)")
    p_solve.add_argument("data", nargs="?", help="Optional data (.dat)")
    p_solve.add_argument("--solver", choices=["highs", "gurobi"], default="highs", help="Solver to use (default highs)")
    p_solve.add_argument("--out", choices=["json", "py", "lp", "mps"], default="json", help="Output format")
    p_solve.add_argument("--out-file", help="Write output to file instead of stdout")
    p_solve.add_argument("--solver-settings", help="Path to a JSON object containing backend-native solver settings")

    # batch-solve subcommand
    p_batch = subparsers.add_parser(
        "batch-solve",
        help="Solve all instances in a ZIP archive",
        description=(
            "Solve a batch ZIP archive containing exactly one .mod file and one or more .dat files. "
            "The archive may also include highs.json or gurobi.json for solver settings."
        ),
    )
    p_batch.add_argument(
        "archive",
        help="ZIP containing one .mod file, one or more .dat files, and optional highs.json or gurobi.json",
    )
    p_batch.add_argument("--solver", choices=["highs", "gurobi"], default="highs", help="Solver to use (default highs)")

    # compare subcommand
    p_compare = subparsers.add_parser("compare", help="Compare two models for MILP equivalence")
    p_compare.add_argument("left_model", help="Path to the left model (.mod)")
    p_compare.add_argument("right_model", help="Path to the right model (.mod)")
    p_compare.add_argument("--left-data", help="Optional data (.dat) for the left model")
    p_compare.add_argument("--right-data", help="Optional data (.dat) for the right model")
    p_compare.add_argument(
        "--strategy",
        choices=["concrete", "abstract"],
        default="abstract",
        help="Comparison strategy (default abstract)",
    )
    p_compare.add_argument("--out-file", help="Write comparison JSON to file instead of stdout")

    # genai group
    p_genai = subparsers.add_parser("genai", help="Generative AI utilities")
    genai_sub = p_genai.add_subparsers(dest="genai_cmd")

    p_genai_list = genai_sub.add_parser("list-models", help="List LLM models")
    p_genai_list.add_argument("provider", nargs="?", choices=["openai", "google", "ollama"], default="openai")
    p_genai_list.add_argument("--prefix", dest="prefix", help="Optional prefix filter for model listing")

    genai_sub.add_parser("list-methods", help="List generative methods")

    p_genai_generate = genai_sub.add_parser("generate", help="Generate model+data from a prompt")
    p_genai_generate.add_argument("prompt", help="Prompt for generation")
    p_genai_generate.add_argument("--model-file", required=True, help="Path to write generated model (.mod)")
    p_genai_generate.add_argument("--data-file", required=True, help="Path to write generated data (.dat)")
    p_genai_generate.add_argument("--llm-model", dest="llm_model", help="LLM model name (e.g. gpt-5)")
    p_genai_generate.add_argument(
        "--provider", choices=["openai", "google", "ollama"], help="LLM provider to use for generation"
    )
    p_genai_generate.add_argument("--iterations", type=int, default=5, help="Max iterations for generative loop")
    p_genai_generate.add_argument("--out-file", help="Write generation statistics to file")
    p_genai_insight = genai_sub.add_parser("insight", help="Generate, solve, and summarise solution in lay terms (markdown)")
    p_genai_insight.add_argument("prompt", help="Prompt for insight generation")
    p_genai_insight.add_argument(
        "--provider", choices=["openai", "google", "ollama"], help="LLM provider to use for generation/feedback"
    )
    p_genai_insight.add_argument("--llm-model", dest="llm_model", help="LLM model name (e.g. gpt-5)")
    p_genai_insight.add_argument("--iterations", type=int, default=5, help="Max iterations for generative loop")
    p_genai_insight.add_argument(
        "--solver", choices=["highs", "gurobi"], default="highs", help="Solver to use for solving the generated model"
    )
    p_genai_insight.add_argument("--out-file", help="Write markdown insight to file instead of stdout")

    p_genai_ask = genai_sub.add_parser("ask", help="Ask for feedback on an existing model+data")
    p_genai_ask.add_argument("prompt", help="Prompt for feedback")
    p_genai_ask.add_argument("--model-file", required=True, help="Path to model (.mod)")
    p_genai_ask.add_argument("--data-file", required=True, help="Path to data (.dat)")
    p_genai_ask.add_argument("--llm-model", dest="llm_model", help="LLM model name (e.g. gpt-5)")
    p_genai_ask.add_argument("--provider", choices=["openai", "google", "ollama"], help="LLM provider to use")
    p_genai_ask.add_argument("--out-file", help="Write feedback JSON to file")

    return parser


def _handle_solve(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    data_path = Path(args.data) if args.data else None
    if not _validate_input_file(model_path, "model"):
        return 2
    if data_path and not _validate_input_file(data_path, "data"):
        return 2

    solver_key = "gurobi" if args.solver == "gurobi" else "scipy"
    try:
        solver_settings = _load_solver_settings(args.solver_settings)
        if args.out == "json":
            with redirect_stdout(sys.stderr):
                results = _run_solve(model_path, data_path, solver_key, solver_settings)
            _write_json_result(results, args.out_file)
            return 0

        if args.out == "py":
            code = _export_py(model_path, data_path, solver_key)
            if args.out_file:
                _write_text(Path(args.out_file), code)
            else:
                print(code)
            return 0

        out_path = _validate_export_path(args.out, args.out_file)
        with redirect_stdout(sys.stderr):
            _export_lp_mps(model_path, data_path, out_path)
        return 0
    except Exception as exc:
        if isinstance(exc, (FileNotFoundError, ValueError)) and str(exc).startswith("--out"):
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(f"Error during solve/export: {exc}", file=sys.stderr)
        return 1


def _handle_batch_solve(args: argparse.Namespace) -> int:
    try:
        with redirect_stdout(sys.stderr):
            report = batch_solve(args.archive, solver=args.solver)
        failed = any(
            "ERROR" in str(instance.get("status", "")).upper()
            or "FAIL" in str(instance.get("status", "")).upper()
            for instance in report.get("instances", [])
        )
        if failed:
            print("Batch solve completed with failed instances; see generated reports", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"Error during batch solve: {exc}", file=sys.stderr)
        return 1


def _handle_compare(args: argparse.Namespace) -> int:
    left_model_path = Path(args.left_model)
    right_model_path = Path(args.right_model)
    left_data_path = Path(args.left_data) if args.left_data else None
    right_data_path = Path(args.right_data) if args.right_data else None

    input_files = (
        (left_model_path, "left model"),
        (right_model_path, "right model"),
        (left_data_path, "left data"),
        (right_data_path, "right data"),
    )
    for path, label in input_files:
        if path and not _validate_input_file(path, label):
            return 2

    try:
        with redirect_stdout(sys.stderr):
            result = _compare_models(
                left_model_path,
                right_model_path,
                left_data_path,
                right_data_path,
                args.strategy,
            )
        out_text = json.dumps(result, indent=2, sort_keys=True, default=str)
        if args.out_file:
            _write_text(Path(args.out_file), out_text)
        else:
            print(out_text)
        return 0
    except Exception as exc:
        print(f"Error during compare: {exc}", file=sys.stderr)
        return 1


def _genai_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {}
    if getattr(args, "llm_model", None):
        kwargs["model_name"] = args.llm_model
    if getattr(args, "provider", None):
        kwargs["llm_provider"] = args.provider
    return kwargs


def _emit_output(text: str, out_file: Optional[str]) -> None:
    if out_file:
        _write_text(Path(out_file), text)
    else:
        print(text)


def _handle_genai_list_models(args: argparse.Namespace) -> int:
    provider = args.provider
    prefix = getattr(args, "prefix", None)
    listers = {
        "openai": list_openai_models,
        "google": list_gemini_models,
        "ollama": list_ollama_models,
    }
    try:
        list_models = listers[provider]
        models = list_models(prefix=prefix) if prefix else list_models()
        print("\n".join(models))
        return 0
    except Exception as exc:
        print(f"Error listing models for {provider}: {exc}", file=sys.stderr)
        return 3


def _handle_genai_list_methods(_args: argparse.Namespace) -> int:
    methods = (
        ("SyntAGM", "pyopl_generative"),
        ("Standard", "pyopl_standard"),
        ("Chain of Thought", "pyopl_chain_of_thought"),
        ("Tree of Thoughts", "pyopl_tree_of_thoughts"),
        ("CAFA", "pyopl_cafa"),
        ("Chain of Experts", "pyopl_chain_of_experts"),
        ("Reflexion", "pyopl_reflexion"),
    )
    for label, key in methods:
        print(f"{label}: {key}")
    return 0


def _handle_genai_generate(args: argparse.Namespace) -> int:
    try:
        stats = generative_solve(
            args.prompt,
            args.model_file,
            args.data_file,
            iterations=getattr(args, "iterations", 5),
            return_statistics=True,
            **_genai_kwargs(args),
        )
        text = json.dumps(stats, indent=2, sort_keys=True, default=str)
        _emit_output(text, getattr(args, "out_file", None))
        return 0
    except Exception as exc:
        print(f"Error during generative_solve: {exc}", file=sys.stderr)
        return 4


def _handle_genai_ask(args: argparse.Namespace) -> int:
    try:
        feedback = generative_feedback(
            args.prompt,
            args.model_file,
            args.data_file,
            **_genai_kwargs(args),
        )
        text = json.dumps(feedback, indent=2, sort_keys=True, default=str)
        _emit_output(text, getattr(args, "out_file", None))
        return 0
    except Exception as exc:
        print(f"Error during generative_feedback: {exc}", file=sys.stderr)
        return 4


def _unique_insight_paths() -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tmp_dir = Path(os.getcwd()) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"gen_pyopl_{timestamp}"
    model_path = tmp_dir / f"{base_name}.mod"
    data_path = tmp_dir / f"{base_name}.dat"
    suffix = 1
    while model_path.exists() or data_path.exists():
        model_path = tmp_dir / f"{base_name}_{suffix}.mod"
        data_path = tmp_dir / f"{base_name}_{suffix}.dat"
        suffix += 1
    return model_path, data_path


def _insight_summary(feedback) -> str:
    if isinstance(feedback, dict):
        return feedback.get("feedback") or feedback.get("summary") or json.dumps(feedback, indent=2)
    return str(feedback)


def _handle_genai_insight(args: argparse.Namespace) -> int:
    model_path, data_path = _unique_insight_paths()
    try:
        generative_solve(
            args.prompt,
            str(model_path),
            str(data_path),
            iterations=getattr(args, "iterations", 5),
            return_statistics=True,
            **_genai_kwargs(args),
        )
    except Exception as exc:
        print(f"Error during generation: {exc}", file=sys.stderr)
        return 4

    solver_key = "gurobi" if getattr(args, "solver", "highs") == "gurobi" else "scipy"
    try:
        with redirect_stdout(sys.stderr):
            results = _run_solve(model_path, data_path, solver_key)
    except Exception as exc:
        print(f"Error solving generated model: {exc}", file=sys.stderr)
        return 1

    solution = json.dumps(results, indent=2, sort_keys=True, default=str)
    feedback_prompt = (
        "Translate the following optimization solution into clear, non-technical language targeting a lay user. "
        f"Include key findings and suggested next steps.\n\nSolution:\n{solution}"
    )
    try:
        feedback = generative_feedback(
            feedback_prompt,
            str(model_path),
            str(data_path),
            **_genai_kwargs(args),
        )
    except Exception as exc:
        print(f"Error during feedback/translation: {exc}", file=sys.stderr)
        return 4

    markdown = (
        "# GenAI Insight\n\n" f"## Problem Description\n\n{args.prompt}\n\n" f"## Insight\n\n{_insight_summary(feedback)}\n"
    )
    _emit_output(markdown, getattr(args, "out_file", None))
    return 0


def _handle_genai(args: argparse.Namespace) -> int:
    handlers = {
        "insight": _handle_genai_insight,
        "list-models": _handle_genai_list_models,
        "list-methods": _handle_genai_list_methods,
        "generate": _handle_genai_generate,
        "ask": _handle_genai_ask,
    }
    command = getattr(args, "genai_cmd", None)
    handler = handlers.get(command) if isinstance(command, str) else None
    if handler is None:
        print("Unknown command", file=sys.stderr)
        return 2
    return handler(args)


def _dispatch_command(args: argparse.Namespace) -> int:
    # Default/no-command => launch IDE (preserve existing behaviour)
    if not args.command:
        ide = OPLIDE(debug=False)
        ide.mainloop()
        return 0

    # HANDLE IDE SUBCOMMAND (explicit IDE launch)
    if args.command == "ide":
        ide = OPLIDE(debug=getattr(args, "debug", False))
        ide.mainloop()
        return 0

    # HANDLE OTHER SUBCOMMANDS
    if args.command == "solve":
        return _handle_solve(args)

    if args.command == "batch-solve":
        return _handle_batch_solve(args)

    if args.command == "compare":
        return _handle_compare(args)

    if args.command == "genai":
        return _handle_genai(args)

    # Unknown command
    print("Unknown command", file=sys.stderr)
    return 2


def main(argv: Optional[list[str]] = None) -> int:
    return _dispatch_command(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

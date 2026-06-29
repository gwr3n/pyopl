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
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

from . import generative_feedback, generative_solve, solve
from .genai._strategy_base import (
    list_gemini_models,
    list_ollama_models,
    list_openai_models,
)
from .linear_problem_highs import export_linear_problem
from .pyopl_core import OPLCompiler
from .pyopl_ide_bootstrap import OPLIDE
from .scipy_codegen_csc import SciPyCSCCodeGenerator


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _run_solve(model_path: Path, data_path: Optional[Path], solver_key: str):
    try:
        results = solve(str(model_path), str(data_path) if data_path else None, solver=solver_key)
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
    compiler = OPLCompiler()
    ast, _code_str, data_dict = compiler.compile_model(model_code, data_code, solver="scipy")
    problem = SciPyCSCCodeGenerator(ast, data_dict).build_problem()
    return export_linear_problem(problem, out_file)


def main(argv: Optional[list[str]] = None) -> int:
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

    args = parser.parse_args(argv)

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
        model_path = Path(args.model)
        data_path = Path(args.data) if args.data else None
        if not model_path.exists():
            print(f"Error: model file not found: {model_path}", file=sys.stderr)
            return 2
        if data_path and not data_path.exists():
            print(f"Error: data file not found: {data_path}", file=sys.stderr)
            return 2

        solver_key = "gurobi" if args.solver == "gurobi" else "scipy"

        try:
            if args.out == "json":
                with redirect_stdout(sys.stderr):
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

            if args.out in ("lp", "mps"):
                if not args.out_file:
                    print(f"Error: --out {args.out} requires --out-file", file=sys.stderr)
                    return 2
                out_path = Path(args.out_file)
                if out_path.suffix.lower() != f".{args.out}":
                    print(f"Error: --out {args.out} requires an output file ending in .{args.out}", file=sys.stderr)
                    return 2
                with redirect_stdout(sys.stderr):
                    _export_lp_mps(model_path, data_path, out_path)
                return 0
        except Exception as e:
            print(f"Error during solve/export: {e}", file=sys.stderr)
            return 1

    if args.command == "genai":
        cmd = getattr(args, "genai_cmd", None)
        # genai insight: generate model+data -> solve -> ask for lay-summary
        if cmd == "insight":
            prompt = args.prompt
            provider = getattr(args, "provider", None)
            llm_model = getattr(args, "llm_model", None)
            iterations = getattr(args, "iterations", 5)
            solver_key = "gurobi" if getattr(args, "solver", "highs") == "gurobi" else "scipy"

            # Build unique tmp filenames using same scheme as IDE
            import os
            from datetime import datetime

            display_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            safe_ts = display_ts.replace(":", "-").replace(" ", "_")
            tmp_dir = Path(os.getcwd()) / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"gen_pyopl_{safe_ts}"
            model_path = tmp_dir / f"{base_name}.mod"
            data_path = tmp_dir / f"{base_name}.dat"
            i = 1
            while model_path.exists() or data_path.exists():
                model_path = tmp_dir / f"{base_name}_{i}.mod"
                data_path = tmp_dir / f"{base_name}_{i}.dat"
                i += 1

            try:
                gen_kwargs: dict = {}
                if llm_model:
                    gen_kwargs["model_name"] = llm_model
                if provider:
                    gen_kwargs["llm_provider"] = provider
                stats = generative_solve(
                    prompt,
                    str(model_path),
                    str(data_path),
                    iterations=iterations,
                    return_statistics=True,
                    **gen_kwargs,
                )
            except Exception as e:
                print(f"Error during generation: {e}", file=sys.stderr)
                return 4

            # Solve generated model
            try:
                with redirect_stdout(sys.stderr):
                    results = _run_solve(Path(model_path), Path(data_path), solver_key)
            except Exception as e:
                print(f"Error solving generated model: {e}", file=sys.stderr)
                return 1

            # Compose a feedback prompt asking to explain the results in lay terms
            sol_text = json.dumps(results, indent=2, sort_keys=True, default=str)
            feedback_prompt = f"Translate the following optimization solution into clear, non-technical language targeting a lay user. Include key findings and suggested next steps.\n\nSolution:\n{sol_text}"

            try:
                fb_kwargs: dict = {}
                if llm_model:
                    fb_kwargs["model_name"] = llm_model
                if provider:
                    fb_kwargs["llm_provider"] = provider
                feedback = generative_feedback(
                    feedback_prompt,
                    str(model_path),
                    str(data_path),
                    **fb_kwargs,
                )
            except Exception as e:
                print(f"Error during feedback/translation: {e}", file=sys.stderr)
                return 4

            # feedback may be a dict with 'feedback' or a string
            summary = None
            if isinstance(feedback, dict):
                summary = feedback.get("feedback") or feedback.get("summary") or json.dumps(feedback, indent=2)
            else:
                summary = str(feedback)

            # Include original problem description and format as Markdown
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                try:
                    prompt_text = json.dumps(prompt, indent=2, sort_keys=True, default=str)
                except Exception:
                    prompt_text = str(prompt)

            md = "# GenAI Insight\n\n"
            md += "## Problem Description\n\n"
            md += prompt_text + "\n\n"
            md += "## Insight\n\n"
            md += summary + "\n"
            out_file = getattr(args, "out_file", None)
            if out_file:
                _write_text(Path(out_file), md)
            else:
                print(md)
            return 0
        if cmd == "list-models":
            provider = args.provider
            prefix = getattr(args, "prefix", None)
            try:
                if provider == "openai":
                    models = list_openai_models(prefix=prefix) if prefix else list_openai_models()
                elif provider == "google":
                    models = list_gemini_models(prefix=prefix) if prefix else list_gemini_models()
                else:
                    models = list_ollama_models(prefix=prefix) if prefix else list_ollama_models()
                print("\n".join(models))
                return 0
            except Exception as e:
                print(f"Error listing models for {provider}: {e}", file=sys.stderr)
                return 3

        if cmd == "list-methods":
            methods = [
                ("SyntAGM", "pyopl_generative"),
                ("Standard", "pyopl_standard"),
                ("Chain of Thought", "pyopl_chain_of_thought"),
                ("Tree of Thoughts", "pyopl_tree_of_thoughts"),
                ("CAFA", "pyopl_cafa"),
                ("Chain of Experts", "pyopl_chain_of_experts"),
                ("Reflexion", "pyopl_reflexion"),
            ]
            for label, key in methods:
                print(f"{label}: {key}")
            return 0

        if cmd == "generate":
            prompt = args.prompt
            model_out = args.model_file
            data_out = args.data_file
            try:
                gen_kwargs2: dict = {}
                if getattr(args, "llm_model", None):
                    gen_kwargs2["model_name"] = getattr(args, "llm_model")
                if getattr(args, "provider", None):
                    gen_kwargs2["llm_provider"] = getattr(args, "provider")
                stats = generative_solve(
                    prompt,
                    model_out,
                    data_out,
                    iterations=getattr(args, "iterations", 5),
                    return_statistics=True,
                    **gen_kwargs2,
                )
                out_text = json.dumps(stats, indent=2, sort_keys=True, default=str)
                out_file = getattr(args, "out_file", None)
                if out_file:
                    _write_text(Path(out_file), out_text)
                else:
                    print(out_text)
                return 0
            except Exception as e:
                print(f"Error during generative_solve: {e}", file=sys.stderr)
                return 4

        if cmd == "ask":
            prompt = args.prompt
            model_file = args.model_file
            data_file = args.data_file
            try:
                fb_kwargs2: dict = {}
                if getattr(args, "llm_model", None):
                    fb_kwargs2["model_name"] = getattr(args, "llm_model")
                if getattr(args, "provider", None):
                    fb_kwargs2["llm_provider"] = getattr(args, "provider")
                feedback = generative_feedback(
                    prompt,
                    model_file,
                    data_file,
                    **fb_kwargs2,
                )
                out_text = json.dumps(feedback, indent=2, sort_keys=True, default=str)
                out_file = getattr(args, "out_file", None)
                if out_file:
                    _write_text(Path(out_file), out_text)
                else:
                    print(out_text)
                return 0
            except Exception as e:
                print(f"Error during generative_feedback: {e}", file=sys.stderr)
                return 4

    # Unknown command
    print("Unknown command", file=sys.stderr)
    return 2

    # Unknown command
    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

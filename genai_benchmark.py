import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from pyopl import solve


# Ensure parent directory exists
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# Atomic JSON write to avoid partial files on crash
def _dump_json_atomic(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


# Resolve dataset file path relative to this script
def _dataset_file(dataset_name: str) -> Path:
    root = Path(__file__).resolve().parent
    return root / "gen_ai" / "datasets" / dataset_name / f"{dataset_name}.json"


# Extract a float number from various formats
def _extract_number(value: Any) -> Optional[float]:
    # Try to coerce to float directly
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    # Try to pull first number from a string
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


# Extract objective value from result object
def _extract_objective(result: Any) -> Optional[float]:
    # Common dict keys
    if isinstance(result, dict):
        for k in ("objective_value", "objective", "obj_value", "objectiveValue", "obj"):
            if k in result:
                num = _extract_number(result[k])
                if num is not None:
                    return num
    # Object attributes
    for attr in ("objective_value", "objective", "obj_value", "objectiveValue", "obj"):
        if hasattr(result, attr):
            num = _extract_number(getattr(result, attr))
            if num is not None:
                return num
    # Fallback: try string parsing
    try:
        text = str(result)
        m = re.search(r"(objective|obj(?:ective)?_?value)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)
        if m:
            return float(m.group(2))
    except Exception:
        pass
    return None


# Infer optimization direction from model file content
def _get_direction_from_model(model_file: str):
    try:
        with open(model_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None
    if re.search(r"\bminimize\b", content, re.IGNORECASE):
        return "min"
    if re.search(r"\bmaximize\b", content, re.IGNORECASE):
        return "max"
    return None


# Unify single/batch processing into one function
def _process_item(
    index: int,
    item: Any,
    args: Any,
    mode: Any,
    solve_fn: Callable[..., Any],
    models_dir: str,
    alignment_check: bool,
    few_shot: Optional[bool] = None,
) -> tuple[dict, bool]:
    entry: dict[str, Any] = {
        "index": index,
        "solver": args.solver,
        "tolerance": args.tolerance,
        "logic": args.logic,
    }

    prompt = item.get("en_question") if isinstance(item, dict) else None
    expected_raw = item.get("en_answer") if isinstance(item, dict) else None

    if not prompt:
        entry.update({"error": "Selected item has no 'en_question'.", "exit_code": 2})
        return entry, False

    expected = _extract_number(expected_raw)
    if expected is None:
        entry.update(
            {
                "expected_objective": None,
                "error": f"Could not parse numeric en_answer from: {expected_raw}",
                "exit_code": 2,
            }
        )
        return entry, False

    entry["expected_objective"] = expected

    # Per-index output files
    model_path = os.path.join(models_dir, f"gen_pyopl_model_{index}.mod")
    data_path = os.path.join(models_dir, f"gen_pyopl_data_{index}.dat")
    _ensure_parent_dir(model_path)
    _ensure_parent_dir(data_path)

    # Step 1-2: Generate model and data
    t0 = time.perf_counter()
    try:
        gen_kwargs: dict[str, Any] = dict(
            llm_provider=args.provider,
            model_name=args.gpt,
            mode=mode,
            iterations=args.iterations,
            return_statistics=True,
            alignment_check=alignment_check,
        )
        if few_shot is not None:
            gen_kwargs["few_shot"] = few_shot
        if args.logic == "SyntAGM":
            gen_kwargs["mask_error_details"] = bool(args.mask_error_details)
            gen_kwargs["mask_lineno"] = bool(args.mask_lineno)

        gen = solve_fn(
            prompt,
            model_path,
            data_path,
            **gen_kwargs,
        )
        entry["generation_assessment"] = gen.get("assessment")
        entry["generation_iterations"] = gen.get("iterations")
        entry["syntax_errors"] = gen.get("syntax_errors")
        entry["cost"] = gen.get("cost")
        entry["duration_seconds"] = time.perf_counter() - t0
    except Exception as e:
        entry.update(
            {
                "duration_seconds": time.perf_counter() - t0,
                "error": f"generative_solve failed: {e}",
                "exit_code": 3,
            }
        )
        return entry, False

    # Step 3: Solve and compare
    try:
        result = solve(model_path, data_path, solver=args.solver)
        obj = _extract_objective(result)
        if obj is None:
            entry.update(
                {
                    "observed_objective": None,
                    "error": f"Could not extract objective_value from result: {result}",
                    "exit_code": 5,
                }
            )
            ok = False
        else:
            diff = abs(obj - expected)
            ok = diff <= args.tolerance
            entry.update({"observed_objective": obj, "abs_diff": diff, "pass": ok})
    except Exception as e:
        entry.update({"observed_objective": None, "error": f"solve failed: {e}", "exit_code": 4})
        ok = False

    # Infer direction if model exists
    direction = None
    if os.path.exists(model_path):
        try:
            direction = _get_direction_from_model(model_path)
        except Exception:
            direction = None
    entry["direction"] = direction

    # Success exit code if no earlier errors
    if "exit_code" not in entry:
        entry["exit_code"] = 0 if ok else 1

    return entry, ok


# Main entry point
def main() -> int:
    import logging

    parser = argparse.ArgumentParser(description="Run problems from a dataset with generative_solve and compare objective.")
    parser.add_argument(
        "--dataset",
        default="ComplexOR",
        help="The dataset to be used: NL4OPT, NLP4LP, IndustryOR, ComplexOR (default), StochasticOR.",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for generative_solve.")
    parser.add_argument("--provider", default="openai", help="Provider for the GPT model.")
    parser.add_argument("--gpt", default="gpt-4.1", help="GPT model to use for generation.")
    parser.add_argument("--grammar", default="bnf", help="Grammar to use for generation (none, code, bnf).")
    parser.add_argument("--solver", default="gurobi", choices=["scipy", "gurobi"], help="Solver to use for pyopl.solve.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance for equality check.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Solve all problems in the dataset and save results.")
    group.add_argument("--index", type=int, help="Index of the problem in the JSON problem list (default: 0).")
    parser.add_argument(
        "--logic",
        default="SyntAGM",
        choices=["standard", "chain_of_thought", "tree_of_thoughts", "reflexion", "cafa", "chain_of_experts", "SyntAGM"],
        help="Generative logic to use: standard, chain_of_thought, tree_of_thoughts, reflexion, cafa, chain_of_experts, or SyntAGM (default).",
    )
    # Ablation flags (only valid for --logic SyntAGM)
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot prompting (SyntAGM logic only).",
    )
    parser.add_argument(
        "--no-alignment-check",
        action="store_true",
        help="Disable alignment check (SyntAGM logic only).",
    )
    parser.add_argument(
        "--mask-error-details",
        action="store_true",
        help="Mask OPLCompiler syntax error details during SyntAGM validation (ablation flag).",
    )
    parser.add_argument(
        "--mask-lineno",
        action="store_true",
        help="Mask OPLCompiler syntax error line numbers during SyntAGM validation (ablation flag).",
    )

    # If called with no CLI args, default to showing help
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return 0

    # Default: always check alignment in benchmark mode unless explicitly disabled for SyntAGM
    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Select implementation based on --logic
    logic_to_module = {
        "SyntAGM": "pyopl.genai.pyopl_generative",
        "reflexion": "pyopl.genai.pyopl_reflexion",
        "tree_of_thoughts": "pyopl.genai.pyopl_tree_of_thoughts",
        "chain_of_thought": "pyopl.genai.pyopl_chain_of_thought",
        "cafa": "pyopl.genai.pyopl_cafa",
        "chain_of_experts": "pyopl.genai.pyopl_chain_of_experts",
        "standard": "pyopl.genai.pyopl_standard",
    }
    mod_name = logic_to_module.get(args.logic)
    if not mod_name:
        print(f"Unknown logic: {args.logic}", file=sys.stderr)
        return 2
    # Enforce ablation flags only for SyntAGM
    if args.logic != "SyntAGM" and (
        args.no_few_shot or args.no_alignment_check or args.mask_error_details or args.mask_lineno
    ):
        print(
            "--no-few-shot, --no-alignment-check, --mask-error-details, and --mask-lineno are only allowed with --logic SyntAGM.",
            file=sys.stderr,
        )
        return 2

    impl = importlib.import_module(mod_name)
    # Assign once to broadly-typed variables to satisfy mypy
    solve_fn: Callable[..., Any] = getattr(impl, "generative_solve")
    GrammarType: Any = getattr(impl, "Grammar")
    logger_names = [impl.__name__]

    # Configure module loggers for visibility
    for name in set(logger_names):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
            h = logging.StreamHandler(sys.stdout)
            h.setLevel(logging.DEBUG)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
            lg.addHandler(h)
        lg.propagate = False

    # Resolve grammar to the selected module's Grammar enum (case-insensitive)
    try:
        mode = getattr(GrammarType, args.grammar.upper())
    except AttributeError:
        valid = [g.name.lower() for g in GrammarType]
        raise ValueError(f"Unknown grammar: {args.grammar}. Valid options: {valid}")

    # Determine alignment_check and few_shot for SyntAGM ablations
    ALIGNMENT_CHECK = True
    if args.logic == "SyntAGM" and args.no_alignment_check:
        ALIGNMENT_CHECK = False
    few_shot_opt: Optional[bool] = None
    if args.logic == "SyntAGM" and args.no_few_shot:
        few_shot_opt = False

    # Load dataset
    if args.dataset in ["NL4OPT", "NLP4LP", "IndustryOR", "ComplexOR", "ReSocratic", "StochasticOR"]:
        dataset_path = _dataset_file(args.dataset)
    else:
        raise ValueError(
            "Unknown dataset: {}. Supported: NL4OPT, NLP4LP, IndustryOR, ComplexOR, ReSocratic, StochasticOR.".format(
                args.dataset
            )
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list) or not dataset:
        print("Dataset is empty or not a list.", file=sys.stderr)
        return 2

    # Output directories
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    base_dir = os.path.join("gen_ai", args.dataset, args.logic, args.grammar, args.gpt, str(args.iterations))
    # Add ablation tag subfolder only for SyntAGM and only when flags set
    if args.logic == "SyntAGM":
        tags = []
        if args.no_few_shot:
            tags.append("fewshot_off")
        if args.no_alignment_check:
            tags.append("align_off")
        if args.mask_error_details:
            tags.append("mask_errors_on")
        if args.mask_lineno:
            tags.append("mask_lineno_on")
        if tags:
            base_dir = os.path.join(base_dir, "+".join(tags))
    base_dir = os.path.join(base_dir, timestamp)

    models_dir = os.path.join(base_dir, "models")
    results_json_path = os.path.join(base_dir, f"{args.dataset}_results.json")

    _ensure_parent_dir(results_json_path)
    os.makedirs(models_dir, exist_ok=True)

    # Unified flow: choose indices based on --all
    if args.all:
        indices: list[int] = list(range(len(dataset)))
    else:
        idx = 0 if args.index is None else args.index
        if idx < 0 or idx >= len(dataset):
            print(f"Index {idx} out of range. Dataset size: {len(dataset)}", file=sys.stderr)
            return 2
        indices = [idx]

    results: list[dict[str, Any]] = []
    all_ok = True
    last_entry: dict[str, Any] | None = None
    last_ok = False

    for i in indices:
        entry, ok = _process_item(i, dataset[i], args, mode, solve_fn, models_dir, ALIGNMENT_CHECK, few_shot=few_shot_opt)
        results.append(entry)
        last_entry, last_ok = entry, ok
        all_ok = all_ok and ok

        if args.all:
            _dump_json_atomic(results_json_path, results)
            print(f"Wrote results for {len(results)} problems to {results_json_path}")

    # If single, print summary similar to previous behavior
    if not args.all and last_entry is not None:
        if "error" in last_entry:
            print(last_entry["error"], file=sys.stderr)
            return int(last_entry.get("exit_code", 1))

        # Parity with previous single-branch logging
        if last_entry.get("generation_assessment") is not None:
            print(f"generative_solve completed. Assessment: {last_entry.get('generation_assessment')}")

        print("Summary:")
        print(
            json.dumps(
                {
                    "index": last_entry["index"],
                    "solver": last_entry["solver"],
                    "expected_objective": last_entry.get("expected_objective"),
                    "observed_objective": last_entry.get("observed_objective"),
                    "abs_diff": last_entry.get("abs_diff"),
                    "tolerance": last_entry["tolerance"],
                    "pass": last_entry.get("pass", False),
                    "direction": last_entry.get("direction"),
                    "logic": last_entry["logic"],
                    "generation_duration_seconds": last_entry.get("duration_seconds"),
                },
                indent=2,
            )
        )
        return 0 if last_ok else 1

    # Batch mode exit
    return 0 if all_ok else 1


if __name__ == "__main__":
    """
    Usage examples:

    Run SyntAGM on the full dataset ReSocratic using openai/gpt-5:
        ```
        python genai_benchmark.py --provider openai --gpt gpt-5 --dataset StochasticOR --logic SyntAGM --all
        ```
    """
    raise SystemExit(main())

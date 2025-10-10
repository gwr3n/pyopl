import argparse
import json
import os
import re
import sys
from typing import Any, Optional

from pyopl import solve
from pyopl.pyopl_generative_openai import Grammar, generative_solve


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NL4OPT problem with generative_solve and compare objective.")
    parser.add_argument("--json", default="gen_ai/NL4OPT.json", help="Path to NL4OPT JSON file.")
    parser.add_argument("--index", type=int, default=0, help="Index of the problem in the JSON list.")
    parser.add_argument("--model", default="tmp/gen_pyopl_model.mod", help="Output path for generated .mod file.")
    parser.add_argument("--data", default="tmp/gen_pyopl_data.dat", help="Output path for generated .dat file.")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations for generative_solve (used with --all)."
    )
    parser.add_argument("--gpt", default="gpt-5-mini", help="GPT model to use for generation.")
    parser.add_argument("--grammar", default="code", help="Grammar to use for generation (none, code, bnf).")
    parser.add_argument("--solver", default="gurobi", choices=["scipy", "gurobi"], help="Solver to use for pyopl.solve.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance for equality check.")
    # NEW: batch mode to solve all problems
    parser.add_argument("--all", action="store_true", help="Solve all problems in NL4OPT.json and save results to --results.")

    parser.add_argument(
        "--results", default="gen_ai/NL4OPT_results.json", help="Output path for batch results JSON (used with --all)."
    )
    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    if args.grammar == "none":
        mode = Grammar.NONE
    elif args.grammar == "code":
        mode = Grammar.CODE
    elif args.grammar == "bnf":
        mode = Grammar.BNF
    else:
        raise ValueError(f"Unknown grammar: {args.grammar}. Valid options: {[g.name.lower() for g in Grammar]}")

    # Load dataset
    with open(args.json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list) or not dataset:
        print("Dataset is empty or not a list.", file=sys.stderr)
        return 2

    # NEW: batch processing branch
    if args.all:
        _ensure_parent_dir(args.results)
        results = []
        all_ok = True

        for i, item in enumerate(dataset):
            prompt = item.get("en_question")
            expected_raw = item.get("en_answer")
            entry = {
                "index": i,
                "solver": args.solver,
                "tolerance": args.tolerance,
            }

            if not prompt:
                entry.update({"error": "Selected item has no 'en_question'."})
                results.append(entry)
                all_ok = False
                continue

            expected = _extract_number(expected_raw)
            if expected is None:
                entry.update({"expected_objective": None, "error": f"Could not parse numeric en_answer from: {expected_raw}"})
                results.append(entry)
                all_ok = False
                continue

            entry["expected_objective"] = expected

            # Use per-index files to avoid overwriting
            model_root, model_ext = os.path.splitext(args.model)
            data_root, data_ext = os.path.splitext(args.data)
            model_path = f"{model_root}_{i}{model_ext or ''}"
            data_path = f"{data_root}_{i}{data_ext or ''}"

            _ensure_parent_dir(model_path)
            _ensure_parent_dir(data_path)

            # Step 1-2: Generate model and data
            try:
                result = generative_solve(
                    prompt,
                    model_path,
                    data_path,
                    model_name=args.gpt,
                    mode=mode,
                    iterations=args.iterations,
                    return_statistics=True,
                )
                entry["generation_assessment"] = result["assessment"]
                entry["generation_iterations"] = result["iterations"]
                entry["syntax_errors"] = result["syntax_errors"]
            except Exception as e:
                entry.update({"error": f"generative_solve failed: {e}"})
                results.append(entry)
                all_ok = False
                continue

            # Step 3: Solve and compare
            try:
                result = solve(model_path, data_path, solver=args.solver)
                obj = _extract_objective(result)
                if obj is None:
                    entry.update(
                        {"observed_objective": None, "error": f"Could not extract objective_value from result: {result}"}
                    )
                    all_ok = False
                else:
                    diff = abs(obj - expected)
                    ok = diff <= args.tolerance
                    entry.update(
                        {
                            "observed_objective": obj,
                            "abs_diff": diff,
                            "pass": ok,
                        }
                    )
                    if not ok:
                        all_ok = False
            except Exception as e:
                entry.update({"observed_objective": None, "error": f"solve failed: {e}"})
                all_ok = False

            results.append(entry)

            with open(args.results, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Wrote results for {len(results)} problems to {args.results}")

        return 0 if all_ok else 1

    # Single problem branch
    if args.index < 0 or args.index >= len(dataset):
        print(f"Index {args.index} out of range. Dataset size: {len(dataset)}", file=sys.stderr)
        return 2

    item = dataset[args.index]
    prompt = item.get("en_question")
    expected_raw = item.get("en_answer")

    if not prompt:
        print("Selected item has no 'en_question'.", file=sys.stderr)
        return 2

    expected = _extract_number(expected_raw)
    if expected is None:
        print(f"Could not parse numeric en_answer from: {expected_raw}", file=sys.stderr)
        return 2

    # Ensure output dirs exist
    _ensure_parent_dir(args.model)
    _ensure_parent_dir(args.data)

    # Step 1-2: Generate model and data
    try:
        result = generative_solve(
            prompt, args.model, args.data, model_name=args.gpt, mode=mode, iterations=args.iterations, return_statistics=True
        )
        assessment = result["assessment"]
        print(f"generative_solve completed. Assessment: {assessment}")
    except Exception as e:
        print(f"generative_solve failed: {e}", file=sys.stderr)
        return 3

    # Step 3: Solve and compare
    try:
        result = solve(args.model, args.data, solver=args.solver)
        obj = _extract_objective(result)
    except Exception as e:
        print(f"solve failed: {e}", file=sys.stderr)
        return 4

    if obj is None:
        print(f"Could not extract objective_value from result: {result}", file=sys.stderr)
        return 5

    diff = abs(obj - expected)
    ok = diff <= args.tolerance

    print("Summary:")
    print(
        json.dumps(
            {
                "index": args.index,
                "solver": args.solver,
                "expected_objective": expected,
                "observed_objective": obj,
                "abs_diff": diff,
                "tolerance": args.tolerance,
                "pass": ok,
            },
            indent=2,
        )
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

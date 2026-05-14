import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


class TestGenAIBenchmarkContinue(unittest.TestCase):
    def test_continue_resumes_from_first_missing_index(self) -> None:
        # Import here so patch targets resolve correctly.
        import genai_benchmark

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # Create a tiny fake dataset.
            dataset_path = td_path / "dataset.json"
            dataset = [
                {"en_question": "q0", "en_answer": 1},
                {"en_question": "q1", "en_answer": 2},
                {"en_question": "q2", "en_answer": 3},
            ]
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

            # Create an existing run directory with partial results (0 and 1 done).
            dataset_name = "ChallengeOR"
            logic = "SyntAGM"
            grammar = "bnf"
            gpt = "gpt-test"
            iterations = "5"
            base_root = td_path / "gen_ai" / dataset_name / logic / grammar / gpt / iterations
            run_id = "20260101T000000"
            run_dir = base_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            existing_results_path = run_dir / f"{dataset_name}.json"
            existing_results = [
                {"index": 0, "exit_code": 0},
                {"index": 1, "exit_code": 0},
            ]
            existing_results_path.write_text(json.dumps(existing_results), encoding="utf-8")

            # Dummy impl module and solve function.
            call_log: list[int] = []

            def dummy_generative_solve(prompt: Any, model_path: str, data_path: str, **kwargs: Any) -> dict[str, Any]:
                # Record which prompt we were asked to solve.
                # We infer the index from the file name.
                idx = int(Path(model_path).stem.split("_")[-1])
                call_log.append(idx)
                Path(model_path).write_text("minimize\n", encoding="utf-8")
                Path(data_path).write_text("// data\n", encoding="utf-8")
                return {"assessment": "ok", "iterations": 1, "cost": 0, "syntax_errors": None}

            class DummyGrammar:
                BNF: str = "BNF"

            dummy_module = SimpleNamespace(generative_solve=dummy_generative_solve, Grammar=DummyGrammar, __name__="dummy")

            def dummy_solve(model_path: str, data_path: str, solver: str = "gurobi") -> dict[str, Any]:
                # Objective value matches expected for all indices.
                idx = int(Path(model_path).stem.split("_")[-1])
                return {"objective_value": float(idx + 1)}

            argv = [
                "genai_benchmark.py",
                "--provider",
                "openai",
                "--gpt",
                gpt,
                "--dataset",
                dataset_name,
                "--logic",
                logic,
                "--all",
                "--continue",
                run_id,
            ]

            buf = io.StringIO()
            with (
                patch.object(genai_benchmark, "_dataset_file", return_value=dataset_path),
                patch.object(genai_benchmark.importlib, "import_module", return_value=dummy_module),
                patch.object(genai_benchmark, "solve", side_effect=dummy_solve),
                patch.object(genai_benchmark.sys, "argv", argv),
                redirect_stdout(buf),
            ):
                # Ensure relative output paths land inside the temp directory.
                cwd_before = os.getcwd()
                os.chdir(str(td_path))
                try:
                    ret = genai_benchmark.main()
                finally:
                    os.chdir(cwd_before)

            self.assertEqual(ret, 0)

            # Should have only run the missing index 2.
            self.assertEqual(call_log, [2])

            # Existing results file should now have 3 entries.
            payload = json.loads(existing_results_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 3)
            self.assertEqual({e.get("index") for e in payload}, {0, 1, 2})

            out = buf.getvalue()
            self.assertIn("Resuming from existing results", out)

    def test_continue_without_value_resumes_latest(self) -> None:
        import genai_benchmark

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            dataset_path = td_path / "dataset.json"
            dataset = [
                {"en_question": "q0", "en_answer": 1},
                {"en_question": "q1", "en_answer": 2},
            ]
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

            dataset_name = "ChallengeOR"
            logic = "SyntAGM"
            grammar = "bnf"
            gpt = "gpt-test"
            iterations = "5"
            base_root = td_path / "gen_ai" / dataset_name / logic / grammar / gpt / iterations

            # Two runs; latest should be chosen.
            old_dir = base_root / "20260101T000000"
            new_dir = base_root / "20260101T000001"
            old_dir.mkdir(parents=True, exist_ok=True)
            new_dir.mkdir(parents=True, exist_ok=True)

            (old_dir / f"{dataset_name}.json").write_text(json.dumps([{"index": 0, "exit_code": 0}]), encoding="utf-8")
            new_results_path = new_dir / f"{dataset_name}.json"
            new_results_path.write_text(json.dumps([{"index": 0, "exit_code": 0}]), encoding="utf-8")

            call_log: list[int] = []

            def dummy_generative_solve(prompt: Any, model_path: str, data_path: str, **kwargs: Any) -> dict[str, Any]:
                idx = int(Path(model_path).stem.split("_")[-1])
                call_log.append(idx)
                Path(model_path).write_text("minimize\n", encoding="utf-8")
                Path(data_path).write_text("// data\n", encoding="utf-8")
                return {"assessment": "ok", "iterations": 1, "cost": 0, "syntax_errors": None}

            class DummyGrammar:
                BNF: str = "BNF"

            dummy_module = SimpleNamespace(generative_solve=dummy_generative_solve, Grammar=DummyGrammar, __name__="dummy")

            def dummy_solve(model_path: str, data_path: str, solver: str = "gurobi") -> dict[str, Any]:
                idx = int(Path(model_path).stem.split("_")[-1])
                return {"objective_value": float(idx + 1)}

            argv = [
                "genai_benchmark.py",
                "--provider",
                "openai",
                "--gpt",
                gpt,
                "--dataset",
                dataset_name,
                "--logic",
                logic,
                "--all",
                "--continue",
            ]

            buf = io.StringIO()
            with (
                patch.object(genai_benchmark, "_dataset_file", return_value=dataset_path),
                patch.object(genai_benchmark.importlib, "import_module", return_value=dummy_module),
                patch.object(genai_benchmark, "solve", side_effect=dummy_solve),
                patch.object(genai_benchmark.sys, "argv", argv),
                redirect_stdout(buf),
            ):
                cwd_before = os.getcwd()
                os.chdir(str(td_path))
                try:
                    ret = genai_benchmark.main()
                finally:
                    os.chdir(cwd_before)

            self.assertEqual(ret, 0)
            # Only missing index 1 should run.
            self.assertEqual(call_log, [1])
            payload = json.loads(new_results_path.read_text(encoding="utf-8"))
            self.assertEqual({e.get("index") for e in payload}, {0, 1})


if __name__ == "__main__":
    unittest.main()

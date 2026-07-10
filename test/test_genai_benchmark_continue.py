import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


class DummyGrammar:
    NONE: str = "NONE"
    BNF: str = "BNF"
    CODE: str = "CODE"


def run_main_in_tmp(
    genai_benchmark: Any,
    td_path: Path,
    dataset_path: Path,
    argv: list[str],
    *,
    dummy_module: Any,
    solve_side_effect: Any,
) -> tuple[int, str, str]:
    buf = io.StringIO()
    err = io.StringIO()
    with (
        patch.object(genai_benchmark, "_dataset_file", return_value=dataset_path),
        patch.object(genai_benchmark.importlib, "import_module", return_value=dummy_module),
        patch.object(genai_benchmark, "solve", side_effect=solve_side_effect),
        patch.object(genai_benchmark.sys, "argv", argv),
        redirect_stdout(buf),
        redirect_stderr(err),
    ):
        cwd_before = os.getcwd()
        os.chdir(str(td_path))
        try:
            ret = genai_benchmark.main()
        finally:
            os.chdir(cwd_before)
    return ret, buf.getvalue(), err.getvalue()


class TestGenAIBenchmarkHelpers(unittest.TestCase):
    def test_json_result_helpers_and_latest_run(self) -> None:
        from tools import genai_benchmark

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results_path = root / "results.json"
            genai_benchmark._dump_json_atomic(str(results_path), [{"index": 1}, ["ignored"], {"index": "bad"}])

            loaded = genai_benchmark._load_results_json(str(results_path))
            self.assertEqual(loaded, [{"index": 1}, {"index": "bad"}])
            self.assertEqual(genai_benchmark._completed_indices(loaded), {1})

            old_run = root / "20260101T000000"
            new_run = root / "20260101T000001"
            old_run.mkdir()
            new_run.mkdir()
            (old_run / "ChallengeOR_results.json").write_text("[]", encoding="utf-8")
            (new_run / "ChallengeOR_results.json").write_text("[]", encoding="utf-8")

            self.assertEqual(
                genai_benchmark._find_results_file(str(new_run), ["missing.json", "ChallengeOR_results.json"]),
                str(new_run / "ChallengeOR_results.json"),
            )
            self.assertEqual(
                genai_benchmark._find_latest_run_dir(str(root), ["ChallengeOR_results.json"]),
                (str(new_run), "ChallengeOR_results.json"),
            )

    def test_extract_number_objective_and_direction(self) -> None:
        from tools import genai_benchmark

        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.mod"
            model_path.write_text("maximize profit;", encoding="utf-8")

            self.assertEqual(genai_benchmark._extract_number("objective = -1.25e2"), -125.0)
            self.assertIsNone(genai_benchmark._extract_number("no number"))
            self.assertEqual(genai_benchmark._extract_objective({"obj": "$42.5"}), 42.5)
            self.assertEqual(genai_benchmark._extract_objective(SimpleNamespace(objectiveValue="7")), 7.0)
            self.assertEqual(genai_benchmark._extract_objective("objective_value: 9.5"), 9.5)
            self.assertEqual(genai_benchmark._get_direction_from_model(str(model_path)), "max")
            self.assertIsNone(genai_benchmark._get_direction_from_model(str(Path(td) / "missing.mod")))

    def test_process_item_error_paths(self) -> None:
        from tools import genai_benchmark

        args = SimpleNamespace(
            solver="gurobi", tolerance=1e-6, logic="standard", provider="openai", gpt="gpt-test", iterations=1
        )

        with tempfile.TemporaryDirectory() as td:
            no_prompt, ok = genai_benchmark._process_item(
                0, {"en_answer": 1}, args, DummyGrammar.BNF, lambda *a, **k: {}, td, True
            )
            bad_answer, ok2 = genai_benchmark._process_item(
                1, {"en_question": "q", "en_answer": "none"}, args, DummyGrammar.BNF, lambda *a, **k: {}, td, True
            )

            def bad_generate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("generation down")

            gen_fail, ok3 = genai_benchmark._process_item(
                2, {"en_question": "q", "en_answer": 3}, args, DummyGrammar.BNF, bad_generate, td, True
            )

        self.assertFalse(ok)
        self.assertEqual(no_prompt["exit_code"], 2)
        self.assertFalse(ok2)
        self.assertEqual(bad_answer["exit_code"], 2)
        self.assertFalse(ok3)
        self.assertEqual(gen_fail["exit_code"], 3)

    def test_process_item_solve_failures_and_mismatch(self) -> None:
        from tools import genai_benchmark

        args = SimpleNamespace(
            solver="gurobi",
            tolerance=0.1,
            logic="SyntAGM",
            provider="openai",
            gpt="gpt-test",
            iterations=1,
            syntax_error_reporting="masked",
        )

        def dummy_generate(prompt: str, model_path: str, data_path: str, **kwargs: Any) -> dict[str, Any]:
            Path(model_path).write_text("minimize cost;", encoding="utf-8")
            Path(data_path).write_text("// data", encoding="utf-8")
            return {"assessment": "ok", "iterations": 1, "syntax_errors": [], "cost": {}}

        with tempfile.TemporaryDirectory() as td:
            with patch.object(genai_benchmark, "solve", return_value={"status": "no objective"}):
                no_obj, ok = genai_benchmark._process_item(
                    0, {"en_question": "q", "en_answer": 1}, args, DummyGrammar.BNF, dummy_generate, td, False, few_shot=False
                )
            with patch.object(genai_benchmark, "solve", side_effect=RuntimeError("solver down")):
                solve_fail, ok2 = genai_benchmark._process_item(
                    1, {"en_question": "q", "en_answer": 1}, args, DummyGrammar.BNF, dummy_generate, td, False, few_shot=False
                )
            with patch.object(genai_benchmark, "solve", return_value={"objective_value": 3.0}):
                mismatch, ok3 = genai_benchmark._process_item(
                    2, {"en_question": "q", "en_answer": 1}, args, DummyGrammar.BNF, dummy_generate, td, False, few_shot=False
                )

        self.assertFalse(ok)
        self.assertEqual(no_obj["exit_code"], 5)
        self.assertFalse(ok2)
        self.assertEqual(solve_fail["exit_code"], 4)
        self.assertFalse(ok3)
        self.assertEqual(mismatch["exit_code"], 1)
        self.assertEqual(mismatch["direction"], "min")

    def test_process_item_uses_milp_equivalence_when_model_and_data_present(self) -> None:
        from tools import genai_benchmark

        args = SimpleNamespace(
            solver="gurobi", tolerance=1e-6, logic="standard", provider="openai", gpt="gpt-test", iterations=1
        )

        def dummy_generate(prompt: str, model_path: str, data_path: str, **kwargs: Any) -> dict[str, Any]:
            Path(model_path).write_text("minimize cost;", encoding="utf-8")
            Path(data_path).write_text("// generated data", encoding="utf-8")
            return {"assessment": "ok", "iterations": 1, "syntax_errors": [], "cost": {}}

        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(
                    genai_benchmark, "linear_problem_from_opl", side_effect=["expected", "generated"]
                ) as compile_mock,
                patch.object(genai_benchmark, "compare", return_value=True) as compare_mock,
                patch.object(genai_benchmark, "solve") as solve_mock,
            ):
                entry, ok = genai_benchmark._process_item(
                    0,
                    {"en_question": "q", "model": "expected model", "data": "expected data"},
                    args,
                    DummyGrammar.BNF,
                    dummy_generate,
                    td,
                    True,
                )

        self.assertTrue(ok)
        self.assertEqual(entry["comparison"], "milp_equivalence")
        self.assertEqual(entry["exit_code"], 0)
        self.assertTrue(entry["pass"])
        compile_mock.assert_any_call("expected model", "expected data")
        compile_mock.assert_any_call("minimize cost;", "// generated data")
        compare_mock.assert_called_once_with("expected", "generated", tolerance=1e-6)
        solve_mock.assert_not_called()

    def test_process_item_falls_back_to_objective_when_only_en_answer_present(self) -> None:
        from tools import genai_benchmark

        args = SimpleNamespace(
            solver="gurobi", tolerance=0.1, logic="standard", provider="openai", gpt="gpt-test", iterations=1
        )

        def dummy_generate(prompt: str, model_path: str, data_path: str, **kwargs: Any) -> dict[str, Any]:
            Path(model_path).write_text("minimize cost;", encoding="utf-8")
            Path(data_path).write_text("// data", encoding="utf-8")
            return {"assessment": "ok", "iterations": 1, "syntax_errors": [], "cost": {}}

        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(genai_benchmark, "linear_problem_from_opl") as compile_mock,
                patch.object(genai_benchmark, "compare") as compare_mock,
                patch.object(genai_benchmark, "solve", return_value={"objective_value": 10.05}) as solve_mock,
            ):
                entry, ok = genai_benchmark._process_item(
                    0,
                    {"en_question": "q", "en_answer": 10.0},
                    args,
                    DummyGrammar.BNF,
                    dummy_generate,
                    td,
                    True,
                )

        self.assertTrue(ok)
        self.assertEqual(entry["comparison"], "objective")
        self.assertEqual(entry["exit_code"], 0)
        self.assertAlmostEqual(entry["abs_diff"], 0.05)
        solve_mock.assert_called_once()
        compile_mock.assert_not_called()
        compare_mock.assert_not_called()


class TestGenAIBenchmarkContinue(unittest.TestCase):
    def test_main_no_args_prints_help(self) -> None:
        from tools import genai_benchmark

        buf = io.StringIO()
        with patch.object(genai_benchmark.sys, "argv", ["genai_benchmark.py"]), redirect_stdout(buf):
            ret = genai_benchmark.main()

        self.assertEqual(ret, 0)
        self.assertIn("Run problems from a dataset", buf.getvalue())

    def test_single_index_success_prints_summary(self) -> None:
        from tools import genai_benchmark

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset_path = td_path / "dataset.json"
            dataset_path.write_text(json.dumps([{"en_question": "q", "en_answer": "10"}]), encoding="utf-8")

            def dummy_generative_solve(prompt: Any, model_path: str, data_path: str, **kwargs: Any) -> dict[str, Any]:
                Path(model_path).write_text("maximize profit;", encoding="utf-8")
                Path(data_path).write_text("// data", encoding="utf-8")
                return {"assessment": "ok", "iterations": 1, "cost": {}, "syntax_errors": []}

            dummy_module = SimpleNamespace(generative_solve=dummy_generative_solve, Grammar=DummyGrammar, __name__="dummy")
            argv = [
                "genai_benchmark.py",
                "--dataset",
                "ChallengeOR",
                "--logic",
                "standard",
                "--index",
                "0",
                "--gpt",
                "gpt-test",
            ]

            ret, out, err = run_main_in_tmp(
                genai_benchmark,
                td_path,
                dataset_path,
                argv,
                dummy_module=dummy_module,
                solve_side_effect=lambda *args, **kwargs: SimpleNamespace(objective_value="10"),
            )

        self.assertEqual(ret, 0)
        self.assertEqual(err, "")
        self.assertIn("generative_solve completed", out)
        self.assertIn('"pass": true', out)

    def test_main_validation_errors(self) -> None:
        from tools import genai_benchmark

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset_path = td_path / "dataset.json"
            dataset_path.write_text(json.dumps([{"en_question": "q", "en_answer": 1}]), encoding="utf-8")
            dummy_module = SimpleNamespace(generative_solve=lambda *a, **k: {}, Grammar=DummyGrammar, __name__="dummy")

            ret, _out, err = run_main_in_tmp(
                genai_benchmark,
                td_path,
                dataset_path,
                ["genai_benchmark.py", "--dataset", "ChallengeOR", "--logic", "standard", "--no-few-shot"],
                dummy_module=dummy_module,
                solve_side_effect=lambda *a, **k: {},
            )
            ret2, _out2, err2 = run_main_in_tmp(
                genai_benchmark,
                td_path,
                dataset_path,
                ["genai_benchmark.py", "--dataset", "ChallengeOR", "--logic", "standard", "--index", "9"],
                dummy_module=dummy_module,
                solve_side_effect=lambda *a, **k: {},
            )

        self.assertEqual(ret, 2)
        self.assertIn("only allowed with --logic SyntAGM", err)
        self.assertEqual(ret2, 2)
        self.assertIn("out of range", err2)

    def test_all_fresh_run_emits_resume_hint_and_writes_results(self) -> None:
        from tools import genai_benchmark

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset_path = td_path / "dataset.json"
            dataset_path.write_text(json.dumps([{"en_question": "q0", "en_answer": 1}]), encoding="utf-8")

            base_root = td_path / "gen_ai" / "ChallengeOR" / "SyntAGM" / "bnf" / "gpt-test" / "5"
            old_run = base_root / "20260101T000000"
            old_run.mkdir(parents=True)
            (old_run / "ChallengeOR_results.json").write_text("[]", encoding="utf-8")

            def dummy_generative_solve(prompt: Any, model_path: str, data_path: str, **kwargs: Any) -> dict[str, Any]:
                Path(model_path).write_text("minimize cost;", encoding="utf-8")
                Path(data_path).write_text("// data", encoding="utf-8")
                return {"assessment": "ok", "iterations": 1, "cost": {}, "syntax_errors": []}

            dummy_module = SimpleNamespace(generative_solve=dummy_generative_solve, Grammar=DummyGrammar, __name__="dummy")
            argv = ["genai_benchmark.py", "--dataset", "ChallengeOR", "--logic", "SyntAGM", "--all", "--gpt", "gpt-test"]

            ret, out, err = run_main_in_tmp(
                genai_benchmark,
                td_path,
                dataset_path,
                argv,
                dummy_module=dummy_module,
                solve_side_effect=lambda *args, **kwargs: {"objective_value": 1.0},
            )

        self.assertEqual(ret, 0)
        self.assertEqual(err, "")
        self.assertIn("To resume it", out)
        self.assertIn("Wrote results", out)

    def test_continue_resumes_from_first_missing_index(self) -> None:
        # Import here so patch targets resolve correctly.
        from tools import genai_benchmark

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

            existing_results_path = run_dir / f"{dataset_name}_results.json"
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
        from tools import genai_benchmark

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

            (old_dir / f"{dataset_name}_results.json").write_text(json.dumps([{"index": 0, "exit_code": 0}]), encoding="utf-8")
            new_results_path = new_dir / f"{dataset_name}_results.json"
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

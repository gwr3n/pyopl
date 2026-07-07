import asyncio
import importlib
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT_LOGGER = logging.getLogger()
_ROOT_LOGGER_LEVEL = _ROOT_LOGGER.level
_ROOT_LOGGER_HANDLERS = list(_ROOT_LOGGER.handlers)

_pyopl_mcp = importlib.import_module("pyopl._pyopl_mcp")
_rhetor_mcp = importlib.import_module("pyopl._rhetor_mcp")
pyopl_mcp = importlib.import_module("pyopl.pyopl_mcp")
rhetor_mcp = importlib.import_module("pyopl.rhetor_mcp")

_ROOT_LOGGER.handlers[:] = _ROOT_LOGGER_HANDLERS
_ROOT_LOGGER.setLevel(_ROOT_LOGGER_LEVEL)


class TestPyOPLMCP(unittest.TestCase):
    def test_solver_normalization_and_backend_mapping(self):
        self.assertEqual(_pyopl_mcp._normalize_solver(None), "scipy")
        self.assertEqual(_pyopl_mcp._normalize_solver(" highs "), "scipy")
        self.assertEqual(_pyopl_mcp._normalize_solver("GUROBI"), "gurobi")
        self.assertEqual(_pyopl_mcp._normalize_solver("custom"), "custom")
        self.assertEqual(_pyopl_mcp._solve_backend("gurobi"), "gurobi")
        self.assertEqual(_pyopl_mcp._solve_backend("highs"), "scipy")

    def test_export_py_from_files_reads_inputs_and_normalizes_solver(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.mod"
            data_path = Path(tmp_dir) / "data.dat"
            model_path.write_text("model text", encoding="utf-8")
            data_path.write_text("data text", encoding="utf-8")

            with patch.object(_pyopl_mcp, "_compile_to_python", return_value="compiled") as compile_mock:
                result = _pyopl_mcp.export_py_from_files(model_path, data_path, solver="highs")

        self.assertEqual(result, "compiled")
        compile_mock.assert_called_once_with("model text", "data text", solver="highs")

    def test_export_py_from_strings_delegates_to_compiler_helper(self):
        with patch.object(_pyopl_mcp, "_compile_to_python", return_value="compiled") as compile_mock:
            result = _pyopl_mcp.export_py_from_strings("model", "data", solver="gurobi")

        self.assertEqual(result, "compiled")
        compile_mock.assert_called_once_with("model", "data", solver="gurobi")

    def test_compile_to_python_uses_normalized_solver(self):
        class FakeCompiler:
            def compile_model(self, model_text, data_text, solver):
                return {"ast": True}, f"code:{solver}:{model_text}:{data_text}", {"data": True}

        with patch.object(_pyopl_mcp, "OPLCompiler", return_value=FakeCompiler()) as compiler_cls:
            result = _pyopl_mcp._compile_to_python("m", "d", solver="highs")

        self.assertEqual(result, "code:scipy:m:d")
        compiler_cls.assert_called_once_with()

    def test_solve_from_files_and_strings_delegate_to_solver(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.mod"
            data_path = Path(tmp_dir) / "data.dat"
            model_path.write_text("m", encoding="utf-8")
            data_path.write_text("d", encoding="utf-8")

            with patch.object(_pyopl_mcp, "solve", return_value={"status": "OK"}) as solve_mock:
                result = _pyopl_mcp.solve_from_files(model_path, data_path, solver="highs")

            self.assertEqual(result, {"status": "OK"})
            solve_mock.assert_called_once_with(str(model_path), str(data_path), solver="scipy")

        captured = {}

        def fake_solve_from_files(model_file, data_file, solver):
            captured["model_text"] = Path(model_file).read_text(encoding="utf-8")
            captured["data_text"] = Path(data_file).read_text(encoding="utf-8")
            captured["solver"] = solver
            return {"status": "OK"}

        with patch.object(_pyopl_mcp, "solve_from_files", side_effect=fake_solve_from_files) as solve_files_mock:
            result = _pyopl_mcp.solve_from_strings("model text", "data text", solver="gurobi")

        self.assertEqual(result, {"status": "OK"})
        self.assertEqual(solve_files_mock.call_count, 1)
        self.assertEqual(captured, {"model_text": "model text", "data_text": "data text", "solver": "gurobi"})

    def test_pyopl_tools_and_public_wrapper_delegate(self):
        with patch.object(_pyopl_mcp, "solve_from_files", return_value={"status": "OK"}) as solve_mock:
            self.assertEqual(_pyopl_mcp.solve_files_tool("m.mod", "d.dat", "highs"), {"status": "OK"})
        solve_mock.assert_called_once_with("m.mod", "d.dat", "highs")

        with patch.object(_pyopl_mcp, "export_py_from_strings", return_value="code") as export_mock:
            self.assertEqual(_pyopl_mcp.export_py_strings_tool("m", None, "scipy"), "code")
        export_mock.assert_called_once_with("m", None, "scipy")

        with patch.object(_pyopl_mcp, "compare_model_strings", return_value={"equivalent": True}) as compare_mock:
            self.assertEqual(_pyopl_mcp.compare_model_strings_tool("left", "right", None, None), {"equivalent": True})
        compare_mock.assert_called_once_with("left", "right", None, None)

        with patch.object(pyopl_mcp.mcp, "run") as run_mock:
            pyopl_mcp.main()
        run_mock.assert_called_once_with()

    def test_compare_model_strings_returns_equivalence_result_dict(self):
        left_model = """
            dvar float+ x;

            minimize 2 * x;

            subject to {
                x <= 3;
            }
            """
        right_model = """
            dvar float+ y;

            minimize 2 * y;

            subject to {
                2 * y <= 6;
            }
            """

        result = _pyopl_mcp.compare_model_strings_tool(left_model, right_model)

        self.assertEqual(result["status"], "equivalent")
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["level"], "solver_implied")
        self.assertIn("normalized both models", result["proof_steps"])
        self.assertIsNone(result["counterexample"])


class TestRhetorMCP(unittest.TestCase):
    def test_provider_normalization_and_llm_kwargs(self):
        self.assertEqual(_rhetor_mcp._normalize_solver("highs"), "scipy")
        self.assertEqual(_rhetor_mcp._solve_backend("GUROBI"), "gurobi")
        self.assertEqual(_rhetor_mcp._normalize_provider(None), _rhetor_mcp.DEFAULT_PROVIDER)
        self.assertEqual(_rhetor_mcp._normalize_provider(" Gemini "), "google")
        self.assertEqual(_rhetor_mcp._normalize_provider("ollama"), "ollama")
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            _rhetor_mcp._normalize_provider("anthropic")

        self.assertEqual(_rhetor_mcp._build_llm_kwargs(), {})
        self.assertEqual(
            _rhetor_mcp._build_llm_kwargs(llm_model="gpt-x", provider="gemini"),
            {"model_name": "gpt-x", "llm_provider": "google"},
        )

    def test_thread_and_coroutine_helpers_return_results(self):
        self.assertEqual(_rhetor_mcp._run_in_thread(lambda x, y=0: x + y, 2, y=3), 5)
        self.assertEqual(_rhetor_mcp._run_coro_in_thread(lambda: asyncio.sleep(0, result="done")), "done")

    def test_generate_uses_graphchain_when_available(self):
        async def fake_async(prompt, model_file, data_file, **kwargs):
            return {"prompt": prompt, "model_file": model_file, "data_file": data_file, "kwargs": kwargs}

        with (
            patch.object(_rhetor_mcp, "_try_import_graphchain", return_value=fake_async),
            patch.object(
                _rhetor_mcp, "_run_coro_in_thread", side_effect=lambda factory: asyncio.run(factory())
            ) as coro_runner,
        ):
            result = _rhetor_mcp._generate_with_best_available_backend(
                "prompt",
                "model.mod",
                "data.dat",
                iterations=7,
                llm_model="model-x",
                provider="gemini",
            )

        self.assertEqual(result["prompt"], "prompt")
        self.assertEqual(result["kwargs"]["model_name"], "model-x")
        self.assertEqual(result["kwargs"]["iterations"], 7)
        self.assertEqual(result["kwargs"]["llm_provider"], "google")
        coro_runner.assert_called_once()

    def test_generate_falls_back_to_sync_backend(self):
        with (
            patch.object(_rhetor_mcp, "_try_import_graphchain", return_value=None),
            patch.object(_rhetor_mcp, "_run_in_thread", return_value={"stats": True}) as runner,
        ):
            result = _rhetor_mcp._generate_with_best_available_backend(
                "prompt",
                "model.mod",
                "data.dat",
                iterations=2,
                llm_model="m",
                provider="ollama",
            )

        self.assertEqual(result, {"stats": True})
        args, kwargs = runner.call_args
        self.assertIs(args[0], _rhetor_mcp.generative_solve)
        self.assertEqual(args[1:4], ("prompt", "model.mod", "data.dat"))
        self.assertEqual(kwargs["iterations"], 2)
        self.assertTrue(kwargs["return_statistics"])
        self.assertEqual(kwargs["model_name"], "m")
        self.assertEqual(kwargs["llm_provider"], "ollama")

    def test_feedback_and_listing_helpers(self):
        with patch.object(_rhetor_mcp, "_run_in_thread", return_value="feedback") as runner:
            result = _rhetor_mcp._ask_for_feedback("prompt", "m.mod", "d.dat", llm_model="m", provider="openai")

        self.assertEqual(result, "feedback")
        args, kwargs = runner.call_args
        self.assertIs(args[0], _rhetor_mcp.generative_feedback)
        self.assertEqual(args[1:4], ("prompt", "m.mod", "d.dat"))
        self.assertEqual(kwargs, {"model_name": "m", "llm_provider": "openai"})

        with patch.object(_rhetor_mcp, "LLMProvider", [type("P", (), {"value": "p1"})(), type("P", (), {"value": "p2"})()]):
            self.assertEqual(_rhetor_mcp.list_providers(), ["p1", "p2"])
        with patch.object(_rhetor_mcp, "LLMProvider", object()):
            self.assertEqual(_rhetor_mcp.list_providers(), ["openai", "google", "ollama"])

        with (
            patch.object(_rhetor_mcp, "list_openai_models", return_value=["gpt"]),
            patch.object(_rhetor_mcp, "list_gemini_models", return_value=["gemini"]),
            patch.object(_rhetor_mcp, "list_ollama_models", return_value=["llama"]),
        ):
            self.assertEqual(_rhetor_mcp.list_models("openai"), ["gpt"])
            self.assertEqual(_rhetor_mcp.list_models("google", prefix="gem"), ["gemini"])
            self.assertEqual(_rhetor_mcp.list_models("ollama"), ["llama"])

    def test_rhetor_tools_and_public_wrapper_delegate(self):
        self.assertEqual(_rhetor_mcp.list_methods_tool(), _rhetor_mcp.METHODS)

        with patch.object(_rhetor_mcp, "list_models", return_value=["m"]) as list_models_mock:
            self.assertEqual(_rhetor_mcp.list_models_tool("gemini", "g"), ["m"])
        list_models_mock.assert_called_once_with(provider="gemini", prefix="g")

        with patch.object(_rhetor_mcp, "_generate_with_best_available_backend", return_value={"ok": True}) as gen_mock:
            self.assertEqual(_rhetor_mcp.generate_tool("p", "m.mod", "d.dat", "model", "openai", 3), {"ok": True})
        gen_mock.assert_called_once_with("p", "m.mod", "d.dat", iterations=3, llm_model="model", provider="openai")

        with patch.object(_rhetor_mcp, "_ask_for_feedback", return_value="fb") as feedback_mock:
            self.assertEqual(_rhetor_mcp.ask_tool("p", "m.mod", "d.dat", "model", "openai"), "fb")
        feedback_mock.assert_called_once_with("p", "m.mod", "d.dat", llm_model="model", provider="openai")

        with patch.object(rhetor_mcp.mcp, "run") as run_mock:
            rhetor_mcp.main()
        run_mock.assert_called_once_with()

    def test_insight_tool_success_and_error_paths(self):
        with (
            patch.object(_rhetor_mcp, "_generate_with_best_available_backend", return_value={"generated": True}) as gen_mock,
            patch.object(_rhetor_mcp, "solve", return_value={"status": "OPTIMAL", "objective_value": 1}) as solve_mock,
            patch.object(_rhetor_mcp, "_ask_for_feedback", return_value={"summary": "Looks good"}) as feedback_mock,
        ):
            result = _rhetor_mcp.insight_tool("Make a model", provider="openai", llm_model="m", iterations=2, solver="highs")

        self.assertTrue(Path(result["model_path"]).parent.exists())
        self.assertTrue(result["model_path"].endswith(".mod"))
        self.assertEqual(result["stats"], {"generated": True})
        self.assertEqual(result["results"]["status"], "OPTIMAL")
        self.assertEqual(result["feedback"], {"summary": "Looks good"})
        self.assertIn("Looks good", result["markdown"])
        gen_mock.assert_called_once()
        solve_mock.assert_called_once()
        feedback_mock.assert_called_once()

        with (
            patch.object(_rhetor_mcp, "_generate_with_best_available_backend", return_value={"generated": True}),
            patch.object(_rhetor_mcp, "solve", side_effect=RuntimeError("solve failed")),
            patch.object(_rhetor_mcp, "_ask_for_feedback", side_effect=RuntimeError("feedback failed")),
        ):
            result = _rhetor_mcp.insight_tool("Make a model")

        self.assertIn("Error solving generated model", result["results"]["error"])
        self.assertIn("Error generating feedback", result["feedback"]["error"])
        self.assertIn("feedback", result["markdown"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from pyopl import pyopl_cli


class TestCLI(unittest.TestCase):
    def test_cli_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            pyopl_cli.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_cli_batch_help_describes_recursive_archive_contents(self):
        out_buf = io.StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stdout(out_buf):
            pyopl_cli.main(["batch-solve", "--help"])

        self.assertEqual(cm.exception.code, 0)
        help_text = " ".join(out_buf.getvalue().split())
        self.assertIn("every archive folder containing exactly one .mod file and one or more .dat files", help_text)
        self.assertIn("Folders may be nested", help_text)
        self.assertIn("processing continues when an instance fails", help_text)
        self.assertIn("highs.json or gurobi.json", help_text)

    def test_cli_batch_compare_help_describes_archive_contents(self):
        out_buf = io.StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stdout(out_buf):
            pyopl_cli.main(["batch-compare", "--help"])

        self.assertEqual(cm.exception.code, 0)
        help_text = " ".join(out_buf.getvalue().split())
        self.assertIn("exactly one .mod file and one or more .dat files", help_text)
        self.assertIn("--strategy {concrete,abstract}", help_text)

    def test_cli_solve_lot_sizing_highs_json(self):
        model = Path("pyopl/opl_models/lot_sizing/lot_sizing.mod")
        data = Path("pyopl/opl_models/lot_sizing/lot_sizing.dat")
        self.assertTrue(model.exists(), f"Model not found: {model}")
        self.assertTrue(data.exists(), f"Data not found: {data}")

        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = pyopl_cli.main(
                [
                    "solve",
                    str(model),
                    str(data),
                    "--solver",
                    "highs",
                    "--out",
                    "json",
                ]
            )

        self.assertEqual(ret, 0)
        out = buf.getvalue().strip()
        self.assertTrue(out, "No output produced")
        self.assertTrue(out.startswith("{"), out[:200])
        self.assertNotIn("Running HiGHS", out)
        self.assertNotIn("PyOPL/SciPy-HiGHS", out)
        payload = json.loads(out)
        self.assertTrue("status" in payload or "objective_value" in payload)

    def test_cli_solve_missing_model(self):
        argv = ["solve", "does_not_exist.mod"]
        err_buf = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err_buf):
            ret = pyopl_cli.main(argv)
        self.assertNotEqual(ret, 0)
        self.assertIn("model file not found", err_buf.getvalue())

    def test_cli_batch_solve(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive = Path(tmp_dir) / "batch.zip"
            with zipfile.ZipFile(archive, "w") as batch_archive:
                batch_archive.writestr("model.mod", "model")
                batch_archive.writestr("data.dat", "data")

            with patch("pyopl.pyopl_cli.batch_solve", return_value={}) as batch_mock:
                ret = pyopl_cli.main(["batch-solve", str(archive), "--solver", "gurobi"])

        self.assertEqual(ret, 0)
        batch_mock.assert_called_once_with(str(archive), solver="gurobi")

    def test_cli_batch_solve_progress(self):
        with patch("pyopl.pyopl_cli.batch_solve_with_progress", return_value={}) as progress_mock:
            ret = pyopl_cli.main(["batch-solve", "batch.zip", "--progress"])

        self.assertEqual(ret, 0)
        progress_mock.assert_called_once_with("batch.zip", solver="highs")

    def test_cli_batch_solve_returns_nonzero_for_failed_instance(self):
        for status in ("ERROR", "FAILED", "EXECUTION_ERROR"):
            with self.subTest(status=status):
                report = {"instances": [{"data": "data.dat", "status": status}]}
                with patch("pyopl.pyopl_cli.batch_solve", return_value=report):
                    ret = pyopl_cli.main(["batch-solve", "batch.zip"])

                self.assertNotEqual(ret, 0)

    def test_cli_batch_compare(self):
        report = {"instances": [{"data": "data.dat", "status": "different", "equivalent": False}]}
        with patch("pyopl.pyopl_cli.batch_compare", return_value=report) as batch_mock:
            ret = pyopl_cli.main(["batch-compare", "left.zip", "right.zip", "--strategy", "concrete"])

        self.assertEqual(ret, 0)
        batch_mock.assert_called_once_with("left.zip", "right.zip", strategy="concrete")

    def test_cli_batch_compare_returns_nonzero_for_error(self):
        report = {"instances": [{"data": "data.dat", "status": "ERROR", "equivalent": None}]}
        with patch("pyopl.pyopl_cli.batch_compare", return_value=report):
            ret = pyopl_cli.main(["batch-compare", "left.zip", "right.zip"])

        self.assertNotEqual(ret, 0)

    def test_cli_batch_handlers_report_exceptions(self):
        err_buf = io.StringIO()
        with patch("pyopl.pyopl_cli.batch_solve", side_effect=RuntimeError("archive failed")), redirect_stderr(err_buf):
            ret = pyopl_cli.main(["batch-solve", "batch.zip"])

        self.assertEqual(ret, 1)
        self.assertIn("Error during batch solve: archive failed", err_buf.getvalue())

        err_buf = io.StringIO()
        with patch("pyopl.pyopl_cli.batch_compare", side_effect=RuntimeError("comparison failed")), redirect_stderr(err_buf):
            ret = pyopl_cli.main(["batch-compare", "left.zip", "right.zip"])

        self.assertEqual(ret, 1)
        self.assertIn("Error during batch comparison: comparison failed", err_buf.getvalue())

    def test_cli_loads_solver_settings_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = Path(tmp_dir) / "model.mod"
            settings = Path(tmp_dir) / "settings.json"
            model.write_text("dvar float+ x; minimize x; subject to { x >= 1; }", encoding="utf-8")
            settings.write_text('{"time_limit": 2.5}', encoding="utf-8")

            with patch("pyopl.pyopl_cli._run_solve", return_value={"status": "OPTIMAL"}) as solve_mock:
                ret = pyopl_cli.main(["solve", str(model), "--solver", "highs", "--solver-settings", str(settings)])

        self.assertEqual(ret, 0)
        solve_mock.assert_called_once_with(model, None, "scipy", {"time_limit": 2.5})

    def test_cli_solver_settings_loader_rejects_missing_and_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.json"
            with redirect_stderr(io.StringIO()), self.assertRaises(FileNotFoundError):
                pyopl_cli._load_solver_settings(str(missing))

            settings = Path(tmp_dir) / "settings.json"
            settings.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain an object"):
                pyopl_cli._load_solver_settings(str(settings))

    def test_cli_solver_validation_and_error_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = Path(tmp_dir) / "model.mod"
            model.write_text("model", encoding="utf-8")

            err_buf = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err_buf):
                ret = pyopl_cli.main(["solve", str(model), str(Path(tmp_dir) / "missing.dat")])
            self.assertEqual(ret, 2)
            self.assertIn("data file not found", err_buf.getvalue())

            err_buf = io.StringIO()
            with (
                patch("pyopl.pyopl_cli._run_solve", side_effect=RuntimeError("solver failed")),
                redirect_stdout(io.StringIO()),
                redirect_stderr(err_buf),
            ):
                ret = pyopl_cli.main(["solve", str(model)])
            self.assertEqual(ret, 1)
            self.assertIn("Error during solve/export: solver failed", err_buf.getvalue())

            out_buf = io.StringIO()
            with patch("pyopl.pyopl_cli._export_py", return_value="compiled code"), redirect_stdout(out_buf):
                ret = pyopl_cli.main(["solve", str(model), "--out", "py"])
            self.assertEqual(ret, 0)
            self.assertEqual(out_buf.getvalue().strip(), "compiled code")

            err_buf = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err_buf):
                ret = pyopl_cli.main(["solve", str(model), "--out", "lp", "--out-file", str(Path(tmp_dir) / "model.txt")])
            self.assertEqual(ret, 2)
            self.assertIn("ending in .lp", err_buf.getvalue())

    def test_run_solve_propagates_backend_errors(self):
        with patch("pyopl.pyopl_cli.solve", side_effect=RuntimeError("backend failed")):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                pyopl_cli._run_solve(Path("model.mod"), None, "scipy")

    def test_cli_compare_models_json(self):
        result = {
            "status": "equivalent",
            "equivalent": True,
            "level": "solver_implied",
            "reason": "same normalized model",
            "proof_steps": ["normalized both models"],
            "counterexample": None,
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            left_data = td_path / "left.dat"
            right_data = td_path / "right.dat"
            left_model.write_text("dvar float+ x; minimize x; subject to { x >= 1; }", encoding="utf-8")
            right_model.write_text("dvar float+ y; minimize y; subject to { y >= 1; }", encoding="utf-8")
            left_data.write_text("", encoding="utf-8")
            right_data.write_text("", encoding="utf-8")

            with patch("pyopl.pyopl_cli._compare_models", return_value=result) as compare_mock:
                buf = io.StringIO()
                argv = [
                    "compare",
                    str(left_model),
                    str(right_model),
                    "--left-data",
                    str(left_data),
                    "--right-data",
                    str(right_data),
                ]
                with redirect_stdout(buf):
                    ret = pyopl_cli.main(argv)

            self.assertEqual(ret, 0)
            self.assertEqual(json.loads(buf.getvalue()), result)
            compare_mock.assert_called_once_with(left_model, right_model, left_data, right_data, "abstract")

    def test_cli_compare_passes_manual_observable_mapping(self):
        result = {"status": "equivalent", "equivalent": True}
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            mapping = td_path / "mapping.json"
            left_model.write_text("left", encoding="utf-8")
            right_model.write_text("right", encoding="utf-8")
            mapping.write_text("[]", encoding="utf-8")

            with patch("pyopl.pyopl_cli._compare_models", return_value=result) as compare_mock:
                ret = pyopl_cli.main(
                    [
                        "compare",
                        str(left_model),
                        str(right_model),
                        "--observable-mapping",
                        str(mapping),
                    ]
                )

        self.assertEqual(ret, 0)
        self.assertEqual(compare_mock.call_args.args[-1], "[]")

    def test_cli_compare_dispatches_uniform_indexed_mapping_envelope(self):
        envelope = {
            "schema": "pyopl.uniform_indexed_observable_mapping",
            "version": 1,
            "observables": [
                {
                    "kind": "indexed",
                    "name": "x[{t}]",
                    "left": {"expression": "x[t]", "domains": [{"name": "t", "range": "1..N"}]},
                    "right": {"expression": "x[t]", "domains": [{"name": "t", "range": "1..N"}]},
                }
            ],
            "parameter_contract": {"facts": [], "indexed_facts": []},
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            mapping = td_path / "mapping.json"
            left_model.write_text("left", encoding="utf-8")
            right_model.write_text("right", encoding="utf-8")
            mapping.write_text(json.dumps(envelope), encoding="utf-8")

            with (
                patch("pyopl.pyopl_cli.compare_models", return_value=object()) as compare_mock,
                patch("pyopl.pyopl_cli.comparison_result_to_dict", return_value={"status": "unknown"}),
            ):
                result = pyopl_cli._compare_models(left_model, right_model, None, None, "abstract", json.dumps(envelope))

        self.assertEqual(result, {"status": "unknown"})
        self.assertIn("indexed_observable_mapping", compare_mock.call_args.kwargs)
        self.assertIn("parameter_contract", compare_mock.call_args.kwargs)

    def test_cli_compare_models_uses_grounded_mapping_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            left_model = Path(td) / "left.mod"
            right_model = Path(td) / "right.mod"
            left_model.write_text("left model", encoding="utf-8")
            right_model.write_text("right model", encoding="utf-8")

            with (
                patch("pyopl.pyopl_cli.compare_models", return_value=object()) as compare_mock,
                patch("pyopl.pyopl_cli.comparison_result_to_dict", return_value={"status": "equivalent"}),
            ):
                result = pyopl_cli._compare_models(left_model, right_model, None, None, "abstract", "[]")

            self.assertEqual(result, {"status": "equivalent"})
            compare_mock.assert_called_once_with(
                "left model",
                "right model",
                strategy="abstract",
                left_data_text=None,
                right_data_text=None,
                observable_mapping=(),
            )

            with self.assertRaisesRegex(ValueError, "invalid observable mapping JSON"):
                pyopl_cli._compare_models(left_model, right_model, None, None, "abstract", "{")

    def test_cli_compare_rejects_invalid_mapping_options(self):
        with tempfile.TemporaryDirectory() as td:
            left_model = Path(td) / "left.mod"
            right_model = Path(td) / "right.mod"
            left_model.write_text("left", encoding="utf-8")
            right_model.write_text("right", encoding="utf-8")
            missing_mapping = Path(td) / "missing.json"

            invalid_requests = (
                [
                    "compare",
                    str(left_model),
                    str(right_model),
                    "--observable-mapping",
                    str(missing_mapping),
                    "--generate-mapping",
                ],
                [
                    "compare",
                    str(left_model),
                    str(right_model),
                    "--mapping-out-file",
                    str(Path(td) / "generated.json"),
                ],
                [
                    "compare",
                    str(left_model),
                    str(right_model),
                    "--observable-mapping",
                    str(missing_mapping),
                ],
            )
            for argv in invalid_requests:
                with self.subTest(argv=argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(pyopl_cli.main(argv), 2)

    def test_cli_compare_generates_and_can_save_observable_mapping(self):
        result = {"status": "equivalent", "equivalent": True}
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            mapping = td_path / "generated.json"
            left_model.write_text("left model", encoding="utf-8")
            right_model.write_text("right model", encoding="utf-8")

            with (
                patch(
                    "pyopl.pyopl_cli.generative_indexed_observable_mapping", return_value='[{"name": "x"}]'
                ) as generate_mock,
                patch("pyopl.pyopl_cli._compare_models", return_value=result) as compare_mock,
            ):
                ret = pyopl_cli.main(
                    [
                        "compare",
                        str(left_model),
                        str(right_model),
                        "--generate-mapping",
                        "--mapping-out-file",
                        str(mapping),
                    ]
                )
            mapping_text = mapping.read_text(encoding="utf-8")

        self.assertEqual(ret, 0)
        self.assertEqual(mapping_text, '[{"name": "x"}]')
        self.assertEqual(generate_mock.call_args.args[:2], ("left model", "right model"))
        self.assertEqual(compare_mock.call_args.args[-1], '[{"name": "x"}]')

    def test_cli_genai_generate_mapping_emits_json(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            left_model.write_text("left", encoding="utf-8")
            right_model.write_text("right", encoding="utf-8")
            with patch("pyopl.pyopl_cli.generative_observable_mapping", return_value="[]") as generate_mock:
                output = io.StringIO()
                with redirect_stdout(output):
                    ret = pyopl_cli.main(["genai", "generate-mapping", str(left_model), str(right_model)])

        self.assertEqual(ret, 0)
        self.assertEqual(output.getvalue().strip(), "[]")
        generate_mock.assert_called_once()

    def test_cli_compare_selects_abstract_strategy(self):
        result = {
            "strategy": "abstract",
            "status": "equivalent",
            "equivalent": True,
            "level": "schema_isomorphic",
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            left_model.write_text("dvar float x; minimize x; subject to { x >= 0; }", encoding="utf-8")
            right_model.write_text("dvar float y; minimize y; subject to { y >= 0; }", encoding="utf-8")

            with patch("pyopl.pyopl_cli._compare_models", return_value=result) as compare_mock:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    ret = pyopl_cli.main(["compare", str(left_model), str(right_model), "--strategy", "abstract"])

            self.assertEqual(ret, 0)
            self.assertEqual(json.loads(buf.getvalue()), result)
            compare_mock.assert_called_once_with(left_model, right_model, None, None, "abstract")

    def test_cli_compare_models_outfile(self):
        result = {"status": "different", "equivalent": False, "level": "solver_implied"}
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            out_file = td_path / "compare.json"
            left_model.write_text("dvar float+ x; minimize x; subject to { x >= 1; }", encoding="utf-8")
            right_model.write_text("dvar float+ x; minimize 2 * x; subject to { x >= 1; }", encoding="utf-8")

            with patch("pyopl.pyopl_cli._compare_models", return_value=result):
                ret = pyopl_cli.main(["compare", str(left_model), str(right_model), "--out-file", str(out_file)])

            self.assertEqual(ret, 0)
            self.assertEqual(json.loads(out_file.read_text(encoding="utf-8")), result)

    def test_cli_compare_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            right_model = Path(td) / "right.mod"
            right_model.write_text("dvar float+ x; minimize x; subject to { x >= 1; }", encoding="utf-8")
            err_buf = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err_buf):
                ret = pyopl_cli.main(["compare", "does_not_exist.mod", str(right_model)])

        self.assertEqual(ret, 2)
        self.assertIn("left model file not found", err_buf.getvalue())

    def test_cli_compare_error(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            left_model = td_path / "left.mod"
            right_model = td_path / "right.mod"
            left_model.write_text("dvar float+ x; minimize x; subject to { x >= 1; }", encoding="utf-8")
            right_model.write_text("dvar float+ x; minimize x; subject to { x >= 1; }", encoding="utf-8")

            err_buf = io.StringIO()
            with (
                patch("pyopl.pyopl_cli._compare_models", side_effect=ValueError("bad model")),
                redirect_stdout(io.StringIO()),
                redirect_stderr(err_buf),
            ):
                ret = pyopl_cli.main(["compare", str(left_model), str(right_model)])

        self.assertEqual(ret, 1)
        self.assertIn("Error during compare: bad model", err_buf.getvalue())

    def test_genai_list_models_openai(self):
        argv = ["genai", "list-models", "openai"]
        fake_models = ["gpt-test-1", "gpt-test-2"]
        with patch("pyopl.pyopl_cli.list_openai_models", return_value=fake_models):
            buf = io.StringIO()
            with redirect_stdout(buf):
                ret = pyopl_cli.main(argv)
            self.assertEqual(ret, 0)
            out = buf.getvalue()
            self.assertIn("gpt-test-1", out)

    def test_genai_list_models_elm(self):
        with patch("pyopl.pyopl_cli.list_elm_models", return_value=["elm-model"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                ret = pyopl_cli.main(["genai", "list-models", "elm"])

        self.assertEqual(ret, 0)
        self.assertIn("elm-model", buf.getvalue())

    def test_genai_list_models_reports_provider_errors(self):
        err_buf = io.StringIO()
        with patch("pyopl.pyopl_cli.list_openai_models", side_effect=RuntimeError("offline")), redirect_stderr(err_buf):
            ret = pyopl_cli.main(["genai", "list-models", "openai", "--prefix", "gpt"])

        self.assertEqual(ret, 3)
        self.assertIn("Error listing models for openai: offline", err_buf.getvalue())

    def test_genai_list_methods(self):
        argv = ["genai", "list-methods"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = pyopl_cli.main(argv)
        self.assertEqual(ret, 0)
        out = buf.getvalue()
        self.assertIn("SyntAGM", out)

    def test_genai_generate_and_ask(self):
        gen_stats = {"status": "ok", "iterations": 1}
        feedback = {"feedback": "looks good"}
        with patch("pyopl.pyopl_cli.generative_solve", return_value=gen_stats):
            buf = io.StringIO()
            argv = [
                "genai",
                "generate",
                "Create a small model",
                "--model-file",
                "out.mod",
                "--data-file",
                "out.dat",
            ]
            with redirect_stdout(buf):
                ret = pyopl_cli.main(argv)
            self.assertEqual(ret, 0)
            out = buf.getvalue()
            self.assertIn("iterations", out)

        with patch("pyopl.pyopl_cli.generative_feedback", return_value=feedback):
            buf = io.StringIO()
            argv = [
                "genai",
                "ask",
                "Is this model OK?",
                "--model-file",
                "out.mod",
                "--data-file",
                "out.dat",
            ]
            with redirect_stdout(buf):
                ret = pyopl_cli.main(argv)
            self.assertEqual(ret, 0)
            out = buf.getvalue()
            self.assertIn("feedback", out)

    def test_genai_generate_and_ask_report_backend_errors(self):
        err_buf = io.StringIO()
        with (
            patch("pyopl.pyopl_cli.generative_solve", side_effect=RuntimeError("generation failed")) as generate_mock,
            redirect_stderr(err_buf),
        ):
            ret = pyopl_cli.main(
                [
                    "genai",
                    "generate",
                    "prompt",
                    "--model-file",
                    "model.mod",
                    "--data-file",
                    "data.dat",
                    "--llm-model",
                    "local-model",
                    "--provider",
                    "ollama",
                ]
            )

        self.assertEqual(ret, 4)
        self.assertIn("Error during generative_solve: generation failed", err_buf.getvalue())
        self.assertEqual(
            generate_mock.call_args.kwargs,
            {
                "iterations": 5,
                "return_statistics": True,
                "model_name": "local-model",
                "llm_provider": "ollama",
            },
        )

        err_buf = io.StringIO()
        with (
            patch("pyopl.pyopl_cli.generative_feedback", side_effect=RuntimeError("feedback failed")) as feedback_mock,
            redirect_stderr(err_buf),
        ):
            ret = pyopl_cli.main(
                [
                    "genai",
                    "ask",
                    "prompt",
                    "--model-file",
                    "model.mod",
                    "--data-file",
                    "data.dat",
                    "--llm-model",
                    "local-model",
                    "--provider",
                    "ollama",
                ]
            )

        self.assertEqual(ret, 4)
        self.assertIn("Error during generative_feedback: feedback failed", err_buf.getvalue())
        self.assertEqual(
            feedback_mock.call_args.kwargs,
            {"model_name": "local-model", "llm_provider": "ollama"},
        )

    def test_genai_generate_mapping_handles_missing_inputs_and_errors(self):
        with tempfile.TemporaryDirectory() as td:
            left_model = Path(td) / "left.mod"
            right_model = Path(td) / "right.mod"
            right_model.write_text("right", encoding="utf-8")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    pyopl_cli.main(["genai", "generate-mapping", str(left_model), str(right_model)]),
                    2,
                )

            left_model.write_text("left", encoding="utf-8")
            err_buf = io.StringIO()
            with (
                patch(
                    "pyopl.pyopl_cli.generative_observable_mapping",
                    side_effect=RuntimeError("mapping failed"),
                ) as generate_mock,
                redirect_stderr(err_buf),
            ):
                ret = pyopl_cli.main(
                    [
                        "genai",
                        "generate-mapping",
                        str(left_model),
                        str(right_model),
                        "--llm-model",
                        "local-model",
                        "--provider",
                        "ollama",
                    ]
                )

        self.assertEqual(ret, 4)
        self.assertIn("Error during observable mapping generation: mapping failed", err_buf.getvalue())
        self.assertEqual(generate_mock.call_args.kwargs, {"model_name": "local-model", "llm_provider": "ollama"})

    def test_cli_outfile_json_and_py(self):
        model = Path("pyopl/opl_models/lot_sizing/lot_sizing.mod")
        data = Path("pyopl/opl_models/lot_sizing/lot_sizing.dat")
        self.assertTrue(model.exists(), f"Model not found: {model}")
        self.assertTrue(data.exists(), f"Data not found: {data}")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_json = td_path / "out.json"
            out_py = td_path / "out.py"

            # JSON output to file
            argv = ["solve", str(model), str(data), "--solver", "highs", "--out", "json", "--out-file", str(out_json)]
            ret = pyopl_cli.main(argv)
            self.assertEqual(ret, 0)
            self.assertTrue(out_json.exists())
            txt = out_json.read_text(encoding="utf-8")
            self.assertTrue(txt.strip())
            self.assertIn("status", txt) or self.assertIn("objective_value", txt)

            # py export to file
            argv = ["solve", str(model), str(data), "--solver", "highs", "--out", "py", "--out-file", str(out_py)]
            ret = pyopl_cli.main(argv)
            self.assertEqual(ret, 0)
            self.assertTrue(out_py.exists())
            code = out_py.read_text(encoding="utf-8")
            self.assertTrue(len(code) > 10)

    def test_cli_outfile_lp_and_mps(self):
        model_text = "dvar float+ x; minimize 2 * x + 3; subject to { x >= 1; x <= 4; }"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model = td_path / "model.mod"
            out_lp = td_path / "model.lp"
            out_mps = td_path / "model.mps"
            model.write_text(model_text, encoding="utf-8")

            ret = pyopl_cli.main(["solve", str(model), "--out", "lp", "--out-file", str(out_lp)])
            self.assertEqual(ret, 0)
            self.assertTrue(out_lp.exists())
            lp_text = out_lp.read_text(encoding="utf-8")
            self.assertIn("obj:", lp_text)

            ret = pyopl_cli.main(["solve", str(model), "--out", "mps", "--out-file", str(out_mps)])
            self.assertEqual(ret, 0)
            self.assertTrue(out_mps.exists())
            self.assertGreater(out_mps.stat().st_size, 0)

    def test_cli_lp_mps_requires_out_file(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model.mod"
            model.write_text("dvar float+ x; minimize x; subject to { x >= 1; }", encoding="utf-8")
            err_buf = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err_buf):
                ret = pyopl_cli.main(["solve", str(model), "--out", "lp"])
            self.assertEqual(ret, 2)
            self.assertIn("requires --out-file", err_buf.getvalue())

    def test_genai_generate_outfile(self):
        gen_stats = {"status": "ok", "iterations": 1}
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model_file = td_path / "gen.mod"
            data_file = td_path / "gen.dat"
            out_file = td_path / "gen_stats.json"

            with patch("pyopl.pyopl_cli.generative_solve", return_value=gen_stats):
                argv = [
                    "genai",
                    "generate",
                    "Create a small model",
                    "--model-file",
                    str(model_file),
                    "--data-file",
                    str(data_file),
                    "--out-file",
                    str(out_file),
                ]
                ret = pyopl_cli.main(argv)
                self.assertEqual(ret, 0)
                self.assertTrue(out_file.exists())
                txt = out_file.read_text(encoding="utf-8")
                self.assertIn("iterations", txt)

    def test_genai_insight_pipeline(self):
        # Mock generation, solving, and feedback; verify markdown output to file
        gen_stats = {"status": "generated"}
        solve_res = {"status": "OPTIMAL", "objective_value": 42, "solution": {"x": 1}}
        feedback = {"feedback": "The solver found an optimal solution with objective 42. Recommend increasing capacity."}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_md = td_path / "insight.md"

            with (
                patch("pyopl.pyopl_cli.generative_solve", return_value=gen_stats),
                patch("pyopl.pyopl_cli._run_solve", return_value=solve_res),
                patch("pyopl.pyopl_cli.generative_feedback", return_value=feedback),
            ):

                argv = [
                    "genai",
                    "insight",
                    "Analyze the best production plan",
                    "--out-file",
                    str(out_md),
                ]
                ret = pyopl_cli.main(argv)
                self.assertEqual(ret, 0)
                self.assertTrue(out_md.exists())
                mdtxt = out_md.read_text(encoding="utf-8")
                self.assertIn("GenAI Insight", mdtxt)
                self.assertIn("optimal", mdtxt.lower())

    def test_genai_insight_reports_generation_solve_and_feedback_errors(self):
        with tempfile.TemporaryDirectory() as td, patch("pyopl.pyopl_cli.os.getcwd", return_value=td):
            err_buf = io.StringIO()
            with (
                patch("pyopl.pyopl_cli.generative_solve", side_effect=RuntimeError("generation failed")),
                redirect_stderr(err_buf),
            ):
                ret = pyopl_cli.main(["genai", "insight", "prompt"])
            self.assertEqual(ret, 4)
            self.assertIn("Error during generation: generation failed", err_buf.getvalue())

            err_buf = io.StringIO()
            with (
                patch("pyopl.pyopl_cli.generative_solve", return_value={}),
                patch("pyopl.pyopl_cli._run_solve", side_effect=RuntimeError("solve failed")),
                redirect_stderr(err_buf),
            ):
                ret = pyopl_cli.main(["genai", "insight", "prompt"])
            self.assertEqual(ret, 1)
            self.assertIn("Error solving generated model: solve failed", err_buf.getvalue())

            err_buf = io.StringIO()
            with (
                patch("pyopl.pyopl_cli.generative_solve", return_value={}),
                patch("pyopl.pyopl_cli._run_solve", return_value={"status": "OPTIMAL"}),
                patch("pyopl.pyopl_cli.generative_feedback", side_effect=RuntimeError("feedback failed")),
                redirect_stderr(err_buf),
            ):
                ret = pyopl_cli.main(["genai", "insight", "prompt"])
            self.assertEqual(ret, 4)
            self.assertIn("Error during feedback/translation: feedback failed", err_buf.getvalue())

    def test_insight_helpers_summarize_feedback_and_avoid_path_collisions(self):
        self.assertEqual(pyopl_cli._insight_summary({"feedback": "feedback"}), "feedback")
        self.assertEqual(pyopl_cli._insight_summary({"summary": "summary"}), "summary")
        self.assertEqual(pyopl_cli._insight_summary({"details": "other"}), '{\n  "details": "other"\n}')
        self.assertEqual(pyopl_cli._insight_summary("plain feedback"), "plain feedback")

        fixed_datetime = Mock()
        fixed_datetime.now.return_value.strftime.return_value = "2026-01-02_03-04-05"
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td) / "tmp"
            tmp_dir.mkdir()
            (tmp_dir / "gen_pyopl_2026-01-02_03-04-05.mod").touch()

            with patch.object(pyopl_cli, "datetime", fixed_datetime), patch("pyopl.pyopl_cli.os.getcwd", return_value=td):
                model_path, data_path = pyopl_cli._unique_insight_paths()

        self.assertEqual(model_path.name, "gen_pyopl_2026-01-02_03-04-05_1.mod")
        self.assertEqual(data_path.name, "gen_pyopl_2026-01-02_03-04-05_1.dat")

    def test_cli_dispatches_ide_and_reports_unknown_genai_command(self):
        with patch("pyopl.pyopl_cli.OPLIDE") as ide_cls:
            self.assertEqual(pyopl_cli.main([]), 0)
            self.assertEqual(pyopl_cli.main(["ide", "--debug"]), 0)

        self.assertEqual(ide_cls.call_args_list[0].kwargs, {"debug": False})
        self.assertEqual(ide_cls.call_args_list[1].kwargs, {"debug": True})
        self.assertEqual(ide_cls.return_value.mainloop.call_count, 2)

        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            self.assertEqual(pyopl_cli.main(["genai"]), 2)
        self.assertIn("Unknown command", err_buf.getvalue())


if __name__ == "__main__":
    unittest.main()

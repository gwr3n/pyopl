import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pyopl import pyopl_cli


class TestCLI(unittest.TestCase):
    def test_cli_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            pyopl_cli.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

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


if __name__ == "__main__":
    unittest.main()

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

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
            ret = pyopl_cli.main([
                "--solve",
                str(model),
                str(data),
                "--highs",
                "--out",
                "json",
            ])

        self.assertEqual(ret, 0)
        out = buf.getvalue().strip()
        self.assertTrue(out, "No output produced")
        payload = json.loads(out)
        self.assertTrue("status" in payload or "objective_value" in payload)


if __name__ == "__main__":
    unittest.main()

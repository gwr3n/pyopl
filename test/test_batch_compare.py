import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pyopl.batch_compare import batch_compare


def _write_archive(path, model_name, data):
    with zipfile.ZipFile(path, "w") as batch_archive:
        batch_archive.writestr(model_name, "model")
        for data_name, contents in data.items():
            batch_archive.writestr(data_name, contents)


class TestBatchCompare(unittest.TestCase):
    def test_compares_matching_data_basenames_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = root / "left.zip"
            right = root / "right.zip"
            _write_archive(left, "left/model.mod", {"left/a.dat": "left a", "left/only.dat": "left only"})
            _write_archive(right, "model.mod", {"a.dat": "right a", "right/extra.dat": "right only"})
            result = SimpleNamespace(
                status="equivalent",
                equivalent=True,
                level="schema_isomorphic",
                reason="same model",
                proof_steps=("matched",),
                counterexample=None,
            )

            with patch("pyopl.batch_compare.compare_models", return_value=result) as compare_mock:
                report = batch_compare(left, right, strategy="concrete")

            self.assertEqual([instance["data"] for instance in report["instances"]], ["a.dat"])
            compare_mock.assert_called_once_with(
                "model",
                "model",
                strategy="concrete",
                left_data_text="left a",
                right_data_text="right a",
            )
            payload = json.loads((root / "left_vs_right.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["instances"][0]["equivalent"])
            markdown = (root / "left_vs_right.md").read_text(encoding="utf-8")
            self.assertIn("| a.dat | equivalent | True | schema_isomorphic |", markdown)

    def test_records_comparison_errors_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = root / "left.zip"
            right = root / "right.zip"
            _write_archive(left, "model.mod", {"a.dat": "a", "b.dat": "b"})
            _write_archive(right, "model.mod", {"a.dat": "a", "b.dat": "b"})

            with patch(
                "pyopl.batch_compare.compare_models",
                side_effect=[
                    ValueError("invalid"),
                    SimpleNamespace(
                        status="different",
                        equivalent=False,
                        level="normalized",
                        reason="different",
                        proof_steps=(),
                        counterexample=None,
                    ),
                ],
            ):
                report = batch_compare(left, right)

            self.assertEqual([instance["status"] for instance in report["instances"]], ["ERROR", "different"])

    def test_rejects_archives_without_matching_data_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = root / "left.zip"
            right = root / "right.zip"
            _write_archive(left, "model.mod", {"a.dat": "a"})
            _write_archive(right, "model.mod", {"b.dat": "b"})

            with self.assertRaisesRegex(ValueError, "matching .dat filenames"):
                batch_compare(left, right)

    def test_rejects_duplicate_data_basenames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = root / "left.zip"
            right = root / "right.zip"
            _write_archive(left, "model.mod", {"one/a.dat": "a", "two/a.dat": "a"})
            _write_archive(right, "model.mod", {"a.dat": "a"})

            with self.assertRaisesRegex(ValueError, "duplicate data filename"):
                batch_compare(left, right)

    def test_requires_zip_extensions(self):
        with self.assertRaisesRegex(ValueError, r"\.zip extension"):
            batch_compare("left.json", "right.zip")


if __name__ == "__main__":
    unittest.main()

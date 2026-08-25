"""Compare matching model instances contained in two PyOPL batch archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .model_equivalence import compare_models, comparison_result_to_dict


def _is_metadata_member(member: Path) -> bool:
    return any(part.startswith("._") or part in {".DS_Store", "Thumbs.db"} for part in member.parts)


def _archive_members(batch_archive: zipfile.ZipFile) -> list[Path]:
    return [
        Path(info.filename)
        for info in batch_archive.infolist()
        if not info.is_dir() and not _is_metadata_member(Path(info.filename))
    ]


def _read_archive(archive: Path) -> tuple[Path, str, dict[str, tuple[Path, str]]]:
    if archive.suffix.lower() != ".zip":
        raise ValueError("Batch archive must have a .zip extension")
    if not archive.is_file():
        raise FileNotFoundError(f"Batch archive not found: {archive}")

    with zipfile.ZipFile(archive) as batch_archive:
        members = _archive_members(batch_archive)
        models = [member for member in members if member.suffix.lower() == ".mod"]
        if len(models) != 1:
            raise ValueError(f"Batch archive '{archive}' must contain exactly one .mod file; found {len(models)}")
        data_files = [member for member in members if member.suffix.lower() == ".dat"]
        if not data_files:
            raise ValueError(f"Batch archive '{archive}' must contain at least one .dat file")

        data_by_name: dict[str, tuple[Path, str]] = {}
        for data_file in data_files:
            key = data_file.name.lower()
            if key in data_by_name:
                raise ValueError(f"Batch archive '{archive}' contains duplicate data filename '{data_file.name}'")
            data_by_name[key] = (data_file, batch_archive.read(data_file.as_posix()).decode("utf-8"))
        model = models[0]
        return model, batch_archive.read(model.as_posix()).decode("utf-8"), data_by_name


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def _make_markdown(records: list[dict[str, Any]]) -> str:
    columns = ["data", "status", "equivalent", "level", "reason", "proof_steps", "counterexample"]
    lines = [
        "# Batch comparison results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(_markdown_value(record.get(column)) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def _compare_data_pair(
    left_model_text: str,
    right_model_text: str,
    left_data: tuple[Path, str],
    right_data: tuple[Path, str],
    strategy: str,
) -> dict[str, Any]:
    data_name = left_data[0].name
    try:
        result = compare_models(
            left_model_text,
            right_model_text,
            strategy=strategy,
            left_data_text=left_data[1],
            right_data_text=right_data[1],
        )
        return {
            "data": data_name,
            "left_data": left_data[0].as_posix(),
            "right_data": right_data[0].as_posix(),
            **comparison_result_to_dict(result, strategy=strategy),
        }
    except Exception as error:
        return {
            "data": data_name,
            "left_data": left_data[0].as_posix(),
            "right_data": right_data[0].as_posix(),
            "strategy": strategy,
            "status": "ERROR",
            "equivalent": None,
            "level": None,
            "reason": str(error),
            "proof_steps": [],
            "counterexample": None,
        }


def batch_compare(
    left_zip_path: str | Path,
    right_zip_path: str | Path,
    strategy: str = "abstract",
) -> dict[str, Any]:
    """Compare model instances whose data filenames occur in both archives."""
    if strategy not in {"abstract", "concrete"}:
        raise ValueError("Unsupported comparison strategy; choose abstract or concrete")

    left_archive = Path(left_zip_path)
    right_archive = Path(right_zip_path)
    left_model, left_model_text, left_data = _read_archive(left_archive)
    right_model, right_model_text, right_data = _read_archive(right_archive)
    matching_names = sorted(left_data.keys() & right_data.keys())
    if not matching_names:
        raise ValueError("Batch archives do not contain any matching .dat filenames")

    records = [
        _compare_data_pair(
            left_model_text,
            right_model_text,
            left_data[data_name],
            right_data[data_name],
            strategy,
        )
        for data_name in matching_names
    ]
    report = {
        "left_archive": left_archive.name,
        "right_archive": right_archive.name,
        "left_model": left_model.as_posix(),
        "right_model": right_model.as_posix(),
        "strategy": strategy,
        "instances": records,
    }
    report_stem = f"{left_archive.stem}_vs_{right_archive.stem}"
    json_path = left_archive.with_name(f"{report_stem}.json")
    markdown_path = left_archive.with_name(f"{report_stem}.md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_make_markdown(records), encoding="utf-8")
    return report

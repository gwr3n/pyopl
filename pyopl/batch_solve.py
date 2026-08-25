"""Solve all model instances contained in a PyOPL batch archive."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .pyopl_core import solve

_CONFIG_NAMES = {"gurobi.json": "gurobi", "highs.json": "scipy"}
_SOLVER_NAMES = {"scipy": "scipy", "highs": "scipy", "gurobi": "gurobi"}


def _is_metadata_member(member: Path) -> bool:
    """Identify common filesystem metadata entries without relying on a folder name."""
    return any(part.startswith("._") or part in {".DS_Store", "Thumbs.db"} for part in member.parts)


def _archive_members(batch_archive: zipfile.ZipFile) -> list[Path]:
    """Return non-directory, non-filesystem-metadata archive members."""
    return [
        Path(info.filename)
        for info in batch_archive.infolist()
        if not info.is_dir() and not _is_metadata_member(Path(info.filename))
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def _make_markdown(records: list[dict[str, Any]]) -> str:
    report_columns = {"data", "solver", "status", "objective_value", "message"}
    include_message = any(record.get("solver") != "gurobi" for record in records)
    stat_names = sorted(
        {str(name) for record in records for name in record.get("stats", {}) if str(name) not in report_columns},
        key=str.lower,
    )
    columns = ["data", "solver", "status", "objective_value", *stat_names]
    if include_message:
        columns.insert(4, "message")
    lines = [
        "# Batch solve results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        stats = record.get("stats", {})
        values = [
            _markdown_value(
                stats.get(column)
                if column == "message" and not record.get(column) and isinstance(stats, dict)
                else stats.get(column) if column in stat_names and isinstance(stats, dict) else record.get(column)
            )
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _extract_safely(batch_archive: zipfile.ZipFile, destination: Path) -> None:
    destination_root = destination.resolve()
    for member in batch_archive.infolist():
        member_path = (destination / member.filename).resolve()
        if destination_root not in member_path.parents and member_path != destination_root:
            raise ValueError(f"Archive member escapes extraction directory: {member.filename}")
    batch_archive.extractall(destination)


def _solver_configuration(members: list[Path], extraction_root: Path, solver_name: str) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for member in members:
        if _CONFIG_NAMES.get(member.name.lower()) != solver_name:
            continue
        try:
            settings = json.loads((extraction_root / member).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid solver configuration '{member}': {error}") from error
        if not isinstance(settings, dict):
            raise ValueError(f"Solver configuration '{member}' must contain a JSON object")
    return settings


def _solve_data_file(
    model_path: Path, data_path: Path, data_name: str, solver_name: str, settings: dict[str, Any]
) -> dict[str, Any]:
    public_solver = "highs" if solver_name == "scipy" else solver_name
    try:
        result = solve(str(model_path), str(data_path), solver=solver_name, solver_settings=settings)
        return {"data": data_name, "solver": public_solver, **_json_safe(result)}
    except Exception as error:
        return {
            "data": data_name,
            "solver": public_solver,
            "status": "ERROR",
            "message": str(error),
            "solution": {},
            "objective_value": None,
            "stats": {},
        }


def batch_solve(zip_path: str | Path, solver: str = "highs") -> dict[str, Any]:
    """Solve every ``.dat`` file in *zip_path* and write sibling reports."""
    archive = Path(zip_path)
    if archive.suffix.lower() != ".zip":
        raise ValueError("Batch archive must have a .zip extension")
    if not archive.is_file():
        raise FileNotFoundError(f"Batch archive not found: {archive}")
    try:
        solver_name = _SOLVER_NAMES[solver.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("Unsupported solver; choose highs or gurobi") from error

    with tempfile.TemporaryDirectory(prefix="pyopl-batch-") as temporary_directory:
        extraction_root = Path(temporary_directory)
        with zipfile.ZipFile(archive) as batch_archive:
            members = _archive_members(batch_archive)
            models = [member for member in members if member.suffix.lower() == ".mod"]
            if len(models) != 1:
                raise ValueError("Batch archive must contain exactly one .mod file; " f"found {len(models)}")
            data_files = [member for member in members if member.suffix.lower() == ".dat"]
            if not data_files:
                raise ValueError("Batch archive must contain at least one .dat file")
            _extract_safely(batch_archive, extraction_root)
            model_path = extraction_root / models[0]
            settings = _solver_configuration(members, extraction_root, solver_name)

        records: list[dict[str, Any]] = []
        for data_file in sorted(data_files, key=lambda path: str(path).lower()):
            records.append(
                _solve_data_file(
                    model_path,
                    extraction_root / data_file,
                    data_file.as_posix(),
                    solver_name,
                    settings,
                )
            )

    report = {"archive": archive.name, "model": models[0].as_posix(), "instances": records}
    json_path = archive.with_suffix(".json")
    markdown_path = archive.with_suffix(".md")
    json_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_make_markdown(records), encoding="utf-8")
    return report

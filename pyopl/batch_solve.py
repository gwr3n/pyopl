"""Solve all model instances contained in a PyOPL batch archive."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
import zipfile
from pathlib import Path
from threading import Event
from typing import Any, Callable, Optional

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


def _model_batches(members: list[Path]) -> list[tuple[Path, list[Path]]]:
    folders = sorted({member.parent for member in members}, key=lambda path: path.as_posix().lower())
    batches: list[tuple[Path, list[Path]]] = []
    for folder in folders:
        folder_members = [member for member in members if member.parent == folder]
        models = [member for member in folder_members if member.suffix.lower() == ".mod"]
        data_files = sorted(
            (member for member in folder_members if member.suffix.lower() == ".dat"),
            key=lambda path: path.as_posix().lower(),
        )
        if len(models) == 1 and data_files:
            batches.append((models[0], data_files))
    return batches


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
    report_columns = {"model", "data", "solver", "status", "objective_value", "message"}
    include_message = any(record.get("solver") != "gurobi" for record in records)
    stat_names = sorted(
        {str(name) for record in records for name in record.get("stats", {}) if str(name) not in report_columns},
        key=str.lower,
    )
    columns = ["model", "data", "solver", "status", "objective_value", *stat_names]
    if include_message:
        columns.insert(5, "message")
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


def _load_partial_records(json_path: Path, solver: str) -> list[dict[str, Any]]:
    if not json_path.is_file():
        return []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        return []
    return [record for record in payload["instances"] if isinstance(record, dict) and record.get("solver") == solver]


def _write_report(
    archive: Path,
    models: list[str],
    records: list[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {"archive": archive.name, "models": models, "instances": records}
    if len(models) == 1:
        report["model"] = models[0]
    json_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_make_markdown(records), encoding="utf-8")
    return report


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
    model_path: Path,
    model_name: str,
    data_path: Path,
    data_name: str,
    solver_name: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    public_solver = "highs" if solver_name == "scipy" else solver_name
    try:
        result = solve(str(model_path), str(data_path), solver=solver_name, solver_settings=settings)
        return {"model": model_name, "data": data_name, "solver": public_solver, **_json_safe(result)}
    except Exception as error:
        return {
            "model": model_name,
            "data": data_name,
            "solver": public_solver,
            "status": "ERROR",
            "message": str(error),
            "solution": {},
            "objective_value": None,
            "stats": {},
        }


def _format_duration(seconds: float) -> str:
    total_minutes = max(0, int(seconds / 60))
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remaining_minutes, 60)
    parts = []
    if days:
        parts.append(f"{days} day" if days == 1 else f"{days} days")
    if hours:
        parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
    if minutes:
        parts.append(f"{minutes} minute" if minutes == 1 else f"{minutes} minutes")
    return " ".join(parts) if parts else "less than 1 minute"


def _solve_batch_instances(
    archive: Path,
    extraction_root: Path,
    batches: list[tuple[Path, list[Path]]],
    solver_name: str,
    settings: dict[str, Any],
    models: list[str],
    records: list[dict[str, Any]],
    completed: set[tuple[Any, Any]],
    json_path: Path,
    markdown_path: Path,
    progress_callback: Optional[Callable[[dict[str, Any]], None]],
    stop_event: Optional[Event],
) -> dict[str, Any]:
    total_instances = sum(len(data_files) for _, data_files in batches)
    solved_instances = len(completed)
    elapsed_solution_time = 0.0
    if progress_callback:
        progress_callback(
            {
                "event": "started",
                "total": total_instances,
                "completed": solved_instances,
                "remaining": total_instances - solved_instances,
                "average_solution_time": 0.0,
            }
        )
    for model, data_files in batches:
        for data_file in data_files:
            if stop_event is not None and stop_event.is_set():
                if progress_callback:
                    progress_callback(
                        {
                            "event": "stopped",
                            "total": total_instances,
                            "completed": solved_instances,
                            "remaining": total_instances - solved_instances,
                            "average_solution_time": (elapsed_solution_time / solved_instances if solved_instances else 0.0),
                        }
                    )
                return _write_report(archive, models, records, json_path, markdown_path)
            instance_key = (model.as_posix(), data_file.as_posix())
            if instance_key in completed:
                continue
            if progress_callback:
                progress_callback(
                    {
                        "event": "instance_started",
                        "model": model.as_posix(),
                        "data": data_file.as_posix(),
                        "total": total_instances,
                        "completed": solved_instances,
                        "remaining": total_instances - solved_instances,
                        "average_solution_time": (elapsed_solution_time / solved_instances if solved_instances else 0.0),
                    }
                )
            started_at = time.perf_counter()
            record = _solve_data_file(
                extraction_root / model,
                model.as_posix(),
                extraction_root / data_file,
                data_file.as_posix(),
                solver_name,
                settings,
            )
            elapsed_solution_time += time.perf_counter() - started_at
            records.append(record)
            completed.add(instance_key)
            solved_instances += 1
            _write_report(archive, models, records, json_path, markdown_path)
            if progress_callback:
                progress_callback(
                    {
                        "event": "progress",
                        "model": model.as_posix(),
                        "data": data_file.as_posix(),
                        "total": total_instances,
                        "completed": solved_instances,
                        "remaining": total_instances - solved_instances,
                        "average_solution_time": elapsed_solution_time / solved_instances,
                    }
                )
    return _write_report(archive, models, records, json_path, markdown_path)


def batch_solve(
    zip_path: str | Path,
    solver: str = "highs",
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    stop_event: Optional[Event] = None,
) -> dict[str, Any]:
    """Solve each folder containing one ``.mod`` and one or more ``.dat`` files."""
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
            batches = _model_batches(members)
            if not batches:
                raise ValueError("Batch archive must contain a folder with exactly one .mod and at least one .dat file")
            _extract_safely(batch_archive, extraction_root)
            settings = _solver_configuration(members, extraction_root, solver_name)

        models = [model.as_posix() for model, _ in batches]
        json_path = archive.with_suffix(".json")
        markdown_path = archive.with_suffix(".md")
        records = _load_partial_records(json_path, "highs" if solver_name == "scipy" else solver_name)
        completed = {(record.get("model"), record.get("data")) for record in records}
        return _solve_batch_instances(
            archive,
            extraction_root,
            batches,
            solver_name,
            settings,
            models,
            records,
            completed,
            json_path,
            markdown_path,
            progress_callback,
            stop_event,
        )


def batch_solve_with_progress(zip_path: str | Path, solver: str = "highs") -> dict[str, Any]:
    """Run batch solving with a small Tk progress window and a stop button."""
    import tkinter as tk
    from queue import Empty
    from tkinter import ttk

    events: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_batch_solve_worker, args=(str(zip_path), solver, events))

    root = tk.Tk()
    root.title("PyOPL batch solve")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=16)
    frame.grid()
    status = ttk.Label(frame, text="Preparing batch solve...\n\n\n\n\n", anchor="w", justify="left")
    status.grid(row=0, column=0, sticky="w")
    progress = ttk.Progressbar(frame, length=360, mode="determinate")
    progress.grid(row=1, column=0, pady=(10, 8))
    root.update_idletasks()
    root.minsize(root.winfo_width(), root.winfo_height())

    def close_window() -> None:
        root.destroy()

    def show_close_button() -> None:
        stop_button.configure(text="Close", command=close_window, state="normal")

    def stop_batch() -> None:
        if process.is_alive():
            process.terminate()
        show_close_button()

    stop_button = ttk.Button(frame, text="Stop", command=stop_batch)
    stop_button.grid(row=2, column=0, sticky="e")

    def poll_events() -> None:
        try:
            while True:
                event = events.get_nowait()
                if event["event"] == "started":
                    progress.configure(maximum=event["total"], value=event["completed"])
                    status.configure(
                        text=(
                            f"Benchmark: {Path(zip_path).name}\n"
                            "Average solution time per instance: n/a\n"
                            "Estimated time to completion: n/a"
                        )
                    )
                elif event["event"] == "instance_started":
                    average = event["average_solution_time"]
                    has_completed_instances = event["completed"] > 0
                    estimated_completion = average * event["remaining"]
                    average_text = (
                        f"Average solution time per instance: {average:.2f}s"
                        if has_completed_instances
                        else "Average solution time per instance: n/a"
                    )
                    estimated_completion_text = (
                        f"Estimated time to completion: {_format_duration(estimated_completion)}"
                        if has_completed_instances
                        else "Estimated time to completion: n/a"
                    )
                    status.configure(
                        text=(
                            f"Benchmark: {Path(zip_path).name}\n"
                            f"Current model: {Path(event['model']).name}\n"
                            f"Current data: {Path(event['data']).name}\n"
                            f"Solved {event['completed']} of {event['total']}\n"
                            f"{average_text}\n" + estimated_completion_text
                        )
                    )
                elif event["event"] in {"progress", "stopped"}:
                    progress.configure(value=event["completed"])
                    average = event["average_solution_time"]
                    estimated_completion = average * event["remaining"]
                    status.configure(
                        text=(
                            f"Benchmark: {Path(zip_path).name}\n"
                            f"Current model: {Path(event.get('model', '')).name}\n"
                            f"Current data: {Path(event.get('data', '')).name}\n"
                            f"Solved {event['completed']} of {event['total']}\n"
                            f"Average solution time per instance: {average:.2f}s\n"
                            f"Estimated time to completion: {_format_duration(estimated_completion)}"
                        )
                    )
                    if event["event"] == "stopped":
                        stop_button.configure(state="disabled")
                elif event["event"] == "finished":
                    show_close_button()
        except Empty:
            pass
        if process.is_alive() or not events.empty():
            root.after(100, poll_events)

    process.start()
    root.after(100, poll_events)
    root.mainloop()
    process.join()
    if process.exitcode not in (0, None):
        report_path = Path(zip_path).with_suffix(".json")
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(report, dict):
                    return report
            except (OSError, json.JSONDecodeError):
                pass
        raise RuntimeError("Batch solve process was interrupted")
    report_path = Path(zip_path).with_suffix(".json")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Batch solve did not produce a report") from error
    if not isinstance(report, dict):
        raise RuntimeError("Batch solve produced an invalid report")
    return report


def _batch_solve_worker(zip_path: str, solver: str, events: multiprocessing.Queue) -> None:
    stop_event = Event()
    try:
        batch_solve(zip_path, solver=solver, progress_callback=events.put, stop_event=stop_event)
    finally:
        events.put({"event": "finished"})

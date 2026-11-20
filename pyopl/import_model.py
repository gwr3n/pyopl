"""Import linear LP/MPS models as a generic OPL model and data instance."""

from __future__ import annotations

import math
import multiprocessing
import traceback
from pathlib import Path
from typing import Callable

import highspy


def start_import(input_path: str | Path, model_path: str | Path, data_path: str | Path):
    """Run native HiGHS parsing outside the caller's event loop."""
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    stopped = context.Event()
    process = context.Process(
        target=_import_worker,
        args=(str(input_path), str(model_path), str(data_path), messages, stopped),
        daemon=True,
    )
    try:
        process.start()
    except Exception:
        messages.close()
        raise
    return process, messages, stopped


def _import_worker(input_path: str, model_path: str, data_path: str, messages, stopped) -> None:
    try:
        convert_file(
            input_path,
            model_path,
            data_path,
            progress=lambda stage, percent: messages.put(("progress", (stage, percent))),
            stop_requested=stopped.is_set,
        )
    except Exception as exc:
        messages.put(("error", (type(exc).__name__, str(exc), traceback.format_exc())))
    else:
        messages.put(("done", None))
    finally:
        messages.close()
        messages.join_thread()


def convert_file(
    input_path: str | Path,
    model_path: str | Path,
    data_path: str | Path,
    *,
    progress: Callable[[str, int], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    input_path = Path(input_path)
    model_path = Path(model_path)
    data_path = Path(data_path)
    if input_path.suffix.lower() not in (".lp", ".mps"):
        raise ValueError(f"Expected a .mps or .lp file, got: {input_path}")

    last_progress: tuple[str, int] | None = None

    def report(stage: str, percent: int) -> None:
        nonlocal last_progress
        if stop_requested is not None and stop_requested():
            raise InterruptedError("Model import stopped")
        if progress is not None and last_progress != (stage, percent):
            progress(stage, percent)
            last_progress = (stage, percent)

    report("Reading model...", 0)
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    status = highs.readModel(str(input_path))
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"Failed to read optimization model {input_path}: {status}")
    model = highs.getModel()
    if model.hessian_.dim_:
        raise ValueError("Quadratic objectives are not supported by the OPL importer")
    lp = model.lp_
    if lp.num_col_ == 0:
        raise ValueError("Models without variables are not supported by the OPL importer")

    report("Converting matrix...", 10)
    matrix_rows, rhs, senses = _constraint_data(lp, _matrix_rows(lp, report))
    report("Converting variables...", 70)
    binaries, integers = _variable_types(lp, report)
    report("Writing model and data...", 85)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    existing_model, existing_data = model_path.exists(), data_path.exists()
    try:
        _write_mod(model_path, lp.sense_ == highspy.ObjSense.kMinimize, bool(matrix_rows))
        report("Writing data...", 90)
        _write_dat(data_path, lp, matrix_rows, rhs, senses, binaries, integers, report)
        report("Import complete", 100)
    except Exception:
        if not existing_model:
            model_path.unlink(missing_ok=True)
        if not existing_data:
            data_path.unlink(missing_ok=True)
        raise


def _matrix_rows(lp: highspy.HighsLp, report: Callable[[str, int], None]) -> list[list[tuple[int, float]]]:
    matrix = lp.a_matrix_
    starts, indices, values = matrix.start_, matrix.index_, matrix.value_
    rows: list[list[tuple[int, float]]] = [[] for _ in range(lp.num_row_)]
    if matrix.format_ == highspy.MatrixFormat.kRowwise:
        for row_idx in range(lp.num_row_):
            if row_idx % 256 == 0:
                report("Converting matrix...", 10 + 50 * row_idx // max(1, lp.num_row_))
            rows[row_idx] = [(indices[j] + 1, values[j]) for j in range(starts[row_idx], starts[row_idx + 1])]
    elif matrix.format_ == highspy.MatrixFormat.kColwise:
        for col in range(lp.num_col_):
            if col % 256 == 0:
                report("Converting matrix...", 10 + 50 * col // lp.num_col_)
            for j in range(starts[col], starts[col + 1]):
                rows[indices[j]].append((col + 1, values[j]))
    else:
        raise ValueError(f"Unsupported HiGHS matrix format: {matrix.format_}")
    return rows


def _constraint_data(
    lp: highspy.HighsLp, rows: list[list[tuple[int, float]]]
) -> tuple[list[list[tuple[int, float]]], list[float], list[str]]:
    matrix_rows: list[list[tuple[int, float]]] = []
    rhs: list[float] = []
    senses: list[str] = []
    lower_bounds, upper_bounds = lp.row_lower_, lp.row_upper_
    for row_entries, lower, upper in zip(rows, lower_bounds, upper_bounds):
        if lower == upper and math.isfinite(lower):
            matrix_rows.append(row_entries)
            rhs.append(lower)
            senses.append("E")
        else:
            if math.isfinite(upper):
                matrix_rows.append(row_entries)
                rhs.append(upper)
                senses.append("L")
            if math.isfinite(lower):
                matrix_rows.append(row_entries)
                rhs.append(lower)
                senses.append("G")
    return matrix_rows, rhs, senses


def _variable_types(lp: highspy.HighsLp, report: Callable[[str, int], None]) -> tuple[list[int], list[int]]:
    binaries: list[int] = []
    integers: list[int] = []
    integrality, lower_bounds, upper_bounds = lp.integrality_, lp.col_lower_, lp.col_upper_
    for col in range(lp.num_col_):
        if col % 256 == 0:
            report("Converting variables...", 70 + 15 * col // lp.num_col_)
        kind = integrality[col] if integrality else highspy.HighsVarType.kContinuous
        if kind not in (highspy.HighsVarType.kContinuous, highspy.HighsVarType.kInteger) and kind.name != "kImplicitInteger":
            raise ValueError(f"Unsupported variable type at column {col + 1}: {kind}")
        binary = kind != highspy.HighsVarType.kContinuous and lower_bounds[col] == 0 and upper_bounds[col] == 1
        binaries.append(int(binary))
        integers.append(int(kind != highspy.HighsVarType.kContinuous and not binary))
    return binaries, integers


def _write_mod(path: Path, minimize: bool, has_rows: bool) -> None:
    objective = "minimize" if minimize else "maximize"
    path.write_text(
        """int numVars = ...;
range Vars = 1..numVars;
float objOffset = ...;
float objCoef[Vars] = ...;
float lb[Vars] = ...;
float ub[Vars] = ...;
int hasLb[Vars] = ...;
int hasUb[Vars] = ...;
int isBinary[Vars] = ...;
int isInteger[Vars] = ...;
"""
        + (
            """int numConstraints = ...;
range Constraints = 1..numConstraints;
tuple MatrixElement {
  int v;
  float val;
};
{MatrixElement} matrix[Constraints] = ...;
float rhs[Constraints] = ...;
string sense[Constraints] = ...;
"""
            if has_rows
            else ""
        )
        + """
{int} BinaryVars = {v | v in Vars: isBinary[v] == 1};
{int} IntegerVars = {v | v in Vars: isInteger[v] == 1};
{int} ContinuousVars = {v | v in Vars: isBinary[v] == 0 && isInteger[v] == 0};
dvar boolean xBinary[BinaryVars];
dvar int xInteger[IntegerVars];
dvar float xContinuous[ContinuousVars];
dexpr float x[v in Vars] =
  sum(b in BinaryVars: b == v) xBinary[b] +
  sum(i in IntegerVars: i == v) xInteger[i] +
  sum(c in ContinuousVars: c == v) xContinuous[c];
"""
        + f"{objective} objOffset + sum(v in Vars) objCoef[v] * x[v];\n"
        + """subject to {
  forall(v in Vars) {
    if (hasLb[v] == 1)
      lower_bound: x[v] >= lb[v];
    if (hasUb[v] == 1)
      upper_bound: x[v] <= ub[v];
  }
"""
        + (
            """
  forall(c in Constraints) {
    ct_row:
      if (sense[c] == "E")
        sum(e in matrix[c]) e.val * x[e.v] == rhs[c];
      else if (sense[c] == "L")
        sum(e in matrix[c]) e.val * x[e.v] <= rhs[c];
      else
        sum(e in matrix[c]) e.val * x[e.v] >= rhs[c];
  }
"""
            if has_rows
            else ""
        )
        + "}\n",
        encoding="utf-8",
    )


def _write_dat(
    path: Path,
    lp: highspy.HighsLp,
    rows: list[list[tuple[int, float]]],
    rhs: list[float],
    senses: list[str],
    binaries: list[int],
    integers: list[int],
    report: Callable[[str, int], None],
) -> None:
    costs, lower_bounds, upper_bounds = lp.col_cost_, lp.col_lower_, lp.col_upper_
    sense_values = _wrap_dat_values([f'"{sense}"' for sense in senses])
    with path.open("w", encoding="utf-8") as output:
        output.write(
            f"numVars = {lp.num_col_};\n"
            f"objOffset = {lp.offset_};\n"
            f"objCoef = [{_wrap_dat_values([str(float(value)) for value in costs])}];\n"
            f"lb = [{_wrap_dat_values([str(float(value) if math.isfinite(value) else 0.0) for value in lower_bounds])}];\n"
            f"ub = [{_wrap_dat_values([str(float(value) if math.isfinite(value) else 0.0) for value in upper_bounds])}];\n"
            f"hasLb = [{_wrap_dat_values([str(int(math.isfinite(value))) for value in lower_bounds])}];\n"
            f"hasUb = [{_wrap_dat_values([str(int(math.isfinite(value))) for value in upper_bounds])}];\n"
            f"isBinary = [{_wrap_dat_values([str(value) for value in binaries])}];\n"
            f"isInteger = [{_wrap_dat_values([str(value) for value in integers])}];\n"
        )
        if rows:
            output.write(f"numConstraints = {len(rows)};\nmatrix = [\n")
            for index, row in enumerate(rows):
                if index:
                    output.write(",\n")
                output.write("  { " + _wrap_dat_values([f"<{var}, {coef}>" for var, coef in row]) + " }")
                if index % 4096 == 0:
                    report("Writing data...", 90 + 9 * index // len(rows))
            output.write("\n];\n")
            output.write(f"rhs = [{_wrap_dat_values([str(value) for value in rhs])}];\n")
            output.write(f"sense = [{sense_values}];\n")


def _wrap_dat_values(values: list[str]) -> str:
    return ",\n    ".join(", ".join(values[start : start + 80]) for start in range(0, len(values), 80))

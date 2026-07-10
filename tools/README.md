# PyOPL Tools

The `tools` folder contains developer-facing utilities for running PyOPL examples,
testing generated OPL models, benchmarking GenAI modelling workflows, and capturing
test output. These scripts are not part of the installed `pyopl` command-line entry
point; they are meant to be run from the repository root while developing or
validating the project.

Run commands from the project root:

```bash
python -m tools.<module_name>
```

## Available Tools

### `examples.py`

Runs one of the packaged OPL example models in `pyopl/opl_models` with a selected
solver. The file contains an `Example` enum for model selection and a `Solver` enum
for choosing between the SciPy/HiGHS and Gurobi backends.

Edit these selectors near the bottom of the file before running it:

```python
EXAMPLE_SELECTOR = Example.LOT_SIZING
SOLVER_SELECTOR = Solver.GUROBI
```

Then run:

```bash
python -m tools.examples
```

Use this script when you want a quick manual check that a bundled model and data
file can be parsed, generated, solved, and printed through PyOPL.

### `_sparse_example.py`

Builds a sparse vehicle-routing style example in two forms: a dense matrix-based
OPL model and a tuple-set-based OPL model. It writes temporary `.mod` and `.dat`
files, reports file sizes, solves both formulations, prints solver status and
objective values, and removes the temporary files afterward.

This module is normally called through `examples.py` by setting:

```python
EXAMPLE_SELECTOR = Example.SPARSE_EXAMPLE
```

Use it to compare dense matrix data with tuple-set data and to exercise PyOPL's
tuple handling on larger sparse instances.

### `genai_modelling.py`

Provides simple smoke-test functions for the GenAI modelling API in
`pyopl.genai.pyopl_generative`:

- `test_generative_solve()` generates a model and data file from a natural-language
	optimization prompt.
- `test_generative_feedback()` asks for feedback on an existing generated model
	and data file.

Generated files are written under `tmp/` as `gen_pyopl_model.mod` and
`gen_pyopl_data.dat`.

Run it with:

```bash
python -m tools.genai_modelling
```

The script currently enables `test_generative_solve()` by default. Toggle
`test_solve` and `test_feedback` in the `__main__` block to choose which smoke
test to run.

### `genai_benchmark.py`

Runs benchmark problems from JSON datasets in `gen_ai/datasets/<dataset>/` through
one of the GenAI modelling strategies, then validates the generated result. If the
dataset item includes reference model/data text, validation uses MILP equivalence;
otherwise it solves the generated model and compares the objective value with the
dataset answer.

Common single-problem run:

```bash
python -m tools.genai_benchmark --dataset ComplexOR --index 0 --solver gurobi
```

Run a full dataset:

```bash
python -m tools.genai_benchmark --dataset ComplexOR --logic SyntAGM --all
```

Resume the latest interrupted full run:

```bash
python -m tools.genai_benchmark --dataset ComplexOR --logic SyntAGM --all --continue latest
```

Important options include:

- `--dataset`: dataset name, such as `ComplexOR`, `NL4OPT`, `NLP4LP`,
	`IndustryOR`, `ReSocratic`, `StochasticOR`, `ChallengeOR`, or `SmallOR`.
- `--logic`: GenAI strategy, such as `SyntAGM`, `standard`, `chain_of_thought`,
	`tree_of_thoughts`, `reflexion`, `cafa`, or `chain_of_experts`.
- `--grammar`: generation mode, such as `bnf`, `code`, or `none`, depending on
	the selected strategy.
- `--provider` and `--gpt`: LLM provider and model name.
- `--iterations`: maximum generation iterations.
- `--solver`: `scipy` or `gurobi` for solving generated models.
- `--tolerance`: absolute tolerance for objective comparison or MILP equivalence.
- `--no-few-shot`, `--no-alignment-check`, and `--syntax-error-reporting`: SyntAGM
	ablation and syntax-reporting controls.

Batch results are written under:

```text
gen_ai/<dataset>/<logic>/<grammar>/<model>/<iterations>/<timestamp>/
```

Each run folder contains generated model/data files in `models/` and a JSON results
file named `<dataset>_results.json`.

### `test_logger.py`

Runs the unittest suite or a single dotted unittest target and writes the full test
output to a file. This is useful when test output is long or when preserving logs
for later inspection.

Run the full discovered suite and write `unittest_results.txt`:

```bash
python -m tools.test_logger
```

Run one test target:

```bash
python -m tools.test_logger \
	--test test.test_problems.TestPyOPLProblems.test_complex_workforce_planning \
	--output tmp/unittest_results.txt
```

Useful options include:

- `--test`: dotted unittest module, class, or test method to run.
- `--start-dir`: discovery start directory, defaulting to `test`.
- `--pattern`: discovery pattern, defaulting to `test*.py`.
- `--top-level-dir`: optional project top-level directory for discovery.
- `--output`: output file path, defaulting to `unittest_results.txt`.
- `--verbosity`: unittest verbosity level.

## Notes

- Run these scripts from the repository root so relative paths resolve correctly.
- Some tools require optional services or packages, such as a configured LLM
	provider or a working Gurobi installation.
- Generated benchmark and smoke-test artifacts are written to `gen_ai/` or `tmp/`;
	review those folders before committing changes.

---
name: pyopl-generate
description: "Generate a PyOPL .mod model and matching .dat data from a natural-language optimization problem. Use when asked to generate, formulate, create, or draft a PyOPL model, or to perform the IDE Generate Model & Data action. Includes grammar lookup, bounded compile-and-revise iterations, semantic alignment review, and file creation without Rhetor MCP."
argument-hint: "Describe the optimization problem and optionally provide target .mod and .dat paths"
---

# Generate PyOPL Model And Data

## Inputs

Determine:

- The complete problem description, including objective, decisions, constraints, and supplied data.
- Target `.mod` and `.dat` paths. Infer clear sibling paths from context; ask only when the destination is materially ambiguous.
- The iteration limit, defaulting to 5 and never using fewer than 1 attempt.
- Whether the user wants compilation only or also a representative solve.

Keep the original problem description unchanged as the semantic reference for every later assessment and revision.

## Procedure

1. Acquire syntax guidance.
   - Call the PyOPL MCP `read_pyopl_grammar_tool` when available.
   - Otherwise read the repository's bundled PyOPL grammar documentation.
   - Use this reference for syntax, not as a source of problem requirements.

2. Formulate privately.
   - Identify sets and ranges, input parameters, decision variables, objective, and constraints.
   - Select binary, integer, or float domains from the problem meaning.
   - Resolve units, signs, index domains, and links between decisions and constraints.
   - If data is absent, prepare a small plausible mock instance and disclose that assumption.

3. Create complete artifacts.
   - Write one complete `.mod` file and one matching `.dat` file.
   - Label the objective and constraints.
   - Add concise comments explaining parameters, variables, and non-obvious formulation choices.
   - Keep instance values in `.dat` unless the grammar requires inline values.

4. Compile the pair.
   - Prefer PyOPL MCP `export_py_strings_tool` with the complete model and data strings. This is a compile-only check even though it returns generated Python.
   - Do not call a Rhetor MCP tool.
   - On failure, capture the exact compiler error.

5. Revise syntax or semantics within the attempt budget.
   - Make only the changes needed to resolve the reported compiler error.
   - Preserve the intended formulation and return to step 4.
   - Count the initial draft as attempt 1. Stop after the configured limit.

6. Assess semantic alignment after a successful compile.
   - Compare the model and data against the original problem description.
   - Check objective direction and terms, every stated constraint, variable domains and indices, data coverage, signs, units, missing links, and unintended restrictions.
   - If misaligned and attempts remain, make the smallest alignment correction and return to step 4.
   - Compilation success is necessary but not sufficient for alignment.

7. Optionally solve.
   - Call PyOPL MCP `solve_strings_tool` only when the user requests a solve or a representative run is useful to detect infeasibility or bad data.
   - Treat solver status and values as runtime evidence, not proof of semantic alignment.

8. Finish.
   - Ensure the final complete contents are present at the target paths.
   - Report paths, attempts used, compile status, alignment assessment, mock-data assumptions, and solve status when run.
   - If the budget ends with errors or misalignment, retain the latest artifacts but label them as unresolved rather than valid.

## Revision Discipline

- Always revise both files as one formulation, even when only one file changes.
- Do not replace a sound formulation merely to change style.
- Do not introduce requirements absent from the problem statement.
- Do not hide compiler diagnostics or silently weaken constraints to obtain a successful solve.

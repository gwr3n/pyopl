---
name: pyopl-generate
description: "Generate a PyOPL .mod model and matching .dat data from a natural-language optimization problem. Use when asked to generate, formulate, create, or draft a PyOPL model, or to perform the IDE Generate Model & Data action. Includes grammar lookup, relevant few-shot exemplar retrieval, literate modeling, bounded compile-and-revise iterations, semantic alignment review, and file creation without Rhetor MCP."
argument-hint: "Describe the optimization problem and optionally provide target .mod/.dat paths or an additional exemplar folder"
---

# Generate PyOPL Model And Data

## Inputs

Determine:

- The complete problem description, including objective, decisions, constraints, and supplied data.
- Any attached images containing problem text, tables, charts, or diagrams.
- Target `.mod` and `.dat` paths. Infer clear sibling paths from context; ask only when the destination is materially ambiguous.
- Any additional exemplar folder named by the user. Treat it as an addition to the bundled corpus, not a replacement.
- The iteration limit, defaulting to 5 and never using fewer than 1 attempt.
- Whether the user wants compilation only or also a representative solve.

Keep the original problem description unchanged as the semantic reference for every later assessment and revision.

Before generating artifacts, identify unresolved ambiguity. If in doubt, ask a concise clarification question when different reasonable answers would change the objective, decision variables, constraints, domains, indices, units, or interpretation of supplied data. Do not infer these requirements from an exemplar. Missing instance values alone are not blocking when a small disclosed mock instance is appropriate. If the user explicitly requests best effort without clarification, use the least restrictive reasonable interpretation and record every material assumption.

## Procedure

1. Resolve material ambiguity.
   - Check the problem description, attachments, and supplied context before formulation.
   - Ask only the smallest set of questions needed to distinguish materially different formulations.
   - Do not create or compile artifacts until blocking clarification is answered, unless the user explicitly authorized a best-effort result.

2. Acquire syntax guidance.
   - Call the PyOPL MCP `read_pyopl_grammar_tool` when available.
   - Otherwise read the repository's bundled PyOPL grammar documentation.
   - Use this reference for syntax, not as a source of problem requirements.

3. Retrieve relevant few-shot exemplars.
   - Search the installed package corpus at `pyopl/opl_models` by default. Resolve it from the active Python installation rather than assuming the repository checkout is the installed package.
   - Also search an additional folder when the user supplies one. Search recursively and retain the bundled corpus as a source.
   - When the prompt includes images, transcribe readable text and briefly describe relevant tables, labels, and relationships; append that compact context to the retrieval query. Continue using the original text and images as the semantic source of truth.
   - Treat each exemplar as a complete triple: a Markdown problem description plus matching `.mod` and `.dat` files. Prefer files with the same stem as the description; otherwise use the single clear model/data pair in that description's folder. Skip incomplete or ambiguous pairs.
   - Rank descriptions by semantic relevance to the unchanged problem description and use up to 3 of the strongest matches. If semantic ranking is unavailable, select only clearly relevant examples by direct inspection. Continue without exemplars when none are usable.
   - Use exemplars for PyOPL syntax, formulation patterns, structure, and literate presentation only. Never copy their requirements, names, indices, or data unless the user's problem independently calls for them.
   - Keep the selected exemplar set fixed across syntax and alignment revisions so later attempts retain the same grounding.

4. Formulate privately.
   - Identify sets and ranges, input parameters, decision variables, objective, and constraints.
   - Select binary, integer, or float domains from the problem meaning.
   - Resolve units, signs, index domains, and links between decisions and constraints.
   - If data is absent, prepare a small plausible mock instance and disclose that assumption.

5. Create complete artifacts using literate modeling.
   - Write one complete `.mod` file and one matching `.dat` file.
   - Organize declarations, objective, and constraints in a readable order that follows the problem formulation.
   - Label the objective and every constraint with problem-domain names rather than generic numbering when practical.
   - Add concise comments that explain the role and units of sets, parameters, and decision variables, plus the intent of each non-obvious objective term and constraint family.
   - Keep comments synchronized with the mathematics and favor explanatory intent over restating syntax. This literate presentation is part of the generated artifact, not optional decoration.
   - Keep instance values in `.dat` unless the grammar requires inline values.

6. Compile the pair.
   - Prefer PyOPL MCP `export_py_strings_tool` with the complete model and data strings. This is a compile-only check even though it returns generated Python.
   - Do not call a Rhetor MCP tool.
   - On failure, capture the exact compiler error.

7. Revise syntax or semantics within the attempt budget.
   - Make only the changes needed to resolve the reported compiler error.
   - Preserve the intended formulation and return to step 5.
   - Count the initial draft as attempt 1. Stop after the configured limit.

8. Assess semantic alignment after a successful compile.
   - Compare the model and data against the original problem description.
   - Check objective direction and terms, every stated constraint, variable domains and indices, data coverage, signs, units, missing links, and unintended restrictions.
   - If misaligned and attempts remain, make the smallest alignment correction and return to step 5.
   - Compilation success is necessary but not sufficient for alignment.

9. Optionally solve.
   - Call PyOPL MCP `solve_strings_tool` only when the user requests a solve or a representative run is useful to detect infeasibility or bad data.
   - Treat solver status and values as runtime evidence, not proof of semantic alignment.

10. Finish.
   - Ensure the final complete contents are present at the target paths.
   - Report paths, attempts used, compile status, alignment assessment, exemplar sources used, mock-data assumptions, and solve status when run.
   - If the budget ends with errors or misalignment, retain the latest artifacts but label them as unresolved rather than valid.

## Revision Discipline

- Always revise both files as one formulation, even when only one file changes.
- Do not replace a sound formulation merely to change style.
- Do not introduce requirements absent from the problem statement.
- Do not hide compiler diagnostics or silently weaken constraints to obtain a successful solve.

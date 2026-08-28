---
name: pyopl-ask
description: "Review, explain, critique, or correct an existing PyOPL .mod and .dat pair. Use when asked a question about a model, for feedback, validation, debugging, or the IDE Ask action. Grounds feedback in the files and PyOPL grammar, validates any proposed complete revisions, preserves unrelated semantics, and never uses Rhetor MCP."
argument-hint: "Ask a question and identify the .mod and .dat files"
---

# Ask About A PyOPL Model

## Inputs

Require the user's question and an existing `.mod` file. Use its matching `.dat` file when present or state explicitly that the review is model-only.

Read the complete current files before answering. Preserve these originals for revision comparison.

If in doubt, ask a concise clarification question before answering or proposing changes when multiple reasonable interpretations of the question could materially change the review scope, intended formulation, equivalence requirement, or correction. Resolve uncertainty from the complete files when possible, but do not use compiler or solver success to guess intent. If the user explicitly requests best effort without clarification, distinguish the chosen interpretation and assumptions from verified facts.

## Procedure

1. Establish evidence.
   - Read the complete model and data files.
   - Call PyOPL MCP `read_pyopl_grammar_tool` when syntax or supported features matter; otherwise use the bundled grammar documentation.
   - Never call a Rhetor MCP `ask`, `generate`, or `insight` tool.

2. Resolve material ambiguity.
   - Use the question, complete files, and stated context to identify the intended scope and behavior.
   - Ask only the smallest set of questions needed to distinguish materially different answers or revisions.
   - Do not answer under an unstated interpretation or prepare a revision while blocking ambiguity remains, unless the user explicitly authorized a best-effort result.

3. Interpret the question.
   - Answer the specific question first.
   - Inspect the objective, relevant constraints, variable domains, indices, parameter declarations, and data values that bear on it.
   - Distinguish verified facts, modeling judgments, and assumptions.

4. Gather only relevant validation evidence.
   - For syntax or semantic compiler questions, call `export_py_strings_tool` with the complete model/data strings.
   - For feasibility, objective, or assignment questions, call `solve_strings_tool` and report solver status along with relevant outputs.
   - For behavior-preservation questions, call `compare_model_strings_tool` with originals and candidates, choosing abstract or concrete comparison to match the question.
   - Do not solve merely to answer a static syntax question.

5. Produce critical, specific feedback.
   - Check objective intent, constraint logic and signs, domains, indexing, data consistency, units, missing links, and unnecessary restrictions as relevant.
   - Point to concrete declarations or constraints rather than giving generic optimization advice.
   - If no correction is needed, say so and do not manufacture a revision.

6. Propose revisions only when necessary or requested.
   - Make the smallest changes needed to answer the question or correct the defect.
   - Preserve unrelated structure and semantics from the original files.
   - Label objective and constraints and retain or add concise literate comments where revised content needs explanation.
   - Prepare complete candidate model and data contents, not diffs. If one side is unchanged, carry its full original content into the candidate pair.

7. Validate every proposed revision.
   - Compile the complete candidate pair with `export_py_strings_tool`.
   - Compare it with the user's request and originals for alignment and preservation of unrelated behavior.
   - When equivalence is intended, use `compare_model_strings_tool`; do not infer equivalence from similar text or equal results on one instance.
   - If validation fails, repair minimally and retry, for at most 5 total candidate attempts.
   - Expose replacement contents only after successful compilation and alignment. Otherwise provide feedback plus the unresolved validation issue.

8. Return the result.
   - Give a direct answer under `Feedback`.
   - Add `Validated revision` only when complete replacement contents passed validation.
   - Report the validation performed and any assumptions or remaining limitations.
   - Do not overwrite the user's files unless they explicitly ask to apply the validated revision.

## Review Boundaries

- Compilation proves accepted PyOPL syntax and compiler semantics, not that the optimization formulation expresses the intended real-world problem.
- A feasible or optimal solve does not prove correctness.
- Concrete comparison applies to supplied instances; abstract comparison may require finite data to ground indexed schemas.

# Generative Feedback Validation Roadmap

**Status:** Completed  
**Completed:** 2026-08-24

## Goal

Ensure model or data revisions proposed by `generative_feedback` are compilable, aligned with the user's request, and consistent with the unchanged companion file before they are offered for application.

## Behavioral Contract

- Feedback without revisions returns without compilation or alignment calls.
- A model-only or data-only revision is combined with the unchanged companion file before validation.
- Proposed revisions are compiled with `OPLCompiler`.
- Compilable revisions are checked against the user's request and original model/data with a feedback-specific alignment prompt.
- Failed compilation or alignment enters a bounded repair loop.
- Only validated files that differ from the originals are returned.
- If validation cannot succeed, revision fields are omitted and the feedback explains that the proposal was withheld.
- Existing model and data files are never modified by feedback validation.

## Implementation Phases

### 1. Canonical Feedback Workflow (Completed)

- Add a shared feedback module that owns prompt construction, response validation, candidate assembly, compilation, alignment checking, and repair.
- Keep provider, grammar, image, progress, temperature, and stop-sequence behavior compatible with the existing public API.
- Add explicit validation controls with validation enabled by default.

### 2. Focused Validation Coverage (Completed)

Add `unittest` coverage for:

- feedback with no revisions;
- valid model-only and data-only revisions;
- candidate compilation using the unchanged companion file;
- syntax repair and revalidation;
- alignment repair and revalidation;
- companion-file changes introduced by repair;
- exhausted validation attempts and fail-closed behavior;
- malformed feedback, repair, and alignment payloads.

### 3. Strategy Deduplication (Completed)

- Re-export the canonical `generative_feedback` implementation from every GenAI strategy module.
- Remove strategy-local feedback implementations.
- Preserve the strategy module API used by the IDE and tests.
- Update type stubs to expose the shared validation options consistently.

### 4. Integration And Documentation (Completed)

- Run focused GenAI helper, CLI, MCP, and IDE tests.
- Run static diagnostics for all touched modules.
- Document the validated-feedback behavior in the changelog and user-facing API documentation.

## Completion Criteria

- No strategy contains its own `generative_feedback` function body.
- Invalid or misaligned revisions are never returned as applicable revisions.
- Valid partial revisions remain partial unless repair changes the companion file.
- Existing callers continue to use `generative_feedback` without required argument changes.
- Focused tests and static checks pass.

## Outcome

Implemented in `pyopl.genai.feedback_validation`. Every strategy re-exports the canonical function, and focused GenAI, CLI, MCP, and IDE tests verify the shared public behavior.

## Final Verification

- `generative_feedback` has one implementation in `pyopl.genai.feedback_validation`.
- All seven GenAI strategy modules directly re-export the canonical function.
- Revision candidates are compiled as complete model/data pairs and checked for feedback-specific alignment.
- Bounded repair handles compilation and alignment failures; exhausted validation fails closed.
- Type stubs, changelog, and user guide document the shared validation controls.
- Focused and integration-facing tests pass; repository quality tooling was run successfully on 2026-08-24.

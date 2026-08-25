# Exemplar Review Roadmap

## Goal

Replace the current abrupt `Save Exemplar...` action with a reviewed workflow:

1. Distill the current session into a concise optimization-problem description using an LLM.
2. Use three existing exemplar triplets as few-shot examples.
3. Present the proposed description, model, and data in a review popup with three tabs.
4. Save the exemplar only after the user accepts the reviewed content.

The saved exemplar remains a triplet:

- `<name>.mod`: OPL model
- `<name>.dat`: OPL data
- `<name>.md`: concise natural-language problem description

The raw session transcript is input to distillation but is not saved as the description.

## Current Implementation Anchor

The existing flow is in `pyopl/pyopl_ide_bootstrap.py`:

- `save_exemplar()` asks for a name, saves the current session, reads the complete `.pyopl_session` file as the description, and writes immediately.
- `_resolve_exemplar_destination()` validates the destination beneath `opl_models`.
- `_write_exemplar()` writes the model, data, and description atomically.

The existing RAG implementation already loads model/data/description triplets and renders few-shot examples. The new flow should reuse those helpers and keep `_write_exemplar()` as the only final persistence path.

## Phase 1: Define the Draft Contract

Create a small in-memory draft representation containing:

```text
name
model
data
description
source_session
```

Preparation must not create the destination folder or write any exemplar files. A cancelled or closed review must discard the draft without side effects.

## Phase 2: Add Description Distillation

Add a focused GenAI operation for producing the description.

### Inputs

- Current model text
- Current data text
- Current session transcript or session content
- Three retrieved exemplar triplets from `pyopl/opl_models`

### Prompt requirements

The prompt should require the model to:

- Return only a concise plain-text problem description.
- Describe decisions, constraints, objective, and important data relationships.
- Match the style of the existing `.md` examples.
- Avoid reproducing OPL syntax.
- Avoid mentioning the session, LLM, or prompting process.
- Generalize implementation-specific names where practical.
- Stay within a defined output-length limit.

Use the existing RAG retrieval path to select the three examples. If fewer than three examples are available, use the available examples and report that condition through logging or status text. The prompt should state clearly that the examples should only be used for styling and should not affect the generated problem description, which must be ground on current model and data text, as well as session information.

### Failure behavior

- If no GenAI provider/model is selected, show an actionable error.
- If generation fails, do not save an exemplar.
- Preserve the current model and data in the review flow where practical, but require a non-empty description before acceptance.
- Normalize the response so markdown fences and accidental leading labels do not become part of the saved description.

## Phase 3: Separate Preparation from Persistence

Refactor `save_exemplar()` into these stages:

```text
collect name
-> save/flush current session state
-> collect model and data
-> distill description
-> open review dialog
-> accept draft
-> revalidate destination
-> write triplet atomically
```

Required behavior:

- Do not create the destination during preparation.
- Do not overwrite an existing exemplar.
- Re-check the destination immediately before saving.
- Treat Cancel, window close, and Escape as discard operations.
- Save the edited description with the original model and data snapshots.
- Keep `_write_exemplar()` responsible for atomic file publication and temporary-folder cleanup.

## Phase 4: Build the Three-Tab Review Popup

Implement a modal `tk.Toplevel` using the existing `ttk.Notebook` patterns.

### Description tab

- Editable text widget.
- Initially selected when the window opens.
- Displays the LLM-distilled description.
- Provides the value that will be written to `<name>.md`.

### Model tab

- Read-only, syntax-highlighted text widget containing the proposed model.
- The original model snapshot is written to the saved `.mod` file.

### Data tab

- Read-only, syntax-highlighted text widget containing the proposed data.
- The original data snapshot is written to the saved `.dat` file.

### Dialog actions

- `Accept & Save`
- `Cancel`

The dialog should be transient to the IDE, grab input while open, and return an accepted draft to the caller rather than writing files itself. The Accept action should be disabled while required fields are invalid. Filesystem errors should use the existing error-reporting style.

Align dialog style with the rest of the IDE.

## Phase 5: Validate the Draft

Before acceptance, validate:

- Exemplar name is valid.
- Description is non-empty.
- Model is non-empty.
- Data is valid according to the existing application rules, including whether empty data is permitted.
- Model and data can be encoded as UTF-8.
- Destination does not already exist.

Optionally run the existing parser/compiler validation against edited model and data. Initially treat validation failures as warnings unless the application already requires successful compilation for an exemplar.

## Phase 6: Preserve Provenance Correctly

The session transcript should be provided to the distillation prompt but should not be written into the `.md` description. RAG retrieval expects `.md` files to contain problem descriptions; storing full transcripts would reduce retrieval quality.

If provenance is needed later, add a separate metadata mechanism rather than changing the triplet contract. Possible metadata includes source session ID, creation time, selected provider/model, and validation status.

## Phase 7: Add Focused Tests

Extend the existing IDE tests with coverage for:

- Distillation prompt construction with exactly three exemplar triplets.
- Generated description entering the review draft.
- Cancellation writing no files.
- Window close and Escape writing no files.
- Acceptance saving edited description, model, and data.
- Destination revalidation after the review window is open.
- Missing GenAI selection producing a clear error.
- Distillation failure producing no exemplar.
- Atomic-write failure removing temporary artifacts.
- Existing name and destination validation remaining intact.

Keep popup tests lightweight by extracting draft and acceptance logic into testable helpers or using fake Tk widgets where appropriate. Avoid requiring a display server for ordinary unit tests.

## Phase 8: Documentation and Release Notes

Update the user documentation to explain that:

- `Save Exemplar...` first generates a description for review.
- The description, model, and data can be reviewed and edited in separate tabs.
- Nothing is saved until `Accept & Save` is selected.
- The `.md` file contains the distilled problem description, not the raw session transcript.

Add the completed behavior to the `Unreleased` section of `CHANGELOG.md` under `Changed` or `Added`.

## Recommended Delivery Order

1. Extract or reuse exemplar loading and add the draft contract.
2. Implement the distillation prompt and response normalization.
3. Add prompt and distillation unit tests.
4. Implement the three-tab review dialog.
5. Refactor `save_exemplar()` around draft preparation and acceptance.
6. Add cancellation, acceptance, and persistence tests.
7. Update user documentation and `CHANGELOG.md`.
8. Run focused IDE tests, then the broader affected test suite.

## Completion Criteria

The work is complete when:

- Saving an exemplar never writes files before explicit acceptance.
- The description is LLM-distilled from the session with three-shot context from existing exemplars.
- The user can review and edit description, model, and data independently.
- Accepted edits are saved as one atomic exemplar triplet.
- Cancelled, closed, invalid, and failed flows leave no partial exemplar behind.
- Tests cover both the review gate and the existing atomic persistence guarantees.

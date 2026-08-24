# Exemplar Retrieval Roadmap

## Objective

Add a File-menu workflow for finding and loading reusable OPL exemplars from the packaged `pyopl/opl_models` collection and an optional `./opl_models` collection in the current working directory.

## Delivered

- Added `Retrieve Exemplar...` immediately after `Open...` in the File menu.
- Added a modal popup with a search field, scrollable model list, Retrieve button, and Cancel behavior.
- Discover complete exemplar triplets (`.txt` description, `.mod` model, and `.dat` data) recursively in both supported roots.
- Show all discovered exemplars alphabetically when the popup opens.
- Reuse `rank_problem_descriptions`, the existing semantic RAG scorer, to order results for non-empty searches.
- Run semantic ranking in a background thread and ignore stale results when the user changes the query quickly.
- Load the selected model and data through the IDE's existing editor path.
- Added regression tests for menu placement and preservation of RAG ranking order.

## Follow-up Phases

### 1. Interaction polish

- Add a description preview pane for the selected exemplar.
- Display the source collection or relative path when names are duplicated.
- Add keyboard-focused list navigation and a clearer loading state.

### 2. Search resilience

- Decide whether unavailable embedding dependencies should show an actionable warning or a lexical fallback.
- Add tests for duplicate roots, incomplete triplets, empty descriptions, and ranking failures.
- Consider debouncing keystrokes if model encoding becomes expensive on larger collections.

### 3. Retrieval safety and workflow integration

- Confirm behavior when the current editor has unsaved changes before replacing it.
- Add an explicit status message or confirmation when model and data are loaded from different roots.
- Verify packaged-resource discovery after building and installing a wheel.

### 4. End-to-end validation

- Exercise the popup manually on macOS with packaged and local exemplars.
- Validate search ordering with the configured embedding model and a representative set of optimization queries.
- Run the broader IDE and GenAI test suites after any changes to shared RAG helpers.

## Validation Baseline

The focused IDE unittest module currently passes with `58` tests:

```text
./venv/bin/python -m unittest test.test_pyopl_ide_typing
```

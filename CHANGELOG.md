# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added focused regression coverage for SciPy CSC variable-domain resolution, iterator unrolling, bound evaluation, expression emission, and Boolean dispatch helpers.

### Fixed

- Isolated exemplar embedding searches from the Tk process, cleaned up active search workers when the dialog closes, and debounced searches while typing to prevent native shutdown crashes.

### Changed

### Removed

## [v2.4.2] - 2026-08-24

### Added

- Added regression coverage for fully indexed Gurobi constraint names used in IIS reporting.
- Added standard Cut, Copy, and Paste Edit-menu commands, plus `Ctrl/Cmd+/` and `Ctrl/Cmd+.` shortcuts for toggling hash comments and `§` paragraph markers in the model and data editors.
- Added Edit-menu commands and `Ctrl/Cmd+]` and `Ctrl/Cmd+[` shortcuts for indenting and unindenting editor lines.

### Fixed

- Made comment and paragraph operations atomic for undo and preserve active multi-line selections.
- Ensured generated hash comments always begin at column zero while preserving indentation when uncommented.
- Ensured line-based editor commands respect Tk's end-exclusive multi-line selections.
- Preserved commented content indentation when removing paragraph markers.

### Changed

- Made the Rhetor File menu platform-appropriate by using the native macOS application-menu Quit command and retaining File-menu termination commands on Windows and Linux.
- Added platform-aware `Ctrl/Cmd+M` and `Ctrl/Cmd+D` accelerators and keyboard shortcuts for opening model and data files.

### Removed

## [v2.4.1] - 2026-08-24

### Added

- Added a three-tab review gate for saving GenAI-distilled exemplar descriptions with syntax-highlighted model and data previews.

### Fixed

### Changed

- Changed `Save Exemplar...` to generate descriptions without blocking the IDE, require explicit acceptance before atomically saving a triplet, and avoid storing raw session transcripts as exemplar descriptions.

### Removed

## [v2.4.0] - 2026-08-24

### Added

- Added a File menu command for saving the current model, data, and complete session as a working-directory RAG exemplar triplet.
- Added compile, alignment, and bounded repair validation for model/data revisions proposed by `generative_feedback`.
- Added Gurobi IIS diagnostics for infeasible solves, including conflicting linear and general constraints and variable bounds in the IDE output.

### Fixed

- Documented and scoped the Rhetor IDE bridge URL opening to its validated HTTP loopback endpoint for security scanning.
- Narrowed optional session snapshot hashes before dictionary lookup to satisfy static type checking.
- Invalidated newly selected session preview, session diff, and GenAI review tabs so their text paints without requiring mouse entry on macOS.
- Named unsaved model and data artifacts with a shared solve-time `tmp_YYYY-MM-DD_HH-MM-SS` basename.
- Added visual separation between solver statistics and the top-level solver message in IDE output.

### Changed

- Expanded default GenAI RAG retrieval to recursively include model/data/description triplets from an optional working-directory `opl_models` folder.
- Centralized `generative_feedback` so every GenAI strategy exposes one validated implementation.
- Deduplicated output-session model and data snapshots through content-addressed SHA-256 stores in `.pyopl_session`.
- Simplified IDE session-history restoration by isolating snapshot and artifact normalization.

### Removed

## [v2.3.0] - 2026-08-23

### Added

- Added regression coverage for exact continuous strict implications, nested syntax, and backend bound requirements.
- Added a tracked roadmap for implication parity and shared numerical-policy constants.
- Added right-associative nested implications and cross-backend parity tests for Boolean forms and all six integer comparison operators.
- Added shared affine interval utilities and regression coverage for interval containment, non-finite inputs, indexed aggregation, and bound-analysis exception handling.
- Added DOcplex cross-check coverage with environment-gated CPLEX runner configuration and a generic reproducibility roadmap for the dedicated tests.

### Fixed

- Fixed right-hand binary consequent generation for specialized Gurobi implication indicators and cleaned up DOcplex cross-check lint failures.
- Made Gurobi equality and composite implication antecedents biconditional, removed generic arbitrary big-M fallback, and added SciPy equality consequents.
- Lowered strict affine-to-binary implications exactly using Gurobi contrapositives and SciPy finite-bound rows, removing the continuous epsilon dead zone for supported on/off patterns.
- Rejected unsupported continuous comparison truth reification instead of silently approximating it with an epsilon-separated feasible set.
- Hardened SciPy big-M bound derivation for parser unary/parenthesized expressions, conservative indexed fallback bounds, non-finite values, and partially initialized metadata.

### Changed

- Consolidated runtime and development dependency declarations in `pyproject.toml`; CI and documentation now install from package metadata.

### Removed

- Removed the duplicate `requirements.txt` dependency manifest.

## [v2.2.0] - 2026-08-21

### Added

- Added drag-and-drop opening for model/data files and visual GenAI attachments in the IDE.
- Added line-number gutters and explicit `§` model-section folding to the model and data editors.

### Fixed

- Made `§` section folds include an immediately following brace block, including nested sections and blank paragraphs.
- Preserved folded `§` sections when line insertions or deletions shift their ranges in the IDE.
- Persisted folded sections for up to 100 recently edited files in the workspace-local `.pyopl_session` file.
- Disabled GenAI composer inputs and model/data editors while Generate or Ask requests are active.
- Restored the Solve menu's Stop lifecycle after the command was renamed and shortened its running label to "Stop".
- Corrected static typing for IDE drag-and-drop widget registration.

### Changed

- Documented model/data comment syntax and clarified the IDE-only semantics of `§` section markers in the PyOPL grammar.

### Removed

## [v2.1.0] - 2026-08-21

### Added

- Added MCP tools for reading and writing the live Rhetor IDE model and data editors.
- Expanded tuple-array declarations and tuple comprehensions.
- Added strict tuple schema validation and broader tuple feature coverage.
- Added additional boolean reification coverage for Gurobi code generation.

### Fixed

- Preserve syntax-error tracebacks in the IDE output history before the model compilation failure message.
- Prevented explicit MCP `null` editor updates from clearing or partially changing Rhetor IDE editor contents.
- Hardened tuple-array handling and tuple validation across solver backends.
- Solver results now report only variables declared in the PyOPL model, excluding linearisation auxiliaries.

### Changed

- Streamlined solver helper logic and improved boolean reification behavior.
- Simplified Gurobi boolean-expression code generation to reduce control-flow complexity.
- Renamed the IDE run-menu command to "Solve" for a more concise label.
- Corrected solver-settings template links and wording in the user guide.

## [v2.0.0] - 2026-08-21

### Added

- Added backend-native solver settings support for Gurobi and HiGHS.
- Added HiGHS solver output and improved solver timing and status reporting.
- Added filtered-iterator environment generation for more efficient sparse model compilation.
- Modularized IDE support, tuple-set helpers, and MILP equivalence proof helpers.

### Fixed

- Hardened compiler validation and boolean simplification helpers.
- Improved comprehension evaluation, computed parameter maps, iterator typing, and conditional rewriting.

### Changed

- Improved type annotations and code formatting across the modeling and equivalence helpers.

### Removed

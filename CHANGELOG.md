# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added regression coverage for exact continuous strict implications, nested syntax, and backend bound requirements.
- Added a tracked roadmap for implication parity and shared numerical-policy constants.
- Added right-associative nested implications and cross-backend parity tests for Boolean forms and all six integer comparison operators.
- Added shared affine interval utilities and regression coverage for interval containment, non-finite inputs, indexed aggregation, and bound-analysis exception handling.

### Fixed

- Made Gurobi equality and composite implication antecedents biconditional, removed generic arbitrary big-M fallback, and added SciPy equality consequents.
- Lowered strict affine-to-binary implications exactly using Gurobi contrapositives and SciPy finite-bound rows, removing the continuous epsilon dead zone for supported on/off patterns.
- Rejected unsupported continuous comparison truth reification instead of silently approximating it with an epsilon-separated feasible set.
- Hardened SciPy big-M bound derivation for parser unary/parenthesized expressions, conservative indexed fallback bounds, non-finite values, and partially initialized metadata.

### Changed

### Removed

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

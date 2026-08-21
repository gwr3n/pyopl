# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Expanded tuple-array declarations and tuple comprehensions.
- Added strict tuple schema validation and broader tuple feature coverage.
- Added additional boolean reification coverage for Gurobi code generation.

### Fixed

- Hardened tuple-array handling and tuple validation across solver backends.
- Solver results now report only variables declared in the PyOPL model, excluding linearisation auxiliaries.

### Changed

- Streamlined solver helper logic and improved boolean reification behavior.
- Renamed the IDE run-menu command to "Solve" for a more concise label.
- Corrected solver-settings template links and wording in the user guide.

## [2.0.0] - 2026-08-21

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

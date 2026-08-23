# DOcplex Cross-Check Roadmap

## Objective

Cross-check the PyOPL results for every problem-oriented test in
[`test/test_problems.py`](../test/test_problems.py) against an equivalent
DOcplex model. Preserve the same input data, objective direction, feasibility
requirements, and relevant variable bounds in both implementations.

## Reference Verification Environment

The following reference environment was used for the verification results
recorded below. It documents one successful setup; these exact versions and
paths are not requirements for other platforms.

- OS: macOS on Apple Silicon.
- Python environment: `venv-310`.
- Python: `3.10.20`.
- DOcplex: `2.32.264`.
- CPLEX Python bindings: `22.1.1.0`.
- CPLEX Studio: `CPLEX_Studio2211`.
- Native OPL runner: the Studio runner for the host platform.
- PyOPL backends: SciPy/HiGHS and Gurobi.

CPLEX Studio 22.1.1 Python bindings provide wrappers for Python 3.8, 3.9,
and 3.10. Select a Python version supported by the installed CPLEX release;
the version used by the main project environment may not be supported.

### Environment setup

Create a project-local virtual environment using a Python version supported by
the installed CPLEX release. The following example uses Python 3.10 and names
the environment `venv-310`; choose another name if appropriate:

```sh
python3.10 -m venv venv-310
PYTHON=./venv-310/bin/python
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install "docplex==2.32.264"
"$PYTHON" --version
"$PYTHON" -c \
   "import docplex, cplex; print(docplex.__version__, cplex.__version__)"
```

CPLEX Python bindings and the native OPL runner are external CPLEX Studio
components and may require a separate licensed installation. If the Python
wrapper is absent, install the platform-specific wrapper from the local Studio
installation. Replace the placeholders with paths supplied by that
installation:

```sh
"$PYTHON" -m pip install \
   /path/to/CPLEX_Studio/cplex/python/3.10/<platform>
```

Use the wrapper directory matching the operating system, architecture, and
Python version. Avoid an unqualified top-level `setup.py` when the Studio
installation contains more than one platform wrapper; select the matching
platform directory explicitly.

Set the native OPL runner path for the current platform before running the
cross-checks:

```sh
export CPLEX_STUDIO=/path/to/CPLEX_Studio
export OPLRUN="$CPLEX_STUDIO/opl/bin/<platform>/oplrun"
"$OPLRUN" -version
```

The dedicated test module reads both paths from environment variables. All
tests in the module are skipped when either variable is unset. The module uses
`OPLRUN` to invoke the native runner, while `CPLEX_STUDIO` explicitly
identifies the matching Studio installation; set both variables to the same
installation before running the tests.

## DOcplex/OPL Compatibility

The intended DOcplex API is `docplex.mp.model_reader.ModelReader`, with
`build_opl_model(mod, data)` accepting OPL model and data files. In this
environment, DOcplex 2.32.264 invokes an `oplcpolpgen` executable that is not
provided by CPLEX Studio 22.1.1. Studio 22.1.1 provides `oplrun` instead.

The dedicated cross-check module therefore uses this verified compatibility
path:

1. Write the model and data strings to temporary `.mod` and `.dat` files.
2. Preserve the exact files for both PyOPL solver calls.
3. If Studio requires a compatibility adjustment for a tuple-indexed data
    array, derive a temporary DOcplex-only `.dat` file. Do not alter the data
    used by PyOPL.
4. Run `oplrun -e temporary.lp model.mod data.dat`.
5. Load the exported LP with `ModelReader.read()`.
6. Solve the imported DOcplex model with CPLEX.
7. Map each PyOPL declared-variable solution into a DOcplex `new_solution()`.
    Match names exactly first, then normalize punctuation, brackets, quotes, and
    integral float formatting such as `8.0` versus `8`.
8. Require the candidate solution to pass `is_valid_solution()` and evaluate
    the objective with `candidate.get_value(model.objective_expr)`.

## Comparison protocol

For each target:

1. Recreate the model and data in DOcplex, keeping names and index structure as
   close to the PyOPL source as practical.
2. Solve with PyOPL's supported backends and with DOcplex/CPLEX.
3. Compare solve status and objective value using a documented numerical
   tolerance. For models with multiple optimal solutions, compare constraint
   feasibility and objective value rather than exact variable assignments.
4. Record solver versions, runtime, status, objective values, tolerance, and
   any unsupported-language differences.
5. Add a focused regression test or fixture for each completed cross-check.

The dedicated tests must remain separate from `test/test_problems.py`. Do not
add cross-check imports, helpers, or test methods to the existing problem test
module. Put new work in
[`test/test_docplex_cross_checks.py`](../test/test_docplex_cross_checks.py).

## Resume Workflow

To continue one model at a time:

1. Select the next unchecked method in the inventory.
2. Read that method in `test/test_problems.py` and copy its model/data strings
   into the dedicated cross-check module. Keep the source test unchanged.
3. Run the original PyOPL model with both `scipy` and `gurobi`.
4. Run the same OPL model through the native `oplrun` export and DOcplex/CPLEX
   path described above.
5. Compare statuses and objective values with a tolerance of `1e-6` unless
   the model requires a documented alternative.
6. Evaluate every PyOPL solution as a fixed DOcplex candidate and check
   feasibility plus objective value.
7. Run only the relevant dedicated test first, then broader tests if needed.
8. Record the result, environment, compatibility adaptations, and any
   unsupported construct in this roadmap before checking the inventory item.

Focused command:

```sh
"$PYTHON" -m unittest \
  test.test_docplex_cross_checks.TestDocplexCrossChecks.test_v2_features -v
```

Run the complete dedicated cross-check module with:

```sh
"$PYTHON" -m unittest test.test_docplex_cross_checks -v
```

The existing PyOPL problem test can be run independently with:

```sh
"$PYTHON" -m unittest \
  test.test_problems.TestPyOPLProblems.test_v2_features -v
```

The final report should identify targets that cannot be translated directly to
DOcplex because they are parser, AST, code-generation, or error-handling tests.
Those tests remain in the inventory and receive an explicit applicability
decision.

## Work phases

- [x] Inventory and classify all targets.
- [x] Add shared comparison helpers and numerical-tolerance policy.
- [x] Cross-check small baseline models first. The first target is
   `test_v2_features`.
- [x] Cross-check tuple, string-indexed, and nested-data models.
- [x] Cross-check scheduling, routing, inventory, and stochastic models.
- [x] Review unsupported or non-optimization tests and document decisions.
- [x] Run the complete cross-check suite and publish the results.

## Problem inventory

All entries below are methods in [`test/test_problems.py`](../test/test_problems.py).
The checkbox tracks the DOcplex cross-check, not the existing PyOPL test.

### Stochastic and larger optimization models

- [x] `test_stochastic_economic_lot_scheduling` (line 118)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_economic_lot_scheduling`.
   - Verified on 2026-08-22 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy objective: `1790.1485714285648`; Gurobi objective:
      `1790.1485714285714`; CPLEX objective: approximately
      `1790.14857142857`.
   - Both PyOPL assignments passed DOcplex feasibility validation and their
      fixed-solution objective evaluations matched the CPLEX objective within
      `1e-6`.
   - Studio 22.1.1 requires OPL declarations such as `float x[...]` rather than
      PyOPL's accepted `param float x[...]` spelling in the DOcplex fixture.
   - Studio exports some multidimensional arrays with zero-based hash names
      such as `a#0#0#0`; the dedicated mapper converts PyOPL's one-based `a`
      names before matching them.
- [x] `test_stochastic_job_shop_scheduling_2` (line 350)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_job_shop_scheduling_2`.
   - Verified on 2026-08-22 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `21.75`.
   - Both PyOPL decision assignments (`start` and `z`) were fixed in DOcplex,
      solved successfully, and matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 requires scalar inputs and scenario-indexed arrays in the
      DOcplex fixture to use compatible declarations/positional data. The
      original keyed data remains unchanged for PyOPL; the test derives a
      temporary DOcplex-only data file.
- [x] `test_stochastic_job_shop_scheduling` (line 549)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_job_shop_scheduling`.
   - Verified on 2026-08-22 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `21.75`.
   - Each PyOPL start-time schedule was fixed in a fresh DOcplex model and
      solved successfully, matching the CPLEX objective within `1e-6`.
   - Studio 22.1.1 requires positional scenario arrays in the DOcplex
      fixture. The original keyed `prob` and `p` data remains unchanged for
      PyOPL; the test derives a temporary DOcplex-only data file.
   - The model declares sequencing variables for unconstrained/self or
      cross-machine operation pairs. The mapper ignores those omitted
      variables, and CPLEX completes the non-unique sequencing choices.
- [x] `test_stochastic_plane_landing_2` (line 738)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_plane_landing_2`.
   - Verified on 2026-08-22 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `5.0`.
   - Both PyOPL assignments were fixed in fresh DOcplex models and solved
      successfully, matching the CPLEX objective within `1e-6`.
   - Studio 22.1.1 requires integer `0/1` feasibility masks instead of
      Boolean parameters, and positional rather than keyed array data. The
      original Boolean/keyed data remains unchanged for PyOPL; the test
      derives a temporary DOcplex-only compatibility data/model path.
- [x] `test_stochastic_vrp` (line 910)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_vrp`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `54.0`.
   - Both PyOPL solutions were fixed in fresh DOcplex models for route,
      serving, and delivery decisions; each remained feasible and matched the
      CPLEX objective within `1e-6`.
   - The exact keyed scenario data remains the PyOPL fixture. The test derives
      a temporary DOcplex-only positional array fixture for Studio 22.1.1.
- [x] `test_multistage_stochastic_portfolio` (line 1191)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_multistage_stochastic_portfolio`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective approximately
      `102.504950495`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The scenario-pair information links, proportional transaction costs, and
      expected-wealth floor were accepted directly by native `oplrun`; no
      DOcplex-only data adaptation was required.
- [x] `test_stochastic_multi_echelon` (line 1386)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_multi_echelon`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `2127.75`.
   - Both PyOPL assignments were mapped into fresh DOcplex solutions, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The exact source model and data are extracted for the PyOPL runs. The
      temporary native OPL fixture uses Studio-compatible declarations,
      renamed tuple fields, inline positional numeric parameters, and a
      renamed probability parameter; these adaptations preserve the same
      sets, values, constraints, and objective.
- [x] `test_stochastic_plane_landing` (line 1713)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_plane_landing`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 required removal of the labeled objective and positional
      array data in the temporary DOcplex fixture; the original keyed data
      remains unchanged for PyOPL.
- [x] `test_hotel_rostering` (line 1912)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_hotel_rostering`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
   - This large source test is skipped by `test_problems.py` as cumbersome;
      the dedicated cross-check also fixes each PyOPL assignment in DOcplex,
      confirms feasibility, and matches the CPLEX objective within `1e-6`.
   - The initial fixed-assignment failure was caused by the shared mapper:
      normalized one-based names such as `x_5_16` were matched before
      zero-based exported names such as `x#5#16`, causing distinct PyOPL
      variables to collide. Conflict refinement identified the shift-17
      coverage row and collided `x#5#16`/`x#11#16` values. The mapper now
      prefers exact and zero-based exported-name matches before normalization.
   - Native OPL required the derived `shiftConflict` parameter to use integer
      values with an explicit `== 1` filter; the source fixture is unchanged.
- [x] `test_portfolio_diversification` (line 2263)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_portfolio_diversification`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective
      `104.98674763599055`.
   - Both PyOPL portfolio assignments were accepted as feasible DOcplex
      candidates and their objective evaluations matched CPLEX within `1e-6`.
   - Studio 22.1.1 requires integer `0/1` leaf flags instead of Boolean
      parameters and positional rather than keyed tuple-array data. The
      original Boolean/keyed fixture remains unchanged for PyOPL; the test
      derives a temporary DOcplex-only model/data pair.
- [x] `test_RMAB_relaxation_tuples` (line 2439)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_RMAB_relaxation_tuples`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `2.25`.
   - Both PyOPL occupation-measure solutions were accepted as feasible
      DOcplex candidates and their objective evaluations matched CPLEX within
      `1e-6`.
   - Studio 22.1.1 requires positional rather than keyed multidimensional
      parameter data. The original tuple-keyed `cost` and `P` fixture remains
      unchanged for PyOPL; the test derives a temporary DOcplex-only data file.
- [x] `test_RMAB_relaxation_dense` (line 2612)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_RMAB_relaxation_dense`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `2.25`.
   - Both PyOPL occupation-measure solutions were accepted as feasible
      DOcplex candidates and their objective evaluations matched CPLEX within
      `1e-6`.
   - The dense positional `cost` and `P` data is accepted directly by Studio
      22.1.1; no DOcplex-only compatibility adaptation was required.
- [x] `test_column_generation` (line 2787)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_column_generation`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - Both gated modes matched across all solvers: the master mode
      (`RunPricing = 0`) produced objective `3.0`, and the pricing mode
      (`RunPricing = 1`) produced objective `-0.1`.
   - Both PyOPL assignments in both modes were fixed in cloned DOcplex
      models, remained feasible, and matched the corresponding CPLEX
      objective within `1e-6`.
   - Studio 22.1.1 accepts the dense positional `itemLen`, `demand`, `a`, and
      `dual` data directly; the source keyed data remains unchanged for PyOPL
      and the test derives a temporary DOcplex-only data file.
- [x] `test_TOPSIS` (line 2987)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_TOPSIS`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all selected alternative 3 with objective
      `0.6459025501213678`.
   - Both PyOPL selections were accepted as feasible DOcplex candidates and
      their objective evaluations matched CPLEX within `1e-6`.
   - Studio 22.1.1 requires integer `0/1` orientation flags instead of
      Boolean parameters. The original Boolean fixture remains unchanged for
      PyOPL; the test derives a temporary DOcplex-only model/data pair.
- [x] `test_asset_location` (line 3128)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_asset_location`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The source tuple-generated cells, arcs, edges, computed permissions,
       flow connectivity, and simple-path constraints were accepted directly
       by Studio 22.1.1; no DOcplex-only data adaptation was required.
- [x] `test_vrp` (line 3343)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_vrp`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The temporary DOcplex fixture renames the reserved tuple field `to`,
       uses integer `0/1` depot flags, positional parameter arrays, explicit
       external data declarations, and a flattened equivalent MTZ constraint.
       The exact source model and data remain unchanged for PyOPL.
- [x] `test_vrp_2` (line 3527)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_vrp_2`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The numeric node indexing, distance matrix, vehicle constraints, and
       MTZ subtour elimination were accepted directly by Studio 22.1.1; no
       DOcplex-only model or data adaptation was required.
- [x] `test_stochastic_lot_sizing` (line 3629)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_stochastic_lot_sizing`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The temporary DOcplex fixture uses explicit external declarations,
       positional scenario probabilities and demand arrays, and equivalent
       explicit nonanticipativity equalities because Studio 22.1.1 does not
       support a SUM expression in an aggregate filter. The source fixture
       remains unchanged for PyOPL.
- [x] `test_newsvendor` (line 3768)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_newsvendor`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The temporary DOcplex fixture converts Python-style comments and uses
       explicit scalar data declarations for Studio 22.1.1; the source model
       and commented data remain unchanged for PyOPL.
- [x] `test_static_stochastic_knapsack` (line 3869)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_static_stochastic_knapsack`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The temporary DOcplex fixture uses positional scenario probabilities,
       weights, and values plus explicit external declarations for Studio
       22.1.1; the chance constraint, computed expected values, and Big-M
       coefficients preserve the source formulation.
- [x] `test_p_dispersion` (line 4000)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_p_dispersion`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, and Gurobi.
    - Gurobi and CPLEX both produced the optimal objective `31.0`.
    - The PyOPL Gurobi assignment was fixed in a DOcplex model, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - SciPy remains intentionally excluded because the source test documents
       that this implication class is unsupported by the SciPy backend. The
       temporary DOcplex fixture uses a concise equivalent formulation with
       explicit labels and Studio-compatible declarations.
- [x] `test_complex_workforce_planning_3` (line 4079)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_complex_workforce_planning_3`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The source training, staffing, production-capacity, and backlog model
       was accepted directly by Studio 22.1.1 after the standard external
       parameter declaration and objective-label adaptations.
- [x] `test_complex_workforce_planning_2` (line 4316)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_complex_workforce_planning_2`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The source training, workforce deployment, production, and
       inventory/backlog model was accepted by Studio 22.1.1 after the standard
       external parameter declaration and objective-label adaptations.
- [x] `test_complex_workforce_planning_1` (line 4456)
    - Dedicated cross-check: `TestDocplexCrossChecks.test_complex_workforce_planning_1`.
    - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
       22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
    - SciPy, Gurobi, and CPLEX all produced matching optimal objectives.
    - Both PyOPL assignments were fixed in cloned DOcplex models, remained
       feasible, and matched the CPLEX objective within `1e-6`.
    - The temporary DOcplex fixture renames reserved `prod`, externalizes the
       rate, penalty, and demand arrays, and uses Studio-compatible indexed
       sums. Omitted zero delivery variables are ignored only when native OPL
       does not export them; nonzero omissions fail validation.

### Core planning, routing, and graph models

- [x] `test_wagner_whitin_backorders` (line 5057)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_wagner_whitin_backorders`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `980.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The DOcplex fixture uses external Studio-compatible parameter
      declarations and an equivalent explicit first-period balance; the
      source `param` declarations and conditional balance remain unchanged
      for PyOPL.
- [x] `test_production_planning_conditional_compare_solvers_1` (line 5137)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_production_planning_conditional_compare_solvers_1`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `30.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The DOcplex fixture uses external Studio-compatible parameters and an
      equivalent explicit first-period balance; the source conditional model
      remains unchanged for PyOPL.
- [x] `test_production_planning_conditional_compare_solvers_2` (line 5198)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_production_planning_conditional_compare_solvers_2`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy and Gurobi solved the source PyOPL model, while CPLEX solved the
      exact bounded compatibility formulation; all produced objective `30.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 rejects the source strict comparison on `dvar float+`.
      With the source bounds
      `0 <= Q[t] <= 410`, the implication is exactly equivalent to
      `Q[t] <= 410 * order[t]`, which is used by the CPLEX fixture and the
      SciPy lowering. Gurobi uses the equivalent contrapositive indicator
      `order[t] == 0 => Q[t] <= 0`.
- [x] `test_production_planning_compare_solvers` (line 5259)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_production_planning_compare_solvers`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `240.0`.
   - The PyOPL assignment was fixed in a cloned DOcplex model, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The positional multidimensional cost, demand, and capacity data was
      accepted directly by Studio 22.1.1; no DOcplex-only adaptation was
      required.
- [x] `test_vehicle_routing_with_nested_tuples_dat` (line 5319)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_vehicle_routing_with_nested_tuples_dat`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `30.5`.
   - The PyOPL routing assignment was fixed in a cloned DOcplex model,
      remained feasible, and matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 treats the nested tuple field name `to` as reserved. The
      exact source fixture remains unchanged for PyOPL; the DOcplex path uses
      a temporary model with that field renamed to `destination`.
- [x] `test_vehicle_routing_with_nested_tuples` (line 5514)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_vehicle_routing_with_nested_tuples`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `30.5`.
   - The PyOPL routing assignment was fixed in a cloned DOcplex model,
      remained feasible, and matched the CPLEX objective within `1e-6`.
   - The inline nested tuple sets were accepted directly by Studio 22.1.1
      after renaming the reserved tuple field `to` to `destination` in the
      temporary DOcplex fixture; the source model remains unchanged.
- [x] `test_wagner_whitin_linear` (line 5613)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_wagner_whitin_linear`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `630.0`.
   - The complete PyOPL assignment was fixed in cloned DOcplex models,
      remained feasible, and matched the CPLEX objective within `1e-6`.
   - The exact source fixture uses inline data; the DOcplex path derives a
      temporary model/data pair with those same values externalized for
      native `oplrun`.
- [x] `test_wagner_whitin_model_data` (line 5705)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_wagner_whitin_model_data`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `630.0`.
   - The complete PyOPL assignment was fixed in a cloned DOcplex model,
      remained feasible, and matched the CPLEX objective within `1e-6`.
   - The external positional data was accepted directly by Studio 22.1.1;
      the source fixture and its expected assignment remain unchanged.
- [x] `test_wagner_whitin_implication` (line 5808)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_wagner_whitin_implication`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy and Gurobi solved the source PyOPL model, while CPLEX solved the
      exact bounded compatibility formulation; all produced objective `630.0`.
   - The complete PyOPL assignment was fixed in a cloned DOcplex model,
      remained feasible, and matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 rejects the source continuous strict comparison
      `x[t] > 0`. With `0 <= x[t] <= 150` and
      binary `y[t]`, the source implication is exactly equivalent to
      `x[t] <= 150 * y[t]`, which is used by the CPLEX fixture and the SciPy
      lowering. Gurobi uses the equivalent contrapositive indicator
      `y[t] == 0 => x[t] <= 0`.
- [x] `test_job_shop` (line 5920)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_job_shop`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced makespan `13.0`.
   - Both PyOPL schedules were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The cross-check fixes only `start` decisions because the full ordered-pair
      `z` disjunction array contains non-applicable/self-pair variables and is
      non-unique; CPLEX completes those sequencing choices.
- [x] `test_warehouse_location` (line 5992)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_warehouse_location`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `270.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The model's positional numeric data is accepted directly by Studio 22.1.1;
      no DOcplex-only compatibility data adaptation was required.
- [x] `test_graph_coloring_tuples` (line 6056)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_graph_coloring_tuples`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `2.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The tuple-indexed `z` variables were mapped successfully through the
      shared name normalizer; no DOcplex-only data adaptation was required.
- [x] `test_graph_coloring_matrix` (line 6121)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_graph_coloring_matrix`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `2.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - Native OPL exports only adjacency-constrained `z` variables; the
      cross-check omits self and non-edge auxiliaries that are unconstrained
      in the exported model. The source fixture remains unchanged.
- [x] `test_vehicle_routing_matrix_dat` (line 6184)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_vehicle_routing_matrix_dat`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `30.5`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The external positional matrix data was accepted directly by Studio
      22.1.1; no DOcplex-only compatibility adaptation was required.
- [x] `test_vehicle_routing_with_tuples_dat` (line 6236)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_vehicle_routing_with_tuples_dat`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `30.5`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The exact external tuple data remains the PyOPL fixture. The temporary
      DOcplex model renames the reserved tuple field `to`; no data adaptation
      was required.
- [x] `test_vehicle_routing_with_tuples` (line 6285)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_vehicle_routing_with_tuples`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `30.5`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The inline tuple set is accepted directly by Studio 22.1.1; the
      temporary DOcplex model renames the reserved tuple field `to`.
- [x] `test_basic_production_planning_gurobi` (line 6378)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test only parses the model and asserts that
      `GurobiCodeGenerator.generate_code()` returns a string; it does not
      solve the model or expose a result to compare with CPLEX.
   - The original focused test remains the regression coverage for this
      Gurobi code-generation contract.
- [x] `test_basic_production_planning_scipy` (line 6406)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test only parses the model and asserts that
      `SciPyCodeGenerator.generate_code()` returns a string; it does not
      solve the model or expose a result to compare with CPLEX.
   - The original focused test remains the regression coverage for this
      SciPy code-generation contract.
- [x] `test_knapsack_pyopl_vs_cplex_output` (line 6464)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_knapsack_pyopl_vs_cplex_output`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `10.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The repository model and data files were used unchanged.
- [x] `test_knapsackp_pyopl_vs_cplex_output` (line 6475)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_knapsackp_pyopl_vs_cplex_output`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `498.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The exact repository model and data files remain unchanged for PyOPL. The
      temporary DOcplex fixture renames `Value` to avoid Studio's predefined
      data namespace and declares both `Value` and `Use` arrays with `= ...;`,
      which is required for external OPL data binding.
- [x] `test_inventory_routing_pyopl_vs_cplex_output` (line 6486)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_inventory_routing_pyopl_vs_cplex_output`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `103.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The exact source model and keyed data remain unchanged for PyOPL. The
      temporary DOcplex fixture uses Studio-compatible `float` declarations
      and positional arrays; the source file's `param float` declarations,
      keyed arrays, and hash comments are not accepted by Studio 22.1.1.
- [x] `test_tsp_model_parsing_and_codegen_gurobi` (line 6502)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test validates TSP AST index constraints,
      Gurobi code-generation patterns, and generated Python syntax; it does
      not solve the model or expose a result to compare with CPLEX.
   - The original focused test remains the regression coverage for this
      parsing and Gurobi code-generation contract.
- [x] `test_tsp_model_parsing_and_codegen_scipy` (line 6542)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test validates TSP AST index constraints,
      SciPy code-generation patterns, and generated Python syntax; it does
      not solve the model or expose a result to compare with CPLEX.
   - The original focused test remains the regression coverage for this
      parsing and SciPy code-generation contract.
- [x] `test_knapsack_problem_compare_solvers` (line 6579)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_knapsack_problem_compare_solvers`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `10.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The generated source fixture and data are used unchanged for PyOPL. The
      temporary DOcplex fixture inlines the same data because Studio reports a
      duplicate-data error when these external arrays are supplied through the
      generated `.dat` file.
- [x] `test_knapsack_problem_scipy` (line 6637)
   - Applicability decision: not applicable as a separate DOcplex solver-result
      cross-check. The source test checks parser/code-generator return types
      and only verifies that `solve_with_scipy` returns a result containing a
      status; it does not assert an objective or compare a solution with CPLEX.
   - The exact generated knapsack formulation is covered by the dedicated
      `test_knapsack_problem_compare_solvers` cross-check above, while the
      original focused test remains the regression for the SciPy API contract.
- [x] `test_assignment_problem_compare_solvers` (line 6709)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_assignment_problem_compare_solvers`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `10.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`; exact variable
      equality was not required because the assignment model has alternate
      optimal solutions.
   - The generated 2x2 model has no external data and was accepted directly by
      Studio 22.1.1.
- [x] `test_knapsack_problem` (line 6765)
   - Applicability decision: not applicable as a separate DOcplex solver-result
      cross-check. The source test only checks parser/code-generator return
      types and does not solve the model or expose an objective for comparison
      with CPLEX.
   - The exact generated knapsack formulation is covered by the dedicated
      `test_knapsack_problem_compare_solvers` cross-check above, while the
      original focused test remains the regression for the Gurobi parsing API.
- [x] `test_multi_resource_knapsack_problem` (line 6801)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test only checks parser/code-generator return
      types for a multi-resource formulation; it does not solve the model or
      expose an objective for comparison with CPLEX.
   - The original focused test remains the regression coverage for this
      generated multi-resource Gurobi parsing and code-generation contract.
- [x] `test_transportation_problem` (line 6847)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test only invokes `run_test_case_gurobi`, which
      checks parsing and Gurobi code-generation return types without solving
      the transportation model or exposing an objective for CPLEX comparison.
   - The original focused test remains the regression coverage for this
      generated transportation Gurobi code-generation contract.
- [x] `test_simple_assignment_problem` (line 6865)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test only invokes `run_test_case_gurobi`, which
      checks parsing and Gurobi code-generation return types without solving
      the assignment model or exposing an objective for CPLEX comparison.
   - The equivalent solver behavior is covered by the dedicated
      `test_assignment_problem_compare_solvers` cross-check above, while the
      original focused test remains the regression for this code-generation
      contract.

### Small models and language-feature exercises

These entries should first receive an applicability decision. Where a test
does solve an optimization model, it can be promoted to a normal DOcplex
cross-check target.

- [x] `test_v2_features` (line 35)
   - Scope: use the exact `model_code` and `data_code` strings from the test.
   - Compare SciPy and Gurobi objective values with the DOcplex/CPLEX OPL
      objective value.
   - Load the OPL model exported by Studio into DOcplex and evaluate each PyOPL
      solution as a fixed DOcplex candidate, confirming that differing optimal
      assignments still produce the same value.
   - A dedicated test is implemented in
      [`test/test_docplex_cross_checks.py`](../test/test_docplex_cross_checks.py)
      as `TestDocplexCrossChecks.test_v2_features`. It compares SciPy and Gurobi
      objectives with the DOcplex objective, then evaluates both PyOPL assignments
      as fixed DOcplex solutions and checks feasibility and objective value.
   - Verified on 2026-08-22 with Python 3.10, DOcplex 2.32.264, CPLEX 22.1.1.0,
      native `oplrun`, SciPy/HiGHS, and Gurobi. All three objectives were `0.0`,
      and both PyOPL assignments were feasible in DOcplex and evaluated to `0.0`.
    - Studio 22.1.1 requires positional data for this tuple-indexed array and
       DOcplex 2.32.264 expects the unavailable `oplcpolpgen`, so the test
       preserves the exact source data for PyOPL and derives a temporary
       DOcplex-only compatible data file for the native `oplrun` export.
- [x] `test_iterator_scoping_sum_and_forall` (line 4591)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_iterator_scoping_sum_and_forall`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `8.0`.
   - Both PyOPL assignments were accepted as feasible DOcplex candidates and
      their objective evaluations matched the CPLEX objective within `1e-6`.
   - The original iterator-scoping model was accepted directly by Studio
      22.1.1; no DOcplex-only model or data adaptation was required.
- [x] `test_employee_rostering` (line 4649)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_employee_rostering`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `15.0`.
   - Both PyOPL assignments were accepted as feasible DOcplex candidates and
      their objective evaluations matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 requires positional demand data and a temporary rectangular
      employee-by-shift preference array; the exact keyed source fixture remains
      unchanged for PyOPL.
- [x] `test_minl_maxl_in_index_constraint` (line 4717)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_minl_maxl_in_index_constraint`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `0.0`.
   - PyOPL assignments were accepted as feasible DOcplex candidates and their
      objective evaluations matched the CPLEX objective within `1e-6`.
   - The exact `minl/maxl` index-filter model was accepted directly by Studio
      22.1.1; no DOcplex-only model or data adaptation was required.
- [x] `test_tuple_set_comprehension_pairs` (line 4784)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_tuple_set_comprehension_pairs`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `0.0`.
   - The 15 generated tuple-pair variables from the row-major comprehension
      were accepted as a feasible DOcplex candidate and matched the CPLEX
      objective within `1e-6`.
   - The exact tuple-set comprehension model and external `a` data were
      accepted directly by Studio 22.1.1; no adaptation was required.
- [x] `test_param_multi_index_rhs_expression_initialization` (line 4854)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_param_multi_index_rhs_expression_initialization`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `21.0`.
   - The PyOPL assignment was accepted as a feasible DOcplex candidate and
      its objective evaluation matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 requires `float` instead of PyOPL's `param float`
      declaration syntax; the temporary DOcplex model preserves both
      expression initializers unchanged.
- [x] `test_tuples_index_specifiers` (line 4905)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_tuples_index_specifiers`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `2.0`.
   - Both tuple-indexed PyOPL variables were accepted as a feasible DOcplex
      candidate and their objective evaluation matched CPLEX within `1e-6`.
   - The exact tuple index specifiers, tuple literals, and external tuple-set
      data were accepted directly by Studio 22.1.1; no adaptation was required.
- [x] `test_min_max` (line 4958)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_min_max`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective approximately
      `0.4864864865`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The cross-check maps the four declared `x` variables explicitly and lets
      CPLEX complete the aggregate-`max` auxiliary introduced during export.
- [x] `test_minl_maxl` (line 5004)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_minl_maxl`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `0.52`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`; the `x` values
      differ because the model has multiple optimal solutions.
   - The exact variadic `minl/maxl` model and external data were accepted
      directly by Studio 22.1.1. Aggregate `maxl` auxiliaries were completed
      by CPLEX rather than mapped as declared PyOPL decisions.
- [x] `test_not_operator_in_forall_and_constraint` (line 5374)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test only parses a model and inspects its AST
      for `not` nodes; it does not solve the model or expose an objective for
      comparison with CPLEX.
   - The original focused test remains the regression coverage for logical-NOT
      parsing and AST representation.
- [x] `test_and_or_operators_in_constraint_and_implication` (line 5408)
   - Applicability decision: not applicable to a DOcplex solver-result
      cross-check. The source test inspects AST `and`/`or` nodes and generated
      Gurobi source text; it does not solve the model or expose an objective
      for comparison with CPLEX.
   - The original focused test remains the regression coverage for logical
      AND/OR parsing and Gurobi code generation.
- [x] `test_composite_boolean_implication` (line 5458)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_composite_boolean_implication`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `0.0`.
   - Both PyOPL boolean assignments were fixed in cloned DOcplex models,
      remained feasible, and matched the CPLEX objective within `1e-6`.
   - The composite `AND`/`OR`/`NOT` implication was accepted directly by
      Studio 22.1.1; solver-specific auxiliary variables were left for each
      backend to derive rather than mapped as declared PyOPL decisions.
- [x] `test_multi_indexed_variable_and_constraint` (line 6883)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_multi_indexed_variable_and_constraint`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `0.0`.
   - All 12 three-dimensional PyOPL variables were accepted as a feasible
      DOcplex candidate and their objective evaluation matched CPLEX within
      `1e-6`.
   - The exact multi-indexed variable and nested `forall`/`sum` constraint
      model was accepted directly by Studio 22.1.1; no adaptation was required.
- [x] `test_tuple_field_access_and_nested_tuple_set` (line 6916)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_tuple_field_access_and_nested_tuple_set`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `0.0`.
   - Both nested tuple-indexed PyOPL variables were accepted as a feasible
      DOcplex candidate and their objective evaluation matched CPLEX within
      `1e-6`.
   - The exact inline nested tuple set and nested field accesses were accepted
      directly by Studio 22.1.1; no adaptation was required.
- [x] `test_inline_and_external_data_mix` (line 6948)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_inline_and_external_data_mix`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `15.0`.
   - The PyOPL assignments were accepted as feasible DOcplex candidates and
      their objective evaluations matched the CPLEX objective within `1e-6`.
   - The exact mixed inline/external model and data were accepted directly by
      Studio 22.1.1; no adaptation was required.
- [x] `test_filtered_sum_and_nested_forall` (line 6992)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_filtered_sum_and_nested_forall`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `3.0`.
   - Both PyOPL assignments were accepted as feasible DOcplex candidates and
      their objective evaluations matched the CPLEX objective within `1e-6`;
      the selected off-diagonal assignments differ between solvers.
   - The exact filtered-sum and nested-`forall` model was accepted directly by
      Studio 22.1.1; no adaptation was required.
- [x] `test_simple_blending_problem` (line 7026)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_simple_blending_problem`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `275.0`.
   - Both PyOPL assignments were accepted as feasible DOcplex candidates and
      their objective evaluations matched the CPLEX objective within `1e-6`.
   - The exact two-variable blending model was accepted directly by Studio
      22.1.1; no adaptation was required.
- [x] `test_blending_string_sets_list_index_error` (line 7071)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_blending_string_sets_list_index_error`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `342.5`.
   - Both PyOPL assignments were accepted as feasible DOcplex candidates and
      their objective evaluations matched the CPLEX objective within `1e-6`.
   - The exact string-indexed 1D/2D list-parameter model and positional data
      were accepted directly by Studio 22.1.1; no adaptation was required.
- [x] `test_workforce_planning_conditional_vs_explicit` (line 7162)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_workforce_planning_conditional_vs_explicit`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - The explicit and conditional PyOPL formulations both produced objective
      `61700.0`, and their reported variable assignments matched.
   - Both formulations were independently accepted by CPLEX with the same
      objective; every PyOPL assignment was fixed in its corresponding cloned
      DOcplex model and remained feasible within `1e-6`.
   - The temporary DOcplex models use Studio-compatible external declarations
      and constraint-label placement; the source formulations remain unchanged.
- [x] `test_rich_opl_model` (line 7472)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_rich_opl_model`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `25.0`.
   - Both PyOPL tuple-indexed boolean assignments were accepted as feasible
      DOcplex candidates and their objective evaluations matched CPLEX within
      `1e-6`.
   - The exact rich tuple/set model and external data were accepted directly
      by Studio 22.1.1; no adaptation was required.
- [x] `test_mini_graph_coloring_with_neq_and_implication` (line 7534)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_mini_graph_coloring_with_neq_and_implication`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `2.0`.
   - Both PyOPL color assignments were fixed in cloned DOcplex models,
      remained feasible, and matched the CPLEX objective within `1e-6`.
   - The exact tuple edge set, numeric `!=` constraints, and equality-based
      implication were accepted directly by Studio 22.1.1.
- [x] `test_food_blending_problem` (line 7580)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_food_blending_problem`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective approximately
      `35766.6666667`.
   - Both PyOPL slack and mix assignments were fixed in cloned DOcplex
      models, remained feasible, and matched the CPLEX objective within
      `1e-6`.
   - The exact tuple-valued food and ingredient parameters were accepted
      directly by Studio 22.1.1; no adaptation was required.
- [x] `test_transportation_problem_with_tuples_and_string_sets` (line 7678)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_transportation_problem_with_tuples_and_string_sets`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `1271.5`.
   - All six tuple-arc shipment assignments were accepted as a feasible
      DOcplex candidate and their objective evaluation matched CPLEX within
      `1e-6`.
   - Studio 22.1.1 requires explicit external declarations and positional
      arrays for the string-indexed and tuple-indexed numeric parameters; the
      exact keyed source fixture remains unchanged for PyOPL.
- [x] `test_transportation_problem_with_tuples_and_string_sets_and_string_filtering` (line 7822)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_transportation_problem_with_tuples_and_string_sets_and_string_filtering`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `1271.5`.
   - All six tuple-arc shipment assignments were accepted as a feasible
      DOcplex candidate and their objective evaluation matched CPLEX within
      `1e-6`.
   - Studio 22.1.1 requires explicit external declarations and positional
      arrays for the string-indexed and tuple-indexed numeric parameters; the
      exact keyed source fixture remains unchanged for PyOPL.
- [x] `test_inventory_problem_with_tuples` (line 7966)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_inventory_problem_with_tuples`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `136.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - Studio 22.1.1 does not accept the source tuple-keyed integer capacity
      array through external data. The temporary DOcplex fixture keeps the
      two stores inline, inlines the constant capacity bound and period
      parameters, and leaves the exact source model/data unchanged for PyOPL.
- [x] `test_complex_inventory_problem_with_tuples` (line 8055)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_complex_inventory_problem_with_tuples`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `205.0`.
   - Both PyOPL assignments were fixed in cloned DOcplex models, remained
      feasible, and matched the CPLEX objective within `1e-6`.
   - The temporary DOcplex fixture replaces the tuple store domain with an
      equivalent string-indexed set and uses dense positional parameter arrays;
      the exact tuple-keyed source model/data remain unchanged for PyOPL.
- [x] `test_shortest_path_with_tuples` (line 8156)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_shortest_path_with_tuples`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `5.0`.
   - Both PyOPL integer arc-flow assignments were fixed in cloned DOcplex
      models, remained feasible, and matched the CPLEX objective within
      `1e-6`.
   - The exact source model and data remain unchanged for PyOPL. The
      temporary DOcplex model renames the reserved tuple field `to` to
      `destination`.
- [x] `test_shortest_path_with_tuples_and_strings` (line 8227)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_shortest_path_with_tuples_and_strings`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `200.0`.
   - Both PyOPL string-indexed integer arc-flow assignments were fixed in
      cloned DOcplex models, remained feasible, and matched the CPLEX
      objective within `1e-6`.
   - The exact source model and data were accepted for the PyOPL and native
      OPL runs. The temporary DOcplex model renames the reserved tuple field
      `to` to `destination`; no data adaptation was required.
- [x] `test_logistics_with_tuples_and_strings` (line 8322)
   - Dedicated cross-check: `TestDocplexCrossChecks.test_logistics_with_tuples_and_strings`.
   - Verified on 2026-08-23 with Python 3.10, DOcplex 2.32.264, CPLEX
      22.1.1.0, native `oplrun`, SciPy/HiGHS, and Gurobi.
   - SciPy, Gurobi, and CPLEX all produced objective `20000.0`.
   - Both PyOPL string-indexed shipment assignments were fixed in cloned
      DOcplex models, remained feasible, and matched the CPLEX objective
      within `1e-6`.
   - The exact source model and data were accepted for the PyOPL and native
      OPL runs. The DOcplex fixture uses explicit external declarations for
      the cost, supply, and demand arrays; no data adaptation was required.

## Exclusions and decisions to record

- Tests that only inspect parsing, code generation, or a failure condition do
  not have a solver-result cross-check. Record them as `not applicable` with a
  short reason.
- Tests whose names mention CPLEX already should be checked for whether they
  compare output text, objective values, or both; standardize them on the
  shared comparison protocol where practical.
- DOcplex/CPLEX availability must be detected and reported clearly. A missing
  optional CPLEX installation is an environment status, not a passing result.
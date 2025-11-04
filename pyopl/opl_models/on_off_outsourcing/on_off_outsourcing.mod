/*
On/Off Production with Outsourcing — Literate OPL Model

Narrative (matches the provided description):
- We plan over a discrete horizon of T periods (Periods = 1..T).
- Each period, the plant is either running (run[t] = 1) or idle (run[t] = 0).
- Beginning a run in a period incurs a setup cost (setupCost) via start[t] = 1.
- If running, the plant can produce in-house y[t] up to a capacity cap[t].
- Demand in each period must be fully covered by one of two sources:
    • in-house production y[t], selectable only if the plant runs, or
    • outsourcing o[t].
  This is enforced by a logical OR via binary selectors (zin, zout).
- Costs include setup costs, variable in-house production costs, and outsourcing costs.
- Shutdowns (endRun) are tracked for completeness; the horizon end is treated as a shutdown if running.

Data is expected from a .dat file (e.g., on_off_outsourcing.dat).
*/

// ---------------------------
// Sets and parameters (data)
// ---------------------------

int T = ...;                            // Number of periods in the horizon
range Periods = 1..T;                   // Index set for periods

float demand[Periods] = ...;            // Period demand to be covered
float cap[Periods] = ...;               // In-house capacity when running
float cProd[Periods] = ...;             // Unit cost for in-house production
float cOut[Periods] = ...;              // Unit cost for outsourcing
float setupCost = ...;                  // Cost to start running in a period

// ---------------------------
// Decision variables
// ---------------------------

// Operational state
dvar boolean run[Periods];              // 1 if the plant is running in period t
dvar boolean start[Periods];            // 1 if a run starts at the beginning of period t
dvar boolean endRun[Periods];           // 1 if a run ends after period t (end-of-horizon closure)

// Quantities
dvar float+ y[Periods];                 // In-house production in period t
dvar float+ o[Periods];                 // Outsourced quantity in period t

// Disjunctive coverage selectors
// Exactly one is not required; at least one must be 1. With positive costs,
// the model naturally avoids setting both to 1. Using >= 1 keeps the OR semantics.
dvar boolean zin[Periods];              // 1 if in-house is chosen to fully cover demand in t
dvar boolean zout[Periods];             // 1 if outsourcing is chosen to fully cover demand in t

// ---------------------------
// Objective: total cost
// ---------------------------
// Sum of startup costs when run starts, in-house production costs, and outsourcing costs.
minimize totalCost:
  sum(t in Periods) ( setupCost * start[t] + cProd[t] * y[t] + cOut[t] * o[t] );

// ---------------------------
// Constraints
// ---------------------------
subject to {
  // Startup tracking
  // Assumption: plant is initially OFF before period 1.
  // start[t] is activated exactly on 0→1 transitions of run.
  start[1] == run[1];                  // If we run in period 1, that is a startup
  forall(t in 2..T) {
    // Big-M-free linearization of start[t] = max(0, run[t] - run[t-1])
    start[t] >= run[t] - run[t-1];     // Lower bound: must start if we go from 0 to 1
    start[t] <= run[t];                // Cannot start unless we are running
    start[t] <= 1 - run[t-1];          // Cannot start if we were already running
  }

  // Shutdown tracking
  // endRun[t] indicates 1→0 transitions; at horizon end, close any active run.
  forall(t in 1..T-1) {
    // Linearization of endRun[t] = max(0, run[t] - run[t+1])
    endRun[t] >= run[t] - run[t+1];    // Must end if we go from 1 to 0
    endRun[t] <= run[t];               // Cannot end unless we are running
    endRun[t] <= 1 - run[t+1];         // Cannot end if we continue running
  }
  endRun[T] == run[T];                 // If still running in T, we "end" at the horizon boundary

  // In-house production only when running; limited by capacity
  // If run[t] = 0 then y[t] must be 0; if run[t] = 1 then y[t] ≤ cap[t].
  forall(t in Periods)
    y[t] <= cap[t] * run[t];

  // Logical OR for coverage
  // Each period's demand must be fully covered by at least one source.
  // - If zin[t] = 1 then in-house must produce at least demand[t].
  // - If zout[t] = 1 then outsourcing must supply at least demand[t].
  // - At least one of zin or zout must be active.
  // - In-house coverage can only be selected if the plant is running.
  forall(t in Periods) {
    y[t] >= demand[t] * zin[t];        // Activate full coverage by in-house when zin[t] = 1
    o[t] >= demand[t] * zout[t];       // Activate full coverage by outsourcing when zout[t] = 1
    zin[t] + zout[t] >= 1;             // Logical OR: at least one source covers demand
    zin[t] <= run[t];                  // Cannot choose in-house coverage if not running
  }

  // Note:
  // - We do not force exactly one of (zin[t], zout[t]) to be 1. With positive costs,
  //   the optimizer will not select both since that would be strictly more expensive.
  // - If you want exclusivity, you can add: zin[t] + zout[t] == 1.
  // - If you prefer split coverage (partial y plus partial o), replace the OR block with:
  //     y[t] + o[t] >= demand[t]
  //   and drop zin/zout entirely.
}
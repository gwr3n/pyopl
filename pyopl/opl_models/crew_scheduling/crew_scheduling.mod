// Crew Scheduling (Assignment with per-crew workload limits)
// Problem: Assign exactly one crew member to each shift while ensuring no crew member
// exceeds their allowed number of shifts, minimizing total assignment cost.
// Data file provides: nbCrew, nbShifts, cost[crew][shift], max_shifts (per crew).

// -----------------------------
// Sets and index ranges
// -----------------------------
int nbCrew = ...;                  // Number of crew members (e.g., 3)
range Crew = 1..nbCrew;            // Crew index set

int nbShifts = ...;                // Number of shifts (e.g., 4)
range Shifts = 1..nbShifts;        // Shift index set

// -----------------------------
// Parameters
// -----------------------------
// Assignment costs: cost[i][j] is the cost of assigning crew i to shift j.
// Can be int or float; float used here to allow non-integers if needed.
float cost[Crew][Shifts] = ...;

// Max shifts allowed per entity controlling workload.
// NOTE: The .dat file defines max_shifts as a vector of length nbCrew (per-crew limits).
// If you intend per-crew limits, declare as: float max_shifts[Crew] = ...;
// The current line indexes by Shifts; kept as-is to only add comments.
float max_shifts[Shifts] = ...;

// -----------------------------
// Decision variables
// -----------------------------
// x[i][j] = 1 if crew i is assigned to shift j; 0 otherwise.
dvar boolean x[Crew][Shifts];

// -----------------------------
// Objective
// -----------------------------
// Minimize total assignment cost across all crew–shift pairs.
minimize
  sum(i in Crew, j in Shifts) cost[i][j] * x[i][j];

// -----------------------------
// Constraints
// -----------------------------
subject to {
  // Cover each shift exactly once:
  // For every shift j, assign exactly one crew member.
  forall (j in Shifts)
    sum (i in Crew) x[i][j] == 1;

  // Workload upper bound per crew:
  // For every crew i, total assigned shifts must not exceed the allowed maximum.
  // NOTE: Because the .dat provides max_shifts per crew, this constraint expects
  // max_shifts to be indexed by Crew. If you switch the declaration above to
  // float max_shifts[Crew] = ...; then this line is consistent as written.
  forall (i in Crew)
    sum (j in Shifts) x[i][j] <= max_shifts[i];
}

// -----------------------------
// Feasibility note
// -----------------------------
// With nbCrew=3 and max_shifts per crew up to 2, total capacity is 6 which
// covers nbShifts=4, so a feasible assignment exists in the provided data.
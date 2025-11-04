// Job Shop Scheduling Problem (JSSP) — Literate Commentary
// Context:
// - A set of nbJobs must be processed on nbMachines.
// - Each job visits every machine in a fixed order 1..nbMachines (flow-shop style order).
// - Processing times are given by duration[j][m].
// - Decisions are start times start[j][m] for each operation (j,m).
// - Objective is to minimize the makespan, i.e., the completion time of the last operation.
//
// Modeling assumptions:
// - Single machine per machine index m, capacity = 1.
// - Non-preemptive processing; once an operation starts it runs for its full duration.
// - Integer, discrete time (start and durations are integers).
// - No setup times or release dates; all jobs available at time 0.
// - This model enforces the same machine order for all jobs (m = 1..nbMachines).
//
// About the data file (jobshop.dat):
// - nbJobs = 5, nbMachines = 3.
// - duration[j][m] defines processing times for each job j on machine m.
// - Example: duration[1] = [3,2,4] means Job 1 takes 3 on M1, then 2 on M2, then 4 on M3.

// -------------------------------
// Sets and parameters
// -------------------------------

int nbJobs = ...;
int nbMachines = ...;
range Jobs = 1..nbJobs;
range Machines = 1..nbMachines;
int duration[Jobs][Machines] = ...;

// Big-M constant for disjunctive (no-overlap) constraints on machines.
// M must be "large enough" to deactivate a sequencing inequality when needed.
// Practical tip: a tighter M improves performance. A safe choice is, for each machine m,
// sum_j duration[j][m], or the global upper bound on makespan.
// Here we keep a constant placeholder; tune as needed for your data scale.
int M = 1000;

// -------------------------------
// Decision variables
// -------------------------------

// start[j][m] = start time of job j on machine m (nonnegative integers).
dvar int+ start[Jobs][Machines];

// z[j1][j2][m] = 1 if job j1 is scheduled before job j2 on machine m; 0 otherwise.
// Note: z is defined for all ordered pairs j1 != j2 (constraints skip j1==j2).
// Both directions (j1,j2) and (j2,j1) appear and are tied implicitly by the two big-M inequalities.
dvar boolean z[Jobs][Jobs][Machines];

// makespan = time when the last operation in the schedule completes.
dvar int+ makespan;

// -------------------------------
// Objective: minimize makespan
// -------------------------------
minimize makespan;

// -------------------------------
// Constraints
// -------------------------------
subject to {

  // Nonnegativity of start times (redundant given int+ domain, but explicit for clarity).
  forall(j in Jobs, m in Machines)
    start[j][m] >= 0;

  // Machine capacity (disjunctive) constraints:
  // On each machine m, any two jobs j1 and j2 cannot overlap.
  // We model this with a binary sequencing variable z[j1][j2][m]:
  // - If z[j1][j2][m] = 1, then j1 finishes before j2 starts on machine m.
  // - If z[j1][j2][m] = 0, then j2 finishes before j1 starts on machine m.
  //
  // Important note about "-1" below:
  // - Using "- 1" enforces at least one unit of idle time between back-to-back operations.
  // - If you want to allow operations to abut (end at t, next starts at t),
  //   replace "start[...] <= start[...] - 1 + ..." with "start[...] <= start[...] + ...".
  //   That is, remove the "- 1" terms in both inequalities.
  //
  forall(m in Machines)
    forall(j1 in Jobs, j2 in Jobs: j1 != j2){
      start[j1][m] + duration[j1][m] <=  start[j2][m] - 1 + M * z[j1][j2][m];
      start[j2][m] + duration[j2][m] <=  start[j1][m] - 1 + M * (1 - z[j1][j2][m]);
    }
  // Technological (within-job) precedence:
  // Each job must visit machines in the fixed order 1..nbMachines.
  // Operation on machine m+1 cannot start before completion on machine m.
  //
  // Note: This implements the "fixed order" from the description.
  // If jobs had job-specific machine orders, you would encode that order
  // via a permutation array and reference duration accordingly.
  //
  forall(j in Jobs, m in 1..nbMachines-1)
    start[j][m+1] >= start[j][m] + duration[j][m];
  // Makespan constraint
  forall(j in Jobs)
    makespan >= start[j][nbMachines] + duration[j][nbMachines];
}

// End of model.
//
// Notes on strengthening and scaling:
// - Consider tightening M per machine as M[m] = sum(j in Jobs) duration[j][m].
// - You can add symmetry-breaking on z (e.g., fix z[j1][j2][m] for j1<j2 on one machine)
//   to reduce redundant symmetric solutions.
// - If time is large, consider CP Optimizer or time-indexed formulations instead of big-M MILP.


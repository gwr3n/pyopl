/*
Assignment Problem (classic)

Goal
- Assign each worker to exactly one task and each task to exactly one worker.
- Each assignment (i, j) incurs a known cost cost[i][j].
- Minimize the total cost of the one-to-one pairing.

Data (see assignment.dat)
- W = number of workers, T = number of tasks
- cost is a W x T matrix of assignment costs
  Example:
    W = 3; T = 3;
    cost = [[13, 7, 9],
            [ 8, 7, 6],
            [ 6,12, 8]];
*/

// Basic sizes (provided in the .dat file)
int W = ...;             // number of workers
int T = ...;             // number of tasks

// Index domains
range Workers = 1..W;    // worker indices
range Tasks   = 1..T;    // task indices

// Cost parameters (provided in the .dat file)
// cost[i][j] = cost of assigning worker i to task j
float cost[Workers][Tasks] = ...;

// Decision variables
// x[i][j] = 1 if worker i is assigned to task j, 0 otherwise
dvar boolean x[Workers][Tasks];

// Objective: minimize total assignment cost
minimize
  sum(i in Workers, j in Tasks) cost[i][j] * x[i][j];

// Feasibility constraints: enforce a one-to-one assignment
subject to {
  // Each worker is assigned to exactly one task
  forall(i in Workers)
    sum(j in Tasks) x[i][j] == 1;

  // Each task is assigned to exactly one worker
  forall(j in Tasks)
    sum(i in Workers) x[i][j] == 1;
}

/*
Notes
- With equality on both sides, a feasible solution exists only when W == T.
  For rectangular cases (W != T), either:
    * relax one side to <= and the other to >=, or
    * add dummy workers/tasks with zero (or large) costs to square the matrix.
- The optimal x is a permutation matrix for square instances (W == T).
*/

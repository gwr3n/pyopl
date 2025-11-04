// -----------------------------------------------------------------------------
// Set Covering Problem (0–1 Integer Linear Program)
//
// Problem overview:
// - Universe of elements: Elements = {1..M}.
// - Candidate sets: Sets = {1..N}.
// - cover[j][i] = 1 if set j covers element i; 0 otherwise.
// - cost[j] is the cost of selecting set j.
//
// Goal:
// Choose x[j] ∈ {0,1} to minimize total cost sum_j cost[j] * x[j],
// subject to every element i being covered by at least one selected set.
//
// Data is provided in the companion .dat file (e.g., set_covering.dat).
// -----------------------------------------------------------------------------

// Size parameters (provided by .dat):
// M: number of elements to cover
// N: number of candidate sets
int M = ...; // number of elements
int N = ...; // number of sets

// Index sets for readability:
range Elements = 1..M;
range Sets = 1..N;

// Inputs:
// - cover: a binary matrix indexed as cover[set][element] indicating coverage
// - cost:  cost of picking each set
// Note: In the .dat example, cover has N rows (sets) and M columns (elements).
int cover[Sets][Elements] = ...;
float cost[Sets] = ...;

// Decision variables:
// x[j] = 1 if set j is selected, 0 otherwise.
dvar boolean x[Sets];

// Objective:
// Minimize the sum of costs of the selected sets.
minimize
  sum (j in Sets) cost[j] * x[j];

// Constraints:
// Coverage: every element i must be covered by at least one chosen set.
// The sum over all sets of cover[j][i] * x[j] is the number of chosen sets
// that cover element i; requiring this to be >= 1 ensures coverage.
subject to {
  forall (i in Elements)
    sum (j in Sets) cover[j][i] * x[j] >= 1;
}

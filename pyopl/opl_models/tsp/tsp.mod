// Traveling Salesman Problem (TSP) — MTZ formulation (literate commented)
//
// Problem summary
// ---------------
// A single tour must visit each city exactly once and return to the start,
// using known pairwise travel distances. The model chooses directed arcs and
// their order to form one Hamiltonian cycle that minimizes total distance.
// Subtour-elimination constraints (MTZ) enforce that the chosen arcs form a
// single tour rather than multiple disjoint cycles.
//
// Data interface
// --------------
// N     : number of cities (scalar defined in the .dat file)
// Cities: index set {1..N}
// dist  : NxN matrix of nonnegative distances; dist[i][i] is usually 0.
// The corresponding .dat example provides N=4 and a 4x4 distance matrix.

int N = ...;                      // number of cities (from .dat)
range Cities = 1..N;              // index set of cities 1..N
float dist[Cities][Cities] = ...; // pairwise distances between cities

// Decision variables
// ------------------
// x[i][j] ∈ {0,1} indicates whether the tour uses the directed arc i → j.
// We declare x as nonnegative integers; with the degree-equality constraints,
// x is implicitly forced to be 0/1 (exactly one outgoing and one incoming per city).
//
// u[i] are continuous nonnegative "order" variables used by the MTZ constraints
// to eliminate subtours. Intuitively, u[i] encodes the visit position of city i.
// The formulation treats city 1 as a reference; MTZ constraints are enforced
// only among cities 2..N to break symmetry and cut subtours.

dvar int+ x[Cities][Cities];
dvar float+ u[Cities]; // MTZ order variables (u[1] acts as an anchor)

// Objective
// ---------
// Minimize total distance of the selected arcs. Diagonal terms (i == j) are
// excluded, so self-loops are not considered.

minimize
  sum(i in Cities, j in Cities: i != j) dist[i][j] * x[i][j];

// Constraints
// -----------
// Degree constraints (flow conservation):
// - Exactly one outgoing arc leaves each city.
// - Exactly one incoming arc enters each city.
// Together with integrality of x, these enforce that x behaves as binary
// and that the chosen arcs form one or more disjoint cycles covering all cities.
//
// Subtour elimination (MTZ):
// The classical Miller–Tucker–Zemlin cuts ensure there is only one cycle.
// For i != j in {2..N}:
//     u[i] - u[j] + N * x[i][j] <= N - 1
// Interpretation: if arc i→j is used (x[i][j] = 1), then u[i] + 1 <= u[j]
// (after scaling), which imposes a consistent ordering and prevents subtours.
// If the arc is not used, the inequality is nonbinding.
// Note: Self-loops are already excluded by i != j in sums/objective.

subject to {
  // One outgoing arc from each city
  forall(i in Cities)
    sum(j in Cities: j != i) x[i][j] == 1;

  // One incoming arc to each city
  forall(j in Cities)
    sum(i in Cities: i != j) x[i][j] == 1;

  // Subtour elimination (MTZ) among cities 2..N
  forall(i in 2..N, j in 2..N: i != j)
    u[i] - u[j] + N * x[i][j] <= N - 1;
}

// Notes
// -----
// - The formulation is symmetric with respect to the starting city; any city
//   can be considered the start of the tour. The degree constraints ensure the
//   tour returns to its start automatically.
// - Optionally, explicit constraints x[i][i] == 0 could be added, but they are
//   redundant here because i != j is enforced in the sums and objective.
// - If desired, u can be bounded (e.g., 0 <= u[i] <= N) or made integer
//   without changing optimality; keeping them continuous often solves faster.

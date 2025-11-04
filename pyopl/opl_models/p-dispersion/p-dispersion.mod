/* 
P-Dispersion (Max–Min Dispersion) Model — Kuby (1987)

Problem summary:
Given a set of N candidate sites and pairwise distances, select exactly p sites so that the 
minimum distance between any two selected sites is as large as possible. We maximize z, the
minimum pairwise distance among selected sites.

Modeling approach:
- y[i] indicates whether site i is selected.
- x[i][j] is an auxiliary indicator that activates when both i and j are selected (for i < j).
- z captures the minimum distance among all selected pairs; we enforce z ≤ dist[i][j] whenever
  both i and j are selected, using a Big-M linearization.
- maxD provides a valid upper bound on z to tighten the relaxation.
- Pair indices are restricted to i < j to avoid duplicate pairs and i ≠ j.

Notes:
- Choose M large enough to relax inactive constraints, but not too large to avoid numerical issues.
- A safe choice for M is at least max(dist), e.g., M ≥ maxD.
*/

// Number of candidate sites (provided in .dat)
int N = ...;

// Index set for sites
range Sites = 1..N;

// Number of sites to choose (provided in .dat)
int p = ...;

// Symmetric pairwise distances between sites (provided in .dat)
// dist[i][i] should be zero; only i < j are used in constraints below.
float dist[Sites][Sites] = ...;

// Big-M constant for linearization: should be ≥ an upper bound on achievable z (e.g., maxD)
int M = 10000;

// Decision variables
// y[i] = 1 if site i is selected; 0 otherwise
dvar boolean y[Sites];

// x[i][j] = 1 if both sites i and j are selected (only meaningful when i < j)
dvar boolean x[Sites][Sites];

// z = minimum pairwise distance among all selected site pairs
dvar float+ z;

// Upper bound for z (e.g., global max distance); helps bound the LP relaxation
param float maxD = ...;

// Objective: maximize the minimum distance between any two selected sites
maximize z;

subject to {
  // Select exactly p sites
  sum(i in Sites) y[i] == p;

  // Bound z to aid linearization/relaxation
  z <= maxD;

  // Pairwise linking and min-distance constraints (for unique pairs i < j):
  // - Enforce x[i][j] = 1 only when both y[i] = y[j] = 1.
  // - If x[i][j] = 1, then z ≤ dist[i][j]; otherwise relaxed by Big-M.
  forall(i in Sites, j in Sites : i < j){
    // x[i][j] ≥ y[i] + y[j] - 1  ⇒ x[i][j] becomes 1 only if both y’s are 1
    y[i] + y[j] - 1 <= x[i][j];

    // x[i][j] ≤ y[i] and x[i][j] ≤ y[j]  ⇒ x cannot be 1 unless both y’s are 1
    x[i][j] <= y[i];
    x[i][j] <= y[j];

    // Min-distance enforcement with Big-M:
    // When x[i][j] = 1, we get z ≤ dist[i][j].
    // When x[i][j] = 0, the constraint relaxes to z ≤ dist[i][j] + M.
    z <= dist[i][j] + (1-x[i][j])*M;
  }
}


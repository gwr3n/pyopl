/*
Transportation Problem (balanced)

Narrative:
A set of sources with given supplies ships goods to a set of destinations with given demands
at known per‑unit costs. The decision is how much to ship from each source to each destination
so that all supplies and demands are exactly satisfied while minimizing total transportation cost.

Data expectations (from .dat):
- S: number of sources
- D: number of destinations
- cost[i][j]: per‑unit shipping cost from source i to destination j
- supply[i]: available quantity at source i (nonnegative)
- demand[j]: required quantity at destination j (nonnegative)

Modeling notes:
- This is the balanced case: sum_i supply[i] == sum_j demand[j].
  If totals differ, the model becomes infeasible under equality constraints.
  A common extension for unbalanced data is to introduce a dummy node or relax equalities to
  <= and >= with penalties for unmet demand or unused supply.

Units:
- Choose consistent units (e.g., x in tons, cost in $/ton, objective in $).
*/

// -----------------------------
// Sets and dimensions
// -----------------------------

// Number of sources (set in .dat)
int S = ...;
// Index set for sources
range Sources = 1..S;

// Number of destinations (set in .dat)
int D = ...;
// Index set for destinations
range Destinations = 1..D;

// -----------------------------
// Parameters (input data)
// -----------------------------

// Shipping cost matrix: cost[i][j] is cost from source i to destination j
float cost[Sources][Destinations] = ...;

// Supply available at each source (nonnegative)
float+ supply[Sources] = ...;

// Demand required at each destination (nonnegative)
float+ demand[Destinations] = ...;

// -----------------------------
// Decision variables
// -----------------------------

// Shipment quantity from source i to destination j (nonnegative, continuous)
dvar float+ x[Sources][Destinations];

// -----------------------------
// Objective: minimize total transportation cost
// -----------------------------
minimize
  sum(i in Sources, j in Destinations)
    cost[i][j] * x[i][j];

// -----------------------------
// Constraints
// -----------------------------
subject to {
  // Supply conservation:
  // Ship out of each source exactly its available supply.
  forall(i in Sources)
    sum(j in Destinations) x[i][j] == supply[i];

  // Demand satisfaction:
  // Receive at each destination exactly its required demand.
  forall(j in Destinations)
    sum(i in Sources) x[i][j] == demand[j];
}

/*
Implementation tips:
- Ensure the .dat file defines S, D, cost, supply, demand with compatible dimensions.
- For diagnostics, check that sum(supply) equals sum(demand) in your data.
- To switch to an unbalanced formulation, replace equalities with <= (supply) and >= (demand),
  and model penalties or dummy nodes as appropriate.
*/

/* 
Problem: Budgeted multi-resource knapsack

Narrative
- Each item i has a value Value[i] and consumes amounts Use[r][i] of several resources r.
- Decisions:
  - Take[i] ∈ {0,1}: whether to take item i.
  - Capacity[r] ≥ 0 (integer): how much capacity to allocate to resource r.
- Feasibility:
  - For every resource r, the capacity allocated must cover the total usage of all chosen items:
      sum_i Use[r][i] * Take[i] ≤ Capacity[r]
- Global budget:
  - The sum of capacities across all resources must not exceed a single total budget TotalCapacity:
      sum_r Capacity[r] ≤ TotalCapacity
- Objective:
  - Maximize the total value of chosen items: sum_i Value[i] * Take[i]

Data
- See knapsackp.dat for an example with N items, R resources, a TotalCapacity budget,
  values (Value), and per-resource usage matrix (Use[r][i]).

Notes
- Capacity[r] is modeled as integer to represent discrete budget units per resource; 
  change to dvar float+ if fractional capacities are intended.
- Use[r][i] can be zero; only positive usages bind the per-resource constraints.
- This is equivalent to choosing items subject to a shared budget that must be split 
  across multiple resource types to cover the chosen items’ requirements.
*/

// -----------------------------
// Sets and dimensions
// -----------------------------

// Number of items
int N = ...;

// Index set for items
range Items = 1..N;

// Number of resource types
int R = ...;

// Index set for resources
range Resources = 1..R;

// -----------------------------
// Parameters (data inputs)
// -----------------------------

// Global capacity budget available across all resources
int TotalCapacity = ...;

// Item values (objective coefficients)
float Value[Items];

// Resource usage matrix: Use[r][i] is resource r used by item i
float Use[Resources][Items];

// -----------------------------
// Decision variables
// -----------------------------

// Take[i] = 1 if item i is selected, 0 otherwise
dvar boolean Take[Items];

// Capacity allocated to each resource (integer, nonnegative)
dvar int+ Capacity[Resources];

// -----------------------------
// Objective
// -----------------------------

// Maximize total value of selected items
maximize 
    sum(i in Items) Value[i] * Take[i];

// -----------------------------
// Constraints
// -----------------------------
subject to {

    // Per-resource coverage: allocated capacity must cover chosen items' usage
    forall(r in Resources)
        sum(i in Items) Use[r][i] * Take[i] <= Capacity[r];

    // Global budget on the sum of resource capacities
    sum(r in Resources) Capacity[r] <= TotalCapacity;
}

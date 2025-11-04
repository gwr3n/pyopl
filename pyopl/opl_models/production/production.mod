// -----------------------------------------------------------------------------
// Production Planning model (literate-style commented)
// -----------------------------------------------------------------------------
// Problem intent:
// Over a multi-period horizon, choose production quantities of each product
// in each period to minimize total cost, while meeting demand requirements
// and not exceeding per-period capacity.
// See: production.txt for the narrative description and production.dat for data.
// -----------------------------------------------------------------------------

// ---------------------------
// Index sets and dimensions
// ---------------------------
// Number of products (e.g., 2 in the sample dat)
int nbProducts = ...;
// Product index set: p ∈ {1, ..., nbProducts}
range Products = 1..nbProducts;

// Number of periods (e.g., 3 in the sample dat)
int nbPeriods = ...;
// Period index set: t ∈ {1, ..., nbPeriods}
range Periods = 1..nbPeriods;

// ---------------------------
// Parameters (input data)
// ---------------------------
// Per-unit production cost of product p in period t.
// Example from dat: cost = [ [3, 2, 4], [2, 3, 5] ];
float cost[Products][Periods] = ...;

// Demand and capacity:
// IMPORTANT modeling note about demand indexing:
// - The current model uses 'demand' when enforcing a product-level requirement
//   (see constraint "demand satisfaction" below), i.e., demand should be per
//   product if we intend cumulative production across periods to meet each
//   product’s total demand: demand[Products].
// - However, the provided dat defines: demand = [40, 50, 0]; sized by 'Periods'.
//   If demand is intended to be per-period (total across all products), then the
//   demand-satisfaction constraint should be rewritten as:
//       forall(t in Periods) sum(p in Products) x[p][t] >= demand[t];
//   For now, we keep the model structure and highlight this mismatch for review.
float demand[Periods] = ...;

// Per-period production capacity (across all products).
// Example from dat: capacity = [30, 40, 20];
float capacity[Periods] = ...;

// ---------------------------
// Decision variables
// ---------------------------
// x[p][t] = quantity of product p produced in period t (nonnegative).
dvar float+ x[Products][Periods];

// ---------------------------
// Objective
// ---------------------------
// Minimize total production cost over all products and periods.
minimize
  sum(p in Products, t in Periods) cost[p][t] * x[p][t];

// ---------------------------
// Constraints
// ---------------------------
subject to {
  // Demand satisfaction:
  // For each product, total production across all periods must meet that
  // product’s demand. NOTE: This uses demand[p], which assumes demand is
  // indexed by Products; the current dat provides demand by Periods, so
  // either (a) change demand to float demand[Products] in both .mod/.dat,
  // or (b) change this constraint to be period-based if demand is per-period.
  forall(p in Products)
    sum(t in Periods) x[p][t] >= demand[p];

  // Capacity limits:
  // In each period, total production across all products cannot exceed the
  // available capacity in that period.
  forall(t in Periods)
    sum(p in Products) x[p][t] <= capacity[t];
}

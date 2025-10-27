// Two-Stage Stochastic Production with Recourse (Single Product)
//
// Problem synopsis:
// - Before demand is known, choose an initial production quantity x at cost c_init per unit.
// - After demand is revealed in scenario s, produce extra y[s] if needed at cost c_rec per unit.
// - Excess initial production has no value or penalty; only costs are incurred for produced units.
// - Objective: minimize expected total cost = c_init * x + c_rec * E_s[y[s]].
//
// Modeling notes:
// - Decisions are continuous and nonnegative (quantities).
// - Recourse linkage: y[s] must cover any shortfall relative to realized demand in scenario s.
//   Under minimization, linear inequalities y[s] >= demand[s] - x and y[s] >= 0 (implicit) implement y[s] >= max(0, demand[s] - x).

{string} Scenarios;                                 // scenario labels provided in data

// Parameters
param float c_init;                                  // cost per unit for initial production (first stage)
param float c_rec;                                   // cost per unit for recourse production (second stage)
param float demand[Scenarios];                       // realized demand by scenario
param float prob[Scenarios];                         // scenario probabilities (should sum to 1)

// Decision variables
dvar float+ x;                                       // initial production quantity (first-stage)
dvar float+ y[Scenarios];                            // recourse (additional) production per scenario

// Derived expression: expected recourse quantity
// E[y] = sum_s prob[s] * y[s]
dexpr float expected_recourse_qty = sum(s in Scenarios) prob[s] * y[s];

// Objective: minimize expected total cost = first-stage cost + expected second-stage cost
minimize ExpectedTotalCost:
    c_init * x + c_rec * expected_recourse_qty;

subject to {
  // Link recourse to scenario shortfall: y[s] >= max(0, demand[s] - x)
  // Implemented via linear inequality under minimization
  forall(s in Scenarios)
    LinkShortfall: y[s] >= demand[s] - x;

  // Ground (data-only) check: probabilities sum to one
  ProbSumToOne: (sum(s in Scenarios) prob[s]) == 1;
}
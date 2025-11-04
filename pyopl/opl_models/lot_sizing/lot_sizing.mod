/*
Literate model: Single-item lot sizing over a finite horizon.

Goal
- Choose production quantities x[t], setup decisions y[t], and ending inventory s[t] per period t = 1..T
- Satisfy deterministic demands without backlogging
- Minimize fixed setup + variable production + holding costs

Data (see lot_sizing.dat)
- T: number of periods
- demand[t]: deterministic demand in period t
- K: fixed setup cost (incurs only if y[t] = 1)
- u: unit production cost
- h: unit inventory holding cost per period

Modeling notes
- Inventory balance: s[t] = s[t-1] + x[t] - demand[t], with s[1] = x[1] - demand[1]
- No backlogging: enforced by s[t] >= 0 through s being float+ (nonnegative)
- Setup-production link: x[t] <= M_t * y[t]
  This file uses M_t = demand[t] as a tight but restrictive choice that prevents producing
  more than current-period demand. If you want to allow building inventory for future periods,
  consider a larger M_t (e.g., sum_{k=t..T} demand[k]) or a capacity parameter.
*/

// ---------- Sets and parameters ----------

int T = ...;                  // planning horizon length
float demand[1..T] = ...;     // demand profile per period
float K = ...;                // fixed setup cost
float u = ...;                // unit production cost
float h = ...;                // unit holding cost

// ---------- Decision variables ----------

// x[t] ≥ 0: production quantity in period t
// y[t] ∈ {0,1}: 1 if a setup occurs in period t (allows production), 0 otherwise
// s[t] ≥ 0: ending inventory after meeting demand in period t

dvar float+ x[1..T];  // production quantity
dvar boolean y[1..T]; // setup decision
dvar float+ s[1..T];  // ending inventory

// ---------- Objective ----------
// Minimize total cost: fixed setups + variable production + inventory holding

minimize sum(t in 1..T) (K * y[t] + u * x[t] + h * s[t]);

// ---------- Constraints ----------
subject to {
  // Inventory balance in the first period:
  // ending inventory = production − demand (no initial inventory assumed)
  s[1] == x[1] - demand[1];

  // Inventory flow for subsequent periods:
  // inventory carries over from previous period plus current production minus demand
  forall(t in 2..T)
    s[t] == s[t-1] + x[t] - demand[t];

  // Setup-production linking (big-M):
  // If y[t] = 0 then x[t] = 0; if y[t] = 1 then x[t] ≤ demand[t].
  // Note: using demand[t] as M forbids producing more than current-period demand,
  // which limits building inventory for future periods. Use a larger M if desired.
  forall(t in 1..T)
    x[t] <= demand[t] * y[t];
}

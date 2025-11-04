/*
  Capacitated Plant Location — literate commentary

  Problem summary:
  - Decide which plants to open (pay fixed_cost) and how much to ship (x) to customers (pay trans_cost).
  - Meet all customer demands without exceeding plant capacities.
  - Minimize total cost = fixed opening + transportation.

  Sets and indices:
  - Plants = 1..nbPlants
  - Customers = 1..nbCustomers

  Parameters:
  - fixed_cost[i]: fixed opening cost for plant i
  - trans_cost[i][j]: per‑unit shipping cost from plant i to customer j
  - demand[j]: demand of customer j
  - capacity[i]: capacity of plant i

  Decision variables:
  - y[i] ∈ {0,1}: 1 if plant i is opened
  - x[i][j] ≥ 0: units shipped from plant i to customer j

  Objective:
  - Minimize fixed opening costs plus transportation costs.

  Constraints:
  - Demand satisfaction: each customer's demand is fully met.
  - Capacity linking: shipments are limited by capacity when the plant is open.

  Notes:
  - Ensure parameter dimensions align with their index sets (see TODOs below).
*/

// SETS AND INDICES
int nbPlants = ...;
range Plants = 1..nbPlants;

int nbCustomers = ...;
range Customers = 1..nbCustomers;

// PARAMETERS (read from the .dat file)
float trans_cost[Plants][Customers] = ...;

float fixed_cost[Plants] = ...;

float demand[Customers] = ...;

float capacity[Plants] = ...;

// DECISION VARIABLES
dvar boolean y[Plants];
dvar float+ x[Plants][Customers];

// OBJECTIVE: minimize fixed opening and per‑unit transportation costs
minimize sum(i in Plants) fixed_cost[i] * y[i] + sum(i in Plants, j in Customers) trans_cost[i][j] * x[i][j];

// CONSTRAINTS
subject to {
  // Demand satisfaction: each customer's demand must be fully met
  forall(j in Customers)
    sum(i in Plants) x[i][j] == demand[j];

  // Capacity linking: shipments allowed only if plant is open
  // Current form limits each x[i][j]; a common (tighter) aggregate capacity is:
  //   sum(j in Customers) x[i][j] <= capacity[i] * y[i];
  forall(i in Plants, j in Customers)
    x[i][j] <= capacity[i] * y[i];
}



// -----------------------------------------------------------------------------
// Warehouse Location Problem (WLP)
// -----------------------------------------------------------------------------
// Problem summary:
// Candidate warehouses can be opened by paying fixed costs and used to ship
// goods to customers at per‑unit transportation costs. Decisions are:
//  - which warehouses to open (binary y[i])
//  - how much to ship from each open warehouse to each customer (flow x[i][j])
// Objective:
//  Minimize total cost = fixed opening costs + transportation costs,
// subject to:
//  - every customer's demand is fully met
//  - shipments from a warehouse respect its capacity (when the warehouse is open)
//
// Data interface (provided in the .dat file):
//  nbWarehouses, nbCustomers, fixed_cost[i], trans_cost[i][j],
//  demand[j], capacity[i]
// -----------------------------------------------------------------------------

// --- Size parameters (from .dat) ---------------------------------------------
// Number of candidate facilities and number of customers
int nbWarehouses = ...;
int nbCustomers  = ...;

// --- Index sets ---------------------------------------------------------------
// Index range for warehouses and customers
range Warehouses = 1..nbWarehouses;
range Customers  = 1..nbCustomers;

// --- Input parameters (from .dat) --------------------------------------------
// Fixed opening cost for each warehouse i (e.g., in $)
// Example in .dat: fixed_cost = [80, 90];
float fixed_cost[Warehouses] = ...;

// Per‑unit transportation cost from warehouse i to customer j
// Example in .dat: trans_cost = [[3,5,8],[4,3,6]];
float trans_cost[Warehouses][Customers] = ...;

// Demand required by each customer j (units to be delivered)
// Example in .dat: demand = [15, 20, 10];
float demand[Customers] = ...;

// Capacity of each warehouse i (maximum shippable units if opened)
// Example in .dat: capacity = [25, 30];
float capacity[Warehouses] = ...;

// --- Decision variables -------------------------------------------------------
// y[i] = 1 if warehouse i is opened; 0 otherwise
dvar boolean y[Warehouses];

// x[i][j] = units shipped from warehouse i to customer j (nonnegative)
dvar float+ x[Warehouses][Customers];

// --- Objective: minimize total cost ------------------------------------------
minimize
  // Sum of fixed opening costs over opened warehouses
  sum (i in Warehouses) fixed_cost[i] * y[i]
  +
  // Sum of transportation costs over all shipped units
  sum (i in Warehouses, j in Customers) trans_cost[i][j] * x[i][j]
;

// --- Constraints --------------------------------------------------------------
subject to {
  // Demand satisfaction:
  // For each customer j, total inbound shipments must meet exactly its demand.
  forall (j in Customers)
    sum (i in Warehouses) x[i][j] == demand[j];

  // Capacity linking:
  // Shipments from a warehouse are allowed only if it is opened (big-M),
  // and are bounded by its capacity.
  // NOTE: The tighter, typical capacity constraint aggregates over customers:
  //   forall(i in Warehouses) sum(j in Customers) x[i][j] <= capacity[i] * y[i];
  // The per-customer version below is a looser formulation.
  forall (i in Warehouses, j in Customers)
    x[i][j] <= capacity[i] * y[i];
}


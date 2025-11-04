/* 
Literate overview
- Problem: Single-vehicle VRP with a single depot at node 1. The vehicle must leave the depot, visit each customer exactly once, and return to the depot.
- Cost: Distance cost on each selected arc.
- Subtour elimination and capacity: Cumulative load variables propagate along selected arcs; this both respects capacity and breaks subtours among customers.
- Data expectations (as in the .dat):
  * N: total number of nodes (1 depot + N-1 customers)
  * dist[i][j]: travel distances/costs between nodes
  * demand[i]: demand at node i (0 at the depot)
  * capacity: vehicle capacity (must be >= sum of all demands for feasibility in single-vehicle case)
*/

/* 
Index sets and parameters
- Nodes = {1..N}; node 1 is the depot, 2..N are customers.
- dist[i][j] can be asymmetric; model does not assume symmetry.
- demand[1] is typically 0 (depot), positive for customers.
*/
// Vehicle Routing Problem (VRP, single vehicle)
int N = ...;
range Customers = 2..N;
range Nodes = 1..N;
float dist[Nodes][Nodes] = ...;
float demand[Nodes] = ...;
float capacity = ...;

/* 
Decision variables
- x[i][j] = 1 if the arc i -> j is used in the tour, 0 otherwise.
- load[i] = cumulative load carried immediately after visiting node i.
  * load[1] is fixed to 0 (vehicle leaves depot empty in this formulation).
  * For customers, load[i] is at least their demand and propagates through arcs.
*/
dvar boolean x[Nodes][Nodes];
dvar float+ load[Nodes];

/* 
Objective
- Minimize total travel distance across all selected arcs.
- Self-loops (i == j) do not contribute to the objective and are disallowed for customers by degree constraints below; x[1][1] is unconstrained but irrelevant.
*/
minimize sum(i in Nodes, j in Nodes: i != j) dist[i][j] * x[i][j];

subject to {
  /* 
  Depot degree constraints
  - Exactly one arc leaves the depot (start of the route).
  - Exactly one arc enters the depot (end of the route, returning).
  */
  sum(j in Customers) (x[1][j]) == 1;
  sum(i in Customers) (x[i][1]) == 1;

  /* 
  Customer degree constraints
  - Each customer has exactly one outgoing and one incoming arc.
  - Self-loops are explicitly excluded via j != i and i != j.
  */
  forall(i in Customers)
    sum(j in Nodes: j != i) (x[i][j]) == 1;
  forall(j in Customers)
    sum(i in Nodes: i != j) (x[i][j]) == 1;

  /* 
  Load initialization
  - Vehicle departs the depot empty in this formulation.
  - If a different departure load is desired, adjust this equality accordingly.
  */
  load[1] == 0;

  /* 
  Load lower bounds at customers
  - Ensures the vehicle load immediately after serving customer i is at least its demand.
  - Combined with propagation below, this ties loads to route structure and prevents subtours.
  */
  forall(i in Customers)
    load[i] >= demand[i];

  /* 
  Load propagation and capacity (MTZ-style with capacity)
  - For any selected customer-to-customer arc i -> j:
      load[j] >= load[i] + demand[j]
    up to a relaxation term when x[i][j] = 0.
  - The big-M term uses capacity; when x[i][j] = 0, the constraint does not bind.
  - This enforces capacity and eliminates customer-only subtours.
  - Note: Arcs to/from the depot are excluded here; load is anchored at the depot by load[1] == 0.
  */
  forall(i in Customers, j in Customers: i != j)
    load[j] >= load[i] + demand[j] - capacity * (1 - x[i][j]);
}

/* 
Notes and extensions
- To explicitly forbid x[1][1] (a redundant tightening), add: x[1][1] == 0.
- For multiple vehicles, replace depot degree constraints with fleet-size bounds and add vehicle index or flow conservation per vehicle.
- If demands are collected (pickup) rather than delivered, adjust load semantics accordingly.
*/

# Overview of OPL Example Models

This document provides a summary of the OPL (Optimization Programming Language) example models included in the `pyopl/opl_models` folder. Each example demonstrates a classic optimization problem, with a brief description and the main modeling features.

---

## Assignment Problem (`assignment`)
**Files:** `assignment.mod`, `assignment.dat`
- **Description:** Assigns workers to tasks such that each worker is assigned to exactly one task and each task to exactly one worker, minimizing the total assignment cost.
- **Features:** Binary decision variables, cost matrix, one-to-one assignment constraints.

---

## Crew Pairing Problem (`crew_pairing`)
**Files:** `crew_pairing.mod`, `crew_pairing.dat`
- **Description:** Selects a set of crew pairings to cover all flights at minimum cost.
- **Features:** Binary variables for pairings, coverage constraints for flights.

---

## Crew Scheduling Problem (`crew_scheduling`)
**Files:** `crew_scheduling.mod`, `crew_scheduling.dat`
- **Description:** Assigns crew to shifts, ensuring each shift is covered and crew do not exceed maximum allowed shifts.
- **Features:** Binary assignment variables, shift coverage, crew workload limits.

---

## Graph Coloring Problem (`graph_coloring`)
**Files:** `graph_coloring.mod`, `graph_coloring.dat`
- **Description:** Assigns colors to graph nodes so that adjacent nodes have different colors, minimizing the number of colors used.
- **Features:** Integer color variables, big-M encoding for adjacency constraints, minimization of maximum color.

---

## Inventory Routing Problem (`inventory_routing`)
**Files:** `inventory_routing.mod`, `inventory_routing.dat`
- **Description:** Determines delivery schedules and quantities for multiple customers over time, balancing inventory holding and transportation costs.
- **Features:** Integer delivery and inventory variables, vehicle capacity constraints, time-dependent routing, cost minimization.

---

## Job Shop Scheduling Problem (`jobshop`)
**Files:** `jobshop.mod`, `jobshop.dat`
- **Description:** Schedules jobs on machines to minimize the makespan, ensuring no overlap and respecting job order.
- **Features:** Integer start times, binary sequencing variables, makespan minimization, precedence constraints.

---

## Knapsack Problems (`knapsack`)
**Files:** `knapsack.mod`, `knapsackp.mod`, `knapsack.dat`, `knapsackp.dat`
- **Description:** Selects items to maximize value without exceeding capacity. The `knapsackp` variant handles multiple resources.
- **Features:** Binary selection variables, capacity/resource constraints, value maximization.

---

## Maintenance Planning (Average-Cost MDP) (`maintenance`)
**Files:** maintenance.mod, maintenance.dat
- **Description:** Finds a stationary long-run maintenance policy for a deteriorating asset by minimizing steady-state average cost, given state-dependent action costs and probabilistic condition transitions.
- **Features:** Occupation-measure (steady-state) decision variables, average-cost objective, state flow-balance constraints, probability normalization, action feasibility mask, row-stochastic transition checks.

---

## Lot Sizing Problem (`lot_sizing`)
**Files:** `lot_sizing.mod`, `lot_sizing.dat`
- **Description:** Determines production quantities and setups over time to meet demand at minimum cost.
- **Features:** Continuous and binary variables, inventory balance, setup and production costs.

---

## On/Off Production with Outsourcing (`on_off_outsourcing`)
**Files:** `on_off_outsourcing.mod`, `on_off_outsourcing.dat`
- **Description:** Plans production over time with binary on/off decisions and optional outsourcing to satisfy demand at minimum total cost.
- **Features:** Binary setup (on/off) variables, production and inventory variables, capacity active when on, demand balance, outsourcing quantities and costs, setup and production costs.

---

## P-Dispersion Problem (`p-dispersion`)
**Files:** `p-dispersion.mod`, `p-dispersion.txt`
- **Description:** Selects exactly p locations from a set of candidates to maximize the minimum pairwise distance among the selected locations.
- **Features:** Binary selection variables, cardinality constraint (sum of selections equals p), auxiliary minimum-distance variable, pairwise distance constraints (big-M/indicator), max–min objective.

---

## Plant Location Problem (`plant_location`)
**Files:** `plant_location.mod`, `plant_location.dat`
- **Description:** Decides which plants to open and how to supply customers to minimize fixed and transportation costs.
- **Features:** Binary plant opening variables, flow variables, demand satisfaction, capacity constraints.

---

## Production Planning Problem (`production`)
**Files:** `production.mod`, `production.dat`
- **Description:** Plans production quantities for multiple products and periods to meet demand at minimum cost.
- **Features:** Continuous production variables, demand and capacity constraints, cost minimization.

---

## Set Covering Problem (`set_covering`)
**Files:** `set_covering.mod`, `set_covering.txt`
- **Description:** Selects a minimum-cost collection of sets so that every element is covered by at least one chosen set.
- **Features:** Binary selection variables, coverage constraints (each element covered ≥ 1), cost minimization.

---

## Set Partitioning Problem (`set_partitioning`)
**Files:** `set_partitioning.mod`, `set_partitioning.dat`
- **Description:** Selects sets to partition all elements exactly once at minimum cost.
- **Features:** Binary selection variables, partitioning constraints, cost minimization.

---

## Stochastic Production Problem (`stochastic_production`)
**Files:** `stochastic_production.mod`, `stochastic_production.txt`
- **Description:** Plans production over time under demand uncertainty using scenario-based stochastic programming to minimize expected total cost.
- **Features:** Scenario probabilities, nonanticipativity constraints on first-stage decisions, production and inventory variables, capacity limits, per-scenario demand balance, expected-cost minimization.

---

## Stochastic Scheduling Problem (`stochastic_scheduling`)
**Files:** `stochastic_scheduling.mod`, `stochastic_scheduling.txt`
- **Description:** Schedules jobs on machines under uncertainty using scenario-based stochastic programming to minimize expected performance (e.g., makespan or tardiness).
- **Features:** Scenario probabilities, nonanticipativity on first-stage sequencing/scheduling decisions, machine capacity (no-overlap) and precedence constraints, per-scenario timing constraints, expected-objective minimization.

---

## Transportation Problem (`transportation`)
**Files:** `transportation.mod`, `transportation.dat`
- **Description:** Determines optimal shipping quantities from sources to destinations to minimize cost, meeting supply and demand.
- **Features:** Flow variables, supply and demand constraints, cost minimization.

---

## Traveling Salesman Problem (`tsp`)
**Files:** `tsp.mod`, `tsp.dat`
- **Description:** Finds the shortest possible route visiting each city exactly once and returning to the start.
- **Features:** Integer routing variables, subtour elimination constraints, distance minimization.

---

## Vehicle Routing Problem (`vehicle_routing`)
**Files:** `vehicle_routing.mod`, `vehicle_routing.dat`
- **Description:** Routes a vehicle to serve all customers with capacity constraints, minimizing total distance.
- **Features:** Binary routing variables, load tracking, capacity and routing constraints.

---

## Warehouse Location Problem (`warehouse_location`)
**Files:** `warehouse_location.mod`, `warehouse_location.dat`
- **Description:** Decides which warehouses to open and how to supply customers to minimize costs.
- **Features:** Binary warehouse opening variables, flow variables, demand satisfaction, capacity constraints.

---

## Workforce Planning Problem (`workforce_planning`)
**Files:** `workforce_planning.mod`, `workforce_planning.dat`
- **Description:** Plans workforce levels over time to meet demand and minimize costs, considering hiring, firing, and training.
- **Features:** Integer and binary variables, workforce balance, cost minimization, hiring/firing/training constraints.

---

Each example includes a `.mod` file (model) and a `.dat` file (data). These models are useful for learning and benchmarking optimization techniques in OPL.

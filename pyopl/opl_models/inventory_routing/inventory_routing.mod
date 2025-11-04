/*
Inventory Replenishment for Multiple Stores (3 periods)

Problem summary:
- Plan deliveries to each store across a 3-period horizon.
- Satisfy known period demands while respecting storage capacity at each store.
- Costs include per-unit transport (store-specific) and per-unit holding per period.
- No stockouts: end-of-period inventory is constrained to be nonnegative and governed by flow balance.
- Objective: minimize total transport + holding cost.

Indices and sets:
- Stores: set of store identifiers (string-labeled, provided in .dat).
- Periods: discrete planning periods 1..3.

Units and conventions:
- All quantities (demand, deliveries, inventory) are in the same unit (e.g., cases).
- Costs are expressed per unit (transport) and per unit per period (holding).
- Inventory inv[s][t] is end-of-period inventory at store s after demand in period t.
- deliver[s][t] is the quantity shipped to store s that becomes available in period t.

Feasibility notes:
- Because deliveries are nonnegative and unconstrained (no fleet/arc capacity in this simple model),
  stockouts are avoided by ensuring inv >= 0 through flow balance.
- Storage capacity applies to end-of-period inventory, not incoming deliveries during the period.
*/

/* --------------------------
   Sets and planning horizon
   -------------------------- */

{string} Stores = ...;              // Set of store identifiers; defined in the .dat file
range Periods = 1..3;               // Three-period horizon; adjust if you extend the planning window

/* --------------------------
   Parameters (data inputs)
   --------------------------
   Provided via the .dat file.

   OPL note: you can declare as `float holding_cost;` etc. (without `param`) if preferred.
*/

param float holding_cost;                    // Holding cost per unit per period
param float transport_cost[Stores];          // Per-unit transport cost to each store
param float capacity[Stores];                // Max end-of-period inventory allowed at each store
param float demand[Stores][Periods];         // Exogenous demand at each store and period
param float init_inv[Stores];                // Initial inventory at each store before period 1

/* --------------------------
   Decision variables
   --------------------------
   inv[s][t]    ≥ 0 : end-of-period inventory at store s in period t
   deliver[s][t]≥ 0 : quantity delivered to store s in period t
*/

dvar float+ inv[Stores][Periods];            // Inventory level at store s in period t
dvar float+ deliver[Stores][Periods];        // Delivery quantity to store s in period t

/* --------------------------
   Objective
   --------------------------
   Minimize total transport + holding cost over the horizon.

   Intuition:
   - Transport cost penalizes shipping more.
   - Holding cost penalizes carrying inventory forward.
   - The optimal trade-off often pushes inventory toward just-in-time subject to capacity.
*/
minimize
    // Transport cost
    sum(s in Stores, t in Periods) transport_cost[s] * deliver[s][t]
  +
    // Holding cost
    sum(s in Stores, t in Periods) holding_cost * inv[s][t]
;

/* --------------------------
   Constraints
   --------------------------
   1) Inventory flow balance:
      inv[s][t] = inv[s][t-1] + deliver[s][t] - demand[s][t]
      with period-1 using init_inv as inv[s][0].

   2) Capacity: inv[s][t] ≤ capacity[s].

   3) Nonnegativity: inv, deliver ≥ 0 (already enforced by float+, restated for clarity).
*/

subject to {
    // Flow balance in period 1: start from initial inventory, add deliveries, subtract demand
    forall(s in Stores)
        inv[s][1] == init_inv[s] + deliver[s][1] - demand[s][1];

    // Flow balance in later periods: carry over prior inventory
    forall(s in Stores, t in Periods : t > 1)
        inv[s][t] == inv[s][t - 1] + deliver[s][t] - demand[s][t];

    // Storage capacity: limit end-of-period inventory
    forall(s in Stores, t in Periods)
        inv[s][t] <= capacity[s];

    // Explicit nonnegativity (variables are float+; included for readability and solver logs)
    forall(s in Stores, t in Periods)
        inv[s][t] >= 0;

    forall(s in Stores, t in Periods)
        deliver[s][t] >= 0;
}

/*
Modeling extensions (not included here, but commonly added):
- Supplier or vehicle capacity limits on total deliveries per period.
- Fixed delivery costs and binary visit decisions (routing/inventory-routing coupling).
- Lost sales or backlogging if stockouts are permitted (replace equality with slack variables).
- Service-level constraints or safety stock minimums.
- Time-varying holding/transport costs or lead times.
*/


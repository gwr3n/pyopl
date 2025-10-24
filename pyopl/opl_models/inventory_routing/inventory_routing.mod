{string} Stores = ...;
range Periods = 1..3;

# Parameters
param float holding_cost;                    # holding cost per unit per period
param float transport_cost[Stores];          # per-unit transport cost to each store
param float capacity[Stores];                # storage capacity at each store
param float demand[Stores][Periods];         # demand by store and period
param float init_inv[Stores];                # initial inventory at each store (period 0)

# Decision variables
dvar float+ inv[Stores][Periods];            # inventory level at store s in period t
dvar float+ deliver[Stores][Periods];        # delivery quantity to store s in period t

# Objective: minimize transport cost + holding cost
minimize
    sum(s in Stores, t in Periods) transport_cost[s] * deliver[s][t]
  + sum(s in Stores, t in Periods) holding_cost * inv[s][t]
;

# Constraints
subject to {
    # Initial inventory balance for period 1
    forall(s in Stores)
        inv[s][1] == init_inv[s] + deliver[s][1] - demand[s][1];

    # Inventory balance for later periods
    forall(s in Stores, t in Periods : t > 1)
        inv[s][t] == inv[s][t - 1] + deliver[s][t] - demand[s][t];

    # Capacity limits: inventory cannot exceed store capacity
    forall(s in Stores, t in Periods)
        inv[s][t] <= capacity[s];

    # Non-negativity (variables are declared float+ so these are enforced, included for clarity)
    forall(s in Stores, t in Periods)
        inv[s][t] >= 0;

    forall(s in Stores, t in Periods)
        deliver[s][t] >= 0;
}


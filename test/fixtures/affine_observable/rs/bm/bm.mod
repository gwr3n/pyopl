/*
Benchmark formulation (BM) for the static-dynamic stochastic lot-sizing
penalty-cost model with backorders, based on the Tarim-Kingsman formulation
used as BM by Tunc et al.

The model tracks expected end-of-period net inventory directly. Binary
lastOrder[i][t] identifies period i as the latest replenishment through t,
so the appropriate demand aggregation and normal-loss envelope are selected.
*/

int N = ...;                         // number of demand periods
float setupCost = ...;               // fixed cost per replenishment
float holdingCost = ...;             // unit holding cost per period
float backorderCost = ...;           // unit backorder cost per period
float bigM = ...;                    // upper bound on an order quantity
float lossBigM = ...;                // deactivates loss cuts for unselected cycles
int pieces = ...;                    // number of normal-loss tangents
float meanDemand[1..N] = ...;        // period demand means
float stdDemand[1..N] = ...;         // independent period demand deviations
float prefixMean[0..N] = ...;        // expected demand over periods 1..t
float cycleMean[1..N][1..N] = ...;   // expected demand over [i,t]
float cycleStd[1..N][1..N] = ...;    // demand deviation over [i,t]
float normalTail[1..pieces] = ...;   // 1-Phi(z_k)
float normalDensity[1..pieces] = ...;// phi(z_k)

dvar boolean order[1..N];            // 1 when a replenishment is scheduled
dvar float+ quantity[1..N];          // expected order quantity
dvar float inventory[1..N];          // expected end-of-period net inventory
dvar float+ backorders[1..N];        // approximated expected backorders
dvar boolean lastOrder[1..N][1..N];  // latest replenishment i through period t

minimize ExpectedTotalCost:
  sum(t in 1..N)
    (setupCost * order[t]
     + holdingCost * inventory[t]
     + (holdingCost + backorderCost) * backorders[t]);

subject to {
  // The first period starts the first replenishment cycle.
  InitialOrder: order[1] == 1;

  // Expected inventory conservation under independent stochastic demand.
  FirstInventoryBalance:
    inventory[1] == quantity[1] - meanDemand[1];

  forall(t in 2..N)
    InventoryBalance:
      inventory[t] == inventory[t-1] + quantity[t] - meanDemand[t];

  forall(t in 1..N)
    ActivateOrderQuantity: quantity[t] <= bigM * order[t];

  // Exactly one period is the most recent replenishment through each t.
  forall(t in 1..N)
    SelectLatestOrder: sum(i in 1..t) lastOrder[i][t] == 1;

  forall(i in 1..N, t in i..N)
    LatestMustBeOrder: lastOrder[i][t] <= order[i];

  forall(i in 1..N, t in i..N, r in i+1..t)
    NoLaterOrder: lastOrder[i][t] <= 1 - order[r];

  forall(i in 1..N, t in i..N)
    SelectIfLatest:
      lastOrder[i][t] >= order[i] - sum(r in i+1..t) order[r];

  // Conditional normal first-order loss tangents. If i is the latest order,
  // inventory[t] = S[i] - E[D(i,t)], hence L = sigma*phi - tail*inventory.
  forall(i in 1..N, t in i..N, k in 1..pieces)
    LossEnvelope:
      backorders[t] >=
        cycleStd[i][t] * normalDensity[k]
        - normalTail[k] * inventory[t]
        - lossBigM * (1 - lastOrder[i][t]);
}
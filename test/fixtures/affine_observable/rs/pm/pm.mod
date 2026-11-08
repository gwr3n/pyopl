/*
Extended formulation (PM) for the static-dynamic stochastic lot-sizing
penalty-cost model with backorders, following equations (3)-(9) of Tunc et al.

The binary arc x[i][j] selects replenishment cycle [i,j). q[i][j] is the
cumulative expected order quantity attached to that cycle. H[i][j][t]
underestimates expected backorders through a lower piecewise-linear envelope
of the normal first-order loss function.
*/

int N = ...;                         // number of demand periods
float setupCost = ...;               // fixed cost per replenishment
float holdingCost = ...;             // unit holding cost per period
float backorderCost = ...;           // unit backorder cost per period
float bigM = ...;                    // upper bound on cumulative order quantity
int pieces = ...;                    // number of normal-loss tangents
float meanDemand[1..N] = ...;        // period demand means
float stdDemand[1..N] = ...;         // independent period demand deviations
float prefixMean[0..N] = ...;        // expected demand over periods 1..t
float cycleMean[1..N][1..N] = ...;   // expected demand over [i,t]
float cycleStd[1..N][1..N] = ...;    // demand deviation over [i,t]
float normalTail[1..pieces] = ...;   // 1-Phi(z_k)
float normalDensity[1..pieces] = ...;// phi(z_k)

dvar boolean x[1..N][2..N+1];       // 1 when [i,j) is a selected cycle
dvar float+ q[1..N][2..N+1];        // cumulative expected order quantity
dvar float+ H[1..N][2..N+1][1..N];  // approximated expected backorders

minimize ExpectedTotalCost:
  sum(i in 1..N, j in i+1..N+1)
    (setupCost * x[i][j]
     + sum(t in i..j-1)
         (holdingCost * (q[i][j] - prefixMean[t] * x[i][j])
          + (holdingCost + backorderCost) * H[i][j][t]));

subject to {
  // Equations (4)-(6): selected arcs form a path from period 1 to N+1.
  forall(t in 2..N)
    CycleFlow:
      sum(i in 1..t-1) x[i][t] == sum(j in t+1..N+1) x[t][j];

  FirstCycle: sum(j in 2..N+1) x[1][j] == 1;
  LastCycle: sum(i in 1..N) x[i][N+1] == 1;

  // Equation (7): inactive arcs carry no cumulative quantity.
  forall(i in 1..N, j in i+1..N+1)
    ActivateQuantity: q[i][j] <= bigM * x[i][j];

  // Equation (8): cumulative order quantities cannot decrease at a cycle boundary.
  forall(t in 2..N)
    NondecreasingQuantity:
      sum(i in 1..t-1) q[i][t] <= sum(j in t+1..N+1) q[t][j];

  // Equation (9): eleven tangent cuts lower-bound each normal loss function.
  forall(i in 1..N, j in i+1..N+1, t in i..j-1, k in 1..pieces)
    LossEnvelope:
      H[i][j][t] >=
        (cycleMean[i][t] * normalTail[k]
         + cycleStd[i][t] * normalDensity[k]) * x[i][j]
        - normalTail[k] * (q[i][j] - prefixMean[i-1] * x[i][j]);
}
// Strengthened single-commodity flow formulation F1' (paper equations 1-3 and 7-10).
int N = ...;
range Cities = 1..N;
float cost[Cities][Cities] = ...;

dvar boolean x[Cities][Cities];
dvar float+ flow[Cities][Cities];

minimize totalTravelCost:
  sum(i in Cities, j in Cities: i != j) cost[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    oneOutgoingArc: sum(j in Cities: j != i) x[i][j] == 1;
  forall(j in Cities)
    oneIncomingArc: sum(i in Cities: i != j) x[i][j] == 1;

  // City 1 supplies one unit to every other city.
  rootSupply:
    sum(j in Cities: j != 1) flow[1][j]
      - sum(i in Cities: i != 1) flow[i][1] == N - 1;
  forall(i in 2..N)
    cityDemand:
      sum(j in Cities: j != i) flow[j][i]
        - sum(j in Cities: j != i) flow[i][j] == 1;

  forall(j in 2..N)
    rootArcCapacity: flow[1][j] <= (N - 1) * x[1][j];
  forall(i in 2..N, j in Cities: i != j)
    tightenedArcCapacity: flow[i][j] <= (N - 2) * x[i][j];
}
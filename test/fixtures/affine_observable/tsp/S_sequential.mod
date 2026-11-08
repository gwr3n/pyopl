// Sequential Miller-Tucker-Zemlin formulation (paper equations 1-3 and 6).
int N = ...;
range Cities = 1..N;
float cost[Cities][Cities] = ...;

dvar boolean x[Cities][Cities];
dvar float+ order[Cities];

minimize totalTravelCost:
  sum(i in Cities, j in Cities: i != j) cost[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    oneOutgoingArc: sum(j in Cities: j != i) x[i][j] == 1;
  forall(j in Cities)
    oneIncomingArc: sum(i in Cities: i != j) x[i][j] == 1;

  forall(i in 2..N, j in 2..N: i != j)
    sequenceOrder: order[i] - order[j] + N * x[i][j] <= N - 1;
}
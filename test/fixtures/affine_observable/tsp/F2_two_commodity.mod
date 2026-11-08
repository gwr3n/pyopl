// Two-commodity flow formulation (paper equations 1-3 and 11-16).
int N = ...;
range Cities = 1..N;
float cost[Cities][Cities] = ...;

dvar boolean x[Cities][Cities];
dvar float+ forwardFlow[Cities][Cities];
dvar float+ reverseFlow[Cities][Cities];

minimize totalTravelCost:
  sum(i in Cities, j in Cities: i != j) cost[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    oneOutgoingArc: sum(j in Cities: j != i) x[i][j] == 1;
  forall(j in Cities)
    oneIncomingArc: sum(i in Cities: i != j) x[i][j] == 1;

  forwardRootSupply:
    sum(j in Cities: j != 1) (forwardFlow[1][j] - forwardFlow[j][1]) == N - 1;
  forall(i in 2..N)
    forwardCityDemand:
      sum(j in Cities: j != i) (forwardFlow[i][j] - forwardFlow[j][i]) == -1;

  reverseRootDemand:
    sum(j in Cities: j != 1) (reverseFlow[1][j] - reverseFlow[j][1]) == -(N - 1);
  forall(i in 2..N)
    reverseCitySupply:
      sum(j in Cities: j != i) (reverseFlow[i][j] - reverseFlow[j][i]) == 1;

  forall(i in Cities, j in Cities: i != j)
    useSelectedArc:
      forwardFlow[i][j] + reverseFlow[i][j] == (N - 1) * x[i][j];
}
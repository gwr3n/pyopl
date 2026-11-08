// Multi-commodity flow formulation (paper equations 1-3 and 17-22).
int N = ...;
range Cities = 1..N;
range Commodities = 2..N;
float cost[Cities][Cities] = ...;

dvar boolean x[Cities][Cities];
dvar float+ flow[Cities][Cities][Commodities];

minimize totalTravelCost:
  sum(i in Cities, j in Cities: i != j) cost[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    oneOutgoingArc: sum(j in Cities: j != i) x[i][j] == 1;
  forall(j in Cities)
    oneIncomingArc: sum(i in Cities: i != j) x[i][j] == 1;

  forall(i in Cities, j in Cities, k in Commodities: i != j)
    useSelectedArc: flow[i][j][k] <= x[i][j];

  // Commodity k travels from city 1 to city k.
  forall(k in Commodities)
    commoditySource:
      sum(j in Cities: j != 1) flow[1][j][k]
        - sum(i in Cities: i != 1) flow[i][1][k] == 1;
  forall(k in Commodities)
    commoditySink:
      sum(i in Cities: i != k) flow[i][k][k]
        - sum(j in Cities: j != k) flow[k][j][k] == 1;
  forall(k in Commodities, city in 2..N: city != k)
    commodityBalance:
      sum(i in Cities: i != city) flow[i][city][k]
        - sum(j in Cities: j != city) flow[city][j][k] == 0;
}
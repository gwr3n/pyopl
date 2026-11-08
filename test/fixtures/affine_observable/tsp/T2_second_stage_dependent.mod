// Second-stage-dependent formulation (paper equations 1-3, 25, and 27-30).
int N = ...;
range Cities = 1..N;
range Stages = 1..N;
float cost[Cities][Cities] = ...;

dvar boolean x[Cities][Cities];
// The paper notes that these stage variables may be continuous in T2.
dvar float+ stageArc[Cities][Cities][Stages];

minimize totalTravelCost:
  sum(i in Cities, j in Cities: i != j) cost[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    oneOutgoingArc: sum(j in Cities: j != i) x[i][j] == 1;
  forall(j in Cities)
    oneIncomingArc: sum(i in Cities: i != j) x[i][j] == 1;
  forall(i in Cities, j in Cities: i != j)
    linkArcToStage: x[i][j] == sum(t in Stages) stageArc[i][j][t];

  forall(j in Cities)
    enterCityOnce:
      sum(i in Cities, t in Stages: i != j) stageArc[i][j][t] == 1;
  forall(i in Cities)
    leaveCityOnce:
      sum(j in Cities, t in Stages: j != i) stageArc[i][j][t] == 1;
  forall(t in Stages)
    oneArcAtEachStage:
      sum(i in Cities, j in Cities: i != j) stageArc[i][j][t] == 1;
  forall(i in 2..N)
    consecutiveVisit:
      sum(j in Cities, t in Stages: j != i) t * stageArc[i][j][t]
        - sum(k in Cities, t in Stages: k != i) t * stageArc[k][i][t] == 1;

  forall(j in 2..N)
    leaveRootAtFirstStage: stageArc[1][j][1] == x[1][j];
  forall(i in 2..N)
    enterRootAtLastStage: stageArc[i][1][N] == x[i][1];
}
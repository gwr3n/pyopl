// Third-stage-dependent formulation (paper equations 1-3, 25, and 31-33).
int N = ...;
range Cities = 1..N;
range Stages = 1..N;
float cost[Cities][Cities] = ...;

dvar boolean x[Cities][Cities];
// The paper notes that these stage variables may be continuous in T3.
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

  leaveRootFirst:
    sum(j in Cities: j != 1) stageArc[1][j][1] == 1;
  enterRootLast:
    sum(i in Cities: i != 1) stageArc[i][1][N] == 1;
  forall(i in 2..N, t in 2..N)
    stageContinuity:
      sum(j in Cities: j != i) stageArc[i][j][t]
        - sum(k in Cities: k != i) stageArc[k][i][t - 1] == 0;
}
// First-stage-dependent formulation (paper equations 1-3 and 23-26).
int N = ...;
range Cities = 1..N;
range Stages = 1..N;
float cost[Cities][Cities] = ...;

dvar boolean x[Cities][Cities];
dvar boolean stageArc[Cities][Cities][Stages];

minimize totalTravelCost:
  sum(i in Cities, j in Cities: i != j) cost[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    oneOutgoingArc: sum(j in Cities: j != i) x[i][j] == 1;
  forall(j in Cities)
    oneIncomingArc: sum(i in Cities: i != j) x[i][j] == 1;
  forall(i in Cities, j in Cities: i != j)
    linkArcToStage: x[i][j] == sum(t in Stages) stageArc[i][j][t];

  tourLength:
    sum(i in Cities, j in Cities, t in Stages: i != j) stageArc[i][j][t] == N;
  forall(i in 2..N)
    consecutiveVisit:
      sum(j in Cities, t in Stages: j != i) t * stageArc[i][j][t]
        - sum(k in Cities, t in Stages: k != i) t * stageArc[k][i][t] == 1;

  forall(j in 2..N)
    leaveRootAtFirstStage: stageArc[1][j][1] - x[1][j] == 0;
  forall(i in 2..N)
    enterRootAtLastStage: stageArc[i][1][N] - x[i][1] == 0;
  forall(i in 2..N, j in Cities: i != j)
    noOtherFirstStageArc: stageArc[i][j][1] == 0;
  forall(i in Cities, j in 2..N: i != j)
    noOtherLastStageArc: stageArc[i][j][N] == 0;
}
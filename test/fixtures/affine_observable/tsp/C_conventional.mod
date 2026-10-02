// Conventional Dantzig-Fulkerson-Johnson formulation (paper equations 1-4).
int N = ...;
range Cities = 1..N;
float cost[Cities][Cities] = ...;

int subsetCount = ...;
range Subsets = 1..subsetCount;
int subsetMember[Subsets][Cities] = ...;
int subsetSize[Subsets] = ...;

// x[i][j] is one exactly when the tour traverses arc (i,j).
dvar boolean x[Cities][Cities];

minimize totalTravelCost:
  sum(i in Cities, j in Cities: i != j) cost[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    oneOutgoingArc: sum(j in Cities: j != i) x[i][j] == 1;

  forall(j in Cities)
    oneIncomingArc: sum(i in Cities: i != j) x[i][j] == 1;

  // Each listed proper subset not containing city 1 must omit an internal arc.
  forall(s in Subsets)
    eliminateSubtour:
      sum(i in Cities, j in Cities:
          i != j && subsetMember[s][i] == 1 && subsetMember[s][j] == 1)
        x[i][j] <= subsetSize[s] - 1;
}
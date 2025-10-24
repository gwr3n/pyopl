// Transportation Problem
int S = ...;
range Sources = 1..S;
int D = ...;
range Destinations = 1..D;
float cost[Sources][Destinations] = ...;
float+ supply[Sources] = ...;
float+ demand[Destinations] = ...;

dvar float+ x[Sources][Destinations];

minimize sum(i in Sources, j in Destinations) cost[i][j] * x[i][j];

subject to {
  forall(i in Sources)
    sum(j in Destinations) x[i][j] == supply[i];
  forall(j in Destinations)
    sum(i in Sources) x[i][j] == demand[j];
}

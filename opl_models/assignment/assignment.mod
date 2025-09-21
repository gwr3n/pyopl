// Assignment Problem (classic)
int W = ...;
int T = ...;
range Workers = 1..W;
range Tasks = 1..T;
float cost[Workers][Tasks] = ...;
dvar boolean x[Workers][Tasks];

minimize sum(i in Workers, j in Tasks) cost[i][j] * x[i][j];

subject to {
  forall(i in Workers)
    sum(j in Tasks) (x[i][j]) == 1;
  forall(j in Tasks)
    sum(i in Workers) (x[i][j]) == 1;
}

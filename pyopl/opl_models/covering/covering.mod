// Set Covering Problem (variant)
int nbSets = ...;
int nbElements = ...;
range Sets = 1..nbSets;
range Elements = 1..nbElements;
float cost[Sets] = ...;
boolean a[Sets][Elements] = ...;

dvar boolean x[Sets];

minimize sum(i in Sets) cost[i] * x[i];

subject to {
  forall(j in Elements)
    sum(i in Sets) (a[i][j]) * x[i] >= 1;
}


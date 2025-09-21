// Set Covering Problem
int M = ...; // number of elements
int N = ...; // number of sets
range Elements = 1..M;
range Sets = 1..N;
int cover[Sets][Elements] = ...;
float cost[Sets] = ...;

dvar boolean x[Sets];

minimize sum(j in Sets) cost[j] * x[j];

subject to {
  forall(i in Elements)
    sum(j in Sets) cover[j][i] * x[j] >= 1;
}

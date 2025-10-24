// Vehicle Routing Problem (VRP, single vehicle)
int N = ...;
range Customers = 2..N;
range Nodes = 1..N;
float dist[Nodes][Nodes] = ...;
float demand[Nodes] = ...;
float capacity = ...;

dvar boolean x[Nodes][Nodes];
dvar float+ load[Nodes];

minimize sum(i in Nodes, j in Nodes: i != j) dist[i][j] * x[i][j];

subject to {
  sum(j in Customers) (x[1][j]) == 1;
  sum(i in Customers) (x[i][1]) == 1;
  forall(i in Customers)
    sum(j in Nodes: j != i) (x[i][j]) == 1;
  forall(j in Customers)
    sum(i in Nodes: i != j) (x[i][j]) == 1;
  load[1] == 0;
  forall(i in Customers)
    load[i] >= demand[i];
  forall(i in Customers, j in Customers: i != j)
    load[j] >= load[i] + demand[j] - capacity * (1 - x[i][j]);
}

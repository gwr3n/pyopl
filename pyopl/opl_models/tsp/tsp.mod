// Traveling Salesman Problem (TSP)
int N = ...;
range Cities = 1..N;
float dist[Cities][Cities] = ...;

dvar int+ x[Cities][Cities];
dvar float+ u[Cities]; // for subtour elimination

minimize sum(i in Cities, j in Cities: i != j) dist[i][j] * x[i][j];

subject to {
  forall(i in Cities)
    sum(j in Cities: j != i) x[i][j] == 1;
  forall(j in Cities)
    sum(i in Cities: i != j) x[i][j] == 1;
  // Subtour elimination (MTZ)
  forall(i in 2..N, j in 2..N: i != j)
    u[i] - u[j] + N * x[i][j] <= N-1;
}

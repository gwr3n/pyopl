# P-Dispersion Problem (Kuby, 1987)
int N = ...;
range Sites = 1..N;
int p = ...;
float dist[Sites][Sites] = ...;
int M = 10000;

# Decision variables
dvar boolean y[Sites];
dvar boolean x[Sites][Sites];
dvar float+ z;

# Upper bound for z (max inter-site distance)
param float maxD = ...;

maximize z;

subject to {
  # Select exactly p sites
  sum(i in Sites) y[i] == p;

  # Bound z to aid linearization
  z <= maxD;

  # If both i and j are selected, z cannot exceed their separation
  forall(i in Sites, j in Sites : i < j){
    y[i] + y[j] - 1 <= x[i][j];
    x[i][j] <= y[i];
    x[i][j] <= y[j];
    z <= dist[i][j] + (1-x[i][j])*M;
  }
}

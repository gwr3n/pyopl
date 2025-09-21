// Lot Sizing Problem (Single Item)
int T = ...;
float demand[1..T] = ...;
float K = ...;
float u = ...;
float h = ...;

// Decision variables
dvar float+ x[1..T]; // production quantity
dvar boolean y[1..T]; // setup decision
dvar float+ s[1..T]; // inventory

minimize sum(t in 1..T) (K * y[t] + u * x[t] + h * s[t]);

subject to {
  s[1] == x[1] - demand[1];
  forall(t in 2..T)
    s[t] == s[t-1] + x[t] - demand[t];
  forall(t in 1..T)
    x[t] <= demand[t] * y[t];
}

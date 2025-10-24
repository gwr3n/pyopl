// Plant Location Problem
int nbPlants = ...;
range Plants = 1..nbPlants;

int nbCustomers = ...;
range Customers = 1..nbCustomers;

float trans_cost[Plants][Plants] = ...;
float fixed_cost[Plants] = ...;
float demand[Plants] = ...;
float capacity[Plants] = ...;

dvar boolean y[Plants];
dvar float+ x[Plants][Customers];

minimize sum(i in Plants) fixed_cost[i] * y[i] + sum(i in Plants, j in Customers) trans_cost[i][j] * x[i][j];

subject to {
  forall(j in Customers)
    sum(i in Plants) x[i][j] == demand[j];
  forall(i in Plants, j in Customers)
    x[i][j] <= capacity[i] * y[i];
}


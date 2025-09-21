// Warehouse Location Problem
int nbWarehouses = ...;
int nbCustomers = ...;

range Warehouses = 1..nbWarehouses;
range Customers = 1..nbCustomers;

float fixed_cost[Warehouses] = ...;
float trans_cost[Warehouses][Customers] = ...;
float demand[Customers] = ...;
float capacity[Warehouses] = ...;

dvar boolean y[Warehouses];
dvar float+ x[Warehouses][Customers];

minimize sum(i in Warehouses) fixed_cost[i] * y[i] + sum(i in Warehouses, j in Customers) trans_cost[i][j] * x[i][j];

subject to {
  forall(j in Customers)
    sum(i in Warehouses) x[i][j] == demand[j];
  forall(i in Warehouses, j in Customers)
    x[i][j] <= capacity[i] * y[i];
}


// Production Planning
int nbProducts = ...;
range Products = 1..nbProducts;
int nbPeriods = ...;
range Periods = 1..nbPeriods;
float cost[Products][Periods] = ...;
float demand[Periods] = ...;
float capacity[Periods] = ...;

dvar float+ x[Products][Periods];

minimize sum(p in Products, t in Periods) cost[p][t] * x[p][t];

subject to {
  forall(p in Products)
    sum(t in Periods) x[p][t] >= demand[p];
  forall(t in Periods)
    sum(p in Products) x[p][t] <= capacity[t];
}

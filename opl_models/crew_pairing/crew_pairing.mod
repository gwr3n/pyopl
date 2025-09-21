// Crew Pairing Problem
int nbPairings = ...;
range Pairings = 1..nbPairings;
int nbFlights = ...;
range Flights = 1..nbFlights;
float cost[Pairings] = ...;
boolean a[Pairings][Flights] = ...;

dvar boolean x[Pairings];

minimize sum(i in Pairings) cost[i] * x[i];

subject to {
  forall(j in Flights)
    sum(i in Pairings) (a[i][j]) * (x[i]) >= 1;
}


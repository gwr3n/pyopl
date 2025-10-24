// Crew Scheduling
int nbCrew = ...;
range Crew = 1..nbCrew;
int nbShifts = ...;
range Shifts = 1..nbShifts;

float cost[Crew][Shifts] = ...;
float max_shifts[Shifts] = ...;

dvar boolean x[Crew][Shifts];

minimize sum(i in Crew, j in Shifts) cost[i][j] * x[i][j];

subject to {
  forall(j in Shifts)
    sum(i in Crew) (x[i][j]) == 1;
  forall(i in Crew)
    sum(j in Shifts) (x[i][j]) <= max_shifts[i];
}


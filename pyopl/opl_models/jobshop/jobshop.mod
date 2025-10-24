// Job Shop Scheduling Problem
int nbJobs = ...;
int nbMachines = ...;
range Jobs = 1..nbJobs;
range Machines = 1..nbMachines;
int duration[Jobs][Machines] = ...;
int M = 1000;

dvar int+ start[Jobs][Machines];
dvar boolean z[Jobs][Jobs][Machines];
dvar int+ makespan;

minimize makespan;

subject to {
  // Each job must be processed on each machine in order
  forall(j in Jobs, m in Machines)
    start[j][m] >= 0;
  // No overlap on machines (simplified)
  forall(m in Machines)
    forall(j1 in Jobs, j2 in Jobs: j1 != j2){
      start[j1][m] + duration[j1][m] <=  start[j2][m] - 1 + M * z[j1][j2][m];
      start[j2][m] + duration[j2][m] <=  start[j1][m] - 1 + M * (1 - z[j1][j2][m]);
    }
  // Each job must be processed on each machine in order
  forall(j in Jobs, m in 1..nbMachines-1)
    start[j][m+1] >= start[j][m] + duration[j][m];
  // Makespan constraint
  forall(j in Jobs)
    makespan >= start[j][nbMachines] + duration[j][nbMachines];
}


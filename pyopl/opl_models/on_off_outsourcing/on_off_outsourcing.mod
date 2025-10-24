// On/Off Production with Outsourcing
int T = ...;
range Periods = 1..T;

float demand[Periods] = ...;
float cap[Periods] = ...;
float cProd[Periods] = ...;
float cOut[Periods] = ...;
float setupCost = ...;

// Decision variables
// Operational state
dvar boolean run[Periods];
dvar boolean start[Periods];
dvar boolean endRun[Periods];

// Quantities
dvar float+ y[Periods];   // in-house production
dvar float+ o[Periods];   // outsourced quantity

// Disjunctive coverage selectors
dvar boolean zin[Periods];   // choose in-house to fully cover demand
dvar boolean zout[Periods];  // choose outsourcing to fully cover demand

minimize totalCost:
  sum(t in Periods) ( setupCost * start[t] + cProd[t] * y[t] + cOut[t] * o[t] );

subject to {
  // Startup tracking (assume plant initially off before period 1)
  start[1] == run[1];
  forall(t in 2..T) {
    start[t] >= run[t] - run[t-1];
    start[t] <= run[t];
    start[t] <= 1 - run[t-1];
  }

  // Shutdown tracking (end of horizon is treated as a shutdown if running)
  forall(t in 1..T-1) {
    endRun[t] >= run[t] - run[t+1];
    endRun[t] <= run[t];
    endRun[t] <= 1 - run[t+1];
  }
  endRun[T] == run[T];

  // Production only when running; limited by capacity
  forall(t in Periods)
    y[t] <= cap[t] * run[t];

  // Logical OR: at least one source fully covers demand in each period
  forall(t in Periods) {
    y[t] >= demand[t] * zin[t];
    o[t] >= demand[t] * zout[t];
    zin[t] + zout[t] >= 1;       // in-house OR outsourcing covers demand
    zin[t] <= run[t];             // can only pick in-house coverage if running
  }
}
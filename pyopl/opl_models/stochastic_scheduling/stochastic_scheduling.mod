// University Lab Presentation Scheduling with Stochastic Demand and Overtime Recourse
//
// Problem mapping (literate overview):
// - Horizon: three consecutive days (index set Days = 1..T, T=3).
// - Regular capacity: baseCap[d] hours per day (given as 8 each day in data).
// - Demand uncertainty: five equally likely scenarios of arrivals per day,
//   structured so that two scenarios share the same Day-1 attendance and three share another,
//   requiring non-anticipativity before Day-2.
// - Decisions:
//     * o[s][d]  = overtime hours used on day d in scenario s (integer, >= 0).
//     * p[s][d]  = number of presentations processed on day d in scenario s (integer, >= 0).
//     * b[s][d]  = backlog (students waiting) at end of day d in scenario s (integer, >= 0).
// - Non-anticipative commitments:
//     * o0_D1    = Day-1 overtime (same for all scenarios).
//     * o0_D2[g] = Day-2 overtime per information group g (same for scenarios sharing Day-1 history).
// - Flow logic: backlog carries unmet demand forward; execution adapts to realized attendance.
// - Service level: by end of day 3, at least serviceRate (95%) of total scenario demand must be processed in each scenario.
// - Cost: overtime costs otCost per hour; objective minimizes expected overtime cost across scenarios.
//
// Modeling notes:
// - Presentations are one-hour indivisible slots; hence integer variables for p, o, b.
// - Daily processing is limited by (regular capacity + overtime) and availability (arrivals + backlog).
// - Regular capacity baseCap is fixed; overtime is the adjustable lever at each stage.

// -------------------- Sets and indices --------------------
int T = ...;                 // number of days (T=3)
range Days = 1..T;           // day indices
range Days2 = 2..T;          // days 2..T for rolling constraints
{string} Scenarios;          // scenario labels
{int} Groups;                // group identifiers used by groupDay2

// -------------------- Parameters --------------------
param float prob[Scenarios] = ...;          // scenario probabilities (sum to 1)
param int arrivals[Scenarios][Days] = ...;  // arrivals per day per scenario (students)
param int baseCap[Days] = ...;              // regular capacity per day (hours)
param float otCost = ...;                   // cost per hour of overtime
param float serviceRate = 0.95;             // required fraction served by end of horizon

// Information structure for Day-2 decisions: scenarios that share the same
// Day-1 history are grouped together via a group identifier.
// Scenarios with the same groupDay2 value are indistinguishable before Day 2.
param int groupDay2[Scenarios] = ...;       // e.g., {S1->1, S2->1, S3->2, S4->2, S5->2}

// Derived: total demand per scenario (compile-time computed indexed parameter)
param int totalDemand[s in Scenarios] = sum(d in Days) arrivals[s][d];

// -------------------- Decision variables --------------------
dvar int+ o[Scenarios][Days];   // overtime hours used on day d in scenario s
// processed presentations and backlog are scenario specific (recourse)
dvar int+ p[Scenarios][Days];   // processed on day d in scenario s
dvar int+ b[Scenarios][Days];   // backlog at end of day d in scenario s

// Non-anticipative overtime commitments:
// - o0_D1 is the Day-1 overtime decided before any uncertainty is revealed
// - o0_D2[g] is the Day-2 overtime decided after Day-1 is observed but before Day-2 is realized
//   (same within each information group g in Groups)
dvar int+ o0_D1;               // Day-1 overtime (same for all scenarios)
dvar int+ o0_D2[Groups];       // Day-2 overtime per info group

// -------------------- Objective --------------------
// Objective (ExpectedOvertimeCost): minimize expected overtime cost across scenarios and days.
// Uses scenario probabilities prob[s] and per-day overtime o[s][d].
minimize ExpectedOvertimeCost =
  otCost * ( sum(s in Scenarios, d in Days) prob[s] * o[s][d] );

// -------------------- Constraints --------------------
subject to {
  // Day-1 flow balance and capacity per scenario
  forall(s in Scenarios) {
    Balance_D1:
      b[s][1] == arrivals[s][1] - p[s][1];
    Capacity_D1:
      p[s][1] <= baseCap[1] + o[s][1];
    Availability_D1:
      p[s][1] <= arrivals[s][1];
  }

  // Days 2..T rolling balance and capacity per scenario and day
  forall(s in Scenarios, d in Days2) {
    Balance_Rolling:
      b[s][d] == b[s][d-1] + arrivals[s][d] - p[s][d];
    Capacity_Rolling:
      p[s][d] <= baseCap[d] + o[s][d];
    Availability_Rolling:
      p[s][d] <= b[s][d-1] + arrivals[s][d];
  }

  // ServiceLevel: at least serviceRate fraction of scenario demand served by end of Day T
  forall(s in Scenarios)
    ServiceLevel:
      sum(d in Days) p[s][d] >= serviceRate * totalDemand[s];

  // ProbSumToOne: scenario probabilities must sum to 1 (ground constraint)
  ProbSumToOne:
    (sum(s in Scenarios) prob[s]) == 1;

  // ---------------- Non-anticipativity ----------------
  // O_NA_D1: identical Day-1 overtime decision across all scenarios
  forall(s in Scenarios) O_NA_D1:
    o[s][1] == o0_D1;

  // O_NA_D2_link: identical Day-2 overtime decision for scenarios that share the same Day-1 history.
  // FIX: Replace dynamic index o0_D2[groupDay2[s]] with a 0/1 mask over Groups.
  // The boolean (groupDay2[s] == g) is ground and treated as 0/1, selecting exactly one g.
  forall(s in Scenarios) O_NA_D2_link:
    o[s][2] == sum(g in Groups) (groupDay2[s] == g) * o0_D2[g];

  // (No non-anticipativity on Day-3 because Day-2 realizations fully reveal the scenario
  // under the enriched tree in the provided data.)
}
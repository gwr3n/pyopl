/*------------------------------------------------------------------------------
 WORKFORCE PLANNING MODEL (Literate, documented version)

 Problem summary (what this model optimizes):
 - Multi-period workforce planning with multiple skill levels and task types.
 - Decisions per period:
   * hire/fire by skill
   * train between adjacent skill levels (s -> s+1)
   * assign worker-hours by skill to tasks
   * use overtime (bounded per worker)
 - Must meet each task's demand using qualified skills.
 - Productive capacity is workforce × productivity plus overtime.
 - Hires/fires/training are bounded each period.
 - Total headcount limited by managerial span of control.
 - Per-period spend (hire, fire, training, regular wages, overtime) must not exceed budget.
 - Objective: minimize total cost over all periods.

 Modeling notes and units:
 - Time periods are discrete (e.g., months).
 - Skills are ordered; training moves workers only to the next higher skill.
 - assign[s][t][p] is measured in worker-hours in period p.
 - productivity[s] is worker-hours per worker per period (regular time).
 - overtime[s][p] is overtime hours (int) for skill s in period p.
 - workforce[s][p], hire, fire, train are headcounts (integers).
 - Costs (hiring/firing/training/wages/overtime) are in monetary units per respective unit.
 - The demand satisfaction constraint sums only over qualified skills; the model does not
   explicitly forbid assigning unqualified skills to a task, but since assignment increases
   cost and does not help meet demand, optimal solutions will set such assignments to 0.
   If you prefer to enforce this structurally, you can add:
      forall(s in Skills, t in Tasks, p in Periods: skillsRequired[t][s]==0)
         assign[s][t][p] == 0;

 Data files:
 - See the .dat file for a small example with S=2 skills, K=2 tasks, T=3 periods.

------------------------------------------------------------------------------*/

//---------------------------
// Dimensions and index sets
//---------------------------

int T = ...;     // Number of time periods
int S = ...;     // Number of skill levels (ordered: 1..S)
int K = ...;     // Number of task/job types

range Periods   = 1..T;    // p
range Skills    = 1..S;    // s
range SkillTrans= 1..S-1;  // training arcs: from s to s+1
range Tasks     = 1..K;    // t

//----------------------------------
// Parameters (inputs from .dat file)
//----------------------------------

// Costs per head (hire/fire) by skill; currency per worker
float hiringCost[Skills];
float firingCost[Skills];

// Training cost per worker trained from s to s+1; only defined for s in 1..S-1
float trainingCost[SkillTrans];

// Labor costs per hour by skill; wage for regular hours, otWage for overtime hours
float wage[Skills];
float otWage[Skills];

// Regular-time productivity: worker-hours produced per worker per period
float productivity[Skills];

// Max overtime hours allowed per worker per period
float maxOvertime[Skills];

// Initial headcount by skill at the start of period 1
int initialWorkforce[Skills];

// Task demand in worker-hours per task per period
int demand[Tasks][Periods];

// Qualification matrix: 1 if skill s can perform task t, else 0
int skillsRequired[Tasks][Skills];

// Per-period spend limit (budget) across all cost categories
float budget[Periods];

// Operational bounds on hires/fires per skill per period (headcounts)
int maxHire[Skills][Periods];
int maxFire[Skills][Periods];

// Managerial span of control: each manager can oversee up to spanControl workers
int spanControl;
int nManagers;

//----------------------
// Decision variables
//----------------------

// Headcount flows (integers, nonnegative)
dvar int+ hire[Skills][Periods];              // workers hired at skill s in period p
dvar int+ fire[Skills][Periods];              // workers fired at skill s in period p
dvar int+ train[SkillTrans][Periods];         // workers trained from s to s+1, launched in period p

// Assignment and capacity variables
dvar int+ assign[Skills][Tasks][Periods];     // worker-hours at skill s on task t in period p (regular + overtime)
dvar int+ overtime[Skills][Periods];          // overtime hours for skill s in period p

// Workforce state (end-of-period headcount at skill s in period p)
dvar int+ workforce[Skills][Periods];

//----------------------
// Objective: Minimize total cost across the horizon
// Components: hire + fire + training + regular wages (on regular hours) + overtime wages
//----------------------
minimize
  sum(s in Skills, p in Periods) (hiringCost[s] * hire[s][p] + firingCost[s] * fire[s][p])
+ sum(s in SkillTrans, p in Periods) (trainingCost[s] * train[s][p])
// Pay regular wage on regular hours only: (assigned hours - overtime)
+ sum(s in Skills, p in Periods) (wage[s] * (sum(t in Tasks) assign[s][t][p] - overtime[s][p]))
// Pay overtime at the full overtime rate
+ sum(s in Skills, p in Periods) (otWage[s] * overtime[s][p]);

//----------------------
// Constraints
//----------------------
subject to {

  //--- Workforce balance in the first period (no incoming training yet)
  // End-of-period workforce = initial + hires - fires
  workforce[1][1] == initialWorkforce[1] + hire[1][1] - fire[1][1];
  forall(s in 2..S)
    workforce[s][1] == initialWorkforce[s] + hire[s][1] - fire[s][1];

  //--- Workforce transitions for later periods
  // Skill 1 (lowest): loses trainees going to skill 2 from previous period
  forall(p in 2..T)
    workforce[1][p] == workforce[1][p-1] + hire[1][p] - fire[1][p] - train[1][p-1];

  // Intermediate skills 2..S-1: gain trainees from lower, lose trainees to higher
  forall(s in 2..S-1, p in 2..T)
    workforce[s][p] == workforce[s][p-1] + hire[s][p] - fire[s][p] + train[s-1][p-1] - train[s][p-1];

  // Highest skill S: gains trainees from S-1 only (no outflow upward)
  forall(p in 2..T)
    workforce[S][p] == workforce[S][p-1] + hire[S][p] - fire[S][p] + train[S-1][p-1];

  //--- Capacity: assigned hours cannot exceed regular capacity plus overtime
  // Regular capacity = workforce × productivity
  forall(s in Skills, p in Periods)
    sum(t in Tasks) assign[s][t][p] <= workforce[s][p] * productivity[s] + overtime[s][p];

  //--- Overtime per worker cap
  forall(s in Skills, p in Periods)
    overtime[s][p] <= workforce[s][p] * maxOvertime[s];

  // Ensure overtime hours are a subset of assigned hours (overtime must be worked on tasks)
  forall(s in Skills, p in Periods)
    overtime[s][p] <= sum(t in Tasks) assign[s][t][p];

  //--- Operational bounds on hires/fires per period
  forall(s in Skills, p in Periods)
    hire[s][p] <= maxHire[s][p];
  forall(s in Skills, p in Periods)
    fire[s][p] <= maxFire[s][p];

  //--- Cannot fire more than available at the start of the period
  // Period 1: limited by initial workforce
  forall(s in Skills)
    fire[s][1] <= initialWorkforce[s];
  // Later periods: limited by last period's end-of-period workforce
  forall(s in Skills, p in 2..T)
    fire[s][p] <= workforce[s][p-1];

  //--- Training launch bounds
  // Training in period p consumes workers available at the start of p
  forall(s in SkillTrans)
    train[s][1] <= initialWorkforce[s];
  forall(s in SkillTrans, p in 2..T)
    train[s][p] <= workforce[s][p-1];

  //--- Demand satisfaction by qualified skills
  // For each task and period, sum of assigned hours from qualified skills must cover demand
  forall(t in Tasks, p in Periods)
    sum(s in Skills : skillsRequired[t][s] == 1) assign[s][t][p] >= demand[t][p];

  // Optional tightening (not required for correctness due to cost structure):
  // forall(s in Skills, t in Tasks, p in Periods : skillsRequired[t][s]==0)
  //   assign[s][t][p] == 0;

  //--- Managerial span-of-control constraint on total headcount per period
  // Total workforce cannot exceed managers × spanControl
  forall(p in Periods)
    sum(s in Skills) workforce[s][p] <= nManagers * spanControl;

  //--- Per-period budget constraint
  // Sum of all costs incurred in a period must not exceed that period's budget
  forall(p in Periods)
    sum(s in Skills)
      (hiringCost[s] * hire[s][p]
     + firingCost[s] * fire[s][p]
     + wage[s] * (sum(t in Tasks) assign[s][t][p] - overtime[s][p])   // regular wage on regular hours
     + otWage[s] * overtime[s][p])                                    // overtime wage on overtime hours
     + sum(s in SkillTrans) trainingCost[s] * train[s][p]
    <= budget[p];
}

/*------------------------------------------------------------------------------
 Validation and extensions (for model users/readers):

 - Feasibility: If budget or hire/fire bounds are too tight to meet demand,
   the model may become infeasible. Consider increasing budget, allowing more
   hiring/overtime, or relaxing demand as a soft constraint with penalties.

 - Integrality: Variables are integers to reflect headcounts and whole hours.
   If large instances are slow, you can relax assign/overtime to floats.

 - Training timing: Training launched in period p reduces the source skill in p
   and increases the destination skill in p+1 (via balance). If training should
   be completed within the same period, adjust the balance equations accordingly.

 - Qualifications: To strictly forbid unqualified assignment, add the optional
   constraint shown above.

 - Alternative objectives: You can also minimize unmet demand penalties,
   or maximize service level under a budget, by adjusting the objective and
   adding slack variables on demand.

------------------------------------------------------------------------------*/

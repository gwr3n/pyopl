// WORKFORCE PLANNING MODEL
// This model addresses workforce planning by optimizing hiring, firing, training, and assignment of workers to tasks
// while considering costs, productivity, and workforce constraints over a defined time horizon.
//
// ASSUMPTIONS:
// * Time is discretized into periods.
// * There is a finite and known set of skill levels and tasks.
// * Productivity is normalized per worker per period.
// * Overtime is allowed only up to a specified maximum per worker.
// * All monetary values (costs, wages) and worker-hours are known input data.

int T = ...;
int S = ...;
int K = ...;

range Periods = 1..T;
range Skills = 1..S;
range SkillTrans = 1..S-1;
range Tasks = 1..K;

float hiringCost[Skills];
float firingCost[Skills];
float trainingCost[SkillTrans];
float wage[Skills];
float otWage[Skills];
float productivity[Skills];
float maxOvertime[Skills];
int initialWorkforce[Skills];
int demand[Tasks][Periods];
int skillsRequired[Tasks][Skills];
float budget[Periods];
int maxHire[Skills][Periods];
int maxFire[Skills][Periods];
int spanControl;
int nManagers;

dvar int+ hire[Skills][Periods];
dvar int+ fire[Skills][Periods];
dvar int+ train[SkillTrans][Periods];
dvar int+ assign[Skills][Tasks][Periods];
dvar int+ overtime[Skills][Periods];
dvar int+ workforce[Skills][Periods];

minimize
sum(s in Skills, p in Periods) (hiringCost[s] * hire[s][p] + firingCost[s] * fire[s][p])
+ sum(s in SkillTrans, p in Periods) trainingCost[s] * train[s][p]
+ sum(s in Skills, p in Periods) wage[s] * sum(t in Tasks) assign[s][t][p]
+ sum(s in Skills, p in Periods) otWage[s] * overtime[s][p];

subject to {
	workforce[1][1] == initialWorkforce[1] + hire[1][1] - fire[1][1];
	forall(s in 2..S)
		workforce[s][1] == initialWorkforce[s] + hire[s][1] - fire[s][1];

	forall(p in 2..T)
		workforce[1][p] == workforce[1][p-1] + hire[1][p] - fire[1][p] - train[1][p-1];

	forall(s in 2..S-1, p in 2..T)
		workforce[s][p] == workforce[s][p-1] + hire[s][p] - fire[s][p] + train[s-1][p-1] - train[s][p-1];

	forall(p in 2..T)
		workforce[S][p] == workforce[S][p-1] + hire[S][p] - fire[S][p] + train[S-1][p-1];

	forall(s in Skills, p in Periods)
		sum(t in Tasks) assign[s][t][p] <= workforce[s][p]*productivity[s] + overtime[s][p];

	forall(s in Skills, p in Periods)
		overtime[s][p] <= workforce[s][p]*maxOvertime[s];

	forall(s in Skills, p in Periods)
		hire[s][p] <= maxHire[s][p];
	forall(s in Skills, p in Periods)
		fire[s][p] <= maxFire[s][p];

	forall(s in Skills)
		fire[s][1] <= initialWorkforce[s];
	forall(s in Skills, p in 2..T)
		fire[s][p] <= workforce[s][p-1];

	forall(s in SkillTrans)
		train[s][1] <= initialWorkforce[s];
	forall(s in SkillTrans, p in 2..T)
		train[s][p] <= workforce[s][p-1];

	forall(t in Tasks, p in Periods)
		sum(s in Skills : skillsRequired[t][s]==1) assign[s][t][p] >= demand[t][p];

	forall(p in Periods)
		sum(s in Skills) workforce[s][p] <= nManagers * spanControl;

	forall(p in Periods)
		sum(s in Skills)
		(hiringCost[s]*hire[s][p] + firingCost[s]*fire[s][p] + wage[s]*sum(t in Tasks) assign[s][t][p] + otWage[s]*overtime[s][p])
		+ sum(s in SkillTrans)
		trainingCost[s]*train[s][p]
		<= budget[p];
}
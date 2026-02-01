/*
Maintenance MDP via Average-Cost Linear Programming (occupation-measure / steady-state LP)

Intent:
- States i ∈ S are deterioration levels (e.g., Good/Poor/Failed).
- Actions a ∈ A are maintenance decisions (e.g., DoNothing/Repair/Replace).
- P[i][a][j] is the probability of moving to next state j from state i under action a.
- cost[i][a] is the immediate cost of taking action a in state i.

Decision variables:
- x[i][a] = steady-state (long-run) probability of being in state i and choosing action a.

LP:
- Minimize long-run average cost: sum_{i,a} cost[i,a] * x[i,a]
- Subject to:
  (i) flow balance for each state,
  (ii) normalization of total probability mass,
  (iii) feasibility masking for disallowed (state, action) pairs,
  (iv) nonnegativity (via float+).
*/

/********************
 * Sets and indices *
 ********************/
{int} States = ...;                  // State set S (e.g., {1,2,3})
{int} Actions = ...;                 // Action set A (e.g., {1,2,3})

/****************
 * Input data   *
 ****************/
param float   cost[States][Actions] = ...;          // Immediate cost C(i,a)
param float   P[States][Actions][States] = ...;     // Transition probability P(j | i,a)
param boolean allowed[States][Actions] = ...;       // True iff action a is allowed in state i

/************************
 * Decision variables   *
 ************************/
// x[i][a] ≥ 0 is the occupation measure (steady-state joint probability of (state=i, action=a)).
dvar float+ x[States][Actions];

/****************
 * Objective     *
 ****************/
minimize AverageCost:
  // Long-run expected average cost per unit time
  sum(i in States, a in Actions) cost[i][a] * x[i][a];

/****************
 * Constraints   *
 ****************/
subject to {
  // Normalization: total steady-state probability mass equals 1
  Normalize:
    sum(i in States, a in Actions) x[i][a] == 1;

  // Flow balance: probability mass in each state equals probability mass transitioning into it
  forall(j in States)
    FlowBalance:
      sum(a in Actions) x[j][a]
        ==
      sum(i in States, a in Actions) x[i][a] * P[i][a][j];

  // Action feasibility mask: disallowed actions must carry zero probability mass
  // Boolean is coerced to {0,1}, so if allowed[i][a]=false then x[i][a] <= 0.
  forall(i in States, a in Actions)
    AllowedMask:
      x[i][a] <= allowed[i][a];

  // Data sanity (optional but safe): enforce stochastic transitions only for allowed actions
  forall(i in States, a in Actions : allowed[i][a])
    RowStochastic:
      sum(j in States) P[i][a][j] == 1;
}


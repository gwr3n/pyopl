A maintenance planner manages a single asset that can be in any of several discrete condition states representing levels of deterioration, and at each decision epoch must choose one maintenance action from a given set such as doing nothing, repairing, or replacing; taking an action in a state incurs an immediate cost specified by an input table, and the asset then transitions probabilistically to a next condition state according to a provided transition matrix that depends on both the current state and chosen action. Some actions may be infeasible in certain states, and this is captured by an input feasibility indicator that forces the decision to assign zero long-run usage to any disallowed state–action combination. The goal is to find a stationary, long-run operating policy that minimizes the steady-state average cost per time step, modeled using occupation measures: for each state and action, a nonnegative decision variable represents the long-run fraction of time the system is in that state while selecting that action. These occupation measures must sum to one overall to represent a valid steady-state probability distribution, and they must satisfy steady-state flow conservation for every state, meaning that the total long-run probability mass assigned to being in a state (summed over actions) equals the total long-run probability mass flowing into that state from all other state–action pairs weighted by their transition probabilities. For data consistency, the transition probabilities for any allowed state–action pair are required to form a proper probability distribution over next states, and the optimal solution of this linear program yields both the minimum achievable long-run average cost and, by normalizing within each state, an equivalent stationary randomized maintenance policy.

SETS (with meaning)

States
+-------+--------+
| Value | Name   |
+-------+--------+
| 1     | Good   |
| 2     | Poor   |
| 3     | Failed |
+-------+--------+

Actions
+-------+-----------+
| Value | Name      |
+-------+-----------+
| 1     | Do nothing|
| 2     | Repair    |
| 3     | Replace   |
+-------+-----------+


IMMEDIATE COSTS: cost[i][a]

+----------------+------------+--------+---------+
| State \ Action | 1 Do nothing| 2 Repair| 3 Replace|
+----------------+------------+--------+---------+
| 1 Good         | 0.0        | 2.0    | 10.0    |
| 2 Poor         | 0.0        | 4.0    | 10.0    |
| 3 Failed       | 50.0       | 20.0   | 10.0    |
+----------------+------------+--------+---------+


FEASIBILITY MASK: allowed[i][a]

+----------------+-------------+----------+-----------+
| State \ Action | 1 Do nothing| 2 Repair | 3 Replace |
+----------------+-------------+----------+-----------+
| 1 Good         | true        | false    | true      |
| 2 Poor         | true        | true     | true      |
| 3 Failed       | false       | false    | true      |
+----------------+-------------+----------+-----------+


TRANSITION PROBABILITIES: P[i][a][j]
(Columns are next state j; each row is a distribution over next states.)

Next-state labels: 1=Good, 2=Poor, 3=Failed

+---------------+-----------+------------+-----------+-------------+
| Current state | Action    | Next: 1    | Next: 2   | Next: 3     |
+---------------+-----------+------------+-----------+-------------+
| 1 Good        | 1 Do nothing | 0.7     | 0.3       | 0.0         |
| 1 Good        | 2 Repair     | 1.0     | 0.0       | 0.0         |
| 1 Good        | 3 Replace   | 1.0      | 0.0       | 0.0         |
| 2 Poor        | 1 Do nothing | 0.0     | 0.6       | 0.4         |
| 2 Poor        | 2 Repair     | 0.5     | 0.5       | 0.0         |
| 2 Poor        | 3 Replace   | 1.0      | 0.0       | 0.0         |
| 3 Failed      | 1 Do nothing | 0.0     | 0.0       | 1.0         |
| 3 Failed      | 2 Repair     | 0.0     | 0.0       | 1.0         |
| 3 Failed      | 3 Replace   | 1.0      | 0.0       | 0.0         |
+---------------+-----------+------------+-----------+-------------+

Note: Feasibility (allowed[i][a]) determines which of the above action rows are permitted:
- State 1: actions 1 and 3 allowed (action 2 disallowed)
- State 2: actions 1, 2, 3 allowed
- State 3: action 3 allowed only (actions 1 and 2 disallowed)

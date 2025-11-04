/* 
Set Partitioning Model (literate commentary)

Problem intent:
- Given a collection of candidate sets (i in Sets) with costs cost[i],
  choose a subset so that every element (j in Elements) is covered by
  exactly one chosen set, minimizing total cost.

Data interface (from .dat):
- nbSets: number of candidate sets (e.g., 3)
- nbElements: number of elements to cover (e.g., 4)
- cost[i]: cost of selecting set i (e.g., [5,6,7])
- a[i][j]: coverage indicator; 1 if set i covers element j, else 0
  (e.g., rows describe which elements each set covers)

Feasibility note:
- The equality “== 1” enforces exact partitioning (no under- or over-coverage).
- If no combination yields exactly one cover per element, the model is infeasible.
*/
int nbSets = ...;      // number of candidate sets to choose from
int nbElements = ...;  // number of elements that must be covered exactly once

// Index sets
range Sets = 1..nbSets;
range Elements = 1..nbElements;

// Parameters supplied by the .dat file
float cost[Sets] = ...;        // selection cost for each set i
int   a[Sets][Elements] = ...; // binary coverage matrix: a[i][j] in {0,1}

/*
Decision variables:
x[i] = 1 if set i is selected in the solution, 0 otherwise.
*/
dvar boolean x[Sets];

/*
Objective:
Minimize the total cost of the selected sets.
*/
minimize sum(i in Sets) cost[i] * x[i];

/*
Partitioning constraints:
For each element j, exactly one chosen set must cover it.

Interpretation:
- sum_i a[i][j] * x[i] counts how many chosen sets cover element j.
- Setting this sum equal to 1 enforces “covered by exactly one set”.
*/
subject to {
  forall(j in Elements)
    sum(i in Sets) (a[i][j] * x[i]) == 1;
}

/*
Model usage:
- Use with set_partitioning.dat that defines nbSets, nbElements, cost, and a.
- Example (from the attached .dat):
  nbSets = 3; nbElements = 4;
  cost = [5,6,7];
  a = [[1,0,0,1],[0,1,1,0],[0,0,1,1]];
*/


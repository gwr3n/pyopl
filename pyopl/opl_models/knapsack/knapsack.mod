/*
  0-1 Knapsack (literate, narrative style)

  Problem (in words):
  - A set of items is available. Each item i has a weight weight[i] and a value value[i].
  - We want to pick a subset of items to maximize the total value while keeping the total weight
    within a given capacity C. Each item is either taken (1) or not taken (0).

  Data used here (example):
  - In knapsack.dat, there are N = 5 items with:
      weight = [2, 3, 4, 5, 5]
      value  = [2, 3, 4, 5, 5]
      C      = 10
    This model reads weight and value from the .dat file. The capacity C is currently set in the
    model (see below), but could be read from the .dat as well.

  Model outline:
  - Index set Items identifies items.
  - Parameters weight[i], value[i] define data per item.
  - Decision variables x[i] ∈ {0,1} indicate selection of item i.
  - Objective: maximize sum_i value[i] * x[i].
  - Constraint: sum_i weight[i] * x[i] ≤ C.

  Notes:
  - This is the classic 0-1 knapsack. If you wanted the fractional variant, x would be continuous
    with 0 ≤ x[i] ≤ 1. If you wanted “bounded” knapsack (e.g., take up to k copies), x would be
    integer with an upper bound per item.
  - The arrays weight and value must be aligned (same length and indexing).
*/

/*
  Index set of items
  ------------------
  The .dat file provides arrays of length 5 in this example. We mirror that with a fixed range 1..5.
  If your data size varies, you can drive this from the .dat using:
    int N = ...;           // provided in the .dat
    range Items = 1..N;    // then use N here
  For the current example, we keep the explicit 1..5.
*/
range Items = 1..5;

/*
  Data interface (arrays read from .dat)
  --------------------------------------
  Both arrays are indexed over Items and are supplied via the .dat file.
  Example (from knapsack.dat):
    weight = [2,3,4,5,5];
    value  = [2,3,4,5,5];
  Consistency requirement:
  - weight and value must have the same length and correspond element-wise.
*/
float weight[Items] = ...;
float value[Items]  = ...;

/*
  Knapsack capacity
  -----------------
  The capacity C limits the total weight of chosen items.
  - In this model we set C = 10 explicitly to match the example.
  - If you prefer to read C from the .dat (knapsack.dat defines C = 10), replace the line below with:
      float C = ...;
*/
float C = 10;

/*
  Decision variables
  ------------------
  x[i] = 1 if item i is selected, else 0.
  - Boolean (0/1) means each item is either taken or not taken (no fractions).
*/
dvar boolean x[Items];

/*
  Objective
  ---------
  Maximize the sum of values of selected items.
  This is the dot product value · x = sum_i value[i] * x[i].
*/
maximize
  sum (i in Items) value[i] * x[i];

/*
  Capacity constraint
  -------------------
  The total weight of selected items must not exceed C.
  This enforces feasibility for the knapsack.
*/
subject to {
  sum (i in Items) weight[i] * x[i] <= C;
}

/*
  Usage
  -----
  - Pair this model with knapsack.dat.
  - If you switch to reading C from the .dat, change C as noted above.
  - After solving, inspect x[i] to see which items are chosen, and compute totals if desired.
*/
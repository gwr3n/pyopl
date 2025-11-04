// -----------------------------------------------------------------------------
// Crew Pairing Problem (Set Covering Formulation)
// -----------------------------------------------------------------------------
// Problem summary:
// Given a set of feasible crew pairings (each is a valid sequence of flights)
// and an associated cost per pairing, select a subset of pairings so that
// every flight is covered by at least one selected pairing while minimizing
// total cost.
//
// Data is provided via the .dat file (see crew_pairing.dat):
// - nbPairings: number of candidate pairings
// - nbFlights: number of flights to cover
// - cost[i]: cost of pairing i
// - a[i][j] = 1 if pairing i covers flight j, else 0
//
// This model is a classic set covering problem, commonly used for airline
// crew pairing, where "sets" are pairings and "elements" are flights.
// -----------------------------------------------------------------------------

// ----------------------------
// Sets and basic parameters
// ----------------------------

// Number of candidate pairings available.
int nbPairings = ...;

// Index set of pairings: Pairings = {1, 2, ..., nbPairings}.
range Pairings = 1..nbPairings;

// Number of flights that must be covered.
int nbFlights = ...;

// Index set of flights: Flights = {1, 2, ..., nbFlights}.
range Flights = 1..nbFlights;

// ----------------------------
// Input data
// ----------------------------

// cost[i] is the cost of selecting pairing i.
// Typically derived from block time, hotel costs, or other operational metrics.
float cost[Pairings] = ...;

// a[i][j] is the incidence matrix:
// a[i][j] = 1 if pairing i covers flight j; 0 otherwise.
boolean a[Pairings][Flights] = ...;

// ----------------------------
// Decision variables
// ----------------------------

// x[i] = 1 if pairing i is selected in the solution; 0 otherwise.
dvar boolean x[Pairings];

// ----------------------------
// Objective
// ----------------------------

// Minimize the total cost of the selected pairings.
// Each selected pairing i contributes cost[i] to the objective.
minimize sum(i in Pairings) cost[i] * x[i];

// ----------------------------
// Constraints
// ----------------------------

// Flight coverage:
// For every flight j, the sum of selected pairings that cover j must be >= 1.
// This ensures each flight is covered by at least one chosen pairing.
// Note: Using ">= 1" allows overlapping coverage (multiple pairings can cover
// the same flight). If exactly one coverage is required, change to "== 1"
// (only if the pairing set guarantees feasibility under that restriction).
subject to {
  forall(j in Flights)
    sum(i in Pairings) (a[i][j]) * (x[i]) >= 1;
}

// ----------------------------
// Notes and extensions
// ----------------------------
// - To discourage redundant coverage without enforcing equality, add a small
//   penalty for overlapping coverage or introduce column generation methods.
// - Additional constraints frequently used in practice include maximum duty time,
//   base balancing, and incompatibility constraints between pairings.
// - If costs are integral, "float" can be changed to "int" for cost.
// - This compact model assumes the pairing set is pre-generated and feasible.
// -----------------------------------------------------------------------------


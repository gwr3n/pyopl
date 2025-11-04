/*
Proper Graph Coloring via big-M linearization

Problem
- Assign a positive integer color to each node of a graph so that adjacent nodes have different colors.
- Minimize the number of colors used, modeled as the maximum color assigned to any node.

Data (see graph_coloring.dat)
- nbNodes: number of nodes.
- Edges: set of (source, dest) node pairs that define adjacency, e.g., { <1,2>, <2,3> }.

Modeling choices
- Colors are bounded in [1, nbNodes]. This makes M = nbNodes a valid big-M.
- Instead of using "color[u] != color[v]" directly, we encode the disjunction
  color[u] >= color[v] + 1  OR  color[v] >= color[u] + 1
  using a binary z[e] and big-M constraints.

Notes
- This is a proper coloring (no equal colors on adjacent nodes).
- The formulation may have symmetry (permuting colors), which can be reduced with
  additional symmetry-breaking constraints if needed (not included here).
*/

/// -------------------------------
/// Sets, indices, and input data
/// -------------------------------

// Number of nodes in the graph (from .dat)
int nbNodes = ...;

// Canonical node index set: 1, 2, ..., nbNodes
range Nodes = 1..nbNodes;

// Edge is a tuple of endpoints (1-based indices into Nodes)
tuple Edge {
    int source;
    int dest;
};

// Set of undirected or directed edges (treated as adjacency) provided in .dat
{Edge} Edges = ...;

/// --------------------------------------
/// Decision variables and their semantics
/// --------------------------------------

/*
color[i] = integer color assigned to node i
- Restricted to be positive (>= 1) and further bounded by nbNodes below.
- We keep it integer to represent discrete colors.
*/
dvar int+ color[Nodes];

/*
maxColor = maximum color used across all nodes
- The objective minimizes this, effectively minimizing the number of colors used.
*/
dvar int+ maxColor;

/*
z[e] = binary selector for edge e = (u, v)
- z[e] = 0 enforces color[u] >= color[v] + 1
- z[e] = 1 enforces color[v] >= color[u] + 1
This encodes "color[u] != color[v]" without using "!=" directly.
*/
dvar boolean z[Edges];

/// -------------
/// Objective
/// -------------

/*
Minimize the maximum color used; this equals the chromatic number upper-bounded
by nbNodes (the trivial bound).
*/
minimize maxColor;

/// -------------
/// Constraints
/// -------------

subject to {

    // -- Color bounds: 1 <= color[i] <= nbNodes for all nodes i
    // These bounds justify using M = nbNodes in the big-M constraints below.
    forall(i in Nodes) color[i] >= 1;
    forall(i in Nodes) color[i] <= nbNodes;

    // -- Adjacency constraints via big-M (no direct "!=")
    // For each edge e = (u, v), exactly one of the two inequalities must hold:
    //   (1) color[u] >= color[v] + 1        if z[e] = 0
    //   (2) color[v] >= color[u] + 1        if z[e] = 1
    // The inactive side is relaxed by subtracting/adding M = nbNodes.
    forall(e in Edges)
        color[e.source] >= color[e.dest] + 1 - nbNodes * z[e];

    forall(e in Edges)
        color[e.dest] >= color[e.source] + 1 - nbNodes * (1 - z[e]);

    // -- Link maxColor to node colors: maxColor >= color[i] for all i
    // Minimizing maxColor then minimizes the number of colors used.
    forall(i in Nodes) maxColor >= color[i];
}
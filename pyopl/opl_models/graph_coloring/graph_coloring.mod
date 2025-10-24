// Proper Graph Coloring Problem (no !=, uses big-M encoding)
int nbNodes = ...;
range Nodes = 1..nbNodes;

tuple Edge {
    int source;
    int dest;
};

{Edge} Edges = ...;

dvar int+ color[Nodes];
dvar int+ maxColor;
dvar boolean z[Edges]; // auxiliary binary for big-M encoding

minimize maxColor;

subject to {
    // Each node's color is at least 1 and at most nbNodes
    forall(i in Nodes) color[i] >= 1;
    forall(i in Nodes) color[i] <= nbNodes;
    // Adjacent nodes must have different colors (big-M encoding)
    forall(e in Edges)
        color[e.source] >= color[e.dest] + 1 - nbNodes * z[e];
    forall(e in Edges)
        color[e.dest] >= color[e.source] + 1 - nbNodes * (1 - z[e]);
    // maxColor is at least as large as any color used
    forall(i in Nodes) maxColor >= color[i];
}
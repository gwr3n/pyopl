import os
import shutil
import tempfile

from pyopl import solve


# --- Random Vehicle Routing Problem Data Generator ---
def generate_vehicle_routing_data(num_nodes, density=0.4, cost_range=(5.0, 20.0), seed=42):
    """
    Generate a random sparse cost matrix and corresponding tuple-based arc list for a vehicle routing problem.
    - num_nodes: number of nodes in the problem
    - density: fraction of possible arcs to include (excluding self-loops)
    - cost_range: (min, max) cost for arcs
    - seed: random seed for replicability
    Returns: arcs, cost_matrix, num_nodes
    """
    import random

    random.seed(seed)
    M = 1e20  # Large number for missing arcs
    cost_matrix = [[M for _ in range(num_nodes)] for _ in range(num_nodes)]
    arcs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue  # No self-loops
            if random.random() < density:
                cost = round(random.uniform(*cost_range), 2)
                cost_matrix[i][j] = cost
                arcs.append((i + 1, j + 1, cost))  # OPL is 1-based
    return arcs, cost_matrix, num_nodes


# --- Vehicle Routing Problem Data (from test_vehicle_routing_with_tuples) ---
def get_vehicle_routing_data():
    # Tuple-based data
    arcs = [(1, 2, 10.0), (2, 3, 12.5), (3, 1, 8.0)]
    num_nodes = 3
    # Matrix-based data (dense, but with zeros for missing arcs)
    M = 1e20  # A large number to represent "missing" arcs
    cost_matrix = [[M, 10.0, M], [M, M, 12.5], [8.0, M, M]]
    return arcs, cost_matrix, num_nodes


def write_matrix_mod_dat(mod_path, dat_path, matrix, num_nodes):
    with open(mod_path, "w") as f:
        f.write(
            f"""
// Matrix-based vehicle routing problem
range Nodes = 1..{num_nodes};
float cost[Nodes][Nodes] = ...;
dvar boolean x[Nodes][Nodes];
minimize sum(i in Nodes, j in Nodes) cost[i][j] * x[i][j];
subject to {{
  forall(i in Nodes)
    sum(j in Nodes) (x[i][j]) == 1;
  forall(j in Nodes)
    sum(i in Nodes) (x[i][j]) == 1;
}}
"""
        )
    with open(dat_path, "w") as f:
        f.write("cost = [\n")
        for idx, row in enumerate(matrix):
            line = "  [" + ", ".join(str(v) for v in row) + "]"
            if idx < len(matrix) - 1:
                line += ","
            f.write(line + "\n")
        f.write("];")


def write_tuple_mod_dat(mod_path, dat_path, arcs, num_nodes):
    with open(mod_path, "w") as f:
        f.write(
            f"""
// Tuple-based vehicle routing problem
tuple Arc {{ int from; int to; float cost; }};
{{Arc}} arcs;
dvar boolean x[arcs];
minimize sum(a in arcs) a.cost * x[a];
subject to {{
  forall(i in 1..{num_nodes})
    sum(a in arcs : a.from == i) (x[a]) == 1;
  forall(j in 1..{num_nodes})
    sum(a in arcs : a.to == j) (x[a]) == 1;
}}
"""
        )
    with open(dat_path, "w") as f:
        f.write("arcs = {")
        for idx, t in enumerate(arcs):
            comma = "," if idx < len(arcs) - 1 else ""
            f.write(f" <{t[0]},{t[1]},{t[2]}>{comma}")
        f.write("};\n")


def file_size(path):
    return os.path.getsize(path)


def run_sparse_example(solver):
    # Selector variables for data type and random instance parameters
    USE_RANDOM = True  # Set to True for random instance, False for fixed example
    RANDOM_NODES = 100  # Number of nodes for random instance
    RANDOM_DENSITY = 0.1  # Arc density for random instance
    RANDOM_SEED = 42  # Random seed

    if USE_RANDOM:
        arcs, cost_matrix, num_nodes = generate_vehicle_routing_data(
            num_nodes=RANDOM_NODES, density=RANDOM_DENSITY, seed=RANDOM_SEED
        )
        print(f"Generated random instance with {num_nodes} nodes, density={RANDOM_DENSITY}, seed={RANDOM_SEED}")
    else:
        arcs, cost_matrix, num_nodes = get_vehicle_routing_data()
    temp_dir = tempfile.mkdtemp(prefix="pyopl_sparse_")
    try:
        # Matrix version
        matrix_mod = os.path.join(temp_dir, "matrix.mod")
        matrix_dat = os.path.join(temp_dir, "matrix.dat")
        write_matrix_mod_dat(matrix_mod, matrix_dat, cost_matrix, num_nodes)
        with open(matrix_mod) as f:
            print("--- matrix.mod ---")
            print(f.read())
        if num_nodes <= 10:
            with open(matrix_dat) as f:
                print("--- matrix.dat ---")
                print(f.read())
        # Tuple version
        tuple_mod = os.path.join(temp_dir, "tuple.mod")
        tuple_dat = os.path.join(temp_dir, "tuple.dat")
        write_tuple_mod_dat(tuple_mod, tuple_dat, arcs, num_nodes)
        with open(tuple_mod) as f:
            print("--- tuple.mod ---")
            print(f.read())
        if num_nodes <= 10:
            with open(tuple_dat) as f:
                print("--- tuple.dat ---")
                print(f.read())
        # File sizes
        matrix_dat_size = file_size(matrix_dat)
        tuple_dat_size = file_size(tuple_dat)
        print(f"Matrix .dat size: {matrix_dat_size} bytes")
        print(f"Tuple .dat size: {tuple_dat_size} bytes")
        print(f"Number of arcs (nonzero entries): {len(arcs)}")

        print(f"\n--- Solving matrix version with {solver} ---")
        result_matrix = solve(matrix_mod, matrix_dat, solver=solver)
        print(f"Status: {result_matrix.get('status')}")
        print(f"Objective: {result_matrix.get('objective_value')}")
        print(f"Stats: {result_matrix.get('stats')}")
        print(f"\n--- Solving tuple version with {solver} ---")
        result_tuple = solve(tuple_mod, tuple_dat, solver=solver)
        print(f"Status: {result_tuple.get('status')}")
        print(f"Objective: {result_tuple.get('objective_value')}")
        print(f"Stats: {result_tuple.get('stats')}")
    finally:
        shutil.rmtree(temp_dir)
        print(f"Temporary files cleaned up from {temp_dir}")

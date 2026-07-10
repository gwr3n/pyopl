from enum import Enum, auto
from pathlib import Path

from pyopl import solve

OPL_MODELS_DIR = Path(__file__).resolve().parents[1] / "pyopl" / "opl_models"


def _model_path(relative_path: str) -> str:
    return str(OPL_MODELS_DIR / relative_path)


# Example definitions
def run_knapsack(solver):
    """Run the classic Knapsack problem using the selected solver."""
    model = _model_path("knapsack/knapsack.mod")
    data = _model_path("knapsack/knapsack.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_knapsackp(solver):
    """Run the Knapsack problem variant using the selected solver."""
    model = _model_path("knapsack/knapsackp.mod")
    data = _model_path("knapsack/knapsackp.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_lot_sizing(solver):
    """Run the Lot Sizing problem (single item) using the selected solver."""
    model = _model_path("lot_sizing/lot_sizing.mod")
    data = _model_path("lot_sizing/lot_sizing.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_tsp(solver):
    """Run the Traveling Salesman Problem (TSP) using the selected solver."""
    model = _model_path("tsp/tsp.mod")
    data = _model_path("tsp/tsp.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_vehicle_routing(solver):
    """Run the Vehicle Routing Problem (single vehicle) using the selected solver."""
    model = _model_path("vehicle_routing/vehicle_routing.mod")
    data = _model_path("vehicle_routing/vehicle_routing.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_set_covering(solver):
    """Run the Set Covering Problem using the selected solver."""
    model = _model_path("set_covering/set_covering.mod")
    data = _model_path("set_covering/set_covering.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


# --- Additional Examples ---
def run_assignment(solver):
    """Run the Assignment Problem using the selected solver."""
    model = _model_path("assignment/assignment.mod")
    data = _model_path("assignment/assignment.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_transportation(solver):
    """Run the Transportation Problem using the selected solver."""
    model = _model_path("transportation/transportation.mod")
    data = _model_path("transportation/transportation.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_production(solver):
    """Run the Production Planning Problem using the selected solver."""
    model = _model_path("production/production.mod")
    data = _model_path("production/production.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_plant_location(solver):
    """Run the Plant Location Problem using the selected solver."""
    model = _model_path("plant_location/plant_location.mod")
    data = _model_path("plant_location/plant_location.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_crew_scheduling(solver):
    """Run the Crew Scheduling Problem using the selected solver."""
    model = _model_path("crew_scheduling/crew_scheduling.mod")
    data = _model_path("crew_scheduling/crew_scheduling.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_graph_coloring(solver):
    """Run the Graph Coloring Problem using the selected solver."""
    model = _model_path("graph_coloring/graph_coloring.mod")
    data = _model_path("graph_coloring/graph_coloring.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_set_partitioning(solver):
    """Run the Set Partitioning Problem using the selected solver."""
    model = _model_path("set_partitioning/set_partitioning.mod")
    data = _model_path("set_partitioning/set_partitioning.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_warehouse_location(solver):
    """Run the Warehouse Location Problem using the selected solver."""
    model = _model_path("warehouse_location/warehouse_location.mod")
    data = _model_path("warehouse_location/warehouse_location.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_jobshop(solver):
    """Run the Job Shop Scheduling Problem using the selected solver."""
    model = _model_path("jobshop/jobshop.mod")
    data = _model_path("jobshop/jobshop.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_crew_pairing(solver):
    """Run the Crew Pairing Problem using the selected solver."""
    model = _model_path("crew_pairing/crew_pairing.mod")
    data = _model_path("crew_pairing/crew_pairing.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_covering(solver):
    """Run the Set Covering Problem (variant) using the selected solver."""
    model = _model_path("covering/covering.mod")
    data = _model_path("covering/covering.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_workforce_planning(solver):
    """Run the Workforce Planning Problem using the selected solver."""
    model = _model_path("workforce_planning/workforce_planning.mod")
    data = _model_path("workforce_planning/workforce_planning.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_inventory_routing(solver):
    """Run the Inventory Routing Problem using the selected solver."""
    model = _model_path("inventory_routing/inventory_routing.mod")
    data = _model_path("inventory_routing/inventory_routing.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_sparse_example(solver):
    import logging

    from tools._sparse_example import run_sparse_example

    logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(name)s: %(message)s")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    run_sparse_example(solver_name)


def run_p_dispersion(solver):
    """Run the P-Dispersion Problem using the selected solver."""
    model = _model_path("p-dispersion/p-dispersion.mod")
    data = _model_path("p-dispersion/p-dispersion.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


def run_on_off_outsourcing(solver):
    """Run the On-Off Outsourcing Problem using the selected solver."""
    model = _model_path("on_off_outsourcing/on_off_outsourcing.mod")
    data = _model_path("on_off_outsourcing/on_off_outsourcing.dat")
    solver_name = "scipy" if solver == Solver.SCIPY else "gurobi"
    result = solve(model, data, solver=solver_name)
    print(result)


# Enum for examples


class Example(Enum):
    """Enumeration of available example problems."""

    KNAPSACK = auto()  # Classic 0-1 Knapsack Problem
    KNAPSACKP = auto()  # Multi-Resource Knapsack Problem with Partitioned Capacities
    LOT_SIZING = auto()  # Lot Sizing Problem (single item)
    TSP = auto()  # Traveling Salesman Problem (TSP)
    VEHICLE_ROUTING = auto()  # Vehicle Routing Problem (single vehicle)
    SET_COVERING = auto()  # Set Covering Problem
    ASSIGNMENT = auto()  # Assignment Problem
    TRANSPORTATION = auto()  # Transportation Problem
    PRODUCTION = auto()  # Production Planning Problem
    PLANT_LOCATION = auto()  # Plant Location Problem
    CREW_SCHEDULING = auto()  # Crew Scheduling Problem
    GRAPH_COLORING = auto()  # Graph Coloring Problem
    SET_PARTITIONING = auto()  # Set Partitioning Problem
    WAREHOUSE_LOCATION = auto()  # Warehouse Location Problem
    JOBSHOP = auto()  # Job Shop Scheduling Problem
    CREW_PAIRING = auto()  # Crew Pairing Problem
    COVERING = auto()  # Set Covering Problem
    WORKFORCE_PLANNING = auto()  # Workforce Planning Problem
    INVENTORY_ROUTING = auto()  # Inventory Routing Problem
    SPARSE_EXAMPLE = auto()  # Sparse Data Example
    P_DISPERSION = auto()  # P-Dispersion Problem
    ON_OFF_OUTSOURCING = auto()  # On-Off Outsourcing Problem


# List of available examples and their functions (all now take solver argument)
EXAMPLES = {
    Example.KNAPSACK: run_knapsack,
    Example.KNAPSACKP: run_knapsackp,
    Example.LOT_SIZING: run_lot_sizing,
    Example.TSP: run_tsp,
    Example.VEHICLE_ROUTING: run_vehicle_routing,
    Example.SET_COVERING: run_set_covering,
    Example.ASSIGNMENT: run_assignment,
    Example.TRANSPORTATION: run_transportation,
    Example.PRODUCTION: run_production,
    Example.PLANT_LOCATION: run_plant_location,
    Example.CREW_SCHEDULING: run_crew_scheduling,
    Example.GRAPH_COLORING: run_graph_coloring,
    Example.SET_PARTITIONING: run_set_partitioning,
    Example.WAREHOUSE_LOCATION: run_warehouse_location,
    Example.JOBSHOP: run_jobshop,
    Example.CREW_PAIRING: run_crew_pairing,
    Example.COVERING: run_covering,
    Example.WORKFORCE_PLANNING: run_workforce_planning,
    Example.INVENTORY_ROUTING: run_inventory_routing,
    Example.SPARSE_EXAMPLE: run_sparse_example,
    Example.P_DISPERSION: run_p_dispersion,
    Example.ON_OFF_OUTSOURCING: run_on_off_outsourcing,
}


# Enum for solvers
class Solver(Enum):
    SCIPY = auto()
    GUROBI = auto()


# Selector: set this to one of the Example enum values
EXAMPLE_SELECTOR = Example.LOT_SIZING  # e.g. Example.LOT_SIZING, Example.TSP, etc.

# Selector: set this to one of the Solver enum values
SOLVER_SELECTOR = Solver.GUROBI  # e.g. Solver.SCIPY, Solver.GUROBI

# Run the selected example
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] %(name)s: %(message)s")

    func = EXAMPLES.get(EXAMPLE_SELECTOR)
    if func:
        func(SOLVER_SELECTOR)
    else:
        print(f"Unknown example selector: {EXAMPLE_SELECTOR}")

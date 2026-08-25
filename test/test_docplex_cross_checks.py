import ast
import inspect
import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from importlib.util import find_spec

from pyopl.pyopl_core import solve

CPLEX_STUDIO = os.environ.get("CPLEX_STUDIO", "")
OPLRUN = os.environ.get("OPLRUN", "")


def build_docplex_model(model_path, data_path):
    """Export OPL with Studio 22.1.1, then load the LP into DOcplex."""
    from docplex.mp.model_reader import ModelReader  # type: ignore[import-untyped]

    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False) as lp_file:
        lp_path = lp_file.name
    try:
        completed = subprocess.run(
            [OPLRUN, "-e", lp_path, model_path, data_path],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"oplrun failed with exit status {completed.returncode}: " f"{completed.stdout}\n{completed.stderr}"
            )
        return ModelReader.read(lp_path)
    finally:
        os.remove(lp_path)


def map_solution_variables(docplex_model, pyopl_solution):
    variables = {variable.name: variable for variable in docplex_model.iter_variables()}

    def normalize(name):
        name = re.sub(r"(?<=\d)\.0+(?=[^\d]|$)", "", name.lower())
        return re.sub(r"[^a-z0-9]", "", name)

    def zero_based(name):
        return re.sub(r"\d+", lambda match: str(int(match.group()) - 1), name)

    normalized_variables = {normalize(name): variable for name, variable in variables.items()}
    zero_based_variables = {normalize(name): variable for name, variable in variables.items() if "#" in name}
    mapped = {}
    for name, value in pyopl_solution.items():
        variable = None
        if name.startswith("z"):
            pair_names = re.findall(r"\(([^()]*)\)", name)
            if len(pair_names) == 2:
                first = tuple(int(value.strip()) for value in pair_names[0].split(","))
                second = tuple(int(value.strip()) for value in pair_names[1].split(","))
                if first == second or first[2] != second[2] or first >= second:
                    continue
        if name.startswith("a_") or name.startswith("a["):
            variable = zero_based_variables.get(normalize(zero_based(name)))
        if variable is None:
            variable = variables.get(name) or zero_based_variables.get(normalize(zero_based(name)))
        if variable is None:
            variable = normalized_variables.get(normalize(name))
        if variable is None:
            raise AssertionError(f"No DOcplex variable matches PyOPL variable {name!r}; available: {sorted(variables)}")
        mapped[variable] = value
    return mapped


@unittest.skipUnless(
    CPLEX_STUDIO and OPLRUN,
    "DOcplex cross-checks require CPLEX_STUDIO and OPLRUN environment variables",
)
class TestDocplexCrossChecks(unittest.TestCase):
    def test_multi_knapsack_pyopl_vs_cplex_output(self):
        """Compare the multi-resource knapsack model with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/multi_knapsack/multi_knapsack.mod")
        data_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/multi_knapsack/multi_knapsack.dat")
        results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file,
        ):
            with open(model_path) as source_mod_file:
                docplex_model_code = source_mod_file.read().replace("Value", "knapsackObjectiveCoefficient")
                docplex_model_code = docplex_model_code.replace(
                    "float knapsackObjectiveCoefficient[Items];",
                    "float knapsackObjectiveCoefficient[Items] = ...;",
                ).replace(
                    "float Use[Resources][Items];",
                    "float Use[Resources][Items] = ...;",
                )
                docplex_mod_file.write(docplex_model_code)
            with open(data_path) as source_dat_file:
                docplex_dat_file.write(source_dat_file.read().replace("Value", "knapsackObjectiveCoefficient"))
            docplex_model_path = docplex_mod_file.name
            docplex_data_path = docplex_dat_file.name
        try:
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_knapsack_pyopl_vs_cplex_output(self):
        """Compare the repository knapsack model with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/knapsack/knapsack.mod")
        data_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/knapsack/knapsack.dat")
        results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
        docplex_model = build_docplex_model(model_path, data_path)
        docplex_solution = docplex_model.solve(log_output=False)
        self.assertIsNotNone(docplex_solution)
        docplex_objective = docplex_solution.objective_value

        for solver, result in results.items():
            with self.subTest(solver=solver):
                self.assertEqual(result["status"], "OPTIMAL")
                self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                fixed_model = docplex_model.clone()
                mapped_solution = map_solution_variables(docplex_model, result["solution"])
                for variable, value in mapped_solution.items():
                    fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                candidate = fixed_model.solve(log_output=False)
                self.assertIsNotNone(candidate)
                self.assertTrue(candidate.is_valid_solution())
                self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)

    def test_inventory_routing_pyopl_vs_cplex_output(self):
        """Compare the inventory-routing fixture with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_path = os.path.join(
            os.path.dirname(__file__),
            "../pyopl/opl_models/inventory_routing/inventory_routing.mod",
        )
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../pyopl/opl_models/inventory_routing/inventory_routing.dat",
        )
        results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file,
        ):
            with open(model_path) as source_mod_file:
                docplex_model_code = source_mod_file.read().replace("param float ", "float ")
                for declaration in (
                    "float holding_cost;",
                    "float transport_cost[Stores];",
                    "float capacity[Stores];",
                    "float demand[Stores][Periods];",
                    "float init_inv[Stores];",
                ):
                    docplex_model_code = docplex_model_code.replace(declaration, declaration[:-1] + " = ...;")
            docplex_mod_file.write(docplex_model_code)
            docplex_dat_file.write("""Stores = { "StoreA", "StoreB", "StoreC" };
holding_cost = 0.5;
transport_cost = [2.0, 3.0, 1.5];
capacity = [100, 80, 60];
init_inv = [10, 5, 8];
demand = [[10, 12, 8], [5, 7, 6], [8, 9, 7]];
""")
            docplex_model_path = docplex_mod_file.name
            docplex_data_path = docplex_dat_file.name
        try:
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_knapsack_problem_compare_solvers(self):
        """Compare the generated knapsack fixture with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
range Items = 1..5;
param float weight[1..5];
param float value[1..5];
param float C;

dvar boolean x[1..5];

maximize sum (i in Items) (value[i] * x[i]);

subject to {
    sum (i in Items) (weight[i] * x[i]) <= C;
}
"""
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as model_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as data_file,
        ):
            docplex_model_code = model_code.replace("param float weight[1..5];", "float weight[1..5] = [2, 3, 4, 5, 5];")
            docplex_model_code = docplex_model_code.replace("param float value[1..5];", "float value[1..5] = [2, 3, 4, 5, 5];")
            docplex_model_code = docplex_model_code.replace("param float C;", "float C = 10;")
            model_file.write(docplex_model_code)
            data_file.write("")
            model_path = model_file.name
            data_path = data_file.name
        try:
            results = {
                "gurobi": solve(model_path, data_path, solver="gurobi"),
                "scipy": solve(model_path, data_path, solver="scipy"),
            }
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_assignment_problem_compare_solvers(self):
        """Compare the generated assignment fixture with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
dvar boolean assign[1..2][1..2];
range Persons = 1..2;
range Tasks = 1..2;

minimize sum (p in Persons) (sum (t in Tasks) (5 * assign[p][t]));

subject to {
    forall (p in Persons)
        sum (t in Tasks) (assign[p][t]) == 1;
    forall (t in Tasks)
        sum (p in Persons) (assign[p][t]) == 1;
}
"""
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as model_file:
            model_file.write(model_code)
            model_path = model_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as data_file:
            data_path = data_file.name
        try:
            results = {
                "gurobi": solve(model_path, data_path, solver="gurobi"),
                "scipy": solve(model_path, data_path, solver="scipy"),
            }
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_stochastic_multi_echelon(self):
        """Compare the source test's two-stage supply-chain model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_stochastic_multi_echelon)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = re.sub(r"\bprod\b", "production", model_code.replace("param float", "float"))
        docplex_model_code = re.sub(r"\bprob\b", "scenarioProb", docplex_model_code)
        docplex_model_code = docplex_model_code.replace(
            "float scenarioProb[Scenarios] = ...;",
            "float scenarioProb[Scenarios] = [0.25, 0.50, 0.25];",
        )
        docplex_model_code = docplex_model_code.replace(
            "float scenarioProb[Scenarios];",
            "float scenarioProb[Scenarios] = [0.25, 0.50, 0.25];",
        )
        docplex_model_code = docplex_model_code.replace("minimize TotalExpectedCost:", "minimize")
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_model_code = (
            docplex_model_code.replace("string p; string w;", "string plant; string warehouse;")
            .replace("string w; string c;", "string warehouse; string customer;")
            .replace("string s; string c;", "string scenario; string customer;")
            .replace("pw.p", "pw.plant")
            .replace("pw.w", "pw.warehouse")
            .replace("wc.w", "wc.warehouse")
            .replace("wc.c", "wc.customer")
            .replace("<s,c>", "<s,c>")
        )
        docplex_data_code = """
        Plants = { "P1", "P2" }; Warehouses = { "W1", "W2" };
        Customers = { "C1", "C2" }; Scenarios = { "Low", "Base", "High" };
        PlantWarehouses = { <"P1","W1">, <"P1","W2">, <"P2","W1">, <"P2","W2"> };
        WarehouseCustomers = { <"W1","C1">, <"W1","C2">, <"W2","C1">, <"W2","C2"> };
        ScenarioCustomers = { <"Low","C1">, <"Low","C2">, <"Base","C1">, <"Base","C2">, <"High","C1">, <"High","C2"> };
        prodCost = [[4.0,4.2,4.1], [4.5,4.4,4.6]];
        shipPWCost = [[1,1,1.1], [1.4,1.4,1.5], [1.3,1.2,1.2], [0.9,0.9,1]];
        shipWCCost = [[2,2.1,2.1], [2.8,2.9,2.9], [2.7,2.8,2.8], [1.9,2,2]];
        holdPlantCost = [[0.5,0.5,0.5], [0.6,0.6,0.6]];
        holdWHCost = [[0.8,0.8,0.8], [0.8,0.8,0.8]];
        shortageCost = [[12,12,12], [12,12,12]];
        prodCap = [[90,95,95], [80,85,85]];
        shipPWCap = [[70,70,70], [50,50,50], [50,50,50], [70,70,70]];
        shipWCCap = [[60,60,60], [60,60,60], [60,60,60], [60,60,60]];
        invPlantCap = [[40,40,40], [40,40,40]];
        invWHCap = [[50,50,50], [50,50,50]];
        initPlantInv = [10,8]; initWHInv = [12,10];
        demand = [[45,50,48], [35,38,40], [55,60,58], [45,48,50], [65,72,70], [55,58,60]];
        expDemandP1 = [55,45];
        """
        parameter_names = (
            "scenarioProb",
            "prodCost",
            "shipPWCost",
            "shipWCCost",
            "holdPlantCost",
            "holdWHCost",
            "shortageCost",
            "prodCap",
            "shipPWCap",
            "shipWCCap",
            "invPlantCap",
            "invWHCap",
            "initPlantInv",
            "initWHInv",
            "demand",
            "expDemandP1",
        )
        for name in parameter_names:
            assignment = re.search(rf"{name}\s*=\s*(\[.*?\]);", docplex_data_code, re.DOTALL)
            declaration = re.search(rf"float {name}\[[^;]+\];", docplex_model_code)
            if assignment and declaration:
                docplex_model_code = (
                    docplex_model_code[: declaration.start()]
                    + declaration.group(0)[:-1]
                    + " = "
                    + assignment.group(1)
                    + ";"
                    + docplex_model_code[declaration.end() :]
                )
                docplex_data_code = docplex_data_code[: assignment.start()] + docplex_data_code[assignment.end() :]
        for collection, iterator, field, index in (
            ("PlantWarehouses", "pw", "p", "p"),
            ("PlantWarehouses", "pw", "w", "w"),
            ("WarehouseCustomers", "wc", "w", "w"),
            ("WarehouseCustomers", "wc", "c", "c"),
        ):
            docplex_model_code = docplex_model_code.replace(
                f"sum({iterator} in {collection} : {iterator}.{field} == {index})",
                f"sum({iterator} in {collection}) ({iterator}.{field} == {index} ? 1 : 0)",
            )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path, data_path = mod_file.name, dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
                docplex_model_path = docplex_mod_file.name
            with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file:
                docplex_dat_file.write(docplex_data_code)
                docplex_data_path = docplex_dat_file.name
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_solution.objective_value, places=6)
                    mapped_solution = {
                        (
                            "production_" + name[len("prod_") :]
                            if name.startswith("prod_")
                            else "production" + name[len("prod") :] if name.startswith("prod[") else name
                        ): value
                        for name, value in result["solution"].items()
                    }
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, mapped_solution))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_solution.objective_value,
                        places=6,
                    )
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_job_shop(self):
        """Compare the deterministic job-shop model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbJobs = ...;
        int nbMachines = ...;
        range Jobs = 1..nbJobs;
        range Machines = 1..nbMachines;
        int duration[Jobs][Machines] = ...;
        int M = 1000;
        dvar int+ start[Jobs][Machines];
        dvar boolean z[Jobs][Jobs][Machines];
        dvar int+ makespan;
        minimize makespan;
        subject to {
            forall(j in Jobs, m in Machines) start[j][m] >= 0;
            forall(m in Machines)
                forall(j1 in Jobs, j2 in Jobs : j1 != j2) {
                    start[j1][m] + duration[j1][m]
                        <= start[j2][m] - 1 + M * z[j1][j2][m];
                    start[j2][m] + duration[j2][m]
                        <= start[j1][m] - 1 + M * (1 - z[j1][j2][m]);
                }
            forall(j in Jobs, m in 1..nbMachines-1)
                start[j][m+1] >= start[j][m] + duration[j][m];
            forall(j in Jobs)
                makespan >= start[j][nbMachines] + duration[j][nbMachines];
        }
        """
        data_code = """
        nbJobs = 3;
        nbMachines = 2;
        duration = [[3, 2], [2, 4], [5, 1]];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        docplex_model_path = model_path + ".docplex.mod"
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model_code = (
                model_code.replace("{string} Employees = ...;", "{string} Employees = ...;")
                .replace("{string} Shifts = ...;", "{string} Shifts = ...;")
                .replace("int demand[Shifts];", "int demand[Shifts] = ...;")
                .replace("int pref[Pairs];", "int pref[Pairs] = ...;")
            )
            with open(docplex_model_path, "w") as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
            with open(docplex_data_path, "w") as docplex_dat_file:
                docplex_dat_file.write(data_code)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    start_solution = {name: value for name, value in result["solution"].items() if name.startswith("start")}
                    mapped_solution = map_solution_variables(docplex_model, start_solution)
                    for variable, value in mapped_solution.items():
                        if variable.name.startswith("start"):
                            fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_multistage_stochastic_portfolio(self):
        """Compare the multi-stage stochastic portfolio model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        range Assets = 1..2;
        range Scenarios = 1..4;
        float W0 = ...;
        float tc_buy = ...;
        float tc_sell = ...;
        float prob[Scenarios] = ...;
        float R1[Assets][Scenarios] = ...;
        float R2[Assets][Scenarios] = ...;
        dvar float+ x1[Assets];
        dvar float+ b1[Assets];
        dvar float+ s1[Assets];
        dvar float+ x2[Assets][Scenarios];
        dvar float+ b2[Assets][Scenarios];
        dvar float+ s2[Assets][Scenarios];
        dexpr float prev2[a in Assets][sc in Scenarios] = R1[a][sc] * x1[a];
        dexpr float termW[sc in Scenarios] = sum(a in Assets) R2[a][sc] * x2[a][sc];
        dexpr float Eterm = sum(sc in Scenarios) prob[sc] * termW[sc];
        maximize Eterm;
        subject to {
            forall(a in Assets) x1[a] == b1[a] - s1[a];
            W0 + (1 - tc_sell) * sum(a in Assets) s1[a]
                - (1 + tc_buy) * sum(a in Assets) b1[a] == 0;
            forall(a in Assets) x1[a] <= sum(i in Assets) x1[i];
            forall(sc in Scenarios, a in Assets)
                x2[a][sc] == prev2[a][sc] + b2[a][sc] - s2[a][sc];
            forall(sc in Scenarios)
                (1 - tc_sell) * sum(a in Assets) s2[a][sc]
                    - (1 + tc_buy) * sum(a in Assets) b2[a][sc] == 0;
            forall(sc in Scenarios, a in Assets)
                x2[a][sc] <= sum(i in Assets) x2[i][sc];
            forall(a in Assets) {
                x2[a][1] == x2[a][2];
                x2[a][3] == x2[a][4];
                b2[a][1] == b2[a][2];
                b2[a][3] == b2[a][4];
                s2[a][1] == s2[a][2];
                s2[a][3] == s2[a][4];
            }
            Eterm >= 99;
            sum(sc in Scenarios) prob[sc] == 1;
        }
        """
        data_code = """
        W0 = 100;
        tc_buy = 0.01;
        tc_sell = 0.01;
        prob = [0.25, 0.25, 0.25, 0.25];
        R1 = [[1.02, 1.02, 1.02, 1.02], [1.05, 1.05, 0.98, 0.98]];
        R2 = [[1.01, 1.03, 1.00, 1.02], [1.10, 0.95, 1.08, 0.92]];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(
                        docplex_model,
                        {
                            name: value
                            for name, value in result["solution"].items()
                            if not name.startswith("u[Depot]") and not name.startswith("u_Depot")
                        },
                    )
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_wagner_whitin_linear(self):
        """Compare the deterministic Wagner-Whitin model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int T = 5;
        float demand[1..T] = [20, 40, 30, 10, 50];
        float unit_cost = 2;
        float setup_cost = 100;
        float holding_cost = 1;
        dvar float x[1..T];
        dvar float s[0..T];
        dvar boolean y[1..T];
        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);
        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T)
                x[t] <= y[t] * sum(tt in t..T) demand[tt];
            forall(t in 1..T) {
                x[t] >= 0;
                x[t] <= 150;
                s[t] >= 0;
            }
            y[1] == 1;
            y[2] == 0;
            y[3] == 0;
            y[4] == 0;
            y[5] == 1;
        }
        """
        docplex_model_code = """
        int T = ...;
        float demand[1..T] = ...;
        float unit_cost = ...;
        float setup_cost = ...;
        float holding_cost = ...;
        dvar float x[1..T];
        dvar float s[0..T];
        dvar boolean y[1..T];
        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);
        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T)
                x[t] <= y[t] * sum(tt in t..T) demand[tt];
            forall(t in 1..T) {
                x[t] >= 0;
                x[t] <= 150;
                s[t] >= 0;
            }
            y[1] == 1;
            y[2] == 0;
            y[3] == 0;
            y[4] == 0;
            y[5] == 1;
        }
        """
        data_code = """
        T = 5;
        demand = [20, 40, 30, 10, 50];
        unit_cost = 2;
        setup_cost = 100;
        holding_cost = 1;
        """

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file:
            dat_file.write(data_code)
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(
                        docplex_model,
                        {
                            name: value
                            for name, value in result["solution"].items()
                            if not name.startswith("u[Depot]") and not name.startswith("u_Depot")
                        },
                    )
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(data_path)

    def test_wagner_whitin_backorders(self):
        """Compare the Wagner-Whitin backorders model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbPeriods = ...;
        range T = 1..nbPeriods;
        float demand[T] = ...;
        float setup_cost[T] = ...;
        float prod_cost[T] = ...;
        float hold_cost[T] = ...;
        float penalty_cost[T] = ...;
        dvar float+ Q[T];
        dvar boolean y[T];
        dvar float I[T];
        dvar float+ h[T];
        dvar float+ p[T];
        minimize sum(t in T) (setup_cost[t] * y[t]
            + prod_cost[t] * Q[t] + hold_cost[t] * h[t]
            + penalty_cost[t] * p[t]);
        subject to {
            balance_1: I[1] == Q[1] - demand[1];
            forall(t in 2..nbPeriods)
                balance_t: I[t] == I[t-1] + Q[t] - demand[t];
            forall(t in T)
                prod_link: Q[t] <= y[t] * sum(k in t..nbPeriods) demand[k];
            forall(t in T) hold_lb: h[t] >= I[t];
            forall(t in T) penal_lb: p[t] >= -I[t];
        }
        """
        docplex_model_code = model_code
        data_code = """
        nbPeriods = 6;
        demand = [20, 40, 30, 10, 50, 60];
        setup_cost = [100, 80, 100, 120, 110, 90];
        prod_cost = [5, 5, 5, 5, 5, 5];
        hold_cost = [1, 1, 1, 1, 1, 1];
        penalty_cost = [2, 2, 2, 2, 2, 2];
        """

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file:
            dat_file.write(data_code)
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(
                        docplex_model,
                        {
                            name: value
                            for name, value in result["solution"].items()
                            if not name.startswith("z")
                            or tuple(int(index) for index in re.findall(r"\d+", name))
                            in {(1, 2), (1, 4), (2, 1), (2, 3), (3, 2), (3, 4), (4, 1), (4, 3)}
                        },
                    )
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(data_path)

    def test_production_planning_conditional_compare_solvers_1(self):
        """Compare the first conditional production-planning model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int N = 6;
        range T = 1..N;
        dvar float+ Q[T];
        dvar float+ I[T];
        dvar boolean order[T];
        float demand[T] = ...;
        float K = 5;
        float h = 1;
        minimize sum(t in T) (K * order[t] + h * I[t]);
        subject to {
            forall(t in T) {
                Q[t] <= 410;
                if (t == 1) {
                    I[1] == Q[1] - demand[1];
                } else {
                    I[t] == I[t-1] + Q[t] - demand[t];
                }
                (order[t] == 0) => (Q[t] <= 0);
            }
        }
        """
        docplex_model_code = """
        int N = ...;
        range T = 1..N;
        dvar float+ Q[T];
        dvar float+ I[T];
        dvar boolean order[T];
        float demand[T] = ...;
        float K = ...;
        float h = ...;
        minimize sum(t in T) (K * order[t] + h * I[t]);
        subject to {
            I[1] == Q[1] - demand[1];
            forall(t in 2..N) I[t] == I[t-1] + Q[t] - demand[t];
            forall(t in T) {
                Q[t] <= 410;
                (order[t] == 0) => (Q[t] <= 0);
            }
        }
        """
        data_code = """
        N = 6;
        demand = [80, 60, 70, 90, 50, 60];
        K = 5;
        h = 1;
        """

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file:
            dat_file.write(data_code)
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(data_path)

    def test_production_planning_conditional_compare_solvers_2(self):
        """Compare the second conditional production-planning model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int N = 6;
        range T = 1..N;
        dvar float+ Q[T];
        dvar float+ I[T];
        dvar boolean order[T];
        float demand[T] = ...;
        float K = 5;
        float h = 1;
        minimize sum(t in T) (K * order[t] + h * I[t]);
        subject to {
            forall(t in T) {
                Q[t] <= 410;
                if (t == 1) {
                    I[1] == Q[1] - demand[1];
                } else {
                    I[t] == I[t-1] + Q[t] - demand[t];
                }
                (Q[t] > 0) => (order[t] == 1);
            }
        }
        """
        docplex_model_code = """
        int N = ...;
        range T = 1..N;
        dvar float+ Q[T];
        dvar float+ I[T];
        dvar boolean order[T];
        float demand[T] = ...;
        float K = ...;
        float h = ...;
        minimize sum(t in T) (K * order[t] + h * I[t]);
        subject to {
            I[1] == Q[1] - demand[1];
            forall(t in 2..N) I[t] == I[t-1] + Q[t] - demand[t];
            forall(t in T) {
                Q[t] <= 410;
                Q[t] <= 410 * order[t];
            }
        }
        """
        data_code = """
        N = 6;
        demand = [80, 60, 70, 90, 50, 60];
        K = 5;
        h = 1;
        """

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file:
            dat_file.write(data_code)
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(data_path)

    def test_production_planning_compare_solvers(self):
        """Compare the production-planning allocation model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbProducts = ...;
        range Products = 1..nbProducts;
        int nbPeriods = ...;
        range Periods = 1..nbPeriods;
        float cost[Products][Periods] = ...;
        float demand[Periods] = ...;
        float capacity[Periods] = ...;
        dvar float+ x[Products][Periods];
        minimize sum(p in Products, t in Periods) cost[p][t] * x[p][t];
        subject to {
            forall(p in Products) sum(t in Periods) x[p][t] >= demand[p];
            forall(t in Periods) sum(p in Products) x[p][t] <= capacity[t];
        }
        """
        docplex_model_code = model_code
        data_code = """
        nbProducts = 2;
        nbPeriods = 3;
        cost = [[3, 2, 4], [2, 3, 5]];
        demand = [40, 50, 0];
        capacity = [30, 40, 20];
        """

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file:
            dat_file.write(data_code)
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(data_path)

    def test_vehicle_routing_with_nested_tuples(self):
        """Compare the inline nested-tuple vehicle-routing model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple Node {
            int id;
            float x;
            float y;
        };
        tuple Arc {
            Node from;
            Node to;
            float cost;
        };
        {Node} nodes = { <1,0.0,0.0>, <2,1.0,0.0>, <3,0.0,1.0> };
        {Arc} arcs = { < <1,0.0,0.0>, <2,1.0,0.0>, 10.0 >,
            < <2,1.0,0.0>, <3,0.0,1.0>, 12.5 >,
            < <3,0.0,1.0>, <1,0.0,0.0>, 8.0 > };
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            forall(n in nodes)
                sum(a in arcs : a.from.id == n.id) x[a] == 1;
            forall(n in nodes)
                sum(a in arcs : a.to.id == n.id) x[a] == 1;
        }
        """
        docplex_model_code = model_code.replace("Node to;", "Node destination;")
        docplex_model_code = docplex_model_code.replace("a.to.id", "a.destination.id")

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file:
            docplex_data_path = docplex_dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_wagner_whitin_model_data(self):
        """Compare the external-data Wagner-Whitin model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int T = ...;
        float demand[1..T] = ...;
        float unit_cost = ...;
        float setup_cost = ...;
        float holding_cost = ...;
        dvar float x[1..T];
        dvar float s[0..T];
        dvar boolean y[1..T];
        minimize sum(t in 1..T)
            (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);
        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T)
                x[t] <= (sum(tt in t..T) demand[t]) * y[t];
            forall(t in 1..T) {
                x[t] >= 0;
                x[t] <= 150;
                s[t] >= 0;
            }
        }
        """
        docplex_model_code = model_code
        data_code = """
        T = 5;
        demand = [20, 40, 30, 10, 50];
        unit_cost = 2;
        setup_cost = 100;
        holding_cost = 1;
        """

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file:
            dat_file.write(data_code)
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(data_path)

    def test_wagner_whitin_implication(self):
        """Compare the Wagner-Whitin strict-implication model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int T = 5;
        float demand[1..T] = [20, 40, 30, 10, 50];
        float unit_cost = 2;
        float setup_cost = 100;
        float holding_cost = 1;
        dvar float x[1..T];
        dvar float s[0..T];
        dvar boolean y[1..T];
        minimize sum(t in 1..T)
            (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);
        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T) {
                x[t] >= 0;
                x[t] <= 150;
                s[t] >= 0;
            }
            forall(t in 1..T)
                (x[t] > 0) => (y[t] == 1);
        }
        """
        docplex_model_code = model_code.replace(
            "(x[t] > 0) => (y[t] == 1);",
            "x[t] <= 150 * y[t];",
        )

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file:
            mod_file.write(model_code)
            model_path = mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file:
            docplex_data_path = docplex_dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_vehicle_routing_with_nested_tuples_dat(self):
        """Compare nested tuple-indexed routing decisions with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple Node { int id; float x; float y; }
        tuple Arc { Node from; Node to; float cost; }
        {Node} nodes = ...;
        {Arc} arcs = ...;
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            forall(n in nodes)
                sum(a in arcs : a.from.id == n.id) x[a] == 1;
            forall(n in nodes)
                sum(a in arcs : a.to.id == n.id) x[a] == 1;
        }
        """
        docplex_model_code = model_code.replace(
            "Node to;",
            "Node destination;",
        ).replace(
            "a.to.id",
            "a.destination.id",
        )
        data_code = """
        nodes = { <1,0.0,0.0>, <2,1.0,0.0>, <3,0.0,1.0> };
        arcs = { < <1,0.0,0.0>, <2,1.0,0.0>, 10.0 >,
                 < <2,1.0,0.0>, <3,0.0,1.0>, 12.5 >,
                 < <3,0.0,1.0>, <1,0.0,0.0>, 8.0 > };
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
                docplex_model_path = docplex_mod_file.name
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(
                        docplex_model,
                        {name.replace("prod", "production", 1): value for name, value in result["solution"].items()},
                    )
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            if "docplex_model_path" in locals():
                os.remove(docplex_model_path)

    def test_vehicle_routing_with_tuples_dat(self):
        """Compare external tuple-indexed routing decisions with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple Arc {
            int from;
            int to;
            float cost;
        };
        {Arc} arcs = ...;
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            forall(i in 1..3)
                sum(a in arcs : a.from == i) x[a] == 1;
            forall(j in 1..3)
                sum(a in arcs : a.to == j) x[a] == 1;
        }
        """
        docplex_model_code = model_code.replace(
            "int to;",
            "int destination;",
        ).replace(
            "a.to",
            "a.destination",
        )
        data_code = """
        arcs = { <1,2,10.0>, <2,3,12.5>, <3,1,8.0> };
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_mod_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_vehicle_routing_with_tuples(self):
        """Compare inline tuple-indexed routing decisions with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple Arc {
            int from;
            int to;
            float cost;
        };
        {Arc} arcs = { <1,2,10.0>, <2,3,12.5>, <3,1,8.0> };
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            forall(i in 1..3)
                sum(a in arcs : a.from == i) x[a] == 1;
            forall(j in 1..3)
                sum(a in arcs : a.to == j) x[a] == 1;
        }
        """
        docplex_model_code = model_code.replace(
            "int to;",
            "int destination;",
        ).replace(
            "a.to",
            "a.destination",
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
        ):
            mod_file.write(model_code)
            docplex_mod_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_shortest_path_with_tuples(self):
        """Compare the integer tuple-arc shortest-path model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple Arc { int from; int to; float cost; }

        int N = ...;
        range Nodes = 1..N;
        {Arc} arcs = ...;
        int source = ...;
        int dest = ...;
        dvar int+ x[arcs];

        minimize sum(a in arcs) a.cost * x[a];

        subject to {
            forall(i in Nodes) (
                sum(a in arcs: a.from == i) x[a] - sum(a in arcs: a.to == i) x[a] == ((i == source) ? 1 : ((i == dest) ? -1 : 0))
            );
        }
        """
        data_code = """
        N = 5;
        source = 1;
        dest = 5;
        arcs = {
        <1, 2, 2.0>,
        <1, 3, 3.0>,
        <2, 3, 1.0>,
        <2, 4, 1.0>,
        <3, 4, 1.0>,
        <4, 5, 2.0>,
        <3, 5, 5.0>
        };
        """
        docplex_model_code = model_code.replace("int to;", "int destination;").replace("a.to", "a.destination")

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_mod_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_shortest_path_with_tuples_and_strings(self):
        """Compare the string tuple-arc shortest-path model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_shortest_path_with_tuples_and_strings)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("string to;", "string destination;").replace("a.to", "a.destination")

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_mod_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_logistics_with_tuples_and_strings(self):
        """Compare the string-indexed logistics model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_logistics_with_tuples_and_strings)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code
        for declaration in (
            "float cost[Factories][Warehouses];",
            "int supply[Factories];",
            "int demand[Warehouses];",
        ):
            docplex_model_code = docplex_model_code.replace(declaration, declaration[:-1] + " = ...;")

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_mod_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_graph_coloring_tuples(self):
        """Compare the tuple-indexed graph coloring model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        tuple Edge { int source; int dest; }
        {Edge} Edges = ...;
        dvar int+ color[Nodes];
        dvar int+ maxColor;
        dvar boolean z[Edges];
        minimize maxColor;
        subject to {
            forall(i in Nodes) color[i] >= 1;
            forall(i in Nodes) color[i] <= nbNodes;
            forall(e in Edges)
                color[e.source] >= color[e.dest] + 1 - nbNodes * z[e];
            forall(e in Edges)
                color[e.dest] >= color[e.source] + 1 - nbNodes * (1 - z[e]);
            forall(i in Nodes) maxColor >= color[i];
        }
        """
        data_code = """
        nbNodes = 4;
        Edges = { <1,2>, <2,3>, <3,4>, <4,1> };
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_graph_coloring_matrix(self):
        """Compare the adjacency-matrix graph coloring model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        int adj[Nodes][Nodes] = ...;
        dvar int+ color[Nodes];
        dvar int+ maxColor;
        dvar boolean z[Nodes][Nodes];
        minimize maxColor;
        subject to {
            forall(i in Nodes) color[i] >= 1;
            forall(i in Nodes) color[i] <= nbNodes;
            forall(i in Nodes, j in Nodes : adj[i][j] == 1)
                color[i] >= color[j] + 1 - nbNodes * z[i][j];
            forall(i in Nodes, j in Nodes : adj[i][j] == 1)
                color[j] >= color[i] + 1 - nbNodes * (1-z[i][j]);
            forall(i in Nodes) maxColor >= color[i];
        }
        """
        data_code = """
        nbNodes = 4;
        adj = [[0,1,0,1], [1,0,1,0], [0,1,0,1], [1,0,1,0]];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(
                        docplex_model,
                        {
                            name: value
                            for name, value in result["solution"].items()
                            if not name.startswith("z")
                            or tuple(int(index) for index in re.findall(r"\d+", name))
                            in {(1, 2), (1, 4), (2, 1), (2, 3), (3, 2), (3, 4), (4, 1), (4, 3)}
                        },
                    )
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_vehicle_routing_matrix_dat(self):
        """Compare the matrix-based vehicle-routing model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        float cost[Nodes][Nodes] = ...;
        dvar boolean x[Nodes][Nodes];
        minimize sum(i in Nodes, j in Nodes) cost[i][j] * x[i][j];
        subject to {
            forall(i in Nodes) sum(j in Nodes) x[i][j] == 1;
            forall(j in Nodes) sum(i in Nodes) x[i][j] == 1;
        }
        """
        data_code = """
        nbNodes = 3;
        cost = [
            [1000, 10.0, 1000],
            [1000, 1000, 12.5],
            [8.0, 1000, 1000]
        ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    fixed_model = docplex_model.clone()
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_asset_location(self):
        """Compare the tuple-generated asset-location model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_asset_location)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]

        docplex_model_code = model_code.replace("param float", "float")
        docplex_model_code = docplex_model_code.replace("param int", "int")
        docplex_model_code = docplex_model_code.replace("param boolean", "int")
        docplex_model_code = docplex_model_code.replace("dexpr float", "float")
        docplex_model_code = docplex_model_code.replace("minimize obj:", "minimize")
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_model_code = docplex_model_code.replace("param int NR = ...;", "int NR = ...;")
        docplex_model_code = docplex_model_code.replace("param int NC = ...;", "int NC = ...;")
        docplex_model_code = docplex_model_code.replace("param int NT = ...;", "int NT = ...;")
        docplex_model_code = docplex_model_code.replace(
            "float b[v in V] = (((v.i == Ai && v.j == Aj) ? 1 : ((v.i == Bi && v.j == Bj) ? -1 : 0)));",
            "float b[v in V] = ((v.i == Ai && v.j == Aj) ? 1 : ((v.i == Bi && v.j == Bj) ? -1 : 0));",
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_vrp(self):
        """Compare the string-indexed MTZ vehicle-routing model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_vrp)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float").replace("param int", "int")
        docplex_model_code = docplex_model_code.replace("param boolean", "int")
        docplex_model_code = docplex_model_code.replace("boolean is_depot;", "int is_depot;")
        docplex_model_code = docplex_model_code.replace("string to;", "string destination;")
        docplex_model_code = docplex_model_code.replace("a.to", "a.destination")
        docplex_model_code = docplex_model_code.replace("{Node} NODES;", "{Node} NODES = ...;")
        docplex_model_code = docplex_model_code.replace("{Arc}  ARCS;", "{Arc}  ARCS = ...;")
        docplex_model_code = docplex_model_code.replace("{string} NODE_NAMES;", "{string} NODE_NAMES = ...;")
        docplex_model_code = docplex_model_code.replace("{string} DEPOT_NAMES;", "{string} DEPOT_NAMES = ...;")
        docplex_model_code = docplex_model_code.replace("float demand[NODE_NAMES];", "float demand[NODE_NAMES] = ...;")
        docplex_model_code = docplex_model_code.replace("int is_depot[NODE_NAMES];", "int is_depot[NODE_NAMES] = ...;")
        docplex_model_code = docplex_model_code.replace("float distance[ARCS];", "float distance[ARCS] = ...;")
        docplex_model_code = docplex_model_code.replace("int NUM_VEHICLES;", "int NUM_VEHICLES = ...;")
        docplex_model_code = docplex_model_code.replace("float VEHICLE_CAPACITY;", "float VEHICLE_CAPACITY = ...;")
        docplex_model_code = docplex_model_code.replace("minimize total_distance:", "minimize")
        docplex_model_code = re.sub(
            r"forall\(i in NODE_NAMES : !is_depot\[i\]\)\s+"
            r"forall\(j in NODE_NAMES : \(!is_depot\[i\] && !is_depot\[j\] && i != j\)\) \{\s*"
            r"// There may be multiple arcs between i and j \(in ARCS\). Apply MTZ to all arcs from i to j\.\s*"
            r"forall\(a in ARCS : a\.from == i && a\.destination == j\) \{\s*"
            r"u\[i\] - u\[j\] \+ VEHICLE_CAPACITY \* x\[a\] <= VEHICLE_CAPACITY - demand\[j\];\s*"
            r"\}\s*\}",
            "forall(a in ARCS : is_depot[a.from] == 0 && is_depot[a.destination] == 0 "
            "&& a.from != a.destination) "
            "u[a.from] - u[a.destination] + VEHICLE_CAPACITY * x[a] "
            "<= VEHICLE_CAPACITY - demand[a.destination];",
            docplex_model_code,
        )
        for expression in ("n", "i", "j", "a.from", "a.destination"):
            docplex_model_code = docplex_model_code.replace(f"!is_depot[{expression}]", f"is_depot[{expression}] == 0")
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_data_code = data_code.replace("true", "1").replace("false", "0")
        docplex_data_code = re.sub(
            r"demand\s*=\s*\[.*?\];",
            "demand = [0.0, 2.0, 1.5, 1.0];",
            docplex_data_code,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"is_depot\s*=\s*\[.*?\];",
            "is_depot = [1, 0, 0, 0];",
            docplex_data_code,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"distance\s*=\s*\[.*?\];",
            "distance = [4.0, 6.0, 8.0, 4.0, 6.0, 8.0, 5.0, 7.0, 5.0, 4.0, 7.0, 4.0];",
            docplex_data_code,
            flags=re.DOTALL,
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file:
            docplex_dat_file.write(docplex_data_code)
            docplex_data_path = docplex_dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(
                        docplex_model,
                        {
                            name: value
                            for name, value in result["solution"].items()
                            if not name.startswith("u[Depot]") and not name.startswith("u_Depot")
                        },
                    )
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_vrp_2(self):
        """Compare the numeric-indexed MTZ vehicle-routing model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_vrp_2)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("int dist[Nodes][Nodes];", "int dist[Nodes][Nodes] = ...;")
        docplex_model_code = docplex_model_code.replace("int demand[1..n];", "int demand[1..n] = ...;")
        docplex_model_code = docplex_model_code.replace("int n = ...;", "int n = ...;")
        docplex_model_code = docplex_model_code.replace("int m = ...;", "int m = ...;")
        docplex_model_code = docplex_model_code.replace("int Q = ...;", "int Q = ...;")
        docplex_model_code = docplex_model_code.replace("minimize\n            sum", "minimize sum")

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_stochastic_lot_sizing(self):
        """Compare the stochastic lot-sizing model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_stochastic_lot_sizing)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float").replace("param int", "int")
        docplex_model_code = re.sub(
            r"\{string\} Scenarios = \.\.\.;",
            "{string} Scenarios = ...;",
            docplex_model_code,
        )
        docplex_model_code = docplex_model_code.replace("float p[Scenarios];", "float p[Scenarios] = ...;")
        docplex_model_code = docplex_model_code.replace(
            "float demand[Scenarios][Periods];", "float demand[Scenarios][Periods] = ...;"
        )
        for declaration in (
            "init_inventory",
            "init_backlog",
            "order_cost",
            "holding_cost",
            "backlog_cost",
            "order_cap",
            "terminal_value",
        ):
            docplex_model_code = docplex_model_code.replace(f"float {declaration};", f"float {declaration} = ...;")
        docplex_model_code = docplex_model_code.replace("minimize expected_total_cost:", "minimize")
        docplex_model_code = re.sub(
            r"forall\(t in Periods, s1 in Scenarios, s2 in Scenarios :.*?\)\s*" r"x\[s1\]\[t\] == x\[s2\]\[t\];",
            'x["S1"][1] == x["S2"][1];\n'
            'x["S1"][1] == x["S3"][1];\n'
            'x["S1"][1] == x["S4"][1];\n'
            'x["S1"][2] == x["S2"][2];\n'
            'x["S1"][2] == x["S3"][2];\n'
            'x["S1"][3] == x["S2"][3];',
            docplex_model_code,
            flags=re.DOTALL,
        )
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_data_code = """
        T = 3;
        Scenarios = { "S1", "S2", "S3", "S4" };
        p = [0.25, 0.25, 0.25, 0.25];
        demand = [[80, 70, 60], [80, 70, 140], [80, 110, 60], [130, 110, 140]];
        init_inventory = 15;
        init_backlog = 0;
        order_cost = 5.0;
        holding_cost = 1.0;
        backlog_cost = 9.0;
        order_cap = 120.0;
        terminal_value = 0.0;
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file,
        ):
            docplex_mod_file.write(docplex_model_code)
            docplex_dat_file.write(docplex_data_code)
            docplex_model_path = docplex_mod_file.name
            docplex_data_path = docplex_dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_newsvendor(self):
        """Compare the tuple-scenario newsvendor model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_newsvendor)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float")
        docplex_model_code = docplex_model_code.replace("#", "//")
        docplex_model_code = docplex_model_code.replace("float revenue_per_unit;", "float revenue_per_unit = ...;")
        docplex_model_code = docplex_model_code.replace("float cost_per_unit;", "float cost_per_unit = ...;")
        docplex_model_code = docplex_model_code.replace("float salvage_value;", "float salvage_value = ...;")
        docplex_model_code = docplex_model_code.replace("maximize expected_profit:", "maximize")
        docplex_data_code = """
        revenue_per_unit = 10;
        cost_per_unit = 6;
        salvage_value = 2;
        Scenarios = { <400,0.1>, <600,0.2>, <700,0.4>, <800,0.2>, <1000,0.1> };
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file:
            docplex_dat_file.write(docplex_data_code)
            docplex_data_path = docplex_dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_static_stochastic_knapsack(self):
        """Compare the chance-constrained stochastic knapsack model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_static_stochastic_knapsack)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float")
        docplex_model_code = docplex_model_code.replace("dexpr float", "float")
        docplex_model_code = docplex_model_code.replace("float C = ...;", "float C = ...;")
        docplex_model_code = docplex_model_code.replace("minimize expected_value:", "minimize")
        docplex_model_code = docplex_model_code.replace("maximize expected_value:", "maximize")
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_data_code = """
        Scenarios = { "S1", "S2", "S3", "S4" };
        C = 9.0;
        epsilon = 0.25;
        p = [0.25, 0.25, 0.25, 0.25];
        w = [[2.0, 5.0, 1.0, 1.0, 1.0], [2.2, 5.6, 1.2, 1.1, 1.0],
             [1.8, 5.1, 1.0, 1.2, 1.0], [2.5, 6.0, 1.5, 1.0, 1.2]];
        v = [[9.0, 13.0, 5.0, 2.0, 1.5], [8.5, 12.5, 5.0, 2.0, 1.5],
             [9.0, 12.0, 4.5, 2.5, 1.2], [8.0, 13.5, 5.0, 1.8, 1.0]];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file,
        ):
            docplex_mod_file.write(docplex_model_code)
            docplex_dat_file.write(docplex_data_code)
            docplex_model_path = docplex_mod_file.name
            docplex_data_path = docplex_dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_p_dispersion(self):
        """Compare the implication-based p-dispersion model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_p_dispersion)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = """
        int N = ...;
        range Sites = 1..N;
        int p = ...;
        int dist[Sites][Sites] = ...;
        dvar boolean y[Sites];
        dvar float+ z;
        float maxD = ...;
        maximize z;
        subject to {
            SelectSites: sum(i in Sites) y[i] == p;
            BoundZ: z <= maxD;
            forall(i in Sites, j in Sites : i < j)
                PairDistance: (y[i] + y[j] >= 2) => (z <= dist[i][j]);
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        try:
            result = solve(model_path, data_path, solver="gurobi")
            self.assertEqual(result["status"], "OPTIMAL")
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)

            fixed_model = docplex_model.clone()
            mapped_solution = map_solution_variables(docplex_model, result["solution"])
            for variable, value in mapped_solution.items():
                fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
            candidate = fixed_model.solve(log_output=False)
            self.assertIsNotNone(candidate)
            self.assertTrue(candidate.is_valid_solution())
            self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_complex_workforce_planning_3(self):
        """Compare the training and production workforce model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_complex_workforce_planning_3)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float").replace("param int", "int")
        docplex_model_code = docplex_model_code.replace("minimize total_cost:", "minimize")
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_complex_workforce_planning_2(self):
        """Compare the second workforce planning model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_complex_workforce_planning_2)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float").replace("param int", "int")
        docplex_model_code = docplex_model_code.replace("float D1[T] = ...;", "float D1[T] = ...;")
        docplex_model_code = docplex_model_code.replace("float D2[T] = ...;", "float D2[T] = ...;")
        docplex_model_code = docplex_model_code.replace("int S0 = ...;", "int S0 = ...;")
        docplex_model_code = docplex_model_code.replace("minimize totalCost:", "minimize")
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)

    def test_complex_workforce_planning_1(self):
        """Compare the first workforce planning model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_complex_workforce_planning_1)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = re.sub(r"float\s+rate\[Food\]\s*=\s*\[10, 6\];", "float rate[Food] = ...;", model_code)
        docplex_model_code = re.sub(r"float\s+pen\[Food\]\s*=\s*\[0\.5, 0\.6\];", "float pen[Food] = ...;", docplex_model_code)
        docplex_model_code = re.sub(r"int\s+demand\[Food\]\[Weeks\];", "int demand[Food][Weeks] = ...;", docplex_model_code)
        docplex_model_code = docplex_model_code.replace("prod[Food][Weeks]", "production[Food][Weeks]")
        docplex_model_code = docplex_model_code.replace("prod[f][t]", "production[f][t]")
        docplex_model_code = docplex_model_code.replace("prod[f][t] / rate[f]", "production[f][t] / rate[f]")
        docplex_model_code = re.sub(
            r"sum\(f in Food, w in Weeks, t in w\+1\.\.8\)",
            "sum(f in Food, w in Weeks, t in Weeks : t > w)",
            docplex_model_code,
        )
        docplex_model_code = re.sub(r"sum\(t in w\.\.8\)", "sum(t in Weeks : t >= w)", docplex_model_code)
        docplex_model_code = re.sub(r"sum\(w in 1\.\.t\)", "sum(w in Weeks : w <= t)", docplex_model_code)
        docplex_model_code = docplex_model_code.replace(
            "minimize\n            // Labor costs", "minimize\n            // Labor costs"
        )
        docplex_model_code = re.sub(
            r"forall\(([^\n]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_data_code = """
        rate = [10, 6];
        pen = [0.5, 0.6];
        demand = [[10000, 10000, 12000, 12000, 16000, 16000, 20000, 20000],
                  [6000, 7200, 8400, 10800, 10800, 12000, 12000, 12000]];
        """

        def studio_name(name):
            production_match = re.match(r"prod(?:_|\[)(I|II)[,_](\d+)\]?", name)
            if production_match:
                return f'production("{production_match.group(1)}")({production_match.group(2)})'
            delivery_match = re.match(r"y(?:_|\[)(I|II)[,_](\d+)[,_](\d+)\]?", name)
            if delivery_match:
                food_index = 0 if delivery_match.group(1) == "I" else 1
                return "y#{}#{}#{}".format(
                    food_index,
                    int(delivery_match.group(2)) - 1,
                    int(delivery_match.group(3)) - 1,
                )
            return name

        def map_studio_solution(solution, docplex_model):
            exported_names = {variable.name for variable in docplex_model.iter_variables()}
            converted = {}
            for name, value in solution.items():
                converted_name = studio_name(name)
                if converted_name.startswith("y#") and converted_name not in exported_names:
                    if abs(float(value)) > 1e-9:
                        raise AssertionError(f"Native OPL omitted nonzero delivery variable {name!r}")
                    continue
                converted[converted_name] = value
            return map_solution_variables(docplex_model, converted)

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as docplex_dat_file,
        ):
            docplex_mod_file.write(docplex_model_code)
            docplex_dat_file.write(docplex_data_code)
            docplex_model_path = docplex_mod_file.name
            docplex_data_path = docplex_dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_studio_solution(result["solution"], docplex_model)
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)
            os.remove(docplex_data_path)

    def test_warehouse_location(self):
        """Compare the fixed-charge warehouse model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbWarehouses = ...;
        int nbCustomers = ...;
        range Warehouses = 1..nbWarehouses;
        range Customers = 1..nbCustomers;
        float fixed_cost[Warehouses] = ...;
        float trans_cost[Warehouses][Customers] = ...;
        float demand[Customers] = ...;
        float capacity[Warehouses] = ...;
        dvar boolean y[Warehouses];
        dvar float+ x[Warehouses][Customers];
        minimize sum(i in Warehouses) fixed_cost[i] * y[i]
            + sum(i in Warehouses, j in Customers) trans_cost[i][j] * x[i][j];
        subject to {
            forall(j in Customers) sum(i in Warehouses) x[i][j] == demand[j];
            forall(i in Warehouses, j in Customers) x[i][j] <= capacity[i] * y[i];
        }
        """
        data_code = """
        nbWarehouses = 2;
        nbCustomers = 3;
        fixed_cost = [80, 90];
        trans_cost = [[3, 5, 8], [4, 3, 6]];
        demand = [15, 20, 10];
        capacity = [25, 30];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_portfolio_diversification(self):
        """Compare the scenario-tree portfolio model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} Assets = ...;
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        tuple Arc { int parent; int child; };
        {Arc} Arcs = ...;
        int root = ...;
        boolean isLeaf[Nodes] = ...;
        float prob[Nodes] = ...;
        float ret[Arcs][Assets] = ...;
        float initHold[Assets] = ...;
        float transCost = ...;
        float maxShare = ...;
        dvar float+ h[Nodes][Assets];
        dvar float+ buy[Nodes][Assets];
        dvar float+ sell[Nodes][Assets];
        dvar float+ wealth[Nodes];
        maximize sum(n in Nodes : isLeaf[n]) prob[n] * wealth[n];
        subject to {
            forall(a in Assets) h[root][a] == initHold[a] + buy[root][a] - sell[root][a];
            sum(a in Assets) ((1 + transCost) * buy[root][a])
                == sum(a in Assets) ((1 - transCost) * sell[root][a]);
            forall(a in Assets) sell[root][a] <= initHold[a];
            forall(ar in Arcs, a in Assets)
                h[ar.child][a] == ret[ar][a] * h[ar.parent][a]
                    + buy[ar.child][a] - sell[ar.child][a];
            forall(ar in Arcs)
                sum(a in Assets) ((1 + transCost) * buy[ar.child][a])
                    == sum(a in Assets) ((1 - transCost) * sell[ar.child][a]);
            forall(ar in Arcs, a in Assets)
                sell[ar.child][a] <= ret[ar][a] * h[ar.parent][a];
            forall(n in Nodes, a in Assets : !isLeaf[n])
                h[n][a] <= maxShare * sum(b in Assets) h[n][b];
            forall(ar in Arcs : isLeaf[ar.child])
                wealth[ar.child] == sum(a in Assets) ret[ar][a] * h[ar.parent][a];
            forall(n in Nodes : !isLeaf[n]) wealth[n] == 0;
        }
        """
        data_code = """
        Assets = { "StockA", "StockB", "BondC" };
        nbNodes = 7;
        Arcs = { <1,2>, <1,3>, <2,4>, <2,5>, <3,6>, <3,7> };
        root = 1;
        isLeaf = [ false, false, false, true, true, true, true ];
        prob = [ 0, 0, 0, 0.25, 0.25, 0.25, 0.25 ];
        initHold = [ 40, 35, 25 ];
        transCost = 0.01;
        maxShare = 0.60;
        ret = [
            <1,2> [1.08, 1.03, 1.01],
            <1,3> [0.95, 1.06, 1.02],
            <2,4> [1.10, 1.02, 1.01],
            <2,5> [0.92, 1.04, 1.01],
            <3,6> [1.07, 0.98, 1.01],
            <3,7> [0.90, 1.08, 1.02]
        ];
        """
        docplex_model_code = (
            model_code.replace("boolean isLeaf", "int isLeaf")
            .replace(
                "maximize sum(n in Nodes : isLeaf[n]) prob[n] * wealth[n];",
                "maximize sum(n in Nodes : isLeaf[n] == 1) prob[n] * wealth[n];",
            )
            .replace(": !isLeaf[n]", ": isLeaf[n] == 0")
            .replace(": isLeaf[ar.child]", ": isLeaf[ar.child] == 1")
        )
        docplex_data_code = data_code.replace(
            "isLeaf = [ false, false, false, true, true, true, true ];",
            "isLeaf = [ 0, 0, 0, 1, 1, 1, 1 ];",
        )
        docplex_data_code = docplex_data_code.replace(
            """ret = [
            <1,2> [1.08, 1.03, 1.01],
            <1,3> [0.95, 1.06, 1.02],
            <2,4> [1.10, 1.02, 1.01],
            <2,5> [0.92, 1.04, 1.01],
            <3,6> [1.07, 0.98, 1.01],
            <3,7> [0.90, 1.08, 1.02]
        ];""",
            """ret = [
            [1.08, 1.03, 1.01],
            [0.95, 1.06, 1.02],
            [1.10, 1.02, 1.01],
            [0.92, 1.04, 1.01],
            [1.07, 0.98, 1.01],
            [0.90, 1.08, 1.02]
        ];""",
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model_path = model_path + ".docplex.mod"
            docplex_data_path = data_path + ".docplex.dat"
            with open(docplex_model_path, "w") as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
            with open(docplex_data_path, "w") as docplex_dat_file:
                docplex_dat_file.write(docplex_data_code)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            os.remove(model_path)
            os.remove(data_path)
            for path in (model_path + ".docplex.mod", data_path + ".docplex.dat"):
                if os.path.exists(path):
                    os.remove(path)

    def test_RMAB_relaxation_tuples(self):
        """Compare the tuple-indexed RMAB occupation-measure LP with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} Arms = ...;
        {int} States = ...;
        {string} Actions = ...;
        float cost[Arms][States][Actions] = ...;
        float P[Arms][States][Actions][States] = ...;
        float Budget = ...;
        dvar float+ x[Arms][States][Actions];
        minimize sum(k in Arms, i in States, a in Actions) cost[k][i][a] * x[k][i][a];
        subject to {
            forall(k in Arms) sum(i in States, a in Actions) x[k][i][a] == 1;
            forall(k in Arms, j in States)
                sum(a in Actions) x[k][j][a]
                    == sum(i in States, a in Actions) x[k][i][a] * P[k][i][a][j];
            sum(k in Arms, i in States) x[k][i]["Active"] <= Budget;
            forall(k in Arms, i in States, a in Actions)
                sum(j in States) P[k][i][a][j] == 1;
        }
        """
        data_code = """
        Arms = { "A1", "A2" };
        States = { 1, 2 };
        Actions = { "Passive", "Active" };
        Budget = 1;
        cost = [
            <"A1",1,"Passive"> 0.0,
            <"A1",1,"Active"> 0.5,
            <"A1",2,"Passive"> 3.0,
            <"A1",2,"Active"> 3.5,
            <"A2",1,"Passive"> 0.0,
            <"A2",1,"Active"> 0.5,
            <"A2",2,"Passive"> 4.0,
            <"A2",2,"Active"> 4.5
        ];
        P = [
            <"A1",1,"Passive",1> 0.80,
            <"A1",1,"Passive",2> 0.20,
            <"A1",1,"Active",1> 0.95,
            <"A1",1,"Active",2> 0.05,
            <"A1",2,"Passive",1> 0.20,
            <"A1",2,"Passive",2> 0.80,
            <"A1",2,"Active",1> 0.60,
            <"A1",2,"Active",2> 0.40,
            <"A2",1,"Passive",1> 0.70,
            <"A2",1,"Passive",2> 0.30,
            <"A2",1,"Active",1> 0.90,
            <"A2",1,"Active",2> 0.10,
            <"A2",2,"Passive",1> 0.10,
            <"A2",2,"Passive",2> 0.90,
            <"A2",2,"Active",1> 0.50,
            <"A2",2,"Active",2> 0.50
        ];
        """
        docplex_model_code = model_code.replace("minimize sum", "minimize\n            sum")
        docplex_data_code = """
        Arms = { "A1", "A2" };
        States = { 1, 2 };
        Actions = { "Passive", "Active" };
        Budget = 1;
        cost = [
            [[0.0, 0.5], [3.0, 3.5]],
            [[0.0, 0.5], [4.0, 4.5]]
        ];
        P = [
            [[[0.80, 0.20], [0.95, 0.05]], [[0.20, 0.80], [0.60, 0.40]]],
            [[[0.70, 0.30], [0.90, 0.10]], [[0.10, 0.90], [0.50, 0.50]]]
        ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        docplex_model_path = model_path + ".docplex.mod"
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_model_path, "w") as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
            with open(docplex_data_path, "w") as docplex_dat_file:
                docplex_dat_file.write(docplex_data_code)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path, docplex_model_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_RMAB_relaxation_dense(self):
        """Compare the dense RMAB occupation-measure LP with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} Arms = ...;
        {int} States = ...;
        {string} Actions = ...;
        float cost[Arms][States][Actions] = ...;
        float P[Arms][States][Actions][States] = ...;
        float Budget = ...;
        dvar float+ x[Arms][States][Actions];
        minimize sum(k in Arms, i in States, a in Actions) cost[k][i][a] * x[k][i][a];
        subject to {
            forall(k in Arms) sum(i in States, a in Actions) x[k][i][a] == 1;
            forall(k in Arms, j in States)
                sum(a in Actions) x[k][j][a]
                    == sum(i in States, a in Actions) x[k][i][a] * P[k][i][a][j];
            sum(k in Arms, i in States) x[k][i]["Active"] <= Budget;
            forall(k in Arms, i in States, a in Actions)
                sum(j in States) P[k][i][a][j] == 1;
        }
        """
        data_code = """
        Arms = { "A1", "A2" };
        States = { 1, 2 };
        Actions = { "Passive", "Active" };
        Budget = 1;
        cost = [
            [[0.0, 0.5], [3.0, 3.5]],
            [[0.0, 0.5], [4.0, 4.5]]
        ];
        P = [
            [[[0.80, 0.20], [0.95, 0.05]], [[0.20, 0.80], [0.60, 0.40]]],
            [[[0.70, 0.30], [0.90, 0.10]], [[0.10, 0.90], [0.50, 0.50]]]
        ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_TOPSIS(self):
        """Compare the ground-computed TOPSIS selection model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int NA = ...;
        int NC = ...;
        range Alternatives = 1..NA;
        range Criteria = 1..NC;
        string AltName[Alternatives] = ...;
        string CritName[Criteria] = ...;
        float X[Alternatives][Criteria] = ...;
        float w[Criteria] = ...;
        boolean is_benefit[Criteria] = ...;
        float denom[c in Criteria] = sqrt(sum(i in Alternatives) X[i][c] * X[i][c]);
        float r[i in Alternatives][c in Criteria] = X[i][c] / denom[c];
        float v[i in Alternatives][c in Criteria] = r[i][c] * w[c];
        float v_max[c in Criteria] = max(i in Alternatives) v[i][c];
        float v_min[c in Criteria] = min(i in Alternatives) v[i][c];
        float v_plus[c in Criteria] = (is_benefit[c]) ? v_max[c] : v_min[c];
        float v_minus[c in Criteria] = (is_benefit[c]) ? v_min[c] : v_max[c];
        float Splus[i in Alternatives] = sum(c in Criteria)
            (v[i][c] - v_plus[c]) * (v[i][c] - v_plus[c]);
        float Sminus[i in Alternatives] = sum(c in Criteria)
            (v[i][c] - v_minus[c]) * (v[i][c] - v_minus[c]);
        float dplus[i in Alternatives] = sqrt(Splus[i]);
        float dminus[i in Alternatives] = sqrt(Sminus[i]);
        float Ci[i in Alternatives] = dminus[i] / (dplus[i] + dminus[i]);
        dvar boolean y[Alternatives];
        maximize sum(i in Alternatives) Ci[i] * y[i];
        subject to { sum(i in Alternatives) y[i] == 1; }
        """
        data_code = """
        NA = 3;
        NC = 2;
        AltName = ["Phone A", "Phone B", "Phone C"];
        CritName = ["Price", "Camera"];
        X = [[800, 7], [600, 4], [1200, 10]];
        w = [0.4, 0.6];
        is_benefit = [false, true];
        """
        docplex_model_code = (
            model_code.replace("boolean is_benefit", "int is_benefit")
            .replace(
                "(is_benefit[c]) ? v_max[c] : v_min[c]",
                "is_benefit[c] == 1 ? v_max[c] : v_min[c]",
            )
            .replace(
                "(is_benefit[c]) ? v_min[c] : v_max[c]",
                "is_benefit[c] == 1 ? v_min[c] : v_max[c]",
            )
        )
        docplex_data_code = data_code.replace("is_benefit = [false, true];", "is_benefit = [0, 1];")

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        docplex_model_path = model_path + ".docplex.mod"
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_model_path, "w") as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
            with open(docplex_data_path, "w") as docplex_dat_file:
                docplex_dat_file.write(docplex_data_code)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path, docplex_model_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_column_generation(self):
        """Compare both gated column-generation submodels with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} I = ...;
        {string} P = ...;
        int L = ...;
        int itemLen[I] = ...;
        int demand[I] = ...;
        int a[P][I] = ...;
        float dual[I] = ...;
        int RunPricing = ...;
        dvar int+ x[P];
        dvar int+ y[I];
        dexpr float PricingValue = sum(i in I) dual[i] * y[i];
        dexpr float ReducedCost = 1 - PricingValue;
        minimize (RunPricing == 1) * ReducedCost
            + (RunPricing != 1) * sum(p in P) x[p];
        subject to {
            forall(i in I)
                (RunPricing != 0) || (sum(p in P) a[p][i] * x[p] >= demand[i]);
            forall(p in P)
                (RunPricing != 0) || (sum(i in I) itemLen[i] * a[p][i] <= L);
            (RunPricing != 1) || (sum(i in I) itemLen[i] * y[i] <= L);
            (RunPricing != 1) || (PricingValue >= 1.000001);
        }
        """
        data_template = """
        I = {{ "A", "B", "C" }};
        L = 10;
        itemLen = [ "A" 2, "B" 3, "C" 5 ];
        demand = [ "A" 4, "B" 3, "C" 2 ];
        P = {{ "p1", "p2", "p3", "p4" }};
        a = [
            "p1" [ 5, 0, 0 ],
            "p2" [ 0, 3, 0 ],
            "p3" [ 0, 0, 2 ],
            "p4" [ 2, 2, 0 ]
        ];
        dual = [ "A" 0.10, "B" 0.35, "C" 0.55 ];
        RunPricing = {run_pricing};
        """
        docplex_model_code = model_code.replace("int itemLen", "int itemLen").replace(
            "minimize (RunPricing == 1) * ReducedCost",
            "minimize\n            (RunPricing == 1) * ReducedCost",
        )
        docplex_data_template = """
        I = {{ "A", "B", "C" }};
        L = 10;
        itemLen = [ 2, 3, 5 ];
        demand = [ 4, 3, 2 ];
        P = {{ "p1", "p2", "p3", "p4" }};
        a = [ [5, 0, 0], [0, 3, 0], [0, 0, 2], [2, 2, 0] ];
        dual = [ 0.10, 0.35, 0.55 ];
        RunPricing = {run_pricing};
        """

        for run_pricing in (0, 1):
            data_code = data_template.format(run_pricing=run_pricing)
            docplex_data_code = docplex_data_template.format(run_pricing=run_pricing)
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            ):
                mod_file.write(model_code)
                dat_file.write(data_code)
                model_path = mod_file.name
                data_path = dat_file.name
            docplex_model_path = model_path + ".docplex.mod"
            docplex_data_path = data_path + ".docplex.dat"
            try:
                results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
                with open(docplex_model_path, "w") as docplex_mod_file:
                    docplex_mod_file.write(docplex_model_code)
                with open(docplex_data_path, "w") as docplex_dat_file:
                    docplex_dat_file.write(docplex_data_code)
                docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
                docplex_solution = docplex_model.solve(log_output=False)
                self.assertIsNotNone(docplex_solution)
                docplex_objective = docplex_solution.objective_value
                for solver, result in results.items():
                    with self.subTest(run_pricing=run_pricing, solver=solver):
                        self.assertEqual(result["status"], "OPTIMAL")
                        self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                        fixed_model = docplex_model.clone()
                        for variable, value in map_solution_variables(docplex_model, result["solution"]).items():
                            fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                        candidate = fixed_model.solve(log_output=False)
                        self.assertIsNotNone(candidate)
                        self.assertTrue(candidate.is_valid_solution())
                        self.assertAlmostEqual(
                            candidate.get_value(fixed_model.objective_expr),
                            docplex_objective,
                            places=6,
                        )
            finally:
                for path in (model_path, data_path, docplex_model_path, docplex_data_path):
                    if os.path.exists(path):
                        os.remove(path)

    def test_v2_features(self):
        """Compare PyOPL objectives and assignments with DOcplex OPL."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} Products = ...;
        range Periods = 1..3;
        tuple Site { string name; int capacity; }
        tuple Lane { Site source; Site destination; float miles; }
        {Site} Sites = ...;
        {Lane} Lanes = ...;
        float demand[Products][Periods] = ...;
        float productionCost[Products][Periods] = ...;
        float transportCost[Lanes] = ...;
        {int} PeakPeriods = { t | t in Periods : t >= 2 };
        {Site} LargeSites = ...;
        dvar boolean open[Products];
        dvar float+ production[Products][Periods];
        dvar float+ shipped[Lanes];
        dexpr float productCost[p in Products] = sum(t in Periods) productionCost[p][t] * production[p][t];
        minimize sum(p in Products) productCost[p] + sum(l in Lanes) transportCost[l] * shipped[l];
        subject to {
            forall(t in PeakPeriods) sum(p in Products) production[p][t] <= sum(s in LargeSites) s.capacity;
            forall(p in Products, t in Periods) production[p][t] >= demand[p][t] * open[p];
            (open["tea"] == 1) => (open["coffee"] == 1);
            forall(l in Lanes) shipped[l] <= l.miles;
            (open["tea"] == 1 && open["coffee"] == 1) => sum(l in Lanes) shipped[l] >= 10;
        }
        """
        data_code = """
        Products = { "tea", "coffee" };
        LargeSites = { <"north", 100> };
        Sites = { <"north", 100>, <"south", 60> };
        Lanes = { <<"north", 100>, <"south", 60>, 12.5>, <<"south", 60>, <"north", 100>, 8.0> };
        demand = [ [20, 18, 16], [15, 17, 19] ];
        productionCost = [ [4.0, 4.2, 4.1], [3.5, 3.8, 3.7] ];
        transportCost = [
            <<"north", 100>, <"south", 60>, 12.5> 0.8,
            <<"south", 60>, <"north", 100>, 8.0> 1.1
        ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_data_path = data_path + ".docplex.dat"
            with open(data_path) as source_data, open(docplex_data_path, "w") as target_data:
                target_data.write(
                    source_data.read().replace(
                        'transportCost = [\n            <<"north", 100>, <"south", 60>, 12.5> 0.8,\n            <<"south", 60>, <"north", 100>, 8.0> 1.1\n        ];',
                        "transportCost = [ 0.8, 1.1 ];",
                    )
                )
            docplex_model = build_docplex_model(model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                self.assertEqual(result["status"], "OPTIMAL")
                self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                self.assertTrue(candidate.is_valid_solution())
                self.assertAlmostEqual(
                    candidate.get_value(docplex_model.objective_expr),
                    docplex_objective,
                    places=6,
                )
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_iterator_scoping_sum_and_forall(self):
        """Compare iterator-scoping objectives and assignments with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int a = 2;
        int b = 2;
        int c = 2;
        range I = 1..a;
        range J = 1..b;
        range K = 1..c;

        tuple T { int i; int j; int k; }
        {T} Triples =
          { <i,j,k> | i in I, j in J, k in K };

        dvar boolean x[I][J][K];

        maximize sum(c in Triples) x[c.i][c.j][c.k];

        subject to {
          forall(i in I, j in J)
            sum(k in K) x[i][j][k] >= 0;
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_employee_rostering(self):
        """Compare the employee rostering model with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple Pair { string e; string s; }

        {string} Employees = ...;
        {string} Shifts = ...;

        {Pair} Pairs = { <e,s> | e in Employees, s in Shifts };

        int demand[Shifts];
        int pref[Pairs];

        dvar boolean x[Employees][Shifts];

        maximize sum(e in Employees, s in Shifts) pref[<e,s>] * x[e][s];

        subject to {
          forall (e in Employees)
            sum (s in Shifts) x[e][s] == 1;

          forall (s in Shifts)
            sum (e in Employees) x[e][s] == demand[s];
        }
        """
        data_code = """
        Employees = { "Alex", "Bri", "Casey", "Drew", "Evan" };
        Shifts = { "Morning", "Midday", "Evening" };

        demand = [ "Morning" 2, "Midday" 2, "Evening" 1 ];

        pref = [
          <"Alex","Morning"> 3, <"Alex","Midday"> 1, <"Alex","Evening"> 0,
          <"Bri","Morning"> 2, <"Bri","Midday"> 3, <"Bri","Evening"> 1,
          <"Casey","Morning"> 1, <"Casey","Midday"> 2, <"Casey","Evening"> 3,
          <"Drew","Morning"> 0, <"Drew","Midday"> 3, <"Drew","Evening"> 2,
          <"Evan","Morning"> 3, <"Evan","Midday"> 0, <"Evan","Evening"> 2
        ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        docplex_model_path = model_path + ".docplex.mod"
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model_code = (
                model_code.replace("int demand[Shifts];", "int demand[Shifts] = ...;")
                .replace("int pref[Pairs];", "int pref[Employees][Shifts] = ...;")
                .replace("pref[<e,s>] * x[e][s]", "pref[e][s] * x[e][s]")
            )
            with open(docplex_model_path, "w") as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
            with open(docplex_data_path, "w") as docplex_dat_file:
                docplex_dat_file.write("""
                    Employees = { "Alex", "Bri", "Casey", "Drew", "Evan" };
                    Shifts = { "Morning", "Midday", "Evening" };
                    demand = [ 2, 2, 1 ];
                    pref = [
                      [ 3, 1, 0 ],
                      [ 2, 3, 1 ],
                      [ 1, 2, 3 ],
                      [ 0, 3, 2 ],
                      [ 3, 0, 2 ]
                    ];
                    """)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path, docplex_model_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_minl_maxl_in_index_constraint(self):
        """Compare minl/maxl index filtering with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int a = 3;
        int b = 3;
        range Rows = 1..a;
        range Cols = 1..b;

        tuple Pair {
          int i;
          int j;
          int i2;
          int j2;
        }

        {Pair} Pairs =
          { <i,j,i2,j2> |
              i in Rows, j in Cols,
              i2 in Rows, j2 in Cols :
              ((i < i2) || (i == i2 && j < j2)) && (i != i2) && (j != j2)
          };

        dvar float+ y[Pairs];

        minimize sum(pr in Pairs) y[pr];

        subject to {
          forall(pr in Pairs, m in Rows, n in Cols :
            (minl(pr.i, pr.i2) < m && m < maxl(pr.i, pr.i2)) &&
            (minl(pr.j, pr.j2) < n && n < maxl(pr.j, pr.j2))
          )
            y[pr] >= 0;
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_tuple_set_comprehension_pairs(self):
        """Compare tuple-set comprehension pairs with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int a = ...;
        int b = 3;
        range Rows = 1..a;
        range Cols = 1..b;

        tuple Pair {
          int i;
          int j;
          int i2;
          int j2;
        }

        {Pair} Pairs =
          { <i,j,i2,j2> |
              i in Rows, j in Cols,
              i2 in Rows, j2 in Cols :
              ((i-1)*b + j) < ((i2-1)*b + j2)
          };

        dvar boolean y[Pairs];

        minimize sum(pr in Pairs) y[pr];

        subject to {
          forall(pr in Pairs) y[pr] >= 0;
        }
        """
        data_code = """
        a = 2;
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        docplex_model_path = model_path + ".docplex.mod"
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model_code = model_code
            with open(docplex_model_path, "w") as docplex_mod_file:
                docplex_mod_file.write(docplex_model_code)
            with open(docplex_data_path, "w") as docplex_dat_file:
                docplex_dat_file.write(data_code)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path, docplex_model_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_param_multi_index_rhs_expression_initialization(self):
        """Compare multi-index RHS parameter initialization with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {int} I = ...;
        {int} J = {1, 2};

        param float W[i in I][j in J] = i + j;
        param float X[i in I, j in J] = i + j;

        dvar float z;

        maximize z;
        subject to {
          z == sum(i in I, j in J) W[i][j];
          z == sum(i in I, j in J) X[i][j];
        }
        """
        data_code = """
        I = {1, 2, 3};
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model_path = model_path + ".docplex.mod"
            docplex_data_path = data_path + ".docplex.dat"
            with open(docplex_model_path, "w") as docplex_mod_file:
                docplex_mod_file.write(model_code.replace("param float ", "float "))
            with open(docplex_data_path, "w") as docplex_dat_file:
                docplex_dat_file.write(data_code)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path, docplex_model_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_tuples_index_specifiers(self):
        """Compare tuple index specifiers with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple K { int i; string s; }
        {K} KS = ...;

        dvar float x[KS];

        maximize x[<1, "A">] + x[<2, "B">];

        subject to {
          x[<1, "A">] <= 1;
          x[<2, "B">] <= 1;
          x[<1, "A">] >= 0;
          x[<2, "B">] >= 0;
        }
        """
        data_code = """
        KS = { <1, "A">, <2, "B"> };
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_min_max(self):
        """Compare min/max expressions with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int N = 4;
        float a[1..N] = ...;
        dvar float x[1..N];

        minimize max(i in 1..N) (a[i] * x[i]);
        subject to {
          sum(i in 1..N) x[i] == 1;
          forall(i in 1..N) x[i] >= 0;
          min(i in 1..N) x[i] >= 0.1;
        }
        """
        data_code = """
        a = [2.0, 4.5, 1.0, 3.0];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for name, value in result["solution"].items():
                        if name.startswith("x") and "maxagg" not in name:
                            index = int(re.search(r"\d+", name).group())
                            fixed_model.add_constraint(fixed_model.get_var_by_name(f"x({index})") == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(fixed_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_minl_maxl(self):
        """Compare variadic minl/maxl expressions with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        range N = 1..5;
        float a[N] = ...;
        dvar float x[N];
        dvar float y;

        minimize maxl(y, 0);

        subject to {
          sum(i in N) x[i] == 1;
          forall(i in N) x[i] >= 0;
          forall(i in N) y >= a[i] * x[i];
          minl(x[1], x[2], x[3], x[4], x[5]) >= 0.1;
          maxl(x[2], x[3], x[4], x[5]) <= 0.7;
        }
        """
        data_code = """
        a = [2.3, 4.7, 1.1, 3.5, 5.2];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for name, value in result["solution"].items():
                        if name.startswith("x") or name == "y":
                            index_match = re.search(r"\d+", name)
                            variable_name = f"x({index_match.group()})" if index_match else "y"
                            fixed_model.add_constraint(fixed_model.get_var_by_name(variable_name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(fixed_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_composite_boolean_implication(self):
        """Compare composite boolean implication semantics with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        dvar boolean a;
        dvar boolean b;
        dvar boolean c;
        dvar boolean d;

        minimize a;
        subject to {
          (a == 1) && (b == 1) => (c == 1) || !(d == 1);
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for name, value in result["solution"].items():
                        if name in {"a", "b", "c", "d"}:
                            fixed_model.add_constraint(fixed_model.get_var_by_name(name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(fixed_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_multi_indexed_variable_and_constraint(self):
        """Compare multi-indexed variables and constraints with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        dvar float+ x[1..2][1..3][1..2];
        range I = 1..2;
        range J = 1..3;
        range K = 1..2;
        minimize sum(i in I, j in J, k in K) x[i][j][k];
        subject to {
          forall(i in I, j in J)
            sum(k in K) x[i][j][k] <= 5;
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_tuple_field_access_and_nested_tuple_set(self):
        """Compare nested tuple field access with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        tuple Inner { int id; float val; }
        tuple Outer { Inner inner; float weight; }
        {Outer} outers = { < <1, 2.5>, 3.0 >, < <2, 4.0>, 1.5 > };
        dvar float+ x[outers];
        minimize sum(o in outers) o.inner.val * x[o];
        subject to {
          forall(o in outers) x[o] <= o.weight;
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_inline_and_external_data_mix(self):
        """Compare inline and external data handling with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int N = ...;
        range I = 1..N;
        float cost[I] = ...;
        dvar float x[I];
        minimize sum(i in 1..N) cost[i] * x[i];
        subject to {
          forall(i in I) x[i] >= 0;
          sum(i in I) x[i] == 10;
        }
        """
        data_code = """
        N = 3;
        cost = [2.0, 3.0, 1.5];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_filtered_sum_and_nested_forall(self):
        """Compare filtered sums and nested forall constraints with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        range I = 1..3;
        range J = 1..3;
        dvar boolean x[I][J];
        minimize sum(i in I, j in J) x[i][j];
        subject to {
          forall(i in I)
            sum(j in J : j != i) x[i][j] == 1;
          forall(j in J)
            sum(i in I : i != j) x[i][j] == 1;
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_simple_blending_problem(self):
        """Compare the simple blending LP with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        dvar float blendA;
        dvar float blendB;

        minimize 2.5 * blendA + 3.0 * blendB;

        subject to {
          blendA + blendB == 100;
          0.3 * blendA + 0.6 * blendB >= 45;
          0.1 * blendA + 0.2 * blendB <= 20;
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], 275.0, places=6)
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_blending_string_sets_list_index_error(self):
        """Compare string-indexed list parameters with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} Products = ...;
        {string} Resources = ...;

        float Consumption[Products][Resources] = ...;
        float Capacity[Resources] = ...;
        float Demand[Products] = ...;
        float InsideCost[Products] = ...;
        float OutsideCost[Products] = ...;

        dvar float+ Inside[Products];
        dvar float+ Outside[Products];

        minimize sum(p in Products)
          (InsideCost[p] * Inside[p] + OutsideCost[p] * Outside[p]);

        subject to {
          forall(r in Resources)
            sum(p in Products) Consumption[p][r] * Inside[p] <= Capacity[r];
          forall(p in Products)
            Inside[p] + Outside[p] >= Demand[p];
        }
        """
        data_code = """
        Products = { "ProdA", "ProdB" };
        Resources = { "Res1", "Res2" };
        Consumption = [ [ 1.0, 2.0 ], [ 0.5, 1.5 ] ];
        Capacity = [ 100.0, 80.0 ];
        Demand = [ 40.0, 50.0 ];
        InsideCost = [ 2.0, 3.0 ];
        OutsideCost = [ 5.0, 6.0 ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], 342.5, places=6)
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_workforce_planning_conditional_vs_explicit(self):
        """Compare equivalent explicit and conditional workforce models with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_workforce_planning_conditional_vs_explicit)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {
                    "explicit_model",
                    "conditional_model",
                    "data_code",
                }:
                    literals[target.id] = ast.literal_eval(node.value)

        explicit_model = literals["explicit_model"]
        conditional_model = literals["conditional_model"]
        data_code = literals["data_code"]

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as explicit_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as conditional_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as data_file,
        ):
            explicit_file.write(explicit_model)
            conditional_file.write(conditional_model)
            data_file.write(data_code)
            explicit_path = explicit_file.name
            conditional_path = conditional_file.name
            data_path = data_file.name
        docplex_explicit_path = explicit_path + ".docplex.mod"
        docplex_conditional_path = conditional_path + ".docplex.mod"
        try:
            pyopl_results = {
                (formulation, solver): solve(path, data_path, solver=solver)
                for formulation, path in (("explicit", explicit_path), ("conditional", conditional_path))
                for solver in ("scipy", "gurobi")
            }
            for result in pyopl_results.values():
                self.assertEqual(result["status"], "OPTIMAL")

            for solver in ("scipy", "gurobi"):
                self.assertAlmostEqual(
                    pyopl_results[("explicit", solver)]["objective_value"],
                    pyopl_results[("conditional", solver)]["objective_value"],
                    places=6,
                )

            def studio_model(model_code):
                model_code = re.sub(
                    r"^\s*((?:int|float)\s+\w+(?:\[[^;]+\])?);",
                    lambda match: match.group(0).replace(";", " = ...;"),
                    model_code,
                    flags=re.MULTILINE,
                )
                return re.sub(
                    r"forall\(([^\n]*)\)\s*\n\s*(\w+):",
                    r"\2: forall(\1)",
                    model_code,
                )

            with open(docplex_explicit_path, "w") as docplex_file:
                docplex_file.write(studio_model(explicit_model))
            with open(docplex_conditional_path, "w") as docplex_file:
                docplex_file.write(studio_model(conditional_model))

            docplex_models = {
                "explicit": build_docplex_model(docplex_explicit_path, data_path),
                "conditional": build_docplex_model(docplex_conditional_path, data_path),
            }
            docplex_solutions = {formulation: model.solve(log_output=False) for formulation, model in docplex_models.items()}
            for solution in docplex_solutions.values():
                self.assertIsNotNone(solution)
            self.assertAlmostEqual(
                docplex_solutions["explicit"].objective_value,
                docplex_solutions["conditional"].objective_value,
                places=6,
            )
            for formulation, model in docplex_models.items():
                for solver in ("scipy", "gurobi"):
                    with self.subTest(formulation=formulation, solver=solver):
                        result = pyopl_results[(formulation, solver)]
                        self.assertAlmostEqual(
                            result["objective_value"],
                            docplex_solutions[formulation].objective_value,
                            places=6,
                        )
                        fixed_model = model.clone()
                        for variable, value in map_solution_variables(model, result["solution"]).items():
                            fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                        candidate = fixed_model.solve(log_output=False)
                        self.assertIsNotNone(candidate)
                        self.assertTrue(candidate.is_valid_solution())
                        self.assertAlmostEqual(
                            candidate.objective_value, docplex_solutions[formulation].objective_value, places=6
                        )
        finally:
            for path in (
                explicit_path,
                conditional_path,
                data_path,
                docplex_explicit_path,
                docplex_conditional_path,
            ):
                if os.path.exists(path):
                    os.remove(path)

    def test_rich_opl_model(self):
        """Compare the rich tuple-indexed knapsack model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int N = ...;
        range Items = 1..N;

        tuple Product {
          int id;
          float profit;
          float weight;
        }

        {Product} products = ...;
        float capacity = ...;

        dvar boolean take[products];

        maximize sum(p in products) p.profit * take[p];

        subject to {
          sum(p in products) p.weight * take[p] <= capacity;
          forall(p in products) {
            take[p] + (1 - take[p]) == 1;
          }
        }
        """
        data_code = """
        N = 4;
        products = { <1, 10.0, 2.0>, <2, 15.0, 3.0>, <3, 7.0, 1.5>, <4, 8.0, 2.5> };
        capacity = 5.0;
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], 25.0, places=6)
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_mini_graph_coloring_with_neq_and_implication(self):
        """Compare graph coloring with != and implication against DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int N = 4;
        range V = 1..N;
        tuple Edge { int u; int v; }
        {Edge} arcs = { <1,2>, <2,3>, <3,4>, <4,1> };
        dvar int+ color[V];
        dvar int+ maxColor;
        minimize maxColor;
        subject to {
          forall(i in V) {
            color[i] >= 1;
            color[i] <= 4;
            maxColor >= color[i];
          }
          forall(e in arcs)
            color[e.u] != color[e.v];
          (color[1] == 2) => (color[2] >= 2);
        }
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write("")
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], 2.0, places=6)
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for name, value in result["solution"].items():
                        if name.startswith("color_"):
                            index = int(re.search(r"\d+", name).group())
                            fixed_model.add_constraint(fixed_model.get_var_by_name(f"color({index})") == value)
                        elif name == "maxColor":
                            fixed_model.add_constraint(fixed_model.get_var_by_name(name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_food_blending_inline_problem(self):
        """Compare the food blending LP with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} Foods = ...;
        {string} Ingredients = ...;
        tuple foodType { float demand; float price; float protein; float fat; }
        tuple ingredientType { float capacity; float price; float protein; float fat; }
        foodType Food[Foods] = ...;
        ingredientType Ingredient[Ingredients] = ...;
        float MaxProduction = ...;
        float ProcCost = ...;
        dvar float+ slack[Foods];
        dvar float+ Mix[Ingredients][Foods];
        maximize sum(f in Foods, ing in Ingredients)
          (Food[f].price - Ingredient[ing].price - ProcCost) * Mix[ing][f]
          - sum(f in Foods) slack[f];
        subject to {
          forall(f in Foods) sum(ing in Ingredients) Mix[ing][f] == Food[f].demand + 10 * slack[f];
          forall(ing in Ingredients) sum(f in Foods) Mix[ing][f] <= Ingredient[ing].capacity;
          sum(ing in Ingredients, f in Foods) Mix[ing][f] <= MaxProduction;
          forall(f in Foods) sum(ing in Ingredients) (Ingredient[ing].protein - Food[f].protein) * Mix[ing][f] >= 0;
          forall(f in Foods) sum(ing in Ingredients) (Ingredient[ing].fat - Food[f].fat) * Mix[ing][f] <= 0;
        }
        """
        data_code = """
        Foods = { "Meal1", "Meal2", "Meal3" };
        Ingredients = { "Chicken", "Beef", "Soy" };
        Food = [ <3000, 9, 30, 10>, <2000, 7, 25, 15>, <1000, 6, 20, 12> ];
        Ingredient = [ <5000, 4, 35, 6>, <5000, 5, 28, 18>, <5000, 3, 22, 14> ];
        MaxProduction = 14000;
        ProcCost = 1.5;
        """
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], 35766.66666666666, places=6)
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for variable, value in map_solution_variables(docplex_model, result["solution"]).items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            for path in (model_path, data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_food_blending_problem(self):
        """Compare the source food blending model with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_food_blending_problem)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float")

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_path = docplex_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], 35766.66666666666, places=6)
                    self.assertAlmostEqual(result["objective_value"], docplex_solution.objective_value, places=6)
                    fixed_model = docplex_model.clone()
                    for variable, value in map_solution_variables(docplex_model, result["solution"]).items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_solution.objective_value, places=6)
        finally:
            for path in (model_path, data_path, docplex_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_transportation_problem_with_tuples_and_string_sets(self):
        """Compare the source tuple-arc transportation model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_transportation_problem_with_tuples_and_string_sets)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float")
        docplex_model_code = re.sub(
            r"^(\s*float\s+\w+(?:\[[^;]+\])?);",
            r"\1 = ...;",
            docplex_model_code,
            flags=re.MULTILINE,
        )
        docplex_data_code = re.sub(
            r"supply\s*=\s*\[.*?\];",
            "supply = [ 350, 600 ];",
            data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"demand\s*=\s*\[.*?\];",
            "demand = [ 325, 300, 275 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"cost\s*=\s*\[.*?\];",
            "cost = [ 0, 2.5, 1.7, 2.5, 1.8, 1.4 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"capacity\s*=\s*\[.*?\];",
            "capacity = [ 200, 250, 200, 300, 300, 400 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"min_shipment\s*=\s*\[.*?\];",
            "min_shipment = [ 0, 0, 0, 0, 0, 50 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_path = docplex_file.name
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_data_path, "w") as docplex_data_file:
                docplex_data_file.write(docplex_data_code)
            docplex_model = build_docplex_model(docplex_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for variable, value in map_solution_variables(docplex_model, result["solution"]).items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            for path in (model_path, data_path, docplex_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_transportation_problem_with_tuples_and_string_sets_and_string_filtering(self):
        """Compare tuple-arc transportation with direct string filtering."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(
            TestPyOPLProblems.test_transportation_problem_with_tuples_and_string_sets_and_string_filtering
        )
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)

        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float")
        docplex_model_code = re.sub(
            r"^(\s*float\s+\w+(?:\[[^;]+\])?);",
            r"\1 = ...;",
            docplex_model_code,
            flags=re.MULTILINE,
        )
        docplex_data_code = re.sub(
            r"supply\s*=\s*\[.*?\];",
            "supply = [ 350, 600 ];",
            data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"demand\s*=\s*\[.*?\];",
            "demand = [ 325, 300, 275 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"cost\s*=\s*\[.*?\];",
            "cost = [ 0, 2.5, 1.7, 2.5, 1.8, 1.4 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"capacity\s*=\s*\[.*?\];",
            "capacity = [ 200, 250, 200, 300, 300, 400 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_data_code = re.sub(
            r"min_shipment\s*=\s*\[.*?\];",
            "min_shipment = [ 0, 0, 0, 0, 0, 50 ];",
            docplex_data_code,
            count=1,
            flags=re.DOTALL,
        )

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_path = docplex_file.name
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_data_path, "w") as docplex_data_file:
                docplex_data_file.write(docplex_data_code)
            docplex_model = build_docplex_model(docplex_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for variable, value in map_solution_variables(docplex_model, result["solution"]).items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            for path in (model_path, data_path, docplex_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_inventory_problem_with_tuples(self):
        """Compare the tuple-store inventory model with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_inventory_problem_with_tuples)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)

        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param float", "float")
        docplex_model_code = re.sub(
            r"^(\s*(?:int|float)\s+\w+(?:\[[^;]+\])?);",
            r"\1 = ...;",
            docplex_model_code,
            flags=re.MULTILINE,
        )
        docplex_model_code = docplex_model_code.replace(
            "{Store} Stores;",
            '{Store} Stores = { <"S1">, <"S2"> };',
        )
        docplex_model_code = docplex_model_code.replace("int Capacity[Stores] = ...;", "")
        docplex_model_code = docplex_model_code.replace("Capacity[s]", "100")
        docplex_model_code = docplex_model_code.replace("int Demand[Periods] = ...;", "int Demand[Periods] = [ 1, 2, 3 ];")
        docplex_model_code = docplex_model_code.replace(
            "int OrderingCost[Periods] = ...;", "int OrderingCost[Periods] = [ 10, 13, 15 ];"
        )
        docplex_model_code = docplex_model_code.replace("int HoldingCost = ...;", "int HoldingCost = 1;")
        docplex_data_code = re.sub(r"Stores\s*=\s*\{.*?\};", "", data_code, count=1, flags=re.DOTALL)
        docplex_data_code = re.sub(r"Capacity\s*=\s*\[.*?\];", "", docplex_data_code, count=1, flags=re.DOTALL)
        docplex_data_code = re.sub(r"Demand\s*=\s*\[.*?\];", "", docplex_data_code, count=1, flags=re.DOTALL)
        docplex_data_code = re.sub(r"OrderingCost\s*=\s*\[.*?\];", "", docplex_data_code, count=1, flags=re.DOTALL)
        docplex_data_code = re.sub(r"HoldingCost\s*=\s*\d+\s*;", "", docplex_data_code, count=1)

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_path = docplex_file.name
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_data_path, "w") as docplex_data_file:
                docplex_data_file.write(docplex_data_code)
            docplex_model = build_docplex_model(docplex_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for variable, value in map_solution_variables(docplex_model, result["solution"]).items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            for path in (model_path, data_path, docplex_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_complex_inventory_problem_with_tuples(self):
        """Compare the complex tuple-store inventory model with DOcplex and CPLEX."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_complex_inventory_problem_with_tuples)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)

        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = re.sub(
            r"tuple Store\s*\{\s*string name;\s*\}\s*\{Store\} Stores;",
            '{string} Stores = { "StoreA", "StoreB" };',
            model_code,
            count=1,
            flags=re.DOTALL,
        )
        docplex_model_code = docplex_model_code.replace("param float", "float").replace("param int", "int")
        docplex_model_code = re.sub(
            r"^(\s*(?:int|float)\s+\w+(?:\[[^;]+\])?);",
            r"\1 = ...;",
            docplex_model_code,
            flags=re.MULTILINE,
        )
        docplex_data_code = """
        Capacity = [ 100, 100 ];
        Demand = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ];
        TransportCost = [ [ 10, 12, 15 ], [ 8, 11, 13 ] ];
        HoldingCost = 1;
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            docplex_file.write(docplex_model_code)
            model_path = mod_file.name
            data_path = dat_file.name
            docplex_path = docplex_file.name
        docplex_data_path = data_path + ".docplex.dat"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_data_path, "w") as docplex_data_file:
                docplex_data_file.write(docplex_data_code)
            docplex_model = build_docplex_model(docplex_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    for variable, value in map_solution_variables(docplex_model, result["solution"]).items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(candidate.objective_value, docplex_objective, places=6)
        finally:
            for path in (model_path, data_path, docplex_path, docplex_data_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_stochastic_economic_lot_scheduling(self):
        """Compare the stochastic ELSP model across PyOPL and DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        range Items = 1..5;
        range Periods = 1..4;
        range Scenarios = 1..3;
        int NItems = 5;
        float prob[Scenarios] = ...;
        float demand[Scenarios][Items][Periods] = ...;
        float init_inv[Items] = ...;
        float proc_time[Items] = ...;
        float capacity[Periods] = ...;
        float start_setup_time[Items] = ...;
        float start_setup_cost[Items] = ...;
        float seq_setup_time[Items][Items] = ...;
        float seq_setup_cost[Items][Items] = ...;
        float prod_cost[Items] = ...;
        float hold_cost[Items] = ...;
        float back_cost[Items] = ...;
        float max_prod[Items] = ...;
        dvar float+ x[Items][Periods];
        dvar boolean y[Items][Periods];
        dvar boolean first[Items][Periods];
        dvar boolean a[Items][Items][Periods];
        dvar int+ u[Items][Periods];
        dvar float+ inv[Items][Periods][Scenarios];
        dvar float+ bo[Items][Periods][Scenarios];
        minimize
            sum(i in Items, t in Periods) prod_cost[i] * x[i][t]
            + sum(i in Items, t in Periods) start_setup_cost[i] * first[i][t]
            + sum(i in Items, j in Items, t in Periods) seq_setup_cost[i][j] * a[i][j][t]
            + sum(s in Scenarios) prob[s] *
                sum(i in Items, t in Periods)
                    (hold_cost[i] * inv[i][t][s] + back_cost[i] * bo[i][t][s]);
        subject to {
            forall(t in Periods)
                sum(i in Items) proc_time[i] * x[i][t]
                + sum(i in Items) start_setup_time[i] * first[i][t]
                + sum(i in Items, j in Items) seq_setup_time[i][j] * a[i][j][t]
                <= capacity[t];
            forall(i in Items, t in Periods) {
                x[i][t] <= max_prod[i] * y[i][t];
                first[i][t] <= y[i][t];
            }
            forall(j in Items, t in Periods)
                sum(i in Items) a[i][j][t] + first[j][t] == y[j][t];
            forall(i in Items, t in Periods)
                sum(j in Items) a[i][j][t] <= y[i][t];
            forall(i in Items, t in Periods) a[i][i][t] == 0;
            forall(t in Periods) sum(i in Items) first[i][t] <= 1;
            forall(t in Periods, i in Items) sum(j in Items) first[j][t] >= y[i][t];
            forall(i in Items, t in Periods) u[i][t] <= NItems * y[i][t];
            forall(i in Items, j in Items, t in Periods : i != j)
                u[i][t] - u[j][t] + 1 <= (1 - a[i][j][t]) * NItems;
            forall(i in Items, s in Scenarios)
                inv[i][1][s] - bo[i][1][s] == init_inv[i] + x[i][1] - demand[s][i][1];
            forall(i in Items, t in Periods, s in Scenarios : t > 1)
                inv[i][t][s] - bo[i][t][s] ==
                    inv[i][t-1][s] - bo[i][t-1][s] + x[i][t] - demand[s][i][t];
            sum(s in Scenarios) prob[s] == 1;
        }
        """
        data_code = """
        prob = [0.3, 0.5, 0.2];
        demand = [
            [ [18,15,20,16], [12,10,13,11], [20,18,22,19], [8,7,9,8], [14,12,16,13] ],
            [ [22,18,24,19], [14,12,16,13], [24,21,26,23], [10,9,11,10], [17,15,19,16] ],
            [ [20,17,22,18], [13,11,15,12], [22,20,24,21], [9,8,10,9], [16,14,18,15] ]
        ];
        init_inv = [10, 8, 12, 6, 9];
        proc_time = [1.2, 1.0, 1.4, 0.9, 1.1];
        capacity = [120, 115, 125, 118];
        start_setup_time = [4, 3, 5, 2, 4];
        start_setup_cost = [40, 35, 45, 25, 38];
        seq_setup_time = [
            [0,3,4,2,3], [3,0,2,3,2], [4,2,0,4,3], [2,3,4,0,2], [3,2,3,2,0]
        ];
        seq_setup_cost = [
            [0,22,28,18,20], [21,0,19,20,18], [27,20,0,26,23], [17,19,25,0,16], [20,18,22,17,0]
        ];
        prod_cost = [5.0, 4.5, 6.0, 3.8, 5.2];
        hold_cost = [0.8, 0.7, 0.9, 0.6, 0.75];
        back_cost = [7.5, 6.5, 8.0, 5.5, 7.0];
        max_prod = [35, 28, 40, 22, 30];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    candidate = docplex_model.new_solution(map_solution_variables(docplex_model, result["solution"]))
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.get_value(docplex_model.objective_expr),
                        docplex_objective,
                        places=6,
                    )
        finally:
            os.remove(model_path)
            os.remove(data_path)

    def test_stochastic_job_shop_scheduling(self):
        """Compare the here-and-now stochastic job-shop model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        range Jobs = 1..3;
        range Ops = 1..3;
        range Machines = 1..3;
        {string} Scenarios = ...;
        tuple Operation { int j; int o; int m; }
        {Operation} Operations = ...;
        float prob[Scenarios] = ...;
        int due[Jobs] = ...;
        int p[Scenarios][Jobs][Ops] = ...;
        int pMax[j in Jobs][o in Ops] = max(s in Scenarios) p[s][j][o];
        int M = sum(j in Jobs, o in Ops) pMax[j][o];
        dvar int+ start[Jobs][Ops];
        dvar boolean z[Operations][Operations];
        dvar int+ Cmax[Scenarios];
        dvar boolean bMeet[Scenarios][Jobs];
        dvar boolean sat[Scenarios];
        minimize sum(s in Scenarios) prob[s] * Cmax[s];
        subject to {
            forall(j in Jobs, o in Ops) start[j][o] <= M;
            forall(s in Scenarios, j in Jobs, o in Ops : o < 3)
                start[j][o+1] >= start[j][o] + p[s][j][o];
            forall(s in Scenarios, e1 in Operations, e2 in Operations :
                (e1.m == e2.m) && ((e1.j < e2.j) ||
                ((e1.j == e2.j) && (e1.o < e2.o)))) {
                start[e1.j][e1.o] + p[s][e1.j][e1.o]
                    <= start[e2.j][e2.o] + M * (1 - z[e1][e2]);
                start[e2.j][e2.o] + p[s][e2.j][e2.o]
                    <= start[e1.j][e1.o] + M * z[e1][e2];
            }
            forall(s in Scenarios, j in Jobs)
                Cmax[s] >= start[j][3] + p[s][j][3];
            forall(s in Scenarios, j in Jobs)
                bMeet[s][j] == (start[j][3] + p[s][j][3] <= due[j]);
            forall(s in Scenarios) {
                sat[s] <= bMeet[s][1];
                sat[s] <= bMeet[s][2];
                sat[s] <= bMeet[s][3];
                sum(j in Jobs) bMeet[s][j] >= 3 * sat[s];
            }
            sum(s in Scenarios) prob[s] * sat[s] >= 0.25;
            sum(s in Scenarios) prob[s] == 1;
        }
        """
        data_code = """
        Scenarios = { "S1", "S2", "S3", "S4" };
        Operations = {
            <1,1,1>, <1,2,2>, <1,3,3>, <2,1,2>, <2,2,1>, <2,3,3>,
            <3,1,1>, <3,2,3>, <3,3,2>
        };
        due = [15, 21, 21];
        prob = ["S1" 0.25, "S2" 0.25, "S3" 0.10, "S4" 0.40];
        p = [
            "S1" [[3,2,4], [2,4,3], [4,3,2]],
            "S2" [[4,3,5], [5,4,4], [5,3,3]],
            "S3" [[3,2,4], [4,6,4], [4,4,2]],
            "S4" [[5,3,6], [4,5,4], [6,4,3]]
        ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_data_path = data_path + ".docplex.dat"
            with open(docplex_data_path, "w") as target_data:
                target_data.write(
                    data_code.replace(
                        '        prob = ["S1" 0.25, "S2" 0.25, "S3" 0.10, "S4" 0.40];',
                        "        prob = [0.25, 0.25, 0.10, 0.40];",
                    )
                    .replace(
                        '            "S1" [[3,2,4], [2,4,3], [4,3,2]],',
                        "            [[3,2,4], [2,4,3], [4,3,2]],",
                    )
                    .replace(
                        '            "S2" [[4,3,5], [5,4,4], [5,3,3]],',
                        "            [[4,3,5], [5,4,4], [5,3,3]],",
                    )
                    .replace(
                        '            "S3" [[3,2,4], [4,6,4], [4,4,2]],',
                        "            [[3,2,4], [4,6,4], [4,4,2]],",
                    )
                    .replace(
                        '            "S4" [[5,3,6], [4,5,4], [6,4,3]]',
                        "            [[5,3,6], [4,5,4], [6,4,3]]",
                    )
                )
            docplex_model = build_docplex_model(model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    fixed_model = build_docplex_model(model_path, docplex_data_path)
                    fixed_variables = {variable.name: variable for variable in fixed_model.iter_variables()}
                    for variable, value in mapped_solution.items():
                        if variable.name.startswith("start"):
                            fixed_model.add_constraint(fixed_variables[variable.name] == value)
                    fixed_solution = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(fixed_solution)
                    self.assertAlmostEqual(
                        fixed_solution.objective_value,
                        docplex_objective,
                        places=6,
                    )
        finally:
            os.remove(model_path)
            os.remove(data_path)
            if os.path.exists(data_path + ".docplex.dat"):
                os.remove(data_path + ".docplex.dat")

    def test_stochastic_plane_landing_2(self):
        """Compare the stochastic aircraft landing model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        {string} Aircraft = ...;
        int nbSlots = ...;
        range Slots = 1..nbSlots;
        {string} Scenarios = ...;
        int feasible[Aircraft][Slots] = ...;
        int nominal[Aircraft] = ...;
        float prob[Scenarios] = ...;
        int delay[Aircraft][Scenarios] = ...;
        float alpha = ...;
        dvar boolean x[Aircraft][Slots];
        dvar int+ duration;
        dexpr int assigned[a in Aircraft] = sum(t in Slots) t * x[a][t];
        minimize duration;
        subject to {
            forall(a in Aircraft) sum(t in Slots) x[a][t] == 1;
            forall(t in Slots) sum(a in Aircraft) x[a][t] <= 1;
            forall(a in Aircraft, t in Slots) x[a][t] <= feasible[a][t];
            forall(a in Aircraft) duration >= assigned[a];
            sum(s in Scenarios) prob[s] *
                ((sum(a in Aircraft)
                    (assigned[a] >= nominal[a] + delay[a][s])) ==
                    sum(a in Aircraft) 1) >= alpha;
            sum(s in Scenarios) prob[s] == 1;
        }
        """
        data_code = """
        Aircraft = { "A1", "A2", "A3", "A4" };
        nbSlots = 6;
        Scenarios = { "S1", "S2", "S3", "S4" };
        feasible = [
            "A1" [1, 1, 1, 0, 0, 0],
            "A2" [0, 1, 1, 1, 0, 0],
            "A3" [0, 0, 1, 1, 1, 0],
            "A4" [0, 0, 0, 1, 1, 1]
        ];
        nominal = ["A1" 1, "A2" 2, "A3" 3, "A4" 4];
        prob = ["S1" 0.25, "S2" 0.25, "S3" 0.25, "S4" 0.25];
        delay = [
            "A1" [0, 1, 0, 2],
            "A2" [0, 0, 1, 1],
            "A3" [0, 1, 1, 2],
            "A4" [0, 0, 1, 1]
        ];
        alpha = 0.5;
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_data_path = data_path + ".docplex.dat"
            with open(docplex_data_path, "w") as target_data:
                target_data.write(
                    data_code.replace(
                        '        nominal = ["A1" 1, "A2" 2, "A3" 3, "A4" 4];',
                        "        nominal = [1, 2, 3, 4];",
                    )
                    .replace(
                        '        prob = ["S1" 0.25, "S2" 0.25, "S3" 0.25, "S4" 0.25];',
                        "        prob = [0.25, 0.25, 0.25, 0.25];",
                    )
                    .replace(
                        '            "A1" [1, 1, 1, 0, 0, 0],',
                        "            [1, 1, 1, 0, 0, 0],",
                    )
                    .replace(
                        '            "A2" [0, 1, 1, 1, 0, 0],',
                        "            [0, 1, 1, 1, 0, 0],",
                    )
                    .replace(
                        '            "A3" [0, 0, 1, 1, 1, 0],',
                        "            [0, 0, 1, 1, 1, 0],",
                    )
                    .replace(
                        '            "A4" [0, 0, 0, 1, 1, 1]',
                        "            [0, 0, 0, 1, 1, 1]",
                    )
                    .replace(
                        '            "A1" [0, 1, 0, 2],',
                        "            [0, 1, 0, 2],",
                    )
                    .replace(
                        '            "A2" [0, 0, 1, 1],',
                        "            [0, 0, 1, 1],",
                    )
                    .replace(
                        '            "A3" [0, 1, 1, 2],',
                        "            [0, 1, 1, 2],",
                    )
                    .replace(
                        '            "A4" [0, 0, 1, 1]',
                        "            [0, 0, 1, 1]",
                    )
                )
            docplex_model = build_docplex_model(model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    mapped = map_solution_variables(docplex_model, result["solution"])
                    fixed_model = build_docplex_model(model_path, docplex_data_path)
                    fixed_variables = {variable.name: variable for variable in fixed_model.iter_variables()}
                    for variable, value in mapped.items():
                        if variable.name.startswith("x") or variable.name == "duration":
                            fixed_model.add_constraint(fixed_variables[variable.name] == value)
                    fixed_solution = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(fixed_solution)
                    self.assertAlmostEqual(fixed_solution.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            if os.path.exists(data_path + ".docplex.dat"):
                os.remove(data_path + ".docplex.dat")

    def test_stochastic_plane_landing(self):
        """Compare the linked-reliability landing model with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_stochastic_plane_landing)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param ", "")
        docplex_model_code = docplex_model_code.replace("minimize MinimizeMakespan:", "minimize")
        docplex_model_code = re.sub(
            r"forall\(([^)]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_data_code = data_code
        for keyed_rows, positional_rows in (
            (
                '"A1" [ 0, 1, 0, 2 ],\n                "A2" [ 0, 0, 1, 1 ],\n                "A3" [ 0, 1, 1, 2 ],\n                "A4" [ 0, 0, 1, 1 ]',
                "[ 0, 1, 0, 2 ],\n                [ 0, 0, 1, 1 ],\n                [ 0, 1, 1, 2 ],\n                [ 0, 0, 1, 1 ]",
            ),
            (
                '"A1" [ 1, 1, 1, 0, 0, 0 ],\n                "A2" [ 0, 1, 1, 1, 0, 0 ],\n                "A3" [ 0, 0, 1, 1, 1, 0 ],\n                "A4" [ 0, 0, 0, 1, 1, 1  ]',
                "[ 1, 1, 1, 0, 0, 0 ],\n                [ 0, 1, 1, 1, 0, 0 ],\n                [ 0, 0, 1, 1, 1, 0 ],\n                [ 0, 0, 0, 1, 1, 1  ]",
            ),
        ):
            docplex_data_code = docplex_data_code.replace(keyed_rows, positional_rows)
        docplex_data_code = docplex_data_code.replace(
            '"A1" 1,\n                "A2" 2,\n                "A3" 3,\n                "A4" 4',
            "1, 2, 3, 4",
        ).replace(
            '"A1" [ 0, 1, 0, 2 ],\n                "A2" [ 0, 0, 1, 1 ],',
            "[ 0, 1, 0, 2 ],\n                [ 0, 0, 1, 1 ],",
        )
        docplex_data_code = re.sub(r'(?m)^(\s*)"A[1-4]"(\s+)', r"\1", docplex_data_code)

        model_path = data_path = docplex_model_path = docplex_data_path = None
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path, data_path = mod_file.name, dat_file.name
        docplex_data_path = data_path + ".docplex.dat"
        docplex_model_path = model_path + ".docplex.mod"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_data_path, "w") as dat_file:
                dat_file.write(docplex_data_code)
            with open(docplex_model_path, "w") as mod_file:
                mod_file.write(docplex_model_code)
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_solution.objective_value, places=6)
                    fixed_model = docplex_model.clone()
                    mapped = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped.items():
                        if variable.name.startswith("x") or variable.name == "duration":
                            fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    fixed_solution = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(fixed_solution)
                    self.assertTrue(fixed_solution.is_valid_solution())
                    self.assertAlmostEqual(fixed_solution.objective_value, docplex_solution.objective_value, places=6)
        finally:
            for path in (model_path, data_path, docplex_model_path, docplex_data_path):
                if path and os.path.exists(path):
                    os.remove(path)

    def test_hotel_rostering(self):
        """Compare the hotel rostering MILP with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        from test.test_problems import TestPyOPLProblems

        source = inspect.getsource(TestPyOPLProblems.test_hotel_rostering)
        tree = ast.parse(textwrap.dedent(source))
        literals = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"model_code", "data_code"}:
                    literals[target.id] = ast.literal_eval(node.value)
        model_code = literals["model_code"]
        data_code = literals["data_code"]
        docplex_model_code = model_code.replace("param ", "")
        docplex_model_code = docplex_model_code.replace("boolean shiftConflict", "int shiftConflict")
        docplex_model_code = docplex_model_code.replace(" : shiftConflict[e][s][t])", " : shiftConflict[e][s][t] == 1)")
        docplex_model_code = re.sub(
            r"forall\(([^)]*)\)\s+([A-Za-z_]\w*):",
            r"\2: forall(\1)",
            docplex_model_code,
        )
        docplex_model_code = docplex_model_code.replace("minimize TotalWeightedPenalty =", "minimize")

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path, data_path = mod_file.name, dat_file.name
        docplex_model_path = model_path + ".docplex.mod"
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            with open(docplex_model_path, "w") as mod_file:
                mod_file.write(docplex_model_code)
            docplex_model = build_docplex_model(docplex_model_path, data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_solution.objective_value, places=6)
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, result["solution"])
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    fixed_solution = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(fixed_solution)
                    self.assertTrue(fixed_solution.is_valid_solution())
                    self.assertAlmostEqual(fixed_solution.objective_value, docplex_solution.objective_value, places=6)
        finally:
            for path in (model_path, data_path, docplex_model_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_stochastic_vrp(self):
        """Compare the two-stage stochastic VRP with DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        source = self._stochastic_vrp_fixture()
        model_code, data_code, docplex_model_code, docplex_data = source
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        docplex_data_path = data_path + ".docplex.dat"
        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as docplex_mod_file:
            docplex_mod_file.write(docplex_model_code)
            docplex_model_path = docplex_mod_file.name
        try:
            with open(docplex_data_path, "w") as compatibility_data:
                compatibility_data.write(docplex_data)
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_model = build_docplex_model(docplex_model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value
            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    fixed_model = docplex_model.clone()
                    mapped = map_solution_variables(fixed_model, result["solution"])
                    for variable, value in mapped.items():
                        if variable.name.startswith(("x", "serve", "deliv")):
                            fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    fixed_solution = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(fixed_solution)
                    self.assertTrue(fixed_solution.is_valid_solution())
                    self.assertAlmostEqual(fixed_solution.objective_value, docplex_objective, places=6)
        finally:
            os.remove(model_path)
            os.remove(data_path)
            os.remove(docplex_model_path)
            if os.path.exists(docplex_data_path):
                os.remove(docplex_data_path)

    @staticmethod
    def _stochastic_vrp_fixture():
        model_code = """
        int N = ...; range Nodes = 0..N; range Customers = 1..N;
        int nbVeh = ...; range Vehicles = 1..nbVeh; {string} Scenarios = ...;
        float cost[Nodes][Nodes] = ...; float travel[Scenarios][Nodes][Nodes] = ...;
        float prob[Scenarios] = ...; float dem[Scenarios][Customers] = ...;
        float s_time[Customers] = ...; float tw_open[Customers] = ...;
        float tw_close[Customers] = ...; float cap = ...; float penalty = ...; float H = ...;
        dvar boolean x[Vehicles][Nodes][Nodes];
        dvar boolean serve[Vehicles][Customers][Scenarios];
        dvar float+ deliv[Vehicles][Customers][Scenarios]; dvar float+ unmet[Customers][Scenarios];
        dvar float+ time[Vehicles][Nodes][Scenarios]; dvar float+ load[Vehicles][Nodes][Scenarios];
        minimize sum(v in Vehicles, i in Nodes, j in Nodes: i != j) cost[i][j] * x[v][i][j]
          + sum(s in Scenarios) prob[s] * penalty * sum(i in Customers) unmet[i][s];
        subject to {
          forall(v in Vehicles, i in Nodes) x[v][i][i] == 0;
          forall(v in Vehicles) { sum(j in Customers) x[v][0][j] <= 1; sum(i in Customers) x[v][i][0] <= 1;
            sum(j in Customers) x[v][0][j] == sum(i in Customers) x[v][i][0]; }
          sum(v in Vehicles, j in Customers) x[v][0][j] <= nbVeh;
          forall(v in Vehicles, i in Customers) sum(j in Nodes: j != i) x[v][i][j] == sum(j in Nodes: j != i) x[v][j][i];
          forall(i in Customers) { sum(v in Vehicles, j in Nodes: j != i) x[v][j][i] <= 1;
            sum(v in Vehicles, j in Nodes: j != i) x[v][i][j] <= 1;
            sum(v in Vehicles, j in Nodes: j != i) x[v][i][j] == sum(v in Vehicles, j in Nodes: j != i) x[v][j][i]; }
          forall(s in Scenarios, v in Vehicles, i in Customers) { serve[v][i][s] <= sum(j in Nodes: j != i) x[v][i][j]; serve[v][i][s] <= sum(j in Nodes: j != i) x[v][j][i]; }
          forall(s in Scenarios, i in Customers) sum(v in Vehicles) serve[v][i][s] <= 1;
          forall(s in Scenarios, v in Vehicles, i in Customers) deliv[v][i][s] <= dem[s][i] * serve[v][i][s];
          forall(s in Scenarios, i in Customers) { sum(v in Vehicles) deliv[v][i][s] + unmet[i][s] == dem[s][i]; unmet[i][s] <= dem[s][i] * (1 - sum(v in Vehicles) serve[v][i][s]); }
          forall(v in Vehicles, s in Scenarios) time[v][0][s] == 0;
          forall(v in Vehicles, s in Scenarios, n in Nodes) time[v][n][s] <= H;
          forall(v in Vehicles, s in Scenarios, j in Customers) time[v][j][s] >= travel[s][0][j] - H * (1 - x[v][0][j]);
          forall(v in Vehicles, s in Scenarios, i in Customers, j in Customers: i != j) time[v][j][s] >= time[v][i][s] + s_time[i] * serve[v][i][s] + travel[s][i][j] - H * (1 - x[v][i][j]);
          forall(v in Vehicles, s in Scenarios, i in Customers) { time[v][i][s] >= tw_open[i] * serve[v][i][s]; time[v][i][s] <= tw_close[i] + H * (1 - serve[v][i][s]); }
          forall(v in Vehicles, s in Scenarios) load[v][0][s] == 0;
          forall(v in Vehicles, s in Scenarios, n in Nodes) load[v][n][s] <= cap;
          forall(v in Vehicles, s in Scenarios, j in Customers) load[v][j][s] >= deliv[v][j][s] - cap * (1 - x[v][0][j]);
          forall(v in Vehicles, s in Scenarios, i in Customers, j in Customers: i != j) load[v][j][s] >= load[v][i][s] + deliv[v][j][s] - cap * (1 - x[v][i][j]);
        }
        """
        data_code = """
        N = 3; nbVeh = 2; Scenarios = { "S1", "S2" };
        cost = [[0,10,12,15],[10,0,6,8],[12,6,0,7],[15,8,7,0]];
        prob = ["S1" 0.5, "S2" 0.5];
        travel = ["S1" [[0,10,12,15],[10,0,6,8],[12,6,0,7],[15,8,7,0]], "S2" [[0,11,13,16],[11,0,7,9],[13,7,0,8],[16,9,8,0]]];
        dem = ["S1" [4,3,5], "S2" [5,2,6]]; s_time = [5,4,6]; tw_open = [10,15,20]; tw_close = [40,50,60]; cap = 10; penalty = 100; H = 100;
        """
        docplex_data = (
            data_code.replace('prob = ["S1" 0.5, "S2" 0.5];', "prob = [0.5, 0.5];")
            .replace(
                'travel = ["S1" [[0,10,12,15],[10,0,6,8],[12,6,0,7],[15,8,7,0]], "S2" [[0,11,13,16],[11,0,7,9],[13,7,0,8],[16,9,8,0]]];',
                "travel = [[[0,10,12,15],[10,0,6,8],[12,6,0,7],[15,8,7,0]], [[0,11,13,16],[11,0,7,9],[13,7,0,8],[16,9,8,0]]];",
            )
            .replace('dem = ["S1" [4,3,5], "S2" [5,2,6]];', "dem = [[4,3,5], [5,2,6]];")
        )
        return model_code, data_code, model_code, docplex_data

    def test_stochastic_job_shop_scheduling_2(self):
        """Compare the stochastic job-shop model across PyOPL and DOcplex."""
        if find_spec("cplex") is None:
            self.skipTest("DOcplex OPL cross-check requires the cplex runtime")

        model_code = """
        int nbJobs = ...;
        int nbMachines = ...;
        int K = ...;
        range Jobs = 1..nbJobs;
        range Machines = 1..nbMachines;
        range Ops = 1..K;
        {string} Scenarios = { "S1", "S2", "S3", "S4" };
        float prob[Scenarios] = ...;
        int due[Jobs] = ...;
        int route[Jobs][Ops] = ...;
        int p[Scenarios][Jobs][Ops] = ...;
        int M = ...;
        int H = ...;
        dvar int+ start[Jobs][Ops];
        dvar boolean z[Jobs][Jobs][Machines];
        dvar int+ Cmax[Scenarios];
        dvar boolean sat[Scenarios];
        minimize sum(s in Scenarios) prob[s] * Cmax[s];
        subject to {
            forall(j in Jobs, k in Ops) start[j][k] <= H;
            forall(s in Scenarios) Cmax[s] <= H;
            forall(s in Scenarios, j in Jobs, k in 1..K-1)
                start[j][k+1] >= start[j][k] + p[s][j][k];
            forall(m in Machines, j1 in Jobs, j2 in Jobs : j1 < j2)
                z[j1][j2][m] + z[j2][j1][m] == 1;
            forall(s in Scenarios, m in Machines, j1 in Jobs, j2 in Jobs : j1 < j2) {
                sum(k1 in Ops) (route[j1][k1] == m) *
                    (start[j1][k1] + p[s][j1][k1])
                    <= sum(k2 in Ops) (route[j2][k2] == m) * start[j2][k2]
                        + M * z[j1][j2][m];
                sum(k2 in Ops) (route[j2][k2] == m) *
                    (start[j2][k2] + p[s][j2][k2])
                    <= sum(k1 in Ops) (route[j1][k1] == m) * start[j1][k1]
                        + M * (1 - z[j1][j2][m]);
            }
            forall(s in Scenarios, j in Jobs)
                Cmax[s] >= start[j][3] + p[s][j][3];
            forall(s in Scenarios, j in Jobs)
                sat[s] <= (start[j][3] + p[s][j][3] <= due[j]);
            forall(s in Scenarios)
                sat[s] >= sum(j in Jobs)
                    (start[j][3] + p[s][j][3] <= due[j]) - (nbJobs - 1);
            sum(s in Scenarios) prob[s] * sat[s] >= 0.25;
            sum(s in Scenarios) prob[s] == 1;
        }
        """
        data_code = """
        nbJobs = 3;
        nbMachines = 3;
        K = 3;
        M = 42;
        H = 42;
        Scenarios = { "S1", "S2", "S3", "S4" };
        prob = [ 0.25, 0.25, 0.10, 0.40 ];
        due = [ 15, 21, 21 ];
        route = [ [1, 2, 3], [2, 1, 3], [1, 3, 2] ];
        p = [
            "S1" [ [3, 2, 4], [2, 4, 3], [4, 3, 2] ],
            "S2" [ [4, 3, 5], [5, 4, 4], [5, 3, 3] ],
            "S3" [ [3, 2, 4], [4, 6, 4], [4, 4, 2] ],
            "S4" [ [5, 3, 6], [4, 5, 4], [6, 4, 3] ]
        ];
        """

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as mod_file,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as dat_file,
        ):
            mod_file.write(model_code)
            dat_file.write(data_code)
            model_path = mod_file.name
            data_path = dat_file.name
        try:
            results = {solver: solve(model_path, data_path, solver=solver) for solver in ("scipy", "gurobi")}
            docplex_data_path = data_path + ".docplex.dat"
            with open(data_path) as source_data, open(docplex_data_path, "w") as target_data:
                target_data.write(
                    source_data.read()
                    .replace(
                        '        Scenarios = { "S1", "S2", "S3", "S4" };\n',
                        "",
                    )
                    .replace(
                        '            "S1" [',
                        "            [",
                    )
                    .replace(
                        '            "S2" [',
                        "            [",
                    )
                    .replace(
                        '            "S3" [',
                        "            [",
                    )
                    .replace(
                        '            "S4" [',
                        "            [",
                    )
                )
            docplex_model = build_docplex_model(model_path, docplex_data_path)
            docplex_solution = docplex_model.solve(log_output=False)
            self.assertIsNotNone(docplex_solution)
            docplex_objective = docplex_solution.objective_value

            for solver, result in results.items():
                with self.subTest(solver=solver):
                    self.assertEqual(result["status"], "OPTIMAL")
                    self.assertAlmostEqual(result["objective_value"], docplex_objective, places=6)
                    solution = {
                        name: value
                        for name, value in result["solution"].items()
                        if not re.match(r"z(?:_|\[)(\d+)(?:_|,|\])\1", name)
                        and (name.startswith("start") or name.startswith("z"))
                    }
                    fixed_model = docplex_model.clone()
                    mapped_solution = map_solution_variables(docplex_model, solution)
                    for variable, value in mapped_solution.items():
                        fixed_model.add_constraint(fixed_model.get_var_by_name(variable.name) == value)
                    candidate = fixed_model.solve(log_output=False)
                    self.assertIsNotNone(candidate)
                    self.assertTrue(candidate.is_valid_solution())
                    self.assertAlmostEqual(
                        candidate.objective_value,
                        docplex_objective,
                        places=6,
                    )
        finally:
            os.remove(model_path)
            os.remove(data_path)
            if os.path.exists(data_path + ".docplex.dat"):
                os.remove(data_path + ".docplex.dat")

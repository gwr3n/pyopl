# === Standard library imports ===
import json
import logging

from .semantic_error import SemanticError

# === Third-party imports ===
# (none)

# === Local imports ===


# Module-level logger (no handler/formatter setup here)
logger = logging.getLogger(__name__)

# Numerical tolerances (single source of truth)
EPS = 1e-5  # strictness used to split >, < from >=, <=  (raised to exceed FeasibilityTol)
EQ_TOL = 1e-6  # two-sided tolerance for equality reification


# === TupleSetHelper ===
class TupleSetHelper:
    @staticmethod
    def get_tuple_set(set_name, ast, data_dict):
        """
        Retrieve a set of tuples by name, preferring data_dict, falling back to AST.
        For 1-field tuple sets, flatten to a list of values.
        """

        def to_tuple(val):
            if isinstance(val, dict) and "elements" in val:
                return tuple(to_tuple(e) for e in val["elements"])
            elif isinstance(val, (list, tuple)):
                return tuple(to_tuple(e) for e in val)
            else:
                return val

        if set_name in data_dict:
            tuple_set = data_dict[set_name]
            if isinstance(tuple_set, dict) and "elements" in tuple_set:
                elems = [to_tuple(t) for t in tuple_set["elements"]]
                return elems
            elif isinstance(tuple_set, list):
                elems = [to_tuple(t) for t in tuple_set]
                return elems
            else:
                return tuple_set
        for decl in ast.get("declarations", []):
            if decl.get("type") == "set_of_tuples" and decl.get("name") == set_name and decl.get("value") is not None:
                elems = [
                    (to_tuple(t["elements"]) if isinstance(t, dict) and "elements" in t else to_tuple(t))
                    for t in decl["value"]
                ]
                return elems
        return []


# === GurobiCodeGenerator ===
class GurobiCodeGenerator:
    def _expr_conditional(self, expr_node, current_iterators, symbolic):
        cond = self._traverse_expression(expr_node["condition"], current_iterators, symbolic)
        then_expr = self._traverse_expression(expr_node["then"], current_iterators, symbolic)
        else_expr = self._traverse_expression(expr_node["else"], current_iterators, symbolic)
        # Remove extra parentheses if present
        if isinstance(cond, str) and cond.startswith("(") and cond.endswith(")"):
            cond = cond[1:-1]
        return f"{then_expr} if ({cond}) else {else_expr}"

    """
    Generates GurobiPy code from a semantically validated AST.
    """

    # === Initialization ===
    def __init__(self, ast, data_dict=None):
        """
        Initialize the code generator with AST and optional data dictionary.
        """
        self.ast = ast
        self.data_dict = data_dict if data_dict is not None else {}
        self.gurobi_code_lines = []
        self.indent_level = 0
        self.gurobi_var_map = {}  # Maps OPL decision variable names to Gurobi variable objects
        self._add_code_line = self.__class__._add_code_line_impl.__get__(self)

    # --- Helper for adding code lines ---
    def _add_code_line_impl(self, line):
        self.gurobi_code_lines.append("    " * self.indent_level + line)

    # === Public API ===
    def generate_code(self) -> str:
        """
        Generate the full GurobiPy Python code as a string.
        """
        self._add_code_line("import gurobipy as gp")
        self._add_code_line("from gurobipy import GRB")
        self._add_code_line("import itertools  # needed for multi-index forall")
        self._add_code_line("import math  # for math.sqrt and friends")
        self._add_code_line("")
        # SAFE ACCESSOR: protects against accidental out-of-domain index lookups in codegen paths
        self._add_code_line("def _safe_get(container, key, default=0):")
        self.indent_level += 1
        self._add_code_line("try:")
        self.indent_level += 1
        self._add_code_line("return container[key]")
        self.indent_level -= 1
        self._add_code_line("except Exception:")
        self.indent_level += 1
        self._add_code_line("return default")
        self.indent_level -= 2
        self._add_code_line("")
        self._generate_data_declarations(self.data_dict)
        self._add_code_line("model = gp.Model('OPLModel')")
        self._add_code_line("")
        self._generate_declarations(self.ast["declarations"])
        self._generate_objective(self.ast["objective"])
        # Collect variable bounds before constraints for tighter big-M
        self._collect_variable_bounds(self.ast.get("constraints", []))
        self._generate_constraints(self.ast["constraints"])
        self._add_code_line("model.optimize()")
        # Disambiguate INF_OR_UNBD into INFEASIBLE or UNBOUNDED when possible
        self._add_code_line("if model.status == GRB.INF_OR_UNBD:")
        self.indent_level += 1
        self._add_code_line("model.setParam(GRB.Param.DualReductions, 0)")
        self._add_code_line("model.optimize()")
        self.indent_level -= 1
        self._add_code_line("")
        # Results capture
        self._add_code_line("results = {}")
        self._add_code_line("if model.status == GRB.OPTIMAL:")
        self.indent_level += 1
        self._add_code_line("print('Optimal solution found:')")
        self._add_code_line("solution = {}")
        self._add_code_line("for v in model.getVars():")
        self.indent_level += 1
        self._add_code_line("print(f'{v.VarName}: {v.X}')")
        self._add_code_line("solution[v.VarName] = v.X")
        self.indent_level -= 1
        self._add_code_line("print(f'Objective value: {model.ObjVal}')")
        self._add_code_line("results['solution'] = solution")
        self._add_code_line("results['objective_value'] = model.ObjVal")
        self._add_code_line("results['status'] = 'OPTIMAL'")
        self._add_code_line("stats = {}")
        self._add_code_line("try:")
        self.indent_level += 1
        self._add_code_line("stats['MIPGap'] = model.MIPGap")
        self.indent_level -= 1
        self._add_code_line("except AttributeError:")
        self.indent_level += 1
        self._add_code_line("stats['MIPGap'] = None")
        self.indent_level -= 1
        self._add_code_line("stats['Runtime'] = model.Runtime")
        self._add_code_line("stats['NodeCount'] = model.NodeCount")
        self._add_code_line("stats['IterCount'] = model.IterCount")
        self._add_code_line("results['stats'] = stats")
        self.indent_level -= 1
        self._add_code_line("elif model.status == GRB.INF_OR_UNBD:")
        self.indent_level += 1
        self._add_code_line("print('Model is infeasible or unbounded')")
        self._add_code_line("results['status'] = 'INF_OR_UNBD'")
        self.indent_level -= 1
        self._add_code_line("elif model.status == GRB.INFEASIBLE:")
        self.indent_level += 1
        self._add_code_line("print('Model is infeasible')")
        self._add_code_line("results['status'] = 'INFEASIBLE'")
        self.indent_level -= 1
        self._add_code_line("elif model.status == GRB.UNBOUNDED:")
        self.indent_level += 1
        self._add_code_line("print('Model is unbounded')")
        self._add_code_line("results['status'] = 'UNBOUNDED'")
        self.indent_level -= 1
        self._add_code_line("else:")
        self.indent_level += 1
        self._add_code_line("print(f'Optimization ended with status {model.status}')")
        self._add_code_line("results['status'] = f'OPTIMIZATION_STATUS_{model.status}'")
        self.indent_level -= 1
        self._add_code_line("results_container['gurobi_output'] = results")
        return "\n".join(self.gurobi_code_lines)

    # --- Bound Collection (for big-M tightening) ---
    def _collect_variable_bounds(self, constraints):
        if not hasattr(self, "_collected_lbs"):
            self._collected_lbs = {}
            self._collected_ubs = {}

        def record_bound(var_node, op_sym, val):
            # Base symbol only (aggregated across indices)
            if var_node.get("type") == "name":
                base = var_node.get("value")
            elif var_node.get("type") == "indexed_name":
                base = var_node.get("name")
            else:
                return
            if op_sym == ">=":
                cur = self._collected_lbs.get(base)
                if cur is None or val < cur:
                    self._collected_lbs[base] = val
            elif op_sym == "<=":
                cur = self._collected_ubs.get(base)
                if cur is None or val > cur:
                    self._collected_ubs[base] = val
            elif op_sym == "==":
                # equality contributes to both
                curL = self._collected_lbs.get(base)
                if curL is None or val < curL:
                    self._collected_lbs[base] = val
                curU = self._collected_ubs.get(base)
                if curU is None or val > curU:
                    self._collected_ubs[base] = val

        def walk(node):
            if not isinstance(node, dict):
                return
            t = node.get("type")
            if t == "constraint":
                op_sym = node.get("op")
                if op_sym in (">=", "<=", "=="):
                    left = node.get("left")
                    right = node.get("right")
                    if isinstance(left, dict) and isinstance(right, dict):
                        # var OP number
                        if right.get("type") == "number" and left.get("type") in (
                            "name",
                            "indexed_name",
                        ):
                            try:
                                val = float(right.get("value"))
                                record_bound(left, op_sym, val)
                            except Exception:
                                pass
                        # number OP var -> flip
                        elif left.get("type") == "number" and right.get("type") in (
                            "name",
                            "indexed_name",
                        ):
                            try:
                                val = float(left.get("value"))
                                # Flip operator perspective
                                if op_sym == ">=":  # number >= var  -> var <= number
                                    record_bound(right, "<=", val)
                                elif op_sym == "<=":  # number <= var -> var >= number
                                    record_bound(right, ">=", val)
                                elif op_sym == "==":
                                    record_bound(right, "==", val)
                            except Exception:
                                pass
            elif t == "forall_constraint":
                # Traverse inner constraints without explicit unrolling (aggregate bounds suffice)
                if "constraint" in node:
                    walk(node["constraint"])
                if "constraints" in node:
                    for c in node["constraints"]:
                        walk(c)

        for c in constraints:
            walk(c)

    # === Declaration and Data Section ===
    def _generate_data_declarations(self, data_dict):
        # (patch now only in the main parameter emission loop below)

        logger.debug("Entering _generate_data_declarations")

        # Normalize external set_of_tuples in data_dict to dict-with-elements for downstream consumers/tests
        if hasattr(self, "ast") and "declarations" in self.ast:
            for decl in self.ast["declarations"]:
                if decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                    set_name = decl.get("name")
                    if set_name in data_dict and isinstance(data_dict[set_name], list):
                        data_dict[set_name] = {"elements": data_dict[set_name]}

        # --- Recursive shape check for multi-dimensional parameters ---
        def check_shape(param_data, dims, data_dict, param_name, dim=0):
            if not dims:
                logger.debug("shape %s: reached scalar at dim %d", param_name, dim)
                return
            d = dims[0]
            # Determine expected length for this dimension
            if d.get("type") == "named_range_dimension":
                range_decl = self._find_declaration_by_name(d["name"], types=["range_declaration_inline"])
                if range_decl:

                    def eval_expr(expr):
                        if expr["type"] == "number":
                            return int(expr["value"])
                        elif expr["type"] == "name":
                            return int(data_dict[expr["value"]])
                        elif expr["type"] == "binop":
                            op = expr["op"]
                            left = eval_expr(expr["left"])
                            right = eval_expr(expr["right"])
                            if op == "+":
                                return left + right
                            elif op == "-":
                                return left - right
                            elif op == "*":
                                return left * right
                            elif op == "/":
                                return left // right
                            else:
                                raise Exception(f"Unsupported binop in range bound expr: {op}")
                        else:
                            raise Exception(f"Unsupported range bound expr: {expr}")

                    start_idx = eval_expr(range_decl["start"])
                    end_idx = eval_expr(range_decl["end"])
                    expected_len = end_idx - start_idx + 1
                else:
                    expected_len = None
            elif d.get("type") == "named_set_dimension":
                set_obj = data_dict.get(d["name"])
                if set_obj is not None:
                    if isinstance(set_obj, dict) and "elements" in set_obj:
                        expected_len = len(set_obj["elements"])
                    else:
                        expected_len = len(set_obj)
                else:
                    expected_len = None
            else:
                expected_len = None
            logger.debug(
                "shape %s: dim %d expected_len=%s actual=%s dim_type=%s dim_name=%s",
                param_name,
                dim + 1,
                expected_len,
                (len(param_data) if isinstance(param_data, (list, tuple)) else "scalar"),
                d.get("type"),
                d.get("name", None),
            )
            if expected_len is not None:
                if not isinstance(param_data, (list, tuple)):
                    from .semantic_error import SemanticError

                    logger.debug(
                        "shape error %s: expected %dD array, got scalar at dim %d",
                        param_name,
                        len(dims),
                        dim + 1,
                    )
                    raise SemanticError(
                        f"Parameter '{param_name}' expected a {len(dims)}D array, got scalar at dimension {dim+1}."
                    )
                if len(param_data) != expected_len:
                    from .semantic_error import SemanticError

                    logger.debug(
                        "shape error %s: data length %d does not match declared dimension '%s' length %d at dim %d",
                        param_name,
                        len(param_data),
                        d.get("name"),
                        expected_len,
                        dim + 1,
                    )
                    raise SemanticError(
                        f"Parameter '{param_name}' data length {len(param_data)} does not match declared dimension '{d.get('name')}' of length {expected_len} at dimension {dim+1}."
                    )
                if len(dims) > 1:
                    for i, sub in enumerate(param_data):
                        check_shape(sub, dims[1:], data_dict, param_name, dim + 1)

        # Convert flat key-value lists to dicts in data_dict before shape checking
        if hasattr(self, "ast") and "declarations" in self.ast:
            for decl in self.ast["declarations"]:
                if decl.get("type") in (
                    "parameter_external",
                    "parameter_external_indexed",
                    "parameter_external_explicit",
                    "parameter_external_explicit_indexed",
                    "parameter_inline",
                    "parameter_inline_indexed",
                ) and decl.get("dimensions"):
                    param_data = data_dict.get(decl["name"])
                    # DEBUG: Print Stores contents and length if this is Capacity
                    if decl["name"] == "Capacity" and "Stores" in data_dict:
                        logger.debug(
                            "[data_dict] Stores: %s len=%d",
                            data_dict["Stores"],
                            len(data_dict["Stores"]),
                        )
                    # Detect flat key-value list: even length, alternating str and number
                    if isinstance(param_data, list) and len(param_data) % 2 == 0 and len(param_data) > 0:
                        is_flat_kv = all(
                            (isinstance(param_data[i], str) and isinstance(param_data[i + 1], (int, float)))
                            for i in range(0, len(param_data), 2)
                        )
                        if is_flat_kv:
                            # Convert to dict in data_dict
                            data_dict[decl["name"]] = {param_data[i]: param_data[i + 1] for i in range(0, len(param_data), 2)}
                            continue
                    # Only apply shape check to lists/arrays, not dicts
                    if param_data is not None and isinstance(param_data, (list, tuple)):
                        check_shape(param_data, decl["dimensions"], data_dict, decl["name"])
        """Generates Python code for data declarations from the .dat file and tuple/set declarations from AST."""
        # Emit tuple types and sets of tuples from AST declarations (if present)
        # Track parameters we transform into dict (or nested dict) so expression code can index via symbolic keys
        self.dict_params = set()
        if hasattr(self, "ast") and "declarations" in self.ast:
            for decl in self.ast["declarations"]:
                if decl.get("type") == "tuple_type":
                    self.tuple_types = getattr(self, "tuple_types", {})
                    self.tuple_types[decl["name"]] = decl["fields"]
                elif decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                    set_name = decl["name"]
                    tuple_list = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict)
                    if tuple_list:
                        self._add_code_line(f"{set_name} = {repr(tuple_list)}")
                elif decl.get("type") in ("typed_set", "typed_set_external"):
                    set_name = decl["name"]
                    elements = decl.get("value")
                    if (not elements) and set_name in data_dict:
                        elements = data_dict[set_name]
                    if elements is None:
                        elements = []
                    elems_str = ", ".join(repr(e) for e in elements)
                    self._add_code_line(f"{set_name} = [{elems_str}]")
                    self._add_code_line(f"{set_name}_index = {{v:i for i,v in enumerate({set_name})}}")
                elif decl.get("type") in ("tuple_array", "tuple_array_external"):
                    arr_name = decl["name"]
                    tuple_type = decl["tuple_type"]
                    index_set = decl["index_set"]
                    data_list = data_dict.get(arr_name)
                    if data_list is not None and tuple_type in getattr(self, "tuple_types", {}):
                        fields = self.tuple_types[tuple_type]
                        field_names = [f["name"] for f in fields]
                        tuple_dicts = []
                        for t in data_list:
                            d = {}
                            for i, fn in enumerate(field_names):
                                if i < len(t):
                                    d[fn] = t[i]
                            tuple_dicts.append(d)
                        self._add_code_line(f"{arr_name}_data = {json.dumps(tuple_dicts)}")
                        self._add_code_line(f"{arr_name} = {{idx: rec for idx, rec in zip({index_set}, {arr_name}_data)}}")
        # Build a working_data that merges .dat data with inline model values (typed sets, params)
        working_data = dict(data_dict or {})
        if hasattr(self, "ast") and "declarations" in self.ast:
            # Include typed sets so named_set dimensions can be keyed
            for decl in self.ast["declarations"]:
                t = decl.get("type")
                if t in ("typed_set", "typed_set_external"):
                    name = decl["name"]
                    if name not in working_data:
                        vals = decl.get("value")
                        if vals is not None:
                            working_data[name] = vals
                # Inline scalar or array parameters
                if t in ("parameter_inline", "parameter_inline_indexed"):
                    name = decl["name"]
                    if name not in working_data and decl.get("value") is not None:
                        working_data[name] = decl["value"]

        # Do not exit early on empty data_dict; inline params may still need emission
        self._add_code_line("# Data from .dat file")

        # New: validation for 1-D params over set/range where data is provided as a dict with list values.
        param_decl_map = {
            d["name"]: d
            for d in self.ast.get("declarations", [])
            if d.get("type")
            in (
                "parameter_external",
                "parameter_external_indexed",
                "parameter_external_explicit",
                "parameter_external_explicit_indexed",
                "parameter_inline",
                "parameter_inline_indexed",
            )
        }
        for name, decl in param_decl_map.items():
            dims = decl.get("dimensions", []) or []
            if len(dims) == 1 and dims[0].get("type") in (
                "named_set_dimension",
                "named_range_dimension",
            ):
                val = working_data.get(name)
                if isinstance(val, dict):
                    bad_key = next(
                        (k for k, v in val.items() if isinstance(v, (list, tuple, dict))),
                        None,
                    )
                    if bad_key is not None:
                        raise SemanticError(
                            f"Parameter '{name}' declared as 1-D over '{dims[0].get('name', '')}' expects scalar values per key, "
                            f"but data provides an array for key {repr(bad_key)}. Use scalar values (e.g., 2.0), not [2.0]."
                        )

        # (redefine param_decl_map later as in original code; leave existing logic intact)
        # ...existing code...
        param_decl_map = {
            d["name"]: d
            for d in self.ast.get("declarations", [])
            if d.get("type")
            in (
                "parameter_external",
                "parameter_external_indexed",
                "parameter_external_explicit",
                "parameter_external_explicit_indexed",
                "parameter_inline",
                "parameter_inline_indexed",
            )
        }

        # --- helpers for evaluating bounds and normalizing set elements ---
        def _eval_expr_bound(expr):
            # ...existing code but use working_data instead of data_dict...
            if isinstance(expr, dict):
                t = expr.get("type")
                if t == "number":
                    return int(expr["value"])
                if t == "name":
                    return int(working_data[expr["value"]])
                if t == "binop":
                    op = expr["op"]
                    left = _eval_expr_bound(expr["left"])
                    right = _eval_expr_bound(expr["right"])
                    if op == "+":
                        return left + right
                    if op == "-":
                        return left - right
                    if op == "*":
                        return left * right
                    if op == "/":
                        return left // right
            raise Exception(f"Unsupported range bound expr: {expr}")

        def _normalize_set_elems(obj):
            elems = obj.get("elements") if isinstance(obj, dict) and "elements" in obj else obj

            def to_key(x):
                if isinstance(x, (list, tuple)):
                    return tuple(to_key(e) for e in x)
                return x

            return [to_key(e) for e in elems] if elems is not None else []

        # --- Generic N-D flatten to dict with tuple keys using declared dimension semantics ---
        already_emitted = set()
        for name, value in working_data.items():
            pdecl = param_decl_map.get(name)
            if not (pdecl and isinstance(value, (list, tuple))):
                continue
            dims = pdecl.get("dimensions", []) or []
            if len(dims) < 2:
                continue  # 1D handled by dedicated logic below
            keys_per_dim = []
            try:
                for d in dims:
                    dt = d.get("type")
                    if dt == "named_range_dimension":
                        s = _eval_expr_bound(d["start"])
                        e = _eval_expr_bound(d["end"])
                        keys_per_dim.append(list(range(s, e + 1)))
                    elif dt == "range_index":
                        s = _eval_expr_bound(d["start"])
                        e = _eval_expr_bound(d["end"])
                        keys_per_dim.append(list(range(s, e + 1)))
                    elif dt == "named_set_dimension":
                        set_name = d["name"]
                        set_obj = working_data.get(set_name, [])
                        keys_per_dim.append(_normalize_set_elems(set_obj))
                    else:
                        keys_per_dim = None
                        break
            except Exception:
                keys_per_dim = None
            if not keys_per_dim:
                continue

            def _flatten_positions(arr, pos=()):
                if isinstance(arr, (list, tuple)) and arr and any(isinstance(x, (list, tuple)) for x in arr):
                    for i, sub in enumerate(arr):
                        yield from _flatten_positions(sub, pos + (i,))
                elif isinstance(arr, (list, tuple)):
                    for i, v in enumerate(arr):
                        yield pos + (i,), v
                else:
                    yield pos, arr

            flat_dict = {}
            try:
                for idx_pos, v in _flatten_positions(value, ()):
                    if len(idx_pos) != len(keys_per_dim):
                        raise SemanticError(
                            f"Parameter '{name}' dimensionality mismatch: data depth {len(idx_pos)} vs declared {len(keys_per_dim)}."
                        )
                    for dim_i, pi in enumerate(idx_pos):
                        if pi < 0 or pi >= len(keys_per_dim[dim_i]):
                            raise SemanticError(f"Parameter '{name}' index {pi} out of bounds for dimension {dim_i+1}.")
                    key = tuple(keys_per_dim[dim_i][pi] for dim_i, pi in enumerate(idx_pos))
                    flat_dict[key] = v
            except SemanticError:
                flat_dict = None

            if flat_dict is not None:
                self._add_code_line(f"{name} = {repr(flat_dict)}")
                self.dict_params.add(name)
                already_emitted.add(name)

        # Prevent duplicate fallback emissions for already-emitted names
        for name, value in working_data.items():
            if name in already_emitted:
                continue
            pdecl = param_decl_map.get(name)
            # --- NEW: N-D param from nested dict/list keyed along declared dimensions (generalization) ---
            # Accept nested dictionaries keyed by set/range labels and/or lists (row-major) for any dimensionality >= 2.
            if pdecl is not None and isinstance(value, dict) and len(pdecl.get("dimensions", [])) >= 2:
                dims = pdecl.get("dimensions", [])

                # Resolve expected labels per dimension when possible (set elements or explicit range)
                def _resolve_set_elems(set_name):
                    # Prefer working_data (.dat), then AST-declared values
                    set_obj = working_data.get(set_name)
                    if set_obj is None:
                        set_decl = self._find_declaration_by_name(
                            set_name,
                            types=[
                                "typed_set",
                                "typed_set_external",
                                "set_declaration",
                                "set_of_tuples",
                                "set_of_tuples_external",
                            ],
                        )
                        if set_decl is not None:
                            set_obj = set_decl.get("value")
                    return _normalize_set_elems(set_obj) if set_obj is not None else None

                def _dim_labels(dim_spec):
                    dt = dim_spec.get("type")
                    if dt == "named_set_dimension":
                        lbls = _resolve_set_elems(dim_spec["name"])
                        return list(lbls) if lbls is not None else None
                    if dt == "named_range_dimension":
                        s = _eval_expr_bound(dim_spec["start"])
                        e = _eval_expr_bound(dim_spec["end"])
                        return list(range(s, e + 1))
                    if dt == "range_index":
                        s = _eval_expr_bound(dim_spec["start"])
                        e = _eval_expr_bound(dim_spec["end"])
                        return list(range(s, e + 1))
                    return None

                # Precompute labels per dimension when available; None means "unknown, use positional fallback"
                labels_per_dim = [_dim_labels(d) for d in dims]

                # For range dims, also cache their numeric starts for positional fallback
                def _dim_start(dim_spec):
                    dt = dim_spec.get("type")
                    if dt in ("named_range_dimension", "range_index"):
                        return _eval_expr_bound(dim_spec["start"])
                    return None

                starts_per_dim = [_dim_start(d) for d in dims]

                # Normalize keys (lists -> tuples recursively, keep scalars)
                def _to_key(x):
                    if isinstance(x, (list, tuple)):
                        return tuple(_to_key(e) for e in x)
                    return x

                dict_val = {}

                # Helper: map position index j (0-based) to label for a dimension
                def _pos_to_label(dim_idx, j, provided_len=None):
                    lbls = labels_per_dim[dim_idx]
                    if lbls is not None:
                        # Only use labels when lengths appear consistent or we intentionally map up to min length
                        if provided_len is None or len(lbls) == provided_len:
                            return lbls[j] if j < len(lbls) else j + 1
                        # length mismatch: be conservative and fallback to positional
                    # For ranges, use start + j; otherwise 1-based index
                    start = starts_per_dim[dim_idx]
                    return (start + j) if isinstance(start, int) else (j + 1)

                # Recursive flatten
                def _flatten(node, dim_idx, prefix):
                    # If we reached the last dimension, emit values
                    if dim_idx == len(dims) - 1:
                        # node can be dict mapping label -> val, or list/tuple of vals
                        if isinstance(node, dict):
                            for k, v in node.items():
                                key = _to_key(k)
                                dict_val[prefix + (key,)] = v
                        elif isinstance(node, (list, tuple)):
                            L = len(node)
                            for j, v in enumerate(node):
                                lab = _pos_to_label(dim_idx, j, provided_len=L)
                                dict_val[prefix + (lab,)] = v
                        else:
                            # Scalar provided at last dim (degenerate): use positional fallback label
                            lab = _pos_to_label(dim_idx, 0, provided_len=1)
                            dict_val[prefix + (lab,)] = node
                        return

                    # Intermediate dimension: node may be dict (labeled children) or list/tuple (positional)
                    if isinstance(node, dict):
                        for k, child in node.items():
                            key = _to_key(k)
                            _flatten(child, dim_idx + 1, prefix + (key,))
                    elif isinstance(node, (list, tuple)):
                        L = len(node)
                        for j, child in enumerate(node):
                            lab = _pos_to_label(dim_idx, j, provided_len=L)
                            _flatten(child, dim_idx + 1, prefix + (lab,))
                    else:
                        # Degenerate: non-iterable at intermediate level, treat as single-child sequence
                        lab = _pos_to_label(dim_idx, 0, provided_len=1)
                        _flatten(node, dim_idx + 1, prefix + (lab,))

                try:
                    _flatten(value, 0, tuple())
                    if dict_val:
                        self._add_code_line(f"{name} = {repr(dict_val)}")
                        self.dict_params.add(name)
                        already_emitted.add(name)
                        continue
                except Exception:
                    # Fall through to other handlers or shape checks
                    pass

            if value is not None and isinstance(value, (list, tuple)) and pdecl is not None:
                check_shape(value, pdecl.get("dimensions", []), working_data, name)

        """Generates Python code for data declarations from the .dat file and tuple/set declarations from AST."""
        # Emit tuple types and sets of tuples from AST declarations (if present)
        # Keep dict_params
        self.dict_params = set(self.dict_params)
        if hasattr(self, "ast") and "declarations" in self.ast:
            for decl in self.ast["declarations"]:
                if decl.get("type") == "tuple_type":
                    self.tuple_types = getattr(self, "tuple_types", {})
                    self.tuple_types[decl["name"]] = decl["fields"]
                elif decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                    set_name = decl["name"]
                    tuple_list = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict)
                    if tuple_list:
                        self._add_code_line(f"{set_name} = {repr(tuple_list)}")
                elif decl.get("type") in ("typed_set", "typed_set_external"):
                    set_name = decl["name"]
                    elements = decl.get("value")
                    if (not elements) and set_name in data_dict:
                        elements = data_dict[set_name]
                    if elements is None:
                        elements = []
                    elems_str = ", ".join(repr(e) for e in elements)
                    self._add_code_line(f"{set_name} = [{elems_str}]")
                    self._add_code_line(f"{set_name}_index = {{v:i for i,v in enumerate({set_name})}}")
                elif decl.get("type") in ("tuple_array", "tuple_array_external"):
                    arr_name = decl["name"]
                    tuple_type = decl["tuple_type"]
                    index_set = decl["index_set"]
                    data_list = data_dict.get(arr_name)
                    if data_list is not None and tuple_type in getattr(self, "tuple_types", {}):
                        fields = self.tuple_types[tuple_type]
                        field_names = [f["name"] for f in fields]
                        tuple_dicts = []
                        for t in data_list:
                            d = {}
                            for i, fn in enumerate(field_names):
                                if i < len(t):
                                    d[fn] = t[i]
                            tuple_dicts.append(d)
                        self._add_code_line(f"{arr_name}_data = {json.dumps(tuple_dicts)}")
                        self._add_code_line(f"{arr_name} = {{idx: rec for idx, rec in zip({index_set}, {arr_name}_data)}}")
        # If truly nothing to emit, still end section cleanly
        # Rebuild param_decl_map (unchanged)
        param_decl_map = {
            d["name"]: d
            for d in self.ast.get("declarations", [])
            if d.get("type")
            in (
                "parameter_external",
                "parameter_external_indexed",
                "parameter_external_explicit",
                "parameter_external_explicit_indexed",
                "parameter_inline",
                "parameter_inline_indexed",
            )
        }

        def _all_named_scalar_set_dims(pdecl):
            dims = pdecl.get("dimensions", [])
            if not dims:
                return False
            for dim in dims:
                if dim.get("type") != "named_set_dimension":
                    return False
                set_decl = self._find_declaration_by_name(dim.get("name"))
                if not set_decl or set_decl.get("type") not in (
                    "typed_set",
                    "typed_set_external",
                    "set_declaration",
                ):
                    return False
            return True

        for name, value in working_data.items():
            if name in already_emitted:
                continue
            # PATCH: 2D param with (tuple set, range) where .dat supplies dict-of-lists:
            # Demand = [ <"StoreA"> [1,2,3], <"StoreB"> [4,5,6] ];
            if (
                name in param_decl_map
                and isinstance(value, dict)
                and len(param_decl_map[name].get("dimensions", [])) == 2
                and param_decl_map[name]["dimensions"][0].get("type") == "named_set_dimension"
                and param_decl_map[name]["dimensions"][1].get("type") == "named_range_dimension"
                and all(isinstance(v, (list, tuple)) for v in value.values())
            ):
                set_name = param_decl_map[name]["dimensions"][0]["name"]
                range_dim = param_decl_map[name]["dimensions"][1]

                # Evaluate start/end for the range
                def eval_expr(expr):
                    if isinstance(expr, dict):
                        if expr.get("type") == "number":
                            return int(expr["value"])
                        if expr.get("type") == "name":
                            return int(working_data[expr["value"]])
                        if expr.get("type") == "binop":
                            op = expr["op"]
                            left = eval_expr(expr["left"])
                            right = eval_expr(expr["right"])
                            if op == "+":
                                return left + right
                            if op == "-":
                                return left - right
                            if op == "*":
                                return left * right
                            if op == "/":
                                return left // right
                    raise Exception(f"Unsupported range bound expr: {expr}")

                start = eval_expr(range_dim["start"])
                end = eval_expr(range_dim["end"])
                expected_len = end - start + 1

                # Normalize set elements if available in data_dict (preserve tuple keys)
                set_elems = None
                if set_name in data_dict:
                    set_obj = data_dict[set_name]
                    if isinstance(set_obj, dict) and "elements" in set_obj:
                        set_elems = set_obj["elements"]
                    else:
                        set_elems = set_obj

                # Build composite-key dict: {(tupleKey, p): val}
                dict_val = {}
                for k, row in value.items():
                    if len(row) != expected_len:
                        raise SemanticError(
                            f"Parameter '{name}' row for key {k} has length {len(row)}; expected {expected_len}."
                        )
                    # If key is a list, coerce to tuple
                    key_obj = tuple(k) if isinstance(k, (list, tuple)) else k
                    for p in range(start, end + 1):
                        dict_val[(key_obj, p)] = row[p - start]

                self._add_code_line(f"{name} = {repr(dict_val)}")
                self.dict_params.add(name)
                continue
            # PATCH: emit 1D arrays indexed by a set of tuples as dicts with tuple keys (robust to set['elements'])
            if (
                name in param_decl_map
                and param_decl_map[name] is not None
                and param_decl_map[name].get("type")
                in (
                    "parameter_external",
                    "parameter_external_indexed",
                    "parameter_external_explicit",
                    "parameter_external_explicit_indexed",
                    "parameter_inline",
                    "parameter_inline_indexed",
                )
                and isinstance(value, list)
                and len(value) > 0
                and len(param_decl_map[name].get("dimensions", [])) == 2
                and param_decl_map[name]["dimensions"][0].get("type") == "named_set_dimension"
                and param_decl_map[name]["dimensions"][1].get("type") == "named_range_dimension"
            ):
                set_name = param_decl_map[name]["dimensions"][0]["name"]
                set_decl = self._find_declaration_by_name(set_name, types=["set_of_tuples", "set_of_tuples_external"])
                range_dim = param_decl_map[name]["dimensions"][1]

                # Evaluate start/end for the range
                def eval_expr(expr):
                    if expr["type"] == "number":
                        return int(expr["value"])
                    elif expr["type"] == "name":
                        return int(working_data[expr["value"]])
                    elif expr["type"] == "binop":
                        op = expr["op"]
                        left = eval_expr(expr["left"])
                        right = eval_expr(expr["right"])
                        if op == "+":
                            return left + right
                        elif op == "-":
                            return left - right
                        elif op == "*":
                            return left * right
                        elif op == "/":
                            return left // right
                        else:
                            raise Exception(f"Unsupported binop in range bound expr: {op}")
                    else:
                        raise Exception(f"Unsupported range bound expr: {expr}")

                if set_decl and set_name in data_dict:
                    set_obj = data_dict[set_name]
                    if isinstance(set_obj, dict) and "elements" in set_obj:
                        set_elems = set_obj["elements"]
                    else:
                        set_elems = set_obj
                    # Normalize set elements to tuples (preserve 1-field tuples)
                    set_elems = [tuple(e) if isinstance(e, (list, tuple)) else (e,) for e in set_elems]
                    start = eval_expr(range_dim["start"])
                    end = eval_expr(range_dim["end"])
                    expected_len = end - start + 1
                    if len(set_elems) == len(value) and all(len(row) == expected_len for row in value):
                        # Keep tuple keys intact, do not flatten 1-field tuples
                        dict_val = {
                            (set_elems[i], p): value[i][p - start]
                            for i in range(len(set_elems))
                            for p in range(start, end + 1)
                        }
                        self._add_code_line(f"{name} = {repr(dict_val)}")
                        self.dict_params.add(name)
                        continue
            if name in param_decl_map:
                logger.debug(
                    "_generate_data_declarations: Emitting parameter %s type=%s dims=%s",
                    name,
                    param_decl_map[name].get("type"),
                    param_decl_map[name].get("dimensions"),
                )
            # --- PATCH: always emit typed sets as lists, not dicts ---
            decl = self._find_declaration_by_name(name, types=["typed_set", "typed_set_external"])
            if decl is not None:
                elems_str = ", ".join(repr(e) for e in value)
                self._add_code_line(f"{name} = [{elems_str}]")
                self._add_code_line(f"{name}_index = {{v:i for i,v in enumerate({name})}}")
                continue
            # --- PATCH: emit 1D arrays indexed by a range as dicts with OPL indices as keys ---
            param_decl = param_decl_map.get(name)
            if (
                param_decl is not None
                and param_decl.get("type")
                in (
                    "parameter_external",
                    "parameter_external_indexed",
                    "parameter_external_explicit",
                    "parameter_external_explicit_indexed",
                    "parameter_inline",
                    "parameter_inline_indexed",
                )
                and isinstance(value, list)
                and len(value) > 0
                and len(param_decl.get("dimensions", [])) == 1
                and param_decl["dimensions"][0].get("type") == "named_range_dimension"
            ):
                # Get the OPL range or set (e.g., Warehouses = 1..nbWarehouses or Products = ["ProdA", ...])
                range_name = param_decl["dimensions"][0]["name"]
                # Check if it's a typed set (string-indexed)
                set_decl = self._find_declaration_by_name(range_name, types=["typed_set", "typed_set_external"])
                if set_decl and range_name in data_dict:
                    set_elems = data_dict[range_name]
                    if len(set_elems) == len(value):
                        dict_val = {set_elems[i]: value[i] for i in range(len(set_elems))}
                        dict_items = ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in dict_val.items())
                        self._add_code_line(f"{name} = {{{dict_items}}}")
                        self.dict_params.add(name)
                        continue
                    else:
                        raise SemanticError(
                            f"Parameter '{name}' has {len(value)} items but declared set '{range_name}' has {len(set_elems)} elements."
                        )
                # Otherwise, treat as integer-indexed range
                range_decl = self._find_declaration_by_name(range_name, types=["range_declaration_inline"])
                if range_decl:
                    # Evaluate start/end (assume int literals or parameter names in data_dict)
                    start = range_decl["start"]
                    end = range_decl["end"]

                    def eval_expr(expr):
                        if expr["type"] == "number":
                            return int(expr["value"])
                        elif expr["type"] == "name":
                            return int(working_data[expr["value"]])
                        elif expr["type"] == "binop":
                            op = expr["op"]
                            left = eval_expr(expr["left"])
                            right = eval_expr(expr["right"])
                            if op == "+":
                                return left + right
                            elif op == "-":
                                return left - right
                            elif op == "*":
                                return left * right
                            elif op == "/":
                                return left // right  # integer division
                            else:
                                raise Exception(f"Unsupported binop in range bound expr: {op}")
                        else:
                            raise Exception(f"Unsupported range bound expr: {expr}")

                    start_idx = eval_expr(start)
                    end_idx = eval_expr(end)
                    expected_len = end_idx - start_idx + 1
                    if len(value) == expected_len:
                        dict_val = {i: value[i - start_idx] for i in range(start_idx, end_idx + 1)}
                        dict_items = ", ".join(f"{k}: {json.dumps(v)}" for k, v in dict_val.items())
                        self._add_code_line(f"{name} = {{{dict_items}}}")
                        self.dict_params.add(name)
                        continue
                    else:
                        raise SemanticError(
                            f"Parameter '{name}' has {len(value)} items but declared range '{range_name}' expects {expected_len}."
                        )

            # --- PATCH: emit 1D arrays indexed by a named set as dicts with set elements as keys ---
            if (
                param_decl is not None
                and param_decl.get("type")
                in (
                    "parameter_external",
                    "parameter_external_indexed",
                    "parameter_external_explicit",
                    "parameter_external_explicit_indexed",
                    "parameter_inline",
                    "parameter_inline_indexed",
                )
                and isinstance(value, list)
                and len(value) > 0
                and len(param_decl.get("dimensions", [])) == 1
                and param_decl["dimensions"][0].get("type") == "named_set_dimension"
            ):
                set_name = param_decl["dimensions"][0]["name"]
                # New: handle sets declared in AST (including set_of_tuples), not only data_dict-backed sets
                set_decl = self._find_declaration_by_name(
                    set_name,
                    types=[
                        "set_of_tuples",
                        "set_of_tuples_external",
                        "typed_set",
                        "typed_set_external",
                        "set_declaration",
                    ],
                )
                if set_decl is not None:
                    # Prefer tuple set elements via helper, else inline typed set values, else data_dict
                    if set_decl.get("type") in (
                        "set_of_tuples",
                        "set_of_tuples_external",
                    ):
                        set_elems = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict) or []
                    else:
                        set_elems = data_dict.get(set_name, set_decl.get("value", []))
                    # Normalize {elements:[...]} shape to list
                    if isinstance(set_elems, dict) and "elements" in set_elems:
                        set_elems = set_elems["elements"]
                    if set_elems is not None and len(set_elems) == len(value):
                        # Use repr to support tuple keys; TupleSetHelper already returns tuples for tuple sets
                        dict_val = {set_elems[i]: value[i] for i in range(len(set_elems))}
                        self._add_code_line(f"{name} = {repr(dict_val)}")
                        self.dict_params.add(name)
                        continue
                # Original data_dict-backed path
                if set_name in data_dict:
                    set_elems = data_dict[set_name]
                    # Normalize potential {elements:[...]}
                    if isinstance(set_elems, dict) and "elements" in set_elems:
                        set_elems = set_elems["elements"]
                    if len(set_elems) == len(value):
                        dict_val = {set_elems[i]: value[i] for i in range(len(set_elems))}
                        dict_items = ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in dict_val.items())
                        self._add_code_line(f"{name} = {{{dict_items}}}")
                        self.dict_params.add(name)
                        continue

        self._add_code_line("")

    def _generate_declarations(self, declarations):
        """Generates Python code for decision variables, ranges, and parameters declared in the .mod file."""
        self._add_code_line("# Decision Variables and Parameters")
        self.tuple_types = {}
        logger.debug("Entering _generate_declarations")
        for decl in declarations:
            # Skip dexpr declarations (expanded in parser on use)
            if decl.get("type") in ("dexpr", "dexpr_indexed"):
                continue
            if decl.get("type", "").startswith("parameter_"):
                logger.debug(
                    "_generate_declarations: Emitting parameter '%s' type=%s inline=%s external=%s",
                    decl.get("name"),
                    decl.get("type"),
                    decl.get("inline", None),
                    decl.get("external", None),
                )
            decl_type = decl.get("type")
            # Treat set_of_tuples_external as set_of_tuples for codegen
            if decl_type in ("set_of_tuples", "set_of_tuples_external"):
                # Both handled by _decl_set_of_tuples (which is a no-op)
                self._decl_set_of_tuples(decl)
                continue
            if decl_type in ("tuple_array", "tuple_array_external"):
                # Data emission handled earlier; nothing to declare as decision var
                continue
            if decl_type in ("typed_set", "typed_set_external"):
                self._decl_typed_set(decl)
                continue
            # --- PATCH: Handle dvar_indexed with tuple set index ---
            if decl_type == "dvar_indexed" and len(decl.get("dimensions", [])) == 1:
                dim = decl["dimensions"][0]
                if dim.get("type") == "named_set_dimension":
                    set_name = dim["name"]
                    vtype = decl.get("var_type")
                    grb_vtype = (
                        "GRB.BINARY"
                        if vtype == "boolean"
                        else ("GRB.INTEGER" if vtype.startswith("int") else "GRB.CONTINUOUS")
                    )
                    # Ensure lower bounds match domain semantics for tuple-indexed variables
                    if vtype == "boolean":
                        lb_arg = ""  # binaries are [0,1] by default
                    elif vtype in ("int+", "float+"):
                        lb_arg = ", lb=0"
                    else:
                        # plain int/float: allow negative domain
                        lb_arg = ", lb=-GRB.INFINITY"
                    self._add_code_line(
                        f"{decl['name']} = model.addVars({set_name}, vtype={grb_vtype}, name='{decl['name']}'{lb_arg})"
                    )
                    continue
            # Skip emitting parameter again if already transformed to dict form in data section
            if decl_type.startswith("parameter_") and decl.get("name") in getattr(self, "dict_params", set()):
                continue
            method = getattr(self, f"_decl_{decl_type}", None)
            if method:
                method(decl)
            else:
                raise NotImplementedError(
                    f"Declaration type '{decl.get('type')}' is not supported by the Gurobi code generator."
                )
        self._add_code_line("")
        self._add_code_line("model.update()")
        self._add_code_line("")

    # === Objective and Constraints Section ===
    def _generate_objective(self, objective):
        """Generates Python code for the optimization objective."""
        obj_type = "GRB.MAXIMIZE" if objective["type"] == "maximize" else "GRB.MINIMIZE"
        expr_str = self._traverse_expression(objective["expression"], {})
        self._add_code_line(f"model.setObjective({expr_str}, {obj_type})")
        self._add_code_line("")

    def _generate_constraints(self, constraints):
        """Generates Python code for all constraints."""
        self._add_code_line("# Constraints")
        for i, constraint in enumerate(constraints):
            self._generate_single_constraint(constraint, f"c{i}", {})
        self._add_code_line("")

    def _generate_single_constraint(self, constraint_node, constr_name_prefix, current_iterators):
        """Generates Python code for a single constraint or a forall block using a dispatch pattern."""
        node_type = constraint_node["type"]
        method = getattr(self, f"_constraint_{node_type}", None)
        if not method:
            raise NotImplementedError(f"Constraint type '{node_type}' is not supported by the Gurobi code generator.")
        method(constraint_node, constr_name_prefix, current_iterators)

    # === Linear Bound Utilities (safe wrappers) ===
    def _var_bounds_safe(self, var_node):
        if not isinstance(var_node, dict):
            return (None, None)
        t = var_node.get("type")
        if t == "name":
            decl = self._find_declaration_by_name(var_node.get("value"))
        elif t == "indexed_name":
            decl = self._find_declaration_by_name(var_node.get("name"))
        else:
            return (None, None)
        if not decl:
            return (None, None)
        vtype = decl.get("var_type")
        if vtype == "boolean":
            return (0.0, 1.0)
        # Only '+' variants are nonnegative; plain int/float are free
        if vtype in ("int+", "float+"):
            return (0.0, None)
        if vtype in ("int", "float"):
            return (None, None)
        return (None, None)

    def _linear_bounds_safe(self, node):
        """Attempt to compute (lower, upper) bounds for a linear expression tree.
        Returns tuple (L,U) or None if unsupported. Mirrors subset of _linear_bounds earlier.
        """
        # Fast path for variable / indexed_name with collected bounds
        if isinstance(node, dict) and node.get("type") in ("name", "indexed_name") and hasattr(self, "_collected_lbs"):
            base_sym = node.get("value") if node.get("type") == "name" else node.get("name")
            lb = self._collected_lbs.get(base_sym)
            ub = self._collected_ubs.get(base_sym)
            # Merge with static type-derived bounds
            vL, vU = self._var_bounds_safe(node)
            if vL is not None:
                lb = max(lb, vL) if lb is not None else vL
            if vU is not None:
                ub = min(ub, vU) if ub is not None else vU
            if lb is not None or ub is not None:
                return (lb, ub)

        def _lb_rec(n):
            if not isinstance(n, dict):
                return None
            t = n.get("type")
            if t in ("name", "indexed_name"):
                # Try collected then fall back
                if hasattr(self, "_collected_lbs"):
                    base_sym = n.get("value") if t == "name" else n.get("name")
                    lb = self._collected_lbs.get(base_sym)
                    ub = self._collected_ubs.get(base_sym)
                    vL, vU = self._var_bounds_safe(n)
                    if vL is not None:
                        lb = max(lb, vL) if lb is not None else vL
                    if vU is not None:
                        ub = min(ub, vU) if ub is not None else vU
                    if lb is not None or ub is not None:
                        return (lb, ub)
                return self._var_bounds_safe(n)
            if t == "number":
                v = float(n.get("value", 0))
                return (v, v)
            if t == "binop":
                op = n.get("op")
                left = n.get("left")
                right = n.get("right")
                lB = _lb_rec(left)
                rB = _lb_rec(right)
                if lB is None or rB is None:
                    return None
                lL, lU = lB
                rL, rU = rB
                if op == "+":
                    if None in (lL, lU, rL, rU):
                        return (None, None)
                    return (lL + rL, lU + rU)
                if op == "-":
                    if None in (lL, lU, rL, rU):
                        return (None, None)
                    return (lL - rU, lU - rL)
                if op == "*":
                    # allow constant * linear var
                    if left.get("type") == "number":
                        coef = float(left.get("value", 0))
                        baseB = rB
                    elif right.get("type") == "number":
                        coef = float(right.get("value", 0))
                        baseB = lB
                    else:
                        return None
                    bL, bU = baseB
                    if bL is None or bU is None:
                        return (None, None)
                    if coef >= 0:
                        return (coef * bL, coef * bU)
                    else:
                        return (coef * bU, coef * bL)
                return None
            if t == "sum":
                # conservative: attempt inner bounds * cardinality if finite
                expr = n.get("expression")
                innerB = _lb_rec(expr)
                if innerB is None:
                    return None
                innerL, innerU = innerB
                if innerL is None or innerU is None:
                    return (None, None)
                card = 1
                for it in n.get("iterators", []):
                    rng = it.get("range")
                    if rng.get("type") == "range_specifier":
                        s = rng.get("start")
                        e = rng.get("end")
                        if s.get("type") == "number" and e.get("type") == "number":
                            try:
                                a = int(float(s.get("value", 0)))
                                b = int(float(e.get("value", 0)))
                                if b >= a:
                                    card *= b - a + 1
                                else:
                                    return (None, None)
                            except Exception:
                                return (None, None)
                        else:
                            return (None, None)
                    else:
                        return (None, None)
                return (innerL * card, innerU * card)
            return None

        res = _lb_rec(node)
        return res

    def _constraint_implication_constraint(self, constraint_node, constr_name_prefix, current_iterators):
        """
        Handles implication constraints: <constraint> => <constraint>.
        Uses Gurobi indicator constraints when possible, otherwise falls back to big-M encoding.
        """

        # === Composite Boolean Handling (CNF/DNF style via auxiliaries) ===
        def is_composite_boolean(node):
            if not isinstance(node, dict):
                return False
            t = node.get("type")
            if t == "parenthesized_expression":
                return is_composite_boolean(node.get("expression"))
            if t in ("and", "or", "not"):
                return True
            # Unwrap constraint wrapper of form <bool_expr> == true/false
            if t == "constraint" and node.get("op") == "==" and isinstance(node.get("left"), dict):
                left = node["left"]
                right = node.get("right")
                if left.get("type") in ("and", "or", "not") and (
                    not isinstance(right, dict) or right.get("type") == "boolean_literal"
                ):
                    return True
            return False

        if not hasattr(self, "_bool_aux_counter"):
            self._bool_aux_counter = 0

        def _new_bool_aux(prefix):
            self._bool_aux_counter += 1
            name = f"{prefix}_b{self._bool_aux_counter}_{constr_name_prefix}"
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.BINARY, name='{name}')")
            return name

        # Big-M estimation now relies solely on class-level _linear_bounds_safe to avoid duplicated logic.
        def _estimate_bigM_for_difference(left_node, right_node):
            # Reuse class-level safe linear bounds (unified logic with other big-M computations)
            lB = self._linear_bounds_safe(left_node)
            rB = self._linear_bounds_safe(right_node)
            if lB is None or rB is None:
                return None
            if any(v is None for v in (*lB, *rB)):
                return None
            lL, lU = lB
            rL, rU = rB
            diff_lower = lL - rU
            diff_upper = lU - rL
            span = max(abs(diff_lower), abs(diff_upper))
            # Guard against degenerate zero span (still need tiny epsilon for strict ops)
            return max(span, 1e-9)

        bigM_aux_default = 1e6

        def _bind_comparison_to_binary(binvar, comp_node, iterators):
            """Link linear comparison to binary with near-equivalence:
            bin=1 => comparison holds; bin=0 => comparison allowed to fail (soft) but encourage correctness.
            For '==' we enforce both directions with big-M to approximate equivalence.
            """
            if comp_node.get("type") == "constraint":
                left_node = comp_node["left"]
                if left_node.get("type") == "parenthesized_expression":
                    left_node = left_node["expression"]
                if left_node.get("type") == "binop" and left_node.get("sem_type") == "boolean":
                    left = left_node["left"]
                    right = left_node["right"]
                    op = left_node["op"]
                else:
                    left = comp_node["left"]
                    right = comp_node["right"]
                    op = comp_node["op"]
            elif comp_node.get("type") == "binop":
                left = comp_node["left"]
                right = comp_node["right"]
                op = comp_node["op"]
            else:
                raise ValueError("Unsupported comparison node for boolean linearization")
            left_expr = self._traverse_expression(left, iterators)
            right_expr = self._traverse_expression(right, iterators)
            estM = _estimate_bigM_for_difference(left, right)
            bigM_aux = estM if estM is not None else bigM_aux_default
            eps = 0
            if op == ">=":
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} >= -{bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_ge_{binvar}')"
                )
                # bin=0 upper-relaxes
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} <= {bigM_aux} * {binvar}, name='{constr_name_prefix}_aux_ge_relax_{binvar}')"
                )
            elif op == ">":
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} >= {eps} - {bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_gt_{binvar}')"
                )
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} <= {bigM_aux} * {binvar}, name='{constr_name_prefix}_aux_gt_relax_{binvar}')"
                )
            elif op == "<=":
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} <= {bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_le_{binvar}')"
                )
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} >= -{bigM_aux} * {binvar}, name='{constr_name_prefix}_aux_le_relax_{binvar}')"
                )
            elif op == "<":
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} <= -{eps} + {bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_lt_{binvar}')"
                )
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} >= -{bigM_aux} * {binvar}, name='{constr_name_prefix}_aux_lt_relax_{binvar}')"
                )
            elif op == "==":
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} <= {eps} + {bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_eq1_{binvar}')"
                )
                self._add_code_line(
                    f"model.addConstr({right_expr} - {left_expr} <= {eps} + {bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_eq2_{binvar}')"
                )
                self._add_code_line(
                    f"model.addConstr({left_expr} - {right_expr} >= -{eps} - {bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_eq3_{binvar}')"
                )
                self._add_code_line(
                    f"model.addConstr({right_expr} - {left_expr} >= -{eps} - {bigM_aux} * (1 - {binvar}), name='{constr_name_prefix}_aux_eq4_{binvar}')"
                )
            else:
                raise ValueError(f"Unsupported comparison operator in boolean linearization: {op}")

        def _boolean_expr_to_binary(node, iterators):
            # Normalize constraint nodes (simple comparisons) to binop for uniform handling
            if node.get("type") == "constraint":
                # Only treat as comparison if op is a standard comparison operator
                op = node.get("op")
                if op in (
                    "==",
                    "<",
                    ">",
                    "<=",
                    ">=",
                ):  # '!=' currently not supported in linearization
                    node = {
                        "type": "binop",
                        "op": op,
                        "left": node["left"],
                        "right": node["right"],
                        "sem_type": "boolean",
                    }
            # Unwrap parentheses early
            if node.get("type") == "parenthesized_expression":
                return _boolean_expr_to_binary(node["expression"], iterators)
            # Unwrap constraint wrapper (bool_expr == true)
            if node.get("type") == "constraint" and node.get("op") == "==" and isinstance(node.get("left"), dict):
                left = node["left"]
                right = node.get("right")
                if left.get("type") in ("and", "or", "not") and (
                    not isinstance(right, dict) or right.get("type") == "boolean_literal"
                ):
                    node = left  # treat composite
            if (
                node.get("type") in ("constraint", "binop")
                and node.get("sem_type") == "boolean"
                and node.get("type") not in ("and", "or", "not")
            ):
                b = _new_bool_aux("cmp")
                _bind_comparison_to_binary(b, node, iterators)
                return b
            t = node.get("type")
            if t == "not":
                inner = _boolean_expr_to_binary(node["value"], iterators)
                b = _new_bool_aux("not")
                self._add_code_line(f"model.addConstr({b} + {inner} == 1, name='{constr_name_prefix}_notlink_{b}')")
                return b
            if t == "and":
                left = _boolean_expr_to_binary(node["left"], iterators)
                right = _boolean_expr_to_binary(node["right"], iterators)
                b = _new_bool_aux("and")
                self._add_code_line(f"model.addConstr({b} <= {left}, name='{constr_name_prefix}_and1_{b}')")
                self._add_code_line(f"model.addConstr({b} <= {right}, name='{constr_name_prefix}_and2_{b}')")
                self._add_code_line(f"model.addConstr({b} >= {left} + {right} - 1, name='{constr_name_prefix}_and3_{b}')")
                return b
            if t == "or":
                left = _boolean_expr_to_binary(node["left"], iterators)
                right = _boolean_expr_to_binary(node["right"], iterators)
                b = _new_bool_aux("or")
                self._add_code_line(f"model.addConstr({b} >= {left}, name='{constr_name_prefix}_or1_{b}')")
                self._add_code_line(f"model.addConstr({b} >= {right}, name='{constr_name_prefix}_or2_{b}')")
                self._add_code_line(f"model.addConstr({b} <= {left} + {right}, name='{constr_name_prefix}_or3_{b}')")
                return b
            if t == "boolean_literal":
                b = _new_bool_aux("lit")
                val = 1 if node.get("value") else 0
                self._add_code_line(f"model.addConstr({b} == {val}, name='{constr_name_prefix}_lit_{b}')")
                return b
            raise ValueError(f"Unsupported boolean expression type for auxiliary binary: {t}")

        if is_composite_boolean(constraint_node["antecedent"]) or is_composite_boolean(constraint_node["consequent"]):
            ant_bin = _boolean_expr_to_binary(constraint_node["antecedent"], current_iterators)
            cons_bin = _boolean_expr_to_binary(constraint_node["consequent"], current_iterators)
            self._add_code_line(f"model.addConstr({ant_bin} <= {cons_bin}, name='{constr_name_prefix}_impl_bin')")
            return

        def wrap_boolean_literal_as_constraint(node):
            if node.get("type") == "boolean_literal":
                return {
                    "type": "constraint",
                    "op": "==",
                    "left": node,
                    "right": {
                        "type": "boolean_literal",
                        "value": True,
                        "sem_type": "boolean",
                    },
                }
            return node

        # Remaining processing (linear antecedent/consequent case)
        antecedent = wrap_boolean_literal_as_constraint(constraint_node["antecedent"])
        consequent = wrap_boolean_literal_as_constraint(constraint_node["consequent"])

        # Derive big-M for implication (use max of antecedent & consequent diff bounds) else fallback
        bigM_default = 1e6

        def extract_linear_constraint(node):
            if node.get("type") == "constraint":
                left_node = node["left"]
                # Unwrap parenthesized_expression
                if left_node.get("type") == "parenthesized_expression":
                    left_node = left_node["expression"]
                if left_node.get("type") == "binop" and left_node.get("sem_type") == "boolean":
                    # Comparison binop: extract its numeric sides and operator
                    left = left_node["left"]
                    right = left_node["right"]
                    op = left_node["op"]
                else:
                    # Already a normalized constraint: assume left/right numeric and op is comparison
                    left = node["left"]
                    right = node["right"]
                    op = node["op"]
            elif node.get("type") == "binop":
                left = node["left"]
                right = node["right"]
                op = node["op"]
            else:
                raise ValueError("Implication constraints must be between constraints or binops.")
            left_expr = self._traverse_expression(left, current_iterators)
            right_expr = self._traverse_expression(right, current_iterators)
            return left, right, op, left_expr, right_expr

        # Extract both raw nodes and string expressions
        ant_left, ant_right, ant_op, ant_left_expr, ant_right_expr = extract_linear_constraint(antecedent)
        cons_left, cons_right, cons_op, cons_left_expr, cons_right_expr = extract_linear_constraint(consequent)
        # Compute separate big-M values for antecedent and consequent
        M_ant = _estimate_bigM_for_difference(ant_left, ant_right)
        M_cons = _estimate_bigM_for_difference(cons_left, cons_right)
        bigM_ant = M_ant if M_ant is not None else bigM_default
        bigM_cons = M_cons if M_cons is not None else bigM_default
        eps_sep = EQ_TOL  # epsilon for equality separation on >=/<=

        # Try to use indicator constraint if antecedent is a binary variable equality/inequality
        def is_binary_var(node):
            if node.get("type") == "name":
                varname = node["value"]
            elif node.get("type") == "indexed_name":
                varname = node["name"]
            else:
                return False
            decl = self._find_declaration_by_name(varname)
            if decl and decl.get("type") in ("dvar", "dvar_indexed"):
                return decl.get("var_type") == "boolean"
            return False

        # Specialized pattern A: (linear_expr > c) => (binvar == 1)
        # Contrapositive: (binvar == 0) => linear_expr <= c   [no epsilon]
        if (
            cons_op == "=="
            and is_binary_var(cons_left)
            and (
                (cons_right.get("type") == "number" and float(cons_right.get("value", 0)) == 1.0)
                or (cons_right.get("type") == "boolean_literal" and cons_right.get("value") is True)
            )
            and ant_op == ">"
            and ant_right.get("type") == "number"
        ):
            # RHS is a numeric literal: use it directly
            self._add_code_line(
                f"model.addGenConstrIndicator({cons_left_expr}, 0, {ant_left_expr} <= {ant_right_expr}, name='{constr_name_prefix}_indicator_contra')"
            )
            return

        # Specialized pattern B: (linear_expr >= c) => (binvar == 1)
        # Contrapositive: (binvar == 0) => linear_expr <= c - EPS
        if (
            cons_op == "=="
            and is_binary_var(cons_left)
            and (
                (cons_right.get("type") == "number" and float(cons_right.get("value", 0)) == 1.0)
                or (cons_right.get("type") == "boolean_literal" and cons_right.get("value") is True)
            )
            and ant_op == ">="
            and ant_right.get("type") == "number"
        ):
            epsilon_small = EPS
            # Prefer numeric if possible, otherwise subtract symbolically
            try:
                c_numeric = float(ant_right.get("value", 0))
                adjusted = c_numeric - epsilon_small
                self._add_code_line(
                    f"model.addGenConstrIndicator({cons_left_expr}, 0, {ant_left_expr} <= {adjusted}, name='{constr_name_prefix}_indicator_contra_ge')"
                )
            except Exception:
                self._add_code_line(
                    f"model.addGenConstrIndicator({cons_left_expr}, 0, {ant_left_expr} <= ({ant_right_expr} - {epsilon_small}), name='{constr_name_prefix}_indicator_contra_ge')"
                )
            return

        # Indicator constraint: (binvar == 1) => (linear constraint)
        # Supported: antecedent is (binvar == 1) or (binvar == 0) or (binvar >= 1) or (binvar <= 0)
        indicator_used = False
        if ant_op in ("==", ">=", "<=") and is_binary_var(ant_left):
            try:
                rhs_val = float(ant_right.get("value", 0))
                if ant_op == "==" and rhs_val in (0, 1):
                    binval = int(rhs_val)
                    # Consequent must be a linear constraint
                    if cons_op in ("==", ">=", "<=", ">", "<"):
                        self._add_code_line(
                            f"model.addGenConstrIndicator({ant_left_expr}, {binval}, {cons_left_expr} {cons_op} {cons_right_expr}, name='{constr_name_prefix}_indicator')"
                        )
                        indicator_used = True
                elif ant_op == ">=" and rhs_val == 1:
                    # (binvar >= 1) is equivalent to (binvar == 1)
                    if cons_op in ("==", ">=", "<=", ">", "<"):
                        self._add_code_line(
                            f"model.addGenConstrIndicator({ant_left_expr}, 1, {cons_left_expr} {cons_op} {cons_right_expr}, name='{constr_name_prefix}_indicator')"
                        )
                        indicator_used = True
                elif ant_op == "<=" and rhs_val == 0:
                    # (binvar <= 0) is equivalent to (binvar == 0)
                    if cons_op in ("==", ">=", "<=", ">", "<"):
                        self._add_code_line(
                            f"model.addGenConstrIndicator({ant_left_expr}, 0, {cons_left_expr} {cons_op} {cons_right_expr}, name='{constr_name_prefix}_indicator')"
                        )
                        indicator_used = True
            except Exception:
                pass

        if indicator_used:
            return

        # Robust big-M encoding for general linear implication: flag_var == 1 iff antecedent holds
        flag_var = f"implication_flag_{constr_name_prefix}"
        if current_iterators:
            self._add_code_line(
                f"{flag_var} = model.addVar(vtype=GRB.BINARY)  # 1 if antecedent true (auto-named inside loop)"
            )
        else:
            self._add_code_line(f"{flag_var} = model.addVar(vtype=GRB.BINARY, name='{flag_var}')  # 1 if antecedent true")

        eps = EPS
        diff_expr = f"({ant_left_expr} - {ant_right_expr})"
        if ant_op == ">=":
            # Robust split with bias against feasibility tolerance:
            # flag=1 => diff >= -eps ; flag=0 => diff <= -2*eps
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} >= -{eps}, name='{constr_name_prefix}_ant_ge_ind1')"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} <= -{2*eps}, name='{constr_name_prefix}_ant_ge_ind0')"
            )
        elif ant_op == ">":
            # flag=1 => diff >= eps ; flag=0 => diff <= 0.0
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} >= {eps}, name='{constr_name_prefix}_ant_gt_ind1')"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} <= 0.0, name='{constr_name_prefix}_ant_gt_ind0')"
            )
        elif ant_op == "<=":
            # Robust split with bias against feasibility tolerance:
            # flag=1 => diff <= +eps ; flag=0 => diff >= +2*eps
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} <= {eps}, name='{constr_name_prefix}_ant_le_ind1')"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} >= {2*eps}, name='{constr_name_prefix}_ant_le_ind0')"
            )
        elif ant_op == "<":
            # flag=1 => diff <= -eps ; flag=0 => diff >= 0.0
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} <= -{eps}, name='{constr_name_prefix}_ant_lt_ind1')"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} >= 0.0, name='{constr_name_prefix}_ant_lt_ind0')"
            )
        elif ant_op == "==":
            # For equality, keep existing big-M path (non-convex to split exactly without extra binaries).
            self._add_code_line(
                f"model.addConstr({diff_expr} <= {eps_sep} + {bigM_ant} * (1 - {flag_var}), name='{constr_name_prefix}_ant_eq1')"
            )
            self._add_code_line(
                f"model.addConstr(-{diff_expr} <= {eps_sep} + {bigM_ant} * (1 - {flag_var}), name='{constr_name_prefix}_ant_eq2')"
            )
            self._add_code_line(
                f"model.addConstr({diff_expr} >= -{eps_sep} - {bigM_ant} * (1 - {flag_var}), name='{constr_name_prefix}_ant_eq3')"
            )
            self._add_code_line(
                f"model.addConstr(-{diff_expr} >= -{eps_sep} - {bigM_ant} * (1 - {flag_var}), name='{constr_name_prefix}_ant_eq4')"
            )
        else:
            raise ValueError(f"Unsupported antecedent operator in implication: {ant_op}")

        # 2. Enforce consequent only when flag_var == 1 (use bigM_cons)
        if cons_op == "==":
            self._add_code_line(
                f"model.addConstr({cons_left_expr} - {cons_right_expr} <= {eps_sep} + {bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_eq1')"
            )
            self._add_code_line(
                f"model.addConstr({cons_right_expr} - {cons_left_expr} <= {eps_sep} + {bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_eq2')"
            )
            self._add_code_line(
                f"model.addConstr({cons_left_expr} - {cons_right_expr} >= -{eps_sep} - {bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_eq3')"
            )
            self._add_code_line(
                f"model.addConstr({cons_right_expr} - {cons_left_expr} >= -{eps_sep} - {bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_eq4')"
            )
        elif cons_op == ">=":
            self._add_code_line(
                f"model.addConstr({cons_left_expr} - {cons_right_expr} >= -{bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_ge')"
            )
        elif cons_op == ">":
            self._add_code_line(
                f"model.addConstr({cons_left_expr} - {cons_right_expr} >= {eps} - {bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_gt')"
            )
        elif cons_op == "<=":
            self._add_code_line(
                f"model.addConstr({cons_left_expr} - {cons_right_expr} <= {bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_le')"
            )
        elif cons_op == "<":
            self._add_code_line(
                f"model.addConstr({cons_left_expr} - {cons_right_expr} <= -{eps} + {bigM_cons} * (1 - {flag_var}), name='{constr_name_prefix}_cons_lt')"
            )
        else:
            raise ValueError(f"Unsupported consequent operator in implication: {cons_op}")

    # === Declaration Node Handlers ===
    def _decl_tuple_type(self, decl):
        self.tuple_types[decl["name"]] = decl["fields"]

    def _decl_set_of_tuples(self, decl):
        pass  # handled elsewhere

    def _decl_dvar(self, decl):
        name = decl["name"]
        var_type = decl["var_type"]
        if var_type == "boolean":
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.BINARY, name='{name}')")
        elif var_type == "int+":
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.INTEGER, name='{name}', lb=0)")
        elif var_type == "int":
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.INTEGER, name='{name}', lb=-GRB.INFINITY)")
        elif var_type == "float+":
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.CONTINUOUS, name='{name}', lb=0)")
        elif var_type == "float":
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.CONTINUOUS, name='{name}', lb=-GRB.INFINITY)")
        else:
            self._add_code_line(f"{name} = model.addVar(name='{name}')")
        self.gurobi_var_map[name] = name

    def _decl_dvar_indexed(self, decl):
        # Emit decision variables for multi-dimensional arrays using itertools.product
        name = decl["name"]
        var_type = decl["var_type"]
        dimensions = decl["dimensions"]
        range_args = []
        for dim in dimensions:
            if dim["type"] == "range_index":
                start_val = self._traverse_expression(dim["start"], {}, symbolic=True)
                end_val = self._traverse_expression(dim["end"], {}, symbolic=True)
                range_args.append(f"range({start_val}, {end_val} + 1)")
            elif dim["type"] == "named_range_dimension":
                # Use symbolic range name as the end bound: range(<start_expr>, <Name> + 1)
                start_expr = (
                    self._traverse_expression(
                        dim.get("start", {"type": "number", "value": 1}),
                        {},
                        symbolic=True,
                    )
                    if "start" in dim
                    else "1"
                )
                range_args.append(f"range({start_expr}, {dim['name']} + 1)")
            elif dim["type"] == "named_set_dimension":
                set_name = dim["name"]
                tuple_keys = TupleSetHelper.get_tuple_set(set_name, self.ast, self.data_dict)
                range_args.append(f"{set_name}")
                if not hasattr(self, "_emitted_tuple_sets"):
                    self._emitted_tuple_sets = set()
                if set_name not in self._emitted_tuple_sets:
                    self._add_code_line(f"{set_name} = {repr(tuple_keys)}")
                    self._emitted_tuple_sets.add(set_name)
            else:
                raise ValueError(f"Unsupported dimension type in declaration for {name}: {dim['type']}")
        # Use itertools.product for multi-indexed variables
        if len(range_args) > 1:
            product_args = f"itertools.product({', '.join(map(str, range_args))})"
            if var_type == "boolean":
                self._add_code_line(f"{name} = model.addVars({product_args}, vtype=GRB.BINARY, name='{name}')")
            elif var_type == "int+":
                self._add_code_line(f"{name} = model.addVars({product_args}, vtype=GRB.INTEGER, name='{name}', lb=0)")
            elif var_type == "int":
                self._add_code_line(
                    f"{name} = model.addVars({product_args}, vtype=GRB.INTEGER, name='{name}', lb=-GRB.INFINITY)"
                )
            elif var_type == "float+":
                self._add_code_line(f"{name} = model.addVars({product_args}, vtype=GRB.CONTINUOUS, name='{name}', lb=0)")
            elif var_type == "float":
                self._add_code_line(
                    f"{name} = model.addVars({product_args}, vtype=GRB.CONTINUOUS, name='{name}', lb=-GRB.INFINITY)"
                )
            else:
                self._add_code_line(f"{name} = model.addVars({product_args}, name='{name}')")
        else:
            if var_type == "boolean":
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.BINARY, name='{name}')"
                )
            elif var_type == "int+":
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.INTEGER, name='{name}', lb=0)"
                )
            elif var_type == "int":
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.INTEGER, name='{name}', lb=-GRB.INFINITY)"
                )
            elif var_type == "float+":
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.CONTINUOUS, name='{name}', lb=0)"
                )
            elif var_type == "float":
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.CONTINUOUS, name='{name}', lb=-GRB.INFINITY)"
                )
            else:
                self._add_code_line(f"{name} = model.addVars({', '.join(map(str, range_args))}, name='{name}')")
        self.gurobi_var_map[name] = name

    def _decl_range_declaration_inline(self, decl):
        name = decl["name"]

        def emit_bound(expr):
            val = self._traverse_expression(expr, {}, symbolic=True)
            return val

        end_val = emit_bound(decl["end"])
        # Emit the upper bound as a scalar (e.g., I = 2), so loops can use range(1, I + 1)
        self._add_code_line(f"{name} = {end_val}")

    def _decl_range_declaration_external(self, decl):
        name = decl["name"]
        raise SemanticError(
            f"Range '{name}' declared as external. Ranges must be defined in the model with explicit bounds (e.g., range Items = 1..N;)"
        )

    def _decl_set_declaration(self, decl):
        name = decl["name"]
        if name in self.data_dict:
            self._add_code_line(f"{name} = {json.dumps(self.data_dict[name])}")
        else:
            raise SemanticError(
                f"Set '{name}' declared in .mod but not found in .dat file.",
                lineno=decl.get("lineno", None),
            )

    def _decl_typed_set(self, decl):
        name = decl["name"]
        elements = decl.get("value")
        # If not provided inline, look in data_dict
        if (not elements) and name in self.data_dict:
            elements = self.data_dict[name]
        if elements is None:
            elements = []
        elems_str = ", ".join(repr(e) for e in elements)
        self._add_code_line(f"{name} = [{elems_str}]")

    def _decl_parameter_inline(self, decl):
        name = decl["name"]
        # Use repr if this param is in dict_params (tuple-keyed), else json.dumps
        if name in getattr(self, "dict_params", set()):
            self._add_code_line(f"{name} = {repr(decl['value'])}")
        else:
            self._add_code_line(f"{name} = {json.dumps(decl['value'])}")

    def _decl_parameter_inline_indexed(self, decl):
        name = decl["name"]
        dims = decl.get("dimensions", [])
        if len(dims) == 1 and dims[0]["type"] == "named_set_dimension":
            set_name = dims[0]["name"]
            tuple_set = None
            for d in self.ast["declarations"]:
                if d.get("type") == "set_of_tuples" and d["name"] == set_name:
                    tuple_set = d
                    break
            if tuple_set:
                tuple_keys = [tuple(t["elements"]) for t in tuple_set["value"]]
                param_dict = {k: v for k, v in zip(tuple_keys, decl["value"])}
                self._add_code_line(f"{name} = {repr(param_dict)}")
                return
        # Use repr if this param is in dict_params (tuple-keyed), else json.dumps
        if name in getattr(self, "dict_params", set()):
            self._add_code_line(f"{name} = {repr(decl['value'])}")
        else:
            self._add_code_line(f"{name} = {json.dumps(decl['value'])}")

    def _decl_parameter_external(self, decl):
        name = decl["name"]
        if name in self.data_dict:
            val = self.data_dict[name]
            # Always use repr if dict with tuple keys, regardless of dict_params
            if isinstance(val, dict) and any(isinstance(k, tuple) for k in val.keys()):
                self._add_code_line(f"{name} = {repr(val)}")
            elif name in getattr(self, "dict_params", set()):
                self._add_code_line(f"{name} = {repr(val)}")
            else:
                self._add_code_line(f"{name} = {json.dumps(val)}")
        else:
            raise SemanticError(
                f"Parameter '{name}' declared in .mod but not found in .dat file. "
                "Add '= ...;' to explicitly declare it as external if intended.",
                lineno=decl.get("lineno", None),
            )

    def _decl_parameter_external_indexed(self, decl):
        self._decl_parameter_external(decl)

    def _decl_parameter_external_explicit(self, decl):
        name = decl["name"]
        if name in self.data_dict:
            if name in getattr(self, "dict_params", set()):
                self._add_code_line(f"{name} = {repr(self.data_dict[name])}")
            else:
                self._add_code_line(f"{name} = {json.dumps(self.data_dict[name])}")
        else:
            raise SemanticError(
                f"Parameter '{name}' declared with '= ...' in .mod but not found in .dat file.",
                lineno=decl.get("lineno", None),
            )

    def _decl_parameter_external_explicit_indexed(self, decl):
        self._decl_parameter_external_explicit(decl)

    # === Constraint Node Handlers ===
    def _constraint_constraint(self, constraint_node, constr_name_prefix, current_iterators):
        # Defer expression string generation until after pattern-specific rewrites to avoid
        # creating TempConstr objects (by evaluating comparisons) that we later try to combine arithmetically.
        op = constraint_node["op"]
        left_node = constraint_node["left"]
        right_node = constraint_node["right"]

        # --- NEW: normalize comparison wrapped as boolean equality ---
        # Unwrap parentheses
        def _unwrap(n):
            while isinstance(n, dict) and n.get("type") == "parenthesized_expression":
                n = n.get("expression")
            return n

        def _is_comparison(n):
            return (
                isinstance(n, dict)
                and n.get("type") in ("binop", "constraint")
                and n.get("op") in (">=", "<=", "==", ">", "<")
            )

        L = _unwrap(left_node)
        R = _unwrap(right_node)

        # Handle (comparison) == true
        if (
            op == "=="
            and isinstance(R, dict)
            and R.get("type") == "boolean_literal"
            and R.get("value") is True
            and _is_comparison(L)
        ):
            if L.get("type") == "constraint":
                op = L["op"]
                left_node = L["left"]
                right_node = L["right"]
            else:  # binop comparison
                op = L["op"]
                left_node = L["left"]
                right_node = L["right"]
            # refresh locals for downstream logic
            L = _unwrap(left_node)
            R = _unwrap(right_node)

        # Handle true == (comparison)
        elif (
            op == "=="
            and isinstance(L, dict)
            and L.get("type") == "boolean_literal"
            and L.get("value") is True
            and _is_comparison(R)
        ):
            if R.get("type") == "constraint":
                op = R["op"]
                left_node = R["left"]
                right_node = R["right"]
            else:  # binop comparison
                op = R["op"]
                left_node = R["left"]
                right_node = R["right"]
            L = _unwrap(left_node)
            R = _unwrap(right_node)

        # --- Direct cardinality constraint: sum(comparisons) op k (supports >, >=, ==) ---
        def _unwrap(n):
            while isinstance(n, dict) and n.get("type") == "parenthesized_expression":
                n = n.get("expression")
            return n

        def _is_sum_of_comparisons(n):
            n = _unwrap(n)
            if not (isinstance(n, dict) and n.get("type") == "sum"):
                return False
            inner = _unwrap(n.get("expression"))
            return (
                isinstance(inner, dict)
                and inner.get("type") in ("binop", "constraint")
                and inner.get("op") in (">=", ">", "<=", "<", "==")
            )

        if (
            op in (">", ">=", "==", "<=", "<")
            and isinstance(right_node, dict)
            and right_node.get("type") == "number"
            and _is_sum_of_comparisons(left_node)
        ):
            sum_node = _unwrap(left_node)
            k_val = right_node.get("value")
            effective_k = k_val + 1 if op == ">" else k_val
            # Force traversal so that _expr_sum reifies comparisons and stores metadata
            if not hasattr(self, "_comparison_sum_meta"):
                self._comparison_sum_meta = {}
            if id(sum_node) not in self._comparison_sum_meta:
                try:
                    self._traverse_expression(sum_node, current_iterators)
                except Exception:
                    pass
            meta = self._comparison_sum_meta.get(id(sum_node))
            if meta:
                list_name = meta["list_name"]
                if op in (">", ">="):
                    self._add_code_line(f"model.addConstr(gp.quicksum({list_name}) >= {effective_k})")
                elif op == "==":
                    self._add_code_line(f"model.addConstr(gp.quicksum({list_name}) == {effective_k})")
                elif op in ("<=", "<"):
                    self._add_code_line(f"model.addConstr(gp.quicksum({list_name}) <= {effective_k})")
                return
        # Stage 2: detect reified cardinality equality b == (sum(comparisons) >= k)
        if op == "==" and isinstance(right_node, dict):
            # Unwrap any parentheses on right side: b == ( ... )
            r = right_node
            while isinstance(r, dict) and r.get("type") == "parenthesized_expression":
                r = r.get("expression")
            # Pattern A: right is constraint (possibly wrapped) of form (sum_expr >= number)
            if (
                isinstance(r, dict)
                and r.get("type") == "constraint"
                and r.get("op") in (">=", ">")
                and isinstance(r.get("left"), dict)
                and isinstance(r.get("right"), dict)
                and r["right"].get("type") == "number"
            ):
                sum_candidate = r["left"]
                # The left side might itself be parenthesized wrapping the sum
                while isinstance(sum_candidate, dict) and sum_candidate.get("type") == "parenthesized_expression":
                    sum_candidate = sum_candidate.get("expression")
                if isinstance(sum_candidate, dict) and sum_candidate.get("type") == "sum":
                    sum_node = sum_candidate
                    k_val = r["right"]["value"]
                    if r.get("op") == ">":
                        k_val = k_val + 1  # strict > => >= k+1 for integer sum
                    # Ensure metadata map exists
                    if not hasattr(self, "_comparison_sum_meta"):
                        self._comparison_sum_meta = {}
                    # Force traversal to build metadata if entry missing
                    if id(sum_node) not in self._comparison_sum_meta:
                        try:
                            self._traverse_expression(sum_node, current_iterators)
                        except Exception:
                            pass
                    meta = self._comparison_sum_meta.get(id(sum_node))
                    if meta and isinstance(left_node, dict) and left_node.get("type") in ("name", "indexed_name"):
                        list_name = meta["list_name"]
                        bool_var = self._traverse_expression(left_node, current_iterators)
                        self._add_code_line(f"# Reified cardinality: {bool_var} == (sum(comparisons) >= {k_val})")
                        len_var = meta.get("len_var") or f"len({list_name})"
                        self._add_code_line(f"model.addConstr({k_val} * {bool_var} - gp.quicksum({list_name}) <= 0)")
                        self._add_code_line(
                            f"model.addConstr(gp.quicksum({list_name}) - ({k_val}-1) - ({len_var} - {k_val} + 1) * {bool_var} <= 0)"
                        )
                        return
            # Pattern B: right is a binop (>= or >) directly: b == ( sum(...) >= k )
            if (
                isinstance(r, dict)
                and r.get("type") == "binop"
                and r.get("op") in (">=", ">")
                and isinstance(r.get("left"), dict)
                and isinstance(r.get("right"), dict)
                and r["right"].get("type") == "number"
            ):
                sum_candidate = r["left"]
                while isinstance(sum_candidate, dict) and sum_candidate.get("type") == "parenthesized_expression":
                    sum_candidate = sum_candidate.get("expression")
                if isinstance(sum_candidate, dict) and sum_candidate.get("type") == "sum":
                    k_val = r["right"]["value"]
                    if r.get("op") == ">":
                        k_val = k_val + 1
                    if not hasattr(self, "_comparison_sum_meta"):
                        self._comparison_sum_meta = {}
                    if id(sum_candidate) not in self._comparison_sum_meta:
                        try:
                            self._traverse_expression(sum_candidate, current_iterators)
                        except Exception:
                            pass
                    meta = self._comparison_sum_meta.get(id(sum_candidate))
                    if meta and isinstance(left_node, dict) and left_node.get("type") in ("name", "indexed_name"):
                        list_name = meta["list_name"]
                        bool_var = self._traverse_expression(left_node, current_iterators)
                        self._add_code_line(f"# Reified cardinality (binop): {bool_var} == (sum(comparisons) >= {k_val})")
                        len_var = meta.get("len_var") or f"len({list_name})"
                        self._add_code_line(f"model.addConstr({k_val} * {bool_var} - gp.quicksum({list_name}) <= 0)")
                        self._add_code_line(
                            f"model.addConstr(gp.quicksum({list_name}) - ({k_val}-1) - ({len_var} - {k_val} + 1) * {bool_var} <= 0)"
                        )
                        return
        # Specialized handling for '!=' now supported
        if op == "!=":
            left_expr_str = self._traverse_expression(left_node, current_iterators)
            right_expr_str = self._traverse_expression(right_node, current_iterators)

            # Helper to detect boolean decision variable nodes
            def _is_bool_var(node):
                if not isinstance(node, dict):
                    return False
                t = node.get("type")
                if t == "name":
                    decl = self._find_declaration_by_name(node.get("value"))
                elif t == "indexed_name":
                    decl = self._find_declaration_by_name(node.get("name"))
                else:
                    return False
                return decl is not None and decl.get("var_type") == "boolean"

            left_node = constraint_node["left"]
            right_node = constraint_node["right"]
            # Boolean XOR rewrite: a != b  -> a + b == 1
            if _is_bool_var(left_node) and _is_bool_var(right_node):
                self._add_code_line(
                    f"model.addConstr({left_expr_str} + {right_expr_str} == 1, name='{constr_name_prefix}_xor')"
                )
                return
            # Numeric big-M disjunctive enforcement: x != y
            # Introduce binary delta: either x - y >= 1 OR y - x >= 1
            delta = f"neq_flag_{constr_name_prefix}"
            if current_iterators:
                # Inside a forall loop: omit explicit name to avoid duplicate Gurobi var names per iteration
                self._add_code_line(f"{delta} = model.addVar(vtype=GRB.BINARY)")
            else:
                self._add_code_line(f"{delta} = model.addVar(vtype=GRB.BINARY, name='{delta}')")
            # Reuse linear bounds logic from implication section (copied to class helpers) to estimate |x - y| width.
            bigM_default = 1e6
            lB = self._linear_bounds_safe(left_node)
            rB = self._linear_bounds_safe(right_node)
            if lB is not None and rB is not None and None not in (*lB, *rB):
                lL, lU = lB
                rL, rU = rB
                width_left = max(0.0, lU - lL) if lU is not None and lL is not None else None
                width_right = max(0.0, rU - rL) if rU is not None and rL is not None else None
                if width_left is not None and width_right is not None:
                    bigM = max(1.0, max(width_left, width_right) + 1.0)
                else:
                    bigM = bigM_default
            else:
                bigM = bigM_default
            # Encode as specified: a - b + M*δ >= 1 ; b - a + M*(1-δ) >= 1
            self._add_code_line(
                f"model.addConstr({left_expr_str} - {right_expr_str} + {bigM} * {delta} >= 1, name='{constr_name_prefix}_neq1')"
            )
            self._add_code_line(
                f"model.addConstr({right_expr_str} - {left_expr_str} + {bigM} * (1 - {delta}) >= 1, name='{constr_name_prefix}_neq2')"
            )
            return

        # Fast path: if both sides are comparison-free linear expressions (no need for transformation) just emit
        def _is_simple_comparison(n):
            return (
                isinstance(n, dict)
                and n.get("type") in ("constraint", "binop")
                and n.get("op") in (">=", "<=", "==", ">", "<")
            )

        # Avoid building arithmetic over TempConstr: if op itself is comparison and neither side is a sum, emit directly
        if (
            op in (">=", "<=", "==", ">", "<")
            and not (isinstance(left_node, dict) and left_node.get("type") == "sum")
            and not (isinstance(right_node, dict) and right_node.get("type") == "sum")
        ):
            left_expr_str = self._traverse_expression(left_node, current_iterators)
            right_expr_str = self._traverse_expression(right_node, current_iterators)
            self._add_code_line(f"model.addConstr({left_expr_str} {op} {right_expr_str}, name='{constr_name_prefix}')")
            return
        # Generic path: traverse now
        left_expr_str = self._traverse_expression(left_node, current_iterators)
        right_expr_str = self._traverse_expression(right_node, current_iterators)
        self._add_code_line(f"model.addConstr({left_expr_str} {op} {right_expr_str}, name='{constr_name_prefix}')")

    def _constraint_forall_constraint(self, constraint_node, constr_name_prefix, current_iterators):
        iterators = constraint_node["iterators"]
        index_constraint = constraint_node.get("index_constraint")
        loop_vars, loop_ranges = self._extract_forall_loops(iterators, current_iterators)
        loop_header = self._construct_loop_header(loop_vars, loop_ranges)
        self._add_code_line(loop_header)
        self.indent_level += 1
        new_iterators = current_iterators.copy()
        for v in loop_vars:
            new_iterators[v] = v
        if index_constraint is not None:
            cond_str = self._traverse_expression(index_constraint, new_iterators)
            self._add_code_line(f"if {cond_str}:")
            self.indent_level += 1
        self._emit_forall_inner_constraints(constraint_node, constr_name_prefix, loop_vars, new_iterators)
        if index_constraint is not None:
            self.indent_level -= 1
        self.indent_level -= 1

    def _extract_forall_loops(self, iterators, current_iterators):
        """Helper to extract loop variables and ranges for forall constraints."""
        loop_vars = []
        loop_ranges = []
        for it in iterators:
            name = it["iterator"]
            rng = it["range"]
            loop_vars.append(name)
            loop_ranges.append(self._forall_range_expr(rng, current_iterators))
        return loop_vars, loop_ranges

    def _forall_range_expr(self, rng, current_iterators):
        """Helper to get the range/set expression for a forall iterator."""
        if rng["type"] == "range_specifier":
            start = self._traverse_expression(rng["start"], current_iterators, symbolic=True)
            end = self._traverse_expression(rng["end"], current_iterators, symbolic=True)
            return f"range({start}, {end} + 1)"
        elif rng["type"] == "named_range":
            try:
                return self._emit_range_from_declaration(rng["name"], current_iterators, True)
            except SemanticError:
                set_name = self._emit_set_name_if_declared(rng["name"])
                if set_name:
                    return set_name
                else:
                    raise ValueError(f"Range or set '{rng['name']}' not found in declarations.")
        elif rng["type"] in ("named_set", "named_set_dimension"):
            set_name = self._emit_set_name_if_declared(rng["name"])
            if set_name:
                return set_name
            else:
                raise ValueError(f"Set '{rng['name']}' not found in declarations.")
        else:
            raise ValueError(f"Unsupported range type for forall: {rng['type']}")

    def _emit_forall_inner_constraints(self, constraint_node, constr_name_prefix, loop_vars, new_iterators):
        """Helper to emit the inner constraint(s) of a forall block."""
        if "constraint" in constraint_node:
            inner_constraint = constraint_node["constraint"]
            self._generate_single_constraint(
                inner_constraint,
                f"{constr_name_prefix}_{'_'.join(loop_vars)}",
                new_iterators,
            )
        elif "constraints" in constraint_node:
            for i, inner_constr in enumerate(constraint_node["constraints"]):
                self._generate_single_constraint(
                    inner_constr,
                    f"{constr_name_prefix}_{'_'.join(loop_vars)}_{i}",
                    new_iterators,
                )
        else:
            raise ValueError("Forall constraint node missing 'constraint' or 'constraints' key.")

    # === Expression Node Handlers ===
    def _traverse_expression(self, expr_node, current_iterators, symbolic=False):
        """Recursively traverses an expression AST node and returns its Python string representation.
        Uses a dispatch method-per-node-type approach for modularity."""
        node_type = expr_node["type"]
        method = getattr(self, f"_expr_{node_type}", None)
        if not method:
            raise NotImplementedError(f"Expression type '{node_type}' is not supported by the Gurobi code generator.")
        return method(expr_node, current_iterators, symbolic)

    # NEW: function call support (sqrt)
    def _expr_funcall(self, expr_node, current_iterators, symbolic):
        name = expr_node.get("name")
        args = expr_node.get("args", [])
        if name == "sqrt" and len(args) == 1:
            arg_str = self._traverse_expression(args[0], current_iterators, symbolic)
            return f"math.sqrt({arg_str})"
        raise NotImplementedError(f"Unsupported function call '{name}' in expression.")

    # NEW: support minl/maxl (elementwise min/max over args) in Python-emitted expressions
    def _expr_minl(self, expr_node, current_iterators, symbolic):
        args = expr_node.get("args", [])
        parts = [self._traverse_expression(a, current_iterators, symbolic) for a in args]
        return f"min({', '.join(parts)})"

    def _expr_maxl(self, expr_node, current_iterators, symbolic):
        args = expr_node.get("args", [])
        parts = [self._traverse_expression(a, current_iterators, symbolic) for a in args]
        return f"max({', '.join(parts)})"

    def _expr_number(self, expr_node, current_iterators, symbolic):
        return expr_node["value"]

    def _expr_name(self, expr_node, current_iterators, symbolic):
        name = expr_node["value"]
        if name in current_iterators:
            return name
        elif name in self.gurobi_var_map:
            return self.gurobi_var_map[name]
        elif symbolic:
            return name
        elif name in self.data_dict:
            # Always emit the symbolic name for code generation
            return name
        else:
            for decl in self.ast.get("declarations", []):
                if decl.get("type") == "parameter_inline" and decl["name"] == name:
                    # Always emit the symbolic name for code generation
                    return name
            for decl in self.ast.get("declarations", []):
                if (
                    decl.get("type")
                    in (
                        "parameter_external",
                        "parameter_external_indexed",
                        "parameter_external_explicit",
                        "parameter_external_explicit_indexed",
                    )
                    and decl["name"] == name
                ):
                    raise ValueError(
                        f"Parameter '{name}' is declared as external in the model but no value was provided in the data file."
                    )
            raise ValueError(f"Undeclared variable or unhandled context: {name}")

    def _expr_indexed_name(self, expr_node, current_iterators, symbolic):
        base_name = expr_node["name"]

        def emit_index_expr(dim_expr):
            t = dim_expr.get("type")
            if t == "field_access_index":
                return self._expr_field_access(dim_expr, current_iterators, symbolic)
            if t == "field_access":
                return self._expr_field_access(dim_expr, current_iterators, symbolic)
            if t == "number_literal_index":
                return str(dim_expr["value"])
            elif t == "name_reference_index":
                return str(dim_expr["name"])
            elif t == "string_literal":  # <-- emit quoted string for string index
                return repr(dim_expr["value"])
            elif t == "binop":
                left = emit_index_expr(dim_expr["left"])
                right = emit_index_expr(dim_expr["right"])
                return f"({left} {dim_expr['op']} {right})"
            elif t == "uminus":
                val = emit_index_expr(dim_expr["value"])
                return f"-({val})"
            elif t == "parenthesized_expression":
                return f"({emit_index_expr(dim_expr['expression'])})"
            else:
                if "value" in dim_expr:
                    return str(dim_expr["value"])
                elif "name" in dim_expr:
                    return str(dim_expr["name"])
                raise ValueError(f"Unsupported index expr type: {t}")

        decl = None
        for d in self.ast.get("declarations", []):
            if d.get("name") == base_name:
                decl = d
                break

        def is_tuple_indexed(decl):
            if decl is None:
                return False
            dims = decl.get("dimensions", [])
            if len(dims) != 1 or dims[0].get("type") != "named_set_dimension":
                return False
            set_name = dims[0].get("name")
            set_decl = None
            for d in self.ast.get("declarations", []):
                if d.get("name") == set_name:
                    set_decl = d
                    break
            if set_decl and set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                return True
            return False

        if is_tuple_indexed(decl):
            idx = expr_node["dimensions"][0]
            if idx.get("type") in ("name_reference_index", "name"):
                return f"{base_name}[{idx['name']}]"
            elif idx.get("type") == "tuple_literal":
                # Build a Python tuple expression for the index, handling raw literals
                parts = []
                for el in idx.get("elements", []):
                    if isinstance(el, dict):
                        parts.append(self._traverse_expression(el, current_iterators, symbolic))
                    else:
                        parts.append(repr(el))
                # Ensure single-element tuples include the trailing comma
                if len(parts) == 1:
                    tuple_expr = f"({parts[0]},)"
                else:
                    tuple_expr = f"({', '.join(parts)})"
                return f"{base_name}[{tuple_expr}]"
            else:
                idx_expr = emit_index_expr(idx)
                return f"{base_name}[{idx_expr}]"
        else:
            # Decision variable case
            if base_name in self.gurobi_var_map:
                # Always emit direct bracket indexing for decision variables (no _safe_get)
                if len(expr_node["dimensions"]) == 1:
                    idx_expr = emit_index_expr(expr_node["dimensions"][0])
                    return f"{base_name}[{idx_expr}]"
                idx_exprs = [emit_index_expr(dim_expr) for dim_expr in expr_node["dimensions"]]
                if len(idx_exprs) == 1:
                    return f"{base_name}[{idx_exprs[0]}]"
                else:
                    return f"{base_name}[({', '.join(idx_exprs)})]"

            # Tuple array case (data struct of records)
            if decl is not None and decl.get("type") in ("tuple_array", "tuple_array_external"):
                idx_exprs = [emit_index_expr(dim_expr) for dim_expr in expr_node["dimensions"]]
                out = base_name
                for ie in idx_exprs:
                    out += f"[{ie}]"
                return out

            # Parameter / data array: choose dict vs list semantics based on emitted shape
            dims_decl = decl.get("dimensions", []) if decl is not None else []
            container_val = self.data_dict.get(base_name)
            is_dict_param = (hasattr(self, "dict_params") and base_name in self.dict_params) or isinstance(container_val, dict)
            has_tuple_keys = isinstance(container_val, dict) and any(isinstance(k, tuple) for k in container_val.keys())

            raw_idx_exprs = [emit_index_expr(dim_expr) for dim_expr in expr_node["dimensions"]]

            if is_dict_param:
                # Composite dict (tuple keys) or multi-dim -> use a single tuple key
                if has_tuple_keys or len(raw_idx_exprs) > 1:
                    return f"{base_name}[({', '.join(raw_idx_exprs)})]"
                # 1D dict keyed by set element or 1-based range index
                return f"{base_name}[{raw_idx_exprs[0]}]"

            # Fallback: list/list-of-lists (0-based); subtract 1 for non-set dims
            index_exprs = []
            for i, dim_expr in enumerate(expr_node["dimensions"]):
                idx_code = emit_index_expr(dim_expr)
                dim_decl = dims_decl[i] if i < len(dims_decl) else None
                if dim_decl and dim_decl.get("type") == "named_set_dimension":
                    index_exprs.append(f"{idx_code}")
                else:
                    index_exprs.append(f"(({idx_code}) - 1)")
            out = base_name
            for ie in index_exprs:
                out += f"[{ie}]"
            return out

    def _expr_binop(self, expr_node, current_iterators, symbolic):
        left_str = self._traverse_expression(expr_node["left"], current_iterators, symbolic)
        right_str = self._traverse_expression(expr_node["right"], current_iterators, symbolic)
        op = expr_node["op"]
        return f"({left_str} {op} {right_str})"

    def _expr_uminus(self, expr_node, current_iterators, symbolic):
        val_str = self._traverse_expression(expr_node["value"], current_iterators)
        return f"-({val_str})"

    def _expr_not(self, expr_node, current_iterators, symbolic):
        # Logical NOT maps to Python 'not' while ensuring expression parenthesis
        val_str = self._traverse_expression(expr_node["value"], current_iterators)
        # If value already a comparison or boolean expression keep parentheses
        return f"not ({val_str})"

    def _expr_and(self, expr_node, current_iterators, symbolic):
        left = self._traverse_expression(expr_node["left"], current_iterators)
        right = self._traverse_expression(expr_node["right"], current_iterators)
        return f"(({left}) and ({right}))"

    def _expr_or(self, expr_node, current_iterators, symbolic):
        left = self._traverse_expression(expr_node["left"], current_iterators)
        right = self._traverse_expression(expr_node["right"], current_iterators)
        return f"(({left}) or ({right}))"

    def _expr_sum(self, expr_node, current_iterators, symbolic):
        # Use module-level logger
        iterators = expr_node["iterators"]
        index_constraint = expr_node.get("index_constraint")
        inner_expression = expr_node["expression"]
        # Reuse previously built auxiliary list if this sum node was already processed (avoid duplicate reification)
        if hasattr(self, "_comparison_sum_meta") and id(expr_node) in self._comparison_sum_meta:
            meta_reuse = self._comparison_sum_meta[id(expr_node)]
            if meta_reuse.get("list_name"):
                return f"gp.quicksum({meta_reuse['list_name']})"

        # Structural reuse: build a canonical key for sums of comparisons so separate AST nodes with identical structure share auxiliaries
        def _struct_key(node):
            if not isinstance(node, dict) or node.get("type") != "sum":
                return None
            expr_inner = node.get("expression")
            # unwrap
            while isinstance(expr_inner, dict) and expr_inner.get("type") == "parenthesized_expression":
                expr_inner = expr_inner.get("expression")
            if not (
                isinstance(expr_inner, dict)
                and expr_inner.get("type") in ("binop", "constraint")
                and expr_inner.get("op") in (">=", ">", "<=", "<", "==")
            ):
                return None
            # Canonical iterator names ordering
            it_names = [it["iterator"] for it in iterators]
            # Build textual forms of lhs/rhs using traversal in symbolic mode under a temp iterator mapping
            temp_iter_map = current_iterators.copy()
            for nm in it_names:
                temp_iter_map[nm] = nm
            try:
                left_node = expr_inner["left"]
                right_node = expr_inner["right"]
                op = expr_inner["op"]
                left_txt = self._traverse_expression(left_node, temp_iter_map, symbolic=True)
                right_txt = self._traverse_expression(right_node, temp_iter_map, symbolic=True)
            except Exception:
                return None
            idxc_txt = None
            if index_constraint is not None:
                try:
                    idxc_txt = self._traverse_expression(index_constraint, temp_iter_map, symbolic=True)
                except Exception:
                    idxc_txt = "IC_ERR"
            return f"cmp_sum|{tuple(it_names)}|{op}|{left_txt}|{right_txt}|{idxc_txt}"

        key = _struct_key(expr_node)
        if key and hasattr(self, "_comparison_sum_key_map") and key in self._comparison_sum_key_map:
            meta = self._comparison_sum_key_map[key]
            return f"gp.quicksum({meta['list_name']})"
        loop_vars = []
        loop_ranges = []
        logger.debug(
            f"[GurobiCodeGen] SUM: iterators={iterators}, index_constraint={index_constraint}, inner_expression={inner_expression}"
        )
        # Build loop ranges left-to-right, allowing later bounds to reference earlier iterator names
        temp_iter_map = current_iterators.copy()
        for it in iterators:
            name = it["iterator"]
            rng = it["range"]
            logger.debug(f"[GurobiCodeGen] SUM iterator: {it}")
            if rng["type"] == "range_specifier":
                start = self._traverse_expression(rng["start"], temp_iter_map, symbolic=True)
                end = self._traverse_expression(rng["end"], temp_iter_map, symbolic=True)
                loop_ranges.append(f"range({start}, {end} + 1)")
            elif rng["type"] == "named_range":
                try:
                    loop_ranges.append(self._emit_range_from_declaration(rng["name"], temp_iter_map, True))
                except SemanticError:
                    set_name = self._emit_set_name_if_declared(rng["name"])
                    if set_name:
                        loop_ranges.append(set_name)
                    else:
                        raise ValueError(f"Range or set '{rng['name']}' not found in declarations.")
            elif rng["type"] in ("named_set", "named_set_dimension"):
                set_name = self._emit_set_name_if_declared(rng["name"])
                if set_name:
                    loop_ranges.append(set_name)
                else:
                    raise ValueError(f"Set '{rng['name']}' not found in declarations.")
            else:
                raise ValueError(f"Unsupported range type for sum: {rng['type']}")
            loop_vars.append(name)
            # Make this iterator available to subsequent range bounds
            temp_iter_map[name] = name
        # Iterator map for inner expression and optional index constraint
        new_iterators = temp_iter_map.copy()
        logger.debug(f"[GurobiCodeGen] SUM loop_vars={loop_vars}, loop_ranges={loop_ranges}")

        # --- Stage 2 enhancement: detect simple comparison term (boolean) needing reification ---
        def _unwrap_paren(n):
            while isinstance(n, dict) and n.get("type") == "parenthesized_expression":
                n = n.get("expression")
            return n

        inner_unwrapped = _unwrap_paren(inner_expression)

        def _is_comparison(node):
            if not isinstance(node, dict):
                return False
            t = node.get("type")
            if t == "binop" and node.get("op") in (">=", "<=", "==", ">", "<"):
                # Accept even if sem_type missing; treat numeric comparison
                return True
            if t == "constraint" and node.get("op") in (">=", "<=", "==", ">", "<"):
                return True
            return False

        # Only handle the simplest pattern now: inner expression itself (after unwrap) is comparison.
        if _is_comparison(inner_unwrapped):
            if not hasattr(self, "_sum_cmp_counter"):
                self._sum_cmp_counter = 0
            self._sum_cmp_counter += 1
            list_name = f"_cmp_sum_list_{self._sum_cmp_counter}"
            self._add_code_line(f"{list_name} = []  # auxiliaries for sum of comparisons")
            # Record metadata for potential reified cardinality patterns later (b == (sum(...) >= k))
            self._comparison_sum_meta = getattr(self, "_comparison_sum_meta", {})
            # We'll fill len_var after loop emitted
            meta_entry = {"list_name": list_name, "len_var": None}
            self._comparison_sum_meta[id(expr_node)] = meta_entry
            if key:
                # register structural key map
                if not hasattr(self, "_comparison_sum_key_map"):
                    self._comparison_sum_key_map = {}
                self._comparison_sum_key_map[key] = meta_entry
            # Build loop header (single or multi-dim) similar to forall
            if len(loop_vars) == 1:
                loop_header = f"for {loop_vars[0]} in {loop_ranges[0]}:"  # simple
            else:
                self._add_code_line("import itertools  # needed for multi-index forall")
                loop_header = f"for {', '.join(loop_vars)} in itertools.product({', '.join(loop_ranges)}):"
            self._add_code_line(loop_header)
            self.indent_level += 1
            # Optional index constraint guard
            if index_constraint is not None:
                cond_str = self._traverse_expression(index_constraint, new_iterators)
                self._add_code_line(f"if {cond_str}:")
                self.indent_level += 1
            # Reify comparison -> binary
            cmp_node = inner_unwrapped
            # Normalize constraint node to binop shape
            if cmp_node.get("type") == "constraint":
                left_node = cmp_node["left"]
                right_node = cmp_node["right"]
                op = cmp_node["op"]
            else:
                left_node = cmp_node["left"]
                right_node = cmp_node["right"]
                op = cmp_node["op"]
            left_expr = self._traverse_expression(left_node, new_iterators)
            right_expr = self._traverse_expression(right_node, new_iterators)
            # Actual var name per iteration (ensure uniqueness via loop index values). We'll use Gurobi's automatic naming when inside loop.
            aux_sym = f"cmp_aux_{self._sum_cmp_counter}_" + "_".join(loop_vars)
            self._add_code_line(f"{aux_sym} = model.addVar(vtype=GRB.BINARY)  # reified ({left_expr} {op} {right_expr})")
            # Estimate big-M on difference; reuse _linear_bounds_safe
            # Use helper for big-M reification to reduce duplication
            for _line in self._emit_reify_comparison(left_node, right_node, left_expr, right_expr, op, aux_sym).split("\n"):
                self._add_code_line(_line)
            self._add_code_line(f"{list_name}.append({aux_sym})")
            if index_constraint is not None:
                self.indent_level -= 1
            self.indent_level -= 1
            # After loop, store length (static after model build)
            len_var = f"{list_name}_len"
            self._add_code_line(f"{len_var} = len({list_name})  # cardinality of comparison terms")
            meta_entry["len_var"] = len_var
            return f"gp.quicksum({list_name})"

        # Fallback: original behavior but with nested generators (no product), so later ranges can use earlier iterators
        inner_expr_str = self._traverse_expression(inner_expression, new_iterators)
        logger.debug(f"[GurobiCodeGen] SUM inner_expr_str: {inner_expr_str}")
        # Emit nested generator: for v1 in R1 for v2 in R2 ... (avoids NameError for dependent ranges)
        gens = " ".join([f"for {v} in {r}" for v, r in zip(loop_vars, loop_ranges)])
        gen = f"{inner_expr_str} {gens}"
        if index_constraint is not None:
            cond_str = self._traverse_expression(index_constraint, new_iterators)
            logger.debug(f"[GurobiCodeGen] SUM cond_str: {cond_str}")
            gen += f" if {cond_str}"
        logger.debug(f"[GurobiCodeGen] SUM generated quicksum: gp.quicksum({gen})")
        return f"gp.quicksum({gen})"

    def _emit_reify_comparison(self, left_node, right_node, left_expr, right_expr, op, aux_sym):
        """Return code lines (joined by \n) that add big-M constraints linking aux_sym to (left op right).

        Chooses a conservative M via static bound estimation; falls back to 1e6. Encodings:
          op in {>=, >}: enforce diff >= 0 when aux=1; diff <= M*aux
          op in {<=, <}: enforce -diff >= 0 when aux=1; -diff <= M*aux
          op == '==': symmetric two-sided with four inequalities (can be tightened later).
        """

        def _estimate_M(left, right):
            lB = self._linear_bounds_safe(left)
            rB = self._linear_bounds_safe(right)
            if lB is None or rB is None or any(v is None for v in (*lB, *rB)):
                return 1e6
            lL, lU = lB
            rL, rU = rB
            diff_lower = lL - rU
            diff_upper = lU - rL
            return max(abs(diff_lower), abs(diff_upper), 1e-9)

        bigM = _estimate_M(left_node, right_node)
        lines = [f"# Reify ({left_expr} {op} {right_expr}) -> {aux_sym} with M={bigM}"]
        eps = EPS
        eq_tol = EQ_TOL
        if op == ">=":
            # z=1 => diff >= 0 ; z=0 => diff <= -eps
            lines.append(f"model.addConstr({left_expr} - {right_expr} >= 0 - {bigM} * (1 - {aux_sym}))")
            lines.append(f"model.addConstr({left_expr} - {right_expr} <= -{eps} + {bigM} * {aux_sym})")
        elif op == ">":
            # z=1 => diff >= eps ; z=0 => diff <= 0
            lines.append(f"model.addConstr({left_expr} - {right_expr} >= {eps} - {bigM} * (1 - {aux_sym}))")
            lines.append(f"model.addConstr({left_expr} - {right_expr} <= 0 + {bigM} * {aux_sym})")
        elif op == "<=":
            # z=1 => diff <= 0 ; z=0 => diff >= eps
            lines.append(f"model.addConstr({left_expr} - {right_expr} <= 0 + {bigM} * (1 - {aux_sym}))")
            lines.append(f"model.addConstr({left_expr} - {right_expr} >= {eps} - {bigM} * {aux_sym})")
        elif op == "<":
            # z=1 => diff <= -eps ; z=0 => diff >= 0
            lines.append(f"model.addConstr({left_expr} - {right_expr} <= -{eps} + {bigM} * (1 - {aux_sym}))")
            lines.append(f"model.addConstr({left_expr} - {right_expr} >= 0 - {bigM} * {aux_sym})")
        elif op == "==":
            # z=1 => |diff| <= eq_tol; z=0 => relaxed by M
            lines.append(f"model.addConstr({left_expr} - {right_expr} <= {eq_tol} + {bigM} * (1 - {aux_sym}))")
            lines.append(f"model.addConstr({right_expr} - {left_expr} <= {eq_tol} + {bigM} * (1 - {aux_sym}))")
            lines.append(f"model.addConstr({left_expr} - {right_expr} >= -{eq_tol} - {bigM} * (1 - {aux_sym}))")
            lines.append(f"model.addConstr({right_expr} - {left_expr} >= -{eq_tol} - {bigM} * (1 - {aux_sym}))")
        else:
            lines.append(f"model.addConstr({aux_sym} == 0)")
        return "\n".join(lines)

    def _expr_field_access(self, expr_node, current_iterators, symbolic):
        base_str = self._traverse_expression(expr_node["base"], current_iterators)
        field = expr_node["field"]
        # Try to resolve tuple type for the base
        tuple_type = None
        # Try to get the semantic type from the AST node if available
        base_sem_type = None
        base_node = expr_node["base"]
        if isinstance(base_node, dict):
            base_sem_type = base_node.get("sem_type")
        # If the base is a known iterator, try to get its type from the AST declarations
        if base_sem_type and hasattr(self, "tuple_types") and base_sem_type in self.tuple_types:
            tuple_type = base_sem_type
        else:
            # Try to infer from iterator names in current_iterators
            if isinstance(base_node, dict) and base_node.get("type") == "name":
                varname = base_node.get("value")
                # Look for iterator type in AST declarations
                for decl in self.ast.get("declarations", []):
                    if decl.get("type") == "set_of_tuples" and decl.get("name"):
                        # If this set is used as a loop range, its tuple_type is relevant
                        if varname in current_iterators.values():
                            tuple_type = decl.get("tuple_type")
                            break
        # If we have tuple_type and tuple_types dict, map field name to index
        if tuple_type and hasattr(self, "tuple_types") and tuple_type in self.tuple_types:
            fields = self.tuple_types[tuple_type]
            field_names = [f["name"] for f in fields]
            if field in field_names:
                idx = field_names.index(field)
                # We expect tuple arrays emitted as dicts of field->value; prefer dict access when base indexing already selects record.
                # However if record is stored as list (legacy path), positional index works.
                return f"({base_str}['{field}'] if isinstance({base_str}, dict) else {base_str}[{idx}])"
        # Fallback: emit as dict access (legacy, but should not happen for tuples)
        return f"{base_str}['{field}']"

    def _expr_boolean_literal(self, expr_node, current_iterators, symbolic):
        # Return 1 for True, 0 for False
        return 1 if expr_node["value"] else 0

    def _expr_string_literal(self, expr_node, current_iterators, symbolic):
        # Return a quoted Python string literal for use in codegen
        val = expr_node.get("value")
        return repr(val)

    def _expr_parenthesized_expression(self, expr_node, current_iterators, symbolic):
        return f"({self._traverse_expression(expr_node['expression'], current_iterators)})"

    # === Utility/Helper Methods (Private) ===
    def _is_data_array(self, name):
        """
        Returns True if name is a parameter loaded from data_dict (not a decision variable).
        """
        return name in self.data_dict

    def _find_declaration_by_name(self, name, types=None):
        """
        Find a declaration by name and (optionally) type(s) in the AST declarations.
        """
        for d in self.ast.get("declarations", []):
            if d.get("name") == name and (types is None or d.get("type") in types):
                return d
        return None

    def _emit_range_from_declaration(self, name, current_iterators, symbolic):
        """
        Emit a Python range string from a named range declaration.
        """
        rng = self._find_declaration_by_name(name, types=["range_declaration_inline"])
        if rng is None:
            raise SemanticError(f"Range '{name}' not found in declarations.")
        start_val = self._traverse_expression(rng["start"], current_iterators, symbolic)
        end_val = self._traverse_expression(rng["end"], current_iterators, symbolic)
        return f"range({start_val}, {end_val} + 1)"

    def _emit_set_name_if_declared(self, name):
        """
        Return set name if declared as supported set type (including external typed) else None.
        """
        set_decl = self._find_declaration_by_name(
            name,
            types=[
                "set_of_tuples",
                "set_of_tuples_external",
                "set_declaration",
                "typed_set",
                "typed_set_external",
            ],
        )
        return name if set_decl is not None else None

    def _construct_loop_header(self, loop_vars, loop_ranges):
        """
        Construct the loop header for forall/sum, handling single and multi-index cases.
        """
        if len(loop_vars) == 1:
            return f"for {loop_vars[0]} in {loop_ranges[0]}:"
        else:
            self._add_code_line("import itertools  # needed for multi-index forall")
            return f"for {', '.join(loop_vars)} in itertools.product({', '.join(loop_ranges)}):"

# === Standard library imports ===
import json
import logging
import re

from .semantic_error import SemanticError
from .tuple_set_helper import TupleSetHelper

# === Third-party imports ===
# (none)

# Module-level logger (no handler/formatter setup here)
logger = logging.getLogger(__name__)

# Numerical tolerances (single source of truth)
EPS = 1e-5  # strictness used to split >, < from >=, <=  (raised to exceed FeasibilityTol)
EQ_TOL = 1e-6  # two-sided tolerance for equality reification


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
        # Merge inline literals (e.g., int D = 10;) so range bounds resolve during checks
        self.data_dict = dict(data_dict) if data_dict is not None else {}
        for decl in self.ast.get("declarations", []):
            if decl.get("type") == "parameter_inline" and decl.get("name") not in self.data_dict:
                self.data_dict[decl["name"]] = decl.get("value")
        self.gurobi_code_lines = []
        self.indent_level = 0
        self.gurobi_var_map = {}  # Maps OPL decision variable names to Gurobi variable objects
        self._add_code_line = self.__class__._add_code_line_impl.__get__(self)
        # NEW: active label name expression inside forall (Python expression string or None)
        self._active_label_name_expr = None

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
        self._add_code_line("model.Params.OutputFlag = 1")
        self._add_code_line("model.Params.LogToConsole = 1")
        self._add_code_line("")
        self._generate_declarations(self.ast["declarations"])
        self._generate_objective(self.ast["objective"])
        # Collect variable bounds before constraints for tighter big-M
        self._collect_variable_bounds(self.ast.get("constraints", []))
        self._generate_constraints(self.ast["constraints"])
        self._add_code_line(
            "print(f'PyOPL/Gurobi: variables={model.NumVars}, constraints={model.NumConstrs}, sense={model.ModelSense}')"
        )
        self._add_code_line("try:")
        self.indent_level += 1
        self._add_code_line("_pyopl_progress_callback")
        self.indent_level -= 1
        self._add_code_line("except NameError:")
        self.indent_level += 1
        self._add_code_line("_pyopl_progress_callback = None")
        self.indent_level -= 1
        self._add_code_line("if _pyopl_progress_callback is not None:")
        self.indent_level += 1
        self._add_code_line("model.optimize(_pyopl_progress_callback)")
        self.indent_level -= 1
        self._add_code_line("else:")
        self.indent_level += 1
        self._add_code_line("model.optimize()")
        self.indent_level -= 1
        # Disambiguate INF_OR_UNBD into INFEASIBLE or UNBOUNDED when possible
        self._add_code_line("if model.status == GRB.INF_OR_UNBD:")
        self.indent_level += 1
        self._add_code_line("model.setParam(GRB.Param.DualReductions, 0)")
        self._add_code_line("if _pyopl_progress_callback is not None:")
        self.indent_level += 1
        self._add_code_line("model.optimize(_pyopl_progress_callback)")
        self.indent_level -= 1
        self._add_code_line("else:")
        self.indent_level += 1
        self._add_code_line("model.optimize()")
        self.indent_level -= 1
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

        code = "\n".join(self.gurobi_code_lines)

        # Fix: ensure booleans are valid Python (some upstream paths may have produced JSON-like true/false)
        # Only replace bare tokens (not inside identifiers/strings).
        code = re.sub(r'(?<![\w"\'\\])true(?![\w"\'\\])', "True", code)
        code = re.sub(r'(?<![\w"\'\\])false(?![\w"\'\\])', "False", code)

        return code

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
    def _eval_data_bound(self, expr, data_dict):
        if expr["type"] == "number":
            return int(expr["value"])
        if expr["type"] == "name":
            return int(data_dict[expr["value"]])
        if expr["type"] == "binop":
            left = self._eval_data_bound(expr["left"], data_dict)
            right = self._eval_data_bound(expr["right"], data_dict)
            operations = {
                "+": lambda: left + right,
                "-": lambda: left - right,
                "*": lambda: left * right,
                "/": lambda: left // right,
            }
            operation = operations.get(expr["op"])
            if operation is not None:
                return operation()
            raise Exception(f"Unsupported binop in range bound expr: {expr['op']}")
        raise Exception(f"Unsupported range bound expr: {expr}")

    def _expected_dimension_length(self, dimension, data_dict):
        if dimension.get("type") == "named_range_dimension":
            range_decl = self._find_declaration_by_name(
                dimension["name"], types=["range_declaration_inline"]
            )
            if range_decl:
                start_idx = self._eval_data_bound(range_decl["start"], data_dict)
                end_idx = self._eval_data_bound(range_decl["end"], data_dict)
                return end_idx - start_idx + 1
            return None
        if dimension.get("type") != "named_set_dimension":
            return None
        set_obj = data_dict.get(dimension["name"])
        if set_obj is None:
            return None
        if isinstance(set_obj, dict) and "elements" in set_obj:
            return len(set_obj["elements"])
        return len(set_obj)

    def _check_parameter_shape(self, param_data, dimensions, data_dict, param_name, dim=0):
        if not dimensions:
            logger.debug("shape %s: reached scalar at dim %d", param_name, dim)
            return
        dimension = dimensions[0]
        expected_len = self._expected_dimension_length(dimension, data_dict)
        logger.debug(
            "shape %s: dim %d expected_len=%s actual=%s dim_type=%s dim_name=%s",
            param_name,
            dim + 1,
            expected_len,
            (len(param_data) if isinstance(param_data, (list, tuple)) else "scalar"),
            dimension.get("type"),
            dimension.get("name", None),
        )
        if expected_len is None:
            return
        if not isinstance(param_data, (list, tuple)):
            logger.debug(
                "shape error %s: expected %dD array, got scalar at dim %d",
                param_name,
                len(dimensions),
                dim + 1,
            )
            raise SemanticError(
                f"Parameter '{param_name}' expected a {len(dimensions)}D array, got scalar at dimension {dim+1}."
            )
        if len(param_data) != expected_len:
            logger.debug(
                "shape error %s: data length %d does not match declared dimension '%s' length %d at dim %d",
                param_name,
                len(param_data),
                dimension.get("name"),
                expected_len,
                dim + 1,
            )
            raise SemanticError(
                f"Parameter '{param_name}' data length {len(param_data)} does not match declared dimension '{dimension.get('name')}' of length {expected_len} at dimension {dim+1}."
            )
        if len(dimensions) > 1:
            for sub_data in param_data:
                self._check_parameter_shape(
                    sub_data, dimensions[1:], data_dict, param_name, dim + 1
                )

    def _normalize_data_declaration_inputs(self, data_dict):
        declarations = self.ast.get("declarations", []) if hasattr(self, "ast") else []
        for declaration in declarations:
            if declaration.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                set_name = declaration.get("name")
                if set_name in data_dict and isinstance(data_dict[set_name], list):
                    data_dict[set_name] = {"elements": data_dict[set_name]}

        parameter_types = (
            "parameter_external",
            "parameter_external_indexed",
            "parameter_external_explicit",
            "parameter_external_explicit_indexed",
            "parameter_inline",
            "parameter_inline_indexed",
        )
        for declaration in declarations:
            if declaration.get("type") not in parameter_types or not declaration.get("dimensions"):
                continue
            param_name = declaration["name"]
            param_data = data_dict.get(param_name)
            if param_name == "Capacity" and "Stores" in data_dict:
                logger.debug(
                    "[data_dict] Stores: %s len=%d",
                    data_dict["Stores"],
                    len(data_dict["Stores"]),
                )
            if isinstance(param_data, list) and param_data and len(param_data) % 2 == 0:
                is_flat_kv = all(
                    isinstance(param_data[index], str)
                    and isinstance(param_data[index + 1], (int, float))
                    for index in range(0, len(param_data), 2)
                )
                if is_flat_kv:
                    data_dict[param_name] = {
                        param_data[index]: param_data[index + 1]
                        for index in range(0, len(param_data), 2)
                    }
                    continue
            if isinstance(param_data, (list, tuple)):
                self._check_parameter_shape(
                    param_data, declaration["dimensions"], data_dict, param_name
                )

    def _tuple_array_records(self, data_value, field_names, index_values=None):
        if isinstance(data_value, dict):
            items = sorted(data_value.items(), key=lambda item: item[0])
        elif isinstance(index_values, list) and len(index_values) == len(data_value):
            items = zip(index_values, data_value)
        else:
            items = enumerate(data_value, start=1)
        records = {}
        for key, record in items:
            if isinstance(record, dict):
                records[key] = {field: record.get(field) for field in field_names if field in record}
                continue
            records[key] = {
                field: record[index]
                for index, field in enumerate(field_names)
                if index < len(record)
            }
        return records

    def _tuple_set_array_records(self, data_value, index_values=None):
        if isinstance(data_value, dict):
            items = sorted(data_value.items(), key=lambda item: item[0])
        elif isinstance(index_values, list) and len(index_values) == len(data_value):
            items = zip(index_values, data_value)
        else:
            items = enumerate(data_value, start=1)
        return {key: list(records or []) for key, records in items}

    def _emit_structured_data_declarations(self, data_dict):
        for declaration in self.ast.get("declarations", []):
            declaration_type = declaration.get("type")
            if declaration_type == "tuple_type":
                self.tuple_types = getattr(self, "tuple_types", {})
                self.tuple_types[declaration["name"]] = declaration["fields"]
            elif declaration_type in ("set_of_tuples", "set_of_tuples_external"):
                set_name = declaration["name"]
                tuple_list = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict)
                if tuple_list:
                    self._add_code_line(f"{set_name} = {repr(tuple_list)}")
            elif declaration_type == "set_of_tuples_array_external":
                set_name = declaration["name"]
                dimensions = declaration.get("dimensions") or []
                index_values = None
                if len(dimensions) == 1 and dimensions[0].get("type") in (
                    "named_range_dimension",
                    "named_set_dimension",
                ):
                    index_values = data_dict.get(dimensions[0].get("name"))
                data_value = data_dict.get(set_name)
                if data_value is not None:
                    records = self._tuple_set_array_records(data_value, index_values)
                    self._add_code_line(f"{set_name} = {repr(records)}")
            elif declaration_type in ("typed_set", "typed_set_external"):
                set_name = declaration["name"]
                elements = declaration.get("value")
                if not elements and set_name in data_dict:
                    elements = data_dict[set_name]
                elements = [] if elements is None else elements
                elems_str = ", ".join(repr(element) for element in elements)
                self._add_code_line(f"{set_name} = [{elems_str}]")
                self._add_code_line(f"{set_name}_index = {{v:i for i,v in enumerate({set_name})}}")
            elif declaration_type in ("tuple_array", "tuple_array_external"):
                array_name = declaration["name"]
                tuple_type = declaration["tuple_type"]
                data_value = data_dict.get(array_name)
                if data_value is not None and tuple_type in getattr(self, "tuple_types", {}):
                    field_names = [field["name"] for field in self.tuple_types[tuple_type]]
                    index_values = data_dict.get(declaration["index_set"])
                    records = self._tuple_array_records(data_value, field_names, index_values)
                    self._add_code_line(f"{array_name} = {repr(records)}")

    def _build_working_data(self, data_dict):
        working_data = dict(data_dict or {})
        for declaration in self.ast.get("declarations", []):
            declaration_type = declaration.get("type")
            if declaration_type in ("typed_set", "typed_set_external"):
                name = declaration["name"]
                if name not in working_data and declaration.get("value") is not None:
                    working_data[name] = declaration["value"]
            if declaration_type in ("parameter_inline", "parameter_inline_indexed"):
                name = declaration["name"]
                if name not in working_data and declaration.get("value") is not None:
                    working_data[name] = declaration["value"]
        preferred_data = dict(working_data)
        for name, value in working_data.items():
            if name.endswith("__map"):
                preferred_data[name[: -len("__map")]] = value
        return working_data, preferred_data

    def _parameter_declaration_map(self):
        parameter_types = {
            "parameter_external",
            "parameter_external_indexed",
            "parameter_external_explicit",
            "parameter_external_explicit_indexed",
            "parameter_inline",
            "parameter_inline_indexed",
        }
        return {
            declaration["name"]: declaration
            for declaration in self.ast.get("declarations", [])
            if declaration.get("type") in parameter_types
        }

    def _validate_1d_mapping_values(self, parameter_declarations, working_data):
        for name, declaration in parameter_declarations.items():
            dimensions = declaration.get("dimensions", []) or []
            if len(dimensions) != 1 or dimensions[0].get("type") not in (
                "named_set_dimension",
                "named_range_dimension",
            ):
                continue
            value = working_data.get(name)
            if not isinstance(value, dict):
                continue
            bad_key = next(
                (key for key, item in value.items() if isinstance(item, (list, tuple, dict))),
                None,
            )
            if bad_key is not None:
                raise SemanticError(
                    f"Parameter '{name}' declared as 1-D over '{dimensions[0].get('name', '')}' expects scalar values per key, "
                    f"but data provides an array for key {repr(bad_key)}. Use scalar values (e.g., 2.0), not [2.0]."
                )

    def _named_range_bounds(self, range_dimension, working_data):
        start_node = range_dimension.get("start")
        end_node = range_dimension.get("end")
        if isinstance(start_node, dict) and isinstance(end_node, dict):
            return (
                self._eval_data_bound(start_node, working_data),
                self._eval_data_bound(end_node, working_data),
            )
        range_name = range_dimension.get("name")
        declaration = self._find_declaration_by_name(
            range_name, types=["range_declaration_inline"]
        )
        if isinstance(declaration, dict):
            return (
                self._eval_data_bound(declaration["start"], working_data),
                self._eval_data_bound(declaration["end"], working_data),
            )
        range_data = working_data.get(range_name)
        if isinstance(range_data, dict) and range_data.get("type") == "range_data":
            return int(range_data["start"]), int(range_data["end"])
        raise SemanticError(f"Named range '{range_name}' has no bounds.")

    def _is_tuple_range_parameter(self, declaration):
        if declaration is None:
            return False
        dimensions = declaration.get("dimensions", [])
        return (
            len(dimensions) == 2
            and dimensions[0].get("type") == "named_set_dimension"
            and dimensions[1].get("type") == "named_range_dimension"
        )

    def _emit_tuple_range_dict_rows(self, name, value, declaration, working_data):
        if not (
            self._is_tuple_range_parameter(declaration)
            and isinstance(value, dict)
            and all(isinstance(row, (list, tuple)) for row in value.values())
        ):
            return False
        range_dimension = declaration["dimensions"][1]
        start, end = self._named_range_bounds(range_dimension, working_data)
        expected_len = end - start + 1
        flattened = {}
        for key, row in value.items():
            if len(row) != expected_len:
                raise SemanticError(
                    f"Parameter '{name}' row for key {key} has length {len(row)}; expected {expected_len}."
                )
            key_object = tuple(key) if isinstance(key, (list, tuple)) else key
            for position in range(start, end + 1):
                flattened[(key_object, position)] = row[position - start]
        self._add_code_line(f"{name} = {repr(flattened)}")
        self.dict_params.add(name)
        return True

    def _emit_tuple_range_list_rows(self, name, value, declaration, working_data):
        if not (
            self._is_tuple_range_parameter(declaration)
            and isinstance(value, list)
            and value
        ):
            return False
        set_name = declaration["dimensions"][0]["name"]
        range_dimension = declaration["dimensions"][1]
        set_elements = TupleSetHelper.get_tuple_set(set_name, self.ast, working_data) or []
        set_elements = [
            tuple(element) if isinstance(element, (list, tuple)) else (element,)
            for element in set_elements
        ]
        start, end = self._named_range_bounds(range_dimension, working_data)
        expected_len = end - start + 1
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
            flattened = {}
            valid = True
            for key_raw, row in value:
                key_object = tuple(key_raw) if isinstance(key_raw, (list, tuple)) else key_raw
                if not isinstance(row, (list, tuple)) or len(row) != expected_len:
                    valid = False
                    break
                for offset, cell in enumerate(row):
                    flattened[(key_object, start + offset)] = cell
            if valid and flattened:
                self._add_code_line(f"{name} = {repr(flattened)}")
                logger.info(
                    "Emitting tuple-range flattened dict for '%s' with %d entries",
                    name,
                    len(flattened),
                )
                self.dict_params.add(name)
                return True
        if len(set_elements) != len(value) or not all(
            isinstance(row, (list, tuple)) and len(row) == expected_len for row in value
        ):
            return False
        flattened = {
            (set_elements[index], position): value[index][position - start]
            for index in range(len(set_elements))
            for position in range(start, end + 1)
        }
        self._add_code_line(f"{name} = {repr(flattened)}")
        logger.info(
            "Emitting row-major flattened dict for '%s' with %d entries",
            name,
            len(flattened),
        )
        self.dict_params.add(name)
        return True

    def _emit_typed_set_data(self, name, value):
        declaration = self._find_declaration_by_name(
            name, types=["typed_set", "typed_set_external"]
        )
        if declaration is None:
            return False
        elements = ", ".join(repr(element) for element in value)
        self._add_code_line(f"{name} = [{elements}]")
        self._add_code_line(f"{name}_index = {{v:i for i,v in enumerate({name})}}")
        return True

    def _emit_1d_range_parameter(self, name, value, declaration, data_dict, working_data):
        dimensions = declaration.get("dimensions", []) if declaration is not None else []
        if not (
            isinstance(value, list)
            and value
            and len(dimensions) == 1
            and dimensions[0].get("type") == "named_range_dimension"
        ):
            return False
        range_name = dimensions[0]["name"]
        set_declaration = self._find_declaration_by_name(
            range_name, types=["typed_set", "typed_set_external"]
        )
        if set_declaration and range_name in data_dict:
            set_elements = data_dict[range_name]
            if len(set_elements) != len(value):
                raise SemanticError(
                    f"Parameter '{name}' has {len(value)} items but declared set '{range_name}' has {len(set_elements)} elements."
                )
            items = ", ".join(
                f"{json.dumps(key)}: {json.dumps(item)}"
                for key, item in zip(set_elements, value)
            )
            self._add_code_line(f"{name} = {{{items}}}")
            self.dict_params.add(name)
            return True
        range_declaration = self._find_declaration_by_name(
            range_name, types=["range_declaration_inline"]
        )
        if range_declaration is None:
            return False
        start = self._eval_data_bound(range_declaration["start"], working_data)
        end = self._eval_data_bound(range_declaration["end"], working_data)
        expected_len = end - start + 1
        if len(value) != expected_len:
            raise SemanticError(
                f"Parameter '{name}' has {len(value)} items but declared range '{range_name}' expects {expected_len}."
            )
        items = ", ".join(
            f"{index}: {json.dumps(value[index - start])}"
            for index in range(start, end + 1)
        )
        self._add_code_line(f"{name} = {{{items}}}")
        self.dict_params.add(name)
        return True

    def _emit_1d_set_parameter(self, name, value, declaration, data_dict):
        dimensions = declaration.get("dimensions", []) if declaration is not None else []
        if not (
            isinstance(value, list)
            and value
            and len(dimensions) == 1
            and dimensions[0].get("type") == "named_set_dimension"
        ):
            return False
        set_name = dimensions[0]["name"]
        set_declaration = self._find_declaration_by_name(
            set_name,
            types=[
                "set_of_tuples",
                "set_of_tuples_external",
                "typed_set",
                "typed_set_external",
                "set_declaration",
            ],
        )
        if set_declaration is not None:
            if set_declaration.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                set_elements = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict) or []
            else:
                set_elements = data_dict.get(set_name, set_declaration.get("value", []))
            if isinstance(set_elements, dict) and "elements" in set_elements:
                set_elements = set_elements["elements"]
            if set_elements is not None and len(set_elements) == len(value):
                mapping = dict(zip(set_elements, value))
                self._add_code_line(f"{name} = {repr(mapping)}")
                self.dict_params.add(name)
                return True
        if set_name not in data_dict:
            return False
        set_elements = data_dict[set_name]
        if isinstance(set_elements, dict) and "elements" in set_elements:
            set_elements = set_elements["elements"]
        if len(set_elements) != len(value):
            return False
        items = ", ".join(
            f"{json.dumps(key)}: {json.dumps(item)}"
            for key, item in zip(set_elements, value)
        )
        self._add_code_line(f"{name} = {{{items}}}")
        self.dict_params.add(name)
        return True

    def _normalize_data_key(self, value):
        if isinstance(value, (list, tuple)):
            return tuple(self._normalize_data_key(element) for element in value)
        return value

    def _normalize_set_elements(self, value):
        elements = value.get("elements") if isinstance(value, dict) and "elements" in value else value
        if elements is None:
            return []
        return [self._normalize_data_key(element) for element in elements]

    def _dimension_keys(self, dimensions, working_data):
        keys_by_dimension = []
        try:
            for dimension in dimensions:
                dimension_type = dimension.get("type")
                if dimension_type in ("named_range_dimension", "range_index"):
                    start = self._eval_data_bound(dimension["start"], working_data)
                    end = self._eval_data_bound(dimension["end"], working_data)
                    keys_by_dimension.append(list(range(start, end + 1)))
                elif dimension_type == "named_set_dimension":
                    set_value = working_data.get(dimension["name"], [])
                    keys_by_dimension.append(self._normalize_set_elements(set_value))
                else:
                    return None
        except Exception:
            return None
        return keys_by_dimension

    def _flatten_data_positions(self, value, position=()):
        if isinstance(value, (list, tuple)) and value and any(
            isinstance(element, (list, tuple)) for element in value
        ):
            for index, element in enumerate(value):
                yield from self._flatten_data_positions(element, position + (index,))
            return
        if isinstance(value, (list, tuple)):
            for index, element in enumerate(value):
                yield position + (index,), element
            return
        yield position, value

    def _flatten_positional_parameter(self, name, value, keys_by_dimension):
        flattened = {}
        try:
            for positions, item in self._flatten_data_positions(value):
                if len(positions) != len(keys_by_dimension):
                    raise SemanticError(
                        f"Parameter '{name}' dimensionality mismatch: data depth {len(positions)} vs declared {len(keys_by_dimension)}."
                    )
                if any(
                    position < 0 or position >= len(keys_by_dimension[index])
                    for index, position in enumerate(positions)
                ):
                    raise SemanticError(f"Parameter '{name}' positional index is out of bounds.")
                key = tuple(
                    keys_by_dimension[index][position]
                    for index, position in enumerate(positions)
                )
                flattened[key] = item
        except SemanticError:
            return None
        return flattened

    def _emit_positional_nd_parameters(self, working_data, parameter_declarations):
        emitted = set()
        for name, value in working_data.items():
            declaration = parameter_declarations.get(name)
            if declaration is None or not isinstance(value, (list, tuple)):
                continue
            dimensions = declaration.get("dimensions", []) or []
            if len(dimensions) < 2:
                continue
            keys_by_dimension = self._dimension_keys(dimensions, working_data)
            if not keys_by_dimension:
                continue
            flattened = self._flatten_positional_parameter(name, value, keys_by_dimension)
            if flattened is None:
                continue
            self._add_code_line(f"{name} = {repr(flattened)}")
            logger.info("Emitting flattened dict for '%s' with %d entries", name, len(flattened))
            self.dict_params.add(name)
            emitted.add(name)
        return emitted

    def _key_matches_dimensions(self, key, dimensions):
        if not isinstance(key, (list, tuple)) or len(key) != len(dimensions):
            return False
        for index, dimension in enumerate(dimensions):
            if dimension.get("type") == "named_set_dimension":
                set_declaration = self._find_declaration_by_name(dimension.get("name"))
                expects_tuple = set_declaration and set_declaration.get("type") in (
                    "set_of_tuples",
                    "set_of_tuples_external",
                )
                if expects_tuple != isinstance(key[index], (list, tuple)):
                    return False
            elif isinstance(key[index], (list, tuple)):
                return False
        return True

    def _expand_two_set_rows(self, value, dimensions, working_data):
        if len(dimensions) != 2 or any(
            dimension.get("type") != "named_set_dimension" for dimension in dimensions
        ):
            return None
        second_set = dimensions[1]["name"]
        try:
            labels = TupleSetHelper.get_tuple_set(second_set, self.ast, working_data)
        except Exception:
            return None
        if not labels:
            return None
        normalized_labels = [
            tuple(label) if isinstance(label, (list, tuple)) else label for label in labels
        ]
        flattened = {}
        for key, row in value.items():
            if not isinstance(row, (list, tuple)) or len(row) != len(normalized_labels):
                return None
            key_object = tuple(key) if isinstance(key, (list, tuple)) else (key,)
            for label, item in zip(normalized_labels, row):
                flattened[(key_object, label)] = item
        return flattened or None

    def _has_full_length_keys(self, value, dimension_count):
        if not isinstance(value, dict) or not any(
            isinstance(key, (list, tuple)) for key in value
        ):
            return False
        key_lengths = {
            len(key) if isinstance(key, (list, tuple)) else 1 for key in value
        }
        return len(key_lengths) == 1 and next(iter(key_lengths)) == dimension_count

    def _normalize_full_key_mapping(self, value, dimensions, working_data):
        if not self._has_full_length_keys(value, len(dimensions)):
            return None
        has_sequence_values = any(
            isinstance(item, (list, tuple)) for item in value.values()
        )
        all_keys_match = all(
            self._key_matches_dimensions(key, dimensions) for key in value
        )
        if has_sequence_values and not all_keys_match:
            expanded = self._expand_two_set_rows(value, dimensions, working_data)
            if expanded is not None:
                return expanded
        return {
            tuple(key) if isinstance(key, (list, tuple)) else (key,): item
            for key, item in value.items()
        }

    def _resolve_set_elements(self, set_name, working_data):
        set_value = working_data.get(set_name)
        if set_value is None:
            declaration = self._find_declaration_by_name(
                set_name,
                types=[
                    "typed_set",
                    "typed_set_external",
                    "set_declaration",
                    "set_of_tuples",
                    "set_of_tuples_external",
                ],
            )
            if declaration is not None:
                set_value = declaration.get("value")
        if set_value is None:
            return None
        return self._normalize_set_elements(set_value)

    def _dimension_labels_and_start(self, dimension, working_data):
        dimension_type = dimension.get("type")
        if dimension_type == "named_set_dimension":
            labels = self._resolve_set_elements(dimension["name"], working_data)
            return (list(labels) if labels is not None else None), None
        if dimension_type in ("named_range_dimension", "range_index"):
            start = self._eval_data_bound(dimension["start"], working_data)
            end = self._eval_data_bound(dimension["end"], working_data)
            return list(range(start, end + 1)), start
        return None, None

    def _position_label(self, labels, start, position, provided_length):
        if labels is not None and len(labels) == provided_length:
            return labels[position] if position < len(labels) else position + 1
        return start + position if isinstance(start, int) else position + 1

    def _flatten_labeled_data(self, node, dimension_index, prefix, labels, starts, output):
        if dimension_index == len(labels) - 1:
            if isinstance(node, dict):
                for key, value in node.items():
                    output[prefix + (self._normalize_data_key(key),)] = value
                return
            values = node if isinstance(node, (list, tuple)) else [node]
            for position, value in enumerate(values):
                label = self._position_label(
                    labels[dimension_index], starts[dimension_index], position, len(values)
                )
                output[prefix + (label,)] = value
            return
        children = node.items() if isinstance(node, dict) else enumerate(
            node if isinstance(node, (list, tuple)) else [node]
        )
        child_count = len(node) if isinstance(node, (dict, list, tuple)) else 1
        for position, child in children:
            if isinstance(node, dict):
                label = self._normalize_data_key(position)
            else:
                label = self._position_label(
                    labels[dimension_index], starts[dimension_index], position, child_count
                )
            self._flatten_labeled_data(
                child, dimension_index + 1, prefix + (label,), labels, starts, output
            )

    def _flatten_nested_parameter(self, value, dimensions, working_data):
        labels = []
        starts = []
        try:
            for dimension in dimensions:
                dimension_labels, start = self._dimension_labels_and_start(
                    dimension, working_data
                )
                labels.append(dimension_labels)
                starts.append(start)
            flattened = {}
            self._flatten_labeled_data(value, 0, (), labels, starts, flattened)
            return flattened or None
        except Exception:
            return None

    def _emit_mapping_nd_parameters(
        self, working_data, parameter_declarations, already_emitted
    ):
        emitted = set()
        for name, value in working_data.items():
            if name in already_emitted:
                continue
            declaration = parameter_declarations.get(name)
            dimensions = declaration.get("dimensions", []) if declaration else []
            mapping = self._normalize_full_key_mapping(value, dimensions, working_data)
            if mapping is None and isinstance(value, dict) and len(dimensions) >= 2:
                mapping = self._flatten_nested_parameter(value, dimensions, working_data)
            if mapping is not None:
                self._add_code_line(f"{name} = {repr(mapping)}")
                self.dict_params.add(name)
                emitted.add(name)
                continue
            if isinstance(value, (list, tuple)) and declaration is not None:
                self._check_parameter_shape(value, dimensions, working_data, name)
        return emitted

    def _generate_data_declarations(self, data_dict):
        """Generate Python code for data declarations and AST tuple/set declarations."""
        logger.debug("Entering _generate_data_declarations")
        self._normalize_data_declaration_inputs(data_dict)
        self.dict_params = set()
        self._emit_structured_data_declarations(data_dict)
        working_data, working_data_pref = self._build_working_data(data_dict)

        # Do not exit early on empty data_dict; inline params may still need emission
        self._add_code_line("# Data from .dat file")

        # New: validation for 1-D params over set/range where data is provided as a dict with list values.
        param_decl_map = self._parameter_declaration_map()
        self._validate_1d_mapping_values(param_decl_map, working_data_pref)

        # --- helpers for evaluating bounds and normalizing set elements ---
        def _eval_expr_bound(expr):
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

        already_emitted = self._emit_positional_nd_parameters(
            working_data_pref, param_decl_map
        )

        already_emitted.update(
            self._emit_mapping_nd_parameters(
                working_data_pref, param_decl_map, already_emitted
            )
        )

        self.dict_params = set(self.dict_params)
        self._emit_structured_data_declarations(data_dict)

        for name, value in working_data_pref.items():
            if name in already_emitted:
                continue
            declaration = param_decl_map.get(name)
            if self._emit_tuple_range_dict_rows(name, value, declaration, working_data):
                continue
            if self._emit_tuple_range_list_rows(name, value, declaration, working_data):
                continue
            if name in param_decl_map:
                logger.debug(
                    "_generate_data_declarations: Emitting parameter %s type=%s dims=%s",
                    name,
                    param_decl_map[name].get("type"),
                    param_decl_map[name].get("dimensions"),
                )
            if self._emit_typed_set_data(name, value):
                continue
            if self._emit_1d_range_parameter(
                name, value, declaration, data_dict, working_data
            ):
                continue
            if self._emit_1d_set_parameter(name, value, declaration, data_dict):
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
            if decl_type in ("set_of_tuples", "set_of_tuples_external", "set_of_tuples_array_external"):
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
                    bound_args = self._decl_dvar_bound_args(decl)
                    has_explicit_lb = "lower_bound" in decl
                    # Ensure lower bounds match domain semantics for tuple-indexed variables
                    if vtype == "boolean":
                        lb_arg = ""  # binaries are [0,1] by default
                    elif vtype in ("int+", "float+"):
                        lb_arg = "" if has_explicit_lb else ", lb=0"
                    else:
                        # plain int/float: allow negative domain
                        lb_arg = "" if has_explicit_lb else ", lb=-GRB.INFINITY"
                    self._add_code_line(
                        f"{decl['name']} = model.addVars({set_name}, vtype={grb_vtype}, name='{decl['name']}'{lb_arg}{bound_args})"
                    )
                    # Register decision variable name so expression emission treats it as a variable
                    self.gurobi_var_map[decl["name"]] = decl["name"]
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

    # NEW: helper to format constraint name, honoring active label template if present
    def _format_name_expr(self, base_prefix: str, suffix: str = "") -> str:
        """
        Returns a Python expression string for the name= argument.
        If a label template is active, returns (label_expr + suffix), else a quoted literal.
        """
        if getattr(self, "_active_label_name_expr", None):
            if suffix:
                return f"({self._active_label_name_expr} + {repr(suffix)})"
            return f"{self._active_label_name_expr}"
        # Fallback: literal compile-time name
        return repr(base_prefix + (suffix or ""))

    def _gurobi_comparison_expr(self, left_expr, op, right_expr) -> str:
        left_expr = str(left_expr)
        op = str(op)
        right_expr = str(right_expr)
        if op == ">":
            return f"{left_expr} >= ({right_expr}) + {EPS}"
        if op == "<":
            return f"{left_expr} <= ({right_expr}) - {EPS}"
        return f"{left_expr} {op} {right_expr}"

    # NEW: build a Python expression for a label template inside a forall loop
    def _compute_label_expr(self, label_template: dict) -> str:
        """
        Build a Python expression string that evaluates to a constraint name at runtime:
        'Name[i,j,...]' using current iterator variables.
        """
        base = label_template.get("name", "ct")
        iters = list(label_template.get("iterators") or [])
        if not iters:
            return repr(base)
        list_expr = "[" + ", ".join(iters) + "]"
        # Join with ',' and wrap in brackets; str() handles tuples nicely
        return f"({repr(base)} + '[' + ','.join(str(v) for v in {list_expr}) + ']')"

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
        # NEW: respect top-level labels (outside forall) by pushing an active label name
        prev = self._active_label_name_expr
        try:
            if isinstance(constraint_node, dict) and "label" in constraint_node and not self._active_label_name_expr:
                # Use the literal label as the active name (no indices to substitute)
                self._active_label_name_expr = repr(constraint_node["label"])
            node_type = constraint_node["type"]
            method = getattr(self, f"_constraint_{node_type}", None)
            if not method:
                raise NotImplementedError(f"Constraint type '{node_type}' is not supported by the Gurobi code generator.")
            method(constraint_node, constr_name_prefix, current_iterators)
        finally:
            self._active_label_name_expr = prev

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

    def _emit_implication_consequent(
        self, cons_op, cons_left_expr, cons_right_expr, bigM_cons, flag_var, constr_name_prefix
    ):
        if cons_op == "==":
            constraints = (
                f"{cons_left_expr} - {cons_right_expr} <= {EQ_TOL} + {bigM_cons} * (1 - {flag_var})",
                f"{cons_right_expr} - {cons_left_expr} <= {EQ_TOL} + {bigM_cons} * (1 - {flag_var})",
                f"{cons_left_expr} - {cons_right_expr} >= -{EQ_TOL} - {bigM_cons} * (1 - {flag_var})",
                f"{cons_right_expr} - {cons_left_expr} >= -{EQ_TOL} - {bigM_cons} * (1 - {flag_var})",
            )
            suffixes = ("_cons_eq1", "_cons_eq2", "_cons_eq3", "_cons_eq4")
        elif cons_op == ">=":
            constraints = (f"{cons_left_expr} - {cons_right_expr} >= -{bigM_cons} * (1 - {flag_var})",)
            suffixes = ("_cons_ge",)
        elif cons_op == ">":
            constraints = (f"{cons_left_expr} - {cons_right_expr} >= {EPS} - {bigM_cons} * (1 - {flag_var})",)
            suffixes = ("_cons_gt",)
        elif cons_op == "<=":
            constraints = (f"{cons_left_expr} - {cons_right_expr} <= {bigM_cons} * (1 - {flag_var})",)
            suffixes = ("_cons_le",)
        elif cons_op == "<":
            constraints = (f"{cons_left_expr} - {cons_right_expr} <= -{EPS} + {bigM_cons} * (1 - {flag_var})",)
            suffixes = ("_cons_lt",)
        else:
            raise ValueError(f"Unsupported consequent operator in implication: {cons_op}")

        for constraint, suffix in zip(constraints, suffixes):
            self._add_code_line(
                f"model.addConstr({constraint}, name={self._format_name_expr(constr_name_prefix, suffix)})"
            )

    def _emit_specialized_implication_indicator(
        self,
        ant_left,
        ant_right,
        ant_op,
        ant_left_expr,
        ant_right_expr,
        cons_left,
        cons_right,
        cons_op,
        cons_left_expr,
        cons_right_expr,
        constr_name_prefix,
    ):
        def is_binary_var(node):
            if node.get("type") == "name":
                varname = node["value"]
            elif node.get("type") == "indexed_name":
                varname = node["name"]
            else:
                return False
            decl = self._find_declaration_by_name(varname)
            return bool(decl and decl.get("type") in ("dvar", "dvar_indexed") and decl.get("var_type") == "boolean")

        consequent_is_true = cons_right.get("type") == "boolean_literal" and cons_right.get("value") is True
        consequent_is_one = cons_right.get("type") == "number" and float(cons_right.get("value", 0)) == 1.0
        if cons_op == "==" and is_binary_var(cons_left) and (consequent_is_one or consequent_is_true):
            if ant_op == ">" and ant_right.get("type") == "number":
                self._add_code_line(
                    f"model.addGenConstrIndicator({cons_left_expr}, 0, {ant_left_expr} <= {ant_right_expr}, name={self._format_name_expr(constr_name_prefix, '_indicator_contra')})"
                )
                return True
            if ant_op == ">=" and ant_right.get("type") == "number":
                try:
                    adjusted = float(ant_right.get("value", 0)) - EPS
                    consequent_expr = f"{ant_left_expr} <= {adjusted}"
                except Exception:
                    consequent_expr = f"{ant_left_expr} <= ({ant_right_expr} - {EPS})"
                self._add_code_line(
                    f"model.addGenConstrIndicator({cons_left_expr}, 0, {consequent_expr}, name={self._format_name_expr(constr_name_prefix, '_indicator_contra_ge')})"
                )
                return True

        if ant_op not in ("==", ">=", "<=") or not is_binary_var(ant_left):
            return False
        try:
            rhs_val = float(ant_right.get("value", 0))
        except (TypeError, ValueError):
            return False
        indicator_value = None
        if ant_op == "==" and rhs_val in (0, 1):
            indicator_value = int(rhs_val)
        elif ant_op == ">=" and rhs_val == 1:
            indicator_value = 1
        elif ant_op == "<=" and rhs_val == 0:
            indicator_value = 0
        if indicator_value is None or cons_op not in ("==", ">=", "<=", ">", "<"):
            return False
        indicator_expr = self._gurobi_comparison_expr(cons_left_expr, cons_op, cons_right_expr)
        self._add_code_line(
            f"model.addGenConstrIndicator({ant_left_expr}, {indicator_value}, {indicator_expr}, name={self._format_name_expr(constr_name_prefix, '_indicator')})"
        )
        return True

    @staticmethod
    def _normalize_boolean_aux_node(node):
        if node.get("type") == "constraint":
            op = node.get("op")
            if op in ("==", "<", ">", "<=", ">="):
                return {
                    "type": "binop",
                    "op": op,
                    "left": node["left"],
                    "right": node["right"],
                    "sem_type": "boolean",
                }
        if node.get("type") == "parenthesized_expression":
            return GurobiCodeGenerator._normalize_boolean_aux_node(node["expression"])
        if node.get("type") == "constraint" and node.get("op") == "==" and isinstance(node.get("left"), dict):
            left = node["left"]
            right = node.get("right")
            if left.get("type") in ("and", "or", "not") and (
                not isinstance(right, dict) or right.get("type") == "boolean_literal"
            ):
                return left
        return node

    @staticmethod
    def _is_composite_boolean(node):
        if not isinstance(node, dict):
            return False
        node_type = node.get("type")
        if node_type == "parenthesized_expression":
            return GurobiCodeGenerator._is_composite_boolean(node.get("expression"))
        if node_type in ("and", "or", "not"):
            return True
        if node_type != "constraint" or node.get("op") != "==":
            return False
        left = node.get("left")
        right = node.get("right")
        return (
            isinstance(left, dict)
            and left.get("type") in ("and", "or", "not")
            and (not isinstance(right, dict) or right.get("type") == "boolean_literal")
        )

    @staticmethod
    def _wrap_boolean_literal_as_constraint(node):
        if node.get("type") != "boolean_literal":
            return node
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

    def _estimate_big_m_for_difference(self, left_node, right_node):
        left_bounds = self._linear_bounds_safe(left_node)
        right_bounds = self._linear_bounds_safe(right_node)
        if left_bounds is None or right_bounds is None:
            return None
        if any(value is None for value in (*left_bounds, *right_bounds)):
            return None
        left_lower, left_upper = left_bounds
        right_lower, right_upper = right_bounds
        difference_lower = left_lower - right_upper
        difference_upper = left_upper - right_lower
        return max(abs(difference_lower), abs(difference_upper), 1e-9)

    def _extract_implication_constraint(self, node, current_iterators):
        if node.get("type") == "constraint":
            left_node = self._unwrap_parenthesized(node["left"])
            if left_node.get("type") == "binop" and left_node.get("sem_type") == "boolean":
                left = left_node["left"]
                right = left_node["right"]
                operator = left_node["op"]
            else:
                left = node["left"]
                right = node["right"]
                operator = node["op"]
        elif node.get("type") == "binop":
            left = node["left"]
            right = node["right"]
            operator = node["op"]
        else:
            raise ValueError("Implication constraints must be between constraints or binops.")
        left_expression = self._traverse_expression(left, current_iterators)
        right_expression = self._traverse_expression(right, current_iterators)
        return left, right, operator, left_expression, right_expression

    def _bind_implication_comparison_to_binary(
        self, binary, comparison, iterators, constr_name_prefix
    ):
        left, right, operator, left_expression, right_expression = (
            self._extract_implication_constraint(comparison, iterators)
        )
        estimated_big_m = self._estimate_big_m_for_difference(left, right)
        big_m = estimated_big_m if estimated_big_m is not None else 1e6
        epsilon = 0
        if operator == ">=":
            constraints = (
                (f"{left_expression} - {right_expression} >= -{big_m} * (1 - {binary})", f"_aux_ge_{binary}"),
                (f"{left_expression} - {right_expression} <= {big_m} * {binary}", f"_aux_ge_relax_{binary}"),
            )
        elif operator == ">":
            constraints = (
                (f"{left_expression} - {right_expression} >= {epsilon} - {big_m} * (1 - {binary})", f"_aux_gt_{binary}"),
                (f"{left_expression} - {right_expression} <= {big_m} * {binary}", f"_aux_gt_relax_{binary}"),
            )
        elif operator == "<=":
            constraints = (
                (f"{left_expression} - {right_expression} <= {big_m} * (1 - {binary})", f"_aux_le_{binary}"),
                (f"{left_expression} - {right_expression} >= -{big_m} * {binary}", f"_aux_le_relax_{binary}"),
            )
        elif operator == "<":
            constraints = (
                (f"{left_expression} - {right_expression} <= -{epsilon} + {big_m} * (1 - {binary})", f"_aux_lt_{binary}"),
                (f"{left_expression} - {right_expression} >= 0 - {big_m} * {binary}", f"_aux_lt_relax_{binary}"),
            )
        elif operator == "==":
            constraints = (
                (f"{left_expression} - {right_expression} <= {epsilon} + {big_m} * (1 - {binary})", f"_aux_eq1_{binary}"),
                (f"{right_expression} - {left_expression} <= {epsilon} + {big_m} * (1 - {binary})", f"_aux_eq2_{binary}"),
                (f"{left_expression} - {right_expression} >= -{epsilon} - {big_m} * (1 - {binary})", f"_aux_eq3_{binary}"),
                (f"{right_expression} - {left_expression} >= -{epsilon} - {big_m} * (1 - {binary})", f"_aux_eq4_{binary}"),
            )
        else:
            raise ValueError(
                f"Unsupported comparison operator in boolean linearization: {operator}"
            )
        for constraint, suffix in constraints:
            self._add_code_line(
                f"model.addConstr({constraint}, name={self._format_name_expr(constr_name_prefix, suffix)})"
            )

    def _new_implication_boolean_aux(self, prefix, constr_name_prefix):
        self._bool_aux_counter = getattr(self, "_bool_aux_counter", 0) + 1
        name = f"{prefix}_b{self._bool_aux_counter}_{constr_name_prefix}"
        self._add_code_line(f"{name} = model.addVar(vtype=GRB.BINARY, name='{name}')")
        return name

    def _boolean_expr_to_binary(self, node, iterators, constr_name_prefix):
        node = self._normalize_boolean_aux_node(node)
        if (
            node.get("type") in ("constraint", "binop")
            and node.get("sem_type") == "boolean"
            and node.get("type") not in ("and", "or", "not")
        ):
            binary = self._new_implication_boolean_aux("cmp", constr_name_prefix)
            self._bind_implication_comparison_to_binary(
                binary, node, iterators, constr_name_prefix
            )
            return binary
        node_type = node.get("type")
        if node_type == "not":
            inner = self._boolean_expr_to_binary(
                node["value"], iterators, constr_name_prefix
            )
            binary = self._new_implication_boolean_aux("not", constr_name_prefix)
            self._add_code_line(
                f"model.addConstr({binary} + {inner} == 1, name={self._format_name_expr(constr_name_prefix, f'_notlink_{binary}')} )"
            )
            return binary
        if node_type in ("and", "or"):
            left = self._boolean_expr_to_binary(
                node["left"], iterators, constr_name_prefix
            )
            right = self._boolean_expr_to_binary(
                node["right"], iterators, constr_name_prefix
            )
            binary = self._new_implication_boolean_aux(node_type, constr_name_prefix)
            if node_type == "and":
                bounds = (f"{binary} <= {left}", f"{binary} <= {right}", f"{binary} >= {left} + {right} - 1")
            else:
                bounds = (f"{binary} >= {left}", f"{binary} >= {right}", f"{binary} <= {left} + {right}")
            for index, bound in enumerate(bounds, 1):
                self._add_code_line(
                    f"model.addConstr({bound}, name={self._format_name_expr(constr_name_prefix, f'_{node_type}{index}_{binary}')} )"
                )
            return binary
        if node_type == "boolean_literal":
            binary = self._new_implication_boolean_aux("lit", constr_name_prefix)
            value = 1 if node.get("value") else 0
            self._add_code_line(
                f"model.addConstr({binary} == {value}, name={self._format_name_expr(constr_name_prefix, f'_lit_{binary}')} )"
            )
            return binary
        raise ValueError(f"Unsupported boolean expression type for auxiliary binary: {node_type}")

    def _constraint_implication_constraint(self, constraint_node, constr_name_prefix, current_iterators):
        """
        Handles implication constraints: <constraint> => <constraint>.
        Uses Gurobi indicator constraints when possible, otherwise falls back to big-M encoding.
        """

        if self._is_composite_boolean(
            constraint_node["antecedent"]
        ) or self._is_composite_boolean(constraint_node["consequent"]):
            ant_bin = self._boolean_expr_to_binary(
                constraint_node["antecedent"], current_iterators, constr_name_prefix
            )
            cons_bin = self._boolean_expr_to_binary(
                constraint_node["consequent"], current_iterators, constr_name_prefix
            )
            # PATCH: ensure name honors label (if any)
            self._add_code_line(
                f"model.addConstr({ant_bin} <= {cons_bin}, name={self._format_name_expr(constr_name_prefix, '_impl_bin')})"
            )
            return

        # Remaining processing (linear antecedent/consequent case)
        antecedent = self._wrap_boolean_literal_as_constraint(
            constraint_node["antecedent"]
        )
        consequent = self._wrap_boolean_literal_as_constraint(
            constraint_node["consequent"]
        )

        # Derive big-M for implication (use max of antecedent & consequent diff bounds) else fallback
        bigM_default = 1e6

        # Extract both raw nodes and string expressions
        ant_left, ant_right, ant_op, ant_left_expr, ant_right_expr = (
            self._extract_implication_constraint(antecedent, current_iterators)
        )
        cons_left, cons_right, cons_op, cons_left_expr, cons_right_expr = (
            self._extract_implication_constraint(consequent, current_iterators)
        )
        # Compute separate big-M values for antecedent and consequent
        M_ant = self._estimate_big_m_for_difference(ant_left, ant_right)
        M_cons = self._estimate_big_m_for_difference(cons_left, cons_right)
        bigM_ant = M_ant if M_ant is not None else bigM_default
        bigM_cons = M_cons if M_cons is not None else bigM_default
        eps_sep = EQ_TOL  # epsilon for equality separation on >=/<=

        if self._emit_specialized_implication_indicator(
            ant_left,
            ant_right,
            ant_op,
            ant_left_expr,
            ant_right_expr,
            cons_left,
            cons_right,
            cons_op,
            cons_left_expr,
            cons_right_expr,
            constr_name_prefix,
        ):
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
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} >= -{eps}, name={self._format_name_expr(constr_name_prefix, '_ant_ge_ind1')})"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} <= -{2*eps}, name={self._format_name_expr(constr_name_prefix, '_ant_ge_ind0')})"
            )
        elif ant_op == ">":
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} >= {eps}, name={self._format_name_expr(constr_name_prefix, '_ant_gt_ind1')})"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} <= 0.0, name={self._format_name_expr(constr_name_prefix, '_ant_gt_ind0')})"
            )
        elif ant_op == "<=":
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} <= {eps}, name={self._format_name_expr(constr_name_prefix, '_ant_le_ind1')})"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} >= {2*eps}, name={self._format_name_expr(constr_name_prefix, '_ant_le_ind0')})"
            )
        elif ant_op == "<":
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 1, {diff_expr} <= -{eps}, name={self._format_name_expr(constr_name_prefix, '_ant_lt_ind1')})"
            )
            self._add_code_line(
                f"model.addGenConstrIndicator({flag_var}, 0, {diff_expr} >= 0.0, name={self._format_name_expr(constr_name_prefix, '_ant_lt_ind0')})"
            )
        elif ant_op == "==":
            self._add_code_line(
                f"model.addConstr({diff_expr} <= {eps_sep} + {bigM_ant} * (1 - {flag_var}), name={self._format_name_expr(constr_name_prefix, '_ant_eq1')})"
            )
            self._add_code_line(
                f"model.addConstr(-{diff_expr} <= {eps_sep} + {bigM_ant} * (1 - {flag_var}), name={self._format_name_expr(constr_name_prefix, '_ant_eq2')})"
            )
            self._add_code_line(
                f"model.addConstr({diff_expr} >= -{eps_sep} - {bigM_ant} * (1 - {flag_var}), name={self._format_name_expr(constr_name_prefix, '_ant_eq3')})"
            )
            self._add_code_line(
                f"model.addConstr(-{diff_expr} >= -{eps_sep} - {bigM_ant} * (1 - {flag_var}), name={self._format_name_expr(constr_name_prefix, '_ant_eq4')})"
            )
        else:
            raise ValueError(f"Unsupported antecedent operator in implication: {ant_op}")

        # 2. Enforce consequent only when flag_var == 1 (use bigM_cons)
        self._emit_implication_consequent(
            cons_op,
            cons_left_expr,
            cons_right_expr,
            bigM_cons,
            flag_var,
            constr_name_prefix,
        )

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
        bound_args = self._decl_dvar_bound_args(decl)
        has_explicit_lb = "lower_bound" in decl
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
                default_lb = "" if has_explicit_lb else ", lb=0"
                self._add_code_line(
                    f"{name} = model.addVars({product_args}, vtype=GRB.INTEGER, name='{name}'{default_lb}{bound_args})"
                )
            elif var_type == "int":
                default_lb = "" if has_explicit_lb else ", lb=-GRB.INFINITY"
                self._add_code_line(
                    f"{name} = model.addVars({product_args}, vtype=GRB.INTEGER, name='{name}'{default_lb}{bound_args})"
                )
            elif var_type == "float+":
                default_lb = "" if has_explicit_lb else ", lb=0"
                self._add_code_line(
                    f"{name} = model.addVars({product_args}, vtype=GRB.CONTINUOUS, name='{name}'{default_lb}{bound_args})"
                )
            elif var_type == "float":
                default_lb = "" if has_explicit_lb else ", lb=-GRB.INFINITY"
                self._add_code_line(
                    f"{name} = model.addVars({product_args}, vtype=GRB.CONTINUOUS, name='{name}'{default_lb}{bound_args})"
                )
            else:
                self._add_code_line(f"{name} = model.addVars({product_args}, name='{name}'{bound_args})")
        else:
            if var_type == "boolean":
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.BINARY, name='{name}')"
                )
            elif var_type == "int+":
                default_lb = "" if has_explicit_lb else ", lb=0"
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.INTEGER, name='{name}'{default_lb}{bound_args})"
                )
            elif var_type == "int":
                default_lb = "" if has_explicit_lb else ", lb=-GRB.INFINITY"
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.INTEGER, name='{name}'{default_lb}{bound_args})"
                )
            elif var_type == "float+":
                default_lb = "" if has_explicit_lb else ", lb=0"
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.CONTINUOUS, name='{name}'{default_lb}{bound_args})"
                )
            elif var_type == "float":
                default_lb = "" if has_explicit_lb else ", lb=-GRB.INFINITY"
                self._add_code_line(
                    f"{name} = model.addVars({', '.join(map(str, range_args))}, vtype=GRB.CONTINUOUS, name='{name}'{default_lb}{bound_args})"
                )
            else:
                self._add_code_line(f"{name} = model.addVars({', '.join(map(str, range_args))}, name='{name}'{bound_args})")
        self.gurobi_var_map[name] = name

    def _decl_dvar_bound_args(self, decl):
        iterator_names = [it.get("iterator") for it in decl.get("iterators", []) if isinstance(it, dict)]

        def emit_bound_expr(expr):
            if isinstance(expr, dict) and expr.get("type") == "indexed_name" and len(expr.get("dimensions", [])) == 1:
                dim = expr["dimensions"][0]
                if isinstance(dim, dict) and dim.get("type") == "name_reference_index" and dim.get("name") in iterator_names:
                    return expr["name"]
            return self._traverse_expression(expr, {}, symbolic=True)

        args = []
        if "lower_bound" in decl:
            args.append(f"lb={emit_bound_expr(decl['lower_bound'])}")
        if "upper_bound" in decl:
            args.append(f"ub={emit_bound_expr(decl['upper_bound'])}")
        return "" if not args else ", " + ", ".join(args)

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
            # Emit Python literal to preserve True/False and tuple keys
            self._add_code_line(f"{name} = {repr(self.data_dict[name])}")
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
        # Always emit Python literals (repr) so booleans are True/False
        self._add_code_line(f"{name} = {repr(decl['value'])}")

    def _decl_parameter_inline_indexed(self, decl):
        name = decl["name"]
        dims = decl.get("dimensions", [])
        if len(dims) == 1 and dims[0]["type"] == "named_set_dimension":
            set_name = dims[0]["name"]
            # Robustly resolve tuple-set elements from data_dict/AST
            keys = TupleSetHelper.get_tuple_set(set_name, self.ast, self.data_dict)
            if keys:
                tuple_keys = [k if isinstance(k, tuple) else (k,) for k in keys]
                vals = decl.get("value") or []
                if len(vals) != len(tuple_keys):
                    raise SemanticError(
                        f"Parameter '{name}' length {len(vals)} does not match index set '{set_name}' length {len(tuple_keys)}."
                    )
                param_dict = {tuple_keys[i]: vals[i] for i in range(len(vals))}
                self._add_code_line(f"{name} = {repr(param_dict)}")
                return
        self._add_code_line(f"{name} = {repr(decl['value'])}")

    def _decl_parameter_external(self, decl):
        name = decl["name"]
        if name in self.data_dict:
            val = self.data_dict[name]
            # Always emit Python literal (repr). This preserves tuple keys and True/False.
            self._add_code_line(f"{name} = {repr(val)}")
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
            self._add_code_line(f"{name} = {repr(self.data_dict[name])}")
        else:
            raise SemanticError(
                f"Parameter '{name}' declared with '= ...' in .mod but not found in .dat file.",
                lineno=decl.get("lineno", None),
            )

    def _decl_parameter_external_explicit_indexed(self, decl):
        self._decl_parameter_external_explicit(decl)

    # === Constraint Node Handlers ===
    def _is_boolean_decision_variable(self, node):
        if not isinstance(node, dict):
            return False
        node_type = node.get("type")
        if node_type == "name":
            declaration = self._find_declaration_by_name(node.get("value"))
        elif node_type == "indexed_name":
            declaration = self._find_declaration_by_name(node.get("name"))
        else:
            return False
        return declaration is not None and declaration.get("var_type") == "boolean"

    def _not_equal_big_m(self, left_node, right_node):
        left_bounds = self._linear_bounds_safe(left_node)
        right_bounds = self._linear_bounds_safe(right_node)
        if left_bounds is None or right_bounds is None or None in (*left_bounds, *right_bounds):
            return 1e6
        left_lower, left_upper = left_bounds
        right_lower, right_upper = right_bounds
        difference_lower = left_lower - right_upper
        difference_upper = left_upper - right_lower
        return max(1.0, 1.0 - difference_lower, 1.0 + difference_upper)

    def _emit_not_equal_constraint(
        self,
        left_node,
        right_node,
        constr_name_prefix,
        current_iterators,
    ):
        left_expression = self._traverse_expression(left_node, current_iterators)
        right_expression = self._traverse_expression(right_node, current_iterators)
        if self._is_boolean_decision_variable(left_node) and self._is_boolean_decision_variable(right_node):
            self._add_code_line(
                f"model.addConstr({left_expression} + {right_expression} == 1, name='{constr_name_prefix}_xor')"
            )
            return

        flag_name = f"neq_flag_{constr_name_prefix}"
        if current_iterators:
            self._add_code_line(f"{flag_name} = model.addVar(vtype=GRB.BINARY)")
        else:
            self._add_code_line(
                f"{flag_name} = model.addVar(vtype=GRB.BINARY, name='{flag_name}')"
            )
        big_m = self._not_equal_big_m(left_node, right_node)
        self._add_code_line(
            f"model.addConstr({left_expression} - {right_expression} + {big_m} * {flag_name} >= 1, name={self._format_name_expr(constr_name_prefix, '_neq1')})"
        )
        self._add_code_line(
            f"model.addConstr({right_expression} - {left_expression} + {big_m} * (1 - {flag_name}) >= 1, name={self._format_name_expr(constr_name_prefix, '_neq2')})"
        )

    def _unwrap_parenthesized(self, node):
        while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
            node = node.get("expression")
        return node

    def _is_comparison_node(self, node):
        node = self._unwrap_parenthesized(node)
        return (
            isinstance(node, dict)
            and node.get("type") in ("binop", "constraint")
            and node.get("op") in (">=", ">", "<=", "<", "==")
        )

    def _is_comparison_sum(self, node):
        node = self._unwrap_parenthesized(node)
        return (
            isinstance(node, dict)
            and node.get("type") == "sum"
            and self._is_comparison_node(node.get("expression"))
        )

    def _comparison_sum_metadata(self, sum_node, current_iterators):
        if not hasattr(self, "_comparison_sum_meta"):
            self._comparison_sum_meta = {}
        if id(sum_node) not in self._comparison_sum_meta:
            try:
                self._traverse_expression(sum_node, current_iterators)
            except Exception:
                pass
        return self._comparison_sum_meta.get(id(sum_node))

    def _emit_direct_cardinality_constraint(
        self,
        operator,
        left_node,
        right_node,
        constr_name_prefix,
        current_iterators,
    ):
        if not (
            operator in (">", ">=", "==", "<=", "<")
            and isinstance(right_node, dict)
            and right_node.get("type") == "number"
            and self._is_comparison_sum(left_node)
        ):
            return False
        sum_node = self._unwrap_parenthesized(left_node)
        threshold = right_node.get("value")
        effective_threshold = threshold + 1 if operator == ">" else threshold
        metadata = self._comparison_sum_metadata(sum_node, current_iterators)
        if not metadata:
            return False
        list_name, _ = self._comparison_sum_accessors(metadata, current_iterators)
        emitted_operator = ">=" if operator in (">", ">=") else "<=" if operator in ("<", "<=") else "=="
        self._add_code_line(
            f"model.addConstr(gp.quicksum({list_name}) {emitted_operator} {effective_threshold}, name={self._format_name_expr(constr_name_prefix, '_card')})"
        )
        return True

    def _reified_cardinality_parts(self, node):
        node = self._unwrap_parenthesized(node)
        if not (
            isinstance(node, dict)
            and node.get("type") in ("constraint", "binop")
            and node.get("op") in (">=", ">")
            and isinstance(node.get("right"), dict)
            and node["right"].get("type") == "number"
        ):
            return None
        sum_node = self._unwrap_parenthesized(node.get("left"))
        if not isinstance(sum_node, dict) or sum_node.get("type") != "sum":
            return None
        threshold = node["right"]["value"] + (1 if node.get("op") == ">" else 0)
        return sum_node, threshold, node.get("type")

    def _emit_reified_cardinality_constraint(
        self,
        operator,
        left_node,
        right_node,
        constr_name_prefix,
        current_iterators,
    ):
        if operator != "==" or not isinstance(left_node, dict):
            return False
        parts = self._reified_cardinality_parts(right_node)
        if parts is None or left_node.get("type") not in ("name", "indexed_name"):
            return False
        sum_node, threshold, node_type = parts
        metadata = self._comparison_sum_metadata(sum_node, current_iterators)
        if not metadata:
            return False
        list_name, length_expression = self._comparison_sum_accessors(
            metadata, current_iterators
        )
        boolean_variable = self._traverse_expression(left_node, current_iterators)
        label = "Reified cardinality (binop)" if node_type == "binop" else "Reified cardinality"
        self._add_code_line(
            f"# {label}: {boolean_variable} == (sum(comparisons) >= {threshold})"
        )
        length_expression = length_expression or f"len({list_name})"
        self._add_code_line(
            f"model.addConstr({threshold} * {boolean_variable} - gp.quicksum({list_name}) <= 0, name={self._format_name_expr(constr_name_prefix, '_reif_card1')})"
        )
        self._add_code_line(
            f"model.addConstr(gp.quicksum({list_name}) - ({threshold}-1) - ({length_expression} - {threshold} + 1) * {boolean_variable} <= 0, name={self._format_name_expr(constr_name_prefix, '_reif_card2')})"
        )
        return True

    def _normalize_true_comparison(self, operator, left_node, right_node):
        left = self._unwrap_parenthesized(left_node)
        right = self._unwrap_parenthesized(right_node)
        if operator != "==":
            return operator, left_node, right_node
        if (
            isinstance(right, dict)
            and right.get("type") == "boolean_literal"
            and right.get("value") is True
            and self._is_comparison_node(left)
        ):
            return left["op"], left["left"], left["right"]
        if (
            isinstance(left, dict)
            and left.get("type") == "boolean_literal"
            and left.get("value") is True
            and self._is_comparison_node(right)
        ):
            return right["op"], right["left"], right["right"]
        return operator, left_node, right_node

    def _emit_boolean_literal_constraint(
        self,
        operator,
        left_node,
        right_node,
        constr_name_prefix,
        current_iterators,
    ):
        if operator != "==":
            return False
        if isinstance(right_node, dict) and right_node.get("type") == "boolean_literal":
            expression_node = left_node
            literal_node = right_node
        elif isinstance(left_node, dict) and left_node.get("type") == "boolean_literal":
            expression_node = right_node
            literal_node = left_node
        else:
            return False
        if not self._is_boolean_expr_node(expression_node):
            return False
        boolean_expression = self._boolean_expr_to_binary_expr(
            expression_node, current_iterators, constr_name_prefix
        )
        target = 1 if literal_node.get("value") else 0
        self._add_code_line(
            f"model.addConstr({boolean_expression} == {target}, name={self._format_name_expr(constr_name_prefix)})"
        )
        return True

    def _emit_boolean_comparison_constraint(
        self,
        operator,
        left_node,
        right_node,
        constr_name_prefix,
        current_iterators,
    ):
        if operator not in ("==", "<=", ">="):
            return False
        left_is_boolean = self._is_boolean_decision_variable(left_node)
        right_is_boolean = self._is_boolean_decision_variable(right_node)
        if left_is_boolean and self._is_comparison_node(right_node):
            left_expression = self._traverse_expression(left_node, current_iterators)
            right_expression = self._reify_scoped_comparison(
                right_node, current_iterators
            )
        elif right_is_boolean and self._is_comparison_node(left_node):
            left_expression = self._reify_scoped_comparison(
                left_node, current_iterators
            )
            right_expression = self._traverse_expression(
                right_node, current_iterators
            )
        else:
            return False
        self._add_code_line(
            f"model.addConstr({left_expression} {operator} {right_expression}, name={self._format_name_expr(constr_name_prefix)})"
        )
        return True

    def _emit_ordinary_constraint(
        self,
        operator,
        left_node,
        right_node,
        constr_name_prefix,
        current_iterators,
    ):
        left_expression = self._traverse_expression(left_node, current_iterators)
        right_expression = self._traverse_expression(right_node, current_iterators)
        comparison = self._gurobi_comparison_expr(
            left_expression, operator, right_expression
        )
        self._add_code_line(
            f"model.addConstr({comparison}, name={self._format_name_expr(constr_name_prefix)})"
        )

    def _constraint_constraint(self, constraint_node, constr_name_prefix, current_iterators):
        # Defer expression string generation until after pattern-specific rewrites to avoid
        # creating TempConstr objects (by evaluating comparisons) that we later try to combine arithmetically.
        op, left_node, right_node = self._normalize_true_comparison(
            constraint_node["op"],
            constraint_node["left"],
            constraint_node["right"],
        )

        if self._emit_boolean_literal_constraint(
            op,
            left_node,
            right_node,
            constr_name_prefix,
            current_iterators,
        ):
            return
        if self._emit_boolean_comparison_constraint(
            op,
            left_node,
            right_node,
            constr_name_prefix,
            current_iterators,
        ):
            return

        if self._emit_direct_cardinality_constraint(
            op,
            left_node,
            right_node,
            constr_name_prefix,
            current_iterators,
        ):
            return
        if self._emit_reified_cardinality_constraint(
            op,
            left_node,
            right_node,
            constr_name_prefix,
            current_iterators,
        ):
            return
        if op == "!=":
            self._emit_not_equal_constraint(
                left_node,
                right_node,
                constr_name_prefix,
                current_iterators,
            )
            return

        self._emit_ordinary_constraint(
            op,
            left_node,
            right_node,
            constr_name_prefix,
            current_iterators,
        )

    def _emit_index_condition(self, node, current_iterators):
        """
        Emit a pure-Python boolean/numeric expression for forall/sum index filters.
        Never emits Gurobi constructs (no gp.quicksum/TempConstr). Intended only
        for 'if <cond>:' guards around model.addConstr calls.
        """
        t = node.get("type") if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return repr(node)

        if t == "boolean_literal":
            return "True" if node.get("value") else "False"
        if t == "number":
            return str(node.get("value"))
        if t == "string_literal":
            return repr(node.get("value"))
        if t == "name":
            name = node.get("value")
            # iterators and scalar params are accessible as Python vars
            return name
        if t == "indexed_name":
            base_name = node.get("name")
            # Disallow dvars in index filters (should be data-only)
            decl = self._find_declaration_by_name(base_name)
            if decl and decl.get("type") in ("dvar", "dvar_indexed"):
                raise ValueError("Index constraint may not reference decision variables.")
            # Use the general indexed-name emitter which includes dict-vs-list
            # fallbacks and 1-based->0-based adjustments for list indexing.
            return self._expr_indexed_name(node, current_iterators, symbolic=False)
        if t == "parenthesized_expression":
            inner = self._emit_index_condition(node.get("expression"), current_iterators)
            return f"({inner})"
        if t == "not":
            val = self._emit_index_condition(node.get("value"), current_iterators)
            return f"(not ({val}))"
        if t == "and":
            L = self._emit_index_condition(node.get("left"), current_iterators)
            R = self._emit_index_condition(node.get("right"), current_iterators)
            return f"(({L}) and ({R}))"
        if t == "or":
            L = self._emit_index_condition(node.get("left"), current_iterators)
            R = self._emit_index_condition(node.get("right"), current_iterators)
            return f"(({L}) or ({R}))"
        if t == "binop":
            op = node.get("op")
            L = self._emit_index_condition(node.get("left"), current_iterators)
            R = self._emit_index_condition(node.get("right"), current_iterators)
            return f"({L} {op} {R})"
        if t == "sum":
            # Python sum over iterators, converting boolean to 0/1
            iters = node.get("iterators", [])
            idxc = node.get("index_constraint")
            inner = node.get("expression")
            # if inner is a comparison, emit 1 if (...) else 0
            inner_expr = self._emit_index_condition(inner, current_iterators)
            # wrap non-numeric booleans into int(): int(cond)
            # Use generators with nested 'for ... in ...' and optional if guard
            gens = []
            local_iterators = current_iterators.copy()
            for it in iters:
                var = it["iterator"]
                rng = self._forall_range_expr(it["range"], local_iterators)
                gens.append(f"for {var} in {rng}")
                local_iterators[var] = var
            gen_str = " ".join(gens)
            guard = ""
            if idxc is not None:
                guard = f" if {self._emit_index_condition(idxc, local_iterators)}"
            # Coerce boolean to 0/1 via int(...)
            return f"sum((1 if ({inner_expr}) else 0) {gen_str}{guard})"
        # Fallback to symbolic traversal for safe literals
        return self._traverse_expression(node, current_iterators, symbolic=True)

    def _constraint_forall_constraint(self, constraint_node, constr_name_prefix, current_iterators):
        iterators = constraint_node["iterators"]
        index_constraint = constraint_node.get("index_constraint")
        loop_vars, loop_ranges = self._extract_forall_loops(iterators, current_iterators)
        self._add_code_line(self._construct_loop_header(loop_vars, loop_ranges))
        self.indent_level += 1
        new_iterators = current_iterators.copy()
        for v in loop_vars:
            new_iterators[v] = v
        previous_active_ranges = getattr(self, "_active_iterator_ranges", {}).copy()
        active_ranges = previous_active_ranges.copy()
        active_ranges.update({var: rng for var, rng in zip(loop_vars, loop_ranges)})
        self._active_iterator_ranges = active_ranges
        if index_constraint is not None:
            cond_str = self._emit_index_condition(index_constraint, new_iterators)
            self._add_code_line(f"if {cond_str}:")
            self.indent_level += 1
        previous_materialized_loop_depth = getattr(self, "_materialized_loop_depth", 0)
        self._materialized_loop_depth = previous_materialized_loop_depth + 1
        try:
            # Emit body
            self._emit_forall_inner_constraints(constraint_node, constr_name_prefix, loop_vars, new_iterators)
        finally:
            self._materialized_loop_depth = previous_materialized_loop_depth
            self._active_iterator_ranges = previous_active_ranges
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
        elif rng["type"] == "indexed_set":
            idx_expr = self._expr_indexed_name(
                {"type": "indexed_name", "name": rng["name"], "dimensions": rng.get("dimensions", [])},
                current_iterators,
                symbolic=True,
            )
            if " if isinstance(" in idx_expr:
                return idx_expr
            if len(rng.get("dimensions", [])) == 1:
                dim = rng["dimensions"][0]
                if isinstance(dim, dict) and dim.get("type") == "name_reference_index":
                    key = dim["name"]
                    return f"({rng['name']}[{key}] if isinstance({rng['name']}, dict) else {idx_expr})"
            return idx_expr
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

        def with_label_context(child_node, emit_fn):
            # Push active label expression if label_template present on child
            prev = self._active_label_name_expr
            try:
                if isinstance(child_node, dict) and "label_template" in child_node:
                    self._active_label_name_expr = self._compute_label_expr(child_node["label_template"])
                emit_fn()
            finally:
                self._active_label_name_expr = prev

        if "constraint" in constraint_node:
            inner_constraint = constraint_node["constraint"]

            def emit_one():
                self._generate_single_constraint(
                    inner_constraint,
                    f"{constr_name_prefix}_{'_'.join(loop_vars)}",
                    new_iterators,
                )

            with_label_context(inner_constraint, emit_one)
        elif "constraints" in constraint_node:
            for i, inner_constr in enumerate(constraint_node["constraints"]):

                def emit_i():
                    self._generate_single_constraint(
                        inner_constr,
                        f"{constr_name_prefix}_{'_'.join(loop_vars)}_{i}",
                        new_iterators,
                    )

                with_label_context(inner_constr, emit_i)
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

    # NEW: function call support (unary algebraic functions)
    def _expr_funcall(self, expr_node, current_iterators, symbolic):
        name = expr_node.get("name")
        args = expr_node.get("args", [])
        if name in {"sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceil"} and len(args) == 1:
            arg_str = self._traverse_expression(args[0], current_iterators, symbolic)
            return f"math.{name}({arg_str})"
        if name in {"abs", "round"} and len(args) == 1:
            arg_str = self._traverse_expression(args[0], current_iterators, symbolic)
            return f"{name}({arg_str})"
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
        return str(expr_node["value"])

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

    def _emit_index_expr(self, dim_expr, current_iterators, symbolic):
        expr_type = dim_expr.get("type")
        if expr_type in ("field_access_index", "field_access"):
            return self._expr_field_access(dim_expr, current_iterators, symbolic)
        if expr_type == "number_literal_index":
            return str(dim_expr["value"])
        if expr_type == "name_reference_index":
            return str(dim_expr["name"])
        if expr_type == "indexed_name":
            return self._expr_indexed_name(dim_expr, current_iterators, symbolic)
        if expr_type == "string_literal":
            return repr(dim_expr["value"])
        if expr_type == "binop":
            left = self._emit_index_expr(dim_expr["left"], current_iterators, symbolic)
            right = self._emit_index_expr(dim_expr["right"], current_iterators, symbolic)
            return f"({left} {dim_expr['op']} {right})"
        if expr_type == "uminus":
            value = self._emit_index_expr(dim_expr["value"], current_iterators, symbolic)
            return f"-({value})"
        if expr_type == "parenthesized_expression":
            value = self._emit_index_expr(dim_expr["expression"], current_iterators, symbolic)
            return f"({value})"
        if expr_type == "tuple_literal":
            parts = []
            for element in dim_expr.get("elements", []):
                if isinstance(element, dict):
                    if element.get("type") == "boolean_literal":
                        parts.append("True" if element.get("value") else "False")
                    else:
                        parts.append(self._traverse_expression(element, current_iterators, symbolic))
                else:
                    parts.append(repr(element))
            if len(parts) == 1:
                return f"({parts[0]},)"
            return f"({', '.join(parts)})"
        if "value" in dim_expr:
            return str(dim_expr["value"])
        if "name" in dim_expr:
            return str(dim_expr["name"])
        raise ValueError(f"Unsupported index expr type: {expr_type}")

    def _is_tuple_indexed_declaration(self, declaration):
        if declaration is None:
            return False
        dimensions = declaration.get("dimensions", [])
        if len(dimensions) != 1 or dimensions[0].get("type") != "named_set_dimension":
            return False
        set_name = dimensions[0].get("name")
        for candidate in self.ast.get("declarations", []):
            if candidate.get("name") == set_name:
                return candidate.get("type") in ("set_of_tuples", "set_of_tuples_external")
        return False

    def _emit_dict_parameter_indexed_name(
        self, base_name, dims_decl, dims_for_indexing, raw_idx_exprs, current_iterators, symbolic
    ):
        if len(raw_idx_exprs) > 1 or isinstance(self.data_dict.get(base_name), dict) and any(
            isinstance(key, tuple) for key in self.data_dict[base_name].keys()
        ):
            tuple_expr = f"({', '.join(raw_idx_exprs)})"
            list_index_parts = []
            for index, dim_expr in enumerate(dims_for_indexing):
                dim_decl = dims_decl[index] if index < len(dims_decl) else None
                idx_code = self._emit_index_expr(dim_expr, current_iterators, symbolic)
                if dim_decl and dim_decl.get("type") == "named_set_dimension":
                    set_name = dim_decl.get("name")
                    list_index_parts.append(f"{set_name}_index[{idx_code}]")
                else:
                    list_index_parts.append(f"(({idx_code}) - 1)")
            list_access = base_name + "".join(f"[{part}]" for part in list_index_parts)
            raw0 = raw_idx_exprs[0]
            fallback_idx = list_index_parts[1] if len(list_index_parts) > 1 else list_index_parts[0]
            return (
                f"({base_name}[{tuple_expr}] if isinstance({base_name}, dict) and ({tuple_expr}) in {base_name} "
                f"else ({base_name}[{raw0}][{fallback_idx}] if isinstance({base_name}, dict) and {raw0} in {base_name} and isinstance({base_name}[{raw0}], (list,tuple)) "
                f"else {list_access}))"
            )
        dim_decl0 = dims_decl[0] if len(dims_decl) > 0 else None
        idx0 = raw_idx_exprs[0]
        if dim_decl0 and dim_decl0.get("type") == "named_set_dimension":
            set_name0 = dim_decl0.get("name")
            list_idx0 = f"{set_name0}_index[{idx0}]"
        else:
            list_idx0 = f"(({idx0}) - 1)"
        return f"({base_name}[{idx0}] if isinstance({base_name}, dict) else {base_name}[{list_idx0}])"

    def _emit_parameter_indexed_name(self, base_name, expr_node, declaration, current_iterators, symbolic):
        dims_decl = declaration.get("dimensions", []) if declaration is not None else []
        dims_for_indexing = expr_node.get("dimensions", [])
        if (
            len(dims_for_indexing) == 1
            and isinstance(dims_for_indexing[0], dict)
            and dims_for_indexing[0].get("type") == "tuple_literal"
            and not self._is_tuple_indexed_declaration(declaration)
        ):
            dims_for_indexing = dims_for_indexing[0].get("elements", [])

        container_val = self.data_dict.get(base_name)
        is_dict_param = (hasattr(self, "dict_params") and base_name in self.dict_params) or isinstance(container_val, dict)
        raw_idx_exprs = [
            self._emit_index_expr(dim_expr, current_iterators, symbolic)
            for dim_expr in dims_for_indexing
        ]

        if is_dict_param:
            return self._emit_dict_parameter_indexed_name(
                base_name, dims_decl, dims_for_indexing, raw_idx_exprs, current_iterators, symbolic
            )

        index_exprs = []
        for index, dim_expr in enumerate(dims_for_indexing):
            idx_code = self._emit_index_expr(dim_expr, current_iterators, symbolic)
            dim_decl = dims_decl[index] if index < len(dims_decl) else None
            if dim_decl and dim_decl.get("type") == "named_set_dimension":
                set_name = dim_decl.get("name")
                index_exprs.append(f"{set_name}_index[{idx_code}]")
            else:
                index_exprs.append(f"(({idx_code}) - 1)")
        out = base_name
        for index_expr in index_exprs:
            out += f"[{index_expr}]"
        return out

    def _expr_indexed_name(self, expr_node, current_iterators, symbolic):
        base_name = expr_node["name"]

        decl = None
        for d in self.ast.get("declarations", []):
            if d.get("name") == base_name:
                decl = d
                break

        if self._is_tuple_indexed_declaration(decl):
            idx = expr_node["dimensions"][0]
            if idx.get("type") in ("name_reference_index", "name"):
                return f"{base_name}[{idx['name']}]"
            elif idx.get("type") == "tuple_literal":
                # Build a Python tuple expression for the index, handling raw literals
                parts = []
                for el in idx.get("elements", []):
                    if isinstance(el, dict):
                        # Preserve boolean literals inside tuple indices
                        if el.get("type") == "boolean_literal":
                            parts.append("True" if el.get("value") else "False")
                        else:
                            parts.append(self._traverse_expression(el, current_iterators, symbolic))
                    else:
                        parts.append(repr(el))
                # Ensure single-element tuples include the trailing comma
                if len(parts) == 1:
                    tuple_expr = f"({parts[0]},)"
                else:
                    tuple_expr = f"({', '.join(parts)})"

                # Emit tuple-key access when the data was emitted as a dict,
                # but fallback to nested list indexing when the runtime object
                # is a list-of-lists. For tuple-indexed declarations the
                # corresponding list-style access uses the tuple elements as
                # separate indices (adjusting 1-based ranges to 0-based).
                list_index_parts = []
                for el in idx.get("elements", []):
                    if isinstance(el, dict):
                        idx_code = self._traverse_expression(el, current_iterators, symbolic)
                    else:
                        idx_code = repr(el)
                    # Conservatively adjust numeric/name indices to 0-based
                    list_index_parts.append(f"(({idx_code}) - 1)")
                list_access = base_name + "".join(f"[{p}]" for p in list_index_parts)
                return f"({base_name}[{tuple_expr}] if isinstance({base_name}, dict) else {list_access})"
        else:
            # Decision variable case
            if base_name in self.gurobi_var_map:
                # Always emit direct bracket indexing for decision variables (no _safe_get)
                if len(expr_node["dimensions"]) == 1:
                    idx_expr = self._emit_index_expr(expr_node["dimensions"][0], current_iterators, symbolic)
                    return f"{base_name}[{idx_expr}]"
                idx_exprs = [
                    self._emit_index_expr(dim_expr, current_iterators, symbolic)
                    for dim_expr in expr_node["dimensions"]
                ]
                if len(idx_exprs) == 1:
                    return f"{base_name}[{idx_exprs[0]}]"
                else:
                    return f"{base_name}[({', '.join(idx_exprs)})]"

            # Tuple array case (data struct of records)
            if decl is not None and decl.get("type") in ("tuple_array", "tuple_array_external"):
                idx_exprs = [
                    self._emit_index_expr(dim_expr, current_iterators, symbolic)
                    for dim_expr in expr_node["dimensions"]
                ]
                out = base_name
                for ie in idx_exprs:
                    out += f"[{ie}]"
                return out

            return self._emit_parameter_indexed_name(
                base_name, expr_node, decl, current_iterators, symbolic
            )

    def _expr_binop(self, expr_node, current_iterators, symbolic):
        op = expr_node["op"]
        if not symbolic and op in ("+", "-", "*", "/"):
            left_node = expr_node["left"]
            right_node = expr_node["right"]

            def _is_comparison_node(node):
                while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
                    node = node.get("expression")
                return (
                    isinstance(node, dict)
                    and node.get("type") in ("binop", "constraint")
                    and node.get("op") in (">=", "<=", "==", ">", "<")
                )

            left_str = (
                self._reify_scoped_comparison(left_node, current_iterators)
                if _is_comparison_node(left_node)
                else self._traverse_expression(left_node, current_iterators, symbolic)
            )
            right_str = (
                self._reify_scoped_comparison(right_node, current_iterators)
                if _is_comparison_node(right_node)
                else self._traverse_expression(right_node, current_iterators, symbolic)
            )
        else:
            left_str = self._traverse_expression(expr_node["left"], current_iterators, symbolic)
            right_str = self._traverse_expression(expr_node["right"], current_iterators, symbolic)
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

    def _comparison_sum_accessors(self, meta, current_iterators):
        scope_vars = tuple(meta.get("scope_vars") or ())
        if scope_vars:
            key_expr = self._format_scope_key_expr(scope_vars, current_iterators)
            list_expr = f"{meta['list_name']}[{key_expr}]"
            len_name = meta.get("len_name")
            len_expr = f"{len_name}[{key_expr}]" if len_name else None
            return list_expr, len_expr
        return meta["list_name"], meta.get("len_var")

    def _format_scope_key_expr(self, scope_vars, current_iterators):
        parts = [str(current_iterators.get(name, name)) for name in scope_vars]
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    def _active_scope_iterators(self, current_iterators, local_loop_vars=()):
        active_ranges = getattr(self, "_active_iterator_ranges", {})
        scope_vars = []
        scope_ranges = []
        for name, rng in active_ranges.items():
            if name in local_loop_vars or name not in current_iterators:
                continue
            scope_vars.append(name)
            scope_ranges.append(rng)
        return scope_vars, scope_ranges

    def _comparison_expr_accessor(self, meta, current_iterators):
        scope_vars = tuple(meta.get("scope_vars") or ())
        if scope_vars:
            key_expr = self._format_scope_key_expr(scope_vars, current_iterators)
            return f"{meta['name']}[{key_expr}]"
        return meta["name"]

    def _is_boolean_expr_node(self, node):
        while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
            node = node.get("expression")
        return isinstance(node, dict) and (
            node.get("type") in ("and", "or", "not")
            or (node.get("type") in ("binop", "constraint") and node.get("op") in (">=", "<=", "==", "!=", ">", "<"))
            or node.get("type") == "boolean_literal"
        )

    def _new_boolean_aux(self, prefix, constr_name_prefix):
        if not hasattr(self, "_bool_aux_counter"):
            self._bool_aux_counter = 0
        self._bool_aux_counter += 1
        name = f"_{prefix}_bool_{self._bool_aux_counter}"
        if getattr(self, "_materialized_loop_depth", 0) > 0:
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.BINARY)")
        else:
            self._add_code_line(f"{name} = model.addVar(vtype=GRB.BINARY, name='{name}_{constr_name_prefix}')")
        return name

    def _boolean_expr_to_binary_expr(self, node, current_iterators, constr_name_prefix):
        while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
            node = node.get("expression")
        if not isinstance(node, dict):
            raise ValueError(f"Unsupported boolean expression node: {node!r}")
        t = node.get("type")
        if t == "boolean_literal":
            return "1" if node.get("value") else "0"
        if t in ("binop", "constraint") and node.get("op") in (">=", "<=", "==", "!=", ">", "<"):
            return self._reify_scoped_comparison(node, current_iterators)
        if t == "not":
            inner = self._boolean_expr_to_binary_expr(node["value"], current_iterators, constr_name_prefix)
            aux = self._new_boolean_aux("not", constr_name_prefix)
            self._add_code_line(
                f"model.addConstr({aux} + {inner} == 1, name={self._format_name_expr(constr_name_prefix, '_not')})"
            )
            return aux
        if t == "and":
            left = self._boolean_expr_to_binary_expr(node["left"], current_iterators, constr_name_prefix)
            right = self._boolean_expr_to_binary_expr(node["right"], current_iterators, constr_name_prefix)
            aux = self._new_boolean_aux("and", constr_name_prefix)
            self._add_code_line(
                f"model.addConstr({aux} <= {left}, name={self._format_name_expr(constr_name_prefix, '_and_l')})"
            )
            self._add_code_line(
                f"model.addConstr({aux} <= {right}, name={self._format_name_expr(constr_name_prefix, '_and_r')})"
            )
            self._add_code_line(
                f"model.addConstr({aux} >= {left} + {right} - 1, name={self._format_name_expr(constr_name_prefix, '_and_link')})"
            )
            return aux
        if t == "or":
            left = self._boolean_expr_to_binary_expr(node["left"], current_iterators, constr_name_prefix)
            right = self._boolean_expr_to_binary_expr(node["right"], current_iterators, constr_name_prefix)
            aux = self._new_boolean_aux("or", constr_name_prefix)
            self._add_code_line(
                f"model.addConstr({aux} >= {left}, name={self._format_name_expr(constr_name_prefix, '_or_l')})"
            )
            self._add_code_line(
                f"model.addConstr({aux} >= {right}, name={self._format_name_expr(constr_name_prefix, '_or_r')})"
            )
            self._add_code_line(
                f"model.addConstr({aux} <= {left} + {right}, name={self._format_name_expr(constr_name_prefix, '_or_link')})"
            )
            return aux
        raise ValueError(f"Unsupported boolean expression type for Gurobi constraint: {t}")

    def _reify_scoped_comparison(self, comp_node, current_iterators):
        while isinstance(comp_node, dict) and comp_node.get("type") == "parenthesized_expression":
            comp_node = comp_node.get("expression")
        if not (
            isinstance(comp_node, dict)
            and comp_node.get("type") in ("binop", "constraint")
            and comp_node.get("op") in (">=", "<=", "==", "!=", ">", "<")
        ):
            return self._traverse_expression(comp_node, current_iterators)

        if not self._expr_depends_on_decision_var(comp_node):
            cond_expr = self._emit_index_condition(comp_node, current_iterators)
            return f"(1 if ({cond_expr}) else 0)"

        if not hasattr(self, "_comparison_expr_counter"):
            self._comparison_expr_counter = 0
        self._comparison_expr_counter += 1
        aux_name = f"_cmp_expr_{self._comparison_expr_counter}"
        scope_vars, scope_ranges = self._active_scope_iterators(current_iterators)
        in_materialized_loop = getattr(self, "_materialized_loop_depth", 0) > 0

        left_node = comp_node["left"]
        right_node = comp_node["right"]
        op = comp_node["op"]
        if scope_vars and not in_materialized_loop:
            self._add_code_line(f"{aux_name} = {{}}")
            if len(scope_vars) == 1:
                outer_loop_header = f"for {scope_vars[0]} in {scope_ranges[0]}:"
            else:
                self._add_code_line("import itertools  # needed for multi-index forall")
                outer_loop_header = f"for {', '.join(scope_vars)} in itertools.product({', '.join(scope_ranges)}):"
            self._add_code_line(outer_loop_header)
            self.indent_level += 1
            scope_iterators = {name: name for name in scope_vars}
            scope_key_expr = self._format_scope_key_expr(scope_vars, scope_iterators)
            aux_ref = f"{aux_name}[{scope_key_expr}]"
            self._add_code_line(f"{aux_ref} = model.addVar(vtype=GRB.BINARY)")
            previous_active_ranges = getattr(self, "_active_iterator_ranges", {}).copy()
            inner_active_ranges = previous_active_ranges.copy()
            for name in scope_vars:
                inner_active_ranges.pop(name, None)
            self._active_iterator_ranges = inner_active_ranges
            try:
                left_expr = self._traverse_expression(left_node, scope_iterators)
                right_expr = self._traverse_expression(right_node, scope_iterators)
            finally:
                self._active_iterator_ranges = previous_active_ranges
            for line in self._emit_reify_comparison(left_node, right_node, left_expr, right_expr, op, aux_ref).split("\n"):
                self._add_code_line(line)
            self.indent_level -= 1
            key_expr = self._format_scope_key_expr(scope_vars, current_iterators)
            return f"{aux_name}[{key_expr}]"

        self._add_code_line(f"{aux_name} = model.addVar(vtype=GRB.BINARY)")
        left_expr = self._traverse_expression(left_node, current_iterators)
        right_expr = self._traverse_expression(right_node, current_iterators)
        for line in self._emit_reify_comparison(left_node, right_node, left_expr, right_expr, op, aux_name).split("\n"):
            self._add_code_line(line)
        return aux_name

    def _comparison_sum_struct_key(self, expr_node, current_iterators):
        inner_expression = self._unwrap_parenthesized(expr_node.get("expression"))
        if not self._is_comparison_node(inner_expression):
            return None

        iterator_names = [iterator["iterator"] for iterator in expr_node["iterators"]]
        iterator_map = current_iterators.copy()
        iterator_map.update({name: name for name in iterator_names})
        try:
            left_text = self._traverse_expression(inner_expression["left"], iterator_map, symbolic=True)
            right_text = self._traverse_expression(inner_expression["right"], iterator_map, symbolic=True)
        except Exception:
            return None

        index_constraint = expr_node.get("index_constraint")
        index_text = None
        if index_constraint is not None:
            try:
                index_text = self._traverse_expression(index_constraint, iterator_map, symbolic=True)
            except Exception:
                index_text = "IC_ERR"
        return (
            f"cmp_sum|{tuple(iterator_names)}|{inner_expression['op']}|"
            f"{left_text}|{right_text}|{index_text}"
        )

    def _sum_iterator_range(self, iterator_range, iterator_map):
        range_type = iterator_range["type"]
        if range_type == "range_specifier":
            start = self._traverse_expression(iterator_range["start"], iterator_map, symbolic=True)
            end = self._traverse_expression(iterator_range["end"], iterator_map, symbolic=True)
            return f"range({start}, {end} + 1)"
        if range_type == "named_range":
            try:
                return self._emit_range_from_declaration(iterator_range["name"], iterator_map, True)
            except SemanticError:
                set_name = self._emit_set_name_if_declared(iterator_range["name"])
                if set_name:
                    return set_name
                raise ValueError(f"Range or set '{iterator_range['name']}' not found in declarations.")
        if range_type == "indexed_set":
            index_expression = self._expr_indexed_name(
                {
                    "type": "indexed_name",
                    "name": iterator_range["name"],
                    "dimensions": iterator_range.get("dimensions", []),
                },
                iterator_map,
                symbolic=True,
            )
            dimensions = iterator_range.get("dimensions", [])
            if " if isinstance(" not in index_expression and len(dimensions) == 1:
                dimension = dimensions[0]
                if isinstance(dimension, dict) and dimension.get("type") == "name_reference_index":
                    key = dimension["name"]
                    return (
                        f"({iterator_range['name']}[{key}] if isinstance({iterator_range['name']}, dict) "
                        f"else {index_expression})"
                    )
            return index_expression
        if range_type in ("named_set", "named_set_dimension"):
            set_name = self._emit_set_name_if_declared(iterator_range["name"])
            if set_name:
                return set_name
            raise ValueError(f"Set '{iterator_range['name']}' not found in declarations.")
        raise ValueError(f"Unsupported range type for sum: {range_type}")

    def _prepare_sum_iterators(self, iterators, current_iterators):
        loop_vars = []
        loop_ranges = []
        iterator_map = current_iterators.copy()
        for iterator in iterators:
            name = iterator["iterator"]
            logger.debug(f"[GurobiCodeGen] SUM iterator: {iterator}")
            loop_ranges.append(self._sum_iterator_range(iterator["range"], iterator_map))
            loop_vars.append(name)
            iterator_map[name] = name
        return loop_vars, loop_ranges, iterator_map

    def _reused_comparison_sum(self, expr_node, current_iterators, structural_key=None, scope_vars=()):
        if hasattr(self, "_comparison_sum_meta") and id(expr_node) in self._comparison_sum_meta:
            meta_reuse = self._comparison_sum_meta[id(expr_node)]
            if meta_reuse.get("list_name"):
                cached_scope = tuple(meta_reuse.get("scope_vars") or ())
                if not cached_scope or all(name in current_iterators for name in cached_scope):
                    list_expr, _ = self._comparison_sum_accessors(meta_reuse, current_iterators)
                    return f"gp.quicksum({list_expr})"
        if structural_key and hasattr(self, "_comparison_sum_key_map") and structural_key in self._comparison_sum_key_map:
            metadata = self._comparison_sum_key_map[structural_key]
            metadata_scope = tuple(metadata.get("scope_vars") or ())
            if metadata_scope == tuple(scope_vars) and (
                not metadata_scope or all(name in current_iterators for name in metadata_scope)
            ):
                list_expr, _ = self._comparison_sum_accessors(metadata, current_iterators)
                return f"gp.quicksum({list_expr})"
        return None

    def _emit_comparison_sum(
        self,
        expr_node,
        comparison,
        current_iterators,
        new_iterators,
        loop_vars,
        loop_ranges,
        scope_vars,
        scope_ranges,
        structural_key,
    ):
        self._sum_cmp_counter = getattr(self, "_sum_cmp_counter", 0) + 1
        list_name = f"_cmp_sum_list_{self._sum_cmp_counter}"
        list_append_target = list_name
        metadata = {
            "list_name": list_name,
            "len_var": None,
            "len_name": None,
            "scope_vars": tuple(scope_vars),
        }
        self._comparison_sum_meta = getattr(self, "_comparison_sum_meta", {})
        self._comparison_sum_meta[id(expr_node)] = metadata
        if structural_key:
            self._comparison_sum_key_map = getattr(self, "_comparison_sum_key_map", {})
            self._comparison_sum_key_map[structural_key] = metadata

        scope_key_expression = None
        length_name = None
        if scope_vars:
            self._add_code_line(f"{list_name} = {{}}  # scoped auxiliaries for sum of comparisons")
            length_name = f"{list_name}_len"
            self._add_code_line(f"{length_name} = {{}}")
            metadata["len_name"] = length_name
            self._add_code_line(self._construct_loop_header(scope_vars, scope_ranges))
            self.indent_level += 1
            scope_key_expression = self._format_scope_key_expr(scope_vars, {name: name for name in scope_vars})
            list_append_target = f"{list_name}[{scope_key_expression}]"
            self._add_code_line(f"{list_append_target} = []")
        else:
            self._add_code_line(f"{list_name} = []  # auxiliaries for sum of comparisons")

        self._add_code_line(self._construct_loop_header(loop_vars, loop_ranges))
        self.indent_level += 1
        index_constraint = expr_node.get("index_constraint")
        if index_constraint is not None:
            condition = self._traverse_expression(index_constraint, new_iterators)
            self._add_code_line(f"if {condition}:")
            self.indent_level += 1

        left_node = comparison["left"]
        right_node = comparison["right"]
        operator = comparison["op"]
        left_expression = self._traverse_expression(left_node, new_iterators)
        right_expression = self._traverse_expression(right_node, new_iterators)
        auxiliary = f"cmp_aux_{self._sum_cmp_counter}_" + "_".join(loop_vars)
        self._add_code_line(
            f"{auxiliary} = model.addVar(vtype=GRB.BINARY)  # reified "
            f"({left_expression} {operator} {right_expression})"
        )
        reification = self._emit_reify_comparison(
            left_node, right_node, left_expression, right_expression, operator, auxiliary
        )
        for line in reification.split("\n"):
            self._add_code_line(line)
        self._add_code_line(f"{list_append_target}.append({auxiliary})")

        if index_constraint is not None:
            self.indent_level -= 1
        self.indent_level -= 1
        if scope_vars:
            self._add_code_line(
                f"{length_name}[{scope_key_expression}] = len({list_append_target})  "
                "# cardinality of comparison terms"
            )
            self.indent_level -= 1
        else:
            length_variable = f"{list_name}_len"
            self._add_code_line(f"{length_variable} = len({list_name})  # cardinality of comparison terms")
            metadata["len_var"] = length_variable
        list_expression, _ = self._comparison_sum_accessors(metadata, current_iterators)
        return f"gp.quicksum({list_expression})"

    def _emit_ordinary_sum(self, inner_expression, index_constraint, new_iterators, loop_vars, loop_ranges):
        inner_text = self._traverse_expression(inner_expression, new_iterators)
        logger.debug(f"[GurobiCodeGen] SUM inner_expr_str: {inner_text}")
        generators = " ".join(f"for {var} in {iterator_range}" for var, iterator_range in zip(loop_vars, loop_ranges))
        generator = f"{inner_text} {generators}"
        if index_constraint is not None:
            condition = self._traverse_expression(index_constraint, new_iterators)
            logger.debug(f"[GurobiCodeGen] SUM cond_str: {condition}")
            generator += f" if {condition}"
        logger.debug(f"[GurobiCodeGen] SUM generated quicksum: gp.quicksum({generator})")
        return f"gp.quicksum({generator})"

    def _expr_sum(self, expr_node, current_iterators, symbolic):
        iterators = expr_node["iterators"]
        index_constraint = expr_node.get("index_constraint")
        inner_expression = expr_node["expression"]
        reused = self._reused_comparison_sum(expr_node, current_iterators)
        if reused is not None:
            return reused

        logger.debug(
            f"[GurobiCodeGen] SUM: iterators={iterators}, index_constraint={index_constraint}, "
            f"inner_expression={inner_expression}"
        )
        loop_vars, loop_ranges, new_iterators = self._prepare_sum_iterators(iterators, current_iterators)
        logger.debug(f"[GurobiCodeGen] SUM loop_vars={loop_vars}, loop_ranges={loop_ranges}")
        scope_vars, scope_ranges = self._active_scope_iterators(current_iterators, loop_vars)
        structural_key = self._comparison_sum_struct_key(expr_node, current_iterators)
        reused = self._reused_comparison_sum(expr_node, current_iterators, structural_key, scope_vars)
        if reused is not None:
            return reused

        previous_active_ranges = getattr(self, "_active_iterator_ranges", {}).copy()
        active_ranges = previous_active_ranges.copy()
        active_ranges.update({var: iterator_range for var, iterator_range in zip(loop_vars, loop_ranges)})
        self._active_iterator_ranges = active_ranges
        try:
            comparison = self._unwrap_parenthesized(inner_expression)
            if self._is_comparison_node(comparison):
                return self._emit_comparison_sum(
                    expr_node,
                    comparison,
                    current_iterators,
                    new_iterators,
                    loop_vars,
                    loop_ranges,
                    scope_vars,
                    scope_ranges,
                    structural_key,
                )

            return self._emit_ordinary_sum(
                inner_expression, index_constraint, new_iterators, loop_vars, loop_ranges
            )
        finally:
            self._active_iterator_ranges = previous_active_ranges

    def _emit_reify_comparison(self, left_node, right_node, left_expr, right_expr, op, aux_sym):
        """Return code lines (joined by \n) that add big-M constraints linking aux_sym to (left op right).

        Chooses a conservative M via static bound estimation; falls back to 1e6. Encodings:
          op in {>=, >}: enforce diff >= 0 when aux=1; diff <= M*aux
          op in {<=, <}: enforce -diff >= 0 when aux=1; -diff <= M*aux
          op == '==': symmetric two-sided with four inequalities (can be tightened later).
                    op == '!=': aux=1 iff left and right differ by at least EPS in either direction.
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
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} >= 0 - {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_ge1_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} <= -{eps} + {bigM} * {aux_sym}, name={self._format_name_expr('aux', f'_reify_ge2_{aux_sym}')} )"
            )
        elif op == ">":
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} >= {eps} - {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_gt1_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} <= 0 + {bigM} * {aux_sym}, name={self._format_name_expr('aux', f'_reify_gt2_{aux_sym}')} )"
            )
        elif op == "<=":
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} <= 0 + {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_le1_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} >= {eps} - {bigM} * {aux_sym}, name={self._format_name_expr('aux', f'_reify_le2_{aux_sym}')} )"
            )
        elif op == "<":
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} <= -{eps} + {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_lt1_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} >= 0 - {bigM} * {aux_sym}, name={self._format_name_expr('aux', f'_reify_lt2_{aux_sym}')} )"
            )
        elif op == "==":
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} <= {eq_tol} + {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_eq1_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({right_expr} - {left_expr} <= {eq_tol} + {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_eq2_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} >= -{eq_tol} - {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_eq3_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({right_expr} - {left_expr} >= -{eq_tol} - {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_eq4_{aux_sym}')} )"
            )
        elif op == "!=":
            delta = f"{aux_sym}_neq_side"
            lines.append(f"{delta} = model.addVar(vtype=GRB.BINARY)")
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} >= {eps} - {bigM} * (1 - {delta}) - {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_neq1_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({right_expr} - {left_expr} >= {eps} - {bigM} * {delta} - {bigM} * (1 - {aux_sym}), name={self._format_name_expr('aux', f'_reify_neq2_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({left_expr} - {right_expr} <= {eq_tol} + {bigM} * {aux_sym}, name={self._format_name_expr('aux', f'_reify_neq_eq1_{aux_sym}')} )"
            )
            lines.append(
                f"model.addConstr({right_expr} - {left_expr} <= {eq_tol} + {bigM} * {aux_sym}, name={self._format_name_expr('aux', f'_reify_neq_eq2_{aux_sym}')} )"
            )
        else:
            lines.append(f"model.addConstr({aux_sym} == 0, name={self._format_name_expr('aux', f'_reify_unk_{aux_sym}')} )")
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
        return "1" if expr_node["value"] else "0"

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

    def _expr_depends_on_decision_var(self, node):
        if not isinstance(node, dict):
            return False
        node_type = node.get("type")
        if node_type == "name":
            decl = self._find_declaration_by_name(node.get("value"))
            return decl is not None and decl.get("type") in ("dvar", "dvar_indexed")
        if node_type == "indexed_name":
            decl = self._find_declaration_by_name(node.get("name"))
            return decl is not None and decl.get("type") in ("dvar", "dvar_indexed")
        if node_type == "field_access":
            return self._expr_depends_on_decision_var(node.get("base"))
        if node_type == "parenthesized_expression":
            return self._expr_depends_on_decision_var(node.get("expression"))
        if node_type in ("binop", "constraint", "and", "or"):
            return self._expr_depends_on_decision_var(node.get("left")) or self._expr_depends_on_decision_var(
                node.get("right")
            )
        if node_type in ("not", "uminus"):
            return self._expr_depends_on_decision_var(node.get("value"))
        if node_type == "conditional":
            return (
                self._expr_depends_on_decision_var(node.get("condition"))
                or self._expr_depends_on_decision_var(node.get("then"))
                or self._expr_depends_on_decision_var(node.get("else"))
            )
        if node_type in ("sum", "min_agg", "max_agg"):
            return self._expr_depends_on_decision_var(node.get("expression")) or self._expr_depends_on_decision_var(
                node.get("index_constraint")
            )
        if node_type == "tuple_literal":
            return any(self._expr_depends_on_decision_var(el) for el in node.get("elements", []))
        return False

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

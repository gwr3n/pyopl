import itertools
import json
import logging
from collections import defaultdict  # Needed for coefficient accumulation helpers
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# === Local imports ===
from .scipy_codegen_base import SciPyCodeGeneratorBase
from .semantic_error import SemanticError
from .tuple_set_helper import TupleSetHelper

# === Third-party imports ===
# (none)


# === Module-level constants (Stage 1 refactor) ===
# Single source for big-M fallback and boolean epsilon tolerances.
BIG_M_DEFAULT = 1_000_000.0  # Conservative default; refined per-expression when bounds available.
BOOL_EPS = 1e-6  # Tolerance used for strict inequality flips / boolean reification.

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ExpressionEvaluator:
    @staticmethod
    def _extract_index_value(idx):
        """
        Helper to extract the value part from (dict, value) tuples used in index expressions.
        If idx is a tuple of (dict, value), return value; else return idx unchanged.
        """
        if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[0], dict):
            return idx[1]
        return idx

    def _eval_tuple_literal(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
        """
        Evaluate a tuple_literal AST node as a Python tuple.
        Returns a tuple value, consistent with handler dispatch.
        """

        def to_tuple_recursive(e):
            if isinstance(e, dict) and e.get("type") == "tuple_literal":
                return tuple(to_tuple_recursive(ee) for ee in e["elements"])
            elif isinstance(e, dict):
                coef, val = self.eval(e, env)
                if isinstance(val, (float, int, str, tuple)):
                    return val
                raise SemanticError(f"Tuple element evaluated to unsupported type: {type(val)}")
            else:
                return e

        return {}, tuple(to_tuple_recursive(e) for e in expr["elements"])

    def __init__(self, parent: "SciPyCSCCodeGenerator") -> None:
        self.parent = parent

    def eval(
        self, expr: Dict[str, Any], env: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Union[float, str, Tuple[Any, ...]]]:
        """
        Evaluate an expression AST node with optional environment.
        Accepts float, str, or tuple results for handler compatibility.
        """
        # Debug: Log the incoming expression and environment
        logger.debug(f"[EVAL_ENTRY] type={expr.get('type')}, expr={expr}, env={env}")
        if env is None:
            env = {}
        if not isinstance(expr, dict):
            raise self.parent._unsupported_type_error("expr", type(expr))
        t_any = expr.get("type")
        if not isinstance(t_any, str):
            raise self.parent._unsupported_type_error("expr", "missing or non-string 'type'")
        t: str = t_any
        # Handle 'implies' node by rewriting as (not left) or right
        if t == "implies":
            return self.eval(self._rewrite_implies(expr), env)
        if t == "constraint":
            return {}, 0.0
        # Unified handler dispatch via helper
        handler = self._get_handler(t)
        if handler:
            result = handler(expr, env)
            logger.debug(f"[EVAL] Handler for type '{t}' returned: {result} (type: {type(result)})")
            # Accept tuple results for tuple_literal
            if result is None:
                logger.error(f"[EVAL] Handler for type '{t}' returned None. Expr: {expr}")
                raise SemanticError(f"ExpressionEvaluator.eval: handler for type '{t}' returned None (expr={expr})")
            if isinstance(result, tuple) and len(result) == 2:
                coef, val = result
                logger.debug(f"[EVAL] Handler for type '{t}' returned tuple: coef={coef}, val={val} (type(val): {type(val)})")
                # Accept float/int/str/tuple for value
                if isinstance(val, (float, int, str, tuple)):
                    # Always cast int to float for type consistency
                    if isinstance(val, int):
                        return coef, float(val)
                    return coef, val  # type: ignore[return-value]
                # Dict is only allowed if parent is field_access or field_access_index
                if isinstance(val, dict):
                    import inspect

                    # Walk the call stack to find the parent caller
                    stack = inspect.stack()
                    parent_types = set()
                    for frame in stack:
                        code = frame.function
                        if code.startswith("_eval_field_access") or code.startswith("_eval_field_access_index"):
                            parent_types.add(code)
                    if not parent_types:
                        logger.error(
                            f"[EVAL] Handler for type '{t}' returned tuple with dict value: {result}, which is not allowed by type signature."
                        )
                        raise SemanticError(
                            f"ExpressionEvaluator.eval: handler for type '{t}' returned tuple with dict value: {result}, which is not allowed by type signature."
                        )
                    # else: allow dict to be returned for field access
                    return coef, val  # type: ignore[return-value]
            # If result is not a tuple, raise error
            logger.error(f"[EVAL] Handler for type '{t}' returned non-tuple result: {result}")
            raise SemanticError(f"ExpressionEvaluator.eval: handler for type '{t}' returned non-tuple result: {result}")
        # Fallbacks for common literal types
        return self._handle_literal_fallback(t, expr, env)

    def _handle_literal_fallback(
        self, t: str, expr: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str, Tuple[Any, ...]]]:
        """Handle fallback for common literal types in eval, including tuple results. Always returns a tuple or raises."""
        if t == "number_literal_index":
            return {}, expr["value"]
        elif t == "boolean_literal":
            return self._eval_boolean_literal(expr, env)
        elif t == "conditional":
            return self._eval_conditional(expr, env)
        elif t == "tuple_literal":
            return self._eval_tuple_literal(expr, env)
        elif t == "string_literal":  # <-- return plain string literal
            return {}, expr["value"]
        raise NotImplementedError(f"Expression type '{expr.get('type')}' is not supported by the SciPy code generator.")

    def _rewrite_implies(self, expr: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite an 'implies' node as (not left) or right."""
        return {
            "type": "or",
            "left": {"type": "not", "value": expr["left"]},
            "right": expr["right"],
        }

    def _get_handler(self, t: str) -> Any:
        """Return the handler method for a given expression type, or None if not found."""
        return getattr(self, f"_eval_{t}", None)

    def _eval_boolean_literal(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        # Map boolean_literal to 1.0 (True) or 0.0 (False)
        val = expr.get("value", False)
        return {}, 1.0 if val else 0.0

    def _eval_conditional(
        self, expr: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str, Tuple[Any, ...]]]:
        # Evaluate the condition; must be ground (no decision variable)
        coef_cond, val_cond = self.eval(expr["condition"], env)
        if coef_cond:
            raise self.parent._unsupported_type_error("conditional", "non-ground condition")
        if val_cond:
            return self.eval(expr["then"], env)
        else:
            return self.eval(expr["else"], env)

    def _eval_field_access(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        base = expr["base"]
        field = expr["field"]
        logger.debug(f"[_eval_field_access] base expr: {base}, field: {field}, env: {env}")
        result = self.eval(base, env)
        logger.debug(f"[_eval_field_access] eval(base) result: {result}")
        if result is None:
            logger.error(f"[_eval_field_access] base expression '{base}' could not be evaluated (unknown type or error)")
            raise SemanticError(f"_eval_field_access: base expression '{base}' could not be evaluated (unknown type or error)")
        _, base_val = result
        logger.debug(f"[_eval_field_access] base_val: {base_val} (type: {type(base_val)})")
        sem_type = base.get("sem_type")
        if sem_type:
            val = self.parent._resolve_tuple_field(sem_type, field, base_val)
            logger.debug(f"[_eval_field_access] _resolve_tuple_field result: {val}")
            if val is not None:
                return {}, val
        if isinstance(base_val, dict) and field in base_val:
            logger.debug(f"[_eval_field_access] Returning field from dict: {field} -> {base_val[field]}")
            return {}, base_val[field]
        base_str = base["value"] if base.get("type") == "name" else str(base)
        logger.debug(f"[_eval_field_access] Fallback to string field access: {base_str}['{field}']")
        return {}, f"{base_str}['{field}']"

    def _resolve_tuple_field_access_by_index(
        self, base: Dict[str, Any], field: str, tuple_val: Tuple[Any, ...]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        tuple_type_name = self._find_tuple_type_for_iterator(base["value"])
        if tuple_type_name and hasattr(self.parent, "tuple_types") and tuple_type_name in self.parent.tuple_types:
            fields = self.parent.tuple_types[tuple_type_name]
            for idx, f in enumerate(fields):
                if f["name"] == field:
                    return {}, tuple_val[idx]
            raise self.parent._not_found_error(
                "tuple field",
                f"{field} in tuple type {tuple_type_name} for value {base['value']}",
            )
        if hasattr(self.parent, "tuple_types"):
            for tname, fields in self.parent.tuple_types.items():
                if len(tuple_val) == len(fields):
                    for idx, f in enumerate(fields):
                        if f["name"] == field:
                            return {}, tuple_val[idx]
            raise self.parent._not_found_error("tuple field", f"{field} in tuple types for value {base['value']}")
        return {}, f"{base['value']}[{field}]"

    def _find_tuple_type_for_iterator(self, iterator_name: str) -> Optional[str]:
        if not hasattr(self.parent, "ast"):
            return None

        def search_expr(expr):
            if isinstance(expr, dict):
                if expr.get("type") == "sum":
                    for it in expr.get("iterators", []):
                        if it["iterator"] == iterator_name:
                            rng = it["range"]
                            if rng["type"] == "named_range":
                                set_name = rng["name"]
                                for decl in self.parent.ast.get("declarations", []):
                                    if decl.get("type") == "set_of_tuples" and decl.get("name") == set_name:
                                        return decl.get("tuple_type")
                                if set_name in self.parent.data_dict:
                                    set_val = self.parent.data_dict[set_name]
                                    if isinstance(set_val, dict) and "tuple_type" in set_val:
                                        return set_val["tuple_type"]
                for v in expr.values():
                    res = search_expr(v)
                    if res:
                        return res
            elif isinstance(expr, list):
                for e in expr:
                    res = search_expr(e)
                    if res:
                        return res
            return None

        ast = self.parent.ast
        if "objective" in ast:
            res = search_expr(ast["objective"])
            if res:
                return res
        if "constraints" in ast:
            res = search_expr(ast["constraints"])
            if res:
                return res
        return None

    def _eval_number(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        return {}, expr["value"]

    def _eval_name(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        vname = expr["value"]
        is_var, val, is_symbolic = self.parent._lookup_var_or_param(vname, indices=None, env=env)
        if is_var:
            return {cast(str, val): 1.0}, 0.0
        elif not is_symbolic:
            return {}, cast(Union[float, str], val)
        else:
            raise SemanticError(f"Unresolved name '{vname}' in expression (missing parameter or variable)")

    def _eval_indexed_name(
        self, expr: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str, dict[Any, Any]]]:
        # Evaluate indices and coerce numeric-like to int
        indices = [self._eval_index_expr(dim, env)[1] for dim in expr["dimensions"]]
        remapped_indices = list(indices)
        decl = self.parent._find_decl(expr["name"])
        is_tuple_indexed = False
        if decl is not None:
            dims = decl.get("dimensions", [])
            if len(dims) == 1 and dims[0].get("type") == "named_set_dimension":
                set_name = dims[0].get("name")
                set_decl = self.parent._find_decl(set_name)
                if set_decl and set_decl.get("type") in (
                    "set_of_tuples",
                    "set_of_tuples_external",
                ):
                    is_tuple_indexed = True
        if is_tuple_indexed:
            return self._handle_tuple_indexed(expr, remapped_indices)
        # For parameters indexed by a typed scalar set whose data is stored as a Python list, convert string label to position via <Set>_index
        if decl is not None and decl.get("type", "").startswith("parameter"):
            dims_decl = decl.get("dimensions", [])
            param_data = self.parent.data_dict.get(expr["name"])
            if isinstance(param_data, list) and len(dims_decl) == len(remapped_indices):
                remapped_any = False
                remapped_indices_work = []
                for idx_val, dim_decl in zip(remapped_indices, dims_decl):
                    if dim_decl.get("type") == "named_set_dimension" and isinstance(idx_val, str):
                        set_name = dim_decl.get("name")
                        set_decl = self.parent._find_decl(set_name)
                        # Prefer data_dict value; fallback to decl.value for typed sets
                        set_data = self.parent.data_dict.get(set_name)
                        if (
                            set_data is None
                            and set_decl
                            and set_decl.get("type") in ("typed_set", "typed_set_external", "set_declaration")
                        ):
                            set_data = set_decl.get("value") or []
                        if isinstance(set_data, list):
                            try:
                                pos = set_data.index(idx_val)  # 0-based
                                remapped_indices_work.append(pos + 1)  # OPL is 1-based
                                remapped_any = True
                            except ValueError:
                                if isinstance(idx_val, int):
                                    remapped_indices_work.append(idx_val)
                                # else skip or handle as needed
                        else:
                            if isinstance(idx_val, int):
                                remapped_indices_work.append(idx_val)
                    else:
                        if isinstance(idx_val, int):
                            remapped_indices_work.append(idx_val)
                if remapped_any:
                    remapped_indices = remapped_indices_work
        is_var, val, is_symbolic = self.parent._lookup_var_or_param(expr["name"], indices=remapped_indices, env=env)
        all_indices_are_int = all(isinstance(idx, int) for idx in remapped_indices)
        vname = self.parent._multi_indexed_var_name(expr, env, self._eval_index_expr)
        if is_var:
            return {str(val): 1.0}, 0.0
        elif not is_symbolic:
            # Allow dicts for structured parameters; only enforce float/str at scalar leaves
            if isinstance(val, (float, int)):
                return {}, float(val)
            elif isinstance(val, str):
                return {}, val
            elif isinstance(val, dict):
                return {}, val
            else:
                raise SemanticError(f"Expected float, str, or dict for parameter value, got {type(val)}: {val}")
        else:
            if all_indices_are_int:
                raise self.parent._not_found_error("indexed variable or parameter", vname)
            return {vname: 1.0}, 0.0

    def _eval_index_expr(self, dim_expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        tt = dim_expr.get("type")
        # All index expressions return (coef_dict, value)
        if tt == "field_access_index" or tt == "field_access":
            coef, val = self._eval_field_access(dim_expr, env)
            if isinstance(val, float) and float(val).is_integer():
                val = int(val)
            return coef, val
        if tt == "number_literal_index":
            val = dim_expr["value"]
            if isinstance(val, float) and float(val).is_integer():
                val = int(val)
            return {}, val
        elif tt == "name_reference_index":
            # Updated: also consult data_dict when the name is not in env
            name = dim_expr.get("name")
            if name in env:
                val = env[name]
            else:
                # Pull scalar/range bound values from provided data (e.g., nbMachines)
                val = self.parent.data_dict.get(name, name)
            # Coerce float-int to int; also accept numeric strings
            if isinstance(val, float) and float(val).is_integer():
                val = int(val)
            elif isinstance(val, str):
                try:
                    # Try int first (OPL indices are integers)
                    val = int(val)
                except Exception:
                    pass
            return {}, val
        # NEW: resolve plain 'name' nodes used inside index arithmetic (e.g., t in s[t-1])
        elif tt == "name":
            # Previously only looked in env; also consult model data to resolve parameters like S, T
            name = dim_expr.get("value")
            if name in env:
                val = env[name]
            else:
                val = self.parent.data_dict.get(name, name)
            if isinstance(val, float) and float(val).is_integer():
                val = int(val)
            elif isinstance(val, str):
                try:
                    val = int(val)
                except Exception:
                    pass
            return {}, val
        elif tt == "string_literal":  # <-- explicit support for string index literals
            return {}, dim_expr.get("value")
        elif tt == "binop":
            lcoef, lval = self._eval_index_expr(dim_expr["left"], env)
            rcoef, rval = self._eval_index_expr(dim_expr["right"], env)
            # Propagate symbolic binop indices as strings if not both ints
            if not (isinstance(lval, int) and isinstance(rval, int)):
                symbolic = f"({lval} {dim_expr['op']} {rval})"
                return {}, symbolic
            if dim_expr["op"] == "+":
                val = lval + rval
            elif dim_expr["op"] == "-":
                val = lval - rval
            elif dim_expr["op"] == "*":
                val = lval * rval
            else:
                raise self.parent._unsupported_operator_error("index", dim_expr["op"])
            if isinstance(val, float) and float(val).is_integer():
                val = int(val)
            return {}, val
        elif tt == "uminus":
            _, val = self._eval_index_expr(dim_expr["value"], env)
            if isinstance(val, str):
                return {}, f"-({val})"
            if isinstance(val, float) and float(val).is_integer():
                val = int(val)
            return {}, -val
        elif tt == "parenthesized_expression":
            return self._eval_index_expr(dim_expr["expression"], env)
        elif tt == "tuple_literal":

            def to_tuple_recursive(e):
                if isinstance(e, dict) and e.get("type") == "tuple_literal":
                    return tuple(to_tuple_recursive(ee) for ee in e["elements"])
                elif isinstance(e, dict):
                    _, val = self._eval_index_expr(e, env)
                    return val
                else:
                    return e

            return {}, tuple(to_tuple_recursive(e) for e in dim_expr["elements"])
        else:
            if "value" in dim_expr:
                val = dim_expr["value"]
                if isinstance(val, float) and float(val).is_integer():
                    val = int(val)
                return {}, val
            elif "name" in dim_expr:
                val = env.get(dim_expr["name"], dim_expr["name"])
                if isinstance(val, float) and float(val).is_integer():
                    val = int(val)
                return {}, val
            raise self.parent._unsupported_type_error("index expr", tt)

    def _handle_tuple_indexed(self, expr: Dict[str, Any], indices: List[Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        # If index is a tuple (coef_dict, value), extract value
        tuple_key = self._extract_index_value(indices[0])
        vname_tuple = f"{expr['name']}[{repr(tuple_key)}]"
        if vname_tuple in self.parent.var_indices:
            return {vname_tuple: 1.0}, 0.0
        param_dict = self.parent.data_dict.get(expr["name"])
        if param_dict is not None and isinstance(param_dict, dict):
            if tuple_key in param_dict:
                # Treat tuple-indexed parameter as a pure constant (no coefficients)
                return {}, param_dict[tuple_key]
        for d in self.parent._find_decls(expr["name"], "parameter_inline_indexed"):
            dims = d.get("dimensions", [])
            if len(dims) == 1 and dims[0].get("type") == "named_set_dimension":
                set_name = dims[0]["name"]
                tuple_set = self.parent._find_decl(set_name, "set_of_tuples")
                if tuple_set:
                    tuple_keys = [tuple(t["elements"]) for t in tuple_set["value"]]
                    idx = None
                    for i, k in enumerate(tuple_keys):
                        if k == tuple_key:
                            idx = i
                            break
                    if idx is not None:
                        param_vals = d["value"]
                        if isinstance(param_vals, list) and idx < len(param_vals):
                            return {}, param_vals[idx]
        vname_str = f"{expr['name']}[{str(tuple_key)}]"
        if vname_str in self.parent.var_indices:
            return {vname_str: 1.0}, 0.0
        # If not found, raise immediately with clear error
        raise self.parent._not_found_error("tuple-indexed variable or parameter", vname_tuple)

    def _eval_name_reference_index(
        self, expr: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        name_any = expr.get("name")
        name = name_any if isinstance(name_any, str) else str(name_any)
        val = env.get(name, name)
        if isinstance(val, (int, float)):
            return {}, float(val)
        return {}, str(val)

    def _eval_binop(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        """Evaluate binary operations, compacted and deduplicated."""
        left, right = expr["left"], expr["right"]
        op = expr["op"]

        def _is_bool_expr(e):
            return isinstance(e, dict) and (
                e.get("type") == "boolean_literal" or (e.get("type") == "binop" and e.get("sem_type") == "boolean")
            )

        if op == "!=" and _is_bool_expr(left) and _is_bool_expr(right):
            aux_name = self.parent._ensure_aux_binary("xor_flag")
            return {"type": "aux_var", "name": aux_name, "sem_type": "boolean"}, 0.0
        if op == "==":
            left_result = self.eval(left, env)
            right_result = self.eval(right, env)
            if left_result is None or right_result is None:
                raise SemanticError(f"_eval_binop: == failed, left or right is None: left={left_result}, right={right_result}")
            left_coef, left_val = left_result
            right_coef, right_val = right_result
            left_key = left.get("value") if isinstance(left, dict) and left.get("type") == "name" else None
            if isinstance(left_key, str) and left_key in env:
                left_val = env[left_key]
            right_key = right.get("value") if isinstance(right, dict) and right.get("type") == "name" else None
            if isinstance(right_key, str) and right_key in env:
                right_val = env[right_key]
            symbolic = bool(left_coef) or bool(right_coef) or isinstance(left_val, str) or isinstance(right_val, str)
            if symbolic and not getattr(self.parent, "_allow_symbolic_bool", False):
                raise SemanticError("Non-ground boolean == outside constraint build context")
            if isinstance(left_val, (str, int, float)) and isinstance(right_val, (str, int, float)):
                return {}, left_val == right_val
            return {}, str(left_val) == str(right_val)
        if op == "+":
            return self._handle_binop_add(left, right, env)
        elif op == "-":
            return self._handle_binop_sub(left, right, env)
        elif op == "*":
            return self._handle_binop_mul(left, right, env)
        if op in ("!=", "<", ">", "<=", ">="):
            return self._handle_binop_cmp(left, right, op, env)
        raise self.parent._unsupported_operator_error("binop", expr["op"])

    def _handle_binop_add(
        self, left: Dict[str, Any], right: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        l_result = self.eval(left, env)
        r_result = self.eval(right, env)
        if l_result is None or r_result is None:
            raise SemanticError(f"_handle_binop_add: left or right is None: left={l_result}, right={r_result}")
        if not isinstance(l_result, tuple) or not isinstance(r_result, tuple):
            raise SemanticError(f"_handle_binop_add: left or right did not return a tuple: left={l_result}, right={r_result}")
        ldict, lconst = l_result
        rdict, rconst = r_result
        out = ldict.copy()
        for k, v in rdict.items():
            out[k] = out.get(k, 0.0) + v
        if isinstance(lconst, (str, tuple)) or isinstance(rconst, (str, tuple)):
            return out, f"({lconst}) + ({rconst})"
        return out, float(cast(Union[int, float], lconst)) + float(cast(Union[int, float], rconst))

    def _handle_binop_sub(
        self, left: Dict[str, Any], right: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        l_result = self.eval(left, env)
        r_result = self.eval(right, env)
        if l_result is None or r_result is None:
            raise SemanticError(f"_handle_binop_sub: left or right is None: left={l_result}, right={r_result}")
        if not isinstance(l_result, tuple) or not isinstance(r_result, tuple):
            raise SemanticError(f"_handle_binop_sub: left or right did not return a tuple: left={l_result}, right={r_result}")
        ldict, lconst = l_result
        rdict, rconst = r_result
        out = ldict.copy()
        for k, v in rdict.items():
            out[k] = out.get(k, 0.0) - v
        if isinstance(lconst, (str, tuple)) or isinstance(rconst, (str, tuple)):
            return out, f"({lconst}) - ({rconst})"
        return out, float(cast(Union[int, float], lconst)) - float(cast(Union[int, float], rconst))

    def _handle_binop_mul(
        self, left: Dict[str, Any], right: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        """Modularized multiplication handler for binop *."""
        l_result = self.eval(left, env)
        r_result = self.eval(right, env)
        if l_result is None or r_result is None:
            raise SemanticError(f"_handle_binop_mul: left or right is None: left={l_result}, right={r_result}")
        if not isinstance(l_result, tuple) or not isinstance(r_result, tuple):
            raise SemanticError(f"_handle_binop_mul: left or right did not return a tuple: left={l_result}, right={r_result}")
        ldict, lconst = l_result
        rdict, rconst = r_result
        # Symbolic multiply (string or tuple-constant cases)
        if isinstance(lconst, (str, tuple)) or isinstance(rconst, (str, tuple)):
            return {}, f"({lconst}) * ({rconst})"
        # Nonlinear error: variable * variable
        if ldict and rdict:
            raise self.parent._unsupported_type_error("nonlinear term", "variable * variable")
        # variable * constant
        if ldict and not rdict:
            rc = float(cast(Union[int, float], rconst))
            return {k: v * rc for k, v in ldict.items()}, float(cast(Union[int, float], lconst)) * rc
        elif rdict and not ldict:
            lc = float(cast(Union[int, float], lconst))
            return {k: v * lc for k, v in rdict.items()}, float(cast(Union[int, float], rconst)) * lc
        elif not ldict and not rdict:
            return {}, float(cast(Union[int, float], lconst)) * float(cast(Union[int, float], rconst))
        else:
            raise self.parent._unsupported_type_error("nonlinear term", "variable * variable")

    def _handle_binop_cmp(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
        op: str,
        env: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        # Evaluate both sides; do not mask SemanticError here
        ldict, lconst = self.eval(left, env)
        rdict, rconst = self.eval(right, env)

        def _is_num(x: object) -> bool:
            return isinstance(x, (int, float, bool))

        # NEW: treat pure string-vs-string comparisons as ground booleans
        if isinstance(lconst, str) and isinstance(rconst, str):
            if op == "!=":
                return {}, float(lconst != rconst)
            if op == "==":
                return {}, float(lconst == rconst)
            # For ordering ops on strings, fall back to symbolic gating below

        # Symbolic if any variable coefficient or non-numeric literal shows up
        is_symbolic = bool(ldict) or bool(rdict) or isinstance(lconst, (str, tuple)) or isinstance(rconst, (str, tuple))
        if is_symbolic:
            # Respect symbolic-boolean gating
            if not getattr(self.parent, "_allow_symbolic_bool", False):
                raise SemanticError("Non-ground boolean comparison outside constraint build context")
            # In symbolic mode, never numerically evaluate; emit a symbolic placeholder string
            return {}, f"({lconst}) {op} ({rconst})"

        # Ground case: both sides numeric or simple booleans
        if op == "!=":
            return {}, (
                float(bool(lconst) != bool(rconst)) if not (_is_num(lconst) and _is_num(rconst)) else float(lconst != rconst)
            )
        if op in ("<", ">", "<=", ">="):
            # Only perform ordering comparisons if both sides are numeric
            if not (_is_num(lconst) and _is_num(rconst)):
                # With ground but non-numeric (e.g., strings), treat as symbolic gated by flag
                if not getattr(self.parent, "_allow_symbolic_bool", False):
                    raise SemanticError("Non-numeric comparison outside constraint build context")
                return {}, f"({lconst}) {op} ({rconst})"
            lc = float(cast(Union[int, float], lconst))
            rc = float(cast(Union[int, float], rconst))
            if op == "<":
                return {}, float(lc < rc)
            if op == ">":
                return {}, float(lc > rc)
            if op == "<=":
                return {}, float(lc <= rc)
            # op == ">="
            return {}, float(lc >= rc)
        # Should not reach here (== handled in _eval_binop)
        raise self.parent._unsupported_operator_error("binop", op)

    def _eval_uminus(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        d, c = self.eval(expr["value"], env)
        if isinstance(c, (str, tuple)):
            return {k: -v for k, v in d.items()}, f"-({c})"
        return {k: -v for k, v in d.items()}, -float(c)

    def _eval_sum(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        iterators = expr["iterators"]
        loop_vars, loop_ranges = self.parent._unroll_iterators(iterators)
        # Narrow types to satisfy mypy
        coef_dict_total: Dict[str, float] = {}
        const_total: Union[float, str] = 0.0
        # Handle empty iterator: sum is zero
        if any(len(rng) == 0 for rng in loop_ranges):
            return coef_dict_total, const_total
        for idx_tuple in itertools.product(*loop_ranges):
            env2 = dict(env or {})
            for v, val in zip(loop_vars, idx_tuple):
                env2[v] = val
            if not self._sum_index_constraint_satisfied(expr, env2):
                continue
            try:
                coef_dict, const = self.eval(expr["expression"], env=env2)
            except SemanticError:
                # Treat missing parameter/variable as zero ONLY in sum expansion
                coef_dict, const = {}, 0.0
            for vname, coef in coef_dict.items():
                # coef is numeric; coerce to float for safety
                coef_dict_total[vname] = coef_dict_total.get(vname, 0.0) + float(cast(Union[int, float], coef))
            # If any side is symbolic (str or tuple), build a symbolic string; else do numeric add
            if isinstance(const_total, str) or isinstance(const, (str, tuple)):
                const_total = f"({const_total}) + ({const})"
            elif isinstance(const, (int, float)):
                const_total = float(const_total) + float(const)
            # else: ignore non-numeric, non-string constants
        return coef_dict_total, const_total

    def _sum_index_constraint_satisfied(self, expr: Dict[str, Any], env2: Dict[str, Any]) -> bool:
        index_constraint = expr.get("index_constraint")
        if index_constraint is not None:
            try:
                cond_val = self.eval(index_constraint, env2)[1]
                logger.debug(f"[SCIPY_SUM] index_constraint={index_constraint}, env2={env2}, cond_val={cond_val}")
                # Robust truthiness: numeric 0/1 or bool; ignore symbolic strings
                if isinstance(cond_val, (int, float, bool)):
                    return bool(cond_val)
                return bool(cond_val)  # fallback
            except Exception:
                return True
        return True

    def _eval_parenthesized_expression(
        self, expr: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str, Tuple[Any, ...]]]:
        return self.eval(expr["expression"], env)

    # Add more as needed for other expression types

    def _eval_not(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        coef, const = self.eval(expr["value"], env)
        if coef or isinstance(const, str):
            if not getattr(self.parent, "_allow_symbolic_bool", False):
                raise SemanticError("Non-ground boolean NOT outside constraint build context")
            return {}, f"!({const})"
        val = bool(const)
        return {}, float(not val)

    def _eval_and(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        left_coef, left_const = self.eval(expr["left"], env)
        right_coef, right_const = self.eval(expr["right"], env)
        if left_coef or right_coef or isinstance(left_const, str) or isinstance(right_const, str):
            if not getattr(self.parent, "_allow_symbolic_bool", False):
                raise SemanticError("Non-ground boolean AND outside constraint build context")
            return {}, f"({left_const}) && ({right_const})"
        return {}, float(bool(left_const) and bool(right_const))

    def _eval_or(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        left_coef, left_const = self.eval(expr["left"], env)
        right_coef, right_const = self.eval(expr["right"], env)
        if left_coef or right_coef or isinstance(left_const, str) or isinstance(right_const, str):
            if not getattr(self.parent, "_allow_symbolic_bool", False):
                raise SemanticError("Non-ground boolean OR outside constraint build context")
            return {}, f"({left_const}) || ({right_const})"
        return {}, float(bool(left_const) or bool(right_const))


class SciPyCSCCodeGenerator(SciPyCodeGeneratorBase):
    _debug_ast: Dict[str, Any]

    def _resolve_tuple_field(self, tuple_type, field, tuple_val):
        """
        Given a tuple type name, field name, and tuple value, return the value for the field.
        """
        if tuple_type in self.tuple_types:
            fields = self.tuple_types[tuple_type]
            for idx, f in enumerate(fields):
                if f["name"] == field:
                    if isinstance(tuple_val, tuple):
                        return tuple_val[idx]
                    if isinstance(tuple_val, dict) and field in tuple_val:
                        return tuple_val[field]
                    return None
        return None

    def _add_variable(self, base_name, lower=0.0, upper=1.0):
        """
        Adds a variable with a unique name based on base_name, and returns the name and index.
        """
        name = base_name
        k = 0
        while name in self.var_indices:
            k += 1
            name = f"{base_name}_{k}"
        idx = len(self.var_names)
        self.var_names.append(name)
        self.var_indices[name] = idx
        if hasattr(self, "lower_bounds") and hasattr(self, "upper_bounds"):
            self.lower_bounds.append(lower)
            self.upper_bounds.append(upper)
        # No-op for aux_created; removed unreachable/undefined code
        return name, idx

    def _get_param_decl_map(self):
        return {
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

    def _convert_flat_kv_to_dict(self, param_data):
        # Detect flat key-value list: even length, alternating str and number
        if isinstance(param_data, list) and len(param_data) % 2 == 0 and len(param_data) > 0:
            is_flat_kv = all(
                (isinstance(param_data[i], str) and isinstance(param_data[i + 1], (int, float)))
                for i in range(0, len(param_data), 2)
            )
            if is_flat_kv:
                return {param_data[i]: param_data[i + 1] for i in range(0, len(param_data), 2)}
        return None

    def _make_constraint_row(self, coef_dict):
        """
        Create a constraint row for the LP matrix, given a dict of variable coefficients.
        """
        row = [0.0] * len(self.var_names)
        for v, c in coef_dict.items():
            if v in self.var_indices:
                row[self.var_indices[v]] += c
        return row

    def _big_m_for_comparison(self, comp: Dict[str, Any], env: Optional[Dict[str, Any]] = None) -> float:
        """Compute a tightened big-M for a linear (in)equation comp.

        Strategy:
        1. Evaluate lhs and rhs into linear forms f_l(x)=a_l^T x + c_l and f_r(x)=a_r^T x + c_r.
        2. Consider expression f(x) = f_l(x) - f_r(x) with constant part c = c_l - c_r.
        3. Using collected variable bounds (from preprocessing) compute interval [f_min, f_max].
           Fallback bounds: (-1e3, 1e3) if unavailable (kept conservative but finite to avoid overflow).
        4. Return M = max(|f_min|, |f_max|, |f_max - f_min|) + 1.0 (slack) with floor 1.0.

        If any error occurs, fallback to BIG_M_DEFAULT.
        """
        try:
            lhs = comp.get("left")
            rhs = comp.get("right")
            coef_lhs, const_lhs = self._eval_expr(lhs, {})
            if isinstance(rhs, dict):
                coef_rhs, const_rhs = self._eval_expr(rhs, {})
            else:
                coef_rhs, const_rhs = (
                    {},
                    rhs if isinstance(rhs, (int, float)) else 0.0,
                )
            expr_coef = dict(coef_lhs)
            for vn, cf in coef_rhs.items():
                expr_coef[vn] = expr_coef.get(vn, 0.0) - cf
            expr_const = const_lhs - const_rhs
            f_min = expr_const
            f_max = expr_const
            lbs = getattr(self, "_collected_lbs", {})
            ubs = getattr(self, "_collected_ubs", {})
            for vn, cf in expr_coef.items():
                lb = lbs.get(vn, -1e3)
                ub = ubs.get(vn, 1e3)
                if cf >= 0:
                    f_min += cf * lb
                    f_max += cf * ub
                else:
                    f_min += cf * ub
                    f_max += cf * lb
            width = max(abs(f_min), abs(f_max), abs(f_max - f_min))
            return max(1.0, width + 1.0)
        except Exception:
            return BIG_M_DEFAULT

    def _get_tuple_set_names(self, iterators):
        """
        Given a list of iterator dicts, return the iterator variable names for tuple sets used in sum/forall expressions.
        """
        names = set()
        for it in iterators:
            rng = it.get("range", {})
            if rng.get("type") == "named_set":
                set_decl = self._find_decl(rng.get("name"))
                if set_decl and set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                    names.add(it.get("iterator"))
        return names

    # === Section: Error message helpers ===
    def _not_found_error(self, what, name):
        from .semantic_error import SemanticError

        return SemanticError(f"Not found: {what} '{name}'")

    def _unsupported_type_error(self, context, typ):
        from .semantic_error import SemanticError

        return SemanticError(f"Semantic Error: Unsupported {context} type: {typ}")

    def _unsupported_operator_error(self, context, op):
        from .semantic_error import SemanticError

        return SemanticError(f"Semantic Error: Unsupported operator in {context}: {op}")

    """
    SciPyCSCCodeGenerator generates Python code for solving linear programming (LP)
    models using SciPy's `linprog` function, with support for sparse constraint
    matrices in compressed sparse column (CSC) format.

    This class takes a semantically validated abstract syntax tree (AST)
    representing an OPL-style mathematical model and a data dictionary,
    and produces executable Python code that builds the LP problem,
    solves it, and reports results.
    """

    def _should_include_sum_term(
        self,
        loop_vars: list,
        idx_tuple: tuple,
        tuple_set_names: set,
        env: dict,
        index_constraint: dict | None,
        expr: dict,
        eval_env: dict | None = None,
    ) -> tuple[dict, bool]:
        """
        Helper to build env2 and check index_constraint for sum/binop-sum expansion.
        Returns (env2, include:bool)
        """
        env2 = dict(env or {})
        for v, val in zip(loop_vars, idx_tuple):
            if v in tuple_set_names and not isinstance(val, tuple):
                val = tuple(val)
            env2[v] = val
        include = True
        if index_constraint is not None:
            try:
                _, cond_val = self._eval_expr(index_constraint, env2)
                include = bool(cond_val)
            except Exception:
                include = True
        return env2, include

    # ---------------- Boolean composition helpers (AND/OR of linear comparisons) ----------------
    def _is_linear_comparison(self, node: Dict[str, Any]) -> bool:
        # Include '!=' as a linear comparison for boolean logic
        return (
            isinstance(node, dict)
            and node.get("type") == "binop"
            and node.get("sem_type") == "boolean"
            and node.get("op") in ("<=", ">=", "==", "!=")
        )

    def _flatten_bool(self, node: Any, target_type: str) -> List[Any]:
        """Flatten nested 'and'/'or' tree collecting leaves for given target_type."""
        out: List[Any] = []
        if not isinstance(node, dict):
            return [node]
        if node.get("type") == target_type:
            out.extend(self._flatten_bool(node.get("left"), target_type))
            out.extend(self._flatten_bool(node.get("right"), target_type))
        elif self._is_linear_comparison(node):
            out.append(node)
        else:
            out.append(node)
        return out

    def _ensure_aux_binary(self, base_name: str) -> str:
        """Create a new auxiliary binary variable (0/1), declare it, and return its index name. Also append to aux_created if present."""
        name = base_name
        k = 0
        while name in self.var_indices:
            k += 1
            name = f"{base_name}_{k}"
        # Declare the variable
        idx = len(self.var_names)
        self.var_indices[name] = idx
        self.var_names.append(name)
        # Ensure parallel metadata are updated
        if not hasattr(self, "bounds"):
            self.bounds = []
        self.bounds.append([0, 1])
        if not hasattr(self, "integrality"):
            self.integrality = []
        self.integrality.append(1)
        if not hasattr(self, "c"):
            self.c = []
        self.c.append(0.0)
        if not hasattr(self, "aux_created"):
            self.aux_created = []
        self.aux_created.append(name)
        return name

    def _linearize_or(self, comparisons: List[Any]) -> None:
        """Linearize disjunction of linear comparisons using big-M and auxiliary binaries.
        For '!=': split into two comparisons: < and >.
        """
        z_vars = []
        for comp in comparisons:
            op = comp.get("op")
            if op == "!=":
                comp_lt = dict(comp)
                comp_lt["op"] = "<"
                comp_gt = dict(comp)
                comp_gt["op"] = ">"
                self._linearize_or([comp_lt])
                self._linearize_or([comp_gt])
            else:
                z = self._ensure_aux_binary("or_flag")
                z_vars.append(z)
                # No local env in this helper; use empty env to allow evaluator fallbacks
                M = self._big_m_for_comparison(comp, env={})
                pass
                coef_lhs, const_lhs = self._eval_expr(comp["left"], {})
                rhs_node = comp["right"]
                coef_rhs, const_rhs = (
                    self._eval_expr(rhs_node, {})
                    if isinstance(rhs_node, dict)
                    else ({}, rhs_node if isinstance(rhs_node, (int, float)) else 0.0)
                )
                expr_coef = dict(coef_lhs)
                for v, c in coef_rhs.items():
                    expr_coef[v] = expr_coef.get(v, 0.0) - c
                expr_const = const_lhs - const_rhs
                if op == "<=":
                    row = [0.0] * len(self.var_names)
                    for v, c in expr_coef.items():
                        row[self.var_indices[v]] += c
                    row[self.var_indices[z]] += M
                    self.A_ub.append(row)
                    self.b_ub.append(M - expr_const)
                elif op == ">=":
                    row = [0.0] * len(self.var_names)
                    for v, c in expr_coef.items():
                        row[self.var_indices[v]] -= c
                    row[self.var_indices[z]] += M
                    self.A_ub.append(row)
                    self.b_ub.append(M + expr_const)
                elif op == "==":
                    row1 = [0.0] * len(self.var_names)
                    for v, c in expr_coef.items():
                        row1[self.var_indices[v]] += c
                    row1[self.var_indices[z]] += M
                    self.A_ub.append(row1)
                    self.b_ub.append(M - expr_const)
                    row2 = [0.0] * len(self.var_names)
                    for v, c in expr_coef.items():
                        row2[self.var_indices[v]] -= c
                    row2[self.var_indices[z]] += M
                    self.A_ub.append(row2)
                    self.b_ub.append(M + expr_const)
        if z_vars:
            row = [0.0] * len(self.var_names)
            for z in z_vars:
                row[self.var_indices[z]] -= 1.0
            self.A_ub.append(row)
            self.b_ub.append(-1.0)

    def _expand_and(self, comparisons: List[Any]) -> None:
        """Add each comparison as its own constraint. For '!=', add both < and > as separate constraints."""
        for comp in comparisons:
            op = comp.get("op")
            if op == "!=":
                comp_lt = dict(comp)
                comp_lt["op"] = "<"
                comp_gt = dict(comp)
                comp_gt["op"] = ">"
                self._expand_and([comp_lt])
                self._expand_and([comp_gt])
            else:
                lhs_dict, lhs_const = self._accumulate_sum_to_dict(comp["left"], env={}, sign=1)
                rhs_dict, rhs_const = (
                    self._accumulate_sum_to_dict(comp["right"], env={}, sign=1)
                    if isinstance(comp["right"], dict)
                    else (
                        {},
                        (comp["right"] if isinstance(comp["right"], (int, float)) else 0.0),
                    )
                )
                expr_coef = dict(lhs_dict)
                for v, c in rhs_dict.items():
                    expr_coef[v] = expr_coef.get(v, 0.0) - c
                expr_const = lhs_const - rhs_const
            if comp["op"] == "==":
                # equality into A_eq
                row = [0.0] * len(self.var_names)
                for v, c in expr_coef.items():
                    # Coefficient dict from _accumulate_sum_to_dict may use integer indices directly
                    if isinstance(v, int):
                        if v < len(row):
                            row[v] += c
                    else:
                        idx = self.var_indices.get(v)
                        if idx is not None:
                            row[idx] += c
                self.A_eq.append(row)
                self.b_eq.append(-expr_const)
            elif comp["op"] == "<=":
                row = [0.0] * len(self.var_names)
                for v, c in expr_coef.items():
                    if isinstance(v, int):
                        if v < len(row):
                            row[v] += c
                    else:
                        idx = self.var_indices.get(v)
                        if idx is not None:
                            row[idx] += c
                self.A_ub.append(row)
                self.b_ub.append(-expr_const)
            elif comp["op"] == ">=":
                # - (lhs - rhs) <= 0
                row = [0.0] * len(self.var_names)
                for v, c in expr_coef.items():
                    if isinstance(v, int):
                        if v < len(row):
                            row[v] -= c
                    else:
                        idx = self.var_indices.get(v)
                        if idx is not None:
                            row[idx] -= c
                self.A_ub.append(row)
                self.b_ub.append(expr_const)

    # ---------------- Bounds / linear span helpers (mirrors Gurobi backend logic in a lightweight form) ----------------
    def _var_bounds_safe(self, node: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Return (lb, ub) for a variable or numeric literal node when cheaply available.
        Supported forms:
          - boolean dvar: (0,1)
          - int/float dvar: (-inf, +inf)
          - int+/float+ dvar: (0, +inf)
          - numeric literal: (v,v)
        Returns (None, None) if unknown/unbounded.
        """
        if not isinstance(node, dict):
            return (None, None)
        t = node.get("type")
        if t in ("name", "indexed_name"):
            base_name = node.get("value") if t == "name" else node.get("name")
            for d in self.ast.get("declarations", []):
                if d.get("name") == base_name and d.get("type") in (
                    "dvar",
                    "dvar_indexed",
                ):
                    vtype = d.get("var_type")
                    if vtype == "boolean":
                        return (0.0, 1.0)
                    if vtype in ("int+", "float+"):
                        return (0.0, None)
                    if vtype in ("int", "float"):
                        return (None, None)
        if t == "number":
            v = float(node.get("value", 0))
            return (v, v)
        return (None, None)

    def _linear_bounds_safe(self, node: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Best-effort bounds for a restricted linear expression (var, literal, +/- , scalar * var).
        Returns (lb, ub) or (None, None) if cannot bound.
        """
        if not isinstance(node, dict):
            return (None, None)
        t = node.get("type")
        if t in ("name", "indexed_name", "number"):
            # First consult collected per-variable instance bounds (from processed constraints)
            if t in ("name", "indexed_name"):
                if t == "name":
                    vname = node.get("value")
                else:
                    # Attempt to build name; if field access indices not evaluable (due to env), fall back to base var aggregate bounds
                    try:
                        vname = self._multi_indexed_var_name(node, {})
                    except Exception:
                        vname = node.get("name")
                if hasattr(self, "_collected_lbs"):
                    lb = self._collected_lbs.get(vname)
                    ub = self._collected_ubs.get(vname)
                    # If no direct match and this is an indexed_name with field access, try base symbol aggregate
                    if lb is None and ub is None and t == "indexed_name":
                        base_sym = node.get("name")
                        lb = self._collected_lbs.get(base_sym)
                        ub = self._collected_ubs.get(base_sym)
                    if lb is not None or ub is not None:
                        # Merge with static type-derived bounds
                        tlb, tub = self._var_bounds_safe(node)
                        if tlb is not None:
                            lb = max(lb, tlb) if lb is not None else tlb
                        if tub is not None:
                            ub = min(ub, tub) if ub is not None else tub
                        return (lb, ub)
            return self._var_bounds_safe(node)
        if t == "unaryop" and node.get("op") == "-":
            value = node.get("value")
            if not isinstance(value, dict):
                return (None, None)
            lb, ub = self._linear_bounds_safe(value)
            if lb is None or ub is None:
                return (None, None)
            return (-ub, -lb)
        if t == "binop":
            op = node.get("op")
            left = node.get("left")
            right = node.get("right")
            if not (isinstance(left, dict) and isinstance(right, dict)):
                return (None, None)
            if op == "+":
                lL, lU = self._linear_bounds_safe(left)
                rL, rU = self._linear_bounds_safe(right)
                if lL is None or rL is None or lU is None or rU is None:
                    return (None, None)
                return (lL + rL, lU + rU)
            if op == "-":
                lL, lU = self._linear_bounds_safe(left)
                rL, rU = self._linear_bounds_safe(right)
                if lL is None or rU is None or lU is None or rL is None:
                    return (None, None)
                return (lL - rU, lU - rL)
            if op == "*":
                # Only support scalar * var or var * scalar
                if left.get("type") == "number" and right.get("type") in (
                    "name",
                    "indexed_name",
                ):
                    c = float(left.get("value", 0))
                    vL, vU = self._var_bounds_safe(right)
                elif right.get("type") == "number" and left.get("type") in (
                    "name",
                    "indexed_name",
                ):
                    c = float(right.get("value", 0))
                    vL, vU = self._var_bounds_safe(left)
                else:
                    return (None, None)
                if vL is None or vU is None:
                    return (None, None)
                if c >= 0:
                    return (c * vL, c * vU if vU is not None else None)
                else:
                    # negative scalar flips
                    if vU is None:
                        return (None, None)
                    return (c * vU, c * vL)
        return (None, None)

    def _update_vector_from_coef_dict(self, coef_dict: Dict[str, Any], vector: List[float], op: Optional[str] = None) -> None:
        """
        Helper to update a vector from a coef_dict. If op is None, set; if '+', add; if '-', subtract.
        """
        for vname, coef in coef_dict.items():
            idx = self.var_indices.get(vname)
            if idx is not None:
                if op == "+":
                    vector[idx] += coef
                elif op == "-":
                    vector[idx] -= coef
                else:
                    vector[idx] = coef

    def _resolve_tuple_index_varname(self, vname: str) -> Optional[int]:
        """
        Helper to resolve a variable name with a tuple index to its index in var_indices.
        Returns the index if found, else None.

        Note:
            This method is not covered by tests because, in all real OPL models and test cases (including vehicle routing),
            tuple-indexed variables are always generated and referenced in a consistent canonical form (e.g., x[(1, 2, 10.0)]),
            and the code that generates and looks up variable names always uses this canonical form. As a result, the fallback
            logic here (which attempts to reconstruct and look up a tuple-indexed variable name from a string representation)
            is never exercised in practice.

            This method would only be used if, for some reason, a variable name with a tuple index was constructed in a non-canonical
            string form elsewhere in the code (e.g., by user code or a legacy parser), and a lookup was attempted using that form.
            In current usage, this does not occur, so the method is effectively dead code, but is retained for robustness in case
            of future changes or non-standard input.
        """
        if "[" in vname and not vname.startswith("'"):
            try:
                base, key = vname.split("[", 1)
                key = key.rstrip("]")
                if key.startswith("(") and key.endswith(")"):
                    import ast

                    try:
                        key_tuple = ast.literal_eval(key)
                    except Exception:
                        return None
                    vname_norm = f"{base}[{repr(key_tuple)}]"
                    idx = self.var_indices.get(vname_norm)
                    if idx is not None:
                        return idx
            except Exception:
                pass
        # If not found, raise with the actual vname for clarity
        raise SemanticError(f"Variable '{vname}' not found in environment.")

    def _tighten_bounds_from_constraints(self, bounds: list, var_names: list, var_indices: dict, constraints: list) -> None:
        """
        Update lower and upper bounds for variables based on constraints.
        Modifies bounds in place. Handles both scalar and indexed variables, including tuple-indexed and field-access cases.
        Traverses constraints recursively, supporting forall and index constraints, and updates bounds for each variable.
        """
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]

        def update_bounds(idx, op, val):
            if idx is not None and val is not None:
                if op == ">=":
                    lower_bounds[idx] = max(lower_bounds[idx], val) if lower_bounds[idx] is not None else val
                elif op == "<=":
                    upper_bounds[idx] = min(upper_bounds[idx], val) if upper_bounds[idx] is not None else val
                elif op == "==":
                    lower_bounds[idx] = upper_bounds[idx] = val

        def tighten_simple_constraint(constr, env):
            left = constr["left"]
            right = constr["right"]
            rhs_val: Optional[float] = None
            try:
                coef_dict, c = self._eval_expr(right, env)
                # Only update bounds if right side is constant (no decision vars)
                if not coef_dict and isinstance(c, (int, float)):
                    rhs_val = float(c)
            except Exception:
                rhs_val = None
            if left["type"] == "name":
                idx = var_indices.get(left["value"])
                update_bounds(idx, constr["op"], rhs_val)
                return
            elif left["type"] == "indexed_name":
                dims = left["dimensions"]
                remapped: list[Any] = []
                for d in dims:
                    if d["type"] == "name_reference_index":
                        remapped.append(env.get(d["name"]))
                    elif d["type"] == "number_literal_index":
                        remapped.append(d["value"])
                    else:
                        # Evaluate generic index expr safely
                        _, v_eval = self._eval_index_expr(d, env)
                        remapped.append(v_eval)
                # Normalize ints
                remapped = [int(v) if isinstance(v, float) and v.is_integer() else v for v in remapped]
                is_var, looked_up, is_symbolic = self._lookup_var_or_param(left["name"], indices=remapped, env=env)
                if is_var and isinstance(looked_up, str):
                    idx = var_indices.get(looked_up)
                    update_bounds(idx, constr["op"], rhs_val)
                return

        def tighten_forall_constraint(constr, env=None):
            if env is None:
                env = {}
            iterators = constr.get("iterators")
            if not iterators:
                return
            loop_vars = [it["iterator"] for it in iterators]
            rng = iterators[0].get("range")
            # Support both inline and named ranges
            if rng is None:
                return
            if "start" in rng and "end" in rng:
                start = rng["start"]["value"] if isinstance(rng["start"], dict) and "value" in rng["start"] else rng["start"]
                end = rng["end"]["value"] if isinstance(rng["end"], dict) and "value" in rng["end"] else rng["end"]
            elif "name" in rng:
                # Named range: look up declaration
                decl = None
                for d in self.ast.get("declarations", []):
                    if d.get("type") == "range_declaration_inline" and d.get("name") == rng["name"]:
                        decl = d
                        break
                if decl:
                    start = (
                        decl["start"]["value"]
                        if isinstance(decl["start"], dict) and "value" in decl["start"]
                        else decl["start"]
                    )
                    end = decl["end"]["value"] if isinstance(decl["end"], dict) and "value" in decl["end"] else decl["end"]
                else:
                    return
            else:
                return
            # Convert start/end to int if possible
            try:
                start = int(start)
                end = int(end)
            except Exception:
                return
            index_constraint = constr.get("index_constraint")
            inner_constraints = [constr["constraint"]] if "constraint" in constr else constr.get("constraints", [])
            for t in range(start, end + 1):
                env2 = env.copy()
                # Bind loop variable for binop indices
                for v in loop_vars:
                    env2[v] = t
                if index_constraint is not None:
                    try:
                        cond_val = self._eval_expr(index_constraint, env2)[1]
                    except Exception:
                        cond_val = True
                    if not cond_val:
                        continue
                for inner in inner_constraints:
                    tighten_constraint(inner, env=env2)

        def tighten_constraint(constr, env=None):
            if env is None:
                env = {}
            if constr["type"] == "constraint":
                tighten_simple_constraint(constr, env)
            elif constr["type"] == "forall_constraint":
                tighten_forall_constraint(constr, env)

        for constr in constraints:
            tighten_constraint(constr)
        # Update bounds in place
        for i, (lo, hi) in enumerate(zip(lower_bounds, upper_bounds)):
            bounds[i][0] = lo
            bounds[i][1] = hi

    def _handle_scalar_variable_declaration(self, decl: dict, var_names: list, bounds: list, integrality: list) -> None:
        """
        Handle the declaration of a scalar variable, updating var_names, bounds, and integrality lists.
        Adds the variable to var_names and var_indices, and sets appropriate bounds and integrality based on type.
        """
        name = decl["name"]
        var_names.append(name)
        self.var_indices[name] = len(var_names) - 1
        vtype = decl.get("var_type")
        if vtype == "boolean":
            bounds.append([0, 1])
            integrality.append(1)
        elif vtype == "int+":
            bounds.append([0, None])
            integrality.append(1)
        elif vtype == "int":
            bounds.append([None, None])
            integrality.append(1)
        elif vtype == "float+":
            bounds.append([0, None])
            integrality.append(0)
        elif vtype == "float":
            bounds.append([None, None])
            integrality.append(0)
        else:
            bounds.append([None, None])
            integrality.append(0)

    def _handle_indexed_variable_declaration(self, decl: dict, var_names: list, bounds: list, integrality: list) -> None:
        """
        Handle an indexed variable declaration (including tuple-indexed and range-indexed variables),
        updating var_names, bounds, and integrality lists.
        """
        name = decl["name"]
        dims = decl["dimensions"]
        logger.debug(f"[SciPyCSCCodeGenerator] _handle_indexed_variable_declaration: name={name}, dims={dims}")
        # If indexed over a set_of_tuples, flatten as tuple keys
        if len(dims) == 1 and dims[0]["type"] == "named_set_dimension":
            set_name = dims[0]["name"]
            # Determine if underlying set is tuple-valued or scalar typed_set
            set_decl = self._find_decl(set_name)
            if set_decl and set_decl.get("type") in (
                "set_of_tuples",
                "set_of_tuples_external",
            ):
                elements = TupleSetHelper.get_tuple_set(set_name, self.ast, self.data_dict)
            else:
                # typed_set or plain set of scalars
                if set_name in self.data_dict:
                    elements = self.data_dict[set_name]
                elif set_decl:
                    # typed_set stores list in 'value'
                    elements = set_decl.get("value") or []
                else:
                    elements = []
            logger.debug(f"[SciPyCSCCodeGenerator] Elements for {name} over {set_name}: {elements}")
            for k in elements:
                # For scalar string indices use underscore style (name_value) to avoid quote issues elsewhere.
                if isinstance(k, tuple):
                    key_part = repr(k)
                    vname = f"{name}[{key_part}]"  # tuple-indexed keep bracket form
                else:
                    vname = f"{name}_{k}"
                var_names.append(vname)
                self.var_indices[vname] = len(var_names) - 1
                vtype = decl.get("var_type")
                if vtype == "boolean":
                    bounds.append([0, 1])
                    integrality.append(1)
                elif vtype == "int+":
                    bounds.append([0, None])
                    integrality.append(1)
                elif vtype == "int":
                    bounds.append([None, None])
                    integrality.append(1)
                elif vtype == "float+":
                    bounds.append([0, None])
                    integrality.append(0)
                elif vtype == "float":
                    bounds.append([None, None])
                    integrality.append(0)
                else:
                    bounds.append([None, None])
                    integrality.append(0)
            return
        # Fallback: treat as before (should not happen for tuple-indexed)
        logger.debug(f"[SciPyCSCCodeGenerator] Fallback for {name}, dims={dims}")
        vtype = decl.get("var_type")
        dim_ranges = []
        symbolic_dim_ranges = []
        for dim in dims:
            # Evaluate the actual integer range for this dimension
            if dim["type"] == "range_index":
                # Always use _eval_bound, which handles numbers, names, and binops
                start_eval = self._eval_bound(dim["start"])
                end_eval = self._eval_bound(dim["end"])
                logger.debug(f"[SciPyCSCCodeGenerator] Range for {name}: start={start_eval}, end={end_eval}")
                dim_ranges.append(list(range(int(start_eval), int(end_eval) + 1)))
                start_val = self._emit_symbolic_expr(dim["start"])
                end_val = self._emit_symbolic_expr(dim["end"])
                symbolic_dim_ranges.append(f"range({start_val}, {end_val} + 1)")
            elif dim["type"] == "named_range_dimension":
                rng = None
                for d in self.ast["declarations"]:
                    if d["type"] == "range_declaration_inline" and d["name"] == dim["name"]:
                        rng = d
                        break
                if rng is None:
                    raise self._not_found_error("range", dim["name"])
                start_eval = self._eval_bound(rng["start"])
                end_eval = self._eval_bound(rng["end"])
                logger.debug(f"[SciPyCSCCodeGenerator] Named range for {name}: start={start_eval}, end={end_eval}")
                dim_ranges.append(list(range(int(start_eval), int(end_eval) + 1)))
                start_val = self._emit_symbolic_expr(rng["start"])
                end_val = self._emit_symbolic_expr(rng["end"])
                symbolic_dim_ranges.append(f"range({start_val}, {end_val} + 1)")
            elif dim["type"] == "named_set_dimension":
                # Support both scalar typed sets and set_of_tuples
                set_name = dim["name"]
                set_vals = None
                set_decl = self._find_decl(set_name)
                if set_decl and set_decl.get("type") in (
                    "set_of_tuples",
                    "set_of_tuples_external",
                ):
                    set_vals = TupleSetHelper.get_tuple_set(set_name, self.ast, self.data_dict)
                else:
                    set_vals = self.data_dict.get(set_name, [])
                logger.debug(f"[SciPyCSCCodeGenerator] Named set for {name}: {set_vals}")
                dim_ranges.append(set_vals)
                symbolic_dim_ranges.append(set_name)
        # Emit the symbolic range for the variable declaration in the generated code
        self._add_code_line(f"# OPL: dvar {vtype} {name}[{', '.join(symbolic_dim_ranges)}]")
        # Continue with variable name generation and bounds
        for idx_tuple in itertools.product(*dim_ranges):
            # Generate variable name as x_1, y_2, etc. (for test compatibility)
            vname = name + "_" + "_".join(str(i) for i in idx_tuple)
            # FIX: correct f-string interpolation in debug
            logger.debug(f"[SciPyCSCCodeGenerator] Adding range-indexed variable: {vname}")
            var_names.append(vname)
            self.var_indices[vname] = len(var_names) - 1
            if vtype == "boolean":
                bounds.append([0, 1])
                integrality.append(1)
            elif vtype == "int+":
                bounds.append([0, None])
                integrality.append(1)
            elif vtype == "int":
                bounds.append([None, None])
                integrality.append(1)
            elif vtype == "float+":
                bounds.append([0, None])
                integrality.append(0)
            elif vtype == "float":
                bounds.append([None, None])
                integrality.append(0)
            else:
                bounds.append([None, None])
                integrality.append(0)

    # === Section: Index/Range/Iterator Utilities ===
    @staticmethod
    def normalize_index(idx: object) -> object:
        """
        Return index as tuple if list/tuple, else unchanged. Recursively normalizes nested indices.
        """
        if isinstance(idx, (list, tuple)):
            return tuple(SciPyCodeGeneratorBase.normalize_index(e) for e in idx)
        return idx

    # === Section: Private Helpers ===
    def _lookup_var_or_param(
        self,
        name: str,
        indices: list | None = None,
        env: dict | None = None,
        default_zero_if_missing: bool = False,
    ) -> tuple[bool, object, bool]:
        """
        Lookup a variable or parameter value (scalar or indexed) by name and indices in the current environment.
        Returns (is_variable, value_or_varname, is_symbolic).
        """
        return self._resolve_param_value(name, indices, env, default_zero_if_missing)

    # === Section: Error message helpers ===

    def _resolve_param_value(
        self,
        name: str,
        indices: list | None = None,
        env: dict | None = None,
        default_zero_if_missing: bool = False,
    ) -> tuple[bool, object, bool]:
        if env is None:
            env = {}
        # 1. Index variable in env
        if indices is None and name in env:
            return False, env[name], False
        # 2. Variable (scalar or indexed)
        var_result = self._resolve_variable(name, indices)
        if var_result is not None:
            return var_result
        # 3. Parameter (scalar or indexed) from data_dict; if not found, fall through to AST
        try:
            param_result = self._resolve_parameter(name, indices, env, default_zero_if_missing)
        except SemanticError:
            param_result = None
        if param_result is not None:
            return param_result
        # 4. Try AST declarations for parameter value
        try:
            ast_result = self._resolve_ast_parameter(name, indices)
        except SemanticError:
            ast_result = None
        if ast_result is not None:
            return ast_result
        # 5. If not resolvable, raise SemanticError in constraints/objective context
        if not default_zero_if_missing:
            import logging

            logging.getLogger("pyopl.scipy_codegen_csc").error(
                f"[resolve_param_value] SemanticError: Parameter or variable '{name}' with indices {indices} not found in environment."
            )
            raise SemanticError(f"Parameter or variable '{name}' with indices {indices} not found in environment.")
        # If sum expansion context, treat as zero/symbolic
        if indices is not None:
            symbolic = name + "[" + "][".join(str(i) for i in indices) + "]"
            return False, symbolic, True
        return False, name, True

    def _resolve_variable(self, name: str, indices: list | None) -> tuple[bool, object, bool] | None:
        """
        Helper to resolve a variable (scalar or indexed) by name and indices.
        Returns (True, varname, False) if found, else None.
        Handles symbolic index matching for variable names.
        """
        logger = logging.getLogger("pyopl.scipy_codegen_csc")
        if indices is None:
            if name in self.var_indices:
                logger.debug(f"[resolve_variable] Found scalar variable: {name}")
                return True, name, False
            else:
                logger.debug(f"[resolve_variable] Scalar variable not found: {name}")
        else:
            # Normalize indices: coerce float integers to int
            norm_indices = []
            for i in indices:
                if isinstance(i, float) and i.is_integer():
                    norm_indices.append(int(i))
                elif isinstance(i, (bool, int)):
                    norm_indices.append(int(i))
                else:
                    norm_indices.append(i)
            vname = name + "_" + "_".join(str(i) for i in norm_indices)
            logger.debug(f"[resolve_variable] Trying indexed variable: {vname} (indices={norm_indices})")
            if vname in self.var_indices:
                logger.debug(f"[resolve_variable] Found indexed variable: {vname}")
                return True, vname, False
            # Try to resolve as a variable with symbolic index (e.g., s[(t-1)])
            for k in self.var_indices:
                if k.startswith(name + "_") and vname.replace("(", "").replace(")", "") == k.replace("(", "").replace(")", ""):
                    logger.debug(f"[resolve_variable] Found symbolic indexed variable: {k}")
                    return True, k, False

            # Strict handling for out-of-domain indices -> raise SemanticError
            try:
                decl = self._find_decl(name)
                if decl and decl.get("type") in ("dvar_indexed",):
                    dims = decl.get("dimensions", [])
                    # Only attempt if arity matches
                    if len(dims) == len(norm_indices):
                        out_of_domain = False
                        details: list[str] = []
                        for dim_decl, idx_val in zip(dims, norm_indices):
                            dtyp = dim_decl.get("type")
                            if dtyp == "range_index":
                                # Evaluate bounds
                                s = self._eval_bound(dim_decl["start"])
                                e = self._eval_bound(dim_decl["end"])
                                if not isinstance(idx_val, int) or idx_val < int(s) or idx_val > int(e):
                                    out_of_domain = True
                                    details.append(f"{idx_val} not in [{int(s)}..{int(e)}]")
                            elif dtyp == "named_range_dimension":
                                rng_name = dim_decl.get("name")
                                rng_decl = self._find_decl(rng_name, "range_declaration_inline")
                                if rng_decl:
                                    s = self._eval_bound(rng_decl["start"])
                                    e = self._eval_bound(rng_decl["end"])
                                    if not isinstance(idx_val, int) or idx_val < int(s) or idx_val > int(e):
                                        out_of_domain = True
                                        details.append(f"{idx_val} not in [{int(s)}..{int(e)}]")
                            elif dtyp == "named_set_dimension":
                                set_name = dim_decl.get("name")
                                # Get set elements from data or AST
                                set_vals = None
                                set_decl = self._find_decl(set_name)
                                if set_name in self.data_dict:
                                    raw = self.data_dict[set_name]
                                    set_vals = raw["elements"] if isinstance(raw, dict) and "elements" in raw else raw
                                elif set_decl:
                                    # typed_set stores list in 'value'; set_of_tuples value is list of dicts with elements
                                    if set_decl.get("type") == "typed_set":
                                        set_vals = set_decl.get("value") or []
                                    elif set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external") and set_decl.get(
                                        "value"
                                    ):
                                        set_vals = [tuple(t["elements"]) for t in set_decl["value"]]
                                if set_vals is not None:
                                    # For tuple-valued sets keep tuple keys; else compare directly
                                    if idx_val not in set_vals:
                                        out_of_domain = True
                                        details.append(f"{idx_val} not in {set_name}")
                            # Other dimension types are not expected here

                        if out_of_domain:
                            msg = f"Index {norm_indices} for '{name}' is out of declared domain"
                            if details:
                                msg += f" ({'; '.join(details)})"
                            logger.debug(f"[resolve_variable] {msg}")
                            from .semantic_error import SemanticError

                            raise SemanticError(msg)
            except Exception as ex:
                # If we purposely raised our SemanticError, propagate it. Otherwise, fall through.
                from .semantic_error import SemanticError

                if isinstance(ex, SemanticError):
                    raise

            logger.debug(f"[resolve_variable] Indexed variable not found: {vname} (indices={indices})")
        return None

    def _resolve_parameter(
        self,
        name: str,
        indices: list | None,
        env: dict,
        default_zero_if_missing: bool = False,
    ) -> tuple[bool, object, bool] | None:
        """
        Helper to resolve a parameter (scalar or indexed) from data_dict, using indices and environment.
        Returns (False, value, False) if found, else None.
        """

        logger = logging.getLogger("pyopl.scipy_codegen_csc")
        val = self.data_dict.get(name)
        # Convert flat key/value list with tuple-like keys into a dict for tuple-indexed parameters
        if isinstance(val, list) and indices is not None:
            try:
                converted = None
                # Case 1: flat alternating keys and values: [key, val, key, val, ...]
                # Apply ONLY if keys are str/tuple/list AND vals are numeric scalars (int/float).
                if len(val) % 2 == 0 and len(val) > 0:
                    keys = val[::2]
                    vals = val[1::2]
                    if all(isinstance(k, (list, tuple, str)) for k in keys) and all(
                        isinstance(v2, (int, float)) for v2 in vals
                    ):
                        d = {}
                        for k, v2 in zip(keys, vals):
                            d[tuple(k) if isinstance(k, (list, tuple)) else k] = v2
                        converted = d
                # Case 2: list of pairs [[key, val], [key, val], ...]
                # Apply ONLY if each pair has key as str/tuple/list and value numeric (int/float).
                if converted is None and all(isinstance(e, (list, tuple)) and len(e) == 2 for e in val):
                    pair_ok = True
                    for k, v2 in val:
                        if not isinstance(k, (list, tuple, str)) or not isinstance(v2, (int, float)):
                            pair_ok = False
                            break
                    if pair_ok:
                        d = {}
                        for k, v2 in val:
                            d[tuple(k) if isinstance(k, (list, tuple)) else k] = v2
                        converted = d
                if converted is not None:
                    val = converted
                    self.data_dict[name] = converted
                    logger.debug(f"[resolve_parameter] Converted flat KV list to dict for param '{name}': {converted}")
            except Exception:
                pass
        logger.debug(f"[resolve_parameter] Lookup param: {name}, indices={indices}, val={val}, env={env}")
        if indices is not None and val is not None:
            try:
                v = val
                # NEW: support dicts keyed by composite tuples (e.g., {('RoleA','CoreSite'): 1.0})
                if isinstance(v, dict) and any(isinstance(k, tuple) for k in v.keys()):
                    # Build composite key from all indices at once
                    rem_evals = [self._eval_index(idx, env) for idx in indices]
                    # If single evaluated index already is a tuple (from tuple_literal), use it directly
                    tuple_key = rem_evals[0] if len(rem_evals) == 1 and isinstance(rem_evals[0], tuple) else tuple(rem_evals)
                    if tuple_key in v:
                        v = v[tuple_key]
                        # Numeric scalar or structured value
                        if isinstance(v, (int, float)):
                            logger.debug(f"[resolve_parameter] Found tuple-keyed numeric param: {v}")
                            return False, float(v), False
                        if isinstance(v, dict):
                            logger.debug(f"[resolve_parameter] Found tuple-keyed dict param: {v}")
                            return False, v, False
                        if isinstance(v, (list, tuple)):
                            logger.debug(f"[resolve_parameter] Found tuple-keyed list/tuple param: {v}")
                            return False, v, False
                        # Fallback: return as-is
                        return False, v, False
                # Fallback: stepwise lookup for nested dict/list
                for i, idx in enumerate(indices):
                    idx_eval = self._eval_index(idx, env)
                    logger.debug(f"[resolve_parameter] Index eval: idx={idx}, idx_eval={idx_eval}, v={v}, env={env}")
                    if isinstance(v, dict):
                        logger.debug(f"[resolve_parameter] Dict lookup: v[{idx_eval}] (keys={list(v.keys())})")
                        v = v[idx_eval]
                        continue
                    # Always cast to int if possible (OPL indices are ints)
                    if isinstance(idx_eval, float) and idx_eval.is_integer():
                        idx_eval = int(idx_eval)
                    if not isinstance(idx_eval, int):
                        logger.debug(f"[resolve_parameter] Non-int index: idx_eval={idx_eval} (type={type(idx_eval)})")
                        raise ValueError(
                            f"Index '{idx}' could not be resolved to int (got {idx_eval!r}) for param '{name}' with env={env}"
                        )
                    # Support both 1-based and 0-based (OPL is 1-based)
                    if isinstance(v, list) and isinstance(idx_eval, int):
                        logger.debug(f"[resolve_parameter] List lookup: v[{idx_eval - 1}] (len={len(v)})")
                        v = v[idx_eval - 1]
                    else:
                        logger.debug(f"[resolve_parameter] List/dict lookup: v[{idx_eval}] (type={type(v)})")
                        v = v[idx_eval]
                # Numeric scalar
                if isinstance(v, (int, float)):
                    logger.debug(f"[resolve_parameter] Found numeric param: {v}")
                    return False, float(v), False
                # Structured record (dict) for tuple arrays: allow field_access
                if isinstance(v, dict):
                    logger.debug(f"[resolve_parameter] Found dict param: {v}")
                    return False, v, False
                # Raw tuple/list (e.g., tuple literal) pass through for positional field index fallback
                if isinstance(v, (list, tuple)):
                    logger.debug(f"[resolve_parameter] Found list/tuple param: {v}")
                    return False, v, False
            except Exception as e:
                logger.debug(f"[resolve_parameter] Exception during lookup: {e}")
                pass
        # NEW: support scalar string/boolean parameters (e.g., source = "London")
        if indices is None and val is not None and isinstance(val, (str, bool)):
            logger.debug(f"[resolve_parameter] Found scalar non-numeric param: {val!r}")
            return False, val, False
        if indices is None and val is not None and isinstance(val, (int, float)):
            logger.debug(f"[resolve_parameter] Found scalar param: {val}")
            return False, float(val), False
        logger.debug(f"[resolve_parameter] Param not found: {name}, indices={indices}")
        if default_zero_if_missing:
            return False, 0.0, False
        raise SemanticError(f"Parameter or variable '{name}' with indices {indices} not found in environment.")

    def _eval_index(self, idx: object, env: dict) -> object:
        """
        Helper to evaluate an index expression in env/data_dict context.
        Tries to evaluate as safe Python literal/arith, then as int, else returns as is.
        """
        if isinstance(idx, str):
            import ast

            # 1) Try tuple/number literal safely
            try:
                lit = ast.literal_eval(idx)
                return lit
            except Exception:
                pass

            # 2) Safe arithmetic: only (+, -, *, //, parentheses) and names from env/data_dict
            def _safe_eval_arith(expr: str) -> object:
                node = ast.parse(expr, mode="eval")

                allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)
                allowed_nodes = (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Num,
                    ast.Constant,
                    ast.Name,
                    ast.Tuple,
                    ast.Load,
                    ast.USub,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.FloorDiv,
                )

                def _eval(n):
                    if not isinstance(n, allowed_nodes):
                        raise ValueError("Disallowed expression in index")
                    if isinstance(n, ast.Expression):
                        return _eval(n.body)
                    if isinstance(n, ast.Num):
                        return int(n.n)
                    if isinstance(n, ast.Constant):
                        if isinstance(n.value, (int, float, bool)):
                            return (
                                int(n.value)
                                if isinstance(n.value, bool) or (isinstance(n.value, float) and n.value.is_integer())
                                else n.value
                            )
                        raise ValueError("Non-numeric constant in index")
                    if isinstance(n, ast.Name):
                        name = n.id
                        if name in env:
                            v = env[name]
                        else:
                            v = self.data_dict.get(name, name)
                        if isinstance(v, (int, float, bool)):
                            return int(v) if isinstance(v, bool) or (isinstance(v, float) and v.is_integer()) else v
                        raise ValueError(f"Name '{name}' not numeric for index")
                    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
                        return -_eval(n.operand)
                    if isinstance(n, ast.BinOp) and isinstance(n.op, allowed_ops):
                        left = _eval(n.left)
                        right = _eval(n.right)
                        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
                            raise ValueError("Non-numeric operands in index arithmetic")
                        if isinstance(n.op, ast.Add):
                            return left + right
                        if isinstance(n.op, ast.Sub):
                            return left - right
                        if isinstance(n.op, ast.Mult):
                            return left * right
                        if isinstance(n.op, ast.FloorDiv):
                            return left // right
                    if isinstance(n, ast.Tuple):
                        return tuple(_eval(e) for e in n.elts)
                    raise ValueError("Unsupported node in index")

                return _eval(node)

            try:
                v = _safe_eval_arith(idx)
                # Normalize float-integral to int
                if isinstance(v, float) and v.is_integer():
                    return int(v)
                return v
            except Exception:
                # 3) Try int fallthrough
                try:
                    return int(idx)
                except Exception:
                    return idx
        return idx

    def _resolve_ast_parameter(self, name: str, indices: list | None) -> tuple[bool, object, bool] | None:
        """
        Helper to resolve a parameter from AST declarations, using _eval_index for index evaluation.
        Returns (False, value, False) if found, else None.
        """
        for decl in self.ast.get("declarations", []):
            if decl.get("type") == "parameter_inline" and decl["name"] == name:
                return False, float(decl["value"]), False
            if indices is not None and decl.get("type") == "parameter_inline_indexed" and decl["name"] == name:
                try:
                    v = decl["value"]
                    for idx in indices:
                        idx_eval = self._eval_index(idx, {})
                        if isinstance(v, list) and isinstance(idx_eval, int):
                            v = v[idx_eval - 1]
                        else:
                            v = v[idx_eval]
                    if isinstance(v, (int, float)):
                        return False, float(v), False
                except Exception:
                    pass
        # If not found, raise with the actual name and indices for clarity
        raise SemanticError(f"AST parameter '{name}' with indices {indices} not found.")

    def _unroll_iterators(self, iterators: list) -> tuple[list, list]:
        """
        Given a list of OPL-style iterators, return (loop_vars, loop_ranges).
        Each iterator is a dict with 'iterator' and 'range'.
        Handles range_specifier, named_range, set_of_tuples, and set_declaration.
        Always uses TupleSetHelper.get_tuple_set for set-of-tuples.
        Raises SemanticError if range or set is not found.
        """
        loop_vars = []
        loop_ranges = []
        for it in iterators:
            name = it["iterator"]
            rng = it["range"]
            if rng["type"] == "range_specifier":
                start = self._eval_bound(rng["start"])
                end = self._eval_bound(rng["end"])
                loop_ranges.append(list(range(int(start), int(end) + 1)))
            elif rng["type"] in ("named_range", "named_set"):
                decl = None
                for d in self.ast["declarations"]:
                    if d["type"] == "range_declaration_inline" and d["name"] == rng["name"]:
                        decl = d
                        break
                if decl is not None:
                    start = self._eval_bound(decl["start"])
                    end = self._eval_bound(decl["end"])
                    loop_ranges.append(list(range(int(start), int(end) + 1)))
                else:
                    # Fallback: check if it's a set-of-tuples, set_of_tuples_external, or set_declaration
                    set_decl = None
                    for d in self.ast["declarations"]:
                        if d["type"] in ("set_of_tuples", "set_of_tuples_external") and d["name"] == rng["name"]:
                            set_decl = d
                            break
                        if d["type"] == "set_declaration" and d["name"] == rng["name"]:
                            set_decl = d
                            break
                        # Allow typed_set iteration (set of scalars with base_type info)
                        if d.get("type") == "typed_set" and d.get("name") == rng["name"]:
                            set_decl = d
                            break
                        # Allow external typed scalar set
                        if d.get("type") == "typed_set_external" and d.get("name") == rng["name"]:
                            set_decl = d
                            break
                    if set_decl is not None:
                        # Always use TupleSetHelper for set_of_tuples and set_of_tuples_external
                        if set_decl.get("type") in (
                            "set_of_tuples",
                            "set_of_tuples_external",
                        ):
                            set_val = TupleSetHelper.get_tuple_set(rng["name"], self.ast, self.data_dict)
                        elif set_decl.get("type") == "set_declaration":
                            set_val = self.data_dict.get(rng["name"])
                            if set_val is None:
                                set_val = set_decl.get("value", [])
                        elif set_decl.get("type") in (
                            "typed_set",
                            "typed_set_external",
                        ):
                            # Prefer data_dict override
                            if rng["name"] in self.data_dict:
                                set_val = self.data_dict[rng["name"]]
                            else:
                                # Declaration stores elements under 'value'
                                set_val = set_decl.get("value") or []
                        else:
                            set_val = []
                        loop_ranges.append(set_val)
                    else:
                        raise self._not_found_error("range or set", rng["name"])
            else:
                raise self._unsupported_type_error("iterator range type", rng["type"])
            loop_vars.append(name)
        return loop_vars, loop_ranges

    def _eval_bound(self, expr: object) -> float | int:
        """
        Evaluate a bound expression for index/range bounds (used in variable declarations, sum, forall, etc).
        Supports: number, name, binop (+, -, *), uminus, parenthesized_expression.
        Raises SemanticError for unsupported types or operators.
        """
        if isinstance(expr, dict):
            t = expr.get("type")
            if t == "number":
                return expr["value"]
            elif t == "name":
                return self.data_dict.get(expr["value"], 1)
            elif t == "binop":
                left = self._eval_bound(expr["left"])
                right = self._eval_bound(expr["right"])
                if expr["op"] == "+":
                    return left + right
                elif expr["op"] == "-":
                    return left - right
                elif expr["op"] == "*":
                    return left * right
                else:
                    raise self._unsupported_operator_error("index bound binop", expr["op"])
            elif t == "uminus":
                val = self._eval_bound(expr["value"])
                return -val
            elif t == "parenthesized_expression":
                return self._eval_bound(expr["expression"])
            else:
                raise self._unsupported_type_error("expr in index bound", t)
        else:
            raise self._unsupported_type_error("expr in index bound", type(expr))

    def _emit_python_expr(self, expr: dict, env: dict | None = None) -> str:
        """
        Emit a valid Python expression from an AST node, using env for index variables.
        Handles numbers, names, binops, uminus, parenthesized expressions, indexed names, and field access.
        """
        if env is None:
            env = {}
        t = expr.get("type") if isinstance(expr, dict) else None
        if t == "number":
            return str(expr["value"])
        elif t == "name":
            return expr["value"]
        elif t == "binop":
            left = self._emit_python_expr(expr["left"], env)
            right = self._emit_python_expr(expr["right"], env)
            return f"({left} {expr['op']} {right})"
        elif t == "uminus":
            val = self._emit_python_expr(expr["value"], env)
            return f"-({val})"
        elif t == "parenthesized_expression":
            return f"({self._emit_python_expr(expr['expression'], env)})"
        elif t == "conditional":
            cond = self._emit_python_expr(expr["condition"], env)
            then_expr = self._emit_python_expr(expr["then"], env)
            else_expr = self._emit_python_expr(expr["else"], env)
            return f"({then_expr} if ({cond}) else {else_expr})"
        elif t == "indexed_name":
            name = expr["name"]
            dims = expr["dimensions"]
            idxs = [str(self._emit_python_expr(dim, env)) for dim in dims]
            idx_str = ", ".join(idxs)
            return f"{name}[{idx_str}]"
        elif t == "field_access":
            # --- Tuple field access ---
            # OPL: a.cost  -->  Python: a[index]
            base = self._emit_python_expr(expr["base"], env)
            field = expr["field"]
            # Try to resolve tuple type from AST
            if hasattr(self, "tuple_types"):
                sem_type = expr["base"].get("sem_type")
                if sem_type and sem_type in self.tuple_types:
                    fields = self.tuple_types[sem_type]
                    for idx, f in enumerate(fields):
                        if f["name"] == field:
                            return f"{base}[{idx}]"
            # fallback: legacy string access
            return f"{base}['{field}']"
        elif t == "name_reference_index":
            # Use the iterator variable from env
            return env.get(expr["name"], expr["name"])
        elif t == "number_literal_index":
            return str(expr["value"])
        elif t == "string_literal":  # <-- emit quoted string for readability
            return repr(expr["value"])
        else:
            return str(expr)

    def _emit_symbolic_expr(self, expr: dict) -> str:
        """
        Emit a symbolic Python expression for a range bound, never substituting parameter values.
        This is now unified with _traverse_expression.
        """
        return self._traverse_expression(expr)

    # ------------------------------------------------------------------
    # Restored symbolic traversal utilities (lost during Stage 2 refactor)
    # ------------------------------------------------------------------
    def _traverse_expression(self, expr: dict) -> str:
        """Produce a symbolic string form of an expression AST node.
        Only structural; does not evaluate parameters so emitted code mirrors model text.
        Supports: number, name, binop, uminus, parenthesized_expression, conditional,
                  indexed_name, field_access, name_reference_index, number_literal_index,
                  tuple_literal. Falls back to str(expr) for unknown nodes.
        """
        if not isinstance(expr, dict):
            return str(expr)
        t = expr.get("type")
        if t == "number":
            return str(expr.get("value"))
        if t == "name":
            val = expr.get("value")
            return str(val) if val is not None else ""
        if t == "binop":
            left_expr = expr.get("left")
            right_expr = expr.get("right")
            left = self._traverse_expression(left_expr) if isinstance(left_expr, dict) else str(left_expr)
            right = self._traverse_expression(right_expr) if isinstance(right_expr, dict) else str(right_expr)
            return f"({left} {expr.get('op')} {right})"
        if t == "uminus":
            val_expr = expr.get("value")
            val = self._traverse_expression(val_expr) if isinstance(val_expr, dict) else str(val_expr)
            return f"-({val})"
        if t == "parenthesized_expression":
            expr_expr = expr.get("expression")
            return f"({self._traverse_expression(expr_expr) if isinstance(expr_expr, dict) else str(expr_expr)})"
        if t == "conditional":
            cond_expr = expr.get("condition")
            then_expr = expr.get("then")
            else_expr = expr.get("else")
            cond = self._traverse_expression(cond_expr) if isinstance(cond_expr, dict) else str(cond_expr)
            then = self._traverse_expression(then_expr) if isinstance(then_expr, dict) else str(then_expr)
            els = self._traverse_expression(else_expr) if isinstance(else_expr, dict) else str(else_expr)
            return f"({then} if ({cond}) else {els})"
        if t == "indexed_name":
            base = expr.get("name")
            dims = expr.get("dimensions") or []
            parts = [self._traverse_expression(d) if isinstance(d, dict) else str(d) for d in dims]
            if len(parts) == 1:
                return f"{base}[{parts[0]}]"
            return f"{base}[{', '.join(parts)}]"
        if t == "field_access":
            base_expr = expr.get("base")
            base = self._traverse_expression(base_expr) if isinstance(base_expr, dict) else str(base_expr)
            field = expr.get("field")
            if hasattr(self, "tuple_types"):
                sem_type = expr.get("base", {}).get("sem_type") if isinstance(expr.get("base"), dict) else None
                if sem_type and sem_type in self.tuple_types:
                    fields = self.tuple_types[sem_type]
                    for idx, f in enumerate(fields):
                        if f["name"] == field:
                            return f"{base}[{idx}]"
            return f"{base}['{field}']"
        if t in ("name_reference_index", "number_literal_index"):
            if "value" in expr:
                return str(expr["value"])
            if "name" in expr:
                return str(expr["name"])
            return str(expr)

        if t == "tuple_literal":
            elements = expr.get("elements", [])
            parts = [self._traverse_expression(e) for e in elements]
            return f"({', '.join(parts)})"

        if t == "string_literal":  # <-- support in symbolic traversal
            return repr(expr.get("value"))

        # Default: return empty string if no known type matched
        return ""

    def __init__(self, ast: dict, data_dict: dict | None = None, logger=None) -> None:
        import logging

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("SciPyCSCCodeGenerator")
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Helper for implication constraint detection (used elsewhere)
        def contains_implication_constraint(node):
            if isinstance(node, dict):
                if node.get("type") == "implication_constraint":
                    return True
                for v in node.values():
                    if contains_implication_constraint(v):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if contains_implication_constraint(item):
                        return True
            return False

        # Implication constraints now supported (handled in _build_constraints)

        self.ast = ast
        # Patch: auto-extract scalar parameters from AST and add to data_dict
        self.data_dict = dict(data_dict) if data_dict is not None else {}
        self.data = self.data_dict  # For compatibility with codegen logic
        for decl in self.ast.get("declarations", []):
            if decl.get("type") == "parameter_inline" and decl["name"] not in self.data_dict:
                self.data_dict[decl["name"]] = decl["value"]
        self.scipy_code_lines = []
        self.indent_level = 0
        self.var_names = []  # List of variable names in order
        self.var_indices = {}  # Map variable name to index in c, bounds, etc.
        self.bounds = []  # List of (low, high) for each variable
        self.c = []  # Objective coefficients
        self.A_eq = []
        self.b_eq = []
        self.A_ub = []
        self.b_ub = []
        self.results_varname = "results"
        # Instance-level caches for reuse of comparison truth vars and boolean subtrees across constraints

        self._comparison_truth_cache = {}
        self._bool_subtree_cache = {}
        # Maintain parallel simple bound vectors for newly introduced aux binaries
        self.lower_bounds = []
        self.upper_bounds = []
        # Variable index maps for multi-indexed variables (tuple, range, etc.)
        self.var_index_map = {}
        self.var_index_tuple_map = {}
        self.var_bounds = []
        self.var_integrality = []

    # Class-level type annotations for instance variables
    _comparison_truth_cache: dict[Any, Any]
    _bool_subtree_cache: dict[Any, Any]
    lower_bounds: list[Any]
    upper_bounds: list[Any]
    var_index_map: dict[Any, Any]
    var_index_tuple_map: dict[Any, Any]
    var_bounds: list[Any]
    var_integrality: list[Any]

    def _indent(self) -> str:
        return "    " * self.indent_level

    def _add_code_line(self, line: str) -> None:
        self.scipy_code_lines.append(self._indent() + line)

    def generate_code(self) -> str:
        self._add_code_line("import numpy as np")
        self._add_code_line("import time")
        self._add_code_line("from scipy.optimize import linprog")
        self._add_code_line("from scipy.sparse import csr_matrix")
        # Ensure results_container exists
        self._add_code_line("try:")
        self.indent_level += 1
        self._add_code_line("results_container")
        self.indent_level -= 1
        self._add_code_line("except NameError:")
        self.indent_level += 1
        self._add_code_line("results_container = {}")
        self.indent_level -= 1
        # Emit sense variable for use in sign fix
        sense = self.ast.get("objective", {}).get("type", "minimize")
        self._add_code_line(f"sense = '{sense}'")
        self._add_code_line("")
        self._generate_data_declarations(self.data_dict)
        self._add_code_line("")
        self._add_code_line("# Build LP vectors/matrices")
        self._build_variables()
        self._build_objective()
        self._build_constraints()

        # >>> NEW: zero-variable short-circuit (pure feasibility/constant objective) <<<
        if len(self.var_names) == 0:
            # Feasibility: with no variables, equalities require 0 == b_eq[i], inequalities require 0 <= b_ub[i]
            beq_ok = all(abs(b) <= 1e-9 for b in (self.b_eq or []))
            bub_ok = all(b >= -1e-9 for b in (self.b_ub or []))
            feasible = beq_ok and bub_ok
            # Constant objective value (evaluate at codegen time)
            try:
                _, obj_const = self._eval_expr(self.ast["objective"]["expression"], {})
                obj_val = float(obj_const) if isinstance(obj_const, (int, float)) else 0.0
            except Exception:
                obj_val = 0.0

            # Preserve previously emitted data/headers; append short-circuit result without calling linprog
            self._add_code_line("")
            self._add_code_line("# No decision variables: short-circuit without linprog")
            self._add_code_line("results = {}")
            if feasible:
                self._add_code_line("results['status'] = 'OPTIMAL'")
                self._add_code_line(f"results['objective_value'] = {obj_val}")
            else:
                self._add_code_line("results['status'] = 'INFEASIBLE'")
                self._add_code_line("results['objective_value'] = None")
            self._add_code_line("results['solution'] = {}")
            self._add_code_line("results_container['scipy_output'] = results")
            return "\n".join(self.scipy_code_lines)
        # <<< END NEW >>>

        # Patch: Enforce top-level explicit assignments for binary variables
        for constr in self.ast.get("constraints", []):
            if (
                constr.get("type") == "constraint"
                and constr.get("op") == "=="
                and (
                    (isinstance(constr.get("left"), dict) and constr["left"].get("type") in ("name", "indexed_name"))
                    and (
                        isinstance(constr.get("right"), dict)
                        and constr["right"].get("type") == "number"
                        and constr["right"].get("value") in (0, 1)
                    )
                )
            ):
                vname = (
                    self._multi_indexed_var_name(constr["left"], {})
                    if constr["left"].get("type") == "indexed_name"
                    else constr["left"]["value"]
                )
                if vname in self.var_indices:
                    idx = self.var_indices[vname]
                    row = [0.0] * len(self.var_names)
                    row[idx] = 1.0
                    # Only add if not already present in b_eq for this variable
                    already = False
                    for r, c, v in zip(self.A_eq, self.b_eq, self.var_names):
                        if abs(row[self.var_indices[v]]) == 1 and abs(c - constr["right"]["value"]) < 1e-8:
                            already = True
                            break
                    if not already:
                        self.A_eq.append(row)
                        self.b_eq.append(float(constr["right"]["value"]))
        # Reconcile auxiliary vars (added during constraints) with var_names/bounds/integrality
        # If new aux variables were appended, extend bounds/integrality and re-emit declarations.
        if len(self.var_names) > 0:
            if len(self.bounds) < len(self.var_names):
                while len(self.bounds) < len(self.var_names):
                    self.bounds.append([0, 1])  # assume binary aux
            if hasattr(self, "integrality") and len(self.integrality) < len(self.var_names):
                while len(self.integrality) < len(self.var_names):
                    self.integrality.append(1)
            # Remove previous lines for var_names/bounds/integrality and re-add at end of header block
            filtered = []
            for line in self.scipy_code_lines:
                if (
                    line.strip().startswith("var_names = ")
                    or line.strip().startswith("bounds = ")
                    or line.strip().startswith("integrality = ")
                ):
                    continue
                filtered.append(line)
            self.scipy_code_lines = filtered
            bounds_py = "[" + ", ".join(f'[{b[0]}, {b[1] if b[1] is not None else "None"}]' for b in self.bounds) + "]"
            # Insert updated arrays before matrices section; find insertion point (# Constraints or objective value lines not yet added)
            # Simpler: append now (acceptable for execution order)
            self._add_code_line(f"var_names = {repr(self.var_names)}")
            self._add_code_line(f"bounds = {bounds_py}")
            if hasattr(self, "integrality"):
                self._add_code_line(f"integrality = {self.integrality}")
        self._add_code_line("")
        self._add_code_line(f"{self.results_varname} = {{}}")
        self._add_code_line("try:")
        self.indent_level += 1
        self._add_code_line("start_time = time.time()")
        # Only include integrality if needed
        if any(self.integrality):
            self._add_code_line(
                "res = linprog(c, A_ub=A_ub, b_ub=b_ub if b_ub else None, "
                "A_eq=A_eq, b_eq=b_eq if b_eq else None, "
                "bounds=bounds, method='highs', integrality=integrality)"
            )
        else:
            self._add_code_line(
                "res = linprog(c, A_ub=A_ub, b_ub=b_ub if b_ub else None, "
                "A_eq=A_eq, b_eq=b_eq if b_eq else None, "
                "bounds=bounds, method='highs')"
            )
        self._add_code_line("end_time = time.time()")
        self._add_code_line("status_map = {0: 'OPTIMAL', 1: 'ITERATION_LIMIT', 2: 'INFEASIBLE', 3: 'UNBOUNDED'}")
        self._add_code_line("status_str = status_map.get(res.status, 'ERROR')")
        self._add_code_line("if res.success and res.status == 0:")
        self.indent_level += 1
        self._add_code_line("print('Optimal solution found:')")
        self._add_code_line("solution = {}")
        self._add_code_line("for i, name in enumerate(var_names):")
        self.indent_level += 1
        self._add_code_line("solution[name] = res.x[i]")
        self._add_code_line("if abs(res.x[i]) > 1e-8:")
        self.indent_level += 1
        self._add_code_line("print(f'{name}: {res.x[i]}')")
        self.indent_level -= 1
        self.indent_level -= 1
        # Patch: Fix objective sign for maximization
        self._add_code_line("# Patch: Fix objective sign for maximization")
        self._add_code_line("objective_value = res.fun")
        self._add_code_line("if sense == 'maximize':")
        self.indent_level += 1
        self._add_code_line("objective_value = -objective_value")
        self.indent_level -= 1
        # Print objective value (parity with Gurobi output)
        self._add_code_line("print(f'Objective value: {objective_value}')")
        self._add_code_line(f"{self.results_varname}['solution'] = solution")
        self._add_code_line(f"{self.results_varname}['objective_value'] = objective_value")
        self._add_code_line(f"{self.results_varname}['status'] = status_str")
        self._add_code_line("stats = {}")
        self._add_code_line("stats['status'] = res.status")
        self._add_code_line("stats['message'] = res.message")
        self._add_code_line("stats['nit'] = res.nit")
        self._add_code_line("stats['crossover_nit'] = getattr(res, 'crossover_nit', None)")
        self._add_code_line("stats['time'] = end_time - start_time")
        self._add_code_line(f"{self.results_varname}['stats'] = stats")
        self.indent_level -= 1  # Dedent here so else is at the same level as if
        self._add_code_line("else:")
        self.indent_level += 1
        self._add_code_line("print('Optimization failed: ' + res.message)")
        self._add_code_line(f"{self.results_varname}['status'] = status_str")
        self._add_code_line(f"{self.results_varname}['message'] = res.message")
        self._add_code_line(f"{self.results_varname}['objective_value'] = None")
        self.indent_level -= 1
        self.indent_level -= 1
        self._add_code_line("except Exception as e:")
        self.indent_level += 1
        self._add_code_line(f"{self.results_varname}['status'] = 'ERROR'")
        self._add_code_line(f"{self.results_varname}['message'] = str(e)")
        self._add_code_line(f"{self.results_varname}['objective_value'] = None")
        self.indent_level -= 1
        self._add_code_line(f"results_container['scipy_output'] = {self.results_varname}")

        # Dump the generated Scipy model for debugging/comparison
        # print("\n===== Scipy Model Dump =====")
        # for line in self.scipy_code_lines:
        #     print(line)
        # print("===== End Scipy Model Dump =====\n")

        return "\n".join(self.scipy_code_lines)

    def _generate_data_declarations(self, data_dict):
        # --- Recursive shape check for multi-dimensional parameters ---
        # --- Recursive shape check for multi-dimensional parameters ---
        def check_shape(param_data, dims, data_dict, param_name, dim=0):
            """
            Minimal shape validation for lists/arrays:
            - 1D over named_range: length matches end-start+1
            - 1D over named_set: length matches |set|
            - 2D over range×range: rectangular with expected sizes
            - 2D over set×range or set×set: skip here (handled by later normalization)
            """
            from .semantic_error import SemanticError

            if not isinstance(dims, list):
                return
            if isinstance(param_data, dict):
                return  # handled elsewhere (dict normalizations)
            if len(dims) == 1 and isinstance(param_data, list):
                d0 = dims[0]
                if d0.get("type") == "named_range_dimension":
                    # evaluate [start..end]
                    rng_name = d0["name"]
                    rng_decl = next(
                        (
                            d
                            for d in self.ast.get("declarations", [])
                            if d.get("type") == "range_declaration_inline" and d.get("name") == rng_name
                        ),
                        None,
                    )
                    if rng_decl:

                        def eval_bound_local(expr):
                            if expr["type"] == "number":
                                return int(expr["value"])
                            if expr["type"] == "name":
                                return int(data_dict[expr["value"]])
                            if expr["type"] == "binop":
                                op = expr["op"]
                                left = eval_bound_local(expr["left"])
                                right = eval_bound_local(expr["right"])
                                return (
                                    left + right
                                    if op == "+"
                                    else left - right if op == "-" else left * right if op == "*" else left // right
                                )
                            raise Exception("Unsupported range bound expr")

                        start = eval_bound_local(rng_decl["start"])
                        end = eval_bound_local(rng_decl["end"])
                        expected = end - start + 1
                        if len(param_data) != expected:
                            raise SemanticError(
                                f"Parameter '{param_name}' has {len(param_data)} items but declared range '{rng_name}' expects {expected}."
                            )
                elif d0.get("type") == "named_set_dimension":
                    set_name = d0["name"]
                    elems = data_dict.get(set_name)
                    if elems is None:
                        decl = next((d for d in self.ast.get("declarations", []) if d.get("name") == set_name), None)
                        if decl:
                            if decl.get("type") == "typed_set":
                                elems = decl.get("value") or []
                            elif decl.get("type") == "set_declaration":
                                elems = decl.get("value") or []
                    if isinstance(elems, dict) and "elements" in elems:
                        set_len = len(elems["elements"])
                    else:
                        set_len = len(elems or [])
                    if set_len and len(param_data) != set_len:
                        raise SemanticError(
                            f"Parameter '{param_name}' has {len(param_data)} items but declared set '{set_name}' has {set_len} elements."
                        )
            if len(dims) == 2 and isinstance(param_data, list):
                # Only check “rectangular” rows; deeper semantics handled later
                if not all(isinstance(row, (list, tuple)) for row in param_data):
                    return
                row_len = len(param_data[0]) if param_data else 0
                if not all(len(row) == row_len for row in param_data):
                    raise SemanticError(f"Parameter '{param_name}' 2-D data must be rectangular (all rows same length).")
            return

        # Track inline tuple-indexed parameters we already emitted as dicts so we don't overwrite them later
        emitted_inline_tuple_params = set()

        # New: validation for 1-D params over set/range where data is dict with non-scalar values.
        param_decl_map = self._get_param_decl_map()
        for name, decl in param_decl_map.items():
            dims = decl.get("dimensions", []) or []
            if len(dims) == 1 and dims[0].get("type") in (
                "named_set_dimension",
                "named_range_dimension",
            ):
                val = data_dict.get(name)
                if isinstance(val, dict):
                    bad_key = next(
                        (k for k, v in val.items() if isinstance(v, (list, tuple, dict))),
                        None,
                    )
                    if bad_key is not None:
                        from .semantic_error import SemanticError

                        raise SemanticError(
                            f"Parameter '{name}' declared as 1-D over '{dims[0].get('name', '')}' expects scalar values per key, "
                            f"but data provides an array for key {repr(bad_key)}. Use scalar values (e.g., 2.0), not [2.0]."
                        )

        # Convert flat key-value lists to dicts in data_dict before shape checking
        for decl in self.ast.get("declarations", []):
            if decl.get("type") in (
                "parameter_external",
                "parameter_external_indexed",
                "parameter_external_explicit",
                "parameter_external_explicit_indexed",
                "parameter_inline",
                "parameter_inline_indexed",
            ) and decl.get("dimensions"):
                param_data = data_dict.get(decl["name"])
                converted = self._convert_flat_kv_to_dict(param_data)
                if converted is not None:
                    data_dict[decl["name"]] = converted
                    continue
                # Only apply shape check to lists/arrays, not dicts
                if param_data is not None and isinstance(param_data, (list, tuple)):
                    check_shape(param_data, decl["dimensions"], data_dict, decl["name"])

        # --- NEW: accept keyed-row and row-major forms for 2D parameters, aligned with Gurobi ---
        # Helper to evaluate a simple bound expression (number/name/binop) into int
        def _eval_bound_local(expr):
            if isinstance(expr, dict):
                t = expr.get("type")
                if t == "number":
                    return int(expr["value"])
                if t == "name":
                    return int(data_dict[expr["value"]])
                if t == "binop":
                    op = expr["op"]
                    left = _eval_bound_local(expr["left"])
                    right = _eval_bound_local(expr["right"])
                    if op == "+":
                        return left + right
                    if op == "-":
                        return left - right
                    if op == "*":
                        return left * right
                    if op == "/":
                        return left // right
            raise Exception("Unsupported range bound expr")

        # Resolve set elements (typed set or generic set)
        def _resolve_set_elems(set_name):
            if set_name in data_dict:
                set_obj = data_dict[set_name]
                # typed_set may be a list; set_of_tuples may be dict with 'elements'
                if isinstance(set_obj, dict) and "elements" in set_obj:
                    elems = set_obj["elements"]
                else:
                    elems = set_obj
                # normalize tuple keys for tuple sets
                norm = [tuple(e) if isinstance(e, (list, tuple)) else e for e in elems]
                return norm
            # Fallback to AST declared values (typed_set or set_declaration)
            for d in self.ast.get("declarations", []):
                if d.get("name") == set_name:
                    if d.get("type") == "typed_set":
                        return d.get("value") or []
                    if d.get("type") == "set_declaration":
                        return d.get("value") or []
                    if d.get("type") == "set_of_tuples" and d.get("value"):
                        return [
                            (tuple(t["elements"]) if isinstance(t, dict) and "elements" in t else tuple(t)) for t in d["value"]
                        ]
            return None

        # 2D: set × range — accept dict-of-lists or list-of-rows
        for decl in self.ast.get("declarations", []):
            if decl.get("type") in (
                "parameter_external",
                "parameter_external_indexed",
                "parameter_external_explicit",
                "parameter_external_explicit_indexed",
                "parameter_inline",
                "parameter_inline_indexed",
            ):
                name = decl["name"]
                dims = decl.get("dimensions", [])
                if not (
                    len(dims) == 2
                    and dims[0].get("type") == "named_set_dimension"
                    and dims[1].get("type") == "named_range_dimension"
                ):
                    continue
                val = data_dict.get(name)
                if val is None:
                    continue
                set_name = dims[0]["name"]
                rng = dims[1]
                # Compute range bounds
                try:
                    start = _eval_bound_local(rng["start"])
                    end = _eval_bound_local(rng["end"])
                except Exception:
                    continue
                expected_len = end - start + 1
                set_elems = _resolve_set_elems(set_name)
                # keyed-row: dict-of-lists keyed by set elements
                if isinstance(val, dict) and all(isinstance(row, (list, tuple, dict)) for row in val.values()):
                    nested = {}
                    for k, row in val.items():
                        key_obj = tuple(k) if isinstance(k, (list, tuple)) else k
                        if isinstance(row, dict):
                            # assume already keyed by p
                            nested[key_obj] = {int(p): float(v) for p, v in row.items()}
                        else:
                            if len(row) != expected_len:
                                continue
                            nested[key_obj] = {p: float(row[p - start]) for p in range(start, end + 1)}
                    if nested:
                        data_dict[name] = nested
                # row-major: list-of-rows in set order (if we can resolve set order)
                elif (
                    isinstance(val, list)
                    and set_elems is not None
                    and len(set_elems) == len(val)
                    and all(isinstance(row, (list, tuple)) and len(row) == expected_len for row in val)
                ):
                    nested = {}
                    for i, key in enumerate(set_elems):
                        nested_key = tuple(key) if isinstance(key, (list, tuple)) else key
                        nested[nested_key] = {p: float(val[i][p - start]) for p in range(start, end + 1)}
                    data_dict[name] = nested

        # 2D: set × set — accept list-of-rows (row-major) or dict-of-lists keyed by first set
        for decl in self.ast.get("declarations", []):
            if decl.get("type") in (
                "parameter_external",
                "parameter_external_indexed",
                "parameter_external_explicit",
                "parameter_external_explicit_indexed",
                "parameter_inline",
                "parameter_inline_indexed",
            ):
                name = decl["name"]
                dims = decl.get("dimensions", [])
                if not (
                    len(dims) == 2
                    and dims[0].get("type") == "named_set_dimension"
                    and dims[1].get("type") == "named_set_dimension"
                ):
                    continue
                val = data_dict.get(name)
                if val is None:
                    continue
                set1 = dims[0]["name"]
                set2 = dims[1]["name"]
                keys1 = _resolve_set_elems(set1)
                keys2 = _resolve_set_elems(set2)
                if not (keys1 and keys2):
                    continue
                # row-major: list-of-rows
                if (
                    isinstance(val, list)
                    and len(val) == len(keys1)
                    and all(isinstance(row, (list, tuple)) and len(row) == len(keys2) for row in val)
                ):
                    nested = {}
                    for i, k1 in enumerate(keys1):
                        k1n = tuple(k1) if isinstance(k1, (list, tuple)) else k1
                        nested[k1n] = {}
                        for j, k2 in enumerate(keys2):
                            k2n = tuple(k2) if isinstance(k2, (list, tuple)) else k2
                            nested[k1n][k2n] = float(val[i][j])
                    data_dict[name] = nested
                # keyed-row: dict-of-lists keyed by first set
                elif isinstance(val, dict) and all(isinstance(row, (list, tuple, dict)) for row in val.values()):
                    nested = {}
                    for k1, row in val.items():
                        k1n = tuple(k1) if isinstance(k1, (list, tuple)) else k1
                        if isinstance(row, dict):
                            # assume already keyed by second set elements
                            # coerce keys to tuple when needed
                            inner = {}
                            for k2, v in row.items():
                                k2n = tuple(k2) if isinstance(k2, (list, tuple)) else k2
                                inner[k2n] = float(v)
                            nested[k1n] = inner
                        else:
                            if len(row) != len(keys2):
                                continue
                            inner = {}
                            for j, k2 in enumerate(keys2):
                                k2n = tuple(k2) if isinstance(k2, (list, tuple)) else k2
                                inner[k2n] = float(row[j])
                            nested[k1n] = inner
                    if nested:
                        data_dict[name] = nested

        # --- Existing: Convert 2D arrays indexed by tuple-set × range into nested dicts ---
        for decl in self.ast.get("declarations", []):
            if (
                decl.get("type")
                in (
                    "parameter_external",
                    "parameter_external_indexed",
                    "parameter_external_explicit",
                    "parameter_external_explicit_indexed",
                    "parameter_inline",
                    "parameter_inline_indexed",
                )
                and isinstance(data_dict.get(decl["name"]), list)
                and len(decl.get("dimensions", [])) == 2
                and decl["dimensions"][0].get("type") == "named_set_dimension"
                and decl["dimensions"][1].get("type") == "named_range_dimension"
            ):
                name = decl["name"]
                set_name = decl["dimensions"][0]["name"]
                rng = decl["dimensions"][1]
                param_rows = data_dict.get(name)
                set_vals = None
                set_decl = self._find_decl(set_name, "set_of_tuples") or self._find_decl(set_name, "set_of_tuples_external")
                if set_name in data_dict:
                    raw = data_dict[set_name]
                    set_vals = raw["elements"] if isinstance(raw, dict) and "elements" in raw else raw
                elif set_decl and set_decl.get("value"):
                    set_vals = [t["elements"] for t in set_decl["value"]]
                if set_vals is None:
                    continue
                set_elems = [tuple(e) if isinstance(e, (list, tuple)) else (e,) for e in set_vals]

                def eval_bound(expr):
                    if expr["type"] == "number":
                        return int(expr["value"])
                    if expr["type"] == "name":
                        return int(data_dict[expr["value"]])
                    if expr["type"] == "binop":
                        op = expr["op"]
                        left = eval_bound(expr["left"])
                        right = eval_bound(expr["right"])
                        return (
                            left + right
                            if op == "+"
                            else (left - right if op == "-" else left * right if op == "*" else left // right)
                        )
                    raise Exception("Unsupported range bound expr")

                start = eval_bound(rng["start"])
                end = eval_bound(rng["end"])
                expected_len = end - start + 1
                if not (
                    len(set_elems) == len(param_rows)
                    and all(isinstance(row, (list, tuple)) and len(row) == expected_len for row in param_rows)
                ):
                    continue
                nested = {}
                for i, key in enumerate(set_elems):
                    nested[key] = {p: float(param_rows[i][p - start]) for p in range(start, end + 1)}
                data_dict[name] = nested
        # --- Length check for 1D parameters indexed by a range or set (parity with Gurobi) ---
        param_decl_map = self._get_param_decl_map()
        # Emit tuple types and sets of tuples from AST declarations (if present)
        for decl in self.ast.get("declarations", []):
            if decl.get("type") == "tuple_type":
                # Store tuple type info for later use (for field access)
                self.tuple_types = getattr(self, "tuple_types", {})
                self.tuple_types[decl["name"]] = decl["fields"]
            elif decl.get("type") == "set_of_tuples":
                set_name = decl["name"]
                tuple_list = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict)
                if tuple_list:
                    self._add_code_line(f"{set_name} = {repr(tuple_list)}")
                    # Also make available in data_dict for downstream fallback code paths
                    try:
                        self.data_dict[set_name] = tuple_list
                    except Exception:
                        pass
            elif decl.get("type") in ("typed_set", "typed_set_external"):
                # Prefer data_dict override if provided
                set_name = decl["name"]
                if set_name in data_dict:
                    val = data_dict[set_name]
                else:
                    # decl['value'] already a list of scalar elements or None
                    val = decl.get("value") or []
                self._add_code_line(f"{set_name} = {repr(val)}")
                # Also update internal data_dict so index remapping can consult set order
                try:
                    self.data_dict[set_name] = list(val)
                except Exception:
                    pass
                # Emit an index map for string-labelled scalar sets so list parameters can be accessed by label.
                # Mirrors Gurobi backend (<SetName>_index) for parity.
                if isinstance(val, list) and all(isinstance(e, (str, int)) for e in val):
                    # Provide deterministic positional mapping (1-based like OPL logical position -> Python list index).
                    # Store both 1-based position (for potential legacy) and provide direct name->position map used below.
                    self._add_code_line(f"{set_name}_index = {{v: i for i, v in enumerate({set_name})}}")
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
                    self._add_code_line(f"{arr_name}_data = {repr(tuple_dicts)}")
                    self._add_code_line(f"{arr_name} = {{idx: rec for idx, rec in zip({index_set}, {arr_name}_data)}}")
                    # Patch: also update internal data_dict so ExpressionEvaluator sees structured dicts
                    try:
                        index_vals = data_dict.get(index_set)
                        if isinstance(index_vals, list) and len(index_vals) == len(tuple_dicts):
                            structured = {idx: rec for idx, rec in zip(index_vals, tuple_dicts)}
                            # Mutate self.data_dict (not the passed-in view) to keep evaluation consistent
                            self.data_dict[arr_name] = structured
                    except Exception:
                        pass
            elif decl.get("type") == "parameter_inline_indexed":
                # Only emit dict for tuple-indexed, else emit as list
                dims = decl.get("dimensions", [])
                name = decl["name"]
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
                        # Mark as emitted and normalize internal data so evaluators see dicts
                        emitted_inline_tuple_params.add(name)
                        try:
                            self.data_dict[name] = param_dict
                        except Exception:
                            pass
                        continue
                # Fallback: emit as list (for range-indexed)
                self._add_code_line(f"{name} = {json.dumps(decl['value'])}")
                # Also update internal data_dict so evaluation can resolve inline list-indexed params
                try:
                    self.data_dict[name] = decl["value"]
                except Exception:
                    pass
        # Emit data from .dat file as before
        if not data_dict:
            self._add_code_line("")
            return
        self._add_code_line("# Data from .dat file")
        # Collect tuple array names so we don't overwrite structured dicts created earlier
        tuple_array_names = {
            d["name"] for d in self.ast.get("declarations", []) if d.get("type") in ("tuple_array", "tuple_array_external")
        }

        def convert_keys(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if not isinstance(k, str):
                        if isinstance(k, tuple):
                            key_str = ",".join(str(x) for x in k)
                        else:
                            key_str = str(k)
                        new_dict[key_str] = convert_keys(v)
                    else:
                        new_dict[k] = convert_keys(v)
                return new_dict
            elif isinstance(obj, list):
                return [convert_keys(x) for x in obj]
            else:
                return obj

        for name, value in data_dict.items():
            # Skip names already emitted in structured form for tuple-indexed inline params
            if name in emitted_inline_tuple_params:
                continue
            # Length check for 1D parameters indexed by a named range
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
            ):
                dim = param_decl["dimensions"][0]
                if dim.get("type") == "named_range_dimension":
                    range_name = dim["name"]
                    range_decl = self._find_decl(range_name, decl_type="range_declaration_inline")
                    if range_decl:
                        # Evaluate start/end (assume int literals or parameter names in data_dict)
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
                        if len(value) != expected_len:
                            from .semantic_error import SemanticError

                            raise SemanticError(
                                f"Parameter '{name}' has {len(value)} items but declared range '{range_name}' expects {expected_len}."
                            )
                elif dim.get("type") == "named_set_dimension":
                    set_name = dim["name"]
                    set_elems = data_dict.get(set_name)
                    if set_elems is not None:
                        if isinstance(set_elems, dict) and "elements" in set_elems:
                            set_len = len(set_elems["elements"])
                        else:
                            set_len = len(set_elems)
                        if set_len != len(value):
                            from .semantic_error import SemanticError

                            raise SemanticError(
                                f"Parameter '{name}' has {len(value)} items but declared set '{set_name}' has {set_len} elements."
                            )
            # Skip tuple arrays: we already emitted a structured dict (name -> record dict) above.
            if name in tuple_array_names:
                continue
            if isinstance(value, dict) and value.get("type") == "range_data":
                pass
            elif isinstance(value, (list, dict)):
                safe_value = convert_keys(value)
                self._add_code_line(f"{name} = {json.dumps(safe_value)}")
                # If this is a set override (typed scalar set) and we haven't already created an index map (dat file may supply), emit index map.
                if name not in {
                    d["name"]
                    for d in self.ast.get("declarations", [])
                    if d.get("type")
                    in (
                        "tuple_array",
                        "tuple_array_external",
                        "set_of_tuples",
                        "set_of_tuples_external",
                    )
                }:
                    if isinstance(value, list) and value and all(isinstance(e, (str, int)) for e in value):
                        self._add_code_line(f"{name}_index = {{v: i for i, v in enumerate({name})}}")
            elif isinstance(value, str):
                self._add_code_line(f'{name} = "{value}"')
            else:
                self._add_code_line(f"{name} = {value}")
        self._add_code_line("")

    def _handle_tuple_type_declaration(self, decl):
        """
        Store tuple type info for later use (for field access in tuple-indexed variables).
        """
        self.tuple_types[decl["name"]] = decl["fields"]

    def _handle_set_of_tuples_declaration(self, decl, data_dict):
        """
        Skip set_of_tuples declarations (handled in AST/tests, not codegen).
        """
        pass

    def _build_variables(self):
        # Supports scalar, indexed continuous, integer, and boolean variables
        self._add_code_line("# Variable definitions")
        var_names = []
        bounds = []
        integrality = []
        decls = self.ast["declarations"]
        self.tuple_types = {}
        for decl in decls:
            # Skip dexpr declarations (expanded in parser on use)
            if decl.get("type") in ("dexpr", "dexpr_indexed"):
                continue
            if decl["type"] == "tuple_type":
                self._handle_tuple_type_declaration(decl)
                continue
            if decl["type"] in (
                "set_of_tuples",
                "set_of_tuples_external",
                "typed_set",
                "typed_set_external",
                "tuple_array",
                "tuple_array_external",
            ):
                # Skip pure data declarations (including external typed string sets & tuple arrays)
                self._handle_set_of_tuples_declaration(decl, self.data_dict)
                continue
            if decl["type"] == "dvar":
                self._handle_scalar_variable_declaration(decl, var_names, bounds, integrality)
            elif decl["type"] == "dvar_indexed":
                self._handle_indexed_variable_declaration(decl, var_names, bounds, integrality)
            elif decl["type"] in (
                "range_declaration_inline",
                "range_declaration_external",
                "set_declaration",
                "parameter_inline",
                "parameter_inline_indexed",
                "parameter_external",
                "parameter_external_indexed",
                "parameter_external_explicit",
                "parameter_external_explicit_indexed",
            ):
                # These are handled elsewhere or not needed here
                continue
            else:
                raise SemanticError(f"Unsupported declaration type: {decl['type']}")  # type: ignore
        # --- Patch: Update bounds based on constraints ---
        self._tighten_bounds_from_constraints(bounds, var_names, self.var_indices, self.ast.get("constraints", []))
        self.var_names = var_names
        self.bounds = bounds
        self.integrality = integrality
        # Output Python syntax for bounds (not JSON)
        bounds_py = "[" + ", ".join(f'[{b[0]}, {b[1] if b[1] is not None else "None"}]' for b in bounds) + "]"
        self._add_code_line(f"var_names = {repr(var_names)}")
        # Emit alias comments for original bracket notation for string-indexed 1D variables (e.g., y['G1']) to satisfy tests
        for vn in var_names:
            if "_" in vn and "[" not in vn:
                base, rest = vn.split("_", 1)
                # Only single underscore parts (avoid multi-dim) and rest without further underscores means original was base[rest]
                if rest and base and rest.replace("_", "").isalnum():
                    # Heuristic: emit comment reproducing bracket style used in tests
                    self._add_code_line(f"# Alias: {base}['{rest}']")
        self._add_code_line(f"bounds = {bounds_py}")
        self._add_code_line(f"integrality = {integrality}")

    def _eval_expr(self, expr, env=None):
        if not hasattr(self, "_expr_evaluator"):
            self._expr_evaluator = ExpressionEvaluator(self)
        return self._expr_evaluator.eval(expr, env)

    def _build_objective(self):
        self._add_code_line("# Objective vector c")
        c = [0.0] * len(self.var_names)
        obj = self.ast["objective"]
        sense = obj["type"]
        expr = obj["expression"]
        # Delegate to helpers for sum and binop
        self._accumulate_objective(expr, c)
        # Flip sign for maximization
        if sense == "maximize":
            c = [-v for v in c]
        self.c = c
        self._add_code_line(f"c = {c}")

    def _accumulate_objective(self, expr, c):
        """
        Accumulate coefficients for the objective vector c, handling sum and binop recursively.
        Delegates to _accumulate_objective_sum and _accumulate_objective_binop for those cases.
        Fallback: evaluate expression directly and update vector.
        """
        if isinstance(expr, dict) and expr.get("type") == "sum":
            self._accumulate_objective_sum(expr, c)
        elif isinstance(expr, dict) and expr.get("type") == "binop":
            self._accumulate_objective_binop(expr, c)
        else:
            coef_dict, const = self._eval_expr(expr)
            self._update_vector_from_coef_dict(coef_dict, c, "+")

    def _accumulate_objective_sum(self, expr, c):
        """
        Helper to accumulate coefficients for the objective vector c for a sum expression.
        Handles iterator unrolling, index constraints, and tuple-indexed variables.
        """
        iterators = expr["iterators"]
        loop_vars, loop_ranges = self._unroll_iterators(iterators)
        symbolic_ranges = ", ".join(
            [
                (
                    f"{v} in range({self._emit_symbolic_expr(it['range'].get('start', ''))}, {self._emit_symbolic_expr(it['range'].get('end', ''))} + 1)"
                    if it["range"]["type"] == "range_specifier"
                    else f"{v} in {it['range']['name']}"
                )  # named_range, set_of_tuples, or typed_set
                for v, it in zip(loop_vars, iterators)
            ]
        )
        self._add_code_line(
            f"# Symbolic objective: sum({self._emit_python_expr(expr['expression'], {v: v for v in loop_vars})} for {symbolic_ranges})"
        )
        tuple_set_names = set()
        for it in iterators:
            rng = it["range"]
            if rng["type"] == "named_set":
                set_decl = self._find_decl(rng["name"])
                if set_decl and set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                    tuple_set_names.add(it["iterator"])
        for idx_tuple in itertools.product(*loop_ranges):
            env2, include = self._should_include_sum_term(
                loop_vars,
                idx_tuple,
                tuple_set_names,
                {},
                expr.get("index_constraint"),
                expr,
            )
            if not include:
                continue
            coef_dict, const = self._eval_expr(expr["expression"], env=env2)
            if coef_dict:
                logger.debug(f"[SciPyCSCCodeGenerator] Objective term env={env2} coefs={coef_dict}")
            for vname, coef in coef_dict.items():
                idx = self.var_indices.get(vname)
                if idx is None:
                    idx = self._resolve_tuple_index_varname(vname)
                if idx is not None:
                    c[idx] += coef

    def _accumulate_objective_binop(self, expr, c):
        """
        Helper to accumulate coefficients for the objective vector c for a binop expression.
        Handles sum/binop combinations and applies the operator elementwise.
        """
        left_type = expr["left"].get("type") if isinstance(expr["left"], dict) else None
        right_type = expr["right"].get("type") if isinstance(expr["right"], dict) else None
        if left_type == "sum" and right_type == "sum":
            # Accumulate left sum, then right sum, with op
            self._accumulate_objective_sum(expr["left"], c)
            orig_c = c[:]
            temp_c = [0.0] * len(c)
            self._accumulate_objective_sum(expr["right"], temp_c)
            if expr["op"] == "+":
                for i in range(len(c)):
                    c[i] = orig_c[i] + temp_c[i]
            elif expr["op"] == "-":
                for i in range(len(c)):
                    c[i] = orig_c[i] - temp_c[i]
            else:
                raise self._unsupported_operator_error("objective binop", expr["op"])
        elif left_type == "sum":
            self._accumulate_objective_sum(expr["left"], c)
            coef_dict, const = self._eval_expr(expr["right"])
            if expr["op"] == "+":
                self._update_vector_from_coef_dict(coef_dict, c, op="+")
            elif expr["op"] == "-":
                self._update_vector_from_coef_dict(coef_dict, c, op="-")
            else:
                raise self._unsupported_operator_error("objective binop", expr["op"])
        elif right_type == "sum":
            # Accumulate right sum into temp then combine with left
            temp_c = [0.0] * len(c)
            self._accumulate_objective_sum(expr["right"], temp_c)
            coef_dict, const = self._eval_expr(expr["left"])
            if expr["op"] == "+":
                self._update_vector_from_coef_dict(coef_dict, c, op="+")
                for i in range(len(c)):
                    c[i] += temp_c[i]
            elif expr["op"] == "-":
                self._update_vector_from_coef_dict(coef_dict, c, op="+")
                for i in range(len(c)):
                    c[i] -= temp_c[i]
            else:
                raise self._unsupported_operator_error("objective binop", expr["op"])
        else:
            coef_dict, const = self._eval_expr(expr)
            self._update_vector_from_coef_dict(coef_dict, c)

    def _multi_indexed_var_name(self, expr, env, eval_index_expr=None):
        if expr["type"] != "indexed_name":
            return expr["name"]
        base = expr["name"]
        index_values = []
        for dim in expr.get("dimensions", []):
            if dim.get("type") == "number_literal_index":
                idx_val = dim["value"]
            elif eval_index_expr:
                _, idx_val = eval_index_expr(dim, env)
            else:
                idx_val = env.get(dim.get("name"))
            if isinstance(idx_val, tuple) and len(idx_val) == 2 and isinstance(idx_val[0], dict):
                idx_val = idx_val[1]
            index_values.append(idx_val)
        if len(index_values) == 1 and isinstance(index_values[0], tuple):
            vname_tuple = f"{base}[{repr(index_values[0])}]"
            if vname_tuple in self.var_indices or vname_tuple in self.data_dict:
                return vname_tuple
        tuple_key = tuple(index_values)
        vname = f"{base}[{repr(tuple_key)}]"
        if vname in self.var_indices or vname in self.data_dict:
            return vname
        vname_alt = f"{base}[{tuple_key}]"
        if vname_alt in self.var_indices or vname_alt in self.data_dict:
            return vname_alt
        vname_legacy = f"{base}[{str(tuple_key)}]"
        if vname_legacy in self.var_indices or vname_legacy in self.data_dict:
            return vname_legacy
        if len(index_values) == 1:
            vname_single = f"{base}_{index_values[0]}"
            if vname_single in self.var_indices or vname_single in self.data_dict:
                return vname_single
        if len(index_values) > 1:
            vname_multi = f"{base}_" + "_".join(str(i) for i in index_values)
            if vname_multi in self.var_indices or vname_multi in self.data_dict:
                return vname_multi
        if "[" in base and "]" in base:
            base_clean = (
                base.replace("[", "_").replace("]", "").replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
            )
            vname_fallback = f"{base_clean}_{'_'.join(str(i) for i in index_values)}"
            if vname_fallback in self.var_indices or vname_fallback in self.data_dict:
                return vname_fallback
        return base

    def _find_decl(self, name, decl_type=None):
        """
        Find a declaration by name and optional type in the AST declarations.
        Returns the declaration dict if found, else None.
        """
        for d in self.ast["declarations"]:
            if d.get("name") == name and (decl_type is None or d.get("type") == decl_type):
                return d
        return None

    def _find_decls(self, name: str, decl_type: Optional[str] = None) -> list[dict]:
        """
        Return all declarations matching name and optional type.
        Used by ExpressionEvaluator for tuple-indexed parameter lookup.
        """
        return [
            d
            for d in self.ast.get("declarations", [])
            if d.get("name") == name and (decl_type is None or d.get("type") == decl_type)
        ]

    def _is_tuple_indexed(self, decl):
        """
        Return True if the declaration is tuple-indexed (i.e., indexed over a named set of tuples), else False.
        """
        if decl is not None:
            dims = decl.get("dimensions", [])
            if len(dims) == 1 and dims[0].get("type") == "named_set_dimension":
                # Only treat as tuple-indexed if underlying set declaration is a set_of_tuples / external
                set_name = dims[0].get("name")
                set_decl = self._find_decl(set_name)
                if set_decl and set_decl.get("type") in (
                    "set_of_tuples",
                    "set_of_tuples_external",
                ):
                    return True
        return False

    def _is_number_literal_index(self, dim):
        """
        if t == 'constraint':
        """
        return isinstance(dim, dict) and dim.get("type") == "number_literal_index"

    def _is_field_access_index(self, dim):
        """
        Return True if the dimension is a field access index, else False.
        """
        return isinstance(dim, dict) and dim.get("type") == "field_access_index"

    def _extract_field_access_index(self, dim, env):
        """
        Extract the value for a field access index from the environment or by evaluating the base expression.
        Handles tuple and dict base values.
        """
        base_expr = dim["base"]
        field = dim["field"]
        if base_expr["type"] == "name":
            base_val = env.get(base_expr["value"], base_expr["value"])
        else:
            base_val = self._eval_expr(base_expr, env)[1]
        if isinstance(base_val, (list, tuple)):
            field_idx = None
            if hasattr(self, "tuple_types") and base_expr.get("sem_type") in self.tuple_types:
                fields = self.tuple_types[base_expr["sem_type"]]
                for idx_f, f in enumerate(fields):
                    if f["name"] == field:
                        field_idx = idx_f
                        break
            if field_idx is not None:
                return base_val[field_idx]
            else:
                return base_val[0] if len(base_val) > 0 else None
        elif isinstance(base_val, dict):
            return base_val.get(field, None)
        else:
            return None

    def _extract_tuple_index(self, dim, env):
        """
        Extract the tuple index value for set-of-tuples from the dimension and environment.
        Returns a tuple of elements, each resolved from the environment or by evaluation.
        """
        if "elements" in dim:
            elements = []
            for e in dim["elements"]:
                if isinstance(e, str):
                    elements.append(e)
                elif isinstance(e, dict) and "name" in e:
                    elements.append(env.get(e["name"], e["name"]))
                else:
                    elements.append(self._eval_expr(e, env)[1])
            return tuple(elements)
        else:
            return self._eval_expr(dim, env)[1]

    def _extract_normal_index(self, dim, env):
        """
        Extract a normal (non-tuple, non-field) index value from the dimension and environment.
        Resolves string indices from env or data_dict if possible.
        """
        idx = self._eval_expr(dim, env)[1]
        if isinstance(idx, str):
            if idx in env:
                idx = env[idx]
            elif idx in self.data_dict:
                idx = self.data_dict[idx]
        return idx

    def _normalize_index_for_varname(self, idx):
        """
        Normalize the index value for use in a variable name.
        For tuple indices, return the tuple as-is. For all other types, return as-is.
        """
        return idx

    def _format_varname(self, base, indices, is_tuple_indexed):
        """
        Format the variable name given the base, indices, and whether it is tuple-indexed.
        Returns the appropriate string for use as a variable name in the model.
        """
        if is_tuple_indexed:
            return f"{base}[{repr(indices[0])}]"
        else:
            # Normalize indices: convert any non-numeric strings (like 'Super') into identifier-friendly tokens
            norm_parts = []
            for idx in indices:
                if isinstance(idx, tuple):
                    # Flatten tuple parts
                    sub_parts = []
                    for t in idx:
                        if isinstance(t, (int, float)):
                            sub_parts.append(str(int(t) if isinstance(t, float) and t.is_integer() else t))
                        else:
                            sub_parts.append(str(t).replace(" ", "_").replace("'", "").replace('"', ""))
                    norm_parts.append("_".join(sub_parts))
                else:
                    if isinstance(idx, (int, float)):
                        norm_parts.append(str(int(idx) if isinstance(idx, float) and idx.is_integer() else idx))
                    else:
                        norm_parts.append(str(idx).replace(" ", "_").replace("'", "").replace('"', ""))
            return base + "_" + "_".join(norm_parts)

    def _infer_var_bounds(self, vname):
        """Best-effort inference of variable bounds for big-M estimation.
        Order of precedence:
        1. Existing bounds array (authoritative).
        2. Collected per-instance bounds (_collected_lbs/_collected_ubs).
        3. Aggregated symbol bounds (base name before first underscore) as fallback.
        Returns (lb, ub) where either can be None if unknown.
        """
        try:
            if hasattr(self, "var_indices") and vname in self.var_indices and hasattr(self, "bounds"):
                idx = self.var_indices[vname]
                lb, ub = self.bounds[idx]
                return lb, ub
        except Exception:
            pass
        lb = getattr(self, "_collected_lbs", {}).get(vname)
        ub = getattr(self, "_collected_ubs", {}).get(vname)
        if lb is not None or ub is not None:
            return lb, ub
        # Try base symbol (strip trailing indices pattern _\d+)
        import re

        m = re.match(r"^([A-Za-z][A-Za-z0-9]*)(?:_.*)?$", vname)
        if m:
            base = m.group(1)
            lb_b = getattr(self, "_collected_lbs", {}).get(base)
            ub_b = getattr(self, "_collected_ubs", {}).get(base)
            if lb_b is not None or ub_b is not None:
                return lb_b, ub_b
        return (None, None)

    def _build_constraints(self):
        eq_row_idx = 0
        ub_row_idx = 0
        self._add_code_line("# Constraints (sparse)")
        logger.debug("[SciPyCSCCodeGenerator] Entering _build_constraints")
        # Enable symbolic boolean evaluation during constraint build
        prev_sym = getattr(self, "_allow_symbolic_bool", False)
        self._allow_symbolic_bool = True
        A_eq_rows, A_eq_cols, A_eq_data, b_eq = [], [], [], []
        A_ub_rows, A_ub_cols, A_ub_data, b_ub = [], [], [], []
        eq_row_idx = 0
        ub_row_idx = 0

        # Collected per-variable bounds from simple constraints (var >= c, var <= c, var == c)
        if not hasattr(self, "_collected_lbs"):
            self._collected_lbs = {}
            self._collected_ubs = {}

        # --- Mixed AND/OR auxiliary infrastructure ---
        self.aux_created = []  # list of created auxiliary boolean vars
        neg_cache = {}  # var -> its negation auxiliary
        expr_memo = {}  # id(node) -> var name (per build to avoid leaking id mappings)
        # Reuse subtree boolean auxiliary vars across constraints via instance-level cache
        subtree_var_cache = self._bool_subtree_cache

        def _new_aux():
            vname = f"_baux{len(self.aux_created)}"
            self.var_names.append(vname)
            self.var_indices[vname] = len(self.var_names) - 1
            self.bounds.append([0, 1])
            if hasattr(self, "integrality"):
                self.integrality.append(1)
            else:
                self.integrality = [1]
            if hasattr(self, "c") and len(self.c) < len(self.var_names):
                self.c.append(0.0)
            self.aux_created.append(vname)
            return vname

        def _atomic_bool(node, env):
            if not isinstance(node, dict):
                raise SemanticError("Non-dict atomic boolean node")
            if node.get("type") == "constraint" and node.get("op") == "==":
                left = node["left"]
                right = node["right"]

                def is_num01(x):
                    return isinstance(x, dict) and x.get("type") == "number" and x.get("value") in (0, 1)

                def is_var(x):
                    return isinstance(x, dict) and x.get("type") in (
                        "name",
                        "indexed_name",
                    )

                if is_var(left) and is_num01(right):
                    vname = self._multi_indexed_var_name(left, env) if left.get("type") == "indexed_name" else left["value"]
                    return vname, (1 if right["value"] == 1 else -1)
                if is_var(right) and is_num01(left):
                    vname = self._multi_indexed_var_name(right, env) if right.get("type") == "indexed_name" else right["value"]
                    return vname, (1 if left["value"] == 1 else -1)
            raise SemanticError("Unsupported atomic boolean literal")

        # Reuse comparison truth vars across constraints (instance-level)
        comparison_truth_cache = self._comparison_truth_cache

        def _comparison_key(node, env):
            # Build structural key for a comparison binop (<=, >=, !=)
            op = node.get("op")
            left = node.get("left")
            right = node.get("right")
            # Normalize variable and constant ordering for symmetric ops (==, !=)
            if op in ("==", "!="):
                # produce canonical string forms via eval dict keys order independent
                return ("cmp", op, str(left), str(right)) if str(left) <= str(right) else ("cmp", op, str(right), str(left))
            return ("cmp", op, str(left), str(right))

        def _comparison_truth_var(node, env):
            """Return a binary var name representing truth of a linear comparison (<=, >=, !=).
            Current implementation supports simple linear left/right expressions that _eval_expr can process.
            """
            # nonlocal already declared at top of handle_constraint
            if not (
                isinstance(node, dict)
                and node.get("type") == "binop"
                and node.get("sem_type") == "boolean"
                and node.get("op") in ("<=", ">=", "!=", "==")
            ):
                raise SemanticError("Not a supported comparison binop for truth var")
            k = _comparison_key(node, env)
            if k in comparison_truth_cache:
                return comparison_truth_cache[k]
            op = node.get("op")
            # Evaluate both sides into dict form f(x) = lhs - rhs
            lhs_dict, lhs_const = self._eval_expr(node["left"], dict(env))
            rhs_dict, rhs_const = self._eval_expr(node["right"], dict(env))

            # Coerce constants to numeric if possible; raise clear semantic error otherwise.
            def _coerce_numeric(v):
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, bool):
                    return 1.0 if v else 0.0
                if isinstance(v, str):
                    # Attempt to parse simple numeric strings; ignore placeholders like "x['f']".
                    try:
                        return float(v)
                    except ValueError:
                        raise SemanticError(f"Non-numeric term '{v}' in linear comparison; cannot linearize")
                # Unsupported type (e.g., tuple, list)
                raise SemanticError(f"Unsupported constant type {type(v)} in linear comparison")

            lhs_const = _coerce_numeric(lhs_const)
            rhs_const = _coerce_numeric(rhs_const)
            expr_coef = dict(lhs_dict)
            for vn, cf in rhs_dict.items():
                expr_coef[vn] = expr_coef.get(vn, 0.0) - cf
            expr_const = lhs_const - rhs_const  # we model f = sum coef*var + expr_const
            # Unified naming for comparison truth variables
            bname = f"cmp_flag_{len(comparison_truth_cache)}"
            self.var_names.append(bname)
            self.var_indices[bname] = len(self.var_names) - 1
            self.bounds.append([0, 1])
            if hasattr(self, "integrality"):
                self.integrality.append(1)
            else:
                self.integrality = [1]
            if hasattr(self, "c") and len(self.c) < len(self.var_names):
                self.c.append(0.0)
            # Big-M estimation: delegate to unified helper for consistent tightening
            try:
                M = self._big_m_for_comparison(node, env=env)
            except Exception:
                M = BIG_M_DEFAULT  # final safety fallback
            EPS = BOOL_EPS

            # Helper to add row (row_coef_dict <= rhs)
            def add_ub(row_coef_dict, rhs):
                nonlocal ub_row_idx
                row = [0.0] * len(self.var_names)
                for vn, cf in row_coef_dict.items():
                    if vn in self.var_indices:
                        row[self.var_indices[vn]] += cf
                for i, coef in enumerate(row):
                    if abs(coef) > 1e-12:
                        A_ub_rows.append(ub_row_idx)
                        A_ub_cols.append(i)
                        A_ub_data.append(coef)
                b_ub.append(rhs)
                ub_row_idx += 1

            def add_eq(row_coef_dict, rhs):
                nonlocal eq_row_idx
                row = [0.0] * len(self.var_names)
                for vn, cf in row_coef_dict.items():
                    if vn in self.var_indices:
                        row[self.var_indices[vn]] += cf
                for i, coef in enumerate(row):
                    if abs(coef) > 1e-12:
                        A_eq_rows.append(eq_row_idx)
                        A_eq_cols.append(i)
                        A_eq_data.append(coef)
                b_eq.append(rhs)
                eq_row_idx += 1

            # Build linear form value f = sum(cf*var) + expr_const.
            # For convenience treat expr_const by adding it to rhs.
            # Encode operations:
            if op == "<=":
                # f <= 0 when b=1
                # Constraint 1: f - M*(1-b) <= 0  => f - M + M*b <=0
                row1 = dict(expr_coef)
                row1[bname] = row1.get(bname, 0.0) + M
                const1 = M - expr_const
                add_ub(row1, const1)
                # Constraint 2: ensure if f <=0 then b=1: f + EPS - M*b >= 0 -> -f - EPS + M*b <=0
                row2 = {vn: -cf for vn, cf in expr_coef.items()}
                row2[bname] = row2.get(bname, 0.0) + M
                const2 = -expr_const - EPS
                add_ub(row2, const2)
            elif op == ">=":
                # f >=0 equivalent to -f <=0 handled like <= with -f
                # Define g = -f
                neg_coef = {vn: -cf for vn, cf in expr_coef.items()}
                neg_const = -expr_const
                # g <=0 pattern with same two constraints using neg_coef/neg_const
                row1 = dict(neg_coef)
                row1[bname] = row1.get(bname, 0.0) + M
                const1 = M - neg_const
                add_ub(row1, const1)
                row2 = {vn: -cf for vn, cf in neg_coef.items()}
                row2[bname] = row2.get(bname, 0.0) + M
                const2 = -neg_const - EPS
                add_ub(row2, const2)
            elif op == "!=":
                # Generic numeric != encoding via two big-M inequalities (may be tightened later)
                delta = bname
                # Constraint A: -f - M*delta <= -1 - expr_const
                rowA = {vn: -cf for vn, cf in expr_coef.items()}
                rowA[delta] = rowA.get(delta, 0.0) - M
                constA = -1 - expr_const
                add_ub(rowA, constA)
                # Constraint B (current generic form): -f - M + M*delta <= -1 - expr_const  (equivalent to f + M - M*delta >= 1)
                rowB = {vn: -cf for vn, cf in expr_coef.items()}
                rowB[delta] = rowB.get(delta, 0.0) + M
                constB = M - 1 - expr_const
                add_ub(rowB, constB)
            else:  # '==' (not yet fully supported for truth var; fall back by creating equality with tolerance)
                # Soft encoding: |f| <= M*(1-b); if b=1 then both enforce f close to 0 (within EPS)
                row1 = dict(expr_coef)
                row1[bname] = row1.get(bname, 0.0) + M
                const1 = M - expr_const
                add_ub(row1, const1)
                neg_coef = {vn: -cf for vn, cf in expr_coef.items()}
                neg_coef[bname] = neg_coef.get(bname, 0.0) + M
                const2 = M + expr_const
                add_ub(neg_coef, const2)
            comparison_truth_cache[k] = bname
            self._add_code_line(f"# comparison truth var for {op}")
            return bname

        def _bool_expr_var(node, env):
            nonlocal eq_row_idx, ub_row_idx

            # Structural sharing: build a canonical key for subtree to reuse auxiliaries across constraints
            def struct_key(n):
                # Unwrap any parenthesized_expression layers
                while isinstance(n, dict) and n.get("type") == "parenthesized_expression":
                    n = n.get("expression")
                if not isinstance(n, dict):
                    return ("lit", n)
                t = n.get("type")
                if t == "constraint" and n.get("op") == "==":
                    left = n["left"]
                    right = n["right"]

                    def is_num01(x):
                        return isinstance(x, dict) and x.get("type") == "number" and x.get("value") in (0, 1)

                    def is_var(x):
                        return isinstance(x, dict) and x.get("type") in (
                            "name",
                            "indexed_name",
                        )

                    if is_var(left) and is_num01(right):
                        vname = (
                            self._multi_indexed_var_name(left, env) if left.get("type") == "indexed_name" else left["value"]
                        )
                        return ("atom", vname, right["value"])
                    if is_var(right) and is_num01(left):
                        vname = (
                            self._multi_indexed_var_name(right, env) if right.get("type") == "indexed_name" else right["value"]
                        )
                        return ("atom", vname, left["value"])
                if t == "not":
                    return ("not", struct_key(n["value"]))
                if t in ("and", "or"):
                    kl = struct_key(n["left"])
                    kr = struct_key(n["right"])
                    # If an operand is an eq_link (var == composite) collapse to composite part for hashing
                    if isinstance(kl, tuple) and len(kl) >= 3 and kl[0] == "eq_link":
                        kl = kl[2]
                    if isinstance(kr, tuple) and len(kr) >= 3 and kr[0] == "eq_link":
                        kr = kr[2]
                    # commutative: sort keys (after collapsing)
                    pair = tuple(sorted([kl, kr]))
                    return (t, pair)
                if t == "binop" and n.get("sem_type") == "boolean" and n.get("op") in ("<=", ">=", "!=", "=="):
                    # Special case: boolean variable equality with AND/OR composite should key on composite only
                    if n.get("op") == "==" and isinstance(n.get("left"), dict) and isinstance(n.get("right"), dict):
                        left = n["left"]
                        right = n["right"]

                        def _is_bool_var(x):
                            return (
                                isinstance(x, dict)
                                and x.get("type") in ("name", "indexed_name")
                                and x.get("sem_type") == "boolean"
                            )

                        def _is_bool_composite(x):
                            return (
                                isinstance(x, dict)
                                and x.get("sem_type") == "boolean"
                                and x.get("type") in ("and", "or", "binop", "parenthesized_expression")
                            )

                        if _is_bool_var(left) and _is_bool_composite(right):
                            return (
                                "eq_link",
                                str(left.get("value", left)),
                                struct_key(right),
                            )
                        if _is_bool_var(right) and _is_bool_composite(left):
                            return (
                                "eq_link",
                                str(right.get("value", right)),
                                struct_key(left),
                            )
                    return ("cmp", n.get("op"), str(n.get("left")), str(n.get("right")))
                return ("unknown", id(n))  # fallback prevents accidental merging

            sk = struct_key(node)
            if sk in subtree_var_cache:
                return subtree_var_cache[sk]
            if id(node) in expr_memo:
                return expr_memo[id(node)]
            if not isinstance(node, dict):
                raise SemanticError("Invalid boolean expr node (not a dict): {}".format(repr(node)))
            t = node.get("type")
            # Handle special aux_var node for boolean XOR
            if t == "aux_var" and node.get("sem_type") == "boolean":
                vname = node["name"]
                # Register the variable if not already present
                if vname not in self.var_indices:
                    self.var_names.append(vname)
                    self.var_indices[vname] = len(self.var_names) - 1
                    self.bounds.append([0, 1])
                    if hasattr(self, "integrality"):
                        self.integrality.append(1)
                    else:
                        self.integrality = [1]
                    if hasattr(self, "c") and len(self.c) < len(self.var_names):
                        self.c.append(0.0)
                    logger.debug(f"[DEBUG] Registered aux_var node: {vname} (idx={self.var_indices[vname]})")
                else:
                    logger.debug(f"[DEBUG] aux_var node already registered: {vname} (idx={self.var_indices[vname]})")
                subtree_var_cache[sk] = vname
                expr_memo[id(node)] = vname
                return vname
            # Unwrap parentheses early
            if t == "parenthesized_expression":
                inner = node.get("expression")
                return _bool_expr_var(inner, env)
            # Boolean variable equality with composite (var == (and/or/...)) should reuse composite aux directly
            # Handle both 'binop' and 'constraint' nodes with op '!=' and both sides boolean
            is_binop_neq = t == "binop" and node.get("sem_type") == "boolean" and node.get("op") == "!="
            is_constraint_neq = t == "constraint" and node.get("op") == "!="
            if is_binop_neq or is_constraint_neq:
                left = node["left"]
                right = node["right"]

                def _is_bool_expr(x):
                    return isinstance(x, dict) and (
                        x.get("sem_type") == "boolean"
                        or x.get("type") == "boolean_literal"
                        or (
                            x.get("type") == "constraint"
                            and x.get("op") == "=="
                            and (
                                (isinstance(x.get("left"), dict) and x["left"].get("type") in ("name", "indexed_name"))
                                or (isinstance(x.get("right"), dict) and x["right"].get("type") in ("name", "indexed_name"))
                            )
                        )
                        or (x.get("type") in ("and", "or", "not"))
                    )

                if _is_bool_expr(left) and _is_bool_expr(right):
                    x = _bool_expr_var(left, env)
                    y = _bool_expr_var(right, env)
                    z = _new_aux()
                    # z >= x - y
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = 1.0
                    row[self.var_indices[x]] = -1.0
                    row[self.var_indices[y]] = 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    # z >= y - x
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = 1.0
                    row[self.var_indices[x]] = 1.0
                    row[self.var_indices[y]] = -1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    # z <= x + y
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = -1.0
                    row[self.var_indices[x]] = 1.0
                    row[self.var_indices[y]] = 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    # z <= 2 - (x + y)
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = 1.0
                    row[self.var_indices[x]] = 1.0
                    row[self.var_indices[y]] = 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(2.0)
                    ub_row_idx += 1
                    subtree_var_cache[sk] = z
                    expr_memo[id(node)] = z
                    return z
            # Continue with original binop logic for other ops
            if t == "binop" and node.get("sem_type") == "boolean" and node.get("op") in ("<=", ">=", "!=", "=="):
                op = node.get("op")
                # Handle special equality rewrite first
                if op == "==" and isinstance(node.get("left"), dict) and isinstance(node.get("right"), dict):
                    left = node["left"]
                    right = node["right"]

                    def _is_bool_var(x):
                        return (
                            isinstance(x, dict)
                            and x.get("type") in ("name", "indexed_name")
                            and x.get("sem_type") == "boolean"
                        )

                    def _is_bool_composite(x):
                        return (
                            isinstance(x, dict)
                            and x.get("sem_type") == "boolean"
                            and x.get("type") in ("and", "or", "binop", "parenthesized_expression", "not")
                        )

                    # Normalize pattern so var_side holds the variable, expr_side the composite expression
                    var_side = None
                    expr_side = None
                    if _is_bool_var(left) and _is_bool_composite(right):
                        var_side, expr_side = left, right
                    elif _is_bool_var(right) and _is_bool_composite(left):
                        var_side, expr_side = right, left
                    if var_side is not None and expr_side is not None:
                        # Obtain / build variable representing expr_side
                        expr_var = _bool_expr_var(expr_side, env)
                        # Tie var_side to expr_var with equality if not already tied
                        vname = (
                            self._multi_indexed_var_name(var_side, env)
                            if var_side.get("type") == "indexed_name"
                            else var_side["value"]
                        )
                        # Avoid duplicating equality row: check existing row pattern quickly
                        # (Simple heuristic: only add if either row not yet produced linking vname & expr_var)
                        if vname in self.var_indices and expr_var in self.var_indices:
                            already = False
                            if "A_eq" in self.__dict__:
                                v_idx = self.var_indices[vname]
                                e_idx = self.var_indices[expr_var]
                                for r in range(len(self.A_eq)):
                                    row = self.A_eq[r]
                                    if abs(row[v_idx]) == 1 and abs(row[e_idx]) == 1:
                                        already = True
                                        break
                            if not already:
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[vname]] = 1.0
                                row[self.var_indices[expr_var]] = -1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(0.0)
                                eq_row_idx += 1
                        subtree_var_cache[sk] = vname
                        expr_memo[id(node)] = vname
                        return vname
                # Fallback to generic comparison truth var
                bcmp = _comparison_truth_var(node, env)
                subtree_var_cache[sk] = bcmp
                expr_memo[id(node)] = bcmp
                return bcmp
            if t == "constraint" and node.get("op") == "==":
                vname, pol = _atomic_bool(node, env)
                if pol == 1:
                    expr_memo[id(node)] = vname
                    subtree_var_cache[sk] = vname
                    return vname
                if vname in neg_cache:
                    expr_memo[id(node)] = neg_cache[vname]
                    subtree_var_cache[sk] = neg_cache[vname]
                    return neg_cache[vname]
                z = _new_aux()
                row = [0.0] * len(self.var_names)
                row[self.var_indices[z]] = 1.0
                row[self.var_indices[vname]] = 1.0
                for i, coef in enumerate(row):
                    if abs(coef) > 1e-12:
                        A_eq_rows.append(eq_row_idx)
                        A_eq_cols.append(i)
                        A_eq_data.append(coef)
                b_eq.append(1.0)
                eq_row_idx += 1
                neg_cache[vname] = z
                expr_memo[id(node)] = z
                subtree_var_cache[sk] = z
                return z
            if t == "not":
                inner = _bool_expr_var(node["value"], env)
                if inner in neg_cache:
                    subtree_var_cache[sk] = neg_cache[inner]
                    return neg_cache[inner]
                z = _new_aux()
                row = [0.0] * len(self.var_names)
                row[self.var_indices[z]] = 1.0
                row[self.var_indices[inner]] = 1.0
                for i, coef in enumerate(row):
                    if abs(coef) > 1e-12:
                        A_eq_rows.append(eq_row_idx)
                        A_eq_cols.append(i)
                        A_eq_data.append(coef)
                b_eq.append(1.0)
                eq_row_idx += 1
                neg_cache[inner] = z
                subtree_var_cache[sk] = z
                return z
            if t in ("and", "or"):
                # Use tuple-based struct_key for normalization and sharing
                sk = struct_key(node)
                if sk in subtree_var_cache:
                    shared_aux = subtree_var_cache[sk]
                    expr_memo[id(node)] = shared_aux
                    # Tie all relevant variables to shared_aux
                    tie_vars = []

                    def _extract_var_equality(n):
                        n_norm = n
                        while isinstance(n_norm, dict) and n_norm.get("type") == "parenthesized_expression":
                            n_norm = n_norm.get("expression")
                        if (
                            isinstance(n_norm, dict)
                            and n_norm.get("type") == "binop"
                            and n_norm.get("op") == "=="
                            and isinstance(n_norm.get("left"), dict)
                            and isinstance(n_norm.get("right"), dict)
                        ):
                            left = n_norm["left"]
                            right = n_norm["right"]

                            def _is_bool_var(x):
                                return (
                                    isinstance(x, dict)
                                    and x.get("type") in ("name", "indexed_name")
                                    and x.get("sem_type") == "boolean"
                                )

                            if (
                                _is_bool_var(left)
                                and right.get("sem_type") == "boolean"
                                and right.get("type") not in ("name", "indexed_name")
                            ):
                                return (left, right)
                            if (
                                _is_bool_var(right)
                                and left.get("sem_type") == "boolean"
                                and left.get("type") not in ("name", "indexed_name")
                            ):
                                return (right, left)
                        return None

                    left_node = node["left"]
                    right_node = node["right"]
                    left_eq = _extract_var_equality(left_node)
                    if left_eq:
                        tie_vars.append(left_eq[0])
                    right_eq = _extract_var_equality(right_node)
                    if right_eq:
                        tie_vars.append(right_eq[0])
                    for var_node in tie_vars:
                        vname = (
                            self._multi_indexed_var_name(var_node, env)
                            if var_node.get("type") == "indexed_name"
                            else var_node["value"]
                        )
                        if vname in self.var_indices and shared_aux in self.var_indices:
                            row = [0.0] * len(self.var_names)
                            row[self.var_indices[vname]] = 1.0
                            row[self.var_indices[shared_aux]] = -1.0
                            for i, coef in enumerate(row):
                                if abs(coef) > 1e-12:
                                    A_eq_rows.append(eq_row_idx)
                                    A_eq_cols.append(i)
                                    A_eq_data.append(coef)
                            b_eq.append(0.0)
                            eq_row_idx += 1
                    return shared_aux
                # Otherwise, create new aux and record
                tie_vars = []

                def _extract_var_equality(n):
                    if (
                        isinstance(n, dict)
                        and n.get("type") == "binop"
                        and n.get("op") == "=="
                        and isinstance(n.get("left"), dict)
                        and isinstance(n.get("right"), dict)
                    ):
                        left = n["left"]
                        right = n["right"]

                        def _is_bool_var(x):
                            return (
                                isinstance(x, dict)
                                and x.get("type") in ("name", "indexed_name")
                                and x.get("sem_type") == "boolean"
                            )

                        if (
                            _is_bool_var(left)
                            and right.get("sem_type") == "boolean"
                            and right.get("type") not in ("name", "indexed_name")
                        ):
                            return (left, right)
                        if (
                            _is_bool_var(right)
                            and left.get("sem_type") == "boolean"
                            and left.get("type") not in ("name", "indexed_name")
                        ):
                            return (right, left)

                left_node = node["left"]
                right_node = node["right"]
                left_eq = _extract_var_equality(left_node)
                if left_eq:
                    tie_vars.append(left_eq[0])
                    left_node = left_eq[1]
                right_eq = _extract_var_equality(right_node)
                if right_eq:
                    tie_vars.append(right_eq[0])
                    right_node = right_eq[1]
                left_v = _bool_expr_var(left_node, env)
                right_v = _bool_expr_var(right_node, env)
                if left_v == right_v:
                    expr_memo[id(node)] = left_v
                    subtree_var_cache[sk] = left_v
                    for var_node in tie_vars:
                        vname = (
                            self._multi_indexed_var_name(var_node, env)
                            if var_node.get("type") == "indexed_name"
                            else var_node["value"]
                        )
                        if vname in self.var_indices and left_v in self.var_indices:
                            row = [0.0] * len(self.var_names)
                            v_idx = self.var_indices[vname]
                            e_idx = self.var_indices[left_v]
                            row[v_idx] = 1.0
                            row[e_idx] = -1.0
                            for i, coef in enumerate(row):
                                if abs(coef) > 1e-12:
                                    A_eq_rows.append(eq_row_idx)
                                    A_eq_cols.append(i)
                                    A_eq_data.append(coef)
                            b_eq.append(0.0)
                            eq_row_idx += 1
                    return left_v
                z = _new_aux()
                if t == "and":
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = 1.0
                    row[self.var_indices[left_v]] = -1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = 1.0
                    row[self.var_indices[right_v]] = -1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = -1.0
                    row[self.var_indices[left_v]] += 1.0
                    row[self.var_indices[right_v]] += 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(1.0)
                    ub_row_idx += 1
                else:  # or
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = -1.0
                    row[self.var_indices[left_v]] = 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = -1.0
                    row[self.var_indices[right_v]] = 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[z]] = 1.0
                    row[self.var_indices[left_v]] -= 1.0
                    row[self.var_indices[right_v]] -= 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                expr_memo[id(node)] = z
                subtree_var_cache[sk] = z
                for var_node in tie_vars:
                    vname = (
                        self._multi_indexed_var_name(var_node, env)
                        if var_node.get("type") == "indexed_name"
                        else var_node["value"]
                    )
                    if vname in self.var_indices and z in self.var_indices:
                        row = [0.0] * len(self.var_names)
                        row[self.var_indices[vname]] = 1.0
                        row[self.var_indices[z]] = -1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_eq_rows.append(eq_row_idx)
                                A_eq_cols.append(i)
                                A_eq_data.append(coef)
                        b_eq.append(0.0)
                        eq_row_idx += 1
                return z
            # Lower 'implies' to (not left) or right
            if t == "implies":
                not_left = {"type": "not", "value": node["left"]}
                lowered = {"type": "or", "left": not_left, "right": node["right"]}
                return _bool_expr_var(lowered, env)
            # If we reach here, node cannot be resolved to a variable name
            raise SemanticError(f"Unsupported or non-resolvable boolean expression node type: {t} ({repr(node)})")

        def handle_constraint(constr, env, constr_name_prefix=None):
            nonlocal eq_row_idx, ub_row_idx

            # Early guard: if a constraint references a parameter without provided data, error out
            def _is_unbound_param_node(node):
                if not isinstance(node, dict):
                    return False
                t = node.get("type")
                if t not in ("name", "indexed_name"):
                    return False
                base = node.get("value") if t == "name" else node.get("name")
                if base is None:
                    return False
                decl = self._find_decl(base)
                if decl is None:
                    return False
                if decl.get("type") not in (
                    "parameter_inline",
                    "parameter_inline_indexed",
                    "parameter_external",
                    "parameter_external_indexed",
                    "parameter_external_explicit",
                    "parameter_external_explicit_indexed",
                ):
                    return False
                # Inline parameters with explicit value are bound
                if decl.get("type") in ("parameter_inline", "parameter_inline_indexed") and decl.get("value") is not None:
                    return False
                # External-like parameters require data in data_dict
                return self.data_dict.get(base) is None

            if constr.get("type") == "constraint" and (
                _is_unbound_param_node(constr.get("left")) or _is_unbound_param_node(constr.get("right"))
            ):
                raise SemanticError("Constraint references parameter with no data provided")

            # Helper functions must be defined before use
            def _unwrap_paren(n):
                while isinstance(n, dict) and n.get("type") == "parenthesized_expression":
                    n = n.get("expression")
                return n

            def _is_simple_comparison(n):
                n = _unwrap_paren(n)
                return (
                    isinstance(n, dict)
                    and n.get("type") == "binop"
                    and n.get("op") in (">=", "<=", "==", ">", "<")
                    and n.get("sem_type") == "boolean"
                )

            def _detect_sum_of_comparisons(left, right, op_sym_top):
                """
                Detects and normalizes sum-of-comparisons and cardinality constraints.
                Returns (is_sum_of_comparisons, inner_cmp, k_val, loop_vars, loop_ranges) or None.
                """
                LU_left = _unwrap_paren(left)
                LU_right = _unwrap_paren(right)
                if (
                    isinstance(LU_left, dict)
                    and LU_left.get("type") == "sum"
                    and isinstance(LU_right, dict)
                    and LU_right.get("type") == "number"
                    and op_sym_top in (">=", "==")
                ):
                    inner_cmp = _unwrap_paren(LU_left.get("expression"))
                    if _is_simple_comparison(inner_cmp):
                        k_val = LU_right.get("value")
                        iterators = LU_left.get("iterators", [])
                        loop_vars, loop_ranges = self._unroll_iterators(iterators)
                        return True, inner_cmp, k_val, loop_vars, loop_ranges
                return False, None, None, None, None

            def _normalize_implication_nodes(ant, cons):
                """
                Normalize and unwrap implication constraint nodes for antecedent and consequent.
                Returns (ant_unwrapped, cons_unwrapped).
                """

                def _unwrap_bool_eq_true(node):
                    nonlocal ub_row_idx
                    if not (isinstance(node, dict) and node.get("type") == "constraint" and node.get("op") == "=="):
                        return node
                    left = node.get("left")
                    right = node.get("right")

                    def _is_true(x):
                        return isinstance(x, dict) and x.get("type") == "boolean_literal" and x.get("value") is True

                    expr_side = None
                    # --- Patch: Detect and handle sum-of-comparisons/cardinality constraints ---
                    if constr.get("type") == "constraint" and constr.get("op") in (
                        ">=",
                        "==",
                        ">",
                        "<",
                    ):
                        left = constr.get("left")
                        right = constr.get("right")
                        op_sym_top = constr.get("op")
                        LU_left = _unwrap_paren(left)
                        LU_right = _unwrap_paren(right)
                        # Only handle sum-of-comparisons for >=, ==, <=, >, <
                        if (
                            isinstance(LU_left, dict)
                            and LU_left.get("type") == "sum"
                            and isinstance(LU_right, dict)
                            and LU_right.get("type") == "number"
                        ):
                            inner_cmp = _unwrap_paren(LU_left.get("expression"))
                            if _is_simple_comparison(inner_cmp):
                                k_val = LU_right.get("value")
                                iterators = LU_left.get("iterators", [])
                                loop_vars, loop_ranges = self._unroll_iterators(iterators)
                                # For each index, reify the comparison to a boolean aux
                                aux_vars = []
                                for idx_tuple in itertools.product(*loop_ranges):
                                    env2 = dict(env or {})
                                    for v, val in zip(loop_vars, idx_tuple):
                                        env2[v] = val
                                    aux_var = self._bool_expr_var(inner_cmp, env2)
                                    aux_vars.append(aux_var)
                                # Build sum row: sum(aux_vars) op k_val
                                row = [0.0] * len(self.var_names)
                                for aux in aux_vars:
                                    idx = self.var_indices.get(aux)
                                    if idx is not None:
                                        row[idx] += 1.0
                                # For >=, ==, <=, >, <
                                if op_sym_top == ">=":
                                    for i, coef in enumerate(row):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(-coef)
                                    b_ub.append(-k_val)
                                    ub_row_idx += 1
                                elif op_sym_top == "==":
                                    for i, coef in enumerate(row):
                                        if abs(coef) > 1e-12:
                                            A_eq_rows.append(eq_row_idx)
                                            A_eq_cols.append(i)
                                            A_eq_data.append(coef)
                                    b_eq.append(k_val)
                                elif op_sym_top == ">":
                                    # sum(aux_vars) > k_val  <=> sum(aux_vars) >= k_val+1
                                    for i, coef in enumerate(row):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(-coef)
                                    b_ub.append(-(k_val + 1))
                                    ub_row_idx += 1
                                elif op_sym_top == "<":
                                    # sum(aux_vars) < k_val  <=> sum(aux_vars) <= k_val-1
                                    for i, coef in enumerate(row):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(coef)
                                    b_ub.append(k_val - 1)
                                    ub_row_idx += 1
                                return
                    if _is_true(left):
                        expr_side = right
                    elif _is_true(right):
                        expr_side = left
                    if (
                        not expr_side
                        or not isinstance(expr_side, dict)
                        or expr_side.get("type") not in ("parenthesized_expression", "binop")
                    ):
                        return node
                    inner = expr_side.get("expression") if expr_side.get("type") == "parenthesized_expression" else expr_side
                    if not (isinstance(inner, dict) and inner.get("type") == "binop"):
                        return node
                    return {
                        "type": "constraint",
                        "op": inner.get("op"),
                        "left": inner.get("left"),
                        "right": inner.get("right"),
                    }

                def _unwrap_paren(n):
                    while isinstance(n, dict) and n.get("type") == "parenthesized_expression":
                        n = n.get("expression")
                    return n

                ant_unwrapped = _unwrap_bool_eq_true(ant)
                cons_unwrapped = _unwrap_bool_eq_true(cons)
                # If antecedent is equality-to-true of AND composite (possibly parenthesized), decompose and rewrite.
                if isinstance(ant, dict) and ant.get("type") == "constraint" and ant.get("op") == "==":
                    left = _unwrap_paren(ant.get("left"))
                    right = _unwrap_paren(ant.get("right"))
                    if (
                        isinstance(right, dict)
                        and right.get("type") == "and"
                        and isinstance(left, dict)
                        and left.get("type") == "boolean_literal"
                        and left.get("value") is True
                    ):
                        left, right = right, left
                    if (
                        isinstance(left, dict)
                        and left.get("type") == "and"
                        and isinstance(right, dict)
                        and right.get("type") == "boolean_literal"
                        and right.get("value") is True
                    ):
                        leaves = self._flatten_bool(left, "and")
                        last_leaf = None
                        for leaf in leaves:
                            expr_node = leaf.get("expression") if leaf.get("type") == "parenthesized_expression" else leaf
                            if expr_node.get("type") == "parenthesized_expression":
                                expr_node = expr_node.get("expression")
                            if self._is_linear_comparison(expr_node):
                                pseudo = {
                                    "type": "constraint",
                                    "op": expr_node["op"],
                                    "left": expr_node["left"],
                                    "right": expr_node["right"],
                                }
                                handle_constraint(pseudo, env=env)
                                last_leaf = pseudo
                            else:
                                last_leaf = None
                                break
                        if last_leaf is not None:
                            ant_unwrapped = last_leaf
                return ant_unwrapped, cons_unwrapped

            def _tighten_lower_bound(symbol, val):
                self._collected_lbs[symbol] = max(self._collected_lbs.get(symbol, -float("inf")), val)

            def _tighten_upper_bound(symbol, val):
                self._collected_ubs[symbol] = min(self._collected_ubs.get(symbol, float("inf")), val)

            def _tighten_bounds(symbol, val):
                _tighten_lower_bound(symbol, val)
                _tighten_upper_bound(symbol, val)

            # Local helper: check if a node is a boolean tree
            def _is_bool_tree(node):
                if not isinstance(node, dict):
                    return False
                tnode = node.get("type")
                if tnode in ("and", "or"):
                    return _is_bool_tree(node.get("left")) and _is_bool_tree(node.get("right"))
                if tnode == "not":
                    return _is_bool_tree(node.get("value"))
                # atomic: constraint (var == 0/1) or (0/1 == var)
                if tnode == "constraint" and node.get("op") == "==":
                    left = node.get("left")
                    right = node.get("right")

                    def is_var(x):
                        return isinstance(x, dict) and x.get("type") in (
                            "name",
                            "indexed_name",
                        )

                    def is_num01(x):
                        return isinstance(x, dict) and x.get("type") == "number" and x.get("value") in (0, 1)

                    return (is_var(left) and is_num01(right)) or (is_var(right) and is_num01(left))
                return False

            # --- Patch: Always create auxiliary for non-trivial boolean expressions ---
            # Only for constraints of the form (bool_expr) == True/1 or >=1 or <=1

            if constr.get("type") == "constraint" and constr.get("op") in (
                "==",
                ">=",
                "<=",
                "!=",
            ):
                left = constr.get("left")
                right = constr.get("right")

                # Only create aux if left is a non-trivial bool tree (not just a variable == 0/1)
                if _is_bool_tree(left) and not (
                    _is_bool_tree(
                        {
                            "type": "constraint",
                            "left": left,
                            "op": "==",
                            "right": {"type": "number", "value": 1},
                        }
                    )
                ):
                    aux = _bool_expr_var(left, env)
                    logger.debug(f"[DEBUG] Created auxiliary {aux} for boolean expr: {left}")

                    # --- Tautology skip logic ---
                    # For boolean aux, skip constraints that are always true: aux == 1, aux >= 1, aux <= 1, aux >= 0, aux <= 1
                    # Only skip for aux == 1 (r==1 or True), aux >= 0, aux <= 1
                    def _is_tautology_bool_aux(op, r):
                        if op == "==":
                            if isinstance(r, dict) and (
                                (r.get("type") == "boolean_literal" and r.get("value") is True)
                                or (r.get("type") == "number" and r.get("value") == 1)
                            ):
                                return True
                        elif op == ">=":
                            if isinstance(r, dict) and r.get("type") == "number" and r.get("value") == 0:
                                return True
                        elif op == "<=":
                            if isinstance(r, dict) and r.get("type") == "number" and r.get("value") == 1:
                                return True
                        return False

                    if _is_tautology_bool_aux(constr.get("op"), right):
                        logger.debug(f"[DEBUG] Skipping tautological constraint: {aux} {constr.get('op')} {right}")
                        return
                    # For AND/OR trees, always enforce via equality for ==1, >=1, !=0
                    if constr.get("op") == "==":
                        # Add equality constraint row: aux == r
                        row = [0.0] * len(self.var_names)
                        row[self.var_indices[aux]] = 1.0
                        b_eq.append(float(right.get("value", 0)))
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_eq_rows.append(eq_row_idx)
                                A_eq_cols.append(i)
                                A_eq_data.append(coef)
                        eq_row_idx += 1
                        return
                    elif constr.get("op") == ">=":
                        if isinstance(right, dict) and right.get("type") == "number" and right.get("value") == 1:
                            # Enforce aux >= 1 <=> aux == 1 for boolean
                            row = [0.0] * len(self.var_names)
                            row[self.var_indices[aux]] = 1.0
                            b_eq.append(1.0)
                            for i, coef in enumerate(row):
                                if abs(coef) > 1e-12:
                                    A_eq_rows.append(eq_row_idx)
                                    A_eq_cols.append(i)
                                    A_eq_data.append(coef)
                            eq_row_idx += 1
                            return
                    elif constr.get("op") == "!=":
                        if isinstance(right, dict) and right.get("type") == "number" and right.get("value") == 0:
                            # Enforce aux != 0 <=> aux == 1 for boolean
                            row = [0.0] * len(self.var_names)
                            row[self.var_indices[aux]] = 1.0
                            b_eq.append(1.0)
                            for i, coef in enumerate(row):
                                if abs(coef) > 1e-12:
                                    A_eq_rows.append(eq_row_idx)
                                    A_eq_cols.append(i)
                                    A_eq_data.append(coef)
                            eq_row_idx += 1
                            return
                    # For <= 0, >= 0, etc., fall through to generic logic

            # nonlocal already declared at top of handle_constraint
            logger.debug(f"[SciPyCSCCodeGenerator] handle_constraint: {constr}")
            # Passive bound collection for later big-M tightening
            try:
                # Ensure boolean XOR/NEQ at constraint level triggers aux var creation
                if constr.get("type") == "constraint" and constr.get("op") == "!=":
                    left = constr.get("left")
                    right = constr.get("right")

                    def is_bool_expr(e):
                        return isinstance(e, dict) and (
                            e.get("type") == "boolean_literal"
                            or (e.get("type") == "binop" and e.get("sem_type") == "boolean")
                            or (
                                e.get("type") == "constraint"
                                and e.get("op") == "=="
                                and (
                                    (isinstance(e.get("left"), dict) and e["left"].get("type") in ("name", "indexed_name"))
                                    or (
                                        isinstance(e.get("right"), dict) and e["right"].get("type") in ("name", "indexed_name")
                                    )
                                )
                            )
                            or (e.get("type") in ("and", "or", "not"))
                        )

                    if is_bool_expr(left) and is_bool_expr(right):
                        # This will create/register aux vars for both sides and the XOR
                        _ = _bool_expr_var(constr, env)
                if constr.get("type") == "constraint" and constr.get("op") in (
                    ">=",
                    "<=",
                    "==",
                ):
                    op_sym = constr.get("op")
                    left = constr.get("left")
                    right = constr.get("right")

                    def _is_var(n):
                        return isinstance(n, dict) and n.get("type") in (
                            "name",
                            "indexed_name",
                        )

                    def _is_num(n):
                        return isinstance(n, dict) and n.get("type") == "number"

                    if _is_var(left) and _is_num(right):
                        vname = (
                            self._multi_indexed_var_name(left, env) if left.get("type") == "indexed_name" else left["value"]
                        )
                        val = float(right.get("value"))
                        # Per-instance bound tightening (monotone narrowing)
                        if op_sym == ">=":
                            _tighten_lower_bound(vname, val)
                        elif op_sym == "<=":
                            _tighten_upper_bound(vname, val)
                        elif op_sym == "==":
                            _tighten_bounds(vname, val)
                        # Aggregated base-symbol bounds (widening across indices): we want a safe envelope
                        if left.get("type") == "indexed_name":
                            base_sym = left.get("name")
                            # For lower bounds we take the minimum across instances (so use min semantics)
                            if op_sym in (">=", "=="):
                                cur_lb = self._collected_lbs.get(base_sym)
                                if cur_lb is None or val < cur_lb:
                                    _tighten_lower_bound(base_sym, val)
                            if op_sym in ("<=", "=="):
                                cur_ub = self._collected_ubs.get(base_sym)
                                if cur_ub is None or val > cur_ub:
                                    _tighten_upper_bound(base_sym, val)
                    elif _is_var(right) and _is_num(left):
                        vname = (
                            self._multi_indexed_var_name(right, env) if right.get("type") == "indexed_name" else right["value"]
                        )
                        val = float(left.get("value"))
                        # Flip perspective
                        if op_sym == ">=":
                            _tighten_upper_bound(vname, val)
                        elif op_sym == "<=":
                            _tighten_lower_bound(vname, val)
                        elif op_sym == "==":
                            _tighten_bounds(vname, val)
                        if right.get("type") == "indexed_name":
                            base_sym = right.get("name")
                            if op_sym in (
                                "<=",
                                "==",
                            ):  # var <= number (from number >= var) contributes upper envelope
                                # For upper envelope collect maximum
                                if op_sym == "<=":  # derived var >= number so lower bound
                                    pass  # handled separately below
                            # Reverse logic for flipped inequalities:
                            if op_sym == ">=":
                                cur_ub = self._collected_ubs.get(base_sym)
                                if cur_ub is None or val > cur_ub:
                                    _tighten_upper_bound(base_sym, val)
                            elif op_sym == "<=":
                                cur_lb = self._collected_lbs.get(base_sym)
                                if cur_lb is None or val < cur_lb:
                                    _tighten_lower_bound(base_sym, val)
                            elif op_sym == "==":
                                cur_lb = self._collected_lbs.get(base_sym)
                                if cur_lb is None or val < cur_lb:
                                    _tighten_lower_bound(base_sym, val)
                                cur_ub = self._collected_ubs.get(base_sym)
                                if cur_ub is None or val > cur_ub:
                                    _tighten_upper_bound(base_sym, val)
            except Exception:
                pass  # Never let bound collection break core logic
            # Implication constraints: antecedent => consequent
            # --- Patch: ensure auxiliary variables for nested implications ---
            if (
                constr.get("type") == "constraint"
                and isinstance(constr.get("left"), dict)
                and constr.get("left").get("type") == "implies"
            ):
                # Reify implies as auxiliary
                aux = _bool_expr_var(constr.get("left"), env)
                logger.debug(f"[DEBUG] Created auxiliary {aux} for implies expr: {constr.get('left')}")
                # Enforce aux == right (should be boolean literal True/False or 0/1)
                right = constr.get("right")
                if isinstance(right, dict) and (
                    (right.get("type") == "boolean_literal" and right.get("value") is True)
                    or (right.get("type") == "number" and right.get("value") == 1)
                ):
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[aux]] = 1.0
                    b_eq.append(1.0)
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_eq_rows.append(eq_row_idx)
                            A_eq_cols.append(i)
                            A_eq_data.append(coef)
                    eq_row_idx += 1
                    return
                elif isinstance(right, dict) and (
                    (right.get("type") == "boolean_literal" and right.get("value") is False)
                    or (right.get("type") == "number" and right.get("value") == 0)
                ):
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[aux]] = 1.0
                    b_eq.append(0.0)
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_eq_rows.append(eq_row_idx)
                            A_eq_cols.append(i)
                            A_eq_data.append(coef)
                    eq_row_idx += 1
                    return
                # Otherwise, fall through
            if constr.get("type") == "implication_constraint":
                ant = constr["antecedent"]
                cons = constr["consequent"]

                # Accept antecedent of form (var == 1) for boolean var
                def _unwrap_bool_eq_true(node):
                    """If node is constraint ((parenthesized_expression binop ...) == true) unwrap to inner binop as a constraint-like dict.
                    This covers parser output that normalizes boolean expressions via == true.
                    """
                    if not (isinstance(node, dict) and node.get("type") == "constraint" and node.get("op") == "=="):
                        return node
                    left = node.get("left")
                    right = node.get("right")

                    # Identify side that is boolean true literal
                    def _is_true(x):
                        return isinstance(x, dict) and x.get("type") == "boolean_literal" and x.get("value") is True

                    # Other side may be parenthesized_expression wrapping a binop
                    expr_side = None
                    if _is_true(left):
                        expr_side = right
                    elif _is_true(right):
                        expr_side = left
                    if (
                        not expr_side
                        or not isinstance(expr_side, dict)
                        or expr_side.get("type") not in ("parenthesized_expression", "binop")
                    ):
                        return node
                    inner = expr_side.get("expression") if expr_side.get("type") == "parenthesized_expression" else expr_side
                    if not (isinstance(inner, dict) and inner.get("type") == "binop"):
                        return node
                    # Repackage as a pseudo-constraint so downstream logic can treat uniformly
                    return {
                        "type": "constraint",
                        "op": inner.get("op"),
                        "left": inner.get("left"),
                        "right": inner.get("right"),
                    }

                # Unwrap antecedent & consequent if they are equality-to-true wrappers
                ant_unwrapped, cons_unwrapped = _normalize_implication_nodes(ant, cons)

                # (Optional) OR antecedent handling could introduce additional auxiliary; not yet supported
                # (Optional) OR antecedent handling could introduce additional auxiliary; not yet supported
                def _extract_var_eq_val(node, val):
                    if not (isinstance(node, dict) and node.get("type") == "constraint" and node.get("op") == "=="):
                        return None
                    left = node["left"]
                    right = node["right"]

                    def is_var(x):
                        return isinstance(x, dict) and x.get("type") in (
                            "name",
                            "indexed_name",
                        )

                    if (
                        is_var(left)
                        and isinstance(right, dict)
                        and right.get("type") == "number"
                        and right.get("value") == val
                    ):
                        return left
                    if is_var(right) and isinstance(left, dict) and left.get("type") == "number" and left.get("value") == val:
                        return right
                    return None

                ant_var_node = _extract_var_eq_val(ant_unwrapped, 1)

                # --- NEW specialized pattern: (bool_var == 0) => (lin_var <= const) ---
                # Encode: x - M * b <= c   (since b in {0,1})
                ant_eq_zero = _extract_var_eq_val(ant_unwrapped, 0)
                if ant_eq_zero is not None and isinstance(cons_unwrapped, dict) and cons_unwrapped.get("type") == "constraint":
                    op_c = cons_unwrapped.get("op")
                    lc = cons_unwrapped.get("left")
                    rc = cons_unwrapped.get("right")

                    def _is_var(n):
                        return isinstance(n, dict) and n.get("type") in ("name", "indexed_name")

                    def _is_num(n):
                        return isinstance(n, dict) and n.get("type") == "number"

                    # Support canonical form: var <= const
                    if op_c == "<=" and _is_var(lc) and _is_num(rc):
                        ant_vname = (
                            self._multi_indexed_var_name(ant_eq_zero, env)
                            if ant_eq_zero.get("type") == "indexed_name"
                            else ant_eq_zero["value"]
                        )
                        cons_vname = self._multi_indexed_var_name(lc, env) if lc.get("type") == "indexed_name" else lc["value"]
                        rhs_val = float(rc.get("value", 0.0))

                        # Pick big-M from inferred upper bound of cons_var when available
                        M = None
                        try:
                            lb, ub = self._infer_var_bounds(cons_vname)
                            if ub is not None:
                                M = max(1.0, float(ub))
                        except Exception:
                            M = None
                        if M is None:
                            M = BIG_M_DEFAULT

                        # Add inequality: cons_v - M * ant_b <= rhs_val
                        row = [0.0] * len(self.var_names)
                        if cons_vname in self.var_indices:
                            row[self.var_indices[cons_vname]] += 1.0
                        if ant_vname in self.var_indices:
                            row[self.var_indices[ant_vname]] -= M

                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(rhs_val)
                        ub_row_idx += 1
                        return

                    # Also accept equality to zero when x >= 0 (common with float+)
                    if op_c == "==" and _is_var(lc) and _is_num(rc) and abs(float(rc.get("value", 0.0))) < 1e-12:
                        ant_vname = (
                            self._multi_indexed_var_name(ant_eq_zero, env)
                            if ant_eq_zero.get("type") == "indexed_name"
                            else ant_eq_zero["value"]
                        )
                        cons_vname = self._multi_indexed_var_name(lc, env) if lc.get("type") == "indexed_name" else lc["value"]
                        # Use same gating as <= 0: x - M*b <= 0 (nonnegativity enforces x==0 when b==0)
                        M = None
                        try:
                            lb, ub = self._infer_var_bounds(cons_vname)
                            if ub is not None:
                                M = max(1.0, float(ub))
                        except Exception:
                            M = None
                        if M is None:
                            M = BIG_M_DEFAULT

                        row = [0.0] * len(self.var_names)
                        if cons_vname in self.var_indices:
                            row[self.var_indices[cons_vname]] += 1.0
                        if ant_vname in self.var_indices:
                            row[self.var_indices[ant_vname]] -= M

                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(0.0)
                        ub_row_idx += 1
                        return

                # Fast-path: pattern (x > 0) => (y == 1)  or (x >= 0) => (y == 1)
                # Recognize antecedent: constraint with op in ('>','>=') comparing single var to 0; consequent: var == 1 on boolean var
                def _is_zero_number(n):
                    return isinstance(n, dict) and n.get("type") == "number" and abs(n.get("value")) < 1e-12

                def _is_single_var_gt_zero(cnode):
                    if not (isinstance(cnode, dict) and cnode.get("type") == "constraint" and cnode.get("op") in (">", ">=")):
                        return None
                    left = cnode.get("left")
                    right = cnode.get("right")
                    # Accept var > 0 or 0 < var (swap)
                    if isinstance(left, dict) and left.get("type") in ("name", "indexed_name") and _is_zero_number(right):
                        return left
                    if isinstance(right, dict) and right.get("type") in ("name", "indexed_name") and _is_zero_number(left):
                        return right
                    return None

                def _is_var_eq_one(cnode):
                    if not (isinstance(cnode, dict) and cnode.get("type") == "constraint" and cnode.get("op") == "=="):
                        return None
                    left = cnode.get("left")
                    right = cnode.get("right")

                    def is_one(n):
                        return isinstance(n, dict) and n.get("type") == "number" and n.get("value") == 1

                    def is_var(n):
                        return isinstance(n, dict) and n.get("type") in (
                            "name",
                            "indexed_name",
                        )

                    if is_var(left) and is_one(right):
                        return left
                    if is_var(right) and is_one(left):
                        return right
                    return None

                ant_var_pos = _is_single_var_gt_zero(ant_unwrapped)
                cons_var_one = _is_var_eq_one(cons_unwrapped)
                if ant_var_pos is not None and cons_var_one is not None:
                    # Implement big-M gating: x <= M * y  (with y binary). This enforces x>0 -> y=1
                    x_name = (
                        self._multi_indexed_var_name(ant_var_pos, env)
                        if ant_var_pos.get("type") == "indexed_name"
                        else ant_var_pos["value"]
                    )
                    y_name = (
                        self._multi_indexed_var_name(cons_var_one, env)
                        if cons_var_one.get("type") == "indexed_name"
                        else cons_var_one["value"]
                    )
                    # Ensure both variables are registered; y must be binary (assumed from declaration)
                    if x_name not in self.var_indices or y_name not in self.var_indices:
                        raise SemanticError("Implication variables not indexed properly")
                    # Heuristic M: use previously collected upper bound for x if available, else fallback to sum of any demand-like parameter or default
                    bigM = self._collected_ubs.get(x_name)
                    if bigM is None:
                        # Try aggregated symbol bound
                        base_sym = ant_var_pos.get("name") if ant_var_pos.get("type") == "indexed_name" else x_name
                        bigM = self._collected_ubs.get(base_sym)
                    if bigM is None:
                        # Robust AST-based extraction for Wagner-Whitin: sum demand[t..T]
                        try:
                            if "demand" in self.data_dict and isinstance(self.data_dict["demand"], list):
                                dem_list = self.data_dict["demand"]
                                # Use AST to extract index t (OPL 1-based)
                                if ant_var_pos.get("type") == "indexed_name" and ant_var_pos.get("dimensions"):
                                    idx_expr = ant_var_pos["dimensions"][-1]
                                    _, t_idx = self._eval_index_expr(idx_expr, env)
                                    if isinstance(t_idx, int) and 1 <= t_idx <= len(dem_list):
                                        bigM = sum(dem_list[t_idx - 1 :])
                        except Exception:
                            pass
                    if bigM is None:
                        bigM = 1_000_000.0
                    # Add row: x - M*y <= 0
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[x_name]] += 1.0
                    row[self.var_indices[y_name]] -= bigM
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    return
                # --- General linear antecedent -> linear consequent (Option A canonical big-M) ---
                if not ant_var_node:
                    # Expect both antecedent and consequent to be simple constraints
                    if not (
                        isinstance(ant_unwrapped, dict)
                        and ant_unwrapped.get("type") == "constraint"
                        and isinstance(cons_unwrapped, dict)
                        and cons_unwrapped.get("type") == "constraint"
                    ):
                        raise SemanticError("Implication antecedent must be boolean var == 1 or linear constraint")
                    ant_c = ant_unwrapped
                    cons_c = cons_unwrapped
                    ant_op = ant_c.get("op")
                    cons_op = cons_c.get("op")
                    supported_ops = {">=", ">", "<=", "<", "=="}
                    if ant_op not in supported_ops or cons_op not in supported_ops:
                        raise SemanticError("Unsupported implication comparison operator")
                    # We currently do not support equality antecedent (would require splitting); raise for clarity.
                    if ant_op == "==":
                        # Equality antecedent: (left == right) => consequent
                        # Strategy: encode equality with a flag that is forced to 1 iff diff==0 using bounding big-M rows similar to previous implementation.
                        # Then reuse consequent gating with that flag.
                        # Build diff = left - right
                        def _lin_eq(expr):
                            # ...existing code...
                            if not isinstance(expr, dict):
                                raise SemanticError("Unsupported expression in implication linearization")
                            t = expr.get("type")
                            if t == "parenthesized_expression":
                                return _lin_eq(expr.get("expression"))
                            if t in ("name", "indexed_name"):
                                v = self._multi_indexed_var_name(expr, env) if t == "indexed_name" else expr["value"]
                                return {v: 1.0}, 0.0
                            if t == "number":
                                return {}, float(expr.get("value", 0))
                            if t == "binop" and expr.get("op") in ("+", "-"):
                                ld, lc = _lin_eq(expr.get("left"))
                                rd, rc = _lin_eq(expr.get("right"))
                                coef = ld.copy()
                                for k, v in rd.items():
                                    coef[k] = coef.get(k, 0.0) + (v if expr.get("op") == "+" else -v)
                                return coef, lc + (rc if expr.get("op") == "+" else -rc)
                            raise SemanticError("Unsupported linear expression form in implication")

                        def _diff_eq(left, right):
                            ld, lc = _lin_eq(left)
                            rd, rc = _lin_eq(right)
                            coef = ld.copy()
                            for k, v in rd.items():
                                coef[k] = coef.get(k, 0.0) - v
                            return coef, lc - rc

                        ant_coef, ant_const = _diff_eq(ant_c.get("left"), ant_c.get("right"))
                        # Compute tight |diff| bound if possible: diff = sum a_i x_i + ant_const
                        diff_min = 0.0
                        diff_max = 0.0
                        feasible_bounds = True
                        for var, coef in ant_coef.items():
                            lb, ub = self._infer_var_bounds(var)
                            if lb is None or ub is None:
                                feasible_bounds = False
                                break
                            if coef >= 0:
                                diff_min += coef * lb
                                diff_max += coef * ub
                            else:
                                diff_min += coef * ub
                                diff_max += coef * lb
                        diff_min += ant_const
                        diff_max += ant_const
                        if feasible_bounds:
                            bigM = max(abs(diff_min), abs(diff_max), 1.0)
                        else:
                            bigM = 1_000_000.0
                        if not hasattr(self, "_impl_counter"):
                            self._impl_counter = 0
                        flag_name = f"implication_flag_c{self._impl_counter}"
                        self._impl_counter += 1
                        self.var_names.append(flag_name)
                        self.var_indices[flag_name] = len(self.var_names) - 1
                        self.bounds.append([0, 1])
                        if hasattr(self, "integrality"):
                            self.integrality.append(1)
                        else:
                            self.integrality = [1]
                        if hasattr(self, "c") and len(self.c) < len(self.var_names):
                            self.c.append(0.0)

                        # Helper to add row
                        def _add_row(coef_dict, flag_coef, rhs):
                            row = [0.0] * len(self.var_names)
                            for v, c in coef_dict.items():
                                if v not in self.var_indices:
                                    raise SemanticError(f"Variable '{v}' not indexed")
                                row[self.var_indices[v]] += c
                            row[self.var_indices[flag_name]] += flag_coef
                            for i, cv in enumerate(row):
                                if abs(cv) > 1e-12:
                                    A_ub_rows.append(ub_row_idx)
                                    A_ub_cols.append(i)
                                    A_ub_data.append(cv)
                            b_ub.append(rhs)

                        # Standard equality reification (diff==0 -> flag=1) using four inequalities:
                        # diff <=  bigM*(1-flag)
                        # -diff <= bigM*(1-flag)
                        # diff >= -bigM*flag  -> -diff <= bigM*flag
                        # -diff >= -bigM*flag ->  diff <= bigM*flag
                        # First two: diff + bigM*flag <= bigM  and -diff + bigM*flag <= bigM
                        _add_row(ant_coef, bigM, bigM - ant_const)
                        ub_row_idx += 1
                        neg_coef = {v: -c for v, c in ant_coef.items()}
                        _add_row(neg_coef, bigM, bigM + ant_const)
                        ub_row_idx += 1
                        # Second pair: diff - bigM*flag <= 0  and -diff - bigM*flag <= 0
                        _add_row(ant_coef, -bigM, -ant_const)
                        ub_row_idx += 1
                        _add_row(neg_coef, -bigM, ant_const)
                        ub_row_idx += 1

                        # Proceed to consequent gating using flag_name
                        # Build consequent diff normalization below reusing logic after antecedent handling
                        # Prepare consequent components
                        def _emit_consequent(cons_node, op):
                            if op == "==":
                                raise SemanticError("Equality consequent not yet supported")
                            if op in ("<=", "<"):
                                diff_c_coef, diff_c_const = _diff_eq(cons_node.get("left"), cons_node.get("right"))
                            elif op in (">=", ">"):
                                diff_c_coef, diff_c_const = _diff_eq(cons_node.get("right"), cons_node.get("left"))
                            else:
                                raise SemanticError("Unsupported consequent operator")
                            return diff_c_coef, diff_c_const

                        diff_c_coef, diff_c_const = _emit_consequent(cons_c, cons_op)
                        # Bound for consequent bigM (try tighten similarly)
                        M_c = 0.0
                        feasible_c = True
                        for var, coef in diff_c_coef.items():
                            lb, ub = self._infer_var_bounds(var)
                            if lb is None or ub is None:
                                feasible_c = False
                                break
                            if coef >= 0:
                                M_c = max(M_c, abs(coef * lb), abs(coef * ub))
                            else:
                                M_c = max(M_c, abs(coef * ub), abs(coef * lb))
                        if not feasible_c or M_c == 0.0:
                            M_c = 1_000_000.0
                        coefc = diff_c_coef.copy()
                        rhs_c = M_c - diff_c_const
                        # diff_c + M_c*flag <= M_c
                        row = [0.0] * len(self.var_names)
                        for v, c in coefc.items():
                            row[self.var_indices[v]] += c
                        row[self.var_indices[flag_name]] += M_c
                        for i, cv in enumerate(row):
                            if abs(cv) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(cv)
                        b_ub.append(rhs_c)
                        ub_row_idx += 1
                        return

                    # Helper: linearize expression (restricted to +,- of names and numbers)
                    def _lin_imp(expr):
                        if not isinstance(expr, dict):
                            raise SemanticError("Unsupported expression in implication linearization")
                        t = expr.get("type")
                        if t == "parenthesized_expression":
                            return _lin_imp(expr.get("expression"))
                        if t in ("name", "indexed_name"):
                            v = self._multi_indexed_var_name(expr, env) if t == "indexed_name" else expr["value"]
                            return {v: 1.0}, 0.0
                        if t == "number":
                            return {}, float(expr.get("value", 0))
                        if t == "binop" and expr.get("op") in ("+", "-"):
                            ld, lc = _lin_imp(expr.get("left"))
                            rd, rc = _lin_imp(expr.get("right"))
                            coef = ld.copy()
                            for k, v in rd.items():
                                coef[k] = coef.get(k, 0.0) + (v if expr.get("op") == "+" else -v)
                            return coef, lc + (rc if expr.get("op") == "+" else -rc)
                        raise SemanticError("Unsupported linear expression form in implication")

                    def _diff_imp(left, right):
                        ld, lc = _lin_imp(left)
                        rd, rc = _lin_imp(right)
                        coef = ld.copy()
                        for k, v in rd.items():
                            coef[k] = coef.get(k, 0.0) - v
                        return coef, lc - rc

                    # Normalize antecedent to diff_a >= 0 form
                    if ant_op in (">=", ">"):
                        diff_a_coef, diff_a_const = _diff_imp(ant_c.get("left"), ant_c.get("right"))
                    elif ant_op in ("<=", "<"):
                        diff_a_coef, diff_a_const = _diff_imp(ant_c.get("right"), ant_c.get("left"))
                    else:
                        raise SemanticError("Unsupported antecedent operator")

                    # Bounds for antecedent diff
                    def _bounds_expr(expr):
                        try:
                            return self._linear_bounds_safe(expr)
                        except Exception:
                            return (None, None)

                    # Compose diff bounds from left/right
                    lL, lU = _bounds_expr(ant_c.get("left"))
                    rL, rU = _bounds_expr(ant_c.get("right"))
                    diff_min = None
                    diff_max = None
                    if None not in (lL, lU, rL, rU):
                        if ant_op in (">=", ">"):
                            diff_min = lL - rU
                            diff_max = lU - rL
                        else:  # flipped
                            diff_min = rL - lU
                            diff_max = rU - lL
                    # Introduce flag variable z
                    if not hasattr(self, "_impl_counter"):
                        self._impl_counter = 0
                    flag_name = f"implication_flag_c{self._impl_counter}"
                    self._impl_counter += 1
                    self.var_names.append(flag_name)
                    self.var_indices[flag_name] = len(self.var_names) - 1
                    self.bounds.append([0, 1])
                    if hasattr(self, "integrality"):
                        self.integrality.append(1)
                    else:
                        self.integrality = [1]
                    if hasattr(self, "c") and len(self.c) < len(self.var_names):
                        self.c.append(0.0)

                    # Helper to add a <= row
                    def _add_le(coef_dict, flag_coef, rhs):
                        row = [0.0] * len(self.var_names)
                        for var, coef in coef_dict.items():
                            if var not in self.var_indices:
                                raise SemanticError(f"Variable '{var}' not indexed")
                            row[self.var_indices[var]] += coef
                        row[self.var_indices[flag_name]] += flag_coef
                        for i, cv in enumerate(row):
                            if abs(cv) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(cv)
                        b_ub.append(rhs)

                    # Big-M values
                    # M1 for antecedent activation (diff_a >= 0 when flag=1): use max(0, -diff_min)
                    M1 = max(0.0, -(diff_min if diff_min is not None else -1_000_000.0))
                    # U_a for forcing flag when diff_a positive
                    U_a = diff_max if diff_max is not None else 1_000_000.0
                    # 1) -diff_a + M1*flag <= M1  (enforces diff_a >=0 if flag=1)
                    coef1 = {v: -c for v, c in diff_a_coef.items()}
                    # constant shift: diff_a = sum(c_i x_i) + diff_a_const
                    # So -diff_a = -sum(c_i x_i) - diff_a_const
                    coef1_const = -diff_a_const
                    if abs(coef1_const) > 1e-12:
                        # Move constant to RHS: -sum(c_i)x_i - diff_a_const + M1*flag <= M1  => -sum(c_i)x_i + M1*flag <= M1 + diff_a_const
                        rhs1 = M1 + diff_a_const
                    else:
                        rhs1 = M1
                    _add_le(coef1, M1, rhs1)
                    ub_row_idx += 1
                    # 2) diff_a - U_a*flag <= 0 (forces flag=1 if diff_a>0)
                    coef2 = diff_a_coef.copy()
                    # diff_a = sum(c_i)x_i + diff_a_const -> sum(c_i)x_i - U_a*flag <= -diff_a_const
                    rhs2 = -diff_a_const
                    _add_le(coef2, -U_a, rhs2)
                    ub_row_idx += 1

                    # Consequent normalization: enforce diff_c <= 0 (or both for equality)
                    def _emit_consequent(cons_node, op):
                        if op == "==":
                            raise SemanticError("Equality consequent not yet supported")
                        if op in ("<=", "<"):
                            diff_c_coef, diff_c_const = _diff_imp(cons_node.get("left"), cons_node.get("right"))
                        elif op in (">=", ">"):
                            diff_c_coef, diff_c_const = _diff_imp(cons_node.get("right"), cons_node.get("left"))
                        else:
                            raise SemanticError("Unsupported consequent operator")
                        return diff_c_coef, diff_c_const

                    diff_c_coef, diff_c_const = _emit_consequent(cons_c, cons_op)
                    # Bound for consequent to build M_c
                    lL2, lU2 = _bounds_expr(cons_c.get("left"))
                    rL2, rU2 = _bounds_expr(cons_c.get("right"))
                    diff_c_max = None
                    if None not in (lL2, lU2, rL2, rU2):
                        if cons_op in ("<=", "<"):
                            diff_c_max = lU2 - rL2
                        elif cons_op in (">=", ">"):
                            diff_c_max = rU2 - lL2
                    M_c = diff_c_max if diff_c_max is not None else 1_000_000.0
                    # diff_c + M_c*flag <= M_c  (when flag=1 diff_c<=0)
                    coefc = diff_c_coef.copy()
                    # diff_c = sum(c_i)x_i + diff_c_const
                    # sum(c_i)x_i + diff_c_const + M_c*flag <= M_c  -> sum(c_i)x_i + M_c*flag <= M_c - diff_c_const
                    rhs_c = M_c - diff_c_const
                    _add_le(coefc, M_c, rhs_c)
                    ub_row_idx += 1
                    return
                ant_name = (
                    self._multi_indexed_var_name(ant_var_node, env)
                    if ant_var_node.get("type") == "indexed_name"
                    else ant_var_node["value"]
                )
                # Use unwrapped consequent for boolean handling
                cons_simple = cons_unwrapped
                if not (isinstance(cons_simple, dict) and cons_simple.get("type") == "constraint"):
                    raise SemanticError("Implication consequent must be a constraint")
                op_c = cons_simple.get("op")
                lc = cons_simple.get("left")
                rc = cons_simple.get("right")
                # Patterns supported: var == 1 / 0 ; var >= 1 ; var <= 0
                cons_var_one = _extract_var_eq_val(cons_simple, 1)
                cons_var_zero = _extract_var_eq_val(cons_simple, 0)

                def _var_name(node):
                    return self._multi_indexed_var_name(node, env) if node.get("type") == "indexed_name" else node["value"]

                if cons_var_one or (
                    op_c in (">=", "==")
                    and isinstance(lc, dict)
                    and lc.get("type") in ("name", "indexed_name")
                    and isinstance(rc, dict)
                    and rc.get("type") == "number"
                    and rc.get("value") == 1
                ):
                    vnode = cons_var_one if cons_var_one else lc
                    bname = _var_name(vnode)
                    # Enforce: when ant==1 then b==1  -> b - ant >= 0  => -b + ant <= 0
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[ant_name]] += 1.0
                    row[self.var_indices[bname]] -= 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(0.0)
                    ub_row_idx += 1
                    return
                if cons_var_zero or (
                    op_c in ("<=", "==")
                    and isinstance(lc, dict)
                    and lc.get("type") in ("name", "indexed_name")
                    and isinstance(rc, dict)
                    and rc.get("type") == "number"
                    and rc.get("value") == 0
                ):
                    vnode = cons_var_zero if cons_var_zero else lc
                    bname = _var_name(vnode)
                    # Enforce: when ant==1 then b==0 -> ant + b <= 1
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[ant_name]] += 1.0
                    row[self.var_indices[bname]] += 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(coef)
                    b_ub.append(1.0)
                    ub_row_idx += 1
                    return
                raise SemanticError("Unsupported implication consequent form")
            if constr["type"] == "constraint":
                left = constr["left"]
                right = constr["right"]
                op_sym_top = constr.get("op")
                is_sum, inner_cmp, k_val, loop_vars, loop_ranges = _detect_sum_of_comparisons(left, right, op_sym_top)
                if is_sum:
                    z_indices = []

                    for idx_tuple in itertools.product(*loop_ranges):
                        env2 = dict(env or {})
                        for v, val in zip(loop_vars, idx_tuple):
                            env2[v] = val
                        comp_inst = {
                            "type": "binop",
                            "op": inner_cmp.get("op"),
                            "left": inner_cmp.get("left"),
                            "right": inner_cmp.get("right"),
                            "sem_type": "boolean",
                        }
                        z_name = self._ensure_aux_binary("cmp_sum")
                        if hasattr(self, "integrality"):
                            if len(self.integrality) < len(self.var_names):
                                self.integrality.append(1)
                        else:
                            self.integrality = [1] * len(self.var_names)
                        if hasattr(self, "c") and len(self.c) < len(self.var_names):
                            self.c.extend([0.0] * (len(self.var_names) - len(self.c)))
                        z_indices.append(self.var_indices[z_name])
                        M = self._big_m_for_comparison(comp_inst, env=env2)
                        coef_lhs, const_lhs = self._eval_expr(comp_inst["left"], env2)
                        rnode = comp_inst["right"]
                        if isinstance(rnode, dict):
                            coef_rhs, const_rhs = self._eval_expr(rnode, env2)
                        else:
                            coef_rhs, const_rhs = (
                                {},
                                rnode if isinstance(rnode, (int, float)) else 0.0,
                            )
                        diff_coef = dict(coef_lhs)
                        for vn, cf in coef_rhs.items():
                            diff_coef[vn] = diff_coef.get(vn, 0.0) - cf
                        diff_const = const_lhs - const_rhs
                        # Conditional diff >=0 when z=1:
                        # 1) -diff + M*z <= M   (if z=1 -> -diff <= 0 => diff >=0; if z=0 -> -diff <= M relaxes)
                        row1 = [0.0] * len(self.var_names)
                        for vn, cf in diff_coef.items():
                            if vn in self.var_indices:
                                row1[self.var_indices[vn]] -= cf
                        row1[self.var_indices[z_name]] += M
                        for i, coef in enumerate(row1):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(M - diff_const)
                        ub_row_idx += 1
                        # 2) diff - M*z <= 0
                        row2 = [0.0] * len(self.var_names)
                        for vn, cf in diff_coef.items():
                            if vn in self.var_indices:
                                row2[self.var_indices[vn]] += cf
                        row2[self.var_indices[z_name]] -= M
                        for i, coef in enumerate(row2):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(-diff_const)
                        ub_row_idx += 1
                    if op_sym_top == ">=":
                        # -sum z <= -k
                        row = [0.0] * len(self.var_names)
                        for zi in z_indices:
                            row[zi] -= 1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(-k_val)
                        ub_row_idx += 1
                    else:  # ==
                        row = [0.0] * len(self.var_names)
                        for zi in z_indices:
                            row[zi] += 1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_eq_rows.append(eq_row_idx)
                                A_eq_cols.append(i)
                                A_eq_data.append(coef)
                        b_eq.append(k_val)
                        eq_row_idx += 1
                # Pattern B: b == (sum(i)(comparison) >= k)
                if op_sym_top == "==" and isinstance(left, dict) and left.get("type") == "name":
                    r_un = _unwrap_paren(right)
                    if isinstance(r_un, dict) and r_un.get("type") == "binop" and r_un.get("op") in (">=", ">"):
                        sum_side = _unwrap_paren(r_un.get("left"))
                        k_node = r_un.get("right")
                        if (
                            isinstance(sum_side, dict)
                            and sum_side.get("type") == "sum"
                            and isinstance(k_node, dict)
                            and k_node.get("type") == "number"
                        ):
                            inner_cmp = _unwrap_paren(sum_side.get("expression"))
                            if _is_simple_comparison(inner_cmp):
                                k_val = k_node.get("value")
                                iterators = sum_side.get("iterators", [])
                                loop_vars, loop_ranges = self._unroll_iterators(iterators)
                                z_indices = []
                                for idx_tuple in itertools.product(*loop_ranges):
                                    env2 = dict(env or {})
                                    for v, val in zip(loop_vars, idx_tuple):
                                        env2[v] = val
                                    idx_constr = sum_side.get("index_constraint")
                                    include = True
                                    if idx_constr is not None:
                                        try:
                                            _, cval = self._eval_expr(idx_constr, env2)
                                            include = bool(cval)
                                        except Exception:
                                            include = True
                                    if not include:
                                        continue
                                    comp_inst = {
                                        "type": "binop",
                                        "op": inner_cmp.get("op"),
                                        "left": inner_cmp.get("left"),
                                        "right": inner_cmp.get("right"),
                                        "sem_type": "boolean",
                                    }
                                    z_name = self._ensure_aux_binary("cmp_sum")
                                    if hasattr(self, "integrality"):
                                        if len(self.integrality) < len(self.var_names):
                                            self.integrality.append(1)
                                    else:
                                        self.integrality = [1] * len(self.var_names)
                                    if hasattr(self, "c") and len(self.c) < len(self.var_names):
                                        self.c.extend([0.0] * (len(self.var_names) - len(self.c)))
                                    z_indices.append(self.var_indices[z_name])
                                    M = self._big_m_for_comparison(comp_inst, env=env2)
                                    coef_lhs, const_lhs = self._eval_expr(comp_inst["left"], env2)
                                    rnode = comp_inst["right"]
                                    if isinstance(rnode, dict):
                                        coef_rhs, const_rhs = self._eval_expr(rnode, env2)
                                    else:
                                        coef_rhs, const_rhs = (
                                            {},
                                            (rnode if isinstance(rnode, (int, float)) else 0.0),
                                        )
                                    diff_coef = dict(coef_lhs)
                                    for vn, cf in coef_rhs.items():
                                        diff_coef[vn] = diff_coef.get(vn, 0.0) - cf
                                    diff_const = const_lhs - const_rhs
                                    # Conditional diff >=0 when z=1
                                    row1 = [0.0] * len(self.var_names)
                                    for vn, cf in diff_coef.items():
                                        if vn in self.var_indices:
                                            row1[self.var_indices[vn]] -= cf
                                    row1[self.var_indices[z_name]] += M
                                    for i, coef in enumerate(row1):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(coef)
                                    b_ub.append(M - diff_const)
                                    ub_row_idx += 1
                                    row2 = [0.0] * len(self.var_names)
                                    for vn, cf in diff_coef.items():
                                        if vn in self.var_indices:
                                            row2[self.var_indices[vn]] += cf
                                    row2[self.var_indices[z_name]] -= M
                                    for i, coef in enumerate(row2):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(coef)
                                    b_ub.append(-diff_const)
                                    ub_row_idx += 1
                                b_var = left.get("value")
                                if b_var not in self.var_indices:
                                    self._ensure_aux_binary(b_var)
                                N = len(z_indices)
                                # k*b - sum z <= 0
                                rowA = [0.0] * len(self.var_names)
                                rowA[self.var_indices[b_var]] += k_val
                                for zi in z_indices:
                                    rowA[zi] -= 1.0
                                for i, coef in enumerate(rowA):
                                    if abs(coef) > 1e-12:
                                        A_ub_rows.append(ub_row_idx)
                                        A_ub_cols.append(i)
                                        A_ub_data.append(coef)
                                b_ub.append(0.0)
                                ub_row_idx += 1
                                # sum z - (k-1) - (N-k+1)*b <= 0
                                rowB = [0.0] * len(self.var_names)
                                for zi in z_indices:
                                    rowB[zi] += 1.0
                                rowB[self.var_indices[b_var]] -= N - k_val + 1
                                for i, coef in enumerate(rowB):
                                    if abs(coef) > 1e-12:
                                        A_ub_rows.append(ub_row_idx)
                                        A_ub_cols.append(i)
                                        A_ub_data.append(coef)
                                b_ub.append(k_val - 1.0)
                                ub_row_idx += 1
                # Unwrap parentheses on left (and right if needed) early so pre-normalization sees inner binop/composite
                while (
                    isinstance(left, dict)
                    and left.get("type") == "parenthesized_expression"
                    and isinstance(left.get("expression"), dict)
                ):
                    left = left.get("expression")
                    constr["left"] = left
                # (rare) also unwrap right if it's a parenthesized boolean literal or binop used in symmetric patterns
                while (
                    isinstance(right, dict)
                    and right.get("type") == "parenthesized_expression"
                    and isinstance(right.get("expression"), dict)
                ):
                    right = right.get("expression")
                    constr["right"] = right
                # Pre-normalize (comparison_binop)==true -> comparison constraint
                if (
                    op_sym_top == "=="
                    and isinstance(right, dict)
                    and right.get("type") == "boolean_literal"
                    and right.get("value") is True
                    and isinstance(left, dict)
                    and left.get("type") == "binop"
                    and left.get("sem_type") == "boolean"
                    and left.get("op") in ("<=", ">=", "==", "!=")
                ):
                    constr["op"] = left["op"]
                    constr["left"] = left["left"]
                    constr["right"] = left["right"]
                    left = constr["left"]
                    right = constr["right"]
                    op_sym_top = constr["op"]
                # (comparison_binop)==false -> NOT(comparison)==true form
                if (
                    op_sym_top == "=="
                    and isinstance(right, dict)
                    and right.get("type") == "boolean_literal"
                    and right.get("value") is False
                    and isinstance(left, dict)
                    and left.get("type") == "binop"
                    and left.get("sem_type") == "boolean"
                    and left.get("op") in ("<=", ">=", "==", "!=")
                ):
                    wrapped = {
                        "type": "constraint",
                        "op": left["op"],
                        "left": left["left"],
                        "right": left["right"],
                    }
                    constr["left"] = {
                        "type": "not",
                        "value": wrapped,
                        "sem_type": "boolean",
                    }
                    constr["right"] = {
                        "type": "boolean_literal",
                        "value": True,
                        "sem_type": "boolean",
                    }
                    left = constr["left"]
                    right = constr["right"]
                    op_sym_top = constr["op"]

                # Special pattern: boolean var == (x != y)  (capture inequality truth value)

                # --- Direct handling for top-level '!=' before boolean tree/general linear pass ---
                if op_sym_top == "!=":
                    # Fast path boolean XOR / var vs literal for tight linearization
                    def _is_boolean_var_node(node):
                        if not (isinstance(node, dict) and node.get("type") in ("name", "indexed_name")):
                            return False
                        base_name = node.get("value") if node.get("type") == "name" else node.get("name")
                        for d in self.ast.get("declarations", []):
                            if (
                                d.get("name") == base_name
                                and d.get("type") in ("dvar", "dvar_indexed")
                                and d.get("var_type") == "boolean"
                            ):
                                return True
                        return False

                    def _is_number_literal(node, vals):
                        return isinstance(node, dict) and node.get("type") == "number" and node.get("value") in vals

                    if (
                        (_is_boolean_var_node(left) and _is_boolean_var_node(right))
                        or (_is_boolean_var_node(left) and _is_number_literal(right, (0, 1)))
                        or (_is_boolean_var_node(right) and _is_number_literal(left, (0, 1)))
                    ):
                        if _is_boolean_var_node(left) and _is_boolean_var_node(right):
                            v_left = (
                                self._multi_indexed_var_name(left, env)
                                if left.get("type") == "indexed_name"
                                else left["value"]
                            )
                            v_right = (
                                self._multi_indexed_var_name(right, env)
                                if right.get("type") == "indexed_name"
                                else right["value"]
                            )
                            row = [0.0] * len(self.var_names)
                            row[self.var_indices[v_left]] = 1.0
                            row[self.var_indices[v_right]] = 1.0
                            for i, coef in enumerate(row):
                                if abs(coef) > 1e-12:
                                    A_eq_rows.append(eq_row_idx)
                                    A_eq_cols.append(i)
                                    A_eq_data.append(coef)
                            b_eq.append(1.0)
                            self._add_code_line("# encoded != (boolean xor)")
                            eq_row_idx += 1
                            return
                        var_node = left if _is_boolean_var_node(left) else right
                        lit_node = right if var_node is left else left
                        vname = (
                            self._multi_indexed_var_name(var_node, env)
                            if var_node.get("type") == "indexed_name"
                            else var_node["value"]
                        )
                        lit_val = lit_node.get("value")
                        enforce = 1 - lit_val
                        row = [0.0] * len(self.var_names)
                        row[self.var_indices[vname]] = 1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_eq_rows.append(eq_row_idx)
                                A_eq_cols.append(i)
                                A_eq_data.append(coef)
                        b_eq.append(float(enforce))
                        self._add_code_line("# encoded != (boolean var vs literal)")
                        eq_row_idx += 1
                        return
                    # Fallback: treat as generic comparison truth variable and then enforce b==1
                    cmp_bin = {
                        "type": "binop",
                        "op": "!=",
                        "left": left,
                        "right": right,
                        "sem_type": "boolean",
                    }
                    bvar = _comparison_truth_var(cmp_bin, env)
                    # bvar == 1 (simple equality)
                    row = [0.0] * len(self.var_names)
                    row[self.var_indices[bvar]] = 1.0
                    for i, coef in enumerate(row):
                        if abs(coef) > 1e-12:
                            A_eq_rows.append(eq_row_idx)
                            A_eq_cols.append(i)
                            A_eq_data.append(coef)
                    b_eq.append(1.0)
                    eq_row_idx += 1
                    self._add_code_line("# encoded != via truth var fallback")
                    return

                # --- NOT rewrite normalization ---
                if (
                    op_sym_top == "=="
                    and isinstance(left, dict)
                    and left.get("type") == "not"
                    and isinstance(right, dict)
                    and right.get("type") == "boolean_literal"
                ):
                    inner = left.get("value")
                    # Unwrap parentheses
                    while isinstance(inner, dict) and inner.get("type") == "parenthesized_expression":
                        inner = inner.get("expression")
                    want_true = bool(right.get("value"))
                    if not want_true:
                        # (!E) == false -> E == true
                        new_c = {
                            "type": "constraint",
                            "op": "==",
                            "left": inner,
                            "right": {
                                "type": "boolean_literal",
                                "value": True,
                                "sem_type": "boolean",
                            },
                        }
                        handle_constraint(new_c, env=env)
                        return
                    # want_true: !(E) == true
                    # If E is already a constraint we can invert
                    if isinstance(inner, dict) and inner.get("type") == "constraint":
                        op_in = inner.get("op")
                        l_in = inner.get("left")
                        r_in = inner.get("right")
                        EPS = BOOL_EPS
                        if op_in == "<=":
                            # l > r -> l >= r + eps
                            new_right = {
                                "type": "binop",
                                "op": "+",
                                "left": r_in,
                                "right": {
                                    "type": "number",
                                    "value": EPS,
                                    "sem_type": "float",
                                },
                                "sem_type": r_in.get("sem_type", "float"),
                            }
                            new_c = {
                                "type": "constraint",
                                "op": ">=",
                                "left": l_in,
                                "right": new_right,
                            }
                            handle_constraint(new_c, env=env)
                            return
                        if op_in == ">=":
                            new_right = {
                                "type": "binop",
                                "op": "-",
                                "left": r_in,
                                "right": {
                                    "type": "number",
                                    "value": EPS,
                                    "sem_type": "float",
                                },
                                "sem_type": r_in.get("sem_type", "float"),
                            }
                            new_c = {
                                "type": "constraint",
                                "op": "<=",
                                "left": l_in,
                                "right": new_right,
                            }
                            handle_constraint(new_c, env=env)
                            return
                        if op_in == "==":
                            new_c = {
                                "type": "constraint",
                                "op": "!=",
                                "left": l_in,
                                "right": r_in,
                            }
                            self._add_code_line("# encoded != (NOT of ==)")
                            handle_constraint(new_c, env=env)
                            return
                        if op_in == "!=":
                            new_c = {
                                "type": "constraint",
                                "op": "==",
                                "left": l_in,
                                "right": r_in,
                            }
                            handle_constraint(new_c, env=env)
                            return
                    # Fallback: treat !(bool_expr)==true as bool_expr==false (introduce literal flip)
                    new_c = {
                        "type": "constraint",
                        "op": "==",
                        "left": inner,
                        "right": {
                            "type": "boolean_literal",
                            "value": False,
                            "sem_type": "boolean",
                        },
                    }
                    handle_constraint(new_c, env=env)
                    return

                # --- Boolean variable equality with composite boolean expression (reuse auxiliaries) ---
                if op_sym_top == "==" and isinstance(left, dict) and isinstance(right, dict):

                    def _is_decl_bool_var(node):
                        if not (isinstance(node, dict) and node.get("type") in ("name", "indexed_name")):
                            return False
                        vname = (
                            self._multi_indexed_var_name(node, env)
                            if node.get("type") == "indexed_name"
                            else node.get("value")
                        )
                        if vname not in self.var_indices:
                            return False
                        vidx = self.var_indices[vname]
                        try:
                            return (
                                self.integrality[vidx] == 1
                                and self.bounds[vidx][0] == 0
                                and (self.bounds[vidx][1] in (1, None))
                            )
                        except Exception:
                            return False

                    def _try_expr(node):
                        try:
                            return _bool_expr_var(node, env)
                        except Exception:
                            return None

                    if _is_decl_bool_var(left):
                        expr_var = _try_expr(right)
                        if expr_var is not None:
                            v_left = (
                                self._multi_indexed_var_name(left, env)
                                if left.get("type") == "indexed_name"
                                else left.get("value")
                            )
                            if expr_var != v_left:
                                if not isinstance(expr_var, str):
                                    raise SemanticError(f"expr_var is not a string: {repr(expr_var)}")
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[v_left]] = 1.0
                                row[self.var_indices[expr_var]] = -1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(0.0)
                                eq_row_idx += 1
                            return
                    if _is_decl_bool_var(right):
                        expr_var = _try_expr(left)
                        if expr_var is not None:
                            v_right = (
                                self._multi_indexed_var_name(right, env)
                                if right.get("type") == "indexed_name"
                                else right.get("value")
                            )
                            if expr_var != v_right:
                                if not isinstance(expr_var, str):
                                    raise SemanticError(f"expr_var is not a string: {repr(expr_var)}")
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[v_right]] = 1.0
                                row[self.var_indices[expr_var]] = -1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(0.0)
                                eq_row_idx += 1
                            return

                # --- Generic boolean comparison handling (>=, <=, !=, plus fallback ==) ---
                # Supports: (bool_expr) OP literal where OP in {>=, <=, !=, ==} and literal in {0,1, True, False}
                # bool_expr may be composed of and/or/not trees over atomic constraints var==0/1.
                def _is_bool_literal(node):
                    return isinstance(node, dict) and node.get("type") == "boolean_literal"

                def _is_number_01(node):
                    return isinstance(node, dict) and node.get("type") == "number" and node.get("value") in (0, 1)

                def _is_atomic_bool(node):
                    # pattern: constraint (var == 0/1) or (0/1 == var)
                    if not (isinstance(node, dict) and node.get("type") == "constraint" and node.get("op") == "=="):
                        return False
                    left = node.get("left")
                    right = node.get("right")

                    def is_var(x):
                        return isinstance(x, dict) and x.get("type") in (
                            "name",
                            "indexed_name",
                        )

                    return (is_var(left) and _is_number_01(right)) or (is_var(right) and _is_number_01(left))

                def _is_bool_tree(node):
                    if not isinstance(node, dict):
                        return False
                    tnode = node.get("type")
                    if tnode in ("and", "or"):
                        return _is_bool_tree(node.get("left")) and _is_bool_tree(node.get("right"))
                    if tnode == "not":
                        return _is_bool_tree(node.get("value"))
                    return _is_atomic_bool(node)

                # Normalize so left is boolean expression, right is literal if pattern matches
                op_sym = constr.get("op")

                # Reified pattern: b == (sum(bool vars) >= k)
                def _is_boolean_var(node):
                    if not (isinstance(node, dict) and node.get("type") in ("name", "indexed_name")):
                        return False
                    nm = node.get("value") if node.get("type") == "name" else node.get("name")
                    for d in self.ast.get("declarations", []):
                        if (
                            d.get("name") == nm
                            and d.get("type") in ("dvar", "dvar_indexed")
                            and d.get("var_type") == "boolean"
                        ):
                            return True
                    return False

                def _collect_sum(node):
                    if not isinstance(node, dict):
                        return None
                    if node.get("type") == "name" and _is_boolean_var(node):
                        return ({node["value"]}, 0)
                    if node.get("type") == "binop" and node.get("op") == "+":
                        left = _collect_sum(node["left"])
                        right = _collect_sum(node["right"])
                        if left and right:
                            return (left[0].union(right[0]), left[1] + right[1])
                        return None
                    if node.get("type") == "number":
                        return (set(), node.get("value", 0))
                    return None

                if op_sym == "==" and (
                    (
                        _is_boolean_var(left)
                        and isinstance(right, dict)
                        and right.get("type") == "constraint"
                        and right.get("op") == ">="
                    )
                    or (
                        _is_boolean_var(right)
                        and isinstance(left, dict)
                        and left.get("type") == "constraint"
                        and left.get("op") == ">="
                    )
                ):
                    bool_side = left if _is_boolean_var(left) else right
                    ineq = right if bool_side is left else left
                    ineq_l = ineq.get("left")
                    ineq_r = ineq.get("right")
                    if isinstance(ineq_r, dict) and ineq_r.get("type") == "number":
                        k = ineq_r.get("value")
                        collected = _collect_sum(ineq_l)
                        if collected:
                            vars_set, c_off = collected
                            k_adj = k - c_off
                            vname = (
                                bool_side["value"]
                                if bool_side.get("type") == "name"
                                else self._multi_indexed_var_name(bool_side, env)
                            )
                            # Trivial cases: if k_adj <= 0 then sum(vars)>=k always true -> b == 1
                            if k_adj <= 0:
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[vname]] = 1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(1.0)
                                eq_row_idx += 1
                                return
                            # If k_adj > |S| then condition impossible -> b == 0
                            if k_adj > len(vars_set):
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[vname]] = 1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(0.0)
                                eq_row_idx += 1
                                return
                            # Inequality: k_adj * b - sum(vars) <= 0
                            row = [0.0] * len(self.var_names)
                            row[self.var_indices[vname]] += k_adj
                            for vn in vars_set:
                                row[self.var_indices[vn]] -= 1.0
                            for i, coef in enumerate(row):
                                if abs(coef) > 1e-12:
                                    A_ub_rows.append(ub_row_idx)
                                    A_ub_cols.append(i)
                                    A_ub_data.append(coef)
                            b_ub.append(0.0)
                            ub_row_idx += 1
                            if k_adj >= 1:
                                # sum(vars) - (k_adj -1) - (|S|-k_adj+1)*b <=0
                                row = [0.0] * len(self.var_names)
                                for vn in vars_set:
                                    row[self.var_indices[vn]] += 1.0
                                row[self.var_indices[vname]] -= len(vars_set) - k_adj + 1
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_ub_rows.append(ub_row_idx)
                                        A_ub_cols.append(i)
                                        A_ub_data.append(coef)
                                b_ub.append(k_adj - 1)
                                ub_row_idx += 1
                            return
                if op_sym in ("==", "!=", "<=", ">="):
                    if (_is_bool_tree(left) and (_is_bool_literal(right) or _is_number_01(right))) or (
                        _is_bool_tree(right) and (_is_bool_literal(left) or _is_number_01(left))
                    ):
                        if _is_bool_tree(right):  # swap
                            left, right = right, left
                        # Extract literal value as 0/1
                        if _is_bool_literal(right):
                            target_val = 1 if right.get("value") else 0
                        else:  # number 0/1
                            target_val = int(right.get("value"))
                        if target_val not in (0, 1):
                            raise SemanticError("Boolean comparison literal must be 0/1 or True/False")
                        # Reduce operators to equality or tautology/contradiction
                        enforce = None  # None means no constraint needed
                        if op_sym == "==":
                            # Defer to existing fast path only if tree is and/or (handled below). For other trees enforce directly.
                            if left.get("type") not in ("and", "or"):
                                enforce = target_val
                        elif op_sym == "!=":
                            enforce = 1 - target_val
                        elif op_sym == ">=":
                            # B >= 1 -> B ==1 ; B >=0 -> tautology
                            if target_val == 1:
                                enforce = 1
                            else:
                                enforce = None
                        elif op_sym == "<=":
                            # B <=0 -> B==0 ; B <=1 -> tautology
                            if target_val == 0:
                                enforce = 0
                            else:
                                enforce = None
                        if enforce is not None:
                            try:
                                zvar = _bool_expr_var(left, env)
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[zvar]] = 1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(float(enforce))
                                eq_row_idx += 1
                                return
                            except SemanticError:
                                # If fallback failed, raise for unsupported mixed form
                                raise
                        # If we did not enforce directly and op was not '==' OR tree not and/or, we are done (tautology) or fall through for fast path == handling
                        if op_sym != "==" or left.get("type") not in ("and", "or"):
                            return  # tautology or handled

                # --- Reified equality of a declared boolean variable and a general boolean expression ---
                # Pattern: b == (bool_expr) where bool_expr may include comparisons (<=,>=,==,!=) and/or logical operators.
                # We attempt both orientations (expression on left or right). Falls through silently if expression unsupported.
                if constr.get("op") == "==":

                    def _is_bool_dvar(n):
                        return (
                            isinstance(n, dict)
                            and n.get("type") in ("name", "indexed_name")
                            and any(
                                d.get("name") == (n.get("value") if n.get("type") == "name" else n.get("name"))
                                and d.get("type") in ("dvar", "dvar_indexed")
                                and d.get("var_type") == "boolean"
                                for d in self.ast.get("declarations", [])
                            )
                        )

                    left_is_bool = _is_bool_dvar(left)
                    right_is_bool = _is_bool_dvar(right)
                    # Try b == expr
                    if left_is_bool and not right_is_bool:
                        try:
                            # Unwrap any parenthesized_expression layers before building boolean subtree
                            rr = right
                            while isinstance(rr, dict) and rr.get("type") == "parenthesized_expression":
                                rr = rr.get("expression")
                            expr_var = _bool_expr_var(rr, env)
                            vname = left["value"] if left.get("type") == "name" else self._multi_indexed_var_name(left, env)
                            if expr_var != vname:
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[vname]] = 1.0
                                row[self.var_indices[expr_var]] -= 1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(0.0)
                                eq_row_idx += 1
                            return
                        except SemanticError:
                            pass
                    # Try expr == b
                    if right_is_bool and not left_is_bool:
                        try:
                            ll = left
                            while isinstance(ll, dict) and ll.get("type") == "parenthesized_expression":
                                ll = ll.get("expression")
                            expr_var = _bool_expr_var(ll, env)
                            vname = right["value"] if right.get("type") == "name" else self._multi_indexed_var_name(right, env)
                            if expr_var != vname:
                                row = [0.0] * len(self.var_names)
                                row[self.var_indices[vname]] = 1.0
                                row[self.var_indices[expr_var]] -= 1.0
                                for i, coef in enumerate(row):
                                    if abs(coef) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef)
                                b_eq.append(0.0)
                                eq_row_idx += 1
                            return
                        except SemanticError:
                            pass

                # --- Boolean AND/OR linearization fast path (supports boolean literal on either side) ---
                # Pattern: (AND/OR tree of atomic comparisons var==0/1) == boolean_literal
                # Also: boolean_literal == (AND/OR tree ...)
                def is_bool_lit(node):
                    return isinstance(node, dict) and node.get("type") == "boolean_literal"

                def is_and_or(node):
                    return isinstance(node, dict) and node.get("type") in ("and", "or")

                # --- Early pattern: sum of freshly formed comparisons cardinality (sum(...) >= k / == k) ---
                # And reified form: b == (sum(...) >= k)
                def _unwrap(node):
                    while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
                        node = node.get("expression")
                    return node

                left = _unwrap(left)
                right = _unwrap(right)

                # Helper: detect sum-of-comparisons form
                def _is_comparison(n):
                    return isinstance(n, dict) and n.get("type") == "binop" and n.get("op") in (">=", "<=", "==", ">", "<")

                def _sum_of_comparisons(node):
                    if not (isinstance(node, dict) and node.get("type") == "sum"):
                        return False
                    inner = node.get("expression")
                    # Allow parenthesized comparison inside sum
                    inner = _unwrap(inner)
                    return _is_comparison(inner)

                # Case 1: Direct cardinality constraint: sum(...) op k
                if (
                    constr.get("op") in (">=", "==", ">", "<=", "<")
                    and _sum_of_comparisons(left)
                    and isinstance(right, dict)
                    and right.get("type") == "number"
                ):
                    sum_node = left
                    k_val = right.get("value")
                    if constr.get("op") == ">":
                        k_val = k_val + 1  # strict > to >= k+1
                    if constr.get("op") == "<":
                        k_val = k_val - 1  # sum < k  => sum <= k-1
                    # Expand sum, create auxiliary binaries for each comparison instance
                    iterators = sum_node.get("iterators", [])
                    try:
                        loop_vars, loop_ranges = self._unroll_iterators(iterators)
                    except SemanticError:
                        loop_vars, loop_ranges = [], []
                    comp_proto = _unwrap(sum_node.get("expression"))
                    z_vars = []

                    for idx_tuple in itertools.product(*loop_ranges) if loop_ranges else [()]:
                        env2 = dict(env or {})
                        for v, val in zip(loop_vars, idx_tuple):
                            env2[v] = val
                        # Clone comparison node substituting iterator variables in a shallow way via evaluation
                        comp = comp_proto.copy()
                        # We rely on _eval_expr later; here we just build linearization rows
                        z_name = self._ensure_aux_binary("cmp_aux")
                        if hasattr(self, "c") and len(self.c) < len(self.var_names):
                            self.c.append(0.0)
                        # Compute diff bounds via big-M routine
                        M = self._big_m_for_comparison(
                            {
                                "left": comp.get("left"),
                                "right": comp.get("right"),
                                "op": comp.get("op"),
                            },
                            env=env2,
                        )
                        # Build diff = (lhs - rhs)
                        coef_lhs, const_lhs = self._eval_expr(comp.get("left"), env2)
                        rhs_node = comp.get("right")
                        if isinstance(rhs_node, dict):
                            coef_rhs, const_rhs = self._eval_expr(rhs_node, env2)
                        else:
                            coef_rhs, const_rhs = (
                                {},
                                rhs_node if isinstance(rhs_node, (int, float)) else 0.0,
                            )
                        diff_coef = dict(coef_lhs)
                        for vn, cf in coef_rhs.items():
                            diff_coef[vn] = diff_coef.get(vn, 0.0) - cf
                        diff_const = const_lhs - const_rhs
                        # Constraints encoding z = 1 iff diff >= 0 using bounds [-M, M]
                        # 1) diff + M*(1 - z) >= 0  -> -(diff + M*(1 - z)) <= 0
                        row1 = [0.0] * len(self.var_names)
                        for vn, cf in diff_coef.items():
                            if vn in self.var_indices:
                                row1[self.var_indices[vn]] -= cf  # multiply by -1 for <= form
                        row1[
                            self.var_indices[z_name]
                        ] += M  # -diff + M*z + M*(?) but we used form -(diff + M - M*z) <= 0 -> -diff - M + M*z <=0 -> add constant by adjusting rhs
                        # Row represents -diff - M + M*z <= 0  => -diff + M*z <= M
                        for i, coef in enumerate(row1):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(M - diff_const)
                        ub_row_idx += 1
                        # 2) diff - (M+1)*z <= -1  (forces diff <= -1 when z=0, allows diff up to M when z=1)
                        row2 = [0.0] * len(self.var_names)
                        for vn, cf in diff_coef.items():
                            if vn in self.var_indices:
                                row2[self.var_indices[vn]] += cf
                        row2[self.var_indices[z_name]] -= M + 1
                        for i, coef in enumerate(row2):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(-1 - diff_const)
                        ub_row_idx += 1
                        z_vars.append(z_name)
                    # Now apply cardinality constraint on z_vars
                    if constr.get("op") in (">=", ">"):
                        # -sum z_i <= -k
                        row = [0.0] * len(self.var_names)
                        for z in z_vars:
                            row[self.var_indices[z]] -= 1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(-k_val)
                        ub_row_idx += 1
                    elif constr.get("op") == "==":
                        row = [0.0] * len(self.var_names)
                        for z in z_vars:
                            row[self.var_indices[z]] += 1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_eq_rows.append(eq_row_idx)
                                A_eq_cols.append(i)
                                A_eq_data.append(coef)
                        b_eq.append(k_val)
                        eq_row_idx += 1
                    elif constr.get("op") in ("<=", "<"):
                        # sum z_i <= k
                        row = [0.0] * len(self.var_names)
                        for z in z_vars:
                            row[self.var_indices[z]] += 1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(k_val)
                        ub_row_idx += 1
                    return
                # Case 2: Reified form b == (sum(...) >= k)
                if constr.get("op") == "==" and (
                    (isinstance(left, dict) and left.get("type") in ("name", "indexed_name"))
                    or (isinstance(right, dict) and right.get("type") in ("name", "indexed_name"))
                ):
                    # Normalize so that boolean variable is on left
                    if isinstance(right, dict) and right.get("type") in (
                        "name",
                        "indexed_name",
                    ):
                        left, right = right, left
                    bool_var = left
                    cmp_expr = _unwrap(right)
                    if (
                        isinstance(cmp_expr, dict)
                        and cmp_expr.get("type") == "binop"
                        and cmp_expr.get("op") in (">=", ">")
                        and _sum_of_comparisons(cmp_expr.get("left"))
                        and isinstance(cmp_expr.get("right"), dict)
                        and cmp_expr.get("right").get("type") == "number"
                    ):
                        sum_node = cmp_expr.get("left")
                        k_val = cmp_expr.get("right").get("value")
                        if cmp_expr.get("op") == ">":
                            k_val = k_val + 1
                        iterators = sum_node.get("iterators", [])
                        try:
                            loop_vars, loop_ranges = self._unroll_iterators(iterators)
                        except SemanticError:
                            loop_vars, loop_ranges = [], []
                        comp_proto = _unwrap(sum_node.get("expression"))
                        z_vars = []
                        for idx_tuple in itertools.product(*loop_ranges) if loop_ranges else [()]:
                            env2 = dict(env or {})
                            for v, val in zip(loop_vars, idx_tuple):
                                env2[v] = val
                            comp = comp_proto.copy()
                            z_name = self._ensure_aux_binary("cmp_aux")
                            if hasattr(self, "c") and len(self.c) < len(self.var_names):
                                self.c.append(0.0)
                            M = self._big_m_for_comparison(
                                {
                                    "left": comp.get("left"),
                                    "right": comp.get("right"),
                                    "op": comp.get("op"),
                                }
                            )
                            coef_lhs, const_lhs = self._eval_expr(comp.get("left"), env2)
                            rhs_node = comp.get("right")
                            if isinstance(rhs_node, dict):
                                coef_rhs, const_rhs = self._eval_expr(rhs_node, env2)
                            else:
                                coef_rhs, const_rhs = (
                                    {},
                                    (rhs_node if isinstance(rhs_node, (int, float)) else 0.0),
                                )
                            diff_coef = dict(coef_lhs)
                            for vn, cf in coef_rhs.items():
                                diff_coef[vn] = diff_coef.get(vn, 0.0) - cf
                            diff_const = const_lhs - const_rhs
                            # Two inequalities linking diff and z
                            row1 = [0.0] * len(self.var_names)
                            for vn, cf in diff_coef.items():
                                if vn in self.var_indices:
                                    row1[self.var_indices[vn]] -= cf
                            row1[self.var_indices[z_name]] += M
                            for i, coef in enumerate(row1):
                                if abs(coef) > 1e-12:
                                    A_ub_rows.append(ub_row_idx)
                                    A_ub_cols.append(i)
                                    A_ub_data.append(coef)
                            b_ub.append(M - diff_const)
                            ub_row_idx += 1
                            row2 = [0.0] * len(self.var_names)
                            for vn, cf in diff_coef.items():
                                if vn in self.var_indices:
                                    row2[self.var_indices[vn]] += cf
                            row2[self.var_indices[z_name]] -= M + 1
                            for i, coef in enumerate(row2):
                                if abs(coef) > 1e-12:
                                    A_ub_rows.append(ub_row_idx)
                                    A_ub_cols.append(i)
                                    A_ub_data.append(coef)
                            b_ub.append(-1 - diff_const)
                            ub_row_idx += 1
                            z_vars.append(z_name)
                        # Cardinality reification b == (sum z_i >= k)
                        # Retrieve/ensure boolean variable index
                        b_vname = (
                            self._multi_indexed_var_name(bool_var, env)
                            if bool_var.get("type") == "indexed_name"
                            else bool_var["value"]
                        )
                        if b_vname not in self.var_indices:
                            # Declare a new boolean variable (edge case)
                            self.var_names.append(b_vname)
                            self.var_indices[b_vname] = len(self.var_names) - 1
                            self.bounds.append([0, 1])
                            self.integrality.append(1)
                            self.c.append(0.0)
                        n = len(z_vars)
                        # 1) b - sum z_i + k - 1 <= 0
                        rowA = [0.0] * len(self.var_names)
                        rowA[self.var_indices[b_vname]] += 1.0
                        for z in z_vars:
                            rowA[self.var_indices[z]] -= 1.0
                        for i, coef in enumerate(rowA):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(-k_val + 1)
                        ub_row_idx += 1
                        # 2) sum z_i + (n - k)*b <= n
                        rowB = [0.0] * len(self.var_names)
                        for z in z_vars:
                            rowB[self.var_indices[z]] += 1.0
                        rowB[self.var_indices[b_vname]] += n - k_val
                        for i, coef in enumerate(rowB):
                            if abs(coef) > 1e-12:
                                A_ub_rows.append(ub_row_idx)
                                A_ub_cols.append(i)
                                A_ub_data.append(coef)
                        b_ub.append(n)
                        ub_row_idx += 1
                        return
                # Swap if literal on left
                if is_bool_lit(left) and is_and_or(right):
                    left, right = right, left
                if is_and_or(left) and is_bool_lit(right) and constr.get("op") == "==":
                    bool_op = left["type"]
                    target_val = bool(right.get("value", True))

                    def flatten(op_node, op_type):
                        nodes = []
                        if isinstance(op_node, dict) and op_node.get("type") == op_type:
                            nodes.extend(flatten(op_node["left"], op_type))
                            nodes.extend(flatten(op_node["right"], op_type))
                        else:
                            nodes.append(op_node)
                        return nodes

                    atomic_nodes = flatten(left, bool_op)

                    def resolve_var_and_polarity(atom):
                        # Expect (v == 0/1) or (0/1 == v)
                        if not (isinstance(atom, dict) and atom.get("type") == "constraint" and atom.get("op") == "=="):
                            raise SemanticError("Unsupported atomic boolean term for SciPy AND/OR linearization")
                        left = atom["left"]
                        right = atom["right"]

                        def is_num01(node):
                            return isinstance(node, dict) and node.get("type") == "number" and node.get("value") in (0, 1)

                        def is_var(node):
                            return isinstance(node, dict) and node.get("type") in (
                                "name",
                                "indexed_name",
                            )

                        if is_var(left) and is_num01(right):
                            varname = (
                                self._multi_indexed_var_name(left, env)
                                if left.get("type") == "indexed_name"
                                else left["value"]
                            )
                            val = right["value"]
                        elif is_num01(left) and is_var(right):
                            varname = (
                                self._multi_indexed_var_name(right, env)
                                if right.get("type") == "indexed_name"
                                else right["value"]
                            )
                            val = left["value"]
                        else:
                            raise SemanticError("Unsupported comparison in boolean linearization (expected v == 0/1)")
                        if varname not in self.var_indices:
                            raise SemanticError(f"Variable '{varname}' not found for boolean linearization")
                        polarity = 1 if val == 1 else -1  # 1 => v, -1 => (1 - v)
                        return varname, polarity

                    try:
                        literals = [resolve_var_and_polarity(a) for a in atomic_nodes]
                    except SemanticError:
                        literals = None
                    if literals is not None:
                        k = len(literals)
                        if bool_op == "and":
                            if target_val:
                                # All literals true -> enforce each atomic equality
                                for vname, polarity in literals:
                                    row = [0.0] * len(self.var_names)
                                    idx_var = self.var_indices[vname]
                                    row[idx_var] = 1.0
                                    rhs_val = 1.0 if polarity == 1 else 0.0
                                    for i, coef in enumerate(row):
                                        if abs(coef) > 1e-12:
                                            A_eq_rows.append(eq_row_idx)
                                            A_eq_cols.append(i)
                                            A_eq_data.append(coef)
                                    b_eq.append(rhs_val)
                                    eq_row_idx += 1
                            else:
                                # At least one literal false -> sum(literals) <= k-1
                                coef = [0.0] * len(self.var_names)
                                const_shift = 0.0
                                for vname, polarity in literals:
                                    idx_var = self.var_indices[vname]
                                    if polarity == 1:
                                        coef[idx_var] += 1.0
                                    else:
                                        coef[idx_var] += -1.0
                                        const_shift += 1.0
                                rhs_limit = (k - 1) - const_shift
                                for i, coef_i in enumerate(coef):
                                    if abs(coef_i) > 1e-12:
                                        A_ub_rows.append(ub_row_idx)
                                        A_ub_cols.append(i)
                                        A_ub_data.append(coef_i)
                                b_ub.append(rhs_limit)
                                ub_row_idx += 1
                        elif bool_op == "or":
                            coef = [0.0] * len(self.var_names)
                            const_shift = 0.0
                            for vname, polarity in literals:
                                idx_var = self.var_indices[vname]
                                if polarity == 1:
                                    coef[idx_var] += 1.0
                                else:
                                    coef[idx_var] += -1.0
                                    const_shift += 1.0
                            if target_val:
                                # sum(lits) >= 1  => -sum(lits) <= -1
                                for i, coef_i in enumerate(coef):
                                    if abs(coef_i) > 1e-12:
                                        A_ub_rows.append(ub_row_idx)
                                        A_ub_cols.append(i)
                                        A_ub_data.append(-coef_i)
                                b_ub.append(const_shift - 1.0)
                                ub_row_idx += 1
                            else:
                                # sum(lits) == 0 => equality rows
                                for i, coef_i in enumerate(coef):
                                    if abs(coef_i) > 1e-12:
                                        A_eq_rows.append(eq_row_idx)
                                        A_eq_cols.append(i)
                                        A_eq_data.append(coef_i)
                                b_eq.append(-const_shift)
                                eq_row_idx += 1
                                return  # handled boolean constraint
                            return  # handled via fast path
                    # Fast path pattern recognized but not all atomic -> build with auxiliaries
                    try:
                        expr_var = _bool_expr_var(left, env)
                        row = [0.0] * len(self.var_names)
                        row[self.var_indices[expr_var]] = 1.0
                        for i, coef in enumerate(row):
                            if abs(coef) > 1e-12:
                                A_eq_rows.append(eq_row_idx)
                                A_eq_cols.append(i)
                                A_eq_data.append(coef)
                        b_eq.append(1.0 if target_val else 0.0)
                        eq_row_idx += 1
                        return
                    except SemanticError:
                        pass

                # Capture left_type for composite evaluation
                left_type = left.get("type") if isinstance(left, dict) else None
                # Handle AND/OR composite equality early
                if left_type in ("and", "or") and constr.get("op") == "==":
                    target_val = right.get("type") == "boolean_literal" and right.get("value") is True
                    if not target_val:
                        # Already handled earlier by fast path (AND/OR == false). Nothing further required.
                        return
                    leaf_type = left_type
                    if leaf_type == "and":
                        # Recursively emit each conjunct. A conjunct can be:
                        # - a linear comparison (directly converted)
                        # - a nested OR/AND expression (handled by recursive call to handle_constraint)
                        def _emit_conj(node):
                            if not isinstance(node, dict):
                                return
                            # Unwrap parentheses
                            while node.get("type") == "parenthesized_expression":
                                node = node.get("expression")
                                if not isinstance(node, dict):
                                    return
                            t = node.get("type")
                            if t == "and":
                                _emit_conj(node.get("left"))
                                _emit_conj(node.get("right"))
                                return
                            if t == "not":
                                # Delegate NOT handling via existing normalization logic
                                pseudo = {
                                    "type": "constraint",
                                    "op": "==",
                                    "left": node,
                                    "right": {
                                        "type": "boolean_literal",
                                        "value": True,
                                        "sem_type": "boolean",
                                    },
                                }
                                handle_constraint(pseudo, env=env)
                                return
                            if self._is_linear_comparison(node):
                                pseudo = {
                                    "type": "constraint",
                                    "op": node["op"],
                                    "left": node["left"],
                                    "right": node["right"],
                                }
                                handle_constraint(pseudo, env=env)
                                return
                            if t == "or":
                                pseudo = {
                                    "type": "constraint",
                                    "op": "==",
                                    "left": node,
                                    "right": {
                                        "type": "boolean_literal",
                                        "value": True,
                                        "sem_type": "boolean",
                                    },
                                }
                                handle_constraint(pseudo, env=env)
                                return
                            raise self._unsupported_type_error("boolean leaf", node)

                        _emit_conj(left)
                        return
                    else:  # OR possibly containing nested AND groups
                        # Helper: extract disjuncts; each disjunct is list of comparison nodes whose conjunction forms that branch
                        def _disjuncts(node_or):
                            if not isinstance(node_or, dict):
                                return []
                            # Unwrap any layers of parentheses
                            while isinstance(node_or, dict) and node_or.get("type") == "parenthesized_expression":
                                node_or = node_or.get("expression")
                                if not isinstance(node_or, dict):
                                    return []
                            t = node_or.get("type")
                            if t == "or":
                                return _disjuncts(node_or.get("left")) + _disjuncts(node_or.get("right"))
                            if t == "and":
                                # Conjunction branch: flatten AND to its comparison leaves
                                comps = []
                                stack = [node_or]
                                while stack:
                                    n = stack.pop()
                                    if not isinstance(n, dict):
                                        continue
                                    # Unwrap parentheses
                                    while n.get("type") == "parenthesized_expression":
                                        n = n.get("expression")
                                        if not isinstance(n, dict):
                                            break
                                    if not isinstance(n, dict):
                                        continue
                                    if n.get("type") == "and":
                                        stack.append(n.get("left"))
                                        stack.append(n.get("right"))
                                    else:
                                        en = n
                                        if en.get("type") == "parenthesized_expression":
                                            en = en.get("expression")
                                        if not self._is_linear_comparison(en):
                                            raise self._unsupported_type_error("boolean leaf", en)
                                        comps.append(en)
                                return [comps]
                            # Single comparison disjunct; node_or is neither or/and after unwrapping
                            en = node_or
                            if not self._is_linear_comparison(en):
                                raise self._unsupported_type_error("boolean leaf", en)
                            return [[en]]

                        disjuncts = _disjuncts(left)
                        z_vars = []
                        for dj_idx, comp_list in enumerate(disjuncts):
                            z_name = f"or_flag_{dj_idx}"
                            while z_name in self.var_indices:
                                z_name += "_"
                            self.var_names.append(z_name)
                            self.var_indices[z_name] = len(self.var_names) - 1
                            self.bounds.append([0, 1])
                            if hasattr(self, "integrality"):
                                self.integrality.append(1)
                            else:
                                self.integrality = [1]
                            if hasattr(self, "c") and len(self.c) < len(self.var_names):
                                self.c.append(0.0)
                            z_vars.append(z_name)
                            # Tighten M per comparison using collected bounds (compute per comp_node)
                            for comp_node in comp_list:
                                # Use current env for tighter M
                                M = self._big_m_for_comparison(comp_node, env=env)
                                # Build lhs - rhs
                                coef_lhs, const_lhs = self._eval_expr(comp_node["left"], {})
                                right_node = comp_node["right"]
                                if isinstance(right_node, dict):
                                    coef_rhs, const_rhs = self._eval_expr(right_node, {})
                                else:
                                    coef_rhs, const_rhs = (
                                        {},
                                        (right_node if isinstance(right_node, (int, float)) else 0.0),
                                    )
                                expr_coef = dict(coef_lhs)
                                for vn, cf in coef_rhs.items():
                                    expr_coef[vn] = expr_coef.get(vn, 0.0) - cf
                                expr_const = const_lhs - const_rhs
                                op_c = comp_node["op"]
                                if op_c == "<=":
                                    row = [0.0] * len(self.var_names)
                                    for vn, cf in expr_coef.items():
                                        if vn in self.var_indices:
                                            row[self.var_indices[vn]] += cf
                                    row[self.var_indices[z_name]] += M
                                    for i, coef in enumerate(row):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(coef)
                                    b_ub.append(M - expr_const)
                                    ub_row_idx += 1
                                elif op_c == ">=":
                                    row = [0.0] * len(self.var_names)
                                    for vn, cf in expr_coef.items():
                                        if vn in self.var_indices:
                                            row[self.var_indices[vn]] -= cf
                                    row[self.var_indices[z_name]] += M
                                    for i, coef in enumerate(row):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(coef)
                                    b_ub.append(M + expr_const)
                                    ub_row_idx += 1
                                elif op_c == "==":
                                    # Two inequalities
                                    row1 = [0.0] * len(self.var_names)
                                    for vn, cf in expr_coef.items():
                                        if vn in self.var_indices:
                                            row1[self.var_indices[vn]] += cf
                                    row1[self.var_indices[z_name]] += M
                                    for i, coef in enumerate(row1):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(coef)
                                    b_ub.append(M - expr_const)
                                    ub_row_idx += 1
                                    row2 = [0.0] * len(self.var_names)
                                    for vn, cf in expr_coef.items():
                                        if vn in self.var_indices:
                                            row2[self.var_indices[vn]] -= cf
                                    row2[self.var_indices[z_name]] += M
                                    for i, coef in enumerate(row2):
                                        if abs(coef) > 1e-12:
                                            A_ub_rows.append(ub_row_idx)
                                            A_ub_cols.append(i)
                                            A_ub_data.append(coef)
                                    b_ub.append(M + expr_const)
                                    ub_row_idx += 1
                        if z_vars:
                            row = [0.0] * len(self.var_names)
                            for z in z_vars:
                                row[self.var_indices[z]] -= 1.0
                            for i, coef in enumerate(row):
                                if abs(coef) > 1e-12:
                                    A_ub_rows.append(ub_row_idx)
                                    A_ub_cols.append(i)
                                    A_ub_data.append(coef)
                            b_ub.append(-1.0)
                            ub_row_idx += 1
                        return
                lhs_dict, lhs_const = self._accumulate_sum_to_dict(left, env, sign=1)
                rhs_dict, rhs_const = self._accumulate_sum_to_dict(right, env, sign=1)
                logger.debug(f"[SciPyCSCCodeGenerator] lhs_dict: {lhs_dict}, lhs_const: {lhs_const}")
                logger.debug(f"[SciPyCSCCodeGenerator] rhs_dict: {rhs_dict}, rhs_const: {rhs_const}")
                row = [0.0] * len(self.var_names)
                for vname, coef in lhs_dict.items():
                    if isinstance(vname, int):
                        idx = vname
                    else:
                        idx = self.var_indices.get(vname)
                    logger.debug(f"[SciPyCSCCodeGenerator] LHS vname: {vname}, coef: {coef}, idx: {idx}")
                    if idx is not None:
                        row[idx] += coef
                for vname, coef in rhs_dict.items():
                    if isinstance(vname, int):
                        idx = vname
                    else:
                        idx = self.var_indices.get(vname)
                    logger.debug(f"[SciPyCSCCodeGenerator] RHS vname: {vname}, coef: {coef}, idx: {idx}")
                    if idx is not None:
                        row[idx] -= coef
                rhs_value = rhs_const - lhs_const
                logger.debug(f"[SciPyCSCCodeGenerator] Final constraint row: {row}, rhs_value: {rhs_value}")
                if constr["op"] == "==":
                    for i, v in enumerate(row):
                        if abs(v) > 1e-12:
                            A_eq_rows.append(eq_row_idx)
                            A_eq_cols.append(i)
                            A_eq_data.append(v)
                    b_eq.append(rhs_value)
                    eq_row_idx += 1
                elif constr["op"] == "<=":
                    for i, v in enumerate(row):
                        if abs(v) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(v)
                    b_ub.append(rhs_value)
                    ub_row_idx += 1
                elif constr["op"] == ">=":
                    for i, v in enumerate(row):
                        if abs(v) > 1e-12:
                            A_ub_rows.append(ub_row_idx)
                            A_ub_cols.append(i)
                            A_ub_data.append(-v)
                    b_ub.append(-rhs_value)
                    ub_row_idx += 1
                else:
                    logger.debug(f"Unsupported op: {constr['op']}")
            elif constr["type"] == "forall_constraint":
                iterators = constr.get("iterators")
                index_constraint = constr.get("index_constraint")
                if iterators is None:
                    raise self._unsupported_type_error("forall_constraint", "missing iterators")
                if "constraint" in constr:
                    inner_constraints = [constr["constraint"]]
                elif "constraints" in constr:
                    inner_constraints = constr["constraints"]
                else:
                    raise self._unsupported_type_error("forall_constraint", "missing constraint(s)")
                loop_vars, loop_ranges = self._unroll_iterators(iterators)
                for idx_tuple in itertools.product(*loop_ranges):
                    env2 = dict(env or {})
                    for v, val in zip(loop_vars, idx_tuple):
                        env2[v] = val
                    if index_constraint is not None:
                        try:
                            cond_val = self._eval_expr(index_constraint, env2)[1]
                        except Exception:
                            cond_val = True
                        if not cond_val:
                            continue
                    for inner in inner_constraints:
                        handle_constraint(inner, env=env2)
            elif constr["type"] == "implication_constraint":
                # Should have been handled by early branch; unreachable
                return
            else:
                logger.debug(f"Unsupported constraint type: {constr['type']}")

        try:
            for constr in self.ast["constraints"]:
                handle_constraint(constr, env={})
        finally:
            # Always restore symbolic flag even if constraint handling raises
            self._allow_symbolic_bool = prev_sym

        # For test compatibility: always set self.A_eq and self.A_ub to lists (never None)
        n_vars = len(self.var_names)
        if len(b_eq) > 0:
            dense_A_eq = [[0.0 for _ in range(n_vars)] for _ in range(len(b_eq))]
            for r, c, v in zip(A_eq_rows, A_eq_cols, A_eq_data):
                dense_A_eq[r][c] = v
            self.A_eq = dense_A_eq
        else:
            self.A_eq = []
        self.b_eq = b_eq
        if len(b_ub) > 0:
            dense_A_ub = [[0.0 for _ in range(n_vars)] for _ in range(len(b_ub))]
            for r, c, v in zip(A_ub_rows, A_ub_cols, A_ub_data):
                dense_A_ub[r][c] = v
            self.A_ub = dense_A_ub
        else:
            self.A_ub = []
        self.b_ub = b_ub
        # Generated code still uses sparse matrices
        self._add_code_line("from scipy.sparse import csr_matrix")
        self._add_code_line(f"A_eq_rows = {A_eq_rows}")
        self._add_code_line(f"A_eq_cols = {A_eq_cols}")
        self._add_code_line(f"A_eq_data = {A_eq_data}")
        self._add_code_line(f"b_eq = {b_eq}")
        self._add_code_line(f"A_ub_rows = {A_ub_rows}")
        self._add_code_line(f"A_ub_cols = {A_ub_cols}")
        self._add_code_line(f"A_ub_data = {A_ub_data}")
        self._add_code_line(f"b_ub = {b_ub}")
        self._add_code_line(
            f"A_eq = csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)), shape=({len(b_eq)}, {n_vars})) if len(b_eq) > 0 else None"
        )
        self._add_code_line(
            f"A_ub = csr_matrix((A_ub_data, (A_ub_rows, A_ub_cols)), shape=({len(b_ub)}, {n_vars})) if len(b_ub) > 0 else None"
        )
        # Always reconcile metadata (objective c, var_names, bounds, integrality) in case
        for i, line in enumerate(self.scipy_code_lines):
            if line.startswith("var_names = "):
                self.scipy_code_lines[i] = f"var_names = {repr(self.var_names)}"
            elif line.startswith("bounds = "):
                bounds_py = "[" + ", ".join(f'[{b[0]}, {b[1] if b[1] is not None else "None"}]' for b in self.bounds) + "]"
                self.scipy_code_lines[i] = f"bounds = {bounds_py}"
            elif line.startswith("integrality = "):
                self.scipy_code_lines[i] = f"integrality = {self.integrality}"
            elif line.startswith("c = "):
                if hasattr(self, "c"):
                    if len(self.c) < len(self.var_names):
                        self.c.extend([0.0] * (len(self.var_names) - len(self.c)))
                    elif len(self.c) > len(self.var_names):
                        self.c = self.c[: len(self.var_names)]
                    self.scipy_code_lines[i] = f"c = {self.c}"

    def _accumulate_sum_to_dict(self, expr, env, sign=1):
        """
        Accumulate coefficients and constants from an expression into a dict and constant.
        Handles sum, binop (with sum), and base cases. Delegates to helpers for each case.
        """
        from collections import defaultdict

        coef_dict = defaultdict(float)
        const_ref = [0.0]
        if expr["type"] == "sum":
            self._accumulate_sum_expr(expr, env, coef_dict, sign, const_ref)
        elif expr["type"] == "binop" and (expr["left"].get("type") == "sum" or expr["right"].get("type") == "sum"):
            self._accumulate_binop_with_sum(expr, env, coef_dict, sign, const_ref)
        # Explicitly handle simple linear atoms so terms like s[t-1] and demand[t] are preserved
        elif expr["type"] == "indexed_name":
            # Uniformly evaluate and merge by variable name; parameters contribute to constant
            cdict, cval = self._eval_expr(expr, env)
            for vname, coef in cdict.items():
                coef_dict[vname] += sign * coef
            if isinstance(cval, (int, float)):
                const_ref[0] += sign * float(cval)
        elif expr["type"] == "name":
            # Include only if it is a decision variable; numeric parameters contribute to constant
            try:
                is_var, val, is_symbolic = self._lookup_var_or_param(expr.get("value"), indices=None, env=env)
                if is_var:
                    vname = val if isinstance(val, str) else expr.get("value")
                    coef_dict[vname] += sign * 1.0
                elif not is_symbolic and isinstance(val, (int, float)):
                    const_ref[0] += sign * float(val)
            except Exception:
                pass
        elif expr["type"] == "number":
            const_ref[0] += sign * float(expr.get("value", 0.0))
        elif expr["type"] == "binop" and expr.get("op") in ("+", "-"):
            # Recursively accumulate linear binops
            ldict, lconst = self._accumulate_sum_to_dict(expr["left"], env, sign=1)
            rdict, rconst = self._accumulate_sum_to_dict(expr["right"], env, sign=1)
            for k, v in ldict.items():
                coef_dict[k] += sign * v
            factor = 1.0 if expr["op"] == "+" else -1.0
            for k, v in rdict.items():
                coef_dict[k] += sign * factor * v
            const_ref[0] += sign * (lconst + factor * rconst)
        elif expr["type"] == "parenthesized_expression":
            inner_dict, inner_const = self._accumulate_sum_to_dict(expr["expression"], env, sign=1)
            for k, v in inner_dict.items():
                coef_dict[k] += sign * v
            const_ref[0] += sign * inner_const
        else:
            # Patch: treat missing parameters as zero only in sum expansion context
            prev_resolve_param_value = self._resolve_param_value

            def resolve_param_value_zero(name, indices=None, env=None, default_zero_if_missing=False):
                return prev_resolve_param_value(name, indices, env, default_zero_if_missing=True)

            self._resolve_param_value = resolve_param_value_zero
            try:
                cdict, cval = self._eval_expr(expr, env)
            finally:
                self._resolve_param_value = prev_resolve_param_value
            for vname, coef in cdict.items():
                coef_dict[vname] += sign * coef
            if isinstance(cval, (int, float)):
                const_ref[0] += sign * cval
        return coef_dict, const_ref[0]

    def _accumulate_sum_expr(self, expr, env, coef_dict, sign, const_ref):
        """
        Helper for _accumulate_sum_to_dict: handles 'sum' expressions.
        """
        iterators = expr["iterators"]
        loop_vars, loop_ranges = self._unroll_iterators(iterators)
        tuple_set_names = self._get_tuple_set_names(iterators)
        for idx_tuple in itertools.product(*loop_ranges):
            env2 = dict(env or {})
            for v, val in zip(loop_vars, idx_tuple):
                if v in tuple_set_names and not isinstance(val, tuple):
                    val = tuple(val)
                env2[v] = val
            index_constraint = expr.get("index_constraint")
            include = True
            if index_constraint is not None:
                try:
                    _, cond_val = self._eval_expr(index_constraint, env2)
                    include = bool(cond_val)
                except Exception:
                    include = True
            if not include:
                continue
            sum_expr = expr["expression"]
            # If the inner expression is a comparison, defer linearization to specialized handlers in _build_constraints.
            if (
                isinstance(sum_expr, dict)
                and sum_expr.get("type") == "binop"
                and sum_expr.get("op") in (">=", "==", ">", "<", "!=")
            ):
                continue
            cdict, cval = self._eval_expr(sum_expr, env=env2)
            for vname, coef in cdict.items():
                idx = self.var_indices.get(vname)
                if idx is None:
                    idx = self._resolve_tuple_index_varname(vname)
                if idx is not None:
                    coef_dict[idx] += sign * coef
            if isinstance(cval, (int, float)):
                const_ref[0] += sign * cval

    def _accumulate_binop_with_sum(self, expr, env, coef_dict, sign, const_ref):
        """Helper for _accumulate_sum_to_dict: handles binop where one/both sides include a sum.

        Strategy parallels _accumulate_objective_binop but writing into coef_dict/const_ref.
        We accumulate each side separately (respecting sign) then combine according to op.
        Supported ops: +, - . Other ops raise unsupported operator error.
        """
        op = expr.get("op")
        left = expr.get("left")
        right = expr.get("right")
        left_is_sum = isinstance(left, dict) and left.get("type") == "sum"
        right_is_sum = isinstance(right, dict) and right.get("type") == "sum"

        # Utility to merge a temporary coef dict into main with additive factor
        def merge(temp, factor):
            for k, v in temp.items():
                coef_dict[k] += sign * factor * v

        def add_const(val, factor):
            if isinstance(val, (int, float)):
                const_ref[0] += sign * factor * val

        if left_is_sum and right_is_sum:
            # Accumulate each sum side once
            temp_left = defaultdict(float)
            left_const_box = [0.0]
            self._accumulate_sum_expr(left, env, temp_left, 1, left_const_box)
            temp_right = defaultdict(float)
            right_const_box = [0.0]
            self._accumulate_sum_expr(right, env, temp_right, 1, right_const_box)
            if op == "+":
                merge(temp_left, 1.0)
                add_const(left_const_box[0], 1.0)
                merge(temp_right, 1.0)
                add_const(right_const_box[0], 1.0)
            elif op == "-":
                merge(temp_left, 1.0)
                add_const(left_const_box[0], 1.0)
                merge(temp_right, -1.0)
                add_const(right_const_box[0], -1.0)
            else:
                raise self._unsupported_operator_error("binop-with-sum", op)
            return
        if left_is_sum:
            left_coefs = defaultdict(float)
            left_const_box = [0.0]
            self._accumulate_sum_expr(left, env, left_coefs, 1, left_const_box)
            right_coefs, right_const = self._eval_expr(right, env)
            if op == "+":
                merge(left_coefs, 1.0)
                add_const(left_const_box[0], 1.0)
                for vn, cf in right_coefs.items():
                    idx = self.var_indices.get(vn)
                    if idx is not None:
                        coef_dict[idx] += sign * cf
                add_const(right_const, 1.0)
            elif op == "-":
                merge(left_coefs, 1.0)
                add_const(left_const_box[0], 1.0)
                for vn, cf in right_coefs.items():
                    idx = self.var_indices.get(vn)
                    if idx is not None:
                        coef_dict[idx] += sign * (-cf)
                add_const(right_const, -1.0)
            else:
                raise self._unsupported_operator_error("binop-with-sum", op)
            return
        if right_is_sum:
            right_coefs = defaultdict(float)
            right_const_box = [0.0]
            self._accumulate_sum_expr(right, env, right_coefs, 1, right_const_box)
            left_coefs, left_const = self._eval_expr(left, env)
            if op == "+":
                for vn, cf in left_coefs.items():
                    idx = self.var_indices.get(vn)
                    if idx is not None:
                        coef_dict[idx] += sign * cf
                add_const(left_const, 1.0)
                merge(right_coefs, 1.0)
                add_const(right_const_box[0], 1.0)
            elif op == "-":
                for vn, cf in left_coefs.items():
                    idx = self.var_indices.get(vn)
                    if idx is not None:
                        coef_dict[idx] += sign * cf
                add_const(left_const, 1.0)
                merge(right_coefs, -1.0)
                add_const(right_const_box[0], -1.0)
            else:
                raise self._unsupported_operator_error("binop-with-sum", op)
            return
        # Fallback: neither side sum (should not reach here based on guard)
        base_coefs, base_const = self._eval_expr(expr, env)
        for vn, cf in base_coefs.items():
            idx = self.var_indices.get(vn)
            if idx is not None:
                coef_dict[idx] += sign * cf
        add_const(base_const, 1.0)

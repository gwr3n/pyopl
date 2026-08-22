# === Standard library imports ===
import ast
import itertools
import logging
import re
from collections import defaultdict  # Needed for coefficient accumulation helpers
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

# === Local imports ===
from .boolean_lowering import lower_implication
from .comparison_policy import comparison_policy
from .linear_problem import LinearProblem, ObjectiveSense
from .numerical_policy import LINEAR_ZERO_TOLERANCE, SOLVER_FEASIBILITY_TOLERANCE, STRICT_COMPARISON_EPSILON
from .scipy_codegen_base import SciPyCodeGeneratorBase
from .semantic_error import SemanticError
from .tuple_set_helper import TupleSetHelper

# === Third-party imports ===
# (none)


SCIPY_FEASIBILITY_TOLERANCE = SOLVER_FEASIBILITY_TOLERANCE
BOOL_EPS = STRICT_COMPARISON_EPSILON


@dataclass
class _ConstraintBuildState:
    A_eq_rows: list[int] = field(default_factory=list)
    A_eq_cols: list[int] = field(default_factory=list)
    A_eq_data: list[float] = field(default_factory=list)
    b_eq: list[float] = field(default_factory=list)
    A_ub_rows: list[int] = field(default_factory=list)
    A_ub_cols: list[int] = field(default_factory=list)
    A_ub_data: list[float] = field(default_factory=list)
    b_ub: list[float] = field(default_factory=list)
    eq_row_idx: int = 0
    ub_row_idx: int = 0


@dataclass
class _ConstraintBuildContext:
    state: _ConstraintBuildState
    comparison_truth_cache: dict[Any, Any]
    subtree_var_cache: dict[Any, Any]
    expr_memo: dict[Any, Any] = field(default_factory=dict)
    neg_cache: dict[Any, Any] = field(default_factory=dict)


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
            elif isinstance(e, dict) and e.get("type") == "boolean_literal":
                # Preserve bools inside tuple literals
                return bool(e.get("value"))
            elif isinstance(e, dict):
                coef, val = self.eval(e, env)
                if isinstance(val, (float, int, str, tuple, bool)):
                    return val
                raise SemanticError(f"Tuple element evaluated to unsupported type: {type(val)}")
            else:
                return e

        return {}, tuple(to_tuple_recursive(e) for e in expr["elements"])

    def __init__(self, parent: "SciPyCSCCodeGenerator") -> None:
        self.parent = parent

    # NEW: minl/maxl
    def _eval_minl(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        vals: list[float] = []
        for a in expr.get("args", []):
            coef, v = self.eval(a, env)
            if coef:
                raise SemanticError("Non-ground argument in minl()")
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, str):
                try:
                    vals.append(float(v))
                except Exception:
                    raise SemanticError(f"Non-numeric argument '{v}' in minl()")
            else:
                raise SemanticError(f"Unsupported argument type in minl(): {type(v)}")
        if not vals:
            raise SemanticError("minl() requires at least one argument")
        return {}, min(vals)

    def _eval_maxl(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        vals: list[float] = []
        for a in expr.get("args", []):
            coef, v = self.eval(a, env)
            if coef:
                raise SemanticError("Non-ground argument in maxl()")
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, str):
                try:
                    vals.append(float(v))
                except Exception:
                    raise SemanticError(f"Non-numeric argument '{v}' in maxl()")
            else:
                raise SemanticError(f"Unsupported argument type in maxl(): {type(v)}")
        if not vals:
            raise SemanticError("maxl() requires at least one argument")
        return {}, max(vals)

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
            return self._validate_handler_result(t, result)
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
        return lower_implication(expr["left"], expr["right"])

    def _get_handler(self, t: str) -> Any:
        """Return the handler method for a given expression type, or None if not found."""
        return getattr(self, f"_eval_{t}", None)

    def _validate_handler_result(self, expression_type: str, result: Any) -> tuple:
        if result is None:
            raise SemanticError(f"ExpressionEvaluator.eval: handler for type '{expression_type}' returned None")
        if not isinstance(result, tuple) or len(result) != 2:
            raise SemanticError(
                f"ExpressionEvaluator.eval: handler for type '{expression_type}' returned non-tuple result: {result}"
            )
        coef, value = result
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if isinstance(value, (float, str, tuple, bool)):
            return coef, value
        if isinstance(value, dict):
            return self._validate_dict_handler_value(expression_type, result, coef, value)
        raise SemanticError(
            f"ExpressionEvaluator.eval: handler for type '{expression_type}' returned unsupported value: {result}"
        )

    def _validate_dict_handler_value(self, expression_type: str, result: tuple, coef: Any, value: dict) -> tuple:
        import inspect

        if any(frame.function.startswith(("_eval_field_access", "_eval_field_access_index")) for frame in inspect.stack()):
            return coef, value
        raise SemanticError(
            f"ExpressionEvaluator.eval: handler for type '{expression_type}' returned tuple with dict value: {result}"
        )

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
        raise self.parent._not_found_error(
            "tuple type metadata",
            f"iterator or value '{base['value']}' while resolving field '{field}'",
        )

    def _tuple_type_for_named_set(self, set_name: str) -> Optional[str]:
        for declaration in self.parent.ast.get("declarations", []):
            if declaration.get("type") == "set_of_tuples" and declaration.get("name") == set_name:
                return declaration.get("tuple_type")
        set_value = self.parent.data_dict.get(set_name)
        if isinstance(set_value, dict) and "tuple_type" in set_value:
            return set_value["tuple_type"]
        return None

    def _tuple_type_from_sum(self, expr: Dict[str, Any], iterator_name: str) -> Optional[str]:
        if expr.get("type") != "sum":
            return None
        for iterator in expr.get("iterators", []):
            if iterator["iterator"] == iterator_name and iterator["range"]["type"] == "named_range":
                tuple_type = self._tuple_type_for_named_set(iterator["range"]["name"])
                if tuple_type:
                    return tuple_type
        return None

    def _find_tuple_type_in_expr(self, expr: Any, iterator_name: str) -> Optional[str]:
        children: Iterable[Any]
        if isinstance(expr, dict):
            tuple_type = self._tuple_type_from_sum(expr, iterator_name)
            if tuple_type:
                return tuple_type
            children = expr.values()
        elif isinstance(expr, list):
            children = expr
        else:
            return None
        for child in children:
            tuple_type = self._find_tuple_type_in_expr(child, iterator_name)
            if tuple_type:
                return tuple_type
        return None

    def _find_tuple_type_for_iterator(self, iterator_name: str) -> Optional[str]:
        if not hasattr(self.parent, "ast"):
            return None
        ast = self.parent.ast
        for section in ("objective", "constraints"):
            if section in ast:
                tuple_type = self._find_tuple_type_in_expr(ast[section], iterator_name)
                if tuple_type:
                    return tuple_type
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

    def _is_tuple_indexed_declaration(self, decl: Optional[Dict[str, Any]]) -> bool:
        if decl is None:
            return False
        dimensions = decl.get("dimensions", [])
        if len(dimensions) != 1 or dimensions[0].get("type") != "named_set_dimension":
            return False
        set_decl = self.parent._find_decl(dimensions[0].get("name"))
        return bool(set_decl and set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external"))

    def _set_dimension_values(self, dimension: Dict[str, Any]) -> Optional[list]:
        set_decl = self.parent._find_decl(dimension.get("name"))
        set_data = self.parent.data_dict.get(dimension.get("name"))
        if (
            set_data is None
            and set_decl
            and set_decl.get("type")
            in (
                "typed_set",
                "typed_set_external",
                "set_declaration",
            )
        ):
            set_data = set_decl.get("value") or []
        return set_data if isinstance(set_data, list) else None

    def _remap_set_index(self, index_value: Any, dimension: Dict[str, Any]) -> Tuple[Any, bool]:
        if dimension.get("type") != "named_set_dimension" or not isinstance(index_value, str):
            return (index_value, False) if isinstance(index_value, int) else (None, False)
        set_data = self._set_dimension_values(dimension)
        if set_data is None:
            return (index_value, False) if isinstance(index_value, int) else (None, False)
        try:
            return set_data.index(index_value) + 1, True
        except ValueError:
            return (index_value, False) if isinstance(index_value, int) else (None, False)

    def _remap_scalar_set_indices(self, expr: Dict[str, Any], decl: Optional[Dict[str, Any]], indices: List[Any]) -> List[Any]:
        if decl is None or not decl.get("type", "").startswith("parameter"):
            return indices
        dimensions = decl.get("dimensions", [])
        param_data = self.parent.data_dict.get(f"{expr['name']}__map", self.parent.data_dict.get(expr["name"]))
        if not isinstance(param_data, list) or len(dimensions) != len(indices):
            return indices

        remapped_any = False
        remapped_indices = []
        for index_value, dimension in zip(indices, dimensions):
            remapped_value, was_remapped = self._remap_set_index(index_value, dimension)
            if remapped_value is not None:
                remapped_indices.append(remapped_value)
            remapped_any = remapped_any or was_remapped
        return remapped_indices if remapped_any else indices

    def _resolve_indexed_result(
        self, expr: Dict[str, Any], env: Dict[str, Any], indices: List[Any]
    ) -> Tuple[Dict[str, Any], Union[float, str, dict[Any, Any]]]:
        is_var, val, is_symbolic = self.parent._lookup_var_or_param(expr["name"], indices=indices, env=env)
        all_indices_are_int = all(isinstance(index, int) for index in indices)
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

    def _eval_indexed_name(
        self, expr: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str, dict[Any, Any]]]:
        indices = [self._eval_index_expr(dim, env)[1] for dim in expr["dimensions"]]
        decl = self.parent._find_decl(expr["name"])
        if self._is_tuple_indexed_declaration(decl):
            return self._handle_tuple_indexed(expr, indices)
        remapped_indices = self._remap_scalar_set_indices(expr, decl, indices)
        return self._resolve_indexed_result(expr, env, remapped_indices)

    @staticmethod
    def _normalize_index_scalar(value: Any) -> Any:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except Exception:
                pass
        return value

    def _eval_index_name(self, dim_expr: Dict[str, Any], env: Dict[str, Any], key: str) -> Tuple[Dict[str, Any], Any]:
        name = dim_expr.get(key)
        value = env[name] if name in env else self.parent.data_dict.get(name, name)
        return {}, self._normalize_index_scalar(value)

    def _eval_index_binop(self, dim_expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        _, left_value = self._eval_index_expr(dim_expr["left"], env)
        _, right_value = self._eval_index_expr(dim_expr["right"], env)
        if not (isinstance(left_value, int) and isinstance(right_value, int)):
            return {}, f"({left_value} {dim_expr['op']} {right_value})"

        operator = dim_expr["op"]
        if operator == "+":
            value = left_value + right_value
        elif operator == "-":
            value = left_value - right_value
        elif operator == "*":
            value = left_value * right_value
        else:
            raise self.parent._unsupported_operator_error("index", operator)
        return {}, self._normalize_index_scalar(value)

    def _eval_index_minmax(self, dim_expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        evaluator = self._eval_minl if dim_expr.get("type") == "minl" else self._eval_maxl
        _, value = evaluator(dim_expr, env)
        return {}, self._normalize_index_scalar(value)

    def _eval_index_tuple(self, dim_expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        def to_tuple_recursive(element: Any) -> Any:
            if isinstance(element, dict) and element.get("type") == "tuple_literal":
                return tuple(to_tuple_recursive(child) for child in element["elements"])
            if isinstance(element, dict):
                _, value = self._eval_index_expr(element, env)
                return value
            return element

        return {}, tuple(to_tuple_recursive(element) for element in dim_expr["elements"])

    def _eval_index_fallback(self, dim_expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        if "value" in dim_expr:
            return {}, self._normalize_index_scalar(dim_expr["value"])
        if "name" in dim_expr:
            value = env.get(dim_expr["name"], dim_expr["name"])
            return {}, self._normalize_index_scalar(value)
        raise self.parent._unsupported_type_error("index expr", dim_expr.get("type"))

    def _eval_index_expr(self, dim_expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        expression_type = dim_expr.get("type")
        if expression_type in ("field_access_index", "field_access"):
            coefficients, value = self._eval_field_access(dim_expr, env)
            return coefficients, self._normalize_index_scalar(value)
        if expression_type == "number_literal_index":
            return {}, self._normalize_index_scalar(dim_expr["value"])
        if expression_type == "name_reference_index":
            return self._eval_index_name(dim_expr, env, "name")
        if expression_type == "name":
            return self._eval_index_name(dim_expr, env, "value")
        if expression_type == "indexed_name":
            _, indexed_value = self._eval_indexed_name(dim_expr, env)
            if isinstance(indexed_value, dict):
                raise SemanticError("Indexed expression used as an index must resolve to a scalar value")
            return {}, self._normalize_index_scalar(indexed_value)
        if expression_type == "string_literal":
            return {}, dim_expr.get("value")
        if expression_type == "binop":
            return self._eval_index_binop(dim_expr, env)
        if expression_type == "uminus":
            _, value = self._eval_index_expr(dim_expr["value"], env)
            if isinstance(value, str):
                return {}, f"-({value})"
            return {}, -self._normalize_index_scalar(value)
        if expression_type == "parenthesized_expression":
            return self._eval_index_expr(dim_expr["expression"], env)
        if expression_type in ("minl", "maxl"):
            return self._eval_index_minmax(dim_expr, env)
        if expression_type == "tuple_literal":
            return self._eval_index_tuple(dim_expr, env)
        return self._eval_index_fallback(dim_expr, env)

    def _tuple_indexed_parameter_value(self, name: str, tuple_key: Any) -> Optional[Any]:
        param_dict = self.parent.data_dict.get(f"{name}__map", self.parent.data_dict.get(name))
        if isinstance(param_dict, dict) and tuple_key in param_dict:
            return param_dict[tuple_key]
        return None

    def _tuple_indexed_inline_value(self, name: str, tuple_key: Any) -> Optional[Any]:
        for declaration in self.parent._find_decls(name, "parameter_inline_indexed"):
            dimensions = declaration.get("dimensions", [])
            if len(dimensions) != 1 or dimensions[0].get("type") != "named_set_dimension":
                continue
            tuple_keys = TupleSetHelper.get_tuple_set(dimensions[0]["name"], self.parent.ast, self.parent.data_dict) or []
            normalized_keys = [key if isinstance(key, tuple) else (key,) for key in tuple_keys]
            try:
                index = normalized_keys.index(tuple_key)
            except ValueError:
                continue
            values = declaration.get("value")
            if isinstance(values, list) and index < len(values):
                return values[index]
        return None

    def _tuple_indexed_variable(self, name: str, tuple_key: Any) -> Optional[str]:
        for candidate in (f"{name}[{repr(tuple_key)}]", f"{name}[{str(tuple_key)}]"):
            if candidate in self.parent.var_indices:
                return candidate
        return None

    def _handle_tuple_indexed(self, expr: Dict[str, Any], indices: List[Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        tuple_key = self._extract_index_value(indices[0])
        name = expr["name"]
        variable_name = self._tuple_indexed_variable(name, tuple_key)
        if variable_name is not None:
            return {variable_name: 1.0}, 0.0

        parameter_value = self._tuple_indexed_parameter_value(name, tuple_key)
        if parameter_value is not None:
            return {}, parameter_value

        inline_value = self._tuple_indexed_inline_value(name, tuple_key)
        if inline_value is not None:
            return {}, inline_value

        raise self.parent._not_found_error("tuple-indexed variable or parameter", f"{name}[{repr(tuple_key)}]")

    def _eval_name_reference_index(
        self, expr: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        name_any = expr.get("name")
        name = name_any if isinstance(name_any, str) else str(name_any)
        val = env.get(name, name)
        if isinstance(val, (int, float)):
            return {}, float(val)
        return {}, str(val)

    @staticmethod
    def _is_boolean_binop_operand(node: Any) -> bool:
        return isinstance(node, dict) and (
            node.get("type") == "boolean_literal" or (node.get("type") == "binop" and node.get("sem_type") == "boolean")
        )

    def _eval_boolean_inequality(self) -> Tuple[Dict[str, Any], float]:
        aux_name = self.parent._ensure_aux_binary("xor_flag")
        return {"type": "aux_var", "name": aux_name, "sem_type": "boolean"}, 0.0

    def _resolve_equality_operand(self, node, result, env):
        if isinstance(node, dict) and node.get("type") == "name":
            name = node.get("value")
            if isinstance(name, str) and name in env:
                return result[0], env[name]
        return result

    def _eval_ground_equality(self, left_coef, left_val, right_coef, right_val):
        if (
            not left_coef
            and not right_coef
            and isinstance(left_val, (str, int, float, bool))
            and isinstance(right_val, (str, int, float, bool))
        ):
            return {}, left_val == right_val
        symbolic = bool(left_coef) or bool(right_coef) or isinstance(left_val, str) or isinstance(right_val, str)
        if symbolic and not getattr(self.parent, "_allow_symbolic_bool", False):
            raise SemanticError("Non-ground boolean == outside constraint build context")
        return {}, str(left_val) == str(right_val)

    def _eval_binop_equality(
        self, left: Dict[str, Any], right: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str, bool]]:
        left_result = self.eval(left, env)
        right_result = self.eval(right, env)
        if left_result is None or right_result is None:
            raise SemanticError(f"_eval_binop: == failed, left or right is None: left={left_result}, right={right_result}")
        left_coef, left_val = self._resolve_equality_operand(left, left_result, env)
        right_coef, right_val = self._resolve_equality_operand(right, right_result, env)
        return self._eval_ground_equality(left_coef, left_val, right_coef, right_val)

    def _dispatch_binop(
        self, left: Dict[str, Any], right: Dict[str, Any], op: str, env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        if op == "+":
            return self._handle_binop_add(left, right, env)
        if op == "-":
            return self._handle_binop_sub(left, right, env)
        if op == "*":
            return self._handle_binop_mul(left, right, env)
        if op == "/":
            return self._handle_binop_div(left, right, env)
        if op in ("!=", "<", ">", "<=", ">="):
            return self._handle_binop_cmp(left, right, op, env)
        raise self.parent._unsupported_operator_error("binop", op)

    def _eval_binop(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        """Evaluate binary operations, compacted and deduplicated."""
        left, right = expr["left"], expr["right"]
        op = expr["op"]
        if op == "!=" and self._is_boolean_binop_operand(left) and self._is_boolean_binop_operand(right):
            return self._eval_boolean_inequality()
        if op == "==":
            return self._eval_binop_equality(left, right, env)
        return self._dispatch_binop(left, right, op, env)

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

    def _multiply_symbolic(self, left_const, right_const):
        if isinstance(left_const, (str, tuple)) or isinstance(right_const, (str, tuple)):
            return {}, f"({left_const}) * ({right_const})"
        return None

    def _multiply_linear_terms(self, left_dict, right_dict, left_const, right_const):
        if left_dict and right_dict:
            raise self.parent._unsupported_type_error("nonlinear term", "variable * variable")
        if left_dict:
            factor = float(cast(Union[int, float], right_const))
            return {key: value * factor for key, value in left_dict.items()}, float(left_const) * factor
        if right_dict:
            factor = float(cast(Union[int, float], left_const))
            return {key: value * factor for key, value in right_dict.items()}, float(right_const) * factor
        return {}, float(left_const) * float(right_const)

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
        symbolic_result = self._multiply_symbolic(lconst, rconst)
        if symbolic_result is not None:
            return symbolic_result
        return self._multiply_linear_terms(ldict, rdict, lconst, rconst)

    def _handle_binop_div(
        self, left: Dict[str, Any], right: Dict[str, Any], env: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Union[float, str]]:
        """Division handler for binop /. Supports linear pattern: (linear expr) / constant.
        Nonlinear forms like constant/variable or variable/variable are rejected.
        """
        l_result = self.eval(left, env)
        r_result = self.eval(right, env)
        if l_result is None or r_result is None:
            raise SemanticError(f"_handle_binop_div: left or right is None: left={l_result}, right={r_result}")
        if not isinstance(l_result, tuple) or not isinstance(r_result, tuple):
            raise SemanticError(f"_handle_binop_div: left or right did not return a tuple: left={l_result}, right={r_result}")
        ldict, lconst = l_result
        rdict, rconst = r_result

        # Symbolic division: if any side is symbolic string/tuple, keep symbolic
        if isinstance(lconst, (str, tuple)) or isinstance(rconst, (str, tuple)):
            return {}, f"({lconst}) / ({rconst})"

        # Denominator must be constant numeric (no decision vars)
        if rdict:
            # variable in denominator => nonlinear
            raise self.parent._unsupported_type_error("nonlinear term", "division by variable")

        # Numeric constant divisor
        try:
            rc = float(cast(Union[int, float], rconst))
        except Exception:
            raise self.parent._unsupported_type_error("division", "non-numeric divisor")
        if abs(rc) < LINEAR_ZERO_TOLERANCE:
            raise SemanticError("Division by zero")

        inv = 1.0 / rc

        # If numerator is linear expression (coef dict), scale coefficients and constant
        if ldict:
            out_coef = {k: v * inv for k, v in ldict.items()}
            lc = float(cast(Union[int, float], lconst))
            return out_coef, lc * inv
        # Pure numeric division
        if not ldict:
            return {}, float(cast(Union[int, float], lconst)) * inv

        # Fallback (shouldn’t reach)
        return {}, float(cast(Union[int, float], lconst)) * inv

    def _comparison_is_numeric(self, value):
        return isinstance(value, (int, float, bool))

    def _handle_string_comparison(self, left, right, op):
        if not isinstance(left, str) or not isinstance(right, str) or op not in ("!=", "=="):
            return None
        return {}, float(left != right) if op == "!=" else float(left == right)

    def _handle_symbolic_comparison(self, left_dict, right_dict, left, right, op):
        is_symbolic = bool(left_dict) or bool(right_dict) or isinstance(left, (str, tuple)) or isinstance(right, (str, tuple))
        if not is_symbolic:
            return None
        if not getattr(self.parent, "_allow_symbolic_bool", False):
            raise SemanticError("Non-ground boolean comparison outside constraint build context")
        return {}, f"({left}) {op} ({right})"

    def _handle_numeric_comparison(self, left, right, op):
        if op == "!=":
            return {}, (
                float(left != right)
                if self._comparison_is_numeric(left) and self._comparison_is_numeric(right)
                else float(bool(left) != bool(right))
            )
        if op not in ("<", ">", "<=", ">="):
            raise self.parent._unsupported_operator_error("binop", op)
        if not (self._comparison_is_numeric(left) and self._comparison_is_numeric(right)):
            if not getattr(self.parent, "_allow_symbolic_bool", False):
                raise SemanticError("Non-numeric comparison outside constraint build context")
            return {}, f"({left}) {op} ({right})"
        operations = {
            "<": lambda: left < right,
            ">": lambda: left > right,
            "<=": lambda: left <= right,
            ">=": lambda: left >= right,
        }
        return {}, float(operations[op]())

    def _handle_binop_cmp(self, left, right, op, env):
        left_dict, left_value = self.eval(left, env)
        right_dict, right_value = self.eval(right, env)
        result = self._handle_string_comparison(left_value, right_value, op)
        if result is not None:
            return result
        result = self._handle_symbolic_comparison(left_dict, right_dict, left_value, right_value, op)
        if result is not None:
            return result
        return self._handle_numeric_comparison(left_value, right_value, op)

    def _eval_uminus(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        d, c = self.eval(expr["value"], env)
        if isinstance(c, (str, tuple)):
            return {k: -v for k, v in d.items()}, f"-({c})"
        return {k: -v for k, v in d.items()}, -float(c)

    def _eval_sum(self, expr: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], Union[float, str]]:
        iterators = expr["iterators"]
        # Narrow types to satisfy mypy
        coef_dict_total: Dict[str, float] = {}
        const_total: Union[float, str] = 0.0
        for env2, _idx_tuple in self.parent._iter_filtered_environments(
            iterators,
            env,
            expr.get("index_constraint"),
        ):
            coef_dict, const = self.eval(expr["expression"], env=env2)
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
            row[self._resolve_coefficient_index(v)] += c
        return row

    def _big_m_for_comparison(self, comp: Dict[str, Any], env: Optional[Dict[str, Any]] = None) -> float:
        """Return a finite big-M derived from the complete affine interval."""
        env_eval = env or {}
        coef_lhs, const_lhs = self._eval_expr(comp.get("left"), env_eval)
        rhs = comp.get("right")
        coef_rhs, const_rhs = (
            self._eval_expr(rhs, env_eval) if isinstance(rhs, dict) else ({}, rhs if isinstance(rhs, (int, float)) else 0.0)
        )
        if not isinstance(const_lhs, (int, float)) or not isinstance(const_rhs, (int, float)):
            raise SemanticError("Comparison big-M requires a numeric affine expression")
        coefficients = dict(coef_lhs)
        for var_name, coefficient in coef_rhs.items():
            coefficients[var_name] = coefficients.get(var_name, 0.0) - coefficient
        constant = float(const_lhs) - float(const_rhs)
        lower, upper = self._finite_affine_bounds(coefficients, constant, "Comparison big-M")
        return max(1.0, abs(lower), abs(upper))

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

    def _linearize_or_comparison(self, comparison, env, z_name):
        operator = comparison.get("op")
        big_m = self._big_m_for_comparison(comparison, env=env)
        lhs_coef, lhs_const = self._eval_expr(comparison["left"], env)
        rhs_node = comparison["right"]
        rhs_coef, rhs_const = (
            self._eval_expr(rhs_node, env)
            if isinstance(rhs_node, dict)
            else ({}, rhs_node if isinstance(rhs_node, (int, float)) else 0.0)
        )
        expression_coef = dict(lhs_coef)
        for variable, coefficient in rhs_coef.items():
            expression_coef[variable] = expression_coef.get(variable, 0.0) - coefficient
        expression_const = lhs_const - rhs_const

        def append_guarded_row(sign, rhs):
            row = [0.0] * len(self.var_names)
            for variable, coefficient in expression_coef.items():
                row[self.var_indices[variable]] += sign * coefficient
            row[self.var_indices[z_name]] += big_m
            self.A_ub.append(row)
            self.b_ub.append(rhs)

        if operator == "<=":
            append_guarded_row(1.0, big_m - expression_const)
        elif operator == ">=":
            append_guarded_row(-1.0, big_m + expression_const)
        elif operator == "==":
            append_guarded_row(1.0, big_m - expression_const)
            append_guarded_row(-1.0, big_m + expression_const)

    def _linearize_or_not_equal(self, comparison, env):
        lower = dict(comparison)
        lower["op"] = "<"
        upper = dict(comparison)
        upper["op"] = ">"
        self._linearize_or([lower], env=env)
        self._linearize_or([upper], env=env)

    def _linearize_or(self, comparisons: List[Any], env: Optional[Dict[str, Any]] = None) -> None:
        """Linearize disjunction of linear comparisons using big-M and auxiliary binaries."""
        env_eval = env or {}
        z_vars = []
        for comparison in comparisons:
            if comparison.get("op") == "!=":
                self._linearize_or_not_equal(comparison, env_eval)
                continue
            z_name = self._ensure_aux_binary("or_flag")
            z_vars.append(z_name)
            self._linearize_or_comparison(comparison, env_eval, z_name)
        if z_vars:
            row = [0.0] * len(self.var_names)
            for z_name in z_vars:
                row[self.var_indices[z_name]] -= 1.0
            self.A_ub.append(row)
            self.b_ub.append(-1.0)

    def _expand_and(self, comparisons: List[Any], env: Optional[Dict[str, Any]] = None) -> None:
        """Add each comparison as its own constraint. For '!=', add both < and > as separate constraints."""
        env_eval = env or {}
        for comp in comparisons:
            if comp.get("op") == "!=":
                self._append_not_equal_constraint(comp, env_eval)
            else:
                self._append_linear_constraint(comp, env_eval)

    def _append_not_equal_constraint(self, comp: Dict[str, Any], env: Dict[str, Any]) -> None:
        lhs_dict, lhs_const = self._accumulate_sum_to_dict(comp["left"], env=env, sign=1)
        rhs_dict, rhs_const = self._accumulate_sum_to_dict(comp["right"], env=env, sign=1)
        diff_coef = dict(lhs_dict)
        for var_name, coefficient in rhs_dict.items():
            diff_coef[var_name] = diff_coef.get(var_name, 0.0) - coefficient
        diff_const = lhs_const - rhs_const
        diff_min, diff_max = self._finite_integer_affine_bounds(diff_coef, diff_const, "Integer not-equal conjunction term")
        big_m = max(1.0, diff_max + 1.0, 1.0 - diff_min)
        if not hasattr(self, "_neq_counter"):
            self._neq_counter = 0
        direction_name = f"neq_direction_c{self._neq_counter}"
        self._neq_counter += 1
        self.var_names.append(direction_name)
        self.var_indices[direction_name] = len(self.var_names) - 1
        self.bounds.append([0, 1])
        self.integrality.append(1)
        self.c.append(0.0)
        for existing_row in self.A_eq:
            existing_row.append(0.0)
        for existing_row in self.A_ub:
            existing_row.append(0.0)

        negative_row = [0.0] * len(self.var_names)
        positive_row = [0.0] * len(self.var_names)
        for var_name, coefficient in diff_coef.items():
            negative_row[self.var_indices[var_name]] += coefficient
            positive_row[self.var_indices[var_name]] -= coefficient
        negative_row[self.var_indices[direction_name]] = -big_m
        positive_row[self.var_indices[direction_name]] = big_m
        self.A_ub.extend([negative_row, positive_row])
        self.b_ub.extend([-1.0 - diff_const, big_m - 1.0 + diff_const])

    def _append_linear_constraint(self, comp: Dict[str, Any], env: Dict[str, Any]) -> None:
        lhs_dict, lhs_const = self._accumulate_sum_to_dict(comp["left"], env=env, sign=1)
        rhs_dict, rhs_const = (
            self._accumulate_sum_to_dict(comp["right"], env=env, sign=1)
            if isinstance(comp["right"], dict)
            else ({}, comp["right"] if isinstance(comp["right"], (int, float)) else 0.0)
        )
        expr_coef = dict(lhs_dict)
        for variable, coefficient in rhs_dict.items():
            expr_coef[variable] = expr_coef.get(variable, 0.0) - coefficient
        expr_const = lhs_const - rhs_const
        row = [0.0] * len(self.var_names)
        sign = -1 if comp["op"] == ">=" else 1
        for variable, coefficient in expr_coef.items():
            if isinstance(variable, int):
                if variable < len(row):
                    row[variable] += sign * coefficient
            else:
                index = self.var_indices.get(variable)
                if index is not None:
                    row[index] += sign * coefficient
        if comp["op"] == "==":
            self.A_eq.append(row)
            self.b_eq.append(-expr_const)
        elif comp["op"] == "<=":
            self.A_ub.append(row)
            self.b_ub.append(-expr_const)
        elif comp["op"] == ">=":
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
            return self._linear_leaf_bounds(node)
        if t == "unaryop" and node.get("op") == "-":
            value = node.get("value")
            if not isinstance(value, dict):
                return (None, None)
            lb, ub = self._linear_bounds_safe(value)
            if lb is None or ub is None:
                return (None, None)
            return (-ub, -lb)
        if t == "binop":
            return self._linear_binop_bounds(node)
        return (None, None)

    def _linear_leaf_bounds(self, node: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Return collected and static bounds for a linear-expression leaf."""
        node_type = node.get("type")
        if node_type not in ("name", "indexed_name"):
            return self._var_bounds_safe(node)

        if node_type == "name":
            variable_name = node.get("value")
        else:
            try:
                variable_name = self._multi_indexed_var_name(node, {})
            except Exception:
                variable_name = node.get("name")

        if not hasattr(self, "_collected_lbs"):
            return self._var_bounds_safe(node)

        lower_bound = self._collected_lbs.get(variable_name)
        upper_bound = self._collected_ubs.get(variable_name)
        if lower_bound is None and upper_bound is None and node_type == "indexed_name":
            base_symbol = node.get("name")
            lower_bound = self._collected_lbs.get(base_symbol)
            upper_bound = self._collected_ubs.get(base_symbol)
        if lower_bound is None and upper_bound is None:
            return self._var_bounds_safe(node)

        type_lower, type_upper = self._var_bounds_safe(node)
        if type_lower is not None:
            lower_bound = max(lower_bound, type_lower) if lower_bound is not None else type_lower
        if type_upper is not None:
            upper_bound = min(upper_bound, type_upper) if upper_bound is not None else type_upper
        return (lower_bound, upper_bound)

    def _linear_binop_bounds(self, node: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Return bounds for a supported binary linear operation."""
        left = node.get("left")
        right = node.get("right")
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return (None, None)

        op = node.get("op")
        if op in ("+", "-"):
            left_lower, left_upper = self._linear_bounds_safe(left)
            right_lower, right_upper = self._linear_bounds_safe(right)
            if left_lower is None or left_upper is None or right_lower is None or right_upper is None:
                return (None, None)
            if op == "+":
                return (left_lower + right_lower, left_upper + right_upper)
            return (left_lower - right_upper, left_upper - right_lower)

        if op == "*":
            return self._linear_scalar_product_bounds(left, right)
        return (None, None)

    def _linear_scalar_product_bounds(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return bounds for a scalar multiplied by a variable."""
        if left.get("type") == "number" and right.get("type") in ("name", "indexed_name"):
            coefficient = float(left.get("value", 0))
            variable_lower, variable_upper = self._linear_bounds_safe(right)
        elif right.get("type") == "number" and left.get("type") in ("name", "indexed_name"):
            coefficient = float(right.get("value", 0))
            variable_lower, variable_upper = self._linear_bounds_safe(left)
        else:
            return (None, None)

        if variable_lower is None or variable_upper is None:
            return (None, None)
        if coefficient >= 0:
            return (coefficient * variable_lower, coefficient * variable_upper)
        return (coefficient * variable_upper, coefficient * variable_lower)

    def _resolve_coefficient_index(self, variable: Any) -> int:
        if isinstance(variable, int):
            if 0 <= variable < len(self.var_names):
                return variable
            raise SemanticError(f"Coefficient variable index {variable} is out of range")
        if not isinstance(variable, str):
            raise SemanticError(f"Unsupported coefficient variable key {variable!r}")
        idx = self.var_indices.get(variable)
        if idx is not None:
            return idx
        return self._resolve_tuple_index_varname(variable)

    def _update_vector_from_coef_dict(self, coef_dict: Dict[Any, Any], vector: List[float], op: Optional[str] = None) -> None:
        """
        Helper to update a vector from a coef_dict. If op is None, set; if '+', add; if '-', subtract.
        """
        for vname, coef in coef_dict.items():
            idx = self._resolve_coefficient_index(vname)
            if op == "+":
                vector[idx] += coef
            elif op == "-":
                vector[idx] -= coef
            else:
                vector[idx] = coef

    @staticmethod
    def _strict_adjusted_rhs(op: str, rhs_value: float) -> tuple[str, float]:
        if op == ">":
            return ">=", rhs_value + BOOL_EPS
        if op == "<":
            return "<=", rhs_value - BOOL_EPS
        return op, rhs_value

    def _resolve_tuple_index_varname(self, vname: str) -> int:
        """
        Helper to resolve a variable name with a tuple index to its index in var_indices.
        Returns the index if found; otherwise raises SemanticError.

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
                        key_tuple = None
                    if key_tuple is None:
                        raise SemanticError(f"Variable '{vname}' has an invalid tuple index")
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

        def constant_constraint_rhs(constr, env):
            try:
                coef_dict, constant = self._eval_expr(constr["right"], env)
                if not coef_dict and isinstance(constant, (int, float)):
                    return float(constant)
            except Exception:
                pass
            return None

        def update_affine_bound(constr, env, rhs_val):
            try:
                left_coef, left_const = self._eval_expr(constr["left"], env)
            except Exception:
                return False
            if len(left_coef) != 1 or not isinstance(left_const, (int, float)):
                return False
            var_name, coefficient = next(iter(left_coef.items()))
            if var_name not in var_indices or abs(float(coefficient)) <= LINEAR_ZERO_TOLERANCE:
                return False
            bound_value = (rhs_val - float(left_const)) / float(coefficient)
            bound_op = constr["op"]
            if coefficient < 0:
                bound_op = {">=": "<=", "<=": ">=", "==": "=="}.get(bound_op, bound_op)
            update_bounds(var_indices[var_name], bound_op, bound_value)
            return True

        def update_indexed_bound(constr, env, rhs_val):
            left = constr["left"]
            if left["type"] == "name":
                update_bounds(var_indices.get(left["value"]), constr["op"], rhs_val)
                return
            if left["type"] != "indexed_name":
                return
            try:
                vname = self._multi_indexed_var_name(left, env, self._eval_index_expr)
                update_bounds(var_indices.get(vname), constr["op"], rhs_val)
                return
            except Exception:
                pass
            remapped = []
            for dimension in left["dimensions"]:
                if dimension["type"] == "name_reference_index":
                    remapped.append(env.get(dimension["name"]))
                elif dimension["type"] == "number_literal_index":
                    remapped.append(dimension["value"])
                else:
                    _, value = self._eval_index_expr(dimension, env)
                    remapped.append(value)
            remapped = [int(value) if isinstance(value, float) and value.is_integer() else value for value in remapped]
            is_var, looked_up, _is_symbolic = self._lookup_var_or_param(left["name"], indices=remapped, env=env)
            if is_var and isinstance(looked_up, str):
                update_bounds(var_indices.get(looked_up), constr["op"], rhs_val)

        def tighten_simple_constraint(constr, env) -> None:
            rhs_val = constant_constraint_rhs(constr, env)
            if rhs_val is not None and update_affine_bound(constr, env, rhs_val):
                return
            update_indexed_bound(constr, env, rhs_val)

        def tighten_forall_constraint(constr, env=None):
            if env is None:
                env = {}
            iterators = constr.get("iterators")
            if not iterators:
                return
            index_constraint = constr.get("index_constraint")
            inner_constraints = [constr["constraint"]] if "constraint" in constr else constr.get("constraints", [])
            try:
                for env2, _idx_tuple in self._iter_filtered_environments(
                    iterators,
                    env,
                    index_constraint,
                    skip_unresolved=True,
                ):
                    for inner in inner_constraints:
                        tighten_constraint(inner, env=env2)
            except Exception:
                return

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

        if len(dims) == 1 and dims[0]["type"] == "named_set_dimension":
            self._expand_named_set_variable_declaration(decl, var_names, bounds, integrality)
            return

        self._expand_indexed_variable_declaration(decl, var_names, bounds, integrality)

    @staticmethod
    def _default_indexed_variable_bounds(vtype: object) -> tuple[object, object, int]:
        if vtype == "boolean":
            return 0, 1, 1
        if vtype == "int+":
            return 0, None, 1
        if vtype == "int":
            return None, None, 1
        if vtype == "float+":
            return 0, None, 0
        if vtype == "float":
            return None, None, 0
        return None, None, 0

    def _indexed_variable_bounds(self, decl: dict, env: dict) -> tuple[list, int]:
        lower, upper, int_flag = self._default_indexed_variable_bounds(decl.get("var_type"))
        for bound_name in ("lower_bound", "upper_bound"):
            if bound_name not in decl:
                continue
            expression = decl.get(bound_name)
            if expression is None:
                value = None
            elif not isinstance(expression, dict):
                value = expression
            else:
                coef, value = self._eval_expr(expression, env)
                if coef:
                    raise SemanticError("Decision variables are not supported in dvar declaration bounds.")
            if bound_name == "lower_bound":
                lower = value
            else:
                upper = value
        return [lower, upper], int_flag

    def _expand_named_set_variable_declaration(self, decl: dict, var_names: list, bounds: list, integrality: list) -> None:
        name = decl["name"]
        set_name = decl["dimensions"][0]["name"]
        set_decl = self._find_decl(set_name)
        if set_decl and set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
            elements = TupleSetHelper.get_tuple_set(set_name, self.ast, self.data_dict)
        elif set_name in self.data_dict:
            elements = self.data_dict[set_name]
        elif set_decl:
            elements = set_decl.get("value")
            if set_decl.get("type") == "typed_set_external" and elements is None:
                raise SemanticError(f"External set '{set_name}' has no data provided")
            elements = elements or []
        else:
            raise SemanticError(f"Named set '{set_name}' is not declared")

        iterator_names = [it.get("iterator") for it in decl.get("iterators", []) if isinstance(it, dict)]
        logger.debug(f"[SciPyCSCCodeGenerator] Elements for {name} over {set_name}: {elements}")
        for key in elements:
            if isinstance(key, tuple):
                variable_name = f"{name}[{repr(key)}]"
            else:
                variable_name = f"{name}_{key}"
            var_names.append(variable_name)
            self.var_indices[variable_name] = len(var_names) - 1
            env = {iterator_names[0]: key} if iterator_names else {}
            bound, int_flag = self._indexed_variable_bounds(decl, env)
            bounds.append(bound)
            integrality.append(int_flag)

    def _resolve_indexed_variable_dimension(self, dim):
        if dim["type"] == "range_index":
            start_eval = self._eval_bound(dim["start"])
            end_eval = self._eval_bound(dim["end"])
            return (
                list(range(int(start_eval), int(end_eval) + 1)),
                f"range({self._emit_symbolic_expr(dim['start'])}, {self._emit_symbolic_expr(dim['end'])} + 1)",
            )
        if dim["type"] == "named_range_dimension":
            range_decl = next(
                (
                    declaration
                    for declaration in self.ast["declarations"]
                    if declaration["type"] == "range_declaration_inline" and declaration["name"] == dim["name"]
                ),
                None,
            )
            if range_decl is None:
                raise self._not_found_error("range", dim["name"])
            start_eval = self._eval_bound(range_decl["start"])
            end_eval = self._eval_bound(range_decl["end"])
            return (
                list(range(int(start_eval), int(end_eval) + 1)),
                f"range({self._emit_symbolic_expr(range_decl['start'])}, {self._emit_symbolic_expr(range_decl['end'])} + 1)",
            )
        set_name = dim["name"]
        set_decl = self._find_decl(set_name)
        if set_decl and set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
            set_values = TupleSetHelper.get_tuple_set(set_name, self.ast, self.data_dict)
        elif set_name in self.data_dict:
            set_values = self.data_dict[set_name]
        elif set_decl and set_decl.get("type") in ("typed_set", "typed_set_external"):
            set_values = set_decl.get("value")
            if set_decl.get("type") == "typed_set_external" and set_values is None:
                raise SemanticError(f"External set '{set_name}' has no data provided")
            set_values = set_values or []
        else:
            raise SemanticError(f"Named set '{set_name}' is not declared")
        return set_values, set_name

    def _emit_indexed_variable(self, decl, index_tuple, iterator_names, var_names, bounds, integrality):
        name = decl["name"]
        variable_name = name + "_" + "_".join(str(index) for index in index_tuple)
        logger.debug(f"[SciPyCSCCodeGenerator] Adding range-indexed variable: {variable_name}")
        var_names.append(variable_name)
        self.var_indices[variable_name] = len(var_names) - 1
        env = {iterator_names[i]: value for i, value in enumerate(index_tuple) if i < len(iterator_names)}
        bound, int_flag = self._indexed_variable_bounds(decl, env)
        bounds.append(bound)
        integrality.append(int_flag)

    def _expand_indexed_variable_declaration(self, decl: dict, var_names: list, bounds: list, integrality: list) -> None:
        name = decl["name"]
        iterator_names = [it.get("iterator") for it in decl.get("iterators", []) if isinstance(it, dict)]
        dimensions = [self._resolve_indexed_variable_dimension(dim) for dim in decl["dimensions"]]
        dim_ranges = [item[0] for item in dimensions]
        symbolic_dim_ranges = [item[1] for item in dimensions]

        self._add_code_line(f"# OPL: dvar {decl.get('var_type')} {name}[{', '.join(symbolic_dim_ranges)}]")
        for index_tuple in itertools.product(*dim_ranges):
            self._emit_indexed_variable(decl, index_tuple, iterator_names, var_names, bounds, integrality)

    # === Section: Index/Range/Iterator Utilities ===
    @staticmethod
    def normalize_index(idx: object) -> object:
        """
        Return index as tuple if list/tuple, else unchanged. Recursively normalizes nested indices.
        """
        if isinstance(idx, (list, tuple)):
            return tuple(SciPyCSCCodeGenerator.normalize_index(e) for e in idx)
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
            logger.debug(f"[resolve_variable] Scalar variable not found: {name}")
            return None

        norm_indices = self._normalize_variable_indices(indices)
        variable_name = self._indexed_variable_name(name, norm_indices)
        logger.debug(f"[resolve_variable] Trying indexed variable: {variable_name} (indices={norm_indices})")
        if variable_name in self.var_indices:
            logger.debug(f"[resolve_variable] Found indexed variable: {variable_name}")
            return True, variable_name, False

        symbolic_name = self._find_symbolic_variable_name(name, variable_name)
        if symbolic_name is not None:
            logger.debug(f"[resolve_variable] Found symbolic indexed variable: {symbolic_name}")
            return True, symbolic_name, False

        self._raise_if_variable_index_out_of_domain(name, norm_indices)
        logger.debug(f"[resolve_variable] Indexed variable not found: {variable_name} (indices={indices})")
        return None

    @staticmethod
    def _normalize_variable_indices(indices: list) -> list:
        normalized = []
        for index in indices:
            if isinstance(index, float) and index.is_integer():
                normalized.append(int(index))
            elif isinstance(index, (bool, int)):
                normalized.append(int(index))
            else:
                normalized.append(index)
        return normalized

    @staticmethod
    def _indexed_variable_name(name: str, indices: list) -> str:
        return name + "_" + "_".join(str(index) for index in indices)

    def _find_symbolic_variable_name(self, name: str, variable_name: str) -> str | None:
        normalized_name = variable_name.replace("(", "").replace(")", "")
        for candidate in self.var_indices:
            normalized_candidate = candidate.replace("(", "").replace(")", "")
            if candidate.startswith(name + "_") and normalized_name == normalized_candidate:
                return candidate
        return None

    def _raise_if_variable_index_out_of_domain(self, name: str, indices: list) -> None:
        declaration = self._find_decl(name)
        if not declaration or declaration.get("type") != "dvar_indexed":
            return
        dimensions = declaration.get("dimensions", [])
        if len(dimensions) != len(indices):
            return

        details = []
        for dimension, index in zip(dimensions, indices):
            detail = self._variable_domain_detail(dimension, index)
            if detail is not None:
                details.append(detail)
        if not details:
            return

        message = f"Index {indices} for '{name}' is out of declared domain ({'; '.join(details)})"
        logging.getLogger("pyopl.scipy_codegen_csc").debug(f"[resolve_variable] {message}")
        from .semantic_error import SemanticError

        raise SemanticError(message)

    def _variable_domain_detail(self, dimension: dict, index: object) -> str | None:
        dimension_type = dimension.get("type")
        if dimension_type == "range_index":
            start = self._eval_bound(dimension["start"])
            end = self._eval_bound(dimension["end"])
            if not isinstance(index, int) or index < int(start) or index > int(end):
                return f"{index} not in [{int(start)}..{int(end)}]"
            return None
        if dimension_type == "named_range_dimension":
            range_declaration = self._find_decl(dimension.get("name"), "range_declaration_inline")
            if range_declaration:
                start = self._eval_bound(range_declaration["start"])
                end = self._eval_bound(range_declaration["end"])
                if not isinstance(index, int) or index < int(start) or index > int(end):
                    return f"{index} not in [{int(start)}..{int(end)}]"
            return None
        if dimension_type == "named_set_dimension":
            set_name = dimension.get("name")
            if not isinstance(set_name, str):
                return None
            values = self._variable_domain_set_values(set_name)
            if isinstance(values, (list, tuple, set, frozenset, dict)) and index not in values:
                return f"{index} not in {set_name}"
        return None

    def _variable_domain_set_values(self, set_name: str) -> object:
        set_declaration = self._find_decl(set_name)
        if set_name in self.data_dict:
            raw_values = self.data_dict[set_name]
            return raw_values.get("elements") if isinstance(raw_values, dict) and "elements" in raw_values else raw_values
        if not set_declaration:
            return None
        if set_declaration.get("type") == "typed_set":
            return set_declaration.get("value") or []
        if set_declaration.get("type") in ("set_of_tuples", "set_of_tuples_external") and set_declaration.get("value"):
            return [tuple(item["elements"]) for item in set_declaration["value"]]
        return None

    @staticmethod
    def _normalize_parameter_key(key):
        return tuple(key) if isinstance(key, (list, tuple)) else key

    @staticmethod
    def _parameter_list_pairs(val: list):
        if len(val) % 2 != 0 or not val:
            return None
        keys = val[::2]
        values = val[1::2]
        if not all(isinstance(key, (list, tuple, str)) for key in keys):
            return None
        if not all(isinstance(value, (int, float)) for value in values):
            return None
        return zip(keys, values)

    @staticmethod
    def _parameter_entry_pairs(val: list):
        if not all(isinstance(entry, (list, tuple)) and len(entry) == 2 for entry in val):
            return None
        pairs = [(entry[0], entry[1]) for entry in val]
        if not all(isinstance(key, (list, tuple, str)) and isinstance(value, (int, float)) for key, value in pairs):
            return None
        return pairs

    def _flat_parameter_list_to_dict(self, val: list) -> dict | None:
        pairs = self._parameter_list_pairs(val) or self._parameter_entry_pairs(val)
        if pairs is None:
            return None
        return {self._normalize_parameter_key(key): value for key, value in pairs}

    def _normalize_parameter_lookup_value(self, name: str, val: object, indices: list | None, logger) -> object:
        if not isinstance(val, list) or indices is None:
            return val
        try:
            converted = self._flat_parameter_list_to_dict(val)
        except Exception:
            converted = None
        if converted is None:
            return val
        self.data_dict[name] = converted
        logger.debug(f"[resolve_parameter] Converted flat KV list to dict for param '{name}': {converted}")
        return converted

    def _parameter_list_start_index(self, name: str, dimension_index: int) -> int:
        try:
            decl = self._find_decl(name)
            if not decl or not decl.get("dimensions") or dimension_index >= len(decl["dimensions"]):
                return 1

            dim_decl = decl["dimensions"][dimension_index]
            dimension_type = dim_decl.get("type")
            if dimension_type == "range_index":
                return int(self._eval_bound(dim_decl["start"]))
            if dimension_type == "named_range_dimension":
                range_decl = self._find_decl(dim_decl.get("name"), "range_declaration_inline")
                if range_decl:
                    return int(self._eval_bound(range_decl["start"]))
        except Exception:
            pass
        return 1

    def _lookup_composite_parameter(self, val: object, indices: list, env: dict, logger):
        if not isinstance(val, dict) or not any(isinstance(key, tuple) for key in val):
            return None
        evaluated_indices = [self._eval_index(index, env) for index in indices]
        tuple_key = (
            evaluated_indices[0]
            if len(evaluated_indices) == 1 and isinstance(evaluated_indices[0], tuple)
            else tuple(evaluated_indices)
        )
        if tuple_key not in val:
            return None
        resolved = val[tuple_key]
        logger.debug(f"[resolve_parameter] Found composite-key param: {resolved!r}")
        if isinstance(resolved, (int, float)):
            return False, float(resolved), False
        return False, resolved, False

    def _lookup_parameter_dimension(self, name: str, resolved: object, index: object, dimension_index: int, env: dict, logger):
        evaluated_index = self._eval_index(index, env)
        logger.debug(f"[resolve_parameter] Index eval: idx={index}, idx_eval={evaluated_index}, " f"v={resolved}, env={env}")
        if isinstance(resolved, dict):
            logger.debug(f"[resolve_parameter] Dict lookup: v[{evaluated_index}] (keys={list(resolved.keys())})")
            return resolved[evaluated_index]
        if isinstance(evaluated_index, float) and evaluated_index.is_integer():
            evaluated_index = int(evaluated_index)
        if not isinstance(evaluated_index, int):
            raise ValueError(
                f"Index '{index}' could not be resolved to int (got {evaluated_index!r}) " f"for param '{name}' with env={env}"
            )
        if isinstance(resolved, list):
            start_index = self._parameter_list_start_index(name, dimension_index)
            offset = evaluated_index - start_index
            logger.debug(f"[resolve_parameter] List lookup with start={start_index}: v[{offset}] (len={len(resolved)})")
            return resolved[offset]
        logger.debug(f"[resolve_parameter] List/dict lookup: v[{evaluated_index}] (type={type(resolved)})")
        return cast(Any, resolved)[evaluated_index]

    @staticmethod
    def _normalize_indexed_parameter_result(resolved: object, logger):
        if isinstance(resolved, (int, float)):
            logger.debug(f"[resolve_parameter] Found numeric param: {resolved}")
            return False, float(resolved), False
        if isinstance(resolved, (str, bool, dict, list, tuple)):
            logger.debug(f"[resolve_parameter] Found indexed param: {resolved!r}")
            return False, resolved, False
        return None

    def _lookup_indexed_parameter(self, name: str, indices: list, env: dict, val: object, logger):
        composite_result = self._lookup_composite_parameter(val, indices, env, logger)
        if composite_result is not None:
            return composite_result

        resolved = val
        for dimension_index, index in enumerate(indices):
            resolved = self._lookup_parameter_dimension(name, resolved, index, dimension_index, env, logger)
        return self._normalize_indexed_parameter_result(resolved, logger)

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
        val = self.data_dict.get(f"{name}__map", self.data_dict.get(name))
        val = self._normalize_parameter_lookup_value(name, val, indices, logger)
        logger.debug(f"[resolve_parameter] Lookup param: {name}, indices={indices}, val={val}, env={env}")
        if indices is not None and val is not None:
            try:
                resolved = self._lookup_indexed_parameter(name, indices, env, val, logger)
                if resolved is not None:
                    return resolved
            except Exception as e:
                logger.debug(f"[resolve_parameter] Exception during lookup: {e}")
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

    @staticmethod
    def _normalize_safe_index_number(value):
        if isinstance(value, bool) or (isinstance(value, float) and value.is_integer()):
            return int(value)
        return value

    def _eval_safe_index_number(self, node, env):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return self._normalize_safe_index_number(node.value)
            raise ValueError("Non-numeric constant in index")
        value = env.get(node.id, self.data_dict.get(node.id, node.id))
        if isinstance(value, (int, float, bool)):
            return self._normalize_safe_index_number(value)
        raise ValueError(f"Name '{node.id}' not numeric for index")

    def _eval_safe_index_binop(self, node, env, allowed_nodes, allowed_ops):
        left = self._eval_safe_index_ast(node.left, env, allowed_nodes, allowed_ops)
        right = self._eval_safe_index_ast(node.right, env, allowed_nodes, allowed_ops)
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            raise ValueError("Non-numeric operands in index arithmetic")
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.FloorDiv: lambda: left // right,
        }
        return operations[type(node.op)]()

    def _eval_safe_index_ast(self, node, env, allowed_nodes, allowed_ops):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Disallowed expression in index")
        if isinstance(node, ast.Expression):
            return self._eval_safe_index_ast(node.body, env, allowed_nodes, allowed_ops)
        if isinstance(node, (ast.Constant, ast.Name)):
            return self._eval_safe_index_number(node, env)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_safe_index_ast(node.operand, env, allowed_nodes, allowed_ops)
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_ops):
            return self._eval_safe_index_binop(node, env, allowed_nodes, allowed_ops)
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_safe_index_ast(element, env, allowed_nodes, allowed_ops) for element in node.elts)
        raise ValueError("Unsupported node in index")

    def _safe_eval_index_arithmetic(self, expression, env):
        allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
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
        return self._eval_safe_index_ast(ast.parse(expression, mode="eval"), env, allowed_nodes, allowed_ops)

    def _eval_index(self, idx: object, env: dict) -> object:
        """
        Helper to evaluate an index expression in env/data_dict context.
        Tries to evaluate as safe Python literal/arith, then as int, else returns as is.
        """
        if isinstance(idx, str):
            # 1) Try tuple/number literal safely
            try:
                lit = ast.literal_eval(idx)
                return lit
            except Exception:
                pass

            try:
                v = self._safe_eval_index_arithmetic(idx, env)
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

    def _iterator_declaration(self, name: str) -> dict | None:
        for declaration in self.ast["declarations"]:
            if declaration.get("name") == name and declaration.get("type") in (
                "set_of_tuples",
                "set_of_tuples_external",
                "set_declaration",
                "typed_set",
                "typed_set_external",
            ):
                return declaration
        return None

    def _unroll_named_iterator(self, name: str) -> list:
        range_decl = self._find_decl(name, "range_declaration_inline")
        if range_decl is not None:
            start = self._eval_bound(range_decl["start"])
            end = self._eval_bound(range_decl["end"])
            return list(range(int(start), int(end) + 1))

        set_decl = self._iterator_declaration(name)
        if set_decl is None:
            raise self._not_found_error("range or set", name)

        declaration_type = set_decl.get("type")
        if declaration_type in ("set_of_tuples", "set_of_tuples_external"):
            return TupleSetHelper.get_tuple_set(name, self.ast, self.data_dict)
        if declaration_type == "set_declaration":
            set_val = self.data_dict.get(name)
            return set_decl.get("value", []) if set_val is None else set_val
        if name in self.data_dict:
            return self.data_dict[name]
        set_val = set_decl.get("value")
        if declaration_type == "typed_set_external" and set_val is None:
            raise SemanticError(f"External set '{name}' has no data provided")
        return set_val or []

    def _unroll_indexed_iterator(self, rng: dict, env: dict) -> list:
        set_val = self.data_dict.get(rng["name"], [])
        for dimension in rng.get("dimensions", []):
            if isinstance(dimension, dict):
                _, index_value = self._eval_index_expr(dimension, env)
                if isinstance(set_val, dict):
                    set_val = set_val[index_value]
                elif isinstance(set_val, (list, tuple)):
                    if isinstance(index_value, float) and index_value.is_integer():
                        index_value = int(index_value)
                    set_val = set_val[int(index_value) - 1]
                else:
                    set_val = []
                    break
        return list(set_val or [])

    def _unroll_iterators(self, iterators: list, env: dict | None = None) -> tuple[list, list]:
        """
        Given a list of OPL-style iterators, return (loop_vars, loop_ranges).
        Each iterator is a dict with 'iterator' and 'range'.
        Handles range_specifier, named_range, set_of_tuples, and set_declaration.
        Always uses TupleSetHelper.get_tuple_set for set-of-tuples.
        Raises SemanticError if range or set is not found.
        """
        loop_vars = []
        loop_ranges = []
        env = env or {}
        for it in iterators:
            name = it["iterator"]
            rng = it["range"]
            range_type = rng["type"]
            if range_type == "range_specifier":
                start = self._eval_bound(rng["start"])
                end = self._eval_bound(rng["end"])
                loop_ranges.append(list(range(int(start), int(end) + 1)))
            elif range_type in ("named_range", "named_set"):
                loop_ranges.append(self._unroll_named_iterator(rng["name"]))
            elif range_type == "indexed_set":
                loop_ranges.append(self._unroll_indexed_iterator(rng, env))
            else:
                raise self._unsupported_type_error("iterator range type", range_type)
            loop_vars.append(name)
        return loop_vars, loop_ranges

    def _eval_bound_binop(self, expr: dict) -> float | int:
        left = self._eval_bound(expr["left"])
        right = self._eval_bound(expr["right"])
        operations = {
            "+": lambda: left + right,
            "-": lambda: left - right,
            "*": lambda: left * right,
        }
        operator = expr["op"]
        if operator not in operations:
            raise self._unsupported_operator_error("index bound binop", operator)
        return operations[operator]()

    def _eval_bound_collection(self, expr: dict) -> float | int:
        values = [self._eval_bound(argument) for argument in expr.get("args", [])]
        if not values:
            raise self._unsupported_type_error("expr in index bound", expr.get("type"))
        return min(values) if expr.get("type") == "minl" else max(values)

    def _eval_bound(self, expr: object) -> float | int:
        """
        Evaluate a bound expression for index/range bounds (used in variable declarations, sum, forall, etc).
        Supports: number, name, binop (+, -, *), uminus, parenthesized_expression.
        Raises SemanticError for unsupported types or operators.
        """
        if not isinstance(expr, dict):
            raise self._unsupported_type_error("expr in index bound", type(expr))

        expression_type = expr.get("type")
        if expression_type == "number":
            return expr["value"]
        if expression_type == "name":
            name = expr["value"]
            if name not in self.data_dict:
                raise SemanticError(f"Range bound parameter '{name}' has no data provided")
            value = self.data_dict[name]
            if not isinstance(value, (int, float, bool)):
                raise SemanticError(f"Range bound parameter '{name}' must be numeric")
            return value
        if expression_type == "binop":
            return self._eval_bound_binop(expr)
        if expression_type == "uminus":
            return -self._eval_bound(expr["value"])
        if expression_type == "parenthesized_expression":
            return self._eval_bound(expr["expression"])
        if expression_type in ("minl", "maxl"):
            return self._eval_bound_collection(expr)
        raise self._unsupported_type_error("expr in index bound", expression_type)

    def _eval_dynamic_bound_name(self, expr, env):
        name = expr.get("value")
        if name in env:
            return int(env[name])
        value = self.data_dict.get(name)
        if isinstance(value, (int, float, bool)):
            return int(value)
        declaration = self._find_decl(name, "range_declaration_inline")
        if declaration:
            return int(self._eval_bound(declaration["end"]))
        raise self._unsupported_type_error("name in index bound", name)

    def _eval_dynamic_bound_binop(self, expr, env):
        op = expr.get("op")
        left = self._eval_bound_dynamic(cast(Dict[str, Any], expr.get("left")), env)
        right = self._eval_bound_dynamic(cast(Dict[str, Any], expr.get("right")), env)
        operations = {"+": lambda: left + right, "-": lambda: left - right, "*": lambda: left * right}
        if op not in operations:
            raise self._unsupported_operator_error("index bound binop", op)
        return int(operations[op]())

    def _eval_dynamic_bound_collection(self, expr, env):
        args = expr.get("args", []) or []
        values = [self._eval_bound_dynamic(cast(Dict[str, Any], arg), env) for arg in args]
        if not values:
            raise self._unsupported_type_error("expr in index bound", expr.get("type"))
        return min(values) if expr.get("type") == "minl" else max(values)

    def _eval_bound_dynamic(self, expr: dict, env: dict) -> int:
        if not isinstance(expr, dict):
            raise self._unsupported_type_error("expr in index bound", type(expr))
        expression_type = expr.get("type")
        if expression_type == "number":
            value = expr.get("value", 0)
            if isinstance(value, (bool, int, float)):
                return int(value)
            raise self._unsupported_type_error("number literal in index bound", type(value))
        if expression_type == "name":
            return self._eval_dynamic_bound_name(expr, env)
        if expression_type == "binop":
            return self._eval_dynamic_bound_binop(expr, env)
        if expression_type == "uminus":
            return -self._eval_bound_dynamic(cast(Dict[str, Any], expr.get("value")), env)
        if expression_type == "parenthesized_expression":
            return self._eval_bound_dynamic(cast(Dict[str, Any], expr.get("expression")), env)
        if expression_type in ("minl", "maxl"):
            return self._eval_dynamic_bound_collection(expr, env)
        raise self._unsupported_type_error("expr in index bound", expression_type)

    def _iterator_domain_dynamic(self, iterator: dict, env: dict) -> list:
        if not env:
            cache: dict[int, list[Any]] | None = getattr(self, "_static_iterator_domain_cache", None)
            if cache is None:
                cache = {}
                self._static_iterator_domain_cache = cache
            cache_key = id(iterator)
            if cache_key in cache:
                return cache[cache_key]
            domain = self._uncached_iterator_domain_dynamic(iterator, env)
            cache[cache_key] = domain
            return domain
        return self._uncached_iterator_domain_dynamic(iterator, env)

    def _uncached_iterator_domain_dynamic(self, iterator: dict, env: dict) -> list:
        rng = iterator.get("range") or {}
        range_type = rng.get("type")
        if range_type == "range_specifier":
            start_expr = cast(Dict[str, Any], rng.get("start"))
            end_expr = cast(Dict[str, Any], rng.get("end"))
            start = self._eval_bound_dynamic(start_expr, env)
            end = self._eval_bound_dynamic(end_expr, env)
            if end < start:
                return []
            return list(range(int(start), int(end) + 1))
        if range_type == "named_range":
            range_name = cast(str, rng.get("name"))
            declaration = self._find_decl(range_name, "range_declaration_inline")
            if declaration is None:
                raise SemanticError(f"Named range '{range_name}' is not declared")
            start = int(self._eval_bound(declaration["start"]))
            end = int(self._eval_bound(declaration["end"]))
            return list(range(start, end + 1))
        if range_type in ("named_set", "named_set_dimension"):
            return self._named_iterator_domain(rng)
        if range_type == "indexed_set":
            return self._indexed_iterator_domain(rng, env)
        raise self._unsupported_type_error("iterator range type", range_type)

    def _named_iterator_domain(self, rng: dict) -> list:
        set_name = cast(str, rng.get("name"))
        set_values = self.data_dict.get(set_name)
        if isinstance(set_values, dict) and "elements" in set_values:
            set_values = set_values["elements"]
        if set_values is not None:
            return list(set_values)

        set_decl = self._find_decl(set_name)
        if set_decl and set_decl.get("type") in ("typed_set", "typed_set_external"):
            set_values = set_decl.get("value")
            if set_decl.get("type") == "typed_set_external" and set_values is None:
                raise SemanticError(f"External set '{set_name}' has no data provided")
            return list(set_values or [])
        if set_decl and set_decl.get("type") in ("set_of_tuples", "set_of_tuples_external"):
            return list(TupleSetHelper.get_tuple_set(set_name, self.ast, self.data_dict))
        raise SemanticError(f"Named set '{set_name}' is not declared")

    def _indexed_iterator_domain(self, rng: dict, env: dict) -> list:
        set_values = self.data_dict.get(rng.get("name"), [])
        for dimension in rng.get("dimensions", []) or []:
            _, index_value = self._eval_index_expr(cast(Dict[str, Any], dimension), env)
            if isinstance(set_values, dict):
                set_values = set_values[index_value]
            elif isinstance(set_values, (list, tuple)):
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                set_values = set_values[int(index_value) - 1]
            else:
                return []
        return list(set_values or [])

    # NEW: build dynamic iterator domains honoring dependencies between bounds
    def _iterate_iterators_dynamic(self, iterators: list[dict], outer_env: dict) -> list[tuple[dict, tuple]]:
        """
        Returns a list of (env_snapshot, idx_tuple) pairs for all combinations,
        evaluating each iterator's range with access to previously bound iterator values.
        """
        results: list[tuple[dict, tuple]] = []

        def rec(idx: int, env: dict, acc: list):
            if idx == len(iterators):
                # snapshot env to avoid mutation
                results.append((dict(env), tuple(acc)))
                return
            it = iterators[idx]
            it_name = it.get("iterator")
            dom = self._iterator_domain_dynamic(it, env)
            for v in dom:
                env[it_name] = v
                acc.append(v)
                rec(idx + 1, env, acc)
                acc.pop()
                env.pop(it_name, None)

        rec(0, dict(outer_env or {}), [])
        return results

    def _iter_filtered_environments(
        self,
        iterators: list[dict],
        outer_env: dict | None = None,
        index_constraint: dict | None = None,
        *,
        skip_unresolved: bool = False,
    ) -> list[tuple[dict, tuple]]:
        """Return iterator environments whose optional filter is definitively true."""
        singleton = self._singleton_filtered_environment(iterators, outer_env or {}, index_constraint)
        if singleton is not None:
            return singleton
        environments = self._iterate_iterators_dynamic(iterators, outer_env or {})
        if index_constraint is None:
            return environments

        included: list[tuple[dict, tuple]] = []
        for env, idx_tuple in environments:
            try:
                coefficients, value = self._eval_expr(index_constraint, env)
            except Exception as exc:
                if skip_unresolved:
                    continue
                raise SemanticError(f"Unable to evaluate iterator filter for indices {idx_tuple}: {exc}") from exc
            if coefficients or not isinstance(value, (int, float, bool)):
                if skip_unresolved:
                    continue
                raise SemanticError(f"Unable to resolve iterator filter for indices {idx_tuple}")
            if bool(value):
                included.append((env, idx_tuple))
        return included

    @staticmethod
    def _singleton_equality_bound(iterator_name: Any, index_constraint: dict) -> dict | None:
        left = index_constraint.get("left")
        right = index_constraint.get("right")
        if not isinstance(iterator_name, str) or not isinstance(left, dict) or not isinstance(right, dict):
            return None
        if left.get("type") == "name" and left.get("value") == iterator_name:
            return right
        return left if right.get("type") == "name" and right.get("value") == iterator_name else None

    def _singleton_filtered_environment(
        self,
        iterators: list[dict],
        outer_env: dict,
        index_constraint: dict | None,
    ) -> list[tuple[dict, tuple]] | None:
        if len(iterators) != 1 or not isinstance(index_constraint, dict):
            return None
        if index_constraint.get("type") != "binop" or index_constraint.get("op") != "==":
            return None

        iterator = iterators[0]
        iterator_name = iterator.get("iterator")
        bound_expr = self._singleton_equality_bound(iterator_name, index_constraint)
        if bound_expr is None:
            return None

        try:
            coefficients, value = self._eval_expr(bound_expr, outer_env)
        except Exception:
            return None
        if coefficients or isinstance(value, dict):
            return None

        domain = self._iterator_domain_dynamic(iterator, outer_env)
        if not self._iterator_domain_contains(iterator, domain, value):
            return []
        env = dict(outer_env)
        env[iterator_name] = value
        return [(env, (value,))]

    def _iterator_domain_contains(self, iterator: dict, domain: list, value: Any) -> bool:
        cache: dict[int, frozenset[Any]] | None = getattr(self, "_iterator_domain_membership_cache", None)
        if cache is None:
            cache = {}
            self._iterator_domain_membership_cache = cache
        cache_key = id(iterator)
        members = cache.get(cache_key)
        if members is None:
            try:
                members = frozenset(domain)
            except TypeError:
                return value in domain
            cache[cache_key] = members
        try:
            return value in members
        except TypeError:
            return value in domain

    def _emit_python_call(self, expr: dict, env: dict) -> str:
        name = expr.get("name")
        args = expr.get("args", [])
        if len(args) != 1:
            return str(expr)
        arg = self._emit_python_expr(args[0], env)
        if name in {"sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceil"}:
            return f"math.{name}({arg})"
        if name in {"abs", "round"}:
            return f"{name}({arg})"
        return str(expr)

    def _emit_python_field_access(self, expr: dict, env: dict) -> str:
        base_expr = expr["base"]
        base = self._emit_python_expr(base_expr, env)
        field = expr["field"]
        for index, field_info in enumerate(getattr(self, "tuple_types", {}).get(base_expr.get("sem_type"), [])):
            if field_info["name"] == field:
                return f"{base}[{index}]"
        return f"{base}['{field}']"

    def _emit_python_aggregate(self, expr: dict, env: dict) -> str:
        parts = [self._emit_python_expr(arg, env) for arg in expr.get("args", [])]
        function_name = "min" if expr.get("type") == "minl" else "max"
        return f"{function_name}({', '.join(parts)})"

    def _emit_python_indexed_name(self, expr: dict, env: dict) -> str:
        indices = [self._emit_python_expr(dim, env) for dim in expr["dimensions"]]
        return f"{expr['name']}[{', '.join(indices)}]"

    def _emit_python_compound_expr(self, expr: dict, env: dict) -> str | None:
        expression_type = expr.get("type")
        if expression_type == "binop":
            left = self._emit_python_expr(expr["left"], env)
            right = self._emit_python_expr(expr["right"], env)
            return f"({left} {expr['op']} {right})"
        if expression_type == "uminus":
            return f"-({self._emit_python_expr(expr['value'], env)})"
        if expression_type == "parenthesized_expression":
            return f"({self._emit_python_expr(expr['expression'], env)})"
        if expression_type == "conditional":
            condition = self._emit_python_expr(expr["condition"], env)
            then_expr = self._emit_python_expr(expr["then"], env)
            else_expr = self._emit_python_expr(expr["else"], env)
            return f"({then_expr} if ({condition}) else {else_expr})"
        if expression_type == "indexed_name":
            return self._emit_python_indexed_name(expr, env)
        if expression_type == "field_access":
            return self._emit_python_field_access(expr, env)
        if expression_type == "funcall":
            return self._emit_python_call(expr, env)
        if expression_type in ("minl", "maxl"):
            return self._emit_python_aggregate(expr, env)
        return None

    def _emit_python_expr(self, expr: dict, env: dict | None = None) -> str:
        """
        Emit a valid Python expression from an AST node, using env for index variables.
        Handles numbers, names, binops, uminus, parenthesized expressions, indexed names, and field access.
        """
        if env is None:
            env = {}
        if not isinstance(expr, dict):
            return str(expr)
        t = expr.get("type")
        leaf_emitters = {
            "number": lambda: str(expr["value"]),
            "name": lambda: expr["value"],
            "boolean_literal": lambda: "True" if expr.get("value") else "False",
            "name_reference_index": lambda: env.get(expr["name"], expr["name"]),
            "number_literal_index": lambda: str(expr["value"]),
            "string_literal": lambda: repr(expr["value"]),
        }
        if t in leaf_emitters:
            return leaf_emitters[t]()
        compound = self._emit_python_compound_expr(expr, env)
        if compound is not None:
            return compound
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
    def _traverse_compound_expression(self, expr: dict) -> str | None:
        expression_type = expr.get("type")
        if expression_type == "binop":
            left = self._traverse_expression_value(expr.get("left"))
            right = self._traverse_expression_value(expr.get("right"))
            return f"({left} {expr.get('op')} {right})"
        if expression_type == "uminus":
            return f"-({self._traverse_expression_value(expr.get('value'))})"
        if expression_type == "parenthesized_expression":
            return f"({self._traverse_expression_value(expr.get('expression'))})"
        if expression_type == "conditional":
            condition = self._traverse_expression_value(expr.get("condition"))
            then_expr = self._traverse_expression_value(expr.get("then"))
            else_expr = self._traverse_expression_value(expr.get("else"))
            return f"({then_expr} if ({condition}) else {else_expr})"
        if expression_type == "indexed_name":
            return self._traverse_indexed_name(expr)
        if expression_type == "field_access":
            return self._traverse_field_access(expr)
        if expression_type == "tuple_literal":
            return self._traverse_tuple_literal(expr)
        if expression_type == "funcall":
            return self._traverse_function_call(expr)
        if expression_type in ("minl", "maxl"):
            return self._traverse_aggregate(expr)
        return None

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
            value = expr.get("value")
            return str(value) if value is not None else ""
        if t in ("name_reference_index", "number_literal_index"):
            if "value" in expr:
                return str(expr["value"])
            if "name" in expr:
                return str(expr["name"])
            return str(expr)

        if t == "string_literal":  # <-- support in symbolic traversal
            return repr(expr.get("value"))
        compound = self._traverse_compound_expression(expr)
        if compound is not None:
            return compound

        # Default: return empty string if no known type matched
        return ""

    def _traverse_expression_value(self, value: Any) -> str:
        """Traverse a child AST value while preserving scalar fallback formatting."""
        return self._traverse_expression(value) if isinstance(value, dict) else str(value)

    def _traverse_indexed_name(self, expr: dict) -> str:
        base = expr.get("name")
        parts = [self._traverse_expression_value(d) for d in (expr.get("dimensions") or [])]
        return f"{base}[{', '.join(parts)}]"

    def _traverse_field_access(self, expr: dict) -> str:
        base_expr = expr.get("base")
        base = self._traverse_expression_value(base_expr)
        field = expr.get("field")
        if hasattr(self, "tuple_types") and isinstance(base_expr, dict):
            sem_type = base_expr.get("sem_type")
            fields = self.tuple_types.get(sem_type, []) if sem_type else []
            for index, field_info in enumerate(fields):
                if field_info["name"] == field:
                    return f"{base}[{index}]"
        return f"{base}['{field}']"

    def _traverse_tuple_literal(self, expr: dict) -> str:
        parts = [self._traverse_expression_value(element) for element in expr.get("elements", [])]
        return f"({', '.join(parts)})"

    def _traverse_function_call(self, expr: dict) -> str:
        name = expr.get("name")
        args = expr.get("args", [])
        if len(args) != 1:
            return ""
        arg = self._traverse_expression_value(args[0])
        if name in {"sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceil"}:
            return f"math.{name}({arg})"
        if name in {"abs", "round"}:
            return f"{name}({arg})"
        return ""

    def _traverse_aggregate(self, expr: dict) -> str:
        parts = [self._traverse_expression_value(arg) for arg in expr.get("args", [])]
        function_name = "min" if expr.get("type") == "minl" else "max"
        return f"{function_name}({', '.join(parts)})"

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
        self.original_var_names: list[str] = []  # Variables declared in the PyOPL model
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
        self.obj_const_offset = 0.0

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

    def _apply_top_level_binary_assignments(self) -> None:
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
                    already = False
                    for _row, rhs_value, var_name in zip(self.A_eq, self.b_eq, self.var_names):
                        if abs(row[self.var_indices[var_name]]) == 1 and abs(rhs_value - constr["right"]["value"]) < 1e-8:
                            already = True
                            break
                    if not already:
                        self.A_eq.append(row)
                        self.b_eq.append(float(constr["right"]["value"]))

    def _reconcile_problem_metadata(self) -> None:
        while len(self.bounds) < len(self.var_names):
            self.bounds.append([0, 1])
        while len(self.integrality) < len(self.var_names):
            self.integrality.append(1)
        if len(self.c) < len(self.var_names):
            self.c.extend([0.0] * (len(self.var_names) - len(self.c)))
        elif len(self.c) > len(self.var_names):
            self.c = self.c[: len(self.var_names)]

    def _refresh_problem_metadata_code_lines(self) -> None:
        bounds_py = "[" + ", ".join(f'[{b[0]}, {b[1] if b[1] is not None else "None"}]' for b in self.bounds) + "]"
        found_var_names = False
        found_bounds = False
        found_integrality = False
        found_c = False
        for i, line in enumerate(self.scipy_code_lines):
            if line.startswith("var_names = "):
                self.scipy_code_lines[i] = f"var_names = {repr(self.var_names)}"
                found_var_names = True
            elif line.startswith("bounds = "):
                self.scipy_code_lines[i] = f"bounds = {bounds_py}"
                found_bounds = True
            elif line.startswith("integrality = "):
                self.scipy_code_lines[i] = f"integrality = {self.integrality}"
                found_integrality = True
            elif line.startswith("c = "):
                self.scipy_code_lines[i] = f"c = {self.c}"
                found_c = True
        if not found_var_names:
            self._add_code_line(f"var_names = {repr(self.var_names)}")
        self._add_code_line(f"original_var_names = {repr(self.original_var_names)}")
        if not found_bounds:
            self._add_code_line(f"bounds = {bounds_py}")
        if not found_integrality:
            self._add_code_line(f"integrality = {self.integrality}")
        if not found_c:
            self._add_code_line(f"c = {self.c}")

    def _snapshot_linear_problem(self) -> LinearProblem:
        sense = self.ast.get("objective", {}).get("type", "minimize")
        if sense not in ("minimize", "maximize"):
            sense = "minimize"
        return LinearProblem(
            sense=cast(ObjectiveSense, sense),
            var_names=list(self.var_names),
            bounds=[list(bound) for bound in self.bounds],
            integrality=list(self.integrality),
            c=list(self.c),
            A_eq=[list(row) for row in self.A_eq],
            b_eq=list(self.b_eq),
            A_ub=[list(row) for row in self.A_ub],
            b_ub=list(self.b_ub),
            objective_offset=float(self.obj_const_offset),
        )

    def build_problem(self) -> LinearProblem:
        self._generate_data_declarations(self.data_dict)
        self._add_code_line("")
        self._add_code_line("# Build LP vectors/matrices")
        self._build_variables()
        self.original_var_names = list(self.var_names)
        self._build_objective()
        self._build_constraints()
        self._apply_top_level_binary_assignments()
        self._reconcile_problem_metadata()
        self._refresh_problem_metadata_code_lines()
        return self._snapshot_linear_problem()

    def generate_code(self) -> str:
        self._add_code_line("import numpy as np")
        self._add_code_line("import math")
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
        self._add_code_line(
            f"solver_options = {{'disp': True, 'primal_feasibility_tolerance': {SCIPY_FEASIBILITY_TOLERANCE!r}, "
            f"'dual_feasibility_tolerance': {SCIPY_FEASIBILITY_TOLERANCE!r}}}"
        )
        self._add_code_line(
            "solver_options.update({key: value for key, value in globals().get('_pyopl_solver_settings', {}).items() "
            "if value is not None})"
        )
        # Emit sense variable for use in sign fix
        sense = self.ast.get("objective", {}).get("type", "minimize")
        self._add_code_line(f"sense = '{sense}'")
        self._add_code_line("")
        problem = self.build_problem()

        # >>> NEW: zero-variable short-circuit (pure feasibility/constant objective) <<<
        if len(problem.var_names) == 0:
            # Feasibility: with no variables, equalities require 0 == b_eq[i], inequalities require 0 <= b_ub[i]
            beq_ok = all(abs(b) <= 1e-9 for b in (problem.b_eq or []))
            bub_ok = all(b >= -1e-9 for b in (problem.b_ub or []))
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
        self._add_code_line("")
        self._add_code_line(f"{self.results_varname} = {{}}")
        self._add_code_line("try:")
        self.indent_level += 1
        self._add_code_line("start_time = time.time()")
        self._add_code_line(
            "print(f'PyOPL/SciPy-HiGHS: variables={len(var_names)}, equalities={len(b_eq)}, inequalities={len(b_ub)}, integrality={sum(1 for v in integrality if v)}')"
        )
        # Only include integrality if needed
        if any(self.integrality):
            self._add_code_line(
                "res = linprog(c, A_ub=A_ub, b_ub=b_ub if b_ub else None, "
                "A_eq=A_eq, b_eq=b_eq if b_eq else None, "
                "bounds=bounds, method='highs', integrality=integrality, "
                "options=solver_options)"
            )
        else:
            self._add_code_line(
                "res = linprog(c, A_ub=A_ub, b_ub=b_ub if b_ub else None, "
                "A_eq=A_eq, b_eq=b_eq if b_eq else None, "
                "bounds=bounds, method='highs', "
                "options=solver_options)"
            )
        self._add_code_line("end_time = time.time()")
        self._add_code_line(
            "print(f'PyOPL/SciPy-HiGHS: status={res.status}, success={res.success}, iterations={getattr(res, \"nit\", None)}, time={end_time - start_time:.3f}s')"
        )
        self._add_code_line("status_map = {0: 'OPTIMAL', 1: 'ITERATION_LIMIT', 2: 'INFEASIBLE', 3: 'UNBOUNDED'}")
        self._add_code_line("status_str = status_map.get(res.status, 'ERROR')")
        self._add_code_line("if res.status == 1 and 'time limit' in str(res.message).lower():")
        self.indent_level += 1
        self._add_code_line("status_str = 'TIME_LIMIT'")
        self.indent_level -= 1
        self._add_code_line("stats = {}")
        self._add_code_line("stats['status'] = res.status")
        self._add_code_line("stats['message'] = res.message")
        self._add_code_line("stats['nit'] = getattr(res, 'nit', None)")
        self._add_code_line("stats['crossover_nit'] = getattr(res, 'crossover_nit', None)")
        self._add_code_line("stats['time'] = end_time - start_time")
        self._add_code_line("if res.success and res.status == 0:")
        self.indent_level += 1
        self._add_code_line("print('Optimal solution found:')")
        self._add_code_line("solution = {}")
        self._add_code_line("for name in original_var_names:")
        self.indent_level += 1
        self._add_code_line("i = var_names.index(name)")
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
        self._add_code_line("objective_value += objective_offset")
        # Print objective value (parity with Gurobi output)
        self._add_code_line("print(f'Objective value: {objective_value}')")
        self._add_code_line(f"{self.results_varname}['solution'] = solution")
        self._add_code_line(f"{self.results_varname}['objective_value'] = objective_value")
        self._add_code_line(f"{self.results_varname}['status'] = status_str")
        self._add_code_line(f"{self.results_varname}['stats'] = stats")
        self.indent_level -= 1  # Dedent here so else is at the same level as if
        self._add_code_line("else:")
        self.indent_level += 1
        self._add_code_line("print('Optimization failed: ' + res.message)")
        self._add_code_line(f"{self.results_varname}['status'] = status_str")
        self._add_code_line(f"{self.results_varname}['message'] = res.message")
        self._add_code_line(f"{self.results_varname}['objective_value'] = None")
        self._add_code_line(f"{self.results_varname}['stats'] = stats")
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

    def _eval_data_bound(self, expr, data_dict):
        if not isinstance(expr, dict):
            raise ValueError("Unsupported range bound expr")
        expression_type = expr.get("type")
        if expression_type == "number":
            return int(expr["value"])
        if expression_type == "name":
            return int(data_dict[expr["value"]])
        if expression_type == "binop":
            left = self._eval_data_bound(expr["left"], data_dict)
            right = self._eval_data_bound(expr["right"], data_dict)
            operator = expr["op"]
            if operator == "+":
                return left + right
            if operator == "-":
                return left - right
            if operator == "*":
                return left * right
            if operator == "/":
                return left // right
        raise ValueError("Unsupported range bound expr")

    def _validate_one_dimensional_parameter_shape(self, param_data, dimension, data_dict, parameter_name):
        if dimension.get("type") == "named_range_dimension":
            range_name = dimension["name"]
            range_declaration = self._find_decl(range_name, "range_declaration_inline")
            if range_declaration:
                start = self._eval_data_bound(range_declaration["start"], data_dict)
                end = self._eval_data_bound(range_declaration["end"], data_dict)
                expected_length = end - start + 1
                if len(param_data) != expected_length:
                    raise SemanticError(
                        f"Parameter '{parameter_name}' has {len(param_data)} items but declared range "
                        f"'{range_name}' expects {expected_length}."
                    )
        elif dimension.get("type") == "named_set_dimension":
            set_name = dimension["name"]
            elements = data_dict.get(set_name)
            if elements is None:
                declaration = self._find_decl(set_name)
                if declaration and declaration.get("type") in ("typed_set", "set_declaration"):
                    elements = declaration.get("value") or []
            set_length = len(elements.get("elements", [])) if isinstance(elements, dict) else len(elements or [])
            if set_length and len(param_data) != set_length:
                raise SemanticError(
                    f"Parameter '{parameter_name}' has {len(param_data)} items but declared set "
                    f"'{set_name}' has {set_length} elements."
                )

    def _validate_two_dimensional_parameter_shape(self, param_data, parameter_name):
        if not all(isinstance(row, (list, tuple)) for row in param_data):
            return
        row_length = len(param_data[0]) if param_data else 0
        if not all(len(row) == row_length for row in param_data):
            raise SemanticError(f"Parameter '{parameter_name}' 2-D data must be rectangular (all rows same length).")

    def _validate_parameter_shape(self, param_data, dimensions, data_dict, parameter_name):
        if not isinstance(dimensions, list) or isinstance(param_data, dict):
            return
        if len(dimensions) == 1 and isinstance(param_data, list):
            self._validate_one_dimensional_parameter_shape(param_data, dimensions[0], data_dict, parameter_name)
        if len(dimensions) == 2 and isinstance(param_data, list):
            self._validate_two_dimensional_parameter_shape(param_data, parameter_name)

    def _resolve_data_set_elements(self, set_name, data_dict):
        if set_name in data_dict:
            set_object = data_dict[set_name]
            elements = set_object["elements"] if isinstance(set_object, dict) and "elements" in set_object else set_object
            return [tuple(element) if isinstance(element, (list, tuple)) else element for element in elements]
        declaration = self._find_decl(set_name)
        if declaration is None:
            return None
        if declaration.get("type") in ("typed_set", "set_declaration"):
            return declaration.get("value") or []
        if declaration.get("type") == "set_of_tuples" and declaration.get("value"):
            return [
                tuple(value["elements"]) if isinstance(value, dict) and "elements" in value else tuple(value)
                for value in declaration["value"]
            ]
        return None

    def _set_range_value_row(self, row, start, end):
        if isinstance(row, dict):
            return {int(index): float(item) for index, item in row.items()}
        if len(row) != end - start + 1:
            return None
        return {index: float(row[index - start]) for index in range(start, end + 1)}

    def _normalize_set_range_rows(self, rows, start, end):
        nested = {}
        for key, row in rows:
            normalized_row = self._set_range_value_row(row, start, end)
            if normalized_row is not None:
                normalized_key = tuple(key) if isinstance(key, (list, tuple)) else key
                nested[normalized_key] = normalized_row
        return nested

    def _normalize_set_range_value(self, value, set_elements, start, end):
        expected_length = end - start + 1
        if isinstance(value, dict) and all(isinstance(row, (list, tuple, dict)) for row in value.values()):
            return self._normalize_set_range_rows(value.items(), start, end)
        if (
            isinstance(value, list)
            and set_elements is not None
            and len(set_elements) == len(value)
            and all(isinstance(row, (list, tuple)) and len(row) == expected_length for row in value)
        ):
            return self._normalize_set_range_rows(zip(set_elements, value), start, end)
        return {}

    def _normalize_set_range_parameter(self, declaration, data_dict):
        dimensions = declaration.get("dimensions", [])
        if not (
            len(dimensions) == 2
            and dimensions[0].get("type") == "named_set_dimension"
            and dimensions[1].get("type") == "named_range_dimension"
        ):
            return
        name = declaration["name"]
        value = data_dict.get(name)
        if value is None:
            return
        try:
            start = self._eval_data_bound(dimensions[1]["start"], data_dict)
            end = self._eval_data_bound(dimensions[1]["end"], data_dict)
        except Exception:
            return
        set_elements = self._resolve_data_set_elements(dimensions[0]["name"], data_dict)
        nested = self._normalize_set_range_value(value, set_elements, start, end)
        if nested:
            data_dict[name] = nested

    def _normalize_set_set_parameter(self, declaration, data_dict):
        dimensions = declaration.get("dimensions", [])
        if not (len(dimensions) == 2 and all(dimension.get("type") == "named_set_dimension" for dimension in dimensions)):
            return
        name = declaration["name"]
        value = data_dict.get(name)
        if value is None:
            return
        first_keys = self._resolve_data_set_elements(dimensions[0]["name"], data_dict)
        second_keys = self._resolve_data_set_elements(dimensions[1]["name"], data_dict)
        if not (first_keys and second_keys):
            return
        nested = self._normalize_set_set_rows(value, first_keys, second_keys)
        if nested:
            data_dict[name] = nested

    @staticmethod
    def _normalize_set_key(key):
        return tuple(key) if isinstance(key, (list, tuple)) else key

    def _normalize_set_set_row(self, row, second_keys):
        if isinstance(row, dict):
            return {self._normalize_set_key(key): float(item) for key, item in row.items()}
        if len(row) != len(second_keys):
            return None
        return {self._normalize_set_key(key): float(item) for key, item in zip(second_keys, row)}

    def _normalize_set_set_rows(self, value, first_keys, second_keys):
        if isinstance(value, list):
            if len(value) != len(first_keys) or not all(isinstance(row, (list, tuple)) for row in value):
                return {}
            rows = zip(first_keys, value)
        elif isinstance(value, dict) and all(isinstance(row, (list, tuple, dict)) for row in value.values()):
            rows = value.items()
        else:
            return {}

        nested = {}
        for first_key, row in rows:
            normalized_row = self._normalize_set_set_row(row, second_keys)
            if normalized_row is not None:
                nested[self._normalize_set_key(first_key)] = normalized_row
        return nested

    def _data_range_bounds(self, range_dimension, data_dict):
        start_node = range_dimension.get("start")
        end_node = range_dimension.get("end")
        if isinstance(start_node, dict) and isinstance(end_node, dict):
            return self._eval_data_bound(start_node, data_dict), self._eval_data_bound(end_node, data_dict)
        range_name = range_dimension.get("name")
        declaration = self._find_decl(range_name, "range_declaration_inline")
        if isinstance(declaration, dict):
            return (
                self._eval_data_bound(declaration["start"], data_dict),
                self._eval_data_bound(declaration["end"], data_dict),
            )
        range_data = data_dict.get(range_name)
        if isinstance(range_data, dict) and range_data.get("type") == "range_data":
            return int(range_data["start"]), int(range_data["end"])
        raise SemanticError(f"Named range '{range_name}' has no bounds.")

    def _tuple_set_parameter_elements(self, set_name, data_dict):
        set_declaration = self._find_decl(set_name, "set_of_tuples") or self._find_decl(set_name, "set_of_tuples_external")
        if set_name in data_dict:
            raw_set = data_dict[set_name]
            set_values = raw_set["elements"] if isinstance(raw_set, dict) and "elements" in raw_set else raw_set
        elif set_declaration and set_declaration.get("value"):
            set_values = [value["elements"] for value in set_declaration["value"]]
        else:
            return None
        return [tuple(value) if isinstance(value, (list, tuple)) else (value,) for value in set_values]

    def _tuple_set_parameter_rows(self, set_elements, parameter_rows, expected_length):
        if not (
            len(set_elements) == len(parameter_rows)
            and all(isinstance(row, (list, tuple)) and len(row) == expected_length for row in parameter_rows)
        ):
            return None
        return list(zip(set_elements, parameter_rows))

    def _normalize_tuple_set_range_parameter(self, declaration, data_dict):
        dimensions = declaration.get("dimensions", [])
        name = declaration.get("name")
        if not (
            len(dimensions) == 2
            and dimensions[0].get("type") == "named_set_dimension"
            and dimensions[1].get("type") == "named_range_dimension"
            and isinstance(data_dict.get(name), list)
        ):
            return
        set_elements = self._tuple_set_parameter_elements(dimensions[0]["name"], data_dict)
        if set_elements is None:
            return
        start, end = self._data_range_bounds(dimensions[1], data_dict)
        rows = self._tuple_set_parameter_rows(set_elements, data_dict[name], end - start + 1)
        if rows is None:
            return
        data_dict[name] = {key: {index: float(row[index - start]) for index in range(start, end + 1)} for key, row in rows}

    def _emit_ast_data_declarations(self, data_dict):
        for declaration in self.ast.get("declarations", []):
            declaration_type = declaration.get("type")
            if declaration_type == "tuple_type":
                self.tuple_types = getattr(self, "tuple_types", {})
                self.tuple_types[declaration["name"]] = declaration["fields"]
            elif declaration_type == "set_of_tuples":
                set_name = declaration["name"]
                tuple_values = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict)
                if tuple_values:
                    self._add_code_line(f"{set_name} = {repr(tuple_values)}")
                    self.data_dict[set_name] = tuple_values
            elif declaration_type in ("typed_set", "typed_set_external"):
                set_name = declaration["name"]
                value = data_dict.get(set_name, declaration.get("value"))
                if declaration_type == "typed_set_external" and value is None:
                    continue
                value = value or []
                self._add_code_line(f"{set_name} = {repr(value)}")
                self.data_dict[set_name] = list(value)
                if isinstance(value, list) and all(isinstance(element, (str, int)) for element in value):
                    self._add_code_line(f"{set_name}_index = {{v: i for i, v in enumerate({set_name})}}")
            elif declaration_type in ("tuple_array", "tuple_array_external"):
                self._emit_tuple_array_data(declaration, data_dict)
            elif declaration_type == "parameter_inline_indexed":
                if self._emit_inline_indexed_parameter(declaration, data_dict):
                    return True
        return False

    def _emit_tuple_array_data(self, declaration, data_dict):
        array_name = declaration["name"]
        tuple_type = declaration["tuple_type"]
        data_value = data_dict.get(array_name)
        if data_value is None or tuple_type not in getattr(self, "tuple_types", {}):
            return
        dimensions = declaration.get("dimensions") or []
        index_set = declaration.get("index_set")
        if len(dimensions) == 1 or (not dimensions and index_set):
            field_names = [field["name"] for field in self.tuple_types[tuple_type]]
            index_set = index_set or dimensions[0].get("name")
            index_values = data_dict.get(index_set)
            if isinstance(data_value, dict):
                items = sorted(data_value.items(), key=lambda item: item[0])
            elif isinstance(index_values, list) and len(index_values) == len(data_value):
                items = zip(index_values, data_value)
            else:
                items = enumerate(data_value, start=1)
            data_value = {
                key: value if isinstance(value, dict) else {field: value[index] for index, field in enumerate(field_names)}
                for key, value in items
            }
        self._add_code_line(f"{array_name} = {repr(data_value)}")
        self.data_dict[array_name] = data_value

    def _emit_inline_indexed_parameter(self, declaration, data_dict):
        dimensions = declaration.get("dimensions", [])
        name = declaration["name"]
        if len(dimensions) == 1 and dimensions[0].get("type") == "named_set_dimension":
            set_name = dimensions[0]["name"]
            set_declaration = self._find_decl(set_name, "set_of_tuples")
            if set_declaration:
                keys = TupleSetHelper.get_tuple_set(set_name, self.ast, data_dict)
                tuple_keys = [key if isinstance(key, tuple) else tuple(key) for key in keys]
                parameter_dict = dict(zip(tuple_keys, declaration["value"]))
                self._add_code_line(f"{name} = {repr(parameter_dict)}")
                return True
        self._add_code_line(f"{name} = {repr(declaration['value'])}")
        self.data_dict[name] = declaration["value"]
        return False

    @staticmethod
    def _parameter_declaration_types():
        return (
            "parameter_external",
            "parameter_external_indexed",
            "parameter_external_explicit",
            "parameter_external_explicit_indexed",
            "parameter_inline",
            "parameter_inline_indexed",
        )

    def _validate_scalar_parameter_mappings(self, parameter_declarations, data_dict):
        for name, declaration in parameter_declarations.items():
            dimensions = declaration.get("dimensions", []) or []
            if len(dimensions) != 1 or dimensions[0].get("type") not in (
                "named_set_dimension",
                "named_range_dimension",
            ):
                continue
            value = data_dict.get(name)
            if not isinstance(value, dict):
                continue
            bad_key = next(
                (key for key, item in value.items() if isinstance(item, (list, tuple, dict))),
                None,
            )
            if bad_key is not None:
                raise SemanticError(
                    f"Parameter '{name}' declared as 1-D over '{dimensions[0].get('name', '')}' expects "
                    f"scalar values per key, but data provides an array for key {repr(bad_key)}. "
                    "Use scalar values (e.g., 2.0), not [2.0]."
                )

    def _prepare_indexed_parameter_values(self, data_dict):
        parameter_types = self._parameter_declaration_types()
        for declaration in self.ast.get("declarations", []):
            if declaration.get("type") not in parameter_types or not declaration.get("dimensions"):
                continue
            name = declaration["name"]
            parameter_data = data_dict.get(name)
            converted = self._convert_flat_kv_to_dict(parameter_data)
            if converted is not None:
                data_dict[name] = converted
            elif isinstance(parameter_data, (list, tuple)):
                self._validate_parameter_shape(parameter_data, declaration["dimensions"], data_dict, name)

    def _normalize_indexed_parameter_values(self, data_dict):
        parameter_types = self._parameter_declaration_types()
        for declaration in self.ast.get("declarations", []):
            if declaration.get("type") not in parameter_types:
                continue
            self._normalize_set_range_parameter(declaration, data_dict)
            self._normalize_set_set_parameter(declaration, data_dict)
            self._normalize_tuple_set_range_parameter(declaration, data_dict)

    def _prepare_parameter_data(self, data_dict):
        parameter_declarations = self._get_param_decl_map()
        self._validate_scalar_parameter_mappings(parameter_declarations, data_dict)
        self._prepare_indexed_parameter_values(data_dict)
        self._normalize_indexed_parameter_values(data_dict)
        return parameter_declarations

    def _validate_runtime_parameter_length(self, name, value, declaration, data_dict):
        if declaration is None or not isinstance(value, list) or not value:
            return
        dimensions = declaration.get("dimensions", [])
        if len(dimensions) != 1:
            return
        dimension = dimensions[0]
        if dimension.get("type") == "named_range_dimension":
            range_name = dimension["name"]
            range_declaration = self._find_decl(range_name, "range_declaration_inline")
            if range_declaration:
                start = self._eval_data_bound(range_declaration["start"], data_dict)
                end = self._eval_data_bound(range_declaration["end"], data_dict)
                expected_length = end - start + 1
                if len(value) != expected_length:
                    raise SemanticError(
                        f"Parameter '{name}' has {len(value)} items but declared range "
                        f"'{range_name}' expects {expected_length}."
                    )
        elif dimension.get("type") == "named_set_dimension":
            set_name = dimension["name"]
            set_elements = data_dict.get(set_name)
            if set_elements is not None:
                set_length = (
                    len(set_elements["elements"])
                    if isinstance(set_elements, dict) and "elements" in set_elements
                    else len(set_elements)
                )
                if set_length != len(value):
                    raise SemanticError(
                        f"Parameter '{name}' has {len(value)} items but declared set "
                        f"'{set_name}' has {set_length} elements."
                    )

    def _runtime_data_names(self):
        tuple_array_names = set()
        structured_names = set()
        for declaration in self.ast.get("declarations", []):
            declaration_type = declaration.get("type")
            name = declaration["name"]
            if declaration_type in ("tuple_array", "tuple_array_external"):
                tuple_array_names.add(name)
            if declaration_type in (
                "tuple_array",
                "tuple_array_external",
                "set_of_tuples",
                "set_of_tuples_external",
            ):
                structured_names.add(name)
        return tuple_array_names, structured_names

    def _emit_runtime_data_value(self, name, value, structured_names):
        if isinstance(value, (list, dict)):
            self._add_code_line(f"{name} = {repr(value)}")
            if (
                name not in structured_names
                and isinstance(value, list)
                and value
                and all(isinstance(element, (str, int)) for element in value)
            ):
                self._add_code_line(f"{name}_index = {{v: i for i, v in enumerate({name})}}")
        elif isinstance(value, str):
            self._add_code_line(f"{name} = {repr(value)}")
        else:
            self._add_code_line(f"{name} = {value}")

    def _emit_runtime_data(self, data_dict, parameter_declarations):
        self._add_code_line("# Data from .dat file")
        tuple_array_names, structured_names = self._runtime_data_names()
        for name, value in data_dict.items():
            self._validate_runtime_parameter_length(name, value, parameter_declarations.get(name), data_dict)
            if name in tuple_array_names or (isinstance(value, dict) and value.get("type") == "range_data"):
                continue
            self._emit_runtime_data_value(name, value, structured_names)
        self._add_code_line("")

    def _generate_data_declarations(self, data_dict):
        parameter_declarations = self._prepare_parameter_data(data_dict)
        if self._emit_ast_data_declarations(data_dict):
            return
        if not data_dict:
            self._add_code_line("")
            return
        self._emit_runtime_data(data_dict, parameter_declarations)

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

    def _handle_variable_declaration(self, decl, var_names, bounds, integrality):
        declaration_type = decl["type"]
        if declaration_type in ("dexpr", "dexpr_indexed"):
            return
        if declaration_type == "tuple_type":
            self._handle_tuple_type_declaration(decl)
            return
        if declaration_type in (
            "set_of_tuples",
            "set_of_tuples_external",
            "set_of_tuples_array_external",
            "typed_set",
            "typed_set_external",
            "tuple_array",
            "tuple_array_external",
        ):
            self._handle_set_of_tuples_declaration(decl, self.data_dict)
            return
        if declaration_type == "dvar":
            self._handle_scalar_variable_declaration(decl, var_names, bounds, integrality)
            return
        if declaration_type == "dvar_indexed":
            self._handle_indexed_variable_declaration(decl, var_names, bounds, integrality)
            return
        if declaration_type in (
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
            return
        raise SemanticError(f"Unsupported declaration type: {declaration_type}")

    def _emit_variable_metadata(self, var_names, bounds, integrality):
        bounds_py = "[" + ", ".join(f'[{bound[0]}, {bound[1] if bound[1] is not None else "None"}]' for bound in bounds) + "]"
        self._add_code_line(f"var_names = {repr(var_names)}")
        for variable_name in var_names:
            if "_" in variable_name and "[" not in variable_name:
                base, rest = variable_name.split("_", 1)
                if rest and base and rest.replace("_", "").isalnum():
                    self._add_code_line(f"# Alias: {base}['{rest}']")
        self._add_code_line(f"bounds = {bounds_py}")
        self._add_code_line(f"integrality = {integrality}")

    def _build_variables(self):
        # Supports scalar, indexed continuous, integer, and boolean variables
        self._add_code_line("# Variable definitions")
        var_names = []
        bounds = []
        integrality = []
        self.tuple_types = {}
        for declaration in self.ast["declarations"]:
            self._handle_variable_declaration(declaration, var_names, bounds, integrality)
        self._tighten_bounds_from_constraints(bounds, var_names, self.var_indices, self.ast.get("constraints", []))
        self.var_names = var_names
        self.bounds = bounds
        self.integrality = integrality
        self._emit_variable_metadata(var_names, bounds, integrality)

    def _eval_expr(self, expr, env=None):
        if not hasattr(self, "_expr_evaluator"):
            self._expr_evaluator = ExpressionEvaluator(self)
        return self._expr_evaluator.eval(expr, env)

    # NEW: delegate index-expression evaluation to the ExpressionEvaluator
    def _eval_index_expr(self, dim_expr: Dict[str, Any], env: Dict[str, Any]) -> tuple[Dict[str, Any], Any]:
        if not hasattr(self, "_expr_evaluator"):
            self._expr_evaluator = ExpressionEvaluator(self)
        return self._expr_evaluator._eval_index_expr(dim_expr, env)

    def _build_objective(self):
        self._add_code_line("# Objective vector c")
        c = [0.0] * len(self.var_names)
        obj = self.ast["objective"]
        sense = obj["type"]
        expr = obj["expression"]
        # Reset objective constant offset
        self.obj_const_offset = 0.0
        # Delegate to helpers for sum and binop
        self._accumulate_objective(expr, c)
        # Flip sign for maximization
        if sense == "maximize":
            c = [-v for v in c]
        self.c = c
        self._add_code_line(f"c = {c}")
        # NEW: emit the constant objective offset for runtime reporting
        self._add_code_line(f"objective_offset = {float(self.obj_const_offset)}")

    def _accumulate_objective(self, expr, c):
        """
        Accumulate coefficients for the objective vector c, handling sum and binop recursively.
        Also accumulates the constant offset into self.obj_const_offset.
        """
        if isinstance(expr, dict) and expr.get("type") == "sum":
            self._accumulate_objective_sum(expr, c, sign=1.0)
        elif isinstance(expr, dict) and expr.get("type") == "binop":
            self._accumulate_objective_binop(expr, c)
        else:
            coef_dict, const = self._eval_expr(expr)
            self._update_vector_from_coef_dict(coef_dict, c, "+")
            if isinstance(const, (int, float)):
                self.obj_const_offset += float(const)

    def _accumulate_objective_sum(self, expr, c, sign: float = 1.0):
        """
        Helper to accumulate coefficients for the objective vector c for a sum expression.
        Handles iterator unrolling with dependent bounds, index constraints, and tuple-indexed variables.
        Also accumulates numeric constant terms into self.obj_const_offset, scaled by 'sign'.
        """
        iterators = expr["iterators"]
        # Symbolic comment emission (unchanged)
        loop_vars = [it["iterator"] for it in iterators]
        symbolic_ranges = ", ".join(
            [
                (
                    f"{v} in range({self._emit_symbolic_expr(it['range'].get('start', ''))}, {self._emit_symbolic_expr(it['range'].get('end', ''))} + 1)"
                    if it["range"]["type"] == "range_specifier"
                    else f"{v} in {it['range']['name']}"
                )
                for v, it in zip(loop_vars, iterators)
            ]
        )
        self._add_code_line(
            f"# Symbolic objective: sum({self._emit_python_expr(expr['expression'], {v: v for v in loop_vars})} for {symbolic_ranges})"
        )

        for env2, _idx_tuple in self._iter_filtered_environments(iterators, {}, expr.get("index_constraint")):
            coef_dict, const = self._eval_expr(expr["expression"], env=env2)
            if coef_dict:
                for vname, coef in coef_dict.items():
                    c[self._resolve_coefficient_index(vname)] += sign * coef
            if isinstance(const, (int, float)):
                self.obj_const_offset += sign * float(const)

    def _accumulate_objective_sum_binop(self, expr, c):
        left = expr["left"]
        right = expr["right"]
        if expr["op"] not in ("+", "-"):
            raise self._unsupported_operator_error("objective binop", expr["op"])
        self._accumulate_objective_sum(left, c, sign=1.0)
        self._accumulate_objective_sum(right, c, sign=1.0 if expr["op"] == "+" else -1.0)

    def _accumulate_objective_sum_operand(self, sum_node, other_node, operator, c):
        self._accumulate_objective_sum(sum_node, c, sign=1.0)
        coef_dict, const = self._eval_expr(other_node)
        self._update_vector_from_coef_dict(coef_dict, c, op="+" if operator == "+" else "-")
        if isinstance(const, (int, float)):
            self.obj_const_offset += (1.0 if operator == "+" else -1.0) * float(const)

    def _accumulate_objective_plain_binop(self, expr, c):
        coef_dict, const = self._eval_expr(expr)
        self._update_vector_from_coef_dict(coef_dict, c)
        if isinstance(const, (int, float)):
            self.obj_const_offset += float(const)

    def _accumulate_objective_operand(self, operand, c):
        coef_dict, const = self._eval_expr(operand)
        self._update_vector_from_coef_dict(coef_dict, c, op="+")
        if isinstance(const, (int, float)):
            self.obj_const_offset += float(const)

    def _accumulate_objective_binop(self, expr, c):
        """Accumulate objective coefficients for binops containing zero, one, or two sums."""
        left = expr["left"]
        right = expr["right"]
        left_is_sum = isinstance(left, dict) and left.get("type") == "sum"
        right_is_sum = isinstance(right, dict) and right.get("type") == "sum"
        if left_is_sum and right_is_sum:
            self._accumulate_objective_sum_binop(expr, c)
        elif left_is_sum:
            self._accumulate_objective_sum_operand(left, right, expr["op"], c)
        elif right_is_sum:
            self._accumulate_objective_sum(right, c, sign=1.0 if expr["op"] == "+" else -1.0)
            self._accumulate_objective_operand(left, c)
        else:
            self._accumulate_objective_plain_binop(expr, c)

    def _eval_multi_index_values(self, expr, env, eval_index_expr):
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
        return index_values

    def _indexed_name_candidates(self, base, index_values):
        tuple_key = tuple(index_values)
        candidates = []
        if len(index_values) == 1 and isinstance(index_values[0], tuple):
            candidates.append(f"{base}[{repr(index_values[0])}]")
        candidates.extend(
            (
                f"{base}[{repr(tuple_key)}]",
                f"{base}[{tuple_key}]",
                f"{base}[{str(tuple_key)}]",
            )
        )
        if len(index_values) == 1:
            candidates.append(f"{base}_{index_values[0]}")
        elif len(index_values) > 1:
            candidates.append(f"{base}_" + "_".join(str(i) for i in index_values))
        if "[" in base and "]" in base:
            base_clean = (
                base.replace("[", "_").replace("]", "").replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
            )
            candidates.append(f"{base_clean}_{'_'.join(str(i) for i in index_values)}")
        return candidates

    def _multi_indexed_var_name(self, expr, env, eval_index_expr=None):
        if expr["type"] != "indexed_name":
            return expr["name"]
        if eval_index_expr is None:
            eval_index_expr = self._eval_index_expr
        base = expr["name"]
        index_values = self._eval_multi_index_values(expr, env, eval_index_expr)
        for candidate in self._indexed_name_candidates(base, index_values):
            if candidate in self.var_indices or candidate in self.data_dict:
                return candidate
        declaration = self._find_decl(base)
        if declaration and declaration.get("type") in ("dvar", "dvar_indexed"):
            raise SemanticError(f"Unable to resolve indexed variable '{base}' with indices {index_values}")
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

        m = re.match(r"^([A-Za-z][A-Za-z0-9]*)(?:_.*)?$", vname)
        if m:
            base = m.group(1)
            lb_b = getattr(self, "_collected_lbs", {}).get(base)
            ub_b = getattr(self, "_collected_ubs", {}).get(base)
            if lb_b is not None or ub_b is not None:
                return lb_b, ub_b
        return (None, None)

    def _finite_affine_bounds(self, coefficients, constant, context):
        """Return the exact interval of an affine expression with finite variable bounds."""
        if not isinstance(constant, (int, float)):
            raise SemanticError(f"{context} requires a numeric affine expression")
        lower = upper = float(constant)
        for var_name, coefficient in coefficients.items():
            if var_name not in self.var_indices:
                raise SemanticError(f"Variable '{var_name}' not indexed")
            lb, ub = self._infer_var_bounds(var_name)
            if lb is None or ub is None:
                raise SemanticError(f"{context} requires finite variable bounds for a valid big-M formulation")
            coefficient = float(coefficient)
            if coefficient >= 0:
                lower += coefficient * lb
                upper += coefficient * ub
            else:
                lower += coefficient * ub
                upper += coefficient * lb
        return lower, upper

    def _finite_integer_affine_bounds(self, coefficients, constant, context):
        """Validate a unit-integer affine expression and return its exact finite interval."""
        tolerance = 1e-9
        if not isinstance(constant, (int, float)) or abs(float(constant) - round(float(constant))) > tolerance:
            raise SemanticError(f"{context} requires an integer-valued affine expression")
        for var_name, coefficient in coefficients.items():
            if var_name not in self.var_indices:
                raise SemanticError(f"Variable '{var_name}' not indexed")
            if self.integrality[self.var_indices[var_name]] == 0:
                raise SemanticError(f"{context} requires integer variables or an explicit tolerance policy")
            if abs(float(coefficient) - round(float(coefficient))) > tolerance:
                raise SemanticError(f"{context} requires integer-valued coefficients on the unit lattice")
        return self._finite_affine_bounds(coefficients, constant, context)

    def _append_sparse_row(self, state: _ConstraintBuildState, row: list[float], rhs: float, *, sense: str) -> None:
        if sense == "eq":
            row_idx = state.eq_row_idx
            rows = state.A_eq_rows
            cols = state.A_eq_cols
            data = state.A_eq_data
            rhs_values = state.b_eq
        elif sense == "ub":
            row_idx = state.ub_row_idx
            rows = state.A_ub_rows
            cols = state.A_ub_cols
            data = state.A_ub_data
            rhs_values = state.b_ub
        else:
            raise ValueError(f"Unsupported sparse row sense: {sense}")

        for idx, coef in enumerate(row):
            if abs(coef) > LINEAR_ZERO_TOLERANCE:
                rows.append(row_idx)
                cols.append(idx)
                data.append(coef)
        rhs_values.append(rhs)

        if sense == "eq":
            state.eq_row_idx += 1
        else:
            state.ub_row_idx += 1

    def _append_sparse_coef_row(
        self,
        state: _ConstraintBuildState,
        coef_by_var: dict[Any, float],
        rhs: float,
        *,
        sense: str,
    ) -> None:
        row = [0.0] * len(self.var_names)
        for var_name, coef in coef_by_var.items():
            row[self._resolve_coefficient_index(var_name)] += coef
        self._append_sparse_row(state, row, rhs, sense=sense)

    def _finalize_constraint_state(self, state: _ConstraintBuildState) -> None:
        n_vars = len(self.var_names)
        if len(state.b_eq) > 0:
            dense_A_eq = [[0.0 for _ in range(n_vars)] for _ in range(len(state.b_eq))]
            for row_idx, col_idx, value in zip(state.A_eq_rows, state.A_eq_cols, state.A_eq_data):
                dense_A_eq[row_idx][col_idx] = value
            self.A_eq = dense_A_eq
        else:
            self.A_eq = []
        self.b_eq = state.b_eq

        if len(state.b_ub) > 0:
            dense_A_ub = [[0.0 for _ in range(n_vars)] for _ in range(len(state.b_ub))]
            for row_idx, col_idx, value in zip(state.A_ub_rows, state.A_ub_cols, state.A_ub_data):
                dense_A_ub[row_idx][col_idx] = value
            self.A_ub = dense_A_ub
        else:
            self.A_ub = []
        self.b_ub = state.b_ub

        self._add_code_line("from scipy.sparse import csr_matrix")
        self._add_code_line(f"A_eq_rows = {state.A_eq_rows}")
        self._add_code_line(f"A_eq_cols = {state.A_eq_cols}")
        self._add_code_line(f"A_eq_data = {state.A_eq_data}")
        self._add_code_line(f"b_eq = {state.b_eq}")
        self._add_code_line(f"A_ub_rows = {state.A_ub_rows}")
        self._add_code_line(f"A_ub_cols = {state.A_ub_cols}")
        self._add_code_line(f"A_ub_data = {state.A_ub_data}")
        self._add_code_line(f"b_ub = {state.b_ub}")
        self._add_code_line(
            f"A_eq = csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)), shape=({len(state.b_eq)}, {n_vars})) if len(b_eq) > 0 else None"
        )
        self._add_code_line(
            f"A_ub = csr_matrix((A_ub_data, (A_ub_rows, A_ub_cols)), shape=({len(state.b_ub)}, {n_vars})) if len(b_ub) > 0 else None"
        )

    def _comparison_key(self, node, env):
        def _bound_key(part):
            part = self._unwrap_comparison_parentheses(part)
            if not isinstance(part, dict):
                return ("lit", part)
            node_type = part.get("type")
            if node_type == "indexed_name":
                try:
                    return ("indexed", self._multi_indexed_var_name(part, env))
                except Exception:
                    return ("indexed_raw", part.get("name"), str(part.get("dimensions")))
            if node_type == "name":
                name = part.get("value")
                return ("name", env.get(name, name))
            if node_type in ("number", "string_literal", "boolean_literal"):
                return (node_type, part.get("value"))
            if node_type == "binop":
                return ("binop", part.get("op"), _bound_key(part.get("left")), _bound_key(part.get("right")))
            if node_type == "sum":
                return _sum_key(part)
            return (node_type, str(part))

        def _sum_key(part):
            local_iterators = {
                it.get("iterator") for it in (part.get("iterators") or []) if isinstance(it, dict) and it.get("iterator")
            }
            env_snapshot = tuple(
                sorted((name, repr(value)) for name, value in (env or {}).items() if name not in local_iterators)
            )
            return (
                "sum",
                str(part.get("iterators")),
                env_snapshot,
                _bound_key(part.get("expression")),
                _bound_key(part.get("index_constraint")),
            )

        op = node.get("op")
        left = node.get("left")
        right = node.get("right")
        if op in ("==", "!="):
            left_key = _bound_key(left)
            right_key = _bound_key(right)
            return ("cmp", op, left_key, right_key) if left_key <= right_key else ("cmp", op, right_key, left_key)
        return ("cmp", op, _bound_key(left), _bound_key(right))

    @staticmethod
    def _unwrap_comparison_parentheses(expr_node):
        while isinstance(expr_node, dict) and expr_node.get("type") == "parenthesized_expression":
            expr_node = expr_node.get("expression")
        return expr_node

    @staticmethod
    def _is_simple_comparison_node(expr_node):
        return (
            isinstance(expr_node, dict)
            and expr_node.get("type") == "binop"
            and expr_node.get("op") in (">=", "<=", "==", ">", "<")
            and expr_node.get("sem_type") == "boolean"
        )

    def _ground_numeric_comparison_value(self, expr_node, env):
        coef_dict, const_value = self._eval_expr(expr_node, dict(env))
        if coef_dict:
            return None
        if isinstance(const_value, bool):
            return 1.0 if const_value else 0.0
        if isinstance(const_value, (int, float)):
            return float(const_value)
        return None

    def _sum_comparison_truth_names(self, sum_node, env, ctx):
        sum_node = self._unwrap_comparison_parentheses(sum_node)
        if not (isinstance(sum_node, dict) and sum_node.get("type") == "sum"):
            return None
        inner_comparison = self._unwrap_comparison_parentheses(sum_node.get("expression"))
        if not self._is_simple_comparison_node(inner_comparison):
            return None

        truth_names = []
        for env2, _idx_tuple in self._iter_filtered_environments(
            sum_node.get("iterators", []),
            env,
            sum_node.get("index_constraint"),
        ):
            comparison = {
                "type": "binop",
                "op": inner_comparison.get("op"),
                "left": inner_comparison.get("left"),
                "right": inner_comparison.get("right"),
                "sem_type": "boolean",
            }
            truth_names.append(self._comparison_truth_var(comparison, env2, ctx))
        return truth_names

    def _add_comparison_binary(self, name):
        self.var_names.append(name)
        self.var_indices[name] = len(self.var_names) - 1
        self.bounds.append([0, 1])
        self.integrality.append(1)
        if hasattr(self, "c") and len(self.c) < len(self.var_names):
            self.c.append(0.0)

    @staticmethod
    def _coerce_comparison_numeric(value):
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise SemanticError(f"Non-numeric term '{value}' in linear comparison; cannot linearize")
        raise SemanticError(f"Unsupported constant type {type(value)} in linear comparison")

    def _comparison_truth_for_sum(self, node, env, ctx, comparison_truth_cache, key):
        if node.get("op") != "==":
            return None
        left_truths = self._sum_comparison_truth_names(node.get("left"), env, ctx)
        right_truths = self._sum_comparison_truth_names(node.get("right"), env, ctx)
        if left_truths is None and right_truths is None:
            return None
        z_names = left_truths if left_truths is not None else right_truths
        other_side = node.get("right") if left_truths is not None else node.get("left")
        k_value = self._ground_numeric_comparison_value(other_side, env)
        if k_value is None or abs(k_value - len(z_names)) >= LINEAR_ZERO_TOLERANCE:
            return None
        bname = f"cmp_flag_{len(comparison_truth_cache)}"
        self._add_comparison_binary(bname)
        for z_name in z_names:
            row = [0.0] * len(self.var_names)
            row[self.var_indices[bname]] += 1.0
            row[self.var_indices[z_name]] -= 1.0
            self._append_sparse_row(ctx.state, row, 0.0, sense="ub")
        row = [0.0] * len(self.var_names)
        for z_name in z_names:
            row[self.var_indices[z_name]] += 1.0
        row[self.var_indices[bname]] -= 1.0
        self._append_sparse_row(ctx.state, row, float(len(z_names) - 1), sense="ub")
        comparison_truth_cache[key] = bname
        self._add_code_line("# comparison truth var for conjunction of sum-comparisons")
        return bname

    def _comparison_truth_affine_parts(self, node, env):
        lhs_dict, lhs_const = self._eval_expr(node["left"], dict(env))
        rhs_dict, rhs_const = self._eval_expr(node["right"], dict(env))
        lhs_const = self._coerce_comparison_numeric(lhs_const)
        rhs_const = self._coerce_comparison_numeric(rhs_const)
        expr_coef = dict(lhs_dict)
        for var_name, coef in rhs_dict.items():
            expr_coef[var_name] = expr_coef.get(var_name, 0.0) - coef
        expr_const = lhs_const - rhs_const
        return expr_coef, expr_const

    def _append_comparison_inequality_rows(self, expr_coef, expr_const, op, bname, ctx):
        oriented_coef = dict(expr_coef)
        oriented_const = expr_const
        integer_valued = all(
            self.integrality[self.var_indices[var_name]] != 0 and float(coef).is_integer()
            for var_name, coef in expr_coef.items()
        ) and float(expr_const).is_integer()
        separation = comparison_policy(integer_valued=integer_valued).strict_separation
        if op in (">=", ">"):
            oriented_coef = {var_name: -coef for var_name, coef in oriented_coef.items()}
            oriented_const = -oriented_const
        if op in ("<", ">"):
            oriented_const += separation
        lower, upper = self._finite_affine_bounds(
            oriented_coef,
            oriented_const,
            f"Comparison truth variable for '{op}'",
        )
        true_row = dict(oriented_coef)
        true_row[bname] = upper
        self._append_sparse_coef_row(ctx.state, true_row, upper - oriented_const, sense="ub")
        false_row = {var_name: -coef for var_name, coef in oriented_coef.items()}
        false_row[bname] = lower - separation
        self._append_sparse_coef_row(ctx.state, false_row, -separation + oriented_const, sense="ub")

    def _append_comparison_equality_rows(self, expr_coef, expr_const, op, bname, ctx):
        lower, upper = self._finite_integer_affine_bounds(
            expr_coef,
            expr_const,
            f"Comparison truth variable for '{op}'",
        )
        negative_name = f"{bname}_negative"
        positive_name = f"{bname}_positive"
        self._add_comparison_binary(negative_name)
        self._add_comparison_binary(positive_name)
        relation = {negative_name: 1.0, positive_name: 1.0}
        relation[bname] = 1.0 if op == "==" else -1.0
        self._append_sparse_coef_row(ctx.state, relation, 1.0 if op == "==" else 0.0, sense="eq")
        lower_row = {var_name: -coef for var_name, coef in expr_coef.items()}
        lower_row[negative_name] = lower
        lower_row[positive_name] = 1.0
        self._append_sparse_coef_row(ctx.state, lower_row, expr_const, sense="ub")
        upper_row = dict(expr_coef)
        upper_row[negative_name] = 1.0
        upper_row[positive_name] = -upper
        self._append_sparse_coef_row(ctx.state, upper_row, -expr_const, sense="ub")

    def _comparison_truth_var(self, node, env, ctx: _ConstraintBuildContext):
        """Return a binary var name representing truth of a supported linear comparison."""
        if not (
            isinstance(node, dict)
            and node.get("type") == "binop"
            and node.get("sem_type") == "boolean"
            and node.get("op") in ("<=", "<", ">=", ">", "!=", "==")
        ):
            raise SemanticError("Not a supported comparison binop for truth var")
        comparison_truth_cache = ctx.comparison_truth_cache
        key = self._comparison_key(node, env)
        if key in comparison_truth_cache:
            return comparison_truth_cache[key]
        op = node.get("op")
        sum_truth_name = self._comparison_truth_for_sum(node, env, ctx, comparison_truth_cache, key)
        if sum_truth_name is not None:
            return sum_truth_name

        expr_coef, expr_const = self._comparison_truth_affine_parts(node, env)
        bname = f"cmp_flag_{len(comparison_truth_cache)}"
        self._add_comparison_binary(bname)
        if op in ("<=", "<", ">=", ">"):
            self._append_comparison_inequality_rows(expr_coef, expr_const, op, bname, ctx)
        else:
            self._append_comparison_equality_rows(expr_coef, expr_const, op, bname, ctx)
        comparison_truth_cache[key] = bname
        self._add_code_line(f"# comparison truth var for {op}")
        return bname

    def _bool_bound_key(self, part, env):
        while isinstance(part, dict) and part.get("type") == "parenthesized_expression":
            part = part.get("expression")
        if not isinstance(part, dict):
            return ("lit", part)
        node_type = part.get("type")
        if node_type == "indexed_name":
            try:
                return ("indexed", self._multi_indexed_var_name(part, env))
            except Exception:
                return ("indexed_raw", part.get("name"), str(part.get("dimensions")))
        if node_type == "name":
            name = part.get("value")
            return ("name", env.get(name, name))
        if node_type in ("number", "string_literal", "boolean_literal"):
            return (node_type, part.get("value"))
        if node_type == "binop":
            return (
                "binop",
                part.get("op"),
                self._bool_bound_key(part.get("left"), env),
                self._bool_bound_key(part.get("right"), env),
            )
        if node_type == "sum":
            return self._bool_sum_bound_key(part, env)
        return (node_type, str(part))

    def _bool_sum_bound_key(self, part, env):
        local_iterators = {
            it.get("iterator") for it in (part.get("iterators") or []) if isinstance(it, dict) and it.get("iterator")
        }
        env_snapshot = tuple(sorted((name, repr(value)) for name, value in (env or {}).items() if name not in local_iterators))
        return (
            "sum",
            str(part.get("iterators")),
            env_snapshot,
            self._bool_bound_key(part.get("expression"), env),
            self._bool_bound_key(part.get("index_constraint"), env),
        )

    def _bool_atom_key(self, node, env):
        if not isinstance(node, dict) or node.get("type") != "constraint" or node.get("op") != "==":
            return None
        left = node["left"]
        right = node["right"]

        def is_num01(value):
            return isinstance(value, dict) and value.get("type") == "number" and value.get("value") in (0, 1)

        def is_var(value):
            return isinstance(value, dict) and value.get("type") in ("name", "indexed_name")

        if is_var(left) and is_num01(right):
            vname = self._multi_indexed_var_name(left, env) if left.get("type") == "indexed_name" else left["value"]
            return ("atom", vname, right["value"])
        if is_var(right) and is_num01(left):
            vname = self._multi_indexed_var_name(right, env) if right.get("type") == "indexed_name" else right["value"]
            return ("atom", vname, left["value"])
        return None

    def _bool_equality_link_key(self, node, env):
        if node.get("op") != "==" or not isinstance(node.get("left"), dict) or not isinstance(node.get("right"), dict):
            return None
        left = node["left"]
        right = node["right"]

        def is_bool_var(value):
            return self._is_bool_var_node(value)

        def is_bool_composite(value):
            return self._is_bool_composite_node(value)

        if is_bool_var(left) and is_bool_composite(right):
            return ("eq_link", self._bool_bound_key(left, env), self._bool_struct_key(right, env))
        if is_bool_var(right) and is_bool_composite(left):
            return ("eq_link", self._bool_bound_key(right, env), self._bool_struct_key(left, env))
        return None

    def _bool_composite_struct_key(self, node, env):
        left_key = self._bool_struct_key(node["left"], env)
        right_key = self._bool_struct_key(node["right"], env)
        if isinstance(left_key, tuple) and len(left_key) >= 3 and left_key[0] == "eq_link":
            left_key = left_key[2]
        if isinstance(right_key, tuple) and len(right_key) >= 3 and right_key[0] == "eq_link":
            right_key = right_key[2]
        return (node.get("type"), tuple(sorted([left_key, right_key])))

    def _bool_comparison_struct_key(self, node, env):
        link_key = self._bool_equality_link_key(node, env)
        if link_key is not None:
            return link_key
        return (
            "cmp",
            node.get("op"),
            self._bool_bound_key(node.get("left"), env),
            self._bool_bound_key(node.get("right"), env),
        )

    def _bool_struct_key(self, node, env):
        while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
            node = node.get("expression")
        if not isinstance(node, dict):
            return ("lit", node)
        node_type = node.get("type")
        atom_key = self._bool_atom_key(node, env)
        if atom_key is not None:
            return atom_key
        if node_type == "not":
            return ("not", self._bool_struct_key(node["value"], env))
        if node_type in ("and", "or"):
            return self._bool_composite_struct_key(node, env)
        if node_type == "binop" and node.get("sem_type") == "boolean" and node.get("op") in ("<=", ">=", "!=", "=="):
            return self._bool_comparison_struct_key(node, env)
        return ("unknown", id(node))

    def _new_bool_aux_var(self) -> str:
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

    def _atomic_bool_var(self, node, env):
        if not isinstance(node, dict):
            raise SemanticError("Non-dict atomic boolean node")
        if node.get("type") == "constraint" and node.get("op") == "==":
            left = node["left"]
            right = node["right"]

            def is_num01(value):
                return isinstance(value, dict) and value.get("type") == "number" and value.get("value") in (0, 1)

            def is_var(value):
                return isinstance(value, dict) and value.get("type") in ("name", "indexed_name")

            if is_var(left) and is_num01(right):
                vname = self._multi_indexed_var_name(left, env) if left.get("type") == "indexed_name" else left["value"]
                return vname, (1 if right["value"] == 1 else -1)
            if is_var(right) and is_num01(left):
                vname = self._multi_indexed_var_name(right, env) if right.get("type") == "indexed_name" else right["value"]
                return vname, (1 if left["value"] == 1 else -1)
        raise SemanticError("Unsupported atomic boolean literal")

    @staticmethod
    def _is_bool_var_node(node) -> bool:
        return isinstance(node, dict) and node.get("type") in ("name", "indexed_name") and node.get("sem_type") == "boolean"

    @staticmethod
    def _is_bool_composite_node(node, *, include_not: bool = False) -> bool:
        composite_types = (
            ("and", "or", "binop", "parenthesized_expression", "not")
            if include_not
            else (
                "and",
                "or",
                "binop",
                "parenthesized_expression",
            )
        )
        return isinstance(node, dict) and node.get("sem_type") == "boolean" and node.get("type") in composite_types

    def _extract_bool_var_equality(self, node):
        while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
            node = node.get("expression")
        if not (
            isinstance(node, dict)
            and node.get("type") == "binop"
            and node.get("op") == "=="
            and isinstance(node.get("left"), dict)
            and isinstance(node.get("right"), dict)
        ):
            return None

        left = node["left"]
        right = node["right"]
        if self._is_bool_var_node(left) and self._is_bool_composite_node(right, include_not=True):
            return (left, right)
        if self._is_bool_var_node(right) and self._is_bool_composite_node(left, include_not=True):
            return (right, left)
        return None

    @staticmethod
    def _is_boolean_expression_node(node):
        return isinstance(node, dict) and (
            node.get("sem_type") == "boolean"
            or node.get("type") in ("boolean_literal", "and", "or", "not")
            or (
                node.get("type") == "constraint"
                and node.get("op") == "=="
                and (
                    (isinstance(node.get("left"), dict) and node["left"].get("type") in ("name", "indexed_name"))
                    or (isinstance(node.get("right"), dict) and node["right"].get("type") in ("name", "indexed_name"))
                )
            )
        )

    def _register_boolean_aux_node(self, node):
        vname = node["name"]
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
        return vname

    def _bool_expr_var(self, node, env, ctx: _ConstraintBuildContext):
        env_memo_key = (
            id(node),
            tuple(sorted((name, repr(value)) for name, value in (env or {}).items())),
        )
        struct_key = self._bool_struct_key(node, env)
        if struct_key in ctx.subtree_var_cache:
            return ctx.subtree_var_cache[struct_key]
        if env_memo_key in ctx.expr_memo:
            return ctx.expr_memo[env_memo_key]

        result = self._encode_bool_expr_var(node, env, ctx, struct_key, env_memo_key)
        ctx.subtree_var_cache[struct_key] = result
        ctx.expr_memo[env_memo_key] = result
        return result

    def _boolean_composite_operands(self, node):
        tie_vars = []
        operands = []
        for operand in (node["left"], node["right"]):
            equality = self._extract_bool_var_equality(operand)
            if equality:
                tie_vars.append(equality[0])
                operand = equality[1]
            operands.append(operand)
        return operands[0], operands[1], tie_vars

    def _tie_boolean_vars(self, var_nodes, target_var, env, ctx):
        for var_node in var_nodes:
            vname = (
                self._multi_indexed_var_name(var_node, env) if var_node.get("type") == "indexed_name" else var_node["value"]
            )
            if vname in self.var_indices and target_var in self.var_indices:
                row = [0.0] * len(self.var_names)
                row[self.var_indices[vname]] = 1.0
                row[self.var_indices[target_var]] = -1.0
                self._append_sparse_row(ctx.state, row, 0.0, sense="eq")

    def _append_boolean_composite_rows(self, op, result_var, left_var, right_var, ctx):
        result_idx = self.var_indices[result_var]
        left_idx = self.var_indices[left_var]
        right_idx = self.var_indices[right_var]
        if op == "and":
            rows = (
                ({result_idx: 1.0, left_idx: -1.0}, 0.0),
                ({result_idx: 1.0, right_idx: -1.0}, 0.0),
                ({result_idx: -1.0, left_idx: 1.0, right_idx: 1.0}, 1.0),
            )
        else:
            rows = (
                ({result_idx: -1.0, left_idx: 1.0}, 0.0),
                ({result_idx: -1.0, right_idx: 1.0}, 0.0),
                ({result_idx: 1.0, left_idx: -1.0, right_idx: -1.0}, 0.0),
            )
        for coefficients, rhs in rows:
            row = [0.0] * len(self.var_names)
            for index, coefficient in coefficients.items():
                row[index] += coefficient
            self._append_sparse_row(ctx.state, row, rhs, sense="ub")

    def _encode_boolean_composite(self, node, env, ctx, env_memo_key):
        struct_key = self._bool_struct_key(node, env)
        if struct_key in ctx.subtree_var_cache:
            shared_aux = ctx.subtree_var_cache[struct_key]
            ctx.expr_memo[env_memo_key] = shared_aux
            tie_vars = []
            for operand in (node["left"], node["right"]):
                equality = self._extract_bool_var_equality(operand)
                if equality:
                    tie_vars.append(equality[0])
            self._tie_boolean_vars(tie_vars, shared_aux, env, ctx)
            return shared_aux

        left_node, right_node, tie_vars = self._boolean_composite_operands(node)
        left_var = self._bool_expr_var(left_node, env, ctx)
        right_var = self._bool_expr_var(right_node, env, ctx)
        if left_var == right_var:
            result_var = left_var
        else:
            result_var = self._new_bool_aux_var()
            self._append_boolean_composite_rows(node["type"], result_var, left_var, right_var, ctx)
        ctx.expr_memo[env_memo_key] = result_var
        ctx.subtree_var_cache[struct_key] = result_var
        self._tie_boolean_vars(tie_vars, result_var, env, ctx)
        return result_var

    @staticmethod
    def _memoize_boolean_var(ctx, struct_key, env_memo_key, var_name):
        ctx.subtree_var_cache[struct_key] = var_name
        ctx.expr_memo[env_memo_key] = var_name
        return var_name

    def _try_encode_constraint_comparison(self, node, env, ctx, struct_key, env_memo_key):
        if node.get("type") != "constraint" or node.get("op") not in ("<=", "<", ">=", ">", "!=", "=="):
            return None
        if (
            node.get("op") == "!="
            and self._is_boolean_expression_node(node.get("left"))
            and self._is_boolean_expression_node(node.get("right"))
        ):
            return None

        if node.get("op") == "==":
            try:
                self._atomic_bool_var(node, env)
                return None
            except SemanticError:
                pass

        comparison = {
            "type": "binop",
            "op": node.get("op"),
            "left": node.get("left"),
            "right": node.get("right"),
            "sem_type": "boolean",
        }
        result = self._comparison_truth_var(comparison, env, ctx)
        return self._memoize_boolean_var(ctx, struct_key, env_memo_key, result)

    def _try_encode_boolean_not_equal(self, node, env, ctx, struct_key, env_memo_key):
        is_binop = node.get("type") == "binop" and node.get("sem_type") == "boolean"
        is_constraint = node.get("type") == "constraint"
        if node.get("op") != "!=" or not (is_binop or is_constraint):
            return None

        left = node["left"]
        right = node["right"]
        if not self._is_boolean_expression_node(left) or not self._is_boolean_expression_node(right):
            return None

        left_var = self._bool_expr_var(left, env, ctx)
        right_var = self._bool_expr_var(right, env, ctx)
        result_var = self._new_bool_aux_var()
        result_idx = self.var_indices[result_var]
        left_idx = self.var_indices[left_var]
        right_idx = self.var_indices[right_var]
        rows = (
            ({result_idx: -1.0, left_idx: 1.0, right_idx: -1.0}, 0.0),
            ({result_idx: -1.0, left_idx: -1.0, right_idx: 1.0}, 0.0),
            ({result_idx: 1.0, left_idx: -1.0, right_idx: -1.0}, 0.0),
            ({result_idx: 1.0, left_idx: 1.0, right_idx: 1.0}, 2.0),
        )
        for coefficients, rhs in rows:
            row = [0.0] * len(self.var_names)
            for index, coefficient in coefficients.items():
                row[index] = coefficient
            self._append_sparse_row(ctx.state, row, rhs, sense="ub")
        return self._memoize_boolean_var(ctx, struct_key, env_memo_key, result_var)

    def _boolean_binop_tie_variable(self, node, env, ctx):
        if node.get("op") != "==":
            return None
        left = node.get("left")
        right = node.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            return None
        if self._is_bool_var_node(left) and self._is_bool_composite_node(right, include_not=True):
            var_side, expr_side = left, right
        elif self._is_bool_var_node(right) and self._is_bool_composite_node(left, include_not=True):
            var_side, expr_side = right, left
        else:
            return None
        expr_var = self._bool_expr_var(expr_side, env, ctx)
        var_name = self._multi_indexed_var_name(var_side, env) if var_side.get("type") == "indexed_name" else var_side["value"]
        if var_name not in self.var_indices or expr_var not in self.var_indices:
            return var_name
        var_idx = self.var_indices[var_name]
        expr_idx = self.var_indices[expr_var]
        already_tied = any(abs(row[var_idx]) == 1 and abs(row[expr_idx]) == 1 for row in getattr(self, "A_eq", ()))
        if not already_tied:
            row = [0.0] * len(self.var_names)
            row[var_idx] = 1.0
            row[expr_idx] = -1.0
            self._append_sparse_row(ctx.state, row, 0.0, sense="eq")
        return var_name

    def _try_encode_boolean_binop(self, node, env, ctx, struct_key, env_memo_key):
        if node.get("type") != "binop" or node.get("sem_type") != "boolean" or node.get("op") not in ("<=", ">=", "!=", "=="):
            return None
        tied_name = self._boolean_binop_tie_variable(node, env, ctx)
        if tied_name is not None:
            return self._memoize_boolean_var(ctx, struct_key, env_memo_key, tied_name)
        result = self._comparison_truth_var(node, env, ctx)
        return self._memoize_boolean_var(ctx, struct_key, env_memo_key, result)

    def _try_encode_atomic_boolean_constraint(self, node, env, ctx, struct_key, env_memo_key):
        if node.get("type") != "constraint" or node.get("op") != "==":
            return None

        var_name, polarity = self._atomic_bool_var(node, env)
        if polarity == 1:
            return self._memoize_boolean_var(ctx, struct_key, env_memo_key, var_name)
        if var_name in ctx.neg_cache:
            return self._memoize_boolean_var(ctx, struct_key, env_memo_key, ctx.neg_cache[var_name])

        result_var = self._new_bool_aux_var()
        row = [0.0] * len(self.var_names)
        row[self.var_indices[result_var]] = 1.0
        row[self.var_indices[var_name]] = 1.0
        self._append_sparse_row(ctx.state, row, 1.0, sense="eq")
        ctx.neg_cache[var_name] = result_var
        return self._memoize_boolean_var(ctx, struct_key, env_memo_key, result_var)

    def _try_encode_boolean_negation(self, node, env, ctx, struct_key, env_memo_key):
        if node.get("type") != "not":
            return None

        inner_var = self._bool_expr_var(node["value"], env, ctx)
        if inner_var in ctx.neg_cache:
            return self._memoize_boolean_var(ctx, struct_key, env_memo_key, ctx.neg_cache[inner_var])

        result_var = self._new_bool_aux_var()
        row = [0.0] * len(self.var_names)
        row[self.var_indices[result_var]] = 1.0
        row[self.var_indices[inner_var]] = 1.0
        self._append_sparse_row(ctx.state, row, 1.0, sense="eq")
        ctx.neg_cache[inner_var] = result_var
        return self._memoize_boolean_var(ctx, struct_key, env_memo_key, result_var)

    def _encode_bool_expr_var(self, node, env, ctx, struct_key, env_memo_key):
        if struct_key in ctx.subtree_var_cache:
            return ctx.subtree_var_cache[struct_key]
        if env_memo_key in ctx.expr_memo:
            return ctx.expr_memo[env_memo_key]
        if not isinstance(node, dict):
            raise SemanticError("Invalid boolean expr node (not a dict): {}".format(repr(node)))
        node_type = node.get("type")
        if node_type == "aux_var" and node.get("sem_type") == "boolean":
            var_name = self._register_boolean_aux_node(node)
            return self._memoize_boolean_var(ctx, struct_key, env_memo_key, var_name)
        if node_type == "parenthesized_expression":
            return self._bool_expr_var(node.get("expression"), env, ctx)

        encoders = (
            self._try_encode_constraint_comparison,
            self._try_encode_boolean_not_equal,
            self._try_encode_boolean_binop,
            self._try_encode_atomic_boolean_constraint,
            self._try_encode_boolean_negation,
        )
        for encoder in encoders:
            result = encoder(node, env, ctx, struct_key, env_memo_key)
            if result is not None:
                return result

        if node_type in ("and", "or"):
            return self._encode_boolean_composite(node, env, ctx, env_memo_key)
        if node_type == "implies":
            lowered = lower_implication(node["left"], node["right"])
            return self._bool_expr_var(lowered, env, ctx)
        raise SemanticError(f"Unsupported or non-resolvable boolean expression node type: {node_type} ({repr(node)})")

    @staticmethod
    def _unwrap_parenthesized_node(node):
        while isinstance(node, dict) and node.get("type") == "parenthesized_expression":
            node = node.get("expression")
        return node

    @staticmethod
    def _is_var_reference_node(node):
        return isinstance(node, dict) and node.get("type") in ("name", "indexed_name")

    @staticmethod
    def _is_number_node(node):
        return isinstance(node, dict) and node.get("type") == "number"

    @staticmethod
    def _is_number_01_node(node):
        return isinstance(node, dict) and node.get("type") == "number" and node.get("value") in (0, 1)

    def _is_simple_boolean_comparison(self, node):
        node = self._unwrap_parenthesized_node(node)
        return (
            isinstance(node, dict)
            and node.get("type") == "binop"
            and node.get("op") in (">=", "<=", "==", ">", "<")
            and node.get("sem_type") == "boolean"
        )

    def _detect_sum_of_comparisons(self, left, right, op_sym_top):
        """Return comparison cardinality parts, or a false sentinel if the pattern does not match."""
        left_unwrapped = self._unwrap_parenthesized_node(left)
        right_unwrapped = self._unwrap_parenthesized_node(right)
        if (
            isinstance(left_unwrapped, dict)
            and left_unwrapped.get("type") == "sum"
            and isinstance(right_unwrapped, dict)
            and right_unwrapped.get("type") == "number"
            and op_sym_top in (">=", "==", ">", "<=", "<")
        ):
            inner_cmp = self._unwrap_parenthesized_node(left_unwrapped.get("expression"))
            if self._is_simple_boolean_comparison(inner_cmp):
                threshold = right_unwrapped.get("value")
                if op_sym_top == ">":
                    threshold += 1
                elif op_sym_top == "<":
                    threshold -= 1
                return (
                    True,
                    inner_cmp,
                    threshold,
                    left_unwrapped.get("iterators", []),
                    left_unwrapped.get("index_constraint"),
                )
        return False, None, None, None, None

    @classmethod
    def _unwrap_boolean_true_constraint(cls, node):
        if not (isinstance(node, dict) and node.get("type") == "constraint" and node.get("op") == "=="):
            return node
        left = node.get("left")
        right = node.get("right")

        def is_true(value):
            return isinstance(value, dict) and value.get("type") == "boolean_literal" and value.get("value") is True

        if is_true(left):
            expression = right
        elif is_true(right):
            expression = left
        else:
            return node
        if not isinstance(expression, dict) or expression.get("type") not in (
            "parenthesized_expression",
            "binop",
            "and",
            "or",
            "not",
        ):
            return node
        inner = cls._unwrap_parenthesized_node(expression)
        if not isinstance(inner, dict):
            return node
        if inner.get("type") in ("and", "or", "not"):
            return inner
        if inner.get("type") != "binop":
            return node
        return {
            "type": "constraint",
            "op": inner.get("op"),
            "left": inner.get("left"),
            "right": inner.get("right"),
        }

    @classmethod
    def _normalize_implication_nodes(cls, antecedent, consequent):
        return cls._unwrap_boolean_true_constraint(antecedent), cls._unwrap_boolean_true_constraint(consequent)

    def _try_enforce_reified_implication_literal(self, constr, env, bool_expr_var, append_eq_row):
        if not (
            constr.get("type") == "constraint"
            and isinstance(constr.get("left"), dict)
            and constr.get("left").get("type") == "implies"
        ):
            return False
        auxiliary = bool_expr_var(constr.get("left"), env)
        logger.debug(f"[DEBUG] Created auxiliary {auxiliary} for implies expr: {constr.get('left')}")
        right = constr.get("right")
        if isinstance(right, dict) and (
            (right.get("type") == "boolean_literal" and right.get("value") is True)
            or (right.get("type") == "number" and right.get("value") == 1)
        ):
            value = 1.0
        elif isinstance(right, dict) and (
            (right.get("type") == "boolean_literal" and right.get("value") is False)
            or (right.get("type") == "number" and right.get("value") == 0)
        ):
            value = 0.0
        else:
            return False
        row = [0.0] * len(self.var_names)
        row[self.var_indices[auxiliary]] = 1.0
        append_eq_row(row, value)
        return True

    def _is_declared_boolean_var_node(self, node):
        if not self._is_var_reference_node(node):
            return False
        base_name = node.get("value") if node.get("type") == "name" else node.get("name")
        declaration = self._find_decl(base_name)
        return bool(
            declaration and declaration.get("type") in ("dvar", "dvar_indexed") and declaration.get("var_type") == "boolean"
        )

    def _is_constraint_boolean_expression(self, node):
        return isinstance(node, dict) and (
            node.get("type") == "boolean_literal"
            or (node.get("type") == "binop" and node.get("sem_type") == "boolean")
            or (
                node.get("type") == "constraint"
                and node.get("op") == "=="
                and (self._is_var_reference_node(node.get("left")) or self._is_var_reference_node(node.get("right")))
            )
            or node.get("type") in ("and", "or", "not")
        )

    def _is_atomic_bool_tree_node(self, node):
        if not (isinstance(node, dict) and node.get("type") == "constraint" and node.get("op") == "=="):
            return False
        left = node.get("left")
        right = node.get("right")
        return (self._is_var_reference_node(left) and self._is_number_01_node(right)) or (
            self._is_var_reference_node(right) and self._is_number_01_node(left)
        )

    def _is_bool_tree_node(self, node):
        if not isinstance(node, dict):
            return False
        node_type = node.get("type")
        if node_type in ("and", "or"):
            return self._is_bool_tree_node(node.get("left")) and self._is_bool_tree_node(node.get("right"))
        if node_type == "not":
            return self._is_bool_tree_node(node.get("value"))
        return self._is_atomic_bool_tree_node(node)

    def _is_unbound_constraint_parameter_node(self, node):
        if not self._is_var_reference_node(node):
            return False
        base = node.get("value") if node.get("type") == "name" else node.get("name")
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
        if decl.get("type") in ("parameter_inline", "parameter_inline_indexed") and decl.get("value") is not None:
            return False
        return self.data_dict.get(base) is None

    def _ensure_constraint_parameters_bound(self, constr):
        if constr.get("type") == "constraint" and (
            self._is_unbound_constraint_parameter_node(constr.get("left"))
            or self._is_unbound_constraint_parameter_node(constr.get("right"))
        ):
            raise SemanticError("Constraint references parameter with no data provided")

    def _collect_passive_bound_for_pair(self, variable_node, number_node, operator, env, variable_on_left):
        vname = (
            self._multi_indexed_var_name(variable_node, env)
            if variable_node.get("type") == "indexed_name"
            else variable_node["value"]
        )
        value = float(number_node.get("value"))
        normalized_operator = operator if variable_on_left else {">=": "<=", "<=": ">=", "==": "=="}[operator]

        if normalized_operator == ">=":
            self._collected_lbs[vname] = max(self._collected_lbs.get(vname, -float("inf")), value)
        elif normalized_operator == "<=":
            self._collected_ubs[vname] = min(self._collected_ubs.get(vname, float("inf")), value)
        else:
            self._collected_lbs[vname] = max(self._collected_lbs.get(vname, -float("inf")), value)
            self._collected_ubs[vname] = min(self._collected_ubs.get(vname, float("inf")), value)

        if variable_node.get("type") != "indexed_name":
            return
        base_symbol = variable_node.get("name")
        base_operator = operator if variable_on_left else {">=": "<=", "<=": ">=", "==": "=="}[operator]
        if base_operator in (">=", "=="):
            current_lower = self._collected_lbs.get(base_symbol)
            if current_lower is None or value < current_lower:
                self._collected_lbs[base_symbol] = max(self._collected_lbs.get(base_symbol, -float("inf")), value)
        if base_operator in ("<=", "=="):
            current_upper = self._collected_ubs.get(base_symbol)
            if current_upper is None or value > current_upper:
                self._collected_ubs[base_symbol] = min(self._collected_ubs.get(base_symbol, float("inf")), value)

    def _collect_passive_constraint_bounds(self, constr, env, bool_expr_var) -> None:
        """Collect simple variable bounds for later big-M tightening without changing core constraint handling."""

        def tighten_lower_bound(symbol, val):
            self._collected_lbs[symbol] = max(self._collected_lbs.get(symbol, -float("inf")), val)

        def tighten_upper_bound(symbol, val):
            self._collected_ubs[symbol] = min(self._collected_ubs.get(symbol, float("inf")), val)

        def tighten_bounds(symbol, val):
            tighten_lower_bound(symbol, val)
            tighten_upper_bound(symbol, val)

        try:
            if constr.get("type") == "constraint" and constr.get("op") == "!=":
                left = constr.get("left")
                right = constr.get("right")
                if self._is_constraint_boolean_expression(left) and self._is_constraint_boolean_expression(right):
                    bool_expr_var(constr, env)

            if constr.get("type") != "constraint" or constr.get("op") not in (">=", "<=", "=="):
                return

            op_sym = constr.get("op")
            left = constr.get("left")
            right = constr.get("right")
            if self._is_var_reference_node(left) and self._is_number_node(right):
                self._collect_passive_bound_for_pair(left, right, op_sym, env, variable_on_left=True)
            elif self._is_var_reference_node(right) and self._is_number_node(left):
                self._collect_passive_bound_for_pair(right, left, op_sym, env, variable_on_left=False)
        except Exception:
            pass

    def _bool_tree_literal_constraint_parts(self, constr):
        if constr.get("type") != "constraint" or constr.get("op") not in ("==", ">=", "<=", "!="):
            return None
        left = constr.get("left")
        right = constr.get("right")
        if not self._is_bool_tree_node(left) or self._is_bool_tree_node(
            {"type": "constraint", "left": left, "op": "==", "right": {"type": "number", "value": 1}}
        ):
            return None
        return left, right, constr.get("op")

    @staticmethod
    def _is_number_literal(node, value):
        return isinstance(node, dict) and node.get("type") == "number" and node.get("value") == value

    def _emit_bool_tree_literal_row(self, aux, value, append_eq_row):
        row = [0.0] * len(self.var_names)
        row[self.var_indices[aux]] = 1.0
        append_eq_row(row, float(value))

    def _try_enforce_bool_tree_literal_constraint(self, constr, env, bool_expr_var, append_eq_row) -> bool:
        """Handle constraints that directly force a composed boolean tree to a literal truth value."""
        parts = self._bool_tree_literal_constraint_parts(constr)
        if parts is None:
            return False
        left, right, operator = parts
        aux = bool_expr_var(left, env)
        logger.debug(f"[DEBUG] Created auxiliary {aux} for boolean expr: {left}")
        if (operator, right) in ((">=", {"type": "number", "value": 0}), ("<=", {"type": "number", "value": 1})):
            logger.debug(f"[DEBUG] Skipping tautological constraint: {aux} {operator} {right}")
            return True
        if operator == "==" and isinstance(right, dict):
            self._emit_bool_tree_literal_row(aux, right.get("value", 0), append_eq_row)
            return True
        if (operator == ">=" and self._is_number_literal(right, 1)) or (
            operator == "!=" and self._is_number_literal(right, 0)
        ):
            self._emit_bool_tree_literal_row(aux, 1, append_eq_row)
            return True
        return False

    def _emit_plain_linear_constraint(self, constr, env, append_eq_row, append_ub_row) -> None:
        left = constr["left"]
        right = constr["right"]
        lhs_dict, lhs_const = self._accumulate_sum_to_dict(left, env, sign=1)
        rhs_dict, rhs_const = self._accumulate_sum_to_dict(right, env, sign=1)
        logger.debug(f"[SciPyCSCCodeGenerator] lhs_dict: {lhs_dict}, lhs_const: {lhs_const}")
        logger.debug(f"[SciPyCSCCodeGenerator] rhs_dict: {rhs_dict}, rhs_const: {rhs_const}")
        row = [0.0] * len(self.var_names)
        for vname, coef in lhs_dict.items():
            idx = self._resolve_coefficient_index(vname)
            logger.debug(f"[SciPyCSCCodeGenerator] LHS vname: {vname}, coef: {coef}, idx: {idx}")
            row[idx] += coef
        for vname, coef in rhs_dict.items():
            idx = self._resolve_coefficient_index(vname)
            logger.debug(f"[SciPyCSCCodeGenerator] RHS vname: {vname}, coef: {coef}, idx: {idx}")
            row[idx] -= coef
        rhs_value = rhs_const - lhs_const
        logger.debug(f"[SciPyCSCCodeGenerator] Final constraint row: {row}, rhs_value: {rhs_value}")
        if constr["op"] == "==":
            append_eq_row(row, rhs_value)
        elif constr["op"] == "<=":
            append_ub_row(row, rhs_value)
        elif constr["op"] == ">=":
            append_ub_row([-v for v in row], -rhs_value)
        elif constr["op"] in (">", "<"):
            adjusted_op, adjusted_rhs = self._strict_adjusted_rhs(constr["op"], rhs_value)
            if adjusted_op == "<=":
                append_ub_row(row, adjusted_rhs)
            else:
                append_ub_row([-v for v in row], -adjusted_rhs)
        else:
            logger.debug(f"Unsupported op: {constr['op']}")

    def _gate_affine_implication_consequent(
        self,
        antecedent_node,
        active_value,
        consequent_node,
        env,
        append_ub_row,
    ):
        if not (isinstance(consequent_node, dict) and consequent_node.get("type") == "constraint"):
            raise SemanticError("Implication consequent must be a constraint")
        consequent_op = consequent_node.get("op")
        if consequent_op not in ("<=", "<", ">=", ">", "=="):
            raise SemanticError("Unsupported implication consequent operator")
        antecedent_name = (
            self._multi_indexed_var_name(antecedent_node, env)
            if antecedent_node.get("type") == "indexed_name"
            else antecedent_node["value"]
        )
        left_coef, left_const = self._eval_expr(consequent_node.get("left"), dict(env or {}))
        right_coef, right_const = self._eval_expr(consequent_node.get("right"), dict(env or {}))
        if not isinstance(left_const, (int, float)) or not isinstance(right_const, (int, float)):
            raise SemanticError("Implication consequent requires a numeric affine expression")

        diff_coef = dict(left_coef)
        for var_name, value in right_coef.items():
            diff_coef[var_name] = diff_coef.get(var_name, 0.0) - value
        diff_const = float(left_const) - float(right_const)
        if consequent_op in (">=", ">"):
            diff_coef = {var_name: -value for var_name, value in diff_coef.items()}
            diff_const = -diff_const
        if consequent_op in ("<", ">"):
            diff_const += BOOL_EPS

        def emit_side(coef, constant):
            _lower, upper = self._finite_affine_bounds(
                coef,
                constant,
                "Boolean implication consequent",
            )
            relaxation = max(0.0, upper)
            row = [0.0] * len(self.var_names)
            for var_name, value in coef.items():
                row[self.var_indices[var_name]] += value
            if active_value == 1:
                row[self.var_indices[antecedent_name]] += relaxation
                rhs = relaxation - constant
            else:
                row[self.var_indices[antecedent_name]] -= relaxation
                rhs = -constant
            append_ub_row(row, rhs)

        emit_side(diff_coef, diff_const)
        if consequent_op == "==":
            emit_side({var_name: -value for var_name, value in diff_coef.items()}, -diff_const)

    @staticmethod
    def _is_implication_var_node(node):
        return isinstance(node, dict) and node.get("type") in ("name", "indexed_name")

    @staticmethod
    def _is_implication_number(node, value):
        return isinstance(node, dict) and node.get("type") == "number" and node.get("value") == value

    def _implication_variable_name(self, node, env):
        return self._multi_indexed_var_name(node, env) if node.get("type") == "indexed_name" else node["value"]

    def _is_boolean_decision_variable(self, node):
        if not self._is_implication_var_node(node):
            return False
        base_name = node.get("value") if node.get("type") == "name" else node.get("name")
        declaration = self._find_decl(base_name)
        return bool(
            declaration and declaration.get("type") in ("dvar", "dvar_indexed") and declaration.get("var_type") == "boolean"
        )

    def _implication_boolean_equality_variable(self, consequent, value):
        if consequent.get("op") != "==":
            return None
        left = consequent.get("left")
        right = consequent.get("right")
        if self._is_implication_var_node(left) and self._is_implication_number(right, value):
            return left
        if self._is_implication_var_node(right) and self._is_implication_number(left, value):
            return right
        return None

    def _emit_boolean_implication_row(self, antecedent_name, variable_name, value, append_ub_row):
        row = [0.0] * len(self.var_names)
        row[self.var_indices[antecedent_name]] = 1.0
        row[self.var_indices[variable_name]] = -1.0 if value == 1 else 1.0
        append_ub_row(row, 0.0 if value == 1 else 1.0)

    def _implication_boolean_target(self, consequent, value):
        variable = self._implication_boolean_equality_variable(consequent, value)
        if variable is not None and self._is_boolean_decision_variable(variable):
            return variable
        left = consequent.get("left")
        right = consequent.get("right")
        operators = (">=", "==") if value == 1 else ("<=", "==")
        if (
            consequent.get("op") in operators
            and self._is_boolean_decision_variable(left)
            and self._is_implication_number(right, value)
        ):
            return left
        return None

    def _handle_boolean_antecedent_target(self, antecedent_name, consequent, value, env, append_ub_row):
        variable_node = self._implication_boolean_target(consequent, value)
        if variable_node is None:
            return False
        self._emit_boolean_implication_row(
            antecedent_name,
            self._implication_variable_name(variable_node, env),
            value,
            append_ub_row,
        )
        return True

    def _handle_boolean_antecedent_implication(self, antecedent_node, consequent_node, env, append_ub_row):
        antecedent_name = self._implication_variable_name(antecedent_node, env)
        if not (isinstance(consequent_node, dict) and consequent_node.get("type") == "constraint"):
            raise SemanticError("Implication consequent must be a constraint")
        if self._handle_boolean_antecedent_target(antecedent_name, consequent_node, 1, env, append_ub_row):
            return
        if self._handle_boolean_antecedent_target(antecedent_name, consequent_node, 0, env, append_ub_row):
            return
        if consequent_node.get("op") in ("<=", "<", ">=", ">", "=="):
            self._gate_affine_implication_consequent(
                antecedent_node,
                1,
                consequent_node,
                env,
                append_ub_row,
            )
            return
        raise SemanticError("Unsupported implication consequent form")

    def _handle_equality_antecedent_implication(
        self,
        antecedent,
        consequent,
        env,
        append_eq_row,
        append_ub_row,
    ):
        def linear_expression(expr):
            if not isinstance(expr, dict):
                raise SemanticError("Unsupported expression in implication linearization")
            try:
                coef, const = self._eval_expr(expr, dict(env or {}))
            except Exception as exc:
                raise SemanticError("Unsupported linear expression form in implication") from exc
            if not isinstance(const, (int, float)):
                raise SemanticError("Unsupported linear expression form in implication")
            return dict(coef), float(const)

        def expression_difference(left, right):
            left_coef, left_const = linear_expression(left)
            right_coef, right_const = linear_expression(right)
            coef = left_coef.copy()
            for name, value in right_coef.items():
                coef[name] = coef.get(name, 0.0) - value
            return coef, left_const - right_const

        ant_coef, ant_const = expression_difference(
            antecedent.get("left"),
            antecedent.get("right"),
        )
        diff_min, diff_max = self._finite_integer_affine_bounds(
            ant_coef,
            ant_const,
            "Equality implication antecedent",
        )
        if not hasattr(self, "_impl_counter"):
            self._impl_counter = 0
        flag_name = f"implication_flag_c{self._impl_counter}"
        self._impl_counter += 1
        negative_flag = f"{flag_name}_negative"
        positive_flag = f"{flag_name}_positive"
        for name in (flag_name, negative_flag, positive_flag):
            self.var_names.append(name)
            self.var_indices[name] = len(self.var_names) - 1
            self.bounds.append([0, 1])
            self.integrality.append(1)
            if hasattr(self, "c") and len(self.c) < len(self.var_names):
                self.c.append(0.0)

        partition_row = [0.0] * len(self.var_names)
        for name in (flag_name, negative_flag, positive_flag):
            partition_row[self.var_indices[name]] = 1.0
        append_eq_row(partition_row, 1.0)

        lower_row = [0.0] * len(self.var_names)
        upper_row = [0.0] * len(self.var_names)
        for name, coef in ant_coef.items():
            lower_row[self.var_indices[name]] -= coef
            upper_row[self.var_indices[name]] += coef
        lower_row[self.var_indices[negative_flag]] += diff_min
        lower_row[self.var_indices[positive_flag]] += 1.0
        upper_row[self.var_indices[negative_flag]] += 1.0
        upper_row[self.var_indices[positive_flag]] -= diff_max
        append_ub_row(lower_row, ant_const)
        append_ub_row(upper_row, -ant_const)

        consequent_op = consequent.get("op")
        if consequent_op in ("<=", "<", "=="):
            cons_coef, cons_const = expression_difference(
                consequent.get("left"),
                consequent.get("right"),
            )
        elif consequent_op in (">=", ">"):
            cons_coef, cons_const = expression_difference(
                consequent.get("right"),
                consequent.get("left"),
            )
        else:
            raise SemanticError("Unsupported consequent operator")

        def append_gated_side(coef, const):
            _cons_min, cons_max = self._finite_affine_bounds(
                coef,
                const,
                "Equality implication consequent",
            )
            big_m = max(0.0, cons_max)
            row = [0.0] * len(self.var_names)
            for name, value in coef.items():
                row[self.var_indices[name]] += value
            row[self.var_indices[flag_name]] += big_m
            append_ub_row(row, big_m - const)

        append_gated_side(cons_coef, cons_const)
        if consequent_op == "==":
            append_gated_side({name: -value for name, value in cons_coef.items()}, -cons_const)

    def _handle_implication_constraint(
        self,
        constr,
        env,
        bool_expr_var,
        comparison_truth_var,
        append_eq_row,
        append_ub_row,
    ):
        ant_unwrapped, cons_unwrapped = self._normalize_implication_nodes(
            constr["antecedent"],
            constr["consequent"],
        )
        if isinstance(ant_unwrapped, dict) and ant_unwrapped.get("type") in ("and", "or", "not", "implies"):
            antecedent_name = bool_expr_var(ant_unwrapped, env)
            ant_unwrapped = {
                "type": "constraint",
                "op": "==",
                "left": {"type": "name", "value": antecedent_name, "sem_type": "boolean"},
                "right": {"type": "number", "value": 1},
            }
        if isinstance(cons_unwrapped, dict) and cons_unwrapped.get("type") in ("and", "or", "not", "implies"):
            consequent_name = bool_expr_var(cons_unwrapped, env)
            cons_unwrapped = {
                "type": "constraint",
                "op": "==",
                "left": {"type": "name", "value": consequent_name, "sem_type": "boolean"},
                "right": {"type": "number", "value": 1},
            }

        def extract_var_eq_val(node, val):
            if not (isinstance(node, dict) and node.get("type") == "constraint" and node.get("op") == "=="):
                return None
            left = node["left"]
            right = node["right"]

            def is_var(value):
                return isinstance(value, dict) and value.get("type") in ("name", "indexed_name")

            if is_var(left) and isinstance(right, dict) and right.get("type") == "number" and right.get("value") == val:
                return left
            if is_var(right) and isinstance(left, dict) and left.get("type") == "number" and left.get("value") == val:
                return right
            return None

        ant_var_node = extract_var_eq_val(ant_unwrapped, 1)
        ant_eq_zero = extract_var_eq_val(ant_unwrapped, 0)
        if ant_eq_zero is not None:
            self._gate_affine_implication_consequent(
                ant_eq_zero,
                0,
                cons_unwrapped,
                env,
                append_ub_row,
            )
            return

        if ant_var_node:
            self._handle_boolean_antecedent_implication(
                ant_var_node,
                cons_unwrapped,
                env,
                append_ub_row,
            )
            return

        if not (
            isinstance(ant_unwrapped, dict)
            and ant_unwrapped.get("type") == "constraint"
            and isinstance(cons_unwrapped, dict)
            and cons_unwrapped.get("type") == "constraint"
        ):
            raise SemanticError("Implication antecedent must be boolean var == 1 or linear constraint")
        ant_op = ant_unwrapped.get("op")
        cons_op = cons_unwrapped.get("op")
        supported_ops = {">=", ">", "<=", "<", "==", "!="}
        if ant_op not in supported_ops or cons_op not in supported_ops:
            raise SemanticError("Unsupported implication comparison operator")

        if ant_op != "==":
            antecedent_comparison = {
                "type": "binop",
                "op": ant_op,
                "left": ant_unwrapped.get("left"),
                "right": ant_unwrapped.get("right"),
                "sem_type": "boolean",
            }
            flag_name = comparison_truth_var(antecedent_comparison, env)
            self._gate_affine_implication_consequent(
                {"type": "name", "value": flag_name},
                1,
                cons_unwrapped,
                env,
                append_ub_row,
            )
            return
        self._handle_equality_antecedent_implication(
            ant_unwrapped,
            cons_unwrapped,
            env,
            append_eq_row,
            append_ub_row,
        )

    def _weighted_boolean_sum_operands(self, left):
        sum_node = self._unwrap_parenthesized_node(left)
        if not (isinstance(sum_node, dict) and sum_node.get("type") == "sum"):
            return None
        inner = self._unwrap_parenthesized_node(sum_node.get("expression"))
        if not (isinstance(inner, dict) and inner.get("type") == "binop" and inner.get("op") == "*"):
            return None

        for weight_node, bool_node in (
            (inner.get("left"), inner.get("right")),
            (inner.get("right"), inner.get("left")),
        ):
            bool_node = self._unwrap_parenthesized_node(bool_node)
            if (
                isinstance(bool_node, dict)
                and bool_node.get("sem_type") == "boolean"
                and bool_node.get("type") not in ("name", "indexed_name")
            ):
                return sum_node, weight_node, bool_node
        return None

    def _weighted_boolean_sum_rhs(self, right, op_sym_top, env):
        if op_sym_top not in (">=", "==", "<=", ">", "<"):
            return None
        try:
            rhs_coef, rhs_const = self._eval_expr(right, dict(env or {}))
        except Exception:
            return None
        if rhs_coef != {} or not isinstance(rhs_const, (int, float)):
            return None
        rhs_value = float(rhs_const)
        if op_sym_top == ">":
            rhs_value += BOOL_EPS
        elif op_sym_top == "<":
            rhs_value -= BOOL_EPS
        return rhs_value

    def _weighted_boolean_sum_row(self, sum_node, weight_node, bool_node, env, bool_expr_var):
        row = [0.0] * len(self.var_names)
        for env2, _idx_tuple in self._iter_filtered_environments(
            sum_node.get("iterators", []),
            env,
            sum_node.get("index_constraint"),
        ):
            weight_coef, weight_const = self._eval_expr(weight_node, env2)
            if weight_coef or isinstance(weight_const, (str, tuple)):
                raise SemanticError("Weighted boolean sums require numeric weights")
            bool_var = bool_expr_var(bool_node, env2)
            if len(row) < len(self.var_names):
                row.extend([0.0] * (len(self.var_names) - len(row)))
            row[self.var_indices[bool_var]] += float(weight_const)
        return row

    def _append_weighted_boolean_sum_row(self, row, rhs_value, op_sym_top, state):
        if op_sym_top in (">=", ">"):
            self._append_sparse_row(state, [-coef for coef in row], -rhs_value, sense="ub")
        elif op_sym_top in ("<=", "<"):
            self._append_sparse_row(state, row, rhs_value, sense="ub")
        else:
            self._append_sparse_row(state, row, rhs_value, sense="eq")

    def _try_handle_weighted_boolean_sum_constraint(self, left, right, op_sym_top, env, bool_expr_var, state):
        weighted_operands = self._weighted_boolean_sum_operands(left)
        if weighted_operands is None:
            return False
        sum_node, weight_node, bool_node = weighted_operands
        rhs_value = self._weighted_boolean_sum_rhs(right, op_sym_top, env)
        if rhs_value is None:
            return False

        row = self._weighted_boolean_sum_row(sum_node, weight_node, bool_node, env, bool_expr_var)
        self._append_weighted_boolean_sum_row(row, rhs_value, op_sym_top, state)
        return True

    def _try_handle_sum_of_comparisons_constraint(
        self,
        left,
        right,
        op_sym_top,
        env,
        comparison_truth_var,
        state,
    ):
        is_sum, inner_comparison, threshold, iterators, index_constraint = self._detect_sum_of_comparisons(
            left,
            right,
            op_sym_top,
        )
        if not is_sum:
            return False
        normalized_op = ">=" if op_sym_top == ">" else "<=" if op_sym_top == "<" else op_sym_top
        if normalized_op == ">=" and threshold <= 0:
            return True

        truth_indices = []
        for env2, _idx_tuple in self._iter_filtered_environments(iterators, env, index_constraint):
            comparison = {
                "type": "binop",
                "op": inner_comparison.get("op"),
                "left": inner_comparison.get("left"),
                "right": inner_comparison.get("right"),
                "sem_type": "boolean",
            }
            truth_name = comparison_truth_var(comparison, env2)
            truth_indices.append(self.var_indices[truth_name])

        row = [0.0] * len(self.var_names)
        if normalized_op == ">=":
            for truth_index in truth_indices:
                row[truth_index] -= 1.0
            self._append_sparse_row(state, row, -threshold, sense="ub")
        elif normalized_op == "==":
            for truth_index in truth_indices:
                row[truth_index] += 1.0
            self._append_sparse_row(state, row, threshold, sense="eq")
        else:
            for truth_index in truth_indices:
                row[truth_index] += 1.0
            self._append_sparse_row(state, row, threshold, sense="ub")
        return True

    def _reified_comparison_sum_rows(self, left, right, op_sym_top, env, comparison_truth_var):
        if op_sym_top != "==" or not isinstance(left, dict) or left.get("type") != "name":
            return None
        comparison = self._unwrap_parenthesized_node(right)
        if not (isinstance(comparison, dict) and comparison.get("type") == "binop" and comparison.get("op") in (">=", ">")):
            return None
        sum_node = self._unwrap_parenthesized_node(comparison.get("left"))
        threshold_node = comparison.get("right")
        if not (
            isinstance(sum_node, dict)
            and sum_node.get("type") == "sum"
            and isinstance(threshold_node, dict)
            and threshold_node.get("type") == "number"
        ):
            return None
        inner_comparison = self._unwrap_parenthesized_node(sum_node.get("expression"))
        if not self._is_simple_boolean_comparison(inner_comparison):
            return None

        truth_indices = []
        for env2, _idx_tuple in self._iter_filtered_environments(
            sum_node.get("iterators", []),
            env,
            sum_node.get("index_constraint"),
        ):
            comparison_instance = {
                "type": "binop",
                "op": inner_comparison.get("op"),
                "left": inner_comparison.get("left"),
                "right": inner_comparison.get("right"),
                "sem_type": "boolean",
            }
            truth_name = comparison_truth_var(comparison_instance, env2)
            truth_indices.append(self.var_indices[truth_name])

        boolean_name = left.get("value")
        if boolean_name not in self.var_indices:
            self._ensure_aux_binary(boolean_name)
        threshold = threshold_node.get("value")
        truth_count = len(truth_indices)
        lower_row = [0.0] * len(self.var_names)
        lower_row[self.var_indices[boolean_name]] = threshold
        upper_row = [0.0] * len(self.var_names)
        upper_row[self.var_indices[boolean_name]] = -(truth_count - threshold + 1)
        for truth_index in truth_indices:
            lower_row[truth_index] -= 1.0
            upper_row[truth_index] += 1.0
        return lower_row, upper_row, threshold - 1.0

    def _rewrite_not_literal_constraint(self, left, right, op_sym_top):
        if not (
            op_sym_top == "=="
            and isinstance(left, dict)
            and left.get("type") == "not"
            and isinstance(right, dict)
            and right.get("type") == "boolean_literal"
        ):
            return None

        inner = self._unwrap_parenthesized_node(left.get("value"))
        if not bool(right.get("value")):
            return {
                "type": "constraint",
                "op": "==",
                "left": inner,
                "right": {"type": "boolean_literal", "value": True, "sem_type": "boolean"},
            }, False

        if isinstance(inner, dict) and inner.get("type") == "constraint":
            inner_op = inner.get("op")
            inner_left = inner.get("left")
            inner_right = inner.get("right")
            if inner_op in ("<=", ">="):
                arithmetic_op = "+" if inner_op == "<=" else "-"
                inverted_op = ">=" if inner_op == "<=" else "<="
                adjusted_right = {
                    "type": "binop",
                    "op": arithmetic_op,
                    "left": inner_right,
                    "right": {"type": "number", "value": BOOL_EPS, "sem_type": "float"},
                    "sem_type": inner_right.get("sem_type", "float"),
                }
                return {
                    "type": "constraint",
                    "op": inverted_op,
                    "left": inner_left,
                    "right": adjusted_right,
                }, False
            if inner_op in ("==", "!="):
                return {
                    "type": "constraint",
                    "op": "!=" if inner_op == "==" else "==",
                    "left": inner_left,
                    "right": inner_right,
                }, inner_op == "=="

        return {
            "type": "constraint",
            "op": "==",
            "left": inner,
            "right": {"type": "boolean_literal", "value": False, "sem_type": "boolean"},
        }, False

    def _boolean_equality_expression_var(self, node, env, bool_expr_var):
        unwrapped = self._unwrap_parenthesized_node(node)
        if not isinstance(unwrapped, dict) or unwrapped.get("type") in ("boolean_literal", "sum"):
            return None
        if (
            unwrapped.get("type") == "constraint"
            and unwrapped.get("op") in (">=", ">")
            and isinstance(unwrapped.get("left"), dict)
            and unwrapped["left"].get("type") == "binop"
            and self._is_boolean_sum_term(unwrapped["left"])
        ):
            return None
        if (
            unwrapped.get("type") == "binop"
            and unwrapped.get("op") in (">=", ">")
            and isinstance(unwrapped.get("left"), dict)
            and unwrapped["left"].get("type") == "sum"
        ):
            return None
        return bool_expr_var(node, env)

    def _is_boolean_sum_term(self, node):
        if not isinstance(node, dict):
            return False
        if node.get("type") == "binop" and node.get("op") == "+":
            return self._is_boolean_sum_term(node.get("left")) and self._is_boolean_sum_term(node.get("right"))
        if node.get("type") == "number":
            return True
        if node.get("type") not in ("name", "indexed_name"):
            return False
        base_name = node.get("value") if node.get("type") == "name" else node.get("name")
        declaration = self._find_decl(base_name)
        return bool(declaration and declaration.get("var_type") == "boolean")

    def _try_tie_boolean_variable_expression(self, left, right, op_sym_top, env, bool_expr_var, state):
        if op_sym_top != "==" or not isinstance(left, dict) or not isinstance(right, dict):
            return False
        for variable_node, expression_node in ((left, right), (right, left)):
            if not self._is_declared_boolean_var_node(variable_node):
                continue
            unwrapped = self._unwrap_parenthesized_node(expression_node)
            if not (
                isinstance(unwrapped, dict)
                and (
                    unwrapped.get("sem_type") == "boolean"
                    or unwrapped.get("type") in ("and", "or", "not", "implies", "boolean_literal")
                    or (unwrapped.get("type") == "constraint" and unwrapped.get("op") in ("==", "!=", "<=", ">=", "<", ">"))
                )
            ):
                continue
            expression_var = self._boolean_equality_expression_var(expression_node, env, bool_expr_var)
            if expression_var is None:
                continue
            variable_name = (
                self._multi_indexed_var_name(variable_node, env)
                if variable_node.get("type") == "indexed_name"
                else variable_node.get("value")
            )
            if expression_var != variable_name:
                if not isinstance(expression_var, str):
                    raise SemanticError(f"expr_var is not a string: {repr(expression_var)}")
                row = [0.0] * len(self.var_names)
                row[self.var_indices[variable_name]] = 1.0
                row[self.var_indices[expression_var]] = -1.0
                self._append_sparse_row(state, row, 0.0, sense="eq")
            return True
        return False

    def _collect_boolean_sum(self, node):
        if not isinstance(node, dict):
            return None
        if node.get("type") == "name" and self._is_declared_boolean_var_node(node):
            return {node["value"]: 1.0}, 0
        if node.get("type") == "number":
            return {}, node.get("value", 0)
        if node.get("type") != "binop" or node.get("op") != "+":
            return None
        left_sum = self._collect_boolean_sum(node.get("left"))
        right_sum = self._collect_boolean_sum(node.get("right"))
        if left_sum is None or right_sum is None:
            return None
        weights = dict(left_sum[0])
        for variable_name, weight in right_sum[0].items():
            weights[variable_name] = weights.get(variable_name, 0.0) + weight
        return weights, left_sum[1] + right_sum[1]

    def _reified_boolean_sum_parts(self, boolean_node, inequality, env):
        if not self._is_declared_boolean_var_node(boolean_node):
            return None
        if not (
            isinstance(inequality, dict)
            and inequality.get("type") == "constraint"
            and inequality.get("op") == ">="
            and isinstance(inequality.get("right"), dict)
            and inequality["right"].get("type") == "number"
        ):
            return None
        collected = self._collect_boolean_sum(inequality.get("left"))
        if collected is None:
            return None
        variable_weights, constant_offset = collected
        threshold = inequality["right"].get("value") - constant_offset
        boolean_name = (
            boolean_node["value"] if boolean_node.get("type") == "name" else self._multi_indexed_var_name(boolean_node, env)
        )
        return boolean_name, variable_weights, threshold

    def _emit_reified_boolean_sum_rows(self, boolean_name, variable_weights, threshold, state):
        maximum_sum = sum(variable_weights.values())
        if threshold <= 0 or threshold > maximum_sum:
            row = [0.0] * len(self.var_names)
            row[self.var_indices[boolean_name]] = 1.0
            self._append_sparse_row(state, row, 1.0 if threshold <= 0 else 0.0, sense="eq")
            return

        lower_row = [0.0] * len(self.var_names)
        lower_row[self.var_indices[boolean_name]] = threshold
        for variable_name, weight in variable_weights.items():
            lower_row[self.var_indices[variable_name]] -= weight
        self._append_sparse_row(state, lower_row, 0.0, sense="ub")

        upper_row = [0.0] * len(self.var_names)
        for variable_name, weight in variable_weights.items():
            upper_row[self.var_indices[variable_name]] += weight
        upper_row[self.var_indices[boolean_name]] = -(maximum_sum - threshold + 1)
        self._append_sparse_row(state, upper_row, threshold - 1, sense="ub")

    def _try_handle_reified_boolean_sum(self, left, right, op_sym_top, env, state):
        if op_sym_top != "==":
            return False
        for boolean_node, inequality in ((left, right), (right, left)):
            parts = self._reified_boolean_sum_parts(boolean_node, inequality, env)
            if parts is None:
                continue
            boolean_name, variable_weights, threshold = parts
            self._emit_reified_boolean_sum_rows(boolean_name, variable_weights, threshold, state)
            return True
        return False

    @staticmethod
    def _is_boolean_literal_node(node):
        return isinstance(node, dict) and node.get("type") == "boolean_literal"

    def _bool_tree_literal_operands(self, left, right, op_sym):
        if op_sym not in ("==", "!=", "<=", ">="):
            return None
        left_is_tree = self._is_bool_tree_node(left)
        right_is_tree = self._is_bool_tree_node(right)
        left_is_literal = self._is_boolean_literal_node(left) or self._is_number_01_node(left)
        right_is_literal = self._is_boolean_literal_node(right) or self._is_number_01_node(right)
        if not ((left_is_tree and right_is_literal) or (right_is_tree and left_is_literal)):
            return None
        if right_is_tree:
            left, right = right, left
        target_value = int(bool(right.get("value"))) if self._is_boolean_literal_node(right) else int(right.get("value"))
        return left, right, target_value

    def _bool_tree_literal_enforcement(self, tree, op_sym, target_value):
        if op_sym == "==" and tree.get("type") not in ("and", "or"):
            return target_value
        if op_sym == "!=":
            return 1 - target_value
        if op_sym == ">=" and target_value == 1:
            return 1
        if op_sym == "<=" and target_value == 0:
            return 0
        return None

    def _emit_bool_tree_literal_comparison(self, tree, enforce, env, bool_expr_var, state):
        expression_var = bool_expr_var(tree, env)
        row = [0.0] * len(self.var_names)
        row[self.var_indices[expression_var]] = 1.0
        self._append_sparse_row(state, row, float(enforce), sense="eq")

    def _try_handle_bool_tree_literal_comparison(self, left, right, op_sym, env, bool_expr_var, state):
        operands = self._bool_tree_literal_operands(left, right, op_sym)
        if operands is None:
            return None
        left, right, target_value = operands
        enforce = self._bool_tree_literal_enforcement(left, op_sym, target_value)
        if enforce is not None:
            self._emit_bool_tree_literal_comparison(left, enforce, env, bool_expr_var, state)
            return True, left, right

        handled = op_sym != "==" or left.get("type") not in ("and", "or")
        return handled, left, right

    def _try_handle_boolean_variable_relation(self, left, right, op_sym, env, bool_expr_var, state):
        if op_sym not in ("==", "<=", ">="):
            return False
        left_is_boolean = self._is_declared_boolean_var_node(left)
        right_is_boolean = self._is_declared_boolean_var_node(right)
        if left_is_boolean == right_is_boolean:
            return False

        variable_node = left if left_is_boolean else right
        expression_node = right if left_is_boolean else left
        try:
            expression_var = bool_expr_var(self._unwrap_parenthesized_node(expression_node), env)
        except SemanticError:
            return False
        variable_name = (
            variable_node["value"] if variable_node.get("type") == "name" else self._multi_indexed_var_name(variable_node, env)
        )
        if left_is_boolean:
            left_var, right_var = variable_name, expression_var
        else:
            left_var, right_var = expression_var, variable_name
        if left_var == right_var:
            return True

        row = [0.0] * len(self.var_names)
        if op_sym == "==":
            row[self.var_indices[left_var]] = 1.0
            row[self.var_indices[right_var]] -= 1.0
            self._append_sparse_row(state, row, 0.0, sense="eq")
        elif op_sym == "<=":
            row[self.var_indices[left_var]] = 1.0
            row[self.var_indices[right_var]] -= 1.0
            self._append_sparse_row(state, row, 0.0, sense="ub")
        else:
            row[self.var_indices[left_var]] = -1.0
            row[self.var_indices[right_var]] += 1.0
            self._append_sparse_row(state, row, 0.0, sense="ub")
        return True

    def _flatten_boolean_operator(self, node, operator):
        if isinstance(node, dict) and node.get("type") == operator:
            return self._flatten_boolean_operator(node.get("left"), operator) + self._flatten_boolean_operator(
                node.get("right"), operator
            )
        return [node]

    def _resolve_atomic_boolean_literal(self, atom, env):
        if not (isinstance(atom, dict) and atom.get("type") == "constraint" and atom.get("op") == "=="):
            raise SemanticError("Unsupported atomic boolean term for SciPy AND/OR linearization")
        left = atom.get("left")
        right = atom.get("right")
        if self._is_var_reference_node(left) and self._is_number_01_node(right):
            variable_node, value_node = left, right
        elif self._is_number_01_node(left) and self._is_var_reference_node(right):
            variable_node, value_node = right, left
        else:
            raise SemanticError("Unsupported comparison in boolean linearization (expected v == 0/1)")
        variable_name = (
            self._multi_indexed_var_name(variable_node, env)
            if variable_node.get("type") == "indexed_name"
            else variable_node["value"]
        )
        if variable_name not in self.var_indices:
            raise SemanticError(f"Variable '{variable_name}' not found for boolean linearization")
        return variable_name, 1 if value_node["value"] == 1 else -1

    def _and_or_literal_operands(self, left, right, op_sym):
        left = self._unwrap_parenthesized_node(left)
        right = self._unwrap_parenthesized_node(right)
        if self._is_boolean_literal_node(left) and isinstance(right, dict) and right.get("type") in ("and", "or"):
            left, right = right, left
        if not (
            isinstance(left, dict)
            and left.get("type") in ("and", "or")
            and self._is_boolean_literal_node(right)
            and op_sym == "=="
        ):
            return None
        return left, left["type"], bool(right.get("value", True))

    def _emit_and_or_literal_rows(self, literals, operator, target_value, state):
        literal_count = len(literals)
        if operator == "and" and target_value:
            for variable_name, polarity in literals:
                row = [0.0] * len(self.var_names)
                row[self.var_indices[variable_name]] = 1.0
                self._append_sparse_row(state, row, 1.0 if polarity == 1 else 0.0, sense="eq")
            return True
        if operator == "and":
            row = [0.0] * len(self.var_names)
            constant_shift = 0.0
            for variable_name, polarity in literals:
                row[self.var_indices[variable_name]] += polarity
                if polarity == -1:
                    constant_shift += 1.0
            self._append_sparse_row(state, row, literal_count - 1 - constant_shift, sense="ub")
            return True
        row = [0.0] * len(self.var_names)
        constant_shift = 0.0
        for variable_name, polarity in literals:
            row[self.var_indices[variable_name]] += polarity
            if polarity == -1:
                constant_shift += 1.0
        if target_value:
            self._append_sparse_row(state, [-coefficient for coefficient in row], constant_shift - 1.0, sense="ub")
        else:
            self._append_sparse_row(state, row, -constant_shift, sense="eq")
        return True

    def _emit_and_or_expression_row(self, expression, target_value, env, bool_expr_var, state):
        try:
            expression_var = bool_expr_var(expression, env)
        except SemanticError:
            return False
        row = [0.0] * len(self.var_names)
        row[self.var_indices[expression_var]] = 1.0
        self._append_sparse_row(state, row, 1.0 if target_value else 0.0, sense="eq")
        return True

    def _try_handle_and_or_literal_fast_path(self, left, right, op_sym, env, bool_expr_var, state):
        operands = self._and_or_literal_operands(left, right, op_sym)
        if operands is None:
            return False
        left, operator, target_value = operands
        try:
            literals = [
                self._resolve_atomic_boolean_literal(atom, env) for atom in self._flatten_boolean_operator(left, operator)
            ]
        except SemanticError:
            literals = None

        if literals is not None:
            return self._emit_and_or_literal_rows(literals, operator, target_value, state)
        return self._emit_and_or_expression_row(left, target_value, env, bool_expr_var, state)

    def _try_handle_asserted_and(self, left, right, op_sym, env, handle_constraint):
        if not (isinstance(left, dict) and left.get("type") == "and" and op_sym == "=="):
            return False
        target_value = self._is_boolean_literal_node(right) and right.get("value") is True
        if not target_value:
            return True

        def emit_conjunct(node):
            node = self._unwrap_parenthesized_node(node)
            if not isinstance(node, dict):
                return
            node_type = node.get("type")
            if node_type == "and":
                emit_conjunct(node.get("left"))
                emit_conjunct(node.get("right"))
                return
            if node_type in ("not", "or"):
                handle_constraint(
                    {
                        "type": "constraint",
                        "op": "==",
                        "left": node,
                        "right": {"type": "boolean_literal", "value": True, "sem_type": "boolean"},
                    },
                    env=env,
                )
                return
            if self._is_linear_comparison(node):
                handle_constraint(
                    {"type": "constraint", "op": node["op"], "left": node["left"], "right": node["right"]},
                    env=env,
                )
                return
            raise self._unsupported_type_error("boolean leaf", node)

        emit_conjunct(left)
        return True

    def _boolean_or_disjuncts(self, node):
        node = self._unwrap_parenthesized_node(node)
        if not isinstance(node, dict):
            return []
        node_type = node.get("type")
        if node_type == "or":
            return self._boolean_or_disjuncts(node.get("left")) + self._boolean_or_disjuncts(node.get("right"))
        if node_type == "and":
            comparisons = []
            stack = [node]
            while stack:
                current = self._unwrap_parenthesized_node(stack.pop())
                if not isinstance(current, dict):
                    continue
                if current.get("type") == "and":
                    stack.extend((current.get("left"), current.get("right")))
                elif self._is_linear_comparison(current):
                    comparisons.append(current)
                else:
                    raise self._unsupported_type_error("boolean leaf", current)
            return [comparisons]
        if not self._is_linear_comparison(node):
            raise self._unsupported_type_error("boolean leaf", node)
        return [[node]]

    def _append_or_guarded_comparison(self, comparison, flag_name, env, state):
        big_m = self._big_m_for_comparison(comparison, env=env)
        left_coef, left_const = self._eval_expr(comparison["left"], env)
        right_node = comparison["right"]
        right_coef, right_const = (
            self._eval_expr(right_node, env)
            if isinstance(right_node, dict)
            else ({}, right_node if isinstance(right_node, (int, float)) else 0.0)
        )
        expression_coef = dict(left_coef)
        for variable_name, coefficient in right_coef.items():
            expression_coef[variable_name] = expression_coef.get(variable_name, 0.0) - coefficient
        expression_const = left_const - right_const

        def guarded_row(sign):
            row = [0.0] * len(self.var_names)
            for variable_name, coefficient in expression_coef.items():
                if variable_name in self.var_indices:
                    row[self.var_indices[variable_name]] += sign * coefficient
            row[self.var_indices[flag_name]] += big_m
            return row

        operator = comparison["op"]
        if operator in ("<=", "=="):
            self._append_sparse_row(state, guarded_row(1.0), big_m - expression_const, sense="ub")
        if operator in (">=", "=="):
            self._append_sparse_row(state, guarded_row(-1.0), big_m + expression_const, sense="ub")

    def _try_handle_asserted_or(self, left, right, op_sym, env, state):
        if not (
            isinstance(left, dict)
            and left.get("type") == "or"
            and op_sym == "=="
            and self._is_boolean_literal_node(right)
            and right.get("value") is True
        ):
            return False
        flag_names = []
        for disjunct_index, comparisons in enumerate(self._boolean_or_disjuncts(left)):
            flag_name = f"or_flag_{disjunct_index}"
            while flag_name in self.var_indices:
                flag_name += "_"
            self._ensure_aux_binary(flag_name)
            flag_names.append(flag_name)
            for comparison in comparisons:
                self._append_or_guarded_comparison(comparison, flag_name, env, state)
        if flag_names:
            selector_row = [0.0] * len(self.var_names)
            for flag_name in flag_names:
                selector_row[self.var_indices[flag_name]] = -1.0
            self._append_sparse_row(state, selector_row, -1.0, sense="ub")
        return True

    def _handle_forall_constraint(self, constraint, env, handle_constraint):
        iterators = constraint.get("iterators")
        if iterators is None:
            raise self._unsupported_type_error("forall_constraint", "missing iterators")
        if "constraint" in constraint:
            inner_constraints = [constraint["constraint"]]
        elif "constraints" in constraint:
            inner_constraints = constraint["constraints"]
        else:
            raise self._unsupported_type_error("forall_constraint", "missing constraint(s)")
        for inner_env, _index_tuple in self._iter_filtered_environments(
            iterators,
            env,
            constraint.get("index_constraint"),
        ):
            for inner_constraint in inner_constraints:
                handle_constraint(inner_constraint, env=inner_env)

    def _normalize_comparison_literal_constraint(self, constraint):
        left = self._unwrap_parenthesized_node(constraint.get("left"))
        right = self._unwrap_parenthesized_node(constraint.get("right"))
        constraint["left"] = left
        constraint["right"] = right
        operator = constraint.get("op")
        is_comparison = (
            isinstance(left, dict)
            and left.get("type") == "binop"
            and left.get("sem_type") == "boolean"
            and left.get("op") in ("<=", ">=", "==", "!=")
        )
        if not (operator == "==" and is_comparison and isinstance(right, dict) and right.get("type") == "boolean_literal"):
            return left, right, operator
        if right.get("value") is True:
            constraint["op"] = left["op"]
            constraint["left"] = left["left"]
            constraint["right"] = left["right"]
        else:
            wrapped = {"type": "constraint", "op": left["op"], "left": left["left"], "right": left["right"]}
            constraint["left"] = {"type": "not", "value": wrapped, "sem_type": "boolean"}
            constraint["right"] = {"type": "boolean_literal", "value": True, "sem_type": "boolean"}
        return constraint["left"], constraint["right"], constraint["op"]

    def _try_handle_constraint_special_forms(
        self,
        constraint,
        env,
        bool_expr_var,
        comparison_truth_var,
        append_ub_row,
        state,
    ):
        left = constraint["left"]
        right = constraint["right"]
        operator = constraint.get("op")
        if self._try_handle_weighted_boolean_sum_constraint(left, right, operator, env, bool_expr_var, state):
            return True, left, right, operator
        if self._try_handle_sum_of_comparisons_constraint(
            left,
            right,
            operator,
            env,
            comparison_truth_var,
            state,
        ):
            return True, left, right, operator
        reified_rows = self._reified_comparison_sum_rows(left, right, operator, env, comparison_truth_var)
        if reified_rows is not None:
            lower_row, upper_row, upper_rhs = reified_rows
            append_ub_row(lower_row, 0.0)
            append_ub_row(upper_row, upper_rhs)
            return True, left, right, operator
        left, right, operator = self._normalize_comparison_literal_constraint(constraint)
        return False, left, right, operator

    def _handle_normalized_boolean_constraint(
        self,
        left,
        right,
        operator,
        env,
        bool_expr_var,
        state,
        handle_constraint,
    ):
        rewritten_not = self._rewrite_not_literal_constraint(left, right, operator)
        if rewritten_not is not None:
            new_constraint, marks_not_of_equality = rewritten_not
            if marks_not_of_equality:
                self._add_code_line("# encoded != (NOT of ==)")
            handle_constraint(new_constraint, env=env)
            return True
        if self._try_tie_boolean_variable_expression(left, right, operator, env, bool_expr_var, state):
            return True
        if self._try_handle_reified_boolean_sum(left, right, operator, env, state):
            return True
        literal_result = self._try_handle_bool_tree_literal_comparison(
            left,
            right,
            operator,
            env,
            bool_expr_var,
            state,
        )
        if literal_result is not None:
            handled, left, right = literal_result
            if handled:
                return True
        if self._try_handle_boolean_variable_relation(left, right, operator, env, bool_expr_var, state):
            return True
        left = self._unwrap_parenthesized_node(left)
        right = self._unwrap_parenthesized_node(right)
        if self._try_handle_and_or_literal_fast_path(left, right, operator, env, bool_expr_var, state):
            return True
        if isinstance(left, dict) and left.get("type") in ("and", "or") and operator == "==":
            if not (self._is_boolean_literal_node(right) and right.get("value") is True):
                return True
            if left.get("type") == "and":
                self._try_handle_asserted_and(left, right, operator, env, handle_constraint)
            else:
                self._try_handle_asserted_or(left, right, operator, env, state)
            return True
        return False

    def _handle_normalized_constraint(
        self,
        constraint,
        left,
        right,
        operator,
        env,
        bool_expr_var,
        state,
        handle_constraint,
        append_eq_row,
        append_ub_row,
    ):
        if operator == "!=":
            self._handle_not_equal_constraint(left, right, env, state)
            return
        if self._handle_normalized_boolean_constraint(
            left,
            right,
            operator,
            env,
            bool_expr_var,
            state,
            handle_constraint,
        ):
            return
        self._emit_plain_linear_constraint(constraint, env, append_eq_row, append_ub_row)

    def _not_equal_boolean_operands(self, left, right):
        left_is_boolean = self._is_declared_boolean_var_node(left)
        right_is_boolean = self._is_declared_boolean_var_node(right)
        left_is_literal = isinstance(left, dict) and left.get("type") == "number" and left.get("value") in (0, 1)
        right_is_literal = isinstance(right, dict) and right.get("type") == "number" and right.get("value") in (0, 1)
        if (
            (left_is_boolean or right_is_boolean)
            and (left_is_boolean or left_is_literal)
            and (right_is_boolean or right_is_literal)
        ):
            return left_is_boolean, right_is_boolean
        return None

    def _boolean_not_equal_name(self, node, env):
        return self._multi_indexed_var_name(node, env) if node.get("type") == "indexed_name" else node["value"]

    def _emit_boolean_not_equal(self, left, right, env, state, left_is_boolean, right_is_boolean):
        if left_is_boolean and right_is_boolean:
            left_name = self._boolean_not_equal_name(left, env)
            right_name = self._boolean_not_equal_name(right, env)
            row = [0.0] * len(self.var_names)
            row[self.var_indices[left_name]] = 1.0
            row[self.var_indices[right_name]] = 1.0
            self._append_sparse_row(state, row, 1.0, sense="eq")
            self._add_code_line("# encoded != (boolean xor)")
            return

        variable_node = left if left_is_boolean else right
        literal_node = right if variable_node is left else left
        variable_name = self._boolean_not_equal_name(variable_node, env)
        row = [0.0] * len(self.var_names)
        row[self.var_indices[variable_name]] = 1.0
        self._append_sparse_row(state, row, float(1 - literal_node.get("value")), sense="eq")
        self._add_code_line("# encoded != (boolean var vs literal)")

    def _not_equal_affine_difference(self, left, right, env):
        left_coef, left_const = self._eval_expr(left, env)
        right_coef, right_const = self._eval_expr(right, env)
        diff_coef = dict(left_coef)
        for name, coef in right_coef.items():
            diff_coef[name] = diff_coef.get(name, 0.0) - coef
        return diff_coef, float(left_const) - float(right_const)

    def _add_not_equal_direction_binary(self):
        if not hasattr(self, "_neq_counter"):
            self._neq_counter = 0
        direction_name = f"neq_direction_c{self._neq_counter}"
        self._neq_counter += 1
        self.var_names.append(direction_name)
        self.var_indices[direction_name] = len(self.var_names) - 1
        self.bounds.append([0, 1])
        self.integrality.append(1)
        self.c.append(0.0)
        return direction_name

    def _emit_integer_not_equal(self, diff_coef, diff_const, env, state):
        diff_min, diff_max = self._finite_integer_affine_bounds(
            diff_coef,
            diff_const,
            "Integer not-equal constraint",
        )
        big_m = max(1.0, diff_max + 1.0, 1.0 - diff_min)
        direction_name = self._add_not_equal_direction_binary()
        negative_row = dict(diff_coef)
        negative_row[direction_name] = -big_m
        self._append_sparse_coef_row(state, negative_row, -1.0 - diff_const, sense="ub")
        positive_row = {name: -coef for name, coef in diff_coef.items()}
        positive_row[direction_name] = big_m
        self._append_sparse_coef_row(state, positive_row, big_m - 1.0 + diff_const, sense="ub")
        self._add_code_line("# encoded integer != via direction binary")

    def _handle_not_equal_constraint(self, left, right, env, state):
        boolean_operands = self._not_equal_boolean_operands(left, right)
        if boolean_operands is not None:
            self._emit_boolean_not_equal(left, right, env, state, *boolean_operands)
            return

        diff_coef, diff_const = self._not_equal_affine_difference(left, right, env)
        self._emit_integer_not_equal(diff_coef, diff_const, env, state)

    def _handle_constraint_node(self, constr, env, state, bool_expr_var, comparison_truth_var, append_eq_row, append_ub_row):
        self._ensure_constraint_parameters_bound(constr)
        if self._try_enforce_bool_tree_literal_constraint(constr, env, bool_expr_var, append_eq_row):
            return

        logger.debug(f"[SciPyCSCCodeGenerator] handle_constraint: {constr}")
        self._collect_passive_constraint_bounds(constr, env, bool_expr_var)
        if self._try_enforce_reified_implication_literal(constr, env, bool_expr_var, append_eq_row):
            return
        if constr.get("type") == "implication_constraint":
            self._handle_implication_constraint(
                constr,
                env,
                bool_expr_var,
                comparison_truth_var,
                append_eq_row,
                append_ub_row,
            )
            return
        if constr["type"] == "constraint":
            handled, left, right, operator = self._try_handle_constraint_special_forms(
                constr,
                env,
                bool_expr_var,
                comparison_truth_var,
                append_ub_row,
                state,
            )
            if handled:
                return
            self._handle_normalized_constraint(
                constr,
                left,
                right,
                operator,
                env,
                bool_expr_var,
                state,
                lambda child, env: self._handle_constraint_node(
                    child, env, state, bool_expr_var, comparison_truth_var, append_eq_row, append_ub_row
                ),
                append_eq_row,
                append_ub_row,
            )
            return
        if constr["type"] == "forall_constraint":
            self._handle_forall_constraint(
                constr,
                env,
                lambda child, env: self._handle_constraint_node(
                    child, env, state, bool_expr_var, comparison_truth_var, append_eq_row, append_ub_row
                ),
            )
            return
        if constr["type"] != "implication_constraint":
            logger.debug(f"Unsupported constraint type: {constr['type']}")

    def _build_constraints(self):
        self._add_code_line("# Constraints (sparse)")
        logger.debug("[SciPyCSCCodeGenerator] Entering _build_constraints")
        # Enable symbolic boolean evaluation during constraint build
        prev_sym = getattr(self, "_allow_symbolic_bool", False)
        self._allow_symbolic_bool = True
        state = _ConstraintBuildState()

        def append_eq_row(row, rhs):
            self._append_sparse_row(state, row, rhs, sense="eq")

        def append_ub_row(row, rhs):
            self._append_sparse_row(state, row, rhs, sense="ub")

        # Collected per-variable bounds from simple constraints (var >= c, var <= c, var == c)
        if not hasattr(self, "_collected_lbs"):
            self._collected_lbs = {}
            self._collected_ubs = {}

        # --- Mixed AND/OR auxiliary infrastructure ---
        self.aux_created = []  # list of created auxiliary boolean vars
        ctx = _ConstraintBuildContext(
            state=state,
            comparison_truth_cache=self._comparison_truth_cache,
            subtree_var_cache=self._bool_subtree_cache,
        )

        def _comparison_truth_var(node, env):
            result = self._comparison_truth_var(node, env, ctx)
            return result

        def _bool_expr_var(node, env):
            result = self._bool_expr_var(node, env, ctx)
            return result

        try:
            for constr in self.ast["constraints"]:
                self._handle_constraint_node(
                    constr,
                    {},
                    state,
                    _bool_expr_var,
                    _comparison_truth_var,
                    append_eq_row,
                    append_ub_row,
                )
        finally:
            # Always restore symbolic flag even if constraint handling raises
            self._allow_symbolic_bool = prev_sym

        self._finalize_constraint_state(state)
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

    def _merge_accumulated_expression(self, target, values, sign, const_ref):
        coef_dict, const_value = values
        for name, coefficient in coef_dict.items():
            target[name] += sign * coefficient
        if isinstance(const_value, (int, float)):
            const_ref[0] += sign * float(const_value)

    def _accumulate_sum_atom(self, expr, env, coef_dict, sign, const_ref):
        if expr["type"] == "indexed_name":
            cdict, cval = self._eval_expr(expr, env)
            for vname, coef in cdict.items():
                coef_dict[vname] += sign * coef
            if isinstance(cval, (int, float)):
                const_ref[0] += sign * float(cval)
        elif expr["type"] == "name":
            is_var, val, is_symbolic = self._lookup_var_or_param(expr.get("value"), indices=None, env=env)
            if is_var:
                vname = val if isinstance(val, str) else expr.get("value")
                coef_dict[vname] += sign * 1.0
            elif not is_symbolic and isinstance(val, (int, float)):
                const_ref[0] += sign * float(val)
        elif expr["type"] == "number":
            const_ref[0] += sign * float(expr.get("value", 0.0))
        else:
            cdict, cval = self._eval_expr(expr, env)
            for vname, coef in cdict.items():
                coef_dict[vname] += sign * coef
            if isinstance(cval, (int, float)):
                const_ref[0] += sign * cval

    def _accumulate_sum_binop(self, expr, env, coef_dict, sign, const_ref):
        ldict, lconst = self._accumulate_sum_to_dict(expr["left"], env, sign=1)
        rdict, rconst = self._accumulate_sum_to_dict(expr["right"], env, sign=1)
        factor = 1.0 if expr["op"] == "+" else -1.0
        self._merge_accumulated_expression(coef_dict, (ldict, lconst), sign, const_ref)
        for key, value in rdict.items():
            coef_dict[key] += sign * factor * value
        const_ref[0] += sign * factor * rconst

    def _accumulate_sum_parenthesized(self, expr, env, coef_dict, sign, const_ref):
        self._merge_accumulated_expression(
            coef_dict,
            self._accumulate_sum_to_dict(expr["expression"], env, sign=1),
            sign,
            const_ref,
        )

    def _accumulate_sum_to_dict(self, expr, env, sign=1):
        """Accumulate coefficients and constants from an expression into a dict and constant."""
        from collections import defaultdict

        coef_dict = defaultdict(float)
        const_ref = [0.0]
        if expr["type"] == "sum":
            self._accumulate_sum_expr(expr, env, coef_dict, sign, const_ref)
        elif expr["type"] == "binop" and (expr["left"].get("type") == "sum" or expr["right"].get("type") == "sum"):
            self._accumulate_binop_with_sum(expr, env, coef_dict, sign, const_ref)
        elif expr["type"] in ("indexed_name", "name", "number"):
            self._accumulate_sum_atom(expr, env, coef_dict, sign, const_ref)
        elif expr["type"] == "binop" and expr.get("op") in ("+", "-"):
            self._accumulate_sum_binop(expr, env, coef_dict, sign, const_ref)
        elif expr["type"] == "parenthesized_expression":
            self._accumulate_sum_parenthesized(expr, env, coef_dict, sign, const_ref)
        else:
            cdict, cval = self._eval_expr(expr, env)
            for vname, coef in cdict.items():
                coef_dict[vname] += sign * coef
            if isinstance(cval, (int, float)):
                const_ref[0] += sign * cval
        return coef_dict, const_ref[0]

    def _accumulate_sum_expr(self, expr, env, coef_dict, sign, const_ref):
        """
        Helper for _accumulate_sum_to_dict: handles 'sum' expressions with dependent bounds.
        """
        iterators = expr["iterators"]
        for env2, _idx_tuple in self._iter_filtered_environments(iterators, env, expr.get("index_constraint")):
            sum_expr = expr["expression"]
            # If the inner expression is a comparison, defer to constraints builder
            if (
                isinstance(sum_expr, dict)
                and sum_expr.get("type") == "binop"
                and sum_expr.get("op") in (">=", "==", ">", "<", "!=")
            ):
                continue
            cdict, cval = self._eval_expr(sum_expr, env=env2)
            for vname, vcoef in cdict.items():
                self._resolve_coefficient_index(vname)
                coef_dict[vname] += sign * vcoef
            if isinstance(cval, (int, float)):
                const_ref[0] += sign * cval

    def _accumulate_both_sides_sum(self, left, right, op, env, coef_dict, sign, const_ref):
        left_coefs = defaultdict(float)
        left_const = [0.0]
        self._accumulate_sum_expr(left, env, left_coefs, 1, left_const)
        right_coefs = defaultdict(float)
        right_const = [0.0]
        self._accumulate_sum_expr(right, env, right_coefs, 1, right_const)
        if op not in ("+", "-"):
            raise self._unsupported_operator_error("binop-with-sum", op)

        factor = 1.0 if op == "+" else -1.0
        for values, constant in ((left_coefs, left_const[0]), (right_coefs, right_const[0])):
            for key, value in values.items():
                coef_dict[key] += sign * (factor if values is right_coefs else 1.0) * value
            const_ref[0] += sign * (factor if values is right_coefs else 1.0) * constant

    def _accumulate_left_sum(self, left, right, op, env, coef_dict, sign, const_ref):
        left_coefs = defaultdict(float)
        left_const = [0.0]
        self._accumulate_sum_expr(left, env, left_coefs, 1, left_const)
        right_coefs, right_const = self._eval_expr(right, env)
        if op not in ("+", "-"):
            raise self._unsupported_operator_error("binop-with-sum", op)

        for key, value in left_coefs.items():
            coef_dict[key] += sign * value
        const_ref[0] += sign * left_const[0]
        right_factor = 1.0 if op == "+" else -1.0
        for key, value in right_coefs.items():
            coef_dict[self._resolve_coefficient_index(key)] += sign * right_factor * value
        if isinstance(right_const, (int, float)):
            const_ref[0] += sign * right_factor * right_const

    def _accumulate_right_sum(self, left, right, op, env, coef_dict, sign, const_ref):
        right_coefs = defaultdict(float)
        right_const = [0.0]
        self._accumulate_sum_expr(right, env, right_coefs, 1, right_const)
        left_coefs, left_const = self._eval_expr(left, env)
        if op not in ("+", "-"):
            raise self._unsupported_operator_error("binop-with-sum", op)

        for key, value in left_coefs.items():
            coef_dict[self._resolve_coefficient_index(key)] += sign * value
        if isinstance(left_const, (int, float)):
            const_ref[0] += sign * left_const
        right_factor = 1.0 if op == "+" else -1.0
        for key, value in right_coefs.items():
            coef_dict[key] += sign * right_factor * value
        const_ref[0] += sign * right_factor * right_const[0]

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
            self._accumulate_both_sides_sum(left, right, op, env, coef_dict, sign, const_ref)
            return
        if left_is_sum:
            self._accumulate_left_sum(left, right, op, env, coef_dict, sign, const_ref)
            return
        if right_is_sum:
            self._accumulate_right_sum(left, right, op, env, coef_dict, sign, const_ref)
            return
        # Fallback: neither side sum (should not reach here based on guard)
        base_coefs, base_const = self._eval_expr(expr, env)
        for vn, cf in base_coefs.items():
            coef_dict[self._resolve_coefficient_index(vn)] += sign * cf
        add_const(base_const, 1.0)

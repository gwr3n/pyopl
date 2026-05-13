# Optimisation Programming Language (OPL) Compiler Lexer and Parser
#
# This compiler is designed to parse a subset of Optimisation Programming Language (OPL)-like syntax,
# focusing on declarations (dvar, param, set, range), objective functions (minimize/maximize),
# and constraints (linear, forall, sum). It does not support all advanced OPL features
# (e.g., piecewise linear functions, logical constraints, complex data structures, external functions).
# It aims for compatibility with core OPL constructs for linear and mixed-integer programming models.

# mypy: disable-error-code=no-redef

# === Standard library imports ===
import json
import keyword
import logging
import os
import sys
import traceback
from io import StringIO
from typing import Any, Optional, cast  # typing helpers

# === Third-party imports ===
from sly import Lexer, Parser  # type: ignore[import-untyped]

try:  # provide '_' decorator symbol explicitly for static analysis
    from sly.yacc import _  # type: ignore
except Exception:  # pragma: no cover
    pass

# === Local imports ===
from .gurobi_codegen import GurobiCodeGenerator
from .scipy_codegen import SciPyCodeGenerator, SciPyCodeGeneratorBase
from .semantic_error import SemanticError

# --- Reserved identifiers that must not appear as model/data names.
# Python keywords are invalid as generated identifiers, and a small built-in set
# remains blocked because code generators may emit those names directly.
RESERVED_PY_IDENTIFIERS: set[str] = set(keyword.kwlist) | set(getattr(keyword, "softkwlist", ())) | {"len"}

# --- Logging Setup ---
# Use module-level logger, and set DEBUG level for development
logger = logging.getLogger(__name__)

SYNTAX_ERROR_REPORTING_MODES = {"full", "line", "masked"}


def _parser_error_with_hint(tok_type: object, tok_val: object) -> str:
    message = f"Syntax error at or near token {tok_type}, value '{tok_val}'."
    if tok_type == "IN":
        return (
            message + " Hint: this implementation does not support filtered/index-comprehension style dvar declarations. "
            "Declare the variable over full index sets and move filtering logic into constraints or tuple/set definitions."
        )
    return (
        message
        + " Hint: rewrite the construct using simpler supported PyOPL syntax, and avoid OPL forms that depend on inline filtering or advanced indexing in declarations."
    )


def _execution_error_with_hint(exc: Exception, backend: str) -> str:
    raw = f"Error during {backend} code execution: {exc}"
    detail = str(exc)
    if "unsupported operand type(s) for -: 'str' and 'str'" in detail:
        return (
            raw + " Hint: a string comparison or string-valued expression is being used inside an algebraic expression. "
            'Do not use tests like k == "K1" inside sums/objectives; encode that logic through data or explicit binary variables.'
        )
    if "unsupported operand type(s) for -: 'gurobipy._core.LinExpr' and 'TempConstr'" in detail:
        return (
            raw + " Hint: a boolean comparison such as (sum(...) >= 1) is being used as if it were a numeric expression. "
            "Replace boolean comparisons in arithmetic with explicit binary variables or separate linear constraints."
        )
    return (
        raw + " Hint: the generated model uses a construct accepted by parsing but not by the backend code generator. "
        "Simplify boolean logic, string tests, and advanced indexed expressions in arithmetic contexts."
    )


def _load_failure_message() -> str:
    return (
        "Failed to load or parse OPL model from file. See errors traceback. "
        "Hint: common fixes are to remove unsupported declaration filters, rewrite keyed .dat arrays into supported plain arrays/key-value forms, and avoid advanced indexed expressions in parameter lookups."
    )


# --- Optional gurobipy import (lazy). Parser should not require gurobi at import time. ---
# Define as Optional[Any] so assigning None is type-safe when gurobipy is unavailable
gp: Optional[Any] = None
GRB: Optional[Any] = None
try:
    import gurobipy as gp  # type: ignore
    from gurobipy import GRB  # type: ignore
except Exception:  # broad: missing lib or license
    gp = None
    GRB = None
    logger.warning("gurobipy unavailable; Gurobi backend will be disabled until installed.")


# --- Symbol Table ---
class SymbolTable:
    """
    Manages symbols (variables, ranges) and their properties within different scopes.
    Supports nested scopes for constructs like 'forall' and 'sum'.
    """

    def __init__(self):
        self.scopes = [{}]  # List of dictionaries, each representing a scope.
        # The last element is the current ( innermost) scope.

    def enter_scope(self):
        """Enters a new, nested scope."""
        self.scopes.append({})
        # Debug: Entered scope (removed print for cleanliness)

    def exit_scope(self):
        """Exits the current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
            # Debug: Exited scope (removed print for cleanliness)
        else:
            raise SemanticError("Cannot exit global scope.")

    def add_symbol(self, name, symbol_type, value=None, dimensions=None, is_dvar=False, lineno=None):
        """
        Adds a symbol to the current scope.
        :param name: Name of the symbol.
        :param symbol_type: Type of the symbol (e.g., 'int', 'float', 'boolean', 'range').
        :param value: For ranges, this holds {'start': int, 'end': int}.
        :param dimensions: For indexed variables, a list of dimension specs.
                           Can now be numeric ranges, named ranges, or named sets.
        :param is_dvar: True if it's a decision variable.
        :param lineno: The line number where the symbol was declared.
        """
        # NEW: reject reserved Python identifiers
        if isinstance(name, str) and name in RESERVED_PY_IDENTIFIERS:
            raise SemanticError(
                f"Identifier '{name}' is reserved and cannot be used in the model (conflicts with Python keywords or built-ins). "
                f"Please rename it.",
                lineno=lineno,
            )
        current_scope = self.scopes[-1]
        if name in current_scope:
            raise SemanticError(f"Symbol '{name}' already declared in this scope.", lineno=lineno)

        current_scope[name] = {
            "type": symbol_type,
            "value": value,
            "dimensions": dimensions,  # This now stores the processed dimension info (range, named_range, named_set)
            "is_dvar": is_dvar,
            "lineno": lineno,  # Store line number
        }
        # Debug: Added symbol (removed print for cleanliness)

    def get_symbol(self, name):
        """
        Retrieves a symbol's information, searching from the innermost to outermost scope.
        :param name: Name of the symbol to retrieve.
        :return: Dictionary containing symbol information.
        :raises SemanticError: If the symbol is not found.
        """
        for scope in reversed(self.scopes):
            if name in scope:
                # Debug: Found symbol (removed print for cleanliness)
                return scope[name]
        # Debug: Symbol not found (removed print for cleanliness)
        raise SemanticError(f"Undeclared symbol '{name}'.")


# --- Lexer ---
class OPLLexer(Lexer):
    """
    Lexer for the OPL-like declarative modeling language.
    Tokenizes the input string into meaningful units for parsing.
    """

    # Order matters for precedence: DOTDOT before NUMBER
    tokens = {
        "DOT",
        "DOTDOT",
        "ELLIPSIS",
        "IN",
        "AND_OP",
        "OR_OP",
        "DVAR",
        "INT",
        "FLOAT",
        "INT_POS",
        "FLOAT_POS",
        "BOOLEAN",
        "STRING",
        "RANGE",
        "PARAM",
        "SET",
        "SUBJECT_TO",
        "MINIMIZE",
        "MAXIMIZE",
        "SUM",
        "FORALL",
        "LE",
        "GE",
        "EQ",
        "NEQ",
        "IMPLIES",
        "NAME",
        "NUMBER",
        "STRING_LITERAL",
        "BOOLEAN_LITERAL",
        "TUPLE",
        "DEXPR",
        "IF",
        "ELSE",
        "AGG_MIN",
        "AGG_MAX",
    }
    # Implication operator: =>
    IMPLIES = r"=>"
    STRING = r"string"
    # Keywords for conditional constraints
    IF = r"\bif\b"
    ELSE = r"\belse\b"

    # Ignore whitespace
    ignore = " \t\r"

    # Define literals (single-character tokens)
    literals = {
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
        "(",
        ")",
        "[",
        "]",
        ":",
        ";",
        ",",
        "{",
        "}",
        "<",
        ">",
        "?",
        "!",
        "|",
        # Note: DOT ('.') is now a token, not a literal; added '!' for logical NOT
    }

    # Define keywords
    TUPLE = r"\btuple\b"
    DVAR = r"\bdvar\b"
    INT_POS = r"\bint\+"
    FLOAT_POS = r"\bfloat\+"
    INT = r"\bint\b"
    FLOAT = r"\bfloat\b"
    BOOLEAN = r"\bboolean\b"
    RANGE = r"\brange\b"
    PARAM = r"\bparam\b"
    SET = r"\bset\b"
    SUBJECT_TO = r"\bsubject\s+to\b"
    MINIMIZE = r"\bminimize\b"
    MAXIMIZE = r"\bmaximize\b"
    AGG_MIN = r"\bmin\b"
    AGG_MAX = r"\bmax\b"
    SUM = r"\bsum\b"
    FORALL = r"\bforall\b"
    IN = r"\bin\b"
    DEXPR = r"\bdexpr\b"

    # Operators
    LE = r"<="
    GE = r">="
    EQ = r"=="  # Using '==' for equality to distinguish from assignment '='
    # Add support for '!=' as not-equal operator
    NEQ = r"!="
    AND_OP = r"&&"
    OR_OP = r"\|\|"

    # --- Token rules ---

    # Boolean literals (must be matched before NAME)
    @_(r"true|false")  # type: ignore
    def BOOLEAN_LITERAL(self, t):
        t.value = t.value.lower()
        return t

    # Identifiers (variable names, etc.)
    NAME = r"[a-zA-Z_][a-zA-Z0-9_]*"

    # Numbers (integers or floats)
    @_(r"\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?")  # type: ignore
    def NUMBER(self, t):
        if "." in str(t.value) or "e" in str(t.value).lower():
            t.value = float(t.value)
        else:
            t.value = int(t.value)
        return t

    # ELLIPSIS, DOTDOT, DOT must be defined after NUMBER to avoid splitting floats
    ELLIPSIS = r"\.\.\."
    DOTDOT = r"\.\."
    DOT = r"\."

    # --- Comment and whitespace rules ---

    # Newlines
    @_(r"\n+")  # type: ignore
    def ignore_newline(self, t):
        self.lineno += t.value.count("\n")

    # Single-line comments (// ...)
    @_(r"//.*")  # type: ignore
    def ignore_line_comment(self, t):
        pass

    # Block comments (/* ... */)
    @_(r"/\*[\s\S]*?\*/")  # type: ignore
    def ignore_block_comment(self, t):
        self.lineno += t.value.count("\n")

    # Hash comments (# ...)
    @_(r"#.*")  # type: ignore
    def ignore_hash_comment(self, t):
        pass

    # String literals
    @_(r'"[^"]*"')  # type: ignore
    def STRING_LITERAL(self, t):
        return t

    def error(self, t):
        raise SemanticError(f"Illegal character '{t.value[0]}'", lineno=self.lineno)


# --- Parser ---
class OPLParser(Parser):
    # debugfile="parser_debug.out"

    # Allow zero or more declarations at the top level
    @_("")  # type: ignore
    def declarations(self, p):
        logger.debug("[DECLARATIONS] Empty declarations list")
        return []

    @_("")  # type: ignore
    def declaration_list(self, p):
        logger.debug("[DECL_LIST] Empty declaration_list")
        return []

    # Set of tuples declaration (inline init): { TupleType } SetName = { <...>, ... };
    @_('"{" NAME "}" NAME "=" "{" tuple_literal_list "}" ";"')  # type: ignore
    def declaration(self, p):
        value = {"elements": p.tuple_literal_list, "tuple_type": p.NAME0}
        self.symbol_table.add_symbol(p.NAME1, "set", value=value, lineno=p.lineno)
        return {
            "type": "set_of_tuples",
            "tuple_type": p.NAME0,
            "name": p.NAME1,
            "value": p.tuple_literal_list,
        }

    # Guard: reject scalar elements in typed set-of-tuples
    @_('"{" NAME "}" NAME "=" "{" element_list "}" ";"')  # type: ignore
    def declaration(self, p):
        raise SemanticError(
            f"Set '{p.NAME1}' is declared as a set of tuples '{{{p.NAME0}}}', but scalar elements were provided. "
            "Use tuple literals like <...>."
        )

    # Typed scalar set of strings: {string} S = { "a", "b" };
    @_('"{" STRING "}" NAME "=" "{" element_list "}" ";"')  # type: ignore
    def declaration(self, p):
        base_type = "string"
        self.symbol_table.add_symbol(
            p.NAME,
            "set",
            value={"base_type": base_type, "elements": p.element_list},
            lineno=p.lineno,
        )
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": p.element_list}

    # Uninitialized typed scalar set: {string} S;
    @_('"{" STRING "}" NAME ";"')  # type: ignore
    def declaration(self, p):
        base_type = "string"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": None}

    # External typed scalar set: {string} S = ...;
    @_('"{" STRING "}" NAME "=" ELLIPSIS ";"')  # type: ignore
    def declaration(self, p):
        base_type = "string"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set_external", "base_type": base_type, "name": p.NAME, "value": None}

    # NEW: Typed scalar set of integers: {int} S = { 1, 2 };
    @_('"{" INT "}" NAME "=" "{" int_element_list "}" ";"')  # type: ignore
    def declaration(self, p):
        base_type = "int"
        self.symbol_table.add_symbol(
            p.NAME, "set", value={"base_type": base_type, "elements": p.int_element_list}, lineno=p.lineno
        )
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": p.int_element_list}

    # NEW: Uninitialized {int} S;
    @_('"{" INT "}" NAME ";"')  # type: ignore
    def declaration(self, p):
        base_type = "int"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": None}

    # NEW: External {int} S = ...;
    @_('"{" INT "}" NAME "=" ELLIPSIS ";"')  # type: ignore
    def declaration(self, p):
        base_type = "int"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set_external", "base_type": base_type, "name": p.NAME, "value": None}

    # NEW: Typed scalar set of floats: {float} S = { 1.0, 2 };
    @_('"{" FLOAT "}" NAME "=" "{" float_element_list "}" ";"')  # type: ignore
    def declaration(self, p):
        base_type = "float"
        self.symbol_table.add_symbol(
            p.NAME, "set", value={"base_type": base_type, "elements": p.float_element_list}, lineno=p.lineno
        )
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": p.float_element_list}

    # NEW: Uninitialized {float} S;
    @_('"{" FLOAT "}" NAME ";"')  # type: ignore
    def declaration(self, p):
        base_type = "float"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": None}

    # NEW: External {float} S = ...;
    @_('"{" FLOAT "}" NAME "=" ELLIPSIS ";"')  # type: ignore
    def declaration(self, p):
        base_type = "float"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set_external", "base_type": base_type, "name": p.NAME, "value": None}

    # NEW: Typed scalar set of booleans: {boolean} S = { true, false };
    @_('"{" BOOLEAN "}" NAME "=" "{" boolean_element_list "}" ";"')  # type: ignore
    def declaration(self, p):
        base_type = "boolean"
        self.symbol_table.add_symbol(
            p.NAME, "set", value={"base_type": base_type, "elements": p.boolean_element_list}, lineno=p.lineno
        )
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": p.boolean_element_list}

    # NEW: Uninitialized {boolean} S;
    @_('"{" BOOLEAN "}" NAME ";"')  # type: ignore
    def declaration(self, p):
        base_type = "boolean"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set", "base_type": base_type, "name": p.NAME, "value": None}

    # NEW: External {boolean} S = ...;
    @_('"{" BOOLEAN "}" NAME "=" ELLIPSIS ";"')  # type: ignore
    def declaration(self, p):
        base_type = "boolean"
        self.symbol_table.add_symbol(p.NAME, "set", value={"base_type": base_type, "elements": None}, lineno=p.lineno)
        return {"type": "typed_set_external", "base_type": base_type, "name": p.NAME, "value": None}

    @_("NAME")  # type: ignore
    def type(self, p):
        # Allow user-defined types (tuple types) as valid types for tuple fields
        return p.NAME

    @_("NAME")  # NEW: allow iterator names inside tuple literals (e.g., <i,j,...>)
    def tuple_element(self, p):
        # Do not require sem_type here; it will be resolved during evaluation
        return {"type": "name", "value": p.NAME}

    # --- Typed set-of-tuples WITH comprehension ---
    # { Pair } Pairs = { <i,j,i2,j2> | i in Rows, j in Cols, i2 in Rows, j2 in Cols : condition };
    @_('"{" NAME "}" NAME "=" "{" tuple_comprehension "}" ";"')  # type: ignore
    def declaration(self, p):
        tuple_type = p.NAME0
        set_name = p.NAME1
        comp = p.tuple_comprehension
        # Register symbol as a set (typed) so later references resolve
        self.symbol_table.add_symbol(
            set_name,
            "set",
            value={"tuple_type": tuple_type},
            is_dvar=False,
            lineno=p.lineno,
        )
        return {
            "type": "set_of_tuples_comprehension",
            "tuple_type": tuple_type,
            "name": set_name,
            "comprehension": comp,
        }

    # tuple_comprehension: <tuple_elems> | sum_index_list [ : condition ]
    @_('"<" tuple_element_list ">" "|" sum_index_list opt_index_constraint')  # type: ignore
    def tuple_comprehension(self, p):
        return {
            "type": "tuple_comprehension",
            "tuple_expr": {"type": "tuple_literal", "elements": p.tuple_element_list},
            "iterators": p.sum_index_list,
            "index_constraint": p.opt_index_constraint,
        }

    # --- DEXPR: decision expressions (expand-on-use) ---

    # Header: [ i in I, j in J ] — keep iterators in scope until RHS parsed
    @_('"[" dexpr_index_list "]"')  # type: ignore
    def dexpr_index_header(self, p):
        # Open a fresh scope for the iterator(s) used by this header
        self.symbol_table.enter_scope()
        # Iterators already added to current scope by dexpr_index_list/dexpr_index
        return {"iterators": p.dexpr_index_list, "_iterator_scope_opened": True}

    # --- Strict OPL nested headers support: [i in I][j in J] ... ---

    # Tail of nested headers: zero or more additional [iterators] groups
    @_('"[" dexpr_index_list "]" dexpr_index_header_tail')  # type: ignore
    def dexpr_index_header_tail(self, p):
        # Concatenate this segment with the remainder of the tail
        return p.dexpr_index_list + p.dexpr_index_header_tail

    @_("")  # type: ignore
    def dexpr_index_header_tail(self, p):
        return []

    # Full nested header(s): one or more [iterators] groups, all sharing a single scope
    @_('"[" dexpr_index_list "]" dexpr_index_header_tail')  # type: ignore
    def dexpr_index_headers(self, p):
        # Single shared scope for all nested headers (strict OPL form)
        self.symbol_table.enter_scope()
        all_iters = p.dexpr_index_list + p.dexpr_index_header_tail
        return {"iterators": all_iters, "_iterator_scope_opened": True}

    @_("dexpr_index_list ',' dexpr_index")  # type: ignore
    def dexpr_index_list(self, p):
        p.dexpr_index_list.append(p.dexpr_index)
        return p.dexpr_index_list

    @_("dexpr_index")  # type: ignore
    def dexpr_index_list(self, p):
        return [p.dexpr_index]

    @_("NAME IN IN_RANGE")  # type: ignore
    def dexpr_index(self, p):
        name = p.NAME
        rng = p.IN_RANGE
        iterator_type = "int"
        if rng["type"] in ("named_range", "named_set"):
            try:
                symbol_info = self.symbol_table.get_symbol(rng["name"])
            except SemanticError:
                # allow forward-declared names; treat as range by default
                symbol_info = {"type": "range", "value": None}
            val = symbol_info.get("value")
            if symbol_info.get("type") == "set" and isinstance(val, dict) and "tuple_type" in val:
                iterator_type = val["tuple_type"]
            elif symbol_info.get("type") == "set" and isinstance(val, dict) and "base_type" in val:
                iterator_type = val["base_type"]
            elif symbol_info.get("type") not in ("range", "set"):
                raise SemanticError(
                    f"Symbol '{rng['name']}' used in dexpr index is not a declared range or set.",
                    lineno=p.lineno,
                )
        # Add iterator to current scope so RHS can reference it
        # Guard against duplicate insertion when ambiguous productions reduce more than once.
        current_scope = self.symbol_table.scopes[-1]
        if name not in current_scope:
            self.symbol_table.add_symbol(name, iterator_type, is_dvar=False, lineno=p.lineno)
        return {"iterator": name, "range": rng}

    # Scalar dexpr: dexpr type Z = expression;
    @_('DEXPR type NAME "=" expression ";"')  # type: ignore
    def declaration(self, p):
        # Store scalar dexpr
        self.symbol_table.add_symbol(
            p.NAME,
            "dexpr",
            value={
                "iterators": [],
                "dimensions": [],
                "expression": p.expression,
                "var_type": p.type,
            },
            lineno=p.lineno,
        )
        return {
            "type": "dexpr",
            "name": p.NAME,
            "var_type": p.type,
            "iterators": [],
            "dimensions": [],
            "expression": p.expression,
        }

    # Indexed dexpr: dexpr type Y[i in I, j in J] = expression;
    @_('DEXPR type NAME dexpr_index_header "=" expression ";"')  # type: ignore
    def declaration(self, p):
        # Convert iterator ranges to declaration-like dimensions so usage checks work
        def to_decl_dim(rng):
            if rng["type"] == "range_specifier":
                # Keep start/end as AST nodes
                return {"type": "range_index", "start": rng["start"], "end": rng["end"]}
            if rng["type"] == "named_range":
                # Attach start/end if available
                try:
                    sym = self.symbol_table.get_symbol(rng["name"])
                    if sym.get("type") == "range" and sym.get("value"):
                        return {
                            "type": "named_range_dimension",
                            "name": rng["name"],
                            "start": sym["value"]["start"],
                            "end": sym["value"]["end"],
                        }
                except SemanticError:
                    pass
                return {"type": "named_range_dimension", "name": rng["name"]}
            if rng["type"] == "named_set":
                return {"type": "named_set_dimension", "name": rng["name"]}
            return {"type": rng["type"], **{k: v for k, v in rng.items() if k != "type"}}

        iterators = p.dexpr_index_header["iterators"]
        dimensions = [to_decl_dim(it["range"]) for it in iterators]

        # Close the iterator scope BEFORE adding the symbol so W persists in the global scope
        try:
            if p.dexpr_index_header.get("_iterator_scope_opened"):
                self.symbol_table.exit_scope()
        except Exception:
            pass

        # Store dexpr in the (now current) outer scope
        self.symbol_table.add_symbol(
            p.NAME,
            "dexpr",
            value={
                "iterators": iterators,
                "dimensions": dimensions,
                "expression": p.expression,
                "var_type": p.type,
            },
            dimensions=dimensions,
            lineno=p.lineno,
        )

        # Return AST declaration
        return {
            "type": "dexpr_indexed",
            "name": p.NAME,
            "var_type": p.type,
            "iterators": iterators,
            "dimensions": dimensions,
            "expression": p.expression,
        }

    # NEW: Indexed dexpr with strict OPL nested headers: dexpr type Y[i in I][j in J] = expression;
    @_('DEXPR type NAME dexpr_index_headers "=" expression ";"')  # type: ignore
    def declaration(self, p):
        def to_decl_dim(rng):
            if rng["type"] == "range_specifier":
                return {"type": "range_index", "start": rng["start"], "end": rng["end"]}
            if rng["type"] == "named_range":
                try:
                    sym = self.symbol_table.get_symbol(rng["name"])
                    if sym.get("type") == "range" and sym.get("value"):
                        return {
                            "type": "named_range_dimension",
                            "name": rng["name"],
                            "start": sym["value"]["start"],
                            "end": sym["value"]["end"],
                        }
                except SemanticError:
                    pass
                return {"type": "named_range_dimension", "name": rng["name"]}
            if rng["type"] == "named_set":
                return {"type": "named_set_dimension", "name": rng["name"]}
            return {"type": rng["type"], **{k: v for k, v in rng.items() if k != "type"}}

        iterators = p.dexpr_index_headers["iterators"]
        dimensions = [to_decl_dim(it["range"]) for it in iterators]

        # Close iterator scope before adding symbol
        try:
            if p.dexpr_index_headers.get("_iterator_scope_opened"):
                self.symbol_table.exit_scope()
        except Exception:
            pass

        self.symbol_table.add_symbol(
            p.NAME,
            "dexpr",
            value={
                "iterators": iterators,
                "dimensions": dimensions,
                "expression": p.expression,
                "var_type": p.type,
            },
            dimensions=dimensions,
            lineno=p.lineno,
        )
        return {
            "type": "dexpr_indexed",
            "name": p.NAME,
            "var_type": p.type,
            "iterators": iterators,
            "dimensions": dimensions,
            "expression": p.expression,
        }

    # Helper: convert index-spec nodes to general expression nodes for substitution
    def _index_to_expr(self, idx):
        if not isinstance(idx, dict):
            return idx
        t = idx.get("type")
        if t == "name_reference_index":
            # Treat as plain name in expression
            sem = idx.get("sem_type", None)
            return {"type": "name", "value": idx["name"], "sem_type": sem}
        if t == "number_literal_index":
            sem = idx.get("sem_type", "int")
            return {"type": "number", "value": idx["value"], "sem_type": sem}
        if t in ("binop", "uminus", "parenthesized_expression", "tuple_literal", "field_access"):
            return idx
        if t == "field_access_index":
            # normalize to field_access
            return {
                "type": "field_access",
                "base": idx["base"],
                "field": idx["field"],
                "sem_type": idx.get("sem_type", None),
            }
        return idx

    # Helper: deep substitute iterator variables with index expressions
    def _subst_iterators(self, expr, mapping):
        if isinstance(expr, dict):
            # Replace plain iterator name nodes
            if expr.get("type") == "name":
                key = expr.get("value")
                if isinstance(key, str) and key in mapping:
                    return self._index_to_expr(mapping[key])
            # Replace iterator references used inside indexed dimensions.
            if expr.get("type") == "name_reference_index":
                key = expr.get("name")
                if isinstance(key, str) and key in mapping:
                    return mapping[key]
            # Recurse
            out = {}
            for k, v in expr.items():
                out[k] = self._subst_iterators(v, mapping)
            return out
        if isinstance(expr, list):
            return [self._subst_iterators(v, mapping) for v in expr]
        return expr

    # --- Conditional expression: (cond) ? thenExpr : elseExpr ---

    @_('"(" expression ")" "?" expression ":" expression')  # type: ignore
    def conditional(self, p):
        cond = p.expression0
        then_expr = p.expression1
        else_expr = p.expression2
        # For now, assume semantic check is done in codegen/eval
        # Set sem_type to then_expr's type (else_expr should match)
        return {
            "type": "conditional",
            "condition": cond,
            "then": then_expr,
            "else": else_expr,
            "sem_type": then_expr["sem_type"],
        }

    """
    Parser for the declarative modeling language.
    Builds an Abstract Syntax Tree (AST) from the tokens and performs semantic analysis.
    """

    # --- External set of tuples declaration: {Arc} arcs = ...; ---
    @_('"{" NAME "}" NAME "=" ELLIPSIS ";"')  # type: ignore
    def declaration(self, p):
        # External set of tuples declaration with ellipsis (e.g., {Arc} arcs = ...;)
        tuple_type = p.NAME0
        set_name = p.NAME1
        self.symbol_table.add_symbol(
            set_name,
            "set",
            value={"tuple_type": tuple_type},
            is_dvar=False,
            lineno=p.lineno,
        )
        return {
            "type": "set_of_tuples_external",
            "tuple_type": tuple_type,
            "name": set_name,
            "value": None,
        }

    # --- Uninitialized set of tuples declaration: {Arc} arcs; ---
    @_('"{" NAME "}" NAME ";"')  # type: ignore
    def declaration(self, p):
        # Uninitialized set of tuples declaration (e.g., {Arc} arcs;)
        tuple_type = p.NAME0
        set_name = p.NAME1
        self.symbol_table.add_symbol(
            set_name,
            "set",
            value={"tuple_type": tuple_type},
            is_dvar=False,
            lineno=p.lineno,
        )
        return {
            "type": "set_of_tuples",
            "tuple_type": tuple_type,
            "name": set_name,
            "value": None,
        }

    # --- Primary expressions (atomic) ---
    # Ensure BOOLEAN_LITERAL is matched before NAME

    # --- NAME primary: consult iterator-context before symbol table ---
    @_("BOOLEAN_LITERAL", "STRING_LITERAL", "NAME")
    def primary(self, p):
        if hasattr(p, "BOOLEAN_LITERAL"):
            return {
                "type": "boolean_literal",
                "value": p.BOOLEAN_LITERAL.lower() == "true",
                "sem_type": "boolean",
            }
        elif hasattr(p, "STRING_LITERAL"):
            return {
                "type": "string_literal",
                "value": p.STRING_LITERAL[1:-1],
                "sem_type": "string",
            }
        elif hasattr(p, "NAME"):
            # SLY attributes are typed as Any; narrow to str for mypy before dict indexing.
            name_any = p.NAME
            if not isinstance(name_any, str):
                raise SemanticError("Invalid identifier token (expected NAME).", lineno=p.lineno)
            name = name_any

            # NEW: check current iterator context first (only active inside sum/forall bodies)
            if self._iterator_context_stack:
                top = self._iterator_context_stack[-1]
                sem = top.get(name)
                if sem is not None:
                    return {"type": "name", "value": name, "sem_type": sem}

            # Fallback: regular symbol table lookup
            symbol_info = self.symbol_table.get_symbol(name)
            # Inline scalar dexpr on use
            if symbol_info.get("type") == "dexpr":
                val = symbol_info.get("value") or {}
                iters = val.get("iterators") or []
                dims = val.get("dimensions") or []
                if iters or dims:
                    raise SemanticError(
                        f"Expected indexed dexpr, but '{name}' is declared with indices. Missing dimensions.",
                        lineno=p.lineno,
                    )
                return self._subst_iterators(val.get("expression"), {})
            if symbol_info.get("dimensions"):
                raise SemanticError(
                    f"Expected scalar variable, but '{name}' is an indexed variable. Missing dimensions.",
                    lineno=p.lineno,
                )
            return {"type": "name", "value": name, "sem_type": symbol_info["type"]}

    # --- sum_expression and forall_expression nonterminals ---

    # OPL-style juxtaposition: sum(i in I : cond) x[i] means sum over x[i]
    @_("SUM sum_index_header nonparen_expression")  # type: ignore
    def sum_expression(self, p):
        logger.debug(f"[PARSER] Enter sum_expression (juxtaposition): SUM {p.sum_index_header} {p.nonparen_expression}")
        iterators = p.sum_index_header["iterators"]
        index_constraint = p.sum_index_header.get("index_constraint")
        sum_body = p.nonparen_expression
        expr_type = sum_body["sem_type"]
        if expr_type == "boolean":
            expr_type = "int"
        # Close iterator scope if opened (legacy behavior)
        if p.sum_index_header.get("_iterator_scope_opened"):
            self.symbol_table.exit_scope()
        # Pop iterator context if pushed
        if p.sum_index_header.get("_iter_ctx_pushed") and self._iterator_context_stack:
            self._iterator_context_stack.pop()
        logger.debug(
            f"[PARSER] Exit sum_expression (juxtaposition): iterators={iterators}, index_constraint={index_constraint}, expr_type={expr_type}"
        )
        return {
            "type": "sum",
            "iterators": iterators,
            "index_constraint": index_constraint,
            "expression": sum_body,
            "sem_type": expr_type,
        }

    # Parenthesized/atomic case: sum(i in I : cond) (x[i] >= 6)
    @_("SUM sum_index_header parenthesized_expression")  # type: ignore
    def sum_expression(self, p):
        logger.debug(f"[PARSER] Enter sum_expression (parenthesized): SUM {p.sum_index_header} {p.parenthesized_expression}")
        iterators = p.sum_index_header["iterators"]
        index_constraint = p.sum_index_header.get("index_constraint")
        parsed_expression = p.parenthesized_expression
        expr_type = parsed_expression["sem_type"]
        result_type = "int" if expr_type == "boolean" else expr_type
        if p.sum_index_header.get("_iterator_scope_opened"):
            self.symbol_table.exit_scope()
        if p.sum_index_header.get("_iter_ctx_pushed") and self._iterator_context_stack:
            self._iterator_context_stack.pop()
        logger.debug(
            f"[PARSER] Exit sum_expression (parenthesized): iterators={iterators}, index_constraint={index_constraint}, expr_type={expr_type}"
        )
        return {
            "type": "sum",
            "iterators": iterators,
            "index_constraint": index_constraint,
            "expression": parsed_expression,
            "sem_type": result_type,
        }

    # Helper nonterminal for bare aggregate bodies.
    # This lets `sum(i in I) a[i] * x[i]` bind the full product into the sum
    # without greedily swallowing surrounding `+`, `-`, or comparison context.
    @_("multiplicative")  # type: ignore
    def nonparen_expression(self, p):
        return p.multiplicative

    @_('"(" expression ")"')  # type: ignore
    def parenthesized_expression(self, p):
        return {
            "type": "parenthesized_expression",
            "expression": p.expression,
            "sem_type": p.expression["sem_type"],
        }

    # Allow sum_expression and forall_expression as valid expressions
    @_("sum_expression")  # type: ignore
    def primary(self, p):
        # Allow sum() constructs wherever a primary is valid in layered grammar
        return p.sum_expression

    # @_('forall_expression') # type: ignore
    # def expression(self, p):
    #     return p.forall_expression

    # Operator precedence table:
    # - DOT (field access) binds tightest, right-associative, so a + b.to parses as a + (b.to), not (a + b).to
    # - Arithmetic operators (+, -, *, /) are left-associative
    # - Comparison operators (==, !=, <=, >=, <, >) and range operator (..) are handled in separate nonterminals
    #   and do not need to be in the precedence table, as they are not parsed as general infix operators.
    # Operator precedence (from lowest to highest binding):
    # 1. Ternary '? :' (treat '?' as lowest)
    # 2. OR
    # 3. AND
    # 4. Add/Sub
    # 5. Mul/Div
    # 6. Unary '!'
    # 7. Field access '.' (DOT)
    precedence = (
        ("right", "?"),  # conditional (lowest precedence among listed)
        ("left", "OR_OP"),
        ("left", "AND_OP"),
        (
            "nonassoc",
            "EQ",
            "NEQ",
            "LE",
            "GE",
            ">",
            "<",
        ),  # comparisons (non-associative)
        ("left", "+", "-"),
        ("left", "*", "/", "%"),
        ("right", "!"),  # unary logical NOT
        ("right", "DOT"),  # field access binds tightest
    )

    # --- Primary expressions (atomic) ---
    # (Removed duplicate stray @_('NAME') decorator and code for primary)

    @_("NUMBER")  # type: ignore
    def primary(self, p):
        sem_type = "int" if isinstance(p.NUMBER, int) else "float"
        return {"type": "number", "value": p.NUMBER, "sem_type": sem_type}

    @_('"(" expression ")"')  # type: ignore
    def primary(self, p):
        return {
            "type": "parenthesized_expression",
            "expression": p.expression,
            "sem_type": p.expression["sem_type"],
        }

    # Helper: detect negative numeric literals (either number < 0 or uminus of a number)
    def _is_negative_literal(self, expr) -> bool:
        try:
            if isinstance(expr, dict):
                t = expr.get("type")
                if t == "number":
                    v = expr.get("value")
                    return isinstance(v, (int, float)) and v < 0
                if t == "uminus":
                    inner = expr.get("value")
                    return isinstance(inner, dict) and inner.get("type") == "number"
        except Exception:
            pass
        return False

    # Signed numeric literal for non-expression contexts (arrays, tuple elements, typed sets, direct param values)
    @_("NUMBER")  # type: ignore
    def signed_number(self, p):
        return p.NUMBER

    @_('"-" NUMBER')  # type: ignore
    def signed_number(self, p):
        n = p.NUMBER
        return -n

    # --- Field access: primary DOT NAME (right-associative, allows chaining) ---
    @_("primary DOT NAME")  # type: ignore
    def primary(self, p):
        logger.debug(
            f"[FIELD_ACCESS] (primary rule triggered) p.primary: {p.primary}, p.NAME: {p.NAME}, type(p.primary): {type(p.primary)}"
        )
        base = p.primary
        field = p.NAME
        # Determine tuple type name from base semantic type and look it up
        base_sem_type = base.get("sem_type")
        tuple_def = None
        if base_sem_type:
            for scope in reversed(self.symbol_table.scopes):
                info = scope.get(base_sem_type)
                if info and info.get("type") == "tuple_type":
                    tuple_def = info
                    break
        if not tuple_def:
            raise SemanticError(f"Field access '{field}' applied to non-tuple expression.")
        fields = tuple_def.get("value", [])
        field_type = None
        for f in fields:
            if f.get("name") == field:
                field_type = f.get("type")
                break
        if not field_type:
            raise SemanticError(f"Unknown field '{field}' for tuple type '{base_sem_type}'.")
        return {
            "type": "field_access",
            "base": base,
            "field": field,
            "sem_type": field_type,
        }

    # --- NEW: simple function calls (currently only sqrt) ---
    @_("NAME '(' expression ')'")  # type: ignore
    def primary(self, p):
        func = p.NAME
        if func != "sqrt":
            raise SemanticError(f"Unsupported function '{func}'. Only sqrt(...) is supported.")
        arg = p.expression
        # Result of sqrt is float
        return {"type": "funcall", "name": "sqrt", "args": [arg], "sem_type": "float"}

    # --- Untyped set literal on LHS: allow only set of tuples; scalar sets must be typed ---
    @_('NAME "=" "{" set_value_list "}" ";"')  # type: ignore
    def declaration(self, p):
        if p.set_value_list and isinstance(p.set_value_list[0], dict) and p.set_value_list[0].get("type") == "tuple_literal":
            return {"type": "set_of_tuples", "name": p.NAME, "value": p.set_value_list}
        raise SemanticError(
            "Scalar sets in model files must be typed. Use '{int}', '{float}', '{boolean}', or '{string}': e.g., {int} S = {1,2};",
            lineno=p.lineno,
        )

    # Accept either a tuple_literal_list or an element_list as set_value_list
    @_("tuple_literal_list")  # type: ignore
    def set_value_list(self, p):
        return p.tuple_literal_list

    @_("element_list")  # type: ignore
    def set_value_list(self, p):
        return p.element_list

    # --- element_list (model parser) for typed scalar sets ---
    @_("STRING_LITERAL")  # type: ignore
    def element_list(self, p):
        return [p.STRING_LITERAL.strip('"')]

    @_('element_list "," STRING_LITERAL')  # type: ignore
    def element_list(self, p):
        p.element_list.append(p.STRING_LITERAL.strip('"'))
        return p.element_list

    # NEW: int_element_list for {int} sets
    @_("signed_number")  # type: ignore
    def int_element_list(self, p):
        v = p.signed_number
        if not (isinstance(v, int) and not isinstance(v, bool)):
            raise SemanticError(f"Expected integer literal in {{int}} set, got '{v}'.")
        return [v]

    @_('int_element_list "," signed_number')  # type: ignore
    def int_element_list(self, p):
        v = p.signed_number
        if not (isinstance(v, int) and not isinstance(v, bool)):
            raise SemanticError(f"Expected integer literal in {{int}} set, got '{v}'.")
        p.int_element_list.append(v)
        return p.int_element_list

    # NEW: float_element_list for {float} sets (allow ints; coerce to float)
    @_("signed_number")  # type: ignore
    def float_element_list(self, p):
        v = p.signed_number
        if isinstance(v, bool):
            raise SemanticError(f"Expected numeric literal in {{float}} set, got '{v}'.")
        return [float(v)]

    @_('float_element_list "," signed_number')  # type: ignore
    def float_element_list(self, p):
        v = p.signed_number
        if isinstance(v, bool):
            raise SemanticError(f"Expected numeric literal in {{float}} set, got '{v}'.")
        p.float_element_list.append(float(v))
        return p.float_element_list

    # NEW: boolean_element_list for {boolean} sets
    @_("BOOLEAN_LITERAL")  # type: ignore
    def boolean_element_list(self, p):
        # Model lexer provides 'true'/'false' (str)
        return [True if p.BOOLEAN_LITERAL == "true" else False]

    @_('boolean_element_list "," BOOLEAN_LITERAL')  # type: ignore
    def boolean_element_list(self, p):
        p.boolean_element_list.append(True if p.BOOLEAN_LITERAL == "true" else False)
        return p.boolean_element_list

    @_("tuple_literal_list ',' tuple_literal")  # type: ignore
    def tuple_literal_list(self, p):
        return p.tuple_literal_list + [p.tuple_literal]

    @_("tuple_literal")  # type: ignore
    def tuple_literal_list(self, p):
        return [p.tuple_literal]

    @_("'<' tuple_element_list '>'")  # type: ignore
    def tuple_literal(self, p):
        return {"type": "tuple_literal", "elements": p.tuple_element_list}

    @_("'<' '>'")  # type: ignore
    def tuple_literal(self, p):
        # Allow empty tuple literal <>
        return {"type": "tuple_literal", "elements": []}

    # Make tuple literal usable as an expression (e.g., as an index into tuple-set–indexed vars/params)
    @_("tuple_literal")  # type: ignore
    def primary(self, p):
        # Keep original tuple_literal node; sem_type not required for index usage
        return p.tuple_literal

    @_("tuple_element_list ',' tuple_element")  # type: ignore
    def tuple_element_list(self, p):
        return p.tuple_element_list + [p.tuple_element]

    @_("tuple_element")  # type: ignore
    def tuple_element_list(self, p):
        return [p.tuple_element]

    # Tuple elements: allow negative numbers via signed_number
    @_("STRING_LITERAL")  # type: ignore
    def tuple_element(self, p):
        return p.STRING_LITERAL.strip('"')

    @_("signed_number")  # type: ignore
    def tuple_element(self, p):
        return p.signed_number

    @_("tuple_literal")  # type: ignore
    def tuple_element(self, p):
        # Allow nested tuple literals as tuple elements
        return p.tuple_literal

    @_("STRING")  # type: ignore
    def type(self, p):
        return "string"

    # --- Tuple type declaration: allow empty tuple types ---
    @_(
        'TUPLE NAME "{" tuple_field_list "}"',  # type: ignore
        'TUPLE NAME "{" tuple_field_list "}" ";"',  # type: ignore
        'TUPLE NAME "{" "}"',  # type: ignore
        'TUPLE NAME "{" "}" ";"',
    )  # type: ignore
    def declaration(self, p):
        # If tuple_field_list is present, use it; else, empty list
        fields = p.tuple_field_list if hasattr(p, "tuple_field_list") else []
        self.symbol_table.add_symbol(p.NAME, "tuple_type", value=fields)
        return {"type": "tuple_type", "name": p.NAME, "fields": fields}

    @_("tuple_field_list tuple_field")  # type: ignore
    def tuple_field_list(self, p):
        return p.tuple_field_list + [p.tuple_field]

    @_("tuple_field")  # type: ignore
    def tuple_field_list(self, p):
        return [p.tuple_field]

    @_('type NAME ";"')  # type: ignore
    def tuple_field(self, p):
        return {"type": p.type, "name": p.NAME}

    """
    Parser for the declarative modeling language.
    Builds an Abstract Syntax Tree (AST) from the tokens and performs semantic analysis.
    """
    tokens = OPLLexer.tokens
    start = "model"  # Explicitly set the start symbol for the parser

    # --- Layered expression grammar to reduce conflicts ---
    # primary already defined elsewhere (boolean literals, NAME, indexed_name, etc.)

    # Parentheses already handled by existing parenthesized_expression rule earlier; avoid duplicate primary rule.

    # unary: logical NOT and unary minus
    @_('"!" unary')
    def unary(self, p):
        inner = p.unary
        return {"type": "not", "value": inner, "sem_type": "boolean"}

    @_('"-" unary')
    def unary(self, p):
        expr_type = p.unary["sem_type"]
        if expr_type == "boolean":
            raise SemanticError("Cannot apply unary minus to a boolean expression.")
        return {"type": "uminus", "value": p.unary, "sem_type": expr_type}

    @_("primary")
    def unary(self, p):
        return p.primary

    # multiplicative
    @_("unary")
    def multiplicative(self, p):
        return p.unary

    @_('multiplicative "*" unary')
    def multiplicative(self, p):
        return self._handle_binop(p.multiplicative, p.unary, "*", getattr(p, "lineno", None))

    @_('multiplicative "/" unary')
    def multiplicative(self, p):
        return self._handle_binop(p.multiplicative, p.unary, "/", getattr(p, "lineno", None))

    # NEW: modulo operator
    @_('multiplicative "%" unary')
    def multiplicative(self, p):
        return self._handle_binop(p.multiplicative, p.unary, "%", getattr(p, "lineno", None))

    # additive
    @_("multiplicative")
    def additive(self, p):
        return p.multiplicative

    @_('additive "+" multiplicative')
    def additive(self, p):
        return self._handle_binop(p.additive, p.multiplicative, "+", getattr(p, "lineno", None))

    @_('additive "-" multiplicative')
    def additive(self, p):
        return self._handle_binop(p.additive, p.multiplicative, "-", getattr(p, "lineno", None))

    # Helper: reject chained comparisons like a <= b <= c early
    def _reject_chained_comparison(self, left_expr, lineno):
        if isinstance(left_expr, dict):
            t = left_expr.get("type")
            op = left_expr.get("op")
            if t in ("binop", "constraint") and op in ("<", ">", "<=", ">=", "=="):
                raise SemanticError(
                    "Chained comparisons (e.g., a <= b <= c) are not supported. "
                    "Split into two constraints: a <= b; b <= c;",
                    lineno=lineno,
                )

    # relational (<, <=, >, >=)
    @_("additive")
    def relational(self, p):
        return p.additive

    @_('relational "<" additive')
    def relational(self, p):
        self._reject_chained_comparison(p.relational, getattr(p, "lineno", None))
        left = p.relational
        right = p.additive
        return {"type": "binop", "op": "<", "left": left, "right": right, "sem_type": "boolean"}

    @_('relational ">" additive')
    def relational(self, p):
        self._reject_chained_comparison(p.relational, getattr(p, "lineno", None))
        left = p.relational
        right = p.additive
        return {"type": "binop", "op": ">", "left": left, "right": right, "sem_type": "boolean"}

    @_("relational LE additive")
    def relational(self, p):
        self._reject_chained_comparison(p.relational, getattr(p, "lineno", None))
        left = p.relational
        right = p.additive
        return {"type": "binop", "op": "<=", "left": left, "right": right, "sem_type": "boolean"}

    @_("relational GE additive")
    def relational(self, p):
        self._reject_chained_comparison(p.relational, getattr(p, "lineno", None))
        left = p.relational
        right = p.additive
        return {"type": "binop", "op": ">=", "left": left, "right": right, "sem_type": "boolean"}

    # equality (==, !=)
    @_("relational")
    def equality(self, p):
        return p.relational

    @_("equality EQ relational")
    def equality(self, p):
        return {
            "type": "binop",
            "op": "==",
            "left": p.equality,
            "right": p.relational,
            "sem_type": "boolean",
        }

    @_("equality NEQ relational")
    def equality(self, p):
        return {
            "type": "binop",
            "op": "!=",
            "left": p.equality,
            "right": p.relational,
            "sem_type": "boolean",
        }

    # logic AND
    @_("equality")
    def logic_and(self, p):
        return p.equality

    @_("logic_and AND_OP equality")
    def logic_and(self, p):
        if p.logic_and.get("sem_type") != "boolean" or p.equality.get("sem_type") != "boolean":
            raise SemanticError("Logical '&&' requires boolean operands.")
        return {
            "type": "and",
            "left": p.logic_and,
            "right": p.equality,
            "sem_type": "boolean",
        }

    # logic OR
    @_("logic_and")
    def logic_or(self, p):
        return p.logic_and

    @_("logic_or OR_OP logic_and")
    def logic_or(self, p):
        if p.logic_or.get("sem_type") != "boolean" or p.logic_and.get("sem_type") != "boolean":
            raise SemanticError("Logical '||' requires boolean operands.")
        return {
            "type": "or",
            "left": p.logic_or,
            "right": p.logic_and,
            "sem_type": "boolean",
        }

    # conditional (ternary)
    @_("logic_or")
    def conditional(self, p):
        return p.logic_or

    # Final expression alias
    @_("conditional")
    def expression(self, p):
        return p.conditional

    def __init__(self) -> None:
        self.symbol_table = SymbolTable()
        # Track last-seen token line for EOF errors
        self._last_lineno = 1
        # NEW: stack of dicts mapping iterator name -> sem_type (e.g., tuple type or base type)
        # Activated while parsing bodies of sum(...) and forall(...)
        self._iterator_context_stack: list[dict[str, str]] = []

    # Helper: build iterator type mapping from sum_index_list entries
    def _iter_types_from_sum_index_list(self, sum_index_list: list[dict[str, Any]]) -> dict[str, str]:
        it_types: dict[str, str] = {}

        for it in sum_index_list or []:
            iterator_obj: object = it.get("iterator")
            if not isinstance(iterator_obj, str):
                continue
            iterator: str = iterator_obj  # now a real str key for mypy

            rng_any = it.get("range")
            rng: dict[str, Any] = rng_any if isinstance(rng_any, dict) else {}

            sem_type: str = "int"  # default

            rtype = rng.get("type")
            if rtype in ("named_range",):
                sem_type = "int"
            elif rtype in ("named_set", "named_set_dimension"):
                rng_name_obj: object = rng.get("name")
                rng_name: Optional[str] = rng_name_obj if isinstance(rng_name_obj, str) else None
                if rng_name:
                    try:
                        sym = self.symbol_table.get_symbol(rng_name)
                        val = sym.get("value")
                        if sym.get("type") == "set" and isinstance(val, dict) and "tuple_type" in val:
                            sem_type = cast(str, val["tuple_type"])
                        elif sym.get("type") == "set" and isinstance(val, dict) and "base_type" in val:
                            sem_type = cast(str, val["base_type"])
                        else:
                            sem_type = "string"
                    except SemanticError:
                        sem_type = "string"
                else:
                    sem_type = "string"
            elif rtype == "range_specifier":
                sem_type = "int"

            it_types[iterator] = sem_type

        return it_types

    def parse(self, tokens):
        # Materialize tokens so we can track the last line for EOF diagnostics
        self.symbol_table = SymbolTable()
        self.current_tokens = list(tokens)
        if self.current_tokens:
            try:
                # SLY tokens carry .lineno
                self._last_lineno = getattr(self.current_tokens[-1], "lineno", 1)
            except Exception:
                self._last_lineno = 1
        else:
            self._last_lineno = 1
        return super().parse(iter(self.current_tokens))

    # --- Custom error method for parser debugging ---
    def error(self, token):
        # Unexpected token
        if token is not None:
            lineno = getattr(token, "lineno", self._last_lineno)
            tok_type = getattr(token, "type", None)
            tok_val = getattr(token, "value", None)
            raise SemanticError(_parser_error_with_hint(tok_type, tok_val), lineno=lineno)
        # Unexpected EOF
        raise SemanticError("Syntax error at end of file (EOF).", lineno=self._last_lineno)

    @_("declarations objective_section constraints_section")  # type: ignore
    def model(self, p):
        # Debug: print model rule reduction
        # print("[DEBUG] model rule reduced")
        return {
            "declarations": p.declarations,
            "objective": p.objective_section,
            "constraints": p.constraints_section,
        }

    @_("declarations constraints_section objective_section")  # type: ignore
    def model(self, p):
        return {
            "declarations": p.declarations,
            "objective": p.objective_section,
            "constraints": p.constraints_section,
        }

    @_("declaration_list declaration")  # type: ignore
    def declaration_list(self, p):
        logger.debug(f"[DECL_LIST] Appending declaration: {p.declaration}")
        return p.declaration_list + [p.declaration]

    @_("declaration")  # type: ignore
    def declaration_list(self, p):
        logger.debug(f"[DECL_LIST] Single declaration: {p.declaration}")
        return [p.declaration]

    @_("declaration_list")  # type: ignore
    def declarations(self, p):
        logger.debug(f"[DECLARATIONS] Reduced to declaration_list: {p.declaration_list}")
        return p.declaration_list

    @_("declaration")  # type: ignore
    def declarations(self, p):
        logger.debug(f"[DECLARATIONS] Single declaration: {p.declaration}")
        return [p.declaration]

    @_('DVAR type NAME ";"')  # type: ignore
    def declaration(self, p):
        # Disallow string decision variables (unsupported in codegen)
        if p.type == "string":
            raise SemanticError(
                "String decision variables are not supported. Use 'string' only for tuple fields or typed scalar sets.",
                lineno=p.lineno,
            )
        self.symbol_table.add_symbol(p.NAME, p.type, is_dvar=True, lineno=p.lineno)
        return {"type": "dvar", "var_type": p.type, "name": p.NAME}

    @_('DVAR type NAME indexed_dimensions ";"')  # type: ignore
    def declaration(self, p):
        # Disallow string decision variables (unsupported in codegen)
        if p.type == "string":
            raise SemanticError(
                "String decision variables are not supported. Use 'string' only for tuple fields or typed scalar sets.",
                lineno=p.lineno,
            )
        processed_dimensions = []
        for dim_spec in p.indexed_dimensions:
            if dim_spec["type"] == "range_index":
                # Always store start/end as AST nodes (do not convert to int)
                # If start/end are int, wrap as AST number nodes for codegen compatibility
                start = dim_spec["start"]
                end = dim_spec["end"]
                if isinstance(start, int):
                    start = {"type": "number", "value": start, "sem_type": "int"}
                if isinstance(end, int):
                    end = {"type": "number", "value": end, "sem_type": "int"}
                processed_dimensions.append({"type": "range_index", "start": start, "end": end})
            elif dim_spec["type"] == "name_reference_index":
                name = dim_spec["name"]
                try:
                    symbol_info = self.symbol_table.get_symbol(name)
                    if symbol_info["type"] == "range":
                        if symbol_info["value"] is not None:
                            processed_dimensions.append(
                                {
                                    "type": "named_range_dimension",
                                    "name": name,
                                    "start": symbol_info["value"]["start"],
                                    "end": symbol_info["value"]["end"],
                                }
                            )
                        else:
                            processed_dimensions.append({"type": "named_range_dimension", "name": name})
                    elif symbol_info["type"] == "set":
                        processed_dimensions.append({"type": "named_set_dimension", "name": name})
                    else:
                        raise SemanticError(
                            f"Symbol '{name}' used as dimension must be a 'range' or 'set', but found '{symbol_info['type']}'.",
                            lineno=p.lineno,
                        )
                except SemanticError as e:
                    raise SemanticError(
                        f"Undeclared symbol '{name}' used as dimension.",
                        lineno=p.lineno,
                    ) from e
            elif dim_spec["type"] == "number_literal_index":
                raise SemanticError(
                    f"Single number index '{dim_spec['value']}' not allowed in variable declaration dimensions. Use 'range' like [1..N] or a named 'set'/'range'.",
                    lineno=p.lineno,
                )
            else:
                raise SemanticError(
                    f"Unsupported dimension type in declaration: {dim_spec['type']}",
                    lineno=p.lineno,
                )

        self.symbol_table.add_symbol(
            p.NAME,
            p.type,
            dimensions=processed_dimensions,
            is_dvar=True,
            lineno=p.lineno,
        )
        return {
            "type": "dvar_indexed",
            "var_type": p.type,
            "name": p.NAME,
            "dimensions": processed_dimensions,
        }

    # --- Range declaration with general integer expressions as bounds ---
    @_('RANGE NAME "=" range_expr DOTDOT range_expr ";"')  # type: ignore
    def declaration(self, p):
        start_node = p.range_expr0
        end_node = p.range_expr1

        # Disallow negative literal bounds (e.g., -3 .. 5 or 3 .. -5)
        if self._is_negative_literal(start_node) or self._is_negative_literal(end_node):
            raise SemanticError("Range bounds must be non-negative literals.", lineno=p.lineno)

        # If both constant numbers, ensure start <= end
        start_is_int = (
            isinstance(start_node, dict) and start_node.get("type") == "number" and isinstance(start_node.get("value"), int)
        )
        end_is_int = isinstance(end_node, dict) and end_node.get("type") == "number" and isinstance(end_node.get("value"), int)
        if start_is_int and end_is_int:
            s_val = start_node["value"]
            e_val = end_node["value"]
            if s_val > e_val:
                raise SemanticError(
                    f"Range start ({s_val}) cannot be greater than end ({e_val}).",
                    lineno=p.lineno,
                )
        # Always store as AST nodes for codegen compatibility
        self.symbol_table.add_symbol(
            p.NAME,
            "range",
            value={"start": start_node, "end": end_node},
            lineno=p.lineno,
        )
        return {
            "type": "range_declaration_inline",
            "name": p.NAME,
            "start": start_node,
            "end": end_node,
        }

    @_('RANGE NAME ";"')  # type: ignore
    def declaration(self, p):
        self.symbol_table.add_symbol(p.NAME, "range", value=None, lineno=p.lineno)
        return {"type": "range_declaration_external", "name": p.NAME}

    @_('SET NAME ";"')  # type: ignore
    def declaration(self, p):
        self.symbol_table.add_symbol(p.NAME, "set", is_dvar=False, lineno=p.lineno)
        return {"type": "set_declaration", "name": p.NAME}

    # --- Start of "param" optional rules and new explicit external parameter syntax ---

    # --- Optional 'param' keyword: allows both 'param type Name' and 'type Name' ---
    @_("PARAM")  # type: ignore
    def opt_PARAM(self, p):
        return True

    # Empty rule: needed to allow omission of 'param' keyword (i.e., 'type Name')
    @_("")  # type: ignore
    def opt_PARAM(self, p):
        return False

    # --- Optional assignment with ellipsis: allows both 'type Name = ...' and 'type Name' ---
    @_('"=" ELLIPSIS')  # type: ignore
    def opt_assign_ellipsis(self, p):
        return True

    # Empty rule: needed to allow omission of '= ...' in parameter declarations
    @_("")  # type: ignore
    def opt_assign_ellipsis(self, p):
        return False

    @_('opt_PARAM type NAME opt_assign_ellipsis ";"')  # type: ignore
    def declaration(self, p):
        """
        Rule for scalar parameter declaration.
        If '= ...' is present, it's explicitly external.
        Otherwise, it's implicitly external.
        """
        name = p.NAME
        var_type = p.type
        has_ellipsis_assignment = p.opt_assign_ellipsis

        if has_ellipsis_assignment:
            # Explicitly external parameter: type Name = ...; or param type Name = ...;
            self.symbol_table.add_symbol(name, var_type, is_dvar=False, lineno=p.lineno)
            return {"type": "parameter_external", "var_type": var_type, "name": name}
        else:
            # Implicitly external parameter: type Name; or param type Name;
            self.symbol_table.add_symbol(name, var_type, is_dvar=False, lineno=p.lineno)
            return {"type": "parameter_external", "var_type": var_type, "name": name}

    @_('opt_PARAM type NAME indexed_dimensions opt_assign_ellipsis ";"')  # type: ignore
    def declaration(self, p):
        """
        Rule for indexed parameter declaration.
        If '= ...' is present, it's explicitly external.
        Otherwise, it's implicitly external.
        """
        name = p.NAME
        var_type = p.type
        processed_dimensions = []
        for dim_spec in p.indexed_dimensions:
            if dim_spec["type"] == "range_index":
                processed_dimensions.append(dim_spec)
            elif dim_spec["type"] == "name_reference_index":
                dim_name = dim_spec["name"]
                try:
                    symbol_info = self.symbol_table.get_symbol(dim_name)
                    if symbol_info["type"] == "range":
                        if symbol_info["value"] is not None:
                            processed_dimensions.append(
                                {
                                    "type": "named_range_dimension",
                                    "name": dim_name,
                                    "start": symbol_info["value"]["start"],
                                    "end": symbol_info["value"]["end"],
                                }
                            )
                        else:
                            processed_dimensions.append({"type": "named_range_dimension", "name": dim_name})
                    elif symbol_info["type"] == "set":
                        processed_dimensions.append({"type": "named_set_dimension", "name": dim_name})
                    else:
                        raise SemanticError(
                            f"Symbol '{dim_name}' used as dimension must be a 'range' or 'set', but found '{symbol_info['type']}'.",
                            lineno=p.lineno,
                        )
                except SemanticError as e:
                    raise SemanticError(e.message, lineno=p.lineno) from e
            elif dim_spec["type"] == "number_literal_index":
                raise SemanticError(
                    f"Single number index '{dim_spec['value']}' not allowed in declaration dimensions. Use 'range' like [1..N] or a named 'set'/'range'.",
                    lineno=p.lineno,
                )
            else:
                raise SemanticError(
                    f"Unsupported dimension type in declaration: {dim_spec['type']}",
                    lineno=p.lineno,
                )

        has_ellipsis_assignment = p.opt_assign_ellipsis

        if has_ellipsis_assignment:
            # Explicitly external indexed parameter: type Name[dims] = ...; or param type Name[dims] = ...;
            self.symbol_table.add_symbol(
                name,
                var_type,
                dimensions=processed_dimensions,
                is_dvar=False,
                lineno=p.lineno,
            )
            return {
                "type": "parameter_external_explicit_indexed",
                "var_type": var_type,
                "name": name,
                "dimensions": processed_dimensions,
            }
        else:
            # Implicitly external indexed parameter: type Name[dims]; or param type Name[dims];
            self.symbol_table.add_symbol(
                name,
                var_type,
                dimensions=processed_dimensions,
                is_dvar=False,
                lineno=p.lineno,
            )
            return {
                "type": "parameter_external_indexed",
                "var_type": var_type,
                "name": name,
                "dimensions": processed_dimensions,
            }

    # --- End of "param" optional rules and new explicit external parameter syntax ---

    @_('indexed_dimensions "[" index_specifier "]"')  # type: ignore
    def indexed_dimensions(self, p):
        """
        Handles multiple dimensions for indexed variables (e.g., [1..2][1..3]).
        Recursively builds a list of index specifiers.
        """
        p.indexed_dimensions.append(p.index_specifier)
        return p.indexed_dimensions

    @_('"[" index_specifier "]"')  # type: ignore
    def indexed_dimensions(self, p):
        """
        Base case for indexed_dimensions: a single dimension.
        """
        return [p.index_specifier]

    @_("INT")  # type: ignore
    def type(self, p):
        return "int"

    @_("FLOAT")  # type: ignore
    def type(self, p):
        return "float"

    @_("INT_POS")  # type: ignore
    def type(self, p):
        return "int+"

    @_("FLOAT_POS")  # type: ignore
    def type(self, p):
        return "float+"

    @_("BOOLEAN")  # type: ignore
    def type(self, p):
        return "boolean"

    @_('MINIMIZE expression ";"')  # type: ignore
    def objective_section(self, p):
        # OPL semantics: allow boolean objectives
        return {"type": "minimize", "expression": p.expression}

    @_('MAXIMIZE expression ";"')  # type: ignore
    def objective_section(self, p):
        # OPL semantics: allow boolean objectives
        return {"type": "maximize", "expression": p.expression}

    # NEW: Objective with label using colon: minimize z: expr;
    @_('MINIMIZE NAME ":" expression ";"')  # type: ignore
    def objective_section(self, p):
        return {"type": "minimize", "label": p.NAME, "expression": p.expression}

    @_('MAXIMIZE NAME ":" expression ";"')  # type: ignore
    def objective_section(self, p):
        return {"type": "maximize", "label": p.NAME, "expression": p.expression}

    # NEW: Objective with label using equals: minimize z = expr;
    @_('MINIMIZE NAME "=" expression ";"')  # type: ignore
    def objective_section(self, p):
        return {"type": "minimize", "label": p.NAME, "expression": p.expression}

    @_('MAXIMIZE NAME "=" expression ";"')  # type: ignore
    def objective_section(self, p):
        return {"type": "maximize", "label": p.NAME, "expression": p.expression}

    # --- Objectives (existing rules above) ---
    # Add explicit lint: reject indexed objective labels like 'minimize z[i]: expr;' or 'minimize z[i] = expr;'
    @_('MINIMIZE NAME indexed_dimensions ":" expression ";"')  # type: ignore
    def objective_section(self, p):
        raise SemanticError(
            "Objective labels must be unindexed. Use 'minimize z: expr;' not 'minimize z[...]: expr;'.",
            lineno=p.lineno,
        )

    @_('MAXIMIZE NAME indexed_dimensions ":" expression ";"')  # type: ignore
    def objective_section(self, p):
        raise SemanticError(
            "Objective labels must be unindexed. Use 'maximize z: expr;' not 'maximize z[...]: expr;'.",
            lineno=p.lineno,
        )

    @_('MINIMIZE NAME indexed_dimensions "=" expression ";"')  # type: ignore
    def objective_section(self, p):
        raise SemanticError(
            "Objective labels must be unindexed. Use 'minimize z = expr;' not 'minimize z[...] = expr;'.",
            lineno=p.lineno,
        )

    @_('MAXIMIZE NAME indexed_dimensions "=" expression ";"')  # type: ignore
    def objective_section(self, p):
        raise SemanticError(
            "Objective labels must be unindexed. Use 'maximize z = expr;' not 'maximize z[...] = expr;'.",
            lineno=p.lineno,
        )

    # --- Constraints section ---
    # Lint: reject a plain label that prefixes a forall, e.g. `c: forall(...) { ... }`
    @_('NAME ":" FORALL forall_index_header constraint')  # type: ignore
    def constraint(self, p):
        # Clean up any iterator context opened by the header to avoid leaking state
        if p.forall_index_header.get("_iter_ctx_pushed") and self._iterator_context_stack:
            self._iterator_context_stack.pop()
        if p.forall_index_header.get("_iterator_scope_opened"):
            self.symbol_table.exit_scope()
        raise SemanticError(
            "Constraint labels may not prefix a forall. To label constraints produced by a forall, put the label inside the forall, e.g.:\n"
            "  forall(i in I) ct: expr;\n"
            "or\n"
            "  forall(i in I) { ct: expr; }",
            lineno=p.lineno,
        )

    @_('NAME ":" FORALL forall_index_header constraint_block')  # type: ignore
    def constraint(self, p):
        # Clean up any iterator context opened by the header to avoid leaking state
        if p.forall_index_header.get("_iter_ctx_pushed") and self._iterator_context_stack:
            self._iterator_context_stack.pop()
        if p.forall_index_header.get("_iterator_scope_opened"):
            self.symbol_table.exit_scope()
        raise SemanticError(
            "Constraint labels may not prefix a forall. To label constraints produced by a forall, put the label inside the forall, e.g.:\n"
            "  forall(i in I) ct: expr;\n"
            "or\n"
            "  forall(i in I) { ct: expr; }",
            lineno=p.lineno,
        )

    # Lint: reject indexed labels like 'ct[i]: ...;' at top level
    @_('NAME indexed_dimensions ":" expression ";"')  # type: ignore
    def constraint(self, p):
        raise SemanticError(
            "Indexed constraint labels are not allowed. Use an unindexed label, e.g., 'ct: ...;'. "
            "To label constraints generated by a forall, put the plain label inside the forall: "
            "forall(i in I) ct: ...;",
            lineno=p.lineno,
        )

    # Lint: reject indexed labels inside forall like 'forall(...) ct[i]: ...;'
    @_('FORALL forall_index_header NAME indexed_dimensions ":" expression ";"')  # type: ignore
    def constraint(self, p):
        # Close any iterator context opened by the header to avoid scope leaks on error
        if p.forall_index_header.get("_iter_ctx_pushed") and self._iterator_context_stack:
            self._iterator_context_stack.pop()
        if p.forall_index_header.get("_iterator_scope_opened"):
            self.symbol_table.exit_scope()
        raise SemanticError(
            "Indexed constraint labels are not allowed inside forall. Use a plain, unindexed label inside the forall: "
            "forall(i in I) ct: ...;",
            lineno=p.lineno,
        )

    @_('SUBJECT_TO "{" constraint_list "}"')  # type: ignore
    def constraints_section(self, p):
        return p.constraint_list

    # Allow empty constraints block: subject to { }
    @_('SUBJECT_TO "{" "}"')  # type: ignore
    def constraints_section(self, p):
        return []

    # --- Constraint list: sequence of constraints, each ending with a semicolon ---
    @_("constraint")  # type: ignore
    def constraint_list(self, p):
        return [p.constraint]

    @_("constraint_list constraint")  # type: ignore
    def constraint_list(self, p):
        p.constraint_list.append(p.constraint)
        return p.constraint_list

    # --- Constraint: either implication or regular constraint, both consume semicolon ---
    @_('expression IMPLIES expression ";"')  # type: ignore
    def constraint(self, p):
        def to_constraint(expr):
            if expr.get("type") == "constraint":
                return expr
            # Any boolean-valued expression (including binop comparisons, 'not', parenthesized, boolean literals)
            if expr.get("sem_type") == "boolean":
                return {
                    "type": "constraint",
                    "op": "==",
                    "left": expr,
                    "right": {
                        "type": "boolean_literal",
                        "value": True,
                        "sem_type": "boolean",
                    },
                }
            if expr.get("type") == "parenthesized_expression":
                return to_constraint(expr["expression"])
            raise SemanticError("Implication sides must be constraints or boolean expressions.")

        antecedent = to_constraint(p.expression0)
        consequent = to_constraint(p.expression1)
        return {
            "type": "implication_constraint",
            "antecedent": antecedent,
            "consequent": consequent,
        }

    @_('expression ";"')  # type: ignore
    def constraint(self, p):
        # Normalize int+ and float+ to int and float for OPL semantics
        def norm_type(t):
            if t == "int+":
                return "int"
            if t == "float+":
                return "float"
            return t

        expr = p.expression
        # If it's already a constraint node (equality to True) pass through
        if expr.get("type") == "constraint":
            return expr
        if expr.get("type") == "binop" and expr.get("op") in (
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
        ):
            op = expr["op"]
            left = expr["left"]
            right = expr["right"]
            left_type = norm_type(left.get("sem_type", None))
            right_type = norm_type(right.get("sem_type", None))
            allowed_types = {"int", "float", "boolean", None}
            if left_type not in allowed_types or right_type not in allowed_types:
                raise SemanticError(
                    f"'{op}' operator only supported for int/float/boolean types, got '{left_type}' and '{right_type}'.",
                    lineno=p.lineno,
                )
            return {"type": "constraint", "op": op, "left": left, "right": right}
        # Fallback: if boolean-valued (e.g., from logical composition), equate to True
        if expr.get("sem_type") == "boolean":
            return {
                "type": "constraint",
                "op": "==",
                "left": expr,
                "right": {
                    "type": "boolean_literal",
                    "value": True,
                    "sem_type": "boolean",
                },
            }
        # If purely arithmetic numeric expression appears alone, this is likely a user error.
        raise SemanticError(
            "Standalone arithmetic expression not allowed as constraint; use comparison (e.g., expr <= value)."
        )

    # Labeled simple constraint: label: expr OP expr;
    @_('NAME ":" expression ";"')  # type: ignore
    def constraint(self, p):
        expr = p.expression
        if expr.get("type") == "binop" and expr.get("op") in (
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
        ):
            return {
                "type": "constraint",
                "label": p.NAME,
                "op": expr["op"],
                "left": expr["left"],
                "right": expr["right"],
            }
        if expr.get("sem_type") == "boolean":
            return {
                "type": "constraint",
                "label": p.NAME,
                "op": "==",
                "left": expr,
                "right": {
                    "type": "boolean_literal",
                    "value": True,
                    "sem_type": "boolean",
                },
            }
        raise SemanticError("Labeled constraints must be comparison or boolean expression.")

    # Labeled forall single constraint: forall(...) label: expr OP expr;
    @_('FORALL forall_index_header NAME ":" expression ";"')  # type: ignore
    def constraint(self, p):
        expr = p.expression
        if expr.get("type") == "binop" and expr.get("op") in (
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
        ):
            base_constraint = {
                "type": "constraint",
                "label": p.NAME,
                "op": expr["op"],
                "left": expr["left"],
                "right": expr["right"],
            }
        elif expr.get("sem_type") == "boolean":
            base_constraint = {
                "type": "constraint",
                "label": p.NAME,
                "op": "==",
                "left": expr,
                "right": {
                    "type": "boolean_literal",
                    "value": True,
                    "sem_type": "boolean",
                },
            }
        else:
            raise SemanticError("forall labeled constraint must be comparison or boolean expression.")
        fc = self._build_forall_constraint(p.forall_index_header, base_constraint, getattr(p, "lineno", None))
        return fc

    # --- NEW: Conditional constraints ---
    # if (<ground_condition>) { <list-of-constraints> } else { <list-of-constraints> }
    @_('IF "(" expression ")" constraint_block ELSE constraint_block')
    def constraint(self, p):
        # Build an AST node; validation and evaluation occur in OPLCompiler
        return {
            "type": "if_constraint",
            "condition": p.expression,
            "then_constraints": p.constraint_block0,
            "else_constraints": p.constraint_block1,
            "lineno": getattr(p, "lineno", None),
        }

    # if (<ground_condition>) { <list-of-constraints> }
    @_('IF "(" expression ")" constraint_block')
    def constraint(self, p):
        return {
            "type": "if_constraint",
            "condition": p.expression,
            "then_constraints": p.constraint_block,
            "else_constraints": None,
            "lineno": getattr(p, "lineno", None),
        }

    # (Boolean standalone constraint rule merged into unified expression ';' rule above)

    def _check_comparison_types(self, left_expr, right_expr, lineno):
        # Patch: allow boolean variables in arithmetic and sum contexts (OPL semantics)
        # Accept any combination of int, float, or boolean for arithmetic and comparison.
        def normalize_type(t):
            if t == "int+":
                return "int"
            if t == "float+":
                return "float"
            return t

        left_type = normalize_type(left_expr.get("sem_type", None))
        right_type = normalize_type(right_expr.get("sem_type", None))
        allowed_types = {"int", "float", "boolean", None}
        if left_type not in allowed_types or right_type not in allowed_types:
            raise SemanticError(f"Type mismatch in comparison: {left_type} vs {right_type}", lineno)
        # Otherwise, allow (int, float, boolean) in any combination
        return

    def _build_forall_constraint(self, forall_index_header, constraint_or_block, lineno):
        # Scope is already open (iter_header_open), and iterators are already added by sum_index.
        iterators = forall_index_header["iterators"]
        index_constraint = forall_index_header.get("index_constraint")
        result = {
            "type": "forall_constraint",
            "iterators": iterators,
            "index_constraint": index_constraint,
        }

        def wrap_implication_if_needed(c):
            if isinstance(c, dict) and c.get("type") == "implication_constraint":
                return c
            if isinstance(c, dict) and c.get("type") == "constraint":
                return c
            if isinstance(c, list):
                return [wrap_implication_if_needed(x) for x in c]
            return c

        # NEW: attach a label_template to labelled constraints inside this forall
        it_names = [it.get("iterator") for it in iterators if isinstance(it, dict) and "iterator" in it]

        def attach_label_template(node):
            if isinstance(node, dict) and "label" in node and "label_template" not in node:
                node["label_template"] = {"name": node["label"], "iterators": list(it_names)}
            return node

        if isinstance(constraint_or_block, list):
            wrapped = [wrap_implication_if_needed(x) for x in constraint_or_block]
            # attach templates
            result["constraints"] = [attach_label_template(x) for x in wrapped if isinstance(x, dict)]
        else:
            single = wrap_implication_if_needed(constraint_or_block)
            result["constraint"] = attach_label_template(single) if isinstance(single, dict) else single
        return result

    # --- Forall constraints: pop iterator context after parsing inner constraint/block ---
    @_("FORALL forall_index_header constraint")  # type: ignore
    def constraint(self, p):
        node = self._build_forall_constraint(p.forall_index_header, p.constraint, getattr(p, "lineno", None))
        # Close parser iterator-context scope
        if p.forall_index_header.get("_iter_ctx_pushed") and self._iterator_context_stack:
            self._iterator_context_stack.pop()
        # Legacy: close symbol-table scope if it was opened
        if p.forall_index_header.get("_iterator_scope_opened"):
            self.symbol_table.exit_scope()
        return node

    @_("FORALL forall_index_header constraint_block")  # type: ignore
    def constraint(self, p):
        node = self._build_forall_constraint(p.forall_index_header, p.constraint_block, getattr(p, "lineno", None))
        if p.forall_index_header.get("_iter_ctx_pushed") and self._iterator_context_stack:
            self._iterator_context_stack.pop()
        if p.forall_index_header.get("_iterator_scope_opened"):
            self.symbol_table.exit_scope()
        return node

    @_('"{" constraint_list "}"')  # type: ignore
    def constraint_block(self, p):
        # Accept implication_constraint(s) in block
        return p.constraint_list

    # Support: (i in Cities, j in Cities: i != j)
    @_('"(" sum_index_list opt_index_constraint ")"')  # type: ignore
    def forall_index_header(self, p):
        iterators = p.sum_index_list
        result = {"iterators": iterators, "index_constraint": p.opt_index_constraint}
        # Push iterator context for body parsing
        iter_types = self._iter_types_from_sum_index_list(iterators)
        self._iterator_context_stack.append(iter_types)
        result["_iter_ctx_pushed"] = True
        return result

    @_("expression DOTDOT expression")  # type: ignore
    def IN_RANGE(self, p):
        start_val = p.expression0
        end_val = p.expression1
        if start_val["sem_type"] not in ["int", "int+"] or end_val["sem_type"] not in ["int", "int+"]:
            raise SemanticError("Range bounds must be integer-valued.", lineno=p.lineno)
        # Disallow negative literal bounds
        if self._is_negative_literal(start_val) or self._is_negative_literal(end_val):
            raise SemanticError("Range bounds must be non-negative literals.", lineno=p.lineno)
        return {"type": "range_specifier", "start": start_val, "end": end_val}

    @_("NAME")  # type: ignore
    def IN_RANGE(self, p):
        # Distinguish between named range and named set
        try:
            sym = self.symbol_table.get_symbol(p.NAME)
            if sym.get("type") == "set":
                return {"type": "named_set", "name": p.NAME}
            else:
                return {"type": "named_range", "name": p.NAME}
        except SemanticError:
            # Fallback treat as named_range; semantic error will surface later if undeclared
            return {"type": "named_range", "name": p.NAME}

    @_("expression DOTDOT expression")  # type: ignore
    def index_specifier(self, p):
        start_val = p.expression0
        end_val = p.expression1
        if start_val["sem_type"] not in ["int", "int+"] or end_val["sem_type"] not in ["int", "int+"]:
            raise SemanticError("Index range bounds must be integer-valued.", lineno=p.lineno)
        # Disallow negative literal bounds
        if self._is_negative_literal(start_val) or self._is_negative_literal(end_val):
            raise SemanticError("Index range bounds must be non-negative literals.", lineno=p.lineno)
        return {"type": "range_index", "start": start_val, "end": end_val}

    # Accept any int-valued expression as a range bound
    @_("expression")  # type: ignore
    def range_expr(self, p):
        expr = p.expression
        if expr["sem_type"] not in ["int", "int+"]:
            raise SemanticError(f"Range bound must be integer-valued, got type '{expr['sem_type']}'.")
        # Disallow negative literal bound
        if self._is_negative_literal(expr):
            raise SemanticError("Range bounds must be non-negative literals.", lineno=p.lineno)
        return expr

    @_("expression")  # type: ignore
    def index_specifier(self, p):
        expr = p.expression
        # If it's a number literal, convert to number_literal_index; reject negative literal indices
        if expr["type"] == "number":
            if isinstance(expr.get("value"), (int, float)) and expr["value"] < 0:
                raise SemanticError("Negative literal indices are not allowed.")
            return {
                "type": "number_literal_index",
                "value": expr["value"],
                "sem_type": expr.get("sem_type", "int"),
            }
        # Reject uminus of a number literal as index
        if expr["type"] == "uminus" and isinstance(expr.get("value"), dict) and expr["value"].get("type") == "number":
            raise SemanticError("Negative literal indices are not allowed.")
        # Existing acceptance logic (binop, uminus of non-literal, etc.)
        if expr["type"] in [
            "binop",
            "uminus",
            "parenthesized_expression",
            "field_access",
            "field_access_index",
            "string_literal",
            "tuple_literal",
        ]:
            if expr["type"] == "number_literal_index" and "sem_type" not in expr:
                expr["sem_type"] = "int"
            return expr
        if expr["type"] == "field_access" and expr.get("sem_type") in ["int", "int+"]:
            return {
                "type": "field_access_index",
                "base": expr["base"],
                "field": expr["field"],
                "sem_type": expr.get("sem_type", None),
            }
        if expr["type"] == "name":
            symbol_info = self.symbol_table.get_symbol(expr["value"])
            return {
                "type": "name_reference_index",
                "name": expr["value"],
                "sem_type": symbol_info["type"],
            }
        raise SemanticError(f"Unsupported index expression type: {expr['type']}.", lineno=p.lineno)

    # Juxtaposition rules for sum_expression expression and forall_expression expression are intentionally omitted
    # to avoid ambiguity and allow sum/forall expressions to be used directly as the LHS of constraints.

    # --- sum_expression and forall_expression nonterminals ---

    # A forall expression is not a value-producing expression and cannot appear in an expression context
    # such as an objective, assignment, or parameter value. It is a statement-level construct used for
    # constraints or for generating multiple constraints, not for producing a value.

    # The rule @_('FORALL forall_index_header expression') for forall_expression as an expression is
    # present for completeness or for future extensions, but it does not correspond to any valid OPL
    # model in standard usage. In practice, OPL models only use forall in the context of constraints
    # (i.e., subject to { forall(...) ...; }), not as a value in an expression.

    # def _build_forall_expression(self, forall_index_header, expression, lineno, debug_prefix=""):
    #     logger.debug(f"[PARSER] Enter {debug_prefix}forall_expression: {forall_index_header} {expression}")
    #     iterators = forall_index_header['iterators']
    #     index_constraint = forall_index_header.get('index_constraint')
    #     self.symbol_table.enter_scope()
    #     for iterator in iterators:
    #         name = iterator['iterator']
    #         rng = iterator['range']
    #         iterator_type = 'int'
    #         if rng['type'] == 'named_range':
    #             try:
    #                 symbol_info = self.symbol_table.get_symbol(rng['name'])
    #             except SemanticError:
    #                 raise SemanticError(f"Symbol '{rng['name']}' used in 'in' clause is not declared.", lineno=lineno)
    #             if symbol_info.get('type') == 'set' and symbol_info.get('value') and isinstance(symbol_info['value'], dict) and 'tuple_type' in symbol_info['value']:
    #                 iterator_type = symbol_info['value']['tuple_type']
    #             elif symbol_info.get('type') not in ('range', 'set'):
    #                 raise SemanticError(f"Symbol '{rng['name']}' used in 'in' clause is not a declared range or set.", lineno=lineno)
    #         self.symbol_table.add_symbol(name, iterator_type, is_dvar=False, lineno=lineno)
    #     parsed_expression = expression
    #     expr_type = parsed_expression['sem_type']
    #     result_type = 'int' if expr_type == 'boolean' else expr_type
    #     self.symbol_table.exit_scope()
    #     logger.debug(f"[PARSER] Exit {debug_prefix}forall_expression: iterators={iterators}, index_constraint={index_constraint}, expr_type={expr_type}")
    #     return {'type': 'forall', 'iterators': iterators, 'index_constraint': index_constraint, 'expression': parsed_expression, 'sem_type': result_type}

    # @_('FORALL forall_index_header expression') # type: ignore
    # def forall_expression(self, p):
    #     return self._build_forall_expression(p.forall_index_header, p.expression, getattr(p, 'lineno', None), debug_prefix="")

    # @_('FORALL "(" forall_index_header ")" expression') # type: ignore
    # def forall_expression(self, p):
    #     return self._build_forall_expression(p.forall_index_header, p.expression, getattr(p, 'lineno', None), debug_prefix="(parens) ")

    # --- New: open a scope as soon as we see '(' starting an iterator header ---
    @_('"("')
    def iter_header_open(self, p):
        # Open a scope for iterators used by sum/forall header
        self.symbol_table.enter_scope()
        # Tag that a scope is open; the iterator additions will go into this scope
        return {"_iterator_scope_opened": True}

    # --- sum_index_header: push iterator context for the upcoming body parse ---
    @_("iter_header_open sum_index_list opt_index_constraint ')'")  # type: ignore
    def sum_index_header(self, p):
        logger.debug(
            f"[PARSER] Enter sum_index_header: sum_index_list={p.sum_index_list}, opt_index_constraint={p.opt_index_constraint}"
        )
        result = {"iterators": p.sum_index_list, "index_constraint": p.opt_index_constraint}
        # Iterator types for context (do NOT change symbol-table scoping)
        iter_types = self._iter_types_from_sum_index_list(p.sum_index_list)
        self._iterator_context_stack.append(iter_types)
        result["_iter_ctx_pushed"] = True
        # Preserve flag if separate scope was opened via iter_header_open
        result["_iterator_scope_opened"] = True
        logger.debug(f"[PARSER] Exit sum_index_header: result={result}")
        return result

    # --- forall_index_header: push iterator context similarly (no scope changes) ---
    @_("iter_header_open sum_index_list opt_index_constraint ')'")  # type: ignore
    def forall_index_header(self, p):
        iterators = p.sum_index_list
        result = {"iterators": iterators, "index_constraint": p.opt_index_constraint, "_iterator_scope_opened": True}
        # Push iterator context for body parsing
        iter_types = self._iter_types_from_sum_index_list(iterators)
        self._iterator_context_stack.append(iter_types)
        result["_iter_ctx_pushed"] = True
        return result

    # Support: legacy header without iter_header_open
    @_('"(" sum_index_list opt_index_constraint ")"')  # type: ignore
    def sum_index_header(self, p):
        logger.debug(
            f"[PARSER] Enter sum_index_header (alt): sum_index_list={p.sum_index_list}, opt_index_constraint={p.opt_index_constraint}"
        )
        result = {"iterators": p.sum_index_list, "index_constraint": p.opt_index_constraint}
        # Push iterator types for body parsing even if no scope was opened
        iter_types = self._iter_types_from_sum_index_list(p.sum_index_list)
        self._iterator_context_stack.append(iter_types)
        result["_iter_ctx_pushed"] = True
        logger.debug(f"[PARSER] Exit sum_index_header (alt): result={result}")
        return result

    # Multi-index: all iterators are in the same scope
    @_('sum_index_list "," sum_index')  # type: ignore
    def sum_index_list(self, p):
        # Do not enter/exit scope here; all iterators are in the same scope
        p.sum_index_list.append(p.sum_index)
        return p.sum_index_list

    @_("sum_index")  # type: ignore
    def sum_index_list(self, p):
        # Do not enter/exit scope here; all iterators are in the same scope
        return [p.sum_index]

    @_("NAME IN IN_RANGE")  # type: ignore
    def sum_index(self, p):
        # Add the iterator symbol with correct type if possible
        current_scope = self.symbol_table.scopes[-1]
        rng = p.IN_RANGE
        iterator_type = "int"
        # If the range is a named set (possibly of tuples) or named range, set iterator type accordingly
        if rng["type"] in ("named_range", "named_set"):
            try:
                symbol_info = self.symbol_table.get_symbol(rng["name"])
            except SemanticError:
                raise SemanticError(
                    f"Symbol '{rng['name']}' used in 'in' clause is not declared.",
                    lineno=p.lineno,
                )
            # tuple-valued set: store tuple type name so field access and index type checks work
            val = symbol_info.get("value")
            if symbol_info.get("type") == "set" and isinstance(val, dict) and "tuple_type" in val:
                iterator_type = val["tuple_type"]
            # typed scalar set: use its base_type (string/int/float/boolean)
            elif symbol_info.get("type") == "set" and isinstance(val, dict) and "base_type" in val:
                iterator_type = val["base_type"]
            elif symbol_info.get("type") not in ("range", "set"):
                raise SemanticError(
                    f"Symbol '{rng['name']}' used in 'in' clause is not a declared range or set.",
                    lineno=p.lineno,
                )
        if p.NAME not in current_scope:
            # Store the tuple/base type name as the type for iterators
            self.symbol_table.add_symbol(p.NAME, iterator_type, is_dvar=False, lineno=p.lineno)
        return {"iterator": p.NAME, "range": p.IN_RANGE}

    # --- Optional index constraint: allows both 'sum(i in I)' and 'sum(i in I : cond)' ---
    # (Single ':' expression opt_index_constraint rule retained earlier; duplicate removed)

    @_('":" expression')  # type: ignore
    def opt_index_constraint(self, p):
        # Fallback: allow any boolean-valued expression (future-proofing for more complex boolean logic)
        return p.expression

    # Empty rule: needed to allow omission of ': constraint' in sum/forall index headers
    @_("")  # type: ignore
    def opt_index_constraint(self, p):
        return None

    # Removed duplicate binary operation rules for expression
    @_('expression "+" expression')  # type: ignore
    def expression(self, p):
        logger.debug(
            f"[BINOP_RULE] '+' left: {p.expression0}, right: {p.expression1}, left type: {p.expression0.get('sem_type', p.expression0.get('type', type(p.expression0)))}, right type: {p.expression1.get('sem_type', p.expression1.get('type', type(p.expression1)))}"
        )
        return self._handle_binop(p.expression0, p.expression1, "+", p.lineno)

    @_('expression "-" expression')  # type: ignore
    def expression(self, p):
        logger.debug(
            f"[BINOP_RULE] '-' left: {p.expression0}, right: {p.expression1}, left type: {p.expression0.get('sem_type', p.expression0.get('type', type(p.expression0)))}, right type: {p.expression1.get('sem_type', p.expression1.get('type', type(p.expression1)))}"
        )
        return self._handle_binop(p.expression0, p.expression1, "-", p.lineno)

    @_('expression "*" expression')  # type: ignore
    def expression(self, p):
        logger.debug(
            f"[BINOP_RULE] '*' left: {p.expression0}, right: {p.expression1}, left type: {p.expression0.get('sem_type', p.expression0.get('type', type(p.expression0)))}, right type: {p.expression1.get('sem_type', p.expression1.get('type', type(p.expression1)))}"
        )
        return self._handle_binop(p.expression0, p.expression1, "*", p.lineno)

    @_('expression "/" expression')  # type: ignore
    def expression(self, p):
        logger.debug(
            f"[BINOP_RULE] '/' left: {p.expression0}, right: {p.expression1}, left type: {p.expression0.get('sem_type', p.expression0.get('type', type(p.expression0)))}, right type: {p.expression1.get('sem_type', p.expression1.get('type', type(p.expression1)))}"
        )
        return self._handle_binop(p.expression0, p.expression1, "/", p.lineno)

    def _handle_binop(self, left_expr, right_expr, op, lineno):
        # Extensive logger debugging for binop typing issues
        logger.debug(f"[BINOP] op: {op}, left_expr: {left_expr}, right_expr: {right_expr}, lineno: {lineno}")

        # If both sides are sum/forall, return a binop node with both as children
        left_is_sum = isinstance(left_expr, dict) and left_expr.get("type") in ("sum", "forall")
        right_is_sum = isinstance(right_expr, dict) and right_expr.get("type") in ("sum", "forall")

        if left_is_sum and right_is_sum:
            result_type = left_expr.get("sem_type") or right_expr.get("sem_type") or "int"
            logger.debug("[BINOP] Both sides are sum/forall: returning binop of two sums/foralls")
            return {
                "type": "binop",
                "op": op,
                "left": left_expr,
                "right": right_expr,
                "sem_type": result_type,
            }

        # DO NOT lift +/- into sum (prevents accidental duplication of unrelated terms)
        # Only allow pushing into sums for multiplicative contexts handled below.

        # If only left is sum/forall, push binop inside left sum/forall (for *, /, % only)
        if left_is_sum and op in ("*", "/", "%"):
            new_body = {"type": "binop", "op": op, "left": left_expr["expression"], "right": right_expr, "sem_type": None}
            sum_node = dict(left_expr)
            sum_node["expression"] = new_body
            sum_node["sem_type"] = left_expr.get("sem_type", right_expr.get("sem_type"))
            return sum_node

        # If only right is sum/forall, push binop inside right sum/forall (for *, /, % only)
        if right_is_sum and op in ("*", "/", "%"):
            new_body = {"type": "binop", "op": op, "left": left_expr, "right": right_expr["expression"], "sem_type": None}
            sum_node = dict(right_expr)
            sum_node["expression"] = new_body
            sum_node["sem_type"] = right_expr.get("sem_type", left_expr.get("sem_type"))
            return sum_node

        # Patch: allow boolean variables in arithmetic and sum contexts (OPL semantics)
        def normalize_type(t):
            if t == "int+":
                return "int"
            if t == "float+":
                return "float"
            return t

        left_type = normalize_type(left_expr.get("sem_type", None))
        right_type = normalize_type(right_expr.get("sem_type", None))
        # Check for tuple types: if either side is a tuple type, error unless it's a field access
        tuple_type_names = set()
        for scope in self.symbol_table.scopes:
            for sym, info in scope.items():
                if info.get("type") == "tuple_type":
                    tuple_type_names.add(sym)
        if left_type in tuple_type_names and left_expr.get("type") != "field_access":
            logger.error(
                f"[BINOP] Cannot use tuple variable '{left_expr.get('value', '?')}' of type '{left_type}' in arithmetic; use a field access like '{left_expr.get('value', '?')}.field'."
            )
            raise SemanticError(
                f"Cannot use tuple variable '{left_expr.get('value', '?')}' of type '{left_type}' in arithmetic; use a field access like '{left_expr.get('value', '?')}.field'.",
                lineno=lineno,
            )
        if right_type in tuple_type_names and right_expr.get("type") != "field_access":
            logger.error(
                f"[BINOP] Cannot use tuple variable '{right_expr.get('value', '?')}' of type '{right_type}' in arithmetic; use a field access like '{right_expr.get('value', '?')}.field'."
            )
            raise SemanticError(
                f"Cannot use tuple variable '{right_expr.get('value', '?')}' of type '{right_type}' in arithmetic; use a field access like '{right_expr.get('value', '?')}.field'.",
                lineno=lineno,
            )
        allowed_types = {"int", "float", "boolean", None}
        if left_type not in allowed_types or right_type not in allowed_types:
            logger.error(f"[BINOP] Type mismatch in arithmetic: {left_type} vs {right_type}")
            raise SemanticError(f"Type mismatch in arithmetic: {left_type} vs {right_type}", lineno)
        # Otherwise, allow (int, float, boolean) in any combination
        result_type = "float" if "float" in [left_type, right_type] else "int"
        logger.debug(f"[BINOP] Returning binop node, result_type: {result_type}")
        return {
            "type": "binop",
            "op": op,
            "left": left_expr,
            "right": right_expr,
            "sem_type": result_type,
        }

    @_("expression ',' arg_list")
    def arg_list(self, p):
        return [p.expression] + p.arg_list

    @_("expression")
    def arg_list(self, p):
        return [p.expression]

    # --- Function calls: sqrt (1 arg), maxl/minl (>=1 arg) ---
    @_("NAME '(' arg_list ')'")  # type: ignore
    def primary(self, p):
        func = p.NAME
        args = p.arg_list
        if func == "sqrt":
            if len(args) != 1:
                raise SemanticError("sqrt(...) takes exactly one argument.", lineno=p.lineno)
            return {"type": "funcall", "name": "sqrt", "args": [args[0]], "sem_type": "float"}
        if func in ("maxl", "minl"):
            if len(args) == 0:
                raise SemanticError(f"{func}(...) requires at least one argument.", lineno=p.lineno)
            # Enforce numeric args at parse-time to catch obvious mistakes early
            for a in args:
                at = a.get("sem_type")
                if at not in ("int", "int+", "float", "float+"):
                    raise SemanticError(f"{func}(...) expects numeric arguments.", lineno=p.lineno)
            sem = "float" if any(a.get("sem_type") in ("float", "float+") for a in args) else "int"
            return {"type": func, "args": args, "sem_type": sem}
        raise SemanticError(f"Unsupported function '{func}'. Only sqrt, maxl, minl are supported.", lineno=p.lineno)

    # min(i in I : cond) expr   — juxtaposition
    @_("AGG_MIN sum_index_header nonparen_expression")
    def min_expression(self, p):
        expr_type = p.nonparen_expression["sem_type"]
        if expr_type not in ("int", "int+", "float", "float+"):
            raise SemanticError("min aggregate expects numeric expression.")
        sem = "float" if expr_type in ("float", "float+") else "int"
        return {
            "type": "min_agg",
            "iterators": p.sum_index_header["iterators"],
            "index_constraint": p.sum_index_header.get("index_constraint"),
            "expression": p.nonparen_expression,
            "sem_type": sem,
        }

    # min(...) (parenthesized body)
    @_("AGG_MIN sum_index_header parenthesized_expression")
    def min_expression(self, p):
        expr_type = p.parenthesized_expression["sem_type"]
        if expr_type not in ("int", "int+", "float", "float+"):
            raise SemanticError("min aggregate expects numeric expression.")
        sem = "float" if expr_type in ("float", "float+") else "int"
        return {
            "type": "min_agg",
            "iterators": p.sum_index_header["iterators"],
            "index_constraint": p.sum_index_header.get("index_constraint"),
            "expression": p.parenthesized_expression,
            "sem_type": sem,
        }

    # max(i in I : cond) expr
    @_("AGG_MAX sum_index_header nonparen_expression")
    def max_expression(self, p):
        expr_type = p.nonparen_expression["sem_type"]
        if expr_type not in ("int", "int+", "float", "float+"):
            raise SemanticError("max aggregate expects numeric expression.")
        sem = "float" if expr_type in ("float", "float+") else "int"
        return {
            "type": "max_agg",
            "iterators": p.sum_index_header["iterators"],
            "index_constraint": p.sum_index_header.get("index_constraint"),
            "expression": p.nonparen_expression,
            "sem_type": sem,
        }

    @_("AGG_MAX sum_index_header parenthesized_expression")
    def max_expression(self, p):
        expr_type = p.parenthesized_expression["sem_type"]
        if expr_type not in ("int", "int+", "float", "float+"):
            raise SemanticError("max aggregate expects numeric expression.")
        sem = "float" if expr_type in ("float", "float+") else "int"
        return {
            "type": "max_agg",
            "iterators": p.sum_index_header["iterators"],
            "index_constraint": p.sum_index_header.get("index_constraint"),
            "expression": p.parenthesized_expression,
            "sem_type": sem,
        }

    # Allow min/max aggregates as primary
    @_("min_expression")
    def primary(self, p):
        return p.min_expression

    @_("max_expression")
    def primary(self, p):
        return p.max_expression

    # Indexed variable reference: x[i], x[i,j], etc.
    @_("NAME indexed_dimensions")  # type: ignore
    def primary(self, p):
        # Look up the symbol and check dimensions
        try:
            symbol_info = self.symbol_table.get_symbol(p.NAME)
        except SemanticError as e:
            raise SemanticError(e.message, lineno=p.lineno) from e

        # Special case: dexpr expansion on use
        if symbol_info.get("type") == "dexpr":
            val = symbol_info.get("value") or {}
            decl_dims = symbol_info.get("dimensions") or val.get("dimensions") or []
            used_dims = p.indexed_dimensions
            if len(decl_dims) != len(used_dims):
                raise SemanticError(
                    f"Incorrect number of dimensions for dexpr '{p.NAME}'. Declared {len(decl_dims)}, but used {len(used_dims)}.",
                    lineno=p.lineno,
                )
            # Build iterator -> used index mapping using declared iterator order
            iterators = val.get("iterators") or []
            idx_map = {}
            for it, used in zip(iterators, used_dims):
                idx_map[it["iterator"]] = used
            # Inline expression with substitution
            inlined = self._subst_iterators(val.get("expression"), idx_map)
            return inlined

        if not symbol_info.get("dimensions"):
            raise SemanticError(
                f"Expected indexed variable, but '{p.NAME}' is a scalar variable.",
                lineno=p.lineno,
            )

        declared_dims = symbol_info["dimensions"]
        used_dims = p.indexed_dimensions
        if len(declared_dims) != len(used_dims):
            raise SemanticError(
                f"Incorrect number of dimensions for '{p.NAME}'. Declared {len(declared_dims)}, but used {len(used_dims)}.",
                lineno=p.lineno,
            )

        processed_dims = []
        for i, (declared_dim_spec, used_index_spec) in enumerate(zip(declared_dims, used_dims)):
            dim_type = declared_dim_spec["type"]
            # Accept index expressions (binop, uminus, parenthesized_expression, field_access, etc.)
            if dim_type in ["range_index", "named_range_dimension"]:
                # Integer/range dimension: enforce integer-typed index
                if used_index_spec["type"] in [
                    "number_literal_index",
                    "name_reference_index",
                    "binop",
                    "uminus",
                    "parenthesized_expression",
                    "field_access",
                    "field_access_index",
                ]:
                    # For number_literal_index, check bounds if declared_dim_spec is range_index and bounds are constant numbers
                    if used_index_spec["type"] == "number_literal_index" and dim_type == "range_index":
                        start_bound = declared_dim_spec["start"]
                        end_bound = declared_dim_spec["end"]
                        # Only check bounds if both are AST number nodes
                        if (
                            isinstance(start_bound, dict)
                            and start_bound.get("type") == "number"
                            and isinstance(end_bound, dict)
                            and end_bound.get("type") == "number"
                        ):
                            s_val = start_bound["value"]
                            e_val = end_bound["value"]
                            if not (s_val <= used_index_spec["value"] <= e_val):
                                raise SemanticError(
                                    f"Index {used_index_spec['value']} for dimension {i+1} of '{p.NAME}' is out of declared range [{s_val}..{e_val}].",
                                    lineno=p.lineno,
                                )
                    # Otherwise, skip static check (defer to codegen/runtime)
                    # For all non-literal indices, check that the semantic type is integer
                    index_sem_type = used_index_spec.get("sem_type", None)
                    if used_index_spec["type"] != "number_literal_index":
                        # Accept field_access as index if its sem_type is int or int+
                        if index_sem_type not in ["int", "int+"]:
                            logger.debug(
                                f"[SEMANTIC] Rejecting index for dim {i+1} of '{p.NAME}': type={used_index_spec['type']}, sem_type={index_sem_type}"
                            )
                            raise SemanticError(
                                f"Index expression for dimension {i+1} of '{p.NAME}' must be integer-valued, got type '{index_sem_type}'.",
                                lineno=p.lineno,
                            )
                        else:
                            logger.debug(
                                f"[SEMANTIC] Accepting index for dim {i+1} of '{p.NAME}': type={used_index_spec['type']}, sem_type={index_sem_type}"
                            )
                    processed_dims.append(used_index_spec)
                else:
                    logger.debug(f"[SEMANTIC] Unsupported index type for integer/range dimension: {used_index_spec['type']}")
                    raise SemanticError(
                        f"Unsupported index type for integer/range dimension: {used_index_spec['type']}",
                        lineno=p.lineno,
                    )
            elif dim_type == "named_set_dimension":
                # Set dimension: allow tuple-typed index if set is a set of tuples
                set_name = declared_dim_spec["name"]
                set_info = self.symbol_table.get_symbol(set_name)
                tuple_type = None
                base_type = None
                if set_info.get("value") and isinstance(set_info["value"], dict):
                    if "tuple_type" in set_info["value"]:
                        tuple_type = set_info["value"]["tuple_type"]
                    if "base_type" in set_info["value"]:
                        base_type = set_info["value"]["base_type"]

                if tuple_type:
                    # Accept index if its sem_type matches the tuple type (or is a tuple_literal)
                    idx_type = used_index_spec.get("type")
                    idx_sem_type = used_index_spec.get("sem_type")
                    if idx_type in ["name_reference_index", "name"] and idx_sem_type == tuple_type:
                        processed_dims.append(used_index_spec)
                    elif idx_type == "tuple_literal":
                        processed_dims.append(used_index_spec)
                    else:
                        raise SemanticError(
                            f"Index expression for tuple set dimension {i+1} of '{p.NAME}' must be of tuple type '{tuple_type}', got type '{idx_sem_type}'.",
                            lineno=p.lineno,
                        )
                else:
                    # Typed scalar set: require the index to match the set's base type (OPL semantics).
                    # If the set is untyped (no base_type), allow string or integer indices for compatibility.
                    idx_sem_type = used_index_spec.get("sem_type")
                    if base_type:
                        if idx_sem_type != base_type:
                            raise SemanticError(
                                f"Index expression for set dimension {i+1} of '{p.NAME}' must be {base_type}-valued, got type '{idx_sem_type}'.",
                                lineno=p.lineno,
                            )
                        processed_dims.append(used_index_spec)
                    else:
                        # Untyped set: accept string or integer iterator indices
                        if idx_sem_type not in ["string", "int", "int+"]:
                            raise SemanticError(
                                f"Index expression for set dimension {i+1} of '{p.NAME}' must be string- or integer-valued, got type '{idx_sem_type}'.",
                                lineno=p.lineno,
                            )
                        processed_dims.append(used_index_spec)
            else:
                raise SemanticError(
                    f"Dimension {i+1} of '{p.NAME}' is not indexable. Declared as type: {dim_type}.",
                    lineno=p.lineno,
                )

        # For tuple_array, expose underlying tuple_type as semantic type so field access works
        sem_type = symbol_info["type"]
        if sem_type == "tuple_array":
            val = symbol_info.get("value") or {}
            tuple_type = val.get("tuple_type")
            if tuple_type:
                sem_type = tuple_type
        return {
            "type": "indexed_name",
            "name": p.NAME,
            "dimensions": processed_dims,
            "sem_type": sem_type,
        }

    # --- Support for parameter declarations with direct value assignment ---
    @_('opt_PARAM type NAME "=" signed_number ";"')  # type: ignore
    def declaration(self, p):
        # Scalar parameter with direct value assignment (allow negatives)
        name = p.NAME
        var_type = p.type
        value = p.signed_number
        self.symbol_table.add_symbol(name, var_type, value=value, is_dvar=False, lineno=p.lineno)
        return {
            "type": "parameter_inline",
            "var_type": var_type,
            "name": name,
            "value": value,
        }

    # NEW: scalar parameter with general expression on RHS (e.g., float C = 5 / 6;)
    @_('opt_PARAM type NAME "=" expression ";"')  # type: ignore
    def declaration(self, p):
        name = p.NAME
        var_type = p.type
        expr = p.expression
        # Downcast literal RHS to parameter_inline for codegen/test compatibility
        if isinstance(expr, dict) and expr.get("type") == "number":
            val = expr.get("value")
            self.symbol_table.add_symbol(name, var_type, value=val, is_dvar=False, lineno=p.lineno)
            return {
                "type": "parameter_inline",
                "var_type": var_type,
                "name": name,
                "value": val,
            }
        # Otherwise, keep as expression (handled later in compile pipeline)
        self.symbol_table.add_symbol(name, var_type, is_dvar=False, lineno=p.lineno)
        return {
            "type": "parameter_inline_expr",
            "var_type": var_type,
            "name": name,
            "expression": expr,
        }

    # --- computed indexed parameter from expression with iterators, e.g.
    # float sqrt_demand[t in T] = sqrt(demand[t]);
    @_('opt_PARAM type NAME dexpr_index_header "=" expression ";"')  # type: ignore
    def declaration(self, p):
        name = p.NAME
        var_type = p.type
        iterators = p.dexpr_index_header["iterators"]

        def to_decl_dim(rng):
            if rng["type"] == "range_specifier":
                return {"type": "range_index", "start": rng["start"], "end": rng["end"]}
            if rng["type"] == "named_range":
                return {"type": "named_range_dimension", "name": rng["name"], "start": rng.get("start"), "end": rng.get("end")}
            if rng["type"] == "named_set":
                return {"type": "named_set_dimension", "name": rng["name"]}
            return {"type": rng["type"], **{k: v for k, v in rng.items() if k != "type"}}

        dimensions = [to_decl_dim(it["range"]) for it in iterators]

        # Close the iterator scope before adding the parameter symbol
        try:
            if p.dexpr_index_header.get("_iterator_scope_opened"):
                self.symbol_table.exit_scope()
        except Exception:
            pass

        # Register symbol in outer scope
        self.symbol_table.add_symbol(
            name,
            var_type,
            dimensions=dimensions,
            is_dvar=False,
            lineno=p.lineno,
        )

        return {
            "type": "parameter_inline_indexed_expr",
            "var_type": var_type,
            "name": name,
            "iterators": iterators,
            "dimensions": dimensions,
            "expression": p.expression,
        }

    # NEW: computed indexed parameter with strict OPL nested headers: float W[i in I][j in J] = ...
    # NEW: float W[i in I][j in J] = ...
    @_('opt_PARAM type NAME dexpr_index_headers "=" expression ";"')  # type: ignore
    def declaration(self, p):
        name = p.NAME
        var_type = p.type
        iterators = p.dexpr_index_headers["iterators"]

        def to_decl_dim(rng):
            if rng["type"] == "range_specifier":
                return {"type": "range_index", "start": rng["start"], "end": rng["end"]}
            if rng["type"] == "named_range":
                return {"type": "named_range_dimension", "name": rng["name"], "start": rng.get("start"), "end": rng.get("end")}
            if rng["type"] == "named_set":
                return {"type": "named_set_dimension", "name": rng["name"]}
            return {"type": rng["type"], **{k: v for k, v in rng.items() if k != "type"}}

        dimensions = [to_decl_dim(it["range"]) for it in iterators]

        # Close iterator scope before adding the symbol
        try:
            if p.dexpr_index_headers.get("_iterator_scope_opened"):
                self.symbol_table.exit_scope()
        except Exception:
            pass

        self.symbol_table.add_symbol(
            name,
            var_type,
            dimensions=dimensions,
            is_dvar=False,
            lineno=p.lineno,
        )

        return {
            "type": "parameter_inline_indexed_expr",
            "var_type": var_type,
            "name": name,
            "iterators": iterators,
            "dimensions": dimensions,
            "expression": p.expression,
        }

    @_('opt_PARAM type NAME indexed_dimensions "=" array_value ";"')  # type: ignore
    def declaration(self, p):
        # Indexed parameter with direct value assignment (e.g., float w[1..5] = [1,2,3,4,5];)
        name = p.NAME
        var_type = p.type
        dimensions = p.indexed_dimensions
        value = p.array_value
        processed_dimensions = []
        for dim_spec in dimensions:
            if dim_spec["type"] == "range_index":
                # Always store start/end as AST nodes (do not convert to int)
                start = dim_spec["start"]
                end = dim_spec["end"]
                if isinstance(start, int):
                    start = {"type": "number", "value": start, "sem_type": "int"}
                if isinstance(end, int):
                    end = {"type": "number", "value": end, "sem_type": "int"}
                processed_dimensions.append({"type": "range_index", "start": start, "end": end})
            elif dim_spec["type"] == "name_reference_index":
                dim_name = dim_spec["name"]
                try:
                    symbol_info = self.symbol_table.get_symbol(dim_name)
                    if symbol_info["type"] == "range":
                        if symbol_info["value"] is not None:
                            processed_dimensions.append(
                                {
                                    "type": "named_range_dimension",
                                    "name": dim_name,
                                    "start": symbol_info["value"]["start"],
                                    "end": symbol_info["value"]["end"],
                                }
                            )
                        else:
                            processed_dimensions.append({"type": "named_range_dimension", "name": dim_name})
                    elif symbol_info["type"] == "set":
                        processed_dimensions.append({"type": "named_set_dimension", "name": dim_name})
                    else:
                        raise SemanticError(
                            f"Symbol '{dim_name}' used as dimension must be a 'range' or 'set', but found '{symbol_info['type']}'.",
                            lineno=p.lineno,
                        )
                except SemanticError as e:
                    raise SemanticError(e.message, lineno=p.lineno) from e
            elif dim_spec["type"] == "number_literal_index":
                raise SemanticError(
                    f"Single number index '{dim_spec['value']}' not allowed in declaration dimensions. Use 'range' like [1..N] or a named 'set'/'range'.",
                    lineno=p.lineno,
                )
            else:
                raise SemanticError(
                    f"Unsupported dimension type in declaration: {dim_spec['type']}",
                    lineno=p.lineno,
                )
        self.symbol_table.add_symbol(
            name,
            var_type,
            dimensions=processed_dimensions,
            value=value,
            is_dvar=False,
            lineno=p.lineno,
        )
        return {
            "type": "parameter_inline_indexed",
            "var_type": var_type,
            "name": name,
            "dimensions": processed_dimensions,
            "value": value,
        }

    # --- Tuple array grammar support ---
    # External tuple array: tupleType Arr[Set] = ...; (declare dimensions so existing indexed variable rule works)
    @_('NAME NAME "[" NAME "]" "=" ELLIPSIS ";"')  # type: ignore
    def declaration(self, p):
        tuple_type = p.NAME0
        array_name = p.NAME1
        index_set = p.NAME2
        dimensions = [{"type": "named_set_dimension", "name": index_set}]
        self.symbol_table.add_symbol(
            array_name,
            "tuple_array",
            value={"tuple_type": tuple_type, "index_set": index_set},
            dimensions=dimensions,
            lineno=p.lineno,
        )
        return {
            "type": "tuple_array_external",
            "tuple_type": tuple_type,
            "name": array_name,
            "index_set": index_set,
            "dimensions": dimensions,
            "value": None,
        }

    # Uninitialized tuple array: tupleType Arr[Set];
    @_('NAME NAME "[" NAME "]" ";"')  # type: ignore
    def declaration(self, p):
        tuple_type = p.NAME0
        array_name = p.NAME1
        index_set = p.NAME2
        dimensions = [{"type": "named_set_dimension", "name": index_set}]
        self.symbol_table.add_symbol(
            array_name,
            "tuple_array",
            value={"tuple_type": tuple_type, "index_set": index_set, "elements": None},
            dimensions=dimensions,
            lineno=p.lineno,
        )
        return {
            "type": "tuple_array",
            "tuple_type": tuple_type,
            "name": array_name,
            "index_set": index_set,
            "dimensions": dimensions,
            "value": None,
        }

    # --- element_list (model parser) for typed scalar sets ---
    # NOTE: Duplicate string-only rules removed here to avoid SLY duplicate productions.
    # The canonical element_list (with string/int/float/boolean variants) is defined earlier in this class.

    # --- Nested array_value support for inline parameter initialization in model files ---
    # Replace minimal array_elements-based rules with nested row_list to allow 2D/3D arrays.
    @_('"[" row_list "]"')
    def array_value(self, p):
        return p.row_list

    # Allow rows to contain general scalar values (NUMBER, STRING_LITERAL, BOOLEAN_LITERAL),
    # not just NUMBER, to match .dat file capabilities.
    @_('row_list "," scalar_value')
    def row_list(self, p):
        return p.row_list + [p.scalar_value]

    @_("scalar_value")
    def row_list(self, p):
        return [p.scalar_value]

    # Nested arrays remain supported
    @_('row_list "," array_value')
    def row_list(self, p):
        return p.row_list + [p.array_value]

    @_("array_value")
    def row_list(self, p):
        return [p.array_value]

    # General scalar values usable in inline model arrays
    @_("signed_number")
    def scalar_value(self, p):
        return p.signed_number

    @_("STRING_LITERAL")
    def scalar_value(self, p):
        return p.STRING_LITERAL

    @_("BOOLEAN_LITERAL")
    def scalar_value(self, p):
        return p.BOOLEAN_LITERAL


# --- Parser for .dat files ---
class OPLDataLexer(Lexer):
    """
    Lexer for OPL .dat files.
    """

    tokens = {
        "BOOLEAN_LITERAL",
        "STRING_LITERAL",
        "NAME",
        "NUMBER",
        "DOTDOT",
    }

    ignore = " \t\r"

    literals = {"=", ";", "{", "}", "[", "]", ",", ":", "<", ">"}

    def __init__(self):
        self.lineno = 1

    # --- Token rules ---

    DOTDOT = r"\.\."

    @_(r"true|false")  # type: ignore
    def BOOLEAN_LITERAL(self, t):
        t.value = t.value.lower() == "true"
        return t

    @_(r'"[^"]*"')  # type: ignore
    def STRING_LITERAL(self, t):
        return t

    # Identifiers (variable names, etc.)
    NAME = r"[a-zA-Z_][a-zA-Z0-9_]*"

    # Signed numbers (integers or floats)
    @_(r"[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)")  # type: ignore
    def NUMBER(self, t):
        if "." in str(t.value) or "e" in str(t.value).lower():
            t.value = float(t.value)
        else:
            t.value = int(t.value)
        return t

    @_(r"\n+")  # type: ignore
    def ignore_newline(self, t):
        self.lineno += t.value.count("\n")

    @_(r"#.*")  # type: ignore
    def ignore_hash_comment(self, t):
        pass

    @_(r"//.*")  # type: ignore
    def ignore_line_comment(self, t):
        pass

    @_(r"/\*[\s\S]*?\*/")  # type: ignore
    def ignore_block_comment(self, t):
        self.lineno += t.value.count("\n")

    def error(self, t):
        raise SemanticError(f"Illegal character in .dat file: '{t.value[0]}'.", lineno=self.lineno)


# --- Parser for .dat files ---
class OPLDataParser(Parser):
    @_('"{" NAME "}" NAME "=" "{" element_list "}" ";"')  # type: ignore
    def data_declaration(self, p):
        # Clear error for typed scalar-set prefix in .dat
        raise SemanticError(
            "Typed scalar set prefixes (e.g. '{string} S = {...};') are not allowed in .dat files. "
            "Declare the typed set in the model (.mod) and use 'S = {...};' in the data file.",
            lineno=getattr(self.lexer, "lineno", None),
        )

    @_('"{" NAME "}" NAME "=" "{" tuple_literal_list "}" ";"')  # type: ignore
    def data_declaration(self, p):
        # Clear error for typed set-of-tuples prefix in .dat
        raise SemanticError(
            "Typed set-of-tuples prefix ('{TupleType} S = {...};') is not allowed in .dat files. "
            "Declare '{TupleType} S;' in the model and use 'S = { <...>, ... };' in the data file.",
            lineno=getattr(self.lexer, "lineno", None),
        )

    @_("tuple_literal")  # type: ignore
    def tuple_element(self, p):
        return p.tuple_literal

    # --- Untyped set-of-tuples assignment: arcs = { <...>, <...> }; ---
    @_('NAME "=" "{" tuple_literal_list "}" ";"')  # type: ignore
    @_('NAME "=" "[" tuple_literal_list "]" ";"')  # type: ignore
    def data_declaration(self, p):
        # Robustly handle all tuple/set/array assignments, including nested tuples
        # NAME = { <tuple>, ... };
        if hasattr(p, "NAME") and hasattr(p, "tuple_literal_list") and len(p) > 2 and p[2] == "{":
            self.data[p.NAME] = p.tuple_literal_list
            return {
                "type": "set_of_tuples_untyped",
                "name": p.NAME,
                "value": p.tuple_literal_list,
            }
        # NAME = [ <tuple>, ... ];
        elif hasattr(p, "NAME") and hasattr(p, "tuple_literal_list") and len(p) > 2 and p[2] == "[":
            self.data[p.NAME] = p.tuple_literal_list
            return {
                "type": "tuple_array_data",
                "name": p.NAME,
                "value": p.tuple_literal_list,
            }
        # {Type} NAME = { <tuple>, ... };
        elif hasattr(p, "NAME0") and hasattr(p, "NAME1") and hasattr(p, "tuple_literal_list"):
            self.data[p.NAME1] = p.tuple_literal_list
            return {
                "type": "set_of_tuples",
                "tuple_type": p.NAME0,
                "name": p.NAME1,
                "value": p.tuple_literal_list,
            }
        # Fallback: try to handle nested tuple or future forms
        elif hasattr(p, "NAME") and hasattr(p, "tuple_literal_list"):
            self.data[p.NAME] = p.tuple_literal_list
            return {
                "type": "tuple_or_set",
                "name": p.NAME,
                "value": p.tuple_literal_list,
            }
        else:
            raise Exception(f"Unrecognized tuple/set data_declaration: {p}")

    # Fix: Only add to the list if p.tuple_literal is not None (prevents extra split on nested commas)
    @_('tuple_literal_list "," tuple_literal')  # type: ignore
    def tuple_literal_list(self, p):
        return p.tuple_literal_list + [p.tuple_literal]

    @_("tuple_literal")  # type: ignore
    def tuple_literal_list(self, p):
        return [p.tuple_literal]

    @_('"<" tuple_element_list ">"')  # type: ignore
    def tuple_literal(self, p):
        return tuple(p.tuple_element_list)

    @_('tuple_element_list "," tuple_element')  # type: ignore
    def tuple_element_list(self, p):
        return p.tuple_element_list + [p.tuple_element]

    @_("tuple_element")  # type: ignore
    def tuple_element_list(self, p):
        return [p.tuple_element]

    @_("NUMBER")  # type: ignore
    def tuple_element(self, p):
        return p.NUMBER

    @_("STRING_LITERAL")  # type: ignore
    def tuple_element(self, p):
        return p.STRING_LITERAL.strip('"')

    @_("BOOLEAN_LITERAL")  # type: ignore
    def tuple_element(self, p):
        return p.BOOLEAN_LITERAL

    """
    Parser for OPL .dat files.
    Builds a dictionary of data.
    """
    tokens = OPLDataLexer.tokens
    start = "data_file"

    def __init__(self):
        self.data = {}
        self.lexer = None
        # Track last token line for EOF diagnostics when lexer is not available
        self._last_token_lineno = 1
        # NEW: keep a per-name line number map
        self.name_linenos = {}

    def parse(self, tokens, lexer=None):
        self.lexer = lexer
        self.data = {}
        # Materialize tokens to capture last token line; feed iterator to SLY
        tok_list = list(tokens)
        if tok_list:
            try:
                self._last_token_lineno = getattr(tok_list[-1], "lineno", 1)
            except Exception:
                self._last_token_lineno = 1
        else:
            self._last_token_lineno = 1
        return super().parse(iter(tok_list))

    @_("data_declaration_list")  # type: ignore
    def data_file(self, p):
        return self.data

    @_("")  # type: ignore
    def data_declaration_list(self, p):
        # Allow empty .dat files
        return []

    @_("data_declaration_list data_declaration")  # type: ignore
    def data_declaration_list(self, p):
        # Accept a sequence of data_declaration statements
        return p.data_declaration_list + [p.data_declaration]

    @_("data_declaration")  # type: ignore
    def data_declaration_list(self, p):
        return [p.data_declaration]

    @_(
        'NAME "=" scalar_value ";"',
        'NAME "=" set_value ";"',
        'NAME "=" array_value ";"',
        'NAME "=" key_value_array ";"',
    )  # type: ignore
    def data_declaration(self, p):
        # NEW: remember the line for this name
        try:
            self.name_linenos[p.NAME] = getattr(self.lexer, "lineno", self._last_token_lineno)
        except Exception:
            pass
        # Handle all scalar, set, array, and key_value_array assignments
        if hasattr(p, "scalar_value"):
            self.data[p.NAME] = p.scalar_value
            return {"type": "param", "name": p.NAME, "value": p.scalar_value}
        elif hasattr(p, "set_value"):
            self.data[p.NAME] = p.set_value
            return {"type": "set", "name": p.NAME, "value": p.set_value}
        elif hasattr(p, "array_value"):
            self.data[p.NAME] = p.array_value
            return {"type": "array", "name": p.NAME, "value": p.array_value}
        elif hasattr(p, "key_value_array"):
            self.data[p.NAME] = p.key_value_array
            return {"type": "key_value_array", "name": p.NAME, "value": p.key_value_array}
        else:
            raise Exception("Unrecognized data_declaration assignment")

    @_('NAME "=" NUMBER DOTDOT NUMBER ";"')  # type: ignore
    def data_declaration(self, p):
        start_val = p.NUMBER0
        end_val = p.NUMBER1
        # Disallow negative range bounds in .dat files
        if not isinstance(start_val, int) or not isinstance(end_val, int):
            raise SemanticError(
                f"Range bounds in .dat file must be integers, got {type(start_val).__name__} and {type(end_val).__name__}.",
                lineno=self.lexer.lineno,
            )
        if start_val < 0 or end_val < 0:
            raise SemanticError(
                f"Range bounds in .dat file must be non-negative, got {start_val}..{end_val}.",
                lineno=self.lexer.lineno,
            )
        if start_val > end_val:
            raise SemanticError(
                f"Range start ({start_val}) cannot be greater than end ({end_val}).",
                lineno=self.lexer.lineno,
            )
        self.data[p.NAME] = {"start": start_val, "end": end_val, "type": "range_data"}
        return {
            "type": "range_assignment_data",
            "name": p.NAME,
            "value": {"start": start_val, "end": end_val},
        }

    # --- Key-value array support ---
    @_('"[" key_value_row_list "]"')  # type: ignore
    def key_value_array(self, p):
        # Return as dict for easy lookup
        return dict(p.key_value_row_list)

    @_('key_value_row_list "," key_value_row')  # type: ignore
    def key_value_row_list(self, p):
        p.key_value_row_list.append(p.key_value_row)
        return p.key_value_row_list

    @_("key_value_row")  # type: ignore
    def key_value_row_list(self, p):
        return [p.key_value_row]

    # String label row: "Seattle" 350
    @_("STRING_LITERAL scalar_value")  # type: ignore
    def key_value_row(self, p):
        return (p.STRING_LITERAL.strip('"'), p.scalar_value)

    # Tuple label row: <...> scalar_value
    @_("tuple_literal scalar_value")  # type: ignore
    def key_value_row(self, p):
        return (p.tuple_literal, p.scalar_value)

    # NEW: String label row with array value: "StoreA" [1,2,3]
    @_("STRING_LITERAL array_value")  # type: ignore
    def key_value_row(self, p):
        return (p.STRING_LITERAL.strip('"'), p.array_value)

    # NEW: Tuple label row with array value: <"StoreA"> [1,2,3]
    @_("tuple_literal array_value")  # type: ignore
    def key_value_row(self, p):
        return (p.tuple_literal, p.array_value)

    # Allow trailing comma (optional)
    @_('key_value_row_list ","')  # type: ignore
    def key_value_row_list(self, p):
        return p.key_value_row_list

    @_("NUMBER")  # type: ignore
    def scalar_value(self, p):
        return p.NUMBER

    @_("STRING_LITERAL")  # type: ignore
    def scalar_value(self, p):
        return p.STRING_LITERAL.strip('"')

    @_("BOOLEAN_LITERAL")  # type: ignore
    def scalar_value(self, p):
        return p.BOOLEAN_LITERAL

    @_('"{" element_list "}"')  # type: ignore
    def set_value(self, p):
        return p.element_list

    @_("scalar_value")  # type: ignore
    def element_list(self, p):
        return [p.scalar_value]

    @_('element_list "," scalar_value')  # type: ignore
    def element_list(self, p):
        p.element_list.append(p.scalar_value)
        return p.element_list

    # --- Nested array support for .dat files ---
    @_('"[" row_list "]"')  # type: ignore
    def array_value(self, p):
        return p.row_list

    @_('row_list "," scalar_value')  # type: ignore
    def row_list(self, p):
        p.row_list.append(p.scalar_value)
        return p.row_list

    @_("scalar_value")  # type: ignore
    def row_list(self, p):
        return [p.scalar_value]

    # Add support for nested arrays (e.g., [ [1,2], [3,4] ])
    @_('row_list "," array_value')  # type: ignore
    def row_list(self, p):
        p.row_list.append(p.array_value)
        return p.row_list

    @_("array_value")  # type: ignore
    def row_list(self, p):
        return [p.array_value]

    def error(self, p):
        # Unexpected token
        if p is not None:
            lineno = getattr(p, "lineno", None)
            if lineno is None:
                lineno = getattr(self.lexer, "lineno", self._last_token_lineno)
            if p.type == "NAME":
                raise SemanticError(
                    f"Syntax error in .dat file at or near token NAME, value '{p.value}'. "
                    "Hint: .dat files must contain plain data assignments such as 'nbJobs = 3;', 'S = { ... };', or 'cost = [ ... ];'. "
                    "Do not include model-style declarations like 'int nbJobs = 3;', 'float cost[I] = ...;', 'param ...', or any 'dvar', 'minimize', or 'subject to' blocks in the .dat file.",
                    lineno=lineno,
                )
            if p.type == "NUMBER":
                raise SemanticError(
                    f"Syntax error in .dat file at or near token NUMBER, value '{p.value}'. "
                    "Hint: keyed arrays in .dat files accept string keys like '\"S1\" 0.25' or tuple keys like '<\"S1\"> 0.25', "
                    "but not bare numeric keys like '1 0.25'. If the parameter is indexed by scenario order, use a plain array such as "
                    "'[0.25, 0.25, 0.25, 0.25]'; otherwise switch to string or tuple-labeled keys that match the model index set.",
                    lineno=lineno,
                )
            if p.type == "<":
                raise SemanticError(
                    "Syntax error in .dat file at or near token <, value '<'. "
                    "Hint: tuple literals '<...>' are only accepted in flat tuple collections such as 'S = { <...>, <...> };' "
                    "or 'A = [ <...>, <...> ];'. Nested tuple structures such as arrays of tuple-lists are not supported in .dat files; "
                    "rewrite them as a supported flat collection or use plain scalar/array data instead.",
                    lineno=lineno,
                )
            raise SemanticError(
                f"Syntax error in .dat file at or near token {p.type}, value '{p.value}'.",
                lineno=lineno,
            )
        # Unexpected EOF
        eof_line = getattr(self.lexer, "lineno", self._last_token_lineno)
        raise SemanticError("Syntax error in .dat file at end of file (EOF).", lineno=eof_line)


class OPLCompiler:
    """
    Orchestrates the OPL compilation process, from parsing .mod and .dat files
    to generating and potentially executing GurobiPy code.
    """

    @staticmethod
    def _normalize_syntax_error_reporting(syntax_error_reporting: str = "full") -> str:
        reporting = syntax_error_reporting.strip().lower()
        if reporting not in SYNTAX_ERROR_REPORTING_MODES:
            valid = ", ".join(sorted(SYNTAX_ERROR_REPORTING_MODES))
            raise ValueError(f"Unknown syntax_error_reporting: {syntax_error_reporting}. Valid options: {valid}")
        return reporting

    def __init__(self, syntax_error_reporting: str = "full"):
        self.model_lexer = OPLLexer()
        self.model_parser = OPLParser()
        self.data_lexer = OPLDataLexer()
        self.data_parser = OPLDataParser()
        self.syntax_error_reporting = self._normalize_syntax_error_reporting(syntax_error_reporting)

    def _raise_masked_syntax_error(self, exc: SemanticError) -> None:
        if self.syntax_error_reporting == "masked":
            raise SyntaxError("Syntax error") from None
        lineno = getattr(exc, "lineno", None)
        if lineno is None:
            lineno = getattr(self.model_parser, "_last_lineno", None)
        if lineno is None:
            lineno = getattr(self.data_parser, "_last_token_lineno", None)
        if lineno is None:
            raise SyntaxError("Syntax error") from None
        raise SyntaxError(f"Syntax error on line {lineno}") from None

    def compile_model(
        self,
        model_code: str,
        data_code: Optional[str] = None,
        solver: str = "gurobi",
        syntax_error_reporting: Optional[str] = None,
    ):
        """
        Compiles an OPL model and optional data into solver-specific code.

        Args:
            model_code (str): The OPL model code string.
            data_code (str, optional): The OPL data code string.
            solver (str, optional): The solver to use ('gurobi' or 'scipy'). Defaults to 'gurobi'.

        Returns:
            tuple: (ast, code_str, data_dict) if successful.

        Raises:
            SemanticError: If there's an error during lexing, parsing, or semantic analysis.
            Exception: For unexpected errors.
        """
        effective_reporting = (
            self.syntax_error_reporting
            if syntax_error_reporting is None
            else self._normalize_syntax_error_reporting(syntax_error_reporting)
        )
        should_mask_errors = effective_reporting != "full"
        if should_mask_errors:
            try:
                return self.compile_model(model_code, data_code, solver=solver, syntax_error_reporting="full")
            except SemanticError as exc:
                previous_reporting = self.syntax_error_reporting
                self.syntax_error_reporting = effective_reporting
                try:
                    self._raise_masked_syntax_error(exc)
                finally:
                    self.syntax_error_reporting = previous_reporting

        data_dict = {}
        model_ast = None
        ast: dict[str, Any] = {}
        code = ""
        if data_code:
            # Tokenize and parse data file
            data_tokens = self.data_lexer.tokenize(data_code)
            data_dict = self.data_parser.parse(data_tokens, lexer=self.data_lexer)

        # Tokenize and parse model file
        model_tokens = list(self.model_lexer.tokenize(model_code))
        # Debug: print token stream for model
        logger.debug("[TOKEN_STREAM] Model tokens:")
        for t in model_tokens:
            logger.debug(f"  type={t.type}, value={t.value}")
        model_ast = self.model_parser.parse(iter(model_tokens))

        # --- Inject parameter assignments from model AST into data_dict ---
        if model_ast and "declarations" in model_ast:
            for decl in model_ast["declarations"]:
                # Handle scalar and array parameter assignments
                if decl.get("type", "").startswith("parameter"):
                    name = decl.get("name")
                    value = decl.get("value")
                    if value is not None:
                        data_dict[name] = value
                # Handle array assignments (e.g., float demand[1..T] = [...])
                if decl.get("type", "") == "parameter_array":
                    name = decl.get("name")
                    value = decl.get("value")
                    if value is not None:
                        data_dict[name] = value

        # Build a single canonical working_data that merges .dat data with inline parameter values
        working_data = dict(data_dict or {})
        # Inject inline declarations (parameters, typed sets, tuple-array inline values) into working_data
        if model_ast and "declarations" in model_ast:
            for decl in model_ast["declarations"]:
                t = decl.get("type")
                name = decl.get("name")  # ensure we have the declaration name for all branches
                if not name:
                    continue
                # Inline scalar/array parameters (parameter_inline / parameter_inline_indexed)
                if t in ("parameter_inline", "parameter_inline_indexed") and decl.get("value") is not None:
                    working_data[name] = decl["value"]
                # Typed sets declared inline
                if t == "typed_set" and decl.get("value") is not None and name not in working_data:
                    working_data[name] = decl["value"]
                # Sets-of-tuples declared inline
                if t == "set_of_tuples" and decl.get("value") is not None:
                    elems = []
                    for v in decl["value"]:
                        if isinstance(v, dict) and "elements" in v:
                            elems.append(v["elements"])
                        else:
                            elems.append(v)
                    working_data[name] = {
                        "elements": elems,
                        "tuple_type": decl.get("tuple_type"),
                    }

        # NEW: reserved-name guard across declarations and data keys (prevents emitting e.g. 'len = {...}')
        def _reject_reserved_names():
            # Check declared names
            declared_names = {
                d.get("name")
                for d in (model_ast.get("declarations") or [])
                if isinstance(d, dict) and isinstance(d.get("name"), str)
            }
            bad_decl = declared_names & RESERVED_PY_IDENTIFIERS
            if bad_decl:
                bad = sorted(bad_decl)[0]
                raise SemanticError(
                    f"Identifier '{bad}' is reserved and cannot be used as a model symbol. "
                    f"Please rename it in the .mod file."
                )
            # Check data keys as well, since generators may emit all keys from data_dict
            bad_data = set(working_data.keys()) & RESERVED_PY_IDENTIFIERS
            if bad_data:
                bad = sorted(bad_data)[0]
                ln = getattr(self.data_parser, "name_linenos", {}).get(bad)
                raise SemanticError(
                    f"Identifier '{bad}' is reserved and cannot appear as a data key (would shadow Python keywords or built-ins). "
                    f"Please rename it in the .dat or model data.",
                    lineno=ln,
                )

        _reject_reserved_names()

        # --- evaluate typed set-of-tuples comprehensions into concrete sets (must happen BEFORE computed params) ---
        # Local helpers used by comprehensions to evaluate integer bounds and named ranges against working_data.
        def eval_bound(expr):
            if isinstance(expr, dict):
                t = expr.get("type")
                if t == "number":
                    return int(expr.get("value"))
                if t == "name":
                    val = working_data.get(expr.get("value"))
                    if isinstance(val, (int, float)):
                        return int(val)
                    raise SemanticError(f"Unknown name in range bound: {expr.get('value')}")
                if t == "binop":
                    op = expr.get("op")
                    left = eval_bound(expr.get("left"))
                    right = eval_bound(expr.get("right"))
                    if op == "+":
                        return left + right
                    if op == "-":
                        return left - right
                    if op == "*":
                        return left * right
                    if op == "/":
                        # integer division for bounds
                        return int(left / right)
            raise SemanticError(f"Unsupported bound expr: {expr}")

        def resolve_named_range(rng_name: str):
            # Try inline range declaration in the model
            rng_decl = next(
                (
                    d
                    for d in (model_ast.get("declarations") or [])
                    if isinstance(d, dict) and d.get("type") == "range_declaration_inline" and d.get("name") == rng_name
                ),
                None,
            )
            if rng_decl:
                s = eval_bound(rng_decl["start"])
                e = eval_bound(rng_decl["end"])
                return s, e
            # Try .dat-provided range
            dv = working_data.get(rng_name)
            if isinstance(dv, dict) and dv.get("type") == "range_data":
                return int(dv["start"]), int(dv["end"])
            raise SemanticError(f"Named range '{rng_name}' not found for set comprehension.")

        if model_ast and "declarations" in model_ast:
            new_decls2 = []
            for decl in model_ast["declarations"]:
                if decl.get("type") != "set_of_tuples_comprehension":
                    new_decls2.append(decl)
                    continue

                comp = decl.get("comprehension") or {}
                tuple_expr = comp.get("tuple_expr")
                iterators = comp.get("iterators") or []
                idxc = comp.get("index_constraint")

                # Domain resolution for each iterator (range, named range, named set)
                def _domain_for_range(rng):
                    if rng["type"] == "range_specifier":
                        s = eval_bound(rng["start"])
                        e = eval_bound(rng["end"])
                        return list(range(int(s), int(e) + 1))
                    if rng["type"] == "named_range":
                        s, e = resolve_named_range(rng["name"])
                        return list(range(int(s), int(e) + 1))
                    if rng["type"] in ("named_set", "named_set_dimension"):
                        set_name = rng["name"]
                        set_obj = working_data.get(set_name, [])
                        if isinstance(set_obj, dict) and "elements" in set_obj:
                            elems = set_obj["elements"]
                        else:
                            elems = set_obj
                        return list(elems or [])
                    raise SemanticError(
                        f"Unsupported iterator range type '{rng['type']}' in set comprehension for '{decl.get('name')}'."
                    )

                it_names = [it["iterator"] for it in iterators]
                domains = [_domain_for_range(it["range"]) for it in iterators]

                # Evaluate tuple expression into a Python tuple under env
                def _eval_tuple(expr, env):
                    if isinstance(expr, dict) and expr.get("type") == "tuple_literal":
                        out = []
                        for el in expr.get("elements", []):
                            out.append(_eval_tuple(el, env))
                        return tuple(out)
                    if isinstance(expr, dict):
                        # Reuse the later eval_expr if present, else minimally handle names and numbers
                        t = expr.get("type")
                        if t == "name":
                            return env.get(expr.get("value"))
                        if t == "number":
                            return expr.get("value")
                        if t == "parenthesized_expression":
                            return _eval_tuple(expr.get("expression"), env)
                    return expr

                tuples = []

                # Nested loops over cartesian product of all iterator domains
                def _recurse(depth, env):
                    if depth == len(it_names):
                        # filter
                        keep = True
                        if idxc is not None:
                            # very simple boolean evaluation: treat nonzero numeric as True
                            def _eval_bool(e, envb):
                                if isinstance(e, dict):
                                    if e.get("type") == "number":
                                        return bool(e.get("value"))
                                    if e.get("type") == "name":
                                        return bool(envb.get(e.get("value")))
                                    if e.get("type") == "parenthesized_expression":
                                        return _eval_bool(e.get("expression"), envb)
                                return bool(e)

                            keep = _eval_bool(idxc, env)
                        if keep:
                            tval = _eval_tuple(tuple_expr, env)

                            # Normalize nested numeric to int when integral
                            def _norm(v):
                                if isinstance(v, float) and v.is_integer():
                                    return int(v)
                                if isinstance(v, tuple):
                                    return tuple(_norm(x) for x in v)
                                return v

                            tuples.append(_norm(tval))
                        return
                    nm = it_names[depth]
                    for v in domains[depth]:
                        env[nm] = v
                        _recurse(depth + 1, env)
                    env.pop(nm, None)

                _recurse(0, {})
                # Mutate working_data and AST: concrete set as list of tuples
                working_data[decl["name"]] = tuples
                new_decls2.append(
                    {
                        "type": "set_of_tuples",
                        "tuple_type": decl.get("tuple_type"),
                        "name": decl.get("name"),
                        "value": tuples,
                    }
                )
            model_ast["declarations"] = new_decls2
            data_dict = dict(working_data)

        # Use working_data for subsequent validation/emission
        data_dict = working_data

        # NEW: evaluate computed indexed parameter declarations and rewrite them into concrete inline params
        if model_ast and "declarations" in model_ast:
            import math

            # Helpers to evaluate simple int bounds from AST using working_data
            def eval_bound(expr):
                if isinstance(expr, dict):
                    t = expr.get("type")
                    if t == "number":
                        return int(expr.get("value"))
                    if t == "name":
                        # named range bound could reference a number param
                        val = working_data.get(expr.get("value"))
                        if isinstance(val, (int, float)):
                            return int(val)
                        raise SemanticError(f"Unknown name in range bound: {expr.get('value')}")
                    if t == "binop":
                        op = expr.get("op")
                        left = eval_bound(expr.get("left"))
                        right = eval_bound(expr.get("right"))
                        if op == "+":
                            return left + right
                        if op == "-":
                            return left - right
                        if op == "*":
                            return left * right
                        if op == "/":
                            return int(left / right)
                raise SemanticError(f"Unsupported bound expr: {expr}")

            # NEW: resolve a named range from declarations or data
            def resolve_named_range(rng_name: str):
                # Try model inline range
                rng_decl = next(
                    (
                        d
                        for d in (model_ast.get("declarations") or [])
                        if d.get("type") == "range_declaration_inline" and d.get("name") == rng_name
                    ),
                    None,
                )
                if rng_decl:
                    s = eval_bound(rng_decl["start"])
                    e = eval_bound(rng_decl["end"])
                    return s, e
                # Try .dat provided range
                data_rng = working_data.get(rng_name)
                if isinstance(data_rng, dict) and data_rng.get("type") == "range_data":
                    return int(data_rng["start"]), int(data_rng["end"])
                raise SemanticError(f"Named range '{rng_name}' not found for computed parameter.")

            # Build tuple metadata for field access in computed params
            tuple_fields_by_type: dict[str, list[str]] = {}
            set_tuple_type_by_name: dict[str, str] = {}
            for d in model_ast.get("declarations") or []:
                if not isinstance(d, dict):
                    continue
                if d.get("type") == "tuple_type":
                    tname_any = d.get("name")
                    if isinstance(tname_any, str):
                        raw_fields = d.get("fields") or []
                        fields: list[str] = []
                        for f in raw_fields:
                            if isinstance(f, dict):
                                fname = f.get("name")
                                if isinstance(fname, str):
                                    fields.append(fname)
                        tuple_fields_by_type[tname_any] = fields
                elif d.get("type") in ("set_of_tuples", "set_of_tuples_external"):
                    name_any = d.get("name")
                    tuple_type_any = d.get("tuple_type")
                    if isinstance(name_any, str) and isinstance(tuple_type_any, str):
                        set_tuple_type_by_name[name_any] = tuple_type_any

            # Evaluate general numeric/boolean/string expression for param RHS. Limited support: number, name,
            # indexed_name, binop, uminus, parenthesis, funcall(sqrt), minl/maxl, and NEW: sum/min_agg/max_agg & field_access.
            def eval_expr(expr, env, iter_meta=None) -> Any:
                t = expr.get("type") if isinstance(expr, dict) else None
                if t == "number":
                    return float(expr.get("value"))
                if t == "boolean_literal":
                    return 1.0 if expr.get("value") else 0.0
                if t == "string_literal":
                    return expr.get("value")
                if t == "name":
                    nm = expr.get("value")
                    if nm in env:
                        v = env[nm]
                        return float(v) if isinstance(v, (int, float)) else v
                    if nm in working_data:
                        return working_data[nm]
                    raise SemanticError(f"Unknown name '{nm}' in computed parameter expression.")
                if t == "conditional":
                    # Support (cond ? then : else) in computed parameter expressions
                    cval = eval_expr(expr.get("condition"), env, iter_meta)
                    ctruth = bool(cval)
                    branch = expr.get("then") if ctruth else expr.get("else")
                    return eval_expr(branch, env, iter_meta)
                if t == "field_access":
                    # Support s.demand where s is an iterator bound to a tuple from a set-of-tuples
                    base = expr.get("base")
                    field = expr.get("field")
                    # Evaluate base value
                    base_val = eval_expr(base, env, iter_meta)
                    # Determine tuple type: prefer base.sem_type, then iterator metadata
                    tuple_type = None
                    if isinstance(base, dict):
                        bst = base.get("sem_type")
                        if isinstance(bst, str) and bst in tuple_fields_by_type:
                            tuple_type = bst
                    if (
                        tuple_type is None
                        and isinstance(base, dict)
                        and base.get("type") == "name"
                        and isinstance(iter_meta, dict)
                    ):
                        itn = base.get("value")
                        meta = iter_meta.get(itn) if itn else None
                        if isinstance(meta, dict):
                            tuple_type = meta.get("tuple_type")
                    if tuple_type is None:
                        raise SemanticError("Cannot resolve tuple type for field access in computed parameter.")
                    fields = tuple_fields_by_type.get(tuple_type) or []
                    try:
                        idx = fields.index(field)
                    except ValueError as e:
                        raise SemanticError(f"Unknown field '{field}' for tuple type '{tuple_type}'.") from e
                    try:
                        val = base_val[idx]
                    except Exception as e:
                        raise SemanticError(f"Field access failed on base value: {e}") from e
                    return float(val) if isinstance(val, (int, float)) else val
                if t == "indexed_name":
                    base = expr.get("name")
                    dims = expr.get("dimensions", [])
                    # Support multi-dimensional indices (range or set based)
                    arr = working_data.get(base)
                    if arr is None:
                        raise SemanticError(f"Parameter '{base}' not found for indexed access.")
                    # Evaluate each index and progressively index into arr
                    cur = arr
                    for dim in dims:
                        idx_val = eval_index(dim, env)
                        # Normalize float-int to int
                        if isinstance(idx_val, float) and idx_val.is_integer():
                            idx_val = int(idx_val)
                        if isinstance(cur, list):
                            if not isinstance(idx_val, (int, float)):
                                raise SemanticError(
                                    f"List parameter '{base}' requires integer indices, got {type(idx_val).__name__}: {idx_val!r}"
                                )
                            pos = int(idx_val) - 1  # OPL is 1-based for range-indexed lists
                            try:
                                cur = cur[pos]
                            except Exception as e:
                                raise SemanticError(f"Index out of bounds for '{base}' at {idx_val}: {e}") from e
                        elif isinstance(cur, dict):
                            # dict can be keyed by ints/strings/tuples depending on declaration
                            try:
                                cur = cur[idx_val]
                            except Exception as e:
                                raise SemanticError(f"Key '{idx_val!r}' not found in parameter '{base}': {e}") from e
                        else:
                            raise SemanticError(f"Cannot index into value of type {type(cur).__name__} for '{base}'.")
                    # At the end, cur is the scalar or structured element
                    if isinstance(cur, (int, float)):
                        return float(cur)
                    return cur
                if t == "sum":
                    # Evaluate sum over iterators, respecting optional index_constraint
                    iters = expr.get("iterators", [])
                    idxc = expr.get("index_constraint")
                    body = expr.get("expression")
                    # Build iterator domains
                    domains = []
                    for it in iters:
                        rng = it["range"]
                        if rng["type"] == "range_specifier":
                            st = eval_bound(rng["start"])
                            en = eval_bound(rng["end"])
                            domains.append(list(range(st, en + 1)))
                        elif rng["type"] == "named_range":
                            st, en = resolve_named_range(rng["name"])
                            domains.append(list(range(st, en + 1)))
                        elif rng["type"] in ("named_set", "named_set_dimension"):
                            set_name = rng["name"]
                            set_obj = working_data.get(set_name, [])
                            if isinstance(set_obj, dict) and "elements" in set_obj:
                                elems = set_obj["elements"]
                            else:
                                elems = set_obj
                            domains.append(list(elems or []))
                        else:
                            raise SemanticError(f"Unsupported range in sum aggregate: {rng['type']}")

                    # Recursive nested loops
                    def rec_sum(depth, local_env):
                        if depth == len(iters):
                            # index constraint filter
                            if idxc is not None:
                                cond_val = eval_expr(idxc, local_env, it_meta)
                                if isinstance(cond_val, (int, float)):
                                    if not bool(cond_val):
                                        return 0.0
                                else:
                                    if not cond_val:
                                        return 0.0
                            v = eval_expr(body, local_env, it_meta)
                            return float(v)
                        it_name = iters[depth]["iterator"]
                        total = 0.0
                        for val in domains[depth]:
                            local_env[it_name] = val
                            total += rec_sum(depth + 1, local_env)
                        local_env.pop(it_name, None)
                        return total

                    # Build iterator metadata map for field access
                    it_meta: dict[str, dict] = {}
                    for it in iters:
                        nm = it["iterator"]
                        rng = it["range"]
                        meta = {}
                        if rng.get("type") in ("named_set", "named_set_dimension"):
                            sname = rng["name"]
                            meta["set"] = sname
                            meta["tuple_type"] = set_tuple_type_by_name.get(sname)
                        it_meta[nm] = meta
                    return rec_sum(0, dict(env))
                if t in ("max_agg", "min_agg"):
                    iters = expr.get("iterators", [])
                    idxc = expr.get("index_constraint")
                    body = expr.get("expression")

                    # Build domains
                    domains = []
                    for it in iters:
                        rng = it["range"]
                        if rng["type"] == "range_specifier":
                            st = eval_bound(rng["start"])
                            en = eval_bound(rng["end"])
                            domains.append(list(range(st, en + 1)))
                        elif rng["type"] == "named_range":
                            st, en = resolve_named_range(rng["name"])
                            domains.append(list(range(st, en + 1)))
                        elif rng["type"] in ("named_set", "named_set_dimension"):
                            set_name = rng["name"]
                            set_obj = working_data.get(set_name, [])
                            if isinstance(set_obj, dict) and "elements" in set_obj:
                                elems = set_obj["elements"]
                            else:
                                elems = set_obj
                            domains.append(list(elems or []))
                        else:
                            raise SemanticError(f"Unsupported range in aggregate: {rng['type']}")

                    # Iterator metadata for field access
                    it_meta: dict[str, dict] = {}
                    for it in iters:
                        nm = it["iterator"]
                        rng = it["range"]
                        meta = {}
                        if rng.get("type") in ("named_set", "named_set_dimension"):
                            sname = rng["name"]
                            meta["set"] = sname
                            meta["tuple_type"] = set_tuple_type_by_name.get(sname)
                        it_meta[nm] = meta

                    best = None

                    def rec_agg(depth, local_env):
                        nonlocal best
                        if depth == len(iters):
                            # filter
                            if idxc is not None:
                                cond_val = eval_expr(idxc, local_env, it_meta)
                                if isinstance(cond_val, (int, float)):
                                    if not bool(cond_val):
                                        return
                                else:
                                    if not cond_val:
                                        return
                            v = float(eval_expr(body, local_env, it_meta))
                            if best is None:
                                best = v
                            else:
                                if t == "max_agg":
                                    if v > best:
                                        best = v
                                else:
                                    if v < best:
                                        best = v
                            return
                        it_name = iters[depth]["iterator"]
                        for val in domains[depth]:
                            local_env[it_name] = val
                            rec_agg(depth + 1, local_env)
                        local_env.pop(it_name, None)

                    rec_agg(0, dict(env))
                    if best is None:
                        raise SemanticError("Aggregate domain is empty in computed parameter expression.")
                    return best
                if t == "and":
                    return bool(eval_expr(expr.get("left"), env)) and bool(eval_expr(expr.get("right"), env))
                if t == "or":
                    return bool(eval_expr(expr.get("left"), env)) or bool(eval_expr(expr.get("right"), env))
                if t == "not":
                    return not bool(eval_expr(expr.get("value"), env))
                if t == "binop":
                    op = expr.get("op")
                    lv = eval_expr(expr.get("left"), env)
                    rv = eval_expr(expr.get("right"), env)
                    # numeric arithmetic
                    if op == "+":
                        return float(lv) + float(rv)
                    if op == "-":
                        return float(lv) - float(rv)
                    if op == "*":
                        return float(lv) * float(rv)
                    if op == "/":
                        return float(lv) / float(rv)
                    if op == "%":
                        return float(lv) % float(rv)
                    # comparisons: support numeric and equality on general types
                    if op in ("<", "<=", ">", ">=", "==", "!="):
                        if op == "<":
                            return 1.0 if (float(lv) < float(rv)) else 0.0
                        if op == "<=":
                            return 1.0 if (float(lv) <= float(rv)) else 0.0
                        if op == ">":
                            return 1.0 if (float(lv) > float(rv)) else 0.0
                        if op == ">=":
                            return 1.0 if (float(lv) >= float(rv)) else 0.0
                        if op == "==":
                            return 1.0 if (lv == rv) else 0.0
                        if op == "!=":
                            return 1.0 if (lv != rv) else 0.0
                    raise SemanticError(f"Unsupported operator in computed parameter expression: {op}")
                if t == "uminus":
                    return -float(eval_expr(expr.get("value"), env))
                if t == "parenthesized_expression":
                    return eval_expr(expr.get("expression"), env)
                if t == "funcall":
                    fname = expr.get("name")
                    args = expr.get("args", [])
                    if fname == "sqrt" and len(args) == 1:
                        return math.sqrt(float(eval_expr(args[0], env)))
                    raise SemanticError(f"Unsupported function '{fname}' in computed parameter expression.")
                if t in ("maxl", "minl"):
                    vals = [eval_expr(e, env) for e in (expr.get("args") or [])]
                    try:
                        nums = [float(v) for v in vals]
                    except Exception:
                        raise SemanticError(f"{t} in parameter must be numeric and ground.")
                    if not nums:
                        raise SemanticError(f"{t} requires at least one argument.")
                    return max(nums) if t == "maxl" else min(nums)
                raise SemanticError(f"Unsupported node in computed parameter expression: {t}")

            def eval_index(idx_expr, env):
                t = idx_expr.get("type")
                if t == "number_literal_index":
                    return idx_expr.get("value")
                if t == "name_reference_index":
                    nm = idx_expr.get("name")
                    return env.get(nm, nm)
                if t == "name":
                    return env.get(idx_expr.get("value"), idx_expr.get("value"))
                if t == "field_access_index" or t == "field_access":
                    raise SemanticError("Field access in computed parameter indices not supported.")
                if t == "binop":
                    op = idx_expr.get("op")
                    left = eval_index(idx_expr.get("left"), env)
                    right = eval_index(idx_expr.get("right"), env)
                    if op == "+":
                        return int(left) + int(right)
                    if op == "-":
                        return int(left) - int(right)
                    if op == "*":
                        return int(left) * int(right)
                    raise SemanticError(f"Unsupported index binop: {op}")
                if t == "uminus":
                    return -int(eval_index(idx_expr.get("value"), env))
                if t == "parenthesized_expression":
                    return eval_index(idx_expr.get("expression"), env)
                if t == "string_literal":
                    return idx_expr.get("value")
                raise SemanticError(f"Unsupported index expr: {t}")

            new_decls = []
            for decl in model_ast["declarations"]:
                if decl.get("type") != "parameter_inline_indexed_expr" and decl.get("type") != "parameter_inline_expr":
                    new_decls.append(decl)
                    continue

                # Shared caster by declared var_type
                def cast_value(v, var_type):
                    if isinstance(var_type, str) and var_type.startswith("int"):
                        return int(round(float(v)))
                    if var_type == "boolean":
                        return bool(round(float(v)))
                    return float(v)

                # Handle scalar parameter from expression
                if decl.get("type") == "parameter_inline_expr":
                    name = decl["name"]
                    var_type = decl.get("var_type") or ""
                    value = cast_value(eval_expr(decl["expression"], {}), var_type)
                    working_data[name] = value
                    new_decls.append(
                        {
                            "type": "parameter_inline",
                            "var_type": var_type,
                            "name": name,
                            "value": value,
                        }
                    )
                    continue

                # Computed indexed param
                name = decl["name"]
                var_type = decl.get("var_type") or ""
                dimensions = decl.get("dimensions", [])
                iterators = decl.get("iterators", [])

                # Support N-dimensional computed parameters (build nested lists in iterator order)
                # Domains for each iterator (respecting ranges and named sets)
                def _domain_for_range(rng):
                    if rng["type"] == "range_specifier":
                        s = eval_bound(rng["start"])
                        e = eval_bound(rng["end"])
                        return list(range(s, e + 1))
                    if rng["type"] == "named_range":
                        s, e = resolve_named_range(rng["name"])
                        return list(range(s, e + 1))
                    if rng["type"] in ("named_set", "named_set_dimension"):
                        set_name = rng["name"]
                        set_obj = working_data.get(set_name, [])
                        if isinstance(set_obj, dict) and "elements" in set_obj:
                            elems = set_obj["elements"]
                        else:
                            elems = set_obj
                        return list(elems or [])
                    raise SemanticError(f"Unsupported iterator range type '{rng['type']}' for computed parameter '{name}'.")

                it_names = [it["iterator"] for it in iterators]
                domains = [_domain_for_range(it["range"]) for it in iterators]

                # Recursively build nested lists in row-major order following iterator sequence
                def build_nested(depth: int, env_map: dict) -> object:
                    if depth == len(iterators):
                        # Ground evaluation at the leaf
                        val = eval_expr(decl["expression"], env_map)
                        return cast_value(val, var_type)
                    acc = []
                    itn = it_names[depth]
                    for v in domains[depth]:
                        env_map[itn] = v
                        acc.append(build_nested(depth + 1, env_map))
                    # Clean up to avoid leaking iterator into sibling branches
                    env_map.pop(itn, None)
                    return acc

                computed_value = build_nested(0, {})

                # Store nested list in working_data and rewrite declaration to an inline indexed parameter
                working_data[name] = computed_value
                new_decls.append(
                    {
                        "type": "parameter_inline_indexed",
                        "var_type": var_type,
                        "name": name,
                        "dimensions": dimensions,
                        "value": computed_value,
                    }
                )
                continue
            # Replace declarations list
            model_ast["declarations"] = new_decls
            # Also update data_dict since generators may consult it directly
            data_dict = dict(working_data)

            # Normalize inline indexed parameter lists into dicts keyed by domain elements/tuples.
            # This makes generated code robust: instead of relying on list-index fallbacks,
            # generators can index parameters by labels (strings/ints/tuples) directly.
            for decl in model_ast.get("declarations") or []:
                if decl.get("type") != "parameter_inline_indexed":
                    continue
                name = decl.get("name")
                if not name or name not in working_data:
                    continue
                val = working_data.get(name)
                # Only normalize when data is a nested list/tuple (produced inline or by computed params)
                if not isinstance(val, (list, tuple)):
                    continue

                dims = decl.get("dimensions", []) or []

                # Build domains list for each declared dimension
                domains = []
                for d in dims:
                    dtyp = d.get("type")
                    if dtyp in ("named_set", "named_set_dimension"):
                        set_name = d.get("name")
                        set_obj = working_data.get(set_name, [])
                        if isinstance(set_obj, dict) and "elements" in set_obj:
                            domain_elems = list(set_obj["elements"])
                        else:
                            domain_elems = list(set_obj or [])
                        domains.append(domain_elems)
                    elif dtyp in ("named_range", "named_range_dimension"):
                        # Try to resolve named range bounds from model declarations or working_data
                        try:
                            s, e = resolve_named_range(d["name"])
                            domains.append(list(range(int(s), int(e) + 1)))
                        except Exception:
                            # Fallback: if a range declaration exists, evaluate its AST bounds
                            rng_decl = next(
                                (
                                    x
                                    for x in (model_ast.get("declarations") or [])
                                    if x.get("name") == d.get("name") and x.get("type") == "range_declaration_inline"
                                ),
                                None,
                            )
                            if rng_decl:
                                start_idx = eval_bound(rng_decl["start"])
                                end_idx = eval_bound(rng_decl["end"])
                                domains.append(list(range(int(start_idx), int(end_idx) + 1)))
                            else:
                                # As a last resort, map to 1..len(val) for this axis
                                domains.append(list(range(1, len(val) + 1)))
                    else:
                        # For anonymous/range_index dims, infer positions 1..N
                        if isinstance(val, (list, tuple)):
                            domains.append(list(range(1, len(val) + 1)))
                        else:
                            domains.append([])

                # Flatten nested list into a mapping keyed by domain elements or tuples
                mapping = {}

                def _rec_flat(depth: int, node, prefix: list):
                    if depth == len(domains):
                        # Build a hashable key for this entry. If the key element
                        # is a list (e.g., tuple elements parsed as lists),
                        # convert it to a tuple so it can be used as a dict key.
                        if len(prefix) == 1:
                            key = prefix[0]
                            if isinstance(key, list):
                                key = tuple(key)
                            mapping[key] = node
                        else:
                            # For multi-dim keys, ensure any inner lists are
                            # converted to tuples before forming the key tuple.
                            safe_prefix = tuple(tuple(p) if isinstance(p, list) else p for p in prefix)
                            mapping[safe_prefix] = node
                        return
                    if not isinstance(node, (list, tuple)):
                        raise SemanticError(
                            f"Parameter '{name}' expected nested list matching declared domains, got {type(node).__name__}"
                        )
                    dom = domains[depth]
                    for i, key in enumerate(dom):
                        if i >= len(node):
                            raise SemanticError(f"Parameter '{name}' data length shorter than domain at dimension {depth+1}")
                        _rec_flat(depth + 1, node[i], prefix + [key])

                try:
                    _rec_flat(0, val, [])
                except SemanticError:
                    # If normalization fails, leave the original list form and continue
                    continue

                # Do NOT overwrite the original AST declaration or the
                # canonical working_data entry (tests and SciPy expect
                # nested lists in decl['value'] and data). Instead, keep
                # the original list form in-place and expose a separate
                # mapping that generators may use if desired.
                # Store the mapping under a distinct key to avoid name
                # collisions with user data.
                working_data[f"{name}__map"] = mapping

        # --- Validate shape of multi-dimensional arrays (use merged working data) ---
        def validate_shape(param_data, dims, param_name, data_dict, dim=0):
            if not dims:
                return
            d = dims[0]
            # New: if this is a 1-D parameter indexed by a set/range and data is a dict,
            # ensure each value is a scalar (not a list/tuple/dict). This catches cases like:
            # transport_cost[Stores] declared, but data provides {"StoreA": [2.0], ...}.
            if (
                len(dims) == 1
                and isinstance(param_data, dict)
                and d.get("type") in ("named_set_dimension", "named_range_dimension")
            ):
                for k, v in param_data.items():
                    if isinstance(v, (list, tuple, dict)):
                        raise SemanticError(
                            f"Parameter '{param_name}' is 1-D over '{d.get('name', '')}' but data value for key {repr(k)} "
                            f"is an array; expected a scalar (e.g., 2.0). Remove extra brackets like [2.0] -> 2.0."
                        )
                return
            d = dims[0]
            expected_len = None
            if d.get("type") == "named_range":
                range_decl = next(
                    (
                        x
                        for x in model_ast["declarations"]
                        if x.get("name") == d["name"] and x.get("type") == "range_declaration_inline"
                    ),
                    None,
                )
                if range_decl:

                    def eval_expr(expr):
                        if expr["type"] == "number":
                            return int(expr["value"])
                        elif expr["type"] == "name":
                            # resolve name from merged working data (may be inline scalar)
                            if expr["value"] not in data_dict:
                                raise SemanticError(f"Range bound refers to unknown name '{expr['value']}'")
                            return int(data_dict[expr["value"]])
                        elif expr["type"] == "binop":
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

                    start = eval_expr(range_decl["start"])
                    end = eval_expr(range_decl["end"])
                    expected_len = end - start + 1
            elif d.get("type") == "named_set_dimension":
                set_obj = data_dict.get(d["name"])
                if set_obj is not None:
                    if isinstance(set_obj, dict) and "elements" in set_obj:
                        expected_len = len(set_obj["elements"])
                    else:
                        expected_len = len(set_obj)
            if expected_len is not None:
                if not isinstance(param_data, (list, tuple)):
                    raise SemanticError(
                        f"Parameter '{param_name}' expected a {len(dims)}D array, got scalar at dimension {dim+1}."
                    )
                if len(param_data) != expected_len:
                    raise SemanticError(
                        f"Parameter '{param_name}' data length {len(param_data)} does not match declared dimension '{d.get('name')}' of length {expected_len} at dimension {dim+1}."
                    )
                if len(dims) > 1:
                    for i, sub in enumerate(param_data):
                        validate_shape(sub, dims[1:], param_name, data_dict, dim + 1)

        # Apply validation for all indexed parameters using merged data_dict
        if model_ast and "declarations" in model_ast:
            for decl in model_ast["declarations"]:
                if decl.get("type") in (
                    "parameter_external",
                    "parameter_external_indexed",
                    "parameter_external_explicit",
                    "parameter_external_explicit_indexed",
                    "parameter_inline",
                    "parameter_inline_indexed",
                ) and decl.get("dimensions"):
                    param_data = data_dict.get(decl["name"])
                    if param_data is not None and isinstance(param_data, (list, tuple)):
                        validate_shape(param_data, decl["dimensions"], decl["name"], data_dict)

        # --- Generate code for the model ---

        # --- Validate types of typed scalar sets ---
        def _is_int(x):
            return isinstance(x, int) and not isinstance(x, bool)

        def _is_num(x):
            return isinstance(x, (int, float)) and not isinstance(x, bool)

        def _is_bool(x):
            return isinstance(x, bool)

        def _is_str(x):
            return isinstance(x, str)

        def validate_typed_sets(model_ast, data_dict):
            if not model_ast or "declarations" not in model_ast:
                return
            for decl in model_ast["declarations"]:
                if decl.get("type") not in ("typed_set", "typed_set_external"):
                    continue
                base = decl.get("base_type")
                name = decl.get("name")

                # Determine source of values: inline value takes precedence if present
                values = decl.get("value")
                if values is None:
                    values = data_dict.get(name)

                if values is None:
                    continue  # Uninitialized/external with no data yet

                # Normalize sets-of-tuples (should not pass here) or dict wrappers
                if isinstance(values, dict) and "elements" in values:
                    values = values["elements"]

                if not isinstance(values, list):
                    raise SemanticError(f"Set '{name}' must be assigned a list of values, got {type(values).__name__}.")

                if base == "int":
                    if not all(_is_int(v) for v in values):
                        raise SemanticError(f"All elements of set '{name}' must be integers.")
                elif base == "float":
                    if not all(_is_num(v) for v in values):
                        raise SemanticError(f"All elements of set '{name}' must be numeric (int/float).")
                    # Coerce to float for consistency
                    coerced = [float(v) for v in values]
                    data_dict[name] = coerced
                elif base == "boolean":
                    if not all(_is_bool(v) for v in values):
                        raise SemanticError(f"All elements of set '{name}' must be booleans (true/false).")
                elif base == "string":
                    if not all(_is_str(v) for v in values):
                        raise SemanticError(f"All elements of set '{name}' must be strings.")

        validate_typed_sets(model_ast, data_dict)
        # --- End typed set validation ---

        def validate_named_ranges(ast: dict, data_dict: dict) -> None:
            """
            Ensure every named range used as an index or iterator is declared with explicit bounds in the model (.mod).
            If the same name is present only in the .dat (as a range assignment), raise a clear SemanticError.
            """
            # Inline range declarations present in the model
            declared_inline: set[str] = {
                n
                for n in (
                    d.get("name")
                    for d in (ast.get("declarations") or [])
                    if isinstance(d, dict) and d.get("type") == "range_declaration_inline"
                )
                if isinstance(n, str)
            }

            # Collect named ranges used as indices in declarations
            used: set[str] = set()
            for d in ast.get("declarations", []) or []:
                if not isinstance(d, dict):
                    continue
                dims = d.get("dimensions", []) or []
                for dim in dims:
                    if isinstance(dim, dict) and dim.get("type") == "named_range_dimension":
                        n = dim.get("name")
                        if isinstance(n, str):
                            used.add(n)

            # Collect named ranges used in forall/sum iterators
            def walk(node: object) -> None:
                if isinstance(node, dict):
                    t = node.get("type")
                    if t in ("forall_constraint", "sum"):
                        iters = node.get("iterators", []) or []
                        if isinstance(iters, list):
                            for it in iters:
                                if not isinstance(it, dict):
                                    continue
                                rng = it.get("range") or {}
                                if isinstance(rng, dict) and rng.get("type") == "named_range":
                                    n = rng.get("name")
                                    if isinstance(n, str):
                                        used.add(n)
                    for v in list(node.values()):
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)

            walk(ast.get("objective", {}))
            walk(ast.get("constraints", []))

            # For each used range, require inline declaration
            for name in sorted(used):
                if name in declared_inline:
                    continue
                # If present in data as range_data, point to the correct fix
                dv = data_dict.get(name)
                if isinstance(dv, dict) and dv.get("type") == "range_data":
                    raise SemanticError(
                        f"Range '{name}' was supplied in the data file, but ranges used for indexing must be declared "
                        f"with explicit bounds in the model file. Declare it in the model (e.g., 'range {name} = 1..N;') "
                        f"and remove it from the .dat."
                    )
                # Otherwise generic not-found
                raise SemanticError(f"Range '{name}' is used as an index but not declared in the model.")

        validate_named_ranges(model_ast, data_dict)

        # --- Generate code for the model ---

        ast = model_ast

        # First, simplify any ground boolean gating/conditions to avoid emitting constant boolean rows.
        try:
            self._simplify_ground_booleans(ast, working_data)
        except SemanticError as e:
            logger.error(f"Ground boolean simplification error: {e}")
            raise

        # After AST is built and data_dict merged (inline + .dat), rewrite conditional constraints:
        try:
            self._evaluate_and_splice_if_constraints(ast, data_dict)
            # Simplify again in case if-constraints introduced new ground booleans
            self._simplify_ground_booleans(ast, working_data)
            self._lower_minmax_aggregates(ast)
            self._lower_maxmin_convex(ast)
            self._split_boolean_and_constraints(ast)  # NEW: split conjunctions
        except SemanticError as e:
            logger.error(f"Conditional constraint error: {e}")
            # Surface the error similarly to other semantic errors
            raise

        if solver == "gurobi":
            code = GurobiCodeGenerator(ast, data_dict).generate_code()
        elif solver == "scipy":
            code = cast(SciPyCodeGeneratorBase, SciPyCodeGenerator(ast, data_dict)).generate_code()
        else:
            raise ValueError(f"Unsupported solver: {solver}")

        return ast, code, data_dict

    def _simplify_ground_booleans(self, ast: dict, env: dict) -> None:
        """
        Constant-fold ground boolean expressions (no decision vars, no iterators) in constraints and forall
        index constraints. This eliminates patterns like (RunPricing != 1) || (lhs <= rhs) == true by:
          - folding ground comparisons, and, or, not
          - dropping tautologies
          - reducing False || X to X, True || X to True
          - reducing True && X to X, False && X to False
          - simplifying (bool_expr) == true/false
        """
        if not isinstance(ast, dict):
            return

        dvars = self._collect_dvar_names(ast.get("declarations", []))

        def is_ground_bool(node: dict) -> bool:
            # ground if it contains no dvar and all names are in env (no iterators here)
            if self._expr_contains_dvar(node, dvars):
                return False
            try:
                # if it evaluates without error, treat as ground
                _ = self._eval_ground_expr(node, env)
                return True
            except Exception:
                return False

        def as_bool_lit(v: bool) -> dict:
            return {"type": "boolean_literal", "value": bool(v), "sem_type": "boolean"}

        def simplify_bool(node: Any) -> Any:
            if not isinstance(node, dict):
                return node
            t = node.get("type")

            # Recurse first
            if t in ("and", "or"):
                left = simplify_bool(node.get("left"))
                right = simplify_bool(node.get("right"))
                node = {"type": t, "left": left, "right": right, "sem_type": "boolean"}
            elif t == "not":
                val = simplify_bool(node.get("value"))
                node = {"type": "not", "value": val, "sem_type": "boolean"}
            elif t == "parenthesized_expression":
                inner = simplify_bool(node.get("expression"))
                node = {"type": "parenthesized_expression", "expression": inner, "sem_type": inner.get("sem_type", None)}
            elif t == "binop" and node.get("sem_type") == "boolean" and node.get("op") in ("<", "<=", ">", ">=", "==", "!="):
                left = simplify_bool(node.get("left"))
                right = simplify_bool(node.get("right"))
                node = {"type": "binop", "op": node.get("op"), "left": left, "right": right, "sem_type": "boolean"}
            else:
                pass

            # Constant fold ONLY boolean nodes (avoid folding numeric names like L -> true)
            is_bool_node = isinstance(node, dict) and (
                node.get("sem_type") == "boolean"
                or node.get("type") in ("and", "or", "not")
                or (node.get("type") == "binop" and node.get("sem_type") == "boolean")
            )
            if is_bool_node and is_ground_bool(node):
                try:
                    val = self._eval_ground_condition(node, env)
                    return as_bool_lit(val)
                except Exception:
                    pass

            # Apply boolean algebra with literals
            if isinstance(node, dict) and node.get("type") in ("and", "or"):
                L = node.get("left")
                R = node.get("right")
                if isinstance(L, dict) and L.get("type") == "boolean_literal":
                    if node["type"] == "or":
                        return as_bool_lit(True) if L.get("value") else R
                    else:
                        return R if L.get("value") else as_bool_lit(False)
                if isinstance(R, dict) and R.get("type") == "boolean_literal":
                    if node["type"] == "or":
                        return as_bool_lit(True) if R.get("value") else L
                    else:
                        return L if R.get("value") else as_bool_lit(False)
            if isinstance(node, dict) and node.get("type") == "not":
                V = node.get("value")
                if isinstance(V, dict) and V.get("type") == "boolean_literal":
                    return as_bool_lit(not V.get("value"))
            return node

        def simplify_constraint(c: dict) -> list[dict]:
            # Returns zero or more constraints (drop tautologies)
            if c.get("type") == "constraint":
                op = c.get("op")
                L = c.get("left")
                R = c.get("right")

                # Only simplify boolean-equality constraints: (bool_expr) == true/false
                if op == "==":
                    if isinstance(R, dict) and R.get("type") == "boolean_literal":
                        Ls = simplify_bool(L)
                        # NEW: unwrap parens so we see the inner comparison
                        while isinstance(Ls, dict) and Ls.get("type") == "parenthesized_expression":
                            Ls = Ls.get("expression")

                        # If L fully reduces to boolean literal, decide
                        if isinstance(Ls, dict) and Ls.get("type") == "boolean_literal":
                            if Ls.get("value") == R.get("value"):
                                return []  # tautology
                            else:
                                return [
                                    {
                                        "type": "constraint",
                                        "op": "==",
                                        "left": {"type": "number", "value": 0, "sem_type": "int"},
                                        "right": {"type": "number", "value": 1, "sem_type": "int"},
                                    }
                                ]

                        # If Ls is a comparison (binop boolean), keep or negate it properly
                        if isinstance(Ls, dict) and Ls.get("type") == "binop" and Ls.get("sem_type") == "boolean":
                            op_any = Ls.get("op")
                            if not isinstance(op_any, str):
                                # Can't safely negate/emit a comparison op; leave as (Ls == true/false)
                                return [{"type": "constraint", "op": "==", "left": Ls, "right": R}]

                            if R.get("value") is True:
                                # (cmp) == true -> cmp
                                return [
                                    {
                                        "type": "constraint",
                                        "op": op_any,
                                        "left": Ls.get("left"),
                                        "right": Ls.get("right"),
                                    }
                                ]
                            else:
                                # (cmp) == false -> negate cmp
                                neg: dict[str, str] = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
                                neg_op = neg.get(op_any)
                                if neg_op is None:
                                    return [{"type": "constraint", "op": "==", "left": Ls, "right": R}]
                                return [
                                    {
                                        "type": "constraint",
                                        "op": neg_op,
                                        "left": Ls.get("left"),
                                        "right": Ls.get("right"),
                                    }
                                ]

                        # Non-ground boolean tree: leave for codegen to linearize
                        return [{"type": "constraint", "op": "==", "left": Ls, "right": R}]

                # Other numeric constraints unchanged
                return [c]

            if c.get("type") == "forall_constraint":
                # Simplify index_constraint first
                new_ic = c.get("index_constraint")
                if isinstance(new_ic, dict):
                    new_ic = simplify_bool(new_ic)
                    # Drop forall if guard is false
                    if isinstance(new_ic, dict) and new_ic.get("type") == "boolean_literal" and new_ic.get("value") is False:
                        return []  # no constraints generated
                    # Remove guard if true
                    if isinstance(new_ic, dict) and new_ic.get("type") == "boolean_literal" and new_ic.get("value") is True:
                        new_ic = None

                out = []
                if "constraint" in c and isinstance(c["constraint"], dict):
                    inner = simplify_constraint(c["constraint"])
                    if inner:
                        if len(inner) == 1:
                            out.append(
                                {
                                    "type": "forall_constraint",
                                    "iterators": c.get("iterators", []),
                                    "index_constraint": new_ic,
                                    "constraint": inner[0],
                                }
                            )
                        else:
                            out.append(
                                {
                                    "type": "forall_constraint",
                                    "iterators": c.get("iterators", []),
                                    "index_constraint": new_ic,
                                    "constraints": inner,
                                }
                            )
                    return out
                if "constraints" in c and isinstance(c["constraints"], list):
                    inner_all = []
                    for cc in c["constraints"]:
                        inner_all.extend(simplify_constraint(cc))
                    if inner_all:
                        if len(inner_all) == 1:
                            out.append(
                                {
                                    "type": "forall_constraint",
                                    "iterators": c.get("iterators", []),
                                    "index_constraint": new_ic,
                                    "constraint": inner_all[0],
                                }
                            )
                        else:
                            out.append(
                                {
                                    "type": "forall_constraint",
                                    "iterators": c.get("iterators", []),
                                    "index_constraint": new_ic,
                                    "constraints": inner_all,
                                }
                            )
                    return out
                return [c]

            # Pass-through for other nodes
            return [c]

        if "constraints" in ast and isinstance(ast["constraints"], list):
            new_list: list[dict] = []
            for c in ast["constraints"]:
                if isinstance(c, dict):
                    new_list.extend(simplify_constraint(c))
                else:
                    new_list.append(c)
            ast["constraints"] = new_list

    # NEW: split (A && B && ...) == true into multiple constraints A==true; B==true; ...
    def _split_boolean_and_constraints(self, ast: dict) -> None:
        def is_true(node: Any) -> bool:
            return isinstance(node, dict) and node.get("type") == "boolean_literal" and node.get("value") is True

        def flatten_and(node: Any) -> list[dict]:
            if isinstance(node, dict) and node.get("type") == "and":
                return flatten_and(node.get("left")) + flatten_and(node.get("right"))
            return [node]

        if not isinstance(ast, dict) or "constraints" not in ast:
            return

        new_cons: list[dict] = []
        for c in ast["constraints"]:
            if (
                isinstance(c, dict)
                and c.get("type") == "constraint"
                and c.get("op") == "=="
                and is_true(c.get("right"))
                and isinstance(c.get("left"), dict)
                and c["left"].get("type") == "and"
            ):
                for part in flatten_and(c["left"]):
                    new_cons.append(
                        {
                            "type": "constraint",
                            "op": "==",
                            "left": part,
                            "right": {"type": "boolean_literal", "value": True, "sem_type": "boolean"},
                            "label": c.get("label"),
                        }
                    )
            else:
                new_cons.append(c)
        ast["constraints"] = new_cons

    # ----------------- NEW: Conditional-constraint compile-time rewrite -----------------

    def _evaluate_and_splice_if_constraints(self, ast: dict, env: dict) -> None:
        """
        Validate groundness of all if-constraint conditions, evaluate them using env,
        and splice only the selected branch into ast['constraints'].

        Extended: if an if-constraint appears inside a forall, rewrite it into two
        forall nodes with augmented index constraints (cond) and (!cond). Conditions
        inside forall must not reference decision variables but may reference
        iterators and parameters.
        """
        if not isinstance(ast, dict) or "constraints" not in ast:
            return

        dvar_names = self._collect_dvar_names(ast.get("declarations", []))

        def contains_dvar(expr: Any) -> bool:
            return self._expr_contains_dvar(expr, dvar_names)

        def is_ground(expr: Any) -> bool:
            # Ground = contains no decision variables and no free iterators.
            return not contains_dvar(expr)

        def and_expr(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
            if a is None:
                return b
            if b is None:
                return a
            return {"type": "and", "left": a, "right": b, "sem_type": "boolean"}

        def not_expr(e: dict) -> dict:
            return {"type": "not", "value": e, "sem_type": "boolean"}

        def normalize_forall_body(fc: dict) -> list[dict]:
            if "constraints" in fc and isinstance(fc["constraints"], list):
                # Filter to dict items only to satisfy typing and avoid None
                return [c for c in fc["constraints"] if isinstance(c, dict)]
            if "constraint" in fc and isinstance(fc["constraint"], dict):
                return [fc["constraint"]]
            return []

        def make_forall(iterators, index_constraint: Optional[dict], body_constraints: list[dict]) -> dict:
            node: dict[str, Any] = {
                "type": "forall_constraint",
                "iterators": iterators,
                "index_constraint": index_constraint,
            }
            if len(body_constraints) == 1:
                node["constraint"] = body_constraints[0]
            else:
                node["constraints"] = body_constraints
            return node

        # Helper: normalized boolean literal
        def _bool_lit(v: bool) -> dict:
            return {"type": "boolean_literal", "value": bool(v), "sem_type": "boolean"}

        # Convert any constraint/boolean-like node to a boolean expression dict.
        # Guarantees a dict is returned.
        def to_bool_expr(node_any: Any) -> dict:
            n = node_any
            # unwrap parentheses
            while isinstance(n, dict) and n.get("type") == "parenthesized_expression":
                n = n.get("expression")
            # Already a boolean expression node
            if isinstance(n, dict):
                t = n.get("type")
                if t in ("and", "or", "not", "boolean_literal"):
                    return cast(dict, n)
                if t == "binop":
                    # binop is used both for arithmetic and comparisons; allow as boolean when used in conditions
                    return cast(dict, n)
                if t == "constraint":
                    op = n.get("op")
                    L = n.get("left")
                    R = n.get("right")
                    # constraint of form (expr == true/false)
                    if op == "==" and isinstance(R, dict) and R.get("type") == "boolean_literal":
                        return (
                            cast(dict, L)
                            if R.get("value") is True
                            else {"type": "not", "value": cast(dict, L), "sem_type": "boolean"}
                        )
                    # General comparison -> boolean expression
                    if op in ("<", "<=", ">", ">=", "==", "!="):
                        return {"type": "binop", "op": op, "left": L, "right": R, "sem_type": "boolean"}
                    # Fallback: equate to true
                    return {"type": "binop", "op": "==", "left": n, "right": _bool_lit(True), "sem_type": "boolean"}
                if t in ("name", "indexed_name", "funcall"):
                    # Treat bare boolean-valued symbol/expression as == true
                    return {"type": "binop", "op": "==", "left": n, "right": _bool_lit(True), "sem_type": "boolean"}
                # Last resort: wrap unknown dict node as == true
                return {"type": "binop", "op": "==", "left": n, "right": _bool_lit(True), "sem_type": "boolean"}
            # Python literal fallback
            if isinstance(n, bool):
                return _bool_lit(n)
            # Numbers/strings -> treat nonzero/nonempty as boolean at eval time; still force a node
            return {
                "type": "binop",
                "op": "==",
                "left": {"type": "number", "value": n, "sem_type": "int" if isinstance(n, int) else "float"},
                "right": _bool_lit(True),
                "sem_type": "boolean",
            }

        # Rewrite a single forall node: split inner if-constraints and ground-antecedent implications
        def rewrite_forall_node(fc: dict) -> list[dict]:
            iterators = fc.get("iterators", [])
            base_ic: Optional[dict] = cast(Optional[dict], fc.get("index_constraint"))
            body = normalize_forall_body(fc)

            regular_constraints: list[dict] = []
            new_foralls: list[dict] = []

            for c in body:
                # Handle if-constraints
                if isinstance(c, dict) and c.get("type") == "if_constraint":
                    cond_any = c.get("condition")
                    if not isinstance(cond_any, dict):
                        raise SemanticError("Malformed if-constraint: missing condition.")
                    cond: dict = cond_any
                    if contains_dvar(cond):
                        raise SemanticError("Condition of if-constraint inside forall must not reference decision variables.")
                    then_list = c.get("then_constraints") or []
                    else_list = c.get("else_constraints") or []
                    if then_list:
                        then_fc = make_forall(
                            iterators, and_expr(base_ic, cond), [cc for cc in then_list if isinstance(cc, dict)]
                        )
                        new_foralls.extend(rewrite_forall_node(then_fc))
                    if else_list:
                        else_fc = make_forall(
                            iterators, and_expr(base_ic, not_expr(cond)), [cc for cc in else_list if isinstance(cc, dict)]
                        )
                        new_foralls.extend(rewrite_forall_node(else_fc))
                    continue

                # Implication with ground antecedent inside forall: push antecedent into index condition
                if isinstance(c, dict) and c.get("type") == "implication_constraint":
                    ant = c.get("antecedent")
                    cons = c.get("consequent")
                    ant_bool: dict = to_bool_expr(ant)
                    if contains_dvar(ant_bool):
                        regular_constraints.append(c)
                    else:
                        guarded_ic: Optional[dict] = and_expr(base_ic, ant_bool)
                        if isinstance(cons, dict):
                            new_foralls.append(make_forall(iterators, guarded_ic, [cons]))
                    continue

                # Keep others
                if isinstance(c, dict):
                    regular_constraints.append(c)

            if regular_constraints:
                new_foralls.append(make_forall(iterators, base_ic, regular_constraints))

            return new_foralls

        # Top-level pass
        out_top: list[dict] = []

        for c in ast.get("constraints", []):
            # Top-level if-constraint
            if isinstance(c, dict) and c.get("type") == "if_constraint":
                cond_any = c.get("condition")
                if not isinstance(cond_any, dict):
                    raise SemanticError("Malformed if-constraint: missing condition.")
                cond: dict = cond_any
                if not is_ground(cond):
                    if contains_dvar(cond):
                        raise SemanticError(
                            "Condition of if-constraint must be ground (must not reference decision variables)."
                        )
                    raise SemanticError("Condition of if-constraint at top level cannot reference iterators.")
                val = self._eval_ground_condition(cond, env)
                chosen_list = (c.get("then_constraints") or []) if val else (c.get("else_constraints") or [])
                for cc in chosen_list:
                    if isinstance(cc, dict) and cc.get("type") == "forall_constraint":
                        out_top.extend(rewrite_forall_node(cc))
                    elif isinstance(cc, dict):
                        out_top.append(cc)
                continue

            # Top-level forall
            if isinstance(c, dict) and c.get("type") == "forall_constraint":
                out_top.extend(rewrite_forall_node(c))
                continue

            # Top-level implication with ground antecedent
            if isinstance(c, dict) and c.get("type") == "implication_constraint":
                ant = c.get("antecedent")
                cons = c.get("consequent")
                ant_bool: dict = to_bool_expr(ant)
                if contains_dvar(ant_bool):
                    out_top.append(c)
                else:
                    if self._eval_ground_condition(ant_bool, env) and isinstance(cons, dict):
                        out_top.append(cons)
                continue

            if isinstance(c, dict):
                out_top.append(c)

        ast["constraints"] = out_top

    def _lower_minmax_aggregates(self, ast: dict) -> None:
        if not isinstance(ast, dict):
            return

        def make_forall(iterators, idxc, cons):
            node = {"type": "forall_constraint", "iterators": iterators, "index_constraint": idxc}
            if isinstance(cons, list):
                node["constraints"] = cons
            else:
                node["constraint"] = cons
            return node

        def agg_to_forall(agg, op_side: str, other):
            # op_side: 'left' means agg on LHS, else RHS
            iters = agg.get("iterators", [])
            idxc = agg.get("index_constraint")
            e = agg.get("expression")

            def wrap(c):
                return make_forall(iters, idxc, c)

            # Builds a list of rewritten constraints per rule
            return wrap if op_side == "wrap" else (iters, idxc, e)

        # Objective rewrite
        obj = ast.get("objective")
        if isinstance(obj, dict):
            expr = obj.get("expression")
            if isinstance(expr, dict) and expr.get("type") in ("max_agg", "min_agg"):
                t = expr["type"]
                if t == "max_agg" and obj.get("type") == "minimize":
                    z = self._gensym("__maxagg_obj")
                    ast["declarations"].append({"type": "dvar", "var_type": "float", "name": z})
                    # forall(i): e(i) <= z
                    iters = expr["iterators"]
                    idxc = expr.get("index_constraint")
                    e = expr["expression"]
                    ast["constraints"].append(
                        make_forall(
                            iters,
                            idxc,
                            {
                                "type": "constraint",
                                "op": "<=",
                                "left": e,
                                "right": {"type": "name", "value": z, "sem_type": "float"},
                            },
                        )
                    )
                    ast["objective"]["expression"] = {"type": "name", "value": z, "sem_type": "float"}
                elif t == "min_agg" and obj.get("type") == "maximize":
                    z = self._gensym("__minagg_obj")
                    ast["declarations"].append({"type": "dvar", "var_type": "float", "name": z})
                    # forall(i): e(i) >= z
                    iters = expr["iterators"]
                    idxc = expr.get("index_constraint")
                    e = expr["expression"]
                    ast["constraints"].append(
                        make_forall(
                            iters,
                            idxc,
                            {
                                "type": "constraint",
                                "op": ">=",
                                "left": e,
                                "right": {"type": "name", "value": z, "sem_type": "float"},
                            },
                        )
                    )
                    ast["objective"]["expression"] = {"type": "name", "value": z, "sem_type": "float"}
                else:
                    raise SemanticError("Non-convex objective: supported only minimize max(...) or maximize min(...).")

        # Constraint rewrite (walk all constraints)
        def rewrite_constraint(c):
            if not isinstance(c, dict):
                return [c]
            if c.get("type") == "constraint":
                L, R, op = c.get("left"), c.get("right"), c.get("op")

                # Helpers to build per-iterator constraints
                def forall_from(agg_side, other_side, opLR):
                    iters = agg_side["iterators"]
                    idxc = agg_side.get("index_constraint")
                    e = agg_side["expression"]
                    cons = {
                        "type": "constraint",
                        "op": opLR,
                        "left": e if opLR in ("<=", ">=") and agg_side is L else other_side,
                        "right": other_side if opLR in ("<=", ">=") and agg_side is L else e,
                    }
                    return [make_forall(iters, idxc, cons)]

                # max-agg convex forms
                if isinstance(L, dict) and L.get("type") == "max_agg" and op == "<=":
                    return forall_from(L, R, "<=")
                if isinstance(R, dict) and R.get("type") == "max_agg" and op == ">=":
                    return forall_from(R, L, ">=")
                # min-agg convex forms
                if isinstance(L, dict) and L.get("type") == "min_agg" and op == ">=":
                    return forall_from(L, R, ">=")
                if isinstance(R, dict) and R.get("type") == "min_agg" and op == "<=":
                    return forall_from(R, L, "<=")

                # Disallow other placements
                if (isinstance(L, dict) and L.get("type") in ("min_agg", "max_agg")) or (
                    isinstance(R, dict) and R.get("type") in ("min_agg", "max_agg")
                ):
                    raise SemanticError("Unsupported non-convex aggregate placement (==, >, <, or reversed forms).")
                return [c]

            if c.get("type") == "forall_constraint":
                inner = []
                if "constraint" in c:
                    for cc in rewrite_constraint(c["constraint"]):
                        inner.append(cc)
                    return (
                        [dict(c, **({"constraints": inner, "constraint": None}))]
                        if len(inner) > 1
                        else [dict(c, **({"constraint": inner[0]}))]
                    )
                elif "constraints" in c:
                    for cc in c["constraints"]:
                        inner.extend(rewrite_constraint(cc))
                    return [dict(c, **({"constraints": inner}))]
                return [c]
            # Pass through others
            return [c]

        if "constraints" in ast:
            newC = []
            for c in ast["constraints"]:
                newC.extend(rewrite_constraint(c))
            ast["constraints"] = newC

    def _collect_dvar_names(self, declarations: list) -> set:
        names = set()
        for d in declarations or []:
            if not isinstance(d, dict):
                continue
            t = d.get("type")
            if t in ("dvar", "dvar_indexed"):
                n = d.get("name")
                if isinstance(n, str):
                    names.add(n)
        return names

    def _expr_contains_dvar(self, node: Any, dvar_names: set) -> bool:
        """
        Returns True if node refers to any decision variable name.
        """
        if isinstance(node, dict):
            t = node.get("type")
            if t == "name":
                v = node.get("value")
                return isinstance(v, str) and v in dvar_names
            if t == "indexed_name":
                n = node.get("name")
                return isinstance(n, str) and n in dvar_names
            # Recurse over children
            for v in node.values():
                if self._expr_contains_dvar(v, dvar_names):
                    return True
            return False
        if isinstance(node, list):
            return any(self._expr_contains_dvar(x, dvar_names) for x in node)
        return False

    def _eval_ground_condition(self, expr: Any, env: dict) -> bool:
        """
        Evaluate a ground boolean expression using provided env (merged inline/.dat data).
        Supports: number, boolean_literal, string_literal, name, indexed_name,
                    binop (arith and comparisons), and/or/not, parenthesized_expression, conditional.
        """
        val = self._eval_ground_expr(expr, env)
        if isinstance(val, (int, float)):
            # nonzero treated as True
            return bool(val)
        if isinstance(val, bool):
            return val
        raise SemanticError(f"Condition does not evaluate to boolean: {expr}")

    def _eval_ground_expr(self, expr: Any, env: dict):
        if not isinstance(expr, dict):
            return expr
        t = expr.get("type")
        if t == "number":
            return expr.get("value")
        if t == "boolean_literal":
            return bool(expr.get("value"))
        if t == "string_literal":
            return expr.get("value")
        if t == "name":
            name = expr.get("value")
            if name in env:
                return env[name]
            # If it's a known scalar set/range name etc., leave as-is or raise
            raise SemanticError(f"Unknown symbol in ground expression: {name}")
        if t == "indexed_name":
            base = expr.get("name")
            if base not in env:
                raise SemanticError(f"Unknown symbol in ground expression: {base}")
            target = env[base]
            dims = expr.get("dimensions", [])
            # Evaluate each index dimension
            for d in dims:
                idx = self._eval_ground_expr(d, env)
                # Coerce booleans to int if needed
                if isinstance(idx, bool):
                    idx = int(idx)
                try:
                    target = target[idx]
                except Exception as e:
                    raise SemanticError(f"Index error in ground expression {base}[...]: {e}") from e
            return target
        if t == "parenthesized_expression":
            return self._eval_ground_expr(expr.get("expression"), env)
        if t == "not":
            return not self._eval_ground_condition(expr.get("value"), env)
        if t == "and":
            return bool(
                self._eval_ground_condition(expr.get("left"), env) and self._eval_ground_condition(expr.get("right"), env)
            )
        if t == "or":
            return bool(
                self._eval_ground_condition(expr.get("left"), env) or self._eval_ground_condition(expr.get("right"), env)
            )
        if t == "conditional":
            cond = self._eval_ground_condition(expr.get("condition"), env)
            return self._eval_ground_expr(expr.get("then") if cond else expr.get("else"), env)
        if t == "binop":
            op = expr.get("op")
            left = self._eval_ground_expr(expr.get("left"), env)
            right = self._eval_ground_expr(expr.get("right"), env)
            try:
                if op == "+":
                    return left + right
                if op == "-":
                    return left - right
                if op == "*":
                    return left * right
                if op == "/":
                    return left / right
                if op == "%":
                    return left % right
                if op == "<":
                    return left < right
                if op == "<=":
                    return left <= right
                if op == ">":
                    return left > right
                if op == ">=":
                    return left >= right
                if op == "==":
                    return left == right
                if op == "!=":
                    return left != right
            except Exception as e:
                raise SemanticError(f"Error evaluating ground binop '{op}': {e}") from e
            raise SemanticError(f"Unsupported operator in ground expression: {op}")
        # Unsupported in conditions
        raise SemanticError(f"Unsupported expression in ground condition: {t}")

    def _lower_maxmin_convex(self, ast: dict) -> None:
        """
        Convex lowering for maxl/minl:
          - Objective: minimize maxl(...) or maximize minl(...): add aux z and epigraph/hypograph.
          - Constraints: expand four convex forms into per-argument linear constraints.
          - Otherwise: raise SemanticError.
        """
        if not isinstance(ast, dict):
            return

        def is_max(n):
            return isinstance(n, dict) and n.get("type") == "maxl"

        def is_min(n):
            return isinstance(n, dict) and n.get("type") == "minl"

        def args_or_err(node):
            args = node.get("args") or []
            if len(args) == 0:
                raise SemanticError("maxl/minl require at least one argument.")
            if len(args) == 1:
                return [args[0]], True
            return args, False

        # Objective
        if "objective" in ast and isinstance(ast["objective"], dict):
            obj = ast["objective"]
            expr = obj.get("expression")
            # unwrap parentheses
            if isinstance(expr, dict) and expr.get("type") == "parenthesized_expression":
                expr = expr.get("expression")

            if obj.get("type") == "minimize" and is_max(expr):
                args, single = args_or_err(expr)
                if single:
                    ast["objective"]["expression"] = args[0]
                else:
                    zname = self._gensym("__maxl_obj")
                    # declare aux continuous variable
                    (ast.get("declarations") or []).append({"type": "dvar", "var_type": "float", "name": zname})
                    # replace objective expression with aux
                    ast["objective"]["expression"] = {"type": "name", "value": zname, "sem_type": "float"}
                    # add epigraph constraints: z >= ei
                    for ei in args:
                        ast["constraints"].append(
                            {
                                "type": "constraint",
                                "op": ">=",
                                "left": {"type": "name", "value": zname, "sem_type": "float"},
                                "right": ei,
                            }
                        )
            elif obj.get("type") == "maximize" and is_min(expr):
                args, single = args_or_err(expr)
                if single:
                    ast["objective"]["expression"] = args[0]
                else:
                    zname = self._gensym("__minl_obj")
                    (ast.get("declarations") or []).append({"type": "dvar", "var_type": "float", "name": zname})
                    ast["objective"]["expression"] = {"type": "name", "value": zname, "sem_type": "float"}
                    # hypograph: z <= ei
                    for ei in args:
                        ast["constraints"].append(
                            {
                                "type": "constraint",
                                "op": "<=",
                                "left": {"type": "name", "value": zname, "sem_type": "float"},
                                "right": ei,
                            }
                        )
            else:
                # If maxl/minl appears anywhere in objective, reject (non-convex usage)
                if self._contains_maxmin(obj.get("expression")):
                    raise SemanticError(
                        "Non-convex objective: maxl/minl allowed only as minimize maxl(...) or maximize minl(...)."
                    )

        # Constraints
        def expand_constraint(cnode: dict) -> list[dict]:
            # Returns a list of linear constraints replacing cnode, or raises on non-convex use.
            if not isinstance(cnode, dict):
                return [cnode]
            t = cnode.get("type")
            if t == "constraint":
                op = cnode.get("op")
                L = cnode.get("left")
                R = cnode.get("right")
                label = cnode.get("label")

                # Helper to attach label if present
                def with_label(cons: dict) -> dict:
                    if label:
                        cons = dict(cons)
                        cons["label"] = label
                    return cons

                # Allowed convex patterns (including reversed sides)
                if op == "<=" and is_max(L):
                    args, single = args_or_err(L)
                    if single:
                        return [with_label({"type": "constraint", "op": "<=", "left": args[0], "right": R})]
                    return [with_label({"type": "constraint", "op": "<=", "left": ei, "right": R}) for ei in args]
                if op == ">=" and is_max(R):
                    args, single = args_or_err(R)
                    if single:
                        return [with_label({"type": "constraint", "op": ">=", "left": L, "right": args[0]})]
                    return [with_label({"type": "constraint", "op": ">=", "left": L, "right": ei}) for ei in args]
                if op == ">=" and is_min(L):
                    args, single = args_or_err(L)
                    if single:
                        return [with_label({"type": "constraint", "op": ">=", "left": args[0], "right": R})]
                    return [with_label({"type": "constraint", "op": ">=", "left": ei, "right": R}) for ei in args]
                if op == "<=" and is_min(R):
                    args, single = args_or_err(R)
                    if single:
                        return [with_label({"type": "constraint", "op": "<=", "left": L, "right": args[0]})]
                    return [with_label({"type": "constraint", "op": "<=", "left": L, "right": ei}) for ei in args]

                # If equality involves maxl/minl, reject
                if op == "==" and (self._contains_maxmin(L) or self._contains_maxmin(R)):
                    raise SemanticError("Non-convex: equality with maxl/minl is not supported.")
                # If maxl/minl appear elsewhere (inside arithmetic), reject
                if self._contains_maxmin(L) or self._contains_maxmin(R):
                    raise SemanticError(
                        "Non-convex or unsupported placement of maxl/minl in constraint. Allowed only in: maxl(...) <= rhs, lhs >= maxl(...), minl(...) >= rhs, lhs <= minl(...)."
                    )
                return [cnode]

            if t == "implication_constraint":
                # Do not allow maxl/minl under implication for now (non-convex in general)
                if self._contains_maxmin(cnode.get("antecedent")) or self._contains_maxmin(cnode.get("consequent")):
                    raise SemanticError("Non-convex: maxl/minl not supported inside implication constraints.")
                return [cnode]

            if t == "forall_constraint":
                # Rewrite children and keep structure
                iterators = cnode.get("iterators", [])
                ic = cnode.get("index_constraint")
                if "constraint" in cnode:
                    expanded = expand_constraint(cnode["constraint"])
                    if len(expanded) == 1:
                        return [dict(cnode, **{"constraint": expanded[0]})]
                    else:
                        node = dict(cnode)
                        node.pop("constraint", None)
                        node["constraints"] = expanded
                        return [node]
                elif "constraints" in cnode and isinstance(cnode["constraints"], list):
                    new_children: list[dict] = []
                    for child in cnode["constraints"]:
                        new_children.extend(expand_constraint(child))
                    return [dict(cnode, **{"iterators": iterators, "index_constraint": ic, "constraints": new_children})]
                return [cnode]

            # Other nodes unchanged
            return [cnode]

        if "constraints" in ast and isinstance(ast["constraints"], list):
            new_cons: list[dict] = []
            for c in ast["constraints"]:
                if isinstance(c, dict):
                    new_cons.extend(expand_constraint(c))
                else:
                    new_cons.append(c)
            ast["constraints"] = new_cons

    # Helper: unique symbol names
    _mm_counter: int = 0

    def _gensym(self, prefix: str) -> str:
        self._mm_counter = getattr(self, "_mm_counter", 0) + 1
        return f"{prefix}_{self._mm_counter}"

    def _contains_maxmin(self, node: Any) -> bool:
        if isinstance(node, dict):
            t = node.get("type")
            if t in ("maxl", "minl"):
                return True
            return any(self._contains_maxmin(v) for v in node.values())
        if isinstance(node, list):
            return any(self._contains_maxmin(x) for x in node)
        return False


# Convenience helper for tests and simple parsing without code generation
def parse_model(model_code: str):
    """Parse a model string and return its AST (no code generation)."""
    compiler = OPLCompiler()
    ast, _code, _data = compiler.compile_model(model_code, data_code=None, solver="gurobi")
    return ast


# --- Utility function to load OPL model from disk ---
def load_opl_model(model_file_name, data_file_name=None, solver="gurobi"):
    """
    Loads an OPL model from a file and optionally a data file,
    then parses it and generates solver-specific code.

    Args:
        model_file_name (str): Path to the .mod or .opl model file.
        data_file_name (str, optional): Path to the .dat data file.
        solver (str, optional): The solver to use ('gurobi' or 'scipy'). Defaults to 'gurobi'.

    Returns:
        tuple: (ast, code_str, data_dict) if successful, (None, None, None) otherwise.
    """
    opl_model_code = ""
    opl_data_code = None
    data_dict = {}

    try:
        with open(model_file_name, "r") as f:
            opl_model_code = f.read()

        if data_file_name:
            if os.path.exists(data_file_name):
                with open(data_file_name, "r") as f:
                    opl_data_code = f.read()
                logger.info(f"Note: Data file '{data_file_name}' loaded.")
            else:
                logger.warning(f"Warning: Data file '{data_file_name}' not found. Proceeding without it.")

        compiler = OPLCompiler()
        ast, code, data_dict = compiler.compile_model(opl_model_code, opl_data_code, solver=solver)

        return ast, code, data_dict

    except FileNotFoundError as e:
        logger.error(f"Error: File not found - {e.filename}")
        return None, None, None
    except SemanticError as e:
        logger.error(f"Error parsing OPL model or data: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading/parsing the model: {e}")
        traceback.print_exc()
        return None, None, None


# --- Function to solve an OPL model ---
def solve(model_file: str, data_file: Optional[str] = None, solver: str = "gurobi") -> dict[str, Any]:
    """
    Solves an OPL model using the specified solver.

    Args:
        model_file (str): Path to the .mod or .opl model file.
        data_file (str, optional): Path to the .dat data file.
        solver (str): The solver to use ('gurobi' or 'scipy').

    Returns:
        dict: A dictionary containing the optimization results if successful,
              or status/error information otherwise.
    """
    if solver == "gurobi":
        return solve_with_gurobi(model_file, data_file)
    elif solver == "scipy":
        return solve_with_scipy(model_file, data_file)
    else:
        raise ValueError(f"Unsupported solver: {solver}")


def solve_with_gurobi(model_file, data_file=None):
    """
    Loads an OPL model and optional data from disk,
    generates GurobiPy code, and executes it to solve the model.
    Prints the GurobiPy model output.

    Returns:
        dict: A dictionary containing the optimization results if successful,
              or status/error information otherwise.
    """
    results = {
        "status": "FAILED",
        "message": "An unexpected error occurred during compilation or execution.",
        "solution": {},
        "objective_value": None,
        "stats": {},
    }

    if not os.path.exists(model_file):
        results["message"] = f"Error: Model file '{model_file}' does not exist."
        logger.error(results["message"])
        return results
    if data_file is not None and not os.path.exists(data_file):
        results["message"] = f"Error: Data file '{data_file}' does not exist."
        logger.error(results["message"])
        return results

    logger.info(f"\n--- Solving OPL Model with Gurobi: {model_file} ---")
    if data_file:
        logger.info(f"--- Using Data File: {data_file} ---")

    loaded_ast, loaded_gurobi_code, loaded_data_dict = load_opl_model(model_file, data_file)

    if loaded_ast and loaded_gurobi_code:
        logger.info("\n--- Loaded AST from file ---")
        logger.info(json.dumps(loaded_ast, indent=2))
        if loaded_data_dict:
            logger.info("\n--- Loaded Data Dictionary from file ---")
            logger.info(json.dumps(_json_safe(loaded_data_dict), indent=2))
        logger.info("\n--- Generated GurobiPy Code ---")
        logger.info(loaded_gurobi_code)

        logger.info("\n--- GurobiPy Model Output ---")
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        exec_globals = {
            "gp": gp,
            "GRB": GRB,
            "results_container": {},  # This will hold the results from the executed code
        }

        try:
            exec(loaded_gurobi_code, exec_globals)
            # Retrieve results from the exec_globals after execution
            if "gurobi_output" in exec_globals["results_container"]:
                results = exec_globals["results_container"]["gurobi_output"]
                # Do not override status to COMPLETED; keep solver's status
            else:
                results["status"] = "EXECUTION_NO_OUTPUT"
                results["message"] = "GurobiPy code executed, but no results captured."
                logger.warning(results["message"])

        except gp.GurobiError as e:
            results["status"] = "GUROBI_ERROR"
            results["message"] = f"Gurobi Error: {e.message}"
            logger.error(results["message"])
        except Exception as e:
            results["status"] = "EXECUTION_ERROR"
            results["message"] = _execution_error_with_hint(e, "GurobiPy")
            logger.error(results["message"])
            traceback.print_exc(file=sys.stdout)  # Print traceback to captured stdout
        finally:
            sys.stdout = old_stdout  # Restore stdout
            logger.info(redirected_output.getvalue())  # Print captured output to original stdout
    else:
        results["message"] = _load_failure_message()
        logger.error(results["message"])

    logger.info("\n" + "=" * 50 + "\n")
    return results


def solve_with_scipy(model_file, data_file=None):
    """
    Loads an OPL model and optional data from disk,
    generates SciPy linprog code, and executes it to solve the model.
    Prints the SciPy linprog model output.

    Returns:
        dict: A dictionary containing the optimization results if successful,
              or status/error information otherwise.
    """
    results = {
        "status": "FAILED",
        "message": "An unexpected error occurred during compilation or execution.",
        "solution": {},
        "objective_value": None,
        "stats": {},
    }

    if not os.path.exists(model_file):
        results["message"] = f"Error: Model file '{model_file}' does not exist."
        logger.error(results["message"])
        return results
    if data_file is not None and not os.path.exists(data_file):
        results["message"] = f"Error: Data file '{data_file}' does not exist."
        logger.error(results["message"])
        return results

    logger.info(f"\n--- Solving OPL Model with SciPy: {model_file} ---")
    if data_file:
        logger.info(f"--- Using Data File: {data_file} ---")

    loaded_ast, loaded_scipy_code, loaded_data_dict = load_opl_model(model_file, data_file, solver="scipy")

    if loaded_ast and loaded_scipy_code:
        logger.info("\n--- Loaded AST from file ---")
        logger.info(json.dumps(loaded_ast, indent=2))
        if loaded_data_dict:
            logger.info("\n--- Loaded Data Dictionary from file ---")
            logger.info(json.dumps(_json_safe(loaded_data_dict), indent=2))
        logger.info("\n--- Generated SciPy linprog Code ---")
        logger.info(loaded_scipy_code)

        logger.info("\n--- SciPy linprog Model Output ---")
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            exec_globals = {
                "json": json,
                "np": __import__("numpy"),
                "linprog": __import__("scipy.optimize", fromlist=["linprog"]).linprog,
                "results_container": {},
            }
            exec(loaded_scipy_code, exec_globals)
            if "scipy_output" in exec_globals["results_container"]:
                results = exec_globals["results_container"]["scipy_output"]
                # Do not override status to COMPLETED; keep solver's status
            else:
                results["status"] = "EXECUTION_NO_OUTPUT"
                results["message"] = "SciPy code executed, but no results captured."
                logger.warning(results["message"])
        except Exception as e:
            results["status"] = "EXECUTION_ERROR"
            results["message"] = _execution_error_with_hint(e, "SciPy")
            logger.error(results["message"])
            traceback.print_exc(file=sys.stdout)
        finally:
            sys.stdout = old_stdout
            logger.info(redirected_output.getvalue())
    else:
        results["message"] = _load_failure_message()
        logger.error(results["message"])

    logger.info("\n" + "=" * 50 + "\n")
    return results


# --- Helper: make dicts with tuple keys JSON-serializable ---
def _json_safe(obj):
    """
    Recursively convert dicts with tuple keys to lists of [key, value] pairs (with keys as lists),
    so they can be safely serialized with json.dumps.
    """
    if isinstance(obj, dict):
        if any(isinstance(k, tuple) for k in obj.keys()):
            return [[list(k) if isinstance(k, tuple) else k, _json_safe(v)] for k, v in obj.items()]
        else:
            return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    else:
        return obj

# Import the single, canonical SemanticError class
from .semantic_error import SemanticError


class SciPyCodeGeneratorBase:
    """
    Abstract base class for SciPy code generators.
    Subclasses must implement the required methods.
    """

    def __init__(self, ast, data_dict=None):
        self.ast = ast
        self.data_dict = data_dict or {}
        self.scipy_code_lines = []
        self.indent_level = 0
        self.var_names = []
        self.var_indices = {}
        self.bounds = []
        self.c = []
        self.A_eq = []
        self.b_eq = []
        self.A_ub = []
        self.b_ub = []
        self.results_varname = "results"
        self.integrality = []
        self.tuple_types = {}

    def _indent(self):
        return "    " * self.indent_level

    def _add_code_line(self, line):
        self.scipy_code_lines.append(self._indent() + line)

    def generate_code(self):
        raise NotImplementedError("Subclasses must implement generate_code()")

    def _build_variables(self):
        raise NotImplementedError("Subclasses must implement _build_variables()")

    def _build_objective(self):
        raise NotImplementedError("Subclasses must implement _build_objective()")

    def _build_constraints(self):
        raise NotImplementedError("Subclasses must implement _build_constraints()")

    def find_declaration(self, name, decl_type=None):
        for d in self.ast.get("declarations", []):
            if d.get("name") == name and (decl_type is None or d.get("type") == decl_type):
                return d
        return None

    def find_declarations(self, name, decl_type=None):
        return [
            d
            for d in self.ast.get("declarations", [])
            if d.get("name") == name and (decl_type is None or d.get("type") == decl_type)
        ]

    @staticmethod
    def normalize_index(idx):
        if isinstance(idx, (list, tuple)):
            return tuple(SciPyCodeGeneratorBase.normalize_index(e) for e in idx)
        return idx

    def _normalize_index(self, idx):
        return self.normalize_index(idx)

    def _tuple_key(self, elements, env=None):
        if env is None:
            env = {}
        key = []
        for e in elements:
            if isinstance(e, dict) and e.get("type") == "name" and e["name"] in env:
                key.append(env[e["name"]])
            elif isinstance(e, dict) and e.get("type") == "name":
                key.append(e["name"])
            elif isinstance(e, dict) and e.get("type") == "number":
                key.append(e["value"])
            elif isinstance(e, dict) and e.get("type") == "string":
                key.append(e["value"])
            elif isinstance(e, dict) and e.get("type") == "tuple_literal":
                key.append(self._tuple_key(e["elements"], env))
            else:
                key.append(e)
        return tuple(key)

    def _find_decl(self, name, decl_type=None):
        return self.find_declaration(name, decl_type)

    def _find_decls(self, name, decl_type=None):
        return self.find_declarations(name, decl_type)

    @staticmethod
    def _unsupported_type_error(context, value):
        return SemanticError(f"Unsupported {context} type: {value}")

    @staticmethod
    def _unsupported_operator_error(context, op):
        return SemanticError(f"Unsupported {context} operator: '{op}'")

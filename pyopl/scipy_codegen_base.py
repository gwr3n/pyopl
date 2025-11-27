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

    def generate_code(self):
        raise NotImplementedError("Subclasses must implement generate_code()")
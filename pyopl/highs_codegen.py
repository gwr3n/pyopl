# Use module-level logger, no handler/formatter setup here
import logging

from .highs_codegen_base import HighsCodeGeneratorBase
from .highs_codegen_csc import HighsCSCCodeGenerator, LinearProblem
from .linear_problem_highs import build_highs_model, export_linear_problem
from .semantic_error import SemanticError

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class HighsCodeGenerator:
    def __new__(cls, ast, data_dict=None, mode="csc"):
        if mode == "csc":
            return HighsCSCCodeGenerator(ast, data_dict)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# For type checks and compatibility
__all__ = [
    "HighsCodeGenerator",
    "HighsCSCCodeGenerator",
    "LinearProblem",
    "build_highs_model",
    "export_linear_problem",
    "HighsCodeGeneratorBase",
    "SemanticError",
]

# Use module-level logger, no handler/formatter setup here
import logging

from .scipy_codegen_base import SciPyCodeGeneratorBase
from .scipy_codegen_csc import SciPyCSCCodeGenerator
from .semantic_error import SemanticError

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class SciPyCodeGenerator:
    def __new__(cls, ast, data_dict=None, mode="csc"):
        if mode == "csc":
            return SciPyCSCCodeGenerator(ast, data_dict)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# For type checks and compatibility
__all__ = [
    "SciPyCodeGenerator",
    "SciPyCSCCodeGenerator",
    "SciPyCodeGeneratorBase",
    "SemanticError",
]

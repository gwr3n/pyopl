"""PyOPL package initialization."""

# Ensure 'icon' is a package so importlib.resources works
import os
import sys

from .genai.pyopl_generative import generative_feedback, generative_solve
from .model_equivalence import compare_models
from .pyopl_core import solve

__version__ = "2.0.0"
__year__ = "2026"
__author__ = "Roberto Rossi"
__email__ = "robros@gmail.com"
__description__ = "A Python library for parsing and solving OPL-like mathematical programming models using multiple solvers."
__license__ = "MIT"
__url__ = "https://github.com/gwr3n/pyopl"

icon_dir = os.path.join(os.path.dirname(__file__), "icon")
if os.path.isdir(icon_dir) and icon_dir not in sys.path:
    sys.path.append(icon_dir)
__all__ = ["solve", "compare_models", "generative_solve", "generative_feedback"]

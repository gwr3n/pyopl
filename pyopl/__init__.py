"""PyOPL package initialization."""

# Ensure 'icon' is a package so importlib.resources works
import os
import sys

from .genai.pyopl_generative import generative_feedback, generative_solve
from .pyopl_core import solve

__version__ = "1.9.0"
__year__ = "2026"
__author__ = "Roberto Rossi"
__email__ = "robros@gmail.com"
__description__ = "A Python library for parsing and solving OPL-like mathematical programming models using multiple solvers."
__license__ = "MIT"
__url__ = "https://github.com/gwr3n/pyopl"

icon_dir = os.path.join(os.path.dirname(__file__), "icon")
if os.path.isdir(icon_dir) and icon_dir not in sys.path:
    sys.path.append(icon_dir)
__all__ = ["solve", "generative_solve", "generative_feedback"]

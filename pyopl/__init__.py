"""PyOPL package initialization."""

# Ensure 'icon' is a package so importlib.resources works
import os
import sys

from .pyopl_core import solve

__version__ = "0.6.0"
__author__ = "Roberto Rossi"
__email__ = "robros@gmail.com"
__description__ = "A Python interface for OPL models with support for multiple solvers."
__license__ = "MIT"
__url__ = "https://github.com/gwr3n/pyopl"

icon_dir = os.path.join(os.path.dirname(__file__), "icon")
if os.path.isdir(icon_dir) and icon_dir not in sys.path:
    sys.path.append(icon_dir)
__all__ = ["solve"]

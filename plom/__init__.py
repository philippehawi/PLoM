# plom/__init__.py

from .core import PLoM
from .preprocessing import Scaler, PCA
from .manifold import DiffusionMaps
from .sampling import ItoSampler

# Optional: Expose plotting functions if you want them at the top level
# from .plotting import plot_2d_scatter, plot_3d_scatter, plot_eigenvalues

# Define the package version (match pyproject.toml)
__version__ = "2.0.0"

# Define what happens with 'from plom import *'
__all__ = [
    "PLoM",
    "Scaler",
    "PCA",
    "DiffusionMaps",
    "ItoSampler"
]
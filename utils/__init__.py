"""
Star pipeline processing modules for filamentous structure analysis.
@Builab 2025
"""

# Core computational functions
from .fit import fit_curves

# I/O utilities
from .io import (
    validate_dataframe,
    load_coordinates,
    read_star,
    write_star,
    combine_star_files
)

from .clean import (
    clean_tubes,
    filter_short_tubes,
    filter_by_direction
)

from .connect import connect_tubes

from .predict import predict_angles

from .sort import group_and_sort

from .scoring import calculate_tube_scores, print_tube_scores

__all__ = [
    # Fit functions
    'fit_curves',
    # Clean functions
    'clean_tubes',
    'filter_short_tubes',
    'filter_by_direction',
    #Connect functions
    'connect_tubes',
    # Predict functions
    'predict_angles',
    # I/O functions,
    'validate_dataframe',
    'load_coordinates',
    'read_star',
    'write_star',
    # Sort function
    'group_and_sort',
    # Scoring
    'calculate_tube_scores',
    'print_tube_scores'
]

__version__ = '1.0.0'

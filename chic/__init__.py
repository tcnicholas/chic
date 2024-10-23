"""
chic: A Python library for coarse-graining hybrid and inorganic frameworks.

This library provides tools for working with complex structures, allowing for
efficient analysis and manipulation of hybrid and inorganic framework materials.

Version modifications:
---------------------
"24.05.24" "0.1.15", "External bonds needs to account for many PBCs."
"24.04.24" "0.1.11", "Correct image type in cut-off neighbour list."
"24.04.24" "0.1.10", "Added pairwise radial cut-offs with rmin and rmax options 
    for neighbour searching."
"23.04.24" "0.1.9", "Added radial cut-off for neighbour searching."
"""


from .structure import Structure
from .net import Net


__version__ = "0.1.17"
__author__ = "Thomas C. Nicholas"
__email__ = "tcnicholas@me.com"
__url__ = "https://github.com/tcnicholas/chic"


__all__ = ['Structure', 'Net']

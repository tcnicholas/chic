"""
27.07.23
@tcnicholas
Functions for tidying up structures.
"""


from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element


def unit_occupancy(struct: Structure):
    """
    Set all atom occupancies to 1.

    :param struct: structure to modify.
    """
    for i,a in enumerate(struct):
        d = a.species.as_dict()
        e = list(d.keys())[0]
        if d[e] != 1:
            struct.replace(i, Element(e))


def no_deuterium(struct: Structure):
    """
    Replace deuterium in structure with hydrogen.

    :param struct: structure to modify.
    """
    ds = [i for i,a in enumerate(struct) if a.specie.symbol == "D"]
    for d in ds:
        struct.replace(d, Element("H"))
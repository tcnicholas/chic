"""
27.07.23
@tcnicholas
Functions for sorting sites.
"""


from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element


class SiteType(Enum):
    A = "a"
    B = "b"


@dataclass
class SortingMethod:
    name: str
    function: Callable[[Structure], List[List[str]]]


InCHI =  {
    "non_metals" : {"H", "He", "B", "C",  "N",  "O",  "F", "Ne",
                   "Si",  "P",  "S", "Cl", "Ar", "Ge", "As", "Se", 
                   "Br", "Kr", "Te",  "I", "Xe", "At", "Rn"},
    "non_metal_exceptions" : {"B", "Si","P"}
}


def sort_sites(structure: Structure, method: str) -> List[List[str]]:

    assert isinstance(structure, Structure), "structure must be a pymatgen.core.Structure instance"
    assert isinstance(method, str), "method must be a string"
    
    methods: Dict[str, SortingMethod] = {
        "all_atoms": SortingMethod("all_atoms", all_atoms),
        "mof": SortingMethod("mof", mof)
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available methods are: {methods.keys()}")
    return methods[method].function(structure)


def all_atoms(structure: Structure) -> List[List[str]]:
    return [[e.symbol] for e in structure.composition.elements]


def mof(structure: Structure) -> List[List[str]]:
    elements = structure.composition.elements
    atom_site = {SiteType.A: [], SiteType.B: []}
    for e in elements:
        site_type = get_site_type(structure, e)
        if site_type is not None:
            atom_site[site_type].append(e.symbol)
        else:
            print(f"Warning: {e.symbol} not assigned to a site type")
    return list(atom_site.values())


def get_site_type(structure: Structure, e: Element) -> Optional[SiteType]:
    if e.is_transition_metal or e.symbol in InCHI["non_metal_exceptions"] or \
       ((e.is_alkali or e.is_alkaline) and e.symbol in {"Li", "Be"}):
        return SiteType.A
    if e.symbol == 'O' and is_ice(structure):
        return SiteType.A
    if e.symbol in InCHI["non_metals"]:
        return SiteType.B
    return None


def is_ice(structure: Structure) -> bool:
    return structure.composition.reduced_formula == "H2O"


def create_site_type_lookup(site_types: List[List[str]]) -> dict:
    lookup = {}
    for i, sublist in enumerate(site_types):
        for item in sublist:
            lookup[item] = i
    return lookup
"""
12.06.21
@tcnicholas
A module for removing disorder from structures.
"""

from pymatgen.core.periodic_table import Element

import numpy as np


def poreOxygen(structure):
    """
    Find oxygen atoms that only bonded to other oxygens and deletes them from
    the Pymatgen Structure object. Often the unresolved electron density in the
    pores of frameworks is modelled by artificial oxygen atoms.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    Returns
    -------
    Modified Structure object.
    """

    # Get indices of oxygen atoms.
    oi = [i for i,a in enumerate(structure) if a.specie.symbol=="O"]

    # Get distance between all oxygen atoms.
    fc = structure.frac_coords[oi]
    d = structure.lattice.get_all_distances(fc,fc)
    np.fill_diagonal(d,100)

    # Identify short O-O distances.

    return structure


def pairwiseElement(structure, element="H", cutoff=(0.0,0.8)):
    """
    Finds where two atoms of the same atomic species lie within a cut-off radius
    (0--0.7 Å by default), deletes them, and places the same atomic species at
    the average position.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    element: str
        Chemical symbol for element to search for occupational disorder.

    cutoff: float or tuple
        Set the cutoff radius. If float given, the range is assumed to start
        from zero. Otherwise, searches between the two values in the tuple.
    """

    # Convert cutoff variable to tuple for lower and upper bounds if required.
    try:
        _ = iter(cutoff)
    except TypeError:
        cutoff = (0,cutoff)
    
    # Get indices of element.
    ei = [i for i,a in enumerate(structure) if a.specie.symbol==element]

    # Get distance between atoms.
    fc = structure.frac_coords[ei]
    d = structure.lattice.get_all_distances(fc,fc)
    np.fill_diagonal(d,100)

    # Identify violations. Loop over all atoms, and find the closest atom.
    ix = []
    for i,a in enumerate(d):

        # Get indices from distance matrix, determine indices from.
        if cutoff[0] < np.amin(a) < cutoff[1]:
            ix.append(tuple(sorted((ei[i],ei[np.argmin(a)]))))

    # Identify unique pairs.
    ix = set(ix)

    # Iterate through violations, find average position, and delete original
    # atoms.
    nPos = []
    for v in ix:
        
        # Get Cartesian postions and average position.
        c1, c2 = structure.cart_coords[list(v)]
        nPos.append((c1 + c2) / 2)
    
    # Remove old atoms and add new atoms.
    structure.remove_sites(np.array(list(ix)).flatten())
    for c in nPos:
        structure.append(   
                            Element(element),
                            structure.lattice.get_fractional_coords(c)
                        )

    # Print report to terminal.
    print(f"* disorder * Averaged {len(ix)} pairs of {element} atoms.")
    
    return structure


def loneElement(structure, element="O", cutoff=2):
    """
    Finds atoms with no nearest neighbours within a cut-off radius and deletes
    them.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    element: str
        Chemical symbol for element to search for occupational disorder.

    cutoff: float
        Radius to search around a given atom and if no other atoms are found
        then the atom will be removed.
    """

    # Get indices of element.
    ei = [i for i,a in enumerate(structure) if a.specie.symbol==element]

    # Get distance from atoms of type 'element' to all other atoms in structure.
    fc = structure.frac_coords[ei]
    d = structure.lattice.get_all_distances(
                                                structure.frac_coords,
                                                structure.frac_coords
                                            )
    # Ignore self-distances.
    np.fill_diagonal(d, 1000)
    d = d[ei]

    # Iterate through atoms and determine whether there are any atoms in the
    # cutoff sphere.
    rmv = []
    for i,a in enumerate(d):

        if 0 == a[a < cutoff].shape[0]:
            rmv.append(ei[i])

    # Removing atoms by index.
    structure.remove_sites(rmv)

    # Print report to terminal.
    print(f"* disorder * Deleted {len(rmv)} {element} atoms.")

    return structure
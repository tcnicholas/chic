"""
04.03.22
@tcnicholas
Input structure from LAMMPS-data file.

This module was written with the coarse-graining of hypothetical ZIFs in mind.
It is has not been tested for other systems. Careful use is advised!
"""

from .bu import buildingUnit
from ase.io import read

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor


def extract_topology(dataFile):
    """
    Parse topology data from LAMMPS data file.
    """
    with open(dataFile, "r") as f:
        lines = f.readlines()
    headers = ["Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"]
    curr_header = None
    header = []
    data = {}
    while lines:
        l = lines.pop(0)
        if curr_header is not None: l = l.strip()
        if not l: continue
        if l.strip() in headers:
            curr_header = l
            if l != "Atoms\n":
                data[l] = []
        if curr_header is None:
            header.append(l)
        elif curr_header is not None and curr_header != "Atoms\n" and l not in headers:
            data[curr_header].append(l)
    return header, data
    

def get_bonds(dataFile):
    """
    Parse LAMMPS data file and extract bond list.
    """
    # get headers and data.
    h, d = extract_topology(dataFile)

    # get bonds.
    bonds = d["Bonds\n"]
    bonds.pop(0)

    # format in NumPy array of [id, type, atom1_id, atom2_id]
    return np.array([x.split() for x in bonds], dtype=np.int64)


def read_lammps(filePath):
    """
    Read LAMMPS data file and create ASE atoms and extract bond information.
    """

    # if sort by id, the property arrays are not automatically re-orderd.
    aseAtoms = read(filePath, format="lammps-data", sort_by_id=False)
    structure = AseAtomsAdaptor.get_structure(aseAtoms)
    return structure, aseAtoms, {}


def get_units_from_LAMMPS(filePath):
    """
    Coarse-graining is a lot easier if you already have the molecular unit
    information (e.g. the molIDs from a LAMMPS data file).
    """
    
    structure, aseAtoms, _ = read_lammps(filePath)
    bonds = get_bonds(filePath)

    assert "mol-id" in aseAtoms.arrays.keys(), "MolIDs not found in file!"

    # extract per-atom properties from the arrays.
    cell = structure.lattice
    aIDs = aseAtoms.arrays["id"]
    molIDs = np.unique(aseAtoms.arrays["mol-id"])
    symbols = np.array(aseAtoms.get_chemical_symbols())

    # create dictionary for mapping aIDs to aIXs.
    aID2aIX = {id_:ix_ for ix_, id_ in enumerate(aIDs)}
    frac_coords = structure.frac_coords

    numA = 1
    numB = 1
    units = {}

    for mol in molIDs:

        # get the atom indices of this unit.
        a_ix = np.where(aseAtoms.arrays["mol-id"]==mol)[0]

        # get the atom IDs associated with this molecule (because the atoms are
        # not sorted by atom ID).
        a_id = aIDs[a_ix]

        # get atomic symbols in this unit.
        sym = symbols[a_ix]

        # get bonds involved with atoms in this unit.
        bs = np.array([b for b in bonds if len(set(b[2:]) & set(a_id))])

        # sort into internal (intra-unit) and external (inter-unit) bonds.
        intraBonds = [b for b in bs if np.all([x in a_id for x in b[2:]])]
        interBonds = [b for b in bs if not np.all([x in a_id for x in b[2:]])]

        # find consistent periodic images of all atoms in the building unit
        # based on minimimum image convention (for which these structures were
        # designed to work with).
        used_bonds = [list(x) for x in intraBonds]
        atoms = {}
        if a_id.shape[0] > 1:
            
            b1 = used_bonds.pop()
            a1 = aID2aIX[b1[2]]
            a2 = aID2aIX[b1[3]]
            _, im = cell.get_distance_and_image(frac_coords[a1],frac_coords[a2])

            atoms[a1] = np.array([0,0,0])
            atoms[a2] = im

            while used_bonds:

                for b in used_bonds:

                    # get the number of atoms in the bond that have been found.
                    num = np.sum([1 for x in list(b[2:]) if aID2aIX[x] in atoms.keys()])
                    if num==1:

                        # find which atom in bond it is, and which it is not.
                        if aID2aIX[b[2]] in atoms.keys():
                            a1 = aID2aIX[b[2]]
                            a2 = aID2aIX[b[3]]
                        else:
                            a1 = aID2aIX[b[3]]
                            a2 = aID2aIX[b[2]]

                        # get the appropriate periodic image of the unrecorded
                        # atom.
                        _, im = cell.get_distance_and_image(frac_coords[a1]+atoms[a1],frac_coords[a2])
                        atoms[a2] = im
                        used_bonds.remove(b)
                    
                    # remove the bond if both atoms are already accounted for.
                    elif num==2:
                        used_bonds.remove(b)
        else:
            atoms[a_ix[0]] = np.array([0,0,0])
        
        # find consistent perioidic images of all external atoms bound to this
        # building unit.
        connectivity = []
        for b in interBonds:
            
            if aID2aIX[b[2]] in atoms.keys():
                a1 = aID2aIX[b[2]]
                a2 = aID2aIX[b[3]]
            else:
                a1 = aID2aIX[b[3]]
                a2 = aID2aIX[b[2]]

            d, im = cell.get_distance_and_image(frac_coords[a1]+atoms[a1],frac_coords[a2])
            
            # for CHIC need to format as a list of bonds:
            #    [[a1, image], [a2, image], d]
            connectivity += [[[a1, atoms[a1]], [a2, im], d]]

        # now format as a CHIC building unit.
        atoms_chic = [[k,v] for k,v in atoms.items()]
        bonds_chic = [[aID2aIX[b[2]],aID2aIX[b[3]]] for b in intraBonds]

        # determine label for building unit.
        if a_id.shape[0] == 1:
            l = f"a{numA}"
            numA += 1
        else:
            l = f"b{numB}"
            numB += 1
        
        units[l] = buildingUnit(structure, atoms_chic, bonds_chic, connectivity)
    
    return units

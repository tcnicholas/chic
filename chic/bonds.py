"""
12.06.23
@tcnicholas
Class for handling the bond vectors.
"""


from typing import List

import numpy as np
from pymatgen.core import Structure


class Bonding:
    """
    A class for representing the bonds in a structure as vectors to extract
    local geometric properties.
    """
    def __init__(
        self,
        struct: Structure,
        labels: List[str],
        bonds: List[List[int]],
    ) -> None:
        """
        Initialise the bonding class.

        :param struct: Pymatgen Structure object.
        :param labels: list of atom labels.
        :param bonds: list of bonds in structure. e.g.
        """
        self._struct = struct
        self._labels = labels
        self._bonds = bonds
        self._vectors = []
        self._bonded_labels = []
        self._bonds_per_atom()

    
    @property
    def labels(self) -> np.ndarray(1):
        """
        Array of atom labels.

        :return: array of atom labels in structure.
        """
        return np.array(self._labels, dtype="<U10")
    

    @property
    def all_bond_lengths(self) -> np.ndarray(1):
        """
        Compute all bond lengths.
        """
        return np.linalg.norm(np.vstack(self._vectors), axis=1)
    

    def bond_by_label(self, label: str) -> np.ndarray(2):
        """
        Return the bond vectors for a particular atom label.

        :param label: atom label.
        :return: bond vectors for atom.
        """
        return self._vectors[self._labels.index(label)]
    

    def bound_to(self, label: str) -> np.ndarray(1):
        """
        Return the labels of units bound to this label.

        :param label: atom label.
        :return: labels of atoms bound to this atom.
        """
        return [sorted(b[0])[0] for b in self._bonds if label in b[0]]
    

    def _get_bond(self, 
        a1: str,
        a2: str, 
        img1: np.ndarray(3),
        img2: np.ndarray(3),
    ):
        """
        Return the MIC bond vector between two atoms in the structure.

        :param a1: atom label 1.
        :param a2: atom label 2.
        :param img1: periodic image of atom 1.
        :param img2: periodic image of atom 2.
        :return: bond vector between atoms.
        """
        a1_frac = self._struct[self._labels.index(a1)].frac_coords + img1
        a2_frac = self._struct[self._labels.index(a2)].frac_coords + img2
        a1_cart = self._struct.lattice.get_cartesian_coords(a1_frac)
        a2_cart = self._struct.lattice.get_cartesian_coords(a2_frac)
        return a2_cart - a1_cart
    

    def _bonds_per_atom(self):
        """
        Store all bonds for each atom as centred on each atom. Note this will
        double count the bonds (e.g. will store for both the bond A1 - B2 for
        both A1 and B2).

        :param structure: Pymatgen Structure object.
        :param labels: list of atom labels.
        :param bonds: list of bonds in structure. e.g. 
                        [(atom1, atom2), (img1, img2)].
        :return: None.
        """

        # labels are in the same order as the atoms in structure.
        vectors = []; bond_labels = []
        for atom1 in self._labels:
            
            # for each atom, find all bond vectors and get label of second atom.
            vs = []; ls = []
            b = [x for x in self._bonds if atom1 in x[0]]

            for x in b:

                # get index of atom1 and atom2 in bond and atom2 label.
                ix1 = x[0].index(atom1)
                ix2 = int(bool(not ix1))
                atom2 = x[0][ix2]

                # get vector.
                v = self._get_bond(atom1, atom2, x[1][ix1], x[1][ix2])

                # append the vector and store label.
                vs.append(v)
                ls.append(atom2)
            
            # append lists to lists.
            vectors.append(np.array(vs, dtype=np.float64))
            bond_labels.append(ls)

        # format vectors as numpy array.
        self._vectors = np.array(vectors, dtype="object")
        self._bonded_labels = np.array(bond_labels, dtype="object")
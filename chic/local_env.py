"""
17.07.21
@tcnicholas
Analysis functions for atom local environments.

Chau-Hardwick order parameter for tetrahedral configurations:
[1] Mol. Phys., 1998, 93, 511--518 (10.1080/002689798169195)

CrystalNN algorithm reported in:
[2] Inorg. Chem. 2021, 60, 3, 1590–-1603 (10.1021/acs.inorgchem.0c02996)
"""

from .sites import one_cn
from .utils import *

from pymatgen.analysis.local_env import CrystalNN

from timeit import default_timer as timer
from collections import ChainMap
from functools import partial
import multiprocessing
import numpy as np
import warnings


def get_nn_dict(cnn, structure, ix):
    """
    For the given atom indices, calculates the nearest neighbours using the
    crystalNN method.

    Args
    ----
    cnn: CrystalNN
        The CrystalNN class object.
    
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    ix: list
        Indices (int) of atoms to get neighbours for.
    """
    return { i:cnn.get_nn_info(structure,i) for i in ix }


def nn_dict(structure, elements):
    """
    Get nearest-neighbour dictionary for structure using crystalNN.
    Rate-limiting step so pool to multiple cores.

    #TODO: CrystalNN works better if you can pass oxidation states to the object
    too, so maybe there is a crude way of guessing the oxidation states?

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    elements: list
        Atomic symbols for elements to search neighbours for.
    """

    warnings.simplefilter("ignore") # Throws warning from pymatgen.
    start = timer()
    cnn = CrystalNN(    weighted_cn=True, cation_anion=False, 
                        distance_cutoffs=(0.5,1), x_diff_weight=0,
                        porous_adjustment=True, search_cutoff=5,
                        fingerprint_length=None)

    ix = [i for i,a in enumerate(structure) if a.specie.symbol in elements 
                and a.specie.symbol not in one_cn]

    cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cores) as pool:
        results = pool.map(partial( get_nn_dict, cnn, structure),
                                    np.array_split(ix, cores))

    warnings.simplefilter('always')

    t = timer() - start
    print(f"* nearest-neighbour * CrystalNN algorithm took {t:.3f} s.")

    return dict(ChainMap(*results))


def get_bond(structure, labels, a1, a2, img1, img2):
    """
    Calculate bond vector between atom1 (at image1) and atom2 (at image2).

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    labels: list
        Atom labels from CIF in correct order such that the coordinates from
        structure can be taken by index.

    a1: str
        Label of atom 1 in bond.

    a2: str
        Label of atom 2 in bond.

    img1: np.ndarray
        (1x3) array of image flags for atom 1.

    img2: np.ndarray
        (1x3) array of image flags for atom 2.
    """

    # get fractional coordiantes of correct images of atoms.
    a1_pos = structure[labels.index(a1)].frac_coords + img1
    a2_pos = structure[labels.index(a2)].frac_coords + img2

    # get cartesian coordinates.
    a1_cart = structure.lattice.get_cartesian_coords(a1_pos)
    a2_cart = structure.lattice.get_cartesian_coords(a2_pos)

    return a2_cart - a1_cart


@njit
def Sg(vectors):
    """
    Calculate the angular component of the Chau-Harwick tetrahedral order
    parameter. It is the normalised sum of the squares of the differences
    between the cosines of the inter-bond angles and the cosine of the ideal
    tetrahedral angle. A perfect tetrahedron has a value of zero, and the
    extreme cases of four superimposed bonds has a value of unity. See eq (3) 
    in ref [1].

    Args
    ----
    vectors: np.ndarray
        Four (1x3) vectors that define the tetrahedron.
    """

    assert len(vectors) == 4, "Tetrahedon must be defined by 4 vectors, " \
        f"{len(vectors)} != 4."

    # iterate through (6) unique pairs of vectors.
    vals = np.zeros(6)
    for i in range(4):
        for j in range(4):
            if i < j:
                vals[int( (j-i-1) + 0.5 * i * (7-i) )] = \
                    ( np.dot( unit(vectors[i]),unit(vectors[j]) ) + (1/3) )**2
                
    return np.sum(vals) * 3 / 32


@njit
def Sk(vectors):
    """
    Calculate the distance component of the Chau-Hardwik tetrahedral order
    parameter. It is a measure of the variance of the radial distances from the
    central atom to the peripheral atoms. A perfect tetrahedron has a value of
    zero. As the configuration deviates from tetrahedralting, the value
    increases with maxiumum value of unity. See eq(4) in ref [1].

    Args
    ----
    vectors: np.ndarray
        Four (1x3) vectors that define the tetrahedron.
    """

    # get the bond lengths from vectors.
    l = np.zeros(4)
    for v in range(4):
        l[v] = np.sqrt(np.sum(np.square(vectors[v])))
    
    # calculate the arithmetic mean of the lengths.
    avgL = np.mean(l)
    
    return np.sum( np.square(l-avgL) / (4*avgL*avgL) ) / 3


class bonding:
    """
    A class for representing the bonds in a structure as vectors to extract
    local geometric properties.
    """

    def __init__(self, structure, labels, bonds):
        """
        Args
        ----
        structure: pymatgen.core.Structure
            Pymatgen Structure object. The order of atoms in the object should
            correspond with the labels provided.

        labels: list
            Atom labels from CIF in correct order such that the coordinates from
            structure can be taken by index.

        bonds: list
            All bonds in the structure (taken from TopoCIF). 
            Format: [(atom1, atom2), (img1, img2)].
        """
        # order of atom bonds for per-atom bond information.
        self._labels = labels

        # store per-atom bond information.
        self._vectors = []
        self._bonded_labels = []
        self._bonds_per_atom(structure, labels, bonds)

        # store unique bond information.
        self._u_vectors = []
        self._u_bond_labels = []
        self._unique_bonds(structure, labels, bonds)

    
    @property
    def lengths(self):
        """
        Calculate all the bond lengths in the structure.
        """
        return np.array([np.linalg.norm(v) for v in self._u_vectors])

    
    @property
    def coordination_number(self):
        """
        Calculate the number of bonds per atoms.
        """
        return np.array([len(vs) for vs in self._vectors])

    
    @property
    def labels(self):
        """
        Array of atom labels.
        """
        return np.array(self._labels, dtype="<U10")
    

    def has_coordination(self, coordination: int = 4, get_count: bool = False):
        """
        Does the structure have any atoms with the given coordination number?

        Args
        ----
        coordination: int
            Coordination number to check for.
        
        get_count: bool
            If True, returns the number of atoms with that coordination number.
        """
        # get number of occurences of that coordination.
        occur = np.count_nonzero(self.coordination_number==coordination)

        if not get_count:
            return bool(occur)

        return occur

    
    def lengths_by_species(self, species: str):
        """
        Calculate bond lengths for a particular species.

        Args
        ----
        species: str
            Chemical symbol of species to get bond lengths of (e.g. "Si").
        """

        v = [ [np.linalg.norm(v) for v in vs]
            for l,vs in zip(self.labels,self._vectors) if l.startswith(species)]
            
        return np.array(v).flatten()

    
    def angles(self, species1: str, species2: str, species3: str, degrees=True,
            average=True):
        """
        Calculate angles with species1--species2--species3, where speices2 is
        the middle atom. E.g. to search for all O-Si-O bond angles:
            
            species1="O", species2="Si", species3="O"

        Args
        ----
        species1: str
            Chemical symbol of the first "outer" atom in angle.

        species2: str
            Chemical symbol of the "middle" atom in angle.

        species3: str
            Chemical symbol of the second "outer" atom in angle.

        degrees: bool
            If True, returns the angles in degrees, else returns in radians.

        average: bool
            Whether to return the mean value for all angles in structure.
        """

        # get indices of middle species (species2).
        m = [i for i,x in enumerate(self._labels) if x.startswith(species2)]

        # iterate through "middle" species and check which are bonded to both 
        # species1 and species2 and get angle.
        angles = []
        for a2,b in zip(m, self._bonded_labels[m]):
            
            # indices of bonded species1 and species2, respectively.
            sp1 = [b.index(l) for l in b if str(l).startswith(species1)]
            sp2 = [b.index(l) for l in b if str(l).startswith(species3)]
            
            # for unique pairings of these "outer" atoms, calculate angle.
            for a1,a3 in unique_pairs(sp1,sp2):
                theta = angle(*self._vectors[a2][[a1,a3]])
                if degrees:
                    theta = rad2deg(theta)
                angles.append(theta)
        
        # if average, return the average angle in the structure.
        if average:
            return np.mean(angles)
        return np.array(angles)


    
    def chau_hardwick(self, component="angle", average=True):
        """
        Calculate the Chau-Harwick bond order parameters for all 4-coordinate
        building units.

        Args
        ----
        component: str
            "angle" or "length". Whether to calculate the angular (Sg) or bond
            length (Sk) component, respecitvely.

        average: bool
            Whether to return the mean value for all tetrahedra in structure.
        """

        component = component.lower().strip()
        
        # identify indices of 4-coordinate atoms.
        four_cn = np.nonzero(self.coordination_number==4)[0]

        # iterate through 4-coordinate atoms.
        ch = np.zeros(four_cn.shape[0])
        for i,a in enumerate(self._vectors[four_cn]):

            if component == "angle":
                ch[i] = Sg(a)
            elif component == "length":
                ch[i] = Sk(a)
            else:
                raise ValueError(f"Specifiy either 'angle' or 'length' for " \
                    "the Chau-Hardwick parameter calculation.")
        
        # if average specified, return the average value for all atoms.
        if average:
            return np.mean(ch)
        return ch

    
    def _bonds_per_atom(self, structure, labels, bonds):
        """
        Store all bonds for each atom as centred on each atom. Note this will
        double count the bonds (e.g. will store for both the bond A1 - B2 for
        both A1 and B2).

        Args
        ----
        structure: pymatgen.core.Structure
            Pymatgen Structure object. The order of atoms in the object should
            correspond with the labels provided.

        labels: list
            Atom labels from CIF in correct order such that the coordinates from
            structure can be taken by index.

        bonds: list
            All bonds in the structure (taken from TopoCIF). 
            Format: [(atom1, atom2), (img1, img2)].
        """

        # get Pymatgen Lattice method for calculating cartesian coordinates.
        gcc = getattr(structure.lattice, "get_cartesian_coords")

        # labels are in the same order as the atoms in structure.
        vectors = []; bond_labels = []
        for atom1 in labels:
            
            # for each atom, find all bond vectors and get label of second atom.
            vs = []; ls = []
            b = [x for x in bonds if atom1 in x[0]]

            for x in b:

                # get index of atom1 and atom2 in bond and atom2 label.
                ix1 = x[0].index(atom1)
                ix2 = int(bool(not ix1))
                atom2 = x[0][ix2]

                # get vector.
                v = get_bond(structure,labels,atom1,atom2,x[1][ix1],x[1][ix2])

                # append the vector and store label.
                vs.append(v)
                ls.append(atom2)
            
            # append lists to lists.
            vectors.append(np.array(vs))
            bond_labels.append(ls)

        # format vectors as numpy array.
        self._vectors = np.array(vectors)
        self._bonded_labels = np.array(bond_labels)
        
    
    def _unique_bonds(self, structure, labels, bonds):
        """
        Get unique bond vectors straight from TopoCIF. Although a little verbose
        it is easier to just "doubly store" the vectors but know these are
        unique rather than sifting though other vectors for unique pairs.

        Args
        ----
        structure: pymatgen.core.Structure
            Pymatgen Structure object. The order of atoms in the object should
            correspond with the labels provided.

        labels: list
            Atom labels from CIF in correct order such that the coordinates from
            structure can be taken by index.

        bonds: list
            All bonds in the structure (taken from TopoCIF). 
            Format: [(atom1, atom2), (img1, img2)].
        """
        self._u_bond_labels = [b[0] for b in bonds]
        self._u_vectors = [get_bond(structure,labels,*b[0],*b[1]) for b in bonds]
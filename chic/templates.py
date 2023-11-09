"""
12.06.23
@tcnicholas
MOF-FF for ZIFs parameters (H and CH3 subsituents). For details of fitting 
methodology and raw parameters, see: 
    
    J. Chem. Theory Comput., 2019, 15, 4, 2420–2432.

or alternatively, the very hepful GitHub repository:

https://github.com/fxcoudert/citable-data/tree/master/107-D%C3%BCrholt_JCTC_2019

The idea is we can generate template classes that can just be fed directly into
the same decorating procedure. They therefore need to be able to point towards
the exact bonds between atom types that are consistent with the LAMMPS input
file, which I will base of the example ones from the original paper.
"""


from dataclasses import dataclass

import numpy as np


@dataclass
class ZIF8_CH3:
    """
    Store the parameters for the ZIF-8(CH3) material.

    default_atom_types: dict of values matching a LAMMPS atom type with the atom
        type defined in Fig. 3 of the above paper. This is to ensure continuity
        of atom type mapping for construction of all materials, and also clear
        parallels with the paper. Note this mapping is that used in the example
        input scripts reported in the GitHub repo.

    charges: dict of charges for each LAMMPS atom type. I could only find these 
        defined in the example data input file, not in the SI itself.
    """

    name = 'ZIF-8(CH3)'

    mofff_parameterised = True

    default_atom_types = {
        1: 'Zn', 
        2: 'H3',
        3: 'H1',
        4: 'C1',
        5: 'C2',
        6: 'C3',
        7: 'N'
    }

    symbol_to_atom_type = {v:k for k,v in default_atom_types.items()}

    charge = {
        1:  0.72900,    # Zn
        2:  0.13730,    # H3
        3:  0.16270,    # H1
        4: -0.19230,    # C1
        5:  0.49380,    # C2
        6: -0.52760,    # C3
        7: -0.34170,    # N
    }

    mass = {
        1: 65.3800,       # Zn
        2:  1.0079,       # H3
        3:  1.0079,       # H1
        4: 12.0107,       # C1
        5: 12.0107,       # C2
        6: 12.0107,       # C3
        7: 14.0067,       # N
    }

    # imidazolate linker atom types.
    atom_types = np.array([5, 4, 4, 7, 7, 3, 3, 6, 2, 2, 2])

    # imidaolate linker atom labels.
    atom_labels = [
        'C2', 'C1a', 'C1b', 'Na', 'Nb', 'H1a', 'H1b', 'C3', 'H3', 'H3', 'H3'
    ]

    coordinates = np.array([
        [ 0.00157540,       1.14179895,      -0.00337380],  # C2        [0]
        [ 0.69389977,      -0.94732783,      -0.00075316],  # C1(a)     [1]
        [-0.69655023,      -0.94536639,      -0.00120301],  # C1(b)     [2]
        [ 1.12676485,       0.37437388,       0.00252608],  # N(a)      [3]
        [-1.12568979,       0.37652139,       0.00280390],  # N(b)      [4]
        [ 1.40677878,      -1.78020224,      -0.01496259],  # H1(a)     [5]
        [-1.41112613,      -1.77696203,      -0.01190953],  # H1(b)     [6]
        [ 0.00157540,       2.64313143,      -0.08304721],  # C3        [7]
        [-0.81449603,       3.07664393,       0.52981185],  # H3        [8]
        [ 0.97054650,       3.05470741,       0.26016854],  # H3        [9]
        [-0.16163473,       2.98456687,      -1.12856641],  # H3        [10]
    ], dtype=np.float128)

    bonds_by_index = np.array([
        [0, 7],     # C2-C3
        [0, 3],     # C2-N(a)
        [0, 4],     # C2-N(b)
        [1, 3],     # C1(a)-N(a)
        [2, 4],     # C1(b)-N(b)
        [1, 2],     # C1(a)-C1(b)
        [1, 5],     # C1(a)-H1(a)
        [2, 6],     # C1(b)-H1(b)
        [7, 8],     # C3-H3
        [7, 9],     # C3-H3
        [7, 10],    # C3-H3
    ])

    # atom-2 is the central atom of the angle in LAMMPS.
    angles_by_index = np.array([
        [3, 0, 4],  # N(a)-C2-N(b)
        [8, 7, 9],  # H3-C3-H3
        [8, 7, 10], # H3-C3-H3
        [9, 7, 10], # H3-C3-H3
        [2, 1, 5],  # C1(b)-C1(a)-H1(a)
        [1, 2, 6],  # C1(a)-C1(b)-H1(b)
        [2, 1, 3],  # C1(b)-C1(a)-N(a)
        [1, 2, 4],  # C1(a)-C1(b)-N(b)
        [0, 7, 8],  # C2-C3-H3
        [0, 7, 9],  # C2-C3-H3
        [0, 7, 10], # C2-C3-H3
        [7, 0, 3],  # C3-C2-N(a)
        [7, 0, 4],  # C3-C2-N(b)
        [5, 1, 3],  # H1(a)-C1(a)-N(a)
        [6, 2, 4],  # H1(b)-C1(b)-N(b)
        [1, 3, 0],  # C1(a)-N(a)-C2
        [2, 4, 0],  # C1(b)-N(b)-C2
    ])

    #TODO: it might be better to "insert" the zinc label into a np.nan value
    #       posiiton in the future to make it easier to generalise.
    # zinc will come last in the 'type' description, so go from inner molecule
    # to outer molecule. these angle lists will need the atom ID of the zinc
    # appending when writing to LAMMPS.
    angles_2body_by_index = np.array([
        [[0, 3], [1, 3]], # if bound through 'a'. C2-N(a)-Zn and C1(a)-N(a)-Zn
        [[0, 4], [2, 4]]  # if bound through 'b'. C2-N(b)-Zn and C1(b)-N(b)-Zn
    ])

    # atoms 2-3 form central bond in LAMMPS.
    dihedrals_by_index = np.array([
        [2, 1, 3, 0],   # C1(b)-C1(a)-N(a)-C2
        [1, 2, 4, 0],   # C1(a)-C1(b)-N(b)-C2
        [5, 1, 3, 0],   # H1(a)-C1(a)-N(a)-C2
        [6, 2, 4, 0],   # H1(b)-C1(b)-N(b)-C2
        [7, 0, 3, 1],   # C3-C2-N(a)-C1(a)
        [7, 0, 4, 2],   # C3-C2-N(b)-C1(b)
        [3, 1, 2, 4],   # N(a)-C1(a)-C1(b)-N(b)
        [5, 1, 2, 6],   # H1(a)-C1(a)-C1(b)-H1(b)
        [3, 0, 7, 8],   # N(a)-C2-C3-H3
        [3, 0, 7, 9],   # N(a)-C2-C3-H3
        [3, 0, 7, 10],  # N(a)-C2-C3-H3
        [4, 0, 7, 8],   # N(b)-C2-C3-H3
        [4, 0, 7, 9],   # N(b)-C2-C3-H3
        [4, 0, 7, 10],  # N(b)-C2-C3-H3
        [3, 0, 4, 2],   # N(a)-C2-N(b)-C1(b)
        [4, 0, 3, 1],   # N(b)-C2-N(a)-C1(a)
        [5, 1, 2, 4],   # H1(a)-C1(a)-C1(b)-N(b)
        [6, 2, 1, 3],   # H1(b)-C1(b)-C1(a)-N(a)
    ])

    # zinc will come last in the 'type' description.
    dihedrals_2body_by_index = np.array([
        [[4, 0, 3], [5, 1, 3],  # if bound through 'a'. N(b)–C2–N(a)–Zn; H1(a)–C1(a)-N(a)–Zn;
         [7, 0, 3], [2, 1, 3]], #                       C3-C2-N(a)-Zn;   C1(b)–C1(a)–N(a)–Zn
        [[3, 0, 4], [6, 2, 4],  # if bound through 'b'. N(a)–C2–N(b)–Zn; H1(b)–C1(b)-N(b)–Zn;
         [7, 0, 4], [1, 2, 4]]  #                       C3-C2-N(b)-Zn;   C1(a)–C1(b)–N(b)–Zn
    ])

    outofplanes_by_index = np.array([
        [0, 7, 3, 4],    # C2–C3–N(a)–N(b)
        [1, 5, 2, 3],    # C1(a)–H1(a)–C1(b)–N(a)
        [2, 6, 1, 4],    # C1(b)–H1(b)–C1(a)–N(b)
    ])

    outofplanes_2body_by_index = np.array([
        [[3, 1, 0]], # if bound through 'a'. N(a)–Zn–C1(a)–C2
        [[4, 2, 0]], # if bound through 'b'. N(b)–Zn–C1(b)–C2
    ])

    bond_types = {
        ('N', 'Zn'):  1,
        ('C3', 'H3'): 2,
        ('C1', 'H1'): 3,
        ('C1', 'C1'): 4,
        ('C1', 'N'):  5,
        ('C2', 'C3'): 6,
        ('C2', 'N'):  7,
    }

    angle_types = {
        ('N', 'Zn', 'N'):   1,
        ('C1', 'C1', 'H1'): 2,
        ('H1', 'C1', 'N'):  3,
        ('C1', 'C1', 'N'):  4,
        ('C3', 'C2', 'N'):  5,
        ('N', 'C2', 'N'):   6,
        ('H3', 'C3', 'H3'): 7,
        ('C2', 'C3', 'H3'): 8,
        ('C1', 'N', 'Zn'):  9,
        ('C2', 'N', 'Zn'):  10,
        ('C1', 'N', 'C2'):  11,
    }

    dihedral_types = {
        ('C1', 'N', 'Zn', 'N'):   1,
        ('C2', 'N', 'Zn', 'N'):   2,
        ('H1', 'C1', 'C1', 'H1'): 3,
        ('H1', 'C1', 'C1', 'N'):  4,
        ('N', 'C1', 'C1', 'N'):   5,
        ('H1', 'C1', 'N', 'Zn'):  6,
        ('H1', 'C1', 'N', 'C2'):  7,
        ('C1', 'C1', 'N', 'Zn'):  8,
        ('C1', 'C1', 'N', 'C2'):  9,
        ('N', 'C2', 'C3', 'H3'):  10,
        ('C3', 'C2', 'N', 'Zn'):  11,
        ('C3', 'C2', 'N', 'C1'):  12,
        ('N', 'C2', 'N', 'Zn'):   13,
        ('N', 'C2', 'N', 'C1'):   14,
    }

    out_of_plane_types = {
        ('C1', 'H1', 'C1', 'N'): 1,
        ('C2', 'C3', 'N', 'N'):  2,
        ('N', 'C1', 'C2', 'Zn'): 3,
    }

    def property_by_symbol(
        self,
        symbol: str, 
        property: str
    ) -> float:
        """
        Return the property for the given atom symbol.
        """
        return getattr(self, property)[self.symbol_to_atom_type[symbol]]
    

    @property
    def atom_charges(self) -> np.ndarray:
        """
        Per-atom charges.
        """
        return np.array([self.charge[i] for i in self.atom_types])
    

    def __len__(self) -> int:
        """
        Return the number of atoms in the building unit.
        """
        return len(self.atom_types)
    


@dataclass
class ZIF8_H:
    """
    Store the parameters for the ZIF-8(H) material.

    default_atom_types: dict of values matching a LAMMPS atom type with the atom
        type defined in Fig. 3 of the above paper. This is to ensure continuity
        of atom type mapping for construction of all materials, and also clear
        parallels with the paper. Note this mapping is that used in the example
        input scripts reported in the GitHub repo.

    charges: dict of charges for each prescribed LAMMPS atom type. I could only
        find these defined in the example data input file, not in the SI itself.
    """

    name = 'ZIF-8(H)'

    mofff_parameterised = True

    default_atom_types = {
        1: 'Zn',
        2: 'H1',
        3: 'H2',
        4: 'C1',
        5: 'C2',
        6: 'N'
    }

    symbol_to_atom_type = {v:k for k,v in default_atom_types.items()}

    charge = {
        1:  0.52120,    # Zn
        2:  0.14200,    # H1
        3:  0.11450,    # H2
        4: -0.15040,    # C1
        5: -0.05190,    # C2
        6: -0.15320,    # N
    }

    mass = {
        1: 65.3800,       # Zn
        2:  1.0079,       # H1
        3:  1.0079,       # H2
        4: 12.0107,       # C1
        5: 12.0107,       # C2
        6: 14.0067,       # N
    }

    # imidazolate linker atom types.
    atom_types = np.array([5, 4, 4, 6, 6, 2, 2, 3])

    # imidaolate linker atom labels.
    atom_labels = ['C2', 'C1a', 'C1b', 'Na', 'Nb', 'H1a', 'H1b', 'H2']

    coordinates = np.array([
        [-0.00000493,       1.13054471,       0.00032007],  # C2        [0]
        [ 0.69661426,      -0.94265861,       0.00009591],  # C1(a)     [1]
        [-0.69662368,      -0.94265876,       0.00008671],  # C1(b)     [2]
        [ 1.12929701,       0.37739266,      -0.00025418],  # N(a)      [3]
        [-1.12928266,       0.37738000,      -0.00024851],  # N(b)      [4]
        [ 1.41029730,      -1.77527570,      -0.01096395],  # H1(a)     [5]
        [-1.41030640,      -1.77527599,      -0.01098257],  # H1(b)     [6]
        [-0.00000493,       2.22845998,      -0.01704101],  # H2        [7]
    ], dtype=np.float128)

    bonds_by_index = np.array([
        [0, 7],        # C2–H2
        [1, 2],        # C1(a)–C1(b)
        [0, 3],        # C2–N(a)
        [0, 4],        # C2–N(b)
        [1, 5],        # C1(a)–H1(a)
        [2, 6],        # C1(b)–H1(b)
        [1, 3],        # C1(a)–N(a)
        [2, 4],        # C1(b)–N(b)
    ])

    angles_by_index = np.array([
        [7, 0, 3],     # H2–C2–N(a)
        [7, 0, 4],     # H2–C2–N(b)
        [3, 0, 4],     # N(a)–C2–N(b)
        [1, 2, 6],     # C1(a)–C1(b)–H1(b)
        [2, 1, 5],     # C1(b)–C1(a)–H1(a)
        [1, 2, 4],     # C1(a)–C1(b)–N(b)
        [2, 1, 3],     # C1(b)–C1(a)–N(a)
        [1, 3, 0],     # C1(a)–N(a)–C2
        [2, 4, 0],     # C1(b)–N(b)–C2
        [5, 1, 3],     # H1(a)–C1(a)–N(a)
        [6, 2, 4],     # H1(b)–C1(b)–N(b)
    ])

    angles_2body_by_index = np.array([
        [[0, 3], [1, 3]], # if bound through 'a'. C2-N(a)-Zn and C1(a)-N(a)-Zn
        [[0, 4], [2, 4]]  # if bound through 'b'. C2-N(b)-Zn and C1(b)-N(b)-Zn
    ])

    dihedrals_by_index = np.array([
        [5, 1, 2, 6],   # H1(a)–C1(a)–C1(b)–H1(b)
        [1, 2, 4, 0],   # C1(a)–C1(b)–N(b)–C2
        [2, 1, 3, 0],   # C1(b)–C1(a)–N(a)–C2
        [5, 1, 3, 0],   # H1(a)–C1(a)–N(a)–C2
        [6, 2, 4, 0],   # H1(b)–C1(b)–N(b)–C2
        [7, 0, 3, 1],   # H2–C2–N(a)–C1(a)
        [7, 0, 4, 2],   # H2–C2–N(b)–C1(b)
        [3, 0, 4, 2],   # N(a)–C2–N(b)–C1(b)
        [4, 0, 3, 1],   # N(b)–C2–N(a)–C1(a)
        [3, 1, 2, 4],   # N(a)–C1(a)–C1(b)–N(b)
        [5, 1, 2, 4],   # H1(a)–C1(a)–C1(b)–N(b)
        [6, 2, 1, 3],   # H1(b)–C1(b)–C1(a)–N(a)
    ])

    # zinc will come last in the 'type' description.
    dihedrals_2body_by_index = np.array([

        # if bound through 'a'.
        [[4, 0, 3], [5, 1, 3],  #   N(b)–C2–N(a)–Zn;    H1(a)–C1(a)-N(a)–Zn;
         [7, 0, 3], [2, 1, 3]], #   H2-C2-N(a)-Zn;      C1(b)–C1(a)–N(a)–Zn

        # if bound through 'b'.
        [[3, 0, 4], [6, 2, 4],  #   N(a)–C2–N(b)–Zn;    H1(b)–C1(b)-N(b)–Zn;
         [7, 0, 4], [1, 2, 4]]  #   H2-C2-N(b)-Zn;      C1(a)–C1(b)–N(b)–Zn
    ])

    outofplanes_by_index = np.array([
        [1, 5, 2, 3], # C1(a)–H1(a)–C1(b)-N(a)
        [2, 5, 1, 4], # C1(b)–H1(b)–C1(a)-N(b)
        [0, 7, 3, 4], # C2-H2–N(a)–N(b)
    ])

    outofplanes_2body_by_index = np.array([
        [[3, 1, 0]], # if bound through 'a'. N(a)–Zn–C1(a)–C2
        [[4, 2, 0]], # if bound through 'b'. N(b)–Zn–C1(b)–C2
    ])

    bond_types = {
        ('N', 'Zn'):  1,
        ('C1', 'H1'): 2,
        ('C2', 'H2'): 3,
        ('C1', 'C1'): 4,
        ('C1', 'N'):  5,
        ('C2', 'N'):  6,
    }

    angle_types = {
        ('N', 'Zn', 'N'):   1,
        ('C1', 'C1', 'H1'): 2,
        ('H1', 'C1', 'N'):  3,
        ('C1', 'C1', 'N'):  4,
        ('H2', 'C2', 'N'):  5,
        ('N', 'C2', 'N'):   6,
        ('C1', 'N', 'Zn'):  7,
        ('C2', 'N', 'Zn'):  8,
        ('C1', 'N', 'C2'):  9,
    }

    dihedral_types = {
        ('C1', 'N', 'Zn', 'N'):   1,
        ('C2', 'N', 'Zn', 'N'):   2,
        ('H1', 'C1', 'C1', 'H1'): 3,
        ('H1', 'C1', 'C1', 'N'):  4,
        ('N', 'C1', 'C1', 'N'):   5,
        ('H1', 'C1', 'N', 'Zn'):  6,
        ('H1', 'C1', 'N', 'C2'):  7,
        ('C1', 'C1', 'N', 'Zn'):  8,
        ('C1', 'C1', 'N', 'C2'):  9,
        ('H2', 'C2', 'N', 'Zn'):  10,
        ('H2', 'C2', 'N', 'C1'):  11,
        ('N', 'C2', 'N', 'Zn'):   12,
        ('N', 'C2', 'N', 'C1'):   13,
    }

    out_of_plane_types = {
        ('C1', 'H1', 'C1', 'N'): 1,
        ('C2', 'H2', 'N', 'N'):  2,
        ('N', 'C1', 'C2', 'Zn'): 3,
    }

    def property_by_symbol(
        self,
        symbol: str, 
        property: str
    ) -> float:
        """
        Return the property for the given atom symbol.
        """
        return getattr(self, property)[self.symbol_to_atom_type[symbol]]
    

    @property
    def atom_charges(self) -> np.ndarray:
        """
        Per-atom charges.
        """
        return np.array([self.charge[i] for i in self.atom_types])
    

    def __len__(self) -> int:
        """
        Return the number of atoms in the building unit.
        """
        return len(self.atom_types)



@dataclass
class ZIF_C4H4:
    """
    Store the parameters for the ZIF-C4H4 material (benzimidazole subsituted).

    This has not been parameterised by MOFF-FF, so we can either not provide any
    parameters, or provide the base parameters for the imidazolate linker and
    then add repulsive terms to avoid clashing imidazolate molecules.
    """

    name = 'ZIF-(C4H4)'

    mofff_parameterised = False

    default_atom_types = {
        1: 'Zn',
        2: 'H1',
        3: 'H2',
        4: 'H3',
        5: 'C1',
        6: 'C2',
        7: 'C3',
        8: 'C4',
        9: 'N',
    }

    symbol_to_atom_type = {v:k for k,v in default_atom_types.items()}

    
    charge = {
        1:  0.0,    # Zn
        2:  0.0,    # H1
        3:  0.0,    # H2
        4:  0.0,    # H3
        5:  0.0,    # C1
        6:  0.0,    # C2
        7:  0.0,    # C3
        8:  0.0,    # C4
        9:  0.0     # N
    }

    mass = {
        1: 65.3800,       # Zn
        2:  1.0079,       # H1
        3:  1.0079,       # H2
        4:  1.0079,       # H3
        5: 12.0107,       # C1
        6: 12.0107,       # C2
        7: 12.0107,       # C3
        8: 12.0107,       # C4
        9: 14.0067,       # N
    }

    # imidazolate linker atom types.
    atom_types = np.array([])

    # imidaolate linker atom labels.
    atom_labels = ['C2', 'C1a', 'C1b', 'Na', 'Nb', 'H1a', 'H1b', 'H2']

    coordinates = np.array([
        [-0.00000089,       1.05603741,       0.00000013],  # C2        [0]
        [-0.73597585,      -0.94068366,       0.00096014],  # C1(a)     [1]
        [ 0.73597694,      -0.94068283,      -0.00096007],  # C1(b)     [2]
        [-1.47976724,      -2.15283610,      -0.01499364],  # C3a       [3]
        [ 1.47977047,      -2.15283402,       0.01499333],  # C3b       [4]
        [-0.73109556,      -3.33834693,       0.01752366],  # C4a       [5]
        [ 0.73109926,      -3.33834545,      -0.01752333],  # C4b       [6]
        [-1.18813096,       0.41266361,      -0.00059483],  # N(a)      [7]
        [ 1.18813076,       0.41266547,       0.00059463],  # N(b)      [8]
        [-0.00000089,       2.00711033,      -0.00000024],  # H2        [9]
        [-2.43004830,      -2.16082845,      -0.04827870],  # H3(a)     [10]
        [ 2.43005058,      -2.16082569,       0.04827899],  # H3(b)     [11]
        [-1.19014189,      -4.16953700,       0.06600101],  # H4(a)     [12]
        [ 1.19014680,      -4.16953597,      -0.06600078]   # H4(b)     [13]
    ], dtype=np.float128)
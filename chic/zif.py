"""
13.06.23
@tcnicholas
Building units for decorating nets.
"""

from typing import List
from dataclasses import dataclass, field

from .utils import replace_a_and_b


@dataclass
class Zinc:
    """
    Store explicit look-up indexing for a given Zinc atom.
    """
    # molecule label.
    mol_id: int = 0

    # atom ID this Zinc atom.
    atom_id: int = 0

    # molecule IDs of imidazole rings bound.
    Im1: int = 0
    Im2: int = 0
    Im3: int = 0
    Im4: int = 0

    # for each Im, also if it is bound to N_a or N_b.
    Im1_N: list = field(default_factory=list)
    Im2_N: list = field(default_factory=list)
    Im3_N: list = field(default_factory=list)
    Im4_N: list = field(default_factory=list)

    # bond ids.
    BondIDs: list = field(default_factory=list)

    # store dictionaries which point to:
    #       bound imidazolate mol id : bonds ids.
    Bond2Im: dict = field(default_factory=dict)
    Angle2Im: dict = field(default_factory=dict)
    Dihedral2Im: dict = field(default_factory=dict)
    OutOfPlane2Im: dict = field(default_factory=dict)


    def bonds(self):
        """
        Spit out Zn-N bonds.
        """
        return [    [self.atom_id,   self.Im1_N[0],   ("N", "Zn")],
                    [self.atom_id,   self.Im2_N[0],   ("N", "Zn")],
                    [self.atom_id,   self.Im3_N[0],   ("N", "Zn")],
                    [self.atom_id,   self.Im4_N[0],   ("N", "Zn")]]
    

    @property
    def bound_im(self):
        """
        Return list of molecule IDs and the particular N ID (and a or b) 
        of the bound imidazolate.

            [ ( mol-ID, [nitrogen atom ID, "a" or "b"] ), ]
        """
        return [(self.Im1, self.Im1_N), (self.Im2, self.Im2_N), 
                    (self.Im3, self.Im3_N), (self.Im4, self.Im4_N)]

    
    @property
    def bound_im_mol_ids(self) -> List[int]:
        """
        List of molecule IDs of the bound imidazolates.

        :return: list of molecule IDs of the bound imidazolates.
        """
        return [self.Im1, self.Im2, self.Im3, self.Im4]
    

    @property
    def bound_im_a_or_b(self):
        return [self.Im1_N[1], self.Im2_N[1], self.Im3_N[1], self.Im4_N[1]]

    
    def a_or_b(self, im_molID):
        """
        Helper function to get N a or b for a given molID.
        """
        bound = zip(self.bound_im_molID, self.bound_im_a_or_b)
        return [a_or_b for ID, a_or_b in bound if ID==im_molID][0]

    
    def which_im(self, im_molID):
        """
        Find which Im_i attribute of the class corresponds to the given 
        imidazolate molecule ID.
        """
        all_ims = zip(self.bound_im_molID, ["Im1","Im2","Im3","Im4"])
        return [im for ID, im in all_ims if ID==im_molID]
    


@dataclass
class Imidizolate:
    """
    Store explicit look-up indexing for each atom type in the imidazolate ring.
    """

    # molecule label.
    mol_id: int = 0
    atom_ids: list = field(default_factory=list)
    atom_labels: list = field(default_factory=list)

    # bound to zinc atoms.
    Zn_a: int = 0
    Zn_b: int = 0

    # topology ids.
    BondIDs: list = field(default_factory=list)
    AngleIDs: list = field(default_factory=list)
    DihedralIDs: list = field(default_factory=list)


    def atom_id_by_label(self, label: str) -> int:
        """
        Return the atom ID for a given atom label.
        """
        return self.atom_ids[self.atom_labels.index(label)]
    

    def topology_indices(self, template, topology_type='bonds'):
        """
        Return list of bonds in the imidazolate ring.
        """
        all_bonds = []
        values = getattr(template, f'{topology_type}_by_index')
        for bond in values:
            ids = [self.atom_ids[i] for i in bond]
            labels = tuple([
                replace_a_and_b(self.atom_labels[i]) for i in bond
            ])
            all_bonds.append([*ids, labels])
        return all_bonds
    

    def topology_indices_2body(self, 
        template, 
        a_or_b: int,
        topology_type='angles',
    ):
        """
        Return list of angles in the imidazolate ring involved in 2-body
        interactions.

        :param template: template object.
        :param a_or_b: 0 or 1, for N_a or N_b.
        :param topology_type: topology type.

        """
        all_angles = []
        values = getattr(template, f'{topology_type}_2body_by_index')[a_or_b]
        for bond in values:
            ids = [self.atom_ids[i] for i in bond]
            labels = tuple([
                replace_a_and_b(self.atom_labels[i]) for i in bond
            ] + ['Zn'])
            all_angles.append([*ids, labels])
        return all_angles

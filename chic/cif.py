"""
27.07.23
@tcnicholas
Handling CIFs.
"""

import warnings
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
from scipy.spatial import cKDTree
from pymatgen.io.cif import CifParser

from .bonds import Bonding
from .tidy import unit_occupancy, no_deuterium
from .utils import remove_symbols, remove_uncertainties


topo_tags = [    
    '_topol_link.node_label_1',
    '_topol_link.node_label_2',
    '_topol_link.distance',
    '_topol_link.site_symmetry_symop_1',
    '_topol_link.site_symmetry_translation_1_x',
    '_topol_link.site_symmetry_translation_1_y',
    '_topol_link.site_symmetry_translation_1_z',
    '_topol_link.site_symmetry_symop_2',
    '_topol_link.site_symmetry_translation_2_x',
    '_topol_link.site_symmetry_translation_2_y',
    '_topol_link.site_symmetry_translation_2_z',
    '_topol_link.type',
    '_topol_link.multiplicity'
]


frac_tags = [ 
    '_atom_site_fract_x',
    '_atom_site_fract_y',
    '_atom_site_fract_z'
]


def parse_bonds(bonds):
    """
    Takes raw TopoCIF input and converts to more useful arrays of bonded-atom
    labels and their respective perioidic images.
    """
    # Extract the images for atoms 1 and 2 for each bond.
    i1s = [np.array(x, dtype=int) for x in list(zip(*bonds[4:7]))]
    i2s = [np.array(x, dtype=int) for x in list(zip(*bonds[8:11]))]

    # strip away any nonsense.
    modified_labels = [
        [remove_symbols(item) for item in sublist] for sublist in bonds[:2]
    ]
    bonds[:2] = modified_labels

    # Return them, along with the two atom-labels per bond.
    return list(zip(list(zip(*bonds[:2])),list(zip(i1s,i2s))))


def match_cif_pym_atoms(cif_dict, structure, tol=1e-5) -> None:
    """
    Match up the atom labels from CIF to the atoms in the Pymatgen Structure
    object based on fractional coordinates with a tolerance. This only works
    when the supplied CIF is in P1 space group, otherwise not all atoms are
    labelled explicitly.

    :param atoms: Dictionary of CIF atom labels as keys and fractional 
        coordinates as values.
    :param structure: Pymatgen Structure object.
    :param tol: Tolerance for matching fractional coordinates.
    :return: None.
    """

    # the coord tags are a list of tuples corresponding to the fractional
    # coordinates of each atom in the CIF. we might need to remove the 
    # uncertainties in brackets.
    coord_tags = list(zip(*[cif_dict[t] for t in frac_tags]))
    coord_tags = [[remove_uncertainties(x) for x in y] for y in coord_tags]
    coords = np.array(coord_tags, dtype=np.float64)
    cif_atoms = {
        remove_symbols(atom_label):coord 
        for atom_label,coord in zip(cif_dict["_atom_site_label"], coords)
    }

    # Create KD-Tree from Pymatgen fractional coords for efficient matching.
    pym_coords = structure.frac_coords % 1.0
    kdtree = cKDTree(pym_coords)

    pym_labels = {}
    try:
        for cif_label, cif_frac_coords in cif_atoms.items():
            dists, indices = kdtree.query(cif_frac_coords, k=1)
            if dists < tol:
                matching_index = int(indices)
                pym_labels[matching_index] = cif_label
            else:
                raise ValueError(
                    f"No matching atom found for CIF atom '{cif_label}'."
                )
            
        for i,a in enumerate(structure):
            a.properties["label"] = remove_symbols(pym_labels[i])
    except:
        warnings.warn('Unable to match CIF atoms to Pymatgen atoms.')


def get_bonding(cif_dict, structure, tol=1e-8):
    """
    Extract bonding information from CIF.
    
    :param parser: Pymatgen CifParser object.
    :param structure: Pymatgen Structure object.
    :param tol: Tolerance for matching fractional coordinates.
    :return: Bonding object.
    """

    # check if all of the topo_tags are in the cif_dict.
    if not all([t in cif_dict for t in topo_tags]):
        return None

    # Get the bonding information from the CIF.
    bonds = [cif_dict[t] for t in topo_tags]
    bonds = parse_bonds(bonds)
    labels = [a.properties["label"] for a in structure]

    # Return the Bonding object.
    return Bonding(structure, labels, bonds)


def build_neighbor_list_from_topocif(struct, bonds):
    """
    Build a neighbor list for each site in the Pymatgen Structure based on a 
    list of bonds.

    :param struct: Pymatgen Structure object.
    :param bonds: List of bonds where each bond is a tuple containing
                  ((label1, label2), (image1, image2)).
    :return: Dictionary mapping each site index to a list of neighboring sites.
    """
    label_to_index = {site.properties["label"]:i for i,site in enumerate(struct)}

    neighbor_list = {i: [] for i in range(len(struct))}

    for (label1, label2), (image1, image2) in bonds:

        index1, index2 = label_to_index.get(label1), label_to_index.get(label2)

        if index1 is not None:
            neighbor_list[index1].append({
                'site': struct[index2],
                'image': image2 - image1,
                'weight': 1.0,
                'site_index': index2,
                'distance': np.linalg.norm(image2 - image1)
            })

        if index2 is not None:
            neighbor_list[index2].append({
                'site': struct[index1],
                'image': image1 - image2,
                'weight': 1.0,
                'site_index': index1,
                'distance': np.linalg.norm(image1 - image2)
            })

    return neighbor_list


def read_cif(
    filename: str,
    primitive: bool = False,
    occupancy_tolerance: float = 100,
    merge_tolerance: float = 0.01,
    site_tolerance: float = 0,
    match_atoms_tolerance: float = 1e-8,
):
    """
    Read CIF with Pymatgen.

    :param filename: filename of CIF.
    :param primitive: whether to return primitive structure.
    :param occupancy_tolerance: occupancy tolerance for CIF parser.
    :param merge_tolerance: merge tolerance for CIF parser.
    :param site_tolerance: site tolerance for CIF parser.
    """

    # load structure from CIF. I find the Pymatgen CIF warnings irritating.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = CifParser(
            filename, 
            occupancy_tolerance = occupancy_tolerance, 
            site_tolerance = site_tolerance
        )
        struct = parser.get_structures(primitive=primitive)[0]

    # tidy the structure
    unit_occupancy(struct)
    no_deuterium(struct)
    struct.merge_sites(tol=merge_tolerance, mode="delete")

    # add the raw CIF labels to the structure properties.
    cif_dict = [x for x in parser.as_dict().values()][0]
    match_cif_pym_atoms(cif_dict, struct, tol=match_atoms_tolerance)

    # also attempt to gather the bonds from the CIF.
    try:
        bonding = get_bonding(cif_dict, struct)
    except:
        bonding = None

    return struct, bonding


def format_bond(
    atom1: str, 
    atom2: str, 
    image: Tuple[float, float, float], 
    distance: float
) -> str:
    """
    Formats the bond information into a string for writing to file.
    """
    return (f'{atom1:>8} {atom2:>8} {distance:8.5f} '
            f'{1:>4} {0:>4} {0:>4} {0:>4} '
            f'{1:>4} {image[0]:4.0f} {image[1]:4.0f} {image[2]:4.0f}  V  1\n')


class TopoCifWriter:

    def __init__(self, 
        parent_structure,
        beads: Dict, 
        bead_bonds: Dict,
        name: str = 'net'
    ):
        """
        Initialises a new instance of the TopoCifWriter class.
        """
        self._parent_structure = parent_structure
        self._beads = beads
        self._bead_bonds = bead_bonds
        self._name = name


    def write_file(self, filename: Union[str, Path], write_bonds: bool=True):
        """
        Writes the content to a file with the provided filename.
        """
        if self._beads is None or self._bead_bonds is None:
            raise ValueError("Beads or Bead Bonds are not initialised.")

        sections = [self._header(), self._cell_loop(), self._positions_loop()]

        if write_bonds and self._bead_bonds:
            sections.append(self._bonds_loop())
        
        content = "".join(sections)
        content += f"#End of data_{self._name}\n\n"

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("w+") as w:
            w.write(content)


    def _header(self) -> str:
        """
        Generates the header of file string.
        """
        form = f"'{self._parent_structure.composition.anonymized_formula}'"
        return (f"data_{self._name}\n"
            f"_chemical_formula_sum {form:>{60-len('_chemical_formula_sum')}}\n")


    def _cell_loop(self) -> str:
        """
        Writes unit cell loop to file string.
        """
        cell_params = np.ravel(self._parent_structure.lattice.parameters)
        _, Z = self._parent_structure.composition.get_reduced_composition_and_factor()
        volume = self._parent_structure.lattice.volume

        cell_string = (f"_cell_length_a\t\t\t{cell_params[0]:.5f}\n"
                       f"_cell_length_b\t\t\t{cell_params[1]:.5f}\n"
                       f"_cell_length_c\t\t\t{cell_params[2]:.5f}\n"
                       f"_cell_angle_alpha\t\t{cell_params[3]:.5f}\n"
                       f"_cell_angle_beta\t\t{cell_params[4]:.5f}\n"
                       f"_cell_angle_gamma\t\t{cell_params[5]:.5f}\n"
                       f"_cell_volume\t\t\t{volume:.5f}\n"
                       f"_cell_formula_units_Z\t\t{int(Z)}\n"
                       "_symmetry_space_group_name_H-M\t'P 1'\n"
                       "_symmetry_Int_Tables_number\t1\n"
                       "loop_\n"
                       "_symmetry_equiv_pos_site_id\n"
                       "_symmetry_equiv_pos_as_xyz\n"
                       "1 x,y,z\n")

        return cell_string


    def _positions_loop(self) -> str:
        """
        Writes atom positions loop.
        """
        positions = ["loop_\n",
                     "_atom_site_label\n",
                     "_atom_site_type_symbol\n",
                     "_atom_site_symmetry_multiplicity\n",
                     "_atom_site_fract_x\n",
                     "_atom_site_fract_y\n",
                     "_atom_site_fract_z\n",
                     "_atom_site_occupancy\n"]

        positions.extend(
            bead.to_topocif_string() + "\n" for bead in self._beads.values()
        )
        return ''.join(positions)


    def _bonds_loop(self) -> str:
        """
        Writes bonds loop to file string in the TopoCIF format.
        """
        bonds = ["loop_\n",
                 "_topol_link.node_label_1\n",
                 "_topol_link.node_label_2\n",
                 "_topol_link.distance\n",
                 "_topol_link.site_symmetry_symop_1\n",
                 "_topol_link.site_symmetry_translation_1_x\n",
                 "_topol_link.site_symmetry_translation_1_y\n",
                 "_topol_link.site_symmetry_translation_1_z\n",
                 "_topol_link.site_symmetry_symop_2\n",
                 "_topol_link.site_symmetry_translation_2_x\n",
                 "_topol_link.site_symmetry_translation_2_y\n",
                 "_topol_link.site_symmetry_translation_2_z\n",
                 "_topol_link.type\n",
                 "_topol_link.multiplicity\n"]

        for edge, images in self._bead_bonds.items():
            atom1 = self._beads[edge[0]].label
            atom2 = self._beads[edge[1]].label
            for image in images:
                bonds.append(format_bond(
                    atom1,
                    atom2, 
                    image['image'],
                    image['bead_distance']
                ))
        return ''.join(bonds)
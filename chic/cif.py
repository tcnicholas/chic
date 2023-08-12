"""
27.07.23
@tcnicholas
Handling CIFs.
"""

import warnings
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
from pymatgen.io.cif import CifParser

from .tidy import unit_occupancy, no_deuterium


def read_cif(
    filename: str,
    primitive: bool = False,
    occupancy_tolerance: float = 100,
    merge_tolerance: float = 0.01,
    site_tolerance: float = 0
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

    return struct


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
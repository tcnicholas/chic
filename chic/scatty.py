"""
27.10.23
@tcnicholas
Converting LAMMPS dump files to scatty input files.
"""

from typing import List
from pathlib import Path

import numpy as np
from ase import Atoms
from pymatgen.core import Element
from pymatgen.core.lattice import Lattice


class ScattyNodeOnlyWriter:

    def __init__(self,
        single_unit_cell: Atoms,
        trajectory: List[Atoms], 
        first_frame: Atoms = None,
        sub_Z: int = 30,
        title: str = 'Scatty-Test',
        box: List[int] = [2, 2, 2],
        precision: int = 5,
    ) -> None:
        """
        Main class for writing the scatty input files.

        Arguments:
            trajectory: The trajectory of the atoms.
            first_frame: The first frame in the trajectory. If None, the first
                frame is taken as the first frame in the trajectory.
            sub_Z: The atomic number to replace all atoms with.
        """

        # store the single unit cell and the trajectory.
        self._single_unit_cell = single_unit_cell
        self._trajectory = trajectory

        # if first frame is not given, take the first frame in the trajectory,
        # and remove this frame from the trajectory.
        self._first_frame = first_frame if first_frame is not None else self._trajectory.pop(0)

        # file admin.
        self._sub_Z = sub_Z
        self._element = Element.from_Z(self._sub_Z).symbol
        self._title = title
        self._box = np.array(box)
        self._lattice = Lattice(self._first_frame.get_cell().array)
        self._single_unit_cell_lattice = Lattice(self._single_unit_cell.get_cell().array)
        self._precision = precision

        # we need to relate each atom in the supercell to the single unit cell.
        self._supercell_mapping = None
        self._supercell_indices = None
        self._normalise_species()
        self._map_supercell_to_single_cell()

    
    def write_atom_files(self, output_dir: Path) -> None:
        """
        Write the atom files for each frame in the trajectory.
        """

        # make sure output_dir is a Path object.
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # gather header for all files.
        header = self._atoms()

        # loop over the trajectory of displacements and write the atom files.
        for file_num, displacements in enumerate(self.displacements(), 1):

            # combine the header and the displacements.
            full_file = header + self._format_displacements(displacements)
            
            # write the file.
            with open(
                output_dir / f'{self._title}_atoms_{file_num:02}.txt', 'w'
            ) as f:
                f.write(full_file)

    
    def displacements(self) -> np.ndarray:
        """
        Calculate the displacements for each atom in the trajectory relative to
        the first frame. These feature in the Scatty program as (u1, u2, u3)
        where

            **u** = u1**a** + u2**b** + u3**c**
        """

        # first get the coordinates of the first frame in frac coordinates.
        first_coords = self._first_frame.get_scaled_positions()

        # next get the coordinates of the trajectory in frac coordinates.
        trajectory_coordinates = np.stack([
            atoms.get_scaled_positions() for atoms in self._trajectory
        ])

        # hence compute the displacements in fractional coordinates.
        displacements = trajectory_coordinates - first_coords[np.newaxis, :, :]

        # return the displacements, taking into account the periodic boundary
        # conditions (using the minimum image convention).
        return displacements - np.round(displacements)
    

    def _format_displacements(self, displacements: np.ndarray) -> str:
        """
        For a given frame of displacements, format the displacements for the
        Scatty program.
        """

        # initialise the string.
        all_atoms = ''

        # template for the atoms block.
        template = (
            'ATOM {:<5d}{:<5d}{:<5d}{:<5d}'
            '{:>24.16E}{:>24.16E}{:>24.16E}'
            f' {self._element}\n'
        )

        # loop over the atoms and format the displacements.
        zipped = zip(
            self._supercell_mapping,
            self._supercell_indices,
            displacements
        )
        for id, cell, disp in zipped:
            all_atoms += template.format(id, *cell, *disp)

        # return the formatted displacements.
        return all_atoms
    
    
    def _atoms(self) -> None:
        """
        Write the atoms block of the Scatty program.

        This will remain a constant for all [title]_atoms_[number].txt files.

        Title: The title of the simulation.
        Cell: A list of the lattice parameters a,b,c in Angstrom units, followed
            by a list of the cell angles alpha, beta, gamma in degrees.
        Box: The number of crystallographic unit cells in the supercell, given
            as three integers corresponding to the number of unit cells along
            the crystallographic axes.
        SITE: The fractional coordinates of the average position of a site in 
            the crystallographic unit cell. Each site is given on a new line, 
            with the keyword SITE followed by the x,y,z fractional coordinates.
            The number of SITE lines should equal the number of sites in the 
            crystallographic unit cell.
        OCC: A list of element symbols, where each element symbol is followed by
            a real number specifying the average occupancy of the site by the 
            given element. The element symbols and occupancies should be
            separated by spaces (e.g., OCC Na 0.5 Li 0.5). Neutron-scattering
            lengths and atomic X-ray form factors for the various elements are
            stored internally in Scatty. In general, the number of OCC lines
            must equal the number of SITE lines, and the OCC lines must be given
            in the same order as the SITE lines. For convenience, however, two
            exceptions to this are possible. First, if only one OCC line is
            given, Scatty will apply the same occupancy to all sites. Second,
            the OCC lines may optionally be omitted if only magnetic scattering
            is calculated; Scatty will then assume that each site is fully 
            occupied by a single type of magnetic ion with magnetic properties
            specified by extra lines
        """

        # get lattice parameters for single unit cell.
        single_cell = Lattice.from_parameters(
            *(self._first_frame.cell.lengths() / self._box),
            *self._first_frame.cell.angles()
        )

        # first write the header.
        h = f'TITLE\t{self._title}\n'
        h += f'CELL {single_cell.a:.4f} {single_cell.b:.4f} {single_cell.c:.4f} '
        h += f'{single_cell.alpha:.4f} {single_cell.beta:.4f} {single_cell.gamma:.4f}\n'
        h += f'BOX {self._box[0]} {self._box[1]} {self._box[2]}\n'

        # next follows the specification of the average position of the atoms
        # in the crystallographic unit cell.
        for atom in self._single_unit_cell.get_scaled_positions():
            h += 'SITE{:>15.10f}{:>15.10f}{:>15.10f}\n'.format(*atom)

        # then the occupancies.
        for _ in self._single_unit_cell.get_atomic_numbers():
            h += f'OCC {self._element} 1.0\n'

        # store the header.
        return h
    
    
    def _map_supercell_to_single_cell(self) -> None:
        """
        Map the atoms in the supercell to the single unit cell. This should 
        identify both the atom ID in the crystallographic unit cell and the
        indices of the supercell in which it resides.
        """

        # first we determine the appropriate lattice for a single unit cell.
        single_cell = Lattice.from_parameters(
            *(self._first_frame.cell.lengths() / self._box),
            *self._first_frame.cell.angles()
        )

        # we then jsue this as a change of basis matrix to determine the
        # fractional coordinates of the atoms in the supercell in terms of the
        # single unit cell.
        temp_atoms = self._first_frame.copy()
        temp_atoms.set_cell(single_cell.matrix, scale_atoms=False)

        self._supercell_mapping = single_cell.get_all_distances(
            temp_atoms.get_scaled_positions(),
            self._single_unit_cell.get_scaled_positions(),
        ).argmin(axis=1) + 1

        # hence go through each atom and subtract the fractional coordinates of
        # the atom in single unit cell which it mapped to, and then determine 
        # the periodic image.
        single_unit_cell_coords = self._single_unit_cell.get_scaled_positions()
        supercell_transformed_coords = single_cell.get_fractional_coords(
            self._first_frame.get_positions()
        )

        all_diffs = []
        for coords, mapping_index in zip(supercell_transformed_coords, self._supercell_mapping):
            diff = np.round(coords-single_unit_cell_coords[mapping_index-1], 4).astype(np.uint)
            all_diffs.append(diff)
        self._supercell_indices = np.array(all_diffs)

    
    def _normalise_species(self) -> None:
        """
        Relabel all atoms with the same element.
        """
        self._single_unit_cell.numbers.fill(self._sub_Z)
        self._first_frame.numbers.fill(self._sub_Z)
        for atoms in self._trajectory:
            atoms.numbers.fill(self._sub_Z)
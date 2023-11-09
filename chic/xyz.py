"""
22.06.23
@tcnicholas
Input and output for xyz files.
"""


from typing import Dict

from ase.io import read

from .utils import kcalmol2eV


def lammps_to_extxyz(
    filename: str,
    dump_filename: str,
    mofff_computes: bool = False,
    add_info: Dict = {},
    write_cif: bool = False,
) -> None:
    """
    Read in LAMMPS dump file and write to extended xyz file.
    """

    # Read in LAMMPS dump file.
    atoms = read(dump_filename, format='lammps-dump-text', index=-1)
    atoms.info |= add_info

    # If using MOF-FF, extract the MOF-FF computed energies.
    columns = ['symbols', 'positions']
    if mofff_computes:

        mofff_headers = [
            'energy',
            'energy_pair',
            'energy_bond',
            'energy_angle',
            'energy_dihedral',
            'energy_improper',
            'energy_kspace',
        ]

        atoms.arrays["forces"] = kcalmol2eV(atoms.get_forces()) + 0
        for i, header in enumerate(mofff_headers, 1):
            atoms.arrays[header] = kcalmol2eV(atoms.arrays.pop(f'c_{i}'))
        columns += mofff_headers + ['forces']

    # Write to extended xyz file.
    atoms.write(filename, format='extxyz', columns=columns, write_results=False)

    if write_cif:
        atoms.write(str(filename).replace('.xyz', '.cif'), format='cif')

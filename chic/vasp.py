"""
20.06.23
@tcnicholas
Handle VASP input and output.
"""

from typing import Union

import ase
import numpy as np
from ase.io.vasp import _symbol_count_from_symbols

from .templates import ZIF8_CH3, ZIF8_H


def lattice_to_string(atoms, fmt=' %21.16f'):
    """
    Convert unit cell into POSCAR format.
    """
    lattice = ''
    for vec in atoms.get_cell():
        for el in vec:
            lattice += fmt % el
        lattice += '\n'
    return lattice


def symbol_count_to_string(sc):
    """
    """
    symbol_string = ''
    for sym, _ in sc:
        symbol_string += ' {:3s}'.format(sym)
    symbol_string += '\n'
    for _, count in sc:
        symbol_string += ' {:3d}'.format(count)
    return symbol_string + '\n'


def write_vasp(
    poscar_filename: str,
    atom_information_filename: str,
    atoms: ase.Atoms,
    template: Union[ZIF8_CH3, ZIF8_H],
    name: str = None,
    atom_data: bool = True
):
    """
    Write structure in POSCAR format, whilst maintaining a clear mapping between
    identifiable atom types in the ligands. That way separate atom types can be
    preserved in the MLP models (if desired).

    :param poscar_filename: name of POSCAR file to write.
    :param atom_information_filename: name of file to write atom information to.
    :param name: name of structure.
    :param atoms: ASE atoms object.
    :param template: template to use for decorating the net (H or CH3).
    :return: None.

    Notes
    -----
    The first line is a comment line reserved for the user to detail their
    simulation.
    The second line is a scaling factor that will be applied to the unscaled
    lattice vectors provided on the following three lines.
    """

    # gather all coordinates and atom types.
    coord = atoms.get_scaled_positions()
    atom_type_symbol = np.array([
        template.default_atom_types[a] for a in atoms.arrays['type']
    ])

    # get all chemical elements in alphabetical order.
    symbols = np.array(atoms.get_chemical_symbols().copy())
    all_elements = sorted(np.unique(symbols))
    sc = _symbol_count_from_symbols(sorted(symbols))
    atom_data = np.vstack([
        atoms.arrays['id'], 
        atoms.arrays['mol-id'],
        atoms.arrays['type'],
        atom_type_symbol
    ]).T

    # generate file header.
    file_str = ' '.join(x[0] for x in sc) + '\n' if name is None else name+'\n'
    file_str += '1.0000000000000000\n'
    file_str += lattice_to_string(atoms)
    file_str += symbol_count_to_string(sc)
    
    # write (direct) coordinates and also gather the specific atom details into
    # an ordered array that matches the VASP POSCAR file.
    stored_labels = []
    file_str += 'Direct\n'
    for sym in all_elements:
        ix = [i for i,s in enumerate(symbols) if s==sym]
        stored_labels.append(atom_data[ix])
        for atom in coord[ix]:
            for dcoord in atom:
                file_str += ' %19.16f' % dcoord
            file_str += '\n'

    stored_labels = np.vstack(stored_labels)
    
    with open(poscar_filename, 'w') as file:
        file.write(file_str)

    header = 'atom_id mol_id, atom_type, symbol'
    np.savetxt(
        atom_information_filename,
        stored_labels, 
        header=header,
        fmt='%s'
    )


def kpoints_gamma(filename: str) -> None:
    """
    Write KPOINTS file for a gamma-point calculation.

    :param filename: name of KPOINTS file to write.
    :return: None.
    """

    kpoints = 'Automatic mesh\n'
    kpoints += '0              ! number of k-points = 0 ->automatic generation scheme\n'
    kpoints += 'Gamma          ! generate a Gamma centered grid\n'
    kpoints += '1  1  1        ! subdivisions N_1, N_2 and N_3 along recipr. l. vectors\n'
    kpoints += '0. 0. 0.       ! optional shift of the mesh (s_1, s_2, s_3)\n'

    with open(filename, 'w') as file:
        file.write(kpoints)
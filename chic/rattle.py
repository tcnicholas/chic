"""
22.06.23
@tcnicholas
Rattling methods.
"""

import numpy as np
from ase.geometry import cellpar_to_cell
from pymatgen.core.lattice import Lattice

from .net import Net
from .bonds import Bonding


def rattle_vectors(size, stddev=0.001, seed=None, rng=None):
    if seed is not None and rng is not None:
        raise ValueError('Please do not provide both seed and rng.')
    if rng is None:
        if seed is None:
            seed = 42
        rng = np.random.default_rng(seed=seed)
    return rng.normal(scale=stddev, size=size)


def rattle_net(
    net: Net,
    random_seed: int,
    rms_atoms: float,
    cell_length_dev: float,
    cell_angle_dev: float,
):
    """
    Rattle the underlying (cg) net prior to re-decoration.
    """

    cell_parameters = net._struct.lattice.parameters
    lengths = cell_parameters[:3]
    angles = cell_parameters[3:]

    # cell perturbation.
    rng = np.random.default_rng(seed=random_seed)

    # generate cell length perturbation.
    n_lengths = np.zeros(3)
    for i,l in enumerate(lengths):
        n_lengths[i] = rng.uniform(
            low=l-(l*cell_length_dev),
            high=l+(l*cell_length_dev)
        )

    # generate cell angle perturbation.
    n_angles = np.zeros(3)
    for i,a in enumerate(angles):
        n_angles[i] = rng.uniform(low=a-cell_angle_dev, high=a+cell_angle_dev)

    # hence set the new cell.
    ncell = cellpar_to_cell(np.concatenate([n_lengths, n_angles]))
    net._struct.lattice = Lattice(ncell)

    # atom perturbation.
    displacements = rattle_vectors(
        size = net._struct.cart_coords.shape, 
        stddev = rms_atoms,
        rng = rng
    )
    for i in range(len(net._struct)):
        net._struct.translate_sites(
            indices=i, vector=displacements[i,:], frac_coords=False
        )

    # update the bonding information.
    net._bonds = Bonding(
        net._struct, 
        net._bonds._labels, 
        net._bonds._bonds
    )


def rattle_atoms(
    net: Net,
    random_seed: int,
    rms_atoms: float,
    cell_length_dev: float,
    cell_angle_dev: float,
) -> None:
    """
    Rattle the decorated net to sample distortions of the rings too. For the 
    coarse-grained models, this likely equates to introducing noise into the
    dataset since we still only represent the linkers as single spherical points
    under the SOAP representation.
    """

    cell_parameters = net._struct.lattice.parameters
    lengths = cell_parameters[:3]
    angles = cell_parameters[3:]

    # cell perturbation.
    rng = np.random.default_rng(seed=random_seed)

    # generate cell length perturbation.
    n_lengths = np.zeros(3)
    for i,l in enumerate(lengths):
        n_lengths[i] = rng.uniform(
            low=l-(l*cell_length_dev),
            high=l+(l*cell_length_dev)
        )

    # generate cell angle perturbation.
    n_angles = np.zeros(3)
    for i,a in enumerate(angles):
        n_angles[i] = rng.uniform(low=a-cell_angle_dev, high=a+cell_angle_dev)

    # hence set the new cell.
    ncell = cellpar_to_cell(np.concatenate([n_lengths, n_angles]))
    net._struct.lattice = Lattice(ncell)

    # atom perturbation.
    coords = net._decorated_atoms[:,-3:]
    displacements = rattle_vectors(
        size = coords.shape, 
        stddev = rms_atoms,
        rng = rng
    )
    net._decorated_atoms[:,-3:] = coords + displacements
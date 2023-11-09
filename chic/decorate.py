"""
12.06.23
@tcnicholas
Generalised decorate code for other imidazolate linkers.
"""

import math
from typing import Union

import numpy as np

from . import templates
from .zif import Zinc, Imidizolate
from .vector import random_vector, compute_angle, align_matrix
from .utils import setattrs


random = np.random.default_rng(seed=42)


def perturb_centroid(
    centroid: np.ndarray(3),
    vectors: np.ndarray((2,3)),
) -> np.ndarray(3):
    """
    Perturb the centroid of the ring slightly.

    :param centroid: centroid of the ring.
    :param vectors: bond vectors to use to align the linker.
    :return: perturbed centroid.
    """
    vectors = vectors.astype(np.float128)
    v_unit = vectors / np.linalg.norm(vectors, axis=1)[:,None]
    rvec = random_vector(random.random(), random.random(), norm=True)
    perturb = np.cross(rvec, v_unit[0])
    perturb /= np.linalg.norm(perturb)

    # original positions of bound zincs.
    m1 = (centroid + vectors[0]).copy()
    m2 = (centroid + vectors[1]).copy()

    # compute new bond vectors.
    centroid += perturb
    v_unit = np.array([m1-centroid, m2-centroid])
    v_unit = v_unit / np.linalg.norm(vectors, axis=1)[:,None]

    # compute new bisecting vector.
    this_way = np.sum(v_unit, axis=0)
    this_way = this_way / np.linalg.norm(this_way)

    return centroid, v_unit, this_way


#TODO: implement other orient methods. In particular, random orientation for 
# defect cases.
def place_linker(
    template: Union[templates.ZIF8_CH3, templates.ZIF8_H],
    centroid: np.ndarray(3),
    vectors: np.ndarray((2,3)),
    orient: str = 'bisect',
    ):
    """
    Determine how to orient the linker in the structure.

    :param template: template linker.
    :param centroid: centroid of the linker ring.
    :param vector: bond vectors to use to align the linker.
    :param orient: orientation method.
    """
    vectors = vectors.astype(np.float128)
    v_unit = vectors / np.linalg.norm(vectors, axis=1)[:,None]
    angle = compute_angle(*v_unit)
    if orient == 'bisect':

        # in the case of 180 angles, perturb the position of the ring slightly
        # and determine a new bisecting vector.
        if math.isclose(angle, np.pi):
            centroid, v_unit, this_way = perturb_centroid(centroid, vectors)

        # define the direction as along the bisecting vector.
        this_way = np.sum(v_unit, axis=0)
        this_way = this_way / np.linalg.norm(this_way)

        # now align molecule along bisecting vector.
        mol_perp = np.cross(*v_unit).astype(np.float128)
        mol_perp /= np.linalg.norm(mol_perp)
        coords = template.coordinates.copy()
        if not np.allclose(np.abs(mol_perp), np.array([0,0,1])):
            rot1 = align_matrix(np.array([0,0,1]), mol_perp)
            coords = np.matmul(coords, rot1)

        dir2 = coords[0].copy()
        rot2 = align_matrix(dir2, this_way)
        coords = np.matmul(coords, rot2)
        coords += centroid
    
    return coords.astype(np.float64)


def assign_closest_nitrogen(
    decorated_atoms: np.ndarray,
    get_frac: callable,
    get_dist: callable,
    a_site_unit: Zinc,
    b_site_unit: Imidizolate, 
    i: int,
) -> None:
    """
    Determine which nitrogen atom in the imidazolate is bound to this a site.

    :param decorated_atoms: array of decorated atoms.
    :param get_frac: function to get fractional coordinates.
    :param get_dist: function to get distance between two points.
    :param a_site_unit: Zinc unit.
    :param b_site_unit: Imidazolate unit.
    :param i: index of imidazolate.
    :return: None.
    """

    # first gather fractional coordinates.
    zn_p = get_frac(decorated_atoms[a_site_unit.atom_id-1][4:])
    na_p = get_frac(decorated_atoms[b_site_unit.atom_id_by_label('Na')-1][4:])
    nb_p = get_frac(decorated_atoms[b_site_unit.atom_id_by_label('Nb')-1][4:])
    
    # check two distances.
    r_na = get_dist(zn_p, na_p)[0]
    r_nb = get_dist(zn_p, nb_p)[0]
    assert r_na != r_nb, 'Unclear bonding assignment for Zn-N!'

    # assignment.
    if r_na < r_nb:
        setattrs(a_site_unit,
            **{f"Im{i}_N":[b_site_unit.atom_id_by_label('Na'), "a"]}
        )
        b_site_unit.Zn_a = a_site_unit.mol_id
    else:
        setattrs(a_site_unit,
            **{f"Im{i}_N":[b_site_unit.atom_id_by_label('Nb'), "b"]}
        )
        b_site_unit.Zn_b = a_site_unit.mol_id
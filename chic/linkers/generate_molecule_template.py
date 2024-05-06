"""
12.06.23
@tcnicholas
Take imidazolate ligand in arbitrary orientation and align the molecule plane
with the xy-plane, center the ring at the origin, and align the ring-centre and
subsituent with the positive y-axis.
"""


import ase
import numpy as np
from ase.io import read


def center_molecule(
    molecule: ase.Atoms, 
    ring_coords: np.ndarray
) -> None:
    """
    Center ring centroid at origin.

    :param molecule: molecule to center.
    :param ring_coords: coordinates of ring atoms.
    """
    ring_center = np.mean(ring_coords, axis=0)
    molecule.positions -= ring_center


def fit_plane(
    points: np.ndarray
) -> np.ndarray(3):
    """
    Fit plane to points in 3D space, and return the normal vector.

    :param points: points to fit plane to.
    """
    centroid = np.mean(points, axis=0)
    cov_mat = np.cov(points - centroid, rowvar=False)
    _, _, vh = np.linalg.svd(cov_mat)
    return vh[-1]


def rotation_matrix_from_vectors(
    vec1: np.ndarray(3), 
    vec2: np.ndarray(3),
) -> np.ndarray((3,3)):
    """
    Find the rotation matrix that aligns vec1 to vec2.

    :param vec1: vector to align.
    :param vec2: vector to align to.
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def align_to_xy_plane(
    points: np.ndarray(3), 
    subset=None
) -> np.ndarray(3):
    """
    Rotate the points so that the plane best fit to the subset lies on the 
    xy-plane.

    :param points: points to rotate.
    :param subset: subset of points to fit plane to.
    """
    subset = points if subset is None else subset
    normal = fit_plane(subset)
    rot_matrix = rotation_matrix_from_vectors(normal, [0, 0, 1])
    rotated_points = np.dot(points, rot_matrix.T)  # rotate all points
    return rotated_points


def calculate_angle(
    vec1: np.ndarray(3), 
    vec2: np.ndarray(3),
) -> float:
    """
    Calculate angle between two vectors (radians).

    :param vec1: vector 1.
    :param vec2: vector 2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return np.arccos(dot_product / norm_product)


def rotation_matrix_z(
    theta: float,
) -> np.ndarray((3,3)):
    """
    Calculate 3D rotation matrix for rotation around z-axis.

    :param theta: angle to rotate by (radians).
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def align_to_y_axis(
    points: np.ndarray(3),
    v1: np.ndarray(3),
):
    """
    Rotate points in the xy plane such that v1 aligns with the y-axis.

    :param points: points to rotate.
    :param v1: vector to align with y-axis.
    """
    y_axis = np.array([0, 1, 0])
    v1[2] = 0

    angle = calculate_angle(v1, y_axis)
    if np.cross(v1, y_axis)[2] < 0:
        angle = -angle

    rot_matrix = rotation_matrix_z(angle)
    rotated_points = np.dot(points, rot_matrix.T)

    return rotated_points


def imidazolate():
    """
    Generate an unsubstituted imidazolate molecule template.

    I started by cutting out a ligand from VEJYUF (ZIF-4) structure.
    """

    im_h = read("Im_raw.xyz")
    ring_pos = np.vstack([im_h.positions[:3,:], im_h.positions[-2:,:]])
    center_molecule(im_h, ring_pos)

    ring_pos = np.vstack([im_h.positions[:3,:], im_h.positions[-2:,:]])
    coords = align_to_xy_plane(im_h.positions, ring_pos)
    im_h.positions = coords

    # point the C2-H2 bond along the y-axis.
    c2_axis = im_h.positions[3,:]-im_h.positions[0,:]
    points = align_to_y_axis(im_h.positions, c2_axis)
    im_h.positions = points

    im_h.write("Im-H.xyz")



def methyl_imidazolate():
    """
    Generate a methyl imidazolate molecule template.

    I started by cutting out a ligand from the example ZIF8-CH3 structure and
    then we re-orient it to get a systematic template.
    """

    im_ch3 = read("ZIF8_CH3_222.xyz")
    ring_pos = np.vstack([im_ch3.positions[:3,:], im_ch3.positions[-2:,:]])
    center_molecule(im_ch3, ring_pos)

    ring_pos = np.vstack([im_ch3.positions[:3,:], im_ch3.positions[-2:,:]])
    coords = align_to_xy_plane(im_ch3.positions, ring_pos)
    im_ch3.positions = coords

    methyl_carbon_vector = im_ch3.positions[3,:]-im_ch3.positions[2,:]
    points = align_to_y_axis(im_ch3.positions, methyl_carbon_vector)
    im_ch3.positions = points

    im_ch3.write("Im-CH3.xyz")
    
    
def benzimidazolate():
    """
    Generate a benzene substituted imidazolate molecule template.
    
    I cut out a ligand from the ZIF-68-GME structure (GITTUZ CSD RefCode).
    """

    im_b = read("bIm_raw.xyz")
    ring_pos = np.vstack([
        im_b.positions[9,:],
        im_b.positions[5,:],
        im_b.positions[13,:],
        im_b.positions[0,:],
        im_b.positions[6,:],

    ])
    center_molecule(im_b, ring_pos)
    coords = align_to_xy_plane(im_b.positions, ring_pos)
    im_b.positions = coords

    # point the C2-H2 bond along the y-axis.
    c2_axis = im_b.positions[12,:]-im_b.positions[9,:]
    points = align_to_y_axis(im_b.positions, c2_axis)
    im_b.positions = points

    im_b.write("Im-C4H4.xyz")


if __name__ == '__main__':
    imidazolate()

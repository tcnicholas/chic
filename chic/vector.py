"""
07.10.23
@tcnicholas
Vector functions.
"""


import numpy as np


def unit(vector):
    """
    Return the unit vector of a given vector.

    Args
    ----
    vector: np.ndarray
        Vector to transform to unit vector.
    """
    return vector / np.sqrt(np.sum(np.square(vector)))


def compute_angle(vector1, vector2):
    """
    Calculate the angle between two vectors (in radians).

    Args
    ----
    vector1: np.ndarray
        Vector from middle point to terminal point 1.

    vector2: np.ndarray
        Vector from middle point to terminal point 2.
    """

    d = np.dot(unit(vector1), unit(vector2))
    
    # np.clip() to be included in numba v0.54 so can update then accordingly.
    if d < -1.0:
        d = -1.0
    elif d > 1.0:
        d = 1.0 
        
    return np.arccos(d)


def random_vector(
    rand1: float, 
    rand2: float, 
    norm: bool = False
) -> np.ndarray(3):
    """
    Generate random vector.

    :param rand1: random number 1.
    :param rand2: random number 2.
    :param norm: whether to normalise the vector.
    :return: random vector.
    """

    # Generate two random variables.
    phi = rand1 * 2 * np.pi
    z = rand2 * 2 - 1

    # Determine final unit vector.
    z2 = z * z
    x = np.sqrt(1 - z2) * np.sin(phi)
    y = np.sqrt(1 - z2) * np.cos(phi)
    
    v = np.array([x,y,z])
    if norm:
        return v / np.linalg.norm(v)
    return v


def renormalize_matrix_svd(R):
    """
    Renormalize a rotation matrix using SVD. I found this to work better for
    some of the cases where numerical inaccuracies were causing problems.

    :param R: rotation matrix.
    :return: renormalized rotation matrix.
    """
    u, s, vh = np.linalg.svd(R, full_matrices=True)
    return np.dot(u, vh)


def align_matrix(f,t):
    """
    Rotation matrix to align v(i) along v(f).
    See: T. Möller, J. F. Hughes; J. Graphics Tools, 1999, 4, 1--4.

    f and t should be numpy arrays.
    """

    # Separate cases for vectors alligned.
    # (1) Opposite directions:
    if np.allclose(f, -1 * t):
        return np.array([[-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float128)
    # (2) Same direction (i.e. do nothing).
    elif np.allclose(f, t):
        return np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float128)
    
    v = np.cross(t, f).astype(np.float128)
    c = np.dot(f, t).astype(np.float128)
    h = (1 - c) / (1 - c**2)
    vx, vy, vz = v
    
    matrix = np.array([  [c + h*vx**2,  h*vx*vy - vz,  h*vx*vz + vy],
              [h*vx*vy+vz,   c+h*vy**2,     h*vy*vz-vx  ],
              [h*vx*vz - vy, h*vy*vz + vx,  c+h*vz**2   ]  ], dtype=np.float64)
    
    return renormalize_matrix_svd(matrix)
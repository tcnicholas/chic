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


def angle(vector1, vector2):
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
"""
29.07.21
@tcnicholas
Module for SOAP analysis.

More on the SOAP descriptor may be found in the following two papers:
[1] Phys. Rev. B, 2013, 87, 184115 (10.1103/PhysRevB.87.184115)
[2] Phys. Chem. Chem. Phys., 2016, 18, 13754--13769 (10.1039/C6CP00415F)
"""

import itertools
import numpy as np
from numba import njit


def calc_soap_dscribe(atoms, parameters, atomic_numbers=None, periodic=True,
        sparse=False, rbf="polynomial"):
    """
    Calculate the SOAP vector for the structure.

    Args
    ----
    parameters: list
        SOAP parameters (r, sigma, n, l, zeta).
        
        r: radial cut-off.
        sigma: broadness of the Gaussian functions (smoothness).
        n: order to which the radial basis set is expanded to.
        l: order to which the angular basis set is expanded to.
        zeta: power to which the normalised SOAP kernel is raised.

    atomic_numbers: list
        Atomic numbers to include in the SOAP analysis. All elements that will
        be encountered need to be included. If None, will just include all
        elements in the structure. Note, undesirable behaviour may occur if
        comparing structures with differnet species if not all elements are
        included for both structures.

    periodic: bool
        Whether to construct a perioidic SOAP.

    sparse:

    rbf: str
        Radial basis function to use ("poylnomial" or DScribe's custom "gto"
        basis set).
    """

    from dscribe.descriptors import SOAP

    # build for all atoms in structure.
    if atomic_numbers is None:
        atomic_numbers = atoms.get_atomic_numbers()
    
    # unpack the SOAP parameters.
    r, sigma, n, l, _ = parameters

    # using DScribe implementation of SOAP, create periodic calculator.
    p_soap = SOAP(  species = np.unique(atomic_numbers),
                    rcut = r,
                    sigma = sigma,
                    nmax = n,
                    lmax = l,
                    periodic = periodic,
                    sparse = sparse,
                    rbf = rbf )
    
    return p_soap.create(atoms)


@njit
def k(zeta: int, v1: np.ndarray, v2: np.ndarray):
    """
    Normalised dot product between two SOAP vectors, raised to the power of zeta
    (5th element of parameters).

    Args
    ----
    zeta: int
        power to raise the similarity kernel to.

    v1: np.ndarray
        SOAP vector of atomic environment 1.

    v2: np.ndarray
        SOAP vector of atomic environment 2.
    """
    k_val = ( np.dot(v1,v2) / np.sqrt( np.dot(v1,v1) * np.dot(v2,v2) ) ) ** zeta

    # remove numerical errors that give k_val slightly higher than unity because
    # are problematic for caclulating the SOAP-distance matrix (taking sqrt).
    if k_val > 1:
        return 1

    return k_val


@njit
def per_cell(zeta: int, s1: np.ndarray, s2: np.ndarray):
    """
    Calculate the similarity between two unit cells given the SOAP vectors for
    atoms in each cell.

    Args
    ----
    parameters: list
        SOAP parameters (r, sigma, n, l, zeta).
        
        r: radial cut-off.
        sigma: broadness of the Gaussian functions (smoothness).
        n: order to which the radial basis set is expanded to.
        l: order to which the angular basis set is expanded to.
        zeta: power to which the normalised SOAP kernel is raised.

    """

    # re-allocate all pairwise comparisons between N (s1.shape[0]) and 
    # M (s2.shape[0]) atoms in structure 1 and 2, respectively.
    k_cell = np.zeros(s1.shape[0] * s2.shape[0])

    for i in range(s1.shape[0]):
        for j in range(s2.shape[0]):
            k_cell[int(s2.shape[0]*i + j)] = k(zeta, s1[i], s2[j])

    return np.mean(k_cell)


def kernel(zeta: int, soap_vectors: list):
    """
    Calculate the SOAP kernel for all structures in list, soap_vectors.

    Args
    ----
    zeta: int
        power to raise the similarity kernel to.

    soap_vectors: list
        SOAP vectors, with each list element corresponding to a new structure.
    """

    # initialise soap similarity kernel.
    k = np.zeros(shape=(len(soap_vectors),len(soap_vectors)))
    for i in range(k.shape[0]):
        for j in range(k.shape[0]):
            # Symmetric matrix so avoid needless work.
            if j >= i:
                k[i,j] = k[j,i] = per_cell(zeta,soap_vectors[i],soap_vectors[j])
    return k


def d_soap(k):
    """
    Transform SOAP similarity kernel to proper distance metric.

    Args
    ----
    k: float or np.ndarray
        SOAP kernel value or array of values to convert to distance metric.
    """
    return np.sqrt(2 - 2*k)


@njit
def heterogeneity(zeta: int, soap_vectors):
    """
    Calculate the X-site heterogeneity, where X is a given site type. Defined as
    the average SOAP distance between all X sites within a given structure. A
    larger heterogeneity value indicates greater variation in local geometry
    about the given site.
    
    Note this is calculated slightly differently from the per-cell comparisons 
    by not including the SOAP similarity of a given atom to itself (i.e. i < j
    rather than i <= j).

    Args
    ----
    zeta: int
        power to raise the similarity kernel to.

    soap_vectors: list
        SOAP vectors, with each list element corresponding to a new structure.
    """

    # get number of soap vectors.
    n = len(soap_vectors)

    # number of unique combinations of all n vectors (excluding i==j).
    N = 0.5 * n * (n - 1)
    
    # pre-allocate array for similarity values.
    h = np.zeros(N)

    # iterate through unique pairs of distinct SOAP vectors.
    for i in range(n):
        for j in range(n):
            if i < j:
                h[int(( j - i-1 ) + 0.5 * i * ( 2*n - i-1 ))] = \
                    k(zeta,soap_vectors[i],soap_vectors[j])

    return d_soap(np.mean(h))
"""
13.06.21
@tcnicholas
Utility module for general functions.
"""


import os
import numpy as np
from numba import njit
from pathlib import Path
from fnmatch import fnmatch


@njit
def unit(vector):
    """
    Return the unit vector of a given vector.

    Args
    ----
    vector: np.ndarray
        Vector to transform to unit vector.
    """
    return vector / np.sqrt(np.sum(np.square(vector)))


@njit
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


@njit
def rad2deg(theta):
    """
    Convert radians to degrees.

    Args
    ----
    theta: float
        Angle in radians.
    """
    return theta * 180 / np.pi


def unique_pairs(l1, l2):
    """
    Given two lists, identify unique pairings for elements where the same
    element may not be repeated.

    Args
    ----
    l1: list
    l2: list
    """
    return list({ tuple(sorted([x,y])) for x in l1 for y in l2 if x!=y })


def siteType_to_idx(sites):
    """
    Convert site letters to indices (where "a" = 0, "b" = 1, etc.).

    Args
    ----
    sites: str or list
        Site label "a", "b", etc. to convert into zero-indexing from "a".
    """

    # If multiple sites passed as single string, remove all whitespace and
    # iterate through string.
    if type(sites) == str:
        sites = sites.lower().replace(" ", "")

    return [ord(s.lower())-ord("a") for s in sites]


def iximg2cart(structure, ix, img):
    """
    Get Cartesian coordinates of atom with index, ix, and image, img, in the
    Pymatgen structure, structure.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    ix: int
        Index of atom in structure.

    img: np.ndarray
        (1x3) array representing the perioidic image of the atom.
    """
    return structure.lattice.get_cartesian_coords(structure[ix].frac_coords+img)


def cart2fracImg(lattice, cartesian_coords):
    """
    Convert cartesian coordinates to fracitonal coordiantes (using Pymatgen
    Lattice object) and then wrap into unit cell and return image.

    Args
    ----
    lattice: Pymatgen.core.Lattice
        Pymatgen Lattice object.

    cartesian_coords: np.ndarray
        Cartesian coords to convert to fractional coords and image flags.
    """

    # Get fractional coordiates.
    f = lattice.get_fractional_coords(cartesian_coords)

    # Wrap coordinates.
    fw = f % 1

    # Get image.
    img = (f - fw).astype(int)

    return fw, img


def from_directory(dirName, format="cif"):
    """
    Retrieve all files from directory. Will search recursively through all
    subfolders in the directory too.

    Args
    ----
    dirName: str
        Name of directory to search.

    format: str
        File format to search for.
    """

    return (os.path.join(path,name) for path,subdirs,files in os.walk(dirName) 
                for name in files if Path(name).suffix == f".{format}")


def separate_cifs(filePath: str):
    """
    Separate catenated CIF into individual datablocks within a temporary
    directory.

    Args
    ----
    filePath: str
        path to input file.
    """
    
    fp = Path(filePath)
    
    # create a temporary dictionary.
    if not os.path.exists(fp.stem):
        os.makedirs(fp.stem)
        
    # iterate through line-by-line and identify datablocks between 'data_' and '#END'.
    with open(fp,"r") as f:
        
        # count number of files.
        c = 1
        lines = []
        read = False
        
        for line in f:
            
            # Start reading new file.
            if line.startswith("data_"):
                read = True
                name = str(line.split("_")[-1].strip())
            
            # End of file.
            elif line.upper().startswith("#END"):
                lines.append(line)
                read = False
                c += 1
            
            # still reading same file, keep storing lines.
            if read and len(line.strip()) != 0:
                lines.append(line)
            
            # finished reading file, export it.
            if not read and len(lines) != 0:
                with open(f"{fp.stem}/{name}.cif","w+") as save:
                    save.writelines(lines)
                
                # re-initialise lines.
                lines = []
    
    return fp.stem


def atoms2html(atoms):
    """
    Convert ASE atoms objet into HTML string to be visualised in iPython
    notebooks using IPython.display.HTML. Insipired by Łukasz Mentel blog post
    at: http://lukaszmentel.com/blog/ase-jupyter-notebook/index.html.

    Args
    ----
    atoms: ase.atoms.Atoms
        ASE Atoms object to visualise.
    """

    # Center the atoms object so the image starts in frame.
    p = atoms.get_positions()
    p -= np.mean(p, axis=0)
    atoms.set_positions(p)

    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile('r+', suffix='.html') as ntf:
        atoms.write(ntf.name, format='html')
        ntf.seek(0)
        html = ntf.read()

    return html


def no_digit(s):
    """
    Remove digits from string.

    Args
    ----
    s: str
        String to remove digits from.
    """
    return "".join((x for x in s if not x.isdigit()))


def middle_element(x):
    """
    Find the middle element of a list. If the length of the list is even, get
    the two elements either side of the middle.

    Args
    ----
    x: list
    """

    # get length of path and the "ideal" center.
    l = len(x)
    c = l / 2

    # if odd, get the middle element.
    if l % 2 != 0:
        return [x[int(np.floor(c))]]
    
    # else get the two elements either side.
    return [x[int(c-1)], x[int(c)]]


def ixImg2str(x):
    """
    Encode tuple(index, image) to a string and return dictioanry to original
    formatting.

    Args
    ----
    x: tuple
        Connvert tuple(index, image) to unqiue string.
    """
    s = f"{str(x[0]).zfill(8)}{''.join((str(i) for i in x[1]))}"
    return s, {s:x}
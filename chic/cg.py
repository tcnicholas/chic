"""
17.07.21
@tcnicholas
Main coarse-graining module.

Inlcudes the functions for "reducing" the structure down to fundamental building
units and then for getting the coarse-grained representation.

#TODO: it is probable that the reduce/cluster_crawl algorithms could be 
re-hashed to work as a recursive function rather than a loop. Might be neater?

#TODO: CrystalNN also works better if you incorporate the charges of the atoms,
so if there is a neat way of guessing those automatically that might improve the
quality of the building unit search.
"""

from .sites import *
from .utils import *
from .bu import *
from .cif import *

from pymatgen.core.structure import Structure as py_structure
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np
from copy import copy


def get_seed(structure, remaining):
    """
    Get atom to start the cluster crawl. Will not start on a one-coordinate
    atom.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    remaining: set
        Set of remaining atoms still to classify.

    Returns
    -------
    remaining   :   updated "remaining" set of atoms;
    unit_seed   :   the seed atom's entry for the unit list. This has the index
                    of the seed atom, as well as the image (by default, set the
                    seed atom image as [0, 0, 0]; i.e. wrapped into unit cell).
    """
    l = copy(remaining)
    seed = None

    while l and (seed is None):

        # Try first atom in list.
        seed = l.pop()

        # Get symbol of seed atom.
        s = structure[seed].specie.symbol

        # Make sure seed atom is not one-coordinate.
        if s in one_cn:
            seed = None
    
    # When a seed atom has been chosen, remove it from the set of atoms still to
    # classify.
    remaining -= {seed}

    # Return updated "remaining atoms to classify" list, and the image of the
    # seed atom (always wrap the seed atom within the unit cell).
    return remaining, [(seed, np.array([0, 0, 0]))]


def get_neighbours(nn, ix, weight, seed, distance=None, structure=None):
    """
    Retrieve neighbours of the seed atom from the nearest-neighbour dictionary.
    The neighbours must be in the set of indices "ix", and above the specified
    weight threshold.

    Args
    ----
    nn:
        Nearest-neighbour dictionary using CrystalNN.

    ix: list
        List of indices of atoms of the correct site-type.

    intraWeight: float
        The minimum weight required to consider an atom to be a bond between a
        building unit atom and an atom of another site-type.

    seed: int
        Index of seed atom for which neighbours are being seached.

    distance: float or tuple
        (optional) set maximum distance allowed for neighbour interactions.

    structure: pymatgen.core.Structure
        Pymatgen Structure object.
    """
    n = [a for a in nn[seed] if (a["site_index"] in ix and a["weight"]>weight)]

    if distance is not None:
        assert structure is not None

        # Get lower and upper limits on the distance.
        try:
            l,u = list(iter(distance)[0:2])
        except:
            l,u = [0,distance]
        
        n = [a for a in n 
                if l < structure.get_distance(seed,a["site_index"]) < u]
    
    return n


def fast_track(remaining, unit, seed, neighbours, elements):
    """
    If any of the neighbours found are in the list of elements "elements", fast
    -track them to unit list, such that they are not searched as "seed" atoms.
    Useful for e.g. one-coordinate atoms (often two organic molecules approach
    most closely via H-atoms, which might lead to the merging of two building
    units).

    Args
    ----
    remaining: set
        Set of atom indices still left to classify.

    unit: list
        List of atoms-images in building unit so far.

    seed: tuple
        Seed atom to search for neighbours (ix, image).

    neighbours:
        List of neighbouring atom indices to search over.

    elements:
        List of atomic symbols for elements that are being fast-tracked.
    """

    ft = []
    for a in range(len(neighbours)-1, -1, -1):
        if neighbours[a]["site"].specie.symbol in elements:
            ft.append(neighbours.pop(a))
    
    # Get index and image for the fast-tracked atoms.
    branchTips = [(a["site_index"], a["image"] + seed[1]) for a in ft]
    ix = {a[0] for a in branchTips}

    # Update the "remaining" set and the buildng unit list.
    unit += branchTips
    remaining -= ix

    return remaining, unit, neighbours


def cluster_crawl(structure, nn, remaining, unit, seed, growing, intraWeight, 
                    intraBond):
    """
    Starting from the seed atoms, find nearest bonded neighbours and add them to
    the "branches" of the growing building unit.

    Args
    ----
    nn:
        Dictionary of neighbours accoridng to CrystalNN algorithm.

    remainig: set
        Indices of atoms in site-type still to classify.

    unit: list
        Current list of tuple(atomIndex, image) in the building unit.

    seed: list
        List of "seed" atoms to search for neighbours. For each atom, the seed
        is reported as the index and image [ix, img].

    growing: bool
        Whether or not the building unit list "unit" has grown during the
        cluster-crawl (used to terminate a search when complete).

    intraWeight: float
        Minimum weight required for an atom to be considered a neighbour in the
        building unit.
    """

    branches = []
    for shoot in seed:

        # Retrieve neighbours of correct site-type AND weighting > intraWeight.
        n = get_neighbours(nn, remaining, intraWeight, shoot[0])

        # If one-coordinate atoms found as neighbours, fast-track them to the
        # unit list (rather than setting them as seeds, because they should only
        # return their current neighbour).
        remaining, unit, neighbours = fast_track(remaining,unit,shoot,n,one_cn)

        # Then mature the new seed atoms into "branches" to search next time.
        branches += [(a["site_index"], a["image"]+shoot[1]) for a in n]

    # Put seed atoms into building unit list after searching for neighbours.
    unit += seed

    # If new atoms found.
    if branches:

        # Remove them from the search list.
        remaining -= {a[0] for a in branches}

        # Re-set seed atoms as the newly found "branches".
        seed = branches
    
    # If no new atoms are found, the building unit is no longer growing, and the
    # algorithm needs to terminate.
    else:
        growing = False

    return remaining, unit, seed, growing


def unqiue_atom_image(aI):
    """
    Take a list of tuple(atomIndex, image) and return unique entries. Separate 
    function required because numpy arrays are not hashable.

    Args
    ----
    aI: tuple
        tuple(atomIndex, image).
    """
    # Create a new, hashable list.
    nl = set([(a[0],tuple(a[1])) for a in aI])

    # Return with numpy array image again.
    return [(a[0],np.array(a[1],dtype=int)) for a in nl]


def get_intrabonds(structure, nn, unit, intraWeight, intraBond):
    """
    Confirm all intra-molecule bonds have been identified. Due to the way atoms
    are removed from the "remaining" set of atoms to search, some intra-bonds
    may be skipped in the cluster-crawl.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    nn:

    unit: list
        List of tuple(atomIndex, Image) for all atoms found in the building unit
        so far in the algorithm.

    intraWeight: float
        The minimum weight for a given bond required for it to be considered
        a bond within the same building unit.

    intraBond: float
        Maximum bond length between two atoms for it to be considered a bond 
        between a building unit atom and an atom of another site-type.
    """

    # get all indices.
    ix_all = {a[0] for a in unit if structure[a[0]].specie.symbol}
    ix_one_cn = {a for a in ix_all if structure[a].specie.symbol in one_cn}

    # create set of bonds.
    bonds = set()
    for a in ix_all - ix_one_cn:
        for b in get_neighbours(nn,ix_all,intraWeight,a,intraBond,structure):
            bonds |= {tuple(sorted([a,b["site_index"]]))}

    return tuple(bonds)


def reduce(structure, nn, sites, t, intraWeight, interWeight, intraBond,
            interBond, cLimit, skipClusterCrawl=False):
    """
    For a given set of sites, perform a cluster-crawl to identify the building
    units and their connectivity.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    nn:

    sites: list
        Specifying atoms in each site-type. One list per site-type. I.e. for
        ZIF-8 (Zn(mIm)2) Zn is an A site, and the C, N, H (imidazolate ring)
        are B sites, so you would pass:

    t: int
        Site-type index.

    intraWeight: float
        The minimum weight for a given bond required for it to be considered
        a bond within the same building unit.

    interWeight: float
        The minimum weight required to consider an atom to be a bond between
        a building unit atom and an atom of another site-type.

    intraBond: float
        Maximum bond length between two atoms for it to be considered a bond 
        between a building unit atom and an atom of another site-type.

    interBond: float
        Maximum bond length between two atoms for it to be considered a bond
        between two building units.

    cLimit: list
        Maximum connectivity allowed for a given building unit. One int per
        site-type should be given. E.g. for Zn(mIm2), where the Zn are A
        sites and the methyl-imidazolate molecules are B sites, you might
        pass cLimit = [4,2]

    skipClusterCrawl: bool
        Skip the cluster crawl algorithm for. Instead, any atoms of that site-
        type are kept as discrete positions.
    """

    # Site-type.
    st = chr(ord("a") + t)

    # Store all building units in a dictionary.
    units = {}

    # Get indices of all atoms with the site-type being searched.
    ix = [i for i,a in enumerate(structure) if a.specie.symbol in sites[t]]

    # Will also need the indices of all atoms *not* in the site-type being
    # searched in order to determine the connectivity of the building unit.
    # (index other).
    o_s = np.concatenate([s for i,s in enumerate(sites) if i!=t]) # other sites
    ixo = [i for i,a in enumerate(structure) if a.specie.symbol in o_s]

    # Keep track of which atoms are remaining.
    r = set(ix)

    # Setup a building unit counter.
    c = 1

    # Iterate over all atoms until all have a home.
    while r:

        # --------------------- #
        # (A) Find buildig unit #
        # --------------------- #     
        # Store atom indices and atom image flags.
        u = []

        # End algorithm if the only species left are one-coordinate atoms.
        if not {structure[a].specie.symbol for a in r} - set(one_cn):
            break
            
        # Otherwise, generate a new seed atom to start a crawl.
        r, s = get_seed(structure, r)
        
        # Perform "cluster-crawl". This repeats until the list of atoms in the
        # unit is no longer growing.
        if not skipClusterCrawl:

            g = True
            while g:

                # From a seed, find all bonded neighbours of the same site-type
                # and append them to a "branch". These "branch" atoms will
                # become new seeds in the next iteration. Once a seed has been
                # searched, it is pushed into the "u" (unit) list.
                r, u, s, g = cluster_crawl(structure, nn, r, u, s, g,
                                            intraWeight, intraBond)

        else:
            
            # Set unit as just the seed if not performing the cluster-crawl for
            # this site-type.
            u += s

        # Note, during cluster-crawl algorithm, the same atom may be added
        # multiple times to ensure all intra-bonds were found. Remove now.
        u = unqiue_atom_image(u)

        # get all intra-bonds.
        b = get_intrabonds(structure, nn, u, intraWeight, intraBond)

        # --------------------- #
        # (B) Find connectivity # 
        # --------------------- #
        # Now that the building unit has been found, need to determine its 
        # connectivity to other site-types.

        # Get indices of atoms in unit.
        ixu = [a[0] for a in u]
        
        # Iterate through atoms in unit and determine connectivity.
        co = []
        for a in u:
            
            if structure[a[0]].specie.symbol not in one_cn:
                
                # Get inter-unit neighbours.
                ne = get_neighbours(nn, ixo, interWeight, a[0], interBond, 
                                    structure)

                # Format bond: ([atom1, image1], [atom2, image2], weight) where
                # atom(i) and image(i) are the indices and image of the atoms,
                # respectivelty; 1 and 2 are the atoms in the unit and external
                # atom, respectively.
                con = [
                        [a,[x["site_index"],x["image"]+a[1]],x["weight"]]
                            for x in ne]

                # If cLimit is not None, sort the connections in order of
                # decreasing weights and choose the first cLimit-entries.
                if cLimit is not None:
                    con = sorted(co, key=lambda x: x[2], reverse=True)[:cLimit]

                co += con
        
        # Format all the collected information in buildingUnit object and store
        # in dictionary.
        units[f"{st}{c}"] = buildingUnit(structure, u, b, co)
        
        # Update counter.
        c += 1

    # Print short report to console.
    print(f"* reduce * Found {c-1} {st.upper()}-type building units.")

    return units


def coarse_grain(structure, units, method, minReqCN, **kwargs):
    """
    Create coarse-grained structure from the building units (found via the
    reduce() method). Updates the units to include the unit connectivity
    information.

    Args
    ----
    structure: pymatgen.core.Structure
        Pymatgen Structure object.

    units: list
        List of tuple(atomIndex, Image) for all atoms found in the building unit
        so far in the algorithm.

    method: str
        The coarse-graining method to use for each building unit. Currently
        implemented:

        (i) "centroid" : place the dummy atom at the centroid of all atoms
            in the building unit. Calls buildingUnit.centroid() in bu.py.

            Optional kwarg
            --------------
            ignore (list) = []: a list of chemical symbols to ignore when 
                calculating the centroid. e.g. for imidazolate (C3N2H3), in
                order to get the centre of the ring it may be desirable to 
                ignore = ["H"]

        (ii) "shortest_path" : finds the shortest path between the
            atoms that connect the building unit to other building units.
            Uses the NetworkX graph representation of the molecule. If the
            shortest path has an odd number of nodes, the dummy atom is
            placed ontop of the mdidle node. Otherwise, the dummy atom is
            placed inbetween the two middle nodes.

            Optional kwarg
            --------------
            (iii) useCycles (bool) = False: searches for cycles (rings) in 
                the building unit, contracts the cycle to a single node, cX,
                and then carries out the smallest path search. If the cX is
                the middle of the shortest path, all nodes in the cycle are
                returned, therefore having the effect that the centroid of
                the ring is used.

    minReqCN: int
        The minimum required connectivity of a building unit for it to be
        kept in the final coarse-grained structure. E.g. solvents are often
        one-coordinate (bound to a particular metal centre) as would be
        removed if minReqCN=2.
    """

    # Store to which building unit each atom belongs such that connectivity
    # between building units may be evaulated.
    ix_u = {}

    # Also store "invalid" units with connectivities below the required value.
    inv_u = []

    for ul, u in units.items():

        # First check if building unit has connectivity above the specified
        # threshold.
        cn, _ = u.cn(get_atoms=True)

        if cn >= minReqCN:

            # Get fractional coordinates and image of coarse-grained site for
            # the building unit. Here, the coarse-graining method should/must be
            # a method of the "buildingUnit".
            c = getattr(u, method)(**kwargs)

            # Set fractional coordinate and image to building unit object.
            u.frac_img = cart2fracImg(structure.lattice, c)

            # At this stage, connectivity is defined via the full-atomistic
            # atoms; however, need to adjust the percieved image of the atom as
            # a result of the exact image of the building unit. Define an
            # "image-shift" that points from the atom in the bond to the
            # discrete position for the building unit.
            for ix, im in zip(u.ix, u.img):

                # For each atom indices, store tuple(unitLabel, image-shift)
                ix_u[ix] = (ul, u.frac_img[1]-im)
        
        else:

            # Remove building unit and flag it as one not to keep in the bond-
            # defining steps below.
            inv_u.append(ul)
            del units[ul]
        
    # Now set bonds (and images) between units.
    for ul, u in units.items():

        # Get coordination number and external atoms.
        # ea = [(ix, img) for atom in externalAtoms]
        cn, ea = u.cn(get_atoms=True)

        # Iterate through external atoms and images, find unit to which they
        # belong, and determine final image of that unit.
        b = []
        for a in ea:
            
            # Get external unit and image.
            eui = ix_u[a[0]]

            if eui[0] not in inv_u:
            
                # Image of bonded other unit is the bonded atom image plus the
                # image shift for that bonded atom (defined above to compensate
                # for the difference in location of the unit centroid, relative 
                # to the bonded atom) minus the image of the unit centred on.
                # Store the bond as the tuple(unitLabel, image).
                b.append((ix_u[a[0]][0], a[1] + ix_u[a[0]][1] - u.frac_img[1]))
        
        # Update building unit property.
        u.unit_bonds = b

    # If building units removed, print to console.
    if inv_u:
        print(
            f"* coarse-grain * Removed {len(inv_u)} units with CN < {minReqCN}."
            )

    return units


def cg_atoms(atoms, units, sites, scale, scaleValue, siteMap, keepSingleAtoms,
                package):
    """
    Get positions for atoms in the coarse-grained structure and the final
    bond description. Returns a dictionary of the lattice, fractional
    coordinates, and bonds. Also provides the option to scale the lattice.

    Args
    ----
    atoms: pymatgen.core.Structure
        Pymatgen Structure object.

    units: list
        List of tuple(atomIndex, Image) for all atoms found in the building unit
        so far in the algorithm.

    sites: list
        Specifying atoms in each site-type. One list per site-type. I.e. for
        ZIF-8 (Zn(mIm)2) Zn is an A site, and the C, N, H (imidazolate ring)
        are B sites, so you would pass:

    scale: str
        Scaling method to be used. Currently supported:
            "min_xx": minimum bond length between any atoms.
            "min_ab": minimum bond length between building units.
            "avg_ab": average bond length between building units.
        
    scaleValue: float
        Length (Å) to scale the characteristic bond length (defined by
        "scale") to.

    siteMap: list
        A list of atoms to map each building unit to. Should be of the same
        length as the number of site-types. E.g. to map Zn(mIm)2 to a
        coarse-grained structure,

            siteMap = ["Si", "O"]

        would map all A sites (Zn) to Si, and all B sites (mIm) to O. If
        not set, will default to "Dummy Species" with labels DA, DB, DC, ...
        Note if creating an ASE Atoms object, real atoms must be used, and
        so siteMap *must* be set.

    keepSingleAtoms: bool
        If True, the chemical identity of the single atom building units
        will be preserved. E.g. for BIF-1-Li ( [LiB(im)]4 ) where Li and B 
        are A sites, the final coarse-grained structure would keep the Li
        and B atoms, but add dummy species for the imidazolate units.

    package: str
        "pymatgen" or "ase". If set, will return the Structure/Atoms object
        of the specified package, respectively. As noted in siteMap, ASE
        requires that real elements are set for the Atoms object.
    """

    # Extract unit cell.
    lattice = atoms.lattice.copy()

    # Extract labels, positions, and images for each building unit.
    l, p, _ = zip(*[(l,*u.frac_img) for l,u in units.items()])

    # Extract bonds in format consistent with TopoCIF specification; i.e.
    # node1_label, node2_label, distance, sym_op1, x1, y1, z1, sym_op2, 
    # x2, y2, z2, link_type, multiplicity. There will be a list of tuples, 
    # one tuple per unit, and the length of each tuple will be the number of
    # bonds stored.
    b = [u.unit_bonds for u in units.values()]
    
    # Determine scaling type now, because can avoid calling next section
    # twice to calculate the bond distances if it is "min_xx" scaling.
    if scale is not None:

        scale = scale.lower()

        if scale == "min_xx":

            # Get all distances (ignoring self-distances along diagonal).
            d = lattice.get_all_distances(p,p)
            np.fill_diagonal(d, 1000)
            
            # Get scale factor and scale the lattice to the new volume.
            sf = ( scaleValue / np.amin(d) )**3
            lattice = lattice.scale(lattice.volume * sf)

        elif scale in ["min_ab", "avg_ab"]:
            
            # Get the bond distances from the formatted bonds.
            _, d = format_bonds(lattice,l,p,b,return_lengths=True)

            # Get scale factor and scale the lattice to new volume.
            if scale == "min_ab":
                sf = ( scaleValue / np.amin(d) )**3
            elif scale == "avg_ab":
                sf = ( scaleValue / np.mean(d) )**3

            lattice = lattice.scale(lattice.volume * sf)
        
        else:
            warnings.warn(f"Scale method {scale} is not supported.")

    # Get the final TopoCIF-formatted bonds.
    b = format_bonds(lattice, l, p, b)

    # The atomMap must provide a one-to-one mapping for every site-type
    # in the structure.
    assert len(siteMap) == len(sites), "Povide a one-to-one " + \
        f"mapping of dummy-sites to atomic symbols " + \
        f"({len(sites)} != {len(siteMap)})"

    # Relabel each atom with a new symbol.
    l, symbols, b = relabel(units, siteMap, keepSingleAtoms, b)

    # Sort structure information into a dictionary.
    s_info = {  "lattice": lattice,
                "symbols": symbols,
                "labels": l,
                "frac_coords": p,
                "bonds": b              }
    
    # If package specified return either a Pymatgen Structure object, or an ASE
    # atoms object.
    s = py_structure(s_info["lattice"],s_info["symbols"],s_info["frac_coords"])

    if package is not None and package.lower() == "ase":
        s = AseAtomsAdaptor.get_atoms(s)
    
    return s_info, s
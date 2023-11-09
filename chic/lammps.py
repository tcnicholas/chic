""" 
05.08.23
@tcnicholas
Handling LAMMPS input and output files.
"""


from pathlib import Path
from datetime import datetime
from typing import Dict, Union
from collections import defaultdict

import numpy as np
from pymatgen.core.periodic_table import Element
from pymatgen.core import PeriodicSite, Lattice
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.core.structure import Structure as PymatgenStructure
from pymatgen.io.lammps.data import LammpsData, lattice_2_lmpbox

from .atomic_cluster import AtomicCluster
from .utils import most_common_value, get_first_n_letters
from .sort_sites import sort_sites, create_site_type_lookup


def guess_species_from_mass(masses: Dict[int, float]) -> str:
    """
    Guess the species from the mass of the atom.

    :param masses: dictionary of atom types to masses.
    :return: dictionary of atom types to species.
    """
    all_masses = np.array(list(masses.values()))
    ref_masses = [el.atomic_mass.real for el in Element]
    diff = np.abs(np.array(ref_masses) - all_masses[:, None])
    atomic_numbers = np.argmin(diff, axis=1) + 1
    symbols = [Element.from_Z(an).symbol for an in atomic_numbers]
    return {i:Element(s) for i, s in enumerate(symbols, 1)}


def atoms_data_to_periodic_sites(
    data: LammpsData,
    lattice: Lattice,
    atom_type_to_species: Dict[int, str]
) -> np.ndarray:
    """
    Convert the atoms data to a list of PeriodicSite objects.

    :param data: LammpsData object.
    :param lattice: Lattice of the structure.
    :param atom_type_to_species: Dictionary of atom types to species.
    :return: array of PeriodicSite objects.
    """
    symbols = [atom_type_to_species[t] for t in data.atoms['type']]
    frac_coords = lattice.get_fractional_coords(data.atoms[['x', 'y', 'z']])
    return np.array([
        PeriodicSite(s, c, lattice, to_unit_cell=True)
        for s, c in zip(symbols, frac_coords)
    ])



def find_nearest_neighbors(
    lattice: Lattice, 
    site_frac_coords: np.ndarray, 
    other_frac_coords: np.ndarray, 
    cutoff_distance: float = 2.0
) -> list:
    """
    Returns the indices, distances, and images of sites near the given site
    within a cutoff_distance.

    :param lattice: Lattice of the structure.
    :param site_frac_coords: Fractional coordinates for which to find neighbors.
    :param other_frac_coords: numpy array of fractional coordinates to search for neighbors.
    :param cutoff_distance: Maximum distance to consider a neighbor.
    :return: List of dictionaries containing the site index, distance, and image.
    """
    neighbors = []
    for i, other_site_coords in enumerate(other_frac_coords):
        dist, image = lattice.get_distance_and_image(
            site_frac_coords, other_site_coords, jimage=None
        )
        if 0 < dist <= cutoff_distance:
            neighbors.append({
                'site_index': i,
                'distance': dist,
                'image': np.array(image)
            })
    return neighbors


def find_periodic_images(
    lattice: Lattice, 
    frac_coords: np.ndarray, 
    cutoff_distance: float = 5.0
):
    """
    Find the periodic images for the given fractional coordinates based on their 
    neighbours within a cut-off distance.

    :param lattice: Lattice of the structure.
    :param frac_coords: numpy array of fractional coordinates for which to find 
        periodic images.
    :param cutoff_distance: Maximum distance to consider a neighbor.
    :return: Dictionary of images for each site.
    """
    
    starting_site_index = 0
    images = defaultdict(list)
    images[starting_site_index] = np.array([0, 0, 0], dtype=np.uint)

    visited = set()
    stack = [starting_site_index]
    
    while stack:
        site_index = stack.pop()
        visited.add(site_index)
        neighbors = find_nearest_neighbors(
            lattice, frac_coords[site_index], frac_coords, 
            cutoff_distance=cutoff_distance
        )
        for neighbor in neighbors:
            neighbor_index = neighbor['site_index']
            neighbor_image = neighbor['image']
            relative_image = neighbor_image + images[site_index]
            
            if neighbor_index not in visited and neighbor_index not in stack:
                stack.append(neighbor_index)
                images[neighbor_index] = relative_image.astype(int)
    
    return images


def read_lammps_data(
    filename: str, 
    keep_bonds: bool = True,
    cluster_by_molecule_id: bool = False,
    intramolecular_cutoff: float = 2.0,
    site_types = None,
    sort_sites_method: str = 'mof',
    atom_style: str = 'full',
):
    """
    Read structure from LAMMPS data file.

    :param filename: path to the lammps data file.
    :param keep_bonds: whether to keep the bonds in the data file and convert
        them to neighbour list dictionary.
    :param cluster_by_molecule_id: whether to cluster the atoms by molecule id.

    Note 1: if using the keep_boonds option, be careful that the bonds can be
    found using MIC, because LAMMPS does not default to include the atom images.

    Note 2: if using the cluster_by_molecule_id option, be careful that the
    molecule ids are not all the same (can be a default write option) which
    would result in a single cluster. I put some checks for this (#TODO: check).

    Note 3: we compile the required data to allow this data file to enable
    subsequent processing an accompanying lammps trajectory file.
    """
    
    # check file exists.
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist.")
    
    # read the file using the Pymatgen LammpsData parser.
    data = LammpsData.from_file(filename, sort_id=True, atom_style=atom_style)
    atom_type_to_species = guess_species_from_mass(
        data.masses.to_dict()['mass']
    )
    lattice = data.box.to_lattice()
    periodic_sites = atoms_data_to_periodic_sites(
        data,
        lattice,
        atom_type_to_species,
    )

    # generate a Pymatgen Structure object and prepare the atomic clusters and
    # neighbour list returns.
    struct = PymatgenStructure.from_sites(periodic_sites)
    atomic_clusters = None
    neighbour_list = None

    # sort the sites using the chic sorting method.
    if site_types is None:
        site_types = sort_sites(struct, sort_sites_method)
    site_type_lookup = create_site_type_lookup(site_types)
    all_sites = get_first_n_letters(len(site_types))

    #TODO: read the bonds and load them into a dictionary with weight = 1, and
    # format as a periodic neighbour with image etc.
    # gather any bonds if present.
    if keep_bonds and data.topology is not None:
        bonds = data.topology['Bonds']
        pass

    # cluster the atoms by molecule id if requested.
    if cluster_by_molecule_id:

        # extract fractional coordinates from periodic sites.
        frac_coords_array = np.array([
            site.frac_coords for site in periodic_sites
        ])

        # check that the molecule ids are not all the same.
        site_indices = np.arange(len(periodic_sites))
        mol_ids = data.atoms['molecule-ID'].to_numpy()
        unique_mol_ids = sorted(np.unique(mol_ids))

        if len(unique_mol_ids) == 1:
            raise ValueError(
                "All molecule ids are the same. Cannot cluster by molecule id."
            )
        
        # cluster the atoms by molecule id. the main components required for the
        # AtomicCluster class are the site indices, species, cartesian 
        # coordinates, and periodic images.
        atomic_clusters = defaultdict()
        site_type_counts = np.ones(len(site_types), dtype=np.uint)
        for mol_id in unique_mol_ids:
            
            # gather the sites for this molecule id.
            these_sites = periodic_sites[mol_ids==mol_id]
            these_sites_indices = site_indices[mol_ids==mol_id]
            these_species = [
                site.species.elements[0].symbol for site in these_sites
            ]

            # assign what atomic cluster type it is.
            site_type_index = site_type_lookup[most_common_value(these_species)]
            site_type = all_sites[site_type_index]
            number = site_type_counts[site_type_index]

            # gather the periodic images for these sites and convert to a set of
            # image consistent Cartesian coordinates.
            images = find_periodic_images(
                lattice, frac_coords_array[mol_ids==mol_id], 
                intramolecular_cutoff
            )
            try:
                cart_coords = lattice.get_cartesian_coords([
                    site.frac_coords+images[i] for i,site in enumerate(these_sites)
                ])
            except:
                failed = Path('.failed_clusters')
                failed.mkdir(exist_ok=True)
                temp_struct = PymatgenStructure.from_sites(these_sites)
                temp_struct.to(filename=f'{failed}/{mol_id}.cif')
                print(f"Failed to cluster molecule {mol_id}.")
                continue

            atomic_clusters[site_type, number] = AtomicCluster(
                these_sites_indices,
                these_species,
                cart_coords,
                images,
            )

            # update counter.
            site_type_counts[site_type_index] += 1
        
    return struct, atomic_clusters, neighbour_list


def process_dump_file(filename, start=0, end=None, step=1):
    """
    Generator function to parse LAMMPS dump files and yield snapshots within the 
    given range.

    :param filename: Path to the LAMMPS trajectory file.
    :param start: Starting snapshot index.
    :param end: Ending snapshot index. If None, process till the last snapshot.
    :param step: Step size between snapshots.
    :return: Yield tuple of snapshot index and the processed snapshot.
    """
    trajectory = parse_lammps_dumps(str(filename))
    for count, snapshot in enumerate(trajectory):
        if count < start:
            continue
        if end is not None and count > end:
            break
        if (count - start) % step != 0:
            continue

        # grab the lattice.
        lattice = snapshot.box.to_lattice()
        
        # sort data by atom-id and then extract the positions.
        snapshot.data.sort_values(by='id', inplace=True)
        frac_coords = lattice.get_fractional_coords(
            snapshot.data[['x','y','z']]
        )
        
        yield count, lattice, frac_coords


class LammpsDataWriter:

    def __init__(self,
        parent_structure,
        beads: Dict, 
        bead_bonds: Dict,
        name: str = 'net',
        decimal_places: int = 6
    ):
        """
        Initialises a new instance of the LammpsWriter class.
        """
        self._parent_structure = parent_structure
        self._beads = beads
        self._bead_bonds = bead_bonds
        self._name = name
        self._decimal_places = decimal_places
        self._n_types = None
        self._bond_to_bond_type = None
        self._box, self._symm_op = lattice_2_lmpbox(parent_structure.lattice)
        self._process_beads()

    
    def write_file(self, 
        filename: Union[str, Path], 
        write_bonds: bool=True,
        atom_style: str = 'full',
        two_way_bonds: bool = False,
    ):
        """
        Writes the content to a file with the provided filename.

        :param filename: The filename to write to.
        :param write_bonds: Whether to write the bonds to the file.
        :param atom_style: The atom style to use (e.g. full, atomic)
        :param two_way_bonds: Whether to write the bonds in both directions.
        """

        sections = [
            self._header(two_way_bonds), 
            self._cell_loop(),
            self._masses_loop(),
            self._positions_loop(atom_style)
        ]

        if write_bonds and self._bead_bonds:
            sections.append(self._bonds_loop(two_way_bonds))
        
        content = "".join(sections)

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("w+") as w:
            w.write(content)

    
    def _header(self, two_way_bonds: bool) -> str:
        """
        Generates the header of file string.

        The header starts with a comment line followed by a blank line. Then the
        number of atoms, bonds, angles, dihedrals, impropers, and atom types are
        written. The number of bonds, angles, dihedrals, impropers, and atom 
        types and then listed. Finally, the box dimensions are written.

        example header:

        # ZIF-8 generated by CHIC (2021-01-01 00:00:00)

        16 atoms
        15 bonds
        0 angles
        0 dihedrals
        0 impropers

        2 atom types
        1 bond types
        0 angle types
        0 dihedral types
        0 improper types

        0.000000 10.000000 xlo xhi
        0.000000 10.000000 ylo yhi
        0.000000 10.000000 zlo zhi

        Masses

        1 28.0855
        2 15.9994

        """
        h = f"# '{self._name}' generated by CHIC ({datetime.now()})\n\n"
        h += f"{len(self._beads)} atoms\n"
        h += f"{len(self._bead_bonds)*(1+two_way_bonds)} bonds\n"
        h += "0 angles\n"
        h += "0 dihedrals\n"
        h += "0 impropers\n\n"

        h += f"{self._n_types['atoms']} atom types\n"
        h += f"{self._n_types['bonds']} bond types\n"
        h += "0 angle types\n"
        h += "0 dihedral types\n"
        h += "0 improper types\n\n"
        
        return h
    

    def _cell_loop(self) -> str:
        """
        Append the cell loop to the content string. This delegates to the
        Pymatgen LammpsBox class.
        """
        return self._box.get_string(self._decimal_places) + '\n\n'
    

    def _masses_loop(self) -> str:
        """
        """
        m = f"Masses\n\n"
        for i, mass in self._masses.items():
            m += f"{i} {mass['mass']:.4f} # {mass['element']}\n"
        return m + '\n'
    

    def _positions_loop(self, atom_style: str) -> str:
        """
        Get the atom positions.
        """

        # first sort the beads by id.
        p = f'Atoms # style = "{atom_style}"\n\n'
        sorted_beads = sorted(self._beads.items(), key=lambda x: x[1].bead_id)
        for bead in sorted_beads:
            p += bead[1].to_lammps_string(
                self._parent_structure.lattice, 
                atom_style,
                self._bead_type_to_atom_type
            ) + '\n'
        return p + '\n'
    

    def _bonds_loop(self, two_way_bonds: bool) -> str:
        """
        Get the bond positions.

        :param two_way_bonds: Whether to write the bonds in both directions.

        Each bond line is of the form:

            id type atom1 atom2
        """
        p = f'Bonds\n\n'
        n_bonds = 1
        for edge, images in self._bead_bonds.items():
            atom1 = self._beads[edge[0]].bead_id
            atom2 = self._beads[edge[1]].bead_id
            bond_type = self._bond_to_bond_type[images[0]['bond_type']]
            p += f"{n_bonds} {bond_type} {atom1} {atom2}\n"
            n_bonds += 1
            if two_way_bonds:
                p += f"{n_bonds} {bond_type} {atom2} {atom1}\n"
                n_bonds += 1
        return p
    

    def _process_beads(self):
        """ 
        The beads in the native chic.Structure format are less easy to identify
        global properties (e.g. number of atom types etc.). This method
        processes the beads and bonds to extract this information.
        """

        # this will be chemical symbols, which we need to convert to unique 
        # atom types, starting from 1.
        unique_bead_types = set((bead.species for bead in self._beads.values()))
        self._bead_type_to_atom_type = {
            bead_type: i for i, bead_type in enumerate(unique_bead_types, 1)
        }
        self._masses = {
            i: {
                'mass': Element(bead).atomic_mass, 
                'element': bead
            } for i, bead in enumerate(unique_bead_types, 1)
        }
        unique_bond_types = set((
            bond['bond_type'] 
            for bonds in self._bead_bonds.values() for bond in bonds
        ))
        self._n_types = {
            'atoms': len(unique_bead_types),
            'bonds': len(unique_bond_types),
        }
        self._bond_to_bond_type = {
            bond_type: i for i, bond_type in enumerate(unique_bond_types, 1)
        }


class LammpsDumpWriter:

    def __init__(self,
        parent_structure,
        beads: Dict, 
        decimal_places: int = 6,
        timestep: int = 0,
        append: bool = True,
    ):
        """
        Initialises a new instance of the LammpsWriter class.
        """
        self._parent_structure = parent_structure
        self._beads = beads
        self._decimal_places = decimal_places
        self._timestep = timestep
        self._append = append
        self._n_types = None
        self._bond_to_bond_type = None
        self._box, self._symm_op = lattice_2_lmpbox(parent_structure.lattice)
        self._process_beads()

    
    def write_file(self, 
        filename: Union[str, Path],
        atom_style: str = 'dump'
    ):
        """
        Writes the content to a file with the provided filename.

        :param filename: The filename to write to.
        :param write_bonds: Whether to write the bonds to the file.
        :param atom_style: The atom style to use (e.g. full, atomic)
        """

        sections = [
            self._header(), 
            self._positions_loop(atom_style)
        ]
        
        content = "".join(sections)

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if self._append else "w+"
        with filename.open(write_mode) as w:
            w.write(content)

    
    def _header(self) -> str:
        """
        Generates the header of file string. Each snapshot shoudl begin with:

            ITEM: TIMESTEP
            <current_timestep>
            ITEM: NUMBER OF ATOMS
            <number_of_atoms>
            ITEM: BOX BOUNDS pp pp pp
            xlo xhi
            ylo yhi
            zlo zhi

        """
        h = "ITEM: TIMESTEP\n"
        h += f"{self._timestep}\n"
        h += "ITEM: NUMBER OF ATOMS\n"
        h += f"{len(self._beads)}\n"
        h += "ITEM: BOX BOUNDS pp pp pp\n"
        h += self._cell_loop() + '\n'
        return h
    

    def _positions_loop(self, atom_style) -> str:
        """
        Get the atom positions. The baic format is:

            ITEM: ATOMS id mol type mass x y z
        """

        # first sort the beads by id.
        if atom_style == 'mass':
            include_mass = self._masses
            p = f'ITEM: ATOMS id mol type mass x y z\n'
        else:
            include_mass = None
            p = f'ITEM: ATOMS id mol type x y z\n'
        sorted_beads = sorted(self._beads.items(), key=lambda x: x[1].bead_id)
        for bead in sorted_beads:
            p += bead[1].to_lammps_string(
                self._parent_structure.lattice, 
                atom_style,
                self._bead_type_to_atom_type,
                include_mass
            ) + '\n'
        return p

    
    def _cell_loop(self) -> str:
        """
        """
        ph = f"{{:.{self._decimal_places}f}}"
        lines = []

        lmp_box, symmop = lattice_2_lmpbox(self._parent_structure.lattice)

        for bound, d in zip(lmp_box.bounds, "xyz"):
            fillers = bound + [d] * 2
            bound_format = " ".join([ph] * 2)
            lines.append(bound_format.format(*fillers))
        if lmp_box.tilt:
            tilt_format = " ".join([ph] * 3)
            lines.append(tilt_format.format(*lmp_box.tilt))
        return "\n".join(lines)
    

    def _process_beads(self):
        """ 
        The beads in the native chic.Structure format are less easy to identify
        global properties (e.g. number of atom types etc.). This method
        processes the beads and bonds to extract this information.
        """

        # this will be chemical symbols, which we need to convert to unique 
        # atom types, starting from 1.
        unique_bead_types = set((bead.species for bead in self._beads.values()))
        self._bead_type_to_atom_type = {
            bead_type: i for i, bead_type in enumerate(unique_bead_types, 1)
        }
        self._masses = {
            i: {
                'mass': Element(bead).atomic_mass, 
                'element': bead
            } for i, bead in enumerate(unique_bead_types, 1)
        }
        self._n_types = {
            'atoms': len(unique_bead_types),
        }
"""
"""


import os
import pickle
import inspect
import warnings
import importlib
import multiprocessing
from functools import partial
from itertools import chain, product
from typing import List, Union, Dict, Tuple
from collections import ChainMap, defaultdict

import numpy as np
from ase import Atoms

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure as PymatgenStructure

from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

from .sort_sites import sort_sites, create_site_type_lookup
from .cif import read_cif, TopoCifWriter
from .gulp import GulpWriter
from .atomic_cluster import (
    Bead,
    AtomicCluster,
    _order_beads,
    _get_cluster_bead_info,
    _determine_second_bead_image,
    
)
from .lammps import (
    find_periodic_images,
    process_dump_file,
    read_lammps_data,
    LammpsDataWriter,
    LammpsDumpWriter,
)
from .utils import (
    CoarseGrainingMethodRegistry,
    crystal_toolkit_display,
    get_first_n_letters,
    site_type_to_index, 
    round_to_precision,
    parse_arguments,
    get_nn_dict,
    timer,
)


class Structure(PymatgenStructure):
    """
    The Structure class is a subclass of pymatgen Structure. It is used to
    store the structure of a material and to perform operations on it pertinent
    to structural coarse-graining.
    """

    coarse_graining_methods = CoarseGrainingMethodRegistry()
    
    def register_methods(self):
        """
        Register methods from the main coarse-graining methods directory.
        """
        method_dir = os.path.join(
            os.path.dirname(__file__),
            'coarse_graining_methods'
        )
        for file in os.listdir(method_dir):
            if file.endswith('.py') and file != '__init__.py':
                module_name = file[:-3]
                module = importlib.import_module(
                    f'.coarse_graining_methods.{module_name}', 'chic'
                )
                for method_name, method_info in module.methods.items():
                    self.coarse_graining_methods.register_from_file(
                        method_name,
                        method_info['func'],
                        method_info['bead_type']
                    )

    def __init__(self, 
        *args,
        site_types = None,
        sort_sites_method: str = 'mof',
        precomputed_atomic_clusters: Dict[str, AtomicCluster] = {},
        precomputed_neighbour_list: Dict[int, list] = None,
        cores: int = 1,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialise a Structure object.

        :param args: positional arguments to pass to pymatgen Structure.
        :param sort_sites_method: method to use for sorting sites. Either 'mof'.
        :param cores: number of cores to use for parallel operations.
        :param kwargs: keyword arguments to pass to pymatgen Structure.

        I seperate the precomputed neighbour list from the neighbour list
        generated during the coarse-graining process. This is because we can
        then update the "guess" neighbour list with the "true" neighbour list
        from the file. 

        :property _site_types: list of lists of element symbols, sorted by
            sort_sites_method.
        :property _neighbours: dictionary of nearest neighbours for each site.
            Each key corresponds to a site index and each value is a list of
            dictionaries containing the nearest neighbours of that site. Each
            dictionary contains the keys 'site', 'image' and 'weight' and
            'site_index'. If you want to assign your own neighbour list, you 
            can do so by setting these properties.
        """
        super().__init__(*args, **kwargs)
        self.register_methods()
        if site_types is None:
            self._site_types = sort_sites(self, sort_sites_method)
        else:
            self._site_types = site_types
        self._site_type_lookup = create_site_type_lookup(self._site_types)
        self._all_sites = get_first_n_letters(len(self._site_types))
        self._neighbour_list = {}
        self._neighbour_list_precomputed = precomputed_neighbour_list
        self._cores = cores
        self._atomic_clusters = precomputed_atomic_clusters
        self._element_to_cluster = {}
        self._verbose = verbose
        self._min_intra_weight = None
        self._min_intra_bond_length = None
        self._max_intra_bond_length = None
        self._minimum_coordination_numbers = None
        self._maximum_coordination_numbers = None
        self._beads = None
        self._bead_bonds = defaultdict(list)
        self._bead_neighbour_list = defaultdict(list)

    
    @timer
    def average_element_pairs(self,
        element: str,
        rmin: float = 0.0,
        rmax: float = 2.0,
        cluster_method: str = 'dbscan'
    ) -> None:
        """
        Finds all sites of a given element and averages their position if they
        are found within a given distance range. The averaging is done by
        clustering the sites and taking the mean of each cluster.

        :param element: element symbol.
        :param rmin: minimum distance between sites to be averaged.
        :param rmax: maximum distance between sites to be averaged.
        :param cluster_method: clustering method to use. Either 'dbscan' or
            'hierarchical'.
        :return: None.
        """

        # gather elements into a separate structure.
        el = Element(element)
        sites = [site for site in self if site.specie == el]
        structure_cutout = PymatgenStructure.from_sites(sites)

        # cluster the sites.
        dist_mat = structure_cutout.distance_matrix
        dist_mat[dist_mat<rmin] = rmax + 0.1
        np.fill_diagonal(dist_mat, 0)

        if cluster_method.lower() == 'dbscan':
            clusters = DBSCAN(
                eps=rmax, min_samples=1, metric='precomputed'
            ).fit_predict(dist_mat)
        elif cluster_method.lower() == 'hierarchical':
            clusters = fcluster(
                linkage(squareform((dist_mat + dist_mat.T)/2)), rmax, 'distance'
            )
        else:
            raise ValueError(f"Unknown cluster_method: {cluster_method}")

        sites = []
        for c in np.unique(clusters):
            inds = np.where(clusters == c)[0]
            species = structure_cutout[inds[0]].species
            coords = structure_cutout[inds[0]].frac_coords
            for n, i in enumerate(inds[1:]):
                offset = structure_cutout[i].frac_coords - coords
                coords = coords + (
                    (offset - np.round(offset)) / (n + 2)).astype(coords.dtype)
            sites.append(PeriodicSite(species, coords, structure_cutout.lattice))
        
        # remove the original elements.
        self.remove_species([el])
        for site in sites:
            self.append(site.species, site.frac_coords)


    @timer
    def get_neighbours_crystalnn(self, 
        cn_equals_one: List[str] = ['H' 'F', 'Cl'],
        cores: int = None,
        **kwargs
    ) -> None:
        """
        Compute neighbourlist using CrystalNN.

        This is currently the bottleneck in this code. It would be worth
        investigating whether this can be sped up.

        :param cn_equals_one: list of elements for which the coordination
            number should be set to 1 and therefore we can ignore from the 
            neighbourlist search.
        :param cores: number of cores to use for parallel operations. If none, 
            use the number of cores specified in the constructor (default = 1).
            If specified, will overwrite the number of cores specified in the
            constructor.
        """

        cores = cores or self._cores

        cnn_kwargs = {
            'weighted_cn': True,
            'cation_anion': False,
            'distance_cutoffs': (0.5, 1),
            'x_diff_weight': 0,
            'porous_adjustment': True,
            'search_cutoff': 5,
            'fingerprint_length': None
        }
        cnn_kwargs.update(kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cnn = CrystalNN(**cnn_kwargs)
            elements = list(chain(*self._site_types))

            site_indices = [
                i for i,a in enumerate(self) if a.specie.symbol in elements
                and a.specie.symbol not in cn_equals_one
            ]

            with multiprocessing.Pool(processes=cores) as pool:
                nlist = pool.map(partial(
                    get_nn_dict, cnn, self),
                    np.array_split(site_indices, cores)
                )
        
        self._neighbour_list = dict(ChainMap(*nlist))


    @classmethod
    def from_structure(cls, 
        structure: PymatgenStructure, 
        atomic_clusters=None,
        **kwargs
    ) -> 'Structure':
        """
        Create a new Structure object from an existing pymatgen Structure.

        :param structure: An existing pymatgen Structure object.
        :return: A new Structure object.
        """
        instance = cls(
            structure.lattice, 
            structure.species, 
            structure.frac_coords, 
            validate_proximity=False,
            **kwargs
        )
        if atomic_clusters:
            instance._atomic_clusters = atomic_clusters
        return instance
    
    
    @classmethod
    def from_cif(cls, filename: str, **kwargs) -> 'Structure':
        """
        Read structures from CIF file.

        :param filename: path to CIF file.
        :return: Structure object.
        """
        read_cif_kwargs = {
            k:v for k, v in kwargs.items() 
            if k in inspect.signature(read_cif).parameters and k != 'self'
        }
        cls_kwargs = {
            k:v for k, v in kwargs.items()
            if k in inspect.signature(cls.__init__).parameters and k != 'self'
        }
        return cls.from_structure(
            read_cif(filename, **read_cif_kwargs), 
            **cls_kwargs
        )
    

    @classmethod
    def from_lammps_data(cls, filename: str, **kwargs) -> 'Structure':
        """
        Read structures from LAMMPS data file.

        :param filename: path to LAMMPS data file.
        :return: Structure object.
        """
        read_lammps_kwargs = {
            k:v for k, v in kwargs.items() 
            if k in inspect.signature(read_lammps_data).parameters and k != 'self'
        }
        cls_kwargs = {
            k:v for k, v in kwargs.items()
            if k in inspect.signature(cls.__init__).parameters and k != 'self'
        }

        # read the structure, clusters, and neighbourlist. the latter two are 
        # only returned if requested!
        struct, clusters, nl = read_lammps_data(filename, **read_lammps_kwargs)
        cls_kwargs['precomputed_atomic_clusters'] = clusters
        cls_kwargs['precomputed_neighbour_list'] = nl
        return cls.from_structure(struct, **cls_kwargs)

    
    def append_lammps_trajectory(self, 
        filename, 
        intramolecular_cutoff=2.0,
        start=0,
        end=None,
        step=1,
        verbose=False,
    ):
        """
        Append a LAMMPS trajectory file to the current structure object. For 
        each cluster, the atomic clusters are updated to reflect the snapshot's 
        structure. Only snapshots in the given range are returned.

        :param filename: Path to the LAMMPS trajectory file.
        :param intramolecular_cutoff: Cutoff for determining atomic clusters.
        :param start: Starting snapshot index.
        :param end: Ending snapshot index. If None, process full trajectory.
        :param step: Step size between snapshots.
        :param verbose: whether to instantiate the new chic classes with timing.
        :param remove_elements: a list of elements to automatically remove from 
            the structure before processing.
        :return: Yield tuple of snapshot index and the corresponding Structure 
            object.
        """

        # parse the dump file one frame at a time.
        snapshots = process_dump_file(filename, start, end, step)
        for count, lattice, frac_coords in snapshots:

            atomic_clusters = defaultdict()
            for label, cluster in self._atomic_clusters.items():

                new_frac_coords = frac_coords[cluster.site_indices]
                new_images = find_periodic_images(
                    lattice, new_frac_coords, intramolecular_cutoff
                )
                new_cart_coords = lattice.get_cartesian_coords(
                    [fc+new_images[i] for i,fc in enumerate(new_frac_coords)]
                )
                atomic_clusters[label] = \
                    AtomicCluster.with_updated_coordinates_and_images(
                        cluster, new_cart_coords, new_images
                    )
            
            # Create the new Structure object using the new lattice, the old
            # species, and the new fractional coordinates.
            snapshot_structure = type(self).from_structure(
                PymatgenStructure(lattice, self.species, frac_coords),
                atomic_clusters=atomic_clusters, verbose=verbose
            )

            yield count, snapshot_structure


    def get_site_type_index(self, item: str) -> int:
        try:
            return self._site_type_lookup[item]
        except KeyError:
            raise ValueError(f"Item {item} not found in any list.")
        
    
    def get_site_type(self, item: str) -> str:
        try:
            return self._all_sites[self._site_type_lookup[item]]
        except KeyError:
            raise ValueError(f"Item {item} not found in any list.")
        
    
    @timer
    def find_atomic_clusters(self,
        skip_sites: List[str] = None,
        min_intra_weight: Union[float, List[float], Dict[str, float]] = None,
        min_intra_bond_length: Union[float,List[float], Dict[str,float]] = None,
        max_intra_bond_length: Union[float,List[float], Dict[str,float]] = None,
        min_inter_weight: Union[float, List[float], Dict[str, float]] = None,
        min_inter_bond_length: Union[float,List[float], Dict[str,float]] = None,
        max_inter_bond_length: Union[float,List[float], Dict[str,float]] = None,
    ):
        """
        Identify atomic clusters for each site-type.

        When determining bonds, the 'weight' is used to determine the likelihood 
        of a bond existing between two sites. A value of 1.0 means that the bond 
        is definitely present, while a value of 0.0 means that the bond is 
        definitely not present. Tuning the required weight for a bond to be
        present can be useful for identifying building units. 
        
        We divide this into two: the minimum weight for intra-unit bonds and the 
        minimum weight for inter-unit bonds. The former determines the minimum 
        weight for a bond to exist within a given unit (e.g. a MOF ligand), 
        while the latter determines the minimum weight for two building units to 
        be considered bonded (e.g. between node and linker).

        :param skip_sites: list of site types to skip when identifying building
            units. For example, if you want to skip the A sites and instead keep
            them as single atoms, set skip_sites = 'a'. You can skip multiple
            sites by passing a list of site types, e.g. skip_sites = ['a', 'b'].
        :param min_intra_weight: minimum weight required intra-unit edges.
        :param min_intra_bond_length: minimum bond length for intra-unit edges.
        :param max_intra_bond_length: maximum bond length for intra-unit edges.
        :return: None.
        """

        if self._neighbour_list is None:
            raise ValueError('Neighbourlist not computed. Please run ' \
                'get_neighbours_crystalnn() first.')

        # parse all neighbour conditions.
        skip_sites = site_type_to_index(skip_sites)
        self._min_intra_weight = parse_arguments(
            0.2, min_intra_weight, self._all_sites
        )
        self._min_intra_bond_length = parse_arguments(
            0.0, min_intra_bond_length, self._all_sites
        )
        self._max_intra_bond_length = parse_arguments(
            None, max_intra_bond_length, self._all_sites
        )
        self._min_inter_weight = parse_arguments(
            0.4, min_inter_weight, self._all_sites
        )
        self._min_inter_bond_length = parse_arguments(
            0.0, min_inter_bond_length, self._all_sites
        )
        self._max_inter_bond_length = parse_arguments(
            None, max_inter_bond_length, self._all_sites
        )

        # compute adjacency matrix using the neighbourlist and any constraints.
        adjacency_matrix = self._get_adjacency_matrix()

        # identify clusters
        clusters = self._identify_clusters(adjacency_matrix)

        # relabel clusters based on site types and cache them.
        self._finalise_clusters(clusters)

        # connect the clusters.
        self._connect_clusters()


    @timer
    def get_coarse_grained_net(self,
        method: str = 'centroid',
        minimum_coordination_numbers: Union[int, List[int]] = None,
        maximum_coordination_numbers: Union[int, List[int]] = None,
        **kwargs
    ):
        """
        Coarse-grain the atomic clusters and return a network representation of
        the structure. The logic of the method is as follows:

            1) For each atomic cluster, place the cg bead(s).
            2) For each atomic cluster, find the neighbours of the cg bead(s),
                alongside the correct periodic image (by tracing along the 
                connected atomistic sites). From here, it will be possible to 
                fully define the cg network.

        Supported methods can be found by calling:

            Structure.coarse_graining_methods.available_methods().

        Note to developers of methods:

        Methods can be implemented in two ways. Firstly, by building on the main 
        Structure class by adding a class method and assigning the 
            
            @coarse_graining_methods.register(<method/name>, <bead/type>) 
        
        decorator. Secondly, by adding a .py file to the 
            
            "/chic/coarse_graining_methods/"

        directory. Each method can be defined as a function, taking the 'self'
        argument. The file must contain a dictionary called 'methods' which
        contains the following information for each method:

            - 'func': the function to call.
            - 'bead_type': the type of bead to use. This can be 'single' or
                'multi'.

        The method will then be automatically registered and available to use.
        """

        # extract method (if implemented).
        method_func = self.coarse_graining_methods.get_method(method.lower())
        if not method_func:
            raise ValueError(f'Unknown method: {method}')
        
        # parse arguments.
        self._minimum_coordination_numbers = parse_arguments(
            2, minimum_coordination_numbers, self._all_sites
        )
        self._maximum_coordination_numbers = parse_arguments(
            4, maximum_coordination_numbers, self._all_sites
        )
        
        # call the coarse-graining method.
        method_func(self, **kwargs)

        # connect the beads.
        self._connect_beads()


    def get_beads(self,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'},
        masses: Dict[str, float] = None,
        force: bool = False
    ) -> List[Bead]:
        """
        Gather beads and assign species according to the mapping.
        """

        if self._beads is not None and not force:
            return self._beads

        def default_factory() -> int:
            """ Start the defaultdict counter at 1. """
            return 1
        
        all_beads = {}
        number_of_species = defaultdict(default_factory)
        for bead_mol_id, (label, cluster) in enumerate(self._atomic_clusters.items(), 1):
            beads = cluster.beads_frac_coords
            bead_ids = cluster.bead_ids
            add_species = bead_type_to_element_mapping[label[0]]

            # make sure the correct number of species are assigned based on the 
            # supplied mapping.
            if isinstance(add_species, str):
                add_species = [add_species] * len(beads)
            elif len(add_species) < len(beads):
                add_species *= len(beads) // len(add_species) + 1
            else:
                add_species = add_species[:len(beads)]
            for species, frac_coord, bead_id in zip(add_species, beads, bead_ids):
                all_beads[bead_id] = Bead(
                    species, 
                    number_of_species[species], 
                    bead_id, 
                    bead_mol_id,
                    frac_coord
                )
                number_of_species[species] += 1

        # store the beads.
        self._beads = all_beads

        return all_beads

    
    def overlay_cg_atomistic_representations(self,
        filename: str = None,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'Ce'},
        crystal_toolkit: bool = False
    ):
        """
        Overlays the atomistic representation ontop of the coarse-grained 
        representation of the struture. This is useful for visualising the 
        atomistic structure in the context of the coarse-grained structure (and 
        that the coarse-graining was correct).
        """

        # get a copy of the atomistic structure.
        overlay_structure = self.copy()

        # gather the beads.
        if self._beads is None:
            self.get_beads(bead_type_to_element_mapping)

        # gather the coarse-grained beads and append them to the structure.
        for bead in self._beads.values():
            overlay_structure.append(bead.species, bead.frac_coord)
            
        # write the structure to file.
        if filename is not None:
            overlay_structure.to(str(filename))

        # display the structure.
        if crystal_toolkit:
            crystal_toolkit_display(overlay_structure)


    def net_to_cif(self, 
        filename: str, 
        write_bonds: bool = True,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'},
        name: str = 'net'
    ) -> None:
        """
        Write the coarse-grained structure to a (Topo)CIF file.

        :param filename: path to CIF to write.
        :param write_bonds: whether to write bonds to the CIF.
        :param net_name: name of the network.
        """
        TopoCifWriter(self, 
            self.get_beads(bead_type_to_element_mapping, force=True), 
            self._bead_bonds, name
        ).write_file(filename, write_bonds)


    def net_to_lammps_data(self,
        filename: str,
        write_bonds: bool = True,
        two_way_bonds: bool = False,
        atom_style: str = 'full',
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'},
        name: str = 'net',
    ) -> None:
        """
        Write the coarse-grained structure to a LAMMPS data file.

        :param filename: path to LAMMPS data file to write.
        :param write_bonds: whether to write bonds to the LAMMPS data file.
        :param two_way_bonds: whether to write bonds in both directions.
        :param atom_style: atom style to use. Either 'full' or 'atomic'.
        :param net_name: name of the network.
        :return: None.
        """
        # we force the get_beads() method to recompute so that the correct
        # mapping is used.
        LammpsDataWriter(self, 
            self.get_beads(bead_type_to_element_mapping, force=True),
            self._bead_bonds, name
        ).write_file(
            filename, 
            write_bonds=write_bonds, 
            atom_style=atom_style, 
            two_way_bonds=two_way_bonds
        )

    
    def net_to_lammps_dump(self,
        filename: str,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'},
        **kwargs
    ):
        """
        """
        LammpsDumpWriter(self,
            self.get_beads(bead_type_to_element_mapping, force=True),
            **kwargs
        ).write_file(filename)


    def net_to_struct(self, 
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'}
    ) -> PymatgenStructure:
        """
        Convert the coarse-grained structure to a pymatgen Structure object.
        """

        # gather the beads.
        if self._beads is None:
            self.get_beads(bead_type_to_element_mapping)
        
        # gather into a list of PeriodicSite objects.
        sites = [
            PeriodicSite(bead.species, bead.frac_coord, self.lattice)
            for bead in self._beads.values()
        ]
    
        return PymatgenStructure.from_sites(sites)
    

    def net_to_ase_atoms(self) -> Atoms:
        """
        Convert the coarse-grained structure to an ASE atoms object.
        """
        
        # first gather beads into a PymatgenStructure object.
        struct = self.net_to_struct()

        # then use the Pymatgen ASE interface to convert to an ASE atoms object.
        return AseAtomsAdaptor.get_atoms(struct)


    def net_to(self, filename, fmt='', **kwargs) -> None:
        """
        Use Pymatgen's writer to write the coarse-grained structure to file.

        :param filename: path to file to write.
        :param fmt: format to write to. If not specified, will be inferred from
            the file extension.
        :param kwargs: keyword arguments to pass to the writer.
        :return: None.
        """
        self.net_to_struct().to(filename, fmt, **kwargs)

    
    def net_to_ase_to(self, filename, fmt=None, **kwargs) -> None:
        """
        Use ASE's writer to write the coarse-grained structure to file.

        :param filename: path to file to write.
        :param fmt: format to write to. If not specified, will be inferred from
            the file extension.
        :param kwargs: keyword arguments to pass to the writer.
        :return: None.
        """
        self.net_to_ase_atoms().write(filename, format=fmt, **kwargs)

    
    def net_to_gulp_framework_nodes(self, filename: str, **kwargs) -> None:
        """
        Reduce the coarse-grained AB2 net to the bare framework consisting of 
        the A nodes only. Then write to a GULP input file.
        """

        # check first that the neighbourlist has been computed, atomic clusters
        # have been identified, and the coarse-grained net has been generated.
        if self._neighbour_list is None:
            raise ValueError('Neighbourlist not computed. Please run ' \
                'get_neighbours_crystalnn() first.')
        if self._atomic_clusters is None:
            raise ValueError('Atomic clusters not identified. Please run ' \
                'find_atomic_clusters() first.')
        if self._beads is None:
            raise ValueError('Coarse-grained net not generated. Please run ' \
                'get_coarse_grained_net() first.')
        
        # we cam now proceed to reduce the net to the framework nodes and write
        # to a GULP input file.
        _gulp_kwargs = {
            'sub_elem': 'Si',
            'rattle': 0.0,
            'k_bond': 200.0,
            'k_angle': 10.0,
            'rtol': 2.0
        }
        _gulp_kwargs.update(kwargs)
        gulp_writer = GulpWriter(self, **_gulp_kwargs)
        gulp_writer.write_file(filename)

    
    def pickle_neighbourlist(self, filename: str):
        """
        Write the current computed neighbour list to a pickle-d file.

        :param filename: name of file to write.
        """
        if not self._neighbour_list:
            raise 


    def _get_adjacency_matrix(self):
        """
        Compute a global adjacency matrix for all intra-unit bonds.
        """
        data = []
        i_indices = []
        j_indices = []

        for site_index, site_neighbours in self._neighbour_list.items():

            this_site = self[site_index]
            this_site_type = self.get_site_type_index(this_site.specie.symbol)

            for neighbour in site_neighbours:

                neighbour_site = neighbour['site']
                neighbour_site_type = self.get_site_type_index(
                    neighbour_site.specie.symbol
                )

                if this_site_type != neighbour_site_type:
                    continue
                
                site_type = self._all_sites[this_site_type]                
                if self._check_neighbour_conditions('intra', neighbour, site_type):
                    data.append(1)
                    i_indices.append(site_index)
                    j_indices.append(neighbour['site_index'])
                else:
                    site_neighbours.remove(neighbour)

        adjacency_matrix = csr_matrix(
            (data, (i_indices, j_indices)), 
            shape=(self.num_sites, self.num_sites)
        ).maximum(
            csr_matrix((data, (j_indices, i_indices)), 
            shape=(self.num_sites, self.num_sites))
        )

        return adjacency_matrix
    

    def _identify_clusters(self, adjacency_matrix):
        """
        Identify clusters within the structure by decomposiong the adjacency
        matrix.
        """
        _, labels = csgraph.connected_components(
            adjacency_matrix, directed=False
        )
        clusters = defaultdict(list)
        for site_index, label in enumerate(labels):
            clusters[label].append(site_index)
        return clusters


    def _check_neighbour_conditions(self, 
        intra_or_inter: str, 
        neighbour: dict, 
        site_type: str
    ):
        """
        Check conditions for a neighbouring site in the cluster.

        This function checks whether the weight and bond length of a 
        neighbouring site satisfy certain conditions that depend on whether the
        neighbour is considered 'intra' or 'inter'. The weight and bond length 
        conditions are determined by the specified 'site_type'.

        :param intra_or_inter: Either 'intra' or 'inter' indicating the type of 
            neighbour.
        :param neighbour: Dictionary containing 'weight' and 'site' (which 
            includes 'nn_distance' for bond length) of the neighbouring site.
        :param site_type: The type of the site which determines the conditions 
            to be checked.
        :returns: True if all conditions are satisfied, False otherwise.
        """
        if intra_or_inter not in ['intra', 'inter']:
            raise ValueError(f"Unknown intra_or_inter: {intra_or_inter}")

        weight = neighbour['weight']
        bond_length = neighbour['site'].nn_distance

        min_length_value = getattr(self,
            f"_min_{intra_or_inter}_bond_length")[site_type]
        max_length_value = getattr(self,
            f"_max_{intra_or_inter}_bond_length")[site_type]

        return (
            weight >= getattr(self, f"_min_{intra_or_inter}_weight")[site_type] 
            and (min_length_value is None or bond_length >= min_length_value) 
            and (max_length_value is None or bond_length <= max_length_value)
        )


    def _prepare_atomic_cluster(self, 
        site_indices_set,
        site_type,
    ):
        """
        """

        images = defaultdict(list)
        images[next(iter(site_indices_set))] = np.array([0,0,0], dtype=np.uint)

        visited = set()
        stack = [next(iter(site_indices_set))]

        # calculate all neighbors upfront
        cluster_neighbors = {
            site_index: [
                neighbour for neighbour in self._neighbour_list[site_index]
            if neighbour['site_index'] in site_indices_set
            and self._check_neighbour_conditions('intra', neighbour, site_type)
            ] 
            for site_index in site_indices_set
        }

        # calculate all external neighbors.
        external_neighbors = {
            site_index: [
                neighbor for neighbor in self._neighbour_list[site_index] 
            if neighbor['site_index'] not in site_indices_set
            and self._check_neighbour_conditions('inter', neighbor, site_type)
            ] 
            for site_index in site_indices_set
        }

        # get image consistency.
        while stack:
            site_index = stack.pop()
            visited.add(site_index)
            for neighbor in cluster_neighbors[site_index]:
                neighbor_index = neighbor['site_index']
                neighbor_image = neighbor['image']
                relative_image = neighbor_image + images[site_index]
                if neighbor_index not in visited and neighbor_index not in stack:
                    stack.append(neighbor_index)
                    images[neighbor_index] = relative_image.astype(int)

        # gather all properties of the cluster.
        sites = [self[i] for i in site_indices_set]
        species = [site.specie for site in sites]
        frac_coords = np.array([site.frac_coords for site in sites])
        sorted_images = np.array([images[i] for i in site_indices_set])
        consistent_frac_coords = frac_coords + sorted_images
        consistent_cart_coords = self.lattice.get_cartesian_coords(
            consistent_frac_coords
        )

        # gather the unique edges and weights from the cluster.
        edges = {}
        for site_index, neighbours in cluster_neighbors.items():
            for neighbour in neighbours:
                neighbour_index = neighbour['site_index']
                if neighbour_index < site_index:
                    continue
                edge = (site_index, neighbour_index)
                if edge not in edges:
                    edges[edge] = {'weight': neighbour['weight']}

        # now add the external edges.
        edges_external = {}
        for site_index, neighbours in external_neighbors.items():
            for neighbour in neighbours:

                # make sure the external site image is corrected for the atomic 
                # cluster image.
                raw_image = np.array(neighbour['image'], dtype=int)
                image = raw_image + images[site_index]
                edge = (site_index, neighbour['site_index'])
                if edge not in edges_external:
                    edges_external[edge] = {
                        'image': raw_image,
                        'weight': neighbour['weight']
                    }

        cluster = AtomicCluster(
            site_indices_set,
            species,
            consistent_cart_coords,
            sorted_images,
            edges,
            edges_external
        )

        return cluster
    

    def _finalise_clusters(self, clusters):
        """
        """

        final_clusters = {}
        element_to_cluster = {}
        site_type_counts = np.ones(len(self._site_types), dtype=np.uint)
        for site_indices in clusters.values():

            site_type = np.unique([
                self.get_site_type_index(self[i].specie.symbol) 
                for i in site_indices
            ])

            assert len(site_type) == 1, 'Multiple site types in cluster.'
            site_type_index = site_type[0]
            site_type = self._all_sites[site_type_index]

            # hence convert to AtomicCluster object.
            cluster = self._prepare_atomic_cluster(site_indices, site_type)
            number = site_type_counts[site_type_index]
            final_clusters[site_type, number] = cluster

            # store the cluster index for each site for quick lookup later.
            for site_index in site_indices:
                element_to_cluster[site_index] = (site_type, number)

            # update counter.
            site_type_counts[site_type_index] += 1

        self._atomic_clusters = final_clusters
        self._element_to_cluster = element_to_cluster

    
    def _connect_clusters(self):
        """
        Connects the clusters in the structure after all of the clusters have 
        been determined. It updates each cluster's list of bound clusters, 
        including the cluster label it's bound to, the site index, the image, 
        and the weight.

        Note: This function should be called only after all atomic clusters have 
            been determined.
        """
        for cluster in self._atomic_clusters.values():
            for edge, info in cluster._edges_external.items():
                info['cluster'] = self._element_to_cluster[edge[1]]

    
    def _wrap_cart_coords_to_frac(self,
        cart_coords: np.ndarray,
        precision: float = 1e-8,
    ) -> Tuple[np.ndarray]:
        """
        Wrap cartesian coordinates to fractional coordinates.

        :param cart_coords: Cartesian coordinates to wrap.
        :param precision: Precision to round to.
        :return: Fractional coordinates and image.
        """
        frac_unwrapped = round_to_precision(
            self.lattice.get_fractional_coords(cart_coords), precision
        ) + 0.0
        frac_wrapped = frac_unwrapped % 1.0
        image = (frac_unwrapped - frac_wrapped).astype(int)
        return frac_wrapped, image


    def _compute_bead_distances(self,
        bead1: np.ndarray,
        bead2: np.ndarray,
        image: np.ndarray,
    ) -> float:
        """
        Compute the distance between two beads.

        :param bead1: Information for bead1.
        :param bead2: Information for bead2.
        :param image: Image of bead2.
        :return: Distance between the two beads.
        """
        cart_coords = self.lattice.get_cartesian_coords(
            [bead1['bead_frac_coords'], bead2['bead_frac_coords']+image]
        )
        return np.linalg.norm(cart_coords[0] - cart_coords[1])

    
    def _update_bead_neighbourlist(self, 
        bead1_id: int, 
        bead2_id: int, 
        image: Tuple[int, int, int],
        weight: float, 
        distance: float
    ) -> None:
        """
        Update the bead neighbourlist.

        :param bead1_id: Bead ID of bead1.
        :param bead2_id: Bead ID of bead2.
        :param image: Image of bead2.
        :param weight: Weight of the bond.
        :param distance: Distance between the two beads.
        """
        bead1_list = self._bead_neighbour_list.get(bead1_id, [])

        # create a set of tuples for existing (bead_id, image) pairs
        existing_pairs = set((d['bead_id'], d['image']) for d in bead1_list)

        # create a tuple for the new pair
        new_pair = (bead2_id, image)

        # append the dictionary only if the pair doesn't exist already
        if new_pair not in existing_pairs:
             self._bead_neighbour_list[bead1_id] = bead1_list + [{
                'bead_id': bead2_id,
                'image': image,
                'weight': weight,
                'bead_distance': distance
            }]

    
    def _update_bead_bonds(self,
        bead1_id,
        bead2_id,
        image,
        weight,
        distance,
        bond_type: str = 'inter'
    ):
        """
        Update the bead bonds.
        """

        edge = (bead1_id, bead2_id)
        existing_images = set((d['image']) for d in self._bead_bonds[edge])
        if image not in existing_images:
            self._bead_bonds[edge].append({
                'image': image,
                'weight': weight,
                'bead_distance': distance,
                'bond_type': bond_type
            })


    def _connect_beads(self):
        """
        Connect the beads in the structure.
        """

        # now beads have been placed, connect up the clusters.
        for local_cluster in self._atomic_clusters.values():

            for (internal_site, external_site), edge_info in local_cluster._edges_external.items():

                bound_cluster = self._atomic_clusters[edge_info['cluster']]

                # atom images involved in bond.
                local_cluster_atom_image = local_cluster.get_image_by_site_index(
                    internal_site
                )
                bound_cluster_atom_image = bound_cluster.get_image_by_site_index(
                    external_site
                )

                # get the local and bound cluster bead indices, bead numbers, and images.
                local_cluster_bead_info = _get_cluster_bead_info(
                    local_cluster, internal_site
                )
                bound_cluster_bead_info = _get_cluster_bead_info(
                    bound_cluster, external_site
                )

                # now determine the correct periodic images for all unique combinations
                # of beads.
                for bead1, bead2 in product(
                    local_cluster_bead_info, 
                    bound_cluster_bead_info
                ):
                    bead2_img = _determine_second_bead_image(
                        bead1, 
                        bead2, 
                        local_cluster_atom_image, 
                        bound_cluster_atom_image, 
                        edge_info
                    )

                    # now order the beads by their bead_ids.
                    bead1, bead2, image = _order_beads(bead1, bead2, bead2_img)
                    bead_distance = self._compute_bead_distances(
                        bead1, bead2, image
                    )

                    # now add the bond to the dictionary if not already there.
                    self._update_bead_neighbourlist(
                        bead1['bead_ids'], 
                        bead2['bead_ids'], 
                        tuple(image), 
                        edge_info['weight'], 
                        bead_distance
                    )

                    self._update_bead_neighbourlist(
                        bead2['bead_ids'], 
                        bead1['bead_ids'], 
                        tuple((-image).astype(int)), 
                        edge_info['weight'], 
                        bead_distance
                    )

                    self._update_bead_bonds(
                        bead1['bead_ids'],
                        bead2['bead_ids'],
                        tuple(image),
                        edge_info['weight'],
                        bead_distance,
                        bond_type='inter'
                    )

    
    @coarse_graining_methods.register('centroid', 'single')
    def _centroid_method(self,
        skip_elements: List[str] = None, 
        precision: float = 1e-8,
    ):
        """
        Place a single bead at the geometric centre of each AtomicCluster. This 
        does not define additional bonds between beads so all external bonds are 
        bound to this one site. Accordingly, we forward all external connections
        to any atom in this cluster directly to the bead.
        """

        # get the centroid of each cluster and assign it as the single bead.
        bead_count = 1
        for cluster in self._atomic_clusters.values():

            # first assign the beads.
            centroid = [cluster.get_centroid(skip_elements=skip_elements)]

            # assign the values to the cluster.
            cluster.assign_beads(
                [bead_count],
                *self._wrap_cart_coords_to_frac(centroid, precision=precision),
                {i: [0] for i in cluster.site_indices},
                internal_bead_bonds=[]
            )
            
            # update the bead count.
            bead_count += 1


    @coarse_graining_methods.register('shortest path', 'single')
    def _shortest_path_method(self):
        raise NotImplementedError('Shortest path method not implemented yet.')

"""
12.06.23
@tcnicholas
The main Structure class for coarse-graining inorganic and framework materials.
"""


import os
import pickle
import inspect
import hashlib
import warnings
import importlib
import multiprocessing
from pathlib import Path
from functools import partial
from collections import namedtuple
from itertools import chain, product
from collections import ChainMap, defaultdict
from typing import List, Union, Dict, Tuple, Optional

import numpy as np
from ase import Atoms
from ase.build import sort

from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import PeriodicNeighbor
from pymatgen.core import Structure as PymatgenStructure

from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import fcluster, linkage

from .cif import read_cif, TopoCifWriter, build_neighbor_list_from_topocif
from .sort_sites import sort_sites, create_site_type_lookup
from .gulp import GulpWriter
from .bonds import Bonding
from .net import Net
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
    Colours,
    crystal_toolkit_display,
    get_first_n_letters,
    site_type_to_index,
    remove_non_letters,
    round_to_precision,
    parse_arguments,
    get_nn_dict,
    get_symbol,
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
        allow_pickle: bool = False,
        **kwargs
    ) -> None:
        """
        Initialise a Structure object.

        I separate the precomputed neighbour list from the neighbour list
        generated during the coarse-graining process. This is because we can
        then update the "guess" neighbour list with the "true" neighbour list
        from the file. 

        Arguments:
            site_types: List of lists of element symbols, sorted by
                sort_sites_method.
            sort_sites_method: Method to use for sorting sites. Either 'mof' or 
                'alphabetical'.
            precomputed_atomic_clusters: Dictionary of precomputed atomic
                clusters. Each key corresponds to a site type and each value is
                an AtomicCluster object.
            precomputed_neighbour_list: Dictionary of precomputed neighbour
                lists. Each key corresponds to a site index and each value is a
                list of dictionaries containing the nearest neighbours of that
                site. Each dictionary contains the keys 'site', 'image' and
                'weight' and 'site_index'. If you want to assign your own
                neighbour list, you can do so by setting these properties
            cores: Number of cores to use for parallel operations.
            verbose: Whether to instantiate the new chic classes with timing.
        """
        super().__init__(*args, **kwargs)
        self.register_methods()
        if site_types is None:
            self._site_types = sort_sites(self, sort_sites_method)
        else:
            self._site_types = site_types
        self._site_type_lookup = create_site_type_lookup(self._site_types)
        self._all_sites = get_first_n_letters(len(self._site_types))

        # we should now delete any species that are not in the site types.
        species_to_remove = {
            x.symbol for x in set(self.species)
        } - set(chain(*self._site_types))

        for species in species_to_remove:
            self.remove_sites_by_symbol(species)

        self._neighbour_list = None
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
        self._bead2cluster = defaultdict(str)
        
        # whether to default pickle.
        self.__pickle_dir__ = None
        self.__allow_pickle__ = allow_pickle
        self.__setup_pickle__()


    @property
    def neighbour_list(self) -> Dict[int, list]:
        """
        Return the computed neighbour list.
        """
        return self._neighbour_list
    

    @property
    def atomic_clusters(self) -> Dict[str, AtomicCluster]:
        """
        Return the determined atomic clusters.
        """
        return self._atomic_clusters
    

    def remove_sites_by_symbol(self, symbol: str) -> None:
        """
        """
        ix = np.where(np.array([x.specie.symbol for x in self])==symbol)[0]
        self.remove_sites(ix)


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

        Arguments:
            element: Element symbol.
            rmin: Minimum distance between sites to be averaged.
            rmax: Maximum distance between sites to be averaged.
            cluster_method: Clustering method to use. Either 'dbscan' or
                'hierarchical'.
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
        cn_equals_one: List[str] = ['H', 'F', 'Cl'],
        cores: int = None,
        reset: bool = False,
        **kwargs
    ) -> None:
        """
        Compute neighbourlist using CrystalNN.

        This is currently the bottleneck in this code. It would be worth
        investigating whether this can be sped up.

        Arguments:
            cn_equals_one: List of elements for which the coordination number
                should be set to 1 and therefore we can ignore from the
                neighbourlist search.

            cores: Number of cores to use for parallel operations. If None,
                use the number of cores specified in the constructor (default =
                1). If specified, will overwrite the number of cores specified
                in the constructor.

            **kwargs:
                weighted_cn – (bool) if set to True, will return fractional
                weights for each potential near neighbor.

                cation_anion – (bool) if set True, will restrict bonding targets
                    to sites with opposite or zero charge. Requires an oxidation 
                    states on all sites in the structure.

                distance_cutoffs – ([float, float]) - if not None, penalizes 
                    neighbor distances greater than sum of covalent radii plus
                    distance_cutoffs[0]. Distances greater than covalent radii 
                    sum plus distance_cutoffs[1] are enforced to zero weight.

                x_diff_weight – (float) - if multiple types of neighbor elements 
                    are possible, this sets preferences for targets with higher
                    electronegativity difference.

                porous_adjustment – (bool) - if True, readjusts Voronoi weights
                    to better describe layered / porous structures

                search_cutoff – (float) cutoff in Angstroms for initial neighbor
                    search; this will be adjusted if needed internally

                fingerprint_length – (int) if a fixed_length CN “fingerprint” is
                    desired from get_nn_data(), set this parameter
        """

        if reset:
            self._neighbour_list = False
        
        if self._neighbour_list is not None:
            return

        # parse the number of cores.
        cores = cores or self._cores

        # set the CrystalNN kwargs.
        cnn_kwargs = {
            'weighted_cn': True,
            'cation_anion': False,
            'distance_cutoffs': (0.5, 1),
            'x_diff_weight': 0,
            'porous_adjustment': True,
            'search_cutoff': 10.0,
            'fingerprint_length': None
        }
        cnn_kwargs.update(kwargs)

        # compute the neighbourlist. we hide the warnings because CrystalNN
        # throws a lot of them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cnn = CrystalNN(**cnn_kwargs)
            elements = list(chain(*self._site_types))

            # get the indices of the sites to compute the neighbourlist for.
            site_indices = [
                i for i,a in enumerate(self) if a.specie.symbol in elements
                and a.specie.symbol not in cn_equals_one
            ]

            # compute the neighbourlist in parallel.
            with multiprocessing.Pool(processes=cores) as pool:
                nlist = pool.map(partial(
                    get_nn_dict, cnn, self),
                    np.array_split(site_indices, cores)
                )
        
        # retrieve the neighbourlist.
        self._neighbour_list = dict(ChainMap(*nlist))

        # we can now update the neighbour list to add the sites with CN=1 by
        # iterating over all of the determined neighbours and adding the 
        # neighbours of the neighbours. first let's add empty lists for the
        # sites with CN=1.
        missed_sites = [
            i for i,a in enumerate(self) if a.specie.symbol in cn_equals_one
        ]
        for site_index in missed_sites:
            self._neighbour_list.setdefault(site_index, [])

        # now infer the neighbours of these sites by iterating over the
        # neighbourlist. we prepare a dictionary to store the neighbours.
        NeighbourData = namedtuple(
            'NeighbourData', 'site image weight site_index'
        )

        # hence gather missed neighbours.
        for site_index, neighbours in self._neighbour_list.items():
            current_site = self[site_index]
            
            for neighbour in neighbours:

                # Get the PeriodicNeighbor object for this neighbour.
                neighbour_site = neighbour['site']
                
                # Check if the current neighbour is one of the specified elements
                if neighbour_site.specie.symbol in cn_equals_one:

                    # Calculate the negative image array.
                    neg_image = -np.array(neighbour['image'])
                    
                    # Create a PeriodicNeighbor object for the site with index 
                    # 'site_index'
                    new_neighbour = NeighbourData(
                        site=PeriodicNeighbor(
                            species=current_site.species,
                            coords=current_site.frac_coords,
                            lattice=self.lattice,
                            properties=None,
                            nn_distance=neighbour_site.nn_distance,
                            index=site_index,
                            image=neg_image
                        ),
                        image=neg_image,
                        weight=neighbour['weight'],
                        site_index=site_index
                    )
                    
                    # Append the new neighbour information to the neighbour list 
                    # of the current neighbour's site
                    self._neighbour_list[neighbour['site_index']].append(
                        new_neighbour._asdict()
                    )
        
        if self.__allow_pickle__:
            with open(self.__pickle_dir__ / 'nl.pickle', 'wb') as f:
                pickle.dump(self._neighbour_list, f, pickle.HIGHEST_PROTOCOL)


    def get_neighbours_by_cutoff(self, 
        rcut: float = 1.8,
        by_element_pair: Dict[Tuple[str, str], Union[float, Tuple[float, float]]] = None
    ) -> None:
        """
        Compute the neighbourlist with potentially different radial cut-offs for 
        different element pairs. The cutoffs for specific pairs can be defined 
        in by_element_pair dictionary with tuples of element symbols as keys and 
        values as either a single max cutoff or a tuple of 
        (min_cutoff, max_cutoff).

        Arguments:
            rcut: a single global cut-off to use for all pairs of atoms. If 
                per-atom-pair cut-offs are set to larger than this value, it
                will be overwritten.
            by_element_pair: specific atom pair distances. Can either be a float
                indicating a cut-off for that pair, or a tuple of floats, which
                act as a distance window (rmin, rmax).
        """

        if by_element_pair is None:
            by_element_pair = {}

        # Normalize the cutoff values in the dictionary
        normalized_by_element_pair = {}
        for key, value in by_element_pair.items():
            if isinstance(value, tuple):
                rmin, rmax = value if len(value) == 2 else (0, value[0])
            else:
                rmin, rmax = 0, value
            normalized_by_element_pair[tuple(sorted(key))] = (rmin, rmax)

        # Determine the global maximum rcut necessary for neighbor search
        max_rcut = max(
            [rcut] + [pair[1] for pair in normalized_by_element_pair.values()]
        )

        # Use the adjusted global rcut to compute initial neighbors.
        ns = self.get_all_neighbors(max_rcut)
        nl = defaultdict(list)
        for i, neighbours in enumerate(ns):
            this_symbol = get_symbol(self[i])
            for neighbour in neighbours:
                other_symbol = get_symbol(self[neighbour.index])
                pair = tuple(sorted((this_symbol, other_symbol)))
                pair_cuts = normalized_by_element_pair.get(pair, (0, rcut))

                # Only include the neighbour if the distance is within the specified 
                # cutoff range.
                if pair_cuts[0] <= neighbour.nn_distance <= pair_cuts[1]:
                    nl[i].append({
                        'site': neighbour,
                        'image': tuple([int(x) for x in neighbour.image]),
                        'weight': 1.0,
                        'site_index': neighbour.index
                    })

        # Sort the dictionary by atom index.
        self._neighbour_list = dict(sorted(nl.items()))


    @classmethod
    def from_structure(cls, 
        structure: PymatgenStructure, 
        atomic_clusters=None,
        **kwargs
    ) -> 'Structure':
        """
        Create a new Structure object from an existing pymatgen Structure.

        Arguments:
            structure: An existing pymatgen Structure object.
            atomic_clusters: Dictionary of precomputed atomic clusters. Each
                key corresponds to a site type and each value is an
                AtomicCluster object.

        Returns:
            A new Structure object.
        """
        instance = cls(
            structure.lattice, 
            structure.species, 
            structure.frac_coords, 
            validate_proximity=False,
            site_properties=structure.site_properties,
            **kwargs
        )
        if atomic_clusters:
            instance._atomic_clusters = atomic_clusters
        return instance

    
    @classmethod
    def from_ase_atoms(cls, atoms: Atoms) -> 'Structure':
        """
        Create a new Structure object from an ASE Atoms object.

        Arguments:
            atoms: An ASE Atoms object.

        Returns:
            A new Structure object.
        """
        structure = AseAtomsAdaptor.get_structure(atoms)
        return cls.from_structure(structure)

    
    @classmethod
    def from_cif(cls, 
        filename: str, 
        **kwargs
    ) -> 'Structure':
        """
        Read structures from CIF.

        Arguments:
            filename: Path to CIF file.

        Returns:
            Structure object.
        """

        read_cif_kwargs = {
            k:v for k, v in kwargs.items() 
            if k in inspect.signature(read_cif).parameters and k != 'self'
        }
        cls_kwargs = {
            k:v for k, v in kwargs.items()
            if k in inspect.signature(cls.__init__).parameters and k != 'self'
        }

        # read the CIF and attempt to extract any bonding information. this will
        # be used to construct the neighbourlist if found. this requires that
        # the CIF is in TopoCIF format and in the P1 space group.
        struct, bonding = read_cif(filename, **read_cif_kwargs)
        if bonding is not None:
            neighbour_list = build_neighbor_list_from_topocif(
                struct, bonding._bonds
            )
        else:
            neighbour_list = None
        
        return cls.from_structure(
            struct,
            **cls_kwargs,
            precomputed_neighbour_list=neighbour_list
        )
    

    @classmethod
    def from_lammps_data(cls, filename: str, **kwargs) -> 'Structure':
        """
        Read structures from LAMMPS data file.

        Arguments:
            filename: Path to LAMMPS data file.
        
        Returns:
            Structure object.
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
        filename: str, 
        intramolecular_cutoff: float = 2.0,
        start: int = 0,
        end: int = None,
        step: int = 1,
        verbose: bool = False,
        gather_columns = None
    ):
        """
        Append a LAMMPS trajectory file to the current structure object. For 
        each cluster, the atomic clusters are updated to reflect the snapshot's 
        structure. Only snapshots in the given range are returned.

        Arguments:
            filename: Path to the LAMMPS trajectory file.
            intramolecular_cutoff: Cutoff for determining atomic clusters.
            start: Starting snapshot index.
            end: Ending snapshot index. If None, process full trajectory.
            step: Step size between snapshots.
            verbose: Whether to instantiate the new chic class with timing.

        Returns:
            Generator of Structure objects.
        """

        # parse the dump file one frame at a time.
        snapshots = process_dump_file(filename, start, end, step, gather_columns)
        for count, lattice, frac_coords, extra_data in snapshots:

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
                PymatgenStructure(lattice, self.species, frac_coords, site_properties=extra_data),
                atomic_clusters=atomic_clusters, verbose=verbose
            )
            
            # add any extra data collected from the file.
            #for label, data in extra_data.items():
            #    snapshot_structure.site_properties[label] = data

            yield count, snapshot_structure
        
    
    @timer
    def find_atomic_clusters(self,
        skip_sites: List[str] = None,
        min_intra_weight: Union[float, List[float], Dict[str, float]] = None,
        min_intra_bond_length: Union[float,List[float], Dict[str,float]] = None,
        max_intra_bond_length: Union[float,List[float], Dict[str,float]] = None,
        min_inter_weight: Union[float, List[float], Dict[str, float]] = None,
        min_inter_bond_length: Union[float,List[float], Dict[str,float]] = None,
        max_inter_bond_length: Union[float,List[float], Dict[str,float]] = None,
        allow_same_type_neighbours: bool = False
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

        Arguments:
            skip_sites: List of site types to skip when identifying building
                units. For example, if you want to skip the A sites and instead
                keep them as single atoms, set skip_sites = 'a'. You can skip
                multiple sites by passing a list of site types, e.g.
                skip_sites = ['a', 'b'].
            min_intra_weight: Minimum weight required intra-unit edges.
            min_intra_bond_length: Minimum bond length for intra-unit edges.
            max_intra_bond_length: Maximum bond length for intra-unit edges.
            min_inter_weight: Minimum weight required inter-unit edges.
            min_inter_bond_length: Minimum bond length for inter-unit edges.
            max_inter_bond_length: Maximum bond length for inter-unit edges.
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
            2.5, max_intra_bond_length, self._all_sites
        )
        self._min_inter_weight = parse_arguments(
            0.4, min_inter_weight, self._all_sites
        )
        self._min_inter_bond_length = parse_arguments(
            0.0, min_inter_bond_length, self._all_sites
        )
        self._max_inter_bond_length = parse_arguments(
            3.0, max_inter_bond_length, self._all_sites
        )

        # compute adjacency matrix using the neighbourlist and any constraints.
        adjacency_matrix = self._get_adjacency_matrix()

        # identify clusters
        clusters = self._identify_clusters(adjacency_matrix)

        # relabel clusters based on site types and cache them.
        self._finalise_clusters(clusters, allow_same_type_neighbours)

        # connect the clusters.
        self._connect_clusters()

    
    def find_metal_clusters(self, elements: Optional[List[str]] = None) -> None:
        """
        Find metal clusters in structure. This assumes that the standard single-node
        approach has been used to find atomic clusters, such that isolated metal
        atoms and isolated oxygen atoms are left after the organic linkers have been
        found.

        Args:
            elements: List of elements to consider as metal. If None, elements from 
                A-type sites plus oxygen are used.

        Returns:
            List of metal clusters.
        """
        metal_clusters = []
        if elements is None:
            elements = self._site_types[0] + ['O']

        # gather appropriate atomic clusters to map to metal clusters. this will be
        # all clusters that contain the metal element (all a-type sites) plus all
        # clusters that have lone oxygens atoms only.
        labels_to_remove = []
        clusters_to_map = []
        for label, cluster in self.atomic_clusters.items():

            if any([Element(element) in cluster._species for element in elements]) and label[0] == 'a':
                clusters_to_map.append(cluster)
                labels_to_remove.append(label)

            elif len(cluster._species) == 1 and Element('O') in cluster._species:
                
                # this is a lone oxygen. however we also need not include oxygen
                # atoms that are not bonded to anything.
                if len(cluster._edges_external) == 0:
                    continue

                clusters_to_map.append(cluster)
                labels_to_remove.append(label)

        # extract all site indices from clusters to map.
        site_indices = []
        for cluster in clusters_to_map:
            site_indices += cluster._site_indices

        # build adjancey matrix for these sites.
        data = []
        i_indices = []
        j_indices = []
        for site_index in site_indices:
            for neighbour in self._neighbour_list[site_index]:
                if neighbour['site_index'] in site_indices:
                    data.append(1)
                    i_indices.append(site_index)
                    j_indices.append(neighbour['site_index'])

        # Build adjacency matrix for these sites.
        num_sites = max(site_indices) + 1
        adjacency_matrix = csr_matrix(
            (data, (i_indices, j_indices)), 
            shape=(num_sites, num_sites)
        )

        # Since the graph is undirected, ensure the adjacency matrix is symmetric
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T - csr_matrix.diagonal(adjacency_matrix)

        # Find connected components
        _, labels = connected_components(
            csgraph=adjacency_matrix, 
            directed=False, 
            return_labels=True
        )

        # Generate clusters based on connected components labels
        clusters = defaultdict(list)
        for site_index, label in enumerate(labels):
            if site_index in site_indices:
                clusters[label].append(site_index)

        # convert clusters to AtomicCluster objects.
        for label, cluster_site_indices in clusters.items():
            cluster = self._prepare_atomic_cluster(cluster_site_indices, 'a')
            metal_clusters.append(cluster)

        # now we can remove the original clusters from the structure and replace
        # them with the metal clusters.
        for label in labels_to_remove:
            self.atomic_clusters.pop(label)

        # find how many a-type sites are left, and renumber them.
        a_sites = [
            cluster for label,cluster in self.atomic_clusters.items() if label[0] == 'a'
        ]

        # hence re-add the metal clusters.
        for i,cluster in enumerate(metal_clusters + a_sites, 1):
            self.atomic_clusters[('a', i)] = cluster

        # then re-connect the clusters.
        self._element_to_cluster = {}
        for label,cluster in self.atomic_clusters.items():
            for site_index in cluster._site_indices:
                self._element_to_cluster[site_index] = label
        
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

        1. For each atomic cluster, place the cg bead(s).

        2. For each atomic cluster, find the neighbours of the cg bead(s),
                alongside the correct periodic image (by tracing along the 
                connected atomistic sites). From here, it will be possible to 
                fully define the cg network.

        Supported methods can be found by calling:

            Structure.coarse_graining_methods.available_methods().

        ---

        Note to developers of coarse-graining methods:

        Methods can be implemented in two ways. Firstly, by building on the main 
        Structure class by adding a class method and assigning the 
            
            @coarse_graining_methods.register(<method/name>, <bead/type>) 
        
        decorator. Secondly, by adding a .py file to the 
            
            "/chic/coarse_graining_methods/"

        directory. Each method can be defined as a function, taking the 'self'
        argument. The file must contain a dictionary called 'methods' which
        contains the following information for each method:

        - *func*: the function to call.
        - *bead_type*: the type of bead to use. This can be "single" or
            "multi".

        The method will then be automatically registered and available to use.

        ---

        Arguments:
            method: Coarse-graining method to use.
            minimum_coordination_numbers: Minimum coordination number for each
                site type.
            maximum_coordination_numbers: Maximum coordination number for each
                site type.
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
            8, maximum_coordination_numbers, self._all_sites
        )
        
        # Reset bead bonds.
        self._bead_bonds = defaultdict(list)
        
        # call the coarse-graining method.
        method_func(self, **kwargs)

        # connect the beads.
        self._connect_beads()


    def get_beads(self,
        bead_type_to_element_mapping: Dict[str, str] = {
            'a': 'Si', 'b': 'O', 'X': 'Ce'
        },
        skip_non_framework: bool = True,
        energy_key: str = 'c_pe_per_atom'
    ) -> List[Bead]:
        """
        Gather beads and assign species according to the mapping.

        Arguments:
            bead_type_to_element_mapping: Mapping of bead types to elements.
            skip_non_framework: Whether to skip non-framework atoms.

        Returns:
            List of Bead objects.
        """

        def default_factory() -> int:
            """ Start the defaultdict counter at 1. """
            return 1
        
        all_beads = {}
        bead_mol_id = 1
        self._bead2cluster = defaultdict(str)
        number_of_species = defaultdict(default_factory)
        for label, cluster in self._atomic_clusters.items():
            
            # first, check if the cluster should be skipped.
            if skip_non_framework and cluster.skip:
                continue
            
            # get the beads and their ids.
            beads = cluster.beads_frac_coords
            bead_ids = cluster.bead_ids

            # get the species to assign to the beads. if we are keeping the 
            # non-framework species, we want to assign them to the 'X' species.
            if cluster.skip:
                add_species = bead_type_to_element_mapping['X']
            else:
                add_species = bead_type_to_element_mapping[label[0]]

            # make sure the correct number of species are assigned based on the 
            # supplied mapping.
            if isinstance(add_species, str):
                add_species = [add_species] * len(beads)
            elif len(add_species) < len(beads):
                add_species *= len(beads) // len(add_species) + 1
            else:
                add_species = add_species[:len(beads)]
                
            # coarse-grained atomic properties.
            elocal = None; fcm = None
            if 'forces' in self.site_properties:

                # translational.
                these_forces = np.array([
                    self.site_properties['forces'][x] for x in cluster.site_indices
                ])
                these_masses = np.array([
                    x.atomic_mass for x in cluster.species
                ])[:, np.newaxis]
                fcm = these_forces.sum(axis=0)

            if energy_key in self.site_properties:
                these_energies = np.array([
                    self.site_properties[energy_key][x] for x in cluster.site_indices
                ])
                elocal = these_energies.sum()

            # gather the bead object properties.
            for species, frac_coord, bead_id in zip(add_species,beads,bead_ids):
                all_beads[bead_id] = Bead(
                    species, 
                    number_of_species[species], 
                    bead_id, 
                    bead_mol_id,
                    frac_coord,
                    energy=elocal,
                    force=fcm
                )
                number_of_species[species] += 1
                self._bead2cluster[bead_id] = label

            # increment the bead mol ID.
            bead_mol_id += 1

        # store the beads.
        self._beads = all_beads

        return all_beads

    
    def overlay_cg_atomistic_representations(self,
        filename: str = None,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'Ce', 'X': 'Be'},
        crystal_toolkit: bool = False,
        skip_non_framework: bool = True
    ) -> None:
        """
        Overlays the atomistic representation ontop of the coarse-grained 
        representation of the struture. This is useful for visualising the 
        atomistic structure in the context of the coarse-grained structure (and 
        that the coarse-graining was correct).

        Arguments:
            filename: path to CIF to write.
            bead_type_to_element_mapping: mapping of bead types to elements.
            crystal_toolkit: whether to display the structure in crystal toolkit.
            skip_non_framework: whether to skip non-framework atoms.
        """

        # get a copy of the atomistic structure.
        overlay_structure = self.copy()

        # get the beads.
        self.get_beads(
            bead_type_to_element_mapping,
            skip_non_framework=skip_non_framework
        )

        # gather the coarse-grained beads and append them to the structure.
        for bead in self._beads.values():
            overlay_structure.append(bead.species, bead.frac_coord)
            
        # write the structure to file.
        if filename is not None:
            overlay_structure.to(str(filename))

        # display the structure.
        if crystal_toolkit:
            crystal_toolkit_display(overlay_structure)

    
    def to_ase(self) -> Atoms:
        """
        Convert structure to ASE Atoms object.
        """
        return AseAtomsAdaptor.get_atoms(self)


    def to_net(self,
        bead_type_to_element_mapping: Dict[str, str] = {
            'a': 'Si', 'b': 'O', 'X': 'Ce'
        },
        name: str = 'Net',
        supercell: Tuple[int, int, int] = None
    ) -> Net:
        """
        Convert the coarse-grained net to a chic.Net object.

        In order to instantiate the Net class, we need to provide the lattice, 
        species, fractional coordinates, and the bonding information. The 
        bonding information needs to be a list of tuples, where each tuple is of 
        the format:
                        
                        ((atom1, atom2), (image1, image2))

        where atom1 and atom2 are the labels of the atoms that are bonded, and
        image1 and image2 are the periodic images of the atoms that are bonded. 
        The periodic images are given as a tuple of integers, where each integer 
        represents the number of unit cells in each direction.

        Arguments:
            bead_type_to_element_mapping: mapping of bead types to elements.
            name: name of the net.
            supercell: whether to make a supercell of the net. If specified, 
                should be a tuple of three integers, where each integer 
                represents the number of unit cells in each direction.

        Returns:
            Net object.
        """

        # gather the beads.
        beads = self.get_beads(
            bead_type_to_element_mapping, 
            skip_non_framework=True
        )

        # now extract the bonding.
        bonds = []
        for edge, images in self._bead_bonds.items():
            atom1 = beads[edge[0]].label
            atom2 = beads[edge[1]].label
            for image in images:
                bonds.append((
                    (atom1, atom2),(tuple(np.zeros(3,dtype=int)),image['image'])
                ))
        
        # hence create a Pymatgen structure.
        struct = PymatgenStructure(
            self.lattice,
            [bead.species for bead in beads.values()],
            [bead.frac_coord for bead in beads.values()],
            validate_proximity=False,
        )

        # add the labels to the structure in the atom properties.
        labels = [bead.label for bead in beads.values()]
        struct.add_site_property('label', labels)

        # create the bonding class.
        bonding = Bonding(struct, labels, bonds)

        # hence create a new Net object.
        net = Net.from_structure(struct, bonding, filename=name)

        # give the option to make a supercell straight away. This will just
        # call the Net.make_supercell() method and re-write the net.
        if supercell is not None:
            net = net.make_supercell(supercell)

        return net


    def net_to_cif(self, 
        filename: str, 
        write_bonds: bool = True,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O', 'X': 'Ce'},
        name: str = 'net',
        skip_non_framework: bool = True
    ) -> None:
        """
        Write the coarse-grained structure to a (Topo)CIF file.

        Arguments:
            filename: path to CIF to write.
            write_bonds: whether to write bonds to the CIF.
            bead_type_to_element_mapping: mapping of bead types to elements.
            name: name of the network.
            skip_non_framework: whether to skip non-framework atoms.
        """
        TopoCifWriter(self, 
            self.get_beads(
                bead_type_to_element_mapping,
                skip_non_framework=skip_non_framework
            ), 
            self._bead_bonds, name
        ).write_file(filename, write_bonds)


    def net_to_lammps_data(self,
        filename: str,
        write_bonds: bool = True,
        write_angles: bool = False,
        two_way_bonds: bool = False,
        atom_style: str = 'full',
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'},
        name: str = 'net',
    ) -> None:
        """
        Write the coarse-grained structure to a LAMMPS data file.

        Arguments:
            filename: Path to LAMMPS data file to write.
            write_bonds: Whether to write bonds to the LAMMPS data file.
            two_way_bonds: Whether to write bonds in both directions.
            atom_style: Atom style to use. Either 'full' or 'atomic'.
            bead_type_to_element_mapping: Mapping of bead types to elements.
            name: Name of the network.
        """
        # we force the get_beads() method to recompute so that the correct
        # mapping is used.
        LammpsDataWriter(self, 
            self.get_beads(bead_type_to_element_mapping),
            self._bead_bonds, name
        ).write_file(
            filename, 
            write_bonds=write_bonds, 
            write_angles=write_angles,
            atom_style=atom_style, 
            two_way_bonds=two_way_bonds
        )

    
    def net_to_lammps_dump(self,
        filename: str,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'},
        **kwargs
    ) -> None:
        """
        Write the coarse-grained structure to a LAMMPS dump file.

        Arguments:
            filename: Path to LAMMPS dump file to write.
            bead_type_to_element_mapping: Mapping of bead types to elements.
            **kwargs: Keyword arguments to pass to the writer.
        """
        LammpsDumpWriter(self,
            self.get_beads(bead_type_to_element_mapping),
            **kwargs
        ).write_file(filename)


    def net_to_struct(self, 
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'}
    ) -> PymatgenStructure:
        """
        Convert the coarse-grained structure to a pymatgen Structure object.

        Arguments:
            bead_type_to_element_mapping: mapping of bead types to elements.

        Returns:
            PymatgenStructure object.
        """

        # gather the beads.
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

        Returns:
            ASE atoms object.
        """
        
        # first gather beads into a PymatgenStructure object.
        struct = self.net_to_struct()

        # then use the Pymatgen ASE interface to convert to an ASE atoms object.
        return sort(AseAtomsAdaptor.get_atoms(struct))


    def net_to(self, filename: str, fmt: str = '', **kwargs) -> None:
        """
        Use Pymatgen's writer to write the coarse-grained structure to file.

        Arguments:
            filename: Path to file to write.
            fmt: Format to write to. If not specified, will be inferred from
                the file extension.
            kwargs: Keyword arguments to pass to the writer.
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.net_to_struct().to(str(filename), fmt, **kwargs)

    
    def net_to_ase_to(self, filename: str, fmt: str = None, **kwargs) -> None:
        """
        Use ASE's writer to write the coarse-grained structure to file.

        Arguments:
            filename: Path to file to write.
            fmt: Format to write to. If not specified, will be inferred from
                the file extension.
            kwargs: Keyword arguments to pass to the writer.
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.net_to_ase_atoms().write(filename, format=fmt, **kwargs)
        
    
    def net_to_extxyz(self,
        filename: str,
        bead_type_to_element_mapping: Dict[str, str] = {'a': 'Si', 'b': 'O'},
        info_dict: dict = None,
        append: bool = False
    ) -> None:
        """
        Write coarse-grained structure to extxyz file. This will attempt to add
        any bead energies and forces, if provided, to the file.
        """
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.get_beads(bead_type_to_element_mapping)
        
        # get sites.
        sites = [
            PeriodicSite(bead.species, bead.frac_coord, self.lattice)
            for bead in self._beads.values()
        ]

        # get atoms.
        columns = ['symbols', 'positions']
        struct = PymatgenStructure.from_sites(sites)
        atoms = AseAtomsAdaptor.get_atoms(struct)

        # check for local energies and forces.
        if all(x.energy is not None for x in self._beads.values()):
            columns.append('elocal')
            energies = np.array([x.energy for x in self._beads.values()])
            atoms.set_array('elocal', energies)
            atoms.info['energy'] = energies.sum()
        if all(x.force is not None for x in self._beads.values()):
            columns.append('forces')
            atoms.set_array(
                'forces', np.array([x.force for x in self._beads.values()])
            )
            
        # add any extra information to structure.
        if info_dict is not None:
            atoms.info.update(info_dict)

        # write to file.
        sort(atoms).write(
            filename, format='extxyz', columns=columns, write_info=True,
            append=append
        )

    
    def net_to_gulp_framework_nodes(self,
        filename: str, 
        keywords: list = None,
        commands: list = None,
        **kwargs
    ) -> None:
        """
        Reduce the coarse-grained AB2 net to the bare framework consisting of 
        the A nodes only. Then write to a GULP input file.

        Arguments:
            filename: Path to GULP input file to write.
            **kwargs: Keyword arguments to pass to the writer.
        """

        # check first that the neighbourlist has been computed, atomic clusters
        # have been identified, and the coarse-grained net has been generated.
        if self._neighbour_list is None:
            raise ValueError('Neighbourlist not computed. Please run ' \
                'get_neighbours_crystalnn() first.')
        if self._atomic_clusters is None:
            raise ValueError('Atomic clusters not identified. Please run ' \
                'find_atomic_clusters() first.')
        
        # gather beads.
        self.get_beads({'a': 'H', 'b': 'O'})

        # select keywords. opti conp bond property molq phon eigenmodes
        if keywords is None:
            keywords = [
                'conp', 'bond', 'property', 'molq', 'phon', 'eigenmodes'
            ]
        
        # we cam now proceed to reduce the net to the framework nodes and write
        # to a GULP input file.
        _gulp_kwargs = {
            'sub_elem': 'H',
            'rattle': 0.0,
            'k_bond': 200.0,
            'k_angle': 10.0,
            'rtol': 2.0,
            'keywords': keywords,
            'commands': commands
        }
        _gulp_kwargs.update(kwargs)

        # instantiate the writer and write to file.
        gulp_writer = GulpWriter(self, **_gulp_kwargs)
        gulp_writer.write_file(filename)

    
    @timer(colour=Colours.OKGREEN)
    def replicate(self, factors: List[int] = None) -> 'Structure':
        """
        Replicate the net [nx, ny, nz] times in each of the respective lattice
        directions. 
        
        Here we replicate the net and the full atomistic structure. We also 
        replicate the neighbour list so that we can assign the new bonds to the 
        supercell without having to recompute the neighbour list.

        Arguments:
            factors: List of factors to replicate the net in each direction.

        Returns:
            New Structure object.
        """

        # default to replicating the net 2x2x2.
        if factors is None:
            factors = [2, 2, 2]
        
        # suggest that user computes the neighbour list first if they haven't 
        # already.
        if not self._neighbour_list:
            warnings.warn('Neighbour list not computed. It would be wise to ' \
                'run get_neighbours_crystalnn() first.')

        # update the Lattice object.
        scale_matrix = np.diag(factors)
        lattice = Lattice(np.dot(scale_matrix, self.lattice.matrix))

        # now replicate the atomistic structure.
        new_sites = []
        orig2new_atoms = {}
        atom_count = 0
        for cell_count, img in enumerate(product(*[range(e) for e in factors])):
            img = np.array(img)
            for site_index, site in enumerate(self):
                orig2new_atoms[str(site_index)+str(img % factors)] = atom_count
                new_frac_coords = (site.frac_coords + img) / factors
                new_sites.append(
                    PeriodicSite(
                        site.species,
                        new_frac_coords,
                        lattice,
                        properties=site.properties
                    )
                )
                atom_count += 1

        # now we can reinstantiate the class.
        new_struct = PymatgenStructure.from_sites(new_sites)
        instance = Structure.from_structure(new_struct)

        # we might need to update stored variables from this instance.
        instance._site_type_lookup = self._site_type_lookup
        instance._site_types = self._site_types
        instance._all_sites = self._all_sites
        instance._cores = self._cores
        instance._verbose = self._verbose

        new_neighbour_list = {}
        for site_index, neighbours in self._neighbour_list.items():

            # we need to add the zero-neighbour entries back into the neighbour
            # list for compatibility with the atomic cluster searching 
            # algorithm.
            if len(neighbours) == 0:
                for img in product(*[range(e) for e in factors]):
                    img = np.array(img)
                    nl1 = orig2new_atoms[str(site_index)+str(img % factors)]
                    new_neighbour_list[nl1] = []
                continue

            # otherwise, we need to replicate the neighbours but with updated
            # periodic images.
            for neighbour in neighbours:
                for img in product(*[range(e) for e in factors]):
                    img = np.array(img)
                    img1_n = img
                    img2_n = img + neighbour['image']

                    # get bead IDs in terms of the new beads.
                    nl1 = orig2new_atoms[str(site_index)+str(img1_n % factors)]
                    nl2 = orig2new_atoms[str(neighbour['site_index'])+str(img2_n % factors)]
                    ni1 = np.zeros(3, dtype=int)
                    ni2 = img2_n // factors

                    # get the new positions of the beads.
                    cart1 = lattice.get_cartesian_coords(
                        new_sites[nl1].frac_coords + ni1
                    )
                    cart2 = lattice.get_cartesian_coords(
                        new_sites[nl2].frac_coords + ni2
                    )

                    # update neighbour list.
                    neighbour_site = new_struct[nl2]
                    neighbour_site.properties['nn_distance'] = np.linalg.norm(cart2-cart1)
                    new_neighbour_list.setdefault(nl1, []).append({
                        'site_index': nl2,
                        'image': ni2,
                        'weight': neighbour['weight'],
                        'distance': np.linalg.norm(cart2-cart1),
                        'site':neighbour_site
                    })
            
        instance._neighbour_list = new_neighbour_list

        """
        if self._beads is not None:

            new_beads = {}
            new_bead_bonds = {}
            orig2new_beads = {}
            num_labels = {"Si":1, "O":1}

            unique_labels = {
                remove_non_letters(bead.label) for bead in self._beads.values()
            }
            num_labels = {label:1 for label in unique_labels}

            # count how many mol-ids there are in a single unit cell. this will 
            # allow us to assign new mol-ids to the replicated unit cells.
            num_mols = len({bead.bead_mol_id for bead in self._beads.values()})

            bead_count = 1
            for cell_count, img in enumerate(product(*[range(e) for e in factors])):

                img = np.array(img)
                for bead in self._beads.values():
                    
                    # get the label and remove any non-letter characters.
                    label = remove_non_letters(bead.label)

                    # prepare new bead.
                    new_beads[bead_count] = Bead(
                        species=label,
                        species_number=num_labels[label], 
                        bead_id=bead_count, 
                        bead_mol_id=bead.bead_mol_id + cell_count * num_mols,
                        frac_coord= (bead.frac_coord + img) / factors
                    )

                    # update old bead-ids to new bead-ids.
                    orig2new_beads[str(bead.bead_id)+str(img%factors)] = bead_count

                    # update counters.
                    num_labels[label] += 1
                    bead_count += 1

            # map old bonds to new images and determine the translation of the bonds
            # for the new periodic images.
            for (n1, n2), all_bonds in self._bead_bonds.items():
                for bond in all_bonds:
                    for img in product(*[range(e) for e in factors]):
                        img = np.array(img)
                        img1_n = img
                        img2_n = img + bond['image']

                        # get bead IDs in terms of the new beads.
                        nl1 = orig2new_beads[str(n1)+str(img1_n % factors)]
                        nl2 = orig2new_beads[str(n2)+str(img2_n % factors)]
                        ni1 = np.zeros(3, dtype=int)
                        ni2 = img2_n // factors

                        # get the new positions of the beads.
                        cart1 = lattice.get_cartesian_coords(
                            new_beads[nl1].frac_coord + ni1
                        )
                        cart2 = lattice.get_cartesian_coords(
                            new_beads[nl2].frac_coord + ni2
                        )

                        # hence assign new clusters.
                        new_bead_bonds.setdefault((nl1, nl2), []).append({
                            'image': ni2,
                            'weight': bond['weight'],
                            'bead_distance': f"{np.linalg.norm(cart2-cart1):.4f}",
                            'bond_type': bond['bond_type']
                        })
        """

        return instance

    
    def pickle_neighbourlist(self, filename: str):
        """
        Write the current computed neighbour list to a pickle-d file.

        :param filename: name of file to write.
        """
        if not self._neighbour_list:
            raise 


    def _get_adjacency_matrix_new(self) -> csr_matrix:
        """
        Compute a global adjacency matrix for all intra-unit bonds.
        """
        data = []
        i_indices = []
        j_indices = []
        
        for site_index, site_neighbours in self._neighbour_list.items():
            this_site = self[site_index]
            this_site_type = self._get_site_type_index(this_site.specie.symbol)
            
            # Use a list comprehension to filter neighbours based on conditions
            filtered_neighbours = [
                neighbour for neighbour in site_neighbours
                if self._get_site_type_index(neighbour['site'].specie.symbol) == this_site_type
                and self._check_neighbour_conditions('intra', neighbour, self._all_sites[this_site_type])
            ]
            
            for neighbour in filtered_neighbours:
                data.append(1)
                i_indices.append(site_index)
                j_indices.append(neighbour['site_index'])
                
            # Update the neighbour list with only the filtered neighbours
            self._neighbour_list[site_index] = filtered_neighbours

        # Construct the CSR matrix using the data and indices
        adjacency_matrix = csr_matrix(
            (data, (i_indices, j_indices)), 
            shape=(self.num_sites, self.num_sites)
        ).maximum(
            csr_matrix((data, (j_indices, i_indices)), 
            shape=(self.num_sites, self.num_sites))
        )
        return adjacency_matrix


    def _get_adjacency_matrix(self) -> csr_matrix:
        """
        Compute a global adjacency matrix for all intra-unit bonds.
        """
        data = []
        i_indices = []
        j_indices = []

        for site_index, site_neighbours in self._neighbour_list.items():

            this_site = self[site_index]
            this_site_type = self._get_site_type_index(this_site.specie.symbol)

            for neighbour in site_neighbours:

                neighbour_site = neighbour['site']
                neighbour_site_type = self._get_site_type_index(
                    neighbour_site.specie.symbol
                )

                if this_site_type != neighbour_site_type:
                    continue
                
                site_type = self._all_sites[this_site_type]                
                if self._check_neighbour_conditions('intra',neighbour,site_type):
                    data.append(1)
                    i_indices.append(site_index)
                    j_indices.append(neighbour['site_index'])
                else:
                    pass
                    #site_neighbours.remove(neighbour)

        adjacency_matrix = csr_matrix(
            (data, (i_indices, j_indices)), 
            shape=(self.num_sites, self.num_sites)
        ).maximum(
            csr_matrix((data, (j_indices, i_indices)), 
            shape=(self.num_sites, self.num_sites))
        )

        return adjacency_matrix
    

    def _identify_clusters(self, adjacency_matrix: csr_matrix):
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
        allow_same_type_neighbours: bool = False
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
                and self._check_neighbour_conditions('inter',neighbor,site_type)
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
        
        # Handle any site indices that didn't get an image during the first pass
        for site_index in site_indices_set:
            if site_index not in images:
            
                # Search for a visited neighbor with a valid image to use as a
                # reference.
                for neighbor in cluster_neighbors[site_index]:
                    neighbor_index = neighbor['site_index']
                    if neighbor_index in images:
                        # Use the visited neighbor's image and relative image to
                        # assign the new image
                        neighbor_image = neighbor['image']
                        images[site_index] = images[neighbor_index] + neighbor_image
                        break

                if site_index not in images:
                    # If no visited neighbor with image was found, raise error.
                    raise ValueError(
                        f"No image found for site index {site_index}, "
                        "and no visited neighbor with an image to use as "
                        "a reference."
                    )

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
    

    def _finalise_clusters(self, clusters, allow_same_type_neighbours):
        """
        """

        final_clusters = {}
        element_to_cluster = {}
        site_type_counts = np.ones(len(self._site_types), dtype=np.uint)
        for site_indices in clusters.values():

            site_type = np.unique([
                self._get_site_type_index(self[i].specie.symbol) 
                for i in site_indices
            ])

            assert len(site_type) == 1, 'Multiple site types in cluster.'
            site_type_index = site_type[0]
            site_type = self._all_sites[site_type_index]

            # hence convert to AtomicCluster object.
            cluster = self._prepare_atomic_cluster(
                site_indices, site_type, allow_same_type_neighbours)
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
    

    def _get_site_type_index(self, item: str) -> int:
        try:
            return self._site_type_lookup[item]
        except KeyError:
            raise ValueError(f"Item {item} not found in any list.")
        
    
    def _get_site_type(self, item: str) -> str:
        try:
            return self._all_sites[self._site_type_lookup[item]]
        except KeyError:
            raise ValueError(f"Item {item} not found in any list.")

    
    @coarse_graining_methods.register('centroid', 'single')
    def _centroid_method(self,
        skip_elements: List[str] = None, 
        precision: float = 1e-8,
    ) -> None:
        """
        Place a single bead at the geometric centre of each AtomicCluster. This 
        does not define additional bonds between beads so all external bonds are 
        bound to this one site. Accordingly, we forward all external connections
        to any atom in this cluster directly to the bead.

        :param skip_elements: list of elements to skip when assigning the
            geometric centroid of the building unit.
        :param precision: precision to round to when assigning bead coordinates.
        :return: None.
        """

        # get the centroid of each cluster and assign it as the single bead.
        bead_count = 1
        non_framework_count = -1
        for label, cluster in self._atomic_clusters.items():

            # first assign the beads.
            centroid = [cluster.get_centroid(skip_elements=skip_elements)]
            
            # skip the cluster if it does not have the required coordination
            # number to other clusters. instead we label with a negative bead
            # ID to indicate that it is not a framework bead.
            if not (
                self._minimum_coordination_numbers[label[0]] <= 
                cluster.coordination_number <=
                self._maximum_coordination_numbers[label[0]]
            ):
                cluster.assign_beads(
                    [non_framework_count],
                    *self._wrap_cart_coords_to_frac(
                        centroid, precision=precision
                    ),
                    {i: [0] for i in cluster.site_indices},
                    internal_bead_bonds=[]
                )
                cluster.skip = True
                non_framework_count -= 1
                continue
            

            # assign the values to the cluster.
            cluster.assign_beads(
                [bead_count],
                *self._wrap_cart_coords_to_frac(centroid, precision=precision),
                {i: [0] for i in cluster.site_indices},
                internal_bead_bonds=[]
            )
            cluster.skip = False
            
            # update the bead count.
            bead_count += 1


    @coarse_graining_methods.register('shortest path', 'single')
    def _shortest_path_method(self):
        """
        """
        raise NotImplementedError('Shortest path method not implemented yet.')
        

    def __setup_pickle__(self):
        """
        If pickle is allowed, neighbours lists are pickled to a hidden
        directory based on the structure's hash key. This will be searched
        before further progress is made.
        """
        self.__pickle_dir__ = self.__hash_key__()
        if self.__pickle_dir__.exists():
            if (self.__pickle_dir__ / 'nl.pickle').exists():
                with open(self.__pickle_dir__ / 'nl.pickle', 'rb') as f:
                    self._neighbour_list = pickle.load(f)
        else:
            self.__pickle_dir__.mkdir(exist_ok=True, parents=True)
            
    
    def tidy(self):
        """"
        Force clearance of the pickled data.
        """
        if self.__allow_pickle__ and self.__pickle_dir__.exists():
            for item in self.__pickle_dir__.iterdir():
                if item.is_file():
                    item.unlink()
            self.__pickle_dir__.rmdir()
        
    def __hash_key__(self):
        """
        Generates a unique hash key for the structure based on lattice
        parameters, atomic positions, and types.
        """
        lattice_str = str(np.round(self.lattice.matrix.flatten(),3))
        atoms_str = '_'.join(sorted([
            f"{atom.species}_{np.round(atom.frac_coords,1)}" for atom in self
        ]))
        unique_str = lattice_str + atoms_str
        hash_key = hashlib.sha256(unique_str.encode()).hexdigest()
        return Path('.' + hash_key)

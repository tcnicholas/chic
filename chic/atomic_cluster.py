"""
27.07.23
@tcnicholas
Atomic cluster functionality.
"""


from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import networkx as nx
from pymatgen.core import Molecule

from .utils import crystal_toolkit_display


class AtomicCluster:
    """
    Represents a cluster of atoms in a structure.
    """
    def __init__(self, 
        site_indices: List[int],
        species: List[str],
        cart_coords: np.ndarray,
        images: np.ndarray,
        edges: Optional[Dict[Tuple[int, int], Dict[str, float]]] = {},
        edges_external: Optional[Dict[Tuple[int, int], Dict[str, float]]] = {}
    ):
        """
        :param site_indices: list of site indices in the cluster. These should
            correspond to the site indices in the main structure.
        :param species: list of species in the cluster.
        :param cart_coords: cartesian coordinates of the cluster. These should
            form a set of image consistent coordinates.
        :param images: images of the cluster. These are the images used to take
            the wrapped fractional coordinates to the cartesian coordinates.
        :param edges: dictionary of edges in the cluster. The keys should be
            the connected nodes, and the values are a dictionary of properties.
        :param edges_external: dictionary of edges in the cluster that connect
            to atoms outside the cluster. The keys should be the connected 
            nodes (internal, external), and the values are a dictionary of 
            properties, which should include the 'image' key.
        """
        assert len(site_indices)==len(species)==len(cart_coords)==len(images), \
            'Input lists must have same length.'
        
        self._site_indices = site_indices
        self._species = species
        self._symbols = [s.symbol for s in species]
        self._cart_coords = cart_coords
        self._images = images
        self._edges = edges
        self._edges_external = edges_external
        self._graph = self._create_graph()
        self._rings_cache = None
        self._bound_to_clusters = None

        self._bead_ids = None
        self._beads_frac_coords = None
        self._beads_images = None
        self._atom_to_bead_index = None
        self._internal_bead_bonds = None

        # whether this cluster is being skipped for the final network. This is
        # usually based on the number of external edges, but could be set
        # manually or via custom coarse-graining protocols.
        self._skip = False


    @property
    def graph(self):
        """
        Getter for _graph attribute.
        """
        return self._graph
    

    @property
    def bead_ids(self):
        """
        Getter for bead_ids attribute. These are the unique IDs assigned to the
        beads for the cluster.
        """
        return self._bead_ids
    

    @property
    def beads_frac_coords(self):
        """
        Getter for beads_frac_coords attribute. These are the fractional
        coordinates of the beads.
        """
        return self._beads_frac_coords
    

    @property
    def beads_images(self):
        """
        Getter for beads_images attribute. These are the images of the beads.
        """
        return self._beads_images
    

    @property
    def atom_to_bead_index(self):
        """
        Getter for atom_to_bead_index attribute. This is a dictionary mapping
        the atom indices to the beads indices.
        """
        return self._atom_to_bead_index
    

    @property
    def internal_bead_bonds(self):
        """
        Getter for internal_bead_bonds attribute. This is a list of tuples
        representing the bonds between beads in the cluster.
        """
        return self._internal_bead_bonds
    

    @property
    def site_indices(self) -> List[int]:
        """
        :return: list of site indices in the cluster.
        """
        return self._site_indices
    

    @property
    def species(self) -> List[str]:
        """
        :return: list of species in the cluster.
        """
        return self._species
    

    @property
    def cart_coords(self) -> np.ndarray:
        """
        :return: cartesian coordinates of the cluster.
        """
        return self._cart_coords
    

    @property
    def coordination_number(self) -> int:
        """
        :return: coordination number of the cluster, i.e. the number of external
            edges.
        """
        return len(self._edges_external)
    

    @property
    def skip(self) -> bool:
        """
        Getter for skip attribute.
        """
        return self._skip
    

    @skip.setter
    def skip(self, value: bool):
        """
        Setter for skip attribute.
        """
        self._skip = value
    

    def get_indices_by_species(self, species: str) -> List[int]:
        """
        Get the indices of the sites in the cluster with a given species.
        """
        return [i for i, s in enumerate(self._symbols) if s == species]
    

    def get_cart_coords_by_species(self, species: str) -> np.ndarray:
        """
        Get the cartesian coordinates of the sites in the cluster with a given
        species.
        """
        return self._cart_coords[self.get_indices_by_species(species)]
    

    def assign_beads(
        self,
        bead_ids: List[int],
        beads_frac_coords: np.ndarray,
        beads_images: np.ndarray,
        atom_to_bead_index: Dict[int, List[int]],
        internal_bead_bonds: List[Tuple[int, int]]
    ):
        """
        Assign beads to the cluster.

        :param bead_ids: list of bead IDs.
        :param beads_frac_coords: fractional coordinates of the beads.
        :param beads_images: images of the beads.
        :param atom_to_bead_index: dictionary mapping the atom indices to the
            beads indices.
        :param internal_bead_bonds: list of tuples representing the bonds
            between beads in the cluster.
        """
        self._bead_ids = bead_ids
        self._beads_frac_coords = beads_frac_coords
        self._beads_images = beads_images
        self._atom_to_bead_index = atom_to_bead_index
        self._internal_bead_bonds = internal_bead_bonds


    def get_centroid(self, 
        skip_elements: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute the centroid of the cluster.

        :param skip_elements: the elements to skip when computing the centroid.
        """
        skip_elements = skip_elements or []
        indices = [
            i for i, symbol in enumerate(self.species) 
            if symbol.symbol not in skip_elements
        ]
        return np.mean(self.cart_coords[indices], axis=0)
    

    def find_rings(self, including: List[str] = None):
        """
        Find all rings in the cluster.

        :param including: list of elements to include in the rings. If None,
            all rings will be returned irrespective of the elements included.
        """

        if self._rings_cache is not None:
            return self._rings_cache

        if len(self._graph) == 1:
            result = {'nodes': self._site_indices, 'edges': None}
            self._rings_cache = result
            return result

        undirected = self._graph.to_undirected()
        directed = undirected.to_directed()
        unique_cycles = set()

        if including is not None:
            including = set(including)

        for cycle in nx.simple_cycles(directed):
            if len(cycle) <= 2:
                continue
            sorted_cycle = sorted(cycle)
            frozenset_cycle = frozenset(sorted_cycle)
            if frozenset_cycle in unique_cycles:
                continue
            unique_cycles.add(frozenset_cycle)

        if not unique_cycles:
            result = {'nodes': self._site_indices, 'edges': None}
            self._rings_cache = result
            return result

        if including is not None:
            cycles_nodes = [
                list(cycle) for cycle in unique_cycles if including & cycle
            ]
        else:
            cycles_nodes = [list(cycle) for cycle in unique_cycles]

        cycles_edges = [
            [(cycle[idx - 1], itm) for idx, itm in enumerate(cycle)] 
            for cycle in cycles_nodes
        ]

        result = [
            {'nodes': n, 'edges': e} for n, e in zip(cycles_nodes, cycles_edges)
        ]

        self._rings_cache = result
        return result


    def to_molecule(self) -> Molecule:
        """
        Convert the AtomicCluster to a pymatgen Molecule object.
        
        Note, this has a lot of shared functionality, but loses the site 
        indexing that I want to keep uniform across all representations.
        """
        return Molecule(self._species, self._cart_coords)
    

    def get_image_by_site_index(self, site_index: int) -> np.ndarray:
        """
        Exctract the periodic image for a given site index.
        """
        return self._images[self._site_indices.index(site_index)]

    
    def visualise(self):
        """
        Visualise the cluster using crystal toolkit.
        """
        crystal_toolkit_display(self.to_molecule().get_centered_molecule())


    @classmethod
    def with_updated_coordinates_and_images(cls, 
        instance, 
        new_cart_coords: np.ndarray, 
        new_images: np.ndarray
    ):
        """
        Create a copy of an existing AtomicCluster instance with updated 
        cartesian coordinates and images. Also set edges_external to None.

        :param instance: An instance of AtomicCluster to be copied.
        :param new_cart_coords: Updated cartesian coordinates.
        :param new_images: Updated images.
        :return: New AtomicCluster instance with updated attributes.
        """
        return cls(
            site_indices=instance.site_indices,
            species=instance.species,
            cart_coords=new_cart_coords,
            images=new_images,
            edges=instance._edges,
            edges_external={}  # Set edges_external to None
        )
    

    def _create_graph(self) -> nx.Graph:
        """
        Build a graph from the cluster.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self._site_indices)
        if self._edges is not None:
            graph.add_edges_from(self._edges)
        return graph
    

    def __len__(self) -> int:
        """
        :return: number of sites in the cluster.
        """
        return len(self._site_indices)
    

    def __repr__(self) -> str:
        """
        :return: string representation of the cluster.
        """
        return f'AtomicCluster("{self.to_molecule().composition}", ' \
            f'site_indices={self._site_indices})'


def _get_cluster_bead_info(
    cluster: AtomicCluster, 
    site: int
):
    """
    Get the bead information for a given site in a cluster.

    :param cluster: AtomicCluster object.
    :param site: Site index in the cluster.
    :return: Numpy array of bead information.
    """
    bead_indices = cluster._atom_to_bead_index[site]                                # internal indexing of beads.
    bead_info = np.array(
        [(cluster._bead_ids[x],                                                     # bead index from global list.
          cluster._beads_frac_coords[x],                                            # bead wrapped fractional coordinates.
          cluster._beads_images[x]) for x in bead_indices],                         # image consistent bead images
        dtype=[('bead_ids','O'), ('bead_frac_coords','O'), ('bead_images','O')]
    )
    return bead_info


def _determine_second_bead_image(
    bead1: np.ndarray, 
    bead2: np.ndarray, 
    local_cluster_atom_image: np.ndarray, 
    bound_cluster_atom_image: np.ndarray, 
    edge_info: Dict[str, Any]
):
    """
    Calculate the distance between two beads.

    :param bead1: Information for bead1.
    :param bead2: Information for bead2.
    :param local_cluster_atom_image: Image of atom in local cluster.
    :param bound_cluster_atom_image: Image of atom in bound cluster.
    :param edge_info: Information of the edge connecting the clusters.
    :return: image of bead2.
    """
    img_diff = edge_info['image'] + local_cluster_atom_image - \
        bead1['bead_images'] - bound_cluster_atom_image
    return (bead2['bead_images'] + img_diff).astype(int)


def _order_beads(
    bead1: np.ndarray, 
    bead2: np.ndarray,
    image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Order the beads by their bead_ids.
    
    :param bead1: Information for bead1.
    :param bead2: Information for bead2.
    :param image: Image of bead2.
    """
    if bead1['bead_ids'] < bead2['bead_ids']:
        return bead1, bead2, (image + 0.0).astype(int)
    else:
        return bead2, bead1, (-image + 0.0).astype(int)
    

@dataclass(order=True)
class Bead:
    """
    A bead is a fractional coordinate and a species.

    The bead-mol-id allows us to group together beads that are part of the same
    molecule. This is information can be used in the LAMMPS data and dump 
    format files.
    """
    species: str
    species_number: int
    bead_id: int
    bead_mol_id: int
    frac_coord: np.ndarray


    def __post_init__(self):
        """ Set the bead-id as the sort index. """
        self.sort_index = self.bead_id


    @property
    def label(self) -> str:
        """
        Return the label of the bead.
        """
        return f"{self.species}{self.species_number}"


    def to_topocif_string(self) -> str:
        """
        Convert the bead to a string in the TOPCIF format.
        """
        return f"{self.label:>8} {self.species:>8} 1 " \
            f"{self.frac_coord[0]:>14.10f} " \
                f"{self.frac_coord[1]:>14.10f} " \
                    f"{self.frac_coord[2]:>14.10f}  1"
    

    def to_lammps_string(self, 
        lattice, 
        atom_style: str, 
        species_to_atom_type: Dict[str, int],
        mass_dict = None
    ) -> str:
        """
        Convert the bead to a string in the LAMMPS format.
        """
        cart_coord = lattice.get_cartesian_coords(self.frac_coord % 1) + 0.0
        atom_type = species_to_atom_type[self.species]
        mass = mass_dict[atom_type]['mass'] if mass_dict is not None else ''
        if atom_style == 'full':
            return self.to_lammps_full(atom_type, cart_coord)
        elif atom_style == 'atomic':
            return self.to_lammps_atomic(atom_type, cart_coord)
        elif atom_style == 'dump':
            return self.to_lammps_dump(atom_type, cart_coord)
        else:
            raise ValueError(
                f"atom_style must be 'full', 'atomic', or 'dump', not {atom_style}"
            )
    

    def to_lammps_full(self, atom_type, cart_coord) -> str:
        """
        Convert the bead to a string in the LAMMPS atom_style full format.
        This contains the following columns for each atom:

            atom-ID molecule-ID atom-type q x y z
        """
        return f'{self.bead_id:>6.0f} {self.bead_mol_id:>6.0f} ' \
            f'{atom_type:>6.0f} {0.0:>8.5f} ' \
            f'{cart_coord[0]:>15.10f} ' \
            f'{cart_coord[1]:>15.10f} ' \
            f'{cart_coord[2]:>15.10f}'
    

    def to_lammps_atomic(self, atom_type, cart_coord) -> str:
        """
        Convert the bead to a string in the LAMMPS atom_style atomic format.
        This contains the following columns for each atom:

            atom-id atom-type x y z
        """
        return f'{self.bead_id:>6.0f} {atom_type:>6.0f} ' \
            f'{cart_coord[0]:>15.10f} ' \
            f'{cart_coord[1]:>15.10f} ' \
            f'{cart_coord[2]:>15.10f}'

    
    def to_lammps_dump(self, atom_type, cart_coord) -> str:
        """
        Convert the bead to a string in the LAMMPS dump format.

            id mol type mass x y z
        """
        return f'{self.bead_id:.0f} {self.bead_mol_id:.0f} {atom_type:.0f} ' \
            f'{cart_coord[0]:.10f} {cart_coord[1]:.10f} {cart_coord[2]:.10f}'


    def __repr__(self) -> str:
        """ Pretty string formatting. """
        return f"Bead(id={self.bead_id}, type={self.species}, frac_coord={self.frac_coord})"
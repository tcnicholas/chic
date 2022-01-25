"""
20.07.21
@tcnicholas
Module for defining the building unit class.
"""


from .utils import *

import networkx as nx
from networkx.algorithms.shortest_paths import all_shortest_paths

from pymatgen.core.composition import Composition
from ase import Atoms
import numpy as np
import itertools


class buildingUnit:
    """
    The building unit class stores information about the fundmental building
    units in the structure (as found via the reduce method). It also has the
    coarse-graining methods for defining discrete the unit as a single, discrete
    position.
    """

    def __init__(self, structure, atoms, bonds, connectivity):
        """
        Args
        ----
        structure: pymatgen.core.Structure
            Pymatgen Structure object.

        atoms: list
            Atom indices and images

        bonds: list
            Internal building unit bonds (pairs of atom indices)

        connectivtiy: list
            Bonds between atoms in the unit and atoms in building units of other
            site-types.

            Formatted as: ([atom1, image1], [atom2, image2], weight)
            
            See the c.reduce() function for implementation details.
        """

        # store information from full-atomistic structure #
        self.ix = [a[0] for a in atoms] # indices
        self.img = [a[1] for a in atoms] # images
        self.sym = [structure[a[0]].specie.symbol for a in atoms] # atom symbols
        self.atoms = np.array([iximg2cart(structure,*a) for a in atoms]) # cart
        self.bonds = bonds # internal building unit bonds
        self.connectivity = connectivity # external bonds to the building unit

        # store information about coarse-grained structure dummy atom fractional
        # coordinates and image
        self._frac_img = None 
        # labels and images of other units bonded to this unit.
        self._unit_bonds = None

        # create NetworkX graph of building unit.
        self.graph = moleculeGraph(self.ix, self.sym, bonds, connectivity)

    
    @property
    def frac_img(self):
        """
        Fractional coordinates and image of building unit.
        """
        return self._frac_img
    

    @frac_img.setter
    def frac_img(self, value):
        """
        Setter of building unit fractional coordinates and image.
        """
        self._frac_img = value

    
    @property
    def unit_bonds(self):
        """
        Labels and images of other units bonded to this unit.
        """
        return self._unit_bonds
    

    @unit_bonds.setter
    def unit_bonds(self, value):
        """
        Settter of labels and images of other units bonded to this unit.
        """
        self._unit_bonds = value
    

    def centroid(self, ignore=[]):
        """
        Get centroid of building unit in cartesian coordinates. Optionally, can
        choose species to ignore in the weighting of the building unit centroid.

        Args
        ----
        ignore: list
            list of atomic symbols to ignore in the calculation of the centroid
            position.
        """
        
        # Get indices of atoms not in the ignore list.
        ix = [i for i in range(len(self.sym)) if self.sym[i] not in ignore]
        return np.mean(self.atoms[ix],axis=0)

    
    def shortest_path(self, useCycles=False):
        """
        Get the middle of the shortest path between the "connecting" nodes of
        the building unit to the external units.
        
        Finds the shortest path using the NetworkX graph, identifies the middle
        atom(s), and places dummy atom at that position (or mid point of the
        bond if the length of path is even). If there are >2 connecting nodes in
        the building unit, the shortest path between all combinations will be
        found, and the average position of each path will be taken.

        Args
        ----
        useCycles: bool
            If True, contracts all cycles to single node at the centre and then
            determines the new path.
        """

        # get connecting atoms.
        ca = self.connecting_atoms()

        # for all combinations of connecting atoms, get the middle node(s) of
        # the shortest path(s).
        c = itertools.combinations([a[0] for a in ca], r=2)
        m = [self.graph.middle_shortest_path(*n, useCycles) for n in c]

        # for combination, i, in m...
        # for path, p, in i...
        # for atom, a, in p...
        return np.mean([
            [   np.mean([self.atoms[self.ix.index(a)] for a in p], axis=0)
                    for p in i]
                        for i in m], axis=0)[0]


    def connecting_atoms(self):
        """
        Get the atoms in the building unit that are bonded to external atoms. If
        multiple "internal" atoms are bound to the same "external" atom, choose
        one with greatest voronoi weight.
        """

        # encode the connectivity in strings and create a dictionary to reformat
        f_con = []; iis = {}
        for b in self.connectivity:

            # extract the internal atom.
            s1, d1 = ixImg2str(b[0])
            iis.update(d1)

            # extract the external atom.
            s2, d2 = ixImg2str(b[1])
            iis.update(d2)
            
            # store the encoded connection with voronoi weight.
            f_con.append([s1, s2, b[2]])
        
        # get unique external atoms.
        ue = {a[1] for a in f_con}

        # get list of internal atoms if bonded the external atom, e, sort by
        # voronoi weight (high->low), and taken atom with highest weight.
        ui = []
        for e in ue:
            ui.append( sorted(  [(i[0],i[2]) for i in f_con if i[1]==e], 
                                key=lambda x: x[1], 
                                reverse=True    )[0][0])
        
        # turn back to original formatting.
        return [iis[a] for a in ui]

    
    def cn(self, get_atoms=False):
        """
        Return the connectivtiy of the building unit. Defined as the number of
        unique "external" atoms the building unit is bonded too. Note: in some
        cases the same index but different iamges are found, and therefore get a
        set of (ix, img) tuples for each atom.

        Args
        ----
        get_atoms: bool
            If True, returns the atom index and image for each "externally"
            connected atom.
        """

        # Store each external atom as (ix, img) tuples.
        ea = set()

        # The external atom and image are always in position [1][0] and [1][1].
        for a in self.connectivity:
            ea |= {(a[1][0],tuple(a[1][1]))}

        # Define coordination number as number of external atoms the unit is
        # bonded to.
        cn = len(ea)
        
        # If get_atoms, also return the unique set of external atoms and images.
        if get_atoms:
            return cn, ea
        
        # Otherwise return the coordination number only.
        return cn

    
    def bu_html(self):
        """
        Create a HTML representation of the building unit which can be displayed
        in e.g. iPython notebook.
        """

        # First create atoms object of molecule.
        mol = Atoms(self.sym, self.atoms)

        # Then convert to HTML string.
        return atoms2html(mol)

    
    def composition(self, asPymatgen=False):
        """
        Get chemical composition of building unit using Pymatgen Composition.

        Args
        ----
        asPymatgen      :   if True, returns the Pymatgen Composition object;
                            otherwise returns the formula sorted by
                            electronegativity.
        """

        # create Pymatgen Composition object.
        comp = Composition("".join(self.sym))
        if asPymatgen:
            return comp

        # return elements sorted by electronegativity.
        return comp.formula


    def __len__(self):
        """
        Define length of building unit as the number of constituent atoms.
        """
        return len(self.sym)


class moleculeGraph:
    """
    A class for representing a molecular building unit as a NetworkX graph.
    Allows more sophisticated coarse-graining methods to be implemented by
    e.g. identifying rings or shortest paths between bonding atoms.

    #TODO
    -----
    Although (hopefully) the molecules should nto have too much disorder at this
    point, and not be too large, NP-complete depth-first searching is quite slow
    and so may want to translate this into native objects that can be spead up
    using i.e. numba or a C extension module.
    """

    def __init__(self, indices, symbols, bonds, connectivity):
        """
        Args
        ----
        indices: list
            Indices of atoms in building unit.

        symbols: list
            Chemical symbols in building unit.

        bonds: list
            Internal building unit bonds (pairs of atom indices)

        connectivtiy: list
            Bonds between atoms in the unit and atoms in building units of other
            site-types.

            Formatted as: ([atom1, image1], [atom2, image2], weight)
            
            See the c.reduce() function for implementation details.
        """

        # identify the atoms in the building unit that are bound to 
        self.connections = [c[0][0] for c in connectivity]
        self.g = self._make_graph(indices, symbols, bonds)

    
    @property
    def cycles(self):
        """
        Get all cycles in building unit. Does not perform well if there is 
        disordered sites.
        """
        return all_cycles(self.g)


    @property
    def simple_cycles(self):
        """
        Get all simple cycles (i.e. cycles that cannot be broken down into
        smaller cycles).
        """

        # get all cycles.
        simple_cycles = self.cycles
        
        # test all cycles are simple.
        for c in simple_cycles:
            
            # make a candidate simple cycle by only
            # keeping nodes of that graph.
            sg = self.g.copy()
            sg.remove_nodes_from(set(self.g.nodes)-set(c))
            
            # try decomposing cycle to smaller cycles.
            # if can, not simple cycle so remove.
            if len(all_cycles(sg)) > 1:
                simple_cycles.remove(c)
                
        return simple_cycles

    
    @property
    def contracted_cycles(self):
        """
        Create a cycle-contracted graph whereby all simple cycles are contracted
        to a single point (centroid of that cycle).
        """

        # create a cycle-contracted graph.
        ccg = self.g.copy()

        # label "contracted nodes" with cX where X is an integer. 
        c = [int(x[1:]) for x in ccg.nodes if type(x) == str and x[0] == "c"]
        c = max(c) + 1 if c else 1

        contracted_cycles = {}
        for i,cycle in enumerate(self.simple_cycles):
            
            # get new name of node.
            node = f"c{i+c}"
            
            # get cycle edges.
            cycle_edges = ([item for item in x] for x in ccg.edges 
                            if any(item in cycle for item in x))
            
            # remove edges of cycle and get other node.
            connected_nodes = [(set(x)-set(cycle)).pop() for x in cycle_edges 
                                if not set(cycle).issuperset((set(x)))]
            
            # store contracted cycle information.
            contracted_cycles[node] = {"nodes":cycle, "c_nodes":connected_nodes}

        # then check for linked cycles.
        contracted_edges = []
        for c1, c2 in itertools.combinations(contracted_cycles.items(), r=2):
            if any(item in c1[1]["nodes"] for item in c2[1]["c_nodes"]):
                contracted_edges.append((c1[0],c2[0]))
        
        # then add new edges to cX nodes (and between cX nodes).
        for cX, info in contracted_cycles.items():
            contracted_edges += ((cX,n) for n in info["c_nodes"])

        # remove old cycle nodes and add new edges.
        ccg.add_edges_from(contracted_edges)
        ccg.remove_nodes_from(itertools.chain.from_iterable([c["nodes"] 
                            for c in contracted_cycles.values()]))

        # update labels with dummy symbol "X".
        for node in ccg.nodes:
            if type(node) == str and node.startswith("c"):
                ccg.nodes[node]["symbol"] = "X"

        # return the NetworkX graph and the information about the original
        # indices of that graph.
        return ccg, contracted_cycles
                
    
    def middle_shortest_path(self, n1, n2, useCycles=False):
        """
        Find the node index/indices at the middle of the shortest path(s) 
        between node1 (n1) and node2 (n2).

        Args
        ----
        n1: int or str
            Label of node 1.

        n2: int or str
            Label of node 2.

        useCycles: bool
            If True, first contract all cycles in graph to single node and then
            search for shortest path.
        """

        useGraph = self.g

        ix_map = {n:n for n in useGraph.nodes}

        if useCycles:
            
            # get cycle-contracted graph (ccg) and information on what nodes
            # make up the cycles and what they were connected to.
            useGraph, ccg_info = self.contracted_cycles

            # add indices of original cycle atoms to ix_map so that if a ring
            # centroid is needed, the centroid is over all atoms in that ring.
            for cX, nodes in ccg_info.items():
                ix_map[cX] = nodes["nodes"]

            # need to redefine n1 and n2 if they were a member of a cycle to the
            # name of that contracted cycle.
            for cX, nodes in ccg_info.items():
                if n1 in nodes["nodes"]:
                    n1 = cX
                if n2 in nodes["nodes"]:
                    n2 = cX

        return [ np.ravel( [ix_map[n] for n in middle_element(p)] ) 
                for p in all_shortest_paths(useGraph, n1, n2) ]

    
    def _make_graph(self, indices, symbols, bonds):
        """
        Construct NetworkX graph from atom indices, symbols, and bonds.

        Args
        ----
        indices: list
            Indices of atoms in building unit.

        symbols: list
            Chemical symbols in building unit.

        bonds: list
            Internal building unit bonds (pairs of atom indices)
        """

        # construct graph from bonds. each node is labelled with the atom index.
        g = nx.Graph()
        g.add_nodes_from(indices)
        g.add_edges_from(bonds)

        # add chemical symbol to each node.
        for ix, s in zip(indices, symbols):
            g.nodes[ix]["symbol"] = s

        return g



def all_cycles(g):
    """
    Get all cycles in graph. Based on algorithm by Joe Jordan (GitHub: 
    https://gist.github.com/joe-jordan/6548029). Operates as an NP-complete
    depth-first search, so will not scale well with graph size. Here, assume
    graph is fully connected.

    Args
    ---
    g: networkx.Graph
        Graph to search for cycles.
    """
    
    # get a node from graph.
    start = list(g.nodes)[0]
    
    cycle_stack = []
    output_cycles = set()

    cycle_stack.append(start)
    
    # create stack with starting indice, and iterator through bonds.
    stack = [(start,iter(g[start]))]

    while stack:
        
        # get the most recent parent node and iterate over all its bonded nodes.
        parent, children = stack[-1]

        try:
            
            # get next bonded node to parent node. will "fail" when no more
            # neighbours are found bonded to the parent.
            child = next(children)
            
            # if the child is not already in the cycle, add it to the list.
            if child not in cycle_stack:
                
                # add child to list, and search over child's bonded nodes in 
                # next iteration.
                cycle_stack.append(child)
                stack.append((child,iter(g[child])))

            else:
                
                # possible cycle found.
                # if node in list already, get index of child.
                i = cycle_stack.index(child)
                
                # requires at least 3 members to be a ring (i.e. not just 
                # returning bond in opposite direction).
                if i < len(cycle_stack) - 2:
                    
                    # get cycle.
                    output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

        except StopIteration:
            
            # otherwise probably run out nodes to search for cycles.
            stack.pop()
            cycle_stack.pop()
    
    return [list(i) for i in output_cycles]


def get_hashable_cycle(cycle):
    """
    Cycle as a tuple in a deterministic order.

    Args
    ----
    cycle: list
        List of node labels in cycle.
    """
    
    # get index of minimum index in cycle.
    m = min(cycle)
    mi = cycle.index(m)
    
    mi_plus_1 = (mi + 1) if (mi < len(cycle) - 1) else 0
    
    if cycle[mi-1] > cycle[mi_plus_1]:
        result = cycle[mi:] + cycle[:mi]
    else:
        result = list(reversed(cycle[:mi_plus_1])) + \
                    list(reversed(cycle[mi_plus_1:]))
        
    return tuple(result)

from typing import Tuple, Dict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from networkx import all_shortest_paths
from chic.atomic_cluster import AtomicCluster

class DihedralAnalysis:
    """
    Identifies the carbonate groups on a 1,3-BDC linker and computes the
    dihedral angles between the carbonate and the ring. Assumes the molecular
    graph for the building unit is reasonable (CrystalNN neighbour list is 
    likely more reliable for this task).

    Attributes:
        None

    Methods:
        get_coord_by_ix(unit: AtomicCluster, ix: int) -> np.ndarray: Retrieves
            coordinate by index.
        compute_dihedral(A, B, C, D) -> float: Computes the dihedral angle 
            between points A, B, C, and D.
        analyse_ligand(unit: AtomicCluster) -> Dict[str, np.ndarray]: Analyses
            ligands in the building unit, computing dihedrals for carbonates.

    Inner Classes:
        Carbonate: Represents a carbonate group with methods to check carbon
            presence and compute dihedrals.
    """

    @staticmethod
    def get_coord_by_ix(unit: AtomicCluster, ix: int) -> np.ndarray:
        """
        Retrieves coordinate by index.

        Arguments:
            unit (AtomicCluster): The building unit.
            ix (int): The index to retrieve.
        
        Returns:
            np.ndarray: The coordinate.
        """
        ix_ = np.argwhere(np.array(unit.site_indices, dtype=int) == ix)[0][0]
        return unit.cart_coords[ix_]

    @staticmethod
    def compute_dihedral(
        A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray
    ) -> float:
        """
        Computes the dihedral angle between points A, B, C, and D.

        Arguments:
            A (np.ndarray): The first point.
            B (np.ndarray): The second point.
            C (np.ndarray): The third point.
            D (np.ndarray): The fourth point.

        Returns:
            float: The dihedral angle in degrees.
        """
        BA, BC = A - B, C - B
        CB, CD = -BC, D - C
        n1, n2 = np.cross(BA, BC), np.cross(CB, CD)
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        d = np.dot(n1, n2)
        sign = 1 if np.dot(np.cross(BA, CD), BC) >= 0 else -1
        return sign * np.arccos(d) * 180 / np.pi

    @dataclass
    class Carbonate:
        """
        Represents a carbonate group with methods to check carbon presence and 
        compute dihedrals.

                            C(3)           O(1)
                                \           /
                                C(2) --- C(1)
                                            \
                                            O(2)

        Attributes:
            o1 (int): The index of the first oxygen atom.
            c1 (int): The index of the first carbon atom.
            o2 (int): The index of the second oxygen atom.
            c2 (int): The index of the second carbon atom.
            c3 (int): The index of the third carbon atom.

        Methods:
            has_c(ix: int) -> bool: Checks if given index ix is a carbon in the 
                carbonate.
            dihedrals(bu: AtomicCluster) -> Tuple[float, float]: Computes and 
                returns dihedral angles for the carbonate based on building unit 
                provided.
        """
        o1: int
        c1: int
        o2: int
        c2: int = None
        c3: int = None

        def has_c(self, ix: int) -> bool:
            """ 
            Checks if given index ix is a carbon in the carbonate.

            Arguments:
                ix (int): The index to check.

            Returns:
                bool: True if the index is a carbon in the carbonate.
            """
            return ix==self.c1

        def dihedrals(self, unit: AtomicCluster) -> Tuple[float, float]:
            """
            Computes and returns dihedral angles for the carbonate based on 
            building unit provided.

            Arguments:
                unit (AtomicCluster): The building unit.

            Returns:
                Tuple[float, float]: The dihedral angles in degrees.
            """
            coords = lambda idx: DihedralAnalysis.get_coord_by_ix(unit, idx)
            dh1 = DihedralAnalysis.compute_dihedral(
                coords(self.o1), coords(self.c1), 
                coords(self.c2), coords(self.c3)
            )
            dh2 = DihedralAnalysis.compute_dihedral(
                coords(self.o2), coords(self.c1), 
                coords(self.c2), coords(self.c3)
            )
            return dh1, dh2

    def analyse_ligand(self, unit: AtomicCluster) -> Dict[str, np.ndarray]:
        """
        Analyzes ligands in a given building unit, computing dihedrals for 
        carbonates.

        Arguments:
            unit (AtomicCluster): The building unit.

        Returns:
            Dict[str, np.ndarray]: A dictionary with dihedral angles for each 
                carbonate.
        """

        # get unique combinations of oxygen atoms to search for paths between.
        oxygen_combinations = combinations([
            x for s,x in zip(unit._symbols,unit.site_indices) if s=='O'], r=2
        )

        backbone = []
        carbonates = []
        for n1, n2 in oxygen_combinations:

            # access the molecular graph automatically found by chic.
            p = list(all_shortest_paths(unit.graph, n1, n2))

            if len(p) == 1:
                p = p[0]

                # this is the path between oxygens on the same carbonate group.
                if len(p) == 3:
                    carbonates.append(self.Carbonate(*p))

                # this is a linking path between the two carbonte groups.
                else:
                    backbone.append(tuple(p[1:-1]))
        
        # each 1,3-BDC ligand should have two carbnate groups, and one obvious
        # shortest path passing through C(2) of the ring.
        assert len(carbonates)==2 and len(set(backbone))==1, "Analysis failed."
        backbone = list(backbone)[0]

        # Assign additional carbon indices based on backbone analysis
        for carbonate in carbonates:
            if carbonate.has_c(backbone[0]):
                carbonate.c2, carbonate.c3 = backbone[1], backbone[2]
            else:
                carbonate.c2, carbonate.c3 = backbone[-2], backbone[-3]

        # Compute dihedrals for each carbonate
        dihedrals = [carbonate.dihedrals(unit) for carbonate in carbonates]
        dihedrals = [sorted(d, key=abs, reverse=True) for d in dihedrals]

        return {
            'carboxylate_a': np.round(dihedrals[0],3), 
            'carboxylate_b': np.round(dihedrals[1],3)
        }
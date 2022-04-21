"""
01.06.21
@tcnicholas
Main structure processing module.

References:
[1] Preprint available on ChemRxiv: 
    https://www.doi.org/10.33774/chemrxiv-2021-bdkwx
[2] Chem. Sci., 2020, 11, 12580
"""


from .cif import *
from .lammpsdata import *
from .sites import *
from .utils import *
from .disorder import *
from .local_env import *
from .cg import *

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from timeit import default_timer as timer
from itertools import chain
from pathlib import Path
import multiprocessing
import warnings
import sys


class Structure:
    """
    Main structure analysing class. Used for coarse-graining structures
    and determining the bonding.

    Reads a CIF and allows you to process them using class methods.
    """

    def __init__(self, filePath: str, sites=None, method="mof", cores=None,
        dataFormat="cif", primitive=False):
        """
        Args
        ----
        filePath: str
            path to input file.

        sites: list
            specifying atoms in each site-type. One list per site-type. I.e. for
            ZIF-8 (Zn(mIm)2) Zn is an A site, and the C, N, H (imidazolate ring)
            are B sites, so you would pass:

                sites = [["Zn"], ["C", "N", "H"]]

        method: str   
            name of sort_sites() algorithm to use (in sites.py). New custom
            methods should be added to the site.py module so that they are
            callable from the main Structure module.
                        
            (i) "mof": developed for work in Ref 1. Works well for a wide range
                of AB2 MOF structures.

            (ii) "allAtoms": makes each element in structure into its own
                site-type class. e.g. for Tb(HCO2)3, the resultant sites would
                be: 
                
                [["Tb"], ["H"], ["C"], ["O"]] 
                
                (not necessarily in that order).

        """
        self.filePath = filePath
        self.units = {}
        
        if dataFormat=="cif":
            self.atoms, self.info, _ = read_cif(filePath, primitive)
        elif dataFormat=="lammps-data":
            self.atoms, _, self.info = read_lammps(filePath)
        else:
            warnings.warn(f"File format {dataFormat} unrecongnised.")
            
        self._sites = sort_sites(self.atoms, method) if sites is None else sites
        self.cg_method = None
        self.cores = multiprocessing.cpu_count() if cores is None else cores

    
    def repair_disorder(self, method="poreOxygen", **kwargs):
        """
        Calls various functions written to resolve occupational disorder into
        discrete positions. Several routines are implemented to remove different
        types of disorder. Modifies the Pymatgen.Structure object "self.atoms".
        For more details, see the documentation for "disorder.py".

        Updates self.atoms class attribute. Also updates the atom indices and so
        any 'reduce' and 'coarse-grain' anaylsis will be deleted.

        Args
        ----
        method  :   str

            (i)     "poreOxygen": finds oxygen atoms with only O-O bonds and
                    deletes them.

            (ii)    "pairwiseElement": finds where two atoms of the same atomic
                    species lie within a cut-off radius (0--0.7 Å by default)
                    and deletes them, and places the same atomic species at the
                    average position. Takes the following kwargs:

                    element: str (e.g. "O")
                    cutoff: float or tuple e.g. 1.0 (= 0.0--1.0) or (0.0, 1.0)

            (iii)   "loneElement": finds atoms with no nearest neighbours within
                    a cut-off radius and deletes them.
        """

        # If make changes to number of atoms, also require the reduce() is
        # re-calculated.
        if len(self.units) > 0:
            warnings.warn(
            "Modifying the number of atoms will invalidate the atom indices" + \
            " for the building units. Deleting units list! Call reduce() again!"
            )
            self.units = {}

        # Apply algorithm.
        self.atoms = eval(f"{method}(self.atoms,**kwargs)")
        
        # If all occurences of an element are removed, also remove the element
        # from the sites categories.
        es = set(np.concatenate(self.sites))
        ne = {a.symbol for a in self.atoms.composition.elements}
        rmv = es - ne

        # Remove from sites lists.
        for s in reversed(range(len(self.sites))):
            for e in reversed(range(len(self.sites[s]))):
                if self.sites[s][e] in rmv:
                    del self.sites[s][e]

    
    def reduce(self, skipSites=None, intraWeight=0.2, interWeight=0.4,
                intraBond=None, interBond=None, cLimit=None):
        """
        Iterates through each site-type and performs a "cluster-crawl" algorithm
        to identify the constituent atoms of each building block.

        Args
        ----
        skipSites: list
            List of "a", "b", "c", etc. sites to skip the cluster crawl
            algorithm for. Instead, any atoms of that site-type are kept as
            discrete positions.

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
        """
        
        # calculate neighbour list for all atoms.
        nn = nn_dict(self.atoms, elements=list(chain(*self.sites)), cores=self.cores)

        # time the reduce() method.
        start = timer()

        # determine which site-types to apply the cluster crawl algorithm for.
        if bool(skipSites):
            skip = siteType_to_idx(skipSites)
        else:
            skip = []
        
        # iterate through site-types and cluster crawl.
        for s in range(len(self.sites)):

            if cLimit is not None:
                cLim = cLimit[s]
            else:
                cLim = None

            # if skipping cluster-crawl for site-type, iterate through all atoms
            # in this category and create single atom building unit.
            if s in skip:

                # find building units for the site. Note the skipClusterCrawl is
                # True, which ensures every atom is identified as a separate
                # building unit.
                u = reduce(self.atoms, nn, self.sites, s, intraWeight,
                            interWeight, intraBond, interBond, cLim,
                            skipClusterCrawl=True)
                
            # otherwise, attempt a cluster-crawl.
            else:

                # find building units for the site.
                u = reduce(self.atoms, nn, self.sites, s, intraWeight,
                            interWeight, intraBond, interBond, cLim)

            # add to dictionary (in latest version of Python you can use the
            # '|' operator, however stick with version 3.5 for now).
            self.units = {**self.units, **u}

        # end timer and print report.
        t = timer() - start
        print(f"* reduce * Reduce algorithm took {t:.3f} s.")
        
    
    def extract_lammps_topology(self):
        """
        If the structure is a LAMMPS data format including topology information
        (e.g. ZIFs), one can extract all of the building units using the mol-id
        and bond information.
        """
        self.units = get_units_from_LAMMPS(self.filePath)


    def coarse_grain(self, method="centroid", minReqCN=2, **kwargs):
        """
        Get coarse-grained structure from building units. Updates the building
        unit properties with the discrete placeholder atoms for each unit (with
        the corresponding image) and the connectivity information to other
        building units.

        Args
        ----
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

        method = method.lower().strip()

        # time the coarse_grain() method.
        start = timer()

        # update building units with coarse-grained information.
        self.units = coarse_grain(  self.atoms,
                                    self.units, 
                                    method, 
                                    minReqCN,
                                    **kwargs    )
        
        # Keep track of which method was used.
        self.cg_method = method

        # end timer and print report.
        t = timer() - start
        print(f"* coarse-grain * Coarse-grain algorithm took {t:.3f} s.")

    
    def get_cg_atoms(self, scale=None, scaleValue=1.0, siteMap=None,
            keepSingleAtoms=False, package=None):
        """
        Create an "atoms" object for the coarse-grained structure. This can
        either be a Pymatgen "Structure" object, or an ASE "Atoms" object.

        Args
        ----
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

        assert self.cg_method is not None, "Structure must be coarse-grained!"

        if package is not None and package.lower() == "ase":
            
            # ASE does not support Dummy Species in the same way as Pymatgen,
            # therefore you need to specify an atomMap.
            assert siteMap is not None, "ASE does not accept dummy species." + \
                f" Provide a siteMap for each ({len(self.sites)}) dummy site." 

        if siteMap is None:

            # Pymatgen has strict criteria as to what is a suitable symbol for a 
            # Dummpy Species such that it is not confused with real atoms. For
            # simplicity, default to lettering i.e. dA, dB, dC... for site1,
            # site2, site3... etc.
            siteMap = [f"D{chr( ord('a')+i )}" for i in range(len(self.sites))]

        # Gather information required to make Pymatgen structure.
        s_info, s = cg_atoms(   self.atoms, self.units, self.sites, scale,
                                scaleValue, siteMap, keepSingleAtoms, package
                            )

        if package is not None:
            return s_info, s
        
        return s_info
        
    
    def atoms_html(self):
        """
        Get HTML string for visualising the current full-atomistic structure in
        an iPython environment.
        """
        
        # first convert Pymatgen Structure object into ASE Atoms object.
        aseAtoms = AseAtomsAdaptor().get_atoms(self.atoms)

        # then return the HTML string.
        return atoms2html(aseAtoms)


    def cg_atoms_html(self, scale=None, scaleValue=1.0, siteMap=None, 
            keepSingleAtoms=False):
        """
        Get HTML string for visualising the ccoarse-grained structure in an
        iPython environment.

        Args
        ----
        scale: str (default=None)
            Scaling method to be used. Currently supported:
                "min_xx": minimum bond length between any atoms.
                "min_ab": minimum bond length between building units.
                "avg_ab": average bond length between building units.
            
        scaleValue: float (default=1.0)
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
        """
        
        # get ASE Atoms object.
        _, aseAtoms = self.get_cg_atoms(scale=scale, scaleValue=scaleValue, 
            siteMap=siteMap, keepSingleAtoms=keepSingleAtoms, package="ase")

        # then return the HTML string.
        return atoms2html(aseAtoms)

    
    def write_cg_cif(self, fileName, scale=None, scaleValue=1.0, siteMap=None, 
            keepSingleAtoms=False, topoCIF=True):
        """
        Write the coarse-grained structure and building unit connectivity to a
        TopoCIF.

        Args
        ----
        fileName: str
            Name of output file

        scale: str (default=None)
            Scaling method to be used. Currently supported:
                "min_xx": minimum bond length between any atoms.
                "min_ab": minimum bond length between building units.
                "avg_ab": average bond length between building units.
            
        scaleValue: float (default=1.0)
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

        topoCIF: bool
            If True, will also append a topoCIF block to CIF containing topology
            information of how the building units are connected. Useful e.g. for
            identifying topology of the framework with ToposPro.
        """

        assert self.cg_method is not None,  "Structure must be coarse-grained!"

        from pymatgen.io.cif import CifWriter

        # use Pymatgen object to obtain basic analysis. by default here, set
        # siteMap to None so that we can define the mapping from the dummy
        # species used in the Pymatgen object.
        s_info, s = self.get_cg_atoms(  scale=scale, 
                                        scaleValue=scaleValue,
                                        siteMap=siteMap, 
                                        keepSingleAtoms=keepSingleAtoms,
                                        package="pymatgen" )

        # Create class for writing CIF.
        cif = writeCIF(Path(self.filePath).stem, s, *s_info.values(), self.info)

        # Write file.
        cif.write_file(fileName, topoCIF)


    def dump(self, fileName):
        """
        Write current structure to CIF.

        fileName: str
            Name of output file
        """

        #TODO: format this to work with custom WriteCIF.
        CifWriter(self.atoms).write_file(fileName+".cif")

    
    def overlay(self, fileName, siteMap, keepSingleAtoms=False,
            return_html=False):
        """
        Write both the coarse-grained and full atomistic structures to same file
        to visualise how the it has been coarse-grained.

        Args
        ----
        fileName: str
            Name of output file

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

        return_html: bool
            Creates HTML-formatted image of the overlaid structures for
            visualising in IPython environments.
        """

        assert self.cg_method is not None,  "Structure must be coarse-grained!"

        from ase.io import write

        # get full atomistic structure.
        s_full = AseAtomsAdaptor().get_atoms(self.atoms)

        # get un-scaled coarse-grained structure.
        _, s_cg = self.get_cg_atoms(    scale=None,
                                        siteMap=siteMap, 
                                        keepSingleAtoms=keepSingleAtoms,
                                        package="ase" )

        # add both ASE Atoms objects together.
        s = s_full + s_cg
        write(fileName, s, format="cif")

        # can return HTML formatted visualisation for iPython HTML, however it
        # is not very clear to see, especially when atoms are overlayed (e.g. 
        # using shortest_path coarse-graining). Preferable to look at CIF.
        if return_html:
            return atoms2html(s)

    
    @property
    def sites(self):
        """
        Elements in full atomistic structure sorted into site-types, e.g. in
        silica (SiO2), [["Si"],["O"]], where Si is an "a" site and oxygen is a
        "b" site.
        """
        return self._sites


    @sites.setter
    def sites(self, sites):
        """
        Setter method for specifying the sites at a later stage.

        Args
        ----
        sites: (list of lists) 
            specifying atoms in each site-type. One list per site-type. I.e. for
            ZIF-8 (Zn(mIm)2) Zn is an A site, and the C, N, H (imidazolate ring)
            are B sites, so you would pass: sites = [["Zn"], ["C", "N", "H"]]
        """
        self._sites = sites



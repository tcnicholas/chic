"""
28.07.21
@tcnicholas

Main module for constructing structure maps using SOAP descriptors for the 
atomic environments and dimensionality reduction algorithms to create the low
dimensional embeddings (typically 2 dimensions).

More on the SOAP descriptor may be found in the following two papers:
[1] Phys. Rev. B, 2013, 87, 184115 (10.1103/PhysRevB.87.184115)
[2] Phys. Chem. Chem. Phys., 2016, 18, 13754--13769 (10.1039/C6CP00415F)
"""

from .cif import *
from .structure import *
from .utils import *
from .local_env import *
from .soapMethods import *

from pymatgen.io.ase import AseAtomsAdaptor

from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import KernelPCA
from umap import UMAP

from timeit import default_timer as timer
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import warnings


class StructureMap:
    """
    Main class for creating structure maps using SOAP descriptors and
    dimensionality reduction algorithms. Each structure is added to the class as 
    a "dataPoint".
    """

    def __init__(self):
        # Store structure data.
        self.names = []
        self.data = []
        self.dirs = {}

        # Store parameter details.
        self._soap_parameters = None
        self._scaling = None
        self._normalise = None

        # Store kernel and embedding details.
        self._k = None
        self.embeddings = set()

    
    def add_directory(self, dirPath, sites=None, sitesMethod=None, 
            suppress_cif_warning=True):
        """
        Add all structures in directory to structure map. Assumes all data is in
        CIF format.

        Args
        ----
        dirPath: str
            path to directory.

        sites: list
            pre-defined sites for the structure, e.g. for SiO2, one would pass
            sites = [["Si"],["O"]], thereby setting the "a" sites as Si, and "b"
            sites as O.

        sitesMethod: str
            if sites not specified, choose method to automatically guess the
            sites (see sites.py for more details).

        supress_cif_warning: bool
            Pymatgen CIF parser does not recognise TopoCIF commands, or ToposPro
            specefic CIF tags, and so throws up errors. Can choose to hide/shows
            these.
        """

        # get all files in directory (and sub-directories). note: once movinng
        # to large datasets, the algorithms will produce quantitatively
        # different embeddings if e.g. the SOAP vectors are slightly changed, or
        # the order of structures is changed. to aim for some degree of exact
        # reproducibility, always sort the files by name.
        af = sorted(from_directory(dirPath, "cif"))
        print(f"* files * adding {len(af)} structures to map.")
        
        # iterate through all files in directory with correct format and add to
        # the structure data list.
        for f in af:
            self.add_structure(f, sites=sites, sitesMethod=sitesMethod, 
                                suppress_cif_warning=suppress_cif_warning)
        

    
    def add_structure(self, filePath, sites=None, sitesMethod=None, 
            suppress_cif_warning=True):
        """
        Add a single structure to map from filePath.

        Args
        ----
        filePath: str
            path to input file.

        sites: list
            pre-defined sites for the structure, e.g. for SiO2, pass

                sites = [["Si"],["O"]]

            thereby setting the "a" sites as Si, and "b" sites as O.

        sitesMethod: str
            if sites not specified, choose method to automatically guess the
            sites (see sites.py for more details).
        """
        if suppress_cif_warning:
            warnings.simplefilter("ignore")

        # for each structure file, create a dataPoint class.
        self.names.append(Path(filePath).stem)
        dp = dataPoint(filePath, sites=sites, sitesMethod=sitesMethod)
        self.data.append(dp)

        warnings.simplefilter("always")

    
    def scale_all(self, method="min_xx", scaleValue=1.0):
        """
        Scale all structures to a uniform characteristic bond length, defined by
        the different scaling methods.

        Args
        ----
        method: str
            Scaling method to be used. Currently supported:
                "min_xx": minimum bond length between any atoms.
                "min_ab": minimum bond length between building units.
                "avg_ab": average bond length between building units.
            
        scaleValue: float
            Length (Å) to scale the characteristic bond length (defined by
            "scale") to.
        """

        # if a kernel matrix is currently stored, wipe it to prevent mixing up
        # different representations and maps.
        self._k = None

        # iterate through all structures and re-scale them.
        for s in self.data:
            s.scale(method=method, scaleValue=scaleValue)

        # also store the scale method.
        self._scaling = method

    
    def normalise_all(self, siteMap):
        """
        Normalise all species in a given siteType to a single element, e.g. to
        turn all A site atoms to Si and all B site atoms to O, 
        siteMap = ["Si", "O"]. Note, because the structure is stored as an ASE
        Atoms object, the "dummy atoms" need to correspond to a real (albeit
        abitrary) element.

        Args
        ----
        siteMap: list
            A list of atoms to map each building unit to. Should be of the same
            length as the number of site-types. E.g. to map Zn(mIm)2 to a
            coarse-grained structure,

                siteMap = ["Si", "O"]

            would map all A sites (Zn) to Si, and all B sites (mIm) to O. If
            not set, will default to "Dummy Species" with labels DA, DB, DC, ...
            Note if creating an ASE Atoms object, real atoms must be used, and
            so siteMap *must* be set.
        """

        # if a kernel matrix is currently stored, wipe it to prevent mixing up
        # different representations and maps.
        self._k = None

        # iterate through all structures and normalise them.
        for s in self.data:
            s.normalise(siteMap=siteMap)

        # also store the normalising species.
        self._normalise = tuple(siteMap)


    
    def calc_soap(self, package="dscribe", atomic_numbers=None, periodic=True,
            sparse=False, rbf="polynomial"):
        """
        Calculate the SOAP vectors for all structures.

        Args
        ----
        package: str
            Name of package to use for calculating the SOAP vecotrs. Currently
            only DScribe is supported (QUIPPY to be added).

        atomic_numbers: list
            Atomic numbers to include in the SOAP analysis. All elements that
            will be encountered need to be included. If None, will just include
            all elements in the structure. Note, undesirable behaviour may
            occur if comparing structures with differnet species if not all
            elements are included for both structures.

        periodic: bool
            Whether to construct a perioidic SOAP.

        sparse:

        rbf: str
            Radial basis function to use ("poylnomial" or DScribe's custom "gto"
            basis set).
        """

        print(f"* soap * calculating SOAP vectors using '{package}'.")
        
        # Iterate through all structures, calculate SOAP power spectrum vector,
        # and store to dataPoint class.
        for s in self.data:
            s.calc_soap(self._soap_parameters, package, atomic_numbers, 
                periodic, sparse, rbf)

    
    def calc_kernel(self, method="a"):
        """
        Calculate SOAP similarity kernel matrix.

        Args
        ----
        method: str
            "a", "b", "all". Which site-type atoms to use in pairwise comparison
            between each cell.
        """

        assert np.all([True for s in self.data if s.soap is not None]), \
            "Calculate SOAP vectors for all structures first!"

        # get SOAP vectors from all structures and store in one list.
        # turn method into indices to extract atom indices from each dataPoint.
        if method.lower() == "all":
            all_soap = [s.soap for s in self.data]

        else:
            if len(method) == 1 and type(method) == str:
                ix = ord(method.lower()) - ord("a")

            elif en(method) > 1 and type(method) in (list, tuple, set):
                ix = [ord(i.lower()) - ord("a") for i in method]

            else:
                raise ValueError(f"Invalid arguement for method: {method}")

            all_soap = [s.soap_by_siteType(ix) for s in self.data]

        # print report to terminal.
        print(f"* kernel * calculating SOAP kernel matrix for {method}-sites.")

        # calculate the similarity kernel. this is sent to external function in 
        # soapMethods.py to use with numba njit decorator because this is quite
        # a slow operation.
        self._k = kernel(self._soap_parameters[4], all_soap)

    
    def dimRed(self, method="MDS", mapName=None, return_embedding=True, 
            n_components=2, **kwargs):
        """
        Perform dimensionality reduction using the specified embedding method.
        Currently support algorithms are "KPCA", "MDS", "TSNE", and "UMAP". The
        number of components (i.e. desired number of dimensions to reduce the
        data to is automatically set to 2, but other options may be chosen).

        For other keyword arguements (algorithm parameter settings), please
        refer to the documentation for each algorithm:
        
        Args
        ----
        method: str
            Which dimensionality reduciton algorithm to use. "KPCA", "MDS", 
            "TSNE", or "UMAP".

        mapName: str
            Optional name for the particular embedding.

        return_embedding: bool
            Return the embedding infomration as a embedding dataclass object. Is
            also stored in the StructureMap.embeddings attribute. Useful for
            IPython environments (see example 2).

        n_components: int
            Number of components to reduce the data to. I.e. for a 2D map,
            n_components=2, for a 3D map, n_components=3.


        Documentation and helpful resources:
        ------------------------------------
        k-PCA:
            scikit-learn documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

        MDS:
            scikit-learn documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

        TSNE:
            scikit-learn documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

            a useful resource advising on parameter settings and interpretation:
            https://distill.pub/2016/misread-tsne/

        UMAP:
            UMAP documentation:
            https://umap-learn.readthedocs.io/en/latest/parameters.html

        """

        assert self._k is not None, "Calculate kernel matrix first!"

        method = method.strip().upper()

        # inititiate embedding scheme algorithm.
        if method == "MDS":
            a = eval(f"{method}(dissimilarity='precomputed', **kwargs)")
        elif method == "KPCA":
            a = KernelPCA(n_components=n_components, kernel="precomputed", **kwargs)
        else:
            a = eval(f"{method}(metric='precomputed', **kwargs)")

        # extract coordinates.
        if method in ["MDS", "TSNE"]:
            c = a.fit(self.d).embedding_
        elif method == "UMAP":
            c = a.fit_transform(self.d)
        elif method == "KPCA":
            # note k-PCA takes the kernel form, not the distance matrix.
            c = a.fit_transform(self._k)
        
        # store coordinates in dataclass, along with information about the
        # scaling/soap_parameter/embedding_parameter settings.
        if mapName is not None and mapName in [m.name for m in self.maps]:
            raise ValueError(f"mapName '{mapName}' already used!")

        if mapName is None:
            mapName = f"map_{len(self.embeddings)+1:02d}"

        # create the dataclass.
        e = embedding(mapName, method, self._soap_parameters, self._scaling, c)

        # Store embedding in embeddings set.
        self.embeddings |= {e}

        # optionally return the embedding.
        if return_embedding:
            return e

    
    def get_dataPoint(self, name: str):
        """
        Get dataPoint object by name.
        """
        return self.data[self.names.index(name)]

    
    def get_soap_value(self, name1: str, name2: str, distance=False):
        """
        Find SOAP similarity value between two structures.

        Args
        ----
        name1: str
            Name of structure 1.
        
        name2: str
            Name of structure 2.

        distance: bool
            If True, returns the SOAP distance, rather than SOAP similarity.
        """

        # retrieve value from SOAP kernel.
        k = self.k[self.names.index(name1), self.names.index(name2)]

        # if distance is True, return the distance metric.
        if distance:
            return d_soap(k)
        return k
    

    def add_data_by_name(self, data):
        """
        A method that allows custom data to be added to each dataPoint object.
        
        Args
        ----
        data : dict

            A dictionary of {name: {"property1":value1, "property2":value2,...}}
            where "name" is the name of one of the dataPoints currently stored.
        """

        # keep track of any "unfound" dataPoints.
        not_found = []

        # iterate through each name in data dictionary and attempt to update the
        # dataPoint's info.
        for name, info in data.items():

            if name not in self.names:
                not_found.append(name)
                continue

            self.data[self.names.index(name)].data.update(info)

        # print report to terminal.
        added = len(data)-len(not_found)
        print(f"* data * Added data to {added} dataPoints.")

        # print names of structures that the data-appending failed for.
        if added < len(self.names):
            missed = len(self.names) - added
            print(f"* data * Data not added for {missed} dataPoints.")


    @property
    def soap_parameters(self):
        """
        Get SOAP parameters currently set.
        """
        return self.soap_parameters


    @soap_parameters.setter
    def soap_parameters(self, value):
        """
        Setter method for the SOAP parameters.

        Args
        ----
        value: list
            SOAP parameters (r, sigma, n, l, zeta).
            
            r: radial cut-off.
            sigma: broadness of the Gaussian functions (smoothness).
            n: order to which the radial basis set is expanded to.
            l: order to which the angular basis set is expanded to.
            zeta: power to which the normalised SOAP kernel is raised.
        """
        self._soap_parameters = tuple(value)
    

    @property
    def k(self):
        """
        Get the SOAP kernel (similarity) matrix.
        """
        return self._k

    
    @property
    def d(self):
        """
        Get the correspodning SOAP distance (dissimilarity) matrix.
        """
        assert self._k is not None, "Calculate kernel matrix first!"
        return d_soap(self._k)

    
    def __len__(self):
        """
        Set the length of the StructureMap as the number of datapoints.
        """
        return len(self.data)


class dataPoint:
    """
    A class to represent each structure in the map, storing both structural
    properties, and information on its position in the calculated structure
    maps.
    """

    def __init__(self, filePath, sites=None, sitesMethod=None):
        # name of file/dataPoint.
        self._name = Path(filePath).stem

        # structural properties stored for calculations.
        self._labels = None
        self._atoms = None
        self._atoms_pym = None 
        self._sites = None
        self._bonds = None
        self._bonds_raw = None
        self._getAtoms(filePath=filePath, sites=sites, sitesMethod=sitesMethod)

        # store parameters used.
        self._scaleFactor = 1
        self._soap_parameters = []
        self._soap = None

        # store additional data for dataPoint.
        self.data = {}

    
    def normalise(self, siteMap):
        """
        Normalise atoms in each site-type.

        Args
        ----
        siteMap: list
            A list of atoms to map each building unit to. Should be of the same
            length as the number of site-types. E.g. to map Zn(mIm)2 to a
            coarse-grained structure,

                siteMap = ["Si", "O"]

            would map all A sites (Zn) to Si, and all B sites (mIm) to O. If
            not set, will default to "Dummy Species" with labels DA, DB, DC, ...
            Note if creating an ASE Atoms object, real atoms must be used, and
            so siteMap *must* be set.
        """

        assert len(siteMap) == len(self._sites), "Provide a site for each " + \
            f"each site-type ({len(self.sites)} atoms required)!"

        # create dictionary to create new list of symbols.
        atom_map = {}
        for site, atom in zip(self._sites, siteMap):
            for e in site:
                atom_map[e] = atom

        # get all atomic symbols and create new list using the atom map.
        sym = self._atoms.get_chemical_symbols()
        new_sym = [atom_map[s] for s in sym]
        
        # update the atoms object and update sites list.
        self._atoms.set_chemical_symbols(new_sym)
        self._sites = [[s] for s in siteMap]

    
    def scale(self, method="min_xx", scaleValue=1.0):
        """
        Scale structures to acheive a uniform characterstic bond length across
        all structures.
        
        Args
        ----
        method: str
            Scaling method to be used. Currently supported:
                "min_xx": minimum bond length between any atoms.
                "min_ab": minimum bond length between building units.
                "avg_ab": average bond length between building units.
                "volume": achieve the specified volume in the unit cell.
            
        scaleValue: float
            Length (Å) to scale the characteristic bond length (defined by
            "scale") to.
        """

        method = method.lower()

        if method in ["min_ab", "avg_ab"]:

            assert self._bonds is not None, f"Cannot perform '{method}' " + \
                f"scaling method without bond information."

            # get bond lengths.
            lengths = self._bonds.lengths

            # calculate the scale factor according to method.
            if method == "min_ab":
                sf = scaleValue / lengths.min()
            else:
                sf = scaleValue / lengths.mean()

        elif method == "min_xx":

            # Get all distances (ignoring self-distances along diagonal).
            d = self._atoms.get_all_distances(mic=True, vector=False)
            np.fill_diagonal(d, 1000)
            
            # Get scale factor and scale the lattice to the new volume.
            sf = scaleValue / np.amin(d)
        
        elif method == "volume":
        
            # get current volume.
            v = self._atoms.get_volume()
            
            # get scale factor.
            sf = (scaleValue / v)**(1/3)
        
        # rescale the atoms cell and store scale factor to dataPoint attribute.
        self._atoms.set_cell(self._atoms.get_cell() * sf, scale_atoms=True)
        self._scaleFactor = sf

        # also try and scale bonds.
        if self._bonds is not None:
            self._scale_bonds(sf)


    def calc_soap(self, parameters, package="dscribe", atomic_numbers=None, 
            periodic=True, sparse=False, rbf="polynomial"):
        """
        Calculate SOAP vector and add to dataPoint class attributes.

        Args
        ----
        parameters: list
            SOAP parameters (r, sigma, n, l, zeta).
            
            r: radial cut-off.
            sigma: broadness of the Gaussian functions (smoothness).
            n: order to which the radial basis set is expanded to.
            l: order to which the angular basis set is expanded to.
            zeta: power to which the normalised SOAP kernel is raised.

        package: str
            Name of package to use for calculating the SOAP vecotrs. Currently
            only DScribe is supported (QUIPPY to be added).

        atomic_numbers: list
            Atomic numbers to include in the SOAP analysis. All elements that
            will be encountered need to be included. If None, will just include
            all elements in the structure. Note, undesirable behaviour may
            occur if comparing structures with differnet species if not all
            elements are included for both structures.

        periodic: bool
            Whether to construct a perioidic SOAP.

        sparse:

        rbf: str
            Radial basis function to use ("poylnomial" or DScribe's custom "gto"
            basis set).
        """

        # store parameters and calculate SOAP vector.
        self._soap_parameters = parameters
        self._soap = eval(f"calc_soap_{package}(self._atoms, parameters, " \
                            "atomic_numbers, periodic, sparse, rbf)")

    
    def ix_of_elements(self, elements):
        """
        Get indices of atoms in structure for a given site-type.

        Args
        ----
        siteType: list
            Atomic numbers of elements to get indices of from ASE Atoms object.
        """
        return [i for i,Z in enumerate(self._atoms.get_chemical_symbols()) 
                if Z in elements]


    def soap_by_siteType(self, siteType):
        """
        Return SOAP vectors if atom in specified siteType.

        Args
        ----
        siteType: int
            Index for siteType ("a" = 0, "b" = 1, etc.)
        """
        # convert sites into numpy array to allow list indexing.
        ix = self.ix_of_elements(np.array(self._sites)[siteType].flatten())
        return self._soap[ix]


    def _getAtoms(self, filePath, sites=None, sitesMethod="all"):
        """
        Reads CIF and creates ASE Atoms object, guesses site-types, and if
        possible, retrieves bond information.

        Args
        ----
        filePath: str
            path to input file.

        sites: list
            pre-defined sites for the structure, e.g. for SiO2, pass

                sites = [["Si"],["O"]]

            thereby setting the "a" sites as Si, and "b" sites as O.

        sitesMethod: str
            if sites not specified, choose method to automatically guess the
            sites (see sites.py for more details).
        """
        # extract structure, structure information and bonds.
        s, info, bonds = read_cif(str(filePath))

        # store pymatgen structure object too (this is probably redundant but
        # used currently as a way of re-scaling the bonds/bond vectors. Need to
        # check if the AseAtomsAdaptor preserves atom indices/absolute frac
        # coords etc.).
        self._atoms_pym = s
        self._bonds_raw = bonds

        # convert Pymatgen to ASE Atoms.
        self._atoms = AseAtomsAdaptor.get_atoms(s)

        # sort the site types.
        self._sites = sort_sites(s, sitesMethod) if sites is None else sites

        # store original atom labels from TopoCIF (matched up to Pymatgen
        # Structure object {see cif.py}).
        if bonds is not None:

            labels = [a.properties["label"] for a in s]
            self._labels = labels

            # process bond information.
            self._bonds = bonding(s, labels, bonds)

    
    def geometric_density(self, siteType, scaled=True):
        """
        Calculate the geometric density. Defined as:
        N_X * 1000 / V, where N_X is the number of sites of type X in the unit
        cell, and V is the volume of the (scaled) unit cell.

        Args
        ----
        siteType: int or list
            Index of site type(s) to use for geometric density. e.g. for A site
            denstiy, pass 0, for B site density, pass 1, for both, pass [0,1].

        scaled: bool
            Whether or not to get the scaled (current) density metric, or the
            original density (pre-scaling). Note, if the structure is not scaled
            the values will be identical.
        """

        # get indices of elements with the given siteType(s)
        ix = self.ix_of_elements(np.array(self._sites)[siteType].flatten())

        # if scaled, return the current unit cell volume (accounting for the 
        # re-scaling of the unit cell); i.e. do nothing. Else, re-scale the
        # density to get the original cell volume before scaling.
        sf = 1.0
        if not scaled:
            sf = self._scaleFactor**3
        
        return len(ix) * 1000 / (self._atoms.cell.volume / sf)

    
    def heterogeneity(self, siteType, zeta):
        """
        Calculate the SOAP heterogeneity value for a given site type. Defined as
        the average SOAP distance between all X sites within a given structure. 
        A larger heterogeneity value indicates greater variation in local
        geometry about the given site.

        Args
        ----
        siteType: int or list
            Index of site type(s) to use for the heterogeneity calculation, e.g.
            for A site heterogeneity, pass 0, for B site heterogeneity, pass 1, 
            for a "general" heterogeneity, pass [0,1].

        zeta: int
            Power to raise the SOAP kernel to.
        """

        assert self._soap is not None, "Require SOAP vectors to be calculated" \
            "to calculate SOAP heterogeneity!"

        # get indices of elements with the given siteType(s)
        soap_vectors = self.soap_by_siteType(siteType)

        # outsource calculation to soapMethods.py
        return heterogeneity(zeta, soap_vectors)

    
    def _scale_bonds(self, scaleValue):
        """
        If the atoms object is scaled, also recompute the bond vectors etc.

        Args
        ----
        scaleValue: float
            Scale factor to multiply all lattice vectors by. Note this will be
            converted into the scale factor to scale the Pymatgen structure
            Lattice volume by.
        """

        new_volume = self._atoms_pym.lattice.volume * (scaleValue**3)
        self._atoms_pym.lattice = self._atoms_pym.lattice.scale(new_volume)

        # process bond information.
        self._bonds = bonding(self._atoms_pym, self._labels, self._bonds_raw)
        
    

    @property
    def name(self):
        """
        Name assigned to datapoint (stem of the file path by default).
        """
        return self._name
    

    @name.setter
    def name(self, value):
        """
        Setter method for assigning new name to datapoint.
        """
        self._name = value


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
        sites       :   (list of lists) specifying atoms in each site-type. One
                        list per site-type. I.e. for ZIF-8 (Zn(mIm)2) Zn is an A
                        site, and the C, N, H (imidazolate ring) are B sites, so
                        you would pass:

                        sites = [["Zn"], ["C", "N", "H"]]
        """
        self._sites = sites
        
    
    @property
    def atoms(self):
        return self._atoms

    
    @property
    def bonds(self):
        return self._bonds

    
    @property
    def labels(self):
        return self._labels

    
    @property
    def soap(self):
        """
        Get the SOAP vector for this structure.
        """
        return self._soap
    

    @soap.setter
    def soap(self, value):
        """
        Set the SOAP vector for this structure. It may be desirable to set the
        SOAP vectors using pre-computed values. Care should be taken to ensure
        all SOAP vectors are of the correct/appropriate length with comparable
        parameters.
        """
        self._soap = value
    

    def __repr__(self):
        """
        Neater formatting.
        """
        return f"dataPoint({self._name})"


@dataclass(frozen=True)
class embedding:
    """
    Store the structure map data and settings in a dataclass. This enables the
    user to calculate multiple embeddings with different settings. Is a frozen
    dataclass so that only one copy of data stored.
    """
    name: str = field(compare=False)
    method: str
    soap_p: tuple
    scaling_method: str
    coords: np.ndarray = field(compare=False, repr=False)

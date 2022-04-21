"""
01.06.21
@tcnicholas
Input/output functions for CIFs.
"""

from .utils import *

from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element

from pathlib import Path
import numpy as np
import warnings


# All tags for defining bonds in TopoCIFs. For more details, see:
# https://www.iucr.org/resources/cif/dictionaries/cif_topology
topo_tags = [    '_topol_link.node_label_1',
                    '_topol_link.node_label_2',
                    '_topol_link.distance',
                    '_topol_link.site_symmetry_symop_1',
                    '_topol_link.site_symmetry_translation_1_x',
                    '_topol_link.site_symmetry_translation_1_y',
                    '_topol_link.site_symmetry_translation_1_z',
                    '_topol_link.site_symmetry_symop_2',
                    '_topol_link.site_symmetry_translation_2_x',
                    '_topol_link.site_symmetry_translation_2_y',
                    '_topol_link.site_symmetry_translation_2_z',
                    '_topol_link.type',
                    '_topol_link.multiplicity'
]

frac_tags = [ '_atom_site_fract_x',
                    '_atom_site_fract_y',
                    '_atom_site_fract_z'
]

# Also extract useful CIF information to retain in output configuration.
info_tags =  [      '_citation_doi',
                    '_citation_year',
                    '_chemical_name_common',
                    #'_chemical_formula_sum',
]


def match_cif_pym_atoms(atoms, structure):
    """
    Match up the atom labels from CIF to the atoms in the Pymatgen Structure
    object. This enables bond information stored in TopoCIF to be related to
    specific atoms in structure.
    """

    # get CIF labels with numbers.
    cif_labels = list(atoms.keys())

    # create tuple(symbol, factional coordinates) for each atom in both formats.
    cif_atoms = [(no_digit(s),tuple(np.round(fc,5))) for s,fc in atoms.items()]
    pym_atoms = [(x.symbol,tuple(np.round(y,5))) 
                    for x,y in zip(structure.species,structure.frac_coords)]

    # create empty array for labels in Pymatgen structure.
    pym_labels = np.empty(len(pym_atoms), dtype="<U10")
    for i, a in enumerate(cif_atoms):

        # get index in Pymatgen Structure that corresponds to this atom.
        ix = [j for j,b in enumerate(pym_atoms) if np.all(a==b)]

        assert ix, "No match found!"
        assert len(ix) == 1, "Duplicate atoms found!"

        # fill in labels array.
        pym_labels[ix[0]] = cif_labels[i]
    
    return pym_labels


def extract_topo(structure, cif_dict):
    """
    Attempt to extract topoCIF data.
    """

    if not all(x in cif_dict.keys() for x in topo_tags):
        return structure, None

    # Extract fractional coordinates from CIF.
    cs = [np.array(x,dtype=np.float64) 
            for x in zip(*[cif_dict[t] for t in frac_tags])]

    # Convert into {atomLabel:coordinate} dictionary.
    atoms = { aL:c for aL,c in zip(cif_dict["_atom_site_label"],cs) }

    # Parse bond information from topoCIF. This will be given in a format as
    # set out in "topo_tags" above.
    bonds = parseBonds([cif_dict[t] for t in topo_tags])

    # Match up atom species in Pymatgen Structure and labels from CIF.
    if True:

        # get labels.
        pym_labels = match_cif_pym_atoms(atoms, structure)

        # assign labels to property of atoms in structure.
        for i,a in enumerate(structure):
            a.properties["label"] = pym_labels[i]

    else:
        warnings.warn(f"Failed to match up Pymatgen-TopoCIF atoms.")
        pym_labels = None

    return structure, bonds


def read_cif(filePath: str, primitive=False):
    """
    Read CIF file with Pymatgen CifParser.

    Args
    ----
    filePath    :   path to input structure.
    """

    # Parse the input file using Pymatgen.
    p = CifParser(Path(filePath), occupancy_tolerance=100, site_tolerance=0)
    pDict = p.as_dict()[list(p.as_dict().keys())[0]]

    # Get extra info from CIF.
    info = fetch_info(pDict)

    # Create Pymatgen "Structure" object. At the moment, only supporting one
    # file at a time, and therefore this should just receive the one and only
    # input structure.
    #TODO: adapt all reads to cycle through all structures in catenated CIF.
    s = p.get_structures(primitive=primitive)[0]

    # Iterate through all atoms, a, in the structure and set occupancy to 1.
    for i,a in enumerate(s):

        d = a.species.as_dict()
        e = list(d.keys())[0]
        
        # Test if occupancy is not unity.
        if d[e] != 1:
            s.replace(i,Element(e))

    # Remove potential "duplicate" atoms found within 0.1 Å of one-another.
    s.merge_sites(tol=0.1, mode="delete")

    # Then get a sorted structure (by electronegativity).
    s = s.get_sorted_structure()

    # Occasionally deuterated structures are give, however some routines require
    # protons only. Automatically convert D --> H.
    ds = [i for i,a in enumerate(s) if a.specie.symbol == "D"]

    for d in ds:
        s.replace(d,Element("H"))#, properties={"index" : d})

    # Consider parsing TopoCIF data. This mainly functions well for structures
    # already processed by this code, thereby achieving interal consistency.

    # Require all CIF "tags" to be present in file.
    cif_dict = [x for x in p.as_dict().values()][0]
    s, bonds = extract_topo(s, cif_dict)

    return s, info, bonds


class writeCIF:
    """
    Write structure to CIF.
    """
    def __init__(
        self,
        name,
        structure,
        lattice,
        symbols,
        labels,
        frac_coords,
        bonds,
        info = {},
        topos = False,
    ):
        """
        Args
        ----

        """
        self.name = name
        self.structure = structure
        self.lattice = lattice
        self.symbols = symbols
        self.labels = labels
        self.frac_coords = np.array(frac_coords)
        self.bonds = bonds
        self.info = info
        self.topos = topos
        self.modify = {}
        self._check_frac_coords_wrapping(frac_coords)
        

    
    def write_file(self, fileName, bonds=True):
        """
        Iterate through CIF loops and write to file.
        """

        sections = ["header", "cellLoop", "positionsLoop"]

        if bonds:
            sections.append("bondsLoop")
        
        f = "".join((getattr(self,f"_{s}")() for s in sections))
        f += f"\n#End of data_{self.name}\n\n"

        with open(Path(fileName),"w+") as w:
            w.writelines(f)
            
    
    def _check_frac_coords_wrapping(self, frac_coords):
        """
        Sometimes fractional coordinates are stored at 1.0 rather than 0.0,
        which can cause problems when re-interpreting the bond information later
        or in other programs (e.g. ToposPro).
        """
        for i, (label, coord) in enumerate(zip(self.labels, self.frac_coords)):
            if np.any(np.round(coord,8) != np.round(coord,8)%1):
                new_frac = np.round(coord,8)%1 # wrap [0,1)
                adjust = new_frac-np.round(coord,8) # vector adjustment for bonds.
                self.frac_coords[i,:] = new_frac
                self.modify[label] = adjust
        
        new_bonds = []
        for i, (l1, l2, d, img1, img2) in enumerate(self.bonds):
        
            # determine how much the second node image needs adjusting.
            tot_adjust = np.zeros(3)
            if l1 in self.modify:
                tot_adjust += self.modify[l1]
            if l2 in self.modify:
                tot_adjust -= self.modify[l2]
            
            # compute new image for node2. assume node1 remains seated within
            # the unit cell.
            n_img2 = (np.array(img2) + tot_adjust).astype(np.int64)
            
            # check the distance is ok.
            c = self.lattice.get_cartesian_coords(
                [self.frac_coords[self.labels.index(l1)],
                self.frac_coords[self.labels.index(l2)]+n_img2])
            n_d = np.linalg.norm(c[1]-c[0])
            assert np.round(n_d,4) == np.round(float(d),4), f"old: {d} new: {n_d}"
            new_bonds.append([l1,l2,n_d,img1,tuple(n_img2)])
            
        self.bonds = new_bonds
            
    
    def _header(self):
        """
        Header of file string.
        """

        h = f"data_{self.name}\n"

        if self.topos is True:
            l = 60 - len("_database_code_TOPOS")
            h += f"_database_code_TOPOS {self.name:>{l}}\n"

        for tag in info_tags:
            if tag in self.info:
                l = int(60 - len(tag))
                h += f"{tag} {self.info[tag]:>{l}}\n"

        form = f"'{self.structure.composition.anonymized_formula}'"

        l = 60 - len("_chemical_formula_sum")
        h += f"_chemical_formula_sum {form:>{l}}\n"
        #h += f";\nFile: {self.name}.\n;\n"

        return h


    def _cellLoop(self):
        """
        Write unit cell loop to file string.
        """
        # Get (a,b,c,alpha,beta,gamma), cell volume, and number of formula units
        cellParams = np.ravel(self.lattice.parameters)
        cellVolume = self.lattice.volume
        _, Z = self.structure.composition.get_reduced_composition_and_factor()

        c = f"_cell_length_a\t\t\t{cellParams[0]:.5f}\n" \
            f"_cell_length_b\t\t\t{cellParams[1]:.5f}\n" \
            f"_cell_length_c\t\t\t{cellParams[2]:.5f}\n" \
            f"_cell_angle_alpha\t\t{cellParams[3]:.5f}\n" \
            f"_cell_angle_beta\t\t{cellParams[4]:.5f}\n" \
            f"_cell_angle_gamma\t\t{cellParams[5]:.5f}\n" \
            f"_cell_volume\t\t\t{cellVolume:.5f}\n" \
            f"_cell_formula_units_Z\t\t{int(Z)}\n" \
            "_symmetry_space_group_name_H-M\t'P 1'\n" \
            "_symmetry_Int_Tables_number\t1\n" \
            "loop_\n" \
            "_symmetry_equiv_pos_site_id\n" \
            "_symmetry_equiv_pos_as_xyz\n" \
            "1 x,y,z\n"

        return c


    def _positionsLoop(self):
        """ Write atom positions loop. """

        p = "loop_\n" \
            "_atom_site_label\n" \
            "_atom_site_type_symbol\n" \
            "_atom_site_symmetry_multiplicity\n" \
            "_atom_site_fract_x\n" \
            "_atom_site_fract_y\n" \
            "_atom_site_fract_z\n" \
            "_atom_site_occupancy\n"

        return p + coords2str(self.labels,self.symbols,self.frac_coords) + "\n"

    
    def _bondsLoop(self):
        """ Writes bond connectivity loop. """

        b = "loop_\n" \
            "_topol_link.node_label_1\n" \
            "_topol_link.node_label_2\n" \
            "_topol_link.distance\n" \
            "_topol_link.site_symmetry_symop_1\n" \
            "_topol_link.site_symmetry_translation_1_x\n" \
            "_topol_link.site_symmetry_translation_1_y\n" \
            "_topol_link.site_symmetry_translation_1_z\n" \
            "_topol_link.site_symmetry_symop_2\n" \
            "_topol_link.site_symmetry_translation_2_x\n" \
            "_topol_link.site_symmetry_translation_2_y\n" \
            "_topol_link.site_symmetry_translation_2_z\n" \
            "_topol_link.type\n" \
            "_topol_link.multiplicity\n"

        return b + bonds2str(self.bonds)


def fetch_info(CIF_dict):
    """
    Extract extra information from CIF.
    """

    info = {}
    for i in info_tags:

        if i in CIF_dict:
            fetch = CIF_dict[i]
            if type(fetch) in [list, tuple]:
                fetch = fetch[0]
            info[i] = f"'{fetch}'"
    
    return info


def format_bonds(lattice, labels, positions, bonds, return_lengths=False):
    """
    For a given list of atom labels and their bonds, format information
    according to the TopoCIF specification.
    """

    # Define a "zero-image" array.
    zero_img = list(np.zeros(3,dtype=int))

    # Create label-position,image dictionary.
    li = {l:p for l,p in zip(labels,positions)}

    # Note, there will be duplicate bonds if simply iterate through all building
    # units because the bonds are defined in both directions. This is not
    # desirable behaviour, and so the atom labels are sorted alphabeticaly, and
    # only unique entries are returned.
    fb = []; ds = []
    for n1, b in zip(labels, bonds):

        for n2, img in b:
            
            # Get fractional coordinates with correct image. The n1 is defined 
            # at its [0,0,0] image.
            c1 = li[n1]
            c2 = np.sum([li[n2],img],axis=0)

            # Calculate Cartesian distance between two nodes.
            cs = lattice.get_cartesian_coords([c1,c2])
            d = np.round(np.linalg.norm(np.subtract(*cs),axis=0),8)

            # Get sorted nodes and re-center bond such that the node1 image is
            # zero.
            nodes = sorted(((n1, zero_img), (n2, img)), key=lambda x: x[0])
            nodes = np.concatenate(np.array(nodes,dtype="object"),dtype="object")
            fb.append( 
                (nodes[0],nodes[2],d,tuple(zero_img),tuple(nodes[3]-nodes[1])) 
            )

            # Also store distances in separate list for ease of extracting later
            # because could be used for re-scaling structures.
            ds.append(d)
    
    # Get unique bonds.
    fb = list(set(fb))

    if return_lengths:
        return fb, ds

    return fb


def relabel(units, siteMap, keepSingleAtoms, bonds=None):
    """
    Relabel sites for coarse-grained structure.
    """
    
    # Convert siteMap from list into dictionary.
    siteMap = {chr(ord("a")+i):siteMap[i] for i in range(len(siteMap))}

    # In order to (a) create a new Pymatgen Structure object, and (b) relabel
    # bonds with the new labels, we need a list of both the symbols alone (a)
    # and symbols+number (b). Store these in (a) symbols and (b) labels lists.
    symbols = []; labels=[]; old2new = {}; l_c = {}
    for l,u in units.items():

        if keepSingleAtoms and len(u) == 1:
            new = u.sym[0]
        else:
            new = siteMap[ "".join(i for i in l if not i.isdigit()) ]
        
        # Get number of that given label so far.
        if new not in l_c:
            l_c[new] = 1
        else:
            l_c[new] += 1

        # Get symbol and label.
        symbols.append(new)
        new = f"{new}{l_c[new]}"
        labels.append(new)

        # Create mapping dictionary to relabel bonds.
        old2new[l] = new

    # If bonds passed to function, also relabel the atoms.
    if bonds is not None:

        nb = [[old2new[x] for x in b[:2]]+list(b[2:]) for b in bonds]

        return labels, symbols, nb

    return labels, symbols


def parseBonds(bonds):
    """
    Takes raw TopoCIF input and converts to more useful arrays of bonded-atom
    labels and their respective perioidic images.
    """
    # Extract the images for atoms 1 and 2 for each bond.
    i1s = [np.array(x, dtype=int) for x in list(zip(*bonds[4:7]))]
    i2s = [np.array(x, dtype=int) for x in list(zip(*bonds[8:11]))]

    # Return them, along with the two atom-labels per bond.
    return list(zip(list(zip(*bonds[:2])),list(zip(i1s,i2s))))


def coords2str(labels, symbols, x):
    """
    Convert list of atom labels and 2D numpy array of coordinates into CIF
    string.
    """
    # Iterate through each line and format atom position line to:
    # Atom-label, atomic symbol, siteSym, frac(x), frac(y), frac(z), occupancy.
    cs = []
    for l,s,a in zip(labels,symbols,x):
        cs.append(" ".join( [f"{l:<8}{s:<5}{1:<3}"] + \
                            [f"{v:>18.12f}" for v in a] + \
                            [f"{1:>3}"]) )
    return "\n".join(cs)


def bonds2str(bonds):
    """
    Convert list of bonds into TopoCIF string.
    """
    bs = []
    for b in bonds:
        bs.append(" ".join(
            [f"{x:>8}" for x in b[:2]] + [f"{b[2]:>8.5f}"] + [f"{1:>3}"] + \
            ["{:>3} {:>3} {:>3}".format(*b[3])] + [f"{1:>3}"] + \
            ["{:>3} {:>3} {:>3}".format(*b[4])] + [f"{'V':>2}{1:>2}"]
        ))
    
    return "\n".join(bs)


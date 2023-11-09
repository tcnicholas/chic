"""
12.06.23
@tcnicholas
The main Net class for decorating frameorks to MOFs.
"""

import inspect
import warnings
from pathlib import Path
from typing import Union
from itertools import combinations, product

import ase
import numpy as np
from ase import Atoms
from ase.build.tools import sort
from pymatgen.core.lattice import Lattice
from ase.calculators.lammps import Prism, convert
from pymatgen.core import Structure as PymatgenStructure

from . import templates
from .cif import read_cif
from .bonds import Bonding
from .zif import Zinc, Imidizolate
from .utils import (
    strip_number,
    setattrs,
    atom2str,
)
from .decorate import place_linker, assign_closest_nitrogen


class Net(PymatgenStructure):    
    """
    The Net class is a subclass of the Pymatgen Structure class. It is used to
    specifically for decorating networks with e.g. ZIF nodes and linkers.
    """
    def __init__(self, 
        *args,
        filename: str,
        supercells: int = 0,
        **kwargs
    ) -> None:
        """
        Initialise net.

        Arguments:
            filename: Filename of net.
            supercells: Number of supercells built so far.
        """

        # initialise the Pymatgen Structure.
        super().__init__(*args, **kwargs)

        # keep track of the filename and name of the net.
        self._filename = Path(filename)
        self._name = str(self._filename.stem).split('_')[0]
        self._bonding = None
        self._supercells = supercells

        # mapping between labels and indices in Pymatgen Structure.
        self._label_index = {
            l.properties["label"]:i for i,l in enumerate(self)
        }

        # prepare empty dictionary for building units.
        self._template = None
        self._building_units = {}
        self._decorated_atoms = []
        self._formula = {"A":0, "B":0}

        # topology.
        self._nbonds = 1
        self._nangles = 1
        self._ndihedrals = 1
        self._noutofplanes = 1
        self._bonds_lammps = []
        self._angles_lammps = []
        self._dihedrals_lammps = []
        self._outofplanes_lammps = []


    @classmethod
    def from_structure(cls,
        structure: PymatgenStructure,
        bonding: Bonding,
        **kwargs
    ) -> 'Net':
        """
        Create a new Net object from an existing Pymatgen Structure.

        Arguments:
            structure: Pymatgen Structure object.
            bonding: Bonding object.

        Returns:
            Net object.
        """
        instance = cls(
            structure.lattice,
            structure.species,
            structure.frac_coords,
            validate_proximity=False,
            site_properties=structure.site_properties,
            **kwargs
        )
        instance._bonding = bonding
        return instance

        
    @classmethod
    def from_topocif(cls,
        filename: str,
        **kwargs
    ) -> 'Net':
        """
        Read structure and bonding form TopoCIF.

        Arguments:
            filename: Filename of TopoCIF.
        
        Returns:
            Net object.
        """

        # check which kwargs are for read_cif and which are for Net.
        read_cif_kwargs = {
            k:v for k, v in kwargs.items() 
            if k in inspect.signature(read_cif).parameters and k != 'self'
        }
        cls_kwargs = {
            k:v for k, v in kwargs.items()
            if k in inspect.signature(cls.__init__).parameters and k != 'self'
        }

        # parse the file. this time we require the bonding information. Raise 
        # a warning if the bonding information is not present.
        struct, bonding = read_cif(filename, **read_cif_kwargs)
        if bonding is None:
            warnings.warn(
                "Bonding information not found in TopoCIF. "
                "Make sure it can be inferred easily enough. "
                "Otherwise consider using chic to generate the bonding first."
            )

        # hence initialise the Net object.
        return cls.from_structure(
            struct, 
            bonding, 
            filename=filename,
            **cls_kwargs
        )
    
    
    def rescale(self,
        scale_value: float = 3.5,
        target: str = 'min_a_b'
    ) -> None:
        """
        Rescale the unit cell.

        The provided network may not be a suitable length-scale for the chemistry
        that is being modelled. This function allows the user to rescale the
        unit cell to a more suitable length-scale. The rescaling is performed
        by scaling the unit cell by a factor that is the ratio of the target
        value to the current value. The target value can be either the minimum
        A-B bond length or the volume of the unit cell.

        Arguments:
            scale_value: Value to scale the unit cell by.
            target: Target value to scale the unit cell by. Can be either
                'min_a_b' or 'volume'.
        """
        if target == 'min_a_b':
            lengths = self._bonding.all_bond_lengths
            sf = scale_value / np.min(lengths)
            new_volume = self.lattice.volume * sf**3
        elif target == 'volume':
            new_volume = scale_value
        self.lattice = self.lattice.scale(new_volume)
        self._bonding = Bonding(
            self, 
            self._bonding._labels, 
            self._bonding._bonds
        )


    def reset_decoration(self) -> None:
        """
        Reset the decorating procedures.
        """

        # global variables.
        self._template = None
        self._building_units = {}
        self._decorated_atoms = []
        self._formula = {"A":0, "B":0}

        # topology.
        self._nbonds = 1
        self._nangles = 1
        self._ndihedrals = 1
        self._noutofplanes = 1
        self._bonds_lammps = []
        self._angles_lammps = []
        self._dihedrals_lammps = []
        self._outofplanes_lammps = []

    
    def add_zif_atoms(self, 
        template: str = 'H'
    ) -> None:
        """
        Decorate the net with zinc nodes and imidazolate linkers with the 
        imidazolate molecules with the provided template.

        Arguments:
            template: Template ligand for decorating the net. Currently
                supported templates are 'H', and 'CH3', with MOF-FF support.
        """

        # get the template class for this material.
        self.reset_decoration()
        template = getattr(templates, f"ZIF8_{template}")()
        self._template = template

        atom_id = 1
        for mol_id, (label, ix) in enumerate(self._label_index.items(), 1):

            centroid = self[ix].coords

            if 'Si' in label:

                # A site nodes are simply zinc atoms in the same position.
                a_site = Zinc(mol_id, atom_id)
                self._building_units[label] = a_site
                self._building_units[mol_id] = a_site
                self._decorated_atoms.append([
                    atom_id,
                    mol_id,
                    1,
                    template.property_by_symbol('Zn', 'charge'),
                    *list(centroid)
                ])
                atom_id += 1
                self._formula["A"] += 1
            
            elif 'O' in label:

                # B site nodes are imidazolate molecules. The imidazolate
                # molecules are placed at the centroids of the O atoms. The
                # orientation of the imidazolate molecule is determined by the
                # bond vectors of the O atoms to the Si atoms.
                a_ids = list(range(atom_id, atom_id+len(template)))
                b_site = Imidizolate(
                    mol_id,
                    a_ids,
                    template.atom_labels,
                )
                self._building_units[label] = b_site
                self._building_units[mol_id] = b_site
                coords = place_linker(
                    template, 
                    centroid, 
                    self._bonding.bond_by_label(label)
                )

                atom_data = zip(*[
                    template.atom_types, template.atom_charges, coords
                ])

                for a_type, charge, coord in atom_data:
                    self._decorated_atoms.append(
                        [atom_id, mol_id, a_type, charge, *list(coord)])
                    atom_id += 1
                self._formula["B"] += 1

        self._decorated_atoms = np.array(self._decorated_atoms)


    def assign_mofff_topology(self) -> None:
        """
        Assign the topology based on the MOF-FF template.
        """

        assert self._template.mofff_parameterised == True, \
            "Template must be parameterised with MOF-FF to add topology info."

        # these functions will be handy throughout.
        get_frac = getattr(self.lattice, "get_fractional_coords")
        get_dist = getattr(self.lattice, "get_distance_and_image")

        # first we assign the atomistic A-to-B site connectivity by identifying
        # which nitrogen atom in the imidazolate is bound to the a site.
        for a_site in range(1, self._formula['A']+1):
            
            a_site_unit = self._building_units[f'Si{a_site}']
            bound = self._bonding.bound_to(f'Si{a_site}')

            for i, b_site in enumerate(bound, 1):

                # assign connectivity between a and b site.
                b_site_unit = self._building_units[b_site]
                setattrs(a_site_unit, **{f"Im{i}": b_site_unit.mol_id})
                assign_closest_nitrogen(
                    self._decorated_atoms, 
                    get_frac, 
                    get_dist, 
                    a_site_unit, 
                    b_site_unit, 
                    i
                )
            
            # hence store Zn–N connectivity.
            zn_bonds = a_site_unit.bonds()
            number_of_bonds = len(zn_bonds)
            a_site_unit.BondIds = list(
                range(self._nbonds, self._nbonds+number_of_bonds)
            )
            for i, (b1, b2, atype) in enumerate(zn_bonds, 1):
                bond_type = self._template.bond_types[atype]
                self._bonds_lammps.append([self._nbonds, bond_type, b1, b2])
                a_site_unit.Bond2Im[getattr(a_site_unit,f'Im{i}')]=[self._nbonds]
                self._nbonds += 1

        # assign all intramolecular bonds, angles, dihedrals, and out-of-planes.
        for b_site in range(1, self._formula['B']+1):
            
            # bonds.
            b_site_unit = self._building_units[f'O{b_site}']
            bonds = b_site_unit.topology_indices(self._template, 'bonds')
            number_of_bonds = len(bonds)
            b_site_unit.BondIds = list(
                range(self._nbonds, self._nbonds+number_of_bonds)
            )
            for i, (b1, b2, atype) in enumerate(bonds, 1):
                bond_type = self._template.bond_types[atype]
                self._bonds_lammps.append([self._nbonds, bond_type, b1, b2])
                self._nbonds += 1
        
            # angles.
            angles = b_site_unit.topology_indices(self._template, 'angles')
            number_of_angles = len(angles)
            b_site_unit.AngleIds = list(
                range(self._nangles, self._nangles+number_of_angles)
            )
            for i, (a1, a2, a3, atype) in enumerate(angles, 1):
                angle_type = self._template.angle_types[atype]
                self._angles_lammps.append(
                    [self._nangles, angle_type, a1, a2, a3]
                )
                self._nangles += 1
            
            # dihedrals.
            dihedrals = b_site_unit.topology_indices(self._template,'dihedrals')
            number_of_dihedrals = len(dihedrals)
            b_site_unit.DihedralIds = list(
                range(self._ndihedrals, self._ndihedrals+number_of_dihedrals)
            )
            for i, (d1, d2, d3, d4, atype) in enumerate(dihedrals, 1):
                dihedral_type = self._template.dihedral_types[atype]
                self._dihedrals_lammps.append(
                    [self._ndihedrals, dihedral_type, d1, d2, d3, d4]
                )
                self._ndihedrals += 1

            # out-of-planes.
            outofplanes = b_site_unit.topology_indices(
                self._template, 'outofplanes'
            )
            number_of_outofplanes = len(outofplanes)
            b_site_unit.OutOfPlaneIds = list(
                range(
                    self._noutofplanes, self._noutofplanes+number_of_outofplanes
                )
            )
            for i, (a1, a2, a3, a4, atype) in enumerate(outofplanes, 1):
                outofplane_type = self._template.out_of_plane_types[atype]
                self._outofplanes_lammps.append(
                    [self._noutofplanes, outofplane_type, a1, a2, a3, a4]
                )
                self._noutofplanes += 1

        # many-body terms. these can all be found by iterating over all a sites
        # and then in turn iterating over unique pairs of b sites.
        for a_site in range(1, self._formula['A']+1):
            
            # 2-body terms.
            a_site_unit = self._building_units[f'Si{a_site}']
            for b_site, (_, a_or_b) in a_site_unit.bound_im:

                b_site_unit = self._building_units[b_site]

                angles = b_site_unit.topology_indices_2body(
                    self._template,
                    ord(a_or_b)-97,
                    'angles',
                )
                number_of_angles = len(angles)
                for i, (a1, a2, atype) in enumerate(angles, 1):
                    angle_type = self._template.angle_types[atype]
                    self._angles_lammps.append([
                        self._nangles, 
                        angle_type, 
                        a1, a2, a_site_unit.atom_id
                    ])
                    self._nangles += 1
                
                dihedrals = b_site_unit.topology_indices_2body(
                    self._template,
                    ord(a_or_b)-97,
                    'dihedrals',
                )
                number_of_dihedrals = len(dihedrals)
                for i, (a1, a2, a3, atype) in enumerate(dihedrals, 1):
                    dihedral_type = self._template.dihedral_types[atype]
                    self._dihedrals_lammps.append([
                        self._ndihedrals, 
                        dihedral_type, 
                        a1, a2, a3, 
                        a_site_unit.atom_id
                    ])
                    self._ndihedrals += 1

                outofplanes = b_site_unit.topology_indices_2body(
                    self._template,
                    ord(a_or_b)-97,
                    'outofplanes',
                )
                number_of_outofplanes = len(outofplanes)
                for i, (a1, a2, a3, atype) in enumerate(outofplanes, 1):
                    outofplane_type = self._template.out_of_plane_types[atype]
                    self._outofplanes_lammps.append([
                        self._noutofplanes, 
                        outofplane_type, 
                        a1, a_site_unit.atom_id, a2, a3
                    ])
                    self._noutofplanes += 1
                
            # 3-body terms.
            #TODO: ponder. I am going to hard-code these because I do not forsee
            # any imidazolate linkers parameterised with MOF-FF to get any more
            # complicated. If they do, I will have to think about this.
            all_pairs = combinations(a_site_unit.bound_im, r=2)
            for (b_site_1, (_,a_or_b_1)), (b_site_2, (_,a_or_b_2)) in all_pairs:

                b_site_unit_1 = self._building_units[b_site_1]
                b_site_unit_2 = self._building_units[b_site_2]

                # the only 3-body angle term is N(x)-Zn-N(y).
                # hence find the atom ID of the relevant nitrogens directly.
                n1 = b_site_unit_1.atom_id_by_label(f'N{a_or_b_1}')
                n2 = b_site_unit_2.atom_id_by_label(f'N{a_or_b_2}')
                angle_type = self._template.angle_types[('N', 'Zn', 'N')]
                self._angles_lammps.append(
                    [self._nangles, angle_type, n1, a_site_unit.atom_id, n2]
                )
                self._nangles += 1

                # dihedrals.
                c1_1 = b_site_unit_1.atom_id_by_label(f'C1{a_or_b_1}')
                c2_1 = b_site_unit_1.atom_id_by_label(f'C2')
                c1_2 = b_site_unit_2.atom_id_by_label(f'C1{a_or_b_2}')
                c2_2 = b_site_unit_2.atom_id_by_label(f'C2')

                dihedral_type = self._template.dihedral_types[('C1', 'N', 'Zn', 'N')]
                self._dihedrals_lammps.append([
                    self._ndihedrals, 
                    dihedral_type, 
                    c1_1, n1, a_site_unit.atom_id, n2
                ])
                self._ndihedrals += 1
                self._dihedrals_lammps.append([
                    self._ndihedrals, 
                    dihedral_type, 
                    c1_2, n2, a_site_unit.atom_id, n1
                ])
                self._ndihedrals += 1

                dihedral_type = self._template.dihedral_types[('C2', 'N', 'Zn', 'N')]
                self._dihedrals_lammps.append([
                    self._ndihedrals, dihedral_type, 
                    c2_1, n1, a_site_unit.atom_id, n2
                ])
                self._ndihedrals += 1
                self._dihedrals_lammps.append([
                    self._ndihedrals, dihedral_type, 
                    c2_2, n2, a_site_unit.atom_id, n1
                ])
                self._ndihedrals += 1
        
        # (adds after the last bond.)
        self._nbonds -= 1
        self._nangles -= 1
        self._ndihedrals -= 1
        self._noutofplanes -= 1


    def to_lammps_data(self,
        filename: str,
        skip_topology: bool = False
    ) -> None:
        """
        Write full LAMMPS data file.

        Arguments:
            filename: Filename of LAMMPS data file.
            skip_topology: Whether to skip the topology section.
        """

        assert self._template is not None, 'Must decorate net first.'

        f = f'{self._template.name}\n\n' \
            f'{len(self._decorated_atoms)} atoms\n' \
            f'{len(self._bonds_lammps)} bonds\n' \
            f'{len(self._angles_lammps)} angles\n' \
            f'{len(self._dihedrals_lammps)} dihedrals\n' \
            f'{len(self._outofplanes_lammps)} impropers\n\n' \
            f'{len(self._template.default_atom_types.keys())} atom types\n'
        
        if skip_topology:
            f += f'0 bond types\n' \
            f'0 angle types\n' \
            f'0 dihedral types\n' \
            f'0 improper types\n\n'
        else:
            f += f'{len(self._template.bond_types.keys())} bond types\n' \
            f'{len(self._template.angle_types.keys())} angle types\n' \
            f'{len(self._template.dihedral_types.keys())} dihedral types\n' \
            f'{len(self._template.out_of_plane_types.keys())} improper types\n\n'

        # use ASE converter to get LAMMPS simulation box parameters.
        p = Prism(self.lattice.matrix)
        xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(),
                                            "distance", "ASE", "real")
        xlo = ylo = zlo = 0.0
        
        # convert the positions to that of the LAMMPS prism.
        decor = np.array(self._decorated_atoms)
        pos = p.vector_to_lammps(decor[:,-3:], wrap=False)
        decor[:,-3:] = pos
        self._decorated_atoms = decor
        
        # simulation box size/parameters.
        # assume origin centred at (0, 0, 0).
        f += \
            f"{xlo:<10.8f} {xhi:<10.8f} xlo xhi\n" + \
            f"{ylo:<10.8f} {yhi:<10.8f} ylo yhi\n" + \
            f"{zlo:<10.8f} {zhi:<10.8f} zlo zhi\n" + \
            f"{xy+0:<10.8f} {xz+0:<10.8f} {yz+0:<10.8f} xy xz yz\n\n"

        # Add atomic masses for each atom.
        f += "Masses\n\n"
        
        for n, (atom_type,symbol) in enumerate(
            self._template.default_atom_types.items(), 1):
            f += f"{n}\t{self._template.mass[atom_type]}\t# {symbol}\n"
        
        f += "\nAtoms\n\n"
        for atom in self._decorated_atoms:
            f += atom2str(atom)

        if skip_topology:
            with open(filename, "w") as file:
                file.write(f)
            return

        f += "\nBonds\n\n"
        for bond in self._bonds_lammps:
            f += "\t{:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f}\n".format(*bond)

        f += "\nAngles\n\n"
        for angle in self._angles_lammps:
            f += "\t{:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f}\n".format(
                *angle)

        f += "\nDihedrals\n\n"
        for dihedral in self._dihedrals_lammps:
            f += "\t{:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f}\n".format(*dihedral)
        
        f += "\nImpropers\n\n"
        for improper in self._outofplanes_lammps:
            f += "\t{:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f}\n".format(*improper)
            
        with open(filename, "w") as file:
            file.write(f)


    #TODO: implement our own wrap function that also stores the correct periodic 
    # image of all atoms so that it will be easier to identify the rigid bodies
    # in LAMMPS.
    def to_ase_atoms(self) -> ase.Atoms:
        """
        Gather the decorated atoms into an ASE Atoms object.

        Returns:
            ASE Atoms object.
        """

        assert self._template is not None, 'Must decorate net first.'

        all_pos = []
        symbols = []
        mol_ids = []
        for atom in self._decorated_atoms:
            all_pos.append(atom[-3:])
            mol_ids.append(atom[1])
            symbols.append(
                strip_number(self._template.default_atom_types[atom[2]])
            )

        # hence compile the atoms object.
        atoms = Atoms(
            symbols, 
            positions=all_pos, 
            cell=self.lattice.matrix,
            pbc=True
        )
        atoms.arrays['mol-id'] = np.array(mol_ids, dtype=int)
        atoms = sort(atoms)
        #atoms.wrap() # it is important we don't wrap the coordinates otherwise
        # the rigid bodies will not be correctly identified in LAMMPS.
        return atoms
    

    def to_ase_to(self, filename: str, fmt:str = None, **kwargs) -> None:
        """
        Use ASE's writer to write the structure to a file.

        Arguments:
            filename: filename of output file.
            fmt: format of output file.
        """
        self.to_ase_atoms().write(filename, format=fmt, **kwargs)
    

    def replicate(self,
        factors: np.ndarray = None,
    ) -> 'Net':
        """
        Make a supercell of the net.

        Arguments:
            factors: Factors by which to replicate the unit cell. Default is
                [2,2,2].

        Returns:
            Net object.
        """

        if factors is None:
            factors = np.array([2,2,2])
        
        factors = np.array(factors)
        if np.all(factors == np.eye(3)):
            return self

        self.reset_decoration()
        self._supercells += 1
        num = {"Si":1, "O":1}
        new_labels = []
        new_positions = {}
        orig2new = {}

        for img in product(*[range(e) for e in factors]):
            img = np.array(img)
            for label, ix in self._label_index.items():
                l = strip_number(label)
                p = self.frac_coords[ix]
                label_n = l + str(num[l])
                new_labels.append(label_n)
                new_positions[label_n] = (p+img) / factors
                num[l] += 1
                orig2new[label+str(img % factors)] = label_n

        scale_factors = np.diag(factors)
        lattice = Lattice(np.dot(scale_factors, self.lattice.matrix))

        # now map old bonds to new images and determine the translation of the 
        # bonds for the new periodic images.
        new_bonds = []
        for (a1,a2), (img1,img2) in self._bonding._bonds:
            for img in product(*[range(e) for e in factors]):

                img = np.array(img)
                img1_n = img1 + img
                img2_n = img2 + img

                nl1 = orig2new[a1+str(img1_n % factors)]
                ni1 = img1_n // factors
                nl2 = orig2new[a2+str(img2_n % factors)]
                ni2 = img2_n // factors

                new_bonds.append( ((nl1, nl2), (ni1, ni2)) )
        
        # now reload the class with everything updated.
        symbols = [strip_number(x) for x in new_labels]
        frac_coords = np.array(list(new_positions.values()))

        # gather the new structure as a Pymatgen Structure object so we can 
        # reinstantiate the class.
        struct = PymatgenStructure(
            lattice,
            symbols,
            frac_coords,
            validate_proximity=False,
        )
        
        # add the full labels to the structure in the atom properties.
        struct.add_site_property('label', new_labels)

        # create the bonding class.
        bonding = Bonding(struct, new_labels, new_bonds)
        
        # reinstantiate the class.
        return Net.from_structure(struct, bonding, filename=self._filename)
    

    def check_closest_approach(self, 
        min_allowed_dist: float = 1.0, # Å
        min_allowed_dist_h: float = 0.5, # Å
    ) -> bool:
        """
        Check the closest approach between pairs of atoms.

        We divide the minimum allowed distances into those involving hydrogen
        atoms and those not involving hydrogen atoms. This is because hydrogen
        will likely have much closer approaches.

        Arguments:
            min_allowed_dist: Minimum allowed distance between two atoms (Å).
            min_allowed_dist_h: Minimum allowed distance between a hydrogen
                atom and any other atom (Å).

        Returns:
            True or False if the closest approach is less than the minimum
            allowed distance.
        """
        atoms = self.decorated_ase_atoms()
        atoms_no_h = atoms[atoms.numbers!=1]
        d1 = atoms_no_h.get_all_distances(mic=True)
        np.fill_diagonal(d1, 1e4)
        d2 = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d2, 1e4)
        return np.any(d1<=min_allowed_dist) or np.any(d2<=min_allowed_dist_h)

    
    def check_need_for_supercell(self,
        min_number_of_atoms: int = 0
    ) -> bool:
        """ 
        Check if the unit cell needs to be expanded. This is required if the
        longest bond length is greater than half the shortest cell length.

        While we can handle the distances properly by considering the specific 
        periodic image of each bond, in programs such as LAMMPS, for efficiency
        the minimum image convention (MIC) is assumed, and will therefore 
        misinterpret the bonds.

        Arguments:
            min_number_of_atoms: Minimum number of atoms in the unit cell.

        Returns:
            True or False if a supercell is needed.
        """

        # return True if the number of atoms is less than the minimum number of
        # atoms. i.e. a supercell is needed.
        atoms_check = len(self) <= min_number_of_atoms

        # return True if half the shortest cell length is less than the longest
        # bond length. i.e. a supercell is needed.
        max_length = self._bonding.all_bond_lengths.max()
        bond_check = max_length >= (np.amin(self.lattice.lengths[0]) / 2)

        # if either condition is True, return True.
        return atoms_check or bond_check
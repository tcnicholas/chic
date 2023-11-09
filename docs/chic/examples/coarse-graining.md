# Coarse-graining basics (ZIF-8 example)

## The task

In this example we take the CIF for the prototypical zeolitic imidazolate
framework, ZIF-8, and coarse-grain it. We check the coarse-graining is 
reasonable by overlaying the atomistic structure and coarse-grained structure
before writing the final net in TopoCIF format.
---
### The Structure class


The majority of the work will be done by the `chic.structure.Structure` class.
```python
from chic import Structure
```

We handle CIF and LAMMPS files separately to other file types because we make
use of the topological data potentially stored in the files. The `Structure`
class can be initialised accordingly:

```python
from pathlib import Path

# path to test example directory.
eg_dir = Path('examples/cg_zif8')

# instantiate a Structure object from a CIF file.
zif8 = Structure.from_cif(eg_dir / 'ZIF-8-sod.cif', cores=4, verbose=True)
```

The number of cores (`cores=4`) specifies how many cores will be used to 
parallelise the neighbour list calculation, which is the bottle neck of the 
code. The `verbose=True` setting determines whether or not timing statistics
will be printed after each structure processing tasks.

### Processing disorder

The `Structure` class builds on `Pymatgen`'s `Structure` class (inheritance), 
thereby giving us direct access to all of the functions you might already know
from `Pymatgen`. Here, our starting structure contains oxygen atoms in the pore.
Given that our main framework does not contain any oxygen atoms, we can remove
these straight away.

```python
# we can apply methods native to the Pymatgen Structure class.
zif8.remove_species('O')
```

Our starting structure has the additional complication of thermal disorder: the
methyl (–CH$_3$) substituent on the imidazolate linker will be free to rotate 
due to low energetic barriers. We therefore have two sites for each of the H 
atoms. For our purposes, it might be beneficial to instead take the "average" 
position for each pair of H atoms.

```python
# we can apply methods native to the CHIC Structure class.
zif8.average_element_pairs('H', rmin=0.0, rmax=0.9)
```

### Coarse-graining

Our structure is now clean and ready to coarse-grain! The three main stages are:

1. computing the neighbour list;
2. finding the atomic clusters (the building blocks of the material that we wish
    to reduce to a simplified representation); and 
3. coarse-graining the atomic clusters.

Each stage requires a corresponding function call.

I find the CrystalNN algorithm works superbly for a wide variety of materials,
and therefore I think is worth the slightly longer compute time (especially for
complicated materials like MOFs!). For a more primitive neighbour list building
option, see the (...) example for a zeolite which instead uses simple radial
cut-offs for determining neighbours.

The next stage is performing a depth-first search (DFS) on the neighbour graph.
This relies on differentiating between atomic cluster *types*, which in `chic`
is achieved by `sorting` the chemical species (an obvious limitation of this
method is when the coordination network is not built up in a *node* and *linker* 
manner). Additional neighbour-determining constraints may be applied at this 
stage (see the [neighbour lists page](../neighbourlists.md)).

Finally we are ready to coarse-grain. The default coarse-graining method is to
place a single bead at the geometric centroid of all atoms in the cluster. When
the `get_coarse_grained_net()` function is called, it also connects up the beads
according to the *inter*-cluster connectivity determined in the previous step.
```python
# compute the neighbour list using the CrystalNN algorithm.
zif8.get_neighbours_crystalnn()

# determine atomic clusters.
zif8.find_atomic_clusters()

# coarse-grain with centroid method.
zif8.get_coarse_grained_net()
```

### Write net to file

So far we have had no confirmation that this process has worked (hopefully you
also haven't received any errors...). One of the easiest ways of checking the 
results is to write the coarse-grained structure to a file that can be opened in
your favourite visualising program (e.g. CrystalMaker, VESTA, OVITO). Three 
main "write" methods currently supported by `chic` are:

1. to a CIF;
2. to a LAMMPS data file; and
3. to a CIF with the atomistic structure and the coarse-grained net overlaid.


```python
# we can write the coarse-grained net with the bond information to a TopoCIF.
zif8.net_to_cif(eg_dir / 'ZIF-8-sod-cg.cif', write_bonds=True, name='ZIF-8-cg')

# we can also write to LAMMPS data file.
zif8.net_to_lammps_data(
    eg_dir / 'ZIF-8-sod-cg.data', write_bonds=True, name='ZIF-8-cg'
)

# we can check the coarse-grained net seems reasonable.
zif8.overlay_cg_atomistic_representations(eg_dir / 'ZIF-8-sod-overlay.cif')
```

Not satisfied with these three options? We can also use `Pymatgen`'s writer 
functionality to output to “cif”, “poscar”, “cssr”, “json”, “xsf”, “mcsqs”, 
“prismatic”, “yaml”, and “fleur-inpgen”. (Note, if you are familiar with
Pymatgen, we haven't called `Structure.to(filename, fmt)` directly because that
would write the atomistic structure to the file).

```python
# call Pymatgen's writer to write to a VASP POSCAR format.
zif8.net_to(eg_dir / 'POSCAR', fmt='poscar')
```

Perhaps you are more familiar with `ASE` and its extensive IO capabilities? Not 
a problem! We can also convert our coarse-grained structure into an `ASE` 
`Atoms` object and use its write function too.

```python
# use ASE writer to write an xyz file.
zif8.net_to_ase_to(eg_dir / 'ZIF-8-cg.xyz')
```

### To Pymatgen and ASE

As suggested in the previous section, it is also possible to extract the 
coarse-grained net as a `Pymatgen` `Structure` object, or as an `ASE` `Atoms`
object.

```python
# net to Pymatgen Structure object.
pym_net = zif8.net_to_struct()

# net to ASE Atoms object.
ase_net = zif8.net_to_ase_atoms()
```

## Summary

The overall structure processing task involves the following stages:

- Read structure from file.
- Compute the neighbour list.
- Find atomic clusters.
- Coarse-grain the structure.
- Output coarse-grained net.

---

## Full code

```python
from pathlib import Path
from chic.structure import Structure

# path to test example directory.
eg_dir = Path('examples/cg_zif8')

# read file and tidy structure.
zif8 = Structure.from_cif(eg_dir / 'ZIF-8-sod.cif', cores=4, verbose=True)
zif8.remove_species('O')
zif8.average_element_pairs('H', rmin=0.0, rmax=0.9)

# coarse-grainiing process.
zif8.get_neighbours_crystalnn()
zif8.find_atomic_clusters()
zif8.get_coarse_grained_net()

# write coarse-grained net in different formats/representations.
zif8.net_to_cif(eg_dir / 'ZIF-8-sod-cg.cif', write_bonds=True, name='ZIF-8-cg')
zif8.net_to_lammps_data(
    eg_dir / 'ZIF-8-sod-cg.data', write_bonds=True, name='ZIF-8-cg'
)
zif8.overlay_cg_atomistic_representations(eg_dir / 'ZIF-8-sod-overlay.cif')
zif8.net_to(eg_dir / 'POSCAR', fmt='poscar')
zif8.net_to_ase_to(eg_dir / 'ZIF-8-cg.xyz')

# net to Pymatgen Structure and ASE Atoms objects.
pym_net = zif8.net_to_struct()
ase_net = zif8.net_to_ase_atoms()
```
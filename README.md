# CHIC
**C**oarse-graining **H**ybrid **I**organic **C**rystals.



## Installation
    
     ```bash
     conda create -n chic23 python=3.8 -y
     conda activate chic23
     conda install -c conda-forge cython pymatgen freud -y
     ```
     

## Tests

     python tests.py


## Examples

### ZIF-8

In this example we take the CIF for the prototypical zeolitic imidazolate
framework, ZIF-8, and coarse-grain it. We check the coarse-graining is 
reasonable by overlaying the atomistic structure and coarse-grained structure
before writing the final net in TopoCIF format.

```python
from chic.structure import Structure

# instantiate a Structure object from a CIF file.
zif8 = Structure.from_cif('examples/ZIF-8-sod.cif', cores=4, verbose=True)

# we can apply methods native to the Pymatgen Structure class directly. Here
# we remove oxygen atoms that are in the pores of the structure.
zif8.remove_species('O')

# we can apply methods native to the CHIC Structure class.
zif8.average_element_pairs('H', rmin=0.0, rmax=0.9)

# a helper function for computing the neighbour list using CrystalNN. This is 
# the bottleneck in the code, and is where Python's multiprocessing module
# is invoked, however works particularly well with MOF systems.
zif8.get_neighbours_crystalnn()

# determine atomic clusters (i.e. zinc nodes and imidazolate linkers).
zif8.find_atomic_clusters()

# coarse-grain with centroid (default) method.
zif8.get_coarse_grained_net()

# we can check the coarse-grained net seems reasonable.
zif8.overlay_cg_atomistic_representations('examples/ZIF-8-sod-overlay.cif')

# we can write the coarse-grained net to a CIF file with the bonds formatted
# for TopoCIF.
zif8.to_cif(
    'examples/ZIF-8-sod-topocif.cif', 
    write_bonds=True, 
    net_name='ZIF-8-cg'
)
```

## Authors

Thomas C. Nicholas

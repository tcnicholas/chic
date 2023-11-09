# CHIC
**C**oarse-graining **H**ybrid **I**organic **C**rystals.

![Schematic of decorating and coarse-graining the sodalite (**sod**) net](/bin/images/sod-decoration.png)


## Getting started

Install with `pip install chic-lib`, and you're ready to go!

By way of a quick example, ZIF-8 (CSD RefCode: FAWCEN) can be coarse-grained
by running:

```python
from chic import Structure

# read in structure and delete oxygen from the pores.
zif8 = Structure.from_cif('FAWCEN.cif')
zif8.remove_species('O')

# compute neighbour list, find atomic clusters, and coarse-grain. 
zif8.get_neighbours_crystalnn()
zif8.find_atomic_clusters()
zif8.get_coarse_grained_net()

# export structure as TopoCIF.
zif8.net_to_cif('ZIF-8-sod-cg.cif', write_bonds=True, name='ZIF-8-cg')
```

Head over to the [chic docs](https://tcnicholas.github.io/chic/) to see examples
and more details!

## ToDo list

- [x] Add docs.
- [ ] Add simple distance cut-off algorithm for neighbour list building.
- [ ] Add custom implementation of optimised CrystalNN algorithm.
- [x] Integrate back-mapping code.
- [ ] Integrate extraction of local energies from LAMMPS dump format.
- [ ] Add registry to Net class for easier future development beyond ZIFs.

## Authors

Thomas C. Nicholas

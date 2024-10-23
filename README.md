![CHIC logo](/docs/source/logo.svg)

**C**oarse-graining **H**ybrid **I**norganic **C**rystals.

## Getting started

Install with `pip install chic-lib`, and you're ready to go!

By way of a quick example, ZIF-8 (CSD RefCode: FAWCEN) can be coarse-grained
by running:

```python
from chic import Structure

# read in structure and delete oxygen from the pores.
struct = Structure.from_cif("ZIF-8-sod.cif")
struct.remove_sites_by_symbol("O")

# compute neighbour list, find atomic clusters, and coarse-grain. 
struct.get_neighbours_crystalnn()
struct.find_atomic_clusters()
struct.get_coarse_grained_net()

# export structure as TopoCIF.
struct.net_to_cif('ZIF-8-sod-cg.cif', write_bonds=True, name='ZIF-8-cg')
```

Head over to the [chic docs](https://tcnicholas.github.io/chic/) to see examples
and more details!

## ToDo list

- [x] Add docs.
- [x] Add simple distance cut-off algorithm for neighbour list building.
- [ ] Add custom implementation of optimised CrystalNN algorithm.
- [x] Integrate back-mapping code.
- [x] Integrate extraction of local energies from LAMMPS dump format.
- [ ] Add registry to Net class for easier future development beyond ZIFs.

## Authors

Thomas C. Nicholas

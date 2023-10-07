# Neighbour lists

At the heart of all structural coarse-graining is the process of (correctly)
identifying atomic neighbours. If the neighbours are assigned incorrectly, 
you might e.g. incorrectly link two clusters or fail to find complete clusters.

The automatic and accurate identification of atomic neighbours is challenging 
(even defining "what a neighbour is" is difficult). 

## Algorithms

### Nearest neighbours

Not yet implemented.

The simplest approach to computing neighbour lists is to use distance cut-offs 
between pairs of atoms.

### CrystalNN

H. Pan *et al.*; *Inorg. Chem.*, **60**, 1590–1603 (2021)

Crystal-near-neighbour ([CrystalNN](https://pubs.acs.org/doi/10.1021/acs.inorgchem.0c02996)) 
is one such algorithm for determining near neighbours. It uses a Voronoi
decomposition and solid angle weights to determine coordination environments,
and is found to be accurate for a diverse set of materials. It is my preferred 
choice of coordination-determining algorithm for MOFs. It is implemented in the 
`pymatgen.analysis.local_env` name space.

## Neighbour lists in chic

The first place we encounter neighbour list building in `chic` is when we want
to find the atomic clusters. A prerequisite to the 
`Structure.find_atomic_clusters()` function is that a neighbour list has already
been computed or assigned. To run the `CrstalNN` neighbour list calculation, 
you can call the following function:

```python
Structure.get_neighbours_crystalnn()
```

### `find_atomic_clusters()`

The *intra* constraints pertain to whether atomic neighbours should be 
considered in the same atomic cluster, while the *inter* constraints determine 
whether atoms in different atomic clusters should be considered bound. 

There are two types of contraints that may be applied.

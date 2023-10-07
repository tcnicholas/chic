# Trajectories

It is sometimes useful to be able to process the evolution of a structure over,
for example, a molecular dynamics trajectory.

In the case of the popular and widely-used molecular dynamics program 
[LAMMPS](https://www.lammps.org/#gsc.tab=0), you typically initiate a simulation 
with a LAMMPS data file which contains the structural (and sometimes topological) 
information about your system. Then, over the course of a simulation a more
"streamlined" file is "dumped" which describes the evolution of the structure
during the simulation.

`chic` provides the option of processing a LAMMPS data file (which contains
sufficient information for the coarse-graining process), and then "sharing" this
information to more efficiently coarse-grain the full trajectory.

## The task

In this example, 


```python
from pathlib import Path
from chic.structure import Structure
```

```python
# path to test example directory.
eg_dir = Path('examples/cg_lammps_output')

# load the LAMMPS data object and cluster by molecule ID.
zif4_400K = Structure.from_lammps_data(
    eg_dir / 'melt.plateau_400K.data', 
    cluster_by_molecule_id=True
)
```

```python
# coarse-grain and output overlay representation.
zif4_400K.get_coarse_grained_net()
zif4_400K.overlay_cg_atomistic_representations(eg_dir / 'overlay.cif')
```

```python
# get the generator of chic Structure for each snapshot.
snapshots = zif4_400K.append_lammps_trajectory(
    eg_dir / 'plateau_400K.dump', start=0, end=10, step=2
)
```

```python
# iterate through each frame index and snapshot. we can treat each snapshot as
# any other Structure class to coarse-grain.
for frame, snapshot in snapshots:
    snapshot.get_coarse_grained_net()
    snapshot.overlay_cg_atomistic_representations(eg_dir / f'snapshot-{frame}-overlay.cif')
```

---
## Full code

```python
from pathlib import Path
from chic.structure import Structure

# path to test example directory.
eg_dir = Path('examples/cg_lammps_output')

# load the LAMMPS data object and cluster by molecule ID.
zif4_400K = Structure.from_lammps_data(
    eg_dir / 'melt.plateau_400K.data', 
    cluster_by_molecule_id=True
)

# coarse-grain starting structure.
zif4_400K.get_coarse_grained_net()
zif4_400K.net_to_cif(eg_dir / 'frame-0.cif', write_bonds=True, name='frame0')

# process snapshots from the trajectory.
snapshots = zif4_400K.append_lammps_trajectory(
    eg_dir / 'plateau_400K.dump', start=0, end=10, step=2
)
for frame, snapshot in snapshots:
    snapshot.get_coarse_grained_net()
    snapshot.net_to_cif(eg_dir / f'frame{frame}.cif')
```
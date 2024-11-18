from load_atoms import view
from IPython.core.display import HTML
from pymatgen.io.ase import AseAtomsAdaptor
from chic.atomic_cluster import AtomicCluster

def view_molecule(cluster: AtomicCluster) -> HTML:
    """
    Convert the building unit to an ASE Atoms object and inspect it with 
    load-atoms visaulisation tool.

    Arguments:
        cluster (AtomicCluster): the building unit to inspect.
    
    Returns:
        The HTML visualisation.
    """
    molecule = AseAtomsAdaptor.get_atoms(cluster.to_molecule())
    return view(molecule, show_bonds=True)
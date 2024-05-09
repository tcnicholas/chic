Dihedral angle analysis
=======================

:code:`chic` can be used to find the organic linkers to assist structural 
analysis. Here, I show how we can calculate dihedral angles for a MOF with
1,3-BDC linkers.

.. raw:: html
   :file: ../_static/1,3-BDC.html

Here, I've included a quick analysis script that can take a 
:class:`AtomicCluster <chic.atomic_cluster.AtomicCluster>` object as an 
argument. Note in particular how I make use of the :code:`AtomicCluster.graph` 
molecular graph object in order to traverse the shortest paths between oxygens
in the molecule.

.. dropdown:: dihedral.py

    .. literalinclude:: ../_static/scripts/dihedral.py

We can load structure in the usual way and search for nodes and linkers. This 
particular MOF can be downloaded from the Cambridge Structural Database (CDC),
with RefCode `JOLFEZ <https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid=JOLFEZ&DatabaseToSearch=Published>`_

    >>> from chic import Structure
    >>> mof = Structure.from_cif("JOLFEZ.cif")
    >>> mof.get_neighbours_crystalnn()
    >>> mof.find_atomic_clusters()
    >>> mof.get_coarse_grained_net()

First we can select a linker molecule from the atomic_clusters dictionary, where
the linkers will default to "B sites" (and zinc nodes will be "A sites").

    >>> linker = mof.atomic_clusters[("b", 2)]
    >>> print(linker)
    AtomicCluster("C8 O4 H4", site_indices=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

We can instantiate our analysis class and analyse our chosen linker molecule:

    >>> from dihedral import DihedralAnalysis
    >>> analysis = DihedralAnalysis()
    >>> results = analysis.analyse_ligand(linker)
    >>> print(results)
    {'carboxylate_a': array([176.342,  -5.238]),
    'carboxylate_b': array([149.77 , -32.647])}


Related content
---------------
.. card:: Torsional flexibility in zinc–benzenedicarboxylate metal–organic frameworks
    :link: https://pubs.rsc.org/en/content/articlelanding/2024/ce/d3ce01078c

    Emily G. Meekel, Thomas C. Nicholas, Ben Slater and Andrew L. Goodwin

    *CrystEngComm*, **26**, 673–680 (2024)
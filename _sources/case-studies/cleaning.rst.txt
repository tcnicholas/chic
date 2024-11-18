Cleaning structures
===================

:code:`chic` can be used to "clean-up" disordered structures. Here, I show how
the thermal disorder in the methyl-substituted imidazolate molecules in a ZIF 
can be averaged to single positions.

In this case study, I will use the :code:`load_atoms` Python package for 
visualising the molecules. You can instsall it easily with 
:code:`pip install load-atoms`. Below is a helper function:

.. dropdown:: viz.py

    .. literalinclude:: ../_static/scripts/viz.py

First, let's load the ZIF-8 structure.

    >>> from chic import Structure
    >>> from viz import view_molecule 
    >>> mof = Structure.from_cif("ZIF-8-sod.cif")
    >>> mof.remove_sites_by_symbol("O")
    >>> mof.get_neighbours_crystalnn()
    >>> mof.find_atomic_clusters()

Let's visualise the cluster.

    >>> disordered_linker = mof.atomic_clusters[("b", 1)]
    >>> view_molecule(disordered_linker)

.. raw:: html
   :file: ../_static/visualisations/mIm-disordered.html

We can "fix" this by averaging pairs of H atoms within a distance of 0.9 Å of
each other. Note, after inserting/deleting atoms, we need to force a rebuild of
the neighbourlist. It is therefore advised to perform structral cleaning before
further analysis.

    >>> mof.average_element_pairs("H", rmax=0.9)
    >>> mof.get_neighbours_crystalnn(force=True)
    >>> mof.find_atomic_clusters()
    >>> ordered_linker = mof.atomic_clusters[("b", 1)]
    >>> view_molecule(ordered_linker)

.. raw:: html
   :file: ../_static/visualisations/mIm-clean.html

We can write the ordered structure back to a CIF using Pymatgen functionality
directly.

    >>> mof.to("ZIF-8-sod-ordered.cif");
chic
====

.. raw:: html
    :file: hide-title.html

.. image:: logo.svg
   :align: center
   :alt: chic logo
   :target: .


.. important::

   This project is under active development. Until version 1.0.0 is released, breaking changes to the API may occur.

:code:`chic` is a Python package for :strong:`c`\ oarse-graining :strong:`h`\ ybrid and :strong:`i`\ norganic :strong:`c`\ rystals.

Quickstart
----------
Install using :code:`pip install chic-lib`, and then use 
:class:`~chic.Structure` to begin processing your structure...

   >>> from chic import Structure
   >>> struct = Structure.from_cif("ZIF-4-cag.cif")
   >>> struct.remove_sites_by_symbol("O")

... find the nodes, linkers, and underlying net...

   >>> struct.get_neighbours_crystalnn()
   >>> struct.find_atomic_clusters()
   >>> struct.get_coarse_grained_net()
   >>> struct.net_to_cif("cag-net.cif")

.. raw:: html
   :file: ./_static/zif4-cg.html

... and redecorate with a different chemistry...

   >>> cag_net = struct.to_net()
   >>> cag_net.add_zif_atoms(template="CH3") # methyl-substituted imidazolate

.. raw:: html
   :file: ./_static/zif4-cg-d.html
   
:code:`chic` is built upon the :class:`pymatgen.core.Structure` class:

   >>> struct
   Structure Summary
   Lattice
      abc : 16.8509 16.8509 16.8509
   angles : 90.0 90.0 90.0
   volume : 4784.860756696228
         A : 16.8509 0.0 1.0318200373876067e-15
         B : -1.0318200373876067e-15 16.8509 1.0318200373876067e-15
         C : 0.0 0.0 16.8509
      pbc : True True True
   PeriodicSite: H (7.669, 7.093, 15.44) [0.4551, 0.4209, 0.9161]
   PeriodicSite: H (7.093, 7.669, 15.44) [0.4209, 0.4551, 0.9161]
   [...]

for which all of the usual functionality is available.


Contributing
------------

:code:`chic` was developed by me, Thomas Nicholas, as part of my PhD research at 
the University of Oxford within the `Goodwin Group <https://goodwingroupox.uk>`_ 
and the `Deringer Group <https://www.chem.ox.ac.uk/people/volker-deringer>`_.
During this time, it has been adapted several times over, with new directions
defined by collaborations too (see :doc:`publications <publications/pub-main>`)!

Please do open an issue or pull request on the 
`GitHub repository <https://github.com/tcnicholas/chic>`_ if you are interested
in suggesting new functionality, identifying and/or fixing a bug, or simply
want to start a dicussion.


.. toctree::
   :maxdepth: 1
   :hidden:

   Quickstart <self>

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API Reference

   api/structure
   api/net
   api/atomic-clusters

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Publications

   publications/pub-main

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Case studies

   case-studies/coarse-graining
   case-studies/topology-data
   case-studies/decorating
   case-studies/dihedrals

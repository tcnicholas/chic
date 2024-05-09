Coarse-graining
===============

Coarse-graining MOFs was at the heart of the chic development. This can be
performed with just a few lines of code.

First, we can load the structure and remove the oxygen from the pores.

   >>> from chic import Structure
   >>> struct = Structure.from_cif("ZIF-8-sod.cif")
   >>> struct.remove_sites_by_symbol("O")

.. raw:: html
   :file: ../_static/zif8.html

Then, we can identify connected atoms, group them into atomic clusters (i.e. 
nodes and linkers), and reduce the structure to it's underyling net:

   >>> struct.get_neighbours_crystalnn()
   >>> struct.find_atomic_clusters()
   >>> struct.get_coarse_grained_net()

.. raw:: html
   :file: ../_static/zif8-cg.html
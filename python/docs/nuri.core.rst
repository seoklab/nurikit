.. Project NuriKit - Copyright 2023 SNU Compbio Lab.
   SPDX-License-Identifier: Apache-2.0

=========
nuri.core
=========

.. automodule:: nuri.core
   :exclude-members: Molecule, Atom, Bond, Neighbor, Mutator, Substructure,
      SubAtom, SubBond, SubNeighbor, AtomData, BondData

   .. autoclass:: Molecule
      :exclude-members: atom, bond, neighbor, bonds, mutator,
         num_atoms, num_bonds, get_conf, set_conf, num_confs, has_bond

      .. automethod:: __init__
      .. automethod:: atom
      .. automethod:: num_atoms
      .. automethod:: bonds
      .. automethod:: bond
      .. automethod:: has_bond
      .. automethod:: num_bonds
      .. automethod:: neighbor
      .. automethod:: mutator
      .. automethod:: get_conf
      .. automethod:: set_conf
      .. automethod:: num_confs

   .. autoclass:: Mutator

   .. autoclass:: Atom
      :no-members:
      :no-undoc-members:

      .. autoproperty:: id
      .. automethod:: count_neighbors
      .. automethod:: count_heavy_neighbors
      .. automethod:: count_hydrogens
      .. automethod:: get_pos
      .. automethod:: set_pos
      .. automethod:: copy_data

   .. autoclass:: Bond
      :no-members:
      :no-undoc-members:

      .. autoproperty:: id
      .. autoproperty:: src
      .. autoproperty:: dst
      .. automethod:: length
      .. automethod:: sqlen
      .. automethod:: rotate
      .. automethod:: copy_data

   .. autoclass:: Neighbor

   .. autoclass:: Substructure

   .. autoclass:: ProxySubstructure
      :no-members:
      :no-undoc-members:
      :members: copy

   .. autoclass:: SubAtom

   .. autoclass:: SubBond

   .. autoclass:: SubNeighbor

   .. autoclass:: AtomData

      .. automethod:: __init__

   .. autoclass:: BondData

      .. automethod:: __init__

------------------
Geometry Utilities
------------------

.. code-block:: python

   from nuri.core import geometry as ngeo

.. automodule:: nuri.core.geometry

.. footbibliography::

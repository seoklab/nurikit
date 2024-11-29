.. Project NuriKit - Copyright 2023 SNU Compbio Lab.
   SPDX-License-Identifier: Apache-2.0

============================
NuriKit Python API Reference
============================

.. automodule:: nuri
   :no-members:
   :no-undoc-members:

.. code-block:: python

   import nuri

----------
Submodules
----------

.. toctree::
   :maxdepth: 1

   nuri.core
   nuri.fmt
   nuri.algo
   nuri.tools

-------------------
Top-level Functions
-------------------

Readers
-------

.. autofunction:: nuri.readfile

.. autofunction:: nuri.readstring

Writers
-------

These functions all release the GIL and are thread-safe. Thread-based
parallelism is recommended for writing multiple molecules in parallel.

.. autofunction:: nuri.to_smiles

.. autofunction:: nuri.to_mol2

.. autofunction:: nuri.to_sdf

--------------------
Top-level Attributes
--------------------

.. py:data:: periodic_table
   :type: .core.PeriodicTable

   The singleton instance of :class:`.core.PeriodicTable` class.

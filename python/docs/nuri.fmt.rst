.. Project NuriKit - Copyright 2023 SNU Compbio Lab.
   SPDX-License-Identifier: Apache-2.0

========
nuri.fmt
========

.. automodule:: nuri.fmt
    :exclude-members: MoleculeReader, CifBlock, CifFrame, CifTable, CifValue,
        readfile, readstring, to_smiles, to_mol2, to_sdf, to_pdb

    .. autoclass:: MoleculeReader

        .. automethod:: __iter__

            Returns itself.

        .. automethod:: __next__

            Returns the next molecule.

    .. autoclass:: CifBlock

        .. automethod:: __init__

    .. autoclass:: CifFrame

        .. automethod:: __init__

    .. autoclass:: CifTable

        .. automethod:: __init__

    .. autoclass:: CifValue

        .. automethod:: __init__

------------------
PDB Format Support
------------------

.. code-block:: python

   from nuri.fmt import pdb

.. automodule:: nuri.fmt.pdb

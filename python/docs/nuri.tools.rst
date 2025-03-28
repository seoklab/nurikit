.. Project NuriKit - Copyright 2024 SNU Compbio Lab.
   SPDX-License-Identifier: Apache-2.0

==========
nuri.tools
==========

.. currentmodule:: nuri.tools

.. automodule:: nuri.tools

--------
TM-tools
--------

.. currentmodule:: nuri.tools.tm

.. code-block:: python

   from nuri.tools import tm as tmtools

.. automodule:: nuri.tools.tm
   :exclude-members: TMAlign

   This module provides ground-up reimplementation of TM-align algorithm based
   on the original TM-align code (version 20220412) by Yang Zhang. This
   implementation aims to reproduce the results of the original code while
   providing improved user interface and maintainability. Refer to the
   following paper for details of the algorithm. :footcite:`tm-align`

   All input structures must have **only single atom per residue** (usually
   ``CA`` atom), as the original TM-align algorithm assumes this.

   .. footbibliography::

   .. autoclass:: TMAlign
      :exclude-members: from_alignment

      .. automethod:: __init__
      .. automethod:: from_alignment

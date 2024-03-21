.. Project NuriKit - Copyright 2023 SNU Compbio Lab.
   SPDX-License-Identifier: Apache-2.0

=========
nuri.algo
=========

.. currentmodule:: nuri.algo

--------------
Ring detection
--------------

.. autofunction:: find_all_rings

SSSR-related routines
=====================

Formally, SSSR (smallest set of smallest rings) is a *minimum cycle basis*
of the molecular graph. As discussed in many literatures, there is no unique
SSSR for a given molecular graph (even for simple molecules such as
2-oxabicyclo[2.2.2]octane), and the SSSR is often counter-intuitive. For
example, the SSSR of cubane (although unique, due to symmetry reasons)
contains only five rings, which is not most chemists would expect.

On the other hand, union of all SSSRs, sometimes called the *relevant rings*
in the literatures, is unique for a given molecule. Chemically speaking, it is
the "all smallest rings" of the molecule and more appropriate for most applications
than SSSR.

Here, we provide two functions to find the relevant rings and SSSR, respectively.
The finding algorithms are based on the algorithm by :cite:t:`algo:sssr`.

.. note::
  The time complexity of the functions when max size is not specified is
  theoretically :math:`\mathcal{O}(\nu E^3)` where :math:`\nu = \mathcal{O}(E)`
  is size of SSSR and :math:`E` is the number of bonds. For most molecules,
  this is in practice :math:`\mathcal{O}(V^3)`, where :math:`V` is the number
  of atoms. See the reference for more details.

.. autofunction:: find_relevant_rings

.. autofunction:: find_sssr

--------------
Other routines
--------------

.. automodule:: nuri.algo
    :exclude-members: find_all_rings, find_relevant_rings, find_sssr

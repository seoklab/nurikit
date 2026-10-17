.. Project NuriKit - Copyright 2023 SNU Compbio Lab.
   SPDX-License-Identifier: Apache-2.0

=========
nuri.desc
=========

Descriptors of a molecular conformation: solvent-accessible surface areas,
analytic molecular surfaces, and the shape complementarity of an interface.

.. automodule:: nuri.desc

Example
=======

Shape complementarity of the barnase-barstar interface (PDB 1BRS without its
waters, chains A and D), with united-atom radii:

.. doctest::

   >>> from nuri.fmt.pdb import read_models
   >>> from nuri.param import assign_radii
   >>> from nuri.desc import shape_complementarity
   >>> model = read_models(test_data / "1brs_clean.pdb")[0]
   >>> radii = assign_radii(model, "connolly")
   >>> def side(chain):
   ...     idxs = [
   ...         i
   ...         for i, atom in enumerate(model.atoms)
   ...         if atom.res_id.chain_id == chain
   ...     ]
   ...     return model.major_conf[idxs], radii[idxs]
   >>> res = shape_complementarity(*side("A"), *side("D"))
   >>> round(res.sc, 3), round(res.distance, 3)
   (0.711, 0.474)

.. footbibliography::

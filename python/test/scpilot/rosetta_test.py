#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from nuri.scpilot import rosetta

SAMPLE_LOG = """
Structure 1:

==================================================

Molecule 1:
\t  Total Atoms: 922
\t Buried Atoms: 266
\tBlocked Atoms: 656
\t   Total Dots: 25258
 Trimmed Surface Dots: 7765
\t Trimmed Area: 520.401

Molecule 2:
\t  Total Atoms: 832
\t Buried Atoms: 268
\tBlocked Atoms: 564
\t   Total Dots: 26359
 Trimmed Surface Dots: 7932
\t Trimmed Area: 535.334

Total/Average for both molecules:
\t  Total Atoms: 1754
\t Buried Atoms: 830
\tBlocked Atoms: 534
\t   Total Dots: 51617
 Trimmed Surface Dots: 15697
\t Trimmed Area: 1055.74


Molecule 1->2:
      Mean Separation: 0.55
    Median Separation: 0.512273
    Mean Shape Compl.: 0.70
  Median Shape Compl.: 0.712243

Molecule 2->1:
      Mean Separation: 0.57
    Median Separation: 0.531862
    Mean Shape Compl.: 0.69
  Median Shape Compl.: 0.700225

Average for both molecules:
      Mean Separation: 0.56
    Median Separation: 0.522067
    Mean Shape Compl.: 0.695
  Median Shape Compl.: 0.706234

==================================================
Shape Complementarity:          0.706234
Interface separation (A):       0.522067
Area buried in interface (A^2): 1055.74
==================================================
"""


def test_parse_verbose():
    r = rosetta.parse_verbose(SAMPLE_LOG)
    assert r.sc == pytest.approx(0.706234)
    assert r.distance == pytest.approx(0.522067)
    assert r.area == pytest.approx(1055.74)
    a, b = r.sides
    assert (a.n_atoms, a.n_buried_atoms, a.n_dots, a.n_trimmed) == (
        922,
        266,
        25258,
        7765,
    )
    assert b.trimmed_area == pytest.approx(535.334)
    assert b.s_median == pytest.approx(0.700225)
    assert a.d_median == pytest.approx(0.512273)


def test_parse_rejects_failed_run():
    with pytest.raises(RuntimeError, match="unparsable"):
        rosetta.parse_verbose("Failed: No atoms defined for molecule 2")


@pytest.mark.skipif(not rosetta.available(), reason="rosetta sc not found")
def test_run_sc_1ar1(test_data):
    r = rosetta.run_sc(test_data / "1ar1.pdb", "H", "L")
    assert r.sc == pytest.approx(0.706, abs=1e-3)
    assert r.sides[0].n_buried_atoms == 266

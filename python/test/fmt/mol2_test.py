#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import List

import numpy as np
import pytest

import nuri
from nuri.core import Hyb, Molecule

mol2_data = """\
@<TRIPOS>MOLECULE
Methane
 1 0 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
 1 C1            -0.0000    -0.0000    -0.0000 C.3
@<TRIPOS>MOLECULE
CHEMBL1087078
  22   25    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1            -7.3016     3.4125     1.5950 C.1
   2 C2            -6.8163     3.4528     0.5267 C.1
   3 C3            -6.2078     3.5033    -0.8127 C.3
   4 O4            -4.8703     3.0044    -0.7474 O.3
   5 C5            -4.1650     2.9826    -1.9035 C.ar
   6 N6            -4.7086     3.4086    -3.0303 N.ar
   7 C7            -4.0263     3.3919    -4.1710 C.ar
   8 C8            -4.6819     3.8812    -5.4075 C.ar
   9 C9            -3.9864     3.8801    -6.6157 C.ar
  10 C10           -4.6040     4.3309    -7.7645 C.ar
  11 C11           -5.9065     4.7944    -7.7166 C.ar
  12 C12           -6.5995     4.8036    -6.5194 C.ar
  13 C13           -5.9943     4.3494    -5.3653 C.ar
  14 N14           -2.7820     2.9532    -4.2369 N.ar
  15 C15           -2.1554     2.5034    -3.1431 C.ar
  16 S16           -0.5363     1.8675    -2.9175 S.3
  17 C17           -0.7945     1.6130    -1.1920 C.2
  18 C18           -0.0127     1.0858     0.0080 C.3
  19 C19           -0.7790     1.6052     1.2407 C.3
  20 C20           -2.2075     1.7525     0.6729 C.3
  21 C21           -2.0135     1.9837    -0.8258 C.2
  22 C22           -2.8314     2.5027    -1.9251 C.ar
@<TRIPOS>BOND
   1    1    2 3
   2    2    3 1
   3    3    4 1
   4    4    5 1
   5    5   22 ar
   6    5    6 ar
   7    6    7 ar
   8    7    8 1
   9    7   14 ar
  10    8   13 ar
  11    8    9 ar
  12    9   10 ar
  13   10   11 ar
  14   11   12 ar
  15   12   13 ar
  16   14   15 ar
  17   15   22 ar
  18   15   16 1
  19   16   17 1
  20   17   21 2
  21   17   18 1
  22   18   19 1
  23   19   20 1
  24   20   21 1
  25   21   22 1
"""


def _verify_mols(mols: List[Molecule]):
    assert len(mols) == 2

    assert len(mols[0]) == 1

    mol = mols[1]
    assert len(mol) == 22
    assert mol.num_bonds() == 25

    for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21]:
        assert mol[i].hyb == Hyb.SP2
        assert mol[i].aromatic

    for i in [17, 18, 19]:
        assert mol[i].hyb == Hyb.SP3

    # Bridge bond between two aromatic rings
    assert not mol.bond(6, 7).aromatic


def test_mol2_file(tmp_path: Path):
    file = tmp_path / "test.mol2"
    file.write_text(mol2_data)

    mols = list(nuri.readfile("mol2", file))
    _verify_mols(mols)


def test_mol2_str():
    mols = list(nuri.readstring("mol2", mol2_data))
    _verify_mols(mols)

    mol2_re = "".join(map(nuri.to_mol2, mols))
    mols_re = list(nuri.readstring("mol2", mol2_re))
    _verify_mols(mols_re)


def test_mol2_options(mol3d: Molecule):
    mol2s = nuri.to_mol2(mol3d)

    mols = list(nuri.readstring("mol2", mol2s))
    assert len(mols) == 2

    mol2s = nuri.to_mol2(mol3d, conf=1)
    mols = list(nuri.readstring("mol2", mol2s))
    assert len(mols) == 1
    assert np.allclose(mol3d.get_conf(1), mols[0].get_conf(0), atol=1e-3)

    with pytest.raises(IndexError):
        mol2s = nuri.to_mol2(mol3d, conf=2)

    mol2s = nuri.to_mol2(mol3d, write_sub=False)
    assert "@<TRIPOS>SUBSTRUCTURE" not in mol2s

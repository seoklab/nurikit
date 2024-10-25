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

sdf_data = """\
L-Alanine
  ABCDEFGH09071717443D
Exported
  6  5  0  0  1  0              3 V2000
   -0.6622    0.5342    0.0000 C   0  0  2  0  0  0
    0.6622   -0.3000    0.0000 C   0  0  0  0  0  0
   -0.7207    2.0817    0.0000 C   1  0  0  0  0  0
   -1.8622   -0.3695    0.0000 N   0  3  0  0  0  0
    0.6220   -1.8037    0.0000 O   0  0  0  0  0  0
    1.9464    0.4244    0.0000 O   0  5  0
  1  2  1  0  0  0
  1  3  1  1  0  0
  1  4  1  0  0  0
  2  5  2  0  0  0
  2  6  1  0  0  0
M  CHG  2   4   1   6  -1
M  ISO  1   3  13
M  END
> 25  <MELTING.POINT>
179.0 - 183.0

> 25  <DESCRIPTION>
PW(W)

$$$$
L-Alanine
 OpenBabel03182412213D
Exported
  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 6 5 0 0 1
M  V30 BEGIN ATOM
M  V30 1 C -0.6622 0.5342 0 0
M  V30 2 C 0.6622 -0.3 0 0
M  V30 3 C -0.7207 2.0817 0 0 MASS=13
M  V30 4 N -1.8622 -0.3695 0 0 CHG=1
M  V30 5 O 0.622 -1.8037 0 0
M  V30 6 O 1.9464 0.4244 0 0 CHG=-1
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 1 3 CFG=1
M  V30 3 1 1 4
M  V30 4 2 2 5
M  V30 5 1 2 6
M  V30 END BOND
M  V30 END CTAB
M  END
>  <MELTING.POINT>
179.0 - 183.0

>  <DESCRIPTION>
PW(W)

$$$$
"""


def _verify_mols(mols: List[Molecule]):
    assert len(mols) == 2

    for mol in mols:
        assert mol[0].hyb == Hyb.SP3

        assert mol[1].hyb == Hyb.SP2
        assert mol[1].conjugated

        assert mol[2].hyb == Hyb.SP3

        assert mol[3].hyb == Hyb.SP3
        assert mol[3].formal_charge == 1

        assert mol[4].formal_charge == 0
        assert mol[4].hyb == Hyb.Terminal
        assert mol[5].formal_charge == -1
        assert mol[5].hyb == Hyb.Terminal

        assert mol.props["MELTING.POINT"] == "179.0 - 183.0"
        assert mol.props["DESCRIPTION"] == "PW(W)"


def test_sdf_file(tmp_path: Path):
    file = tmp_path / "test.sdf"
    file.write_text(sdf_data)

    mols = list(nuri.readfile("sdf", file))
    _verify_mols(mols)


def test_sdf_str():
    mols = list(nuri.readstring("sdf", sdf_data))
    _verify_mols(mols)

    sdf_re = "".join(map(nuri.to_sdf, mols))
    mols_re = list(nuri.readstring("sdf", sdf_re))
    _verify_mols(mols_re)


def test_sdf_options(mol3d: Molecule):
    sdfs = nuri.to_sdf(mol3d)

    mols = list(nuri.readstring("sdf", sdfs))
    assert len(mols) == 2

    sdfs = nuri.to_sdf(mol3d, version=2000)
    mols = list(nuri.readstring("sdf", sdfs))
    assert len(mols) == 2

    sdfs = nuri.to_sdf(mol3d, version=3000)
    mols = list(nuri.readstring("sdf", sdfs))
    assert len(mols) == 2

    sdfs = nuri.to_sdf(mol3d, conf=1)
    mols = list(nuri.readstring("sdf", sdfs))
    assert len(mols) == 1
    assert np.allclose(mol3d.get_conf(1), mols[0].get_conf(0), atol=1e-3)

    with pytest.raises(IndexError):
        sdfs = nuri.to_sdf(mol3d, conf=2)

    with pytest.raises(ValueError, match="Invalid SDF version"):
        sdfs = nuri.to_sdf(mol3d, version=9999)

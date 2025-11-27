#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

import nuri
from nuri.core import Molecule
from nuri.desc import shrake_rupley_sasa


@pytest.fixture()
def phenol():
    sdf = """
     RDKit          3D

 13 13  0  0  0  0  0  0  0  0999 V2000
    2.4823    0.0194   -0.3935 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.1083    0.0083   -0.2127 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4038   -1.1740   -0.1422 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9783   -1.1878    0.0395 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6848   -0.0082    0.1548 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9667    1.1675    0.0823 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4040    1.1892   -0.0978 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.1213    0.0076    0.3925 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9144   -2.1260   -0.2274 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5181   -2.1041    0.0934 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7594   -0.0193    0.2961 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5030    2.1147    0.1708 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9759    2.1126   -0.1557 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  3  4  1  0
  4  5  2  0
  5  6  1  0
  6  7  2  0
  7  2  1  0
  1  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
  7 13  1  0
M  END
"""
    mol = next(nuri.readstring("sdf", sdf))
    return mol


def test_sr_sasa_mol(phenol: Molecule):
    sasa = shrake_rupley_sasa(phenol, nprobe=100)

    ref = np.array(
        [
            25.363962448022551,
            5.0511280666653553,
            13.89060218332973,
            17.678948233328743,
            16.416166216662404,
            15.153384199996067,
            13.89060218332973,
            31.431006180635155,
            22.936139645328353,
            23.785626298859032,
            23.785626298859032,
            26.334086259451073,
            21.237166338266992,
        ]
    )

    np.testing.assert_allclose(sasa, ref, rtol=1e-5)


def test_sr_sasa_coord(phenol: Molecule):
    pts = phenol.get_conf()
    radii = np.array([atom.element.vdw_radius for atom in phenol])

    sasa = shrake_rupley_sasa(pts, radii, nprobe=100)

    ref = np.array(
        [
            25.363962448022551,
            5.0511280666653553,
            13.89060218332973,
            17.678948233328743,
            16.416166216662404,
            15.153384199996067,
            13.89060218332973,
            31.431006180635155,
            22.936139645328353,
            23.785626298859032,
            23.785626298859032,
            26.334086259451073,
            21.237166338266992,
        ]
    )

    np.testing.assert_allclose(sasa, ref, rtol=1e-5)


def test_sr_sasa_errors(phenol: Molecule):
    pts = phenol.get_conf()
    radii = np.array([atom.element.vdw_radius for atom in phenol])

    with pytest.raises(ValueError, match="number of points"):
        shrake_rupley_sasa(pts, radii[:-1])

    with pytest.raises(ValueError, match="number of probes"):
        shrake_rupley_sasa(pts, radii, nprobe=0)

    with pytest.raises(ValueError, match="radius of probes"):
        shrake_rupley_sasa(pts, radii, rprobe=-1.0)

    radii[-2] = -1.0
    with pytest.raises(ValueError, match="radii must be positive"):
        shrake_rupley_sasa(pts, radii)

#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri import algo
from nuri.core import Molecule
from nuri.fmt import readstring


@pytest.fixture()
def sample():
    mol = next(readstring("smi", "CC(=O)OC1CCCC2COC(=O)C21"))
    return mol


def test_crdgen_distgeom(sample: Molecule):
    algo.generate_coords(sample, "DG")

    assert sample.num_confs() == 1

    conf = sample.get_conf()
    assert np.allclose(
        conf,
        [
            [-2.660, 1.200, 0.539],
            [-1.787, 0.051, 0.036],
            [-1.671, -0.981, 0.715],
            [-1.188, 0.148, -1.095],
            [0.111, 0.683, -1.280],
            [0.163, 2.150, -0.882],
            [0.144, 2.286, 0.634],
            [1.209, 1.469, 1.313],
            [1.509, 0.149, 0.783],
            [0.970, -1.042, 1.575],
            [0.686, -2.046, 0.589],
            [0.840, -1.641, -0.629],
            [0.469, -2.281, -1.618],
            [1.205, -0.144, -0.679],
        ],
        atol=1e-2,
        rtol=0,
    )

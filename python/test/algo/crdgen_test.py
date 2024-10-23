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
            [-2.935, 1.012, 0.302],
            [-1.763, 0.055, 0.158],
            [-1.909, -1.037, 0.717],
            [-1.160, 0.101, -0.966],
            [0.103, 0.678, -1.212],
            [0.163, 2.143, -0.808],
            [0.313, 2.278, 0.700],
            [1.506, 1.506, 1.230],
            [1.613, 0.105, 0.744],
            [0.757, -0.897, 1.489],
            [0.689, -1.977, 0.565],
            [0.957, -1.626, -0.639],
            [0.404, -2.209, -1.577],
            [1.262, -0.131, -0.703],
        ],
        atol=1e-2,
        rtol=0,
    )

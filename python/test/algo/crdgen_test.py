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
            [-3.47601, -0.0353863, -0.915114],
            [-2.68436, 0.614118, 0.208193],
            [-3.18419, 0.779729, 1.32576],
            [-1.50221, 1.07873, 0.0670281],
            [-0.477185, 0.435378, -0.6758],
            [0.47463, 1.5616, -1.04135],
            [1.27018, 2.05112, 0.159791],
            [1.9832, 0.912623, 0.871179],
            [1.05167, -0.226333, 1.23773],
            [2.0347, -1.38704, 1.33505],
            [2.1559, -1.84728, -0.00663011],
            [1.16153, -1.48779, -0.732103],
            [1.02281, -1.77099, -1.9257],
            [0.169328, -0.678476, 0.0919661],
        ],
        atol=5e-2,
        rtol=0,
    )

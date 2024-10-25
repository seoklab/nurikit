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
            [-2.98387, 0.925722, 0.117226],
            [-1.7538, 0.0360538, 0.0410949],
            [-1.91009, -1.10937, 0.476756],
            [-1.07767, 0.118291, -1.03789],
            [0.194453, 0.710658, -1.18149],
            [0.213719, 2.16437, -0.735599],
            [0.257627, 2.2591, 0.782613],
            [1.41786, 1.48184, 1.37349],
            [1.57076, 0.0952238, 0.859545],
            [0.676682, -0.933103, 1.51782],
            [0.680156, -1.98755, 0.561759],
            [1.02689, -1.60207, -0.610883],
            [0.364725, -2.05504, -1.55033],
            [1.32255, -0.104127, -0.614107],
        ],
        atol=5e-2,
        rtol=0,
    )

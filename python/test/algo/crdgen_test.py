#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri import algo
from nuri.core import Molecule
from nuri.core.geometry import align_rmsd
from nuri.fmt import readstring


@pytest.fixture()
def sample():
    mol = next(readstring("smi", "CC(=O)OC1CCCC2COC(=O)C21"))
    return mol


def test_crdgen_distgeom(sample: Molecule):
    algo.generate_coords(sample, "DG")

    assert sample.num_confs() == 1

    conf = sample.get_conf()
    assert (
        align_rmsd(
            conf,
            [
                [-3.27734, 1.85008, 0.331878],
                [-2.35362, 0.643631, 0.291369],
                [-2.16021, 0.0192391, 1.33967],
                [-1.8673, 0.31158, -0.842877],
                [-0.478895, 0.165713, -1.10243],
                [0.250714, 1.49762, -1.06163],
                [0.547288, 1.94428, 0.36239],
                [1.27779, 0.871516, 1.15371],
                [0.569783, -0.468935, 1.11893],
                [1.72724, -1.41762, 1.40761],
                [2.3178, -1.63682, 0.130978],
                [1.50836, -1.37125, -0.827468],
                [1.77025, -1.5044, -2.0266],
                [0.168125, -0.904638, -0.275532],
            ],
        )
        <= 0.1
    )

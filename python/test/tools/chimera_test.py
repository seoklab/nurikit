#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import numpy as np
import pytest

from nuri.fmt.pdb import Model, read_models
from nuri.tools import chimera


def _read_calphas(model: Model):
    idxs = [
        i
        for res in model.residues
        for i in res.atom_idxs
        if model.atoms[i].name == "CA"
    ]
    return model.major_conf[idxs]


@pytest.fixture()
def query(test_data: Path):
    model = read_models(test_data / "fixed_ref1.pdb")[0]
    return _read_calphas(model)


@pytest.fixture()
def templ(test_data: Path):
    model = read_models(test_data / "fixed_ref2.pdb")[0]
    return _read_calphas(model)


def test_match_maker(query: np.ndarray, templ: np.ndarray):
    mm = chimera.match_maker(query, templ)

    ref_xform = np.array(
        [
            [-0.14150449, 0.67616094, -0.72303725, 16.25602271],
            [-0.26213529, -0.72990790, -0.63128405, 38.51123346],
            [-0.95460021, 0.10020405, 0.28053089, 17.46343176],
            [0, 0, 0, 1],
        ]
    )
    # fmt: off
    ref_sel = np.array(
        [
            16, 40, 78, 80, 99, 102, 103, 104, 105, 106, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125,
            126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
            139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
            153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
            166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
            179, 180, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
            194, 195, 196,
        ]
    )
    # fmt: on

    np.testing.assert_allclose(mm.transform, ref_xform, rtol=1e-6)
    assert mm.aligned_rmsd == pytest.approx(0.698, abs=1e-3)

    mm.selected.sort()
    np.testing.assert_array_equal(mm.selected, ref_sel)


def test_match_maker_errors(query: np.ndarray, templ: np.ndarray):
    with pytest.raises(ValueError, match="have different number of points"):
        chimera.match_maker(query[:3], templ)

    with pytest.raises(ValueError, match="cutoff must be positive"):
        chimera.match_maker(query, templ, -1)

    with pytest.raises(
        ValueError,
        match="global_ratio must be between 0 and 1",
    ):
        chimera.match_maker(query, templ, global_ratio=1.5)

    with pytest.raises(
        ValueError,
        match="viol_ratio must be between 0 and 1",
    ):
        chimera.match_maker(query, templ, viol_ratio=1.5)

    query[0] = np.nan
    with pytest.raises(RuntimeError, match="alignment failed"):
        chimera.match_maker(query, templ)

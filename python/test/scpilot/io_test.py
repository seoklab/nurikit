#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from nuri.scpilot.io import load_structure


def test_load_1ar1(test_data):
    st = load_structure(test_data / "1ar1.pdb")
    assert len(st) == 3728
    assert st.coords.shape == (3728, 3)
    assert np.all(st.atomic_numbers > 1)
    assert np.all(st.vdw_radii > 1.0)

    counts = {c: int((st.chain_ids == c).sum()) for c in "AHL"}
    assert counts == {"A": 1976, "H": 921, "L": 831}

    hl = st.chains("HL")
    assert len(hl) == 921 + 831
    assert set(hl.chain_ids) == {"H", "L"}
    assert hl.atom_names[0] == "N"
    assert hl.residue_names[0] == "GLU"
    assert hl.residue_seqs[0] == 1

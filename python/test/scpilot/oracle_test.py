#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from nuri.scpilot import rosetta
from nuri.scpilot.io import load_structure
from nuri.scpilot.radii import united_atom_radii
from nuri.scpilot.sc import ScParams, shape_complementarity

pytestmark = pytest.mark.skipif(
    not rosetta.available(), reason="rosetta sc not found"
)


INTERFACES = [
    ("1ar1", "H", "L"),
    ("1ar1", "A", "HL"),
    ("1ar1", "A", "H"),
    ("1ar1", "A", "L"),
    ("2ptc", "E", "I"),
    ("1brs", "A", "D"),
    ("1brs", "B", "E"),
    ("1brs", "C", "F"),
    ("1vfb", "AB", "C"),
    ("1vfb", "A", "B"),
    ("1cho", "EFG", "I"),
]


@pytest.mark.parametrize(("name", "a", "b"), INTERFACES)
def test_matches_rosetta(test_data, name, a, b):
    pdb = test_data / f"{name}.pdb"
    st = load_structure(pdb)
    sa, sb = st.chains(a), st.chains(b)
    ra, _ = united_atom_radii(sa)
    rb, _ = united_atom_radii(sb)

    ours = shape_complementarity(sa.coords, ra, sb.coords, rb, ScParams())
    ref = rosetta.run_sc(pdb, a, b)

    assert ours.sc == pytest.approx(ref.sc, abs=0.02)
    assert ours.distance == pytest.approx(ref.distance, abs=0.1)
    assert ours.area == pytest.approx(ref.area, rel=0.1)
    for mine, theirs in zip(ours.sides, ref.sides):
        assert mine.n_active == pytest.approx(theirs.n_buried_atoms, abs=12)
        assert mine.s_median == pytest.approx(theirs.s_median, abs=0.03)

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri.desc import _shape_complementarity
from nuri.scpilot import rosetta
from nuri.scpilot import sc as pilot
from nuri.scpilot.io import load_structure
from nuri.scpilot.radii import united_atom_radii


def slab(z: float, n: int = 7, spacing: float = 3.0):
    xs = (np.arange(n) - (n - 1) / 2) * spacing
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    return np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, z, dtype=float)]
    )


def test_facing_slabs():
    radii = np.full(49, 1.7)
    a = slab(0.0)
    b = slab(3.4) + np.array([1.5, 1.5, 0.0])
    tight = _shape_complementarity(a, radii, b, radii)
    assert 0.6 < tight["sc"] <= 0.999
    assert tight["distance"] < 1.0
    assert tight["area"] > 0.0
    assert all(s["n_trimmed"] > 0 for s in tight["sides"])

    b_far = slab(4.4) + np.array([1.5, 1.5, 0.0])
    loose = _shape_complementarity(a, radii, b_far, radii)
    assert loose["sc"] < tight["sc"]
    assert loose["distance"] > tight["distance"]


def test_no_interface_raises():
    a = np.zeros((1, 3))
    b = np.array([[30.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no interface"):
        _shape_complementarity(a, [1.7], b, [1.7])


def test_invalid_inputs():
    a = np.zeros((1, 3))
    with pytest.raises(ValueError, match="number of points"):
        _shape_complementarity(a, [1.7, 1.7], a, [1.7])
    with pytest.raises(ValueError, match="positive"):
        _shape_complementarity(a, [0.0], a, [1.7])
    with pytest.raises(ValueError, match="clamp"):
        _shape_complementarity(a, [1.7], a, [1.7], clamp=0.0)
    with pytest.raises(ValueError, match="at least one atom"):
        _shape_complementarity(np.zeros((0, 3)), [], a, [1.7])


def interface(test_data, name, chains_a, chains_b):
    st = load_structure(test_data / f"{name}.pdb")
    sa, sb = st.chains(chains_a), st.chains(chains_b)
    ra, _ = united_atom_radii(sa)
    rb, _ = united_atom_radii(sb)
    return sa.coords, ra, sb.coords, rb


def test_matches_pilot_1ar1(test_data):
    ca, ra, cb, rb = interface(test_data, "1ar1", "H", "L")
    ours = _shape_complementarity(ca, ra, cb, rb)
    ref = pilot.shape_complementarity(ca, ra, cb, rb)

    assert ours["sc"] == pytest.approx(ref.sc, abs=0.02)
    assert ours["distance"] == pytest.approx(ref.distance, abs=0.1)
    assert ours["area"] == pytest.approx(ref.area, rel=0.1)
    for mine, theirs in zip(ours["sides"], ref.sides):
        assert mine["n_atoms"] == theirs.n_atoms
        assert mine["n_active"] == theirs.n_active
        assert mine["s_median"] == pytest.approx(theirs.s_median, abs=0.03)
        assert mine["n_dots"] == pytest.approx(theirs.n_dots, rel=0.02)


@pytest.mark.skipif(not rosetta.available(), reason="rosetta sc not found")
def test_matches_rosetta_1ar1(test_data):
    pdb = test_data / "1ar1.pdb"
    ca, ra, cb, rb = interface(test_data, "1ar1", "H", "L")
    ours = _shape_complementarity(ca, ra, cb, rb)
    ref = rosetta.run_sc(pdb, "H", "L")

    assert ours["sc"] == pytest.approx(ref.sc, abs=0.02)
    assert ours["distance"] == pytest.approx(ref.distance, abs=0.1)
    assert ours["area"] == pytest.approx(ref.area, rel=0.1)
    for mine, theirs in zip(ours["sides"], ref.sides):
        assert mine["n_active"] == pytest.approx(theirs.n_buried_atoms, abs=12)
        assert mine["s_median"] == pytest.approx(theirs.s_median, abs=0.03)

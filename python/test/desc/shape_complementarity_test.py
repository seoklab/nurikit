#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import numpy as np
import pytest

import nuri
from nuri.desc import ScResult, ScSide, SurfacePatch, shape_complementarity
from nuri.fmt.pdb import read_models

SLAB_N = 7
SLAB_ATOMS = SLAB_N**2


def slab(z: float, n: int = SLAB_N, spacing: float = 3.0):
    xs = (np.arange(n) - (n - 1) / 2) * spacing
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    return np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, z, dtype=float)]
    )


@pytest.fixture()
def radii():
    return np.full(SLAB_ATOMS, 1.7)


@pytest.fixture()
def slab_a():
    return slab(0.0)


@pytest.fixture()
def slab_b():
    return slab(3.4) + np.array([1.5, 1.5, 0.0])


@pytest.fixture()
def tight(slab_a: np.ndarray, slab_b: np.ndarray, radii: np.ndarray):
    return shape_complementarity(slab_a, radii, slab_b, radii)


def test_facing_slabs(tight: ScResult, slab_a: np.ndarray, radii: np.ndarray):
    assert 0.6 < tight.sc <= 0.999
    assert tight.distance < 1.0
    assert tight.area > 0.0
    assert all(s.n_trimmed > 0 for s in tight.sides)

    b_far = slab(4.4) + np.array([1.5, 1.5, 0.0])
    loose = shape_complementarity(slab_a, radii, b_far, radii)
    assert loose.sc < tight.sc
    assert loose.distance > tight.distance


def test_no_interface_raises():
    a = np.zeros((1, 3))
    b = np.array([[30.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no interface"):
        shape_complementarity(a, [1.7], b, [1.7])


def test_invalid_inputs():
    a = np.zeros((1, 3))
    with pytest.raises(ValueError, match="at least one atom"):
        shape_complementarity(np.zeros((0, 3)), [], a, [1.7])
    with pytest.raises(ValueError, match="clamp"):
        shape_complementarity(a, [1.7], a, [1.7], clamp=0.0)
    with pytest.raises(ValueError, match="clamp"):
        shape_complementarity(a, [1.7], a, [1.7], clamp=1.5)
    with pytest.raises(ValueError, match="rp"):
        shape_complementarity(a, [1.7], a, [1.7], rp=0.0)
    with pytest.raises(ValueError, match="density"):
        shape_complementarity(a, [1.7], a, [1.7], density=0.0)
    with pytest.raises(ValueError, match="number of points"):
        shape_complementarity(a, [1.7, 1.7], a, [1.7])
    with pytest.raises(ValueError, match="positive"):
        shape_complementarity(a, [0.0], a, [1.7])
    with pytest.raises(ValueError, match="NaN or infinite"):
        shape_complementarity(a, [np.nan], a, [1.7])
    with pytest.raises(ValueError, match="finite"):
        shape_complementarity(a, [1.7], a, [1.7], weight=np.nan)


def _check_side(side: ScSide):
    assert side.n_trimmed <= side.n_buried <= side.n_dots
    assert side.n_active <= side.n_atoms

    dots = side.dots
    assert dots.kind == "ses"
    assert len(dots) == side.n_trimmed
    assert len(side.distance) == side.n_trimmed
    assert len(side.score) == side.n_trimmed
    assert dots.pts.shape == (side.n_trimmed, 3)
    assert dots.normals.shape == (side.n_trimmed, 3)
    assert dots.areas.shape == (side.n_trimmed,)
    assert dots.atoms.shape == (side.n_trimmed,)

    assert side.trimmed_area == pytest.approx(dots.areas.sum(), rel=1e-12)
    assert np.all(dots.atoms >= 0)
    assert np.all(dots.atoms < side.n_atoms)
    assert np.linalg.norm(dots.normals, axis=1) == pytest.approx(
        np.ones(side.n_trimmed), rel=1e-9
    )

    assert sum(dots.count(p) for p in SurfacePatch) == len(dots)
    assert dots.offset(SurfacePatch.Convex) == 0
    patches = list(SurfacePatch)
    for prev, nxt in zip(patches, patches[1:]):
        assert dots.offset(nxt) == dots.offset(prev) + dots.count(prev)

    assert np.median(side.distance) == pytest.approx(side.d_median, rel=1e-12)
    assert np.median(side.score) == pytest.approx(side.s_median, rel=1e-12)


def test_side_consistency(tight: ScResult):
    a, b = tight.sides
    _check_side(a)
    _check_side(b)

    assert tight.sc == pytest.approx(
        0.5 * (a.s_median + b.s_median), rel=1e-12
    )
    assert tight.distance == pytest.approx(
        0.5 * (a.d_median + b.d_median), rel=1e-12
    )
    assert tight.area == pytest.approx(
        a.trimmed_area + b.trimmed_area, rel=1e-12
    )


def test_views_readonly_and_owned(tight: ScResult):
    side = tight.sides[0]
    assert side.dots.pts.flags.writeable is False
    assert side.dots.normals.flags.writeable is False
    assert side.dots.areas.flags.writeable is False
    assert side.dots.atoms.flags.writeable is False
    assert side.distance.flags.writeable is False
    assert side.score.flags.writeable is False

    dots = side.dots
    pts = dots.pts
    total = pts.sum()
    del tight, side, dots
    assert pts.sum() == total


def test_swapped_sides(
    slab_a: np.ndarray, slab_b: np.ndarray, radii: np.ndarray
):
    nuri.seed_thread(42)
    ab = shape_complementarity(slab_a, radii, slab_b, radii)
    nuri.seed_thread(42)
    ba = shape_complementarity(slab_b, radii, slab_a, radii)

    assert ab.sc == pytest.approx(ba.sc, rel=1e-12)
    assert ab.distance == pytest.approx(ba.distance, rel=1e-12)
    assert ab.area == pytest.approx(ba.area, rel=1e-12)

    for mine, theirs in zip(ab.sides, reversed(ba.sides)):
        assert mine.n_atoms == theirs.n_atoms
        assert mine.n_active == theirs.n_active
        assert mine.n_dots == theirs.n_dots
        assert mine.n_buried == theirs.n_buried
        assert mine.n_trimmed == theirs.n_trimmed
        assert mine.trimmed_area == pytest.approx(
            theirs.trimmed_area, rel=1e-12
        )
        assert mine.d_median == pytest.approx(theirs.d_median, rel=1e-12)
        assert mine.s_median == pytest.approx(theirs.s_median, rel=1e-12)


def test_active_masks(
    slab_a: np.ndarray, slab_b: np.ndarray, radii: np.ndarray
):
    extra = slab(10.9) + np.array([1.5, 1.5, 0.0])
    a_ext = np.vstack([slab_a, extra])
    radii_ext = np.full(len(a_ext), 1.7)
    mask = np.ones(len(a_ext), dtype=bool)
    mask[len(slab_a) :] = False

    nuri.seed_thread(5)
    plain = shape_complementarity(slab_a, radii, slab_b, radii)
    nuri.seed_thread(5)
    masked = shape_complementarity(
        a_ext, radii_ext, slab_b, radii, active_a=mask
    )
    nuri.seed_thread(5)
    unmasked = shape_complementarity(a_ext, radii_ext, slab_b, radii)

    assert masked.sc == pytest.approx(plain.sc, rel=1e-12)
    assert masked.distance == pytest.approx(plain.distance, rel=1e-12)
    assert masked.area == pytest.approx(plain.area, rel=1e-12)
    assert masked.sides[0].n_atoms == len(a_ext)
    assert masked.sides[0].n_active == plain.sides[0].n_active
    assert masked.sides[0].n_dots == plain.sides[0].n_dots
    assert unmasked.sides[0].n_active == len(a_ext)
    assert unmasked.sides[0].n_dots > masked.sides[0].n_dots

    with pytest.raises(ValueError, match="active_b size"):
        shape_complementarity(slab_a, radii, slab_b, radii, active_b=mask)
    with pytest.raises(ValueError, match="no interface"):
        shape_complementarity(
            slab_a,
            radii,
            slab_b,
            radii,
            active_a=np.zeros(SLAB_ATOMS, dtype=bool),
        )


def _heavy_atoms(model, chain: str):
    idxs = [
        i
        for i, atom in enumerate(model.atoms)
        if not atom.hetero
        and atom.element.atomic_number > 1
        and atom.res_id.chain_id == chain
    ]
    pts = model.major_conf[idxs]
    radii = [model.atoms[i].element.vdw_radius for i in idxs]
    return pts, radii


def test_barnase_barstar(test_data: Path):
    model = read_models(test_data / "1brs.pdb")[0]
    pts_a, radii_a = _heavy_atoms(model, "A")
    pts_b, radii_b = _heavy_atoms(model, "D")
    assert len(radii_a) > 500
    assert len(radii_b) > 500

    nuri.seed_thread(0)
    res = shape_complementarity(pts_a, radii_a, pts_b, radii_b)
    assert res.sc == pytest.approx(0.660846137477, abs=1e-6)
    assert res.distance == pytest.approx(0.639743237709, abs=1e-6)
    assert res.area == pytest.approx(906.024644525, abs=1e-6)

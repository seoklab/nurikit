#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nuri.scpilot.anal import (
    SasGeometry,
    SesGeometry,
    ses_area,
    two_sphere_ses_area,
)
from nuri.scpilot.io import load_structure
from nuri.scpilot.surface import (
    Patch,
    fibonacci_sphere,
    sample_ses,
    ses_dots,
)


def analytic_areas(coords, radii, rp):
    ses = SesGeometry.build(SasGeometry.from_atoms(coords, radii, rp)[0])
    convex, saddle, concave = ses_area(ses)
    return float(np.nansum(convex)), float(saddle.sum()), float(concave.sum())


def sampled_areas(dots):
    return tuple(dots.area(k) for k in Patch)


def assert_valid_ses(dots, coords, radii, rp):
    np.testing.assert_allclose(np.linalg.norm(dots.normals, axis=1), 1.0)

    sas = np.asarray(radii) + rp
    tree = cKDTree(coords)
    probes = cKDTree(dots.probes).sparse_distance_matrix(
        tree, sas.max(), output_type="coo_matrix"
    )
    assert not np.any(probes.data < sas[probes.col] * (1 - 1e-6))

    inside = cKDTree(dots.pts).sparse_distance_matrix(
        tree, np.max(radii), output_type="coo_matrix"
    )
    assert not np.any(inside.data < np.asarray(radii)[inside.col] * (1 - 1e-6))


def test_fibonacci_sphere_uniform():
    pts = fibonacci_sphere(1000)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0)
    np.testing.assert_allclose(pts.mean(axis=0), 0.0, atol=2e-3)


def test_single_sphere():
    dots = ses_dots(np.zeros((1, 3)), [1.5], rp=1.7, density=50)
    assert np.all(dots.kinds == Patch.CONVEX)
    assert dots.area() == pytest.approx(4 * math.pi * 1.5**2)
    np.testing.assert_allclose(
        dots.normals, (dots.pts - 0.0) / 1.5, atol=1e-12
    )


def test_two_spheres_stamm_table1():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = [1.2, 1.2]
    dots = ses_dots(coords, radii, rp=1.2, density=200)
    convex, torus, _ = two_sphere_ses_area(1.2, 1.2, 2.0, 1.2)
    assert convex + torus == pytest.approx(32.23514, abs=1e-4)
    assert dots.area(Patch.CONVEX) == pytest.approx(convex, rel=2e-3)
    assert dots.area(Patch.TOROIDAL) == pytest.approx(torus, rel=2e-3)
    assert dots.area(Patch.CONCAVE) == 0.0
    assert_valid_ses(dots, coords, radii, 1.2)


def test_two_spheres_rings_merged_and_staggered():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rp, density = 1.2, 200
    sas, _ = SasGeometry.from_atoms(coords, [1.2, 1.2], rp)
    dots = sample_ses(SesGeometry.build(sas), density)
    tor = dots.pts[dots.kinds == Patch.TOROIDAL]

    rl = math.sqrt(2.4**2 - 1.0)
    width = 2.0 * math.atan2(1.0, rl)
    x, ring = np.unique(tor[:, 0].round(9), return_inverse=True)
    assert len(x) == round(rp * width * math.sqrt(density))

    assert len(sas.arcs) == 1
    (arc,) = sas.arcs
    circle = sas.circles[arc.circle]
    for m in range(len(x)):
        p = tor[ring == m]
        phi = (np.arctan2(p @ circle.e2, p @ circle.e1) - arc.phi_beg) % (
            2 * math.pi
        )
        frac = (np.sort(phi) * len(phi) / (2 * math.pi)) % 1.0
        np.testing.assert_allclose(frac, 0.25 + 0.5 * (m % 2), atol=1e-9)


def test_two_spheres_cusp():
    coords = np.array([[0.0, 0.0, 0.0], [4.4, 0.0, 0.0]])
    radii = [1.2, 1.2]
    dots = ses_dots(coords, radii, rp=1.2, density=200)
    convex, torus, _ = two_sphere_ses_area(1.2, 1.2, 4.4, 1.2)
    assert dots.area(Patch.CONVEX) == pytest.approx(convex, rel=2e-3)
    assert dots.area(Patch.TOROIDAL) == pytest.approx(torus, rel=5e-3)
    assert_valid_ses(dots, coords, radii, 1.2)


def test_three_spheres_vs_analytic():
    s = 3.0
    coords = np.array(
        [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s / 2, s * math.sqrt(3) / 2, 0.0]]
    )
    radii = [1.5] * 3
    dots = ses_dots(coords, radii, rp=1.4, density=200)
    ref = analytic_areas(coords, radii, 1.4)
    assert ref[2] > 0.0
    np.testing.assert_allclose(sampled_areas(dots), ref, rtol=5e-3)
    assert_valid_ses(dots, coords, radii, 1.4)


def test_random_cluster_vs_analytic():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 6.0, size=(8, 3))
    radii = rng.uniform(1.4, 1.9, size=8)
    dots = ses_dots(coords, radii, rp=1.4, density=200)
    ref = analytic_areas(coords, radii, 1.4)
    np.testing.assert_allclose(sampled_areas(dots), ref, rtol=5e-3)
    assert_valid_ses(dots, coords, radii, 1.4)


def test_protein_fragment(test_data):
    st = load_structure(test_data / "1ar1.pdb").chains("H")
    st = st.subset(st.residue_seqs <= 12)
    dots = ses_dots(st.coords, st.vdw_radii, rp=1.7, density=15)
    assert_valid_ses(dots, st.coords, st.vdw_radii, 1.7)
    assert np.all(dots.areas > 0.0)
    assert dots.dropped_area < 5e-3 * dots.area()

    for kind in Patch:
        n = int((dots.kinds == kind).sum())
        assert n / dots.area(kind) == pytest.approx(15.0, rel=3e-2)
        scaled = dots.areas[dots.kinds == kind] * 15.0
        lo, hi = np.quantile(scaled, [0.1, 0.9])
        assert 0.85 < lo < hi < 1.15
        lo, hi = np.quantile(scaled, [0.01, 0.99])
        assert 0.4 < lo < hi < 1.6

    assert set(np.unique(dots.atoms)) <= set(range(len(st)))


def test_exact_tangency_no_phantom():
    s = 3.0
    coords = np.array(
        [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s / 2, s * math.sqrt(3) / 2, 0.0]]
    )
    rs = 1.5 + 1.4
    rl = math.sqrt(rs * rs - (s / 2) ** 2)
    tangent = np.array([s / 2, -rl - rs, 0.0])
    coords = np.vstack([coords, tangent])
    dots = ses_dots(coords, [1.5] * 4, rp=1.4, density=400)
    concave = analytic_areas(coords, [1.5] * 4, 1.4)[2]
    assert concave < 3.0
    assert dots.area(Patch.CONCAVE) == pytest.approx(concave)
    n = int((dots.kinds == Patch.CONCAVE).sum())
    assert n / 400 == pytest.approx(concave, rel=5e-2)
    assert_valid_ses(dots, coords, [1.5] * 4, 1.4)


def test_inactive_atoms_only_occlude():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    radii = [1.5, 1.5, 1.5]
    active = np.array([True, False, False])
    dots = ses_dots(coords, radii, rp=1.4, density=50, active=active)
    assert not np.any(dots.atoms == 2)
    convex_owner = dots.atoms[dots.kinds == Patch.CONVEX]
    assert set(convex_owner) == {0}
    assert np.any(dots.kinds == Patch.TOROIDAL)

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nuri.scpilot.anal import (
    SasGeometry,
    SesGeometry,
    ses_area,
    two_sphere_ses_area,
)
from nuri.scpilot.io import load_structure
from nuri.scpilot.surface import Patch, ses_dots

R = 1.5
RP = 1.4
S = 3.0
TRIANGLE = np.array(
    [[0.0, 0.0, 0.0], [S, 0.0, 0.0], [S / 2, S * math.sqrt(3) / 2, 0.0]]
)


def analytic_areas(coords, radii, rp):
    ses = SesGeometry.build(SasGeometry.from_atoms(coords, radii, rp)[0])
    convex, saddle, concave = ses_area(ses)
    return np.array(
        [np.nansum(convex), float(saddle.sum()), float(concave.sum())]
    )


def assert_dots_match(coords, radii, rp, analytic, density=400, rtol=3e-2):
    """Weights reproduce the analytic areas exactly; dot counts follow
    the density on every kind large enough to be counted."""
    dots = ses_dots(coords, radii, rp=rp, density=density)
    weights = np.array([dots.area(k) for k in Patch])
    assert weights.sum() + dots.dropped_area == pytest.approx(
        analytic.sum(), rel=1e-9
    )
    counts = np.array([int((dots.kinds == k).sum()) for k in Patch])
    big = analytic * density >= 500
    np.testing.assert_allclose(counts[big] / density, analytic[big], rtol=rtol)


def triple_vertex():
    rs = R + RP
    cen = TRIANGLE.mean(axis=0)
    a = np.linalg.norm(TRIANGLE[0] - cen)
    return np.array([cen[0], cen[1], math.sqrt(rs * rs - a * a)])


def coincident_case(eps):
    n = np.array([0.3, 0.2, 1.0])
    n /= np.linalg.norm(n)
    return np.vstack([TRIANGLE, triple_vertex() + (R + RP + eps) * n])


def tangent_case(eps):
    rs = R + RP
    t = 0.5 * (TRIANGLE[0] + TRIANGLE[1])
    rl = math.sqrt(rs * rs - (S / 2) ** 2)
    radial = np.array([0.0, -1.0, 0.0])
    return np.vstack([TRIANGLE, t + rl * radial + (rs + eps) * radial])


def test_single_sphere():
    sas, _ = SasGeometry.from_atoms(np.zeros((1, 3)), [1.5], 1.7)
    ses = SesGeometry.build(sas)
    assert sas.sas_area[0] == pytest.approx(4 * math.pi * 3.2**2)
    assert ses.convex_area[0] == pytest.approx(4 * math.pi * 1.5**2)
    assert len(ses.concave) == 0


def test_two_spheres_stamm_table1():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    areas = analytic_areas(coords, [1.2, 1.2], 1.2)
    convex, torus, _ = two_sphere_ses_area(1.2, 1.2, 2.0, 1.2)
    assert convex + torus == pytest.approx(32.23514, abs=1e-4)
    assert areas[0] == pytest.approx(convex)
    assert areas[1] == pytest.approx(torus)
    assert areas[2] == 0.0


def test_two_spheres_cusp():
    coords = np.array([[0.0, 0.0, 0.0], [4.4, 0.0, 0.0]])
    areas = analytic_areas(coords, [1.2, 1.2], 1.2)
    convex, torus, _ = two_sphere_ses_area(1.2, 1.2, 4.4, 1.2)
    assert areas[0] == pytest.approx(convex)
    assert areas[1] == pytest.approx(torus)
    assert_dots_match(coords, [1.2, 1.2], 1.2, areas)


def test_three_spheres_vs_sampled():
    areas = analytic_areas(TRIANGLE, [R] * 3, RP)
    assert areas[2] > 0.0
    assert_dots_match(TRIANGLE, [R] * 3, RP, areas)


def test_random_cluster_vs_sampled():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 6.0, size=(8, 3))
    radii = rng.uniform(1.4, 1.9, size=8)
    assert_dots_match(coords, radii, 1.4, analytic_areas(coords, radii, 1.4))


@pytest.mark.parametrize("case", [coincident_case, tangent_case])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_degenerate_sweep_continuous(case, sign):
    """Areas converge as the offset shrinks; below the cluster radius the
    geometry is treated as exactly coincident."""
    radii = [R] * 4
    exact = analytic_areas(case(0.0), radii, RP)
    assert np.all(np.isfinite(exact))
    prev = None
    for eps in [1e-3, 1e-4, 1e-5]:
        areas = analytic_areas(case(sign * eps), radii, RP)
        assert np.all(np.isfinite(areas))
        if prev is not None:
            np.testing.assert_allclose(areas, prev, rtol=1e-3, atol=1e-1)
        prev = areas
    for eps in [1e-9, 1e-12]:
        np.testing.assert_allclose(
            analytic_areas(case(sign * eps), radii, RP), exact, rtol=1e-3
        )


@pytest.mark.parametrize("case", [coincident_case, tangent_case])
@pytest.mark.parametrize("eps", [1e-1, -1e-2, 1e-3, -1e-4, 1e-5, -1e-5])
def test_degenerate_sweep_vs_sampled(case, eps):
    coords = case(eps)
    assert_dots_match(coords, [R] * 4, RP, analytic_areas(coords, [R] * 4, RP))


def test_exact_coincidence_merges_probes():
    sas, _ = SasGeometry.from_atoms(coincident_case(0.0), [R] * 4, RP)
    assert len(sas.probes) == 5
    assert sorted(np.diff(sas.probe_offsets).tolist()) == [3, 3, 3, 3, 4]
    ses = SesGeometry.build(sas)
    assert len(ses.concave) == 5
    assert all(f.area > 0 for f in ses.concave)
    limit = analytic_areas(coincident_case(1e-5), [R] * 4, RP)
    np.testing.assert_allclose(
        analytic_areas(coincident_case(0.0), [R] * 4, RP), limit, rtol=1e-3
    )


def test_internally_tangent_caps_under_rotation():
    """Two caps of one sphere touching from inside: whether rounding calls
    the pair crossing (a pinch vertex) or nested (hidden), the area is that
    of the outer cap's complement; nothing may survive as a full circle."""
    r0, a_out, a_in = 3.0, 1.0, 0.4
    d_out, d_in = 2 * r0 * math.cos(a_out), 2 * r0 * math.cos(a_in)
    g = a_out - a_in
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, d_out],
            [d_in * math.sin(g), 0.0, d_in * math.cos(g)],
        ]
    )
    radii = np.full(3, r0 - RP)
    exact = 2 * math.pi * r0 * r0 * (1 + math.cos(a_out))
    rng = np.random.default_rng(1)
    for _ in range(100):
        coords = Rotation.random(random_state=rng).apply(base)
        coords += rng.uniform(-5.0, 5.0, size=3)
        sas, _ = SasGeometry.from_atoms(coords, radii, RP)
        assert sas.sas_area[0] == pytest.approx(exact, abs=1e-9)
        assert len(sas.arrangements[0].arcs) == 1
        assert len(sas.probes) <= 1


def test_exact_tangency_has_no_phantom():
    with_tangent = analytic_areas(tangent_case(0.0), [R] * 4, RP)
    without = analytic_areas(tangent_case(1e-1), [R] * 4, RP)
    assert with_tangent[2] == pytest.approx(without[2], abs=1e-9)


@pytest.mark.parametrize("last_residue", [12, 25])
def test_protein_fragment_vs_sampled(test_data, last_residue):
    st = load_structure(test_data / "1ar1.pdb").chains("H")
    st = st.subset(st.residue_seqs <= last_residue)
    analytic = analytic_areas(st.coords, st.vdw_radii, 1.7)
    assert_dots_match(
        st.coords, st.vdw_radii, 1.7, analytic, density=100, rtol=1e-2
    )


def test_inactive_atoms_skip_geometry():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    active = np.array([True, False, False])
    sas, order = SasGeometry.from_atoms(coords, [1.5] * 3, 1.4, active)
    assert (sas.n_active, sas.n_solve) == (1, 2)
    assert len(sas.arrangements) == 2
    np.testing.assert_array_equal(order, [0, 1, 2])
    assert len(sas.arcs) > 0

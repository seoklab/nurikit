#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri.desc import SurfacePatch, analytic_surface, shrake_rupley_sasa

RP = 1.4


@pytest.fixture()
def cluster():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 7.0, size=(12, 3))
    radii = rng.uniform(1.0, 1.9, size=12)
    return pts, radii


def convex_by_atom(geo, n):
    areas = np.zeros(n)
    areas[geo.order] = geo.area(SurfacePatch.Convex)
    return areas


def test_validation(cluster):
    pts, radii = cluster

    with pytest.raises(ValueError, match="does not match number of radii"):
        analytic_surface(pts[:-1], radii, RP)

    bad_radii = radii.copy()
    bad_radii[3] = 0.0
    with pytest.raises(ValueError, match="radii must be positive"):
        analytic_surface(pts, bad_radii, RP)

    with pytest.raises(ValueError, match="kind must be"):
        analytic_surface(pts, radii, RP, kind="foo")

    with pytest.raises(ValueError, match="active size"):
        analytic_surface(pts, radii, RP, active=np.ones(5, dtype=bool))

    with pytest.raises(ValueError, match=r"rp must be in \(0, inf\)"):
        analytic_surface(pts, radii, 0.0)

    bad_pts = pts.copy()
    bad_pts[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        analytic_surface(bad_pts, radii, RP)

    far_pts = pts.copy()
    far_pts[0, 2] = 2e4
    with pytest.raises(ValueError, match="rejected"):
        analytic_surface(far_pts, radii, RP)

    geo = analytic_surface(pts, radii, RP)
    with pytest.raises(ValueError, match="is not a valid SurfacePatch"):
        geo.area(SurfacePatch(5))
    with pytest.raises(TypeError):
        geo.total_area(5)


def test_view_semantics(cluster):
    pts, radii = cluster
    geo = analytic_surface(pts, radii, RP)

    assert geo.order.flags.writeable is False
    assert geo.area(SurfacePatch.Convex).flags.writeable is False

    convex = geo.area(SurfacePatch.Convex)
    total = geo.total_area(SurfacePatch.Convex)
    del geo
    assert convex.sum() == pytest.approx(total)


def test_single_sphere():
    pts = np.zeros((1, 3))
    radii = np.array([1.5])

    sas = analytic_surface(pts, radii, RP, kind="sas")
    assert sas.kind == "sas"
    assert sas.total_area() == pytest.approx(
        4 * np.pi * (1.5 + RP) ** 2, rel=1e-9
    )
    np.testing.assert_array_equal(sas.order, [0])
    assert len(sas.area(SurfacePatch.Toroidal)) == 0
    assert len(sas.area(SurfacePatch.Concave)) == 0
    assert sas.total_area(SurfacePatch.Toroidal) == 0
    assert sas.total_area(SurfacePatch.Concave) == 0

    ses = analytic_surface(pts, radii, RP, kind="ses")
    assert ses.kind == "ses"
    assert ses.total_area() == pytest.approx(4 * np.pi * 1.5**2, rel=1e-9)
    np.testing.assert_array_equal(ses.order, [0])


def test_sas_matches_shrake_rupley(cluster):
    pts, radii = cluster
    geo = analytic_surface(pts, radii, RP, kind="sas")

    ref = shrake_rupley_sasa(pts, radii, nprobe=5000, rprobe=RP)
    np.testing.assert_allclose(
        convex_by_atom(geo, len(radii)), ref, rtol=2e-2, atol=0.5
    )


def test_active_mask(cluster):
    pts, radii = cluster
    n = len(radii)
    active = np.arange(n) % 2 == 0

    full = analytic_surface(pts, radii, RP, kind="sas")
    masked = analytic_surface(pts, radii, RP, kind="sas", active=active)

    assert set(masked.order) == set(np.flatnonzero(active))
    assert len(masked.area(SurfacePatch.Convex)) == active.sum()

    np.testing.assert_allclose(
        convex_by_atom(masked, n)[active],
        convex_by_atom(full, n)[active],
        rtol=1e-12,
    )


def test_ses_totals(cluster):
    pts, radii = cluster
    geo = analytic_surface(pts, radii, RP)
    assert geo.kind == "ses"
    assert analytic_surface(pts, radii).total_area() == geo.total_area()

    by_patch = [geo.total_area(p) for p in SurfacePatch]
    assert geo.total_area() == pytest.approx(sum(by_patch), rel=1e-12)

    for p in SurfacePatch:
        areas = geo.area(p)
        assert (areas >= 0).all()
        assert areas.sum() == pytest.approx(geo.total_area(p), rel=1e-12)

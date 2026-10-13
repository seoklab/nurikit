#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri.desc import _sas_geometry, _ses_geometry
from nuri.scpilot.anal import SasGeometry, SesGeometry
from nuri.scpilot.io import load_structure


def random_cluster(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 15))
    coords = rng.uniform(0.0, 7.0, size=(n, 3))
    return coords, rng.uniform(1.0, 1.9, size=n)


def protein_fragment(test_data, last_residue):
    st = load_structure(test_data / "1ar1.pdb").chains("H")
    st = st.subset(st.residue_seqs <= last_residue)
    return st.coords, st.vdw_radii


def by_atom(order, areas):
    return {int(order[k]): float(areas[k]) for k in range(len(areas))}


def port_areas(coords, radii, rp, active):
    """Port areas keyed by original atom indices: per-atom SAS, per-atom
    convex SES, and the total saddle and concave SES areas."""
    sas = _sas_geometry(coords, radii, rp, active)
    ses = _ses_geometry(coords, radii, rp, active)
    assert sas["kind"] == "sas"
    assert ses["kind"] == "ses"
    assert sas["rp"] == rp
    assert ses["rp"] == rp
    assert len(sas["area_toroidal"]) == 0
    assert len(sas["area_concave"]) == 0
    assert sas["total_area"] == pytest.approx(sas["area_convex"].sum())
    assert ses["total_area"] == pytest.approx(
        ses["area_convex"].sum()
        + ses["area_toroidal"].sum()
        + ses["area_concave"].sum()
    )
    np.testing.assert_array_equal(sas["order"], ses["order"])
    assert len(set(ses["order"].tolist())) == ses["n_atoms"]
    assert len(ses["area_convex"]) <= ses["n_atoms"]
    return (
        by_atom(sas["order"], sas["area_convex"]),
        by_atom(ses["order"], ses["area_convex"]),
        float(ses["area_toroidal"].sum()),
        float(ses["area_concave"].sum()),
    )


def pilot_areas(coords, radii, rp, active):
    sas, order = SasGeometry.from_atoms(coords, radii, rp, active)
    ses = SesGeometry.build(sas)
    return (
        by_atom(order, sas.sas_area[: sas.n_active]),
        by_atom(order, ses.convex_area[: sas.n_active]),
        float(sum(ses.saddle_area[: sas.n_active_arcs])),
        float(sum(f.area for f in ses.concave)),
    )


def assert_dicts_close(ours, ref, atol, what):
    assert ours.keys() == ref.keys(), what
    keys = sorted(ours)
    np.testing.assert_allclose(
        [ours[k] for k in keys],
        [ref[k] for k in keys],
        rtol=0.0,
        atol=atol,
        err_msg=what,
    )
    return len(keys)


def assert_matches_pilot(coords, radii, rp, active=None):
    sas, convex, saddle, concave = port_areas(coords, radii, rp, active)
    p_sas, p_convex, p_saddle, p_concave = pilot_areas(
        coords, radii, rp, active
    )

    n_sas = assert_dicts_close(sas, p_sas, 1e-9, "sas")
    n_convex = assert_dicts_close(convex, p_convex, 1e-10, "convex")
    assert saddle == pytest.approx(p_saddle, rel=1e-8)
    assert concave == pytest.approx(p_concave, rel=1e-8)

    assert n_sas == n_convex > 0
    assert saddle > 0
    assert concave > 0
    return n_convex


@pytest.mark.parametrize("seed", range(5))
def test_random_cluster_matches_pilot(seed):
    coords, radii = random_cluster(seed)
    assert_matches_pilot(coords, radii, 1.4)


@pytest.mark.parametrize("last_residue", [20, 40])
def test_protein_fragment_matches_pilot(test_data, last_residue):
    coords, radii = protein_fragment(test_data, last_residue)
    assert_matches_pilot(coords, radii, 1.7)


def test_masked_cluster_matches_pilot():
    coords, radii = random_cluster(7)
    active = np.zeros(len(radii), dtype=bool)
    active[: len(radii) // 2] = True
    n_convex = assert_matches_pilot(coords, radii, 1.4, active)
    assert n_convex == int(active.sum())

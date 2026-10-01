#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from collections import Counter

import numpy as np
import pytest

from nuri.desc import _ses_geometry
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


def port_areas(coords, radii, rp, active):
    """Port areas keyed by original atom indices: per-atom convex, per
    circle saddle, per 3-atom-probe concave (as a list, duplicates kept),
    and the total active concave area."""
    d = _ses_geometry(coords, radii, rp, active)
    order, ses = d["order"], d["ses"]
    convex = {
        int(order[k]): float(ses["convex_area"][k])
        for k in range(d["n_active"])
    }
    arcs, circles = d["arcs"], d["circles"]
    saddle = Counter()
    for a in range(arcs["n_active"]):
        q = arcs["circ"][a]
        key = tuple(
            sorted((int(order[circles["i"][q]]), int(order[circles["j"][q]])))
        )
        saddle[key] += float(ses["saddle_area"][a])
    adj, off = d["probes"]["atoms"]["adj"], d["probes"]["atoms"]["off"]
    n_probes = d["probes"]["n_active"]
    faces = []
    for p in range(n_probes):
        atoms = adj[off[p] : off[p + 1]]
        if len(atoms) == 3:
            key = tuple(sorted(int(order[i]) for i in atoms))
            faces.append((key, float(ses["face_area"][p])))
    total = float(ses["face_area"][:n_probes].sum())
    return convex, dict(saddle), faces, total


def pilot_areas(coords, radii, rp, active):
    sas, order = SasGeometry.from_atoms(coords, radii, rp, active)
    ses = SesGeometry.build(sas)
    convex = {
        int(order[k]): float(ses.convex_area[k]) for k in range(sas.n_active)
    }
    saddle = Counter()
    for a in range(sas.n_active_arcs):
        c = sas.circles[sas.arcs[a].circle]
        key = tuple(sorted((int(order[c.i]), int(order[c.j]))))
        saddle[key] += float(ses.saddle_area[a])
    faces = []
    for f in ses.concave:
        if len(f.atoms) == 3:
            key = tuple(sorted(int(order[i]) for i in f.atoms))
            faces.append((key, float(f.area)))
    total = float(sum(f.area for f in ses.concave))
    return convex, dict(saddle), faces, total


def unique_faces(faces):
    counts = Counter(key for key, _ in faces)
    return {key: area for key, area in faces if counts[key] == 1}


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
    convex, saddle, faces, total = port_areas(coords, radii, rp, active)
    p_convex, p_saddle, p_faces, p_total = pilot_areas(
        coords, radii, rp, active
    )

    n_convex = assert_dicts_close(convex, p_convex, 1e-10, "convex")
    n_saddle = assert_dicts_close(saddle, p_saddle, 1e-9, "saddle")
    assert total == pytest.approx(p_total, rel=1e-8)

    ours, ref = unique_faces(faces), unique_faces(p_faces)
    common = {k: ours[k] for k in ours.keys() & ref.keys()}
    n_faces = assert_dicts_close(
        common, {k: ref[k] for k in common}, 1e-9, "concave"
    )

    assert n_convex > 0
    assert n_saddle > 0
    assert n_faces > 0
    return n_convex, n_saddle, n_faces


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
    n_convex, _, _ = assert_matches_pilot(coords, radii, 1.4, active)
    assert n_convex == int(active.sum())

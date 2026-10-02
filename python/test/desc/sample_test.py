#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nuri.desc import _ses_dots
from nuri.scpilot.io import load_structure
from nuri.scpilot.surface import Patch, ses_dots


def random_cluster(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 15))
    coords = rng.uniform(0.0, 7.0, size=(n, 3))
    return coords, rng.uniform(1.0, 1.9, size=n)


def protein_fragment(test_data, last_residue):
    st = load_structure(test_data / "1ar1.pdb").chains("H")
    st = st.subset(st.residue_seqs <= last_residue)
    return st.coords, st.vdw_radii


def port_blocks(d):
    off = d["kind_off"]
    assert len(off) == 4
    assert off[0] == 0
    assert off[-1] == len(d["pts"])
    return [slice(off[k], off[k + 1]) for k in range(3)]


def assert_matches_pilot(coords, radii, rp, density, active=None):
    d = _ses_dots(coords, radii, rp, density, active)
    pilot = ses_dots(coords, radii, rp, density, active)

    assert d["rp"] == rp
    assert d["dropped_area"] == pytest.approx(pilot.dropped_area, abs=1e-9)
    assert d["pts"].shape == (len(d["area"]), 3)
    assert d["nrm"].shape == d["pts"].shape

    for kind, block in zip(Patch, port_blocks(d)):
        area = float(d["area"][block].sum())
        assert area == pytest.approx(pilot.area(kind), abs=1e-9), kind
        n_port, n_pilot = (
            block.stop - block.start,
            int(np.count_nonzero(pilot.kinds == kind)),
        )
        tol = 2 if n_pilot < 100 else 0.02 * n_pilot
        assert abs(n_port - n_pilot) <= tol, (kind, n_port, n_pilot)
    return d


@pytest.mark.parametrize("seed", range(3))
def test_random_cluster_matches_pilot(seed):
    coords, radii = random_cluster(seed)
    assert_matches_pilot(coords, radii, 1.4, 15.0)


def test_masked_cluster_matches_pilot():
    coords, radii = random_cluster(7)
    active = np.zeros(len(radii), dtype=bool)
    active[: len(radii) // 2] = True
    d = assert_matches_pilot(coords, radii, 1.4, 15.0, active)
    convex = d["atom"][: d["kind_off"][1]]
    assert len(convex) > 0
    assert np.all(active[convex])


def test_protein_fragment_matches_pilot(test_data):
    coords, radii = protein_fragment(test_data, 40)
    d = assert_matches_pilot(coords, radii, 1.7, 15.0)

    probes = d["pts"] + d["rp"] * d["nrm"]
    dist, nearest = cKDTree(coords).query(probes)
    assert np.all(dist - (radii[nearest] + d["rp"]) >= -1e-9)

    sas = np.asarray(radii) + d["rp"]
    near = cKDTree(probes).sparse_distance_matrix(
        cKDTree(coords), sas.max(), output_type="coo_matrix"
    )
    assert not np.any(near.data < sas[near.col] - 1e-9)

    for block in port_blocks(d):
        area = float(d["area"][block].sum())
        assert area > 0.0
        assert (block.stop - block.start) / area == pytest.approx(
            15.0, rel=0.03
        )

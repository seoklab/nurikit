#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from nuri import algo
from nuri.core import Molecule


@pytest.fixture()
def bcp():
    mol = Molecule()
    with mol.mutator() as mut:
        for _ in range(4):
            mut.add_atom(6)

        mut.add_bond(0, 1)
        mut.add_bond(0, 2)
        mut.add_bond(1, 2)
        mut.add_bond(1, 3)
        mut.add_bond(2, 3)

    mol.sanitize()
    return mol


@pytest.fixture()
def cubane():
    mol = Molecule()
    with mol.mutator() as mut:
        for _ in range(8):
            mut.add_atom(6)

        mut.add_bond(0, 1)
        mut.add_bond(0, 3)
        mut.add_bond(0, 5)
        mut.add_bond(1, 2)
        mut.add_bond(1, 6)
        mut.add_bond(2, 7)
        mut.add_bond(2, 3)
        mut.add_bond(3, 4)
        mut.add_bond(4, 5)
        mut.add_bond(4, 7)
        mut.add_bond(5, 6)
        mut.add_bond(6, 7)

    mol.sanitize()
    return mol


def test_find_all_rings(bcp: Molecule):
    rings = algo.find_all_rings(bcp)
    assert len(rings) == 3

    for ring in rings:
        assert len(ring) == 3 or len(ring) == 4

    ring = rings[0]

    rings = algo.find_all_rings(ring)
    assert len(rings) == 1

    bcp.subs.append(ring)
    rings = algo.find_all_rings(bcp.subs[0])
    assert len(rings) == 1


def test_find_sssr(cubane: Molecule):
    rings = algo.find_sssr(cubane)
    assert len(rings) == 5

    for ring in rings:
        assert len(ring) == 4

    ring = rings[0]

    rings = algo.find_sssr(ring)
    assert len(rings) == 1

    cubane.subs.append(ring)
    rings = algo.find_sssr(cubane.subs[0])
    assert len(rings) == 1


def test_find_relevant(cubane: Molecule):
    rings = algo.find_relevant_rings(cubane)
    assert len(rings) == 6

    for ring in rings:
        assert len(ring) == 4

    ring = rings[0]

    rings = algo.find_relevant_rings(ring)
    assert len(rings) == 1

    cubane.subs.append(ring)
    rings = algo.find_relevant_rings(cubane.subs[0])
    assert len(rings) == 1


def test_ring_size(cubane: Molecule):
    rings = algo.find_all_rings(cubane, 6)
    for ring in rings:
        assert len(ring) <= 6

    with pytest.raises(ValueError, match="must be positive"):
        algo.find_all_rings(cubane, -1)

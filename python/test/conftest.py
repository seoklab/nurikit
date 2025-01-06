#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
import pytest

from nuri.core import BondOrder, Molecule


@pytest.fixture()
def mol():
    mol = Molecule()

    #         H (4)  C (0, implicit 1H) H (8) -- N- (9) -- H (5)
    #             \   |
    #               C (2)              Na+ (10) (<-intentionally unconnected)
    #             /       \
    #  H (6) --  C (3) == C (1) -- H(7)

    with mol.mutator() as mut:
        c0 = mut.add_atom(6).update(implicit_hydrogens=1)
        c1 = mut.add_atom(6)
        c2 = mut.add_atom(6)
        c3 = mut.add_atom(6)
        # 4 - 8
        hs = {i: mut.add_atom(1) for i in range(4, 9)}
        n9 = mut.add_atom(7).update(formal_charge=-1)
        mut.add_atom(11).update(formal_charge=+1)

        mut.add_bond(c0, n9)
        mut.add_bond(c3, c2)
        mut.add_bond(c3, c1, BondOrder.Double)
        mut.add_bond(c1, c2)
        mut.add_bond(c2, c0)
        mut.add_bond(c2, hs[4])
        mut.add_bond(n9, hs[5])
        mut.add_bond(c3, hs[6])
        mut.add_bond(c1, hs[7])
        mut.add_bond(c0, hs[8])

    return mol


@pytest.fixture()
def mol3d(mol: Molecule):
    mol.add_conf(
        [
            [-0.0127, 1.0858, 0.0080],
            [-0.1528, 1.3191, -2.5866],
            [-0.7527, 1.5927, -1.2315],
            [-1.1913, 0.5880, -2.2656],
            [-1.3450, 2.4981, -1.0997],
            [-0.2304, 1.2378, 2.0496],
            [-1.8821, -0.1932, -2.5467],
            [0.6062, 1.5595, -3.3164],
            [0.0021, -0.0041, 0.0020],
            [-0.7003, 1.5596, 1.2166],
            [6.0099, 7.4981, 7.0496],
        ]
    )
    mol.add_conf(
        [
            [-2.8265, 2.3341, -0.0040],
            [-2.4325, 1.6004, 1.0071],
            [-1.3392, 2.1190, 0.1089],
            [-0.4723, 3.2756, 0.6106],
            [-1.2377, 4.5275, 0.5399],
            [-0.6865, 5.3073, 0.8661],
            [-3.6475, 2.7827, -0.5437],
            [-2.7033, 1.0234, 1.8791],
            [-0.8591, 1.4229, -0.5788],
            [-0.1776, 3.0877, 1.6430],
            [10.1054, 2.3061, 4.3717],
        ]
    )
    return mol


@pytest.fixture()
def molsub(mol: Molecule):
    sub = mol.subs.add([2, 3, 7])
    assert sub.num_atoms() == 3
    assert sub.num_bonds() == 1
    return mol


@pytest.fixture()
def mol3dsub(mol3d: Molecule):
    sub = mol3d.subs.add([2, 3, 7, 10])
    assert sub.num_atoms() == 4
    assert sub.num_bonds() == 1
    return mol3d


@pytest.fixture()
def test_data():
    return Path(__file__).parents[2] / "test/test_data"

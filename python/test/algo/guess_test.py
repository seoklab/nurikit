#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from nuri import algo
from nuri.core import Hyb, Molecule


@pytest.fixture()
def arginine():
    mol = Molecule()

    with mol.mutator() as mut:
        mut.add_atom(7)
        mut.add_atom(6)
        mut.add_atom(6)
        mut.add_atom(8)
        mut.add_atom(6)
        mut.add_atom(6)
        mut.add_atom(6)
        mut.add_atom(7)
        mut.add_atom(6)
        mut.add_atom(7)
        mut.add_atom(7)
        mut.add_atom(8)

    mol.add_conf(
        [
            [69.812, 14.685, 89.810],
            [70.052, 14.573, 91.280],
            [71.542, 14.389, 91.604],
            [72.354, 14.342, 90.659],
            [69.227, 13.419, 91.854],
            [67.722, 13.607, 91.686],
            [66.952, 12.344, 92.045],
            [67.307, 11.224, 91.178],
            [66.932, 9.966, 91.380],
            [66.176, 9.651, 92.421],
            [67.344, 9.015, 90.554],
            [71.901, 14.320, 92.798],
        ]
    )

    return mol


@pytest.fixture()
def arginine_bonds(arginine: Molecule):
    with arginine.mutator() as mut:
        mut.add_bond(0, 1)
        mut.add_bond(1, 2)
        mut.add_bond(1, 4)
        mut.add_bond(2, 3)
        mut.add_bond(2, 11)
        mut.add_bond(4, 5)
        mut.add_bond(5, 6)
        mut.add_bond(6, 7)
        mut.add_bond(7, 8)
        mut.add_bond(8, 9)
        mut.add_bond(8, 10)

    return arginine


def test_guess_all(arginine: Molecule):
    with arginine.mutator() as mut:
        algo.guess_everything(mut)

    assert arginine.num_bonds() == 11
    assert arginine[8].hyb == Hyb.SP2


def test_guess_conn(arginine: Molecule):
    with arginine.mutator() as mut:
        algo.guess_connectivity(mut)

    assert arginine.num_bonds() == 11


def test_guess_types(arginine_bonds: Molecule):
    algo.guess_all_types(arginine_bonds)
    assert arginine_bonds[8].hyb == Hyb.SP2


def test_guess_error(arginine: Molecule):
    with arginine.mutator() as mut:
        with pytest.raises(IndexError):
            algo.guess_everything(mut, 100)

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import collections.abc as abc

import pytest

from nuri import core
from nuri.core import (
    Molecule,
    SubAtom,
    SubBond,
    SubNeighbor,
    Substructure,
)


def _owned(mol: Molecule):
    return mol.sub([0, 1, 2, 3])


def _proxy(mol: Molecule):
    return mol.subs.add([0, 1, 2, 3])


@pytest.mark.parametrize("factory", [_owned, _proxy], ids=["owned", "proxy"])
def test_substructure_masquerades_as_owned(mol: Molecule, factory):
    sub = factory(mol)

    assert isinstance(sub, Substructure)
    assert isinstance(sub.atom(0), SubAtom)

    bond = sub.bonds()[0]
    assert isinstance(bond, SubBond)
    assert isinstance(sub.neighbor(bond.src, bond.dst), SubNeighbor)


def test_proxy_types_are_private_virtual_subclasses():
    for name in (
        "ProxySubstructure",
        "ProxySubAtom",
        "ProxySubBond",
        "ProxySubNeighbor",
    ):
        assert not hasattr(core, name)
    assert hasattr(core, "_ProxySubstructure")

    assert issubclass(core._ProxySubstructure, Substructure)
    assert issubclass(core._ProxySubAtom, SubAtom)
    assert issubclass(core._ProxySubBond, SubBond)
    assert issubclass(core._ProxySubNeighbor, SubNeighbor)


def test_virtual_subclass_negatives(mol: Molecule):
    owned = _owned(mol)

    assert not isinstance(object(), Substructure)
    assert not isinstance(owned, SubAtom)
    assert not issubclass(int, Substructure)
    assert not issubclass(SubAtom, Substructure)


def test_container_registrations_intact(mol: Molecule):
    assert isinstance(mol, abc.Sequence)
    assert isinstance(mol[0], abc.Sequence)
    assert isinstance(mol.bonds(), abc.Sequence)
    assert isinstance(mol.subs, abc.Sequence)

    sub = _proxy(mol)
    assert isinstance(sub, abc.Sequence)
    assert isinstance(sub.bonds(), abc.Sequence)
    assert isinstance(sub.props, abc.MutableMapping)


@pytest.mark.parametrize("factory", [_owned, _proxy], ids=["owned", "proxy"])
def test_copy_returns_independent_owned(mol: Molecule, factory):
    sub = factory(mol)
    dup = sub.copy()

    assert isinstance(dup, Substructure)
    assert dup.num_atoms() == sub.num_atoms()
    assert dup.num_bonds() == sub.num_bonds()

    mol.subs.clear()
    assert dup.num_atoms() > 0

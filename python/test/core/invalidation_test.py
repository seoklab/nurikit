#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import gc

import pytest

from nuri.core import Molecule, Substructure
from nuri.core._core import ProxySubstructure


def _assert_invalidated(func, *args, **kwargs):
    with pytest.raises(RuntimeError):
        func(*args, **kwargs)


def _remove_atom(mol: Molecule):
    with mol.mutator() as mut:
        mut.mark_atom_erase(2)


def _remove_bond(mol: Molecule):
    with mol.mutator() as mut:
        mut.mark_bond_erase(2)


@pytest.mark.parametrize(
    "modify",
    [
        _remove_atom,
        _remove_bond,
        Molecule.clear,
        Molecule.clear_atoms,
        Molecule.clear_bonds,
        Molecule.conceal_hydrogens,
    ],
    ids=[
        "remove_atom",
        "remove_bond",
        "clear",
        "clear_atoms",
        "clear_bonds",
        "conceal_hydrogens",
    ],
)
def test_invalidation_all(molsub: Molecule, modify):
    p = molsub.props
    atom = molsub.atom(0)
    a_p = atom.props
    bond = molsub.bond(0)
    b_p = bond.props
    neig = atom[0]

    msub = molsub.subs[0]
    m_p = msub.props
    ma = msub.atom(0)
    ma_p = ma.props
    mb = msub.bond(0)
    mb_p = mb.props
    mn = msub.neighbor(mb.src, mb.dst)

    sub = molsub.sub(bonds=[1])
    s_p = sub.props
    sa = sub.atom(0)
    sa_p = sa.props
    sb = sub.bond(0)
    sb_p = sb.props
    sn = sub.neighbor(sb.src, sb.dst)

    modify(molsub)

    _assert_invalidated(lambda: atom.atomic_number)
    _assert_invalidated(lambda: iter(a_p))
    _assert_invalidated(lambda: bond.order)
    _assert_invalidated(lambda: iter(b_p))
    _assert_invalidated(lambda: neig.src)

    _assert_invalidated(lambda: len(msub))
    _assert_invalidated(lambda: iter(m_p))
    _assert_invalidated(lambda: ma.atomic_number)
    _assert_invalidated(lambda: iter(ma_p))
    _assert_invalidated(lambda: mb.order)
    _assert_invalidated(lambda: iter(mb_p))
    _assert_invalidated(lambda: mn.src)

    _assert_invalidated(lambda: len(sub))
    _assert_invalidated(lambda: sa.atomic_number)
    _assert_invalidated(lambda: iter(sa_p))
    _assert_invalidated(lambda: sb.order)
    _assert_invalidated(lambda: iter(sb_p))
    _assert_invalidated(lambda: sn.src)

    for _ in p:
        pass

    for _ in s_p:
        pass


def _remove_atom_sub(sub: Substructure):
    sub.erase_atom(sub.atom(0))


def _remove_bond_sub(sub: Substructure):
    sub.erase_bond(sub.bond(0))


@pytest.mark.parametrize(
    "modify_msub",
    [
        _remove_atom_sub,
        _remove_bond_sub,
        ProxySubstructure.clear,
        ProxySubstructure.clear_atoms,
        ProxySubstructure.clear_bonds,
    ],
    ids=[
        "remove_atom",
        "remove_bond",
        "clear",
        "clear_atoms",
        "clear_bonds",
    ],
)
def test_invalidation_mol_sub(molsub: Molecule, modify_msub):
    p = molsub.props
    atom = molsub.atom(0)
    a_p = atom.props
    bond = molsub.bond(0)
    b_p = bond.props
    neig = atom[0]

    msub = molsub.subs[0]
    m_p = msub.props
    ma = msub.atom(0)
    ma_p = ma.props
    mb = msub.bond(0)
    mb_p = mb.props
    mn = msub.neighbor(mb.src, mb.dst)

    msub2 = molsub.subs.add()

    sub = molsub.sub(bonds=[1])
    s_p = sub.props
    sa = sub.atom(0)
    sa_p = sa.props
    sb = sub.bond(0)
    sb_p = sb.props
    sn = sub.neighbor(sb.src, sb.dst)

    modify_msub(msub)

    assert atom.atomic_number is not None
    assert iter(a_p) is not None
    assert bond.order is not None
    assert iter(b_p) is not None
    assert neig.src is not None

    assert len(msub) is not None
    _assert_invalidated(lambda: iter(m_p))
    _assert_invalidated(lambda: ma.atomic_number)
    _assert_invalidated(lambda: iter(ma_p))
    _assert_invalidated(lambda: mb.order)
    _assert_invalidated(lambda: iter(mb_p))
    _assert_invalidated(lambda: mn.src)
    _assert_invalidated(lambda: len(msub2))

    assert len(molsub.subs[1]) is not None

    assert len(sub) is not None
    assert sa.atomic_number is not None
    assert iter(sa_p) is not None
    assert sb.order is not None
    assert iter(sb_p) is not None
    assert sn.src is not None

    for _ in p:
        pass

    for _ in s_p:
        pass


@pytest.mark.parametrize(
    "modify_sub",
    [
        _remove_atom_sub,
        _remove_bond_sub,
        Substructure.clear,
        Substructure.clear_atoms,
        Substructure.clear_bonds,
    ],
    ids=[
        "remove_atom",
        "remove_bond",
        "clear",
        "clear_atoms",
        "clear_bonds",
    ],
)
def test_invalidation_sub(molsub: Molecule, modify_sub):
    p = molsub.props
    atom = molsub.atom(0)
    a_p = atom.props
    bond = molsub.bond(0)
    b_p = bond.props
    neig = atom[0]

    msub = molsub.subs[0]
    m_p = msub.props
    ma = msub.atom(0)
    ma_p = ma.props
    mb = msub.bond(0)
    mb_p = mb.props
    mn = msub.neighbor(mb.src, mb.dst)

    sub = molsub.sub(bonds=[1])
    s_p = sub.props
    sa = sub.atom(0)
    sa_p = sa.props
    sb = sub.bond(0)
    sb_p = sb.props
    sn = sub.neighbor(sb.src, sb.dst)

    modify_sub(sub)

    assert atom.atomic_number is not None
    assert iter(a_p) is not None
    assert bond.order is not None
    assert iter(b_p) is not None
    assert neig.src is not None

    assert len(msub) is not None
    assert iter(m_p) is not None
    assert ma.atomic_number is not None
    assert iter(ma_p) is not None
    assert mb.order is not None
    assert iter(mb_p) is not None
    assert mn.src is not None

    assert lambda: len(sub) is not None
    _assert_invalidated(lambda: sa.atomic_number)
    _assert_invalidated(lambda: iter(sa_p))
    _assert_invalidated(lambda: sb.order)
    _assert_invalidated(lambda: iter(sb_p))
    _assert_invalidated(lambda: sn.src)

    for _ in p:
        pass

    for _ in s_p:
        pass


def _run_iterator(it):
    for _ in it:
        pass


def test_invalidation_iterator(mol3dsub: Molecule):
    mol3dsub.props["test"] = "1"
    mol3dsub.atom(0).props["test"] = "1"
    mol3dsub.bond(0).props["test"] = "1"

    mol3dsub.subs[0].props["test"] = "1"
    mol3dsub.subs[0].atom(0).props["test"] = "1"
    mol3dsub.subs[0].bond(0).props["test"] = "1"

    sub = mol3dsub.sub(bonds=mol3dsub.subs[0].parent_bonds())
    sub.props["test"] = "1"

    # The subobjects are intentionally *not* separately kept as variables to
    # test if the intermediates kept alive while iterators are alive.
    iterators = [
        mol3dsub.props.keys(),
        mol3dsub.props.values(),
        mol3dsub.props.items(),
        iter(mol3dsub),
        iter(mol3dsub.bonds()),
        iter(mol3dsub.atom(0)),
        iter(mol3dsub.atom(0).props.keys()),
        iter(mol3dsub.atom(0).props.values()),
        iter(mol3dsub.atom(0).props.items()),
        iter(mol3dsub.bond(0).props.keys()),
        iter(mol3dsub.bond(0).props.values()),
        iter(mol3dsub.bond(0).props.items()),
        mol3dsub.conformers(),
        iter(mol3dsub.subs),
        iter(mol3dsub.subs[0]),
        iter(mol3dsub.subs[0].bonds()),
        iter(mol3dsub.subs[0].atom(0)),
        iter(mol3dsub.subs[0].props.keys()),
        iter(mol3dsub.subs[0].props.values()),
        iter(mol3dsub.subs[0].props.items()),
        iter(mol3dsub.subs[0].atom(0).props.keys()),
        iter(mol3dsub.subs[0].atom(0).props.values()),
        iter(mol3dsub.subs[0].atom(0).props.items()),
        iter(mol3dsub.subs[0].bond(0).props.keys()),
        iter(mol3dsub.subs[0].bond(0).props.values()),
        iter(mol3dsub.subs[0].bond(0).props.items()),
        mol3dsub.subs[0].conformers(),
        iter(sub),
        iter(sub.bonds()),
        iter(sub.atom(0)),
        iter(sub.atom(0).props.keys()),
        iter(sub.atom(0).props.values()),
        iter(sub.atom(0).props.items()),
        iter(sub.bond(0).props.keys()),
        iter(sub.bond(0).props.values()),
        iter(sub.bond(0).props.items()),
        sub.conformers(),
    ]

    mol3dsub.clear()

    del mol3dsub
    del sub
    gc.collect()

    for iterator in iterators:
        with pytest.raises(RuntimeError):
            _run_iterator(iterator)

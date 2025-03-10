#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri.core import BondOrder, Molecule


def test_empty():
    mol = Molecule()
    sub = mol.sub()

    assert not sub
    assert len(sub) == 0
    assert sub.num_atoms() == 0
    assert sub.num_bonds() == 0
    assert not sub.bonds()
    assert sub.num_confs() == 0

    with pytest.raises(IndexError):
        sub[0]

    with pytest.raises(IndexError):
        sub.atom(1)

    with pytest.raises(IndexError):
        sub.bond(0)

    with pytest.raises(IndexError):
        sub.bonds()[1]

    assert sub.molecule is mol


def test_add_atom(mol: Molecule):
    sub = mol.sub()
    sub.add_atoms([mol[i] for i in range(3)])

    assert sub.num_atoms() == 3
    assert sub.parent_atoms() == [0, 1, 2]
    assert sub.num_bonds() == 2
    assert sub.parent_bonds() == [3, 4]

    sa = sub.atom(0)
    sb = sub.bond(0)
    assert sa.atomic_number == 6
    assert sb.order == BondOrder.Single

    sub.add_atoms([2, mol[5]], False)
    assert sub.num_atoms() == 4
    assert sub.parent_atoms() == [0, 1, 2, 5]
    assert sub.num_bonds() == 2
    assert sub.parent_bonds() == [3, 4]

    sa = sub.atom(0)
    with pytest.raises(TypeError):
        sub.add_atoms(["test"])  # pyright: ignore[reportArgumentType]

    assert sa.atomic_number == 6
    assert sub.num_atoms() == 4


def test_add_bond(mol: Molecule):
    sub = mol.sub()
    sub.add_bonds([mol.bond(2), mol.bond(3)])

    assert sub.num_atoms() == 3
    assert sub.parent_atoms() == [1, 2, 3]
    assert sub.num_bonds() == 2
    assert sub.parent_bonds() == [2, 3]

    sa = sub.atom(0)
    sb = sub.bond(0)
    assert sa.atomic_number == 6
    assert sb.order == BondOrder.Double

    sub.add_bonds([1, 2, mol.bond(8)])

    assert sub.num_atoms() == 4
    assert sub.parent_atoms() == [1, 2, 3, 7]
    assert sub.num_bonds() == 4
    assert sub.parent_bonds() == [1, 2, 3, 8]

    sb = sub.bond(0)
    with pytest.raises(TypeError):
        sub.add_bonds(["test"])  # pyright: ignore[reportArgumentType]

    assert sb.order == BondOrder.Single
    assert sub.num_atoms() == 4
    assert sub.num_bonds() == 4


def test_atom(mol3dsub: Molecule):
    sub = mol3dsub.subs[0]

    atom = sub.atom(0)
    assert atom.id == sub[mol3dsub[2]].id

    with pytest.raises(KeyError):
        sub[mol3dsub[0]]

    atom.as_parent().formal_charge = +1
    assert mol3dsub[2].formal_charge == +1

    i = -1
    for i, atom in enumerate(sub, 1):  # noqa: B007
        assert atom.atomic_number in (1, 6, 11)
    assert i == 4

    assert np.allclose(atom.get_pos(), atom.as_parent().get_pos())

    newpos = atom.get_pos() + 1
    atom.set_pos(newpos)
    assert np.allclose(newpos, atom.get_pos())
    assert np.allclose(atom.get_pos(), atom.as_parent().get_pos())


def test_bond(molsub: Molecule):
    sub = molsub.subs[0]
    assert molsub.bond(1).order == BondOrder.Single

    bond = sub.bond(0)
    assert bond.id == sub.bond(molsub.bond(1)).id

    with pytest.raises(KeyError):
        sub.bond(molsub.bond(0))

    bond.as_parent().order = BondOrder.Double
    assert molsub.bond(1).order == BondOrder.Double

    bonds = molsub.subs[0].bonds()
    assert molsub.bond(1).order == BondOrder.Double

    bond = bonds[0]
    assert bond.id == bonds[molsub.bond(1)].id

    with pytest.raises(KeyError):
        bonds[molsub.bond(0)]

    bond.as_parent().order = BondOrder.Single
    assert molsub.bond(1).order == BondOrder.Single

    i = -1
    for i, bond in enumerate(bonds, 1):  # noqa: B007
        assert bond.order == BondOrder.Single
    assert i == 1


def test_contains(molsub: Molecule):
    sub = molsub.subs[0]

    assert 0 in sub
    assert sub[0] in sub
    assert sub[0].as_parent() in sub
    assert molsub[0] not in sub

    mol = molsub.copy()

    assert molsub[2] in sub
    assert mol[0] not in sub
    assert mol[2] not in sub


def test_contains_bond(molsub: Molecule):
    bonds = molsub.subs[0].bonds()

    assert 0 in bonds
    assert bonds[0] in bonds
    assert bonds[0].as_parent() in bonds
    assert molsub.bond(0) not in bonds

    mol = molsub.copy()

    assert molsub.bond(1) in bonds
    assert mol.bond(0) not in bonds
    assert mol.bond(1) not in bonds


def test_neighbors(molsub: Molecule):
    sub = molsub.subs[0]
    nei = sub.neighbor(sub[0], sub[1])

    assert nei.src.id == 0
    assert nei.dst.id == 1

    assert nei.as_parent().src.id == 2
    assert nei.as_parent().dst.id == 3

    sub.add_atoms([1])
    assert sub.parent_atoms() == [1, 2, 3, 7]

    i = -1
    for i, nei in enumerate(sub[1], 1):  # noqa: B007
        assert nei.dst.id == 2
    assert i == 1

    with pytest.raises(ValueError, match="not a neighbor"):
        sub.neighbor(sub[0], sub[1])

    assert (
        molsub.bond(sub[0].as_parent(), sub[1].as_parent()).order
        == BondOrder.Single
    )

    sub.add_bonds([3])
    assert sub.neighbor(sub[0], sub[1]).bond.order == BondOrder.Single


def test_find_bond(molsub: Molecule):
    sub = molsub.subs[0]

    assert not sub.has_bond(sub[0], sub[2])
    with pytest.raises(ValueError, match="no such bond"):
        sub.bond(sub[0], sub[2])

    bond_01 = sub.bond(sub[0], sub[1])
    bond_10 = sub.bond(sub[1], sub[0])
    assert sub.has_bond(sub[0], sub[1])

    bond_01_atoms = sub.bond(sub[0].as_parent(), sub[1].as_parent())
    bond_10_atoms = sub.bond(sub[1].as_parent(), sub[0].as_parent())
    assert sub.has_bond(sub[0].as_parent(), sub[1].as_parent())

    assert bond_01.src.id == bond_10.src.id
    assert bond_01.dst.id == bond_10.dst.id
    assert bond_01.id == bond_10.id

    assert bond_01.src.id == bond_01_atoms.src.id
    assert bond_01.dst.id == bond_01_atoms.dst.id
    assert bond_01.id == bond_01_atoms.id

    assert bond_01.src.id == bond_10_atoms.src.id
    assert bond_01.dst.id == bond_10_atoms.dst.id
    assert bond_01.id == bond_10_atoms.id


def test_find_neighbor(molsub: Molecule):
    sub = molsub.subs[0]

    with pytest.raises(ValueError, match="not a neighbor"):
        sub.neighbor(sub[0], sub[2])

    nei_01 = sub.neighbor(sub[0], sub[1])
    nei_10 = sub.neighbor(sub[1], sub[0])

    nei_01_atoms = sub.neighbor(sub[0].as_parent(), sub[1].as_parent())
    nei_10_atoms = sub.neighbor(sub[1].as_parent(), sub[0].as_parent())

    assert nei_01.src.id == 0
    assert nei_01.dst.id == 1

    assert nei_10.src.id == 1
    assert nei_10.dst.id == 0

    assert nei_01_atoms.src.id == 0
    assert nei_01_atoms.dst.id == 1

    assert nei_10_atoms.src.id == 1
    assert nei_10_atoms.dst.id == 0

    assert (
        nei_01.bond.id
        == nei_10.bond.id
        == nei_01_atoms.bond.id
        == nei_10_atoms.bond.id
    )


def test_erase_atom(molsub: Molecule):
    sub = molsub.subs[0]

    sub.erase_atom(sub[0])
    assert sub.num_atoms() == 2
    assert sub.num_bonds() == 0
    assert molsub[2] not in sub

    with pytest.raises(ValueError, match="atom not in substructure"):
        sub.erase_atom(molsub[2])

    sub.erase_atom(molsub[3])
    assert sub.num_atoms() == 1
    assert molsub[3] not in sub


def test_erase_bond(molsub: Molecule):
    sub = molsub.subs[0]

    pbond = sub.bond(0).as_parent()
    sub.erase_bond(sub.bond(0))
    assert sub.num_bonds() == 0
    assert pbond not in sub.bonds()

    with pytest.raises(ValueError, match="bond not in substructure"):
        sub.erase_bond(pbond)

    sub.refresh_bonds()
    assert sub.num_bonds() == 1

    sub.erase_bond(pbond)
    assert sub.num_bonds() == 0
    assert pbond not in sub.bonds()

    assert sub.num_atoms() == 3


def test_erase_bond_ends(molsub: Molecule):
    sub = molsub.subs[0]

    sub.erase_bond(sub[0], sub[1])
    assert sub.num_bonds() == 0

    with pytest.raises(ValueError, match="bond not in substructure"):
        sub.erase_bond(molsub[2], molsub[3])

    sub.refresh_bonds()
    assert sub.num_bonds() == 1

    sub.erase_bond(molsub[2], molsub[3])
    assert sub.num_bonds() == 0

    assert sub.num_atoms() == 3


def test_add_other(molsub: Molecule):
    molsub.add_from(molsub.subs[0])

    assert len(molsub) == 14
    assert molsub.atom(11).atomic_number == 6
    assert molsub.atom(12).atomic_number == 6
    assert molsub.atom(13).atomic_number == 1

    assert molsub.num_bonds() == 11
    assert molsub.bond(11, 12).order == 1


def test_add_other_external(molsub: Molecule):
    molsub.add_from(molsub.subs[0].copy())

    assert len(molsub) == 14
    assert molsub.atom(11).atomic_number == 6
    assert molsub.atom(12).atomic_number == 6
    assert molsub.atom(13).atomic_number == 1

    assert molsub.num_bonds() == 11
    assert molsub.bond(11, 12).order == 1


def test_bond_refresh(molsub: Molecule):
    sub = molsub.subs[0]
    sub.add_atoms([1], False)

    assert sub.num_bonds() == 1

    with pytest.raises(ValueError, match="no such bond"):
        sub.bond(sub[0], sub[1])

    sub.refresh_bonds()

    assert sub.num_bonds() == 4
    assert sub.bond(sub[0], sub[1]).order == BondOrder.Single

    sub.add_atoms([0], False)
    sub.refresh_bonds()
    assert sub.num_bonds() == 5


def test_implicit_hydrogens(molsub: Molecule):
    sub = molsub.subs[0]

    sub.conceal_hydrogens()

    assert len(sub) == 2

    assert sub[0].implicit_hydrogens == 0
    assert sub[1].implicit_hydrogens == 0

    for atom in sub:
        assert atom.atomic_number != 1


def test_implicit_hydrogens_external(molsub: Molecule):
    sub = molsub.subs[0].copy()

    sub.conceal_hydrogens()

    assert len(sub) == 2

    assert sub[0].implicit_hydrogens == 0
    assert sub[1].implicit_hydrogens == 0

    for atom in sub:
        assert atom.atomic_number != 1


def test_get_conformer(mol3dsub: Molecule):
    sub = mol3dsub.subs[0]

    conf = mol3dsub.get_conf()
    sconf = sub.get_conf()

    assert np.allclose(conf[sub.parent_atoms()], sconf)


def test_set_conformer(mol3dsub: Molecule):
    sub = mol3dsub.subs[0]

    sconf = sub.get_conf()
    sconf += 1

    sub.set_conf(sconf)

    assert np.allclose(sconf, sub.get_conf())
    assert np.allclose(sconf, mol3dsub.get_conf()[sub.parent_atoms()])


def test_conformers(mol3dsub: Molecule):
    sub = mol3dsub.subs[0]

    i = -1
    for i, conf in enumerate(sub.conformers(), 1):  # noqa: B007
        assert conf.shape == (sub.num_atoms(), 3)
    assert i == 2


def test_bond_length(mol3dsub: Molecule):
    sub = mol3dsub.subs[0]

    bond = sub.bond(0)

    l1 = bond.length()
    l2 = np.linalg.norm(bond.src.get_pos() - bond.dst.get_pos())
    assert l1 == pytest.approx(l2)

    lsq1 = bond.sqlen()
    assert lsq1 == pytest.approx(l2**2)


def test_iter_substruct(molsub: Molecule):
    subs = molsub.subs

    for sub in subs:
        sub.id = 100
        assert len(sub) == 3

    assert subs[0].id == 100


def test_add_substruct(molsub: Molecule):
    subs = molsub.subs

    subs.add([7, 8, 9])
    assert len(subs) == 2
    assert subs[1].num_atoms() == 3
    assert subs[1].parent_atoms() == [7, 8, 9]

    subs.add(0, [5, 6, 7])
    assert len(subs) == 3
    assert subs[0].num_atoms() == 3
    assert subs[0].parent_atoms() == [5, 6, 7]


def test_append_substruct(molsub: Molecule):
    subs = molsub.subs
    sub = subs[0]
    sub_copy = sub.copy()

    sub.id = 100

    subs.append(sub_copy)
    assert len(subs) == 2

    subs.append(subs[1])
    assert len(subs) == 3

    assert sub.id == 100
    assert subs[1].id != 100
    assert subs[2].id != 100
    assert subs[1].id == subs[2].id


def test_insert_substruct(molsub: Molecule):
    subs = molsub.subs
    sub = subs[0]
    sub_copy = sub.copy()

    sub.id = 100

    subs.insert(0, sub_copy)
    assert len(subs) == 2

    with pytest.raises(RuntimeError):
        assert sub.id is not None

    subs.insert(2, sub_copy)
    assert len(subs) == 3

    assert subs[1].id == 100
    assert subs[0].id != 100
    assert subs[2].id != 100
    assert subs[0].id == subs[2].id


def test_set_substruct(molsub: Molecule):
    subs = molsub.subs
    sub = subs[0]
    subs[0] = sub

    with pytest.raises(RuntimeError):
        assert sub.id is not None

    newsub = subs[0]
    assert len(newsub) == 3

    subs[0] = newsub.copy()
    newsub = subs[0]
    assert len(newsub) == 3


def test_del_substruct(molsub: Molecule):
    subs = molsub.subs
    del subs[0]
    assert len(subs) == 0


def test_pop_substruct(molsub: Molecule):
    subs = molsub.subs
    sub = subs.pop(0)
    assert len(subs) == 0
    assert len(sub) == 3


def test_clear_substruct(molsub: Molecule):
    subs = molsub.subs
    subs.clear()
    assert len(subs) == 0

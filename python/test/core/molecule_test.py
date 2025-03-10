#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#
# pyright: reportAttributeAccessIssue=false

import numpy as np
import pytest

from nuri.core import AtomData, BondData, BondOrder, Hyb, Molecule


def test_empty():
    mol = Molecule()
    assert not mol
    assert len(mol) == 0
    assert mol.num_atoms() == 0
    assert mol.num_bonds() == 0
    assert not mol.bonds()
    assert mol.num_confs() == 0
    assert mol.num_fragments() == 0
    assert mol.num_sssr() == 0
    assert mol.name == ""
    assert not mol.props
    assert not mol.subs

    with pytest.raises(IndexError):
        mol[1]

    with pytest.raises(IndexError):
        mol.atom(1)

    with pytest.raises(IndexError):
        mol.bond(1)

    with pytest.raises(IndexError):
        mol.bonds()[1]


def test_mutator_errors():
    mol = Molecule()

    with mol.mutator() as mut1:
        mut1.add_atom(6)

        with pytest.raises(RuntimeError):
            mol.clear()

        with pytest.raises(RuntimeError):
            mol.clear_atoms()

        with pytest.raises(RuntimeError):
            mol.clear_bonds()

        with pytest.raises(RuntimeError):
            with mol.mutator():
                pass

        with pytest.raises(RuntimeError):
            with mut1:
                pass

        mut1.add_atom(7)

    assert len(mol) == 2

    with pytest.raises(RuntimeError):
        mut1.add_atom(8)

    with pytest.raises(RuntimeError):
        mut1.__exit__()

    with mut1:
        mut1.add_atom(8)

    assert len(mol) == 3


def test_add_atom():
    mol = Molecule()
    with mol.mutator() as mut:
        atom = mut.add_atom(6)
        assert atom.atomic_number == 6

        with pytest.raises(ValueError, match="invalid atomic number"):
            mut.add_atom(1000)

        with pytest.raises(ValueError, match="invalid hybridization"):
            atom.update(hyb=Hyb(100))

        atom_data = atom.copy_data()

    assert atom_data.atomic_number == 6

    assert mol
    assert len(mol) == 1
    assert mol.num_atoms() == 1

    atom = mol[0]
    assert atom.id == 0
    assert atom.atomic_number == 6

    atom = mol.atom(0)
    assert atom.id == 0
    assert atom.atomic_number == 6

    i = -1
    for i, atom in enumerate(mol, 1):  # noqa: B007
        assert atom.atomic_number == 6
    assert i == 1

    with mol.mutator() as mut:
        atom = mut.add_atom(AtomData(7))
        assert atom.atomic_number == 7


def test_add_bond():
    mol = Molecule()

    with mol.mutator() as mut:
        a1 = mut.add_atom(6)
        a2 = mut.add_atom(6)
        bond = mut.add_bond(a1, a2)
        assert bond.order == 1

        bond_data = bond.copy_data()

        with pytest.raises(ValueError, match="same"):
            mut.add_bond(0, 0)

        with pytest.raises(ValueError, match="duplicate bond"):
            mut.add_bond(0, 1, BondOrder.Double)

        with pytest.raises(IndexError):
            mut.add_bond(0, 2, BondOrder.Double)

        mut.add_atom(8)

        with pytest.raises(ValueError, match="invalid bond order"):
            mut.add_bond(0, 2, BondOrder(1000))

        mut.add_bond(0, 2, BondOrder.Double)

    assert bond_data.order == 1

    assert len(mol) == 3
    assert mol.num_atoms() == 3
    assert mol.num_bonds() == 2

    bond = mol.bond(0)
    assert bond.order == 1

    assert {bond.src.id, bond.dst.id} == {0, 1}
    assert bond.src.atomic_number == 6
    assert bond.dst.atomic_number == 6
    assert mol.has_bond(0, 1)
    assert mol.has_bond(bond.src, bond.dst)

    i = -1
    for i, bond in enumerate(mol.bonds(), 1):
        assert bond.order == i
    assert i == 2

    assert mol.bonds()[0].order == 1

    with mol.mutator() as mut:
        a3 = mut.add_atom(6)
        a4 = mut.add_atom(6)

        assert not mol.has_bond(a3.id, a4.id)
        assert not mol.has_bond(a3, a4)

        mut.add_bond(a3, a4, BondData(BondOrder.Triple))
        mut.add_bond(1, 3, BondData(BondOrder.Aromatic))


def test_add_conformer():
    mol = Molecule()
    with mol.mutator() as mut:
        mut.add_atom(6)
        mut.add_atom(6)

    mol.add_conf(np.arange(6).reshape(2, 3))

    with pytest.raises(ValueError, match="expected 3 columns"):
        mol.add_conf(np.arange(6).reshape(3, 2))

    with pytest.raises(ValueError, match="different number of atoms"):
        mol.add_conf(np.arange(9).reshape(3, 3))

    assert mol.num_confs() == 1

    conf = mol.get_conf()
    assert conf.shape == (2, 3)
    assert conf[0, 2] == 2

    pos = mol[0].get_pos()
    assert pos.shape == (3,)
    assert np.allclose(pos, conf[0])

    with pytest.raises(IndexError):
        mol.get_conf(1)

    mol.add_conf(np.arange(1, 7).reshape(2, 3))

    i = -1
    for i, conf in enumerate(mol.conformers()):
        assert conf.shape == (2, 3)
        assert conf[0, 2] == 2 + i

    assert i == 1


def test_neighbors(mol: Molecule):
    c0 = mol.atom(0)
    assert len(c0) == 3

    i = -1
    for i, nei in enumerate(c0, 1):  # noqa: B007
        assert nei.src.id == 0
        assert nei.dst.id in (2, 8, 9)
    assert i == 3

    assert c0[0].dst.id == 9

    with pytest.raises(IndexError):
        c0[3]

    assert c0.count_neighbors() == 4
    assert c0.count_hydrogens() == 2
    assert c0.count_heavy_neighbors() == 2

    c2 = mol.atom(2)
    assert c2.count_neighbors() == 4
    assert c2.count_hydrogens() == 1
    assert c2.count_heavy_neighbors() == 3


def test_copy(mol: Molecule):
    nol = mol.copy()
    assert len(mol) == len(nol)

    assert mol[0] in mol
    assert mol[0] not in nol

    assert mol.bonds()[0] in mol
    assert mol.bonds()[0] in mol.bonds()
    assert mol.bonds()[0] not in nol
    assert mol.bonds()[0] not in nol.bonds()

    mol.clear()
    assert not mol
    assert nol


def test_update_atom(mol: Molecule):
    atom = mol.atom(0)
    atom.update(atomic_number=7, formal_charge=1, implicit_hydrogens=2)
    assert atom.atomic_number == 7
    assert atom.formal_charge == 1
    assert atom.implicit_hydrogens == 2

    atom.aromatic = True
    assert atom.aromatic


def test_update_bond(mol: Molecule):
    bond = mol.bond(0)
    bond.update(order=BondOrder.Triple)
    assert bond.order == BondOrder.Triple

    bond.aromatic = True
    assert bond.aromatic


def test_find_bond(mol: Molecule):
    with pytest.raises(ValueError, match="no such bond"):
        mol.bond(0, 1)

    bond_02 = mol.bond(0, 2)
    bond_20 = mol.bond(2, 0)
    bond_02_atoms = mol.bond(mol[0], mol[2])

    assert bond_02.src.id == bond_20.src.id
    assert bond_02.dst.id == bond_20.dst.id
    assert bond_02.id == bond_20.id

    assert bond_02.src.id == bond_02_atoms.src.id
    assert bond_02.dst.id == bond_02_atoms.dst.id
    assert bond_02.id == bond_02_atoms.id


def test_find_neighbor(mol: Molecule):
    with pytest.raises(ValueError, match="not a neighbor"):
        mol.neighbor(0, 1)

    nei_02 = mol.neighbor(0, 2)
    nei_20 = mol.neighbor(2, 0)
    nei_02_atoms = mol.neighbor(mol[0], mol[2])

    assert nei_02.src.id == 0
    assert nei_02.dst.id == 2

    assert nei_20.src.id == 2
    assert nei_20.dst.id == 0

    assert nei_02_atoms.src.id == 0
    assert nei_02_atoms.dst.id == 2

    assert nei_02.bond.id == nei_20.bond.id == nei_02_atoms.bond.id


def test_implicit_hydrogens(mol: Molecule):
    mol.conceal_hydrogens()

    assert len(mol) == 6
    assert mol.atom(0).implicit_hydrogens == 2
    for atom in mol:
        assert atom.atomic_number != 1

    mol.reveal_hydrogens()

    assert len(mol) == 12
    assert mol.atom(0).implicit_hydrogens == 0
    assert sum(atom.atomic_number == 1 for atom in mol) == 6


def test_clear_bonds(mol: Molecule):
    mol.name = "test"

    mol.clear_bonds()
    assert not mol.bonds()
    assert mol.num_atoms() == 11
    assert mol.name == "test"


def test_clear_atoms(mol: Molecule):
    mol.name = "test"

    mol.clear_atoms()
    assert not mol
    assert not mol.bonds()
    assert mol.name == "test"


def test_clear_bonds_mutator(mol: Molecule):
    mol.name = "test"

    with mol.mutator() as mut:
        mut.clear_bonds()

    assert not mol.bonds()
    assert mol.num_atoms() == 11
    assert mol.name == "test"


def test_clear_atoms_mutator(mol: Molecule):
    mol.name = "test"

    with mol.mutator() as mut:
        mut.clear_atoms()

    assert not mol
    assert not mol.bonds()
    assert mol.name == "test"


def test_clear_mutator(mol: Molecule):
    mol.name = "test"

    with mol.mutator() as mut:
        mut.clear()

    assert not mol
    assert not mol.bonds()
    assert not mol.name


def test_add_other(mol: Molecule):
    other = Molecule()
    with other.mutator() as mut:
        mut.add_atom(6)
        mut.add_atom(6)
        mut.add_bond(0, 1)

    mol.add_from(other)

    assert len(mol) == 13
    assert mol.atom(11).atomic_number == 6
    assert mol.atom(12).atomic_number == 6
    assert mol.bond(11, 12).order == 1


def test_erase_atoms(mol: Molecule):
    with mol.mutator() as mut:
        mut.mark_atom_erase(0)

        with pytest.raises(IndexError):
            mut.mark_atom_erase(1000)

        mut.mark_atom_erase(mol[0])
        mut.mark_atom_erase(mol[1])

    assert len(mol) == 9


def test_erase_bonds(mol: Molecule):
    with mol.mutator() as mut:
        mut.mark_bond_erase(0, 2)

        with pytest.raises(ValueError, match="no such bond"):
            mut.mark_bond_erase(0, 3)

        mut.mark_bond_erase(mol[0], mol[2])
        mut.mark_bond_erase(2)

        with pytest.raises(IndexError):
            mut.mark_bond_erase(1000)

        mut.mark_bond_erase(mol.bond(0))

    assert mol.num_bonds() == 7


def test_props(mol: Molecule):
    mol.props["test"] = "1"
    assert mol.props["test"] == "1"

    mol.props = {"test": "2"}
    assert mol.props["test"] == "2"

    atom = mol.atom(0)
    atom.props = mol.props
    assert atom.props["test"] == "2"
    mol.props["test"] = "3"
    assert atom.props["test"] == "2"

    bond = mol.bond(0)
    bond.props = mol.props
    assert bond.props["test"] == "3"
    mol.props["test"] = "4"
    assert bond.props["test"] == "3"

    with pytest.raises(TypeError):
        mol.props = {"test": 5}

    assert mol.props["test"] == "4"

    data = AtomData()
    data.props = mol.props
    assert data.props["test"] == "4"
    data.props["test"] = "5"
    assert mol.props["test"] == "4"
    assert data.props["test"] == "5"

    data = BondData()
    data.props = mol.props
    assert data.props["test"] == "4"
    data.props["test"] = "5"
    assert mol.props["test"] == "4"
    assert data.props["test"] == "5"


def test_set_conformer(mol3d: Molecule):
    conf = mol3d.get_conf()
    conf[0] = [100, 200, 300]

    mol3d.set_conf(conf)
    assert np.allclose(mol3d.get_conf()[0], [100, 200, 300])

    atom = mol3d[1]
    atom.set_pos([400, 500, 600])
    assert np.allclose(atom.get_pos(), [400, 500, 600])
    assert np.allclose(mol3d.get_conf()[1], [400, 500, 600])


def test_add_conformer_index(mol3d: Molecule):
    conf = np.zeros((mol3d.num_atoms(), 3))
    mol3d.add_conf(conf, 1)

    assert mol3d.num_confs() == 3
    assert mol3d.get_conf(1).sum() == 0

    conf = np.ones((mol3d.num_atoms(), 3))
    mol3d.add_conf(conf, -1)

    assert mol3d.num_confs() == 4
    assert np.allclose(mol3d.get_conf(-2), conf)


def test_del_conformer(mol3d: Molecule):
    conf = mol3d.get_conf(1)
    mol3d.del_conf(0)

    assert mol3d.num_confs() == 1
    assert np.allclose(mol3d.get_conf(), conf)

    with pytest.raises(IndexError):
        mol3d.del_conf(1)


def test_bond_length(mol3d: Molecule):
    bond = mol3d.bond(0)

    l1 = bond.length()
    l2 = np.linalg.norm(bond.src.get_pos() - bond.dst.get_pos())
    assert l1 == pytest.approx(l2)

    lsq1 = bond.sqlen()
    assert lsq1 == pytest.approx(l2**2)


def test_bond_rotation(mol3d: Molecule):
    original = mol3d.get_conf()

    # 2 -> 0
    bond = mol3d.bond(4)

    rotated = original.copy()
    rotated[[5, 8, 9]] = [
        [-0.504, 0.569, 1.939],
        [0.466, 0.133, -0.219],
        [-0.969, 0.904, 1.108],
    ]

    # Rotate 0 side
    bond.rotate(30)
    assert np.allclose(mol3d.get_conf(), rotated, atol=1e-3)
    bond.rotate(-30)
    assert np.allclose(mol3d.get_conf(), original, atol=1e-3)

    rotated = original.copy()
    rotated[[1, 3, 4, 6, 7]] = [
        [-0.615, 0.817, -2.516],
        [-1.757, 0.682, -1.889],
        [-0.908, 2.670, -1.290],
        [-2.738, 0.231, -1.862],
        [-0.001, 0.555, -3.365],
    ]

    # Rotate 2 side
    bond.rotate(30, True)
    assert np.allclose(mol3d.get_conf(), rotated, atol=1e-3)
    bond.rotate(-30, True)
    assert np.allclose(mol3d.get_conf(), original, atol=1e-3)


def test_clear_conformers(mol3d: Molecule):
    mol3d.clear_confs()
    assert not mol3d.num_confs()

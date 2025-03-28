#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from nuri import periodic_table
from nuri.core import (
    AtomData,
    BondConfig,
    BondData,
    BondOrder,
    Chirality,
    Hyb,
    Molecule,
)
from nuri.core._core import _PropertyMap


def _get_mol():
    m = Molecule()
    with m.mutator() as mut:
        mut.add_atom(6)
        mut.add_atom(6)
        mut.add_bond(0, 1)
    return m


def _get_atom():
    return _get_mol()[0]


def _get_subatom():
    m = _get_mol()
    s = m.subs.add([0])
    return s[0]


def _get_subatom_external():
    m = _get_mol()
    s = m.sub([0])
    return s[0]


@pytest.mark.parametrize(
    "datalike",
    [
        AtomData(),
        _get_atom(),
        _get_subatom(),
        _get_subatom_external(),
    ],
    ids=[
        "AtomData",
        "Atom",
        "ProxySubAtom",
        "SubAtom",
    ],
)
def test_atom_data_interface(datalike: AtomData):
    datalike.hyb = Hyb.SP3D2
    assert datalike.hyb == Hyb.SP3D2

    datalike.implicit_hydrogens = 100
    assert datalike.implicit_hydrogens == 100

    datalike.formal_charge = -100
    assert datalike.formal_charge == -100

    datalike.partial_charge = -100.0
    assert datalike.partial_charge == pytest.approx(-100.0)

    datalike.aromatic = True
    assert datalike.aromatic

    datalike.conjugated = True
    assert datalike.conjugated

    datalike.ring = True
    assert datalike.ring

    datalike.chirality = Chirality.Unknown
    assert datalike.chirality == Chirality.Unknown
    datalike.chirality = None
    assert datalike.chirality == Chirality.Unknown

    datalike.chirality = Chirality.CW
    assert datalike.chirality == Chirality.CW

    datalike.name = "test"
    assert datalike.name == "test"

    # readonly
    assert datalike.atomic_number is not None
    assert datalike.element_symbol is not None
    assert datalike.element_name is not None
    assert datalike.atomic_weight is not None

    datalike.element = periodic_table["Na"]
    assert datalike.element.atomic_number == 11
    datalike.set_element(7)
    assert datalike.element.atomic_number == 7
    datalike.set_element("C")
    assert datalike.element.atomic_number == 6
    datalike.set_element("Oxygen")
    assert datalike.element.atomic_number == 8
    datalike.set_element(periodic_table["Na"])
    assert datalike.element.atomic_number == 11

    iso = datalike.get_isotope()
    assert iso is not None
    iso = datalike.get_isotope(True)
    assert iso is None

    datalike.set_isotope(22)
    assert datalike.get_isotope(True).mass_number == 22

    datalike.set_isotope(datalike.element.isotopes[0])
    assert datalike.get_isotope(True) == datalike.element.isotopes[0]

    datalike.update(
        hyb=Hyb.SP3,
        implicit_hydrogens=0,
        formal_charge=0,
        partial_charge=0.0,
        atomic_number=6,
        aromatic=False,
        conjugated=False,
        ring=False,
        chirality=Chirality.Unknown,
        name="test2",
    )
    assert datalike.hyb == Hyb.SP3
    assert datalike.implicit_hydrogens == 0
    assert datalike.formal_charge == 0
    assert datalike.partial_charge == pytest.approx(0.0)
    assert datalike.atomic_number == 6
    assert not datalike.aromatic
    assert not datalike.conjugated
    assert not datalike.ring
    assert datalike.chirality == Chirality.Unknown
    assert datalike.name == "test2"

    with pytest.raises(ValueError, match="mutually exclusive"):
        datalike.update(
            atomic_number=12,
            element=periodic_table["Mg"],
        )
    assert datalike.atomic_number == 6

    ad = AtomData()
    ad.update_from(datalike)
    assert ad.atomic_number == datalike.atomic_number


def _get_bond():
    return _get_mol().bonds()[0]


def _get_subbond():
    m = _get_mol()
    s = m.subs.add(bonds=[0])
    return s.bonds()[0]


def _get_subbond_external():
    m = _get_mol()
    s = m.sub(bonds=[0])
    return s.bonds()[0]


@pytest.mark.parametrize(
    "datalike",
    [
        BondData(),
        _get_bond(),
        _get_subbond(),
        _get_subbond_external(),
    ],
    ids=[
        "BondData",
        "Bond",
        "ProxySubBond",
        "SubBond",
    ],
)
def test_bond_data_interface(datalike: BondData):
    datalike.order = BondOrder.Double
    assert datalike.order == BondOrder.Double

    assert datalike.approx_order() == pytest.approx(2.0)
    assert not datalike.rotatable()

    datalike.ring = True
    assert datalike.ring

    datalike.aromatic = True
    assert datalike.aromatic

    datalike.conjugated = True
    assert datalike.conjugated

    datalike.config = BondConfig.Unknown
    assert datalike.config == BondConfig.Unknown
    datalike.config = None
    assert datalike.config == BondConfig.Unknown

    datalike.config = BondConfig.Cis
    assert datalike.config == BondConfig.Cis

    datalike.update(
        order=BondOrder.Single,
        aromatic=False,
        conjugated=False,
        ring=False,
        config=BondConfig.Unknown,
        name="test2",
    )
    assert datalike.order == BondOrder.Single
    assert not datalike.aromatic
    assert not datalike.conjugated
    assert not datalike.ring
    assert datalike.config == BondConfig.Unknown
    assert datalike.name == "test2"

    bd = BondData()
    bd.update_from(datalike)
    assert bd.order == datalike.order


@pytest.mark.parametrize(
    "maplike",
    [
        _PropertyMap(),
        Molecule().props,
    ],
    ids=[
        "PropertyMap",
        "ProxyPropertyMap",
    ],
)
def test_propertymap_interface(maplike: _PropertyMap):
    maplike["test"] = "1"
    assert maplike["test"] == "1"
    assert "test" in maplike
    assert len(maplike) == 1
    assert maplike.get("test") == "1"
    assert maplike.get("nonexistent", "2") == "2"

    for it, expected in [
        (maplike, "test"),
        (maplike.keys(), "test"),
        (maplike.values(), "1"),
        (maplike.items(), ("test", "1")),
    ]:
        for k in it:
            assert k == expected
            break
        else:
            pytest.fail("iter failed")

    del maplike["test"]
    assert "test" not in maplike

    maplike["test"] = "1"
    assert maplike.pop("test") == "1"
    assert maplike.pop("test", "2") == "2"

    maplike["test"] = "3"
    assert maplike.pop("test", "4") == "3"

    maplike["test"] = "1"
    assert maplike.popitem() == ("test", "1")

    with pytest.raises(KeyError):
        maplike.popitem()

    maplike["test"] = "1"
    maplike.clear()
    assert not maplike

    ret = maplike.setdefault("test", "1")
    assert ret == "1"

    d = {"test": "2"}
    maplike.update(d)
    assert maplike["test"] == "2"

    maplike.update(test="3")
    assert maplike["test"] == "3"

    maplike.update([("test", "4")])
    assert maplike["test"] == "4"

#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from nuri import periodic_table


def test_isotope_repr():
    iso = periodic_table[6].get_isotope(13)
    assert repr(iso) == "<Isotope 13 C>"


def test_isotopes_iter():
    for iso in periodic_table[6].isotopes:
        assert iso.element.atomic_number == 6

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri.scpilot.io import load_structure
from nuri.scpilot.radii import (
    SIDECHAIN_H,
    template_hydrogens,
    united_atom_radii,
    united_atom_radius,
)

HEAVY_ATOM_COUNTS = {
    "ALA": 5,
    "ARG": 11,
    "ASN": 8,
    "ASP": 8,
    "CYS": 6,
    "GLN": 9,
    "GLU": 9,
    "GLY": 4,
    "HIS": 10,
    "ILE": 8,
    "LEU": 8,
    "LYS": 9,
    "MET": 8,
    "PHE": 11,
    "PRO": 7,
    "SER": 6,
    "THR": 7,
    "TRP": 14,
    "TYR": 12,
    "VAL": 7,
}


@pytest.mark.parametrize(("res", "n"), sorted(HEAVY_ATOM_COUNTS.items()))
def test_template_covers_standard_residue(res, n):
    assert len(SIDECHAIN_H[res]) + 4 == n


def test_template_special_cases():
    assert template_hydrogens("GLY", "CA") == 2
    assert template_hydrogens("PRO", "N") == 0
    assert template_hydrogens("ALA", "N") == 1
    assert template_hydrogens("ALA", "OXT") == 0
    assert template_hydrogens("XYZ", "CA") is None
    assert template_hydrogens("ALA", "CG") is None


def test_radius_rules():
    assert united_atom_radius("C", 0) == pytest.approx(1.80)
    assert united_atom_radius("C", 1) == pytest.approx(1.85)
    assert united_atom_radius("C", 1, aromatic_ch=True) == pytest.approx(1.90)
    assert united_atom_radius("C", 2) == pytest.approx(1.90)
    assert united_atom_radius("C", 3) == pytest.approx(1.95)
    assert united_atom_radius("N", 0) == pytest.approx(1.65)
    assert united_atom_radius("N", 1) == pytest.approx(1.65)
    assert united_atom_radius("N", 2) == pytest.approx(1.70)
    assert united_atom_radius("N", 3) == pytest.approx(1.75)
    assert united_atom_radius("O", 0) == pytest.approx(1.60)
    assert united_atom_radius("O", 1) == pytest.approx(1.70)
    assert united_atom_radius("S", 1) == pytest.approx(1.90)
    assert united_atom_radius("Fe", 0) is None


def test_1ar1_fully_templated(test_data):
    st = load_structure(test_data / "1ar1.pdb")
    radii, fallback = united_atom_radii(st)
    assert not fallback.any()
    assert radii.min() >= 1.60
    assert radii.max() <= 1.95 + 1e-9
    assert np.mean(radii) > np.mean(st.vdw_radii)

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import numpy as np
import pytest

import nuri
from nuri.core import Molecule
from nuri.fmt import pdb
from nuri.param import assign_radii

ELEMENT_RADIUS_ATTR = {"cov": "covalent_radius", "vdw": "vdw_radius"}

pdb_data = """\
ATOM      1  N   MET A   1     -13.991  -6.903  33.129  1.00 22.97           N
ATOM      2  CA  MET A   1     -13.215  -8.093  33.479  1.00 22.29           C
ATOM      3  C   MET A   1     -13.314  -8.351  34.974  1.00 22.05           C
ATOM      4  O   MET A   1     -12.547  -9.129  35.537  1.00 22.25           O
ATOM      5  CB  MET A   1     -11.754  -7.911  33.072  1.00 22.24           C
ATOM      6  CG  MET A   1     -11.575  -7.558  31.609  1.00 22.06           C
ATOM      7  SD  MET A   1     -12.824  -8.351  30.571  1.00 21.21           S
ATOM      8  CE  MET A   1     -12.200 -10.034  30.491  1.00 21.07           C
HETATM    9  O   HOH A 101      -9.123  -4.567  30.000  1.00 30.00           O
"""


@pytest.fixture()
def phenol() -> Molecule:
    return next(nuri.readstring("smi", "Oc1ccccc1"))


@pytest.fixture()
def model(tmp_path: Path) -> pdb.Model:
    file = tmp_path / "test.pdb"
    file.write_text(pdb_data)
    return pdb.read_models(file)[0]


def _element_radii(mol: Molecule, kind: str) -> np.ndarray:
    attr = ELEMENT_RADIUS_ATTR[kind]
    return np.array([getattr(atom.element, attr) for atom in mol])


@pytest.mark.parametrize("kind", ["cov", "vdw"])
def test_mol_element(phenol: Molecule, kind: str):
    radii = assign_radii(phenol, kind)
    assert radii.dtype == np.float64
    assert np.allclose(radii, _element_radii(phenol, kind))


def test_mol_connolly(phenol: Molecule):
    radii = assign_radii(phenol, "connolly")
    assert radii[0] == pytest.approx(1.70)
    assert radii[1] == pytest.approx(1.80)
    assert np.allclose(radii[2:], 1.90)

    phenol.reveal_hydrogens()
    radii = assign_radii(phenol, "connolly", missing="ignore")
    assert (radii[:7] > 0).all()
    assert (radii[7:] < 0).all()

    with pytest.raises(
        ValueError,
        match=r"no connolly radius for atoms: \[7, 8, 9, 10, 11, 12\]",
    ):
        assign_radii(phenol, "connolly")

    radii = assign_radii(phenol, "connolly", missing="vdw")
    assert np.allclose(radii[7:], _element_radii(phenol, "vdw")[7:])
    assert radii[0] == pytest.approx(1.70)


def test_mol_scale(phenol: Molecule):
    phenol.reveal_hydrogens()
    radii = assign_radii(phenol, "connolly", missing="ignore", scale=0.5)
    assert radii[0] == pytest.approx(0.85)
    assert (radii[7:] < 0).all()


def test_mol_template(phenol: Molecule):
    template = np.full(phenol.num_atoms(), -1.0)
    template[0] = 3.0

    radii = assign_radii(phenol, "vdw", template=template)
    assert radii.ctypes.data != template.ctypes.data
    assert template[0] == 3.0
    assert radii[0] == 3.0
    assert np.allclose(radii[1:], _element_radii(phenol, "vdw")[1:])

    radii = assign_radii(phenol, "vdw", scale=2.0, template=template)
    assert radii[0] == 3.0
    assert np.allclose(radii[1:], 2 * _element_radii(phenol, "vdw")[1:])

    with pytest.raises(ValueError, match="template size"):
        assign_radii(phenol, "vdw", template=template[:-1])


def test_model_connolly(model: pdb.Model):
    radii = assign_radii(model, "connolly", missing="ignore")
    assert radii.shape == (9,)
    expected = [1.65, 1.85, 1.80, 1.60, 1.90, 1.90, 1.90, 1.95]
    assert np.allclose(radii[:8], expected)
    assert radii[8] < 0

    with pytest.raises(ValueError, match=r"atoms: \[8\]"):
        assign_radii(model, "connolly")

    radii = assign_radii(model, "connolly", missing="cov")
    assert radii[8] == pytest.approx(model.atoms[8].element.covalent_radius)


@pytest.mark.parametrize("kind", ["cov", "vdw"])
def test_model_element(model: pdb.Model, kind: str):
    attr = ELEMENT_RADIUS_ATTR[kind]
    radii = assign_radii(model, kind)
    expected = [getattr(atom.element, attr) for atom in model.atoms]
    assert np.allclose(radii, expected)


def test_bad_args(phenol: Molecule):
    with pytest.raises(ValueError, match="kind must be"):
        assign_radii(phenol, "bondi")

    with pytest.raises(ValueError, match="missing must be"):
        assign_radii(phenol, "vdw", missing="connolly")

    with pytest.raises(ValueError, match="scale"):
        assign_radii(phenol, "vdw", scale=0.0)

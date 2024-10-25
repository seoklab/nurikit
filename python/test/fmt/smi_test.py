#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import List

import nuri
from nuri.core import Hyb, Molecule

smi_data = """\
C
CC
CCC propane
C1=CC=CC=C1 benzene
"""


def _verify_mols(mols: List[Molecule]):
    assert len(mols) == 4
    assert len(mols[0]) == 1
    assert len(mols[1]) == 2

    assert len(mols[2]) == 3
    assert mols[2].name == "propane"
    for atom in mols[2]:
        assert atom.hyb == Hyb.SP3

    assert len(mols[3]) == 6
    assert mols[3].name == "benzene"
    for atom in mols[3]:
        assert atom.hyb == Hyb.SP2
        assert atom.aromatic


def test_smiles_file(tmp_path: Path):
    file = tmp_path / "test.smi"
    file.write_text(smi_data)

    mols = list(nuri.readfile("smi", file))
    _verify_mols(mols)


def test_smiles_str():
    mols = list(nuri.readstring("smi", smi_data))
    _verify_mols(mols)

    smiles_re = "".join(map(nuri.to_smiles, mols))
    mols_re = list(nuri.readstring("smi", smiles_re))
    _verify_mols(mols_re)

#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import List

import pytest

import nuri
from nuri.core import Molecule, SubstructureCategory

pdb_data = """\
MODEL        1
ATOM      7  N   MET A   1     -13.991  -6.903  33.129  1.00 22.97           N
ATOM      8  CA  MET A   1     -13.215  -8.093  33.479  1.00 22.29           C
ATOM      9  C   MET A   1     -13.314  -8.351  34.974  1.00 22.05           C
ATOM     10  O   MET A   1     -12.547  -9.129  35.537  1.00 22.25           O
ATOM     11  CB  MET A   1     -11.754  -7.911  33.072  1.00 22.24           C
ATOM     12  CG  MET A   1     -11.575  -7.558  31.609  1.00 22.06           C
ATOM     13  SD  MET A   1     -12.824  -8.351  30.571  1.00 21.21           S
ATOM     14  CE  MET A   1     -12.200 -10.034  30.491  1.00 21.07           C
ENDMDL
MODEL        2
ATOM     15  N   GLU A   2     -14.276  -7.680  35.597  1.00 21.61           N
ATOM     16  CA  GLU A   2     -14.539  -7.754  37.021  1.00 21.37           C
ATOM     17  C   GLU A   2     -14.838  -9.192  37.467  1.00 20.56           C
ATOM     18  O   GLU A   2     -14.382  -9.623  38.530  1.00 20.29           O
ATOM     19  CB  GLU A   2     -15.725  -6.832  37.337  1.00 21.91           C
ATOM     20  CG  GLU A   2     -16.178  -6.790  38.801  1.00 23.74           C
ATOM     21  CD  GLU A   2     -17.703  -6.770  38.937  1.00 26.25           C
ATOM     22  OE1 GLU A   2     -18.397  -7.133  37.957  1.00 26.86           O
ATOM     23  OE2 GLU A   2     -18.208  -6.402  40.025  1.00 27.14           O
ATOM     24  N   ASN B   3     -15.590  -9.921  36.639  1.00 19.52           N
ATOM     25  CA  ASN B   3     -16.096 -11.249  36.992  1.00 18.45           C
ATOM     26  C   ASN B   3     -15.210 -12.437  36.598  1.00 17.91           C
ATOM     27  O   ASN B   3     -15.565 -13.584  36.850  1.00 17.64           O
ATOM     28  CB  ASN B   3     -17.500 -11.436  36.413  1.00 18.28           C
ATOM     29  CG  ASN B   3     -18.490 -10.459  36.982  1.00 17.92           C
ATOM     30  OD1 ASN B   3     -18.464 -10.165  38.175  1.00 18.06           O
ATOM     31  ND2 ASN B   3     -19.375  -9.947  36.135  1.00 17.48           N
ENDMDL
"""


def _verify_mols(mols: List[Molecule]):
    assert len(mols) == 2

    assert len(mols[0]) == 8

    mol = mols[1]
    assert len(mol.subs) == 4
    for sub in mol.subs:
        if sub.category == SubstructureCategory.Chain:
            assert sub.name == "A" or sub.name == "B"
        elif sub.category == SubstructureCategory.Residue:
            if sub.id == 2:
                assert sub.name == "GLU"
            elif sub.id == 3:
                assert sub.name == "ASN"
            else:
                pytest.fail("Invalid residue ID")
        else:
            pytest.fail("Invalid substructure category")


def test_pdb_file(tmp_path: Path):
    file = tmp_path / "test.pdb"
    file.write_text(pdb_data)

    mols = list(nuri.readfile("pdb", file))
    _verify_mols(mols)


def test_pdb_str():
    mols = list(nuri.readstring("pdb", pdb_data))
    _verify_mols(mols)

    mols_re = [
        mol
        for pdb_re in map(nuri.to_pdb, mols)
        for mol in nuri.readstring("pdb", pdb_re)
    ]
    _verify_mols(mols_re)

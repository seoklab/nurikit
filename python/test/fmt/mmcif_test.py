#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import numpy as np

import nuri


def test_read_mmcif(test_data: Path):
    cif = test_data / "3cye_part.cif"

    mols = list(nuri.readfile("mmcif", cif, sanitize=False))
    assert len(mols) == 2

    mol = mols[0]
    assert mol.name == "3CYE"
    assert len(mol) == 67
    assert mol.num_bonds() == 1
    assert mol.has_bond(28, 34)
    assert mol.props["model"] == "1"

    assert mol.num_confs() == 2
    assert np.allclose(mol.get_conf(0)[0], mol.get_conf(1)[0])
    assert not np.allclose(mol.get_conf(0)[43], mol.get_conf(1)[43])

    assert len(mol.subs) == 10

    assert mol.subs[0].name == "VAL"
    assert mol.subs[0].num_atoms() == 7

    assert mol.subs[8].props["icode"] == "A"

    mol = mols[1]
    assert mol.name == "3CYE"
    assert len(mol) == 36
    assert mol.num_bonds() == 0
    assert mol.props["model"] == "2"

    assert mol.num_confs() == 2
    assert np.allclose(mol.get_conf(0)[0], mol.get_conf(1)[0])
    assert not np.allclose(mol.get_conf(0)[31], mol.get_conf(1)[31])

    assert len(mol.subs) == 6
    assert mol.subs[0].name == "VAL"
    assert mol.subs[0].num_atoms() == 7

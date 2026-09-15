#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import nuri
from nuri.core import Molecule
from nuri.fmt.cif import read_blocks


def _validate_3cye_part(mols: list[Molecule]):
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

    assert mol.subs[8].id == 169
    assert mol.subs[8].props["chain"] == "A"
    assert mol.subs[8].props["icode"] == "A"
    assert mol.subs[8].props["entity_id"] == "1"

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


def test_read_mmcif(test_data: Path):
    cif = test_data / "3cye_part.cif"

    mols = list(nuri.readfile("mmcif", cif, sanitize=False))
    _validate_3cye_part(mols)


def test_load_mmcif_from_frame(test_data: Path):
    frame = next(read_blocks(test_data / "3cye_part.cif")).data
    mols = frame.as_mols()
    _validate_3cye_part(mols)


def test_load_mmcif_no_atom_sites(tmp_path: Path):
    file = tmp_path / "meta.cif"
    file.write_text("data_meta\n_x.y 1\n")

    frame = next(read_blocks(file)).data
    assert len(frame.as_mols()) == 0


def test_load_mmcif_malformed_row(tmp_path: Path):
    file = tmp_path / "bad.cif"
    file.write_text(
        """data_bad
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
1 N N ALA A notanumber 1.000 2.000 3.000
"""
    )

    frame = next(read_blocks(file)).data
    with pytest.raises(ValueError, match="_atom_site"):
        frame.as_mols()


def test_reader_continues_after_record_error(test_data: Path, caplog):
    caplog.set_level(logging.INFO, logger="nuri")
    models = (test_data / "3cye_part.cif").read_text()
    bad = """data_bad
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
1 N N ALA A invalid 1 2 3
"""
    data = f"""data_empty
_x.y 1
{models}{bad}{models}"""
    reader = nuri.readstring("mmcif", data, sanitize=False)
    _validate_3cye_part([next(reader), next(reader)])
    with pytest.raises(ValueError, match="_atom_site"):
        next(reader)
    _validate_3cye_part(list(reader))
    assert (
        sum("No molecules in block empty" in r.message for r in caplog.records)
        == 1
    )
    with pytest.raises(StopIteration):
        next(reader)

    skipped = list(
        nuri.readstring("mmcif", data, sanitize=False, skip_on_error=True)
    )
    _validate_3cye_part(skipped[:2])
    _validate_3cye_part(skipped[2:])


@pytest.mark.parametrize("skip_on_error", [False, True])
def test_reader_terminal_syntax_error(test_data: Path, skip_on_error: bool):
    models = (test_data / "3cye_part.cif").read_text()
    reader = nuri.readstring(
        "mmcif",
        models + "data_bad\n_x.y 'unterminated\n" + models,
        sanitize=False,
        skip_on_error=skip_on_error,
    )
    _validate_3cye_part([next(reader), next(reader)])
    if not skip_on_error:
        with pytest.raises(ValueError, match="cannot parse cif block"):
            next(reader)
    with pytest.raises(StopIteration):
        next(reader)
    with pytest.raises(StopIteration):
        next(reader)

#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import pytest

from nuri.fmt import cif_ddl2_frame_as_dict, read_cif


def test_read_cif(test_data: Path):
    cif = test_data / "1a8o.cif"

    blocks = list(read_cif(cif))
    assert len(blocks) == 1

    block = blocks[0]
    assert block.name == "1A8O"
    assert not block.save_frames
    assert not block.is_global

    with pytest.raises(IndexError):
        block.save_frames[0]

    frame = block.data
    assert frame.name == "1A8O"
    assert len(frame) == 37

    for table in frame:
        assert len(table) == 1
        assert table[0][0] == "1A8O"
        break

    assert frame[0][0][0] == "1A8O"

    atom_site = frame.prefix_search_first("_atom_site.")
    assert atom_site is not None

    keys = atom_site.keys()
    assert keys == [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.Cartn_x_esd",
        "_atom_site.Cartn_y_esd",
        "_atom_site.Cartn_z_esd",
        "_atom_site.occupancy_esd",
        "_atom_site.B_iso_or_equiv_esd",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    assert len(atom_site) == 644

    row = atom_site[0]
    assert row[2] == "N"  # _atom_site.type_symbol
    assert row[4] is None  # _atom_site.label_alt_id

    for row in atom_site:
        assert row[1] == "1"  # _atom_site.id
        break

    nonexistent = frame.prefix_search_first("_foobar.")
    assert nonexistent is None


def test_read_cif_temporary(test_data: Path):
    data = next(read_cif(test_data / "1a8o.cif")).data
    assert data.name == "1A8O"


def test_convert_ddl2_cif(test_data: Path):
    cif = test_data / "1a8o.cif"

    blocks = list(read_cif(cif))
    assert len(blocks) == 1
    frame = blocks[0].data

    ddl = cif_ddl2_frame_as_dict(frame)

    assert ddl["entry"][0]["id"] == "1A8O"

    atom_site = ddl["atom_site"]
    assert len(atom_site) == 644

    assert atom_site[0]["type_symbol"] == "N"
    assert atom_site[0]["label_alt_id"] is None

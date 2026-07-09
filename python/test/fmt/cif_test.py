#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import logging
from pathlib import Path

import pytest

from nuri.fmt import (
    CifBlock,
    CifFrame,
    CifTable,
    CifValue,
    cif_ddl2_frame_as_dict,
    read_cif,
    write_cif,
)


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


def _read_one(path: Path):
    return next(read_cif(path))


def test_write_cif_build_roundtrip(tmp_path: Path):
    table = CifTable(
        ["atom.id", "atom.name", "atom.alt"],
        [
            ["1", "atom with space", CifValue(None)],
            ["2", "N", None],
        ],
    )
    entry = CifTable(["entry.id"], [["TEST"]])
    block = CifBlock(CifFrame("test", [entry, table]))

    text = write_cif(block)
    assert "_atom.id" in text  # leading underscore is prepended

    out = tmp_path / "out.cif"
    out.write_text(text)

    back = _read_one(out)
    assert back.name == "test"

    ddl = cif_ddl2_frame_as_dict(back.data)
    assert ddl["entry"][0]["id"] == "TEST"

    atoms = ddl["atom"]
    assert len(atoms) == 2
    assert atoms[0]["name"] == "atom with space"
    assert atoms[0]["alt"] is None  # unknown collapses to None on read
    assert atoms[1]["name"] == "N"
    assert atoms[1]["alt"] is None


def test_write_cif_none_defaults_to_unknown():
    table = CifTable(
        ["a.x", "a.y"],
        [[None, None]],
        formatter_kwargs={"a.y": {"null_token": "."}},
    )
    text = write_cif(CifBlock(CifFrame("d", [table])))
    assert "_a.x ?" in text  # x -> unknown (default)
    assert "_a.y ." in text  # y -> inapplicable

    default = CifTable(["a.x"], [[None]])
    text = write_cif(CifBlock(CifFrame("d", [default])))
    assert "_a.x ?" in text  # None defaults to unknown


def test_write_cif_typed_and_mixed_cells():
    table = CifTable(
        ["v.i", "v.f", "v.b", "v.n"],
        [
            [
                7,
                CifValue(1.5, precision=3),
                True,
                CifValue(None, null_token="."),
            ],
            [CifValue(3, width=4), 2.5, False, CifValue(None)],
        ],
    )
    text = write_cif(CifBlock(CifFrame("d", [table])), align=True)
    assert "7" in text
    assert "1.500" in text
    assert "0003" in text
    assert "yes" in text
    assert "no" in text
    assert "." in text
    assert "?" in text


def test_cif_value_string_vs_raw():
    # A numeric-looking str is quoted (kept a string); raw keeps it a bare
    # generic literal (e.g. a standard-uncertainty token).
    table = CifTable(
        ["v.s", "v.r"],
        [["1.234(5)", CifValue("1.234(5)", raw=True)]],
    )
    text = write_cif(CifBlock(CifFrame("d", [table])))
    assert "_v.s '1.234(5)'" in text
    assert "_v.r 1.234(5)" in text


def test_cif_value_bad_type():
    with pytest.raises(TypeError):
        CifValue(["not", "scalar"])


def test_write_cif_frame(tmp_path: Path):
    frame = CifFrame("f", [CifTable(["a.x"], [["1"]])])
    text = write_cif(frame)
    assert text.startswith("data_f")

    out = tmp_path / "frame.cif"
    out.write_text(text)
    assert _read_one(out).data.name == "f"


def test_cif_block_bad_save_frames():
    data = CifFrame("d", [CifTable(["a.x"], [["1"]])])
    s1 = CifFrame("s", [CifTable(["s.y"], [["2"]])])
    s2 = CifFrame("s", [CifTable(["s.z"], [["3"]])])
    with pytest.raises(ValueError, match="Duplicate save frame"):
        CifBlock(data, [s1, s2])

    unnamed = CifFrame("", [CifTable(["s.y"], [["2"]])])
    with pytest.raises(ValueError, match="empty name"):
        CifBlock(data, [unnamed])


def test_write_cif_1a8o_roundtrip(test_data: Path, tmp_path: Path):
    original = _read_one(test_data / "1a8o.cif")
    text = write_cif(original)

    out = tmp_path / "1a8o_out.cif"
    out.write_text(text)
    reparsed = _read_one(out)

    assert reparsed.name == "1A8O"
    assert cif_ddl2_frame_as_dict(original.data) == cif_ddl2_frame_as_dict(
        reparsed.data
    )


def test_write_cif_align(tmp_path: Path):
    table = CifTable(["t.a", "t.b"], [[1, "x"], [2000, "y"]])
    block = CifBlock(CifFrame("x", [table]))

    plain = write_cif(block)
    aligned = write_cif(block, align=True)
    assert plain != aligned
    assert "1    x" in aligned

    expected = {"t": [{"a": "1", "b": "x"}, {"a": "2000", "b": "y"}]}
    for text in (plain, aligned):
        out = tmp_path / "a.cif"
        out.write_text(text)
        assert cif_ddl2_frame_as_dict(_read_one(out).data) == expected


def test_write_cif_unrepresentable():
    block = CifBlock(CifFrame("x", [CifTable(["a"], [["line\n;bad"]])]))
    with pytest.raises(ValueError, match="';'"):
        write_cif(block)


def test_write_cif_global_from_parser(tmp_path: Path, caplog):
    src = tmp_path / "global.cif"
    src.write_text("global_\n_max_height 6.3\n\ndata_b\n_location here\n")

    block = next(read_cif(src))
    assert block.is_global  # global_ blocks are only produced by the parser

    with caplog.at_level(logging.WARNING, logger="nuri"):
        text = write_cif(block)

    assert text.startswith("global_")
    assert any("STAR" in record.message for record in caplog.records)


def test_cif_table_bad_cell():
    with pytest.raises(TypeError):
        CifTable(["a"], [[["not", "a", "scalar"]]])


def test_cif_table_bad_row_length():
    with pytest.raises(ValueError, match="values but the table has"):
        CifTable(["a", "b"], [["1"]])


def test_cif_table_bad_key():
    # Tables are validated at construction, unlike the C++ core.
    with pytest.raises(ValueError, match="Invalid CIF key"):
        CifTable(["bad key"], [["1"]])
    with pytest.raises(ValueError, match="Invalid CIF key"):
        CifTable([""], [["1"]])


def test_cif_table_duplicate_key():
    with pytest.raises(ValueError, match="Duplicate"):
        CifTable(["a.x", "a.x"], [["1", "2"]])
    with pytest.raises(ValueError, match="Duplicate"):
        CifTable(["x", "x"], [["1", "2"]], category="a")


def test_cif_frame_duplicate_key_across_tables():
    with pytest.raises(ValueError, match="Duplicate"):
        CifFrame("d", [CifTable(["a.x"], [["1"]]), CifTable(["a.x"], [["2"]])])


def test_cif_table_category():
    table = CifTable(["id", "type_symbol"], [["1", "N"]], category="atom_site")
    assert table.keys() == ["_atom_site.id", "_atom_site.type_symbol"]

    text = write_cif(CifBlock(CifFrame("d", [table])))
    assert "_atom_site.id" in text
    assert "_atom_site.type_symbol" in text

    with pytest.raises(ValueError, match="Invalid CIF key"):
        CifTable(["id"], [["1"]], category="bad category")


def test_cif_table_formatter_kwargs():
    table = CifTable(
        ["v.i", "v.f", "v.s", "v.b"],
        [[3, 2.5, "1.234(5)", True]],
        formatter_kwargs={
            "v.i": {"width": 4},
            "v.f": {"precision": 3},
            "v.s": {"raw": True},
            "v.b": {"short_form": True},
        },
    )
    text = write_cif(CifBlock(CifFrame("d", [table])))
    assert "_v.i 0003" in text
    assert "_v.f 2.500" in text
    assert "_v.s 1.234(5)" in text  # raw -> unquoted
    assert "_v.b y" in text  # short form


def test_cif_table_formatter_kwargs_strict():
    with pytest.raises(ValueError, match="Unknown key in formatter_kwargs"):
        CifTable(["a"], [["1"]], formatter_kwargs={"b": {"raw": True}})
    with pytest.raises(ValueError, match="Unknown formatter option"):
        CifTable(["a"], [["1"]], formatter_kwargs={"a": {"bogus": True}})
    with pytest.raises(ValueError, match="null_token"):
        CifTable(["a"], [[None]], formatter_kwargs={"a": {"null_token": "x"}})
    with pytest.raises(ValueError, match="precision must be non-negative"):
        CifTable(["a"], [[1.5]], formatter_kwargs={"a": {"precision": -1}})


def test_cif_value_bad_null_token():
    with pytest.raises(ValueError, match="null_token"):
        CifValue(None, null_token="x")


def test_write_cif_bad_type():
    with pytest.raises(TypeError):
        write_cif("not a cif object")

#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""CIF moved from nuri.fmt into the nuri.fmt.cif submodule; the old top-level
names must keep working."""

from pathlib import Path

import nuri.fmt as fmt
from nuri.fmt import cif


def test_class_aliases_are_identical():
    assert fmt.CifValue is cif.Value
    assert fmt.CifTable is cif.Table
    assert fmt.CifFrame is cif.Frame
    assert fmt.CifBlock is cif.Block


def test_function_aliases_are_identical():
    assert fmt.read_cif is cif.read_blocks
    assert fmt.write_cif is cif.write


def test_old_names_importable():
    from nuri.fmt import (  # noqa: F401
        CifBlock,
        CifFrame,
        CifTable,
        CifValue,
        cif_ddl2_frame_as_dict,
        mmcif_load_frame,
        read_cif,
        write_cif,
    )


def test_method_backed_compat_functions():
    frame = cif.Frame("d", [cif.Table(["a.x"], [["1"]])])

    assert fmt.cif_ddl2_frame_as_dict(frame) == frame.as_ddl2_dict()
    assert len(fmt.mmcif_load_frame(frame)) == len(frame.as_mols())


def test_old_api_round_trips(test_data: Path):
    block = next(fmt.read_cif(test_data / "1a8o.cif"))
    text = fmt.write_cif(block)
    assert text.startswith("data_1A8O")
    assert fmt.cif_ddl2_frame_as_dict(block.data)["entry"][0]["id"] == "1A8O"

#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import pytest

import nuri


def test_fmt_notfound():
    with pytest.raises(ValueError, match="Unknown format"):
        list(nuri.readstring("invalid format", ""))


def test_file_nonexistent(tmp_path: Path):
    file = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError):
        list(nuri.readfile("smi", file))


def test_reader_options():
    mols = list(nuri.readstring("smi", "C"))
    assert len(mols) == 1

    with pytest.raises(ValueError, match="Failed to sanitize"):
        mols = list(nuri.readstring("smi", "C(C)(C)(C)(C)(C)"))

    mols = list(nuri.readstring("smi", "C(C)(C)(C)(C)(C)", sanitize=False))
    assert len(mols) == 1
    assert len(mols[0][0]) == 5

    with pytest.raises(ValueError, match="Failed to parse"):
        mols = list(nuri.readstring("smi", "error"))

    mols = list(nuri.readstring("smi", "error", skip_on_error=True))
    assert len(mols) == 0

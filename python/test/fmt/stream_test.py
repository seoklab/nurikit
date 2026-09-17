#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import gzip
import io
import typing
from pathlib import Path

import pytest

import nuri
from nuri.core import Molecule

smi_data = """\
C
CC
CCC propane
C1=CC=CC=C1 benzene
"""

sdf_data = "".join(
    f"""\
mol{i}
  test
comment
  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
    for i in range(3)
)

mol2_data = """\
@<TRIPOS>MOLECULE
Methane
 1 0 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
 1 C1            -0.0000    -0.0000    -0.0000 C.3
@<TRIPOS>MOLECULE
Water
 1 0 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
 1 O1             1.0000     0.0000     0.0000 O.3
"""

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
ENDMDL
CONECT   19   20
END
"""

cif_data = """\
data_batch
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
1 C CA ALA A 1 1 2 3 1
2 N N ALA A 1 4 5 6 1
3 C CA ALA A 1 1 2 3 2
4 N N ALA A 1 4 5 6 2
"""

FORMATS = {
    "smi": (smi_data, True),
    "sdf": (sdf_data, True),
    "mol2": (mol2_data, True),
    "pdb": (pdb_data, True),
    "mmcif": (cif_data, False),
}


def _signature(mol: Molecule):
    return (
        mol.name,
        mol.props.get("model", ""),
        tuple(atom.atomic_number for atom in mol),
        mol.num_bonds(),
    )


def _expected(fmt: str):
    text, sanitize = FORMATS[fmt]
    return [
        _signature(m) for m in nuri.readstring(fmt, text, sanitize=sanitize)
    ]


def _read(fmt: str, stream: typing.IO):
    sanitize = FORMATS[fmt][1]
    return [
        _signature(m) for m in nuri.readstream(fmt, stream, sanitize=sanitize)
    ]


def _write(
    tmp_path: Path,
    text: str,
    opener: typing.Callable[..., typing.IO] = open,
):
    path = tmp_path / "data"
    with opener(path, "wt") as f:
        f.write(text)
    return path


SOURCES = {
    "stringio": lambda _, text: (io.StringIO(text), True),
    "bytesio": lambda _, text: (io.BytesIO(text.encode()), False),
    "text_file": lambda d, text: (open(_write(d, text)), True),
    "binary_file": lambda d, text: (open(_write(d, text), "rb"), False),
    "gzip_rb": lambda d, text: (gzip.open(_write(d, text, gzip.open)), False),
    "gzip_rt": lambda d, text: (
        gzip.open(_write(d, text, gzip.open), "rt"),
        True,
    ),
}


@pytest.fixture(params=sorted(SOURCES))
def make_stream(request: pytest.FixtureRequest, tmp_path: Path):
    def make(text: str):
        return SOURCES[request.param](tmp_path, text)

    return make


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_matches_readstring(
    fmt: str,
    make_stream: typing.Callable[[str], tuple[typing.IO, bool]],
):
    expected = _expected(fmt)
    assert expected

    stream, is_text = make_stream(FORMATS[fmt][0])
    with stream:
        if fmt == "pdb" and is_text:
            with pytest.raises(OSError, match="seek"):
                _read(fmt, stream)
        else:
            assert _read(fmt, stream) == expected


class _Unseekable(io.RawIOBase):
    def __init__(self, data: bytes):
        super().__init__()
        self._buf = io.BytesIO(data)

    def readable(self):
        return True

    def readinto(self, b):
        return self._buf.readinto(b)


def _unseekable(text: str):
    return io.BufferedReader(_Unseekable(text.encode()))


def test_unseekable_binary():
    assert _read("smi", _unseekable(smi_data)) == _expected("smi")

    with pytest.raises(OSError, match="seek"):
        _read("pdb", _unseekable(pdb_data))


class _TrickleBytes(io.BytesIO):
    def read(self, size: int | None = -1):
        if size is None or size < 0 or size > 7:
            size = 7
        return super().read(size)


class _TrickleText(io.StringIO):
    def read(self, size: int | None = -1):
        if size is None or size < 0 or size > 1:
            size = 1
        return super().read(size)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_small_chunks(fmt):
    text = FORMATS[fmt][0]
    expected = _expected(fmt)
    assert _read(fmt, _TrickleBytes(text.encode())) == expected
    if fmt != "pdb":
        assert _read(fmt, _TrickleText(text)) == expected


class _FailingRead:
    def __init__(self, data: bytes, chunk: int, fail_after: int):
        self._buf = io.BytesIO(data)
        self._chunk = chunk
        self._remaining = fail_after

    def read(self, size):
        if self._remaining == 0:
            raise RuntimeError("boom")
        self._remaining -= 1
        return self._buf.read(min(size, self._chunk))


def test_read_error_propagates():
    reader = nuri.readstream(
        "smi", _FailingRead(smi_data.encode(), chunk=8, fail_after=1)
    )
    assert len(next(reader)) == 1
    assert len(next(reader)) == 2
    with pytest.raises(RuntimeError, match="boom"):
        next(reader)
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(reader)


def test_read_error_ignores_skip_on_error():
    reader = nuri.readstream(
        "smi",
        _FailingRead(smi_data.encode(), chunk=8, fail_after=0),
        skip_on_error=True,
    )
    with pytest.raises(RuntimeError, match="boom"):
        next(reader)


class _ReadReturns:
    def __init__(self, value):
        self._value = value

    def read(self, size):
        value, self._value = self._value, None
        return value


def test_read_return_types():
    with pytest.raises(TypeError, match="bytes-like or str"):
        next(nuri.readstream("smi", _ReadReturns(42)))

    assert _read("smi", _ReadReturns(None)) == []
    assert _read("smi", _ReadReturns(bytearray(b"CC\n"))) == _read(
        "smi", io.BytesIO(b"CC\n")
    )
    assert _read("smi", _ReadReturns(memoryview(b"CC\n"))) == _read(
        "smi", io.BytesIO(b"CC\n")
    )


def test_unencodable_text():
    with pytest.raises(UnicodeEncodeError):
        next(nuri.readstream("smi", _ReadReturns("C\udc80\n")))


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_readstring_bytes_like(fmt: str, wrap):
    text, sanitize = FORMATS[fmt]
    mols = nuri.readstring(fmt, wrap(text.encode()), sanitize=sanitize)
    assert [_signature(m) for m in mols] == _expected(fmt)


def test_readstring_rejects_non_text():
    with pytest.raises(TypeError, match="bytes-like or str"):
        nuri.readstring("smi", 42)


def test_readstring_unencodable_text():
    with pytest.raises(UnicodeEncodeError):
        nuri.readstring("smi", "C\udc80\n")


def test_readstring_copies_mutable_buffer():
    buf = bytearray(smi_data.encode())
    mols = nuri.readstring("smi", buf)
    buf[:] = b"\0" * len(buf)
    assert [_signature(m) for m in mols] == _expected("smi")


def test_readstring_large_matches_readfile(tmp_path: Path):
    text = sdf_data * 2000
    path = _write(tmp_path, text)
    expected = [_signature(m) for m in nuri.readfile("sdf", path)]
    assert [_signature(m) for m in nuri.readstring("sdf", text)] == expected
    assert [
        _signature(m) for m in nuri.readstring("sdf", text.encode())
    ] == expected


class _SeekReturnsNone:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self, size):
        return self._buf.read(size)

    def seek(self, offset, whence=0):
        self._buf.seek(offset, whence)

    def tell(self):
        return self._buf.tell()


def test_seek_without_return_value():
    assert _read("pdb", _SeekReturnsNone(pdb_data.encode())) == _expected(
        "pdb"
    )


class _HugeTell(_SeekReturnsNone):
    def tell(self):
        return 1 << 70


def test_tell_overflow():
    with pytest.raises(OverflowError):
        _read("pdb", _HugeTell(pdb_data.encode()))


def test_not_readable():
    with pytest.raises(TypeError, match="readable"):
        nuri.readstream("smi", object())


def test_unknown_format():
    with pytest.raises(ValueError, match="Unknown format"):
        nuri.readstream("nope", io.BytesIO(b""))

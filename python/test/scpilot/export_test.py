#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import io

import numpy as np

from nuri.fmt import readstring
from nuri.scpilot.export import PATCH_NAMES, write_dots
from nuri.scpilot.io import load_structure
from nuri.scpilot.surface import Dots, Patch, ses_dots


def parse_mol2(text):
    head, rest = text.split("@<TRIPOS>ATOM\n")
    body, tail = rest.split("@<TRIPOS>BOND\n")
    counts = head.splitlines()[2].split()
    rows = [line.split() for line in body.splitlines()]
    return counts, rows, tail


def dump(dots, **kwargs):
    out = io.StringIO()
    write_dots(out, dots, **kwargs)
    return parse_mol2(out.getvalue())


def three_spheres():
    coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 2.2, 0.0]])
    radii = np.array([1.7, 1.6, 1.8])
    return ses_dots(coords, radii, rp=1.4, density=30)


def test_layout_matches_dots():
    dots = three_spheres()
    counts, rows, tail = dump(dots, name="probe")

    assert counts[:2] == [str(len(dots)), "0"]
    assert tail == ""
    assert len(rows) == len(dots)

    serial = np.array([int(r[0]) for r in rows])
    np.testing.assert_array_equal(serial, np.arange(1, len(dots) + 1))

    pts = np.array([[float(x) for x in r[2:5]] for r in rows])
    np.testing.assert_allclose(pts, dots.pts, atol=5e-5)

    assert {r[5] for r in rows} == {"H"}
    assert {r[1] for r in rows} == {"DX"}

    names = np.array([r[7] for r in rows])
    for kind in Patch:
        mask = dots.kinds == kind
        assert set(names[mask]) == {PATCH_NAMES[kind]}

    subst = np.array([int(r[6]) for r in rows])
    np.testing.assert_array_equal(subst, 3 * dots.atoms + dots.kinds + 1)
    assert int(counts[2]) == len(np.unique(subst))

    charge = np.array([float(r[8]) for r in rows])
    np.testing.assert_allclose(charge, dots.areas, atol=5e-5)


def test_scalar_and_buried_round_trip():
    dots = three_spheres()
    scalar = np.linspace(-0.9, 0.9, len(dots))
    buried = dots.pts[:, 0] > 1.25

    _, rows, _ = dump(dots, scalar=scalar, buried=buried)

    flags = np.array([r[1] == "DB" for r in rows])
    np.testing.assert_array_equal(flags, buried)
    charge = np.array([float(r[8]) for r in rows])
    np.testing.assert_allclose(charge, scalar, atol=5e-5)


def test_scalar_broadcasts():
    dots = three_spheres()
    _, rows, _ = dump(dots, scalar=1.5, buried=True)
    assert {r[1] for r in rows} == {"DB"}
    assert {float(r[8]) for r in rows} == {1.5}


def test_empty_dots():
    empty = Dots(
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty(0),
        np.empty(0, dtype=int),
        np.empty(0, dtype=np.int8),
        1.4,
    )
    counts, rows, tail = dump(empty)
    assert counts[:3] == ["0", "0", "0"]
    assert rows == []
    assert tail == ""


def test_readable_by_nuri(tmp_path, test_data):
    st = load_structure(test_data / "1ar1.pdb").chains("H")
    st = st.subset(st.residue_seqs <= 12)
    dots = ses_dots(st.coords, st.vdw_radii, rp=1.7, density=15)

    path = tmp_path / "frag.mol2"
    with path.open("w") as f:
        write_dots(f, dots, path.stem)

    mol = next(iter(readstring("mol2", path.read_text())))
    assert mol.num_atoms() == len(dots)
    assert mol.num_bonds() == 0
    assert all(atom.atomic_number == 1 for atom in mol)
    np.testing.assert_allclose(mol.get_conf(0), dots.pts, atol=5e-5)

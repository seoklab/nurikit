#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""United-atom radii for molecular-surface calculations.

Base values follow the "Biosym MS" united-atom set for molecular surface
calculation (CH3 1.95, CH2 1.90, aromatic CH 1.90, C 1.80; amide N 1.65,
amine NH2 1.70, NH3+ 1.75; carbonyl/carboxyl O 1.60, hydroxyl O 1.70; S 1.90;
P 1.80). Radii are derived from each heavy atom's implicit hydrogen count,
which for standard residues comes from the residue templates below.
"""

from __future__ import annotations

import numpy as np

from .io import Structure

BACKBONE = {"N": 1, "CA": 1, "C": 0, "O": 0, "OXT": 0}

SIDECHAIN_H: dict[str, dict[str, int]] = {
    "ALA": {"CB": 3},
    "ARG": {"CB": 2, "CG": 2, "CD": 2, "NE": 1, "CZ": 0, "NH1": 2, "NH2": 2},
    "ASN": {"CB": 2, "CG": 0, "OD1": 0, "ND2": 2},
    "ASP": {"CB": 2, "CG": 0, "OD1": 0, "OD2": 0},
    "CYS": {"CB": 2, "SG": 1},
    "GLN": {"CB": 2, "CG": 2, "CD": 0, "OE1": 0, "NE2": 2},
    "GLU": {"CB": 2, "CG": 2, "CD": 0, "OE1": 0, "OE2": 0},
    "GLY": {},
    "HIS": {"CB": 2, "CG": 0, "ND1": 1, "CD2": 1, "CE1": 1, "NE2": 0},
    "ILE": {"CB": 1, "CG1": 2, "CG2": 3, "CD1": 3},
    "LEU": {"CB": 2, "CG": 1, "CD1": 3, "CD2": 3},
    "LYS": {"CB": 2, "CG": 2, "CD": 2, "CE": 2, "NZ": 3},
    "MET": {"CB": 2, "CG": 2, "SD": 0, "CE": 3},
    "PHE": {"CB": 2, "CG": 0, "CD1": 1, "CD2": 1, "CE1": 1, "CE2": 1, "CZ": 1},
    "PRO": {"CB": 2, "CG": 2, "CD": 2},
    "SER": {"CB": 2, "OG": 1},
    "THR": {"CB": 1, "OG1": 1, "CG2": 3},
    "TRP": {
        "CB": 2,
        "CG": 0,
        "CD1": 1,
        "CD2": 0,
        "NE1": 1,
        "CE2": 0,
        "CE3": 1,
        "CZ2": 1,
        "CZ3": 1,
        "CH2": 1,
    },
    "TYR": {
        "CB": 2,
        "CG": 0,
        "CD1": 1,
        "CD2": 1,
        "CE1": 1,
        "CE2": 1,
        "CZ": 0,
        "OH": 1,
    },
    "VAL": {"CB": 1, "CG1": 3, "CG2": 3},
}
SIDECHAIN_H["MSE"] = {"CB": 2, "CG": 2, "SE": 0, "CE": 3}

AROMATIC_CH = {
    "HIS": {"CD2", "CE1"},
    "PHE": {"CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CD1", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CD1", "CD2", "CE1", "CE2"},
}

BASE = {"C": 1.80, "N": 1.65, "O": 1.60, "S": 1.90, "SE": 1.90, "P": 1.80}


def template_hydrogens(resname: str, atom: str) -> int | None:
    side = SIDECHAIN_H.get(resname)
    if side is None:
        return None
    if atom in side:
        return side[atom]
    if atom in BACKBONE:
        if atom == "N" and resname == "PRO":
            return 0
        if atom == "CA" and resname == "GLY":
            return 2
        return BACKBONE[atom]
    return None


def united_atom_radius(
    element: str, n_h: int, aromatic_ch: bool = False
) -> float | None:
    if element == "C":
        if n_h == 1 and aromatic_ch:
            return 1.90
        return 1.80 + 0.05 * n_h
    if element == "N":
        return 1.65 + 0.05 * max(n_h - 1, 0)
    if element == "O":
        return 1.60 if n_h == 0 else 1.70
    return BASE.get(element)


_SYMBOLS = {6: "C", 7: "N", 8: "O", 15: "P", 16: "S", 34: "SE"}


def united_atom_radii(struct: Structure) -> tuple[np.ndarray, np.ndarray]:
    """Radii for ``struct`` with a mask of atoms that fell back to element
    van der Waals radii (unknown residue or atom name)."""
    radii = struct.vdw_radii.copy()
    fallback = np.zeros(len(struct), dtype=bool)
    for i, (res, atom, z) in enumerate(
        zip(struct.residue_names, struct.atom_names, struct.atomic_numbers)
    ):
        n_h = template_hydrogens(res, atom)
        sym = _SYMBOLS.get(int(z))
        r = None
        if n_h is not None and sym is not None:
            r = united_atom_radius(sym, n_h, atom in AROMATIC_CH.get(res, ()))
        if r is None:
            fallback[i] = True
        else:
            radii[i] = r
    return radii, fallback

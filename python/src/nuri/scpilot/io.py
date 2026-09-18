#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nuri.fmt import pdb


@dataclass(frozen=True)
class Structure:
    coords: np.ndarray
    atomic_numbers: np.ndarray
    vdw_radii: np.ndarray
    atom_names: np.ndarray
    residue_names: np.ndarray
    residue_seqs: np.ndarray
    chain_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.coords)

    def subset(self, mask: np.ndarray) -> Structure:
        return Structure(
            self.coords[mask],
            self.atomic_numbers[mask],
            self.vdw_radii[mask],
            self.atom_names[mask],
            self.residue_names[mask],
            self.residue_seqs[mask],
            self.chain_ids[mask],
        )

    def chains(self, ids: str) -> Structure:
        return self.subset(np.isin(self.chain_ids, list(ids)))


def load_structure(
    path: str | Path,
    model: int = 0,
    heavy_only: bool = True,
    skip_hetero: bool = True,
) -> Structure:
    mdl = pdb.read_models(str(path))[model]
    atoms = mdl.atoms
    coords = np.asarray(mdl.major_conf, dtype=float)

    atomic_numbers = np.fromiter(
        (a.element.atomic_number for a in atoms), dtype=int, count=len(atoms)
    )
    vdw_radii = np.fromiter(
        (a.element.vdw_radius for a in atoms), dtype=float, count=len(atoms)
    )
    atom_names = np.array([a.name for a in atoms])
    hetero = np.fromiter(
        (a.hetero for a in atoms), dtype=bool, count=len(atoms)
    )
    residue_names = np.empty(len(atoms), dtype=object)
    for res in mdl.residues:
        residue_names[np.asarray(res.atom_idxs)] = res.name
    residue_seqs = np.fromiter(
        (a.res_id.res_seq for a in atoms), dtype=int, count=len(atoms)
    )
    chain_ids = np.array([a.res_id.chain_id for a in atoms])

    keep = np.ones(len(atoms), dtype=bool)
    if heavy_only:
        keep &= atomic_numbers > 1
    if skip_hetero:
        keep &= ~hetero

    return Structure(
        coords,
        atomic_numbers,
        vdw_radii,
        atom_names,
        residue_names.astype(str),
        residue_seqs,
        chain_ids,
    ).subset(keep)

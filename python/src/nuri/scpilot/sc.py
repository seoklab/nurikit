#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Shape complementarity statistic of Lawrence & Colman (1993)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .surface import Dots, ses_dots


@dataclass(frozen=True)
class ScParams:
    rp: float = 1.7
    density: float = 15.0
    weight: float = 0.5
    band: float = 1.5
    sep: float = 8.0
    clamp: float = 0.999


@dataclass
class SideResult:
    n_atoms: int
    n_active: int
    n_dots: int
    n_buried: int
    n_trimmed: int
    trimmed_area: float
    d_median: float
    s_median: float


@dataclass
class ScResult:
    sc: float
    distance: float
    area: float
    sides: tuple[SideResult, SideResult]


def active_atoms(coords, tree_other: cKDTree, sep: float) -> np.ndarray:
    """Atoms within ``sep`` of any atom of the other side."""
    d, _ = tree_other.query(coords, k=1)
    return d < sep


def buried_mask(dots: Dots, tree_other: cKDTree, sas_other) -> np.ndarray:
    """Dots whose probe centre overlaps a SAS ball of the other side."""
    sdm = cKDTree(dots.probes).sparse_distance_matrix(
        tree_other, float(np.max(sas_other)), output_type="coo_matrix"
    )
    inside = sdm.data < sas_other[sdm.col]
    buried = np.zeros(len(dots), dtype=bool)
    buried[sdm.row[inside]] = True
    return buried


def trim_peripheral(dots: Dots, buried: np.ndarray, band: float):
    """Buried dots farther than ``band`` from every accessible dot."""
    exposed = dots.pts[~buried]
    keep = buried.copy()
    if len(exposed) == 0 or not buried.any():
        return keep
    n_close = cKDTree(exposed).query_ball_point(
        dots.pts[buried], band, return_length=True
    )
    keep[np.flatnonzero(buried)[n_close > 0]] = False
    return keep


def pair_statistics(mine: Dots, theirs: Dots, weight: float, clamp: float):
    d, idx = cKDTree(theirs.pts).query(mine.pts, k=1)
    dot = np.einsum("ij,ij->i", mine.normals, theirs.normals[idx])
    s = -dot * np.exp(-weight * d * d)
    s = np.clip(s, -clamp, clamp)
    return s, d


@dataclass
class Side:
    n_atoms: int
    active: np.ndarray
    dots: Dots
    buried: np.ndarray
    trimmed: Dots

    @classmethod
    def build(cls, coords, radii, other_coords, other_radii, params):
        coords = np.asarray(coords, dtype=float)
        radii = np.asarray(radii, dtype=float)
        tree_other = cKDTree(other_coords)
        active = active_atoms(coords, tree_other, params.sep)
        dots = ses_dots(coords, radii, params.rp, params.density, active)
        buried = buried_mask(
            dots, tree_other, np.asarray(other_radii, dtype=float) + params.rp
        )
        keep = trim_peripheral(dots, buried, params.band)
        return cls(len(coords), active, dots, buried, dots.subset(keep))

    def result(self, other: Side, params: ScParams) -> SideResult:
        s, d = pair_statistics(
            self.trimmed, other.trimmed, params.weight, params.clamp
        )
        return SideResult(
            n_atoms=self.n_atoms,
            n_active=int(self.active.sum()),
            n_dots=len(self.dots),
            n_buried=int(self.buried.sum()),
            n_trimmed=len(self.trimmed),
            trimmed_area=self.trimmed.area(),
            d_median=float(np.median(d)),
            s_median=float(np.median(s)),
        )


def build_sides(
    coords_a,
    radii_a,
    coords_b,
    radii_b,
    params: ScParams = ScParams(),
) -> tuple[Side, Side]:
    return (
        Side.build(coords_a, radii_a, coords_b, radii_b, params),
        Side.build(coords_b, radii_b, coords_a, radii_a, params),
    )


def shape_complementarity(
    coords_a,
    radii_a,
    coords_b,
    radii_b,
    params: ScParams = ScParams(),
) -> ScResult:
    side_a, side_b = build_sides(coords_a, radii_a, coords_b, radii_b, params)
    if len(side_a.trimmed) == 0 or len(side_b.trimmed) == 0:
        raise ValueError("no interface dots survive trimming")
    a, b = side_a.result(side_b, params), side_b.result(side_a, params)
    return ScResult(
        sc=0.5 * (a.s_median + b.s_median),
        distance=0.5 * (a.d_median + b.d_median),
        area=a.trimmed_area + b.trimmed_area,
        sides=(a, b),
    )

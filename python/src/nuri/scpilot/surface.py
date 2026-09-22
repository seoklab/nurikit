#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Solvent-excluded surface (SES) dot sampler.

Dots are placed inside the exact patch domains of the analytic SES
(:mod:`nuri.scpilot.anal`): convex patches on atom spheres, toroidal
saddles along accessible probe arcs, and concave faces on probe spheres.
Dot weights are normalised so that every patch's sampled area equals its
analytic area; dot counts follow the target density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .anal import Circle, SasGeometry, SesGeometry, TorusArc
from .arrangement import contains


class Patch(IntEnum):
    CONVEX = 0
    TOROIDAL = 1
    CONCAVE = 2


@dataclass
class Dots:
    pts: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    atoms: np.ndarray
    kinds: np.ndarray
    rp: float
    dropped_area: float = 0.0

    def __len__(self) -> int:
        return len(self.pts)

    @property
    def probes(self) -> np.ndarray:
        return self.pts + self.rp * self.normals

    def subset(self, mask: np.ndarray) -> Dots:
        return Dots(
            self.pts[mask],
            self.normals[mask],
            self.areas[mask],
            self.atoms[mask],
            self.kinds[mask],
            self.rp,
        )

    def area(self, kind: Patch | None = None) -> float:
        if kind is None:
            return float(self.areas.sum())
        return float(self.areas[self.kinds == kind].sum())


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    z = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = golden * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


class _DotBuffer:
    def __init__(self) -> None:
        self.pts: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.areas: list[np.ndarray] = []
        self.atoms: list[np.ndarray] = []
        self.kinds: list[np.ndarray] = []
        self.dropped = 0.0

    def add(self, pts, normals, areas, atoms, kind: Patch) -> None:
        if len(pts) == 0:
            return
        self.pts.append(pts)
        self.normals.append(normals)
        self.areas.append(np.broadcast_to(areas, (len(pts),)))
        self.atoms.append(np.broadcast_to(atoms, (len(pts),)))
        self.kinds.append(np.full(len(pts), int(kind), dtype=np.int8))

    def finish(self, rp: float) -> Dots:
        if not self.pts:
            return Dots(
                np.empty((0, 3)),
                np.empty((0, 3)),
                np.empty(0),
                np.empty(0, dtype=int),
                np.empty(0, dtype=np.int8),
                rp,
                self.dropped,
            )
        return Dots(
            np.concatenate(self.pts),
            np.concatenate(self.normals),
            np.concatenate(self.areas).astype(float),
            np.concatenate(self.atoms).astype(int),
            np.concatenate(self.kinds),
            rp,
            self.dropped,
        )


def _lattice_count(area: float, density: float) -> int:
    return max(round(area * density), 1)


def _convex(ses: SesGeometry, density: float, out: _DotBuffer) -> None:
    sas = ses.sas
    lattices: dict[int, np.ndarray] = {}
    for i in np.flatnonzero(ses.convex_area > 0.0):
        area = ses.convex_area[i]
        r = sas.radii[i]
        count = _lattice_count(4.0 * math.pi * r * r, density)
        if count not in lattices:
            lattices[count] = fibonacci_sphere(count)
        dirs = lattices[count]
        dirs = dirs[~sas.arrangements[i].contains(dirs)]
        out.dropped += area * (len(dirs) == 0)
        out.add(
            sas.coords[i] + r * dirs,
            dirs,
            area / max(len(dirs), 1),
            i,
            Patch.CONVEX,
        )


@dataclass
class SaddleRow:
    """One ``beta`` range of one active arc to sample: the arc, its circle,
    the range ``[lo, hi]`` and the area integral of the range."""

    arc: TorusArc
    circle: Circle
    lo: float
    hi: float
    integral: float


def _saddle_rows(ses: SesGeometry) -> list[SaddleRow]:
    """One row per active arc and cusp side. A circle with ``rl >= rp`` has
    no cusp, so its two sides are one arc of ``beta`` and make one merged
    row; a side that is absent has zero width and emits nothing."""
    sas = ses.sas
    rows = []
    for arc in sas.arcs[: sas.n_active_arcs]:
        circle = sas.circles[arc.circle]
        saddle = ses.saddles[arc.circle]
        (lo0, hi0), (lo1, hi1) = saddle.ranges
        int0, int1 = saddle.integral
        if circle.rl >= sas.rp:
            rows.append(SaddleRow(arc, circle, lo0, hi1, int0 + int1))
        else:
            rows.append(SaddleRow(arc, circle, lo0, hi0, int0))
            rows.append(SaddleRow(arc, circle, lo1, hi1, int1))
    return rows


def _toroidal(ses: SesGeometry, density: float, out: _DotBuffer) -> None:
    """Rings of equal ``beta`` width along every saddle row; each ring gets
    ``round(area * density)`` dots spread uniformly in ``phi``, alternate
    rings offset by half a step. A row whose rings all round to zero is
    sampled as one ring; rows that still round to zero are dropped."""
    sas = ses.sas
    rp = sas.rp
    for row in _saddle_rows(ses):
        arc, circle = row.arc, row.circle
        width = row.hi - row.lo
        total = rp * arc.dphi * row.integral

        k_beta = max(round(rp * width * math.sqrt(density)), 1)
        dbeta = width / k_beta
        sin_edge = np.sin(row.lo + np.arange(k_beta + 1) * dbeta)
        beta = row.lo + (np.arange(k_beta) + 0.5) * dbeta
        area = (
            rp
            * arc.dphi
            * (circle.rl * dbeta - rp * (sin_edge[1:] - sin_edge[:-1]))
        )
        k_phi = np.round(area * density).astype(int)
        if not np.any(k_phi > 0):
            beta = np.array([0.5 * (row.lo + row.hi)])
            area = np.array([total])
            k_phi = np.array([round(total * density)])
        keep = np.flatnonzero(k_phi > 0)
        if len(keep) == 0:
            out.dropped += total
            continue
        beta, area, k_phi = beta[keep], area[keep], k_phi[keep]
        area *= total / area.sum()

        cos_b, sin_b = np.cos(beta), np.sin(beta)
        i, j = circle.i, circle.j
        a_i, a_j = circle.a, circle.d - circle.a
        depth_i = (
            np.sqrt(
                sas.radii[i] ** 2
                + 2.0 * rp * (sas.sas[i] - circle.rl * cos_b + a_i * sin_b)
            )
            - sas.radii[i]
        )
        depth_j = (
            np.sqrt(
                sas.radii[j] ** 2
                + 2.0 * rp * (sas.sas[j] - circle.rl * cos_b - a_j * sin_b)
            )
            - sas.radii[j]
        )
        owner = np.where(depth_i <= depth_j, i, j)

        for ring, m in enumerate(keep):
            n = np.arange(k_phi[ring])
            offset = (0.25 + 0.5 * (m % 2)) / k_phi[ring]
            phi = arc.phi_beg + (n / k_phi[ring] + offset) * arc.dphi
            radial = np.cos(phi)[:, None] * circle.e1
            radial += np.sin(phi)[:, None] * circle.e2
            q = circle.centre + circle.rl * radial
            inward = -cos_b[ring] * radial + sin_b[ring] * circle.axis
            out.add(
                q + rp * inward,
                -inward,
                area[ring] / k_phi[ring],
                owner[ring],
                Patch.TOROIDAL,
            )


def _concave(ses: SesGeometry, density: float, out: _DotBuffer) -> None:
    sas = ses.sas
    rp = sas.rp
    lattice = fibonacci_sphere(
        _lattice_count(4.0 * math.pi * rp * rp, density)
    )
    for face in (f for f in ses.concave if f.area > 0.0):
        dirs = lattice[~contains(face.caps, lattice)]
        out.dropped += face.area * (len(dirs) == 0)
        big, small = sas.sas[face.atoms], sas.radii[face.atoms]
        depth = (
            np.sqrt(
                small * small + 2.0 * rp * big * (1.0 - dirs @ face.contacts.T)
            )
            - small
        )
        out.add(
            sas.probes[face.probe].pos + rp * dirs,
            -dirs,
            face.area / max(len(dirs), 1),
            face.atoms[np.argmin(depth, axis=1)],
            Patch.CONCAVE,
        )


def sample_ses(ses: SesGeometry, density: float) -> Dots:
    out = _DotBuffer()
    _convex(ses, density, out)
    _toroidal(ses, density, out)
    _concave(ses, density, out)
    return out.finish(ses.rp)


def ses_dots(
    coords: np.ndarray,
    radii: np.ndarray,
    rp: float = 1.7,
    density: float = 15.0,
    active: np.ndarray | None = None,
) -> Dots:
    """Sample the SES of the spheres ``(coords, radii)`` with roughly
    ``density`` dots per square angstrom.

    Atoms with ``active`` false still occlude the probe but never own a
    convex patch; toroidal and concave patches are generated when at least
    one participating atom is active.
    """
    sas, order = SasGeometry.from_atoms(coords, radii, rp, active)
    dots = sample_ses(SesGeometry.build(sas), density)
    dots.atoms = order[dots.atoms]
    return dots

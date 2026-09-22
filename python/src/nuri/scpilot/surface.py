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

from .anal import SasGeometry, SesGeometry
from .aos import scalars, vectors
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


def _segments(counts: np.ndarray):
    """Expand per-segment ``counts`` into ``(segment id, index within
    segment)`` for every element, plus the start of every segment."""
    seg = np.repeat(np.arange(len(counts)), counts)
    start = np.cumsum(counts) - counts
    return seg, np.arange(len(seg)) - start[seg], start


def _saddle_rows(ses: SesGeometry):
    """One ``(arc, range)`` row per active arc and cusp side: the ``beta``
    range, its area integral, and the arc's circle. Where there is no cusp
    both sides are one arc of ``beta`` and are merged into the first row,
    leaving the second zero-width."""
    sas = ses.sas
    n_arcs = sas.n_active_arcs
    arc = np.repeat(np.arange(n_arcs), 2)
    part = np.tile([0, 1], n_arcs)
    c = scalars(sas.arcs[:n_arcs], "circle", int)[arc]
    ranges = np.array([s.ranges for s in ses.saddles]).reshape(-1, 2, 2)
    integrals = np.array([s.integral for s in ses.saddles]).reshape(-1, 2)
    lo, hi = ranges[c, part].T.copy()
    integral = integrals[c, part].copy()
    whole = scalars(sas.circles, "rl")[c] >= sas.rp
    first, second = whole & (part == 0), whole & (part == 1)
    hi[first] = hi[second]
    integral[first] += integral[second]
    lo[second] = hi[second]
    integral[second] = 0.0
    return arc, c, lo, hi, integral


def _toroidal(ses: SesGeometry, density: float, out: _DotBuffer) -> None:
    """Rings of equal ``beta`` width along every saddle row; each ring gets
    ``round(area * density)`` dots spread uniformly in ``phi``, alternate
    rings offset by half a step. A row whose rings all round to zero is
    sampled as one ring; rows that still round to zero are dropped."""
    sas = ses.sas
    rp = sas.rp
    circles, arcs = sas.circles, sas.arcs
    c_rl, c_a, c_d = (
        scalars(circles, "rl"),
        scalars(circles, "a"),
        scalars(circles, "d"),
    )
    c_i, c_j = scalars(circles, "i", int), scalars(circles, "j", int)
    c_centre, c_axis = vectors(circles, "centre"), vectors(circles, "axis")
    c_e1, c_e2 = vectors(circles, "e1"), vectors(circles, "e2")
    a_dphi, a_phi_beg = scalars(arcs, "dphi"), scalars(arcs, "phi_beg")
    arc, c, lo, hi, integral = _saddle_rows(ses)
    rl, dphi = c_rl[c], a_dphi[arc]
    width = hi - lo
    total = rp * dphi * integral

    k_beta = np.maximum(np.round(rp * width * math.sqrt(density)), 1).astype(
        int
    )
    row, m, first = _segments(k_beta)
    dbeta = width / k_beta
    erow, em, estart = _segments(k_beta + 1)
    sin_edge = np.sin(lo[erow] + em * dbeta[erow])
    edge = estart[row] + m
    beta = lo[row] + (m + 0.5) * dbeta[row]
    area = (
        rp
        * dphi[row]
        * (rl[row] * dbeta[row] - rp * (sin_edge[edge + 1] - sin_edge[edge]))
    )
    k_phi = np.round(area * density).astype(int)

    starved = np.bincount(row, k_phi > 0, minlength=len(arc)) == 0
    collapse = first[starved]
    beta[collapse] = 0.5 * (lo + hi)[starved]
    area[collapse] = total[starved]
    k_phi[collapse] = np.round(total[starved] * density).astype(int)

    keep = k_phi > 0
    out.dropped += float(
        total[np.bincount(row, keep, minlength=len(arc)) == 0].sum()
    )
    row, beta, area, k_phi, m = (
        row[keep],
        beta[keep],
        area[keep],
        k_phi[keep],
        m[keep],
    )
    covered = np.bincount(row, area, minlength=len(arc))
    area *= total[row] / covered[row]

    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cr = c[row]
    i, j = c_i[cr], c_j[cr]
    a_i, a_j, rl_r = c_a[cr], c_d[cr] - c_a[cr], rl[row]
    depth_i = (
        np.sqrt(
            sas.radii[i] ** 2
            + 2.0 * rp * (sas.sas[i] - rl_r * cos_b + a_i * sin_b)
        )
        - sas.radii[i]
    )
    depth_j = (
        np.sqrt(
            sas.radii[j] ** 2
            + 2.0 * rp * (sas.sas[j] - rl_r * cos_b - a_j * sin_b)
        )
        - sas.radii[j]
    )
    owner = np.where(depth_i <= depth_j, i, j)

    ring, n_in_ring, _ = _segments(k_phi)
    r = row[ring]
    cc = c[r]
    offset = (0.25 + 0.5 * (m[ring] % 2)) / k_phi[ring]
    phi = a_phi_beg[arc[r]] + (n_in_ring / k_phi[ring] + offset) * dphi[r]
    radial = np.cos(phi)[:, None] * c_e1[cc] + np.sin(phi)[:, None] * c_e2[cc]
    q = c_centre[cc] + rl[r, None] * radial
    inward = -cos_b[ring, None] * radial + sin_b[ring, None] * c_axis[cc]
    out.add(
        q + rp * inward,
        -inward,
        (area / k_phi)[ring],
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

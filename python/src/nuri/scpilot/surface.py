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
from .arrangement import any_perpendicular


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


def _rotation_to(z_target: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping +z onto the unit vector ``z_target``."""
    ez = z_target
    ex = any_perpendicular(ez)[0]
    ey = np.cross(ez, ex)
    return np.column_stack([ex, ey, ez])


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
    for i in range(sas.n_active):
        area = ses.convex_area[i]
        if area <= 0.0:
            continue
        r = sas.radii[i]
        count = _lattice_count(4.0 * math.pi * r * r, density)
        if count not in lattices:
            lattices[count] = fibonacci_sphere(count)
        dirs = lattices[count]
        dirs = dirs[~sas.arrangements[i].contains(dirs)]
        if len(dirs) == 0:
            out.dropped += area
            continue
        out.add(
            sas.coords[i] + r * dirs,
            dirs,
            area / len(dirs),
            i,
            Patch.CONVEX,
        )


def _segments(counts: np.ndarray):
    """Expand per-segment ``counts`` into ``(segment id, index within
    segment)`` for every element, plus the start of every segment."""
    seg = np.repeat(np.arange(len(counts)), counts)
    start = np.cumsum(counts) - counts
    return seg, np.arange(len(seg)) - start[seg], start


def _toroidal(ses: SesGeometry, density: float, out: _DotBuffer) -> None:
    """Rings of equal ``beta`` width along every active arc's valid ranges;
    each ring gets ``round(area * density)`` dots spread uniformly in
    ``phi``. A range whose rings all round to zero is sampled as one ring;
    ranges that still round to zero are dropped."""
    sas = ses.sas
    rp = sas.rp
    circles, arcs = sas.circles, sas.arcs
    n_arcs = sas.n_active_arcs
    arc = np.repeat(np.arange(n_arcs), 2)
    part = np.tile([0, 1], n_arcs)
    c = arcs.circle[arc]
    lo, hi = ses.saddles.ranges[c, part].T
    rl, dphi = circles.radius[c], arcs.dphi[arc]
    width = hi - lo
    total = rp * dphi * ses.saddles.integral[c, part]

    k_beta = np.maximum(np.round(rp * width * math.sqrt(density)), 1).astype(
        int
    )
    row, m, first = _segments(k_beta)
    dbeta = (width / k_beta)[row]
    e_lo = lo[row] + m * dbeta
    e_hi = e_lo + dbeta
    beta = e_lo + 0.5 * dbeta
    area = (
        rp * dphi[row] * (rl[row] * dbeta - rp * (np.sin(e_hi) - np.sin(e_lo)))
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
    row, beta, area, k_phi = row[keep], beta[keep], area[keep], k_phi[keep]
    covered = np.bincount(row, area, minlength=len(arc))
    area *= (total / np.where(covered > 0.0, covered, 1.0))[row]

    ring, n_in_ring, _ = _segments(k_phi)
    r = row[ring]
    cc = c[r]
    phi = arcs.phi_beg[arc[r]] + (n_in_ring + 0.5) * dphi[r] / k_phi[ring]
    radial = (
        np.cos(phi)[:, None] * circles.e1[cc]
        + np.sin(phi)[:, None] * (circles.e2[cc])
    )
    q = circles.centre[cc] + rl[r, None] * radial
    b = beta[ring]
    inward = (
        -np.cos(b)[:, None] * radial + np.sin(b)[:, None] * circles.axis[cc]
    )
    pts = q + rp * inward
    i, j = circles.pair[cc, 0], circles.pair[cc, 1]
    di = np.linalg.norm(pts - sas.coords[i], axis=1) - sas.radii[i]
    dj = np.linalg.norm(pts - sas.coords[j], axis=1) - sas.radii[j]
    out.add(
        pts,
        -inward,
        (area / k_phi)[ring],
        np.where(di <= dj, i, j),
        Patch.TOROIDAL,
    )


def _concave(ses: SesGeometry, density: float, out: _DotBuffer) -> None:
    sas = ses.sas
    rp = sas.rp
    lattice = fibonacci_sphere(
        _lattice_count(4.0 * math.pi * rp * rp, density)
    )
    for face in ses.concave:
        area = face.area
        if area <= 0.0:
            continue
        centroid = face.contacts.sum(axis=0)
        cn = np.linalg.norm(centroid)
        dirs = lattice
        if cn > 1e-9:
            dirs = lattice @ _rotation_to(centroid / cn).T
        dirs = dirs[~face.arrangement.contains(dirs)]
        if len(dirs) == 0:
            out.dropped += area
            continue
        pts = sas.probes[face.probe] + rp * dirs
        depth = (
            np.linalg.norm(
                pts[:, None, :] - sas.coords[face.atoms][None], axis=2
            )
            - sas.radii[face.atoms][None, :]
        )
        out.add(
            pts,
            -dirs,
            area / len(dirs),
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

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Analytic solvent-accessible and solvent-excluded surfaces.

Every atom sphere is solved as an independent cap arrangement
(:mod:`nuri.scpilot.arrangement`). Vertices are clustered once, globally,
before the per-sphere solves so that toroidal arcs and probe positions
agree. Concave faces are arrangements on the probe spheres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .arrangement import (
    TAU_C,
    Arrangement,
    Caps,
    any_perpendicular,
    cluster_points,
    components,
    covered_arrangement,
    prepare_caps,
    solve,
    solve_caps,
)


@dataclass
class Circles:
    """Probe-centre circles of overlapping sphere pairs (``i < j``)."""

    pair: np.ndarray
    centre: np.ndarray
    radius: np.ndarray
    axis: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    a: np.ndarray
    d: np.ndarray

    def __len__(self) -> int:
        return len(self.radius)

    def half_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """``theta_i, theta_j``: angles at the probe between the axis
        plane and the contact directions toward atoms ``i`` and ``j``."""
        return (
            np.arctan2(self.a, self.radius),
            np.arctan2(self.d - self.a, self.radius),
        )


@dataclass
class TorusArcs:
    """Accessible arcs of the probe circles, in each circle's frame."""

    circle: np.ndarray
    phi_beg: np.ndarray
    dphi: np.ndarray
    v_beg: np.ndarray
    v_end: np.ndarray

    def __len__(self) -> int:
        return len(self.circle)

    @classmethod
    def concat(cls, parts: list[TorusArcs]) -> TorusArcs:
        z = np.empty(0, dtype=int)
        empty = cls(z, np.empty(0), np.empty(0), z, z)
        parts = [empty, *parts]
        return cls(
            *(
                np.concatenate([getattr(x, f) for x in parts])
                for f in ("circle", "phi_beg", "dphi", "v_beg", "v_end")
            )
        )


def prepare(coords, radii, rp, active=None):
    """Order atoms as ``[active | need minus active | occluders]`` and
    drop contained balls.

    Returns ``(order, n_active, n_solve, pairs, d)``: ``order`` maps new to
    old indices, ``pairs`` (``i < j``, sorted) are all overlapping pairs
    in new indices with their distances. Everything downstream assumes
    this shape: no contained or coincident balls, ``rp > 0``, and a pair
    touches a solved sphere iff ``i < n_solve``.
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if rp <= 0.0 or np.any(radii <= 0.0):
        raise ValueError("probe and atom radii must be positive")
    n = len(coords)
    sas = radii + rp
    pairs, d = overlaps(coords, sas)
    inside = contained(n, pairs, d, sas)
    active = (
        np.ones(n, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool).copy()
    )
    active &= ~inside

    i, j = pairs[:, 0], pairs[:, 1]
    keep = (d < sas[i] + sas[j] - TAU_C) & ~inside[i] & ~inside[j]
    pairs, d, i, j = pairs[keep], d[keep], i[keep], j[keep]
    need = active.copy()
    need[j[active[i]]] = True
    need[i[active[j]]] = True

    rank = np.where(active, 0, np.where(need, 1, np.where(inside, 3, 2)))
    order = np.argsort(rank, kind="stable")[: int((rank < 3).sum())]
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(len(order))
    pairs = np.sort(inv[pairs], axis=1)
    o = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return order, int(active.sum()), int(need.sum()), pairs[o], d[o]


def overlaps(coords, sas):
    """All pairs ``(i < j)`` with touching SAS balls and their distances."""
    tree = cKDTree(coords)
    pairs = tree.query_pairs(2.0 * sas.max(), output_type="ndarray")
    if len(pairs) == 0:
        pairs = np.empty((0, 2), dtype=int)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(coords[j] - coords[i], axis=1)
    if np.any(d < 1e-3):
        raise ValueError("coincident atoms")
    touching = d < sas[i] + sas[j]
    return pairs[touching], d[touching]


def contained(n, pairs, d, sas):
    """Balls lying inside another ball (within ``TAU_C``)."""
    i, j = pairs[:, 0], pairs[:, 1]
    inner = d <= np.abs(sas[i] - sas[j]) + TAU_C
    smaller = np.where(sas[i] < sas[j], i, j)
    out = np.zeros(n, dtype=bool)
    out[smaller[inner]] = True
    return out


@dataclass
class SasGeometry:
    """Analytic SAS of atoms ordered by :func:`prepare`.

    Spheres ``< n_solve`` have arrangements; spheres ``< n_active`` own
    surface. Every other sphere only occludes.
    """

    coords: np.ndarray
    radii: np.ndarray
    rp: float
    sas: np.ndarray
    n_active: int
    n_solve: int
    circles: Circles
    arrangements: list[Arrangement]
    probes: np.ndarray
    probe_offsets: np.ndarray
    probe_atoms: np.ndarray
    arcs: TorusArcs

    @property
    def sas_area(self) -> np.ndarray:
        return np.array([arr.area for arr in self.arrangements])

    def atoms_of(self, probe: int) -> np.ndarray:
        return self.probe_atoms[
            self.probe_offsets[probe] : self.probe_offsets[probe + 1]
        ]

    @property
    def probe_active(self) -> np.ndarray:
        """Probes touching at least one active atom (atoms are sorted, so
        the first atom of each probe is its smallest index)."""
        return self.probe_atoms[self.probe_offsets[:-1]] < self.n_active

    @classmethod
    def from_atoms(
        cls, coords, radii, rp: float, active=None
    ) -> tuple[SasGeometry, np.ndarray]:
        """Prepare, permute and build; also returns the new-to-old atom
        index map."""
        order, n_active, n_solve, pairs, d = prepare(coords, radii, rp, active)
        coords = np.asarray(coords, dtype=float)[order]
        radii = np.asarray(radii, dtype=float)[order]
        return cls.build(coords, radii, rp, pairs, d, n_active, n_solve), order

    @classmethod
    def build(
        cls, coords, radii, rp: float, pairs, d, n_active: int, n_solve: int
    ) -> SasGeometry:
        n = len(coords)
        sas = radii + rp
        n_circ = int(np.searchsorted(pairs[:, 0], n_solve))
        circles = _circles(coords, sas, pairs[:n_circ], d[:n_circ])
        nbr_off, nbr_flat = _neighbour_csr(n, pairs)
        offsets, all_caps = _cap_rows(circles, sas, n_solve)

        caps_of: list[Caps] = []
        cap_slot = np.full((n_solve, n), -1, dtype=int)
        covered = np.zeros(n_solve, dtype=bool)
        for i in range(n_solve):
            caps, covered[i] = prepare_caps(
                all_caps.take(slice(offsets[i], offsets[i + 1])), sas[i]
            )
            caps_of.append(caps)
            if not covered[i]:
                other = circles.pair[caps.tag].sum(axis=1) - i
                cap_slot[i, other] = np.arange(len(caps))

        triples, points = _triple_vertices(
            coords,
            sas,
            circles,
            nbr_off,
            nbr_flat,
            pairs[:, 0] * n + pairs[:, 1],
        )
        raw_pts = points.reshape(-1, 3)
        raw_triple = np.repeat(np.arange(len(triples)), 2)
        label = cluster_points(raw_pts, TAU_C)
        n_clusters = int(label.max()) + 1 if len(label) else 0
        reps = np.zeros((n_clusters, 3))
        np.add.at(reps, label, raw_pts)
        reps /= np.maximum(np.bincount(label, minlength=n_clusters), 1)[
            :, None
        ]
        atoms_key = np.unique(
            (label[:, None] * n + triples[raw_triple]).ravel()
        )
        owner = np.full(n_clusters, n, dtype=int)
        np.minimum.at(owner, label, triples[raw_triple, 0])

        tri_off, tri_flat = _inverted_index(triples, n_solve)
        per_sphere = []
        edges = [np.empty((0, 2), dtype=int)]
        cap_off = np.cumsum([0, *(len(c) for c in caps_of)])
        accessible = np.zeros(n_clusters, dtype=bool)
        for i, caps in enumerate(caps_of):
            sel = tri_flat[tri_off[i] : tri_off[i + 1]]
            others = triples[sel][triples[sel] != i].reshape(-1, 2)
            slots = cap_slot[i][others]
            present = np.all(slots >= 0, axis=1)
            sel, slots = sel[present], slots[present]
            edges.append(slots + cap_off[i])
            raw = 2 * np.repeat(sel, 2) + np.tile([0, 1], len(sel))
            local, inv = np.unique(label[raw], return_inverse=True)
            local_reps = reps[local] - coords[i]
            if len(local):
                local_reps /= np.linalg.norm(local_reps, axis=1, keepdims=True)
            per_sphere.append((slots, local, inv, local_reps))
            mine = owner[local] == i
            accessible[local[mine]] = _accessible_on_sphere(
                caps, i, local[mine], local_reps[mine], circles, atoms_key, n
            )
        n_components = _components_per_sphere(cap_off, np.concatenate(edges))

        probe_ids = np.flatnonzero(accessible)
        probe_map = np.full(n_clusters, -1, dtype=int)
        probe_map[probe_ids] = np.arange(len(probe_ids))
        probe_offsets, probe_atoms = _probe_atoms(n, atoms_key, probe_ids)

        arrangements: list[Arrangement] = []
        arc_parts = []
        for i, caps in enumerate(caps_of):
            if covered[i]:
                arrangements.append(covered_arrangement(sas[i], caps))
                continue
            slots, local, inv, local_reps = per_sphere[i]
            crossing = np.zeros((len(caps), len(caps)), dtype=bool)
            crossing[slots[:, 0], slots[:, 1]] = True
            crossing[slots[:, 1], slots[:, 0]] = True
            sign = np.where(circles.pair[caps.tag, 0] == i, 1.0, -1.0)
            arr = solve(
                sas[i],
                caps,
                np.repeat(slots, 2, axis=0),
                inv,
                local_reps,
                accessible[local],
                int(n_components[i]),
                crossing,
                (circles.e1[caps.tag], sign[:, None] * circles.e2[caps.tag]),
            )
            arrangements.append(arr)
            arc_parts.append(_torus_arcs(i, arr, circles, probe_map[local]))

        return cls(
            coords,
            radii,
            rp,
            sas,
            n_active,
            n_solve,
            circles,
            arrangements,
            reps[probe_ids],
            probe_offsets,
            probe_atoms,
            TorusArcs.concat(arc_parts),
        )


def _cap_rows(circles: Circles, sas, n_solve: int):
    """Caps of every solved sphere, gathered from the circle rows.

    Circle ``(i, j)`` cuts sphere ``i`` with axis ``u`` and sphere ``j``
    (when solved) with axis ``-u``; ``cos`` and ``sin`` are the distances
    ``a`` / ``d - a`` and the circle radius over the sphere radius. Rows
    are sorted by sphere; ``offsets`` delimits each sphere's slice and
    ``tag`` is the circle id.
    """
    pi, pj = circles.pair[:, 0], circles.pair[:, 1]
    circ = np.arange(len(circles))
    second = pj < n_solve
    atom = np.concatenate([pi, pj[second]])
    circ = np.concatenate([circ, circ[second]])
    sign = np.concatenate([np.ones(len(pi)), -np.ones(int(second.sum()))])
    order = np.argsort(atom, kind="stable")
    atom, circ, sign = atom[order], circ[order], sign[order]
    offsets = np.searchsorted(atom, np.arange(n_solve + 1))
    r = sas[atom]
    a = np.where(
        sign > 0.0, circles.a[circ], circles.d[circ] - circles.a[circ]
    )
    caps = Caps(
        sign[:, None] * circles.axis[circ],
        a / r,
        circles.radius[circ] / r,
        circ,
    )
    return offsets, caps


def _neighbour_csr(n, pairs) -> tuple[np.ndarray, np.ndarray]:
    """Sorted neighbour lists of every atom as ``(offsets, flat)``."""
    both = np.concatenate([pairs, pairs[:, ::-1]])
    both = both[np.lexsort((both[:, 1], both[:, 0]))]
    return np.searchsorted(both[:, 0], np.arange(n + 1)), both[:, 1]


def _inverted_index(triples, n_solve) -> tuple[np.ndarray, np.ndarray]:
    """Triples touching each solved sphere as ``(offsets, flat)``."""
    atom = triples.ravel()
    order = np.argsort(atom, kind="stable")
    tri = np.repeat(np.arange(len(triples)), 3)[order]
    return np.searchsorted(atom[order], np.arange(n_solve + 1)), tri


def _components_per_sphere(cap_off, edges) -> np.ndarray:
    """Connected components of every sphere's crossing graph, from one
    block-diagonal graph over all caps."""
    n_nodes = int(cap_off[-1])
    labels = components(n_nodes, edges)
    sphere = np.repeat(np.arange(len(cap_off) - 1), np.diff(cap_off))
    uniq = np.unique(sphere * n_nodes + labels)
    return np.bincount(uniq // n_nodes, minlength=len(cap_off) - 1)


def _triple_vertices(coords, sas, circles, nbr_off, nbr_flat, pair_keys):
    """Vertices of all sphere triples touching a solved sphere.

    Candidates are ``(i, j, k)`` with ``i < j < k``, ``(i, j)`` a circle
    and ``k`` a later neighbour of ``i`` that also pairs with ``j``. Each
    triple is intersected once (circle ``(i, j)`` against sphere ``k``)
    so that all three spheres see identical points. Returns
    ``(triples (t, 3) atom indices, points (t, 2, 3))``.
    """
    n = len(coords)
    i, j = circles.pair[:, 0], circles.pair[:, 1]
    nbr_key = np.repeat(np.arange(len(nbr_off) - 1), np.diff(nbr_off)) * n
    nbr_key += nbr_flat
    lo = np.searchsorted(nbr_key, i * n + j, "right")
    count = nbr_off[i + 1] - lo
    circ = np.repeat(np.arange(len(circles)), count)
    k = nbr_flat[
        np.repeat(lo, count)
        + np.arange(int(count.sum()))
        - np.repeat(np.cumsum(count) - count, count)
    ]
    key = j[circ] * n + k
    pos = np.minimum(np.searchsorted(pair_keys, key), len(pair_keys) - 1)
    is_pair = pair_keys[pos] == key
    circ, k = circ[is_pair], k[is_pair]
    triples = np.column_stack([i[circ], j[circ], k])

    t, rl = circles.centre[circ], circles.radius[circ]
    e1, e2 = circles.e1[circ], circles.e2[circ]
    w = t - coords[k]
    g = (sas[k] ** 2 - np.einsum("ij,ij->i", w, w) - rl * rl) / (2.0 * rl)
    a = np.einsum("ij,ij->i", w, e1)
    b = np.einsum("ij,ij->i", w, e2)
    amp2 = a * a + b * b
    hsq = amp2 - g * g
    ok = hsq > 0.0
    triples, t, rl, e1, e2 = triples[ok], t[ok], rl[ok], e1[ok], e2[ok]
    g, a, b, amp2, h = g[ok], a[ok], b[ok], amp2[ok], np.sqrt(hsq[ok])
    pts = []
    for sign in (-1.0, 1.0):
        cos_phi = (a * g + sign * b * h) / amp2
        sin_phi = (b * g - sign * a * h) / amp2
        radial = cos_phi[:, None] * e1 + sin_phi[:, None] * e2
        pts.append(t + rl[:, None] * radial)
    return triples, np.stack(pts, axis=1)


def _circles(coords, sas, pairs, d) -> Circles:
    i, j = pairs[:, 0], pairs[:, 1]
    ri, rj = sas[i], sas[j]
    axis = (coords[j] - coords[i]) / d[:, None]
    a = (d * d + ri * ri - rj * rj) / (2.0 * d)
    rl = np.sqrt(np.maximum(ri * ri - a * a, 0.0))
    centre = coords[i] + a[:, None] * axis
    e1 = any_perpendicular(axis)
    e2 = np.cross(axis, e1)
    return Circles(pairs, centre, rl, axis, e1, e2, a, d)


def _accessible_on_sphere(caps, i, clusters, dirs, circles, atoms_key, n):
    """Accessibility of the clusters owned by sphere ``i`` (their
    smallest atom), decided once against that sphere's caps.

    A cluster is accessible iff it lies outside every cap except those of
    its own atoms. Every ball that could contain a point of sphere ``i``
    overlaps it and therefore is a cap here (or nested inside one), so
    this equals the test against all SAS balls; deciding it once per
    cluster keeps every sphere sharing the cluster consistent.
    """
    if len(clusters) == 0:
        return np.zeros(0, dtype=bool)
    inside = dirs @ caps.axis.T > caps.cos_a
    other = circles.pair[caps.tag].sum(axis=1) - i
    key = clusters[:, None] * n + other[None, :]
    pos = np.minimum(np.searchsorted(atoms_key, key), len(atoms_key) - 1)
    excused = atoms_key[pos] == key
    return ~(inside & ~excused).any(axis=1)


def _probe_atoms(n, keys, probe_ids) -> tuple[np.ndarray, np.ndarray]:
    """Sorted atoms of every probe as ``(offsets, flat)``; ``keys`` are
    the sorted unique ``cluster * n + atom`` incidences."""
    cluster, atom = np.divmod(keys, n)
    lo = np.searchsorted(cluster, probe_ids)
    hi = np.searchsorted(cluster, probe_ids + 1)
    count = hi - lo
    flat = atom[
        np.repeat(lo, count)
        + np.arange(int(count.sum()))
        - np.repeat(np.cumsum(count) - count, count)
    ]
    return np.concatenate([[0], np.cumsum(count)]), flat


def _torus_arcs(i, arr, circles, probe_of_local) -> TorusArcs:
    """Arcs of sphere ``i`` on circles it is the smaller sphere of.

    Those caps share the circle frame, so ``phi`` carries over as is; the
    larger sphere reports nothing. A ``-1`` appended to the probe map
    lets full circles (``-1`` ends) read back ``-1``.
    """
    arcs = arr.arcs
    c = arr.caps.tag[arcs.cap]
    keep = circles.pair[c, 0] == i
    ext = np.append(probe_of_local, -1)
    return TorusArcs(
        c[keep],
        arcs.phi_beg[keep],
        arcs.dphi[keep],
        ext[arcs.v_beg[keep]],
        ext[arcs.v_end[keep]],
    )


@dataclass
class Saddles:
    """Valid generating-arc angle ranges per circle and area per arc.

    The generating arc is parametrised by ``beta``, the angle from the
    inward radial direction; ``beta < 0`` leans toward atom ``i``. Distance
    from the axis is ``rl - rp cos(beta)``, zero at the cusp of a spindle.
    ``ranges`` is ``(n_circles, 2, 2)``: the parts below and above the cusp,
    zero-width where absent.
    """

    ranges: np.ndarray
    area: np.ndarray


@dataclass
class ConcaveFace:
    probe: int
    atoms: np.ndarray
    contacts: np.ndarray
    arrangement: Arrangement

    @property
    def area(self) -> float:
        return self.arrangement.area


@dataclass
class SesGeometry:
    sas: SasGeometry
    convex_area: np.ndarray
    saddles: Saddles
    concave: list[ConcaveFace] = field(default_factory=list)

    @property
    def rp(self) -> float:
        return self.sas.rp

    @classmethod
    def build(cls, sas: SasGeometry) -> SesGeometry:
        ratio = sas.radii[: sas.n_solve] / sas.sas[: sas.n_solve]
        convex = sas.sas_area * ratio * ratio
        return cls(
            sas,
            convex,
            _saddles(sas),
            _concave_faces(sas),
        )


def beta_ranges(rl, rp, theta_i, theta_j) -> np.ndarray:
    """Valid ``beta`` ranges ``(..., 2, 2)`` of the generating arc.

    The arc runs from ``-theta_i`` (touching atom ``i``) to ``theta_j``.
    On a spindle torus (``rl < rp``) the part ``|beta| < b0`` lies beyond
    the axis and is cut out; ``b0`` is zero otherwise, so the two ranges
    simply tile the arc. Absent parts come out zero-width.
    """
    rl, theta_i, theta_j = np.broadcast_arrays(rl, theta_i, theta_j)
    lo, hi = -theta_i, theta_j
    b0 = np.arctan2(np.sqrt(np.maximum(rp * rp - rl * rl, 0.0)), rl)
    below = np.stack([lo, np.maximum(lo, np.minimum(hi, -b0))], axis=-1)
    above = np.stack([np.minimum(hi, np.maximum(lo, b0)), hi], axis=-1)
    return np.stack([below, above], axis=-2)


def saddle_area(rl, rp, ranges, dphi):
    """Saddle area of ``ranges`` ``(..., 2, 2)`` swept over ``dphi``."""
    lo, hi = ranges[..., 0], ranges[..., 1]
    rl = np.asarray(rl)[..., None]
    per_range = rl * (hi - lo) - rp * (np.sin(hi) - np.sin(lo))
    return dphi * rp * per_range.sum(axis=-1)


def _saddles(sas: SasGeometry) -> Saddles:
    circles, arcs = sas.circles, sas.arcs
    theta_i, theta_j = circles.half_angles()
    ranges = beta_ranges(circles.radius, sas.rp, theta_i, theta_j)
    c = arcs.circle
    area = saddle_area(circles.radius[c], sas.rp, ranges[c], arcs.dphi)
    return Saddles(ranges, area)


def _departure_caps(sas: SasGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents of the accessible arcs leaving each probe, as
    ``(offsets, tangents)`` sorted by probe.

    The probe rolls away along each such arc, so the half of its sphere
    facing the tangent is swept and cannot be concave face. For an
    ordinary three-atom vertex these are the three side planes of the
    contact triangle; k-fold vertices and single-vertex circles fall out
    of the same rule. The tangent at a probe on circle ``c`` is
    ``axis x radial``; leaving toward decreasing ``phi`` flips it.
    """
    arcs, circles = sas.arcs, sas.circles
    probe = np.concatenate([arcs.v_beg, arcs.v_end])
    circ = np.concatenate([arcs.circle, arcs.circle])
    sign = np.repeat([1.0, -1.0], len(arcs))
    keep = probe >= 0
    probe, circ, sign = probe[keep], circ[keep], sign[keep]
    radial = (sas.probes[probe] - circles.centre[circ]) / circles.radius[
        circ, None
    ]
    tangents = sign[:, None] * np.cross(circles.axis[circ], radial)
    order = np.argsort(probe, kind="stable")
    offsets = np.searchsorted(probe[order], np.arange(len(sas.probes) + 1))
    return offsets, tangents[order]


def _concave_faces(sas: SasGeometry) -> list[ConcaveFace]:
    probes = sas.probes
    if len(probes) == 0:
        return []
    dep_off, dep_t = _departure_caps(sas)
    tree = cKDTree(probes)
    near = tree.query_ball_point(probes, 2.0 * sas.rp)
    faces = []
    for q in np.flatnonzero(sas.probe_active):
        atoms = sas.atoms_of(q)
        contacts = sas.coords[atoms] - probes[q]
        contacts /= np.linalg.norm(contacts, axis=1, keepdims=True)
        others = np.array([n for n in near[q] if n != q], dtype=int)
        diff = probes[others] - probes[q]
        dist = np.linalg.norm(diff, axis=1)
        tangents = dep_t[dep_off[q] : dep_off[q + 1]]
        cos_a = dist / (2.0 * sas.rp)
        caps = Caps(
            np.concatenate([tangents, diff / dist[:, None]]),
            np.concatenate([np.zeros(len(tangents)), cos_a]),
            np.concatenate(
                [np.ones(len(tangents)), np.sqrt(1.0 - cos_a * cos_a)]
            ),
            np.concatenate([-np.arange(1, len(tangents) + 1), others]),
        )
        faces.append(ConcaveFace(q, atoms, contacts, solve_caps(sas.rp, caps)))
    return faces


def ses_area(ses: SesGeometry):
    """Per-atom convex, per-arc saddle, per-face concave areas."""
    concave = np.array([f.area for f in ses.concave])
    return ses.convex_area, ses.saddles.area, concave


def two_sphere_ses_area(r1: float, r2: float, d: float, rp: float):
    """Closed-form SES area of two intersecting spheres, per patch kind
    (convex, toroidal, concave); Quan & Stamm (2016) eqs. 5.26-5.27."""
    R1, R2 = r1 + rp, r2 + rp
    if d >= R1 + R2:
        return 4 * math.pi * (r1 * r1 + r2 * r2), 0.0, 0.0
    a = (d * d + R1 * R1 - R2 * R2) / (2.0 * d)
    rl = math.sqrt(R1 * R1 - a * a)
    th1 = math.atan2(a, rl)
    th2 = math.atan2(d - a, rl)
    convex = (
        2 * math.pi * (r1 * r1 * (1 + a / R1) + r2 * r2 * (1 + (d - a) / R2))
    )
    ranges = beta_ranges(rl, rp, th1, th2)
    torus = float(saddle_area(rl, rp, ranges, 2.0 * math.pi))
    return convex, torus, 0.0

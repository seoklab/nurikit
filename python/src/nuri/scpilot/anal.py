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
    _merge_coincident,
    any_perpendicular,
    classify_caps,
    cluster_points,
    components,
    covered_arrangement,
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
    keep = ~inside[i] & ~inside[j]
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
    """All pairs ``(i < j)`` whose SAS balls overlap by more than ``TAU_C``
    and their distances; tangent pairs are not overlaps."""
    tree = cKDTree(coords)
    pairs = tree.query_pairs(2.0 * sas.max(), output_type="ndarray")
    if len(pairs) == 0:
        pairs = np.empty((0, 2), dtype=int)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(coords[j] - coords[i], axis=1)
    if np.any(d < 1e-3):
        raise ValueError("coincident atoms")
    overlapping = d < sas[i] + sas[j] - TAU_C
    return pairs[overlapping], d[overlapping]


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
    n_active_circles: int
    n_active_probes: int
    n_active_arcs: int

    @property
    def sas_area(self) -> np.ndarray:
        return np.array([arr.area for arr in self.arrangements])

    def atoms_of(self, probe: int) -> np.ndarray:
        return self.probe_atoms[
            self.probe_offsets[probe] : self.probe_offsets[probe + 1]
        ]

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
        offsets, all_caps, row_of = _cap_rows(
            circles, sas, n_solve, len(pairs)
        )

        caps_of: list[Caps] = []
        slot_of_row = np.empty(len(all_caps), dtype=int)
        for i in range(n_solve):
            sl = slice(offsets[i], offsets[i + 1])
            caps, slot_of_row[sl] = _merge_coincident(
                all_caps.take(sl), sas[i]
            )
            caps_of.append(caps)

        tri = _triple_candidates(
            coords,
            sas,
            circles,
            nbr_off,
            nbr_flat,
            pairs[:, 0] * n + pairs[:, 1],
        )
        solved = tri.triples < n_solve
        slots = _triple_slots(tri.circ, row_of, slot_of_row)
        distinct = (slots >= 0).all(axis=2) & (slots[..., 0] != slots[..., 1])
        kept = (~solved | distinct).all(axis=1)

        covered = np.zeros(n_solve, dtype=bool)
        sph_off, _, sph_edges = _sphere_incidence(
            tri.triples[kept], slots[kept], solved[kept], n_solve
        )
        for i, caps in enumerate(caps_of):
            m = len(caps)
            crossing = np.zeros((m, m), dtype=bool)
            e = sph_edges[sph_off[i] : sph_off[i + 1]]
            crossing[e[:, 0], e[:, 1]] = True
            crossing[e[:, 1], e[:, 0]] = True
            hidden, covered[i], _ = classify_caps(caps, crossing)
            caps_of[i] = caps.take(np.flatnonzero(~hidden))
            renumber = np.where(hidden, -1, np.cumsum(~hidden) - 1)
            sl = slice(offsets[i], offsets[i + 1])
            slot_of_row[sl] = renumber[slot_of_row[sl]]

        slots = _triple_slots(tri.circ, row_of, slot_of_row)
        kept &= (~solved | (slots >= 0).all(axis=2)).all(axis=1)
        triples, slots, solved = tri.triples[kept], slots[kept], solved[kept]
        points = tri.points(kept)

        raw_pts = points.reshape(-1, 3)
        raw_triple = np.repeat(np.arange(len(triples)), 2)
        label, atoms_key, owner = _clusters_by_owner(
            cluster_points(raw_pts, TAU_C), triples[raw_triple], n
        )
        n_clusters = len(owner)
        reps = np.zeros((n_clusters, 3))
        np.add.at(reps, label, raw_pts)
        reps /= np.bincount(label, minlength=n_clusters)[:, None]

        cap_off = np.cumsum([0, *(len(c) for c in caps_of)])
        sph_off, sph_tri, sph_edges = _sphere_incidence(
            triples, slots, solved, n_solve
        )
        sphere = np.repeat(np.arange(n_solve), np.diff(sph_off))
        gcap = sph_edges + cap_off[sphere, None]
        cluster = label[2 * sph_tri[:, None] + np.array([0, 1])]
        incidence = np.unique(
            (cluster[:, :, None] * cap_off[-1] + gcap[:, None, :]).ravel()
        )
        n_components = _components_per_sphere(cap_off, gcap)
        accessible = _accessible(
            Caps.concat(caps_of), cap_off, coords, reps, owner, incidence
        )

        probe_ids = np.flatnonzero(accessible)
        probe_map = np.full(n_clusters, -1, dtype=int)
        probe_map[probe_ids] = np.arange(len(probe_ids))
        probe_offsets, probe_atoms = _probe_atoms(n, atoms_key, probe_ids)

        arrangements: list[Arrangement] = []
        arc_parts = []
        n_active_arcs = 0
        for i, caps in enumerate(caps_of):
            if covered[i]:
                arrangements.append(covered_arrangement(sas[i], caps))
                continue
            rows = slice(sph_off[i], sph_off[i + 1])
            local, inv = np.unique(cluster[rows], return_inverse=True)
            local_reps = reps[local] - coords[i]
            local_reps /= np.linalg.norm(local_reps, axis=1, keepdims=True)
            excused = np.zeros((len(local), len(caps)), dtype=bool)
            excused[
                inv.reshape(-1, 2)[:, :, None], sph_edges[rows, None, :]
            ] = True
            circ, sign = caps.tag >> 1, 1.0 - 2.0 * (caps.tag & 1)
            arr = solve(
                sas[i],
                caps,
                sph_edges[rows],
                local_reps,
                excused,
                accessible[local],
                int(n_components[i]),
                (circles.e1[circ], sign[:, None] * circles.e2[circ]),
            )
            arrangements.append(arr)
            arc_parts.append(_torus_arcs(arr, probe_map[local]))
            n_active_arcs += len(arc_parts[-1]) * (i < n_active)

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
            int(np.searchsorted(circles.pair[:, 0], n_active)),
            int(np.searchsorted(owner[probe_ids], n_active)),
            n_active_arcs,
        )


def _cap_rows(circles: Circles, sas, n_solve: int, n_pairs: int):
    """Caps of every solved sphere, gathered from the circle rows.

    Circle ``(i, j)`` cuts sphere ``i`` with axis ``u`` and sphere ``j``
    (when solved) with axis ``-u``; ``cos`` and ``sin`` are the distances
    ``a`` / ``d - a`` and the circle radius over the sphere radius. Rows
    are sorted by sphere; ``offsets`` delimits each sphere's slice. A cap's
    ``tag`` is ``2 * circle + side`` with side 0 on the circle's first
    sphere; ``row_of[tag]`` is its row, ``-1`` where the sphere is not
    solved.
    """
    pi, pj = circles.pair[:, 0], circles.pair[:, 1]
    circ = np.arange(len(circles))
    second = np.flatnonzero(pj < n_solve)
    atom = np.concatenate([pi, pj[second]])
    tag = np.concatenate([2 * circ, 2 * second + 1])
    a = np.concatenate([circles.a, (circles.d - circles.a)[second]])
    axis = np.concatenate([circles.axis, -circles.axis[second]])
    order = np.argsort(atom, kind="stable")
    atom, tag, a, axis = atom[order], tag[order], a[order], axis[order]
    offsets = np.searchsorted(atom, np.arange(n_solve + 1))
    r = sas[atom]
    caps = Caps(axis, a / r, circles.radius[tag >> 1] / r, tag)
    row_of = np.full(2 * n_pairs, -1, dtype=int)
    row_of[tag] = np.arange(len(tag))
    return offsets, caps, row_of


def _neighbour_csr(n, pairs) -> tuple[np.ndarray, np.ndarray]:
    """Sorted neighbour lists of every atom as ``(offsets, flat)``."""
    both = np.concatenate([pairs, pairs[:, ::-1]])
    both = both[np.lexsort((both[:, 1], both[:, 0]))]
    return np.searchsorted(both[:, 0], np.arange(n + 1)), both[:, 1]


def _components_per_sphere(cap_off, edges) -> np.ndarray:
    """Connected components of every sphere's crossing graph, from one
    block-diagonal graph over all caps."""
    n_nodes = int(cap_off[-1])
    labels = components(n_nodes, edges)
    sphere = np.repeat(np.arange(len(cap_off) - 1), np.diff(cap_off))
    uniq = np.unique(sphere * n_nodes + labels)
    return np.bincount(uniq // n_nodes, minlength=len(cap_off) - 1)


@dataclass
class _TripleCandidates:
    """Sphere triples whose circle ``(i, j)`` meets sphere ``k`` in two
    points, with the intersection geometry kept so that the points can be
    computed for a subset later."""

    triples: np.ndarray
    circ: np.ndarray
    centre: np.ndarray
    radius: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    g: np.ndarray
    a: np.ndarray
    b: np.ndarray
    h: np.ndarray

    def points(self, mask) -> np.ndarray:
        """Both intersection points ``(t, 2, 3)`` of the selected triples,
        ``phi_0 -/+ acos(g / amp)`` written without inverse trig."""
        g, a, b, h = self.g[mask], self.a[mask], self.b[mask], self.h[mask]
        amp2 = a * a + b * b
        t, rl = self.centre[mask], self.radius[mask, None]
        e1, e2 = self.e1[mask], self.e2[mask]
        pts = []
        for sign in (-1.0, 1.0):
            cos_phi = (a * g + sign * b * h) / amp2
            sin_phi = (b * g - sign * a * h) / amp2
            pts.append(
                t + rl * (cos_phi[:, None] * e1 + sin_phi[:, None] * e2)
            )
        return np.stack(pts, axis=1)


def _triple_candidates(coords, sas, circles, nbr_off, nbr_flat, pair_keys):
    """Candidates ``(i, j, k)`` with ``i < j < k``, ``(i, j)`` a circle and
    ``k`` a later neighbour of ``i`` that also pairs with ``j``, kept iff
    circle ``(i, j)`` crosses sphere ``k`` (``h^2 > 0``). Each triple is
    intersected once so that all three spheres see identical points.
    ``circ`` holds the pair ids of ``(i, j)``, ``(i, k)`` and ``(j, k)``.
    """
    n = len(coords)
    i, j = circles.pair[:, 0], circles.pair[:, 1]
    nbr_key = np.repeat(np.arange(len(nbr_off) - 1), np.diff(nbr_off)) * n
    nbr_key += nbr_flat
    lo = np.searchsorted(nbr_key, i * n + j, "right")
    count = nbr_off[i + 1] - lo
    circ = np.repeat(np.arange(len(circles)), count)
    k = nbr_flat[np.repeat(lo, count) + _within(count)]
    key = j[circ] * n + k
    pos = np.minimum(np.searchsorted(pair_keys, key), len(pair_keys) - 1)
    is_pair = pair_keys[pos] == key
    circ, k, pos_jk = circ[is_pair], k[is_pair], pos[is_pair]
    pos_ik = np.searchsorted(pair_keys, i[circ] * n + k)

    t, rl = circles.centre[circ], circles.radius[circ]
    e1, e2 = circles.e1[circ], circles.e2[circ]
    w = t - coords[k]
    g = (sas[k] ** 2 - np.einsum("ij,ij->i", w, w) - rl * rl) / (2.0 * rl)
    a = np.einsum("ij,ij->i", w, e1)
    b = np.einsum("ij,ij->i", w, e2)
    hsq = a * a + b * b - g * g
    ok = hsq > 0.0
    return _TripleCandidates(
        np.column_stack([i[circ], j[circ], k])[ok],
        np.column_stack([circ, pos_ik, pos_jk])[ok],
        t[ok],
        rl[ok],
        e1[ok],
        e2[ok],
        g[ok],
        a[ok],
        b[ok],
        np.sqrt(hsq[ok]),
    )


def _triple_slots(circ, row_of, slot_of_row) -> np.ndarray:
    """Cap slots ``(t, 3, 2)`` of each triple's two caps on each of its
    three spheres, ``-1`` where the sphere has no such cap."""
    cij, cik, cjk = circ[:, 0], circ[:, 1], circ[:, 2]
    rows = np.stack(
        [
            np.column_stack([row_of[2 * cij], row_of[2 * cik]]),
            np.column_stack([row_of[2 * cij + 1], row_of[2 * cjk]]),
            np.column_stack([row_of[2 * cik + 1], row_of[2 * cjk + 1]]),
        ],
        axis=1,
    )
    return np.append(slot_of_row, -1)[rows]


def _sphere_incidence(triples, slots, solved, n_solve):
    """Every (solved sphere, triple touching it) incidence sorted by
    sphere: ``(offsets, triple, the triple's two cap slots there)``."""
    sphere = triples[solved]
    order = np.argsort(sphere, kind="stable")
    tri = np.repeat(np.arange(len(triples)), solved.sum(axis=1))
    offsets = np.searchsorted(sphere[order], np.arange(n_solve + 1))
    return offsets, tri[order], slots[solved][order]


def _clusters_by_owner(label, atoms, n):
    """Relabel clusters so that they are sorted by owner, the smallest atom
    of the cluster, which is the sphere deciding its accessibility.

    Returns ``(label, atoms_key, owner)`` with ``atoms_key`` the sorted
    unique ``cluster * n + atom`` incidences (``atoms`` (k, 3) per raw
    point).
    """
    keys = np.unique((label[:, None] * n + atoms).ravel())
    cluster, atom = np.divmod(keys, n)
    n_clusters = int(label.max()) + 1 if len(label) else 0
    first = np.searchsorted(cluster, np.arange(n_clusters))
    order = np.argsort(atom[first], kind="stable")
    rank = np.empty(n_clusters, dtype=int)
    rank[order] = np.arange(n_clusters)
    keys = np.unique(rank[cluster] * n + atom)
    return rank[label], keys, atom[first][order]


def _accessible(caps, cap_off, coords, reps, owner, incidence) -> np.ndarray:
    """Accessibility of every cluster, decided once on its owner sphere.

    A cluster is accessible iff it lies outside every cap of that sphere
    except those whose crossing points merged into it (``incidence``,
    sorted ``cluster * n_caps + cap`` keys). Every ball that could contain
    a point of the sphere overlaps it and therefore is a cap there (or
    nested inside one), so this equals the test against all SAS balls.
    """
    n_caps = int(cap_off[-1])
    count = np.diff(cap_off)[owner]
    cluster = np.repeat(np.arange(len(owner)), count)
    cap = cap_off[owner[cluster]] + _within(count)
    dirs = reps[cluster] - coords[owner[cluster]]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    inside = np.einsum("ij,ij->i", dirs, caps.axis[cap]) > caps.cos_a[cap]
    key = cluster * n_caps + cap
    pos = np.minimum(np.searchsorted(incidence, key), len(incidence) - 1)
    excused = incidence[pos] == key
    return np.bincount(cluster, inside & ~excused, minlength=len(owner)) == 0


def _within(counts) -> np.ndarray:
    """Index of every element within its segment of ``counts``."""
    return np.arange(int(counts.sum())) - np.repeat(
        np.cumsum(counts) - counts, counts
    )


def _circles(coords, sas, pairs, d) -> Circles:
    i, j = pairs[:, 0], pairs[:, 1]
    ri, rj = sas[i], sas[j]
    axis = (coords[j] - coords[i]) / d[:, None]
    a = (d * d + ri * ri - rj * rj) / (2.0 * d)
    rl = np.sqrt(ri * ri - a * a)
    centre = coords[i] + a[:, None] * axis
    e1 = any_perpendicular(axis)
    e2 = np.cross(axis, e1)
    return Circles(pairs, centre, rl, axis, e1, e2, a, d)


def _probe_atoms(n, keys, probe_ids) -> tuple[np.ndarray, np.ndarray]:
    """Sorted atoms of every probe as ``(offsets, flat)``; ``keys`` are
    the sorted unique ``cluster * n + atom`` incidences."""
    cluster, atom = np.divmod(keys, n)
    lo = np.searchsorted(cluster, probe_ids)
    hi = np.searchsorted(cluster, probe_ids + 1)
    count = hi - lo
    flat = atom[np.repeat(lo, count) + _within(count)]
    return np.concatenate([[0], np.cumsum(count)]), flat


def _torus_arcs(arr, probe_of_local) -> TorusArcs:
    """Arcs of sphere ``i`` on circles it is the smaller sphere of.

    Those caps share the circle frame, so ``phi`` carries over as is; the
    larger sphere reports nothing. A ``-1`` appended to the probe map
    lets full circles (``-1`` ends) read back ``-1``.
    """
    arcs = arr.arcs
    tag = arr.caps.tag[arcs.cap]
    keep = (tag & 1) == 0
    ext = np.append(probe_of_local, -1)
    return TorusArcs(
        tag[keep] >> 1,
        arcs.phi_beg[keep],
        arcs.dphi[keep],
        ext[arcs.v_beg[keep]],
        ext[arcs.v_end[keep]],
    )


@dataclass
class Saddles:
    """Valid generating-arc angle ranges per active circle, their area
    integrals, and the area of every active arc.

    The generating arc is parametrised by ``beta``, the angle from the
    inward radial direction; ``beta < 0`` leans toward atom ``i``. Distance
    from the axis is ``rl - rp cos(beta)``, zero at the cusp of a spindle.
    ``ranges`` is ``(n_circles, 2, 2)``: the parts below and above the cusp,
    zero-width where absent; ``integral`` ``(n_circles, 2)`` is
    ``rl (hi - lo) - rp (sin hi - sin lo)`` per part, so that a saddle
    swept over ``dphi`` has area ``dphi rp sum(integral)``.
    """

    ranges: np.ndarray
    integral: np.ndarray
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
        ratio = sas.radii[: sas.n_active] / sas.sas[: sas.n_active]
        convex = sas.sas_area[: sas.n_active] * ratio * ratio
        return cls(
            sas,
            convex,
            _saddles(sas),
            _concave_faces(sas),
        )


def _pick(cond, x, sin_x, y, sin_y):
    return np.where(cond, x, y), np.where(cond, sin_x, sin_y)


def saddle_ranges(rl, rp, a_i, a_j, sas_i, sas_j):
    """Valid ``beta`` ranges ``(..., 2, 2)`` of the generating arc and the
    area integral ``(..., 2)`` of each.

    ``a_i``, ``a_j`` are the distances from the circle centre to the two
    sphere centres along the axis (``a`` and ``d - a``). The arc runs from
    ``-theta_i`` (touching atom ``i``) to ``theta_j`` with
    ``sin theta = a / R``. On a spindle torus (``rl < rp``) the part
    ``|beta| < b0`` lies beyond the axis and is cut out; ``b0`` is zero
    otherwise, so the two ranges simply tile the arc. Absent parts come out
    zero-width. Endpoints are selected together with their sines, so no
    sine of a selected angle is ever evaluated.
    """
    rl, a_i, a_j, sas_i, sas_j = np.broadcast_arrays(
        rl, a_i, a_j, sas_i, sas_j
    )
    lo, sin_lo = -np.arctan2(a_i, rl), -a_i / sas_i
    hi, sin_hi = np.arctan2(a_j, rl), a_j / sas_j
    root = np.sqrt(np.maximum(rp * rp - rl * rl, 0.0))
    b0, sin_b0 = np.arctan2(root, rl), root / rp
    m, sin_m = _pick(hi < -b0, hi, sin_hi, -b0, -sin_b0)
    below_hi, sin_below_hi = _pick(lo > m, lo, sin_lo, m, sin_m)
    m, sin_m = _pick(lo > b0, lo, sin_lo, b0, sin_b0)
    above_lo, sin_above_lo = _pick(hi < m, hi, sin_hi, m, sin_m)
    ranges = np.stack(
        [np.stack([lo, below_hi], axis=-1), np.stack([above_lo, hi], axis=-1)],
        axis=-2,
    )
    sines = np.stack(
        [
            np.stack([sin_lo, sin_below_hi], axis=-1),
            np.stack([sin_above_lo, sin_hi], axis=-1),
        ],
        axis=-2,
    )
    integral = rl[..., None] * (ranges[..., 1] - ranges[..., 0]) - rp * (
        sines[..., 1] - sines[..., 0]
    )
    return ranges, integral


def _saddles(sas: SasGeometry) -> Saddles:
    """Ranges and integrals of the active circles and areas of the active
    arcs, both prefixes."""
    circles, arcs = sas.circles, sas.arcs
    nc, na = sas.n_active_circles, sas.n_active_arcs
    pair, a, d = circles.pair[:nc], circles.a[:nc], circles.d[:nc]
    ranges, integral = saddle_ranges(
        circles.radius[:nc],
        sas.rp,
        a,
        d - a,
        sas.sas[pair[:, 0]],
        sas.sas[pair[:, 1]],
    )
    c = arcs.circle[:na]
    area = arcs.dphi[:na] * sas.rp * integral[c].sum(axis=-1)
    return Saddles(ranges, integral, area)


def _departure_caps(sas: SasGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents of the accessible arcs leaving each probe, as
    ``(offsets, tangents)`` sorted by probe.

    The probe rolls away along each such arc, so the half of its sphere
    facing the tangent is swept and cannot be concave face. For an
    ordinary three-atom vertex these are the three side planes of the
    contact triangle; k-fold vertices and single-vertex circles fall out
    of the same rule. The tangent at a probe on circle ``c`` is
    ``axis x radial``; leaving toward decreasing ``phi`` flips it. The
    ``-1`` ends of full circles sort before every probe and are never
    sliced.
    """
    arcs, circles = sas.arcs, sas.circles
    probe = np.concatenate([arcs.v_beg, arcs.v_end])
    circ = np.concatenate([arcs.circle, arcs.circle])
    sign = np.repeat([1.0, -1.0], len(arcs))
    radial = sas.probes[probe] - circles.centre[circ]
    tangents = np.cross(circles.axis[circ], radial)
    tangents *= (sign / np.linalg.norm(tangents, axis=1))[:, None]
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
    for q in range(sas.n_active_probes):
        atoms = sas.atoms_of(q)
        contacts = sas.coords[atoms] - probes[q]
        contacts /= np.linalg.norm(contacts, axis=1, keepdims=True)
        others = np.array([n for n in near[q] if n != q], dtype=int)
        diff = probes[others] - probes[q]
        dist = np.linalg.norm(diff, axis=1)
        close = dist < 2.0 * sas.rp
        others, diff, dist = others[close], diff[close], dist[close]
        tangents = dep_t[dep_off[q] : dep_off[q + 1]]
        cos_a = dist / (2.0 * sas.rp)
        caps = Caps(
            np.concatenate([tangents, diff / dist[:, None]]),
            np.concatenate([np.zeros(len(tangents)), cos_a]),
            np.concatenate(
                [np.ones(len(tangents)), np.sqrt(1.0 - cos_a * cos_a)]
            ),
            np.zeros(len(tangents) + len(others), dtype=int),
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
    convex = (
        2 * math.pi * (r1 * r1 * (1 + a / R1) + r2 * r2 * (1 + (d - a) / R2))
    )
    _, integral = saddle_ranges(rl, rp, a, d - a, R1, R2)
    return convex, float(2.0 * math.pi * rp * integral.sum()), 0.0

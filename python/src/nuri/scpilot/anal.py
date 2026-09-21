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
    classify_caps,
    cluster_points,
    components,
    covered_arrangement,
    cross,
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
    this shape: no contained or coincident balls, no two caps of one
    sphere on the same circle, ``rp > 0``, and a pair touches a solved
    sphere iff ``i < n_solve``.
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if rp <= 0.0 or np.any(radii <= 0.0):
        raise ValueError("probe and atom radii must be positive")
    n = len(coords)
    sas = radii + rp
    pairs, d = overlaps(coords, sas)
    inside = contained(n, pairs, d, sas)
    pairs, d = _without(pairs, d, inside)
    inside |= shared_circle_middles(coords, sas, pairs, d)
    pairs, d = _without(pairs, d, inside)
    active = (
        np.ones(n, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool).copy()
    )
    active &= ~inside

    i, j = pairs[:, 0], pairs[:, 1]
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


def shared_circle_middles(coords, sas, pairs, d):
    """Balls whose sphere passes through the circle of two other spheres
    with its centre between theirs on the axis.

    Such a ball lies inside the union of the other two: each of its caps
    is inside the neighbouring sphere's larger cap and the disc of the
    circle is inside both. It has no surface, and keeping it would put two
    caps of one sphere on the same circle. Two circles of sphere ``i``
    coincide iff their centres agree within ``TAU_C`` and their axes are
    parallel to within ``TAU_C / R_i``.
    """
    n = len(coords)
    out = np.zeros(n, dtype=bool)
    if len(pairs) == 0:
        return out
    o = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, d = pairs[o], d[o]
    nbr_off, nbr_flat = _neighbour_csr(n, pairs)
    triples, pid = _overlapping_triples(
        n, pairs, nbr_off, nbr_flat, len(pairs)
    )
    i, j, k = triples.T
    u_ij, t_ij = _circle_plane(coords, sas, i, j, d[pid[:, 0]])
    u_ik, t_ik = _circle_plane(coords, sas, i, k, d[pid[:, 1]])
    same = np.linalg.norm(t_ij - t_ik, axis=1) < TAU_C
    same &= sas[i] * np.linalg.norm(cross(u_ij, u_ik), axis=1) < TAU_C
    triples, u_ij = triples[same], u_ij[same]
    along = np.einsum(
        "tmj,tj->tm", coords[triples] - coords[triples[:, :1]], u_ij
    )
    middle = triples[np.arange(len(triples)), np.argsort(along, axis=1)[:, 1]]
    out[middle] = True
    return out


def _circle_plane(coords, sas, i, j, d):
    """Unit axis and centre of the circle where spheres ``i`` and ``j``
    meet."""
    u = (coords[j] - coords[i]) / d[:, None]
    a = (d * d + sas[i] ** 2 - sas[j] ** 2) / (2.0 * d)
    return u, coords[i] + a[:, None] * u


def _without(pairs, d, dropped):
    keep = ~dropped[pairs[:, 0]] & ~dropped[pairs[:, 1]]
    return pairs[keep], d[keep]


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
        offsets, all_caps, all_tags, row_of = _cap_rows(
            circles, sas, n_solve, len(pairs)
        )

        slot_of_row = _within(np.diff(offsets))
        caps_of = [
            all_caps.take(slice(offsets[i], offsets[i + 1]))
            for i in range(n_solve)
        ]
        tags_of = [
            all_tags[offsets[i] : offsets[i + 1]] for i in range(n_solve)
        ]

        tri = _triple_candidates(
            coords,
            sas,
            circles,
            *_overlapping_triples(n, pairs, nbr_off, nbr_flat, n_circ),
        )
        triples, circ = tri.triples, tri.circ
        solved = triples < n_solve
        slots = _triple_slots(circ, row_of, slot_of_row)

        covered = np.zeros(n_solve, dtype=bool)
        sph_off, _, sph_edges = _sphere_incidence(
            triples, slots, solved, n_solve
        )
        for i, caps in enumerate(caps_of):
            m = len(caps)
            crossing = np.zeros((m, m), dtype=bool)
            e = sph_edges[sph_off[i] : sph_off[i + 1]]
            crossing[e[:, 0], e[:, 1]] = True
            crossing[e[:, 1], e[:, 0]] = True
            hidden, covered[i], _ = classify_caps(caps, crossing)
            caps_of[i] = caps.take(np.flatnonzero(~hidden))
            tags_of[i] = tags_of[i][~hidden]
            renumber = np.where(hidden, -1, np.cumsum(~hidden) - 1)
            sl = slice(offsets[i], offsets[i + 1])
            slot_of_row[sl] = renumber[slot_of_row[sl]]

        slots = _triple_slots(circ, row_of, slot_of_row)
        edge_on = solved & (slots >= 0).all(axis=2)
        has_vertex = (~solved | edge_on).all(axis=1)
        points = tri.points(has_vertex)

        raw_pts = points.reshape(-1, 3)
        raw_triple = np.repeat(np.flatnonzero(has_vertex), 2)
        label, atoms_key, owner = _clusters_by_owner(
            cluster_points(raw_pts, TAU_C), triples[raw_triple], n
        )
        n_clusters = len(owner)
        reps = np.zeros((n_clusters, 3))
        np.add.at(reps, label, raw_pts)
        reps /= np.bincount(label, minlength=n_clusters)[:, None]
        label_of = np.full((len(triples), 2), -1, dtype=int)
        label_of[has_vertex] = label.reshape(-1, 2)

        cap_off = np.cumsum([0, *(len(c) for c in caps_of)])
        sph_off, sph_tri, sph_edges = _sphere_incidence(
            triples, slots, edge_on, n_solve
        )
        sphere = np.repeat(np.arange(n_solve), np.diff(sph_off))
        gcap = sph_edges + cap_off[sphere, None]
        cluster = label_of[sph_tri]
        with_vertex = cluster[:, 0] >= 0
        incidence = np.unique(
            (
                cluster[with_vertex, :, None] * cap_off[-1]
                + gcap[with_vertex, None, :]
            ).ravel()
        )
        n_components = _components_per_sphere(cap_off, gcap)
        accessible = _accessible(
            Caps.concat(caps_of), cap_off, coords, reps, owner, incidence
        )

        probe_ids = np.flatnonzero(accessible)
        probe_map = np.full(n_clusters, -1, dtype=int)
        probe_map[probe_ids] = np.arange(len(probe_ids))
        probe_offsets, probe_atoms = _probe_atoms(n, atoms_key, probe_ids)

        arrangements: list[Arrangement] = [
            covered_arrangement(caps) for caps in caps_of
        ]
        arc_parts = []
        n_active_arcs = 0
        for i in np.flatnonzero(~covered):
            caps = caps_of[i]
            rows = slice(sph_off[i], sph_off[i + 1])
            vrows = np.flatnonzero(with_vertex[rows]) + sph_off[i]
            local, inv = np.unique(cluster[vrows], return_inverse=True)
            local_reps = reps[local] - coords[i]
            local_reps /= np.linalg.norm(local_reps, axis=1, keepdims=True)
            excused = np.zeros((len(local), len(caps)), dtype=bool)
            excused[
                inv.reshape(-1, 2)[:, :, None], sph_edges[vrows, None, :]
            ] = True
            tags = tags_of[i]
            circ, sign = tags >> 1, 1.0 - 2.0 * (tags & 1)
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
            arrangements[i] = arr
            arc_parts.append(_torus_arcs(arr, tags, probe_map[local]))
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
    are sorted by sphere; ``offsets`` delimits each sphere's slice. A row's
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
    caps = Caps(axis, a / r, circles.radius[tag >> 1] / r)
    row_of = np.full(2 * n_pairs, -1, dtype=int)
    row_of[tag] = np.arange(len(tag))
    return offsets, caps, tag, row_of


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


def _overlapping_triples(n, pairs, nbr_off, nbr_flat, n_first):
    """Triples ``(i, j, k)`` with ``i < j < k`` and all three pairs
    overlapping, for ``(i, j)`` among the first ``n_first`` sorted pairs,
    with the pair ids of ``(i, j)``, ``(i, k)`` and ``(j, k)``."""
    keys = pairs[:, 0] * n + pairs[:, 1]
    i, j = pairs[:n_first, 0], pairs[:n_first, 1]
    nbr_key = np.repeat(np.arange(n), np.diff(nbr_off)) * n + nbr_flat
    lo = np.searchsorted(nbr_key, i * n + j, "right")
    count = nbr_off[i + 1] - lo
    p = np.repeat(np.arange(n_first), count)
    k = nbr_flat[np.repeat(lo, count) + _within(count)]
    key = j[p] * n + k
    pos = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
    is_pair = keys[pos] == key
    p, k, pos_jk = p[is_pair], k[is_pair], pos[is_pair]
    pos_ik = np.searchsorted(keys, i[p] * n + k)
    return np.column_stack([i[p], j[p], k]), np.column_stack(
        [p, pos_ik, pos_jk]
    )


def _triple_candidates(coords, sas, circles, triples, pid):
    """Candidates among the overlapping ``triples`` (``(i, j)`` a circle,
    ``pid`` the pair ids of ``(i, j)``, ``(i, k)``, ``(j, k)``), kept iff
    circle ``(i, j)`` crosses sphere ``k`` (``h^2 > 0``). Each triple is
    intersected once so that all three spheres see identical points.
    """
    circ, k = pid[:, 0], triples[:, 2]
    t, rl = circles.centre[circ], circles.radius[circ]
    e1, e2 = circles.e1[circ], circles.e2[circ]
    w = t - coords[k]
    g = (sas[k] ** 2 - np.einsum("ij,ij->i", w, w) - rl * rl) / (2.0 * rl)
    a = np.einsum("ij,ij->i", w, e1)
    b = np.einsum("ij,ij->i", w, e2)
    hsq = a * a + b * b - g * g
    ok = hsq > 0.0
    return _TripleCandidates(
        triples[ok],
        pid[ok],
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


def _sphere_incidence(triples, slots, mask, n_solve):
    """Every (sphere, triple) incidence selected by ``mask`` (t, 3),
    sorted by sphere: ``(offsets, triple, the triple's two cap slots
    there)``."""
    sphere = triples[mask]
    order = np.argsort(sphere, kind="stable")
    tri = np.repeat(np.arange(len(triples)), mask.sum(axis=1))
    offsets = np.searchsorted(sphere[order], np.arange(n_solve + 1))
    return offsets, tri[order], slots[mask][order]


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
    dirs = reps - coords[owner]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    count = np.diff(cap_off)[owner]
    start = np.cumsum(count) - count
    cluster = np.repeat(np.arange(len(owner)), count)
    cap = cap_off[owner[cluster]] + _within(count)
    inside = (
        np.einsum("ij,ij->i", dirs[cluster], caps.axis[cap]) > caps.cos_a[cap]
    )

    inc_cluster, inc_cap = np.divmod(incidence, n_caps)
    sphere_of = np.repeat(np.arange(len(cap_off) - 1), np.diff(cap_off))
    own = sphere_of[inc_cap] == owner[inc_cluster]
    inc_cluster, inc_cap = inc_cluster[own], inc_cap[own]
    inside[start[inc_cluster] + inc_cap - cap_off[owner[inc_cluster]]] = False
    return np.bincount(cluster, inside, minlength=len(owner)) == 0


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
    rl = np.sqrt(np.maximum(ri * ri - a * a, 0.0))
    centre = coords[i] + a[:, None] * axis
    e1 = any_perpendicular(axis)
    e2 = cross(axis, e1)
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


def _torus_arcs(arr, tags, probe_of_local) -> TorusArcs:
    """Arcs of a sphere on circles it is the smaller sphere of (``tags``
    per cap, side bit 0).

    Those caps share the circle frame, so ``phi`` carries over as is; the
    larger sphere reports nothing. A ``-1`` appended to the probe map
    lets full circles (``-1`` ends) read back ``-1``.
    """
    arcs = arr.arcs
    tag = tags[arcs.cap]
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
    caps: Caps
    area: float


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
    tangents = cross(circles.axis[circ], radial)
    tangents *= (sign / np.linalg.norm(tangents, axis=1))[:, None]
    order = np.argsort(probe, kind="stable")
    offsets = np.searchsorted(probe[order], np.arange(len(sas.probes) + 1))
    return offsets, tangents[order]


def _triples(sas: SasGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the three-atom probes and their contact triangles
    ``(k, 3, 3)``."""
    idx = np.flatnonzero(np.diff(sas.probe_offsets) == 3)
    atoms = sas.probe_atoms[sas.probe_offsets[idx, None] + np.arange(3)]
    return idx, sas.coords[atoms]


def _segment_distance(x, p, q) -> np.ndarray:
    e = q - p
    t = np.einsum("ij,ij->i", x - p, e) / np.einsum("ij,ij->i", e, e)
    return np.linalg.norm(x - p - np.clip(t, 0.0, 1.0)[:, None] * e, axis=1)


@dataclass
class ProbeHeights:
    """Per probe: whether its ball reaches its contact triangle (``low``),
    and for low three-atom probes (``planar``) the unit normal of the
    contact plane pointing from the probe toward the triangle with
    ``cos b = h / rp`` for the plane distance ``h < rp``. Probes with more
    than three atoms count as low and are never ``planar``."""

    low: np.ndarray
    planar: np.ndarray
    normal: np.ndarray
    cos_b: np.ndarray
    sin_b: np.ndarray


def _probe_heights(sas: SasGeometry) -> ProbeHeights:
    """Only low faces can be cut, and only low probes can cut, and only in
    the cap of the face beyond the contact plane (ALGORITHMS.md, "Which
    probes can cut a face")."""
    n = len(sas.probes)
    low = np.ones(n, dtype=bool)
    planar = np.zeros(n, dtype=bool)
    normal = np.zeros((n, 3))
    cos_b, sin_b = np.zeros(n), np.zeros(n)
    idx, tri = _triples(sas)
    if len(idx) == 0:
        return ProbeHeights(low, planar, normal, cos_b, sin_b)
    x = sas.probes[idx]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    nrm = cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    signed = np.einsum("ij,ij->i", x - a, nrm)
    inside = np.ones(len(idx), dtype=bool)
    edge_dist = np.empty((3, len(idx)))
    for m, (p, q) in enumerate(((a, b), (b, c), (c, a))):
        inside &= np.einsum("ij,ij->i", cross(q - p, x - p), nrm) >= 0
        edge_dist[m] = _segment_distance(x, p, q)
    h = np.abs(signed)
    dist = np.where(inside, h, edge_dist.min(axis=0))
    is_low = dist < sas.rp
    low[idx] = is_low
    idx, signed, nrm, h = idx[is_low], signed[is_low], nrm[is_low], h[is_low]
    planar[idx] = True
    normal[idx] = -np.sign(signed)[:, None] * nrm
    cos_b[idx] = h / sas.rp
    sin_b[idx] = np.sqrt(1.0 - cos_b[idx] * cos_b[idx])
    return ProbeHeights(low, planar, normal, cos_b, sin_b)


def _triangle_areas(tangents: np.ndarray, rp: float) -> np.ndarray:
    """Areas of the spherical triangles ``{d : d . t_m <= 0}`` bounded by
    the hemispheres of tangent triples ``(k, 3, 3)``: ``rp^2`` times the
    excess ``2 pi - sum of the angles between the three normals``."""
    total = np.zeros(len(tangents))
    for m in range(3):
        t1, t2 = tangents[:, m], tangents[:, (m + 1) % 3]
        total += np.arctan2(
            np.linalg.norm(cross(t1, t2), axis=1),
            np.einsum("ij,ij->i", t1, t2),
        )
    return rp * rp * (2.0 * math.pi - total)


@dataclass
class FaceTriangles:
    """Per probe with exactly three linearly independent departure
    tangents: the hemisphere normals ``tangents`` and the ``corners`` of the
    spherical triangle ``{d : d . t_m <= 0}``, corner ``m`` opposite edge
    ``m`` (the edge on the great circle of ``t_m``). Probes with coplanar
    tangents have no corners and take the unfiltered path."""

    triangular: np.ndarray
    tangents: np.ndarray
    corners: np.ndarray


def _face_triangles(n: int, dep_off, dep_t) -> FaceTriangles:
    triangular = np.diff(dep_off) == 3
    tangents = np.zeros((n, 3, 3))
    corners = np.zeros((n, 3, 3))
    idx = np.flatnonzero(triangular)
    t = dep_t[dep_off[idx, None] + np.arange(3)]
    independent = (
        np.einsum("ij,ij->i", t[:, 0], cross(t[:, 1], t[:, 2])) != 0.0
    )
    triangular[idx[~independent]] = False
    idx, t = idx[independent], t[independent]
    tangents[idx] = t
    for m in range(3):
        v = cross(t[:, (m + 1) % 3], t[:, (m + 2) % 3])
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= -np.sign(np.einsum("ij,ij->i", t[:, m], v))[:, None]
        corners[idx, m] = v
    return FaceTriangles(triangular, tangents, corners)


def _meets_triangle(tri: FaceTriangles, probe, axis, cos_a):
    """Whether the cap ``(axis, cos a)`` on the sphere of ``probe`` meets
    its spherical triangle: the axis is inside, or within ``a`` of an edge
    arc, or within ``a`` of a corner. The squared cosine of the angle from
    the axis to its foot on the great circle of ``t_m`` is
    ``1 - (axis . t_m)^2``; the foot lies on the arc iff it is on the arc's
    side of the plane through the corners' bisector, and since the corners
    are perpendicular to ``t_m`` the foot's component along the bisector is
    the axis's own. All comparisons are on squared cosines, no root is
    taken. Probes without a triangle pass."""
    t, c = tri.tangents[probe], tri.corners[probe]
    along = np.einsum("kmj,kj->km", t, axis)
    inside = np.all(along <= 0.0, axis=1)
    near = np.any(np.einsum("kmj,kj->km", c, axis) > cos_a[:, None], axis=1)
    cos2_a = cos_a * cos_a
    for m in range(3):
        p, q = c[:, (m + 1) % 3], c[:, (m + 2) % 3]
        mid = p + q
        axis_mid = np.einsum("ij,ij->i", axis, mid)
        p_mid = np.einsum("ij,ij->i", p, mid)
        cos2_foot = 1.0 - along[:, m] * along[:, m]
        on_arc = (axis_mid >= 0.0) & (
            axis_mid * axis_mid >= cos2_foot * p_mid * p_mid
        )
        near |= on_arc & (cos2_foot > cos2_a)
    return ~tri.triangular[probe] | inside | near


def _meets_plane_cap(hts: ProbeHeights, probe, axis, cos_a, sin_a):
    """Whether the cap ``(axis, cos a)`` on the sphere of ``probe`` overlaps
    the cap beyond the contact plane, ``d . normal > cos b``: the angle
    between the axes is below ``a + b``. Probes without a plane pass."""
    threshold = cos_a * hts.cos_b[probe] - sin_a * hts.sin_b[probe]
    inner = np.einsum("ij,ij->i", axis, hts.normal[probe])
    return ~hts.planar[probe] | (inner > threshold)


def _probe_pair_caps(
    probes, rp: float, hts: ProbeHeights, tri: FaceTriangles
) -> tuple[np.ndarray, Caps]:
    """Caps cut into every probe sphere by the other probes within
    ``2 rp``, as ``(offsets, caps)`` sorted by probe; each pair is measured
    once and read from both sides with opposite axes. Cutting is
    symmetric, so a pair is dropped for both probes as soon as one side
    cannot be cut: a pair is kept iff its ``cos`` is below 1, the same
    value the cap carries, both probes are low, and each cap meets the
    other probe's beyond-plane cap and spherical triangle."""
    pairs = cKDTree(probes).query_pairs(2.0 * rp, output_type="ndarray")
    if len(pairs) == 0:
        pairs = np.empty((0, 2), dtype=int)
    diff = probes[pairs[:, 1]] - probes[pairs[:, 0]]
    dist = np.linalg.norm(diff, axis=1)
    cos_a = dist / (2.0 * rp)
    close = (cos_a < 1.0) & hts.low[pairs[:, 0]] & hts.low[pairs[:, 1]]
    pairs, diff, dist, cos_a = (
        pairs[close],
        diff[close],
        dist[close],
        cos_a[close],
    )
    axis = diff / dist[:, None]
    sin_a = np.sqrt(1.0 - cos_a * cos_a)
    keep = _meets_plane_cap(hts, pairs[:, 0], axis, cos_a, sin_a)
    keep &= _meets_plane_cap(hts, pairs[:, 1], -axis, cos_a, sin_a)
    idx = np.flatnonzero(keep)
    keep[idx] = _meets_triangle(tri, pairs[idx, 0], axis[idx], cos_a[idx])
    idx = idx[keep[idx]]
    keep[idx] = _meets_triangle(tri, pairs[idx, 1], -axis[idx], cos_a[idx])
    src = pairs[keep].T.ravel()
    axis = np.concatenate([axis[keep], -axis[keep]])
    cos_a, sin_a = np.tile(cos_a[keep], 2), np.tile(sin_a[keep], 2)
    order = np.argsort(src, kind="stable")
    offsets = np.searchsorted(src[order], np.arange(len(probes) + 1))
    return offsets, Caps(axis[order], cos_a[order], sin_a[order])


def _concave_faces(sas: SasGeometry) -> list[ConcaveFace]:
    probes = sas.probes
    if len(probes) == 0:
        return []
    n_active = sas.n_active_probes
    hts = _probe_heights(sas)
    dep_off, dep_t = _departure_caps(sas)
    tri = _face_triangles(len(probes), dep_off, dep_t)
    plain = ~hts.low[:n_active] & tri.triangular[:n_active]
    areas = np.zeros(n_active)
    areas[plain] = _triangle_areas(tri.tangents[:n_active][plain], sas.rp)
    nbr_off, nbr_caps = _probe_pair_caps(probes, sas.rp, hts, tri)
    faces = []
    for q in range(n_active):
        atoms = sas.atoms_of(q)
        contacts = sas.coords[atoms] - probes[q]
        contacts /= np.linalg.norm(contacts, axis=1, keepdims=True)
        tangents = dep_t[dep_off[q] : dep_off[q + 1]]
        hemispheres = Caps(
            tangents, np.zeros(len(tangents)), np.ones(len(tangents))
        )
        if plain[q]:
            faces.append(
                ConcaveFace(q, atoms, contacts, hemispheres, areas[q])
            )
            continue
        caps = Caps.concat(
            [hemispheres, nbr_caps.take(slice(nbr_off[q], nbr_off[q + 1]))]
        )
        arr = solve_caps(sas.rp, caps)
        faces.append(ConcaveFace(q, atoms, contacts, arr.caps, arr.area))
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
    rl = float(np.sqrt(np.maximum(R1 * R1 - a * a, 0.0)))
    convex = (
        2 * math.pi * (r1 * r1 * (1 + a / R1) + r2 * r2 * (1 + (d - a) / R2))
    )
    _, integral = saddle_ranges(rl, rp, a, d - a, R1, R2)
    return convex, float(2.0 * math.pi * rp * integral.sum()), 0.0

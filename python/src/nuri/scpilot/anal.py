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

from .aos import scalars, vectors
from .arrangement import (
    TAU_C,
    Arrangement,
    ArrangementProblem,
    Cap,
    DegenerateGeometryError,
    any_perpendicular,
    cap_components,
    classify_caps,
    cluster_points,
    covered_arrangement,
    cross,
    solve,
    solve_caps,
)


@dataclass
class Circle:
    """Probe-centre circle of the overlapping spheres ``i < j``: centre
    ``c_i + a axis`` on the unit axis from ``i`` to ``j``, radius ``rl``,
    frame ``(e1, e2, axis)``, and the centre distance ``d``."""

    i: int
    j: int
    centre: np.ndarray
    axis: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    rl: float
    a: float
    d: float


@dataclass
class TorusArc:
    """Accessible arc of probe circle ``circle`` from probe ``v_beg`` over
    ``dphi`` to probe ``v_end`` in the circle frame; ``-1`` ends mark a
    full circle."""

    circle: int
    v_beg: int
    v_end: int
    phi_beg: float
    dphi: float


@dataclass
class Probe:
    """Accessible SAS vertex: the probe centre ``pos`` resting on the
    sorted ``atoms``, the smallest of which is the ``owner``, and the unit
    ``tangents`` of the accessible arcs leaving it."""

    pos: np.ndarray
    owner: int
    atoms: np.ndarray
    tangents: list[np.ndarray] = field(default_factory=list)


@dataclass
class Partner:
    """An overlap partner of a sphere and the id of their pair."""

    atom: int
    pair: int


@dataclass
class Sphere:
    """One atom sphere below ``n_enum`` during the SAS build: its overlap
    partners sorted by atom, one cap per partner in the same order (hidden
    caps removed after classification), the ``(triple, corner)``
    incidences of the crossing points on it, whether two of its caps cover
    it, and, below ``n_solve``, its arrangement."""

    partners: list[Partner] = field(default_factory=list)
    caps: list[Cap] = field(default_factory=list)
    incident: list[tuple[int, int]] = field(default_factory=list)
    covered: bool = False
    arrangement: Arrangement | None = None

    def slot(self, atom: int) -> int:
        """Index of the cap made with ``atom``, ``-1`` if there is none:
        a binary search in the partner-sorted caps."""
        lo, hi = 0, len(self.caps)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.caps[mid].partner < atom:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.caps) and self.caps[lo].partner == atom:
            return lo
        return -1


_OTHERS = ((1, 2), (0, 2), (0, 1))


@dataclass
class Triple:
    """Mutually overlapping spheres ``i < j < k`` whose circle ``(i, j)``
    crosses sphere ``k`` in two points: the pair ids of ``(i, j)``,
    ``(i, k)``, ``(j, k)`` and the intersection numbers ``g, a, b, h`` in
    the circle frame. ``edge_on[m]`` says whether both caps of the triple
    survive hiding on sphere ``atoms[m]`` (true above ``n_enum``); a
    triple with every edge on has a vertex, and ``cluster`` then holds the
    cluster ids of its two points."""

    atoms: tuple[int, int, int]
    pairs: tuple[int, int, int]
    g: float
    a: float
    b: float
    h: float
    edge_on: list[bool] = field(default_factory=lambda: [True, True, True])
    has_vertex: bool = False
    cluster: list[int] = field(default_factory=lambda: [-1, -1])

    def others(self, corner: int) -> tuple[int, int]:
        m0, m1 = _OTHERS[corner]
        return self.atoms[m0], self.atoms[m1]


def prepare(coords, radii, rp, active=None):
    """Order atoms as ``[active | need | shell | occluders]`` and drop
    balls without surface.

    ``need`` holds the active atoms and every atom within ``2 TAU_C`` of
    touching one, ``shell`` the same neighbourhood of ``need``: every probe
    that can cut the face of a probe on an active atom has a host in ``shell``
    (ALGORITHMS.md, "Which probes can cut a face", host overlap), which
    needs ``R_min^2 >= 2 rp^2 + 2 TAU_C R_max`` for the SAS radii, about
    ``(sqrt 2 - 1) rp`` for the atom radii.

    Returns ``(order, n_active, n_solve, n_enum, pairs, d)``: ``order``
    maps new to old indices, ``pairs`` (``i < j``, sorted) are all
    overlapping pairs in new indices with their distances. Everything
    downstream assumes this shape: no contained or coincident balls, no
    two caps of one sphere on the same circle, ``rp > 0``, and a pair
    touches a sphere with caps iff ``i < n_enum``.
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if rp <= 0.0 or np.any(radii <= 0.0):
        raise ValueError("probe and atom radii must be positive")
    n = len(coords)
    sas = radii + rp
    if sas.min() ** 2 < 2.0 * rp * rp + 2.0 * TAU_C * sas.max():
        raise ValueError("atom radii must be at least (sqrt 2 - 1) rp")
    near, d_near = near_pairs(coords, sas)
    i, j = near[:, 0], near[:, 1]
    overlapping = d_near < sas[i] + sas[j] - TAU_C
    pairs, d = near[overlapping], d_near[overlapping]
    inside = contained(n, pairs, d, sas)
    pairs, d = _without(pairs, d, inside)
    inside |= shared_circle_middles(coords, sas, pairs, d)
    pairs, d = _without(pairs, d, inside)
    near, _ = _without(near, d_near, inside)
    active = (
        np.ones(n, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool).copy()
    )
    active &= ~inside

    i, j = near[:, 0], near[:, 1]
    need = _neighbourhood(active, i, j)
    shell = _neighbourhood(need, i, j)

    rank = np.where(
        active,
        0,
        np.where(need, 1, np.where(shell, 2, np.where(inside, 4, 3))),
    )
    order = np.argsort(rank, kind="stable")[: int((rank < 4).sum())]
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(len(order))
    pairs = np.sort(inv[pairs], axis=1)
    o = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return (
        order,
        int(active.sum()),
        int(need.sum()),
        int(shell.sum()),
        pairs[o],
        d[o],
    )


def _neighbourhood(mask, i, j):
    """``mask`` together with every atom paired with a masked atom."""
    out = mask.copy()
    out[j[mask[i]]] = True
    out[i[mask[j]]] = True
    return out


def near_pairs(coords, sas):
    """All pairs ``(i < j)`` whose SAS balls come within ``2 TAU_C`` of
    touching, and their distances.

    The pairs closer than ``R_i + R_j - TAU_C`` are the overlaps that carry
    circles; the rest only matter for the neighbourhoods of
    :func:`prepare`: two hosts of one vertex cluster have contact points
    within the cluster's diameter, at most ``2 TAU_C`` (checked in
    :meth:`SasGeometry.build`), without necessarily sharing a circle.
    """
    tree = cKDTree(coords)
    pairs = tree.query_pairs(
        2.0 * sas.max() + 2.0 * TAU_C, output_type="ndarray"
    )
    if len(pairs) == 0:
        pairs = np.empty((0, 2), dtype=int)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(coords[j] - coords[i], axis=1)
    if np.any(d < 1e-3):
        raise ValueError("coincident atoms")
    near = d <= sas[i] + sas[j] + 2.0 * TAU_C
    return pairs[near], d[near]


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
    triples, pid = _triple_rows(_partner_lists(n, pairs), pairs, len(pairs))
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

    Spheres ``< n_active`` own surface, spheres ``< n_solve`` have
    arrangements and torus arcs, spheres ``< n_enum`` have caps and host
    vertices. Every other sphere only occludes.
    """

    coords: np.ndarray
    radii: np.ndarray
    rp: float
    sas: np.ndarray
    n_active: int
    n_solve: int
    n_enum: int
    circles: list[Circle]
    spheres: list[Sphere]
    probes: list[Probe]
    arcs: list[TorusArc]
    n_active_circles: int
    n_active_probes: int
    n_active_arcs: int

    @property
    def arrangements(self) -> list[Arrangement]:
        return [s.arrangement for s in self.spheres[: self.n_solve]]

    @property
    def sas_area(self) -> np.ndarray:
        return scalars(self.arrangements, "area")

    @classmethod
    def from_atoms(
        cls, coords, radii, rp: float, active=None
    ) -> tuple[SasGeometry, np.ndarray]:
        """Prepare, permute and build; also returns the new-to-old atom
        index map."""
        order, n_active, n_solve, n_enum, pairs, d = prepare(
            coords, radii, rp, active
        )
        coords = np.asarray(coords, dtype=float)[order]
        radii = np.asarray(radii, dtype=float)[order]
        geometry = cls.build(
            coords, radii, rp, pairs, d, n_active, n_solve, n_enum
        )
        return geometry, order

    @classmethod
    def build(
        cls,
        coords,
        radii,
        rp: float,
        pairs,
        d,
        n_active: int,
        n_solve: int,
        n_enum: int,
    ) -> SasGeometry:
        sas = radii + rp
        n_circ = int(np.searchsorted(pairs[:, 0], n_enum))
        circles = _circles(coords, sas, pairs[:n_circ], d[:n_circ])
        partners = _partner_lists(len(coords), pairs)
        spheres = [Sphere(p) for p in partners[:n_enum]]
        triples = _crossing_triples(coords, sas, circles, partners, pairs)
        for t, triple in enumerate(triples):
            for corner, atom in enumerate(triple.atoms):
                if atom < n_enum:
                    spheres[atom].incident.append((t, corner))
        _sphere_caps(circles, sas, spheres)
        _hide_caps(spheres, triples)
        raw_pts, raw_tri = _vertex_points(triples, circles)
        clusters = _clusters(coords, spheres, triples, raw_pts, raw_tri)
        probes, probe_map = _probes(clusters)

        arcs: list[TorusArc] = []
        n_active_arcs = 0
        for s in range(n_solve):
            sphere = spheres[s]
            if sphere.covered:
                sphere.arrangement = covered_arrangement(sphere.caps)
                continue
            problem, local = _sphere_problem(
                coords[s], circles, sphere, triples, clusters
            )
            sphere.arrangement = solve(sas[s], problem)
            part = _torus_arcs(sphere, probe_map[local])
            arcs.extend(part)
            if s < n_active:
                n_active_arcs += len(part)
        _departure_tangents(circles, probes, arcs)

        return cls(
            coords,
            radii,
            rp,
            sas,
            n_active,
            n_solve,
            n_enum,
            circles,
            spheres,
            probes,
            arcs,
            int(np.searchsorted(scalars(circles, "i", int), n_active)),
            int(np.searchsorted(scalars(probes, "owner", int), n_active)),
            n_active_arcs,
        )


def _circles(coords, sas, pairs, d) -> list[Circle]:
    i, j = pairs[:, 0], pairs[:, 1]
    ri, rj = sas[i], sas[j]
    axis = (coords[j] - coords[i]) / d[:, None]
    a = (d * d + ri * ri - rj * rj) / (2.0 * d)
    rl = np.sqrt(np.maximum(ri * ri - a * a, 0.0))
    centre = coords[i] + a[:, None] * axis
    e1 = any_perpendicular(axis)
    e2 = cross(axis, e1)
    return [
        Circle(
            int(i[k]),
            int(j[k]),
            centre[k],
            axis[k],
            e1[k],
            e2[k],
            float(rl[k]),
            float(a[k]),
            float(d[k]),
        )
        for k in range(len(pairs))
    ]


def _partner_lists(n: int, pairs) -> list[list[Partner]]:
    """Overlap partners of every atom with their pair ids; ``pairs`` sorted
    by ``(i, j)`` makes every list sorted by atom."""
    partners: list[list[Partner]] = [[] for _ in range(n)]
    for q, (i, j) in enumerate(pairs.tolist()):
        partners[i].append(Partner(j, q))
        partners[j].append(Partner(i, q))
    return partners


def _triple_rows(partners: list[list[Partner]], pairs, n_first: int):
    """Triples ``(i, j, k)`` with ``i < j < k`` and all three pairs
    overlapping, for ``(i, j)`` among the first ``n_first`` sorted pairs,
    with the pair ids of ``(i, j)``, ``(i, k)`` and ``(j, k)``: circle by
    circle, ``k`` runs over the partners of ``i`` after ``j`` that are also
    partners of ``j`` (binary search in ``j``'s list)."""
    atom_of = [scalars(p, "atom", int) for p in partners]
    pair_of = [scalars(p, "pair", int) for p in partners]
    rows = []
    for q in range(n_first):
        i, j = int(pairs[q, 0]), int(pairs[q, 1])
        start = int(np.searchsorted(atom_of[i], j)) + 1
        ks, q_ik = atom_of[i][start:], pair_of[i][start:]
        pos = np.minimum(np.searchsorted(atom_of[j], ks), len(atom_of[j]) - 1)
        found = atom_of[j][pos] == ks
        for k, qik, qjk in zip(ks[found], q_ik[found], pair_of[j][pos[found]]):
            rows.append((i, j, int(k), q, int(qik), int(qjk)))
    arr = np.array(rows, dtype=int).reshape(-1, 6)
    return arr[:, :3], arr[:, 3:]


def _crossing_triples(coords, sas, circles, partners, pairs) -> list[Triple]:
    """Triples whose circle ``(i, j)`` crosses sphere ``k`` (``h^2 > 0``),
    intersected once so that all three spheres see identical points."""
    atoms, pids = _triple_rows(partners, pairs, len(circles))
    circ, k = pids[:, 0], atoms[:, 2]
    t, rl = vectors(circles, "centre")[circ], scalars(circles, "rl")[circ]
    e1, e2 = vectors(circles, "e1")[circ], vectors(circles, "e2")[circ]
    w = t - coords[k]
    g = (sas[k] ** 2 - np.einsum("ij,ij->i", w, w) - rl * rl) / (2.0 * rl)
    a = np.einsum("ij,ij->i", w, e1)
    b = np.einsum("ij,ij->i", w, e2)
    hsq = a * a + b * b - g * g
    ok = np.flatnonzero(hsq > 0.0)
    h = np.sqrt(hsq[ok])
    return [
        Triple(
            tuple(atoms[r].tolist()),
            tuple(pids[r].tolist()),
            float(g[r]),
            float(a[r]),
            float(b[r]),
            float(h_r),
        )
        for r, h_r in zip(ok, h)
    ]


def _sphere_caps(circles: list[Circle], sas, spheres: list[Sphere]) -> None:
    """One cap per partner, in partner order: the circle ``(s, j)`` cuts
    sphere ``s`` with its axis (side 0), the circle ``(i, s)`` with the
    opposite axis (side 1)."""
    for s, sphere in enumerate(spheres):
        for p in sphere.partners:
            c = circles[p.pair]
            if p.atom < s:
                cap = Cap(-c.axis, (c.d - c.a) / sas[s], c.rl / sas[s])
                cap.side = 1
            else:
                cap = Cap(c.axis, c.a / sas[s], c.rl / sas[s])
            cap.partner, cap.circle = p.atom, p.pair
            sphere.caps.append(cap)


def _hide_caps(spheres: list[Sphere], triples: list[Triple]) -> None:
    """Classify every sphere's caps against the crossing graph of its
    incident triples, drop the hidden caps, and record on each triple
    whether both of its caps survive on each of its spheres."""
    for sphere in spheres:
        m = len(sphere.caps)
        crossing = np.zeros((m, m), dtype=bool)
        slots = []
        for t, corner in sphere.incident:
            a, b = triples[t].others(corner)
            sa, sb = sphere.slot(a), sphere.slot(b)
            crossing[sa, sb] = crossing[sb, sa] = True
            slots.append((sa, sb))
        hidden, sphere.covered, _ = classify_caps(sphere.caps, crossing)
        keep = ~hidden
        for (t, corner), (sa, sb) in zip(sphere.incident, slots):
            triples[t].edge_on[corner] = bool(keep[sa] and keep[sb])
        sphere.caps = [c for c, k in zip(sphere.caps, keep) if k]
    for triple in triples:
        triple.has_vertex = all(triple.edge_on)


def _vertex_points(triples: list[Triple], circles: list[Circle]):
    """Both intersection points of every triple with a vertex, as
    ``(raw_pts (2t, 3), raw_tri (2t,))`` with the two points of a triple
    consecutive: ``phi_0 -/+ acos(g / amp)`` written without inverse
    trig."""
    idx = [t for t, triple in enumerate(triples) if triple.has_vertex]
    if not idx:
        return np.empty((0, 3)), np.empty(0, dtype=int)
    sel = [triples[t] for t in idx]
    on = [circles[triple.pairs[0]] for triple in sel]
    t, rl = vectors(on, "centre"), scalars(on, "rl")[:, None]
    e1, e2 = vectors(on, "e1"), vectors(on, "e2")
    g, a, b, h = (scalars(sel, f) for f in ("g", "a", "b", "h"))
    amp2 = a * a + b * b
    pts = []
    for sign in (-1.0, 1.0):
        cos_phi = (a * g + sign * b * h) / amp2
        sin_phi = (b * g - sign * a * h) / amp2
        pts.append(t + rl * (cos_phi[:, None] * e1 + sin_phi[:, None] * e2))
    return np.stack(pts, axis=1).reshape(-1, 3), np.repeat(idx, 2)


@dataclass
class _Cluster:
    rep: np.ndarray
    owner: int
    atoms: np.ndarray
    accessible: bool


def _clusters(coords, spheres, triples, raw_pts, raw_tri) -> list[_Cluster]:
    """Cluster the raw points at ``TAU_C``, decide every cluster on its
    owner sphere, sort the clusters by owner and write their ids into the
    triples."""
    label = cluster_points(raw_pts, TAU_C)
    n_clusters = int(label.max()) + 1 if len(label) else 0
    members: list[list[int]] = [[] for _ in range(n_clusters)]
    for r, g in enumerate(label.tolist()):
        members[g].append(r)

    clusters = []
    for mem in members:
        pts = raw_pts[mem]
        rep = pts.mean(axis=0)
        if np.any(np.linalg.norm(pts - rep, axis=1) > TAU_C):
            raise DegenerateGeometryError("vertex cluster wider than 2 TAU_C")
        tris = [triples[raw_tri[r]] for r in mem]
        atoms = np.unique([a for triple in tris for a in triple.atoms])
        owner = int(atoms[0])
        accessible = _accessible(coords, spheres[owner], owner, rep, tris)
        clusters.append(_Cluster(rep, owner, atoms, accessible))

    order = np.argsort(scalars(clusters, "owner", int), kind="stable")
    rank = np.empty(n_clusters, dtype=int)
    rank[order] = np.arange(n_clusters)
    for r, g in enumerate(label.tolist()):
        triples[raw_tri[r]].cluster[r % 2] = int(rank[g])
    return [clusters[g] for g in order]


def _accessible(coords, sphere: Sphere, owner: int, rep, tris) -> bool:
    """Whether the cluster at ``rep`` lies outside every cap of its owner
    sphere except the caps its member triples make there.

    The owner is the first atom of every member triple that contains it,
    so those caps are the slots of the triple's other two atoms. Every
    ball that could contain a point of the sphere overlaps it and therefore
    is a cap there (or nested inside one), so this equals the test against
    all SAS balls.
    """
    direction = rep - coords[owner]
    direction /= np.linalg.norm(direction)
    inside = np.einsum(
        "ij,j->i", vectors(sphere.caps, "axis"), direction
    ) > scalars(sphere.caps, "cos_a")
    for triple in tris:
        if triple.atoms[0] == owner:
            for atom in triple.atoms[1:]:
                inside[sphere.slot(atom)] = False
    return not inside.any()


def _probes(clusters: list[_Cluster]) -> tuple[list[Probe], np.ndarray]:
    """Accessible clusters in owner order, and the cluster -> probe map
    (``-1`` for inaccessible clusters)."""
    probes: list[Probe] = []
    probe_map = np.full(len(clusters), -1, dtype=int)
    for g, c in enumerate(clusters):
        if c.accessible:
            probe_map[g] = len(probes)
            probes.append(Probe(c.rep, c.owner, c.atoms))
    return probes, probe_map


def _sphere_problem(centre, circles, sphere: Sphere, triples, clusters):
    """The arrangement problem of one SAS sphere from its incident triples:
    crossing edges from the incidences whose caps both survived, vertices
    and their excused caps from those with a vertex, frames from the
    circles of its caps. Returns it with the cluster id of every local
    vertex."""
    m = len(sphere.caps)
    crossing = np.zeros((m, m), dtype=bool)
    incidences = []
    for t, corner in sphere.incident:
        triple = triples[t]
        if not triple.edge_on[corner]:
            continue
        a, b = triple.others(corner)
        sa, sb = sphere.slot(a), sphere.slot(b)
        crossing[sa, sb] = crossing[sb, sa] = True
        if triple.has_vertex:
            incidences.append((triple.cluster, (sa, sb)))

    local = sorted({g for cl, _ in incidences for g in cl})
    index = {g: k for k, g in enumerate(local)}
    excused = np.zeros((len(local), m), dtype=bool)
    for cl, (sa, sb) in incidences:
        for g in cl:
            excused[index[g], sa] = excused[index[g], sb] = True
    reps = np.array([clusters[g].rep for g in local]).reshape(-1, 3) - centre
    reps /= np.linalg.norm(reps, axis=1, keepdims=True)
    accessible = np.array([clusters[g].accessible for g in local], dtype=bool)

    on = [circles[cap.circle] for cap in sphere.caps]
    sign = 1.0 - 2.0 * scalars(sphere.caps, "side")
    problem = ArrangementProblem(
        sphere.caps,
        vectors(on, "e1"),
        sign[:, None] * vectors(on, "e2"),
        crossing,
        cap_components(crossing),
        reps,
        excused,
        accessible,
    )
    return problem, np.array(local, dtype=int)


def _torus_arcs(sphere: Sphere, probe_of_local) -> list[TorusArc]:
    """Arcs of a sphere on its side-0 caps, whose circle frame is the arc
    frame; the other sphere of each circle reports nothing. Full circles
    keep their ``-1`` ends."""

    def probe(v: int) -> int:
        return int(probe_of_local[v]) if v >= 0 else -1

    return [
        TorusArc(
            sphere.caps[arc.cap].circle,
            probe(arc.v_beg),
            probe(arc.v_end),
            arc.phi_beg,
            arc.dphi,
        )
        for arc in sphere.arrangement.arcs
        if sphere.caps[arc.cap].side == 0
    ]


@dataclass
class Saddle:
    """Valid generating-arc angle ranges of one active circle and their
    area integrals.

    The generating arc is parametrised by ``beta``, the angle from the
    inward radial direction; ``beta < 0`` leans toward atom ``i``. Distance
    from the axis is ``rl - rp cos(beta)``, zero at the cusp of a spindle.
    ``ranges`` is ``(2, 2)``: the parts below and above the cusp,
    zero-width where absent; ``integral`` ``(2,)`` is
    ``rl (hi - lo) - rp (sin hi - sin lo)`` per part, so that a saddle
    swept over ``dphi`` has area ``dphi rp sum(integral)``.
    """

    ranges: np.ndarray
    integral: np.ndarray


@dataclass
class ConcaveFace:
    probe: int
    atoms: np.ndarray
    contacts: np.ndarray
    caps: list[Cap]
    area: float


@dataclass
class SesGeometry:
    """Per active atom ``convex_area``, one :class:`Saddle` per active
    circle, ``saddle_area`` per active arc, one face per active probe."""

    sas: SasGeometry
    convex_area: np.ndarray
    saddles: list[Saddle]
    saddle_area: np.ndarray
    concave: list[ConcaveFace] = field(default_factory=list)

    @property
    def rp(self) -> float:
        return self.sas.rp

    @classmethod
    def build(cls, sas: SasGeometry) -> SesGeometry:
        ratio = sas.radii[: sas.n_active] / sas.sas[: sas.n_active]
        convex = sas.sas_area[: sas.n_active] * ratio * ratio
        saddles, saddle_area = _saddles(sas)
        return cls(sas, convex, saddles, saddle_area, _concave_faces(sas))


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


def _saddles(sas: SasGeometry) -> tuple[list[Saddle], np.ndarray]:
    """Ranges and integrals of the active circles and areas of the active
    arcs, both prefixes."""
    circles = sas.circles[: sas.n_active_circles]
    arcs = sas.arcs[: sas.n_active_arcs]
    a, d = scalars(circles, "a"), scalars(circles, "d")
    ranges, integral = saddle_ranges(
        scalars(circles, "rl"),
        sas.rp,
        a,
        d - a,
        sas.sas[scalars(circles, "i", int)],
        sas.sas[scalars(circles, "j", int)],
    )
    saddles = [Saddle(ranges[k], integral[k]) for k in range(len(circles))]
    c = scalars(arcs, "circle", int)
    area = scalars(arcs, "dphi") * sas.rp * integral[c].sum(axis=-1)
    return saddles, area


def _departure_tangents(circles, probes: list[Probe], arcs) -> None:
    """Append to every probe the unit tangents of the accessible arcs
    leaving it.

    The probe rolls away along each such arc, so the half of its sphere
    facing the tangent is swept and cannot be concave face. For an
    ordinary three-atom vertex these are the three side planes of the
    contact triangle; k-fold vertices and single-vertex circles fall out
    of the same rule. The tangent at a probe on circle ``c`` is
    ``axis x radial``; leaving toward decreasing ``phi`` (the ``v_end``
    of an arc) flips it. Full circles have no ends and contribute nothing.
    """
    for arc in arcs:
        circle = circles[arc.circle]
        for v, sign in ((arc.v_beg, 1.0), (arc.v_end, -1.0)):
            if v < 0:
                continue
            probe = probes[v]
            t = np.cross(circle.axis, probe.pos - circle.centre)
            probe.tangents.append(t * (sign / np.linalg.norm(t)))


def _triples(sas: SasGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the three-atom probes and their contact triangles
    ``(k, 3, 3)``."""
    idx = np.array(
        [k for k, p in enumerate(sas.probes) if len(p.atoms) == 3], dtype=int
    )
    atoms = np.array([sas.probes[k].atoms for k in idx], dtype=int)
    return idx, sas.coords[atoms.reshape(-1, 3)]


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
    x = vectors(sas.probes, "pos")[idx]
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


def _face_triangles(probes: list[Probe]) -> FaceTriangles:
    n = len(probes)
    triangular = np.array([len(p.tangents) == 3 for p in probes], dtype=bool)
    tangents = np.zeros((n, 3, 3))
    corners = np.zeros((n, 3, 3))
    idx = np.flatnonzero(triangular)
    t = np.array([probes[k].tangents for k in idx]).reshape(-1, 3, 3)
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
) -> list[list[Cap]]:
    """Caps cut into every probe sphere by the other probes within
    ``2 rp``, one list per probe; each pair is measured once and appended
    to both probes with opposite axes. Cutting is symmetric, so a pair is
    dropped for both probes as soon as one side cannot be cut: a pair is
    kept iff its ``cos`` is below 1, the same value the cap carries, both
    probes are low, and each cap meets the other probe's beyond-plane cap
    and spherical triangle."""
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
    caps: list[list[Cap]] = [[] for _ in range(len(probes))]
    for (p, q), n, c, s in zip(
        pairs[keep].tolist(), axis[keep], cos_a[keep], sin_a[keep]
    ):
        caps[p].append(Cap(n, c, s))
        caps[q].append(Cap(-n, c, s))
    return caps


def _concave_faces(sas: SasGeometry) -> list[ConcaveFace]:
    probes = vectors(sas.probes, "pos")
    if len(probes) == 0:
        return []
    n_active = sas.n_active_probes
    hts = _probe_heights(sas)
    tri = _face_triangles(sas.probes)
    plain = ~hts.low[:n_active] & tri.triangular[:n_active]
    areas = np.zeros(n_active)
    areas[plain] = _triangle_areas(tri.tangents[:n_active][plain], sas.rp)
    nbr_caps = _probe_pair_caps(probes, sas.rp, hts, tri)
    faces = []
    for q in range(n_active):
        atoms = sas.probes[q].atoms
        contacts = sas.coords[atoms] - probes[q]
        contacts /= np.linalg.norm(contacts, axis=1, keepdims=True)
        hemispheres = [Cap(t, 0.0, 1.0) for t in sas.probes[q].tangents]
        if plain[q]:
            faces.append(
                ConcaveFace(q, atoms, contacts, hemispheres, areas[q])
            )
            continue
        caps = hemispheres + nbr_caps[q]
        arr = solve_caps(sas.rp, caps)
        faces.append(ConcaveFace(q, atoms, contacts, arr.caps, arr.area))
    return faces


def ses_area(ses: SesGeometry):
    """Per-atom convex, per-arc saddle, per-face concave areas."""
    concave = np.array([f.area for f in ses.concave])
    return ses.convex_area, ses.saddle_area, concave


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

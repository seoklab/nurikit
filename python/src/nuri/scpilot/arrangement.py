#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Arrangement of spherical caps on a single sphere.

The accessible region is the sphere minus the union of caps. The solver
returns the accessible arcs of every cap circle, the vertices where circles
cross (coincident vertices merged into one), the boundary loops traversed
with the accessible region on the left, and the accessible area from the
Gauss-Bonnet theorem. Every sphere is an independent problem; callers that
need vertices shared between spheres cluster them beforehand and pass the
labels in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .aos import scalars, vectors

TAU_C = 1e-6
_TAU_DIR = 1e-9


class DegenerateGeometryError(RuntimeError):
    pass


@dataclass
class Cap:
    """Cap ``axis . x > cos_a`` on the unit sphere; ``sin_a`` is kept
    alongside so no angle is ever recovered by inverse trig. On a SAS
    sphere the cap also names the ``partner`` atom on the other side of
    its ``circle`` and its ``side`` (0 on the circle's first sphere)."""

    axis: np.ndarray
    cos_a: float
    sin_a: float
    partner: int = -1
    circle: int = -1
    side: int = 0


@dataclass
class Arc:
    """Accessible sub-arc of cap circle ``cap`` from vertex ``v_beg`` over
    ``dphi`` to ``v_end`` in the circle frame; ``-1`` ends mark a full
    circle."""

    cap: int
    v_beg: int
    v_end: int
    phi_beg: float
    dphi: float


@dataclass
class Arrangement:
    caps: list[Cap]
    arcs: list[Arc]
    n_loops: int
    n_patches: int
    area: float

    def contains(self, dirs: np.ndarray) -> np.ndarray:
        return contains(self.caps, dirs)


def caps_from_arrays(
    axis: np.ndarray, cos_a: np.ndarray, sin_a: np.ndarray
) -> list[Cap]:
    return [
        Cap(np.asarray(n, dtype=float), float(c), float(s))
        for n, c, s in zip(axis, cos_a, sin_a)
    ]


def contains(caps: list[Cap], dirs: np.ndarray) -> np.ndarray:
    """True where unit directions ``dirs`` lie inside any cap."""
    inside = dirs @ vectors(caps, "axis").T > scalars(caps, "cos_a")
    return np.any(inside, axis=1)


def any_perpendicular(u: np.ndarray) -> np.ndarray:
    """Unit vectors perpendicular to each row of ``u`` (k, 3)."""
    u = np.asarray(u, dtype=float).reshape(-1, 3)
    onehot = np.zeros_like(u)
    onehot[np.arange(len(u)), np.argmin(np.abs(u), axis=1)] = 1.0
    v = cross(u, onehot)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def circle_frames(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e1 = any_perpendicular(axis)
    return e1, cross(axis, e1)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product of ``(k, 3)`` arrays."""
    return np.column_stack(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ]
    )


def components(n: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component labels ``0 .. k-1`` of ``n`` nodes joined by
    ``edges``, numbered by smallest member.

    Union-find with path halving; the smaller root absorbs the larger, so
    every root is the smallest node of its component.
    """
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    root = np.fromiter((find(x) for x in range(n)), dtype=int, count=n)
    is_root = root == np.arange(n)
    return (np.cumsum(is_root) - 1)[root]


def cluster_points(pts: np.ndarray, tol: float) -> np.ndarray:
    """Labels of connected components of points closer than ``tol``,
    with the pair search on a KD-tree (many points)."""
    if len(pts) < 2:
        return np.arange(len(pts))
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray")
    return components(len(pts), pairs)


def _cluster_dense(pts: np.ndarray, tol: float) -> np.ndarray:
    """Labels of connected components of points closer than ``tol``, with
    every pair tested (few points)."""
    diff = pts[:, None, :] - pts[None, :, :]
    near = np.einsum("ijk,ijk->ij", diff, diff) <= tol * tol
    edges = np.column_stack(np.nonzero(np.triu(near, 1)))
    return components(len(pts), edges)


def prepare_caps(
    caps: list[Cap], radius: float
) -> tuple[list[Cap], bool, np.ndarray]:
    """Merge coincident caps and drop hidden (nested) ones.

    Caps whose circles lie within ``TAU_C`` of each other everywhere (their
    ``radius * (axis, cos, sin)`` vectors that close) are one circle computed
    through different routes, e.g. the departure hemispheres of tangent arcs
    at a pinch; they are merged into their renormalised mean. Returns the
    surviving caps, whether two caps together cover the whole sphere (no
    arrangement is needed then) and the crossing matrix of the survivors.
    """
    caps = _merge_coincident(caps, radius)
    hidden, covered, crossing = classify_caps(caps)
    keep = np.flatnonzero(~hidden)
    return [caps[k] for k in keep], covered, crossing[np.ix_(keep, keep)]


def classify_caps(
    caps: list[Cap], crossing: np.ndarray | None = None
) -> tuple[np.ndarray, bool, np.ndarray]:
    """``(hidden, covered, crossing)`` of a sphere's caps.

    A cap nested inside a larger one is hidden. ``crossing`` may be given
    (see :func:`pair_predicates`); it is returned as used.
    """
    m = len(caps)
    if m < 2:
        return np.zeros(m, dtype=bool), False, np.zeros((m, m), dtype=bool)
    nested, covering, crossing = pair_predicates(caps, crossing)
    c = scalars(caps, "cos_a")
    hidden = np.any(nested & (c[:, None] > c[None, :]), axis=1)
    return hidden, bool(covering.any()), crossing


def pair_predicates(
    caps: list[Cap], crossing: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify every cap pair: ``(nested, covering, crossing)`` boolean
    matrices with a false diagonal.

    With ``g = n_j . n_k`` and ``c, s`` the cosines and sines: nested iff
    ``g >= c_j c_k + s_j s_k`` (``gamma <= |a_j - a_k|``); apart iff
    ``g <= c_j c_k - s_j s_k`` (``gamma >= a_j + a_k`` or
    ``gamma >= 2 pi - a_j - a_k``), which is a cover of the whole sphere when
    ``c_j + c_k < 0`` and disjoint otherwise; crossing in between.

    When the crossing decision was already made elsewhere (SAS spheres
    decide it once per sphere triple), it is passed in and the remaining
    pairs are split into nested and apart by ``g > c_j c_k``, which lies
    ``s_j s_k`` away from either boundary and so never disagrees with a
    crossing decision made by any other route.
    """
    axis = vectors(caps, "axis")
    cosg = axis @ axis.T
    c, s = scalars(caps, "cos_a"), scalars(caps, "sin_a")
    cc = c[:, None] * c[None, :]
    if crossing is None:
        ss = s[:, None] * s[None, :]
        nested = cosg >= cc + ss
        apart = cosg <= cc - ss
        crossing = ~nested & ~apart
    else:
        nested = (cosg > cc) & ~crossing
        apart = ~nested & ~crossing
    covering = apart & (c[:, None] + c[None, :] < 0.0)
    for x in (nested, covering, crossing):
        np.fill_diagonal(x, False)
    return nested, covering, crossing


def _merge_coincident(caps: list[Cap], radius: float) -> list[Cap]:
    vec = radius * np.column_stack(
        [
            vectors(caps, "axis"),
            scalars(caps, "cos_a"),
            scalars(caps, "sin_a"),
        ]
    )
    label = _cluster_dense(vec, TAU_C)
    k = int(label.max()) + 1 if len(label) else 0
    if k == len(caps):
        return caps
    mean = np.zeros((k, 5))
    np.add.at(mean, label, vec)
    axis = mean[:, :3] / np.linalg.norm(mean[:, :3], axis=1, keepdims=True)
    trig = mean[:, 3:] / np.linalg.norm(mean[:, 3:], axis=1, keepdims=True)
    return caps_from_arrays(axis, trig[:, 0], trig[:, 1])


def crossing_points(
    caps: list[Cap], crossing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw intersection points of all crossing cap-circle pairs.

    ``crossing`` is the decision of :func:`pair_predicates`. For a crossing pair
    the point of the intersection line nearest the origin is written in the
    basis ``m = n_1 + n_2``, ``w = n_1 - n_2``:

        base = (c_1 + c_2)/|m|^2 m + (c_1 - c_2)/|w|^2 w

    which has no cancellation and, for nearly parallel axes, moves with the
    input like the geometry does (as ``1/gamma``, not ``1/gamma^2``). The
    half-chord ``h = sqrt(max(1 - |base|^2, 0))`` is clamped so a pair the
    predicate calls crossing but rounding puts at tangency yields two
    coincident points, which clustering merges into a pinch.

    Returns ``(dirs (2e, 3), edges (e, 2))``: the two points of edge ``k``
    are rows ``k`` and ``e + k``.
    """
    axis, cos_a = vectors(caps, "axis"), scalars(caps, "cos_a")
    jj, kk = np.nonzero(np.triu(crossing, 1))
    n1, n2 = axis[jj], axis[kk]
    c1, c2 = cos_a[jj], cos_a[kk]
    mid, dif = n1 + n2, n1 - n2
    base = ((c1 + c2) / np.einsum("ij,ij->i", mid, mid))[:, None] * mid + (
        (c1 - c2) / np.einsum("ij,ij->i", dif, dif)
    )[:, None] * dif
    hsq = np.maximum(1.0 - np.einsum("ij,ij->i", base, base), 0.0)
    perp = cross(n1, n2)
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    h = np.sqrt(hsq)[:, None]
    dirs = np.concatenate([base + h * perp, base - h * perp])
    return dirs, np.column_stack([jj, kk])


def cap_components(crossing: np.ndarray) -> int:
    """Connected components of the caps joined by crossing pairs."""
    m = len(crossing)
    if m == 0:
        return 0
    edges = np.column_stack(np.nonzero(np.triu(crossing, 1)))
    return int(components(m, edges).max()) + 1


def covered_arrangement(caps: list[Cap]) -> Arrangement:
    return Arrangement(caps, [], 0, 0, 0.0)


@dataclass
class ArrangementProblem:
    """A prepared arrangement of one sphere: hygienic caps with their circle
    frames ``(e1, e2)`` (m, 3), the crossing decision (m, m) and its number
    of connected components, and the clustered vertices: unit ``reps``
    (k, 3), ``excused`` (k, m: the caps whose crossing points merged into
    the vertex, which are its incident caps) and ``accessible`` (k)."""

    caps: list[Cap]
    e1: np.ndarray
    e2: np.ndarray
    crossing: np.ndarray
    n_components: int
    reps: np.ndarray
    excused: np.ndarray
    accessible: np.ndarray
    pinched: set[tuple[int, int, int]] = field(default_factory=set)


def local_problem(
    radius: float, caps: list[Cap], crossing: np.ndarray
) -> ArrangementProblem:
    """Prepare a sphere whose vertices are known to nobody else (a probe
    sphere): cross the caps, cluster the points and decide accessibility
    here."""
    dirs, edges = crossing_points(caps, crossing)
    label = _cluster_dense(dirs * radius, TAU_C)
    n_clusters = int(label.max()) + 1 if len(label) else 0
    reps = np.zeros((n_clusters, 3))
    np.add.at(reps, label, dirs)
    reps /= np.linalg.norm(reps, axis=1, keepdims=True)
    excused = np.zeros((n_clusters, len(caps)), dtype=bool)
    excused[label[:, None], np.concatenate([edges, edges])] = True
    inside = reps @ vectors(caps, "axis").T > scalars(caps, "cos_a")
    accessible = ~(inside & ~excused).any(axis=1)
    e1, e2 = circle_frames(vectors(caps, "axis"))
    n_edges = len(edges)
    pinched = {
        (int(label[k]), int(a), int(b))
        for k, (a, b) in enumerate(edges.tolist())
        if label[k] == label[n_edges + k]
    }
    return ArrangementProblem(
        caps,
        e1,
        e2,
        crossing,
        cap_components(crossing),
        reps,
        excused,
        accessible,
        pinched,
    )


def solve_caps(radius: float, caps: list[Cap]) -> Arrangement:
    """Solve an arrangement, clustering vertices locally."""
    caps, covered, crossing = prepare_caps(caps, radius)
    if covered:
        return covered_arrangement(caps)
    return solve(radius, local_problem(radius, caps, crossing))


def solve(radius: float, problem: ArrangementProblem) -> Arrangement:
    arcs = _cap_arcs(problem)
    n_loops, turn_sum, geo_sum = _walk(problem, arcs)

    n_patches = 1 + n_loops - problem.n_components
    chi = 2 * n_patches - n_loops
    area = radius * radius * (2.0 * math.pi * chi - turn_sum + geo_sum)
    return Arrangement(problem.caps, arcs, n_loops, n_patches, float(area))


def _cap_arcs(problem: ArrangementProblem) -> list[Arc]:
    """Accessible arcs of every cap circle, cap by cap.

    The accessible vertices incident to a cap (``excused`` column, masked
    by ``accessible``), sorted by ``phi`` in the cap frame, bound the
    candidate arcs: consecutive vertices with a wrap-around from the last
    to the first (a circle with one vertex yields one full turn, a circle
    without vertices one full circle with ``-1`` ends). A candidate is
    kept iff its midpoint lies outside every cap that crosses its circle;
    disjoint caps cannot contain any of it, and testing them anyway would
    let rounding at an exact tangency contradict the crossing decision.
    """
    caps, e1, e2, reps = problem.caps, problem.e1, problem.e2, problem.reps
    axis, cos_a = vectors(caps, "axis"), scalars(caps, "cos_a")
    sin_a = scalars(caps, "sin_a")
    arcs: list[Arc] = []
    for j in range(len(caps)):
        vs = np.flatnonzero(problem.excused[:, j] & problem.accessible)
        if len(vs) == 0:
            v_beg = v_end = np.array([-1])
            phi_beg, dphi = np.zeros(1), np.full(1, 2.0 * math.pi)
        else:
            dirs = reps[vs]
            phi = np.arctan2(
                np.einsum("ij,j->i", dirs, e2[j]),
                np.einsum("ij,j->i", dirs, e1[j]),
            )
            order = np.argsort(phi, kind="stable")
            v_beg, phi_beg = vs[order], phi[order]
            v_end = np.roll(v_beg, -1)
            dphi = np.roll(phi_beg, -1) - phi_beg
            dphi += 2.0 * math.pi * (dphi <= 0.0)

        mid = phi_beg + 0.5 * dphi
        radial = np.cos(mid)[:, None] * e1[j] + np.sin(mid)[:, None] * e2[j]
        mids = cos_a[j] * axis[j] + sin_a[j] * radial
        inside = (mids @ axis.T > cos_a) & problem.crossing[j]
        for i in np.flatnonzero(~inside.any(axis=1)):
            arcs.append(
                Arc(j, int(v_beg[i]), int(v_end[i]), phi_beg[i], dphi[i])
            )
    return arcs


@dataclass
class Dart:
    """One end of an arc at a vertex: the arc index, the direction of its
    tangent as an angle in the vertex frame, its signed geodesic curvature
    and whether it is the (reversed) in-dart of the arc."""

    arc: int
    angle: float
    kappa: float
    is_in: bool
    snapped: float = 0.0


def _walk(problem: ArrangementProblem, arcs: list[Arc]):
    """Trace boundary loops with the accessible region on the left.

    Arcs are stored with increasing ``phi`` (counter-clockwise around the
    cap axis, cap on the left); traversal therefore runs from ``v_end`` to
    ``v_beg``. Every arc leaving a vertex is an out-dart there and every
    arc arriving is a reversed in-dart; the darts of each vertex are put in
    cyclic order by :func:`_dart_ring`. A full circle has no darts and is
    its own loop. Returns ``(number of loops, sum of turning angles, sum of
    geodesic curvature integrals)``.
    """
    n = len(arcs)
    cos_a = scalars(problem.caps, "cos_a")
    geo_sum = float(
        np.sum(scalars(arcs, "dphi") * cos_a[scalars(arcs, "cap", int)])
    )
    succ = np.arange(n)
    turn = np.zeros(n)

    darts: list[list[Dart]] = [[] for _ in problem.reps]
    for arc, angle_out, angle_in, cot in _dart_angles(problem, arcs):
        a = arcs[arc]
        darts[a.v_end].append(Dart(arc, angle_out, -cot, False))
        darts[a.v_beg].append(Dart(arc, angle_in, cot, True))

    for v, ring in enumerate(darts):
        if not ring:
            continue
        ring = _dart_ring(ring, arcs, problem.crossing, problem.pinched, v)
        for d, prev in zip(ring, [ring[-1], *ring[:-1]]):
            if d.is_in == prev.is_in:
                raise DegenerateGeometryError(
                    f"vertex {v}: darts do not alternate"
                )
            if d.is_in:
                iota = d.angle - prev.angle
                if iota <= -0.5 * math.pi:
                    iota += 2.0 * math.pi
                elif iota > 1.5 * math.pi:
                    iota -= 2.0 * math.pi
                if iota > math.pi + _TAU_DIR:
                    raise DegenerateGeometryError(f"vertex {v}: reflex corner")
                succ[d.arc] = prev.arc
                turn[d.arc] = math.pi - iota

    return _count_cycles(succ), float(turn.sum()), geo_sum


def _dart_angles(problem: ArrangementProblem, arcs: list[Arc]):
    """Per arc with vertices: ``(arc, angle of the out-dart at v_end, angle
    of the reversed in-dart at v_beg, cot alpha of its cap)``.

    The out-dart tangent is ``-(n x u)``, the reversed in-dart tangent
    ``+(n x u)`` for the cap axis ``n`` and the vertex ``u``, projected into
    the tangent plane and measured in the frame ``(ea, eb)`` there.
    """
    idx = np.flatnonzero(scalars(arcs, "v_beg", int) >= 0)
    if len(idx) == 0:
        return []
    caps, reps = problem.caps, problem.reps
    axis = vectors(caps, "axis")
    cot_a = scalars(caps, "cos_a") / scalars(caps, "sin_a")
    cap = scalars(arcs, "cap", int)[idx]
    ea_all = any_perpendicular(reps)

    def angle_at(vertex, sign):
        u = reps[vertex]
        t = cross(axis[cap], u) * sign
        ea = ea_all[vertex]
        eb = cross(u, ea)
        return np.arctan2(
            np.einsum("ij,ij->i", t, eb), np.einsum("ij,ij->i", t, ea)
        )

    angle_out = angle_at(scalars(arcs, "v_end", int)[idx], -1.0)
    angle_in = angle_at(scalars(arcs, "v_beg", int)[idx], 1.0)
    return zip(idx.tolist(), angle_out, angle_in, cot_a[cap])


def _dart_ring(
    ring: list[Dart],
    arcs: list[Arc],
    crossing: np.ndarray,
    pinched: set[tuple[int, int, int]],
    v: int,
) -> list[Dart]:
    """Darts of one vertex in cyclic order.

    Two caps meet at the vertex without crossing there iff they are not a
    crossing pair or both cut points of the pair merged into the vertex
    (``pinched``). The in-dart of one and the out-dart of the other then
    bound a cusp: their angles differ by the merged vertex's offset times
    the curvatures, in either order, so an in-dart whose angular neighbour
    is the out-dart of a touching cap is placed right after it. Everything
    else keeps the raw order, and the raw angle is kept for the corner.
    """
    ring = sorted(ring, key=lambda d: d.angle)
    n = len(ring)
    cap_of = [arcs[d.arc].cap for d in ring]
    for d in ring:
        d.snapped = d.angle
    for i, d in enumerate(ring):
        if not d.is_in:
            continue
        a = cap_of[i]
        best = 2.0 * math.pi
        for j in ((i - 1) % n, (i + 1) % n):
            o, b = ring[j], cap_of[j]
            if j == i or o.is_in or b == a:
                continue
            if crossing[a, b] and (v, min(a, b), max(a, b)) not in pinched:
                continue
            gap = abs(math.remainder(d.angle - o.angle, 2.0 * math.pi))
            if gap < best:
                best = gap
                d.snapped = o.angle
    return sorted(ring, key=lambda d: (d.snapped, d.is_in, d.kappa, d.arc))


def _count_cycles(succ: np.ndarray) -> int:
    seen = np.zeros(len(succ), dtype=bool)
    n_cycles = 0
    for start in range(len(succ)):
        if seen[start]:
            continue
        n_cycles += 1
        i = start
        while not seen[i]:
            seen[i] = True
            i = succ[i]
    return n_cycles

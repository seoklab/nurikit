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
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

TAU_C = 1e-6
_TAU_DIR = 1e-9


class DegenerateGeometryError(RuntimeError):
    pass


@dataclass
class Caps:
    """Caps ``n . x > cos_a`` on the unit sphere; ``sin_a`` is kept
    alongside so no angle is ever recovered by inverse trig."""

    axis: np.ndarray
    cos_a: np.ndarray
    sin_a: np.ndarray

    def __len__(self) -> int:
        return len(self.cos_a)

    @classmethod
    def empty(cls) -> Caps:
        z = np.empty(0)
        return cls(np.empty((0, 3)), z, z.copy())

    def take(self, idx: np.ndarray) -> Caps:
        return Caps(self.axis[idx], self.cos_a[idx], self.sin_a[idx])

    @classmethod
    def concat(cls, parts: list[Caps]) -> Caps:
        parts = [cls.empty(), *parts]
        return cls(
            *(
                np.concatenate([getattr(x, f) for x in parts])
                for f in ("axis", "cos_a", "sin_a")
            )
        )


@dataclass
class Arcs:
    cap: np.ndarray
    v_beg: np.ndarray
    v_end: np.ndarray
    phi_beg: np.ndarray
    dphi: np.ndarray

    def __len__(self) -> int:
        return len(self.cap)


@dataclass
class Arrangement:
    caps: Caps
    arcs: Arcs
    n_loops: int
    n_patches: int
    area: float

    def contains(self, dirs: np.ndarray) -> np.ndarray:
        """True where unit directions ``dirs`` lie inside any cap."""
        return np.any(dirs @ self.caps.axis.T > self.caps.cos_a, axis=1)


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

    Minimum-label propagation with pointer jumping: labels are node ids,
    every round takes the minimum over each edge and then the label of the
    label, so a component collapses to its smallest node in logarithmically
    many rounds.
    """
    label = np.arange(n)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        new = label.copy()
        np.minimum.at(new, a, label[b])
        np.minimum.at(new, b, label[a])
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    is_root = np.zeros(n, dtype=bool)
    is_root[label] = True
    return (np.cumsum(is_root) - 1)[label]


def cluster_points(pts: np.ndarray, tol: float) -> np.ndarray:
    """Labels of connected components of points closer than ``tol``."""
    if len(pts) < 2:
        return np.arange(len(pts))
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(len(pts))
    return components(len(pts), pairs)


def prepare_caps(caps: Caps, radius: float) -> tuple[Caps, bool, np.ndarray]:
    """Merge coincident caps and drop hidden (nested) ones.

    Caps whose circles lie within ``TAU_C`` of each other everywhere (their
    ``radius * (axis, cos, sin)`` vectors that close) are one circle computed
    through different routes, e.g. the departure hemispheres of tangent arcs
    at a pinch; they are merged into their renormalised mean. Returns the
    surviving caps, whether two caps together cover the whole sphere (no
    arrangement is needed then) and the crossing matrix of the survivors.
    """
    caps, _ = _merge_coincident(caps, radius)
    hidden, covered, crossing = classify_caps(caps)
    keep = np.flatnonzero(~hidden)
    return caps.take(keep), covered, crossing[np.ix_(keep, keep)]


def classify_caps(
    caps: Caps, crossing: np.ndarray | None = None
) -> tuple[np.ndarray, bool, np.ndarray]:
    """``(hidden, covered, crossing)`` of a sphere's caps.

    A cap nested inside a larger one is hidden. ``crossing`` may be given
    (see :func:`pair_predicates`); it is returned as used.
    """
    m = len(caps)
    if m < 2:
        return np.zeros(m, dtype=bool), False, np.zeros((m, m), dtype=bool)
    nested, covering, crossing = pair_predicates(caps, crossing)
    c = caps.cos_a
    hidden = np.any(nested & (c[:, None] > c[None, :]), axis=1)
    return hidden, bool(covering.any()), crossing


def pair_predicates(
    caps: Caps, crossing: np.ndarray | None = None
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
    cosg = caps.axis @ caps.axis.T
    c, s = caps.cos_a, caps.sin_a
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


def _merge_coincident(caps: Caps, radius: float) -> tuple[Caps, np.ndarray]:
    """Merged caps and the merged index of every input cap."""
    vec = radius * np.column_stack([caps.axis, caps.cos_a, caps.sin_a])
    label = cluster_points(vec, TAU_C)
    k = len(np.unique(label))
    if k == len(caps):
        return caps, label
    mean = np.zeros((k, 5))
    np.add.at(mean, label, vec)
    axis = mean[:, :3] / np.linalg.norm(mean[:, :3], axis=1, keepdims=True)
    trig = mean[:, 3:] / np.linalg.norm(mean[:, 3:], axis=1, keepdims=True)
    return Caps(axis, trig[:, 0], trig[:, 1]), label


def crossing_points(
    caps: Caps, crossing: np.ndarray
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
    jj, kk = np.nonzero(np.triu(crossing, 1))
    n1, n2 = caps.axis[jj], caps.axis[kk]
    c1, c2 = caps.cos_a[jj], caps.cos_a[kk]
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


def covered_arrangement(caps: Caps) -> Arrangement:
    z = np.empty(0, dtype=int)
    return Arrangement(
        caps, Arcs(z, z, z, np.empty(0), np.empty(0)), 0, 0, 0.0
    )


def solve_caps(radius: float, caps: Caps) -> Arrangement:
    """Solve an arrangement, clustering vertices locally."""
    caps, covered, crossing = prepare_caps(caps, radius)
    if covered:
        return covered_arrangement(caps)
    dirs, edges = crossing_points(caps, crossing)
    label = cluster_points(dirs * radius, TAU_C)
    n_clusters = int(label.max()) + 1 if len(label) else 0
    reps = np.zeros((n_clusters, 3))
    np.add.at(reps, label, dirs)
    reps /= np.linalg.norm(reps, axis=1, keepdims=True)
    excused = np.zeros((n_clusters, len(caps)), dtype=bool)
    excused[label[:, None], np.concatenate([edges, edges])] = True
    inside = reps @ caps.axis.T > caps.cos_a
    accessible = ~(inside & ~excused).any(axis=1)
    return solve(
        radius,
        caps,
        edges,
        reps,
        excused,
        accessible,
        cap_components(crossing),
    )


def solve(
    radius: float,
    caps: Caps,
    edges: np.ndarray,
    reps: np.ndarray,
    excused: np.ndarray,
    accessible: np.ndarray,
    n_components: int,
    frames: tuple[np.ndarray, np.ndarray] | None = None,
) -> Arrangement:
    """Solve an arrangement of prepared ``caps``.

    ``edges`` are the crossing cap pairs and ``n_components`` the number
    of connected components of that graph. Per cluster: ``reps`` (unit
    directions), ``excused`` (clusters x caps: the caps whose crossing
    points merged into the cluster, which are its incident caps) and
    ``accessible``. ``frames`` ``(e1, e2)`` per cap default to
    :func:`circle_frames`.
    """
    m = len(caps)
    e1, e2 = circle_frames(caps.axis) if frames is None else frames
    crossing = np.zeros((m, m), dtype=bool)
    crossing[edges[:, 0], edges[:, 1]] = True
    crossing[edges[:, 1], edges[:, 0]] = True

    vtx, cap = np.nonzero(excused & accessible[:, None])

    arcs = _build_arcs(caps, e1, e2, reps, vtx, cap, crossing)
    n_loops, turn_sum, geo_sum = _walk(caps, reps, arcs)

    n_patches = 1 + n_loops - n_components
    chi = 2 * n_patches - n_loops
    area = radius * radius * (2.0 * math.pi * chi - turn_sum + geo_sum)
    return Arrangement(caps, arcs, n_loops, n_patches, float(area))


def _build_arcs(caps, e1, e2, reps, vtx, cap, crossing) -> Arcs:
    """Accessible arcs of every cap circle.

    ``(vtx, cap)`` are the accessible vertex-cap incidences. Consecutive
    vertices around a circle bound candidate arcs (a circle with one
    vertex yields one full turn, a circle without vertices one full
    circle with ``-1`` ends). A candidate is kept iff its midpoint lies
    outside every cap that crosses its circle; disjoint caps cannot
    contain any of it, and testing them anyway would let rounding at an
    exact tangency contradict the crossing decision.
    """
    m = len(caps)
    dirs = reps[vtx]
    phi = np.arctan2(
        np.einsum("ij,ij->i", dirs, e2[cap]),
        np.einsum("ij,ij->i", dirs, e1[cap]),
    )
    order = np.lexsort((phi, cap))
    vtx, cap, phi = vtx[order], cap[order], phi[order]
    k = len(cap)
    first = np.ones(k, dtype=bool)
    first[1:] = cap[1:] != cap[:-1]
    nxt = np.arange(k) + 1
    nxt[np.flatnonzero(np.roll(first, -1))] = np.flatnonzero(first)
    span = phi[nxt] - phi
    span += 2.0 * math.pi * (span <= 0.0)

    empty = np.flatnonzero(np.bincount(cap, minlength=m) == 0)
    cap_ix = np.concatenate([cap, empty])
    v_beg = np.concatenate([vtx, np.full(len(empty), -1)])
    v_end = np.concatenate([vtx[nxt], np.full(len(empty), -1)])
    phi_beg = np.concatenate([phi, np.zeros(len(empty))])
    dphi = np.concatenate([span, np.full(len(empty), 2.0 * math.pi)])

    mid = phi_beg + 0.5 * dphi
    radial = (
        np.cos(mid)[:, None] * e1[cap_ix] + np.sin(mid)[:, None] * e2[cap_ix]
    )
    mids = (
        caps.cos_a[cap_ix, None] * caps.axis[cap_ix]
        + caps.sin_a[cap_ix, None] * radial
    )
    inside = (mids @ caps.axis.T > caps.cos_a) & crossing[cap_ix]
    ok = ~inside.any(axis=1)
    return Arcs(cap_ix[ok], v_beg[ok], v_end[ok], phi_beg[ok], dphi[ok])


def _walk(caps, reps, arcs):
    """Trace boundary loops with the accessible region on the left.

    Arcs are stored with increasing ``phi`` (counter-clockwise around the
    cap axis, cap on the left); traversal therefore runs from ``v_end`` to
    ``v_beg``. Every arc leaving a vertex is an out-dart there and every
    arc arriving is a reversed in-dart; sorting all darts by (vertex,
    angle, curvature) gives the cyclic order at every vertex at once. A
    full circle has no darts and is its own loop. Returns ``(number of
    loops, sum of turning angles, sum of geodesic curvature integrals)``.
    """
    n = len(arcs)
    geo_sum = float(np.sum(arcs.dphi * caps.cos_a[arcs.cap]))
    succ = np.arange(n)
    turn = np.zeros(n)

    idx = np.flatnonzero(arcs.v_beg >= 0)
    if len(idx):
        vertex, angle, kind, arc = _sorted_darts(caps, reps, arcs, idx)
        first = np.ones(len(vertex), dtype=bool)
        first[1:] = vertex[1:] != vertex[:-1]
        prev = np.arange(len(vertex)) - 1
        prev[first] = np.flatnonzero(np.roll(first, -1))
        bad = kind == kind[prev]
        if bad.any():
            raise DegenerateGeometryError(
                f"vertex {vertex[bad][0]}: darts do not alternate"
            )
        ins = np.flatnonzero(kind == 1)
        iota = angle[ins] - angle[prev[ins]]
        iota += 2.0 * math.pi * (iota < 0.0)
        succ[arc[ins]] = arc[prev[ins]]
        turn[arc[ins]] = math.pi - iota

    n_loops = (
        int(components(n, np.column_stack([np.arange(n), succ])).max()) + 1
        if n
        else 0
    )
    return n_loops, float(turn.sum()), geo_sum


def _sorted_darts(caps, reps, arcs, idx):
    """Darts of the arcs ``idx`` in cyclic order around their vertices.

    Angles are measured in a tangent frame at the vertex. Darts closer
    than ``_TAU_DIR`` (tangent circles, pinches) share one snapped angle
    and are ordered by signed geodesic curvature, right-curving first, so
    the wedge between them is exactly zero; a group straddling the
    ``-pi``/``pi`` seam is merged the same way.
    """
    vertex = np.concatenate([arcs.v_end[idx], arcs.v_beg[idx]])
    kind = np.repeat(np.array([0, 1]), len(idx))
    arc = np.concatenate([idx, idx])
    cap = arcs.cap[arc]
    u = reps[vertex]
    side = 2 * kind - 1
    t = cross(caps.axis[cap], u) * side[:, None]
    kappa = (caps.cos_a / caps.sin_a)[cap] * side
    ea = any_perpendicular(reps)[vertex]
    eb = cross(u, ea)
    angle = np.arctan2(
        np.einsum("ij,ij->i", t, eb), np.einsum("ij,ij->i", t, ea)
    )

    order = np.lexsort((angle, vertex))
    vertex, angle, kappa, kind, arc = (
        x[order] for x in (vertex, angle, kappa, kind, arc)
    )
    m = len(vertex)
    first = np.ones(m, dtype=bool)
    first[1:] = vertex[1:] != vertex[:-1]
    block = np.cumsum(first) - 1
    blk_first = np.flatnonzero(first)
    blk_last = np.flatnonzero(np.roll(first, -1))

    new = first.copy()
    new[1:] |= angle[1:] - angle[:-1] >= _TAU_DIR
    group = np.maximum.accumulate(np.where(new, np.arange(m), 0))
    snapped = angle[group]
    wrap = angle[blk_first] + 2.0 * math.pi - angle[blk_last] < _TAU_DIR
    in_last = group == group[blk_last][block]
    snapped = np.where(in_last & wrap[block], angle[blk_first][block], snapped)

    order = np.lexsort((kappa, snapped, vertex))
    return (x[order] for x in (vertex, snapped, kind, arc))

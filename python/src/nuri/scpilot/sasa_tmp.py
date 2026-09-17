"""
sas_components_ckdtree.py

Compute SAS component sets:
  - I    : intersection points
  - l_m  : circular arcs (or full circles)
  - P_m  : spherical patches (represented via boundary loops)

Implements the 5-step SAS construction described in:
  Quan & Stamm, "Mathematical analysis and calculation of molecular surfaces",
  J. Comput. Phys. 322 (2016) 760–782, Section 6.1. :contentReference[oaicite:1]{index=1}

Inputs:
  centers: (N,3) float array of atom centers
  vdw_radii: (N,) float array of van der Waals radii
  probe_radius: float

Notes:
  - Uses scipy.spatial.cKDTree for:
      (a) neighbor candidate queries among atom centers,
      (b) hidden-point checks (point inside any other SAS ball),
      (c) merging near-duplicate intersection points.
  - Default surface="complete" builds cSAS (boundary of union of SAS-balls).
    surface="exterior" extracts the outer connected component as eSAS (Step 5 in Section 6.1).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree


def _sample_arc_points(arc, n: int = 64) -> np.ndarray:
    """
    Sample polyline points along a CircularArc (or full circle).

    Parameters
    ----------
    arc : CircularArc-like
        Must have attributes: circle_center, circle_radius, u, v, theta_start, delta.
    n : int
        Number of samples along the arc.

    Returns
    -------
    pts : (n,3) ndarray
    """
    n = int(max(2, n))
    thetas = arc.theta_start + np.linspace(0.0, arc.delta, n, endpoint=True)
    c = np.asarray(arc.circle_center, float)
    u = np.asarray(arc.u, float)
    v = np.asarray(arc.v, float)
    r = float(arc.circle_radius)

    # pts = c + r*(cos(t)*u + sin(t)*v)
    ct = np.cos(thetas)[:, None]
    st = np.sin(thetas)[:, None]
    pts = c[None, :] + r * (ct * u[None, :] + st * v[None, :])
    return pts


def plot_sas_components(
    comps,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (9.0, 9.0),
    elev: float = 20.0,
    azim: float = 35.0,
    # what to draw
    show_atoms: bool = True,
    show_sas_spheres: bool = False,
    show_intersection_points: bool = True,
    intersection_points: str = "used",  # "used" or "all"
    show_arcs: bool = True,
    show_loops: bool = False,
    show_patches: bool = True,  # patch boundaries (loops grouped by patch)
    exterior_only: bool = False,  # if comps.exterior_patch_ids is set, filter to that set
    sphere_ids: Optional[Sequence[int]] = None,
    patch_ids: Optional[Sequence[int]] = None,
    # sampling / limits
    arc_samples: int = 80,
    sphere_wire_samples: int = 18,
    max_arcs: Optional[int] = None,
    max_points: Optional[int] = None,
    # styling
    atom_size: float = 18.0,
    ip_size: float = 30.0,
    arc_lw: float = 1.6,
    loop_lw: float = 2.5,
    patch_lw: float = 2.8,
    alpha_sphere: float = 0.10,
    label_ips: bool = False,
    label_arcs: bool = False,
    label_patches: bool = False,
    title: Optional[str] = None,
):
    """
    Visualize SAS components (I, l_m, P_m) in 3D using matplotlib.

    - I (intersection points): shown as scatter points.
    - l_m (arcs/circles): shown as sampled polylines.
    - P_m (spherical patches): shown via their boundary loops (grouped by patch).

    Parameters
    ----------
    comps : SASComponents-like
        Expected fields:
          - centers: (N,3)
          - radii_sas: (N,)
          - I: list of IntersectionPoint-like {id, position}
          - l_m: list of CircularArc-like
          - loops: list of Loop-like {id, sphere, arc_ids}
          - P_m: list of SphericalPatch-like {id, sphere, loop_ids}
          - exterior_patch_ids: optional set of patch ids (for eSAS)
    ax : matplotlib axis or None
        If None, create a new 3D figure.
    sphere_ids : optional list of sphere indices (atom/SAS-sphere indices)
        Restrict drawing to these spheres (arcs/loops/patches connected to them).
    patch_ids : optional list of patch ids
        Restrict drawing to these patches (and their boundary arcs).
    exterior_only : bool
        If True and comps.exterior_patch_ids is present, restrict to exterior patch set.

    Returns
    -------
    (fig, ax)
    """
    # --- figure/axes ---
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    centers = np.asarray(comps.centers, float)
    radii_sas = np.asarray(comps.radii_sas, float)

    sphere_filter: Optional[Set[int]] = (
        set(sphere_ids) if sphere_ids is not None else None
    )

    # Determine which patches to draw
    patch_sel: Optional[Set[int]] = None
    if patch_ids is not None:
        patch_sel = set(int(x) for x in patch_ids)
    elif (
        exterior_only
        and getattr(comps, "exterior_patch_ids", None) is not None
    ):
        patch_sel = set(int(x) for x in comps.exterior_patch_ids)

    # Helper: gather arcs used by selected patches
    def arcs_from_patches(pids: Iterable[int]) -> Set[int]:
        out: Set[int] = set()
        for pid in pids:
            patch = comps.P_m[int(pid)]
            for lid in patch.loop_ids:
                out.update(comps.loops[int(lid)].arc_ids)
        return out

    # Determine arcs to plot
    arcs_to_plot: Set[int]
    if patch_sel is not None:
        arcs_to_plot = arcs_from_patches(patch_sel)
    else:
        arcs_to_plot = set(range(len(comps.l_m)))

    if sphere_filter is not None:
        # keep only arcs incident to selected spheres
        arcs_to_plot = {
            aid
            for aid in arcs_to_plot
            if (
                comps.l_m[aid].sphere_i in sphere_filter
                or comps.l_m[aid].sphere_j in sphere_filter
            )
        }

    arcs_to_plot = set(sorted(arcs_to_plot))
    if max_arcs is not None and len(arcs_to_plot) > int(max_arcs):
        arcs_to_plot = set(sorted(arcs_to_plot)[: int(max_arcs)])

    # Determine which intersection points to plot
    ip_positions_by_id = {
        int(ip.id): np.asarray(ip.position, float) for ip in comps.I
    }

    used_ip_ids: Set[int] = set()
    if intersection_points.lower() == "used":
        for aid in arcs_to_plot:
            arc = comps.l_m[aid]
            if arc.start_ip is not None:
                used_ip_ids.add(int(arc.start_ip))
            if arc.end_ip is not None:
                used_ip_ids.add(int(arc.end_ip))

    if show_intersection_points:
        if intersection_points.lower() == "all":
            ip_ids = sorted(ip_positions_by_id.keys())
        else:
            ip_ids = sorted(used_ip_ids)

        if max_points is not None and len(ip_ids) > int(max_points):
            ip_ids = ip_ids[: int(max_points)]

        if ip_ids:
            I_xyz = np.vstack([ip_positions_by_id[i] for i in ip_ids])
            ax.scatter(
                I_xyz[:, 0],
                I_xyz[:, 1],
                I_xyz[:, 2],
                s=ip_size,
                depthshade=True,
                label="I (intersection pts)",
            )
            if label_ips:
                for i, p in zip(ip_ids, I_xyz):
                    ax.text(p[0], p[1], p[2], f"  I{i}", fontsize=8)

    # Plot arcs/circles
    if show_arcs and arcs_to_plot:
        segs = []
        arc_midpoints = []  # for optional labels
        arc_ids_sorted = sorted(arcs_to_plot)
        for aid in arc_ids_sorted:
            arc = comps.l_m[aid]
            pts = _sample_arc_points(arc, n=arc_samples)
            segs.append(pts)

            # midpoint for label
            theta_mid = arc.theta_start + 0.5 * arc.delta
            mid = arc.circle_center + arc.circle_radius * (
                math.cos(theta_mid) * arc.u + math.sin(theta_mid) * arc.v
            )
            arc_midpoints.append((aid, np.asarray(mid, float)))

        lc = Line3DCollection(segs, linewidths=arc_lw)
        ax.add_collection3d(lc)

        if label_arcs:
            for aid, mid in arc_midpoints:
                ax.text(mid[0], mid[1], mid[2], f"  a{aid}", fontsize=8)

    # Optionally plot loops (thicker, sphere-by-sphere)
    if show_loops and len(comps.loops) > 0:
        for loop in comps.loops:
            if (
                sphere_filter is not None
                and int(loop.sphere) not in sphere_filter
            ):
                continue
            # if patch selection is active, only show loops that belong to selected patches
            if patch_sel is not None:
                in_selected = False
                for pid in patch_sel:
                    if int(loop.id) in set(comps.P_m[int(pid)].loop_ids):
                        in_selected = True
                        break
                if not in_selected:
                    continue

            loop_arc_ids = [aid for aid in loop.arc_ids if aid in arcs_to_plot]
            for aid in loop_arc_ids:
                pts = _sample_arc_points(comps.l_m[aid], n=arc_samples)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=loop_lw)

    # Optionally plot patches via their boundary loops, grouped by patch
    if show_patches and len(comps.P_m) > 0:
        patches_to_draw = (
            patch_sel if patch_sel is not None else set(range(len(comps.P_m)))
        )
        for pid in sorted(patches_to_draw):
            patch = comps.P_m[int(pid)]
            if (
                sphere_filter is not None
                and int(patch.sphere) not in sphere_filter
            ):
                continue

            # draw each boundary loop (as its arcs)
            patch_arc_ids: Set[int] = set()
            for lid in patch.loop_ids:
                patch_arc_ids.update(comps.loops[int(lid)].arc_ids)

            # apply arc filter (patch selection might be None; this keeps consistent with arcs_to_plot)
            patch_arc_ids = [
                aid for aid in sorted(patch_arc_ids) if aid in arcs_to_plot
            ]
            for aid in patch_arc_ids:
                pts = _sample_arc_points(comps.l_m[aid], n=arc_samples)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=patch_lw)

            if label_patches and patch_arc_ids:
                # label at average of arc midpoints
                mids = []
                for aid in patch_arc_ids[: min(10, len(patch_arc_ids))]:
                    arc = comps.l_m[aid]
                    theta_mid = arc.theta_start + 0.5 * arc.delta
                    mids.append(
                        arc.circle_center
                        + arc.circle_radius
                        * (
                            math.cos(theta_mid) * arc.u
                            + math.sin(theta_mid) * arc.v
                        )
                    )
                m = np.mean(np.vstack(mids), axis=0)
                ax.text(m[0], m[1], m[2], f"P{pid}", fontsize=9)

    # Plot atom centers (context)
    if show_atoms and centers.size > 0:
        # optionally filter centers to sphere_filter
        if sphere_filter is not None:
            idx = np.array(sorted(sphere_filter), dtype=int)
            idx = idx[(idx >= 0) & (idx < centers.shape[0])]
            cc = centers[idx]
        else:
            cc = centers
        ax.scatter(
            cc[:, 0],
            cc[:, 1],
            cc[:, 2],
            s=atom_size,
            depthshade=True,
            label="atom centers",
        )

    # Plot SAS spheres as wireframes (expensive if many)
    if show_sas_spheres and centers.size > 0:
        if sphere_filter is not None:
            sphere_idx = sorted(sphere_filter)
        else:
            sphere_idx = range(centers.shape[0])

        # sphere parameterization
        nu = int(max(8, sphere_wire_samples))
        nv = int(max(8, sphere_wire_samples))
        uu = np.linspace(0.0, 2.0 * math.pi, nu)
        vv = np.linspace(0.0, math.pi, nv)
        U, V = np.meshgrid(uu, vv)
        X0 = np.cos(U) * np.sin(V)
        Y0 = np.sin(U) * np.sin(V)
        Z0 = np.cos(V)

        for i in sphere_idx:
            i = int(i)
            c = centers[i]
            R = float(radii_sas[i])
            X = c[0] + R * X0
            Y = c[1] + R * Y0
            Z = c[2] + R * Z0
            ax.plot_wireframe(
                X,
                Y,
                Z,
                rstride=2,
                cstride=2,
                linewidth=0.5,
                alpha=alpha_sphere,
            )

    # Nice axis limits / equal aspect
    if centers.size > 0:
        mins = np.min(centers - radii_sas[:, None], axis=0)
        maxs = np.max(centers + radii_sas[:, None], axis=0)
        ctr = 0.5 * (mins + maxs)
        span = float(np.max(maxs - mins))
        if span <= 0:
            span = 1.0
        half = 0.55 * span
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    if title is not None:
        ax.set_title(title)

    # Optional legend (can get crowded)
    # ax.legend(loc="upper right")

    plt.show()


# -----------------------------
# Dataclasses for output sets
# -----------------------------


@dataclass(frozen=True)
class IntersectionPoint:
    """An SAS intersection point x_m ∈ I with an integer identifier."""

    id: int
    position: np.ndarray  # shape (3,)
    # indices of SAS spheres passing through this point
    spheres: Tuple[int, ...]


@dataclass(frozen=True)
class CircularArc:
    """
    SAS circular arc (or full circle) on intersection circle of two SAS spheres.

    Circle has:
      - center o, unit normal n, radius rc
      - orthonormal basis (u,v) spanning circle plane.

    Arc is parameterized by:
      theta in [-pi, pi) via p(theta)=o + rc*(cos(theta)*u + sin(theta)*v)

    Arc segment is [theta_start, theta_start+delta] along increasing theta (mod 2π).
    """

    id: int
    sphere_i: int
    sphere_j: int
    circle_key: Tuple[int, int]  # (min(i,j), max(i,j))
    circle_center: np.ndarray  # (3,)
    circle_normal: np.ndarray  # (3,) unit
    circle_radius: float
    u: np.ndarray  # (3,) unit
    v: np.ndarray  # (3,) unit, v = normal × u
    start_ip: Optional[int]  # None for full circle
    end_ip: Optional[int]  # None for full circle
    theta_start: float  # in [-pi, pi)
    delta: float  # in (0, 2pi], 2pi for full circle

    @property
    def is_full_circle(self) -> bool:
        return (
            self.start_ip is None
            and self.end_ip is None
            and abs(self.delta - 2.0 * math.pi) < 1e-10
        )


@dataclass(frozen=True)
class Loop:
    """Closed loop on a specific SAS sphere, composed of arc IDs."""

    id: int
    sphere: int
    arc_ids: Tuple[int, ...]


@dataclass(frozen=True)
class SphericalPatch:
    """
    SAS spherical patch on a specific SAS sphere.

    Represented by loop IDs forming its boundary (possibly multiple loops).
    """

    id: int
    sphere: int
    loop_ids: Tuple[int, ...]


@dataclass
class SASComponents:
    centers: np.ndarray  # (N,3) input centers
    radii_sas: np.ndarray  # (N,) SAS radii = vdw + probe
    active_sphere: np.ndarray  # (N,) bool, spheres not contained by another
    eps: float

    I: List[IntersectionPoint]
    l_m: List[CircularArc]
    loops: List[Loop]
    P_m: List[SphericalPatch]

    arcs_by_sphere: Dict[int, List[int]]
    loops_by_sphere: Dict[int, List[int]]
    patches_by_sphere: Dict[int, List[int]]

    # For surface='exterior'
    exterior_patch_ids: Optional[Set[int]] = None


# -----------------------------
# Numeric helpers
# -----------------------------


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, eps: float) -> Optional[np.ndarray]:
    n = _norm(v)
    if n < eps:
        return None
    return v / n


def _wrap_to_2pi(x: float) -> float:
    return x % (2.0 * math.pi)


def _angle_on_circle(
    center: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray
) -> float:
    w = p - center
    return math.atan2(float(np.dot(w, v)), float(np.dot(w, u)))


def _point_on_circle(
    center: np.ndarray, u: np.ndarray, v: np.ndarray, r: float, theta: float
) -> np.ndarray:
    return center + r * (math.cos(theta) * u + math.sin(theta) * v)


# -----------------------------
# Geometry primitives
# -----------------------------


def _sphere_sphere_intersection_circle(
    c1: np.ndarray, R1: float, c2: np.ndarray, R2: float, eps: float
) -> Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]]:
    """
    Return circle = S(c1,R1) ∩ S(c2,R2) as:
      (center o, unit normal n, radius rc, basis u,v)
    or None if no proper intersection circle (disjoint/tangent/contained/degenerate).
    """
    dvec = c2 - c1
    d = _norm(dvec)
    if d < eps:
        return None

    if d >= R1 + R2 - eps:
        return None
    if d <= abs(R1 - R2) + eps:
        return None

    n = dvec / d
    a = (R1 * R1 - R2 * R2 + d * d) / (2.0 * d)
    h2 = R1 * R1 - a * a
    if h2 <= eps:
        return None

    rc = math.sqrt(max(h2, 0.0))
    o = c1 + a * n

    # stable orthonormal basis on plane orthogonal to n
    if abs(float(n[0])) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(n, tmp), eps)
    if u is None:
        tmp = np.array([0.0, 0.0, 1.0])
        u = _unit(np.cross(n, tmp), eps)
        if u is None:
            return None
    v = np.cross(n, u)
    return o, n, float(rc), u, v


def _circle_sphere_intersections(
    circle_center: np.ndarray,
    circle_normal: np.ndarray,
    circle_radius: float,
    u: np.ndarray,
    v: np.ndarray,
    c: np.ndarray,
    R: float,
    eps: float,
) -> List[np.ndarray]:
    """Intersect a 3D circle with a sphere; return 0/1/2 points."""
    h = float(np.dot(c - circle_center, circle_normal))
    if abs(h) > R + eps:
        return []

    c_proj = c - h * circle_normal
    r_plane2 = R * R - h * h
    if r_plane2 < -eps:
        return []
    r_plane = math.sqrt(max(r_plane2, 0.0))

    dvec = c_proj - circle_center
    dx = float(np.dot(dvec, u))
    dy = float(np.dot(dvec, v))
    d = math.hypot(dx, dy)

    if d < eps:
        return []
    if d > circle_radius + r_plane + eps:
        return []
    if d < abs(circle_radius - r_plane) - eps:
        return []

    a = (circle_radius * circle_radius - r_plane * r_plane + d * d) / (2.0 * d)
    h2 = circle_radius * circle_radius - a * a
    if h2 < -eps:
        return []
    hlen = math.sqrt(max(h2, 0.0))

    x2 = a * dx / d
    y2 = a * dy / d

    rx = -dy * (hlen / d)
    ry = dx * (hlen / d)

    p3 = circle_center + u * (x2 + rx) + v * (y2 + ry)
    pts = [p3]
    if hlen > eps:
        p4 = circle_center + u * (x2 - rx) + v * (y2 - ry)
        pts.append(p4)
    return pts


# -----------------------------
# KDTree helpers
# -----------------------------


def _point_hidden_by_any_sphere_kdtree(
    p: np.ndarray,
    *,
    centers: np.ndarray,
    radii: np.ndarray,
    active_ids: np.ndarray,
    tree_active: cKDTree,
    maxR_active: float,
    exclude_global: Set[int],
    eps: float,
) -> bool:
    """
    True if p is strictly inside ANY active sphere (SAS ball) not in exclude_global.
    Candidate centers are found with a ball query of radius maxR_active.
    """
    cand_local = tree_active.query_ball_point(p, r=maxR_active + eps)
    for loc in cand_local:
        gid = int(active_ids[loc])
        if gid in exclude_global:
            continue
        if _norm(p - centers[gid]) < float(radii[gid]) - eps:
            return True
    return False


# -----------------------------
# Union-Find for point merging
# -----------------------------


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _merge_points_ckdtree(
    raw_points: np.ndarray,  # (M,3)
    raw_sphere_sets: List[Set[int]],
    tol: float,
) -> Tuple[np.ndarray, List[Set[int]], np.ndarray]:
    """
    Merge points that are within tol using cKDTree + union-find.

    Returns:
      merged_points: (K,3)
      merged_sphere_sets: list[set[int]] length K
      raw_to_merged: (M,) int mapping
    """
    M = raw_points.shape[0]
    if M == 0:
        return raw_points, [], np.zeros((0,), dtype=int)

    tree = cKDTree(raw_points)
    uf = _UnionFind(M)

    for i in range(M):
        neigh = tree.query_ball_point(raw_points[i], r=tol)
        for j in neigh:
            if j <= i:
                continue
            uf.union(i, j)

    # group members by root
    roots = [uf.find(i) for i in range(M)]
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(roots):
        groups[r].append(i)

    # deterministic order
    root_keys = sorted(groups.keys())
    raw_to_merged = np.empty((M,), dtype=int)

    merged_pts: List[np.ndarray] = []
    merged_sets: List[Set[int]] = []

    for mid, r in enumerate(root_keys):
        members = groups[r]
        merged = np.mean(raw_points[members], axis=0)
        sset: Set[int] = set()
        for m in members:
            sset.update(raw_sphere_sets[m])
            raw_to_merged[m] = mid
        merged_pts.append(merged)
        merged_sets.append(sset)

    return np.vstack(merged_pts), merged_sets, raw_to_merged


# -----------------------------
# Loop "inside" tests (Lemma 6.1)
# -----------------------------


def _closest_point_on_circle(
    x: np.ndarray,
    circle_center: np.ndarray,
    circle_normal: np.ndarray,
    circle_radius: float,
    u: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Closest Euclidean point from x to a 3D circle."""
    w = x - circle_center
    proj = w - float(np.dot(w, circle_normal)) * circle_normal
    proj_norm = _norm(proj)
    if proj_norm < eps:
        return circle_center + circle_radius * u
    return circle_center + circle_radius * (proj / proj_norm)


def _point_on_arc(p: np.ndarray, arc: CircularArc, tol: float) -> bool:
    """Membership test p ∈ arc (as a set) in a tolerance sense."""
    if arc.is_full_circle:
        return True
    if abs(_norm(p - arc.circle_center) - arc.circle_radius) > tol:
        return False
    theta = _angle_on_circle(arc.circle_center, arc.u, arc.v, p)
    d = _wrap_to_2pi(theta - arc.theta_start)
    return d <= arc.delta + tol


def _sample_point_on_arc(arc: CircularArc) -> np.ndarray:
    theta = 0.0 if arc.is_full_circle else (arc.theta_start + 0.5 * arc.delta)
    return _point_on_circle(
        arc.circle_center, arc.u, arc.v, arc.circle_radius, theta
    )


def _point_inside_loop_by_lemma(
    x: np.ndarray,
    loop_arc_ids: Sequence[int],
    arcs: Sequence[CircularArc],
    eps: float,
) -> bool:
    """
    Lemma 6.1 inside-test:
      - find nearest intersection circle among circles used by the loop,
      - compute closest point on that circle,
      - x inside loop iff that closest point lies on the loop.
    """
    if not loop_arc_ids:
        return True

    circle_keys: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for aid in loop_arc_ids:
        circle_keys[arcs[aid].circle_key].append(aid)

    best_d = None
    best_ck = None
    best_closest = None

    for ck, aid_list in circle_keys.items():
        arc0 = arcs[aid_list[0]]
        closest = _closest_point_on_circle(
            x,
            arc0.circle_center,
            arc0.circle_normal,
            arc0.circle_radius,
            arc0.u,
            eps,
        )
        d = _norm(x - closest)
        if best_d is None or d < best_d:
            best_d = d
            best_ck = ck
            best_closest = closest

    assert best_ck is not None and best_closest is not None

    for aid in circle_keys[best_ck]:
        if _point_on_arc(best_closest, arcs[aid], tol=10.0 * eps):
            return True
    return False


# -----------------------------
# SCC (for grouping loops into patches)
# -----------------------------


def _tarjan_scc(graph: Dict[int, List[int]]) -> List[List[int]]:
    """Tarjan SCC algorithm."""
    index = 0
    stack: List[int] = []
    on_stack: Set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    comps: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph[v]:
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in graph.keys():
        if v not in indices:
            strongconnect(v)

    return comps


# -----------------------------
# Exterior extraction (Step 5)
# -----------------------------


def _extract_exterior_patches(
    *,
    centers: np.ndarray,
    radii: np.ndarray,
    active: np.ndarray,
    eps: float,
    arcs: Sequence[CircularArc],
    loops: Sequence[Loop],
    patches: Sequence[SphericalPatch],
    patches_by_sphere: Dict[int, List[int]],
) -> Set[int]:
    """Flood fill patch adjacency from a far-away point to get eSAS."""
    arc_to_patches: Dict[int, List[int]] = defaultdict(list)

    for patch in patches:
        arc_ids: Set[int] = set()
        for lid in patch.loop_ids:
            arc_ids.update(loops[lid].arc_ids)
        for aid in arc_ids:
            arc_to_patches[aid].append(patch.id)

    patch_adj: Dict[int, Set[int]] = {p.id: set() for p in patches}
    for aid, plist in arc_to_patches.items():
        for a in plist:
            for b in plist:
                if a != b:
                    patch_adj[a].add(b)

    active_ids = np.nonzero(active)[0]
    if active_ids.size == 0:
        return set()

    center_mean = np.mean(centers[active_ids], axis=0)
    extent = float(
        np.max(
            np.linalg.norm(centers[active_ids] - center_mean, axis=1)
            + radii[active_ids]
        )
    )
    direction = np.array([1.0, 0.31, 0.17], dtype=float)
    direction = direction / _norm(direction)
    p_inf = (
        center_mean
        + (10.0 * extent + 10.0 * float(np.max(radii[active_ids]))) * direction
    )

    best_i = int(active_ids[0])
    best_val = float("inf")
    for i in active_ids:
        val = _norm(p_inf - centers[int(i)]) - float(radii[int(i)])
        if val < best_val:
            best_val = val
            best_i = int(i)

    vec = p_inf - centers[best_i]
    vec_u = _unit(vec, eps)
    if vec_u is None:
        vec_u = np.array([1.0, 0.0, 0.0])
    x0 = centers[best_i] + float(radii[best_i]) * vec_u

    start_patch = None
    for pid in patches_by_sphere.get(best_i, []):
        patch = patches[pid]
        if not patch.loop_ids:
            start_patch = pid
            break
        ok = True
        for lid in patch.loop_ids:
            if not _point_inside_loop_by_lemma(
                x0, loops[lid].arc_ids, arcs, eps=eps
            ):
                ok = False
                break
        if ok:
            start_patch = pid
            break

    if start_patch is None:
        plist = patches_by_sphere.get(best_i, [])
        start_patch = plist[0] if plist else 0

    visited: Set[int] = set([start_patch])
    q = deque([start_patch])
    while q:
        p = q.popleft()
        for nb in patch_adj[p]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)

    return visited


# -----------------------------
# Main entrypoint
# -----------------------------


def compute_I_lm_Pm(
    centers: np.ndarray,
    vdw_radii: np.ndarray,
    probe_radius: float,
    *,
    surface: str = "complete",
    eps: Optional[float] = None,
    remove_contained: bool = True,
) -> SASComponents:
    """
    Compute I, l_m, and P_m for the SAS.

    surface:
      - "complete": cSAS (boundary of union of SAS-balls)
      - "exterior": eSAS (outer connected component only, per Section 6.1 Step 5)
    """
    centers = np.asarray(centers, dtype=float)
    vdw_radii = np.asarray(vdw_radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N,3)")
    if vdw_radii.ndim != 1 or vdw_radii.shape[0] != centers.shape[0]:
        raise ValueError("vdw_radii must have shape (N,) matching centers")
    if probe_radius < 0:
        raise ValueError("probe_radius must be nonnegative")

    N = centers.shape[0]
    radii_sas = vdw_radii + float(probe_radius)
    maxR = float(np.max(radii_sas)) if N > 0 else 1.0
    if eps is None:
        eps = 1e-6 * maxR
    eps = float(eps)

    if N == 0:
        return SASComponents(
            centers=centers,
            radii_sas=radii_sas,
            active_sphere=np.zeros((0,), dtype=bool),
            eps=eps,
            I=[],
            l_m=[],
            loops=[],
            P_m=[],
            arcs_by_sphere={},
            loops_by_sphere={},
            patches_by_sphere={},
            exterior_patch_ids=set()
            if surface.lower() == "exterior"
            else None,
        )

    # KDTree on all centers for containment pre-pass
    tree_all = cKDTree(centers)

    # 0) remove contained SAS-balls (paper assumes no SAS-ball is included by another)
    active = np.ones(N, dtype=bool)
    if remove_contained:
        for i in range(N):
            # generous radius for candidates; then exact test
            cand = tree_all.query_ball_point(
                centers[i], r=(float(radii_sas[i]) + maxR + eps)
            )
            for j in cand:
                if j == i:
                    continue
                # containment: d + R_i <= R_j
                dij = _norm(centers[i] - centers[j])
                if dij + float(radii_sas[i]) <= float(radii_sas[j]) + eps:
                    active[i] = False
                    break

    active_ids = np.nonzero(active)[0]
    if active_ids.size == 0:
        return SASComponents(
            centers=centers,
            radii_sas=radii_sas,
            active_sphere=active,
            eps=eps,
            I=[],
            l_m=[],
            loops=[],
            P_m=[],
            arcs_by_sphere={},
            loops_by_sphere={},
            patches_by_sphere={},
            exterior_patch_ids=set()
            if surface.lower() == "exterior"
            else None,
        )

    # KDTree on active centers (all subsequent geometry uses only active spheres)
    centers_active = centers[active_ids]
    radii_active = radii_sas[active_ids]
    maxR_active = float(np.max(radii_active))
    tree_active = cKDTree(centers_active)

    # 1) neighbor sets among active spheres (by center distance, then exact sum-of-radii test)
    nA = active_ids.size
    neigh_local: List[Set[int]] = [set() for _ in range(nA)]

    for li in range(nA):
        gi = int(active_ids[li])
        # any neighbor j must satisfy dist(ci,cj) <= R_i + R_j <= R_i + maxR_active
        cand = tree_active.query_ball_point(
            centers_active[li], r=float(radii_active[li]) + maxR_active + eps
        )
        for lj in cand:
            if lj == li:
                continue
            gj = int(active_ids[lj])
            if gj == gi:
                continue
            d = _norm(centers[gi] - centers[gj])
            if d < float(radii_sas[gi]) + float(radii_sas[gj]) + eps:
                neigh_local[li].add(int(lj))

    # 2) intersection circles for each intersecting sphere pair
    circles: Dict[
        Tuple[int, int],
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray],
    ] = {}
    circle_entries: List[
        Tuple[
            int,
            int,
            int,
            int,
            Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray],
        ]
    ] = []

    for li in range(nA):
        gi = int(active_ids[li])
        for lj in neigh_local[li]:
            if lj <= li:
                continue
            gj = int(active_ids[lj])
            circle = _sphere_sphere_intersection_circle(
                centers[gi],
                float(radii_sas[gi]),
                centers[gj],
                float(radii_sas[gj]),
                eps=eps,
            )
            if circle is not None:
                key = (min(gi, gj), max(gi, gj))
                circles[key] = circle
                circle_entries.append((li, lj, gi, gj, circle))

    # 3) intersection points I (raw collection, then merge with KDTree)
    ip_tol = max(10.0 * eps, 1e-9)
    raw_pts: List[np.ndarray] = []
    raw_ssets: List[Set[int]] = []
    pair_to_raw: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for li, lj, gi, gj, (o, n, rc, u, v) in circle_entries:
        common = neigh_local[li].intersection(neigh_local[lj])
        for lk in common:
            gk = int(active_ids[lk])
            # enforce ordering gi < gj < gk to reduce duplicates before merge
            if gk <= max(gi, gj):
                continue

            pts = _circle_sphere_intersections(
                o, n, rc, u, v, centers[gk], float(radii_sas[gk]), eps=eps
            )
            for p in pts:
                # quick on-sphere consistency checks
                if abs(_norm(p - centers[gi]) - float(radii_sas[gi])) > ip_tol:
                    continue
                if abs(_norm(p - centers[gj]) - float(radii_sas[gj])) > ip_tol:
                    continue
                if abs(_norm(p - centers[gk]) - float(radii_sas[gk])) > ip_tol:
                    continue

                # must be exposed (not inside any other SAS-ball)
                if _point_hidden_by_any_sphere_kdtree(
                    p,
                    centers=centers,
                    radii=radii_sas,
                    active_ids=active_ids,
                    tree_active=tree_active,
                    maxR_active=maxR_active,
                    exclude_global={gi, gj, gk},
                    eps=eps,
                ):
                    continue

                rid = len(raw_pts)
                raw_pts.append(np.asarray(p, float))
                raw_ssets.append({gi, gj, gk})

                for a, b in ((gi, gj), (gi, gk), (gj, gk)):
                    pair_to_raw[(min(a, b), max(a, b))].append(rid)

    if raw_pts:
        raw_arr = np.vstack(raw_pts)
        merged_arr, merged_ssets, raw_to_merged = _merge_points_ckdtree(
            raw_arr, raw_ssets, tol=ip_tol
        )

        I_list: List[IntersectionPoint] = [
            IntersectionPoint(
                id=i,
                position=merged_arr[i],
                spheres=tuple(sorted(merged_ssets[i])),
            )
            for i in range(merged_arr.shape[0])
        ]

        pair_to_ip: Dict[Tuple[int, int], List[int]] = {}
        for pair, rlist in pair_to_raw.items():
            mids = [int(raw_to_merged[r]) for r in rlist]
            # unique + sorted
            mids = sorted(set(mids))
            pair_to_ip[pair] = mids
    else:
        I_list = []
        pair_to_ip = {}

    # 4) arcs l_m
    l_m: List[CircularArc] = []
    arcs_by_sphere: Dict[int, List[int]] = defaultdict(list)

    def add_arc(arc: CircularArc) -> None:
        l_m.append(arc)
        arcs_by_sphere[arc.sphere_i].append(arc.id)
        arcs_by_sphere[arc.sphere_j].append(arc.id)

    for (gi, gj), (o, n, rc, u, v) in circles.items():
        ip_ids = pair_to_ip.get((gi, gj), [])

        # helper for exposure check
        def is_circle_exposed_sample(theta: float) -> bool:
            test_p = _point_on_circle(o, u, v, rc, theta)
            return not _point_hidden_by_any_sphere_kdtree(
                test_p,
                centers=centers,
                radii=radii_sas,
                active_ids=active_ids,
                tree_active=tree_active,
                maxR_active=maxR_active,
                exclude_global={gi, gj},
                eps=eps,
            )

        if len(ip_ids) <= 1:
            # no usable subdivision points => full circle if exposed
            if is_circle_exposed_sample(0.0):
                aid = len(l_m)
                add_arc(
                    CircularArc(
                        id=aid,
                        sphere_i=gi,
                        sphere_j=gj,
                        circle_key=(gi, gj),
                        circle_center=o,
                        circle_normal=n,
                        circle_radius=float(rc),
                        u=u,
                        v=v,
                        start_ip=None,
                        end_ip=None,
                        theta_start=0.0,
                        delta=2.0 * math.pi,
                    )
                )
            continue

        angles = [
            _angle_on_circle(o, u, v, I_list[pid].position) for pid in ip_ids
        ]
        order = np.argsort(np.array(angles))
        ip_ids_sorted = [ip_ids[int(k)] for k in order]
        ang_sorted = [angles[int(k)] for k in order]

        # remove near-duplicate angles
        ip_u, ang_u = [], []
        for pid, ang in zip(ip_ids_sorted, ang_sorted):
            if not ang_u:
                ip_u.append(pid)
                ang_u.append(ang)
            else:
                diff = abs((_wrap_to_2pi(ang - ang_u[-1] + math.pi) - math.pi))
                if diff > 1e-7:
                    ip_u.append(pid)
                    ang_u.append(ang)
        ip_ids_sorted, ang_sorted = ip_u, ang_u

        m = len(ip_ids_sorted)
        if m < 2:
            if is_circle_exposed_sample(0.0):
                aid = len(l_m)
                add_arc(
                    CircularArc(
                        aid,
                        gi,
                        gj,
                        (gi, gj),
                        o,
                        n,
                        float(rc),
                        u,
                        v,
                        None,
                        None,
                        0.0,
                        2.0 * math.pi,
                    )
                )
            continue

        for a_idx in range(m):
            b_idx = (a_idx + 1) % m
            pa = ip_ids_sorted[a_idx]
            pb = ip_ids_sorted[b_idx]
            ta = ang_sorted[a_idx]
            tb = ang_sorted[b_idx]
            delta = _wrap_to_2pi(tb - ta)
            if delta < 1e-12:
                continue
            tmid = ta + 0.5 * delta
            midp = _point_on_circle(o, u, v, rc, tmid)
            if _point_hidden_by_any_sphere_kdtree(
                midp,
                centers=centers,
                radii=radii_sas,
                active_ids=active_ids,
                tree_active=tree_active,
                maxR_active=maxR_active,
                exclude_global={gi, gj},
                eps=eps,
            ):
                continue
            aid = len(l_m)
            add_arc(
                CircularArc(
                    aid,
                    gi,
                    gj,
                    (gi, gj),
                    o,
                    n,
                    float(rc),
                    u,
                    v,
                    pa,
                    pb,
                    ta,
                    float(delta),
                )
            )

    # 5) loops on each sphere (Step 3)
    loops: List[Loop] = []
    loops_by_sphere: Dict[int, List[int]] = defaultdict(list)

    for gi in active_ids.tolist():
        incident_arc_ids = arcs_by_sphere.get(gi, [])
        if not incident_arc_ids:
            continue

        adj: Dict[int, List[int]] = defaultdict(list)
        full_circle_arcs: List[int] = []
        for aid in incident_arc_ids:
            arc = l_m[aid]
            if arc.is_full_circle:
                full_circle_arcs.append(aid)
                continue
            adj[arc.start_ip].append(aid)
            adj[arc.end_ip].append(aid)

        unvisited: Set[int] = set(incident_arc_ids)

        # full circles are loops by themselves
        for aid in full_circle_arcs:
            if aid in unvisited:
                lid = len(loops)
                loops.append(Loop(lid, gi, (aid,)))
                loops_by_sphere[gi].append(lid)
                unvisited.remove(aid)

        # build loops by walking arc adjacency at intersection points
        while unvisited:
            start_aid = next(iter(unvisited))
            start_arc = l_m[start_aid]
            if start_arc.is_full_circle:
                unvisited.remove(start_aid)
                continue

            start_ip = start_arc.start_ip
            current_aid = start_aid
            current_ip = start_ip
            loop_arc_ids: List[int] = []
            max_steps = len(incident_arc_ids) + 5

            for _ in range(max_steps):
                loop_arc_ids.append(current_aid)
                unvisited.discard(current_aid)

                arc = l_m[current_aid]
                next_ip = (
                    arc.end_ip if current_ip == arc.start_ip else arc.start_ip
                )

                candidates = [x for x in adj[next_ip] if x != current_aid]
                if not candidates:
                    break
                next_aid = next(
                    (x for x in candidates if x in unvisited), candidates[0]
                )

                current_aid = next_aid
                current_ip = next_ip
                if current_ip == start_ip and current_aid == start_aid:
                    break

                if current_ip == start_ip and start_aid not in unvisited:
                    # we returned to start ip; consider it closed enough
                    break

            lid = len(loops)
            loops.append(Loop(lid, gi, tuple(loop_arc_ids)))
            loops_by_sphere[gi].append(lid)

    # 6) spherical patches on each sphere (Step 4)
    P_m: List[SphericalPatch] = []
    patches_by_sphere: Dict[int, List[int]] = defaultdict(list)

    for gi in active_ids.tolist():
        loop_ids = loops_by_sphere.get(gi, [])
        if not loop_ids:
            pid = len(P_m)
            P_m.append(SphericalPatch(pid, gi, tuple()))
            patches_by_sphere[gi].append(pid)
            continue

        # build directed graph: loop a -> loop b if sample point of a lies inside b
        graph: Dict[int, List[int]] = {lid: [] for lid in loop_ids}
        loop_sample_point = {
            lid: _sample_point_on_arc(l_m[loops[lid].arc_ids[0]])
            for lid in loop_ids
        }

        for a in loop_ids:
            x = loop_sample_point[a]
            for b in loop_ids:
                if a == b:
                    graph[a].append(b)
                else:
                    if _point_inside_loop_by_lemma(
                        x, loops[b].arc_ids, l_m, eps=eps
                    ):
                        graph[a].append(b)

        # strongly connected components correspond to patch-boundary loop sets
        comps = _tarjan_scc(graph)
        for comp in comps:
            pid = len(P_m)
            P_m.append(SphericalPatch(pid, gi, tuple(sorted(comp))))
            patches_by_sphere[gi].append(pid)

    exterior_patch_ids: Optional[Set[int]] = None
    if surface.lower() == "exterior":
        exterior_patch_ids = _extract_exterior_patches(
            centers=centers,
            radii=radii_sas,
            active=active,
            eps=eps,
            arcs=l_m,
            loops=loops,
            patches=P_m,
            patches_by_sphere=patches_by_sphere,
        )

    return SASComponents(
        centers=centers,
        radii_sas=radii_sas,
        active_sphere=active,
        eps=eps,
        I=I_list,
        l_m=l_m,
        loops=loops,
        P_m=P_m,
        arcs_by_sphere=dict(arcs_by_sphere),
        loops_by_sphere=dict(loops_by_sphere),
        patches_by_sphere=dict(patches_by_sphere),
        exterior_patch_ids=exterior_patch_ids,
    )


def main():
    np.random.seed(1234)
    # random 10 points
    centers = np.random.rand(10, 3) * 10.0
    vdw = np.random.rand(10) * 1.5 + 1.0
    probe = 1.4
    comps = compute_I_lm_Pm(centers, vdw, probe, surface="exterior")
    plot_sas_components(comps)


if __name__ == "__main__":
    main()

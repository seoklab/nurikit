# pyright: reportUnusedImport=false
# ruff: noqa

import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import nuri
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from nuri.core import Molecule
from nuri.desc import shrake_rupley_sasa
from scipy.spatial import KDTree
from scipy.spatial import distance as D
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_enable=False)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm


def _any_perpendicular(v: np.ndarray) -> np.ndarray:
    w = np.array(
        [
            math.copysign(v[2], v[0]),
            math.copysign(v[2], v[1]),
            -math.copysign(v[0], v[2]) - math.copysign(v[1], v[2]),
        ]
    )
    return _normalize(w)


def _angle_on_circle(
    pts: np.ndarray,
    cntr: np.ndarray,
    frame: np.ndarray,
    radius: float | None = None,
):
    vs = pts - cntr
    if radius is None:
        radius = np.linalg.norm(vs, axis=-1, keepdims=True)
    vs /= radius
    x, y = frame[:2]
    theta = np.atan2(np.dot(vs, y), np.dot(vs, x))
    return np.where(theta >= 0, theta, 2 * math.pi + theta)


@dataclass
class Contact:
    js: np.ndarray
    dijs: np.ndarray

    def __len__(self):
        return len(self.js)


def _find_contacts_pairs(
    kdt: KDTree,
    sasr: np.ndarray,
    approx_cutoff: float,
    eps: float = 1e-6,
):
    pairs = kdt.query_ball_tree(kdt, approx_cutoff * 2)

    contacts: list[Contact] = []
    for i, (pi, sri, js) in enumerate(zip(kdt.data, sasr, pairs, strict=True)):
        js = np.array(js)
        js = js[js > i]
        if js.size == 0:
            contacts.append(
                Contact(
                    np.empty((0,), dtype=int),
                    np.empty((0,), dtype=float),
                )
            )
            continue

        pjs = kdt.data[js]
        dijs = np.linalg.norm(pjs - pi, axis=-1)
        cmap = (dijs < sri + sasr[js]) & (dijs > np.abs(sri - sasr[js]) + eps)
        contacts.append(Contact(js[cmap], dijs[cmap]))

    return contacts


@dataclass
class Circle:
    id: int

    frame: np.ndarray
    cntr: np.ndarray
    radius: float

    on: list[int]

    def theta(self, pts: np.ndarray) -> np.ndarray:
        return _angle_on_circle(pts, self.cntr, self.frame, self.radius)


@dataclass
class CCI:
    """Circle-Circle Intersection"""

    id: int
    pt: np.ndarray
    circles: list[Circle]


@dataclass
class NodeInfo:
    begin: CCI
    end: CCI


@dataclass
class Arc:
    id: int
    parent: Circle

    tbegin: float = 0.0
    tend: float = 2 * math.pi

    points: NodeInfo | None = None

    @property
    def radian(self) -> float:
        return self.tend - self.tbegin


@dataclass
class Intersection:
    ijk: tuple[int, int, int]

    pijk: CCI

    def __post_init__(self):
        self.ijk = tuple(map(int, self.ijk))  # pyright: ignore


@dataclass
class Toroid:
    ij: tuple[int, int]

    circle: Circle
    phirange: tuple[float, float]

    def __post_init__(self):
        self.ij = tuple(map(int, self.ij))  # pyright: ignore


@dataclass
class ToroidSegment:
    arc: Arc
    parent: Toroid


def _find_toroids(kdt: KDTree, contacts: list[Contact], sasr: np.ndarray):
    toroids: dict[tuple[int, int], Toroid] = {}

    for i, (pi, c, sri) in enumerate(zip(tqdm(kdt.data[:-1]), contacts, sasr)):
        if c.js.size == 0:
            continue

        pjs = kdt.data[c.js]
        uijs = (pjs - pi) / c.dijs[:, None]
        for j, pj, zij, srj, dij in zip(c.js, pjs, uijs, sasr[c.js], c.dijs):
            tij = 0.5 * ((pi + pj) + (pj - pi) * (sri**2 - srj**2) / dij**2)
            Rij = (
                0.5
                * math.sqrt(
                    ((sri + srj) ** 2 - dij**2) * (dij**2 - (sri - srj) ** 2)
                )
                / dij
            )

            yij = _any_perpendicular(zij)
            pk = tij + Rij * yij
            iphi, jphi = _angle_on_circle(
                np.stack([pi, pj]),
                pk,
                np.stack([zij, yij]),
            )
            if jphi < iphi:
                yij = -yij
                iphi = 2 * math.pi - iphi
                jphi = 2 * math.pi - jphi

            toroids[(i, j)] = Toroid(
                ij=(i, j),
                circle=Circle(
                    id=len(toroids),
                    frame=np.stack([np.cross(yij, zij), yij, zij]),
                    cntr=tij,
                    radius=Rij,
                    on=[i, j],
                ),
                phirange=(iphi, jphi),
            )

    return toroids


def _probe_no_contact(
    kdt: KDTree,
    allowed: set[int],
    probe: np.ndarray,
    sasr: np.ndarray,
    cutoff: float,
):
    nbrs = kdt.query_ball_point(probe, cutoff)
    nbrs = list(set(nbrs) - allowed)
    npts = kdt.data[nbrs]
    dists = D.cdist(probe[None], npts).squeeze(0)
    return np.all(dists >= sasr[nbrs] - 1e-6)


def _circle_intersections(
    center: np.ndarray,
    radius: float,
    ci: Circle,
    cj: Circle,
    eps: float = 1e-12,
):
    ui = ci.frame[2]
    uj = cj.frame[2]
    ti = ci.cntr

    uij = np.cross(ui, uj)
    sinw = np.linalg.norm(uij)
    uij /= sinw

    utb = np.cross(uij, ui)
    bij = ti + utb * (np.dot(uj, cj.cntr - ti) / sinw)
    hsq = radius**2 - np.dot(bij - center, bij - center)
    if hsq <= eps:
        return None

    hij = math.sqrt(hsq)
    return bij + hij * uij, bij - hij * uij


def _sas_intersections(
    kdt: KDTree,
    contacts: list[Contact],
    toroids: dict[tuple[int, int], Toroid],
    sasr: np.ndarray,
    cutoff: float,
    eps: float = 1e-12,
):
    xm: list[Intersection] = []

    for i, (pi, c, sri) in enumerate(zip(tqdm(kdt.data[:-1]), contacts, sasr)):
        if len(c) < 2:
            continue

        j: int
        for j in c.js:
            ks = np.intersect1d(c.js, contacts[j].js, assume_unique=True)
            if ks.size == 0:
                continue

            tij = toroids.get((i, j))
            if tij is None:
                continue

            ci = tij.circle
            for k in ks:
                tik = toroids.get((i, k))
                if tik is None:
                    continue

                cj = tik.circle
                inter = _circle_intersections(pi, sri, ci, cj, eps=eps)
                if inter is None:
                    continue

                allowed = {i, j, k}
                for pt in inter:
                    if _probe_no_contact(kdt, allowed, pt, sasr, cutoff):
                        xm.append(
                            Intersection(
                                ijk=(i, j, k),
                                pijk=CCI(id=len(xm), pt=pt, circles=[ci, cj]),
                            )
                        )

    return xm


def _points_on_circle(
    cntr: np.ndarray,
    frame: np.ndarray,
    radius: np.ndarray | float,
    theta: np.ndarray | float,
) -> np.ndarray:
    x, y = frame[:2]
    return cntr + radius * (x * np.cos(theta) + y * np.sin(theta))


def _points_on_circle_center(
    c: Circle,
    theta: np.ndarray | float,
) -> np.ndarray:
    return _points_on_circle(c.cntr, c.frame, c.radius, theta)


def _sas_arcs(
    kdt: KDTree,
    toroids: dict[tuple[int, int], Toroid],
    concave: list[Intersection],
    sasr: np.ndarray,
    cutoff: float,
):
    vertices: dict[tuple[int, int], list[Intersection]] = defaultdict(list)
    for p in concave:
        i, j, k = p.ijk
        vertices[(i, j)].append(p)
        vertices[(i, k)].append(p)
        vertices[(j, k)].append(p)

    segs: list[ToroidSegment] = []
    for (i, j), tij in tqdm(toroids.items()):
        allowed = {i, j}
        x = tij.circle.frame[0]

        vs = vertices[(i, j)]
        if len(vs) < 2:
            test = tij.circle.cntr + tij.circle.radius * x
            if _probe_no_contact(kdt, allowed, test, sasr, cutoff):
                segs.append(
                    ToroidSegment(
                        arc=Arc(id=len(segs), parent=tij.circle),
                        parent=tij,
                    )
                )
            continue

        angles = tij.circle.theta(np.stack([v.pijk.pt for v in vs]))
        order = np.argsort(angles)
        order = np.append(order, order[0])

        angles = angles[order]
        angles[-1] += 2 * math.pi
        vs = [vs[i] for i in order]

        mid = 0.5 * (angles[:-1] + angles[1:])
        tests = _points_on_circle_center(tij.circle, mid[:, None])
        for test, ((b, ba), (e, ea)) in zip(
            tests,
            itertools.pairwise(zip(vs, angles)),
            strict=True,
        ):
            if _probe_no_contact(kdt, allowed, test, sasr, cutoff):
                segs.append(
                    ToroidSegment(
                        arc=Arc(
                            id=len(segs),
                            parent=tij.circle,
                            tbegin=ba,
                            tend=ea,
                            points=NodeInfo(begin=b.pijk, end=e.pijk),
                        ),
                        parent=tij,
                    )
                )

    return segs


def _merge_arcs(sg: nx.MultiGraph, arcs: list[tuple[int, int, int, Arc]]):
    length = np.array([a.radian for *_, a in arcs])
    order = np.argsort(length)

    length_sum: dict[int, float] = defaultdict(float)
    for s, d, _, arc in arcs:
        length_sum[s] += arc.radian
        length_sum[d] += arc.radian

    g = sg.copy()
    idx: int
    equiv_map: dict[int, int] = {}
    key_map: dict[tuple[int, int, int], int] = {}
    for idx in order:
        s, d, k, arc = arcs[idx]
        s = equiv_map.get(s, s)
        d = equiv_map.get(d, d)
        k = key_map.get((*sorted([s, d]), k), k)  # type: ignore
        if s == d:
            continue

        g.remove_edge(s, d, k)

        sel = max(s, d, key=lambda x: length_sum[x])
        nsel = min(s, d, key=lambda x: length_sum[x])
        for k, v in equiv_map.items():
            if v == nsel:
                equiv_map[k] = sel
        equiv_map[nsel] = sel

        for _, nbr, nk, narc in g.edges(nsel, data="arc", keys=True):
            if nbr == sel:
                continue

            nk = g.add_edge(sel, nbr, arc=narc)
            key_map[(*sorted([sel, nbr]), nk)] = nk  # type: ignore

        g.remove_node(nsel)

        if g.number_of_nodes() == g.number_of_edges() and all(
            g.degree[n] == 2  # type: ignore
            for n in g.nodes
        ):
            break

    return [(s, d, k, g[s][d][k]["arc"]) for s, d, k in nx.edge_dfs(g)]


def _loops_on_sphere(all_arcs: list[Arc]):
    sphere_saddle: dict[int, list[Arc]] = defaultdict(list)
    for a in all_arcs:
        for sid in a.parent.on:
            sphere_saddle[sid].append(a)

    sphere_loops: dict[int, list[list[Arc]]] = {}
    for i, arcs in tqdm(sphere_saddle.items()):
        loops = sphere_loops[i] = []

        g = nx.MultiGraph()
        for a in arcs:
            if a.points is None:
                loops.append([a])
                continue

            g.add_edge(a.points.begin.id, a.points.end.id, arc=a)

        for comp in nx.connected_components(g):
            sg = g.subgraph(comp)
            loop = [
                (s, d, k, sg[s][d][k]["arc"]) for s, d, k in nx.edge_dfs(sg)
            ]
            if len(loop) != len(comp) or any(
                sg.degree[n] != 2  # type: ignore
                for n in sg.nodes
            ):
                logging.warning(
                    "Sphere %d doesn't form a proper loop: l: %d, n: %d",
                    i,
                    len(loop),
                    len(comp),
                )
                loop = _merge_arcs(sg, loop)  # type: ignore

            loops.append([arc for *_, arc in loop])

    return sphere_loops


@dataclass
class SphericalPatch:
    id: int

    sid: int
    center: np.ndarray
    radius: float

    loops: list[list[ToroidSegment]] = field(default_factory=list)


def _nearest_point_on_circle(x: np.ndarray, c: Circle, eps: float = 1e-6):
    y, z = c.frame[1:]
    v = x - c.cntr
    proj = v - np.dot(v, z) * z
    pnorm = np.linalg.norm(proj)
    if pnorm < eps:
        return c.cntr + c.radius * y

    return c.cntr + c.radius * proj / pnorm


def _point_on_arc(p: np.ndarray, arc: Arc, tol: float) -> bool:
    if arc.points is None:
        return True

    theta = arc.parent.theta(p).item()
    return arc.tbegin - tol <= theta <= arc.tend + tol


def _point_inside_loop(
    x: np.ndarray,
    loops: list[Arc],
    eps: float = 1e-6,
):
    groups: dict[int, list[Arc]] = defaultdict(list)
    for arc in loops:
        groups[arc.parent.id].append(arc)

    toroids = list(groups.values())
    tests = np.stack(
        [_nearest_point_on_circle(x, a0.parent) for a0, *_ in toroids]
    )
    dists = D.cdist(x[None], tests).squeeze(0)
    sel = int(np.argmin(dists))

    for arc in toroids[sel]:
        if _point_on_arc(x, arc, tol=10.0 * eps):
            return True
    return False


def _sas_spherical_patches(
    kdt: KDTree,
    loops: dict[int, list[list[ToroidSegment]]],
    sasr: np.ndarray,
    cutoff: float,
):
    spherical: list[SphericalPatch] = []

    for i, (pi, sri) in enumerate(zip(tqdm(kdt.data), sasr)):
        ls = loops.get(i)
        if ls is None:
            nbrs = kdt.query_ball_point(pi, sri + cutoff)
            nbrs = [n for n in nbrs if n != i]
            npts = kdt.data[nbrs]
            dists = D.cdist(pi[None], npts).squeeze(0)
            if np.all(dists >= sri + sasr[nbrs]):
                spherical.append(
                    SphericalPatch(
                        id=len(spherical),
                        sid=i,
                        center=pi,
                        radius=sri,
                    )
                )
            continue

        samples = [
            _points_on_circle_center(
                seg.arc.parent,
                0.5 * (seg.arc.tbegin + seg.arc.tend),
            )
            for seg, *_ in ls
        ]

        g = nx.DiGraph()
        g.add_nodes_from(range(len(ls)))
        for j, x in enumerate(samples):
            for k, lk in enumerate(ls):
                if j == k:
                    continue

                if _point_inside_loop(x, [seg.arc for seg in lk]):
                    g.add_edge(j, k)

        for comp in nx.strongly_connected_components(g):
            spherical.append(
                SphericalPatch(
                    id=len(spherical),
                    sid=i,
                    center=pi,
                    radius=sri,
                    loops=[ls[i] for i in comp],
                )
            )

    return spherical


@dataclass
class SasComponents:
    xm: list[Intersection]
    lm: list[ToroidSegment]
    pm: list[SphericalPatch]

    cm: dict[tuple[int, int], Toroid]
    sasr: np.ndarray


def sas_components(pts: np.ndarray, sasr: np.ndarray):
    cutoff = np.max(sasr)

    kdt = KDTree(pts)
    contacts = _find_contacts_pairs(kdt, sasr, cutoff)

    cm = _find_toroids(kdt, contacts, sasr)
    xm = _sas_intersections(kdt, contacts, cm, sasr, cutoff)
    lm = _sas_arcs(kdt, cm, xm, sasr, cutoff)

    lons = _loops_on_sphere([seg.arc for seg in lm])
    lons = {
        k: [[lm[arc.id] for arc in loop] for loop in loops]
        for k, loops in lons.items()
    }
    pm = _sas_spherical_patches(kdt, lons, sasr, cutoff)

    return SasComponents(
        xm=xm,
        lm=lm,
        pm=pm,
        cm=cm,
        sasr=sasr,
    )


def _sas_arc_angle(ei: Arc, ej: Arc, inter: int, patch: SphericalPatch):
    assert ei.points is not None
    assert ej.points is not None

    ei_begin = ei.points.begin.id == inter
    ej_begin = ej.points.begin.id == inter
    p = ei.points.begin.pt if ei_begin else ei.points.end.pt

    vi = np.cross(ei.parent.frame[2], (p - ei.parent.cntr) / ei.parent.radius)
    if ei_begin:
        vi = -vi
    vj = np.cross(ej.parent.frame[2], (p - ej.parent.cntr) / ej.parent.radius)
    if not ej_begin:
        vj = -vj

    z = (p - patch.center) / patch.radius
    y = np.cross(z, vi)
    ccw = ei_begin == (patch.sid == ei.parent.on[0])

    theta = math.atan2(np.dot(y, vj), np.dot(vi, vj))
    if not ccw:
        theta = -theta

    return theta


def _loop_angle_total(loop: list[ToroidSegment], patch: SphericalPatch):
    graph = nx.MultiGraph()
    for seg in loop:
        if seg.arc.points is not None:
            graph.add_edge(
                seg.arc.points.begin.id,
                seg.arc.points.end.id,
                arc=seg.arc,
            )

    asum = 0.0
    for inter in graph.nodes:
        edges = list(graph.edges(inter, data="arc"))
        for (*_, ei), (*_, ej) in itertools.combinations(edges, 2):
            asum += _sas_arc_angle(ei, ej, inter, patch)
    return asum


def _gauss_bonnet_area(patch: SphericalPatch):
    chi = 2 - len(patch.loops)

    asum = sum(_loop_angle_total(loop, patch) for loop in patch.loops)

    lsum = 0.0
    for loop in patch.loops:
        for seg in loop:
            ij = int(patch.sid == seg.parent.ij[1])
            phi = seg.parent.phirange[ij] - 1.5 * math.pi
            if ij == 1:
                phi = -phi
            lsum += seg.arc.radian * math.sin(phi)

    area = patch.radius**2 * (2 * math.pi * chi - (asum + lsum))
    return area


def sas_area(sas: SasComponents):
    sasa = np.zeros(len(sas.sasr))
    for p in sas.pm:
        sasa[p.sid] += _gauss_bonnet_area(p)
    return sasa


@dataclass
class SaddlePatch:
    parent: Toroid
    arc: Arc

    edge_cntrs: np.ndarray
    edge_radii: np.ndarray
    singularity: np.ndarray | None = None

    @classmethod
    def from_arc(
        cls,
        seg: ToroidSegment,
        pts: np.ndarray,
        sasr: np.ndarray,
        rprobe: float,
    ):
        tor = seg.parent
        circ = tor.circle
        ij = np.array(tor.ij)

        # [2, 1]
        sris = sasr[ij, None]
        ris = sris - rprobe
        scales = ris / sris
        # [2, 3]
        cntr = circ.cntr * scales + pts[ij] * (1 - scales)
        radii = circ.radius * scales.squeeze(-1)

        singularity = None
        if circ.radius < rprobe:
            offset = math.sqrt(rprobe**2 - circ.radius**2)
            scales = ris / (ris + offset)
            singularity = circ.cntr * scales + pts[ij] * (1 - scales)

        return cls(
            parent=tor,
            arc=seg.arc,
            edge_cntrs=cntr,
            edge_radii=radii,
            singularity=singularity,
        )


@dataclass
class ConvexPatch:
    id: int

    sid: int
    center: np.ndarray
    radius: float

    loops: list[list[SaddlePatch]] = field(default_factory=list)

    @classmethod
    def from_spherical_patch(
        cls,
        sp: SphericalPatch,
        saddle: list[SaddlePatch],
        rprobe: float,
    ):
        self = cls(
            id=sp.id,
            sid=sp.sid,
            center=sp.center,
            radius=sp.radius - rprobe,
            loops=[[saddle[seg.arc.id] for seg in loop] for loop in sp.loops],
        )
        return self


@dataclass
class ConcavePatch:
    parent: Intersection

    loops: list[list[Arc]] = field(default_factory=list)


def _concave_circles(
    probes: list[Intersection],
    pcm: dict[tuple[int, int], Toroid],
    atoms: np.ndarray,
    sasr: np.ndarray,
    rprobe: float,
):
    circles = [tij.circle for tij in pcm.values()]

    for i, p in enumerate(probes):
        ijk = np.array(p.ijk)
        aijk = atoms[ijk]
        p = p.pijk.pt

        scale = rprobe / sasr[ijk, None]
        pijk = p * (1 - scale) + aijk * scale
        vijk = (p - aijk) / sasr[ijk, None]

        xs = _normalize(np.roll(pijk, -1, axis=0) - pijk)
        ys = _normalize(vijk + np.roll(vijk, -1, axis=0))
        frames = np.stack([xs, ys, np.cross(xs, ys, axis=-1)], axis=1)
        for fr in frames:
            circles.append(
                Circle(
                    id=len(circles),
                    frame=fr,
                    cntr=p,
                    radius=rprobe,
                    on=[i],
                )
            )

    return circles


def _point_inside_triangle(
    pts: np.ndarray,
    normals: np.ndarray,
    signs: np.ndarray,
):
    return np.all(
        np.einsum("...c,nc,n->...n", pts, normals, signs) >= 0,
        axis=-1,
    )


def _concave_inside_angles_signs(
    pts: np.ndarray,
    probes: list[Intersection],
    bcircles: list[list[Circle]],
    sasr: np.ndarray,
):
    angles: dict[int, tuple[float, float]] = {}
    normals = []
    signs = []
    for p, (ci, cj, ck) in zip(probes, bcircles, strict=True):
        ijk = np.array(p.ijk)
        triangle = (pts[ijk] - p.pijk.pt) / sasr[ijk, None]

        for pi, pj, circ in zip(
            triangle,
            np.roll(triangle, -1, axis=0),
            [ci, cj, ck],
        ):
            tb, te = _angle_on_circle(
                np.stack([pi, pj]),
                np.zeros(3),
                circ.frame,
                1.0,
            )
            assert te > tb
            angles[circ.id] = (tb, te)

        test = np.sum(triangle, axis=0) / math.sqrt(
            3 + 2 * np.sum(triangle * np.roll(triangle, -1, axis=0))
        )

        normal = np.stack([ci.frame[2], cj.frame[2], ck.frame[2]])
        normals.append(normal)
        signs.append(np.sign(np.einsum("c,nc->n", test, normal)))

    assert len(angles) == len(probes) * 3
    return angles, np.stack(normals), np.stack(signs)


def _intersection_on_segment(
    point: np.ndarray,
    circle: Circle,
    bounds: tuple[float, float],
    eps: float = 1e-6,
):
    return bounds[0] - eps <= circle.theta(point) <= bounds[1] + eps


def _concave_intersections(
    kdt: KDTree,
    pcircles: list[list[Circle]],
    rprobe: float,
    bangles: dict[int, tuple[float, float]],
    bnormals: np.ndarray,
    bsigns: np.ndarray,
    eps: float = 1e-12,
):
    xm: list[list[CCI]] = [[] for _ in range(len(kdt.data))]
    idgen = itertools.count()

    for xmi, pi, circles, bn, bs in zip(
        tqdm(xm),
        kdt.data,
        pcircles,
        bnormals,
        bsigns,
        strict=True,
    ):
        for ci, cj in itertools.combinations(circles, 2):
            inter = _circle_intersections(pi, rprobe, ci, cj, eps=eps)
            if inter is None:
                continue

            allowed = {*ci.on, *cj.on}
            for pt in inter:
                if (
                    len(ci.on) > 1
                    and len(cj.on) > 1
                    and not _point_inside_triangle((pt - pi) / rprobe, bn, bs)
                ):
                    continue

                if len(ci.on) == 1 and not _intersection_on_segment(
                    pt,
                    ci,
                    bangles[ci.id],
                ):
                    continue

                if len(cj.on) == 1 and not _intersection_on_segment(
                    pt,
                    cj,
                    bangles[cj.id],
                ):
                    continue

                nbrs = kdt.query_ball_point(pt, rprobe - eps)
                if set(nbrs) - allowed:
                    continue

                xmi.append(
                    CCI(
                        id=next(idgen),
                        pt=pt,
                        circles=[ci, cj],
                    )
                )

    return xm


def _concave_arcs(
    i: int,
    kdt: KDTree,
    circles: list[Circle],
    inter: list[CCI],
    rprobe: float,
    bangles: dict[int, tuple[float, float]],
    bnormal: np.ndarray,
    bsign: np.ndarray,
    idgen: itertools.count,
    eps: float = 1e-12,
):
    vertices: dict[int, list[CCI]] = defaultdict(list)
    for p in inter:
        for c in p.circles:
            vertices[c.id].append(p)

    arcs: list[Arc] = []
    for tij in circles:
        vs = vertices[tij.id]
        if len(vs) < 2:
            continue

        angles = tij.theta(np.stack([v.pt for v in vs]))
        order = np.argsort(angles)
        order = np.append(order, order[0])

        angles = angles[order]
        angles[-1] += 2 * math.pi
        vs = [vs[i] for i in order]

        allowed = set(tij.on)
        mid = 0.5 * (angles[:-1] + angles[1:])
        tests = _points_on_circle_center(tij, mid[:, None])
        pcntr = kdt.data[i]
        for test, ((b, ba), (e, ea)) in zip(
            tests,
            itertools.pairwise(zip(vs, angles)),
            strict=True,
        ):
            if len(tij.on) == 1 and not _intersection_on_segment(
                test,
                tij,
                bangles[tij.id],
            ):
                continue

            if len(tij.on) > 1 and not _point_inside_triangle(
                (test - pcntr) / rprobe, bnormal, bsign
            ):
                continue

            nbrs = kdt.query_ball_point(test, rprobe - eps)
            if not set(nbrs) - allowed:
                arcs.append(
                    Arc(
                        id=next(idgen),
                        parent=tij,
                        tbegin=ba,
                        tend=ea,
                        points=NodeInfo(begin=b, end=e),
                    )
                )

    return arcs


def _probe_loops_on_sphere(probe_arcs: list[list[Arc]]):
    probe_loops: list[list[list[Arc]]] = [[] for _ in probe_arcs]
    for i, (arcs, loops) in enumerate(
        zip(probe_arcs, probe_loops, strict=True)
    ):
        g = nx.MultiGraph()
        for a in arcs:
            if a.points is None:
                loops.append([a])
                continue

            g.add_edge(a.points.begin.id, a.points.end.id, arc=a)

        for comp in nx.connected_components(g):
            sg = g.subgraph(comp)
            loop = [
                (s, d, k, sg[s][d][k]["arc"]) for s, d, k in nx.edge_dfs(sg)
            ]
            if len(loop) != len(comp) or any(
                sg.degree[n] != 2  # type: ignore
                for n in sg.nodes
            ):
                logging.warning(
                    "Probe %d doesn't form a proper loop: l: %d, n: %d",
                    i,
                    len(loop),
                    len(comp),
                )
                loop = _merge_arcs(sg, loop)  # type: ignore

            loops.append([arc for *_, arc in loop])

    return probe_loops


def _concave_spherical_patches(
    probes: list[Intersection],
    loops: list[list[list[Arc]]],
):
    concave: list[ConcavePatch] = []

    for pi, li in zip(tqdm(probes), loops, strict=True):
        samples = np.stack(
            [
                _points_on_circle_center(
                    arc.parent, 0.5 * (arc.tbegin + arc.tend)
                )
                for arc, *_ in li
            ]
        )

        g = nx.DiGraph()
        g.add_nodes_from(range(len(li)))
        for j, x in enumerate(samples):
            for k, lk in enumerate(li):
                if j == k:
                    continue

                if _point_inside_loop(x, lk):
                    g.add_edge(j, k)

        for comp in nx.strongly_connected_components(g):
            concave.append(
                ConcavePatch(
                    parent=pi,
                    loops=[li[i] for i in comp],
                )
            )

    return concave


def _concave_patches(
    probes: list[Intersection],
    atoms: np.ndarray,
    sasr: np.ndarray,
    rprobe: float,
):
    pts = np.stack([p.pijk.pt for p in probes])
    kdt = KDTree(pts)

    radii = np.full(len(pts), rprobe)
    contacts = _find_contacts_pairs(kdt, radii, rprobe)

    pcm = _find_toroids(kdt, contacts, radii)
    circles = _concave_circles(probes, pcm, atoms, sasr, rprobe)
    pcircles: list[list[Circle]] = [[] for _ in range(len(pts))]
    for c in circles:
        for i in c.on:
            pcircles[i].append(c)

    bcircles: list[list[Circle]] = [[] for _ in range(len(pts))]
    for c in circles[len(pcm) :]:
        bcircles[c.on[0]].append(c)
    angles, normals, signs = _concave_inside_angles_signs(
        atoms, probes, bcircles, sasr
    )

    pxm = _concave_intersections(kdt, pcircles, rprobe, angles, normals, signs)

    lm_id = itertools.count()
    plm = [
        _concave_arcs(i, kdt, pci, xmi, rprobe, angles, bni, bsi, lm_id)
        for i, (pci, xmi, bni, bsi) in enumerate(
            zip(
                tqdm(pcircles),
                pxm,
                normals,
                signs,
                strict=True,
            )
        )
    ]

    lons = _probe_loops_on_sphere(plm)
    ppm = _concave_spherical_patches(probes, lons)
    return ppm


@dataclass
class SesComponents:
    convex: list[ConvexPatch]
    saddle: list[SaddlePatch]
    concave: list[ConcavePatch]

    rprobe: float


def ses_components(pts: np.ndarray, sas: SasComponents, rprobe: float = 1.4):
    saddle = [
        SaddlePatch.from_arc(seg, pts, sas.sasr, rprobe)
        for seg in tqdm(sas.lm)
    ]
    convex = [
        ConvexPatch.from_spherical_patch(sp, saddle, rprobe)
        for sp in tqdm(sas.pm)
    ]
    concave = _concave_patches(sas.xm, pts, sas.sasr, rprobe)

    return SesComponents(
        convex=convex,
        saddle=saddle,
        concave=concave,
        rprobe=rprobe,
    )


def _read_one(infile: Path, fmt: str):
    mol = next(nuri.readfile(fmt, infile, sanitize=False))
    mol.conceal_hydrogens()
    pts = mol.get_conf()
    radii = np.array([atom.element.vdw_radius for atom in mol])
    return pts, radii


def _arc_positions(arc: Arc, sep: float = 0.1) -> np.ndarray:
    npts = max(2, math.ceil(arc.radian / sep))
    thetas = np.linspace(arc.tbegin, arc.tend, npts, endpoint=False)
    return _points_on_circle_center(arc.parent, thetas[:, None])


def _ses_arc_positions(
    saddle: SaddlePatch,
    ij: int,
    sep: float = 0.1,
):
    npts = max(2, math.ceil(saddle.arc.radian / sep))
    thetas = np.linspace(saddle.arc.tbegin, saddle.arc.tend, npts)
    return _points_on_circle(
        saddle.edge_cntrs[ij],
        saddle.parent.circle.frame,
        saddle.edge_radii[ij],
        thetas[:, None],
    )


def _convex_positions(convex: ConvexPatch, sep: float = 0.1):
    pts = [
        _ses_arc_positions(saddle, int(saddle.parent.ij[1] == convex.sid), sep)
        for loop in convex.loops
        for saddle in loop
    ]
    if not pts:
        return np.empty((0, 3))

    return np.concat(pts)


def _saddle_positions(
    saddle: SaddlePatch,
    rprobe: float,
    sep: float = 0.1,
    singular_only: bool = False,
):
    tor = saddle.parent
    circ = tor.circle
    frame = circ.frame

    z = frame[2]
    tframe = R.from_rotvec((saddle.arc.tbegin - math.pi / 2) * z).apply(frame)
    bframe = R.from_rotvec((saddle.arc.tend - math.pi / 2) * z).apply(frame)
    npts = max(2, math.ceil((tor.phirange[1] - tor.phirange[0]) / sep))
    phis = np.linspace(*tor.phirange, npts)

    if singular_only and saddle.singularity is None:
        return np.empty((0, 3))

    if saddle.singularity is not None:
        singular_phis = _angle_on_circle(
            saddle.singularity,
            circ.cntr + tframe[1] * circ.radius,
            tframe[[2, 1]],
            rprobe,
        )
        phis = phis[
            (phis <= singular_phis.min()) | (phis >= singular_phis.max())
        ]

    top = np.empty((0, 3))
    bottom = np.empty((0, 3))
    if saddle.arc.points is not None:
        top = _points_on_circle(
            circ.cntr + tframe[1] * circ.radius,
            tframe[[2, 1]],
            rprobe,
            phis[:, None],
        )
        bottom = _points_on_circle(
            circ.cntr + bframe[1] * circ.radius,
            bframe[[2, 1]],
            rprobe,
            phis[:, None],
        )

    return np.concat([top, bottom])


def _concave_positions(
    concave: ConcavePatch,
    sep: float = 0.1,
    singular_only: bool = False,
):
    nonsingular = (
        len(concave.loops) == 1
        and len(concave.loops[0]) == 3
        and all(
            len(arc.parent.on) == 1
            and arc.parent.on[0] == concave.parent.pijk.id
            for arc in concave.loops[0]
        )
    )
    if singular_only and nonsingular:
        return np.empty((0, 3))

    arcs = [
        _arc_positions(arc, sep=sep) for loop in concave.loops for arc in loop
    ]
    return np.concat(arcs) if arcs else np.empty((0, 3))


@app.command()
def main(
    inf: Path,
    outf: Path,
    fmt: str = "sdf",
    write_pm: bool = False,
    pm_sep: float = 0.1,
    write_convex: bool = False,
    convex_sep: float = 0.1,
    write_saddle: bool = False,
    saddle_sep: float = 0.1,
    saddle_singular_only: bool = False,
    write_concave: bool = True,
    concave_sep: float = 0.1,
    concave_singular_only: bool = False,
    rprobe: float = 1.4,
):
    pts, radii = _read_one(inf, fmt)

    sas = sas_components(pts, radii + rprobe)

    sasa = sas_area(sas)
    sr_sasa = shrake_rupley_sasa(pts, radii, nprobe=5000)
    print(sasa)
    print(sr_sasa)
    print("SASA comparison:")
    print(f"Gauss-Bonnet SASA: {np.sum(sasa)}")
    print(f"Shrake-Rupley SASA: {np.sum(sr_sasa)}")
    print(f"Difference: {np.sum(sasa) - np.sum(sr_sasa)}")

    ses = ses_components(pts, sas, rprobe=rprobe)

    with open(outf, "w") as f:
        mol = Molecule()

        index = itertools.count(1)
        if write_pm:
            for patch in sas.pm:
                arcs = [
                    _arc_positions(seg.arc, sep=pm_sep)
                    for loop in patch.loops
                    for seg in loop
                ]
                if len(arcs) == 0:
                    continue

                arcs = np.concat(arcs)
                if arcs.size == 0:
                    continue

                i = next(index)
                print(f"{i = }, {patch.sid = }")
                mol.clear()
                with mol.mutator() as m:
                    for _ in arcs:
                        m.add_atom().set_element(1)
                mol.add_conf(arcs)
                f.write(nuri.to_mol2(mol))

        if write_convex:
            for patch in ses.convex:
                arcs = _convex_positions(patch, sep=convex_sep)
                if len(arcs) == 0:
                    continue

                i = next(index)
                print(f"{i = }, {patch.sid = }")

                mol.clear()
                with mol.mutator() as m:
                    for _ in arcs:
                        m.add_atom().set_element(1)
                mol.add_conf(arcs)
                f.write(nuri.to_mol2(mol))

        if write_saddle:
            for patch in ses.saddle:
                arcs = _saddle_positions(
                    patch,
                    ses.rprobe,
                    sep=saddle_sep,
                    singular_only=saddle_singular_only,
                )
                if len(arcs) == 0:
                    continue

                i = next(index)
                print(f"{i = }, {patch.parent.ij = }")

                mol.clear()
                with mol.mutator() as m:
                    for _ in arcs:
                        m.add_atom().set_element(1)
                mol.add_conf(arcs)
                f.write(nuri.to_mol2(mol))

        if write_concave:
            probe_patches: dict[int, list[ConcavePatch]] = defaultdict(list)
            for patch in ses.concave:
                probe_patches[patch.parent.pijk.id].append(patch)

            for pid, patches in probe_patches.items():
                arcs = [
                    _concave_positions(
                        patch,
                        sep=concave_sep,
                        singular_only=concave_singular_only,
                    )
                    for patch in patches
                ]
                if len(arcs) == 0:
                    continue

                arcs = np.concat(arcs)
                if arcs.size == 0:
                    continue

                i = next(index)
                print(f"{i = }, {pid = }")

                mol.clear()
                with mol.mutator() as m:
                    for _ in arcs:
                        m.add_atom().set_element(1)
                mol.add_conf(arcs)
                f.write(nuri.to_mol2(mol))


if __name__ == "__main__":
    app()

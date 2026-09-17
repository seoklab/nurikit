# ruff: noqa
# pyright: reportUnusedImport=false

import itertools
import logging
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import typer
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
class Toroid:
    id: int

    frame: np.ndarray
    cntr: np.ndarray
    radius: float

    on: np.ndarray
    thetas: np.ndarray

    def ij(self, sid: int) -> int:
        return int(self.on[0] != sid)

    def phi(self, pts: np.ndarray) -> np.ndarray:
        return _angle_on_circle(pts, self.cntr, self.frame, self.radius)


@dataclass
class ToroidSegment:
    id: int
    parent: Toroid

    pbegin: float = 0.0
    pend: float = 2 * math.pi
    full: bool = True

    @property
    def radian(self) -> float:
        return self.pend - self.pbegin

    @property
    def test(self) -> np.ndarray:
        phi = 0.5 * (self.pbegin + self.pend)
        return _points_on_circle_center(self.parent, phi)


@dataclass
class Intersection:
    id: int
    pt: np.ndarray

    on: list[int]
    tor: list[Toroid]


@dataclass
class Vertex:
    pt: np.ndarray

    left: ToroidSegment
    left_end: bool

    right: ToroidSegment
    right_begin: bool


@dataclass(kw_only=True)
class Edge(ToroidSegment):
    left: Intersection
    right: Intersection

    def __post_init__(self):
        self.full = False


@dataclass
class Loop:
    arcs: list[ToroidSegment]
    vertices: list[Vertex]

    def __len__(self):
        return len(self.vertices)


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
            it, jt = _angle_on_circle(
                np.stack([pi, pj]),
                pk,
                np.stack([zij, yij]),
            )
            if jt < it:
                yij = -yij
                it = 2 * math.pi - it
                jt = 2 * math.pi - jt

            it = 1.5 * math.pi - it
            jt = jt - 1.5 * math.pi

            toroids[(i, j)] = Toroid(
                id=len(toroids),
                frame=np.stack([np.cross(yij, zij), yij, zij]),
                cntr=tij,
                radius=Rij,
                on=np.array([i, j]),
                thetas=np.array([it, jt]),
            )

    return toroids


def _probe_no_contact(
    kdt: KDTree,
    allowed: set[int],
    probe: np.ndarray,
    sasr: np.ndarray,
    cutoff: float,
    eps: float = 1e-6,
):
    nbrs = kdt.query_ball_point(probe, cutoff)
    nbrs = list(set(nbrs) - allowed)
    npts = kdt.data[nbrs]
    dists = D.cdist(probe[None], npts).squeeze(0)
    return np.all(dists >= sasr[nbrs] - eps)


def _circle_intersections(
    center: np.ndarray,
    radius: float,
    ci: Toroid,
    cj: Toroid,
    eps: float = 1e-6,
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
    eps: float = 1e-6,
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

            for k in ks:
                tik = toroids.get((i, k))
                if tik is None:
                    continue

                inter = _circle_intersections(pi, sri, tij, tik, eps=eps)
                if inter is None:
                    continue

                allowed = {i, j, k}
                for pt in inter:
                    if _probe_no_contact(
                        kdt, allowed, pt, sasr, cutoff, eps=eps
                    ):
                        xm.append(
                            Intersection(
                                id=len(xm),
                                pt=pt,
                                tor=[tij, tik, toroids[(j, k)]],
                                on=[i, j, k],
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
    c: Toroid,
    theta: np.ndarray | float,
) -> np.ndarray:
    return _points_on_circle(c.cntr, c.frame, c.radius, theta)


def _sas_arcs(
    kdt: KDTree,
    toroids: dict[tuple[int, int], Toroid],
    concave: list[Intersection],
    sasr: np.ndarray,
    cutoff: float,
    eps: float = 1e-6,
):
    vertices: dict[tuple[int, int], list[Intersection]] = defaultdict(list)
    for p in concave:
        i, j, k = p.on
        vertices[(i, j)].append(p)
        vertices[(i, k)].append(p)
        vertices[(j, k)].append(p)

    segs: list[ToroidSegment] = []
    for (i, j), tij in tqdm(toroids.items()):
        allowed = {i, j}

        vs = vertices[(i, j)]
        if len(vs) < 2:
            test = tij.cntr + tij.radius * tij.frame[0]
            if _probe_no_contact(kdt, allowed, test, sasr, cutoff, eps=eps):
                segs.append(ToroidSegment(id=len(segs), parent=tij))
            continue

        phis = tij.phi(np.stack([v.pt for v in vs]))
        order = np.argsort(phis)
        order = np.append(order, order[0])

        phis = phis[order]
        phis[-1] += 2 * math.pi
        vs = [vs[i] for i in order]

        mid = 0.5 * (phis[:-1] + phis[1:])
        tests = _points_on_circle_center(tij, mid[:, None])
        for test, ((b, ba), (e, ea)) in zip(
            tests,
            itertools.pairwise(zip(vs, phis)),
            strict=True,
        ):
            if _probe_no_contact(kdt, allowed, test, sasr, cutoff, eps=eps):
                segs.append(
                    Edge(
                        id=len(segs),
                        parent=tij,
                        pbegin=ba,
                        pend=ea,
                        left=b,
                        right=e,
                    )
                )

    return segs


def _merge_arcs(
    sg: nx.MultiGraph,
    edges: list[tuple[int, int, int, Edge]],
    node_equiv: dict[int, int],
    edge_equiv: dict[tuple[int, int, int], int],
) -> list[tuple[int, int, int, Edge]]:
    length = np.array([e.radian for *_, e in edges])
    order = np.argsort(length)

    length_sum: dict[int, float] = defaultdict(float)
    for s, d, _, e in edges:
        length_sum[s] += e.radian
        length_sum[d] += e.radian

    g = sg.copy()
    idx: int
    for idx in order:
        s, d, k, e = edges[idx]
        s = node_equiv.get(s, s)
        d = node_equiv.get(d, d)
        if s == d:
            continue

        k = edge_equiv.get((*sorted([s, d]), k), k)  # type: ignore
        g.remove_edge(s, d, k)

        sel = max(s, d, key=lambda x: length_sum[x])
        nsel = min(s, d, key=lambda x: length_sum[x])
        for old, new in node_equiv.items():
            if new == nsel:
                node_equiv[old] = sel
        node_equiv[nsel] = sel

        for _, nbr, nk, narc in list(g.edges(nsel, keys=True, data="arc")):
            if nbr == sel:
                continue

            k = g.add_edge(sel, nbr, k=nk, arc=narc)
            edge_equiv[(*sorted([sel, nbr]), nk)] = k  # type: ignore

        g.remove_node(nsel)

        if g.number_of_nodes() == g.number_of_edges() and all(
            g.degree[n] == 2  # type: ignore
            for n in g.nodes
        ):
            break

    return [(s, d, k, g[s][d][k]["arc"]) for s, d, k in nx.edge_dfs(g)]


def _traverse_forward(equiv: dict[int, int], arc: Edge, s: int, d: int):
    l = equiv.get(arc.left.id, arc.left.id)
    r = equiv.get(arc.right.id, arc.right.id)
    assert (l, r) == (s, d) or (r, l) == (s, d)

    forward = l == s
    return forward


def _loops_on_sphere(
    sphere_arcs: Sequence[Sequence[ToroidSegment]],
    inter: list[Intersection],
):
    sphere_loops: list[list[Loop]] = [[] for _ in range(len(sphere_arcs))]

    for i, (arcs, loops) in enumerate(zip(tqdm(sphere_arcs), sphere_loops)):
        if not arcs:
            continue

        g = nx.MultiGraph()
        for a in arcs:
            if a.full:
                loops.append(Loop(arcs=[a], vertices=[]))
                continue

            assert isinstance(a, Edge)
            g.add_edge(a.left.id, a.right.id, arc=a)

        n_eq: dict[int, int] = {}
        e_eq: dict[tuple[int, int, int], int] = {}
        for comp in nx.connected_components(g):
            sg: nx.MultiGraph = g.subgraph(comp)  # type: ignore
            loop: list[tuple[int, int, int, Edge]] = [
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
                loop = _merge_arcs(sg, loop, n_eq, e_eq)

            ps, pd, _, pe = loop[-1]
            vertices: list[Vertex] = []
            for cs, cd, _, ce in loop:
                assert pd == cs
                assert inter[cs].id == cs

                vertices.append(
                    Vertex(
                        pt=inter[cs].pt,
                        left=pe,
                        left_end=_traverse_forward(n_eq, pe, ps, pd),
                        right=ce,
                        right_begin=_traverse_forward(n_eq, ce, cs, cd),
                    )
                )
                ps, pd, pe = cs, cd, ce

            loops.append(
                Loop(arcs=[arc for *_, arc in loop], vertices=vertices)
            )

    return sphere_loops


@dataclass
class SphericalPatch:
    sid: int
    center: np.ndarray
    radius: float

    loops: list[Loop] = field(default_factory=list)


def _nearest_point_on_circle(x: np.ndarray, c: Toroid, eps: float = 1e-6):
    y, z = c.frame[1:]
    v = x - c.cntr
    proj = v - np.dot(v, z) * z
    pnorm = np.linalg.norm(proj)
    if pnorm < eps:
        return c.cntr + c.radius * y

    return c.cntr + c.radius * proj / pnorm


def _point_on_arc(
    p: np.ndarray,
    tor: Toroid,
    bounds: tuple[float, float],
    eps: float = 1e-6,
):
    phi = tor.phi(p).item()
    tb, te = bounds
    return (tb - eps <= phi <= te + eps) or (
        tb - eps <= phi + 2 * math.pi <= te + eps
    )


def _point_on_segment(
    p: np.ndarray,
    seg: ToroidSegment,
    eps: float = 1e-5,
) -> bool:
    if seg.full:
        return True

    return _point_on_arc(p, seg.parent, (seg.pbegin, seg.pend), eps=eps)


def _point_inside_loop(x: np.ndarray, loop: Loop, eps: float = 1e-6):
    groups: dict[int, list[ToroidSegment]] = defaultdict(list)
    for arc in loop.arcs:
        groups[arc.parent.id].append(arc)

    toroids = list(groups.values())
    tests = np.stack(
        [_nearest_point_on_circle(x, a0.parent, eps=eps) for a0, *_ in toroids]
    )
    dists = D.cdist(x[None], tests).squeeze(0)
    sel = int(np.argmin(dists))

    xk0 = tests[sel]
    for arc in toroids[sel]:
        if _point_on_segment(xk0, arc, eps=10.0 * eps):
            return True
    return False


def _sphere_loops_to_patch(
    sid: int,
    sri: float,
    pi: np.ndarray,
    ls: list[Loop],
    kdt: KDTree,
    sasr: np.ndarray,
    cutoff: float,
    eps: float = 1e-6,
):
    if not ls:
        nbrs = kdt.query_ball_point(pi, sri + cutoff - eps)
        nbrs = [n for n in nbrs if n != sid]
        npts = kdt.data[nbrs]
        dists = D.cdist(pi[None], npts).squeeze(0)
        if np.all(dists >= sri + sasr[nbrs] - eps):
            yield SphericalPatch(sid=sid, center=pi, radius=sri)
            return

    samples = [loop.arcs[0].test for loop in ls]

    g = nx.DiGraph()
    g.add_nodes_from(range(len(ls)))
    for i, x in enumerate(samples):
        for j, lj in enumerate(ls):
            if i == j:
                continue

            if _point_inside_loop(x, lj, eps=eps):
                g.add_edge(i, j)

    for comp in nx.strongly_connected_components(g):
        yield SphericalPatch(
            sid=sid,
            center=pi,
            radius=sri,
            loops=[ls[n] for n in comp],
        )


def _spherical_patches(
    kdt: KDTree,
    loops: list[list[Loop]],
    sasr: np.ndarray,
    cutoff: float,
    eps: float = 1e-6,
):
    spherical: list[SphericalPatch] = []

    for i, (pi, ls, sri) in enumerate(zip(tqdm(kdt.data), loops, sasr)):
        spherical.extend(
            _sphere_loops_to_patch(i, sri, pi, ls, kdt, sasr, cutoff, eps=eps)
        )

    return spherical


@dataclass
class SasComponents:
    xm: list[Intersection]
    lm: list[ToroidSegment]
    pm: list[SphericalPatch]

    cm: dict[tuple[int, int], Toroid]
    sasr: np.ndarray


def sas_components(pts: np.ndarray, sasr: np.ndarray, eps: float = 1e-6):
    cutoff = np.max(sasr)

    kdt = KDTree(pts)
    contacts = _find_contacts_pairs(kdt, sasr, cutoff, eps=eps)
    cm = _find_toroids(kdt, contacts, sasr)
    xm = _sas_intersections(kdt, contacts, cm, sasr, cutoff, eps=eps)
    lm = _sas_arcs(kdt, cm, xm, sasr, cutoff, eps=eps)

    sphere_arcs: list[list[ToroidSegment]] = [[] for _ in range(len(kdt.data))]
    for a in lm:
        for sid in a.parent.on:
            sphere_arcs[sid].append(a)

    lons = _loops_on_sphere(sphere_arcs, xm)
    pm = _spherical_patches(kdt, lons, sasr, cutoff, eps=eps)

    return SasComponents(
        xm=xm,
        lm=lm,
        pm=pm,
        cm=cm,
        sasr=sasr,
    )


def _vertex_angle(vtx: Vertex, patch: SphericalPatch):
    ci = vtx.left.parent
    cj = vtx.right.parent

    vi = np.cross(ci.frame[2], (vtx.pt - ci.cntr) / ci.radius)
    if not vtx.left_end:
        vi = -vi
    vj = np.cross(cj.frame[2], (vtx.pt - cj.cntr) / cj.radius)
    if not vtx.right_begin:
        vj = -vj

    z = (vtx.pt - patch.center) / patch.radius
    y = np.cross(z, vi)
    theta = abs(math.atan2(np.dot(y, vj), np.dot(vi, vj)))
    return theta


def _loop_angle_total(loop: Loop, patch: SphericalPatch):
    asum = sum(_vertex_angle(vtx, patch) for vtx in loop.vertices)
    return asum


def _gauss_bonnet_area(patch: SphericalPatch):
    chi = 2 - len(patch.loops)

    asum = sum(_loop_angle_total(loop, patch) for loop in patch.loops)

    lsum = 0.0
    for loop in patch.loops:
        for seg in loop.arcs:
            ij = seg.parent.ij(patch.sid)
            lsum += seg.radian * np.sin(-seg.parent.thetas[ij])

    area = patch.radius**2 * (2 * math.pi * chi - (asum + lsum))
    return area, asum, lsum


def sas_area(sas: SasComponents):
    return np.array([_gauss_bonnet_area(p) for p in sas.pm])


@dataclass
class SaddlePatch:
    arc: ToroidSegment

    edge_cntrs: np.ndarray
    edge_radii: np.ndarray
    singular_theta: float = 0.0

    @classmethod
    def from_arc(
        cls,
        seg: ToroidSegment,
        pts: np.ndarray,
        sasr: np.ndarray,
        rprobe: float,
    ):
        tor = seg.parent
        ij = tor.on

        # [2, 1]
        sris = sasr[ij, None]
        ris = sris - rprobe
        scales = ris / sris
        # [2, 3]
        cntr = tor.cntr * scales + pts[ij] * (1 - scales)
        radii = tor.radius * scales.squeeze(-1)

        singular_theta = 0.0
        if tor.radius < rprobe:
            offset = math.sqrt(rprobe**2 - tor.radius**2)
            singular_theta = math.atan2(offset, tor.radius)
            assert singular_theta > 0.0

        return cls(
            arc=seg,
            edge_cntrs=cntr,
            edge_radii=radii,
            singular_theta=singular_theta,
        )


def _loop_from_saddles(
    sas: SphericalPatch,
    rprobe: float,
    loop: Loop,
    saddles: list[SaddlePatch],
):
    segs = {seg.id: seg for seg in loop.arcs}
    for i, seg in segs.items():
        saddle = saddles[seg.id]
        assert saddle.arc.id == seg.id

        ij = saddle.arc.parent.ij(sas.sid)
        center = saddle.edge_cntrs[ij]
        radius = saddle.edge_radii[ij]

        segs[i] = ToroidSegment(
            id=-1,
            parent=Toroid(
                id=-1,
                frame=saddle.arc.parent.frame,
                cntr=center,
                radius=radius,
                on=saddle.arc.parent.on,
                thetas=saddle.arc.parent.thetas,
            ),
            pbegin=saddle.arc.pbegin,
            pend=saddle.arc.pend,
            full=saddle.arc.full,
        )

    scale = rprobe / sas.radius
    vertices = [
        Vertex(
            pt=vtx.pt * (1 - scale) + sas.center * scale,
            left=segs[vtx.left.id],
            left_end=vtx.left_end,
            right=segs[vtx.right.id],
            right_begin=vtx.right_begin,
        )
        for vtx in loop.vertices
    ]

    return Loop(arcs=list(segs.values()), vertices=vertices)


def _ses_convex_from_sas_patch(
    sas: SphericalPatch,
    rprobe: float,
    saddles: list[SaddlePatch],
):
    loops = [
        _loop_from_saddles(sas, rprobe, loop, saddles) for loop in sas.loops
    ]
    ses = SphericalPatch(
        sid=sas.sid,
        center=sas.center,
        radius=sas.radius - rprobe,
        loops=loops,
    )
    return ses


def _concave_circles(
    probes: list[Intersection],
    pcm: dict[tuple[int, int], Toroid],
    atoms: np.ndarray,
    sasr: np.ndarray,
    rprobe: float,
):
    tors = [tij for tij in pcm.values()]

    for i, p in enumerate(probes):
        ijk = np.array(p.on)
        aijk = atoms[ijk]
        p = p.pt

        scale = rprobe / sasr[ijk, None]
        pijk = p * (1 - scale) + aijk * scale
        vijk = (p - aijk) / sasr[ijk, None]

        xs = _normalize(np.roll(pijk, -1, axis=0) - pijk)
        ys = _normalize(vijk + np.roll(vijk, -1, axis=0))
        frames = np.stack([xs, ys, np.cross(xs, ys, axis=-1)], axis=1)
        for fr in frames:
            tors.append(
                Toroid(
                    id=len(tors),
                    frame=fr,
                    cntr=p,
                    radius=rprobe,
                    on=np.array([i]),
                    thetas=np.array([0.0]),
                )
            )

    return tors


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
    bcircles: list[list[Toroid]],
    sasr: np.ndarray,
):
    angles: dict[int, tuple[float, float]] = {}
    normals = []
    signs = []
    for p, (ci, cj, ck) in zip(probes, bcircles, strict=True):
        ijk = p.on
        triangle = (pts[ijk] - p.pt) / sasr[ijk, None]

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


def _concave_intersections(
    kdt: KDTree,
    pcircles: list[list[Toroid]],
    rprobe: float,
    bangles: dict[int, tuple[float, float]],
    bnormals: np.ndarray,
    bsigns: np.ndarray,
    eps: float = 1e-6,
):
    xm: list[list[Intersection]] = [[] for _ in range(len(kdt.data))]
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

                if len(ci.on) == 1 and not _point_on_arc(
                    pt,
                    ci,
                    bangles[ci.id],
                ):
                    continue

                if len(cj.on) == 1 and not _point_on_arc(
                    pt,
                    cj,
                    bangles[cj.id],
                ):
                    continue

                nbrs = kdt.query_ball_point(pt, rprobe - eps)
                if set(nbrs) - allowed:
                    continue

                xmi.append(
                    Intersection(
                        id=next(idgen),
                        pt=pt,
                        on=[*ci.on, *cj.on],
                        tor=[ci, cj],
                    )
                )

    return xm


def _concave_arcs(
    i: int,
    kdt: KDTree,
    circles: list[Toroid],
    inter: list[Intersection],
    rprobe: float,
    bangles: dict[int, tuple[float, float]],
    bnormal: np.ndarray,
    bsign: np.ndarray,
    idgen: itertools.count,
    eps: float = 1e-6,
):
    vertices: dict[int, list[Intersection]] = defaultdict(list)
    for p in inter:
        for c in p.tor:
            vertices[c.id].append(p)

    arcs: list[ToroidSegment] = []
    for tij in circles:
        vs = vertices[tij.id]
        pcntr = kdt.data[i]
        allowed = set(tij.on)

        if len(vs) < 2:
            assert len(tij.on) == 2
            test = tij.cntr + tij.radius * tij.frame[0]

            if not _point_inside_triangle(
                (test - pcntr) / rprobe, bnormal, bsign
            ):
                continue

            nbrs = kdt.query_ball_point(test, rprobe - eps)
            if not set(nbrs) - allowed:
                arcs.append(ToroidSegment(id=next(idgen), parent=tij))

            continue

        angles = tij.phi(np.stack([v.pt for v in vs]))
        order = np.argsort(angles)
        order = np.append(order, order[0])

        angles = angles[order]
        angles[-1] += 2 * math.pi
        vs = [vs[i] for i in order]

        mid = 0.5 * (angles[:-1] + angles[1:])
        tests = _points_on_circle_center(tij, mid[:, None])
        for test, ((b, ba), (e, ea)) in zip(
            tests,
            itertools.pairwise(zip(vs, angles)),
            strict=True,
        ):
            if len(tij.on) == 1 and not _point_on_arc(
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
                    Edge(
                        id=next(idgen),
                        parent=tij,
                        pbegin=ba,
                        pend=ea,
                        left=b,
                        right=e,
                    )
                )

    return arcs


def _concave_patches(
    probes: list[Intersection],
    atoms: np.ndarray,
    sasr: np.ndarray,
    rprobe: float,
    eps: float = 1e-6,
):
    pts = np.stack([p.pt for p in probes])
    kdt = KDTree(pts)

    radii = np.full(len(pts), rprobe)
    contacts = _find_contacts_pairs(kdt, radii, rprobe, eps=eps)

    pcm = _find_toroids(kdt, contacts, radii)
    circles = _concave_circles(probes, pcm, atoms, sasr, rprobe)
    pcircles: list[list[Toroid]] = [[] for _ in range(len(pts))]
    for c in circles:
        for i in c.on:
            pcircles[i].append(c)

    bcircles: list[list[Toroid]] = [[] for _ in range(len(pts))]
    for c in circles[len(pcm) :]:
        bcircles[c.on[0]].append(c)
    angles, normals, signs = _concave_inside_angles_signs(
        atoms, probes, bcircles, sasr
    )

    pxm = _concave_intersections(
        kdt,
        pcircles,
        rprobe,
        angles,
        normals,
        signs,
        eps=eps,
    )

    lm_id = itertools.count()
    plm = [
        _concave_arcs(
            i,
            kdt,
            pci,
            xmi,
            rprobe,
            angles,
            bni,
            bsi,
            lm_id,
            eps=eps,
        )
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

    lons = _loops_on_sphere(plm, [x for xs in pxm for x in xs])
    ppm = _spherical_patches(kdt, lons, radii, rprobe, eps=eps)
    return ppm


@dataclass
class SesComponents:
    convex: list[SphericalPatch]
    saddle: list[SaddlePatch]
    concave: list[SphericalPatch]

    rprobe: float


def ses_components(
    pts: np.ndarray,
    sas: SasComponents,
    rprobe: float = 1.4,
    eps: float = 1e-6,
):
    saddle = [
        SaddlePatch.from_arc(seg, pts, sas.sasr, rprobe)
        for seg in tqdm(sas.lm)
    ]
    convex = [
        _ses_convex_from_sas_patch(sp, rprobe, saddle) for sp in tqdm(sas.pm)
    ]
    concave = _concave_patches(sas.xm, pts, sas.sasr, rprobe, eps=eps)

    return SesComponents(
        convex=convex,
        saddle=saddle,
        concave=concave,
        rprobe=rprobe,
    )


def _saddle_area(saddle: SaddlePatch, rprobe: float):
    tor = saddle.arc.parent

    area = (
        rprobe
        * saddle.arc.radian
        * (
            tor.radius
            * (np.abs(np.sum(tor.thetas)) - 2 * saddle.singular_theta)
            - rprobe
            * (
                np.abs(np.sum(np.sin(tor.thetas)))
                - 2 * math.sin(saddle.singular_theta)
            )
        )
    )
    return area


def ses_area(ses: SesComponents):
    convex = np.array([_gauss_bonnet_area(p) for p in ses.convex])
    saddle = np.array([_saddle_area(p, ses.rprobe) for p in ses.saddle])
    concave = np.array([_gauss_bonnet_area(p) for p in ses.concave])
    return convex, saddle, concave


def _read_one(infile: Path, fmt: str):
    data = np.loadtxt(infile)
    pts = data[:, :3]
    radii = data[:, 3]
    return pts, radii


def _arc_positions(arc: ToroidSegment, sep: float = 0.1) -> np.ndarray:
    npts = max(2, math.ceil(arc.radian / sep))
    phis = np.linspace(arc.pbegin, arc.pend, npts, endpoint=False)
    return _points_on_circle_center(arc.parent, phis[:, None])


def spherical_positions(
    patch: SphericalPatch,
    sep: float = 0.1,
):
    arcs = [
        _arc_positions(seg, sep=sep)
        for loop in patch.loops
        for seg in loop.arcs
    ]
    if len(arcs) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    arcs = np.concat(arcs)
    pts = [v.pt for loop in patch.loops for v in loop.vertices]
    if pts:
        pts = np.stack(pts)
    else:
        pts = np.empty((0, 3))
    return arcs, pts


def saddle_positions(
    saddle: SaddlePatch,
    rprobe: float,
    sep: float = 0.1,
    singular_only: bool = False,
):
    if singular_only and saddle.singular_theta == 0.0:
        return np.empty((0, 3))

    tor = saddle.arc.parent
    frame = tor.frame

    z = frame[2]
    tframe = R.from_rotvec((saddle.arc.pbegin - math.pi / 2) * z).apply(frame)
    bframe = R.from_rotvec((saddle.arc.pend - math.pi / 2) * z).apply(frame)
    npts = max(2, math.ceil(np.abs(np.sum(tor.thetas)) / sep))
    thetas = 1.5 * math.pi + np.linspace(-tor.thetas[0], tor.thetas[1], npts)

    if saddle.singular_theta != 0.0:
        thetas = thetas[
            (thetas <= 1.5 * math.pi - saddle.singular_theta)
            | (thetas >= 1.5 * math.pi + saddle.singular_theta)
        ]

    top = np.empty((0, 3))
    bottom = np.empty((0, 3))
    if not saddle.arc.full:
        top = _points_on_circle(
            tor.cntr + tframe[1] * tor.radius,
            tframe[[2, 1]],
            rprobe,
            thetas[:, None],
        )
        bottom = _points_on_circle(
            tor.cntr + bframe[1] * tor.radius,
            bframe[[2, 1]],
            rprobe,
            thetas[:, None],
        )

    npts = max(2, math.ceil(saddle.arc.radian / sep))
    phis = np.linspace(saddle.arc.pbegin, saddle.arc.pend, npts)[:, None]

    left = _points_on_circle(
        saddle.edge_cntrs[0],
        saddle.arc.parent.frame,
        saddle.edge_radii[0],
        phis,
    )
    right = _points_on_circle(
        saddle.edge_cntrs[1],
        saddle.arc.parent.frame,
        saddle.edge_radii[1],
        phis,
    )

    return np.concat([top, left, bottom, right])


def concave_positions(
    concave: SphericalPatch,
    sep: float = 0.1,
    singular_only: bool = False,
):
    nonsingular = (
        len(concave.loops) == 1
        and len(concave.loops[0]) == 3
        and all(
            len(arc.parent.on) == 1 and arc.parent.on[0] == concave.sid
            for arc in concave.loops[0].arcs
        )
    )
    if singular_only and nonsingular:
        return np.empty((0, 3)), np.empty((0, 3))

    return spherical_positions(concave, sep=sep)


@app.command()
def main(
    inf: Path,
    fmt: str = "sdf",
    load_sas: Path | None = None,
    write_sas: Path | None = None,
    write_pm: Path | None = None,
    pm_sep: float = 0.1,
    write_convex: Path | None = None,
    convex_sep: float = 0.1,
    write_saddle: Path | None = None,
    saddle_sep: float = 0.1,
    saddle_singular_only: bool = False,
    write_concave: Path | None = None,
    concave_sep: float = 0.1,
    concave_singular_only: bool = False,
    rprobe: float = 1.4,
):
    pts, radii = _read_one(inf, fmt)

    if load_sas is not None:
        with open(load_sas, "rb") as f:
            sas: SasComponents = pickle.load(f)
    else:
        sas = sas_components(pts, radii + rprobe)

    if write_sas is not None:
        with open(write_sas, "wb") as f:
            pickle.dump(sas, f)

    sasa = sas_area(sas)

    atom_sasa = np.zeros(len(sas.sasr))
    for p, a in zip(sas.pm, sasa):
        atom_sasa[p.sid] += a[0]

    print(atom_sasa)
    print(f"Gauss-Bonnet SASA: {np.sum(atom_sasa)}")

    ses = ses_components(pts, sas, rprobe=rprobe)
    convex, saddle, concave = ses_area(ses)
    np.savez(
        "area_components.npz",
        pm=sasa,
        convex=convex,
        saddle=saddle,
        concave=concave,
    )

    assert np.all(convex[:, 0] < sasa[:, 0])
    assert np.all(convex[:, 0] >= 0)
    assert np.all(saddle >= 0)
    assert np.all(concave[:, 0] >= 0)

    vsum = np.sum(convex[:, 0])
    ssum = np.sum(saddle)
    csum = np.sum(concave[:, 0])

    print(
        (
            f"SES area: convex = {vsum}, saddle = {ssum}, concave = {csum}, "
            f"total = {vsum + ssum + csum}"
        )
    )


if __name__ == "__main__":
    app()

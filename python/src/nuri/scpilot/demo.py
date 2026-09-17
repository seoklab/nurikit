# ruff: noqa
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportAssignmentType=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nuri
import typer
from nuri.core import Molecule
from scipy.spatial import KDTree
from scipy.spatial import distance as D
from scipy.spatial.transform import Rotation


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm


def _find_contacts_pairs(kdt: KDTree, sasr: np.ndarray, approx_cutoff: float):
    pairs = kdt.query_ball_tree(kdt, approx_cutoff * 2)
    contact_pairs: list[np.ndarray] = []
    for i, (pi, sri, js) in enumerate(zip(kdt.data, sasr, pairs, strict=True)):
        js = np.array(js)
        js = js[js > i]
        if js.size == 0:
            contact_pairs.append(np.empty((2, 0), dtype=int))
            continue

        pjs = kdt.data[js]
        dijs = np.linalg.norm(pjs - pi, axis=-1)
        cmap = (dijs < sri + sasr[js]) & (dijs > np.abs(sri - sasr[js]))
        contact_pairs.append(np.stack([js[cmap], dijs[cmap]]))
    return contact_pairs


def canonical_fibonacci_lattice(npts: int):
    dtheta = 2.0 * math.pi * (3.0 - math.sqrt(5.0)) / 2
    theta = np.arange(npts) * dtheta
    dz = 2.0 / npts
    z = 1.0 - 0.5 * dz + np.arange(npts) * -dz
    r = np.sqrt(1.0 - z * z)

    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return pts


def _npts_for_resolution(radii: np.ndarray, resolution: float) -> np.ndarray:
    """Estimate the number of points needed for a given resolution.

    :param resolution: The desired resolution in points per angstrom squared.
    :returns: The estimated number of points.
    """
    area = 4 * np.pi * radii**2
    npts = np.ceil(area * resolution).astype(int)
    return npts


@dataclass
class ConvexPatch:
    center: np.ndarray

    abs_pos: np.ndarray
    ext_normal: np.ndarray


def _convex_faces(
    kdt: KDTree,
    radii: np.ndarray,
    sasr: np.ndarray,
    atom_npr: np.ndarray,
    all_probes: dict[int, np.ndarray],
    cutoff: float,
):
    convex_faces: dict[int, ConvexPatch] = {}
    for i, (pt, r, sr, npr) in enumerate(zip(kdt.data, radii, sasr, atom_npr)):
        probes = pt + all_probes[npr] * sr
        probe_nbrs: list[list[int]] = kdt.query_ball_point(probes, cutoff)
        exposed = []
        for p, (probe, nbrs) in enumerate(
            zip(probes, probe_nbrs, strict=True)
        ):
            nbrs = [n for n in nbrs if n != i]
            if not nbrs:
                exposed.append(p)
                continue

            dists = D.cdist(probe[None], kdt.data[nbrs]).squeeze(0)
            if np.all(dists > sasr[nbrs]):
                exposed.append(p)

        if exposed:
            normals = all_probes[npr][exposed]
            convex_faces[i] = ConvexPatch(
                center=pt,
                abs_pos=pt + normals * r,
                ext_normal=normals,
            )

    return convex_faces


@dataclass
class SaddlePatch:
    axis: np.ndarray
    cntr: np.ndarray
    radius: float

    abs_pos: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    ext_normal: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


def _solve_phi(v: np.ndarray, R: float, r: float, maxiter: int = 10):
    target = v * R
    phi = v.copy()
    for _ in range(maxiter):
        f = R * phi + r * np.sin(phi) - target
        df = R + r * np.cos(phi)
        dphi = f / df
        phi -= dphi
        if np.all(np.abs(dphi) < 1e-6):
            break
    return phi


def _torus_points(
    R: float,
    r: float,
    resolution: float,
    axis: np.ndarray,
    center: np.ndarray,
    phirange: tuple[float, float] = (0.0, 2 * math.pi),
):
    npts = math.ceil(4 * np.pi**2 * R * r * resolution)

    dtheta = 2.0 * math.pi * (3.0 - math.sqrt(5.0)) / 2
    theta = np.arange(npts) * dtheta

    v = 2 * math.pi * np.mod((np.arange(npts) + 0.5) / npts, 1.0)
    vmin = (R * phirange[0] + r * math.sin(phirange[0])) / R
    vmax = (R * phirange[1] + r * math.sin(phirange[1])) / R

    mask = (vmin <= v) & (v <= vmax)
    theta = theta[mask]
    v = v[mask]
    if v.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    phi = _solve_phi(v, R, r)
    cosu, sinu = np.cos(theta), np.sin(theta)
    cosv, sinv = np.cos(phi), np.sin(phi)

    x = (R + r * cosv) * cosu
    y = (R + r * cosv) * sinu
    z = r * sinv
    ref = np.column_stack([x, y, z])

    px = R * cosu
    py = R * sinu
    pz = np.zeros_like(px)
    probe = np.column_stack([px, py, pz])

    rot, _ = Rotation.align_vectors(axis, [0, 0, 1])
    pts = rot.apply(ref) + center
    probes = rot.apply(probe) + center
    return pts, probes


def _any_perpendicular(v: np.ndarray) -> np.ndarray:
    w = np.array(
        [
            math.copysign(v[2], v[0]),
            math.copysign(v[2], v[1]),
            -math.copysign(v[0], v[2]) - math.copysign(v[1], v[2]),
        ]
    )
    return _normalize(w)


def _saddle_faces(
    tree: KDTree,
    pairs: list[np.ndarray],
    sasr: np.ndarray,
    rprobe: float,
    cutoff: float,
    resolution: float,
):
    saddle_faces: dict[tuple[int, int], SaddlePatch] = {}

    for i, (pi, pair, sri) in enumerate(
        zip(tree.data, pairs, sasr, strict=True)
    ):
        if pair.size == 0:
            continue

        js, dijs = pair
        js = js.astype(int)
        pjs = tree.data[js]
        uijs = (pjs - pi) / dijs[:, None]
        for j, pj, uij, dij in zip(js, pjs, uijs, dijs, strict=True):
            srj = sasr[j]
            tij = 0.5 * ((pi + pj) + (pj - pi) * (sri**2 - srj**2) / dij**2)
            Rij = (
                0.5
                * math.sqrt(
                    ((sri + srj) ** 2 - dij**2) * (dij**2 - (sri - srj) ** 2)
                )
                / dij
            )

            vij = _any_perpendicular(uij)
            pk = tij + Rij * vij

            ik = _normalize(pk - pi)
            iphi = math.pi + math.atan2(np.dot(uij, ik), np.dot(vij, ik))
            jk = _normalize(pk - pj)
            jphi = math.pi + math.atan2(np.dot(uij, jk), np.dot(vij, jk))

            torus_pts, torus_probes = _torus_points(
                R=Rij,
                r=rprobe,
                resolution=resolution,
                axis=uij,
                center=tij,
                phirange=(min(iphi, jphi), max(iphi, jphi)),
            )
            if torus_pts.size == 0:
                saddle_faces[(i, j)] = SaddlePatch(
                    axis=uij,
                    cntr=tij,
                    radius=Rij,
                )
                continue

            probe_nbrs: list[list[int]] = tree.query_ball_point(
                torus_probes, cutoff
            )
            exposed = []
            normals = []
            for probe, nbrs, pt in zip(
                torus_probes,
                probe_nbrs,
                torus_pts,
                strict=True,
            ):
                nbrs = [n for n in nbrs if n != i and n != j]
                if not nbrs:
                    exposed.append(pt)
                    normals.append(probe - pt)
                    continue

                dists = D.cdist(probe[None], tree.data[nbrs]).squeeze(0)
                if np.all(dists > sasr[nbrs]):
                    exposed.append(pt)
                    normals.append(probe - pt)

            saddle_faces[(i, j)] = SaddlePatch(
                axis=uij,
                cntr=tij,
                radius=Rij,
                abs_pos=np.stack(exposed) if exposed else np.empty((0, 3)),
                ext_normal=(
                    _normalize(np.stack(normals))
                    if normals
                    else np.empty((0, 3))
                ),
            )

    return saddle_faces


@dataclass
class ConcavePatch:
    uijk: np.ndarray
    bijk: np.ndarray
    hijk: float
    omega: float

    abs_pos: np.ndarray
    ext_normal: np.ndarray


def _resolve_probe_surface(
    p: np.ndarray,
    pijk: np.ndarray,
    base: np.ndarray,
    solv_pts: np.ndarray,
):
    vijk = _normalize(pijk - p)
    nijk = np.cross(vijk, np.roll(vijk, -1, axis=0), axis=-1)

    refsgn = np.unique(np.sign(np.dot(nijk, _normalize(base - p)))).item()
    sijk = np.sign(np.einsum("nd,md->nm", solv_pts, nijk))

    inside = np.all(sijk == refsgn, axis=-1)
    return solv_pts[inside]


def _test_probe_contact(
    tree: KDTree,
    allowed: set[int],
    probe: np.ndarray,
    sasr: np.ndarray,
    cutoff: float,
    rprobe: float,
    pijk: np.ndarray,
    base: np.ndarray,
    solv_pts: np.ndarray,
):
    nbrs = tree.query_ball_point(probe, cutoff)
    nbrs = [n for n in nbrs if n not in allowed]
    npts = tree.data[nbrs]
    dists = D.cdist(probe[None], npts).squeeze(0)
    if np.any(dists < sasr[nbrs]):
        return np.empty((0, 3)), np.empty((0, 3))

    normals = _resolve_probe_surface(probe, pijk, base, solv_pts)
    pts = probe + rprobe * normals
    return pts, -normals


def _concave_faces(
    tree: KDTree,
    pairs: list[np.ndarray],
    saddle_faces: dict[tuple[int, int], SaddlePatch],
    sasr: np.ndarray,
    solv_pts: np.ndarray,
    rprobe: float,
    cutoff: float,
):
    concave_faces: dict[tuple[int, int, int], list[ConcavePatch]] = (
        defaultdict(list)
    )

    for i, (pi, pair, sri) in enumerate(
        zip(tree.data, pairs, sasr, strict=True)
    ):
        if pair.size == 0:
            continue

        js, dijs = pair
        js = js.astype(int)
        pjs = tree.data[js]
        uijs = (pjs - pi) / dijs[:, None]
        for j, pj, uij in zip(js, pjs, uijs, strict=True):
            ks = np.intersect1d(
                js,
                pairs[j][0].astype(int),
                assume_unique=True,
            )
            if ks.size == 0:
                continue

            torus_ij = saddle_faces[(i, j)]
            tij = torus_ij.cntr

            for k in ks:
                torus_ik = saddle_faces[(i, k)]
                pk = tree.data[k]

                uik = _normalize(pk - pi)

                uijk = np.cross(uij, uik)
                sinw = np.linalg.norm(uijk)
                cosw = np.dot(uij, uik)
                omega = math.atan2(sinw, cosw)

                uijk /= sinw
                utb = np.cross(uijk, uij)
                bijk = tij + utb * (np.dot(uik, torus_ik.cntr - tij) / sinw)
                hsq = sri**2 - np.dot(bijk - pi, bijk - pi)
                if hsq <= 1e-12:
                    continue

                hijk = math.sqrt(hsq)
                pijk = np.stack([pi, pj, pk])
                ijk = {i, j, k}
                ds = np.linalg.norm(pijk[[1, 2, 0]] - pijk[[2, 0, 1]], axis=-1)
                incirc = np.average(pijk, weights=ds, axis=0)

                pts, normals = _test_probe_contact(
                    tree,
                    allowed=ijk,
                    probe=bijk + hijk * uijk,
                    sasr=sasr,
                    cutoff=cutoff,
                    rprobe=rprobe,
                    pijk=pijk,
                    base=incirc,
                    solv_pts=solv_pts,
                )
                if pts.size > 0:
                    concave_faces[(i, j, k)].append(
                        ConcavePatch(
                            uijk=uijk,
                            bijk=bijk,
                            hijk=hijk,
                            omega=omega,
                            abs_pos=pts,
                            ext_normal=normals,
                        )
                    )

                pts, normals = _test_probe_contact(
                    tree,
                    allowed=ijk,
                    probe=bijk - hijk * uijk,
                    sasr=sasr,
                    cutoff=cutoff,
                    rprobe=rprobe,
                    pijk=pijk,
                    base=incirc,
                    solv_pts=solv_pts,
                )
                if pts.size > 0:
                    concave_faces[(i, j, k)].append(
                        ConcavePatch(
                            uijk=uijk,
                            bijk=bijk,
                            hijk=-hijk,
                            omega=omega,
                            abs_pos=pts,
                            ext_normal=normals,
                        )
                    )

    return concave_faces


@dataclass
class Patches:
    index: np.ndarray
    patches: list[ConvexPatch | SaddlePatch | ConcavePatch]

    def __getitem__(self, idx: int):
        return self.patches[self.index[idx]]

    def mask(self, mask: np.ndarray):
        self.index = self.index[mask]


def connolly_ses_gen(
    pts: np.ndarray,
    radii: np.ndarray,
    resolution: float = 5.0,  # points per angstrom squared
    rprobe: float = 1.4,
):
    nprobes = _npts_for_resolution(np.append(radii, rprobe), resolution)
    all_probes: dict[int, np.ndarray] = {
        n: canonical_fibonacci_lattice(n) for n in np.unique(nprobes)
    }
    atom_npr = nprobes[:-1]
    solv_npr = nprobes[-1]

    kdtree = KDTree(pts)
    sasr = radii + rprobe
    cutoff = np.max(sasr)

    convex = _convex_faces(
        kdtree,
        radii,
        sasr,
        atom_npr,
        all_probes,
        cutoff,
    )

    pairs = _find_contacts_pairs(kdtree, sasr, cutoff)
    saddle = _saddle_faces(
        kdtree,
        pairs,
        sasr,
        rprobe,
        cutoff,
        resolution,
    )

    concave = _concave_faces(
        kdtree,
        pairs,
        saddle,
        sasr,
        all_probes[solv_npr],
        rprobe,
        cutoff,
    )

    surface = np.concat(
        [p.abs_pos for p in convex.values()]
        + [p.abs_pos for p in saddle.values()]
        + [p.abs_pos for v in concave.values() for p in v]
    )
    ext_normal = np.concat(
        [p.ext_normal for p in convex.values()]
        + [p.ext_normal for p in saddle.values()]
        + [p.ext_normal for v in concave.values() for p in v]
    )

    parents = list(
        itertools.chain(convex.values(), saddle.values(), *concave.values())
    )
    index = np.array(
        [i for i, p in enumerate(parents) for _ in range(len(p.abs_pos))]
    )
    return surface, ext_normal, Patches(index=index, patches=parents)


def _find_buried_probes(
    probes: KDTree,
    other: KDTree,
    sasr: np.ndarray,
):
    all_nbrs = probes.query_ball_tree(other, np.max(sasr))

    buried = np.zeros(len(probes.data), dtype=np.bool_)
    for i, (probe, nbrs) in enumerate(zip(probes.data, all_nbrs, strict=True)):
        if not nbrs:
            continue

        npts = other.data[nbrs]
        dists = D.cdist(probe[None], npts).squeeze(0)
        if np.any(dists < sasr[nbrs]):
            buried[i] = True

    neighbors = KDTree(probes.data[buried]).query_ball_tree(
        KDTree(probes.data[~buried]),
        1.5,
    )
    for i, nbrs in zip(np.nonzero(buried)[0], neighbors, strict=True):
        if nbrs:
            buried[i] = False

    return buried


def _write_surface(outfile: Path, surface: np.ndarray, probes: np.ndarray):
    mol = Molecule()
    with mol.mutator() as m:
        for _ in surface:
            m.add_atom().set_element(1)

    with open(outfile, "w") as f:
        mol.add_conf(surface)
        f.write(nuri.to_mol2(mol))

        for atom in mol:
            atom.set_element(8)
        mol.set_conf(probes)
        f.write(nuri.to_mol2(mol))


def _calc_sab(
    pa: np.ndarray,
    na: np.ndarray,
    tpb: KDTree,
    nb: np.ndarray,
    parb: Patches,
):
    dists, idxs = tpb.query(pa, k=1)
    nb_up = []
    dists_up = []
    for pi, dij, j in zip(pa, dists, idxs, strict=True):
        patch = parb[j]
        if isinstance(patch, ConvexPatch):
            vij = _normalize(pi - patch.center)
            pj = patch.center + vij * 1.4
            nb_up.append(vij)
            dists_up.append(np.linalg.norm(pi - pj))
        else:
            nb_up.append(nb[j])
            dists_up.append(dij)

    sab = np.einsum("nd,nd->n", na, -np.stack(nb_up)) * np.exp(
        -0.5 * np.stack(dists_up) ** 2
    )
    return sab


def write_mol2(pts: np.ndarray, name: list[str] | None = None):
    mol = Molecule()
    with mol.mutator() as m:
        for _ in pts:
            m.add_atom().set_element(1)

    if name is not None:
        for a, n in zip(mol, name, strict=True):
            a.name = n
    mol.add_conf(pts)
    return nuri.to_mol2(mol)


rdict = {
    6: 1.76,
    7: 1.65,
    8: 1.40,
    9: 1.71,
    15: 2.0425,
    16: 1.85,
    17: 2.07,
    35: 2.22,
    53: 2.36,
}


def _read_one(infile: Path, fmt: str):
    mol = next(nuri.readfile(fmt, infile, sanitize=False))
    mol.conceal_hydrogens()
    pts = mol.get_conf()
    radii = np.array([atom.element.vdw_radius for atom in mol])
    return pts, radii


def _prepare_one(
    pts: np.ndarray,
    radii: np.ndarray,
    resol: float,
    rprobe: float = 1.4,
):
    sasr = radii + rprobe
    surface, ext_normal, parents = connolly_ses_gen(
        pts,
        radii,
        resolution=resol,
        rprobe=rprobe,
    )
    probes = surface + ext_normal * rprobe
    return sasr, surface, ext_normal, probes, parents


def main(
    in1: Path,
    in2: Path,
    out1: Path,
    out2: Path,
    fmt: str = "sdf",
    resol: float = 5.0,
):
    a, r1 = _read_one(in1, fmt)
    b, r2 = _read_one(in2, fmt)

    ta = KDTree(a)
    tb = KDTree(b)

    anear = ta.query_ball_tree(tb, 8.0)
    amask = np.array([bool(nbrs) for nbrs in anear])
    print(f"Molecule 1: {np.sum(amask)}/{len(a)} atoms near molecule 2")
    a = a[amask]
    r1 = r1[amask]

    bnear = tb.query_ball_tree(ta, 8.0)
    bmask = np.array([bool(nbrs) for nbrs in bnear])
    print(f"Molecule 2: {np.sum(bmask)}/{len(b)} atoms near molecule 1")
    b = b[bmask]
    r2 = r2[bmask]

    sr1, s1, n1, p1, par1 = _prepare_one(a, r1, resol)
    print(f"Molecule 1: {len(a)} atoms, {len(s1)} surface points")

    sr2, s2, n2, p2, par2 = _prepare_one(b, r2, resol)
    print(f"Molecule 2: {len(b)} atoms, {len(s2)} surface points")

    tpa = KDTree(p1)
    tb = KDTree(b)
    pa_mask = _find_buried_probes(tpa, tb, sr2)
    tpb = KDTree(p2)
    ta = KDTree(a)
    pb_mask = _find_buried_probes(tpb, ta, sr1)

    s1 = s1[pa_mask]
    n1 = n1[pa_mask]
    par1.mask(pa_mask)
    s2 = s2[pb_mask]
    n2 = n2[pb_mask]
    par2.mask(pb_mask)

    _write_surface(out1, s1, p1[pa_mask])
    _write_surface(out2, s2, p2[pb_mask])

    tsb = KDTree(s2)
    sab = _calc_sab(s1, n1, tsb, n2, par2)
    tsa = KDTree(s1)
    sba = _calc_sab(s2, n2, tsa, n1, par1)

    np.save("sab.npy", sab)
    np.save("sba.npy", sba)

    sc = (np.median(sab) + np.median(sba)) / 2
    print(f"Shape complementarity score: {sc:.4f}")


if __name__ == "__main__":
    typer.run(main)

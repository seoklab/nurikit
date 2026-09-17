# ruff: noqa
"""SES blue-noise / quasi-uniform point sampling (optionally with normals).

This module builds on the analytical SES component data structures from
`data.py` (SphericalPatch / SaddlePatch / SesComponents) and generates a set of
sample points approximately uniformly distributed w.r.t. surface area.

Design goals
------------
1) *Area-proportional* sampling: expected points per patch ~= density * area.
2) *Blue-noise-ish* spacing: achieved via Poisson-disk thinning in 3D with a
   min-distance chosen to match the requested density.
3) Optional *outward normals* from the excluded volume (i.e., pointing toward
   solvent).

Notes
-----
* The Poisson thinning uses Euclidean distance in R^3. For sufficiently dense
  sampling (spacing small relative to local curvature radii), this is a good
  approximation to geodesic distance on the SES.
* For spherical patches, membership is determined using loop boundaries via the
  same geometric primitives as `data.py` (Toroid/ToroidSegment). We use an
  even-odd (parity) rule, with the "inside parity" inferred by matching the
  patch's analytical Gauss–Bonnet area.

Example
-------
>>> import numpy as np
>>> from data import sas_components, ses_components
>>> from ses_sample_points import sample_ses_points
>>> pts = ...   # [N,3] atom centers (angstrom)
>>> vdw = ...   # [N] van der Waals radii (angstrom)
>>> rprobe = 1.4
>>> sas = sas_components(pts, vdw + rprobe)
>>> ses = ses_components(pts, sas, rprobe=rprobe)
>>> X, N, meta = sample_ses_points(ses, density=0.5, seed=0,
...                               return_normals=True, return_meta=True)
>>> X.shape, N.shape
((M, 3), (M, 3))
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import typer

import anal as sesdata

app = typer.Typer(pretty_exceptions_enable=False)

# ------------------------------
# Small helpers
# ------------------------------


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return _normalize(v)


def _phi_on_circle(points: np.ndarray, tor: sesdata.Toroid) -> np.ndarray:
    """Vectorized equivalent of data._angle_on_circle for points on tor."""
    vs = points - tor.cntr
    vs = vs / tor.radius
    x, y = tor.frame[:2]
    theta = np.arctan2(vs @ y, vs @ x)
    return np.where(theta >= 0, theta, 2 * math.pi + theta)


def _phi_in_bounds(
    phi: np.ndarray, pbegin: float, pend: float, eps: float = 1e-8
) -> np.ndarray:
    """Return mask whether phi is inside [pbegin, pend] on an unwrapped circle."""
    return ((pbegin - eps) <= phi) & (phi <= (pend + eps)) | (
        (pbegin - eps) <= (phi + 2 * math.pi)
    ) & ((phi + 2 * math.pi) <= (pend + eps))


@dataclass(frozen=True)
class _CircleGroup:
    """All boundary segments from a single parent Toroid."""

    tor: sesdata.Toroid
    segments: Tuple[sesdata.ToroidSegment, ...]


def _group_loop_arcs(loop: sesdata.Loop) -> List[_CircleGroup]:
    groups: Dict[int, List[sesdata.ToroidSegment]] = {}
    for seg in loop.arcs:
        groups.setdefault(seg.parent.id, []).append(seg)
    out: List[_CircleGroup] = []
    for segs in groups.values():
        out.append(_CircleGroup(tor=segs[0].parent, segments=tuple(segs)))
    return out


def _nearest_points_on_circle(
    points: np.ndarray, tor: sesdata.Toroid, eps: float = 1e-10
) -> np.ndarray:
    """Nearest points on a circle (in 3D) for each query point.

    Matches the logic in data._nearest_point_on_circle but vectorized.
    """
    y, z = tor.frame[1], tor.frame[2]
    v = points - tor.cntr
    proj = v - (v @ z)[:, None] * z[None]
    pnorm = np.linalg.norm(proj, axis=-1)

    out = np.empty_like(points)
    good = pnorm >= eps
    out[good] = tor.cntr + tor.radius * proj[good] / pnorm[good, None]
    out[~good] = tor.cntr + tor.radius * y
    return out


def _inside_loop(
    points: np.ndarray, loop: sesdata.Loop, eps: float = 1e-6
) -> np.ndarray:
    """Vectorized point-in-loop test on a sphere surface.

    This mirrors data._point_inside_loop.

    Returns
    -------
    mask: (N,) bool
        True means the point is considered inside the loop.
    """
    if len(loop.arcs) == 0:
        return np.zeros((len(points),), dtype=bool)

    groups = _group_loop_arcs(loop)

    nearest_pts = []
    dists = []
    onseg = []
    for g in groups:
        p = _nearest_points_on_circle(points, g.tor, eps=eps)
        nearest_pts.append(p)
        dists.append(np.linalg.norm(points - p, axis=-1))

        if any(seg.full for seg in g.segments):
            onseg.append(np.ones((len(points),), dtype=bool))
            continue

        phi = _phi_on_circle(p, g.tor)
        ok = np.zeros((len(points),), dtype=bool)
        for seg in g.segments:
            ok |= _phi_in_bounds(phi, seg.pbegin, seg.pend, eps=10.0 * eps)
        onseg.append(ok)

    dists = np.stack(dists, axis=1)  # [N, G]
    onseg = np.stack(onseg, axis=1)  # [N, G]

    sel = np.argmin(dists, axis=1)  # [N]
    return onseg[np.arange(len(points)), sel]


def _loop_parity(
    points: np.ndarray, loops: Sequence[sesdata.Loop]
) -> np.ndarray:
    if not loops:
        return np.zeros((len(points),), dtype=np.int8)

    acc = np.zeros((len(points),), dtype=np.int8)
    for loop in loops:
        acc ^= _inside_loop(points, loop).astype(np.int8)  # parity via XOR
    return acc


def _infer_patch_parity_ref(
    patch: sesdata.SphericalPatch,
    n_mc: int,
    rng: np.random.Generator,
) -> int:
    """Infer which parity (0/1) corresponds to the patch interior.

    We estimate f1 = area fraction where parity==1 using Monte Carlo on the
    full sphere and select parity_ref such that it matches the analytical patch
    area fraction best.
    """
    area = float(sesdata._gauss_bonnet_area(patch)[0])
    area = abs(area)
    sphere_area = 4.0 * math.pi * patch.radius * patch.radius
    target = 0.0 if sphere_area <= 0 else area / sphere_area
    target = min(max(target, 0.0), 1.0)

    dirs = _random_unit_vectors(n_mc, rng)
    pts = patch.center[None] + patch.radius * dirs
    parity = _loop_parity(pts, patch.loops)
    f1 = float(np.mean(parity))

    if abs(f1 - target) <= abs((1.0 - f1) - target):
        return 1
    return 0


# ------------------------------
# Candidate generation
# ------------------------------


def _sample_spherical_patch_candidates(
    patch: sesdata.SphericalPatch,
    n_candidates: int,
    rng: np.random.Generator,
    mc_samples: int = 512,
    max_tries_factor: int = 200,
) -> np.ndarray:
    """Uniform-by-area candidates on a spherical patch via rejection."""
    if n_candidates <= 0:
        return np.empty((0, 3), dtype=float)

    if not patch.loops:
        dirs = _random_unit_vectors(n_candidates, rng)
        return patch.center[None] + patch.radius * dirs

    parity_ref = _infer_patch_parity_ref(patch, n_mc=mc_samples, rng=rng)

    out: List[np.ndarray] = []
    remaining = n_candidates
    batch = max(512, min(8192, n_candidates * 4))

    tries = 0
    max_tries = max_tries_factor * max(1, n_candidates)
    while remaining > 0 and tries < max_tries:
        tries += batch
        dirs = _random_unit_vectors(batch, rng)
        pts = patch.center[None] + patch.radius * dirs
        parity = _loop_parity(pts, patch.loops)
        keep = pts[parity == parity_ref]
        if keep.size == 0:
            continue

        if len(keep) > remaining:
            keep = keep[:remaining]
        out.append(keep)
        remaining -= len(keep)

    return (
        np.concatenate(out, axis=0) if out else np.empty((0, 3), dtype=float)
    )


def _sample_spherical_patch_candidates_with_normals(
    patch: sesdata.SphericalPatch,
    n_candidates: int,
    rng: np.random.Generator,
    *,
    outward_sign: float,
    mc_samples: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = _sample_spherical_patch_candidates(
        patch,
        n_candidates=n_candidates,
        rng=rng,
        mc_samples=mc_samples,
    )
    if len(pts) == 0:
        return pts, np.empty((0, 3), dtype=float)

    nrm = outward_sign * _normalize(pts - patch.center[None])
    return pts, nrm


def _sample_saddle_candidates_with_normals(
    saddle: sesdata.SaddlePatch,
    rprobe: float,
    n_candidates: int,
    rng: np.random.Generator,
    max_tries_factor: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Area-aware candidates on a saddle (toroidal) patch with normals.

    Parameterization follows `data.saddle_positions`:
      X(phi,theta) = tor.cntr + (R + r*sin(theta))*u(phi) + r*cos(theta)*z
    where u(phi) lies in the torus plane and z is the torus axis.

    The (tube) center at fixed phi is:
      T(phi) = tor.cntr + R*u(phi)

    The vector from the tube center to surface point is:
      V = X - T = r*(sin(theta)*u(phi) + cos(theta)*z)

    For SES, the solvent probe centers lie on the tube-center circle; outward
    normals from the excluded volume therefore point *toward* the tube center,
    i.e., along -V.
    """
    if n_candidates <= 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    tor = saddle.arc.parent
    frame = tor.frame
    z = frame[2]
    Rmaj = float(tor.radius)
    rmin = float(rprobe)

    pbegin = float(saddle.arc.pbegin)
    pend = float(saddle.arc.pend)
    if pend <= pbegin:
        pend = pbegin + 2 * math.pi

    theta_lo = 1.5 * math.pi - float(tor.thetas[0])
    theta_hi = 1.5 * math.pi + float(tor.thetas[1])
    if theta_hi <= theta_lo:
        theta_hi = theta_lo + abs(float(np.sum(tor.thetas)))

    intervals: List[Tuple[float, float]] = [(theta_lo, theta_hi)]
    if saddle.singular_theta > 0.0:
        gap = float(saddle.singular_theta)
        mid = 1.5 * math.pi
        intervals = [(theta_lo, mid - gap), (mid + gap, theta_hi)]
        intervals = [(a, b) for a, b in intervals if b > a]

    lengths = np.array([b - a for a, b in intervals], dtype=float)
    if np.any(lengths <= 0) or len(lengths) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    cum = np.cumsum(lengths)
    total_len = float(cum[-1])

    # Jacobian weight: w(theta) = |R + r*sin(theta)| (up to constant r)
    w_max = abs(Rmaj) + abs(rmin)

    pts_out = np.empty((n_candidates, 3), dtype=float)
    nrm_out = np.empty((n_candidates, 3), dtype=float)
    filled = 0
    tries = 0
    max_tries = max_tries_factor * max(1, n_candidates)
    batch = max(512, min(8192, n_candidates * 4))

    while filled < n_candidates and tries < max_tries:
        tries += batch

        phi = rng.uniform(pbegin, pend, size=batch)

        # Theta uniform on union of intervals.
        u = rng.uniform(0.0, total_len, size=batch)
        sel = np.searchsorted(cum, u, side="right")
        theta = np.empty((batch,), dtype=float)
        for k, (a, b) in enumerate(intervals):
            m = sel == k
            if not np.any(m):
                continue
            u0 = u[m] - (cum[k - 1] if k > 0 else 0.0)
            theta[m] = a + u0

        w = np.abs(Rmaj + rmin * np.sin(theta))
        keep = rng.uniform(0.0, w_max, size=batch) < w
        if not np.any(keep):
            continue

        phi = phi[keep]
        theta = theta[keep]

        uvec = (
            frame[0][None] * np.cos(phi)[:, None]
            + frame[1][None] * np.sin(phi)[:, None]
        )

        pts = (
            tor.cntr[None]
            + (Rmaj + rmin * np.sin(theta))[:, None] * uvec
            + (rmin * np.cos(theta))[:, None] * z[None]
        )

        # Outward-from-excluded-volume normals: toward tube center
        nrm = -(
            np.sin(theta)[:, None] * uvec + np.cos(theta)[:, None] * z[None]
        )
        nrm = _normalize(nrm)

        take = min(len(pts), n_candidates - filled)
        pts_out[filled : filled + take] = pts[:take]
        nrm_out[filled : filled + take] = nrm[:take]
        filled += take

    return pts_out[:filled], nrm_out[:filled]


# ------------------------------
# Poisson-disk thinning in 3D
# ------------------------------


def _poisson_thin_indices(
    points: np.ndarray, order: np.ndarray, r: float
) -> np.ndarray:
    """Return indices of points kept by greedy Poisson-disk thinning."""
    if len(points) == 0:
        return np.empty((0,), dtype=int)
    if r <= 0:
        return np.array(order, dtype=int)

    cell = r / math.sqrt(3.0)
    inv = 1.0 / cell

    grid: Dict[Tuple[int, int, int], List[int]] = {}
    chosen_pts: List[np.ndarray] = []
    chosen_idx: List[int] = []

    def cell_key(p: np.ndarray) -> Tuple[int, int, int]:
        q = np.floor(p * inv).astype(int)
        return int(q[0]), int(q[1]), int(q[2])

    for idx in order:
        p = points[idx]
        key = cell_key(p)

        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = grid.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    if not bucket:
                        continue
                    for j in bucket:
                        if np.linalg.norm(p - chosen_pts[j]) < r:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break

        if not ok:
            continue

        j = len(chosen_pts)
        chosen_pts.append(p)
        chosen_idx.append(int(idx))
        grid.setdefault(key, []).append(j)

    return np.array(chosen_idx, dtype=int)


def _estimate_hex_spacing(density: float) -> float:
    """Approximate nearest-neighbor spacing for a 2D blue-noise set."""
    if density <= 0:
        return float("inf")
    # Hex packing: area per point = sqrt(3)/2 * s^2
    return math.sqrt(2.0 / (math.sqrt(3.0) * density))


def _choose_min_dist(
    candidates: np.ndarray,
    target_n: int,
    density: float,
    seed: int,
    max_iter: int = 24,
) -> Tuple[float, np.ndarray]:
    """Binary search a Poisson radius giving ~target_n points.

    Returns
    -------
    (r, idx)
        r is the min distance used, idx are indices into `candidates`.
    """
    if target_n <= 0 or len(candidates) == 0:
        return 0.0, np.empty((0,), dtype=int)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))

    s = _estimate_hex_spacing(density)
    r0 = 0.85 * s
    r_lo = 0.0
    r_hi = max(1e-8, 2.5 * r0)

    idx_hi = _poisson_thin_indices(candidates, order, r_hi)
    while len(idx_hi) > target_n and r_hi < 1e6:
        r_hi *= 1.5
        idx_hi = _poisson_thin_indices(candidates, order, r_hi)

    if len(idx_hi) > target_n:
        return r_hi, idx_hi[:target_n]

    best_r = r_hi
    best_idx = idx_hi
    best_err = abs(len(best_idx) - target_n)

    for _ in range(max_iter):
        r_mid = 0.5 * (r_lo + r_hi)
        idx_mid = _poisson_thin_indices(candidates, order, r_mid)
        err = abs(len(idx_mid) - target_n)
        if err < best_err:
            best_err, best_r, best_idx = err, r_mid, idx_mid

        # Monotonic: larger r => fewer points.
        if len(idx_mid) > target_n:
            r_lo = r_mid
        else:
            r_hi = r_mid

    if len(best_idx) > target_n:
        best_idx = best_idx[:target_n]
    return best_r, best_idx


# ------------------------------
# Public API
# ------------------------------


@dataclass
class SamplingMeta:
    density: float
    requested_points: int
    returned_points: int
    min_dist: float
    total_area: float
    candidate_points: int
    convex_area: float
    saddle_area: float
    concave_area: float


def sample_ses_points(
    ses: sesdata.SesComponents,
    density: float,
    *,
    oversample: float = 6.0,
    seed: int = 42,
    mc_samples: int = 512,
):
    """Generate quasi-uniform SES sample points.

    Parameters
    ----------
    ses:
        Analytical SES components from `data.ses_components`.
    density:
        Desired sampling density in points per Å^2.
    oversample:
        Candidate oversampling factor before Poisson thinning.
    seed:
        RNG seed.
    mc_samples:
        Monte Carlo samples used per spherical patch to infer loop parity.
    """
    rng = np.random.default_rng(seed)

    if density <= 0:
        x = np.empty((0, 3), dtype=float)
        n = np.empty((0, 3), dtype=float)
        meta = SamplingMeta(
            density=density,
            requested_points=0,
            returned_points=0,
            min_dist=0.0,
            total_area=0.0,
            candidate_points=0,
            convex_area=0.0,
            saddle_area=0.0,
            concave_area=0.0,
        )
        return x, n, meta

    # Areas
    convex_area = np.array(
        [sesdata._gauss_bonnet_area(p)[0] for p in ses.convex]
    )
    saddle_area = np.array(
        [sesdata._saddle_area(p, ses.rprobe) for p in ses.saddle]
    )
    concave_area = np.array(
        [sesdata._gauss_bonnet_area(p)[0] for p in ses.concave]
    )
    total_area = (
        np.sum(convex_area) + np.sum(saddle_area) + np.sum(concave_area)
    )

    # Expected total points.
    expected_total = total_area * density
    requested_points = math.ceil(expected_total)

    # Candidate generation per patch.
    candidates_pts: List[np.ndarray] = []
    candidates_nrm: List[np.ndarray] = []

    def area_ncand(area: float):
        return math.ceil(area * density * oversample)

    for p, a in zip(ses.convex, convex_area):
        n_cand = area_ncand(a)

        pts, nrm = _sample_spherical_patch_candidates_with_normals(
            p,
            n_candidates=n_cand,
            rng=rng,
            outward_sign=+1.0,
            mc_samples=mc_samples,
        )
        if len(pts):
            candidates_pts.append(pts)
            candidates_nrm.append(nrm)

    # Saddle torus candidates (outward points toward tube center)
    for s in ses.saddle:
        area = float(sesdata._saddle_area(s, ses.rprobe))
        n_cand = area_ncand(area)
        if not n_cand:
            continue
        pts, nrm = _sample_saddle_candidates_with_normals(
            s,
            rprobe=ses.rprobe,
            n_candidates=n_cand,
            rng=rng,
        )
        if len(pts):
            candidates_pts.append(pts)
            candidates_nrm.append(nrm)

    # Concave spherical candidates (outward points toward probe center)
    for p in ses.concave:
        area = abs(float(sesdata._gauss_bonnet_area(p)[0]))
        n_cand = area_ncand(area)
        if not n_cand:
            continue
        pts, nrm = _sample_spherical_patch_candidates_with_normals(
            p,
            n_candidates=n_cand,
            rng=rng,
            outward_sign=-1.0,
            mc_samples=mc_samples,
        )
        if len(pts):
            candidates_pts.append(pts)
            candidates_nrm.append(nrm)

    if candidates_pts:
        cand_pts = np.concatenate(candidates_pts, axis=0)
        cand_nrm = np.concatenate(candidates_nrm, axis=0)
    else:
        cand_pts = np.empty((0, 3), dtype=float)
        cand_nrm = np.empty((0, 3), dtype=float)

    if len(cand_pts) == 0 or requested_points == 0:
        x = np.empty((0, 3), dtype=float)
        n = np.empty((0, 3), dtype=float)
        meta = SamplingMeta(
            density=float(density),
            requested_points=int(requested_points),
            returned_points=0,
            min_dist=0.0,
            total_area=float(total_area),
            candidate_points=int(len(cand_pts)),
            convex_area=float(convex_area),
            saddle_area=float(saddle_area),
            concave_area=float(concave_area),
        )
        return x, n, meta

    min_dist, keep_idx = _choose_min_dist(
        cand_pts,
        target_n=requested_points,
        density=density,
        seed=seed,
    )

    x = cand_pts[keep_idx]
    n = cand_nrm[keep_idx]

    meta = SamplingMeta(
        density=float(density),
        requested_points=int(requested_points),
        returned_points=int(len(x)),
        min_dist=float(min_dist),
        total_area=float(total_area),
        candidate_points=int(len(cand_pts)),
        convex_area=float(convex_area),
        saddle_area=float(saddle_area),
        concave_area=float(concave_area),
    )

    return x, n, meta


# ------------------------------
# Optional CLI
# ------------------------------


def _write_xyz(path: Path, pts: np.ndarray):
    np.savetxt(path, pts, fmt="%.8f")


def _write_xyzn(path: Path, pts: np.ndarray, nrm: np.ndarray):
    out = np.concatenate([pts, nrm], axis=1)
    np.savetxt(path, out, fmt="%.8f")


@app.command()
def main(
    inf: Path,
    fmt: str = "sdf",
    load_ses: Path | None = None,
    write_ses: Path | None = None,
    output: Path | None = None,
    rprobe: float = 1.4,
    density: float = 1,
    oversample: float = 6.0,
    # normals: bool = False,
):
    atoms, vdw = sesdata._read_one(inf, fmt=fmt)

    if load_ses is not None:
        with open(load_ses, "rb") as f:
            ses: sesdata.SesComponents = pickle.load(f)
    else:
        sas = sesdata.sas_components(atoms, vdw + rprobe)
        ses = sesdata.ses_components(atoms, sas, rprobe=rprobe)

    if write_ses is not None:
        with open(write_ses, "wb") as f:
            pickle.dump(ses, f)

    pts, nrm, meta = sample_ses_points(
        ses,
        density=density,
        oversample=oversample,
    )
    print(
        f"Sampled {meta.returned_points} points (requested {meta.requested_points})"
    )

    if output is not None:
        sesdata.debug_write_pts(output, pts)


if __name__ == "__main__":
    app()

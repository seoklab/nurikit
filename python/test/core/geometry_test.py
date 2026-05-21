#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from typing import Tuple

import numpy as np
import pytest

from nuri.core import geometry as ngeom


@pytest.fixture(scope="module")
def points():
    gen = np.random.default_rng(42)

    query = gen.random((10, 3)) * 10
    templ = gen.random((10, 3)) * 10

    return query, templ


@pytest.mark.parametrize(
    "method",
    [
        "qcp",
        "kabsch",
        pytest.param(
            "unknown",
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
    ],
)
def test_align(points: Tuple[np.ndarray, np.ndarray], method: str):
    q, t = points

    xform, rmsd = ngeom.align_points(q, t, method=method, reflection=False)

    p = ngeom.transform(xform, q)
    rmsd_calc = np.sqrt(np.mean(np.sum((p - t) ** 2, axis=1)))
    assert np.allclose(rmsd, rmsd_calc, atol=1e-6)

    rmsd2 = ngeom.align_rmsd(q, t, method=method, reflection=False)
    assert np.allclose(rmsd, rmsd2, atol=1e-6)

    xform, rmsd = ngeom.align_points(q, t, method=method, reflection=True)

    p = ngeom.transform(xform, q)
    rmsd_calc = np.sqrt(np.mean(np.sum((p - t) ** 2, axis=1)))
    assert np.allclose(rmsd, rmsd_calc, atol=1e-6)

    rmsd2 = ngeom.align_rmsd(q, t, method=method, reflection=True)
    assert np.allclose(rmsd, rmsd2, atol=1e-6)


@pytest.mark.parametrize(
    "method",
    ["qcp", "kabsch"],
)
def test_align_nonfinite(method: str):
    q = np.ones((10, 3))
    q[0, 0] = np.nan

    t = np.ones((10, 3))

    with pytest.raises(ValueError, match="NaN or infinite values"):
        ngeom.align_points(q, t, method=method, reflection=False)

    with pytest.raises(ValueError, match="NaN or infinite values"):
        ngeom.align_points(q, t, method=method, reflection=True)

    with pytest.raises(ValueError, match="NaN or infinite values"):
        ngeom.align_rmsd(q, t, method=method, reflection=False)

    with pytest.raises(ValueError, match="NaN or infinite values"):
        ngeom.align_rmsd(q, t, method=method, reflection=True)


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(42)
    pts = rng.random((200, 3)) * 10
    query = rng.random((30, 3)) * 10
    return pts, query


def _brute_pairs(pts: np.ndarray, query: np.ndarray, d: float):
    diff = query[:, None, :] - pts[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    return dmat, dmat <= d


class TestOctree:
    def test_construct_and_rebuild(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, _ = cloud
        tree = ngeom.Octree(pts)

        new_pts = pts[:50]
        tree.rebuild(new_pts, bucket_size=8)

        qry = pts[:5]
        idxs, dist = tree.find_neighbors(qry, k=1)
        assert idxs.shape == (5, 2)
        assert dist.shape == (5,)
        assert np.array_equal(idxs[:, 1], np.arange(5))
        assert np.allclose(dist, 0.0)

    def test_find_neighbors_requires_d_or_k(
        self, cloud: Tuple[np.ndarray, np.ndarray]
    ):
        pts, qry = cloud
        tree = ngeom.Octree(pts)

        with pytest.raises(ValueError, match="cutoff distance or number"):
            tree.find_neighbors(qry)

    def test_find_neighbors_by_distance(
        self, cloud: Tuple[np.ndarray, np.ndarray]
    ):
        pts, qry = cloud
        tree = ngeom.Octree(pts)
        d = 1.5

        idxs, dist = tree.find_neighbors(qry, d=d)
        assert idxs.shape[1] == 2
        assert idxs.shape[0] == dist.shape[0]
        assert (dist <= d + 1e-9).all()

        _, mask = _brute_pairs(pts, qry, d)
        assert idxs.shape[0] == int(mask.sum())

        for q_idx in range(qry.shape[0]):
            got = np.sort(idxs[idxs[:, 0] == q_idx, 1])
            expect = np.sort(np.flatnonzero(mask[q_idx]))
            assert np.array_equal(got, expect), f"query {q_idx}"

        rebuilt = np.linalg.norm(pts[idxs[:, 1]] - qry[idxs[:, 0]], axis=-1)
        assert np.allclose(dist, rebuilt, atol=1e-9)

    def test_find_neighbors_by_count(
        self, cloud: Tuple[np.ndarray, np.ndarray]
    ):
        pts, qry = cloud
        tree = ngeom.Octree(pts)
        k = 5

        idxs, dist = tree.find_neighbors(qry, k=k)
        assert idxs.shape == (qry.shape[0] * k, 2)
        assert dist.shape == (qry.shape[0] * k,)

        dmat = np.linalg.norm(qry[:, None, :] - pts[None, :, :], axis=-1)
        expect_idx = np.argsort(dmat, axis=1)[:, :k]
        expect_dist = np.take_along_axis(dmat, expect_idx, axis=1)

        for q_idx in range(qry.shape[0]):
            sel = idxs[:, 0] == q_idx
            nbrs = idxs[sel, 1]
            dists = dist[sel]
            assert nbrs.shape == (k,)

            assert np.array_equal(nbrs, expect_idx[q_idx]), f"query {q_idx}"
            assert np.allclose(dists, expect_dist[q_idx], atol=1e-9)
            assert np.all(np.diff(dists) >= -1e-12)

    def test_find_neighbors_by_count_and_distance(
        self, cloud: Tuple[np.ndarray, np.ndarray]
    ):
        pts, qry = cloud
        tree = ngeom.Octree(pts)
        k = 10
        d = 1.5

        idxs, dist = tree.find_neighbors(qry, d=d, k=k)
        assert idxs.shape[1] == 2
        assert idxs.shape[0] == dist.shape[0]
        assert (dist <= d + 1e-9).all()

        dmat, mask = _brute_pairs(pts, qry, d)
        for q_idx in range(qry.shape[0]):
            sel = idxs[:, 0] == q_idx
            nbrs = idxs[sel, 1]
            dists = dist[sel]

            within = np.flatnonzero(mask[q_idx])
            order = within[np.argsort(dmat[q_idx, within])]
            expect = order[:k]

            assert nbrs.shape[0] == expect.shape[0], f"query {q_idx}"
            assert np.array_equal(nbrs, expect), f"query {q_idx}"
            assert np.allclose(dists, dmat[q_idx, expect], atol=1e-9), (
                f"query {q_idx}"
            )
            assert np.all(np.diff(dists) >= -1e-12)

    def test_query_pairs(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, _ = cloud
        tree = ngeom.Octree(pts)
        d = 1.5

        pairs = tree.query_pairs(d=d)
        assert pairs.ndim == 2
        assert pairs.shape[1] == 2

        n = pts.shape[0]
        dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        iu, ju = np.triu_indices(n, k=1)
        expect_mask = dmat[iu, ju] <= d
        expect_count = int(expect_mask.sum())

        assert pairs.shape[0] == expect_count

        a = np.minimum(pairs[:, 0], pairs[:, 1])
        b = np.maximum(pairs[:, 0], pairs[:, 1])
        assert (a < b).all()

        got = np.unique(a * n + b)
        exp = np.unique(iu[expect_mask] * n + ju[expect_mask])
        assert np.array_equal(got, exp)

    def test_query_pairs_negative_d(
        self, cloud: Tuple[np.ndarray, np.ndarray]
    ):
        pts, _ = cloud
        tree = ngeom.Octree(pts)
        with pytest.raises(ValueError, match="non-negative"):
            tree.query_pairs(d=-1.0)

    def test_query_tree(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, qry = cloud
        tree = ngeom.Octree(pts)
        other = ngeom.Octree(qry)
        d = 1.5

        nbrs = tree.query_tree(other, d=d)
        assert isinstance(nbrs, list)
        assert len(nbrs) == pts.shape[0]

        dmat = np.linalg.norm(pts[:, None, :] - qry[None, :, :], axis=-1)
        for i, arr in enumerate(nbrs):
            expect = np.flatnonzero(dmat[i] <= d)
            assert np.array_equal(np.sort(arr), expect), f"point {i}"

    def test_query_tree_negative_d(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, qry = cloud
        tree = ngeom.Octree(pts)
        other = ngeom.Octree(qry)
        with pytest.raises(ValueError, match="non-negative"):
            tree.query_tree(other, d=-1.0)


class TestVoxelGrid:
    def test_construct_and_rebuild(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, _ = cloud
        grid = ngeom.VoxelGrid(pts, cutoff=1.5)
        assert grid.cutoff == pytest.approx(1.5)

        new_pts = pts[:50]
        grid.rebuild(new_pts, cutoff=2.0)
        assert grid.cutoff == pytest.approx(2.0)

        qry = pts[:5]
        idxs, dist = grid.find_neighbors(qry)
        assert idxs.shape[1] == 2
        assert idxs.shape[0] == dist.shape[0]

        for q_idx in range(5):
            sel = idxs[:, 0] == q_idx
            assert np.any((idxs[sel, 1] == q_idx) & (dist[sel] == 0.0)), (
                f"query {q_idx} should match itself with distance 0"
            )

        grid.rebuild(pts)
        assert grid.cutoff == pytest.approx(2.0)

    @pytest.mark.parametrize("cutoff", [0.0, -1.0])
    def test_invalid_cutoff(
        self,
        cloud: Tuple[np.ndarray, np.ndarray],
        cutoff: float,
    ):
        pts, _ = cloud
        with pytest.raises(ValueError, match="positive"):
            ngeom.VoxelGrid(pts, cutoff=cutoff)

    def test_rebuild_invalid_cutoff(
        self, cloud: Tuple[np.ndarray, np.ndarray]
    ):
        pts, _ = cloud
        grid = ngeom.VoxelGrid(pts, cutoff=1.5)
        with pytest.raises(ValueError, match="positive"):
            grid.rebuild(pts, cutoff=-1.0)
        with pytest.raises(ValueError, match="positive"):
            grid.rebuild(pts, cutoff=0.0)

    def test_find_neighbors(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, qry = cloud
        d = 1.5
        grid = ngeom.VoxelGrid(pts, cutoff=d)

        idxs, dist = grid.find_neighbors(qry)
        assert idxs.shape[1] == 2
        assert idxs.shape[0] == dist.shape[0]
        assert (dist <= d + 1e-9).all()

        _, mask = _brute_pairs(pts, qry, d)
        assert idxs.shape[0] == int(mask.sum())

        for q_idx in range(qry.shape[0]):
            got = np.sort(idxs[idxs[:, 0] == q_idx, 1])
            expect = np.sort(np.flatnonzero(mask[q_idx]))
            assert np.array_equal(got, expect), f"query {q_idx}"

        rebuilt = np.linalg.norm(pts[idxs[:, 1]] - qry[idxs[:, 0]], axis=-1)
        assert np.allclose(dist, rebuilt, atol=1e-9)

    def test_query_pairs(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, _ = cloud
        d = 1.5
        grid = ngeom.VoxelGrid(pts, cutoff=d)

        pairs = grid.query_pairs()
        assert pairs.ndim == 2
        assert pairs.shape[1] == 2

        n = pts.shape[0]
        dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        iu, ju = np.triu_indices(n, k=1)
        expect_mask = dmat[iu, ju] <= d
        expect_count = int(expect_mask.sum())

        assert pairs.shape[0] == expect_count

        a = np.minimum(pairs[:, 0], pairs[:, 1])
        b = np.maximum(pairs[:, 0], pairs[:, 1])
        assert (a < b).all()

        got = np.unique(a * n + b)
        exp = np.unique(iu[expect_mask] * n + ju[expect_mask])
        assert np.array_equal(got, exp)

    def test_query_grid(self, cloud: Tuple[np.ndarray, np.ndarray]):
        pts, qry = cloud
        d = 1.5
        grid = ngeom.VoxelGrid(pts, cutoff=d)
        other = ngeom.VoxelGrid(qry, cutoff=d)

        nbrs = grid.query_grid(other)
        assert isinstance(nbrs, list)
        assert len(nbrs) == pts.shape[0]

        dmat = np.linalg.norm(pts[:, None, :] - qry[None, :, :], axis=-1)
        for i, arr in enumerate(nbrs):
            expect = np.flatnonzero(dmat[i] <= d)
            assert np.array_equal(np.sort(arr), expect), f"point {i}"

    def test_query_grid_mismatched_cutoff(
        self, cloud: Tuple[np.ndarray, np.ndarray]
    ):
        pts, qry = cloud
        grid = ngeom.VoxelGrid(pts, cutoff=1.5)
        other = ngeom.VoxelGrid(qry, cutoff=2.0)

        nbrs = grid.query_grid(other)
        assert len(nbrs) == pts.shape[0]

        dmat = np.linalg.norm(pts[:, None, :] - qry[None, :, :], axis=-1)
        for i, arr in enumerate(nbrs):
            expect = np.flatnonzero(dmat[i] <= grid.cutoff)
            assert np.array_equal(np.sort(arr), expect), f"point {i}"

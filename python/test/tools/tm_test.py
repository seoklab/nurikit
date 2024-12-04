#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Tuple

import numpy as np
import pytest

from nuri.tools import tm as tmtools


@pytest.fixture()
def query():
    return np.array(
        [
            [6.468, 147.247, 1.898],
            [5.575, 143.722, 3.203],
            [5.286, 141.68, 6.406],
            [7.418, 138.579, 5.998],
            [6.865, 136.019, 8.742],
            [7.602, 129.767, 12.504],
            [10.676, 128.272, 13.922],
            [10.549, 125.504, 16.56],
            [12.582, 125.983, 19.831],
            [12.556, 127.662, 23.311],
            [14.358, 130.868, 22.369],
            [13.155, 133.223, 24.992],
            [16.506, 133.627, 26.676],
            [18.315, 134.029, 23.375],
            [17.874, 136.187, 20.28],
            [18.362, 134.986, 16.739],
            [19.367, 136.896, 13.605],
            [17.877, 135.802, 10.337],
            [18.434, 136.495, 6.671],
        ]
    )


@pytest.fixture()
def templ():
    return np.array(
        [
            [-15.904, 0.493, 6.955],
            [-13.38, 2.503, 4.941],
            [-9.748, 3.625, 5.041],
            [-7.877, 6.604, 3.598],
            [-4.273, 6.286, 2.426],
            [-2.033, 9.201, 1.455],
            [1.45, 8.952, -0.051],
            [3.985, 11.104, -1.899],
            [6.007, 10.672, -5.091],
            [9.637, 9.482, -4.874],
            [12.257, 12.088, -5.774],
            [14.161, 9.281, -7.494],
            [12.914, 7.62, -10.666],
            [13.075, 7.523, -14.458],
            [7.187, 9.652, -14.678],
            [5.11, 6.995, -12.92],
            [2.402, 5.651, -15.227],
            [0.074, 4.641, -12.394],
            [0.224, 4.123, -8.629],
            [-0.587, 0.619, -7.393],
        ]
    )


@pytest.fixture()
def refs():
    xform_ref = np.array(
        [
            [0.234082, -0.966243, -0.107608, 128.475],
            [-0.563769, -0.225079, 0.794672, 36.6465],
            [-0.792067, -0.125353, -0.597425, 29.295],
            [0, 0, 0, 1],
        ]
    )
    score_ref = 0.19080501444867484
    return xform_ref, score_ref


@pytest.fixture()
def aln_in():
    return [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
    ]


@pytest.fixture()
def aln_out():
    return np.array(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
        ]
    )


def test_tm_align(
    query: np.ndarray,
    templ: np.ndarray,
    refs: Tuple[np.ndarray, float],
):
    xform_ref, score_ref = refs
    xform, score = tmtools.tm_align(query, templ)

    assert xform == pytest.approx(xform_ref, rel=1e-5)
    assert score == pytest.approx(score_ref, rel=1e-5)


def test_tm_score(
    query: np.ndarray,
    templ: np.ndarray,
    aln_in: List[Tuple[int, int]],
    refs: Tuple[np.ndarray, float],
):
    xform_ref, score_ref = refs
    xform, score = tmtools.tm_score(query, templ, aln_in)

    assert xform == pytest.approx(xform_ref, rel=1e-5)
    assert score == pytest.approx(score_ref, rel=1e-5)


def test_tm_score_self(query: np.ndarray):
    xform, score = tmtools.tm_score(query, query)

    assert xform == pytest.approx(np.eye(4), rel=1e-5)
    assert score == pytest.approx(1.0, rel=1e-5)


def test_tm_align_full(
    query: np.ndarray,
    templ: np.ndarray,
    aln_in: List[Tuple[int, int]],
    aln_out: np.ndarray,
    refs: Tuple[np.ndarray, float],
):
    xform_ref, score_ref = refs
    rmsd_ref = 1.9107255286124583
    query_ss = "CCECCCCECCCCTCEEECC"
    templ_ss = "CCEEEEEEECCCCCCCCCCC"

    tm = tmtools.TMAlign(query, templ, query_ss)
    xform, score = tm.score()

    assert xform == pytest.approx(xform_ref, rel=1e-5)
    assert score == pytest.approx(score_ref, rel=1e-5)
    assert tm.rmsd() == pytest.approx(rmsd_ref, rel=1e-5)
    assert np.all(tm.aligned_pairs() == aln_out)

    tm = tmtools.TMAlign(query, templ, query_ss, templ_ss)
    xform, score = tm.score()

    assert xform == pytest.approx(xform_ref, rel=1e-5)
    assert score == pytest.approx(score_ref, rel=1e-5)
    assert tm.rmsd() == pytest.approx(rmsd_ref, rel=1e-5)
    assert np.all(tm.aligned_pairs() == aln_out)

    tm = tmtools.TMAlign.from_alignment(query, templ, aln_in)
    xform, score = tm.score()

    assert xform == pytest.approx(xform_ref, rel=1e-5)
    assert score == pytest.approx(score_ref, rel=1e-5)
    assert tm.rmsd() == pytest.approx(rmsd_ref, rel=1e-5)
    assert np.all(tm.aligned_pairs() == aln_out)


def test_tm_errors():
    with pytest.raises(ValueError, match="at least 5 residues"):
        tmtools.tm_align(np.zeros((4, 3)), np.zeros((10, 3)))

    with pytest.raises(ValueError, match="must have the same length"):
        tmtools.tm_align(np.zeros((11, 3)), np.zeros((10, 3)), query_ss="C")

    with pytest.raises(ValueError, match="out-of-range"):
        tmtools.tm_score(np.zeros((11, 3)), np.zeros((10, 3)), [(0, 10)])

    with pytest.raises(ValueError, match="must have the same length"):
        tmtools.tm_score(np.zeros((11, 3)), np.zeros((10, 3)))

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import math

import numpy as np
import pytest

from nuri.scpilot.arrangement import (
    Caps,
    DegenerateGeometryError,
    cluster_points,
    prepare_caps,
    solve_caps,
)
from nuri.scpilot.surface import fibonacci_sphere

LATTICE = fibonacci_sphere(2_000_000)


def caps_from(axes, alphas) -> Caps:
    axes = np.asarray(axes, dtype=float).reshape(-1, 3)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    return Caps(axes, np.cos(alphas), np.sin(alphas))


def lattice_area(arr, radius=1.0):
    frac = 1.0 - arr.contains(LATTICE).mean()
    return 4.0 * math.pi * radius * radius * frac


def polar_axis(alpha, psi):
    """Axis whose cap circle of radius ``alpha`` passes through +z."""
    return [
        math.sin(alpha) * math.cos(psi),
        math.sin(alpha) * math.sin(psi),
        math.cos(alpha),
    ]


def test_no_caps():
    arr = solve_caps(2.0, Caps.empty())
    assert len(arr.caps) == 0
    assert arr.n_patches == 1
    assert arr.area == pytest.approx(16.0 * math.pi)


@pytest.mark.parametrize("alpha", [0.3, math.pi / 2, 2.2])
def test_single_cap(alpha):
    arr = solve_caps(1.5, caps_from([[0.0, 0.0, 1.0]], [alpha]))
    assert arr.n_loops == 1
    assert arr.n_patches == 1
    assert arr.area == pytest.approx(
        2.0 * math.pi * 1.5**2 * (1 + math.cos(alpha))
    )


def test_two_crossing_caps():
    arr = solve_caps(1.0, caps_from([[0, 0, 1], [1, 0, 1]], [0.6, 0.5]))
    assert arr.n_loops == 1
    assert arr.n_patches == 1
    assert len(np.unique(arr.arcs.v_beg)) == 2
    assert arr.area == pytest.approx(lattice_area(arr), rel=2e-3)


def test_nested_caps():
    arr = solve_caps(1.0, caps_from([[0, 0, 1], [0.05, 0, 1]], [1.0, 0.3]))
    assert len(arr.caps) == 1
    assert arr.area == pytest.approx(2.0 * math.pi * (1 + math.cos(1.0)))


@pytest.mark.parametrize(
    "eps", [1e-4, -1e-4, 1e-6, -1e-6, 1e-9, -1e-9, 1e-12, -1e-12, 0.0]
)
def test_tangent_caps(eps):
    a1, a2 = 0.5, 0.7
    gamma = a1 + a2 + eps
    axes = [[0, 0, 1], [math.sin(gamma), 0, math.cos(gamma)]]
    arr = solve_caps(1.0, caps_from(axes, [a1, a2]))
    expected = 4 * math.pi - 2 * math.pi * (2 - math.cos(a1) - math.cos(a2))
    assert arr.area == pytest.approx(expected, abs=1e-5)
    assert arr.n_patches == 1


def test_three_circles_through_one_point():
    alphas = [0.4, 0.6, 0.5]
    axes = [polar_axis(a, psi) for a, psi in zip(alphas, [0.0, 2.1, 4.0])]
    arr = solve_caps(1.0, caps_from(axes, alphas))
    ends = np.concatenate([arr.arcs.v_beg, arr.arcs.v_end])
    assert (len(arr.arcs), len(np.unique(ends)), arr.n_loops) == (3, 3, 1)
    assert arr.area == pytest.approx(lattice_area(arr), rel=2e-3)


def test_coincident_caps_merge():
    axes = [[0.0, 0.0, 1.0], [1e-15, 0.0, 1.0]]
    caps, covered, _ = prepare_caps(caps_from(axes, [math.pi / 2] * 2), 2.0)
    assert not covered
    assert len(caps) == 1
    arr = solve_caps(2.0, caps_from(axes, [math.pi / 2] * 2))
    assert arr.area == pytest.approx(2.0 * math.pi * 4.0)


def test_nearby_caps_stay_distinct():
    axes = [[0.0, 0.0, 1.0], [1e-5, 0.0, 1.0]]
    caps, _, _ = prepare_caps(caps_from(axes, [0.7, 0.7]), 1.0)
    assert len(caps) == 2
    arr = solve_caps(1.0, caps_from(axes, [0.7, 0.7]))
    assert arr.area == pytest.approx(
        2.0 * math.pi * (1 + math.cos(0.7)), rel=1e-4
    )


@pytest.mark.parametrize(
    "eps", [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0, -1e-12, -1e-10, -1e-8]
)
def test_internally_tangent_caps(eps):
    a1, a2 = 1.0, 0.3
    gamma = a1 - a2 + eps
    axes = [[0, 0, 1], [math.sin(gamma), 0, math.cos(gamma)]]
    arr = solve_caps(1.0, caps_from(axes, [a1, a2]))
    assert arr.n_patches == 1
    poke = 2.0 * abs(eps) ** 1.5
    assert arr.area == pytest.approx(
        2 * math.pi * (1 + math.cos(a1)), abs=1e-9 + poke
    )


@pytest.mark.parametrize("gamma", [1e-4, 1e-6, 1e-7])
def test_nearly_parallel_crossing_caps(gamma):
    axes = [[0, 0, 1], [math.sin(gamma), 0, math.cos(gamma)]]
    arr = solve_caps(1.0, caps_from(axes, [0.7, 0.7]))
    assert arr.n_patches == 1
    assert arr.area == pytest.approx(
        2 * math.pi * (1 + math.cos(0.7)), abs=4 * gamma
    )


def test_fully_covered():
    arr = solve_caps(1.0, caps_from([[0, 0, 1], [0, 0, -1]], [2.2, 2.2]))
    assert arr.n_patches == 0
    assert arr.area == 0.0


def test_belt_splits_two_patches():
    psis = np.arange(8) * (2 * math.pi / 8)
    axes = [[math.cos(p), math.sin(p), 0.0] for p in psis]
    arr = solve_caps(1.0, caps_from(axes, [0.9] * 8))
    assert arr.n_loops == 2
    assert arr.n_patches == 2
    assert arr.area == pytest.approx(lattice_area(arr), rel=2e-3)


def test_octant_from_hemispheres():
    axes = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    arr = solve_caps(2.0, caps_from(axes, [math.pi / 2] * 3))
    assert arr.n_loops == 1
    assert arr.area == pytest.approx(4.0 * math.pi / 2)


def test_random_caps_match_lattice():
    rng = np.random.default_rng(3)
    for _ in range(5):
        n = rng.integers(3, 12)
        axes = rng.normal(size=(n, 3))
        alphas = rng.uniform(0.2, 1.4, size=n)
        arr = solve_caps(1.0, caps_from(axes, alphas))
        assert arr.area == pytest.approx(lattice_area(arr), abs=3e-3)


def test_cluster_points():
    pts = np.array(
        [[0, 0, 0], [1e-8, 0, 0], [1, 0, 0], [1, 1e-8, 0], [5, 5, 5]]
    )
    labels = cluster_points(pts, 1e-6)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert len(set(labels.tolist())) == 3


def test_degenerate_error_type():
    assert issubclass(DegenerateGeometryError, RuntimeError)

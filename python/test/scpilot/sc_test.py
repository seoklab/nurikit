#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nuri.scpilot.sc import (
    ScParams,
    active_atoms,
    shape_complementarity,
)


def slab(z: float, n: int = 7, spacing: float = 3.0):
    xs = (np.arange(n) - (n - 1) / 2) * spacing
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    return np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, z, dtype=float)]
    )


def test_active_atoms():
    a = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 5.0]])
    np.testing.assert_array_equal(
        active_atoms(a, cKDTree(b), 8.0), [True, False]
    )


def test_facing_slabs_have_high_complementarity():
    radii_a = np.full(49, 1.7)
    radii_b = np.full(49, 1.7)
    a = slab(0.0)
    b = slab(3.4) + np.array([1.5, 1.5, 0.0])
    params = ScParams(density=15.0)
    tight = shape_complementarity(a, radii_a, b, radii_b, params)
    assert 0.6 < tight.sc <= 0.999
    assert tight.distance < 1.0
    assert tight.area > 0.0
    assert all(s.n_trimmed > 0 for s in tight.sides)

    b_far = slab(4.4) + np.array([1.5, 1.5, 0.0])
    loose = shape_complementarity(a, radii_a, b_far, radii_b, params)
    assert loose.sc < tight.sc
    assert loose.distance > tight.distance


def test_no_interface_raises():
    a = np.zeros((1, 3))
    b = np.array([[30.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no interface"):
        shape_complementarity(a, [1.7], b, [1.7])

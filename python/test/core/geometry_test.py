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

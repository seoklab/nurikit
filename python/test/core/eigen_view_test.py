#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Tuple

import numpy as np
import pytest

from nuri.core._core import _EigenViewTestData

_DYNAMIC = np.arange(12, dtype=np.float64).reshape(4, 3)
_SEMI_DYNAMIC = np.arange(15, dtype=np.float64).reshape(5, 3)
_FIXED = np.arange(16, dtype=np.float64).reshape(4, 4)
_VECTOR = np.arange(5, dtype=np.float64)


RESOLVER = {
    "dynamic": (_DYNAMIC, _EigenViewTestData.dynamic),
    "semi_dynamic": (_SEMI_DYNAMIC, _EigenViewTestData.semi_dynamic),
    "fixed": (_FIXED, _EigenViewTestData.fixed),
    "vector": (_VECTOR, _EigenViewTestData.vector),
}


@pytest.mark.parametrize("source", list(RESOLVER))
@pytest.mark.parametrize("readonly", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_view_matches_source(source: str, readonly: bool, transpose: bool):
    exp, resolver = RESOLVER[source]
    data = _EigenViewTestData(**{source: exp})
    if transpose:
        exp = exp.T

    v = resolver(data, readonly, transpose)
    w = resolver(data, readonly, transpose)

    assert v.base is data
    assert not v.flags.owndata
    assert v.flags.writeable == (not readonly)

    assert v.shape == exp.shape
    assert v.ndim == exp.ndim
    assert v.dtype == exp.dtype
    np.testing.assert_array_equal(v, exp)

    # column-major Eigen storage maps back to the numpy orientation; a transpose
    # is a stride swap over the same buffer (F-contiguous for a 2D source)
    assert v.strides == exp.strides
    assert v.flags.c_contiguous == exp.flags.c_contiguous
    assert v.flags.f_contiguous == exp.flags.f_contiguous

    # every call aliases the same internal Eigen buffer, not a fresh copy
    ro = resolver(data, True, transpose)
    rw = resolver(data, False, transpose)
    assert np.shares_memory(v, w)
    assert np.shares_memory(v, ro)
    assert np.shares_memory(v, rw)

    if readonly:
        with pytest.raises(ValueError, match="read-only"):
            v[:] = 999.0
    else:
        v[:] = 999.0
        np.testing.assert_array_equal(w, 999.0)
        np.testing.assert_array_equal(ro, 999.0)
        np.testing.assert_array_equal(rw, 999.0)


def test_empty_unset_member():
    data = _EigenViewTestData()

    cases: List[Tuple[np.ndarray, Tuple[int, ...]]] = [
        (data.dynamic(False, False), (0, 0)),
        (data.semi_dynamic(False, False), (0, 3)),
        (data.vector(False, False), (0,)),
    ]

    for v, shape in cases:
        assert v.shape == shape
        assert v.ndim == len(shape)
        assert v.size == 0
        assert v.dtype == np.float64

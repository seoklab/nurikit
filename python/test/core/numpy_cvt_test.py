#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from nuri.core._core import _py_array_cast_test_helper as cast_test_helper


def _assert_same_array(arr1: np.ndarray, arr2: np.ndarray):
    assert arr1.shape == arr2.shape
    assert arr1.data.obj is arr2.data.obj


def test_cast_1d():
    arr = np.arange(3, dtype=np.float64)

    with pytest.raises(ValueError, match="expected 2D"):
        cast_test_helper(arr, "matrix")

    with pytest.raises(ValueError, match="expected 2D"):
        cast_test_helper(arr, "dynamic")

    out = cast_test_helper(arr, "col_vector")
    _assert_same_array(arr, out)

    out = cast_test_helper(arr, "row_vector")
    _assert_same_array(arr, out)


def test_cast_2d():
    arr = np.arange(18, dtype=np.float64).reshape(6, 3)[::2]

    out = cast_test_helper(arr, "matrix")
    _assert_same_array(arr, out)

    out = cast_test_helper(arr, "dynamic")
    _assert_same_array(arr, out)

    with pytest.raises(ValueError, match="expected 1D"):
        cast_test_helper(arr, "col_vector")

    with pytest.raises(ValueError, match="expected 1D"):
        cast_test_helper(arr, "row_vector")


def test_cast_shape_mismatch():
    arr = np.arange(12, dtype=np.float64)

    with pytest.raises(ValueError, match="expected 3 rows"):
        cast_test_helper(arr.reshape(4, 3), "matrix")

    with pytest.raises(ValueError, match="expected 3 columns"):
        cast_test_helper(arr.reshape(3, 4), "matrix")

    with pytest.raises(ValueError, match="expected 3 elements"):
        cast_test_helper(arr, "col_vector")

    with pytest.raises(ValueError, match="expected 3 elements"):
        cast_test_helper(arr, "row_vector")


def _assert_same_content(arr1: np.ndarray, arr2: np.ndarray):
    assert arr1.shape == arr2.shape
    assert np.allclose(arr1, arr2)
    assert arr1.data.obj is not arr2.data.obj


def test_cast_inner_strided():
    arr = np.arange(18, dtype=np.float64).reshape(3, 6)[:, ::2]

    out = cast_test_helper(arr, "matrix")
    _assert_same_content(arr, out)

    out = cast_test_helper(arr, "dynamic")
    _assert_same_content(arr, out)

    arr = arr[0]

    out = cast_test_helper(arr, "col_vector")
    _assert_same_content(arr, out)

    out = cast_test_helper(arr, "row_vector")
    _assert_same_content(arr, out)


def test_cast_converted():
    arr = np.arange(9, dtype=np.int32).reshape(3, 3)

    out = cast_test_helper(arr, "matrix")
    _assert_same_content(arr, out)

    out = cast_test_helper(arr, "dynamic")
    _assert_same_content(arr, out)

    arr = arr[0]

    out = cast_test_helper(arr, "col_vector")
    _assert_same_content(arr, out)

    out = cast_test_helper(arr, "row_vector")
    _assert_same_content(arr, out)


def test_cast_convert_failed():
    obj = ["data", "cannot", "be", "converted", "to", "numpy", "array"]

    with pytest.raises(ValueError, match="cannot convert"):
        cast_test_helper(obj, "dynamic")

    obj = [[0, 1, 2], [3, 4]]

    with pytest.raises(ValueError, match="cannot convert"):
        cast_test_helper(obj, "dynamic")

    with pytest.raises(TypeError, match="got None"):
        cast_test_helper(None, "dynamic")


def test_cast_empty():
    obj = [[]]
    out = cast_test_helper(obj, "dynamic")
    assert out.shape == (1, 0)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous

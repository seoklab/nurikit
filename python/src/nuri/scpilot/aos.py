#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Records are the unit of storage; arrays are gathered where one kernel
expression runs over many records (the C++ port writes an Eigen expression
there)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def vectors(records: Sequence, name: str) -> np.ndarray:
    """``(k, 3)`` array of the 3-vector field ``name``."""
    out = np.empty((len(records), 3))
    for i, r in enumerate(records):
        out[i] = getattr(r, name)
    return out


def scalars(records: Sequence, name: str, dtype=float) -> np.ndarray:
    """``(k,)`` array of the scalar field ``name``."""
    return np.fromiter(
        (getattr(r, name) for r in records), dtype=dtype, count=len(records)
    )

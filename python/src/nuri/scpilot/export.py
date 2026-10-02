#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Dump sampled SES dots (:mod:`nuri.scpilot.surface`) as a mol2 dot cloud.

The bond section is written empty, so a viewer draws no connectivity over
the ~10^5 dots. Each dot lands in its own substructure, named after its
patch kind and numbered ``3 * owner atom + kind + 1``; the atom name flags
the buried state and the charge column carries one per-dot scalar. In
ChimeraX::

    open dots.mol2
    style sphere; size atomRadius 0.15
    color :CVX cornflowerblue; color :TOR palegreen; color :CCV salmon
    color byattribute charge
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TextIO

import numpy as np

from .surface import Dots

PATCH_NAMES = ("CVX", "TOR", "CCV")


def _atom_lines(
    dots: Dots,
    scalar: np.ndarray,
    buried: np.ndarray,
    subst: np.ndarray,
) -> Iterator[str]:
    rows = zip(dots.pts, dots.kinds, subst, scalar, buried)
    for i, (p, kind, sid, value, flag) in enumerate(rows, 1):
        yield (
            f"{i:>7} {'DB' if flag else 'DX':<8}"
            f"{p[0]:>10.4f}{p[1]:>10.4f}{p[2]:>10.4f}"
            f" {'H':<5} {sid:>7} {PATCH_NAMES[kind]:<8}{value:>10.4f}\n"
        )


def write_dots(
    out: TextIO,
    dots: Dots,
    name: str = "ses_dots",
    scalar: np.ndarray | None = None,
    buried: np.ndarray | None = None,
) -> None:
    """Write ``dots`` to the open text file ``out`` in mol2 format.

    ``scalar`` fills the charge column and defaults to the dot areas;
    ``buried`` marks dots with the ``DB`` atom name instead of ``DX``.
    Both are broadcast against the dots.
    """
    n = len(dots)
    shape = (n,)
    values = dots.areas if scalar is None else np.broadcast_to(scalar, shape)
    flags = (
        np.zeros(n, dtype=bool)
        if buried is None
        else np.broadcast_to(buried, shape)
    )
    subst = 3 * dots.atoms + dots.kinds + 1

    out.write(
        f"@<TRIPOS>MOLECULE\n{name}\n"
        f"{n} 0 {np.unique(subst).size} 0 0\nSMALL\nUSER_CHARGES\n\n"
        f"@<TRIPOS>ATOM\n"
    )
    out.writelines(_atom_lines(dots, values, flags, subst))
    out.write("@<TRIPOS>BOND\n")

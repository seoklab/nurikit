#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

# pyright: reportUnusedImport=false
"""
Project NuriKit: *the* fundamental software platform for chem- and
bio-informatics."""

__all__ = [
    "__version__",
    "periodic_table",
    "readfile",
    "readstring",
    "seed_thread",
    "to_mol2",
    "to_pdb",
    "to_sdf",
    "to_smiles",
]

try:
    from ._version import __full_version__, __version__
except ImportError:
    __version__ = __full_version__ = "unknown"

from . import _log_adapter
from .core import periodic_table, seed_thread
from .fmt import readfile, readstring, to_mol2, to_pdb, to_sdf, to_smiles

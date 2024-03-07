#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

# pyright: reportUnusedImport=false
"""
Project nurikit: *the* fundamental software platform for chem- and
bio-informatics."""

__all__ = [
    "readfile",
    "readstring",
    "periodic_table",
    "__version__",
]

try:
    from ._version import __version__, __full_version__
except ImportError:
    __version__ = __full_version__ = "unknown"

from . import _log_adapter
from .core import periodic_table
from .fmt import readfile, readstring

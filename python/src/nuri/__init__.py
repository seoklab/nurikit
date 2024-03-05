#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#
"""
Project nurikit: *the* fundamental software platform for chem- and
bio-informatics."""

__all__ = [
    "periodic_table",
    "__version__",
]

try:
    from ._version import __version__, __full_version__
except ImportError:
    __version__ = __full_version__ = "unknown"

from .core import periodic_table

try:
    from ._log_adapter import _init as _init_logging
    _init_logging()
    del _init_logging
except Exception:
    import logging as _logging
    _logging.warning(
        "failed to initialize logging for nurikit.", exc_info=True)
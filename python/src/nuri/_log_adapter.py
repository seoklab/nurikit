#
# Project nurikit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import logging
import functools

logger = logging.getLogger("nuri")
_setLevel = logger.setLevel

try:
    from ._log_interface import _init as _init_logging, set_log_level
    _init_logging()
    del _init_logging
except Exception:
    logger.warning(
        "failed to initialize logging for nurikit.", exc_info=True)
else:
    @functools.wraps(_setLevel)
    def set_level_wrapper(level):
        _setLevel(level)
        set_log_level(logger.getEffectiveLevel())

    logger.setLevel = set_level_wrapper

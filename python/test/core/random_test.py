#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import nuri
from nuri.core._core import _random_test_helper


def test_seed_thread():
    nuri.seed_thread()
    seq1 = [_random_test_helper(6) for _ in range(10)]
    nuri.seed_thread()
    seq2 = [_random_test_helper(6) for _ in range(10)]
    assert seq1 != seq2

    nuri.seed_thread(42)
    seq1 = [_random_test_helper(6) for _ in range(10)]
    nuri.seed_thread(42)
    seq2 = [_random_test_helper(6) for _ in range(10)]
    assert seq1 == seq2

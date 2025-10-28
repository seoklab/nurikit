//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/random.h"

#include <random>

namespace nuri {
namespace internal {
  namespace {
    std::seed_seq make_seed_seq(int seed) {
      if (seed >= 0)
        return std::seed_seq({ seed });

      std::random_device rd;
      return std::seed_seq({ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() });
    }
  }  // namespace

  void seed_thread(int seed) {
    std::seed_seq seq = make_seed_seq(seed);
    rng.seed(seq);
  }
}  // namespace internal
}  // namespace nuri

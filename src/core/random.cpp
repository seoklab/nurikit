//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/random.h"

namespace nuri {
namespace internal {
  int set_thread_seed(int seed) {
    if (seed <= 0)
      seed = static_cast<int>(std::random_device()());
    rng.seed(seed);
    return seed;
  }
}  // namespace internal
}  // namespace nuri

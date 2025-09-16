//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_RANDOM_H_
#define NURI_RANDOM_H_

#include <random>

#include <absl/algorithm/container.h>

namespace nuri {
namespace internal {
  // NOLINTNEXTLINE(*-global-variables)
  inline thread_local std::mt19937 rng {};

  template <typename NT>
  NT draw_uid(NT min, NT max) {
    return std::uniform_int_distribution<NT>(min, max - 1)(rng);
  }

  template <typename NT>
  NT draw_uid(NT max) {
    return draw_uid(static_cast<NT>(0), max);
  }

  template <typename RT>
  RT draw_urd(RT min, RT max) {
    return std::uniform_real_distribution<RT>(min, max)(rng);
  }

  template <typename RT>
  RT draw_urd(RT max) {
    return std::uniform_real_distribution<RT>(static_cast<RT>(0), max)(rng);
  }

  template <typename CT, typename RT = typename CT::value_type>
  auto weighted_select(const CT &cutoffs, const RT base = 0) {
    RT max = cutoffs[cutoffs.size() - 1];
    RT p = draw_urd<RT>(base, max);

    auto it = absl::c_lower_bound(cutoffs, p);
    return it - cutoffs.begin();
  }
}  // namespace internal
}  // namespace nuri

#endif /* NURI_RANDOM_H_ */

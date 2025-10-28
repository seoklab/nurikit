//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_RANDOM_H_
#define NURI_RANDOM_H_

#include <algorithm>
#include <iterator>
#include <random>

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
    return draw_urd(static_cast<RT>(0), max);
  }

  template <typename CT, typename RT = typename CT::value_type>
  auto weighted_select(const CT &cutoffs, const RT base = 0) {
    RT max = cutoffs[cutoffs.size() - 1];
    RT p = draw_urd<RT>(base, max);

    auto it = std::lower_bound(std::begin(cutoffs), std::end(cutoffs) - 1, p);
    return it - std::begin(cutoffs);
  }

  extern void set_thread_seed(int seed);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_RANDOM_H_ */

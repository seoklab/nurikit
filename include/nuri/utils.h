//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_UTILS_H_
#define NURI_UTILS_H_

#include <numeric>
#include <type_traits>

namespace nuri {
namespace internal {
  template <bool is_const, class T>
  struct const_if {
    using type = std::conditional_t<is_const, const T, T>;
  };

  template <bool is_const, class T>
  using const_if_t = typename const_if<is_const, T>::type;
}  // namespace internal

template <class Container>
Container mask_to_map(const Container &mask) {
  Container map(mask.size());
  std::inclusive_scan(mask.begin(), mask.end(), map.begin(), std::plus<>(), -1);
  return map;
}
}  // namespace nuri

#endif /* NURI_UTILS_H_ */

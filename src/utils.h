//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURIKIT_UTILS_H_
#define NURIKIT_UTILS_H_

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
}  // namespace nuri

#endif /* NURIKIT_UTILS_H_ */

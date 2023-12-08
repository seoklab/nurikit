//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_GEOMETRY_H_
#define NURI_CORE_GEOMETRY_H_

#include <type_traits>

namespace nuri {
namespace constants {
  extern constexpr inline double kPi =
      3.1415926535897932384626433832795028841971693993751058209749445923078164;

  extern constexpr inline double kTwoPi =
      6.2831853071795864769252867665590057683943387987502116419498891846156328;
}  // namespace constants

template <class DT, std::enable_if_t<std::is_floating_point_v<DT>, int> = 0>
constexpr DT deg2rad(DT deg) {
  return deg * constants::kPi / 180;
}

template <class DT, std::enable_if_t<std::is_integral_v<DT>, int> = 0>
constexpr double deg2rad(DT deg) {
  return deg * constants::kPi / 180;
}
}  // namespace nuri

#endif /* NURI_CORE_GEOMETRY_H_ */

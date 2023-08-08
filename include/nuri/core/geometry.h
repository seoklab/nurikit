//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_GEOMETRY_H_
#define NURI_CORE_GEOMETRY_H_

#include <Eigen/Dense>

#include "nuri/eigen_config.h"

namespace nuri {
namespace constants {
  extern constexpr inline double kPi =
      3.1415926535897932384626433832795028841971693993751058209749445923078164;

  extern constexpr inline double kTwoPi =
      6.2831853071795864769252867665590057683943387987502116419498891846156328;
}  // namespace constants

template <class DT>
constexpr DT deg2rad(DT deg) {
  return deg * constants::kPi / 180;
}

/**
 * @brief A (simplified) re-implementation of Eigen::AngleAxis (for row-major
 *        matrices).
 *
 * @tparam DT The data type of the angle axis (float or double).
 */
template <class DT>
class AngleAxis {
public:
  AngleAxis() = default;

  AngleAxis(double angle, const Vector3<DT> &axis)
      : axis_(axis), angle_(angle) { }

  AngleAxis(double angle, Vector3<DT> &&axis) noexcept
      : axis_(std::move(axis)), angle_(angle) { }

  const Vector3<DT> &axis() const { return axis_; }

  double angle() const { return angle_; }

  Matrix3<DT> to_matrix() const {
    const DT c = std::cos(angle_), s = std::sin(angle_);
    Vector3<DT> sin_axis = axis_ * s, omcos_axis = axis_ * (1 - c);

    Matrix3<DT> m;
    m.diagonal() = omcos_axis.cwiseProduct(axis_).array() + c;

    const DT xy = omcos_axis.x() * axis_.y(), yz = omcos_axis.y() * axis_.z(),
             zx = omcos_axis.z() * axis_.x();

    m(0, 1) = xy - sin_axis.z();
    m(0, 2) = zx + sin_axis.y();
    m(1, 0) = xy + sin_axis.z();
    m(1, 2) = yz - sin_axis.x();
    m(2, 0) = zx - sin_axis.y();
    m(2, 1) = yz + sin_axis.x();

    return m;
  }

private:
  Vector3<DT> axis_;
  DT angle_;
};

using AngleAxisf = AngleAxis<float>;
using AngleAxisd = AngleAxis<double>;
}  // namespace nuri

#endif /* NURI_CORE_GEOMETRY_H_ */

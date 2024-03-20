//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_EIGEN_CONFIG_H_
#define NURI_EIGEN_CONFIG_H_

#include <Eigen/Dense>

#include <absl/log/absl_check.h>

namespace nuri {
using Eigen::Array;
using Eigen::Array3d;
using Eigen::Array3i;
using Eigen::Array4i;
using Eigen::ArrayX;
using ArrayXb = Eigen::ArrayX<bool>;
using Eigen::ArrayXd;
using Eigen::ArrayXi;

using Eigen::Matrix;
using Eigen::Matrix3;
using Eigen::Matrix3d;
using Eigen::Matrix3Xd;
using Eigen::MatrixXd;

using Eigen::Vector;
using Eigen::Vector3;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

/**
 * @brief Cyclic indexer for Eigen types with a given offset.
 * @tparam O Offset of the cyclic index. If 0, it is determined at runtime.
 * @tparam N Size of the cyclic index. If 0, it is determined at runtime.
 *
 * This will produce an index that is cyclic with a given offset, i.e.,
 * (O, O+1, ..., N-1, 0, 1, ..., O-1).
 */
template <int O = 0, int N = 0>
class CyclicIndex {
public:
  static_assert(N > 0, "N must be positive");

  constexpr static int size() { return N; }

  constexpr int operator[](int i) const {
    int before = O + i, after = i - kZeroAt;
    return i < kZeroAt ? before : after;
  }

private:
  constexpr static int kZeroAt = (N - O) % N;
  static_assert(kZeroAt >= 0, "Offset must be smaller than size");
};

template <int O>
class CyclicIndex<O, 0> {
public:
  constexpr CyclicIndex(int n): n_(n), zero_at_((n - O) % n) {
    ABSL_DCHECK(zero_at_ >= 0);
  }

  constexpr int size() const { return n_; }

  constexpr int operator[](int i) const {
    int before = O + i, after = i - zero_at_;
    return i < zero_at_ ? before : after;
  }

private:
  int n_;
  int zero_at_;
};

template <>
class CyclicIndex<0, 0> {
public:
  constexpr CyclicIndex(int n, int offset)
      : n_(n), offset_(offset), zero_at_((n - offset) % n) {
    ABSL_DCHECK(zero_at_ >= 0);
  }

  constexpr int size() const { return n_; }

  constexpr int operator[](int i) const {
    int before = offset_ + i, after = i - zero_at_;
    return i < zero_at_ ? before : after;
  }

private:
  int n_;
  int offset_;
  int zero_at_;
};
}  // namespace nuri

#endif /* NURI_EIGEN_CONFIG_H_ */

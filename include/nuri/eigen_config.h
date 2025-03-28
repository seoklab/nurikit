//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_EIGEN_CONFIG_H_
#define NURI_EIGEN_CONFIG_H_

//! @cond
#include <type_traits>  // IWYU pragma: keep, required for is_class_v

#include <absl/log/absl_check.h>
#include <Eigen/Dense>
//! @endcond

#include "nuri/meta.h"

//! @cond

#ifdef NURI_DEBUG
#define NURI_EIGEN_TMP(type) Eigen::type
#else
#define NURI_EIGEN_TMP(type)                                                   \
  { static_assert(std::is_class_v<Eigen::type>); }                             \
  auto
#endif

#ifdef EIGEN_RUNTIME_NO_MALLOC
#define NURI_EIGEN_ALLOW_MALLOC(flag)                                          \
  Eigen::internal::set_is_malloc_allowed(flag)
#define NURI_EIGEN_MALLOC_ALLOWED() Eigen::internal::is_malloc_allowed()
#else
#define NURI_EIGEN_ALLOW_MALLOC(flag) static_cast<void>(flag)
#define NURI_EIGEN_MALLOC_ALLOWED()   true
#endif

//! @endcond

namespace nuri {
//! @privatesection

using Eigen::Array;
using Eigen::Array3d;
using Eigen::Array3i;
using Eigen::Array4d;
using Eigen::Array4i;
using Eigen::ArrayX;
using ArrayXb = Eigen::ArrayX<bool>;
using ArrayXc = ArrayX<std::int8_t>;
using Eigen::Array2Xd;
using Eigen::Array2Xi;
using Eigen::Array33d;
using Eigen::Array3Xd;
using Eigen::Array4Xd;
using Eigen::ArrayX2d;
using Eigen::ArrayX2i;
using Eigen::ArrayXd;
using Eigen::ArrayXi;
using Eigen::ArrayXX;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using ArrayXXc = ArrayXX<std::int8_t>;

using Eigen::Matrix;
using Eigen::Matrix3;
using Eigen::Matrix3d;
using Eigen::Matrix3Xd;
using Eigen::Matrix4d;
using Eigen::Matrix4Xd;
using Eigen::MatrixX;
using Eigen::MatrixX3d;
using Eigen::MatrixXd;

using Eigen::Vector;
using Eigen::Vector3;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using Eigen::Affine3d;
using Eigen::Translation3d;

template <class Raw, int Options = 0,
          class StrideType =
              std::conditional_t<Raw::IsVectorAtCompileTime,
                                 Eigen::InnerStride<1>, Eigen::OuterStride<>>>
using MutRef = Eigen::Ref<internal::remove_cvref_t<Raw>, Options, StrideType>;

template <class Raw, int Options = 0,
          class StrideType =
              std::conditional_t<Raw::IsVectorAtCompileTime,
                                 Eigen::InnerStride<1>, Eigen::OuterStride<>>>
using ConstRef =
    const Eigen::Ref<const internal::remove_cvref_t<Raw>, Options, StrideType> &;

template <class Raw, int BlockRows = Eigen::Dynamic,
          int BlockCols = Eigen::Dynamic, bool InnerPanel = false>
using MutBlock = Eigen::Block<internal::remove_cvref_t<Raw>, BlockRows,
                              BlockCols, InnerPanel>;

template <class Raw, int BlockRows = Eigen::Dynamic,
          int BlockCols = Eigen::Dynamic, bool InnerPanel = false>
using ConstBlock = const Eigen::Block<const internal::remove_cvref_t<Raw>,
                                      BlockRows, BlockCols, InnerPanel> &;

template <class Raw, int Size = Eigen::Dynamic>
using MutVecBlock = Eigen::VectorBlock<internal::remove_cvref_t<Raw>, Size>;

template <class Raw, int Size = Eigen::Dynamic>
using ConstVecBlock =
    const Eigen::VectorBlock<const internal::remove_cvref_t<Raw>, Size> &;

//! @publicsection

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

//! @privatesection

template <class ML1, class ML2>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
void inplace_transform(ML1 &&m_out, const Affine3d &xform, const ML2 &m) {
  ABSL_DCHECK_EQ(m_out.rows(), m.rows());
  ABSL_DCHECK_EQ(m_out.cols(), m.cols());

  m_out.noalias() = xform.linear() * m;
  m_out.colwise() += xform.translation();
}

namespace internal {
  template <bool Allowed>
  class AllowEigenMallocScoped {
  public:
    AllowEigenMallocScoped(const AllowEigenMallocScoped &) = delete;
    AllowEigenMallocScoped(AllowEigenMallocScoped &&) = delete;
    AllowEigenMallocScoped &operator=(const AllowEigenMallocScoped &) = delete;
    AllowEigenMallocScoped &operator=(AllowEigenMallocScoped &&) = delete;

#ifdef EIGEN_RUNTIME_NO_MALLOC
    AllowEigenMallocScoped(): state_(Eigen::internal::is_malloc_allowed()) {
      Eigen::internal::set_is_malloc_allowed(Allowed);
    }

    ~AllowEigenMallocScoped() {
      Eigen::internal::set_is_malloc_allowed(state_);
    }

  private:
    bool state_;
#else
    AllowEigenMallocScoped() = default;
    ~AllowEigenMallocScoped() { }
#endif
  };
}  // namespace internal
}  // namespace nuri

#endif /* NURI_EIGEN_CONFIG_H_ */

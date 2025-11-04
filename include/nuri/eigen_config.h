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

// NOLINTNEXTLINE(*-naming)
namespace E = Eigen;

using E::Array;
using E::Array3d;
using E::Array3i;
using E::Array4d;
using E::Array4i;
using E::ArrayX;
using ArrayXb = E::ArrayX<bool>;
using ArrayXc = ArrayX<std::int8_t>;
using E::Array2Xd;
using E::Array2Xi;
using E::Array33d;
using E::Array3Xd;
using E::Array4Xd;
using E::ArrayX2d;
using E::ArrayX2i;
using E::ArrayX3d;
using E::ArrayXd;
using E::ArrayXi;
using E::ArrayXX;
using E::ArrayXXd;
using E::ArrayXXi;
using ArrayXXc = ArrayXX<std::int8_t>;

using E::Matrix;
using E::Matrix3;
using E::Matrix3d;
using E::Matrix3Xd;
using E::Matrix4d;
using E::Matrix4Xd;
using E::MatrixX;
using E::MatrixX3d;
using E::MatrixXd;

using E::Vector;
using E::Vector3;
using E::Vector3d;
using E::Vector4d;
using E::VectorXd;

using E::AngleAxisd;
using E::Isometry3d;
using E::Quaterniond;
using E::Translation3d;

template <class DT, int Dim>
using IsometryT = E::Transform<DT, Dim, E::Isometry>;

template <class Raw, int Options = 0,
          class StrideType = std::conditional_t<
              Raw::IsVectorAtCompileTime, E::InnerStride<1>, E::OuterStride<>>>
using MutRef = E::Ref<internal::remove_cvref_t<Raw>, Options, StrideType>;

template <class Raw, int Options = 0,
          class StrideType = std::conditional_t<
              Raw::IsVectorAtCompileTime, E::InnerStride<1>, E::OuterStride<>>>
using ConstRef =
    const E::Ref<const internal::remove_cvref_t<Raw>, Options, StrideType> &;

template <class Raw, int BlockRows = E::Dynamic, int BlockCols = E::Dynamic,
          bool InnerPanel = false>
using MutBlock =
    E::Block<internal::remove_cvref_t<Raw>, BlockRows, BlockCols, InnerPanel>;

template <class Raw, int BlockRows = E::Dynamic, int BlockCols = E::Dynamic,
          bool InnerPanel = false>
using ConstBlock = const E::Block<const internal::remove_cvref_t<Raw>,
                                  BlockRows, BlockCols, InnerPanel> &;

template <class Raw, int Size = E::Dynamic>
using MutVecBlock = E::VectorBlock<internal::remove_cvref_t<Raw>, Size>;

template <class Raw, int Size = E::Dynamic>
using ConstVecBlock =
    const E::VectorBlock<const internal::remove_cvref_t<Raw>, Size> &;

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

template <class ML1, class ML2, class TransformLike>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
void inplace_transform(ML1 &&m_out, const TransformLike &xform, const ML2 &m) {
  const E::Index ri = m.rows(), ci = m.cols();
  const E::Index ro = m_out.rows(), co = m_out.cols();
  ABSL_ASSUME(ri == ro && ci == co);

  for (E::Index i = 0; i < m.cols(); ++i)
    m_out.col(i) = xform * m.col(i);
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
    AllowEigenMallocScoped(): state_(E::internal::is_malloc_allowed()) {
      E::internal::set_is_malloc_allowed(Allowed);
    }

    ~AllowEigenMallocScoped() { E::internal::set_is_malloc_allowed(state_); }

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

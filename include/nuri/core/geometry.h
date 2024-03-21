//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_GEOMETRY_H_
#define NURI_CORE_GEOMETRY_H_

/// @cond
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>
/// @endcond

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  using Array8i = Array<int, 8, 1>;

  class OCTreeNode {
  public:
    OCTreeNode(Array8i &&children, int nleaf)
        : children_(std::move(children)), nleaf_(nleaf) { }

    const Array8i &children() const { return children_; }

    int child(int i) const { return children_[i]; }

    int operator[](int i) const { return child(i); }

    bool leaf() const { return nleaf_ <= 8; }

    int nleaf() const { return nleaf_; }

  private:
    Array8i children_;  // < 0 -> not exist, >= 0 -> index of child
    int nleaf_;         // <= 8 -> leaf, > 8 -> internal node
  };
}  // namespace internal

class OCTree {
public:
  OCTree(): pts_(nullptr) { }

  OCTree(const Matrix3Xd &pts) { rebuild(pts); }

  void rebuild(const Matrix3Xd &pts);

  void find_neighbors_k(const Vector3d &pt, int k, std::vector<int> &idxs,
                        std::vector<double> &distsq) const;

  void find_neighbors_d(const Vector3d &pt, double cutoff,
                        std::vector<int> &idxs,
                        std::vector<double> &distsq) const;

  void find_neighbors_kd(const Vector3d &pt, int k, double cutoff,
                         std::vector<int> &idxs,
                         std::vector<double> &distsq) const;

  const Matrix3Xd &pts() const { return *pts_; }

  const Vector3d &max() const { return max_; }

  const Vector3d &len() const { return len_; }

  const std::vector<internal::OCTreeNode> &nodes() const { return nodes_; }

  int size() const { return static_cast<int>(nodes_.size()); }

  int root() const { return size() - 1; }

  const internal::OCTreeNode &node(int i) const { return nodes_[i]; }

  const internal::OCTreeNode &operator[](int idx) const { return nodes_[idx]; }

private:
  const Matrix3Xd *pts_;
  Vector3d max_;
  Vector3d len_;
  std::vector<internal::OCTreeNode> nodes_;
};

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

template <class MatrixLike>
auto pdistsq(const MatrixLike &m) {
  using DT = typename MatrixLike::Scalar;

  const Eigen::Index n = m.cols();
  ArrayX<DT> distsq(n * (n - 1) / 2);

  for (Eigen::Index i = 0, k = 0; i < n - 1; ++i) {
    Vector3<DT> v = m.col(i);
    for (Eigen::Index j = i + 1; j < n; ++j, ++k) {
      distsq[k] = (v - m.col(j)).squaredNorm();
    }
  }

  return distsq;
}

template <class MatrixLike>
auto pdist(const MatrixLike &m) {
  return pdistsq(m).sqrt();
}

namespace internal {
  constexpr inline double safe_normalizer(double sqn, double eps = 1e-12) {
    return sqn < eps ? 1 : 1 / std::sqrt(sqn);
  }

  template <class VectorLike>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  inline void safe_normalize(VectorLike &&m, double eps = 1e-12) {
    m.array() *= safe_normalizer(m.squaredNorm(), eps);
  }

  template <class VectorLike>
  inline auto safe_normalized(VectorLike &&m, double eps = 1e-12) {
    using T = remove_cvref_t<VectorLike>;
    using Scalar = typename T::Scalar;
    constexpr auto size = T::SizeAtCompileTime;

    Vector<Scalar, size> ret = std::forward<VectorLike>(m);
    safe_normalize(ret, eps);
    return ret;
  }

  template <class MatrixLike>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  inline void safe_colwise_normalize(MatrixLike &&m, double eps = 1e-12) {
    using T = remove_cvref_t<MatrixLike>;
    using Scalar = typename T::Scalar;

    using ArrayLike = decltype(m.colwise().norm().array());
    constexpr auto rows = ArrayLike::RowsAtCompileTime,
                   cols = ArrayLike::ColsAtCompileTime;

    Array<Scalar, rows, cols> norm = m.colwise().norm().array();
    m.array().rowwise() /= (norm < eps).select(1, norm);
  }

  template <class MatrixLike>
  inline auto safe_colwise_normalized(MatrixLike &&m, double eps = 1e-12) {
    using T = remove_cvref_t<MatrixLike>;
    using Scalar = typename T::Scalar;
    constexpr auto rows = T::RowsAtCompileTime, cols = T::ColsAtCompileTime;

    Matrix<Scalar, rows, cols> ret = std::forward<MatrixLike>(m);
    safe_colwise_normalize(ret, eps);
    return ret;
  }
}  // namespace internal

/**
 * @brief Calculate the cosine of the angle between two vectors.
 * @param oa The first vector.
 * @param ob The second vector.
 * @return The cosine of the angle between the two vectors.
 */
inline double cos_angle(const Vector3d &oa, const Vector3d &ob) {
  return oa.dot(ob)
         * internal::safe_normalizer(oa.squaredNorm() * ob.squaredNorm());
}

/**
 * @brief Calculate the cosine of the angle between two vectors.
 * @param o The origin of the angle.
 * @param a The first point.
 * @param b The second point.
 * @return The cosine of the angle between the two vectors o -> a and o -> b.
 */
inline double cos_angle(const Vector3d &o, const Vector3d &a,
                        const Vector3d &b) {
  Vector3d oa = a - o, ob = b - o;
  return cos_angle(oa, ob);
}

/**
 * @brief Calculate sin and cos of half an average angle between vectors,
 *        (0 - idxs[0]), (1 - idxs[1]), ..., (n-1 - idxs[n-1]).
 * @param m The vectors.
 * @param idxs The indices of the vectors to calculate the angle with.
 * @return Pair of (sum of sines, sum of cosines) of half an average angle.
 *         The average angle itself can be directly calculated by
 *         2 * std::atan2(sum_sin, sum_cos).
 * @warning The vectors must be of the same length.
 * @note The behavior is undefined if idxs contains out-of-range indices, or
 *       size of idxs is not equal to the number of vectors.
 * @note The returned sums are always non-negative. Thus, the result could be
 *       interpreted as if they are calculated from the angles in [0, 90]
 *       degrees. That is, the average angle is always in [0, 180] degrees.
 */
template <class Scalar, class Indexer, int N, auto... Extra>
inline std::pair<Scalar, Scalar>
sum_tan2_half(const Matrix<Scalar, 3, N, 0, Extra...> &m, const Indexer &idxs) {
  double csum = (m + m(Eigen::all, idxs)).colwise().norm().sum(),
         ssum = (m - m(Eigen::all, idxs)).colwise().norm().sum();
  return std::make_pair(ssum, csum);
}

/**
 * @brief Calculate sin and cos of half an average angle between consecutive
 *        vectors, i.e., (0, 1), (1, 2), ..., (n-1, 0).
 * @param m The vectors.
 * @return Pair of (sum of sines, sum of cosines) of half an average angle.
 *         The average angle itself can be directly calculated by
 *         2 * std::atan2(sum_sin, sum_cos).
 * @warning The vectors must be of the same length.
 * @note The returned sums are always non-negative. Thus, the result could be
 *       interpreted as if they are calculated from the angles in [0, 90]
 *       degrees. That is, the average angle is always in [0, 180] degrees.
 */
template <class Scalar, int N, auto... Extra>
inline std::pair<Scalar, Scalar>
sum_tan2_half(const Matrix<Scalar, 3, N, 0, Extra...> &m) {
  if constexpr (N == Eigen::Dynamic) {
    CyclicIndex<1> idxs(m.cols());
    return sum_tan2_half(m, idxs);
  }

  CyclicIndex<1, N> idxs;
  return sum_tan2_half(m, idxs);
}

/**
 * @brief Calculate A - (v) - B - (axis) - C - (w) - D dihedral angle along the
 *        axis. Axis should be normalized.
 * @param axis The normalized axis of the dihedral angle.
 * @param v The vector from A to B.
 * @param w The vector from C to D.
 * @return The cosine of the dihedral angle.
 *
 * See https://stackoverflow.com/a/34245697 for the implementation(s) in python.
 */
inline double cos_dihedral(const Vector3d &axis, Vector3d v, Vector3d w) {
  v -= v.dot(axis) * axis;
  w -= w.dot(axis) * axis;
  return v.dot(w)
         * internal::safe_normalizer(v.squaredNorm() * w.squaredNorm());
}

/**
 * @brief Calculate A -> - B -> C -> D dihedral angle.
 * @param a The position of point A.
 * @param b The position of point B.
 * @param c The position of point C.
 * @param d The position of point D.
 * @return The cosine of the dihedral angle.
 *
 * See https://stackoverflow.com/a/34245697 for the implementation(s) in python.
 */
inline double cos_dihedral(const Vector3d &a, const Vector3d &b,
                           const Vector3d &c, const Vector3d &d) {
  Vector3d axis = internal::safe_normalized(c - b);
  return cos_dihedral(axis, a - b, d - c);
}

/**
 * @brief Perform a least-squares fit of a plane to a set of points.
 *
 * @tparam MatrixLike The type of the matrix-like object.
 * @param pts The matrix-like object containing the points. Should be in a
 *        N x 3 shape, where N >= 3.
 * @param normalize Whether to normalize the normal vector. Defaults to true.
 * @return The best-fit plane defined by a 4-vector (a, b, c, d), such that
 *         ax + by + cz + d = 0.
 */
template <class MatrixLike>
Vector4d fit_plane(const MatrixLike &pts, bool normalize = true) {
  Vector3d cntr = pts.rowwise().mean();
  MatrixXd m = pts.colwise() - cntr;
  auto svd = m.jacobiSvd(Eigen::ComputeThinU);

  Vector4d ret;
  ret.head<3>() = svd.matrixU().col(2);
  if (normalize)
    internal::safe_normalize(ret.head<3>());
  ret[3] = -ret.head<3>().dot(cntr);
  return ret;
}
}  // namespace nuri

#endif /* NURI_CORE_GEOMETRY_H_ */

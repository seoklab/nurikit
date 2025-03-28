//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_GEOMETRY_H_
#define NURI_CORE_GEOMETRY_H_

//! @cond
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
#include "nuri/meta.h"

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
  using Points = Eigen::Map<const Matrix3Xd>;

  OCTree(): pts_(nullptr, 3, 0) { }

  template <
      class MatrixLike,
      std::enable_if_t<
          std::is_same_v<internal::remove_cvref_t<typename MatrixLike::Scalar>,
                         double>
              && !MatrixLike::IsRowMajor && MatrixLike::RowsAtCompileTime == 3
              && MatrixLike::InnerStrideAtCompileTime == 1
              && MatrixLike::OuterStrideAtCompileTime == 3,
          int> = 0>
  explicit OCTree(const MatrixLike &pts): pts_(pts.data(), 3, pts.cols()) {
    rebuild();
  }

  template <
      class MatrixLike,
      std::enable_if_t<
          std::is_same_v<internal::remove_cvref_t<typename MatrixLike::Scalar>,
                         double>
              && !MatrixLike::IsRowMajor && MatrixLike::RowsAtCompileTime == 3
              && MatrixLike::InnerStrideAtCompileTime == 1
              && MatrixLike::OuterStrideAtCompileTime == 3,
          int> = 0>
  void rebuild(const MatrixLike &pts) {
    new (&pts_) Points(pts.data(), 3, pts.cols());
    rebuild();
  }

  void find_neighbors_k(const Vector3d &pt, int k, std::vector<int> &idxs,
                        std::vector<double> &distsq) const;

  void find_neighbors_d(const Vector3d &pt, double cutoff,
                        std::vector<int> &idxs,
                        std::vector<double> &distsq) const;

  void find_neighbors_kd(const Vector3d &pt, int k, double cutoff,
                         std::vector<int> &idxs,
                         std::vector<double> &distsq) const;

  const Points &pts() const { return pts_; }

  const Vector3d &max() const { return max_; }

  const Vector3d &len() const { return len_; }

  const std::vector<internal::OCTreeNode> &nodes() const { return nodes_; }

  int size() const { return static_cast<int>(nodes_.size()); }

  int root() const { return size() - 1; }

  const internal::OCTreeNode &node(int i) const { return nodes_[i]; }

  const internal::OCTreeNode &operator[](int idx) const { return nodes_[idx]; }

private:
  void rebuild();

  Points pts_;
  Vector3d max_;
  Vector3d len_;
  std::vector<internal::OCTreeNode> nodes_;
};

namespace constants {
  constexpr double kPi =
      3.1415926535897932384626433832795028841971693993751058209749445923078164;

  constexpr double kTwoPi =
      6.2831853071795864769252867665590057683943387987502116419498891846156328;

  // NOLINTBEGIN(*-identifier-naming)
  constexpr double kCos15 =
      0.9659258262890682867497431997288973676339048390084045504023430763;
  constexpr double kCos30 =
      0.8660254037844386467637231707529361834714026269051903140279034897;
  constexpr double kCos36 =
      0.8090169943749474241022934171828190588601545899028814310677243114;
  constexpr double kCos45 =
      0.7071067811865475244008443621048490392848359376884740365883398690;
  constexpr double kCos54 =
      0.5877852522924731291687059546390727685976524376431459910722724808;
  constexpr double kCos60 = 0.5;
  constexpr double kCos75 =
      0.2588190451025207623488988376240483283490689013199305138140032073;
  constexpr double kCos100 =
      -0.173648177666930348851716626769314796000375677184069387236241378;
  constexpr double kCos102 =
      -0.207911690817759337101742284405125166216584760627723836407181973;
  constexpr double kCos112 =
      -0.374606593415912035414963774501195131000158922253676174103440371;
  constexpr double kCos115 =
      -0.422618261740699436186978489647730181563129301194864623444415159;
  constexpr double kCos125 =
      -0.573576436351046096108031912826157864620433371450986351081027118;
  constexpr double kCos155 =
      -0.906307787036649963242552656754316983267712625175864680871298408;
  constexpr double kCos175 =
      -0.996194698091745532295010402473888046183562672645850974525442277;
  constexpr double kTan10_2 =
      0.0874886635259240052220186694349614581194542763681082291452366622;
  constexpr double kTan15_2 =
      0.1316524975873958534715264574097171035928141022232375735535653257;
  constexpr double kTan116_2 =
      1.6003345290410503553267330811833575255040718469227591484115002297;
  constexpr double kTan155_2 =
      4.5107085036620571342899391172547519686713241944553043587162345185;
  // NOLINTEND(*-identifier-naming)
}  // namespace constants

template <class DT, std::enable_if_t<std::is_floating_point_v<DT>, int> = 0>
constexpr DT deg2rad(DT deg) {
  return deg * constants::kPi / 180;
}

template <class DT, std::enable_if_t<std::is_integral_v<DT>, int> = 0>
constexpr double deg2rad(DT deg) {
  return deg * constants::kPi / 180;
}

template <class DT, std::enable_if_t<std::is_floating_point_v<DT>, int> = 0>
constexpr DT rad2deg(DT rad) {
  return rad * 180 / constants::kPi;
}

template <class DT, std::enable_if_t<std::is_integral_v<DT>, int> = 0>
constexpr double rad2deg(DT rad) {
  return rad * 180 / constants::kPi;
}

template <class MatrixLike>
void pdistsq(MutRef<ArrayX<typename MatrixLike::Scalar>> distsq,
             const MatrixLike &m) {
  using DT = typename MatrixLike::Scalar;
  constexpr Eigen::Index rows = MatrixLike::RowsAtCompileTime;

  const Eigen::Index n = m.cols();
  ABSL_DCHECK(distsq.size() == n * (n - 1) / 2);

  Vector<DT, rows> v(m.rows());
  for (Eigen::Index i = 1, k = 0; i < n; ++i) {
    v = m.col(i);
    for (Eigen::Index j = 0; j < i; ++j, ++k) {
      distsq[k] = (v - m.col(j)).squaredNorm();
    }
  }
}

template <class MatrixLike>
auto pdistsq(const MatrixLike &m) {
  using DT = typename MatrixLike::Scalar;

  const Eigen::Index n = m.cols();
  ArrayX<DT> distsq(n * (n - 1) / 2);
  pdistsq(distsq, m);
  return distsq;
}

template <class MatrixLike>
void pdist(MutRef<ArrayX<typename MatrixLike::Scalar>> dist,
           const MatrixLike &m) {
  pdistsq(dist, m);
  dist = dist.sqrt();
}

template <class MatrixLike>
auto pdist(const MatrixLike &m) {
  auto ret = pdistsq(m);
  ret = ret.sqrt();
  return ret;
}

template <class ArrayLike>
void to_square_form(MutRef<MatrixX<typename ArrayLike::Scalar>> dists,
                    const ArrayLike &pdists, Eigen::Index n) {
  ABSL_DCHECK(dists.rows() == n);
  ABSL_DCHECK(dists.cols() == n);

  dists.diagonal().setZero();
  for (Eigen::Index i = 1, k = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < i; ++j, ++k)
      dists(i, j) = dists(j, i) = pdists[k];
}

template <class ArrayLike>
auto to_square_form(const ArrayLike &pdists, Eigen::Index n) {
  using DT = typename ArrayLike::Scalar;

  MatrixX<DT> dists(n, n);
  to_square_form(dists, pdists, n);
  return dists;
}

template <class ML1, class ML2,
          std::enable_if_t<
              std::is_same_v<typename ML1::Scalar, typename ML2::Scalar>
                  && internal::extract_if_enum_v(ML1::RowsAtCompileTime)
                         == internal::extract_if_enum_v(ML2::RowsAtCompileTime),
              int> = 0>
void cdistsq(MutRef<MatrixX<typename ML1::Scalar>> distsq, const ML1 &a,
             const ML2 &b) {
  ABSL_DCHECK(distsq.rows() == a.cols());
  ABSL_DCHECK(distsq.cols() == b.cols());

  for (Eigen::Index j = 0; j < b.cols(); ++j)
    distsq.col(j) = (a.colwise() - b.col(j)).colwise().squaredNorm();
}

template <class ML1, class ML2>
auto cdistsq(const ML1 &a, const ML2 &b) {
  using DT = typename ML1::Scalar;

  MatrixX<DT> distsq(a.cols(), b.cols());
  cdistsq(distsq, a, b);
  return distsq;
}

template <class ML1, class ML2>
void cdist(MutRef<MatrixX<typename ML1::Scalar>> dist, const ML1 &a,
           const ML2 &b) {
  cdistsq(dist, a, b);
  dist = dist.sqrt();
}

template <class ML1, class ML2>
auto cdist(const ML1 &a, const ML2 &b) {
  auto ret = cdistsq(a, b);
  ret = ret.sqrt();
  return ret;
}

namespace internal {
  constexpr double safe_normalizer(double sqn, double eps = 1e-12) {
    return sqn > eps ? 1 / std::sqrt(sqn) : 0;
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
    constexpr auto max_size = T::MaxSizeAtCompileTime;

    Matrix<Scalar, size, 1, 0, max_size, 1> ret = std::forward<VectorLike>(m);
    safe_normalize(ret, eps);
    return ret;
  }

  template <class MatrixLike>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  inline void safe_colwise_normalize(MatrixLike &&m, double eps = 1e-12) {
    using T = remove_cvref_t<MatrixLike>;
    using Scalar = typename T::Scalar;

    using ArrayLike = decltype(m.colwise().squaredNorm().array());
    constexpr auto cols = ArrayLike::ColsAtCompileTime;
    constexpr auto max_cols = ArrayLike::MaxColsAtCompileTime;

    Array<Scalar, 1, cols, Eigen::RowMajor, 1, max_cols> norm =
        m.colwise().squaredNorm().array();
    m.array().rowwise() *= (norm > eps).select(norm.sqrt().inverse(), 0);
  }

  template <class MatrixLike>
  inline auto safe_colwise_normalized(MatrixLike &&m, double eps = 1e-12) {
    using T = remove_cvref_t<MatrixLike>;
    using Scalar = typename T::Scalar;
    constexpr auto rows = T::RowsAtCompileTime, cols = T::ColsAtCompileTime;
    constexpr auto max_rows = T::MaxRowsAtCompileTime,
                   max_cols = T::MaxColsAtCompileTime;

    Matrix<Scalar, rows, cols, 0, max_rows, max_cols> ret =
        std::forward<MatrixLike>(m);
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

/**
 * @brief Find a vector perpendicular to the given vector.
 *
 * @tparam VectorLike The type of the vector-like object.
 * @param v A vector-like object to generate a perpendicular vector.
 * @param normalize Whether to normalize the perpendicular vector. Defaults to
 *        true.
 * @return A vector perpendicular to the given vector.
 *
 * This function is based on the algorithm proposed by K Whatmough on
 * Mathematics Stack Exchange. @cite core:geom:perpendicular-2023
 */
template <class VectorLike>
Vector3d any_perpendicular(const VectorLike &v, bool normalize = true) {
  Vector3d w = { std::copysign(v[2], v[0]),  //
                 std::copysign(v[2], v[1]),
                 -std::copysign(v[0], v[2]) - std::copysign(v[1], v[2]) };
  if (normalize)
    internal::safe_normalize(w);
  return w;
}

enum class AlignMode : std::uint8_t {
  kMsdOnly = 0x1,
  kXformOnly = 0x2,
  kBoth = kMsdOnly | kXformOnly,
};

/**
 * @brief An implementation of the Kabsch algorithm for aligning two sets of
 *        points. This algorithm is based on the implementation in TM-align.
 * @param query The query points.
 * @param templ The template points.
 * @param mode Selects the return value. Defaults to AlignMode::kBoth. Note that
 *        even if AlignMode::kXformOnly is selected, the MSD value will report a
 *        negative value if the calculation fails.
 * @param reflection Whether to allow reflection. Defaults to false.
 * @return A pair of (transformation matrix, MSD). When this function fails, MSD
 *         is set to a negative value (-1), and the state of the transformation
 *         matrix is left unspecified. This never fails when mode is
 *         AlignMode::kMsdOnly.
 *
 * This implementation has improved stability compared to the TM-align code by
 * integrating a slightly modified version of the *A Robust Eigensover for 3 x 3
 * Symmetric Matrices* algorithm proposed by D Eberly (see more details in the
 * following references) and by improving the pivot selection strategy in the B
 * matrix calculation step.
 *
 * References:
 * - D Eberly.
 *   https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
 *   (Accessed 2024-10-17)
 * - Y Zhang and J Skolnick. *Nucleic Acids Res.* **2005**, *33*, 2302-2309.
 *   DOI:[10.1093/nar/gki524](https://doi.org/10.1093/nar/gki524)
 * - W Kabsch. *Acta Crystallogr. A* **1978**, *34*, 827-828.
 *   DOI:[10.1107/S0567739478001680](https://doi.org/10.1107/S0567739478001680)
 * - W Kabsch. *Acta Crystallogr. A* **1976**, *32*, 922-923.
 *   DOI:[10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873)
 *
 * The following is the full license text for the TM-align code:
 *
 * \code{.unparsed}
 * TM-align: sequence-independent structure alignment of monomer proteins by
 * TM-score superposition. Please report issues to yangzhanglab@umich.edu
 *
 * References to cite:
 * Y Zhang, J Skolnick. Nucl Acids Res 33, 2302-9 (2005)
 *
 * DISCLAIMER:
 * Permission to use, copy, modify, and distribute the Software for any
 * purpose, with or without fee, is hereby granted, provided that the
 * notices on the head, the reference information, and this copyright
 * notice appear in all copies or substantial portions of the Software.
 * It is provided "as is" without express or implied warranty.
 * \endcode
 *
 * This is the copyright text of D Eberly's algorithm:
 *
 * \code{.unparsed}
 * David Eberly, Geometric Tools, Redmond WA 98052
 * Copyright (c) 1998-2024
 * Distributed under the Boost Software License, Version 1.0.
 * https://www.boost.org/LICENSE_1_0.txt
 * https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
 * Version: 6.0.2023.08.08
 * \endcode
 */
extern std::pair<Affine3d, double>
kabsch(const Eigen::Ref<const Matrix3Xd> &query,
       const Eigen::Ref<const Matrix3Xd> &templ,
       AlignMode mode = AlignMode::kBoth, bool reflection = false);

/**
 * @brief Perform quaternion-based superposition of two sets of points.
 * @param query The query points.
 * @param templ The template points.
 * @param mode Selects the return value. Defaults to AlignMode::kBoth. Note that
 *        even if AlignMode::kXformOnly is selected, the MSD value will report a
 *        negative value if the calculation fails.
 * @param reflection Whether to allow reflection. Defaults to false.
 * @param evalprec The precision of eigenvalue calculation. Defaults to 1e-11.
 * @param evecprec The precision of eigenvector calculation. Defaults to 1e-6.
 * @param maxiter The maximum number of Newton-Raphson iterations. Defaults
 *        to 50.
 * @return A pair of (transformation matrix, MSD). When this function fails, MSD
 *         is set to a negative value (-1), and the state of the transformation
 *         matrix is left unspecified. Unlike kabsch(), this function may fail
 *         even when mode is AlignMode::kMsdOnly due to the iterative
 *         root-finding process. Any sufficiently large value of maxiter will
 *         guarantee convergence.
 *
 * This implementation is based on the reference implementation by P Liu and DL
 * Theobald, but modified for better stability and error handling. Also, an
 * option to allow reflection is added based on observations of EA Coutsias, C
 * Seok, and KA Dill (see more details in the following references).
 *
 * References:
 * - EA Coutsias, C Seok, and KA Dill. *J. Comput. Chem.* **2004**, *25* (15),
 *   1849-1857. DOI:[10.1002/jcc.20110](https://doi.org/10.1002/jcc.20110)
 * - P Liu, DK Agrafiotis, and DL Theobald. *J. Comput. Chem.* **2011**, *32*
 *   (1), 185-186. DOI:[10.1002/jcc.21607](https://doi.org/10.1002/jcc.21607)
 * - P Liu, DK Agrafiotis, and DL Theobald. *J. Comput. Chem.* **2010**, *31*
 *   (7), 1561-1563. DOI:[10.1002/jcc.21439](https://doi.org/10.1002/jcc.21439)
 * - DL Theobald. *Acta Crystallogr. A* **2005**, *61* (4), 478-480.
 *   DOI:[10.1107/S0108767305015266](https://doi.org/10.1107/S0108767305015266)
 *
 * The following is the full license text for the reference implementation:
 *
 * \code{.unparsed}
 * Copyright (c) 2009-2016 Pu Liu and Douglas L. Theobald
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endcode
 */
extern std::pair<Affine3d, double>
qcp(const Eigen::Ref<const Matrix3Xd> &query,
    const Eigen::Ref<const Matrix3Xd> &templ, AlignMode mode = AlignMode::kBoth,
    bool reflection = false, double evalprec = 1e-11, double evecprec = 1e-6,
    int maxiter = 50);

/**
 * @brief In-place version of qcp().
 * @param query The query points. On return, the points will be centered at the
 *        origin (unless only single point is given).
 * @param templ The template points. On return, the points will be centered at
 *        the origin (unless only single point is given).
 * @param mode Selects the return value. Defaults to AlignMode::kBoth. Note that
 *        even if AlignMode::kXformOnly is selected, the MSD value will report a
 *        negative value if the calculation fails.
 * @param reflection Whether to allow reflection. Defaults to false.
 * @param evalprec The precision of eigenvalue calculation. Defaults to 1e-11.
 * @param evecprec The precision of eigenvector calculation. Defaults to 1e-6.
 * @param maxiter The maximum number of Newton-Raphson iterations. Defaults
 *        to 50.
 * @return A pair of (transformation matrix, MSD). When this function fails, MSD
 *         is set to a negative value (-1), and the state of the transformation
 *         matrix is left unspecified. Unlike kabsch(), this function may fail
 *         even when mode is AlignMode::kMsdOnly due to the iterative
 *         root-finding process. Any sufficiently large value of maxiter will
 *         guarantee convergence.
 *
 * This implementation is based on the reference implementation by P Liu and DL
 * Theobald, but modified for better stability and error handling. Also, an
 * option to allow reflection is added based on observations of EA Coutsias, C
 * Seok, and KA Dill (see more details in the following references).
 *
 * References:
 * - EA Coutsias, C Seok, and KA Dill. *J. Comput. Chem.* **2004**, *25* (15),
 *   1849-1857. DOI:[10.1002/jcc.20110](https://doi.org/10.1002/jcc.20110)
 * - P Liu, DK Agrafiotis, and DL Theobald. *J. Comput. Chem.* **2011**, *32*
 *   (1), 185-186. DOI:[10.1002/jcc.21607](https://doi.org/10.1002/jcc.21607)
 * - P Liu, DK Agrafiotis, and DL Theobald. *J. Comput. Chem.* **2010**, *31*
 *   (7), 1561-1563. DOI:[10.1002/jcc.21439](https://doi.org/10.1002/jcc.21439)
 * - DL Theobald. *Acta Crystallogr. A* **2005**, *61* (4), 478-480.
 *   DOI:[10.1107/S0108767305015266](https://doi.org/10.1107/S0108767305015266)
 *
 * The following is the full license text for the reference implementation:
 *
 * \code{.unparsed}
 * Copyright (c) 2009-2016 Pu Liu and Douglas L. Theobald
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endcode
 */
extern std::pair<Affine3d, double>
qcp_inplace(MutRef<Matrix3Xd> query, MutRef<Matrix3Xd> templ,
            AlignMode mode = AlignMode::kBoth, bool reflection = false,
            double evalprec = 1e-11, double evecprec = 1e-6, int maxiter = 50);

/**
 * @brief A routine for converting squared pairwise distances to cartesian
 *        coordinates.
 * @param pts Destination to which save the generated coordinates (3d).
 * @param dsqs The squared distances between points. Will be modified in-place.
 * @return Whether the embedding was successful.
 *
 * @note The squared distance matrix must be a N x N symmetric pairwise
 *       squared-distance matrix, where N is the number of points.
 *
 * This implementation is based on the following reference: TF Havel, ID Kuntz,
 * and GM Crippen. *Bull. Math. Biol.* **1983**, *45* (5), 665-720.
 * DOI:[10.1007/BF02460044](https://doi.org/10.1007/BF02460044)
 */
extern bool embed_distances_3d(Eigen::Ref<Matrix3Xd> pts, MatrixXd &dsqs);

/**
 * @brief A routine for converting squared pairwise distances to cartesian
 *        coordinates.
 * @param pts Destination to which save the generated coordinates (4d).
 * @param dsqs The squared distances between points. Will be modified in-place.
 * @return Whether the embedding was successful.
 *
 * @note The squared distance matrix must be a N x N symmetric pairwise
 *       squared-distance matrix, where N is the number of points.
 *
 * This implementation is based on the following reference: TF Havel, ID Kuntz,
 * and GM Crippen. *Bull. Math. Biol.* **1983**, *45* (5), 665-720.
 * DOI:[10.1007/BF02460044](https://doi.org/10.1007/BF02460044)
 */
extern bool embed_distances_4d(Eigen::Ref<Matrix4Xd> pts, MatrixXd &dsqs);
}  // namespace nuri

#endif /* NURI_CORE_GEOMETRY_H_ */

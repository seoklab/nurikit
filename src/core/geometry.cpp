//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <absl/log/absl_log.h>

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  using internal::Array8i;
  using internal::OCTreeNode;

  // Octant index is (xyz):
  // 000 -> +++, 100 -> -++, 010 -> +-+, 110 -> --+,
  // 001 -> ++-, 101 -> -+-, 011 -> +--, 111 -> ---

  const std::array<Vector3d, 8> kOctantMasks = []() {
    std::array<Vector3d, 8> masks;
    for (int i = 0; i < 8; ++i) {
      auto mask_of = [&](int axis) {
        return static_cast<double>(static_cast<bool>(i & (0b100 >> axis)));
      };

      masks[i][0] = mask_of(0);
      masks[i][1] = mask_of(1);
      masks[i][2] = mask_of(2);
    }
    return masks;
  }();

  Vector3d max_of(int octant, const Vector3d &max, const Vector3d &size) {
    return max - size.cwiseProduct(kOctantMasks[octant]);
  }

  // partition begin - end into two parts:
  // +: [begin, ret), -: [ret, end)
  int partition_sub(ArrayXi &idxs, int begin, int end, const Matrix3Xd &pts,
                    const Vector3d &cntr, int axis) {
    auto it =
        std::partition(idxs.begin() + begin, idxs.begin() + end,
                       [&](int idx) { return pts(axis, idx) >= cntr[axis]; });
    return static_cast<int>(it - idxs.begin());
  }

  Array8i partition_octant(ArrayXi &idxs, int begin, int end,
                           const Matrix3Xd &pts, const Vector3d &cntr) {
    Array8i ptrs;
    ptrs[7] = end;

    for (int i = 0; i < 3; ++i) {
      int right = begin;
      int plen = 4 >> i;
      // i == 0 -> j: 3          (plen == 4)
      // i == 1 -> j: 1, 5       (plen == 2)
      // i == 2 -> j: 0, 2, 4, 6 (plen == 1)
      for (int j = plen - 1; j < 8; j += plen << 1) {
        int left = right;
        right = ptrs[j + plen];
        ptrs[j] = partition_sub(idxs, left, right, pts, cntr, i);
      }
    }

    // Now [begin, end) is partitioned into 8 octants so that (xyz) is:
    // 000 (+++): [begin, ptrs[0])
    // 001 (++-): [ptrs[0], ptrs[1])
    // 010 (+-+): [ptrs[1], ptrs[2])
    // 011 (+--): [ptrs[2], ptrs[3])
    // 100 (-++): [ptrs[3], ptrs[4])
    // 101 (-+-): [ptrs[4], ptrs[5])
    // 110 (--+): [ptrs[5], ptrs[6])
    // 111 (---): [ptrs[6], ptrs[7] == end)
    // So the octant index matches end index of each octant.
    return ptrs;
  }

  int build_octree(const Matrix3Xd &pts, std::vector<OCTreeNode> &data,
                   const Vector3d &max, const Vector3d &size, ArrayXi &idxs,
                   const int begin, const int nleaf) {
    Array8i children;

    auto epilog = [&]() {
      int id = static_cast<int>(data.size());
      data.emplace_back(std::move(children), nleaf);
      return id;
    };

    if (nleaf <= 8) {
      children.head(nleaf) = idxs.segment(begin, nleaf);
      return epilog();
    }

    Vector3d half = size * 0.5;
    Array8i ptrs =
        partition_octant(idxs, begin, begin + nleaf, pts, max - half);

    int right = begin;
    for (int i = 0; i < 8; ++i) {
      int left = right;
      right = ptrs[i];

      int nchild = right - left;
      if (nchild <= 0) {
        children[i] = -1;
        continue;
      }

      children[i] = build_octree(pts, data, max_of(i, max, half), half, idxs,
                                 left, nchild);
    }

    return epilog();
  }
}  // namespace

void OCTree::rebuild(const Matrix3Xd &pts) {
  pts_ = &pts;
  max_ = pts.rowwise().maxCoeff();
  len_ = max_ - pts.rowwise().minCoeff();

  ArrayXi idxs(pts.cols());
  std::iota(idxs.begin(), idxs.end(), 0);

  build_octree(pts, nodes_, max_, len_, idxs, 0, static_cast<int>(pts.cols()));
}

namespace {
  using Array8d = Array<double, 8, 1>;

  struct PQEntry {
    bool leaf;
    int idx;
    Vector3d maxs;
    Vector3d size;
    double distsq;
  };

  bool pq_cmp(const PQEntry &p1, const PQEntry &p2) {
    return p1.distsq > p2.distsq;
  }

  using PQ = internal::ClearablePQ<PQEntry, decltype(&pq_cmp)>;

  int octant_idx(const Vector3d &diff) {
    Array3i isneg = (diff.array() < 0.).cast<int>();
    return (isneg.x() << 2) | (isneg.y() << 1) | (isneg.z() << 0);
  }

  Array8d octant_min_distsq(int octant, const Vector3d &ax_distsq,
                            double my_dsq = 0.0) {
    int idxs[8] = { octant,
                    // flip    x  ,             y ,              z,
                    octant ^ 0b100, octant ^ 0b010, octant ^ 0b001,
                    // flip    xy ,             yz,            x z,
                    octant ^ 0b110, octant ^ 0b011, octant ^ 0b101,
                    // flip    xyz
                    octant ^ 0b111 };

    Array8d octant_distsq;
    octant_distsq(idxs)[0] = my_dsq;
    // neighbors
    octant_distsq(idxs).segment<3>(1) = ax_distsq;
    // diagonals
    octant_distsq(idxs).segment<3>(4) = ax_distsq + ax_distsq({ 1, 2, 0 });
    // opposite
    octant_distsq(idxs)[7] = ax_distsq.sum();
    return octant_distsq;
  }

  template <class UnaryPred>
  void find_neighbors_k_impl(const OCTree &oct, const Vector3d &pt, int k,
                             std::vector<int> &idxs,
                             std::vector<double> &distsq, UnaryPred pred) {
    idxs.clear();
    distsq.clear();
    if (k <= 0) {
      ABSL_LOG(WARNING) << "k is not a positive number: " << k;
      return;
    }
    if (k > oct.pts().cols()) {
      ABSL_LOG(WARNING) << "k " << k << " is larger than the number of points "
                        << oct.pts().cols();
    }

    PQ minheap(pq_cmp);
    minheap.push({ false, oct.root(), oct.max(), oct.len(), 0 });

    while (!minheap.empty() && k > 0) {
      auto [leaf, idx, maxs, size, dsq] = minheap.pop_get();
      if (leaf) {
        idxs.push_back(idx);
        distsq.push_back(dsq);
        --k;
        continue;
      }

      const OCTreeNode &node = oct[idx];
      if (node.leaf()) {
        for (int i = 0; i < node.nleaf(); ++i) {
          int cid = node[i];
          double d = (pt - oct.pts().col(cid)).squaredNorm();
          if (pred(d))
            minheap.push({ true, cid, maxs, size, d });
        }
        continue;
      }

      Vector3d half = size * 0.5;
      Vector3d cntr_diff = pt - maxs + half;
      int octant = octant_idx(cntr_diff);
      Array8d octant_distsq =
          octant_min_distsq(octant, cntr_diff.cwiseAbs2(), dsq);

      for (int i = 0; i < 8; ++i) {
        int cid = node[i];
        if (cid < 0 || !pred(octant_distsq[i]))
          continue;

        minheap.push(
            { false, cid, max_of(i, maxs, half), half, octant_distsq[i] });
      }
    }
  }
}  // namespace

void OCTree::find_neighbors_k(const Vector3d &pt, const int k,
                              std::vector<int> &idxs,
                              std::vector<double> &distsq) const {
  find_neighbors_k_impl(*this, pt, k, idxs, distsq,
                        [](double /* dsq */) { return true; });
}

void OCTree::find_neighbors_kd(const Vector3d &pt, const int k,
                               const double cutoff, std::vector<int> &idxs,
                               std::vector<double> &distsq) const {
  find_neighbors_k_impl(*this, pt, k, idxs, distsq,
                        [cutoffsq = cutoff * cutoff](double dsq) {
                          return dsq <= cutoffsq;
                        });
}

namespace {
  void find_candidates_d(const OCTree &oct, const Vector3d &pt,
                         const double cutoffsq, std::vector<int> &idxs,
                         const OCTreeNode &node, const Vector3d &maxs,
                         const Vector3d &size) {
    if (node.leaf()) {
      for (int i = 0; i < node.nleaf(); ++i)
        idxs.push_back(node[i]);

      return;
    }

    Vector3d half = size * 0.5;
    Vector3d cntr_diff = pt - maxs + half;
    int octant = octant_idx(cntr_diff);
    Array8d octant_distsq = octant_min_distsq(octant, cntr_diff.cwiseAbs2());

    for (int i = 0; i < 8; ++i) {
      int cid = node[i];
      if (octant_distsq[i] > cutoffsq || cid < 0)
        continue;

      find_candidates_d(oct, pt, cutoffsq, idxs, oct[cid],
                        max_of(i, maxs, half), half);
    }
  }
}  // namespace

void OCTree::find_neighbors_d(const Vector3d &pt, const double cutoff,
                              std::vector<int> &idxs,
                              std::vector<double> &distsq) const {
  const double cutoffsq = cutoff * cutoff;

  idxs.clear();
  find_candidates_d(*this, pt, cutoffsq, idxs, node(root()), max_, len_);

  distsq.clear();
  erase_if(idxs, [&](int idx) {
    double d = (pts_->col(idx) - pt).squaredNorm();
    bool far = d > cutoffsq;
    if (!far)
      distsq.push_back(d);
    return far;
  });
}

// NOLINTBEGIN(readability-identifier-naming,*-avoid-goto)
/*
 * TMalign license text:
 *
 *   Permission to use, copy, modify, and distribute the Software for any
 *   purpose, with or without fee, is hereby granted, provided that the
 *   notices on the head, the reference information, and this copyright
 *   notice appear in all copies or substantial portions of the Software.
 *   It is provided "as is" without express or implied warranty.
 */
namespace {
  using Array6d = Array<double, 6, 1>;
  using Array9d = Array<double, 9, 1>;

  constexpr double
      kSqrt3 =
          1.7320508075688772935274463415058723669428052538103806280558069794,
      kTol = 1e-2, kEps = 1e-8, kEps2 = 1e-16;
  constexpr int kIp[] = { 0, 1, 3, 1, 2, 4, 3, 4, 5 },
                kIp2312[] = { 1, 2, 0, 1 };

  bool kabsch_prepare_e(Array3d &e, const Array9d &rr, const double spur,
                        const double det) {
    double cof =
        (((((rr[2] * rr[5] - rr[8]) + rr[0] * rr[5]) - rr[7]) + rr[0] * rr[2])
         - rr[6])
        / 3.0;

    double h = spur * spur - cof;
    if (h <= 0)
      return true;

    double g = (spur * cof - det * det) / 2.0 - spur * h;
    double d = h * h * h - g * g;
    d = std::atan2(std::sqrt(nonnegative(d)), -g) / 3.0;

    double sqrth = std::sqrt(h);
    double cth = sqrth * std::cos(d);
    double sth = sqrth * kSqrt3 * std::sin(d);

    e[0] += cth + cth;
    e[1] += -cth + sth;
    e[2] += -cth - sth;

    return false;
  }

  double kabsch_calculate_msd(const Eigen::Ref<const Matrix3Xd> &query,
                              const Eigen::Ref<const Matrix3Xd> &templ,
                              const Vector3d &qm, const Vector3d &tm, Array3d e,
                              const double det) {
    double sd = (query.colwise() - qm).cwiseAbs2().sum()
                + (templ.colwise() - tm).cwiseAbs2().sum();
    e = e.cwiseMax(0).sqrt();
    double d = e[0] + e[1] + std::copysign(e[2], det);

    sd -= d * 2;
    sd = nonnegative(sd);
    return sd / static_cast<double>(templ.cols());
  }

  bool kabsch_calculate_A(Matrix3d &A, const Array9d &rr, const Array3d &e) {
    Array6d ss, ss_sq;

    for (const int col: { 0, 2 }) {
      double ei = e[col];
      ss[0] = (ei - rr[2]) * (ei - rr[5]) - rr[8];
      ss[1] = (ei - rr[5]) * rr[1] + rr[3] * rr[4];
      ss[2] = (ei - rr[0]) * (ei - rr[5]) - rr[7];
      ss[3] = (ei - rr[2]) * rr[3] + rr[1] * rr[4];
      ss[4] = (ei - rr[0]) * rr[4] + rr[1] * rr[3];
      ss[5] = (ei - rr[0]) * (ei - rr[2]) - rr[6];

      ss_sq = ss.square();
      ss_sq = (ss_sq <= kEps2).select(0, ss_sq);
      int idx;
      if (ss_sq[0] >= ss_sq[2]) {
        idx = 0;
        if (ss_sq[0] < ss_sq[5])
          idx = 2;
      } else if (ss_sq[2] >= ss_sq[5]) {
        idx = 1;
      } else {
        idx = 2;
      }

      double ss_sum = 0.0;
      idx *= 3;
      for (int i = 0; i < 3; i++) {
        int k = kIp[i + idx];
        A(i, col) = ss[k];
        ss_sum += ss_sq[k];
      }

      if (ss_sum > kEps) {
        ss_sum = 1.0 / std::sqrt(ss_sum);
      } else {
        ss_sum = 0.0;
      }

      A.col(col) *= ss_sum;
    }

    int c1, c;
    if ((e[0] - e[1]) > (e[1] - e[2])) {
      c1 = 2;
      c = 0;
    } else {
      c1 = 0;
      c = 2;
    }

    double d = A.col(0).dot(A.col(2));
    A.col(c1) = A.col(c1) - d * A.col(c);

    double p = A.col(c1).squaredNorm();
    if (p <= kTol) {
      p = 1.0;
      Array3d a_abs = A.col(c).array().abs();
      int idx = 0;
      for (int i = 0; i < 3; i++) {
        if (p < a_abs[i])
          continue;
        p = a_abs[i];
        idx = i;
      }

      int l = kIp2312[idx];
      int m = kIp2312[idx + 1];
      p = std::sqrt(A(l, c) * A(l, c) + A(m, c) * A(m, c));
      if (p <= kTol)
        return false;

      A(idx, c1) = 0.0;
      A(l, c1) = -A(m, c) / p;
      A(m, c1) = A(l, c) / p;
    } else {
      p = 1.0 / std::sqrt(p);
      A.col(c1) *= p;
    }

    A.col(1) = A.col(2).cross(A.col(0));

    return true;
  }

  bool kabsch_calculate_B(Matrix3d &B, const Matrix3d &A, const Matrix3d &R) {
    for (const int col: { 0, 1 }) {
      B.col(col) = R * A.col(col);

      double d = B.col(col).squaredNorm();
      if (d > kEps)
        d = 1.0 / std::sqrt(d);
      else
        d = 0.0;

      B.col(col) *= d;
    }

    double d = B.col(0).dot(B.col(1));
    B.col(1) = B.col(1) - d * B.col(0);

    double p = B.col(1).squaredNorm();
    if (p <= kTol) {
      p = 1.0;
      Array3d b_abs = B.col(0).array().abs();
      int idx = 0;
      for (int i = 0; i < 3; i++) {
        if (p < b_abs[i])
          continue;

        p = b_abs[i];
        idx = i;
      }

      int k = kIp2312[idx];
      int l = kIp2312[idx + 1];
      p = std::sqrt(B(k, 0) * B(k, 0) + B(l, 0) * B(l, 0));
      if (p <= kTol)
        return false;

      B(idx, 1) = 0.0;
      B(k, 1) = -B(l, 0) / p;
      B(l, 1) = B(k, 0) / p;
    } else {
      p = 1.0 / std::sqrt(p);
      B.col(1) *= p;
    }

    B.col(2) = B.col(0).cross(B.col(1));

    return true;
  }
}  // namespace

std::pair<Affine3d, double> kabsch(const Eigen::Ref<const Matrix3Xd> &query,
                                   const Eigen::Ref<const Matrix3Xd> &templ,
                                   KabschMode mode) {
  std::pair<Affine3d, double> ret { {}, 0.0 };

  Vector3d qs = query.rowwise().sum();
  Vector3d qm = qs.array() / query.cols(), tm = templ.rowwise().mean();

  NURI_EIGEN_TMP(Matrix3d) s = templ * query.transpose();
  Matrix3d R = s - tm * qs.transpose();

  // first 6: lower triangle of R^T * R
  //  last 3: squared off-diagonal elements of R^T * R
  Array9d rr;
  for (int i = 0, k = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j, ++k)
      rr[k] = R.col(i).dot(R.col(j));
  rr.tail<3>() = rr({ 1, 3, 4 }).square();

  double spur = (rr[0] + rr[2] + rr[5]) / 3.0;
  Array3d e = Array3d::Constant(spur);
  const double det = R.determinant();

  if (ABSL_PREDICT_TRUE(spur > 0)) {
    const bool A_ident = kabsch_prepare_e(e, rr, spur, det);

    if (mode != KabschMode::kXformOnly)
      ret.second = kabsch_calculate_msd(query, templ, qm, tm, e, det);

    if (mode == KabschMode::kMsdOnly)
      return ret;

    Matrix3d A = Matrix3d::Identity();
    if (!A_ident && ABSL_PREDICT_FALSE(!kabsch_calculate_A(A, rr, e)))
      goto failure;

    Matrix3d B = Matrix3d::Zero();
    if (ABSL_PREDICT_FALSE(!kabsch_calculate_B(B, A, R)))
      goto failure;

    ret.first.linear() = B * A.transpose();
    ret.first.translation().noalias() = tm - ret.first.linear() * qm;
    return ret;
  }

  if (mode == KabschMode::kMsdOnly) {
    ret.second = kabsch_calculate_msd(query, templ, qm, tm, e, det);
    return ret;
  }

failure:
  ret.second = -1.0;
  return ret;
}
// NOLINTEND(readability-identifier-naming,*-avoid-goto)
}  // namespace nuri

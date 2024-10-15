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
namespace {
  using Array6d = Array<double, 6, 1>;
  using Array9d = Array<double, 9, 1>;

  constexpr double
      kSqrt3 =
          1.7320508075688772935274463415058723669428052538103806280558069794,
      kTol = 1e-2, kEps = 1e-8, kEps2 = 1e-16;
  constexpr int kIp[3][3] = {
    {0, 1, 3},
    {1, 2, 4},
    {3, 4, 5},
  };
  constexpr int kIp2312[] = { 1, 2, 0, 1 };

  bool kabsch_calculate_eigs(Array3d &eigs, const Matrix3d &RtR,
                             const double spur, const double det) {
    const double cof =
        (RtR(0, 0) * RtR(1, 1) - RtR(0, 1) + RtR(0, 0) * RtR(2, 2) - RtR(0, 2)
         + RtR(1, 1) * RtR(2, 2) - RtR(1, 2))
        / 3;

    double h = spur * spur - cof;
    if (h <= 0)
      return true;

    double g = (spur * cof - det * det) / 2.0 - spur * h;
    double d = h * h * h - g * g;
    d = std::atan2(std::sqrt(nonnegative(d)), -g) / 3.0;

    double sqrth = std::sqrt(h);
    double cth = sqrth * std::cos(d);
    double sth = sqrth * kSqrt3 * std::sin(d);

    eigs[0] += cth + cth;
    eigs[1] += -cth + sth;
    eigs[2] += -cth - sth;

    return false;
  }

  double kabsch_calculate_msd(const Eigen::Ref<const Matrix3Xd> &query,
                              const Eigen::Ref<const Matrix3Xd> &templ,
                              const Vector3d &qm, const Vector3d &tm, Array3d e,
                              const double det, const bool reflection) {
    double sd = (query.colwise() - qm).cwiseAbs2().sum()
                + (templ.colwise() - tm).cwiseAbs2().sum();
    e = e.max(0).sqrt();

    double d;
    if (reflection) {
      d = e.sum();
    } else {
      d = e[0] + e[1] + std::copysign(e[2], det);
    }

    sd -= d * 2;
    sd = nonnegative(sd);
    return sd / static_cast<double>(templ.cols());
  }

  bool safe_gram_schmidt(Matrix3d &m, const int pivot, const int axis) {
    const double d = m.col(pivot).dot(m.col(axis));
    m.col(pivot) -= d * m.col(axis);

    const double p = m.col(pivot).squaredNorm();
    if (p > kTol) {
      m.col(pivot) /= std::sqrt(p);
    } else {
      Array3d axis_sq = m.col(axis).array().square();
      int rmin;
      axis_sq.minCoeff(&rmin);

      const int k = kIp2312[rmin], l = kIp2312[rmin + 1];
      const double q = std::sqrt(axis_sq[k] + axis_sq[l]);
      if (q <= kTol)
        return false;

      m(rmin, pivot) = 0.0;
      m(k, pivot) = -m(l, axis) / q;
      m(l, pivot) = m(k, axis) / q;
    }

    return true;
  }

  bool kabsch_form_At(Matrix3d &At, const Matrix3d &RtR, const Array3d &eigs) {
    // Cofactor matrix of R^T * R (lower triangular part)
    Array6d ss, ss_sq;

    for (const int col: { 0, 2 }) {
      double ei = eigs[col];
      ss[0] = (ei - RtR(1, 1)) * (ei - RtR(2, 2)) - RtR(1, 2);
      ss[1] = (ei - RtR(2, 2)) * RtR(1, 0) + RtR(2, 0) * RtR(2, 1);
      ss[2] = (ei - RtR(0, 0)) * (ei - RtR(2, 2)) - RtR(0, 2);
      ss[3] = (ei - RtR(1, 1)) * RtR(2, 0) + RtR(1, 0) * RtR(2, 1);
      ss[4] = (ei - RtR(0, 0)) * RtR(2, 1) + RtR(1, 0) * RtR(2, 0);
      ss[5] = (ei - RtR(0, 0)) * (ei - RtR(1, 1)) - RtR(0, 1);

      ss_sq = ss.square();
      ss_sq = (ss_sq > kEps2).select(ss_sq, 0);

      int rmax;
      ss_sq({ 0, 2, 5 }).maxCoeff(&rmax);

      const double normalizer =
          internal::safe_normalizer(ss_sq(kIp[rmax]).sum(), kEps);
      At.col(col) = normalizer * ss(kIp[rmax]);
    }

    int c1, c;
    if ((eigs[0] - eigs[1]) > (eigs[1] - eigs[2])) {
      c1 = 2;
      c = 0;
    } else {
      c1 = 0;
      c = 2;
    }
    if (!safe_gram_schmidt(At, c1, c))
      return false;

    At.col(1) = At.col(2).cross(At.col(0));
    return true;
  }

  bool kabsch_form_Bt(Matrix3d &Bt, const Matrix3d &At, const Matrix3d &R) {
    Bt.leftCols<2>().noalias() = R * At.leftCols<2>();
    internal::safe_colwise_normalize(Bt.leftCols<2>(), kEps);

    if (!safe_gram_schmidt(Bt, 1, 0))
      return false;

    Bt.col(2) = Bt.col(0).cross(Bt.col(1));
    return true;
  }
}  // namespace

std::pair<Affine3d, double> kabsch(const Eigen::Ref<const Matrix3Xd> &query,
                                   const Eigen::Ref<const Matrix3Xd> &templ,
                                   KabschMode mode, const bool reflection) {
  std::pair<Affine3d, double> ret { {}, 0.0 };

  Vector3d qs = query.rowwise().sum();
  Vector3d qm = qs.array() / query.cols(), tm = templ.rowwise().mean();

  Matrix3d R = templ * query.transpose();
  R.noalias() -= tm * qs.transpose();
  const double det = R.determinant();

  Matrix3d RtR;
  RtR.triangularView<Eigen::Lower>() = R.transpose() * R;
  RtR.triangularView<Eigen::StrictlyUpper>() = RtR.transpose().cwiseAbs2();
  const double spur = RtR.trace() / 3;

  Array3d eigs = Array3d::Constant(spur);

  if (ABSL_PREDICT_TRUE(spur > 0)) {
    const bool A_ident = kabsch_calculate_eigs(eigs, RtR, spur, det);

    if (mode != KabschMode::kXformOnly) {
      ret.second =
          kabsch_calculate_msd(query, templ, qm, tm, eigs, det, reflection);
    }

    if (mode == KabschMode::kMsdOnly)
      return ret;

    Matrix3d At = Matrix3d::Identity();
    if (!A_ident && ABSL_PREDICT_FALSE(!kabsch_form_At(At, RtR, eigs)))
      goto failure;

    Matrix3d Bt = Matrix3d::Zero();
    if (ABSL_PREDICT_FALSE(!kabsch_form_Bt(Bt, At, R)))
      goto failure;

    if (reflection && det < 0)
      Bt.col(2) *= -1;

    ret.first.linear() = Bt * At.transpose();
    ret.first.translation().noalias() = -(ret.first.linear() * qm);
    ret.first.translation() += tm;
    return ret;
  }

  if (mode == KabschMode::kMsdOnly) {
    ret.second =
        kabsch_calculate_msd(query, templ, qm, tm, eigs, det, reflection);
    return ret;
  }

failure:
  ret.second = -1.0;
  return ret;
}
// NOLINTEND(readability-identifier-naming,*-avoid-goto)
}  // namespace nuri

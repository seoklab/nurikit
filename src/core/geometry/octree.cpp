//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
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
  int partition_sub(ArrayXi &idxs, int begin, int end,
                    const OCTree::Points &pts, const Vector3d &cntr, int axis) {
    auto it =
        std::partition(idxs.begin() + begin, idxs.begin() + end,
                       [&](int idx) { return pts(axis, idx) >= cntr[axis]; });
    return static_cast<int>(it - idxs.begin());
  }

  Array8i partition_octant(ArrayXi &idxs, int begin, int end,
                           const OCTree::Points &pts, const Vector3d &cntr) {
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

  int build_octree(const OCTree::Points &pts, std::vector<OCTreeNode> &data,
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

void OCTree::rebuild() {
  max_ = pts_.rowwise().maxCoeff();
  len_ = max_ - pts_.rowwise().minCoeff();

  ArrayXi idxs(pts_.cols());
  std::iota(idxs.begin(), idxs.end(), 0);

  internal::AllowEigenMallocScoped<false> ems;

  build_octree(pts_, nodes_, max_, len_, idxs, 0,
               static_cast<int>(pts_.cols()));
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
    double d = (pts_.col(idx) - pt).squaredNorm();
    bool far = d > cutoffsq;
    if (!far)
      distsq.push_back(d);
    return far;
  });
}
}  // namespace nuri

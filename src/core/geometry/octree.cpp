//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/container/container_ext.h"
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

  int fill_octree_bucket(const OCTree::Points &pts,
                         std::vector<OCTreeNode> &data, ArrayXi &idxs,
                         int begin, const int nleaf) {
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

    children.fill(-1);
    int remaining = nleaf;
    for (int i = 0; i < 8 && remaining > 0; ++i, begin += 8, remaining -= 8) {
      children[i] =
          fill_octree_bucket(pts, data, idxs, begin, nuri::min(8, remaining));
    }
    ABSL_DCHECK_LE(remaining, 0);
    return epilog();
  }

  int build_octree(const OCTree::Points &pts, std::vector<OCTreeNode> &data,
                   const Vector3d &max, const Vector3d &size, ArrayXi &idxs,
                   const int begin, const int nleaf, const int bucket_size) {
    if (nleaf <= bucket_size)
      return fill_octree_bucket(pts, data, idxs, begin, nleaf);

    Vector3d half = size * 0.5;
    Array8i ptrs =
        partition_octant(idxs, begin, begin + nleaf, pts, max - half);

    Array8i children;
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
                                 left, nchild, bucket_size);
    }

    int id = static_cast<int>(data.size());
    data.emplace_back(std::move(children), nleaf);
    return id;
  }
}  // namespace

void OCTree::rebuild() {
  ABSL_DCHECK_GE(bucket_size_, 8);
  ABSL_DCHECK_LE(bucket_size_, 8 * 8);

  max_ = pts_.rowwise().maxCoeff();
  len_ = max_ - pts_.rowwise().minCoeff();

  ArrayXi idxs(pts_.cols());
  std::iota(idxs.begin(), idxs.end(), 0);

  internal::AllowEigenMallocScoped<false> ems;

  build_octree(pts_, nodes_, max_, len_, idxs, 0, static_cast<int>(pts_.cols()),
               bucket_size_);
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
      if (node.nleaf() <= oct.bucket_size()) {
        int li = node.leaf() ? idx : node[0];

        for (int i = 0, remaining = node.nleaf(); remaining > 0;) {
          const OCTreeNode &ln = oct[li];
          ABSL_DCHECK(ln.leaf());

          for (int j = 0; j < ln.nleaf(); ++j) {
            int cid = ln[j];
            double d = (pt - oct.pts().col(cid)).squaredNorm();
            if (pred(d))
              minheap.push({ true, cid, maxs, size, d });
          }

          remaining -= ln.nleaf();
          if (remaining > 0)
            li = node[++i];
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
    if (node.nleaf() <= oct.bucket_size()) {
      const OCTreeNode *ln = node.leaf() ? &node : &oct[node[0]];

      for (int i = 0, rem = node.nleaf(); rem > 0;) {
        ABSL_DCHECK(ln->leaf());

        idxs.insert(idxs.end(), ln->children().begin(),
                    ln->children().begin() + ln->nleaf());

        rem -= ln->nleaf();
        if (rem > 0)
          ln = &oct[node[++i]];
      }

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

namespace {
  class OCTreeBox {
  public:
    explicit OCTreeBox(const OCTree &parent)
        : OCTreeBox(parent, parent.root(), parent.max(), parent.len()) { }

    OCTreeBox(const OCTree &parent, int nid, const Vector3d &max,
              const Vector3d &len)
        : parent_(&parent), nid_(nid), max_(max), len_(len) { }

    std::pair<double, double> minmax_distsq(const OCTreeBox &other) const {
      Vector3d self_min = max_ - len_, other_min = other.max_ - other.len_;
      Vector3d minmax = self_min - other.max_, maxmin = other_min - max_;

      double min_dsq = minmax.cwiseMax(0.0).squaredNorm()
                       + maxmin.cwiseMax(0.0).squaredNorm();
      double max_dsq =
          minmax.cwiseAbs().cwiseMax(maxmin.cwiseAbs()).squaredNorm();

      return { min_dsq, max_dsq };
    }

    bool has_child(int octant) const { return node()[octant] >= 0; }

    OCTreeBox child(int octant) const {
      Vector3d half = len_ * 0.5;
      return OCTreeBox(*parent_, node()[octant], max_of(octant, max_, half),
                       half);
    }

    int id() const { return nid_; }

    const OCTree &tree() const { return *parent_; }
    const OCTreeNode &node() const { return parent_->node(nid_); }

  private:
    internal::Nonnull<const OCTree *> parent_;
    int nid_;
    Vector3d max_;
    Vector3d len_;
  };

  void fill_neighbors_tree(std::vector<std::vector<int>> &idxs,
                           const OCTree &self, const int lp,
                           const OCTree &other, const int rp) {
    const OCTreeNode &ln = self[lp];
    const OCTreeNode &rn = other[rp];

    if (!ln.leaf()) {
      for (int i = 0; i < 8; ++i)
        if (int lc = ln[i]; lc >= 0)
          fill_neighbors_tree(idxs, self, lc, other, rp);

      return;
    }

    if (!rn.leaf()) {
      for (int i = 0; i < 8; ++i)
        if (int rc = rn[i]; rc >= 0)
          fill_neighbors_tree(idxs, self, lp, other, rc);

      return;
    }

    for (int i = 0; i < ln.nleaf(); ++i) {
      std::vector<int> &neighbors = idxs[ln[i]];
      neighbors.insert(neighbors.end(), rn.children().begin(),
                       rn.children().begin() + rn.nleaf());
    }
  }

  void find_neighbors_tree_bucket(std::vector<std::vector<int>> &idxs,
                                  const OCTreeBox &self, const OCTreeBox &other,
                                  const OCTreeNode &self_node,
                                  const OCTreeNode &other_node,
                                  const double cutoffsq) {
    const OCTreeNode *ln = self_node.leaf() ? &self_node
                                            : &self.tree()[self_node[0]];

    for (int i = 0, lrem = self_node.nleaf(); lrem > 0;) {
      ABSL_DCHECK(ln->leaf());

      const OCTreeNode *rn = other_node.leaf() ? &other_node
                                               : &other.tree()[other_node[0]];
      for (int j = 0, rrem = other_node.nleaf(); rrem > 0;) {
        ABSL_DCHECK(rn->leaf());

        for (int k = 0; k < ln->nleaf(); ++k) {
          int lk = (*ln)[k];
          Vector3d lpt = self.tree().pts().col(lk);

          for (int l = 0; l < rn->nleaf(); ++l) {
            int rl = (*rn)[l];

            if ((lpt - other.tree().pts().col(rl)).squaredNorm() <= cutoffsq)
              idxs[lk].push_back(rl);
          }
        }

        rrem -= rn->nleaf();
        if (rrem > 0)
          rn = &other.tree()[other_node[++j]];
      }

      lrem -= ln->nleaf();
      if (lrem > 0)
        ln = &self.tree()[self_node[++i]];
    }
  }

  void find_neighbors_tree_impl(std::vector<std::vector<int>> &idxs,
                                const OCTreeBox &self, const OCTreeBox &other,
                                const double cutoffsq) {
    auto [min_dsq, max_dsq] = self.minmax_distsq(other);

    if (min_dsq > cutoffsq)
      return;
    if (max_dsq <= cutoffsq) {
      fill_neighbors_tree(idxs, self.tree(), self.id(), other.tree(),
                          other.id());
      return;
    }

    const OCTreeNode &ln = self.node();
    const OCTreeNode &rn = other.node();
    if (ln.nleaf() <= self.tree().bucket_size()
        && rn.nleaf() <= other.tree().bucket_size()) {
      find_neighbors_tree_bucket(idxs, self, other, ln, rn, cutoffsq);
      return;
    }

    if (ln.nleaf() <= self.tree().bucket_size()) {
      for (int j = 0; j < 8; ++j) {
        if (!other.has_child(j))
          continue;

        find_neighbors_tree_impl(idxs, self, other.child(j), cutoffsq);
      }
      return;
    }

    if (rn.nleaf() <= other.tree().bucket_size()) {
      for (int i = 0; i < 8; ++i) {
        if (!self.has_child(i))
          continue;

        find_neighbors_tree_impl(idxs, self.child(i), other, cutoffsq);
      }
      return;
    }

    for (int i = 0; i < 8; ++i) {
      if (!self.has_child(i))
        continue;

      OCTreeBox lc = self.child(i);
      for (int j = 0; j < 8; ++j) {
        if (!other.has_child(j))
          continue;

        find_neighbors_tree_impl(idxs, lc, other.child(j), cutoffsq);
      }
    }
  }
}  // namespace

void OCTree::find_neighbors_tree(const OCTree &oct, const double cutoff,
                                 std::vector<std::vector<int>> &idxs) const {
  const double cutoffsq = cutoff * cutoff;

  idxs.resize(pts().cols());
  find_neighbors_tree_impl(idxs, OCTreeBox(*this), OCTreeBox(oct), cutoffsq);
}
}  // namespace nuri

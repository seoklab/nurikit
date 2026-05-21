//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/types/span.h>
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

  int build_octree(const OCTree::Points &pts, std::vector<OCTreeNode> &data,
                   const Vector3d &max, const Vector3d &size, ArrayXi &idxs,
                   const int begin, const int nleaf, const int bucket_size) {
    Array8i children;
    auto epilog = [&]() {
      int id = static_cast<int>(data.size());
      data.emplace_back(std::move(children), begin, nleaf);
      return id;
    };

    if (nleaf <= bucket_size) {
      std::sort(idxs.begin() + begin, idxs.begin() + begin + nleaf);
      if (nleaf <= 8)
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
                                 left, nchild, bucket_size);
    }

    return epilog();
  }
}  // namespace

void OCTree::rebuild() {
  bucket_size_ = std::max(8, bucket_size_);

  max_ = pts_.rowwise().maxCoeff();
  len_ = max_ - pts_.rowwise().minCoeff();

  idxs_.resize(pts_.cols());
  std::iota(idxs_.begin(), idxs_.end(), 0);

  internal::AllowEigenMallocScoped<false> ems;

  build_octree(pts_, nodes_, max_, len_, idxs_, 0,
               static_cast<int>(pts_.cols()), bucket_size_);
}

namespace {
  using Array8d = Array<double, 8, 1>;

  struct NodeEntry {
    int idx;
    Vector3d maxs;
    Vector3d size;
    double min_dsq;
  };

  bool node_cmp(const NodeEntry &a, const NodeEntry &b) {
    return a.min_dsq > b.min_dsq;
  }

  using NodeHeap = internal::ClearablePQ<NodeEntry, decltype(&node_cmp)>;

  struct CandEntry {
    double dsq;
    int idx;
  };

  bool cand_cmp(const CandEntry &a, const CandEntry &b) {
    return a.dsq < b.dsq;
  }

  using CandHeap = internal::ClearablePQ<CandEntry, decltype(&cand_cmp)>;

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

  bool leaf_node(const OCTree &tree, const OCTreeNode &node) {
    return node.nleaf() <= tree.bucket_size();
  }

  absl::Span<const int> node_points(const OCTree &tree,
                                    const OCTreeNode &node) {
    const int *p = node.leaf() ? node.children().data()
                               : tree.idxs().data() + node.begin();
    return absl::MakeConstSpan(p, node.nleaf());
  }

  double maxsq_box_point(const Vector3d &box_max, const Vector3d &box_len,
                         const Vector3d &pt) {
    Vector3d box_min = box_max - box_len;
    Vector3d minmax = box_min - pt, maxmin = pt - box_max;
    double max_dsq =
        minmax.cwiseAbs().cwiseMax(maxmin.cwiseAbs()).squaredNorm();
    return max_dsq;
  }
}  // namespace

void OCTree::find_neighbors_kd(const Vector3d &pt, const int k,
                               std::vector<int> &idxs,
                               std::vector<double> &distsq,
                               double cutoff) const {
  idxs.clear();
  distsq.clear();
  if (k <= 0) {
    ABSL_LOG(WARNING) << "k is not a positive number: " << k;
    return;
  }
  if (k > pts().cols()) {
    ABSL_LOG(WARNING)
        << "k " << k << " is larger than the number of points " << pts().cols();
  }

  double worst_dsq;
  if (cutoff > 0) {
    worst_dsq = cutoff * cutoff;
  } else {
    worst_dsq = maxsq_box_point(max_, len_, pt);
  }

  NodeHeap node_heap(node_cmp);
  CandHeap best_heap(cand_cmp);
  node_heap.push({ root(), max(), len(), 0 });

  while (!node_heap.empty()) {
    auto [idx, maxs, size, min_dsq] = node_heap.pop_get();
    // Nodes are popped in order of increasing min_dsq, so once min_dsq
    // exceeds worst_dsq we cannot improve the answer anymore.
    if (min_dsq > worst_dsq)
      break;

    const OCTreeNode &node = nodes()[idx];
    if (leaf_node(*this, node)) {
      for (const int i: node_points(*this, node)) {
        double d = (pt - pts().col(i)).squaredNorm();
        if (d > worst_dsq)
          continue;

        best_heap.push({ d, i });
        if (static_cast<int>(best_heap.size()) > k) {
          best_heap.pop();
          worst_dsq = best_heap.top().dsq;
        }
      }
      continue;
    }

    Vector3d half = size * 0.5;
    Vector3d cntr_diff = pt - maxs + half;
    int octant = octant_idx(cntr_diff);
    Array8d octant_distsq =
        octant_min_distsq(octant, cntr_diff.cwiseAbs2(), min_dsq);

    for (int i = 0; i < 8; ++i) {
      int cid = node[i];
      if (cid < 0 || octant_distsq[i] > worst_dsq)
        continue;

      node_heap.push({ cid, max_of(i, maxs, half), half, octant_distsq[i] });
    }
  }

  const int n = static_cast<int>(best_heap.size());
  idxs.resize(n);
  distsq.resize(n);
  for (int i = n - 1; i >= 0; --i) {
    auto e = best_heap.pop_get();
    idxs[i] = e.idx;
    distsq[i] = e.dsq;
  }
}

namespace {
  void find_candidates_d(const OCTree &oct, const Vector3d &pt,
                         const double cutoffsq, std::vector<int> &idxs,
                         const OCTreeNode &node, const Vector3d &maxs,
                         const Vector3d &size) {
    if (leaf_node(oct, node)) {
      auto pts = node_points(oct, node);
      idxs.insert(idxs.end(), pts.begin(), pts.end());
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

    bool leaf_node() const { return nuri::leaf_node(tree(), node()); }

    absl::Span<const int> idxs() const {
      ABSL_DCHECK_GE(id(), 0);
      return node_points(tree(), node());
    }

    int id() const { return nid_; }

    const OCTree &tree() const { return *parent_; }
    const OCTreeNode &node() const { return parent_->node(nid_); }
    decltype(auto) pts() const { return tree().pts(); }

  private:
    internal::Nonnull<const OCTree *> parent_;
    int nid_;
    Vector3d max_;
    Vector3d len_;
  };

  template <bool intra>
  void fill_neighbors(std::vector<int> &is, std::vector<int> &js,
                      const OCTreeBox &left, const OCTreeBox &right) {
    auto lp = left.idxs();
    if constexpr (intra) {
      if (left.id() == right.id()) {
        for (int i = 0; i < static_cast<int>(lp.size()) - 1; ++i) {
          is.insert(is.end(), lp.size() - i - 1, lp[i]);
          js.insert(js.end(), lp.begin() + i + 1, lp.end());
        }
        return;
      }
    }

    auto rp = right.idxs();
    for (const int i: lp) {
      is.insert(is.end(), rp.size(), i);
      js.insert(js.end(), rp.begin(), rp.end());
    }
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE inline void
  fill_conditional_buffered(VectorXd &dsqbuf, ArrayXi &jbuf,
                            std::vector<int> &is, std::vector<int> &js,
                            const int i, absl::Span<const int> jsrc,
                            const double cutoffsq) {
    int n = 0;
    for (int p = 0; p < jsrc.size(); ++p) {
      jbuf[n] = jsrc[p];
      n += value_if(dsqbuf[p] <= cutoffsq);
    }
    is.insert(is.end(), n, i);
    js.insert(js.end(), jbuf.data(), jbuf.data() + n);
  }

  template <bool intra>
  void find_neighbors_bucket(std::vector<int> &is, std::vector<int> &js,
                             Matrix3Xd &buffer, VectorXd &dsqbuf, ArrayXi &jbuf,
                             const OCTreeBox &left, const OCTreeBox &right,
                             const double cutoffsq) {
    auto lp = left.idxs();

    auto rp = right.idxs();
    buffer.leftCols(rp.size()) = right.pts()(E::all, rp);

    if constexpr (intra) {
      if (left.id() == right.id()) {
        for (int i = 0; i < static_cast<int>(lp.size()) - 1; ++i) {
          const Vector3d lpt = buffer.col(i);
          const int cnt = static_cast<int>(lp.size()) - i - 1;
          dsqbuf.head(cnt) = (buffer.middleCols(i + 1, cnt).colwise() - lpt)
                                 .colwise()
                                 .squaredNorm();
          fill_conditional_buffered(dsqbuf, jbuf, is, js, lp[i],
                                    lp.subspan(i + 1, cnt), cutoffsq);
        }
        return;
      }
    }

    for (const int i: lp) {
      const Vector3d lpt = left.pts().col(i);
      dsqbuf.head(rp.size()) =
          (buffer.leftCols(rp.size()).colwise() - lpt).colwise().squaredNorm();
      fill_conditional_buffered(dsqbuf, jbuf, is, js, i, rp, cutoffsq);
    }
  }

  template <bool intra>
  void find_neighbors_tree_impl(std::vector<int> &is, std::vector<int> &js,
                                Matrix3Xd &buffer, VectorXd &dsqbuf,
                                ArrayXi &jbuf, const OCTreeBox &left,
                                const OCTreeBox &right, const double cutoffsq) {
    auto [min_dsq, max_dsq] = left.minmax_distsq(right);

    if (min_dsq > cutoffsq)
      return;

    if (max_dsq <= cutoffsq) {
      fill_neighbors<intra>(is, js, left, right);
      return;
    }

    if (left.leaf_node() && right.leaf_node()) {
      find_neighbors_bucket<intra>(is, js, buffer, dsqbuf, jbuf, left, right,
                                   cutoffsq);
      return;
    }

    if (left.leaf_node()) {
      for (int j = 0; j < 8; ++j) {
        if (!right.has_child(j))
          continue;

        find_neighbors_tree_impl<intra>(is, js, buffer, dsqbuf, jbuf, left,
                                        right.child(j), cutoffsq);
      }
      return;
    }

    if (right.leaf_node()) {
      for (int i = 0; i < 8; ++i) {
        if (!left.has_child(i))
          continue;

        find_neighbors_tree_impl<intra>(is, js, buffer, dsqbuf, jbuf,
                                        left.child(i), right, cutoffsq);
      }
      return;
    }

    for (int i = 0; i < 8; ++i) {
      if (!left.has_child(i))
        continue;

      OCTreeBox lc = left.child(i);
      int j0 = 0;
      if constexpr (intra) {
        if (left.id() == right.id())
          j0 = i;
      }
      for (int j = j0; j < 8; ++j) {
        if (!right.has_child(j))
          continue;

        find_neighbors_tree_impl<intra>(is, js, buffer, dsqbuf, jbuf, lc,
                                        right.child(j), cutoffsq);
      }
    }
  }
}  // namespace

void OCTree::find_neighbors_tree(const OCTree &oct, const double cutoff,
                                 std::vector<int> &self,
                                 std::vector<int> &other) const {
  const double cutoffsq = cutoff * cutoff;

  self.clear();
  other.clear();
  Matrix3Xd buffer(3, oct.bucket_size_);
  VectorXd dsqbuf(oct.bucket_size_);
  ArrayXi jbuf(oct.bucket_size_);

  find_neighbors_tree_impl<false>(self, other, buffer, dsqbuf, jbuf,
                                  OCTreeBox(*this), OCTreeBox(oct), cutoffsq);
}

std::vector<std::vector<int>> OCTree::find_neighbors_tree(const OCTree &oct,
                                                          double cutoff) const {
  std::vector<int> is, js;
  find_neighbors_tree(oct, cutoff, is, js);

  std::vector<std::vector<int>> idxs(pts().cols());
  for (int p = 0; p < is.size(); ++p) {
    idxs[is[p]].push_back(js[p]);
  }
  return idxs;
}

void OCTree::find_neighbors_self(double cutoff, std::vector<int> &left,
                                 std::vector<int> &right) const {
  const double cutoffsq = cutoff * cutoff;

  left.clear();
  right.clear();
  Matrix3Xd buffer(3, bucket_size_);
  VectorXd dsqbuf(bucket_size_);
  ArrayXi jbuf(bucket_size_);

  OCTreeBox root(*this);
  find_neighbors_tree_impl<true>(left, right, buffer, dsqbuf, jbuf, root, root,
                                 cutoffsq);
}
}  // namespace nuri

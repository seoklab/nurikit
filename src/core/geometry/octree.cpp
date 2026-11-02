//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/types/span.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  using internal::OCTreeNode;

  using Array8i = Array<int, 8, 1>;

  // Octant index is (xyz):
  // 000 -> +++, 100 -> -++, 010 -> +-+, 110 -> --+,
  // 001 -> ++-, 101 -> -+-, 011 -> +--, 111 -> ---

  using SplitKey = std::pair<double, int>;
  using SplitKeys = absl::Span<SplitKey>;

  // Split begin - end at the rank median into two parts:
  // +: [begin, ret), -: [ret, end)
  int partition_sub(ArrayXi &idxs, int begin, int end, OCTree::Points pts,
                    int axis, SplitKeys keys) {
    for (int i = begin; i < end; ++i)
      keys[i] = { pts(axis, idxs[i]), idxs[i] };

    const int mid = begin + (end - begin) / 2;
    std::nth_element(
        keys.begin() + begin, keys.begin() + mid, keys.begin() + end,
        [](const SplitKey &a, const SplitKey &b) { return a.first > b.first; });

    for (int i = begin; i < end; ++i)
      idxs[i] = keys[i].second;
    return mid;
  }

  Array8i partition_octant(ArrayXi &idxs, int begin, int end,
                           OCTree::Points pts, SplitKeys keys) {
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
        ptrs[j] = partition_sub(idxs, left, right, pts, i, keys);
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

  // XXX: data reallocates during the recursion; hold slots, never references.
  void build_octree(OCTree::Points pts, std::vector<OCTreeNode> &data,
                    ArrayXi &idxs, const int begin, const int nleaf,
                    const int bucket_size, SplitKeys keys, const int slot) {
    Vector3d mins = pts.col(idxs[begin]), maxs = mins;
    for (int i = begin + 1; i < begin + nleaf; ++i) {
      auto pt = pts.col(idxs[i]);
      mins = mins.cwiseMin(pt);
      maxs = maxs.cwiseMax(pt);
    }

    if (nleaf <= bucket_size) {
      std::sort(idxs.begin() + begin, idxs.begin() + begin + nleaf);
      data[slot] = OCTreeNode(0, 0, begin, nleaf, mins, maxs);
      return;
    }

    Array8i ptrs = partition_octant(idxs, begin, begin + nleaf, pts, keys);

    uint8_t mask = 0;
    for (int i = 0, left = begin; i < 8; ++i) {
      mask |= static_cast<uint8_t>(value_if(ptrs[i] > left) << i);
      left = ptrs[i];
    }

    const int first = static_cast<int>(data.size());
    const OCTreeNode node(first, mask, begin, nleaf, mins, maxs);
    data.resize(first + node.num_children());
    data[slot] = node;

    int cid = first;
    for (int i = 0, left = begin; i < 8; ++i) {
      const int right = ptrs[i];
      if (right > left)
        build_octree(pts, data, idxs, left, right - left, bucket_size, keys,
                     cid++);
      left = right;
    }
  }
}  // namespace

void OCTree::rebuild_impl(Points src) {
  bucket_size_ = nuri::max(8, bucket_size_);
  nodes_.clear();

  const int n = static_cast<int>(src.cols());
  if (n == 0) {
    max_.setZero();
    len_.setZero();
    idxs_.resize(0);
    pts_.resize(3, 0);
    return;
  }

  idxs_.resize(n);
  std::iota(idxs_.begin(), idxs_.end(), 0);

  std::vector<SplitKey> keys(n);
  nodes_.resize(1);
  {
    internal::AllowEigenMallocScoped<false> ems;
    build_octree(src, nodes_, idxs_, 0, n, bucket_size_, absl::MakeSpan(keys),
                 root());
  }

  const OCTreeNode &root_node = nodes_.front();
  max_ = root_node.maxs();
  len_ = max_ - root_node.mins();

  pts_ = src(E::all, idxs_);
}

namespace {
  using internal::OCTreeKdScratch;

  double minsq_box_point(const OCTreeNode &node, const Vector3d &pt) {
    return (node.mins() - pt)
        .cwiseMax(pt - node.maxs())
        .cwiseMax(0.0)
        .squaredNorm();
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

namespace {
  class ScalarBest {
  public:
    double offer(double d, int i, double worst_dsq) {
      if (idx_ < 0) {
        dsq_ = d;
        idx_ = i;
        return worst_dsq;
      }

      if (d <= dsq_) {
        dsq_ = d;
        idx_ = i;
      }
      return dsq_;
    }

    void drain(std::vector<int> &idxs, std::vector<double> &distsq) const {
      if (idx_ < 0)
        return;

      idxs.push_back(idx_);
      distsq.push_back(dsq_);
    }

  private:
    double dsq_ = 0;
    int idx_ = -1;
  };

  class HeapBest {
  public:
    HeapBest(int k, OCTreeKdScratch::CandHeap &heap): k_(k), heap_(&heap) { }

    double offer(double d, int i, double worst_dsq) {
      heap_->push({ d, i });
      if (static_cast<int>(heap_->size()) <= k_)
        return worst_dsq;

      heap_->pop();
      return heap_->top().dsq;
    }

    void drain(std::vector<int> &idxs, std::vector<double> &distsq) {
      const int n = static_cast<int>(heap_->size());
      idxs.resize(n);
      distsq.resize(n);
      for (int i = n - 1; i >= 0; --i) {
        auto e = heap_->pop_get();
        idxs[i] = e.idx;
        distsq[i] = e.dsq;
      }
    }

  private:
    int k_;
    internal::Nonnull<OCTreeKdScratch::CandHeap *> heap_;
  };

  template <class Best>
  double scan_leaf(const OCTree &tree, const OCTreeNode &node,
                   const Vector3d &pt, ArrayXd &dsqbuf, Best &best,
                   double worst_dsq) {
    const int cnt = node.nleaf();
    dsqbuf.head(cnt) = (tree.pts().middleCols(node.begin(), cnt).colwise() - pt)
                           .colwise()
                           .squaredNorm();

    for (int i = 0; i < cnt; ++i) {
      double d = dsqbuf[i];
      if (d > worst_dsq)
        continue;

      worst_dsq = best.offer(d, tree.idxs()[i + node.begin()], worst_dsq);
    }
    return worst_dsq;
  }

  template <class Best>
  void find_kd_impl(const OCTree &tree, const Vector3d &pt, double worst_dsq,
                    Best &best, OCTreeKdScratch &scratch,
                    std::vector<int> &idxs, std::vector<double> &distsq) {
    ArrayXd &dsqbuf = scratch.dsqbuf(tree.bucket_size());

    OCTreeKdScratch::NodeHeap &node_heap = scratch.nodes();
    node_heap.emplace(OCTree::root(), 0.0);

    while (!node_heap.empty()) {
      const auto top = node_heap.pop_get();
      // Nodes are popped in order of increasing min_dsq, so once min_dsq
      // exceeds worst_dsq we cannot improve the answer anymore.
      if (top.min_dsq() > worst_dsq)
        break;

      int cur = top.idx();
      while (true) {
        const OCTreeNode &node = tree[cur];
        if (node.leaf()) {
          worst_dsq = scan_leaf(tree, node, pt, dsqbuf, best, worst_dsq);
          break;
        }

        const int first = node.child_begin(), nchild = node.num_children();
        double child_dsq[8];
        int nearest = -1;
        double nearest_dsq = worst_dsq;
        for (int k = 0; k < nchild; ++k) {
          child_dsq[k] = minsq_box_point(tree[first + k], pt);
          if (child_dsq[k] <= nearest_dsq) {
            nearest = k;
            nearest_dsq = child_dsq[k];
          }
        }

        if (nearest < 0)
          break;

        for (int k = 0; k < nchild; ++k) {
          if (k == nearest || child_dsq[k] > worst_dsq)
            continue;

          // XXX: emplace, not push: a pushed entry temp is staged on the stack
          // and reloaded as one 16-byte word (store-forwarding stall).
          node_heap.emplace(first + k, child_dsq[k]);
        }
        cur = first + nearest;
      }
    }

    best.drain(idxs, distsq);
  }
}  // namespace

void OCTree::find_neighbors_kd(const Vector3d &pt, const int k,
                               std::vector<int> &idxs,
                               std::vector<double> &distsq,
                               double cutoff) const {
  KdScratch scratch;
  find_neighbors_kd(pt, k, idxs, distsq, scratch, cutoff);
}

void OCTree::find_neighbors_kd(const Vector3d &pt, const int k,
                               std::vector<int> &idxs,
                               std::vector<double> &distsq, KdScratch &scratch,
                               double cutoff) const {
  idxs.clear();
  distsq.clear();
  scratch.clear();
  if (k <= 0) {
    ABSL_LOG(WARNING) << "k is not a positive number: " << k;
    return;
  }
  if (nodes_.empty())
    return;
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

  if (k == 1) {
    ScalarBest best;
    find_kd_impl(*this, pt, worst_dsq, best, scratch, idxs, distsq);
    return;
  }

  HeapBest best(k, scratch.cands());
  find_kd_impl(*this, pt, worst_dsq, best, scratch, idxs, distsq);
}

namespace {
  void find_candidates_d(const OCTree &oct, const Vector3d &pt,
                         const double cutoffsq, std::vector<int> &idxs,
                         std::vector<double> &distsq, const OCTreeNode &node,
                         ArrayXd &dsqbuf) {
    if (node.leaf()) {
      const int cnt = node.nleaf();
      dsqbuf.head(cnt) =
          (oct.pts().middleCols(node.begin(), cnt).colwise() - pt)
              .colwise()
              .squaredNorm();

      idxs.insert(idxs.end(), oct.idxs().data() + node.begin(),
                  oct.idxs().data() + node.end());
      distsq.insert(distsq.end(), dsqbuf.data(), dsqbuf.data() + cnt);
      return;
    }

    for (int cid = node.child_begin(); cid < node.child_end(); ++cid) {
      if (minsq_box_point(oct[cid], pt) > cutoffsq)
        continue;

      find_candidates_d(oct, pt, cutoffsq, idxs, distsq, oct[cid], dsqbuf);
    }
  }
}  // namespace

void OCTree::find_neighbors_d(const Vector3d &pt, const double cutoff,
                              std::vector<int> &idxs,
                              std::vector<double> &distsq) const {
  const double cutoffsq = cutoff * cutoff;

  idxs.clear();
  distsq.clear();
  if (nodes_.empty())
    return;

  ArrayXd dsqbuf(bucket_size());
  find_candidates_d(*this, pt, cutoffsq, idxs, distsq, node(root()), dsqbuf);

  int n = 0;
  for (int i = 0; i < idxs.size(); ++i) {
    idxs[n] = idxs[i];
    const double dsq = distsq[n] = distsq[i];
    n += value_if(dsq <= cutoffsq);
  }
  idxs.resize(n);
  distsq.resize(n);
}

namespace {
  class OCTreeBox {
  public:
    explicit OCTreeBox(const OCTree &parent)
        : OCTreeBox(parent, OCTree::root()) { }

    OCTreeBox(const OCTree &parent, int nid): parent_(&parent), nid_(nid) { }

    std::pair<double, double> minmax_distsq(const OCTreeBox &other) const {
      Vector3d minmax = node().mins() - other.node().maxs(),
               maxmin = other.node().mins() - node().maxs();

      double min_dsq = minmax.cwiseMax(0.0).squaredNorm()
                       + maxmin.cwiseMax(0.0).squaredNorm();
      double max_dsq =
          minmax.cwiseAbs().cwiseMax(maxmin.cwiseAbs()).squaredNorm();

      return { min_dsq, max_dsq };
    }

    int child_begin() const { return node().child_begin(); }
    int child_end() const { return node().child_end(); }

    OCTreeBox child(int cid) const { return OCTreeBox(*parent_, cid); }

    bool leaf_node() const { return node().leaf(); }

    absl::Span<const int> idxs() const {
      ABSL_DCHECK_GE(id(), 0);
      const int *p = tree().idxs().data() + begin();
      return absl::MakeConstSpan(p, nleaf());
    }

    int begin() const { return node().begin(); }
    int end() const { return node().end(); }
    int nleaf() const { return node().nleaf(); }

    int id() const { return nid_; }

    const OCTree &tree() const { return *parent_; }
    const OCTreeNode &node() const { return parent_->node(nid_); }
    decltype(auto) pts() const { return tree().pts(); }

  private:
    internal::Nonnull<const OCTree *> parent_;
    int nid_;
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
                             VectorXd &dsqbuf, ArrayXi &jbuf,
                             const OCTreeBox &left, const OCTreeBox &right,
                             const double cutoffsq) {
    const auto rp = right.idxs();

    if constexpr (intra) {
      if (left.id() == right.id()) {
        const Matrix3Xd &pts = left.pts();

        for (int i = 0; i < left.nleaf() - 1; ++i) {
          const int cnt = left.nleaf() - i - 1;
          const Vector3d lpt = pts.col(left.begin() + i);
          dsqbuf.head(cnt) =
              (pts.middleCols(left.begin() + i + 1, cnt).colwise() - lpt)
                  .colwise()
                  .squaredNorm();

          fill_conditional_buffered(dsqbuf, jbuf, is, js, left.idxs()[i],
                                    rp.subspan(i + 1, cnt), cutoffsq);
        }
        return;
      }
    }

    const int cnt = right.nleaf();
    auto rpts = right.pts().middleCols(right.begin(), cnt);

    for (int i = left.begin(); i < left.end(); ++i) {
      const Vector3d lpt = left.pts().col(i);
      dsqbuf.head(cnt) = (rpts.colwise() - lpt).colwise().squaredNorm();

      fill_conditional_buffered(dsqbuf, jbuf, is, js, left.tree().idxs()[i], rp,
                                cutoffsq);
    }
  }

  template <bool intra>
  void find_neighbors_tree_impl(std::vector<int> &is, std::vector<int> &js,
                                VectorXd &dsqbuf, ArrayXi &jbuf,
                                const OCTreeBox &left, const OCTreeBox &right,
                                const double cutoffsq) {
    auto [min_dsq, max_dsq] = left.minmax_distsq(right);

    if (min_dsq > cutoffsq)
      return;

    if (max_dsq <= cutoffsq) {
      fill_neighbors<intra>(is, js, left, right);
      return;
    }

    if (left.leaf_node() && right.leaf_node()) {
      find_neighbors_bucket<intra>(is, js, dsqbuf, jbuf, left, right, cutoffsq);
      return;
    }

    if (left.leaf_node()) {
      for (int j = right.child_begin(); j < right.child_end(); ++j)
        find_neighbors_tree_impl<intra>(is, js, dsqbuf, jbuf, left,
                                        right.child(j), cutoffsq);
      return;
    }

    if (right.leaf_node()) {
      for (int i = left.child_begin(); i < left.child_end(); ++i)
        find_neighbors_tree_impl<intra>(is, js, dsqbuf, jbuf, left.child(i),
                                        right, cutoffsq);
      return;
    }

    for (int i = left.child_begin(); i < left.child_end(); ++i) {
      OCTreeBox lc = left.child(i);
      int j0 = right.child_begin();
      if constexpr (intra) {
        if (left.id() == right.id())
          j0 = i;
      }
      for (int j = j0; j < right.child_end(); ++j)
        find_neighbors_tree_impl<intra>(is, js, dsqbuf, jbuf, lc,
                                        right.child(j), cutoffsq);
    }
  }
}  // namespace

void OCTree::find_neighbors_tree(const OCTree &oct, const double cutoff,
                                 std::vector<int> &self,
                                 std::vector<int> &other) const {
  const double cutoffsq = cutoff * cutoff;

  self.clear();
  other.clear();
  if (nodes_.empty() || oct.nodes_.empty())
    return;

  const int cap = nuri::max(bucket_size(), oct.bucket_size());
  VectorXd dsqbuf(cap);
  ArrayXi jbuf(cap);

  find_neighbors_tree_impl<false>(self, other, dsqbuf, jbuf, OCTreeBox(*this),
                                  OCTreeBox(oct), cutoffsq);
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
  if (nodes_.empty())
    return;

  VectorXd dsqbuf(bucket_size());
  ArrayXi jbuf(bucket_size());

  OCTreeBox root(*this);
  find_neighbors_tree_impl<true>(left, right, dsqbuf, jbuf, root, root,
                                 cutoffsq);
}
}  // namespace nuri

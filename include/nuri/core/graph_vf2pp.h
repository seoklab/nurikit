//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_VF2PP_H_
#define NURI_CORE_GRAPH_VF2PP_H_

/// @cond
#include <algorithm>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>
/// @endcond

#include "nuri/eigen_config.h"
#include "nuri/core/graph.h"
#include "nuri/utils.h"

namespace nuri {
enum class MappingType : int {
  kSubgraph,     // Subgraph isomorphism
  kInduced,      // Induced subgraph isomorphism
  kIsomorphism,  // Graph isomorphism
};

namespace internal {
  template <class NT, class ET, class AL1, class AL2>
  int vf2pp_process_bfs_tree(const Graph<NT, ET> &query, ArrayXi &order,
                             ArrayXb &visited, AL1 &curr_conn, AL2 &query_cnts,
                             int root, int i) {
    order[i] = root;
    visited[root] = true;

    int lp = i, rp = i + 1, wp = i + 1;

    for (; i < wp; ++i) {
      int curr = order[i];

      for (auto nei: query.node(curr)) {
        if (!visited[nei.dst().id()]) {
          order[wp++] = nei.dst().id();
          visited[nei.dst().id()] = true;
        }
      }

      if (i < rp)
        continue;

      for (int j = lp; j < rp; ++j) {
        int min_pos = j;
        for (int k = j + 1; k < rp; ++k) {
          if (curr_conn[order[min_pos]] < curr_conn[order[k]]
              || (curr_conn[order[min_pos]] == curr_conn[order[k]]
                  && (query.degree(order[min_pos]) < query.degree(order[k])
                      || (query.degree(order[min_pos]) == query.degree(order[k])
                          && query_cnts[order[min_pos]]
                                 > query_cnts[order[k]])))) {
            min_pos = k;
          }
        }

        --query_cnts[order[min_pos]];
        for (auto nei: query.node(order[min_pos]))
          ++curr_conn[nei.dst().id()];

        std::swap(order[j], order[min_pos]);
      }

      lp = rp;
      rp = wp;
    }

    return i;
  }

  template <class NT, class ET, class AL>
  ArrayXi vf2pp_init_order(const Graph<NT, ET> &query, ConstRef<ArrayXi> qlbl,
                           ConstRef<ArrayXi> tlbl, ArrayXi &target_lcnt,
                           // NOLINTNEXTLINE(*-missing-std-forward)
                           AL &&curr_conn) {
    ArrayXi order = ArrayXi::Constant(query.size(), -1);

    for (int l: tlbl)
      ++target_lcnt[l];
    auto query_cnts = target_lcnt(qlbl);

    ArrayXb visited = ArrayXb::Zero(query.size());
    int iorder = 0;
    for (int i = 0; iorder < query.size() && i < query.size();) {
      if (visited[i]) {
        ++i;
        continue;
      }

      int imin = i;
      for (int j = i + 1; j < query.size(); ++j) {
        if (!visited[j]
            && (query_cnts[imin] > query_cnts[j]
                || (query_cnts[imin] == query_cnts[j]
                    && query.degree(imin) < query.degree(j)))) {
          imin = j;
        }
      }

      iorder = vf2pp_process_bfs_tree(query, order, visited, curr_conn,
                                      query_cnts, imin, iorder);
    }

    return order;
  }

  using Vf2ppLabels = std::vector<std::pair<int, int>>;
  using Vf2ppLabelMap = std::vector<Vf2ppLabels>;

  template <class NT, class ET, class AL>
  std::pair<Vf2ppLabelMap, Vf2ppLabelMap>
  vf2pp_init_r_new_r_inout(const Graph<NT, ET> &query, ConstRef<ArrayXi> qlbl,
                           const ArrayXi &order, ArrayXi &r_inout,
                           // NOLINTNEXTLINE(*-missing-std-forward)
                           ArrayXi &r_new, AL &&visit_count) {
    // r_inout == _labelTmp1, r_new == _labelTmp2
    ABSL_DCHECK_EQ(r_inout.size(), r_new.size());

    Vf2ppLabelMap r_inout_labels(query.size()), r_new_labels(query.size());

    r_inout.setZero();
    r_new.setZero();
    visit_count.setZero();
    for (const int i: order) {
      visit_count[i] = -1;

      for (auto nei: query.node(i)) {
        const int curr = nei.dst().id();
        if (visit_count[curr] > 0) {
          ++r_inout[qlbl[curr]];
        } else if (visit_count[curr] == 0) {
          ++r_new[qlbl[curr]];
        }
      }

      for (auto nei: query.node(i)) {
        const int curr = nei.dst().id();
        const int curr_lbl = qlbl[curr];

        if (r_inout[curr_lbl] > 0) {
          r_inout_labels[i].push_back({ curr_lbl, r_inout[curr_lbl] });
          r_inout[curr_lbl] = 0;
        } else if (r_new[curr_lbl] > 0) {
          r_new_labels[i].push_back({ curr_lbl, r_new[curr_lbl] });
          r_new[curr_lbl] = 0;
        }

        if (visit_count[curr] >= 0)
          ++visit_count[curr];
      }
    }

    return { r_inout_labels, r_new_labels };
  }

  template <MappingType kMt>
  bool vf2pp_r_matches(const Vf2ppLabels &r_node, const ArrayXi &label_tmp) {
    return absl::c_none_of(r_node, [&](std::pair<int, int> p) {
      return kMt == MappingType::kIsomorphism ? label_tmp[p.first] != 0
                                              : label_tmp[p.first] > 0;
    });
  }
}  // namespace internal

template <MappingType kMt, class N1, class E1, class N2, class E2>
class VF2pp {
private:
  const Graph<N1, E1> &query() const { return *query_; }
  const Graph<N2, E2> &target() const { return *target_; }

  auto query_tmp() { return node_tmp_.head(query_->size()); }

  auto conn() { return node_tmp_.head(target_->size()); }

  auto curr_query() const { return query_->node(order_[depth_]); }

  int mapped_target() const { return mapping_[curr_query().id()]; }

  int target_src() const { return src_nei_(0, depth_); }
  void update_target_src(int idx) {
    ABSL_DCHECK_GE(idx, 0);
    ABSL_DCHECK_LT(idx, target().size());

    src_nei_(0, depth_) = idx;
  }

  int target_nei_idx() const { return src_nei_(1, depth_); }
  void update_target_nei_idx(int idx) {
    ABSL_DCHECK_GE(target_src(), 0);
    ABSL_DCHECK_LT(target_src(), target().size());

    ABSL_DCHECK_GE(idx, 0);
    ABSL_DCHECK_LT(idx, target().degree(target_src()));

    src_nei_(1, depth_) = idx;
  }
  void invalidate_target_nei_idx() { src_nei_(1, depth_) = -1; }

  ArrayXi &r_inout_cnt() { return label_tmp1_; }
  ArrayXi &r_new_cnt() { return label_tmp2_; }

public:
  template <class AL1, class AL2>
  VF2pp(const Graph<N1, E1> &query, const Graph<N2, E2> &target,
        AL1 &&query_lbl, AL2 &&target_lbl)
      : query_(&query), target_(&target),  //
        qlbl_(std::forward<AL1>(query_lbl)),
        tlbl_(std::forward<AL2>(target_lbl)),
        mapping_(ArrayXi::Constant(query.size(), -1)),
        src_nei_(Array2Xi::Constant(2, target.size(), -1)),
        node_tmp_(ArrayXi::Zero(nuri::max(query.size(), target.size()))) {
    const int nlabel = nuri::max(qlbl_.maxCoeff(), tlbl_.maxCoeff()) + 1;

    label_tmp1_ = ArrayXi::Zero(nlabel);
    label_tmp2_ = ArrayXi::Zero(nlabel);

    order_ = internal::vf2pp_init_order(query, qlbl_, tlbl_, label_tmp1_,
                                        query_tmp());
    std::tie(r_inout_, r_new_) = internal::vf2pp_init_r_new_r_inout(
        query, qlbl_, order_, r_inout_cnt(), r_new_cnt(), query_tmp());

    node_tmp_.head(nuri::min(query.size(), target.size())).setZero();
  }

  template <class BinaryPred>
  bool next(const BinaryPred &match) {
    while (depth_ >= 0) {
      if (depth_ == query().size()) {
        --depth_;
        return true;
      }

      const auto qn = curr_query();
      const int ti = mapped_target();

      int tj = target_src(), tk = target_nei_idx();
      if (tk >= 0) {
        ABSL_DCHECK_NE(ti, tj);
        sub_pair(qn.id(), ti);
        ++tk;
      } else {
        int qk = 0;
        if (ti < 0) {
          qk = absl::c_find_if(
                   qn, [&](auto nei) { return mapping_[nei.dst().id()] >= 0; })
               - qn.begin();
        } else {
          sub_pair(qn.id(), ti);
        }

        if (qk == qn.degree() || ti >= 0) {
          const int ti2 =
              std::find_if(target().begin() + ti + 1, target().end(),
                           [&](auto tn) {
                             bool candidate = kMt == MappingType::kSubgraph
                                                  ? conn()[tn.id()] >= 0
                                                  : conn()[tn.id()] == 0;
                             return candidate && feas(qn, tn, match);
                           })
              - target().begin();

          if (ti2 < target().size()) {
            add_pair(qn.id(), ti2);
            ++depth_;
          } else {
            --depth_;
          }

          continue;
        }

        tj = mapping_[qn[qk].dst().id()];
        tk = 0;
        update_target_src(tj);
      }

      for (; tk < target().degree(tj); ++tk) {
        auto tn = target().node(tj)[tk].dst();
        if (conn()[tn.id()] > 0 && feas(qn, tn, match)) {
          add_pair(qn.id(), tn.id());
          break;
        }
      }

      if (tk < target().degree(tj)) {
        update_target_nei_idx(tk);
        ++depth_;
      } else {
        invalidate_target_nei_idx();
        --depth_;
      }
    }

    return false;
  }

  const ArrayXi &mapping() const & { return mapping_; }

  ArrayXi &&mapping() && { return std::move(mapping_); }

private:
  void add_pair(const int qi, const int ti) {
    ABSL_DCHECK_GE(ti, 0);
    ABSL_DCHECK_LT(ti, target().size());

    mapping_[qi] = ti;
    conn()[ti] = -1;

    for (auto nei: target().node(ti)) {
      if (conn()[nei.dst().id()] != -1)
        ++conn()[nei.dst().id()];
    }
  }

  void sub_pair(const int qi, const int ti) {
    ABSL_DCHECK_GE(ti, 0);
    ABSL_DCHECK_LT(ti, target().size());

    mapping_[qi] = -1;
    conn()[ti] = 0;

    for (auto nei: target().node(ti)) {
      int curr_conn = conn()[nei.dst().id()];
      if (curr_conn > 0) {
        --conn()[nei.dst().id()];
      } else if (curr_conn == -1) {
        ++conn()[ti];
      }
    }
  }

  bool cut_by_labels(const typename Graph<N1, E1>::ConstNodeRef qn,
                     const typename Graph<N2, E2>::ConstNodeRef tn) {
    // zero init
    r_inout_cnt()(tlbl_)(as_index(tn)).setZero();
    for (auto [lbl, cnt]: r_inout_[qn.id()])
      r_inout_cnt()[lbl] = 0;
    if constexpr (kMt != MappingType::kSubgraph) {
      r_new_cnt()(tlbl_)(as_index(tn)).setZero();
      for (auto [lbl, cnt]: r_new_[qn.id()])
        r_new_cnt()[lbl] = 0;
    }

    for (auto tnei: tn) {
      const int curr = tnei.dst().id();
      if (conn()[curr] > 0)
        --r_inout_cnt()[tlbl_[curr]];
      else if constexpr (kMt != MappingType::kSubgraph) {
        if (conn()[curr] == 0)
          --r_new_cnt()[tlbl_[curr]];
      }
    }

    for (auto [lbl, cnt]: r_inout_[qn.id()])
      r_inout_cnt()[lbl] += cnt;
    if constexpr (kMt != MappingType::kSubgraph) {
      for (auto [lbl, cnt]: r_new_[qn.id()])
        r_new_cnt()[lbl] += cnt;
    }

    const bool r_inout_match =
        internal::vf2pp_r_matches<kMt>(r_inout_[qn.id()], r_inout_cnt());
    if constexpr (kMt == MappingType::kSubgraph)
      return r_inout_match;

    return r_inout_match
           && internal::vf2pp_r_matches<kMt>(r_new_[qn.id()], r_new_cnt());
  }

  template <class BinaryPred>
  bool feas(const typename Graph<N1, E1>::ConstNodeRef qn,
            const typename Graph<N2, E2>::ConstNodeRef tn,
            const BinaryPred &match) {
    if (qlbl_[qn.id()] != tlbl_[tn.id()])
      return false;

    for (auto qnei: qn)
      if (mapping_[qnei.dst().id()] >= 0)
        --conn()[mapping_[qnei.dst().id()]];

    bool is_iso = true;
    for (auto tnei: tn) {
      const int curr_conn = conn()[tnei.dst().id()];
      if (curr_conn < -1)
        ++conn()[tnei.dst().id()];
      else if constexpr (kMt != MappingType::kSubgraph) {
        if (curr_conn == -1)
          is_iso = false;
      }
    }

    if (!is_iso) {
      for (auto qnei: qn) {
        const int ti = mapping_[qnei.dst().id()];
        if (ti >= 0)
          conn()[ti] = -1;
      }
      return false;
    }

    for (auto qnei: qn) {
      const int ti = mapping_[qnei.dst().id()];
      if (ti < 0 || conn()[ti] == -1)
        continue;

      const int curr_conn = conn()[ti];
      conn()[ti] = -1;
      if constexpr (kMt == MappingType::kSubgraph) {
        if (curr_conn < -1)
          return false;
      } else {
        return false;
      }
    }

    return match(qn, tn) && cut_by_labels(qn, tn);
  }

  const Graph<N1, E1> *query_;
  const Graph<N2, E2> *target_;

  Eigen::Ref<const ArrayXi> qlbl_, tlbl_;

  // query.size()
  ArrayXi mapping_, order_;
  internal::Vf2ppLabelMap r_inout_, r_new_;

  // (2, target.size())
  Array2Xi src_nei_;

  // max(label) + 1
  ArrayXi label_tmp1_, label_tmp2_;

  // max(query.size(), target.size())
  ArrayXi node_tmp_;

  int depth_ = 0;
};

template <MappingType kMt, class N1, class E1, class N2, class E2,  //
          class AL1, class AL2>
VF2pp<kMt, N1, E1, N2, E2> make_vf2pp(const Graph<N1, E1> &query,
                                      const Graph<N2, E2> &target, AL1 &&qlbl,
                                      AL2 &&tlbl) {
  return VF2pp<kMt, N1, E1, N2, E2>(query, target, std::forward<AL1>(qlbl),
                                    std::forward<AL2>(tlbl));
}

template <MappingType kMt, class N1, class E1, class N2, class E2, class AL1,
          class AL2, class BinaryPred>
std::pair<ArrayXi, bool> vf2pp(const Graph<N1, E1> &query,
                               const Graph<N2, E2> &target, AL1 &&qlbl,
                               AL2 &&tlbl, const BinaryPred &match) {
  VF2pp<kMt, N1, E1, N2, E2> vf2pp = make_vf2pp<kMt>(
      query, target, std::forward<AL1>(qlbl), std::forward<AL2>(tlbl));
  bool found = vf2pp.next(match);
  return { std::move(vf2pp).mapping(), found };
}

template <class N1, class E1, class N2, class E2, class BinaryPred, class AL1,
          class AL2>
std::pair<ArrayXi, bool>
vf2pp(const Graph<N1, E1> &query, const Graph<N2, E2> &target, AL1 &&qlbl,
      AL2 &&tlbl, const BinaryPred &match, MappingType mt) {
  switch (mt) {
  case MappingType::kSubgraph:
    return vf2pp<MappingType::kSubgraph>(query, target, std::forward<AL1>(qlbl),
                                         std::forward<AL2>(tlbl), match);
  case MappingType::kInduced:
    return vf2pp<MappingType::kInduced>(query, target, std::forward<AL1>(qlbl),
                                        std::forward<AL2>(tlbl), match);
  case MappingType::kIsomorphism:
    return vf2pp<MappingType::kIsomorphism>(
        query, target, std::forward<AL1>(qlbl), std::forward<AL2>(tlbl), match);
  }

  ABSL_LOG(ERROR) << "Invalid mapping type (" << static_cast<int>(mt) << ")";
  return { {}, false };
}

template <MappingType kMt, class N1, class E1, class N2, class E2,
          class BinaryPred>
std::pair<ArrayXi, bool> vf2pp(const Graph<N1, E1> &query,
                               const Graph<N2, E2> &target,
                               const BinaryPred &match) {
  ArrayXi label = ArrayXi::Zero(nuri::max(query.size(), target.size()));
  return vf2pp<kMt>(query, target, label.head(query.size()),
                    label.head(target.size()), match);
}

template <class N1, class E1, class N2, class E2, class BinaryPred>
std::pair<ArrayXi, bool> vf2pp(const Graph<N1, E1> &query,
                               const Graph<N2, E2> &target,
                               const BinaryPred &match, MappingType mt) {
  ArrayXi label = ArrayXi::Zero(nuri::max(query.size(), target.size()));
  return vf2pp(query, target, label.head(query.size()),
               label.head(target.size()), match, mt);
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_VF2PP_H_ */

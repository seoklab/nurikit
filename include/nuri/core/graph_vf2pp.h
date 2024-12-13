//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_VF2PP_H_
#define NURI_CORE_GRAPH_VF2PP_H_

/// @cond
#include <algorithm>
#include <tuple>
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
enum class IsoMapType : int {
  kSubgraph,  // Subgraph isomorphism
  kInduced,   // Induced subgraph isomorphism
  kGraph,     // Graph isomorphism
};

namespace internal {
  template <class GT, bool = internal::GraphTraits<GT>::is_degree_constant_time>
  class Vf2ppDegreeHelper;

  template <class GT>
  class Vf2ppDegreeHelper<GT, true> {
  public:
    explicit Vf2ppDegreeHelper(const GT &graph): graph_(&graph) { }

    int operator[](int i) const { return graph_->degree(i); }

  private:
    const GT *graph_;
  };

  template <class GT>
  class Vf2ppDegreeHelper<GT, false> {
  public:
    explicit Vf2ppDegreeHelper(const GT &graph): degree_(graph.size()) {
      for (int i = 0; i < graph.size(); ++i)
        degree_[i] = graph.degree(i);
    }

    int operator[](int i) const { return degree_[i]; }

  private:
    ArrayXi degree_;
  };

  template <class GT, class AL1, class AL2>
  int vf2pp_process_bfs_tree(const GT &query,
                             const Vf2ppDegreeHelper<GT> &degrees,
                             ArrayXi &order, ArrayXb &visited, AL1 &curr_conn,
                             AL2 &query_cnts, int root, int i) {
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
                  && (degrees[order[min_pos]] < degrees[order[k]]
                      || (degrees[order[min_pos]] == degrees[order[k]]
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

  template <class GT, class AL>
  ArrayXi vf2pp_init_order(const GT &query, ConstRef<ArrayXi> qlbl,
                           ConstRef<ArrayXi> tlbl, ArrayXi &target_lcnt,
                           // NOLINTNEXTLINE(*-missing-std-forward)
                           AL &&curr_conn) {
    ArrayXi order = ArrayXi::Constant(query.size(), -1);

    for (int l: tlbl)
      ++target_lcnt[l];
    auto query_cnts = target_lcnt(qlbl);

    Vf2ppDegreeHelper<GT> degrees(query);
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
                    && degrees[imin] < degrees[j]))) {
          imin = j;
        }
      }

      iorder = vf2pp_process_bfs_tree(query, degrees, order, visited, curr_conn,
                                      query_cnts, imin, iorder);
    }

    return order;
  }

  using Vf2ppLabels = std::vector<std::pair<int, int>>;
  using Vf2ppLabelMap = std::vector<Vf2ppLabels>;

  template <class GT, class AL>
  std::pair<Vf2ppLabelMap, Vf2ppLabelMap>
  vf2pp_init_r_new_r_inout(const GT &query, ConstRef<ArrayXi> qlbl,
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

  template <IsoMapType kMt>
  bool vf2pp_r_matches(const Vf2ppLabels &r_node, const ArrayXi &label_tmp) {
    return absl::c_none_of(r_node, [&](std::pair<int, int> p) {
      return kMt == IsoMapType::kGraph ? label_tmp[p.first] != 0
                                       : label_tmp[p.first] > 0;
    });
  }
}  // namespace internal

template <IsoMapType kMt, class GT, class GU>
class VF2pp {
private:
  const GT &query() const { return *query_; }
  const GU &target() const { return *target_; }

  auto query_tmp() { return node_tmp_.head(query_->size()); }

  ArrayXi &conn() { return node_tmp_; }

  auto curr_node() const { return query_->node(order_[depth_]); }

  int mapped_node() const { return node_map_[curr_node().id()]; }

  auto query_target_ait() const { return query_target_ait_[depth_]; }
  void update_ait(typename GT::const_adjacency_iterator qa,
                  typename GU::const_adjacency_iterator ta) {
    query_target_ait_[depth_] = { qa, ta };
    ta.end() ? --depth_ : ++depth_;
  }

  ArrayXi &r_inout_cnt() { return label_tmp1_; }
  ArrayXi &r_new_cnt() { return label_tmp2_; }

public:
  template <class AL1, class AL2>
  VF2pp(const GT &query, const GU &target, AL1 &&query_lbl, AL2 &&target_lbl)
      : query_(&query), target_(&target),  //
        qlbl_(std::forward<AL1>(query_lbl)),
        tlbl_(std::forward<AL2>(target_lbl)),
        node_map_(ArrayXi::Constant(query.size(), -1)),
        edge_map_(ArrayXi::Constant(query.num_edges(), -1)),
        query_target_ait_(query.size(),
                          { query.adj_end(0), target.adj_end(0) }),
        node_tmp_(ArrayXi::Zero(target.size())) {
    ABSL_DCHECK(!query.empty());
    ABSL_DCHECK_LE(query.size(), target.size());

    const int nlabel = nuri::max(qlbl_.maxCoeff(), tlbl_.maxCoeff()) + 1;

    label_tmp1_ = ArrayXi::Zero(nlabel);
    label_tmp2_ = ArrayXi::Zero(nlabel);

    order_ = internal::vf2pp_init_order(query, qlbl_, tlbl_, label_tmp1_,
                                        query_tmp());
    std::tie(r_inout_, r_new_) = internal::vf2pp_init_r_new_r_inout(
        query, qlbl_, order_, r_inout_cnt(), r_new_cnt(), query_tmp());

    query_tmp().setZero();
  }

  template <class NodeMatch, class EdgeMatch>
  bool next(const NodeMatch &node_match, const EdgeMatch &edge_match) {
    while (depth_ >= 0) {
      if (depth_ == query().size()) {
        --depth_;

        if (map_remaining_edges(edge_match)) {
#ifdef NURI_DEBUG
          first_ = false;
#endif
          return true;
        }
      }

      const auto qn = curr_node();
      const int ti = mapped_node();

      auto [qa, ta] = query_target_ait();
      if (!ta.end()) {
        ABSL_DCHECK(!qa.end());
        ABSL_DCHECK_NE(ti, ta->src().id());

        sub_pair(qn.id(), ti, qa->eid());
        ++ta;
      } else {
        if (ti < 0) {
          auto qb = absl::c_find_if(qn, [&](auto nei) {
            return node_map_[nei.dst().id()] >= 0;
          });
          if (!qb.end())
            qa = qb->dst().find_adjacent(qn);
        } else {
          sub_pair(qn.id(), ti, qa.end() ? -1 : qa->eid());
        }

        if (qa.end() || ti >= 0) {
          const int tj =
              std::find_if(target().begin() + ti + 1, target().end(),
                           [&](auto tn) {
                             bool candidate = kMt == IsoMapType::kSubgraph
                                                  ? conn()[tn.id()] >= 0
                                                  : conn()[tn.id()] == 0;
                             return candidate && feas(qn, tn, node_match);
                           })
              - target().begin();

          if (tj < target().size()) {
            add_pair(qn.id(), tj, -1, -1);
            ++depth_;
          } else {
            --depth_;
          }

          continue;
        }

        ta = target().adj_begin(node_map_[qa->src().id()]);
      }

      ABSL_DCHECK(ta.end() || !qa.end());

      for (; !ta.end(); ++ta) {
        auto tn = ta->dst();
        if (conn()[tn.id()] > 0  //
            && feas(qn, tn, node_match)
            && edge_match(query().edge(qa->eid()), target().edge(ta->eid()))) {
          add_pair(qn.id(), tn.id(), qa->eid(), ta->eid());
          break;
        }
      }

      update_ait(qa, ta);
    }

    return false;
  }

  const ArrayXi &node_map() const & { return node_map_; }
  ArrayXi &&node_map() && { return std::move(node_map_); }

  const ArrayXi &edge_map() const & { return edge_map_; }
  ArrayXi &&edge_map() && { return std::move(edge_map_); }

private:
  void add_pair(const int qn, const int tn, const int qe, const int te) {
    ABSL_DCHECK_GE(tn, 0);
    ABSL_DCHECK_LT(tn, target().size());

    node_map_[qn] = tn;
    if (qe >= 0)
      edge_map_[qe] = te;

    conn()[tn] = -1;
    for (auto nei: target().node(tn)) {
      if (conn()[nei.dst().id()] != -1)
        ++conn()[nei.dst().id()];
    }
  }

  void sub_pair(const int qn, const int tn, const int qe) {
    ABSL_DCHECK_GE(tn, 0);
    ABSL_DCHECK_LT(tn, target().size());

    node_map_[qn] = -1;
    if (qe >= 0)
      edge_map_[qe] = -1;

    conn()[tn] = 0;
    for (auto nei: target().node(tn)) {
      int curr_conn = conn()[nei.dst().id()];
      if (curr_conn > 0) {
        --conn()[nei.dst().id()];
      } else if (curr_conn == -1) {
        ++conn()[tn];
      }
    }
  }

  bool cut_by_labels(const typename GT::ConstNodeRef qn,
                     const typename GU::ConstNodeRef tn) {
    // zero init
    for (auto [lbl, _]: r_inout_[qn.id()])
      r_inout_cnt()[lbl] = 0;
    if constexpr (kMt != IsoMapType::kSubgraph) {
      for (auto [lbl, _]: r_new_[qn.id()])
        r_new_cnt()[lbl] = 0;
    }

    for (auto nei: tn) {
      const int curr = nei.dst().id();
      if (conn()[curr] > 0) {
        --r_inout_cnt()[tlbl_[curr]];
      } else if constexpr (kMt != IsoMapType::kSubgraph) {
        if (conn()[curr] == 0)
          --r_new_cnt()[tlbl_[curr]];
      }
    }

    for (auto [lbl, cnt]: r_inout_[qn.id()])
      r_inout_cnt()[lbl] += cnt;
    if constexpr (kMt != IsoMapType::kSubgraph) {
      for (auto [lbl, cnt]: r_new_[qn.id()])
        r_new_cnt()[lbl] += cnt;
    }

    const bool r_inout_match =
        internal::vf2pp_r_matches<kMt>(r_inout_[qn.id()], r_inout_cnt());
    if constexpr (kMt == IsoMapType::kSubgraph)
      return r_inout_match;

    return r_inout_match
           && internal::vf2pp_r_matches<kMt>(r_new_[qn.id()], r_new_cnt());
  }

  template <class NodeMatch>
  bool feas(const typename GT::ConstNodeRef qn,
            const typename GU::ConstNodeRef tn, const NodeMatch &node_match) {
    if (qlbl_[qn.id()] != tlbl_[tn.id()])
      return false;

    for (auto qnei: qn)
      if (node_map_[qnei.dst().id()] >= 0)
        --conn()[node_map_[qnei.dst().id()]];

    bool is_iso = true;
    for (auto tnei: tn) {
      const int curr_conn = conn()[tnei.dst().id()];
      if (curr_conn < -1)
        ++conn()[tnei.dst().id()];
      else if constexpr (kMt != IsoMapType::kSubgraph) {
        if (curr_conn == -1) {
          is_iso = false;
          break;
        }
      }
    }

    if (!is_iso) {
      for (auto qnei: qn) {
        const int ti = node_map_[qnei.dst().id()];
        if (ti >= 0)
          conn()[ti] = -1;
      }
      return false;
    }

    for (auto qnei: qn) {
      const int ti = node_map_[qnei.dst().id()];
      if (ti < 0 || conn()[ti] == -1)
        continue;

      const int curr_conn = conn()[ti];
      conn()[ti] = -1;
      if constexpr (kMt != IsoMapType::kSubgraph)
        return false;

      if (curr_conn < -1)
        return false;
    }

    return node_match(qn, tn) && cut_by_labels(qn, tn);
  }

  bool is_stale(const typename GT::ConstEdgeRef qe,
                const typename GU::ConstEdgeRef te) {
    const int curr_src = node_map_[qe.src().id()],
              curr_dst = node_map_[qe.dst().id()];

    const bool stale =
        (curr_src != te.src().id() || curr_dst != te.dst().id())
        && (curr_src != te.dst().id() || curr_dst != te.src().id());

#ifdef NURI_DEBUG
    ABSL_DCHECK(!first_ || !stale) << qe.id() << " " << te.id();
#endif

    return stale;
  }

  template <class EdgeMatch>
  bool map_remaining_edges(const EdgeMatch &edge_match) {
    for (auto qe: query().edges()) {
      if (edge_map_[qe.id()] >= 0
          && !is_stale(qe, target().edge(edge_map_[qe.id()]))) {
        continue;
      }

      auto teit = target().find_edge(target().node(node_map_[qe.src().id()]),
                                     target().node(node_map_[qe.dst().id()]));
      ABSL_DCHECK(teit != target().edge_end());

      if (!edge_match(qe, *teit))
        return false;

      edge_map_[qe.id()] = teit->id();
    }

    return true;
  }

  const GT *query_;
  const GU *target_;

  Eigen::Ref<const ArrayXi> qlbl_, tlbl_;

  ArrayXi node_map_, edge_map_;

  // query.size()
  ArrayXi order_;
  internal::Vf2ppLabelMap r_inout_, r_new_;
  std::vector<std::pair<typename GT::const_adjacency_iterator,
                        typename GU::const_adjacency_iterator>>
      query_target_ait_;

  // max(label) + 1
  ArrayXi label_tmp1_, label_tmp2_;

  // max(query.size(), target.size())
  ArrayXi node_tmp_;

  int depth_ = 0;
#ifdef NURI_DEBUG
  bool first_ = true;
#endif
};

template <IsoMapType kMt, class GT, class GU, class AL1, class AL2>
VF2pp<kMt, GT, GU> make_vf2pp(const GT &query, const GU &target, AL1 &&qlbl,
                              AL2 &&tlbl) {
  return VF2pp<kMt, GT, GU>(query, target, std::forward<AL1>(qlbl),
                            std::forward<AL2>(tlbl));
}

struct VF2ppResult {
  ArrayXi node_map;
  ArrayXi edge_map;
  bool found;
};

template <IsoMapType kMt, class GT, class GU, class AL1, class AL2,
          class NodeMatch, class EdgeMatch>
VF2ppResult vf2pp(const GT &query, const GU &target, AL1 &&qlbl, AL2 &&tlbl,
                  const NodeMatch &node_match, const EdgeMatch &edge_match) {
  VF2pp<kMt, GT, GU> vf2pp = make_vf2pp<kMt>(
      query, target, std::forward<AL1>(qlbl), std::forward<AL2>(tlbl));
  bool found = vf2pp.next(node_match, edge_match);
  return { std::move(vf2pp).node_map(), std::move(vf2pp).edge_map(), found };
}

template <class GT, class GU, class NodeMatch, class EdgeMatch, class AL1,
          class AL2>
VF2ppResult vf2pp(const GT &query, const GU &target, AL1 &&qlbl, AL2 &&tlbl,
                  const NodeMatch &node_match, const EdgeMatch &edge_match,
                  IsoMapType mt) {
  switch (mt) {
  case IsoMapType::kSubgraph:
    return vf2pp<IsoMapType::kSubgraph>(                   //
        query, target,                                     //
        std::forward<AL1>(qlbl), std::forward<AL2>(tlbl),  //
        node_match, edge_match);
  case IsoMapType::kInduced:
    return vf2pp<IsoMapType::kInduced>(                    //
        query, target,                                     //
        std::forward<AL1>(qlbl), std::forward<AL2>(tlbl),  //
        node_match, edge_match);
  case IsoMapType::kGraph:
    return vf2pp<IsoMapType::kGraph>(                      //
        query, target,                                     //
        std::forward<AL1>(qlbl), std::forward<AL2>(tlbl),  //
        node_match, edge_match);
  }

  ABSL_UNREACHABLE();
}

template <IsoMapType kMt, class GT, class GU, class NodeMatch, class EdgeMatch>
VF2ppResult vf2pp(const GT &query, const GU &target,
                  const NodeMatch &node_match, const EdgeMatch &edge_match) {
  ArrayXi label = ArrayXi::Zero(nuri::max(query.size(), target.size()));
  return vf2pp<kMt>(query, target, label.head(query.size()),
                    label.head(target.size()), node_match, edge_match);
}

template <class GT, class GU, class NodeMatch, class EdgeMatch>
VF2ppResult vf2pp(const GT &query, const GU &target,
                  const NodeMatch &node_match, const EdgeMatch &edge_match,
                  IsoMapType mt) {
  ArrayXi label = ArrayXi::Zero(nuri::max(query.size(), target.size()));
  return vf2pp(query, target, label.head(query.size()),
               label.head(target.size()), node_match, edge_match, mt);
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_VF2PP_H_ */

//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_VF2PP_H_
#define NURI_CORE_GRAPH_VF2PP_H_

//! @cond
#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
#include "nuri/core/graph.h"
#include "nuri/utils.h"

namespace nuri {
/**
 * @brief The type of isomorphic map to find.
 */
enum class IsoMapType : int {
  //! Subgraph isomorphism.
  kSubgraph,
  //! Induced subgraph isomorphism.
  kInduced,
  //! Graph isomorphism.
  kGraph,
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

/**
 * @brief An implementation of VF2++ algorithm for (sub)graph isomorphism.
 *
 * @tparam kMt The type of subgraph mapping.
 * @tparam GT The type of the query graph ("needle").
 * @tparam GU The type of the target graph ("haystack").
 *
 * This class implements the VF2++ algorithm for subgraph isomorphism. The
 * implementation is based on the reference implementation in LEMON project.
 * Here, we provide two extra functionalities:
 *
 *   1. Support for extra node and edge matching functions.
 *   2. In addition to the node mapping, the algorithm also returns the edge
 *      mapping.
 *
 * See the following paper for details of the algorithm.
 *
 * Reference:
 * - A J&uuml;ttner and P Madarasi. *Discrete Appl. Math.* **2018**, *247*,
 *   69-81.
 *   DOI:[10.1016/j.dam.2018.02.018](https://doi.org/10.1016/j.dam.2018.02.018)
 *   @cite core:graph:vf2pp
 *
 * Here follows the full license text for the LEMON project:
 *
 * \code{.unparsed}
 * Copyright (C) 2003-2012 Egervary Jeno Kombinatorikus Optimalizalasi
 * Kutatocsoport (Egervary Combinatorial Optimization Research Group,
 * EGRES).
 *
 * ===========================================================================
 * Boost Software License, Version 1.0
 * ===========================================================================
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 * \endcode
 */
template <IsoMapType kMt, class GT, class GU>
class VF2pp {
private:
  const GT &query() const { return *query_; }
  const GU &target() const { return *target_; }

  auto query_tmp() { return node_tmp_.head(query_->size()); }

  ArrayXi &conn() { return node_tmp_; }

  auto curr_node(int i) const { return query_->node(order_[i]); }
  auto curr_node() const { return curr_node(depth_); }

  int mapped_node(int i) const { return node_map_[curr_node(i).id()]; }
  int mapped_node() const { return mapped_node(depth_); }

  auto query_target_ait() const { return query_target_ait_[depth_]; }
  void update_ait(typename GT::const_adjacency_iterator qa,
                  typename GU::const_adjacency_iterator ta) {
    query_target_ait_[depth_] = { qa, ta };
  }

  ArrayXi &r_inout_cnt() { return label_tmp1_; }
  ArrayXi &r_new_cnt() { return label_tmp2_; }

public:
  /**
   * @brief Prepare VF2++ algorithm.
   *
   * @tparam AL1 The type of query node label array. Must be representable as
   *         an Eigen 1D array of integers.
   * @tparam AL2 The type of target node label array. Must be representable as
   *         an Eigen 1D array of integers.
   * @param query The query graph.
   * @param target The target graph.
   * @param query_lbl The query node labels. Target nodes with the different
   *        labels will not be considered for matching.
   * @param target_lbl The target node labels. Target nodes must have the same
   *        label with the corresponding query nodes to be considered for
   *        matching.
   *
   * @note Both graphs must not be empty, the query graph must be smaller
   *       or equal to the target graph, and each node labels must not contain
   *       non-negative labels and have same size as the corresponding graph.
   *       The behavior is undefined if any of the conditions are not met.
   * @sa make_vf2pp()
   */
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

  /**
   * @brief Find next subgraph mapping.
   *
   * @tparam NodeMatch A binary predicate that takes two nodes and returns
   *         whether the nodes are matched.
   * @tparam EdgeMatch A binary predicate that takes two edges and returns
   *         whether the edges are matched.
   * @param node_match The node matching function. Must return true if the
   *        (query, target) node pair is matched. Is guaranteed to be called
   *        with topologically feasible and label-matching nodes.
   * @param edge_match The edge matching function. Must return true if the
   *        (query, target) edge pair is matched. Each ends of the edge are
   *        guaranteed to be already-matched nodes. Note that depending on the
   *        query and target graphs, the edge may be matched in reverse order,
   *        i.e., target edge might have source and target nodes swapped
   *        compared to the query edge.
   * @return Whether the mapping is found. Once this function returns false,
   *         all subsequent calls will also return false.
   *
   * @note This function will find overlapping matches by default. If
   *       non-overlapping matches are required, supply node_match and
   *       edge_match functions that return false for already matched nodes
   *       and/or edges.
   * @sa vf2pp()
   */
  template <class NodeMatch, class EdgeMatch>
  bool next(const NodeMatch &node_match, const EdgeMatch &edge_match) {
    while (depth_ >= 0) {
      if (depth_ == query().size()) {
        if (map_remaining_edges(edge_match)) {
#ifdef NURI_DEBUG
          first_ = false;
#endif
          ++depth_;
          return true;
        }

        --depth_;
      } else if (depth_ > query().size()) {
#ifdef NURI_DEBUG
        ABSL_DCHECK(!first_);
#endif
        // previous call to next() returned true
        clear_stale_maps(node_match, edge_match);
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
      ta.end() ? --depth_ : ++depth_;
    }

    return false;
  }

  /**
   * @brief Get current node mapping.
   * @return The node mapping, where the index is the query node ID and the
   *         value is the target node ID.
   * @pre Last call to next() returned true, otherwise the return value is
   *      unspecified.
   */
  const ArrayXi &node_map() const & { return node_map_; }

  /**
   * @brief Move out current node mapping.
   * @return The node mapping, where the index is the query node ID and the
   *         value is the target node ID.
   * @pre Last call to next() returned true, otherwise the return value is
   *      unspecified.
   * @note This function invalidates the current object. Only edge_map() can
   *       be called after this function.
   */
  ArrayXi &&node_map() && { return std::move(node_map_); }

  /**
   * @brief Get current edge mapping.
   * @return The edge mapping, where the index is the query edge ID and the
   *         value is the target edge ID.
   * @pre Last call to next() returned true, otherwise the return value is
   *      unspecified.
   */
  const ArrayXi &edge_map() const & { return edge_map_; }

  /**
   * @brief Move out current edge mapping.
   * @return The edge mapping, where the index is the query edge ID and the
   *         value is the target edge ID.
   * @pre Last call to next() returned true, otherwise the return value is
   *      unspecified.
   * @note This function invalidates the current object. Only node_map() can
   *       be called after this function.
   */
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

  constexpr bool is_stale(const typename GT::ConstEdgeRef qe,
                          const typename GU::ConstEdgeRef te) const {
#ifdef NURI_DEBUG
    const int curr_src = node_map_[qe.src().id()],
              curr_dst = node_map_[qe.dst().id()];

    const bool stale =
        (curr_src != te.src().id() || curr_dst != te.dst().id())
        && (curr_src != te.dst().id() || curr_dst != te.src().id());

    ABSL_DCHECK(!first_ || !stale) << qe.id() << " " << te.id();

    return stale;
#else
    static_cast<void>(*this);
    static_cast<void>(qe);
    static_cast<void>(te);

    return false;
#endif
  }

  template <class EdgeMatch>
  bool map_remaining_edges(const EdgeMatch &edge_match) {
    for (auto qe: query().edges()) {
      if (edge_map_[qe.id()] >= 0) {
        ABSL_DCHECK(!is_stale(qe, target().edge(edge_map_[qe.id()])));
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

  template <class NodeMatch, class EdgeMatch>
  void clear_stale_maps(const NodeMatch &node_match,
                        const EdgeMatch &edge_match) {
    // clear first, some edges are mapped after BFS search
    edge_map_.setConstant(-1);

    // Starting from the deepest stack frame, find first stale node/edge match.
    // Must be started from the deepest frame because the following frames
    // depend on the mapping of the current frame. This will stop at first
    // iteration if non-overlapping match was requested.
    //
    // Last entry will be popped off after this function returns, so must be
    // excluded in this function (< query().size() - 1)
    int i = 0;
    for (; i < query().size() - 1; ++i) {
      int ti = mapped_node(i);
      ABSL_DCHECK_GE(ti, 0);
      // check for stale nodes, if non-overlapping node match was requested
      if (!node_match(curr_node(i), target().node(ti)))
        break;

      auto [qa, ta] = query_target_ait_[i];
      if (qa.end())
        continue;

      // check for stale edges, if non-overlapping edge match was requested
      if (!edge_match(query().edge(qa->eid()), target().edge(ta->eid())))
        break;

      ABSL_DCHECK(!ta.end());
      edge_map_[qa->eid()] = ta->eid();
    }

    // Simulate stack pop when any of node/edge match fails
    // Same here, last edge will be popped off so must be excluded (depth_ > i)
    for (depth_ = query().size() - 1; depth_ > i; --depth_) {
      auto qn = curr_node();
      int ti = mapped_node();
      auto [qa, _] = query_target_ait();

      ABSL_DCHECK_GE(ti, 0);
      sub_pair(qn.id(), ti, qa.end() ? -1 : qa->eid());
      update_ait(query().adj_end(0), target().adj_end(0));
    }
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

/**
 * @brief Prepare VF2++ algorithm.
 *
 * @param query The query graph ("needle").
 * @param target The target graph ("haystack").
 * @param qlbl The query node labels.
 * @param tlbl The target node labels.
 * @return The constructed VF2++ algorithm object.
 *
 * @note Both graphs must not be empty, the query graph must be smaller
 *       or equal to the target graph, and each node labels must not contain
 *       non-negative labels and have same size as the corresponding graph.
 *       The behavior is undefined if any of the conditions are not met.
 * @sa VF2pp::VF2pp()
 */
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

/**
 * @brief Find a query-to-target subgraph mapping.
 *
 * @param query The query graph ("needle").
 * @param target The target graph ("haystack").
 * @param qlbl The query node labels.
 * @param tlbl The target node labels.
 * @param node_match The node matching function. Must return true if the (query,
 *        target) node pair is matched. Is guaranteed to be called with
 *        topologically feasible and label-matching nodes.
 * @param edge_match The edge matching function. Must return true if the (query,
 *        target) edge pair is matched. Each ends of the edge are guaranteed to
 *        be already-matched nodes. Note that depending on the query and target
 *        graphs, the edge may be matched in reverse order, i.e., target edge
 *        might have source and target nodes swapped compared to the query edge.
 * @return The node and edge mapping, and whether the mapping is found. Mappings
 *         are valid only if the last element is true.
 *
 * @note Both graphs must not be empty, the query graph must be smaller
 *       or equal to the target graph, and each node labels must not contain
 *       non-negative labels and have same size as the corresponding graph.
 *       The behavior is undefined if any of the conditions are not met.
 * @sa VF2pp::next()
 */
template <IsoMapType kMt, class GT, class GU, class AL1, class AL2,
          class NodeMatch, class EdgeMatch>
VF2ppResult vf2pp(const GT &query, const GU &target, AL1 &&qlbl, AL2 &&tlbl,
                  const NodeMatch &node_match, const EdgeMatch &edge_match) {
  VF2pp<kMt, GT, GU> vf2pp = make_vf2pp<kMt>(
      query, target, std::forward<AL1>(qlbl), std::forward<AL2>(tlbl));
  bool found = vf2pp.next(node_match, edge_match);
  return { std::move(vf2pp).node_map(), std::move(vf2pp).edge_map(), found };
}

/**
 * @brief Find a query-to-target subgraph mapping.
 *
 * @param query The query graph ("needle").
 * @param target The target graph ("haystack").
 * @param qlbl The query node labels.
 * @param tlbl The target node labels.
 * @param node_match The node matching function. Must return true if the (query,
 *        target) node pair is matched. Is guaranteed to be called with
 *        topologically feasible and label-matching nodes.
 * @param edge_match The edge matching function. Must return true if the (query,
 *        target) edge pair is matched. Each ends of the edge are guaranteed to
 *        be already-matched nodes. Note that depending on the query and target
 *        graphs, the edge may be matched in reverse order, i.e., target edge
 *        might have source and target nodes swapped compared to the query edge.
 * @param mt The type of subgraph mapping.
 * @return The node and edge mapping, and whether the mapping is found. Mappings
 *         are valid only if the last element is true.
 *
 * @note Both graphs must not be empty, the query graph must be smaller
 *       or equal to the target graph, and each node labels must not contain
 *       non-negative labels and have same size as the corresponding graph.
 *       The behavior is undefined if any of the conditions are not met.
 * @sa VF2pp::next()
 */
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

/**
 * @brief Find a query-to-target subgraph mapping assuming all node labels are
 *        equal.
 *
 * @param query The query graph ("needle").
 * @param target The target graph ("haystack").
 * @param node_match The node matching function. Must return true if the (query,
 *        target) node pair is matched. Is guaranteed to be called with
 *        topologically feasible and label-matching nodes.
 * @param edge_match The edge matching function. Must return true if the (query,
 *        target) edge pair is matched. Each ends of the edge are guaranteed to
 *        be already-matched nodes. Note that depending on the query and target
 *        graphs, the edge may be matched in reverse order, i.e., target edge
 *        might have source and target nodes swapped compared to the query edge.
 * @return The node and edge mapping, and whether the mapping is found. Mappings
 *         are valid only if the last element is true.
 *
 * @note Both graphs must not be empty, the query graph must be smaller
 *       or equal to the target graph, and each node labels must not contain
 *       non-negative labels and have same size as the corresponding graph.
 *       The behavior is undefined if any of the conditions are not met.
 * @sa VF2pp::next()
 */
template <IsoMapType kMt, class GT, class GU, class NodeMatch, class EdgeMatch>
VF2ppResult vf2pp(const GT &query, const GU &target,
                  const NodeMatch &node_match, const EdgeMatch &edge_match) {
  ArrayXi label = ArrayXi::Zero(nuri::max(query.size(), target.size()));
  return vf2pp<kMt>(query, target, label.head(query.size()),
                    label.head(target.size()), node_match, edge_match);
}

/**
 * @brief Find a query-to-target subgraph mapping assuming all node labels are
 *        equal.
 *
 * @param query The query graph ("needle").
 * @param target The target graph ("haystack").
 * @param node_match The node matching function. Must return true if the (query,
 *        target) node pair is matched. Is guaranteed to be called with
 *        topologically feasible and label-matching nodes.
 * @param edge_match The edge matching function. Must return true if the (query,
 *        target) edge pair is matched. Each ends of the edge are guaranteed to
 *        be already-matched nodes. Note that depending on the query and target
 *        graphs, the edge may be matched in reverse order, i.e., target edge
 *        might have source and target nodes swapped compared to the query edge.
 * @param mt The type of subgraph mapping.
 * @return The node and edge mapping, and whether the mapping is found. Mappings
 *         are valid only if the last element is true.
 *
 * @note Both graphs must not be empty, the query graph must be smaller
 *       or equal to the target graph, and each node labels must not contain
 *       non-negative labels and have same size as the corresponding graph.
 *       The behavior is undefined if any of the conditions are not met.
 * @sa VF2pp::next()
 */
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

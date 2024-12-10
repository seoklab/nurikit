//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_VF2PP_H_
#define NURI_CORE_GRAPH_VF2PP_H_

/// @cond
#include <algorithm>
#include <utility>

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
  kEdgeSubgraph,  // Subgraph isomorphism
  kNodeSubgraph,  // Induced subgraph isomorphism
  kIsomorphism,   // Graph isomorphism
};

namespace internal {
  template <class NT, class ET, class AL>
  int vf2pp_process_bfs_tree(const Graph<NT, ET> &query, ArrayXi &order,
                             ArrayXb &visited, ArrayXi &curr_conn,
                             AL &query_cnts, int root, int i) {
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

  template <class NT, class ET>
  ArrayXi vf2pp_init_order(const Graph<NT, ET> &query, const ArrayXi &qlbl,
                           const ArrayXi &tlbl, ArrayXi &target_lcnt,
                           ArrayXi &curr_conn, ArrayXb &visited) {
    target_lcnt.setZero();
    for (int l: tlbl)
      ++target_lcnt[l];

    auto query_cnts = target_lcnt(qlbl);

    ArrayXi order = ArrayXi::Constant(query.size(), -1);
    curr_conn.setZero();
    visited.setZero();

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

  using Vf2ppLabelMap = std::vector<std::vector<std::pair<int, int>>>;

  template <class NT, class ET>
  std::pair<Vf2ppLabelMap, Vf2ppLabelMap>
  vf2pp_init_r_new_r_inout(const Graph<NT, ET> &query, const ArrayXi &qlbl,
                           const ArrayXi &order, ArrayXi &r_inout,
                           ArrayXi &r_new, ArrayXi &visit_count) {
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
        } else if (r_new[curr_lbl] > 0) {
          r_new_labels[i].push_back({ curr_lbl, r_new[curr_lbl] });
        }

        if (visit_count[curr] >= 0)
          ++visit_count[curr];
      }
    }

    return { r_inout_labels, r_new_labels };
  }

  template <class NT, class ET>
  void vf2pp_add_pair(const Graph<NT, ET> &target, ArrayXi &q2t, ArrayXi &conn,
                      const int qi, const int ti) {
    ABSL_DCHECK_GE(ti, 0);
    ABSL_DCHECK_LT(ti, target.size());

    conn[ti] = -1;
    q2t[qi] = ti;
    for (auto nei: target.node(ti)) {
      if (conn[nei.dst().id()] != -1)
        ++conn[nei.dst().id()];
    }
  }

  template <class NT, class ET>
  void vf2pp_sub_pair(const Graph<NT, ET> &target, ArrayXi &q2t, ArrayXi &conn,
                      const int qi, const int ti) {
    ABSL_DCHECK_GE(ti, 0);
    ABSL_DCHECK_LT(ti, target.size());

    q2t[qi] = -1;
    conn[ti] = 0;

    for (auto nei: target.node(ti)) {
      int curr_conn = conn[nei.dst().id()];
      if (curr_conn > 0) {
        --conn[nei.dst().id()];
      } else if (curr_conn == -1) {
        ++conn[ti];
      }
    }
  }

  template <MappingType kMt, class NT, class ET>
  bool vf2pp_cut_by_labels(typename Graph<NT, ET>::ConstNodeRef qn,
                           typename Graph<NT, ET>::ConstNodeRef tn,
                           ArrayXi &label_tmp1, ArrayXi &label_tmp2,
                           const ArrayXi &tlbl, const Vf2ppLabelMap &r_inout,
                           const Vf2ppLabelMap &r_new, const ArrayXi &conn) {
    // zero-init
    label_tmp1(tlbl)(as_index(tn)).setZero();
    for (auto [lbl, cnt]: r_inout[qn.id()])
      label_tmp1[lbl] = 0;
    if constexpr (kMt != MappingType::kEdgeSubgraph) {
      label_tmp2(tlbl)(as_index(tn)).setZero();
      for (auto [lbl, cnt]: r_new[qn.id()])
        label_tmp2[lbl] = 0;
    }

    for (auto tnei: tn) {
      const int curr = tnei.dst().id();
      if (conn[curr] > 0)
        --label_tmp1[tlbl[curr]];
      else if constexpr (kMt != MappingType::kEdgeSubgraph) {
        if (conn[curr] == 0)
          --label_tmp2[tlbl[curr]];
      }
    }

    for (auto [lbl, cnt]: r_inout[qn.id()])
      label_tmp1[lbl] += cnt;
    if constexpr (kMt != MappingType::kEdgeSubgraph) {
      for (auto [lbl, cnt]: r_new[qn.id()])
        label_tmp2[lbl] += cnt;
    }

    if constexpr (kMt == MappingType::kEdgeSubgraph) {
      return absl::c_none_of(r_inout[qn.id()], [&](std::pair<int, int> p) {
        return label_tmp1[p.first] > 0;
      });
    }

    if constexpr (kMt == MappingType::kNodeSubgraph) {
      return absl::c_none_of(r_inout[qn.id()],
                             [&](std::pair<int, int> p) {
                               return label_tmp1[p.first] > 0;
                             })
             && absl::c_none_of(r_new[qn.id()], [&](std::pair<int, int> p) {
                  return label_tmp2[p.first] > 0;
                });
    }

    if constexpr (kMt == MappingType::kIsomorphism) {
      return absl::c_none_of(r_inout[qn.id()],
                             [&](std::pair<int, int> p) {
                               return label_tmp1[p.first] != 0;
                             })
             && absl::c_none_of(r_new[qn.id()], [&](std::pair<int, int> p) {
                  return label_tmp2[p.first] != 0;
                });
    }

    ABSL_UNREACHABLE();
  }

  template <MappingType kMt, class NT, class ET>
  bool vf2pp_feas(typename Graph<NT, ET>::ConstNodeRef qn,
                  typename Graph<NT, ET>::ConstNodeRef tn, const ArrayXi &qlbl,
                  const ArrayXi &tlbl, ArrayXi &q2t, ArrayXi &conn,
                  ArrayXi &label_tmp1, ArrayXi &label_tmp2,
                  const Vf2ppLabelMap &r_inout, const Vf2ppLabelMap &r_new) {
    if (qlbl[qn.id()] != tlbl[tn.id()])
      return false;

    for (auto qnei: qn)
      if (q2t[qnei.dst().id()] >= 0)
        --conn[q2t[qnei.dst().id()]];

    bool is_iso = true;
    for (auto tnei: tn) {
      const int curr_conn = conn[tnei.dst().id()];
      if (curr_conn < -1)
        ++conn[tnei.dst().id()];
      else if constexpr (kMt != MappingType::kEdgeSubgraph) {
        if (curr_conn == -1)
          is_iso = false;
      }
    }

    if (!is_iso) {
      for (auto qnei: qn) {
        const int ti = q2t[qnei.dst().id()];
        if (ti >= 0)
          conn[ti] = -1;
      }
      return false;
    }

    for (auto qnei: qn) {
      const int ti = q2t[qnei.dst().id()];
      if (ti < 0 || conn[ti] == -1)
        continue;

      const int curr_conn = conn[ti];
      conn[ti] = -1;
      if constexpr (kMt == MappingType::kEdgeSubgraph) {
        if (curr_conn < -1)
          return false;
      } else {
        return false;
      }
    }

    return vf2pp_cut_by_labels<kMt, NT, ET>(qn, tn, label_tmp1, label_tmp2,
                                            tlbl, r_inout, r_new, conn);
  }

  template <MappingType kMt, class NT, class ET>
  std::pair<ArrayXi, bool>
  vf2pp_ext_match(const Graph<NT, ET> &query, const Graph<NT, ET> &target,
                  const ArrayXi &qlbl, const ArrayXi &tlbl,
                  const ArrayXi &order, ArrayXi &conn, ArrayXi &label_tmp1,
                  ArrayXi &label_tmp2, const Vf2ppLabelMap &r_inout,
                  const Vf2ppLabelMap &r_new) {
    ArrayXi q2t = ArrayXi::Constant(query.size(), -1);

    Array2Xi src_nei = Array2Xi::Constant(2, target.size(), -1);
    conn.setZero();
    label_tmp1.setZero();
    label_tmp2.setZero();
    for (int depth = 0; depth >= 0;) {
      if (depth == order.size())
        return { q2t, true };

      auto qn = query.node(order[depth]);

      const int ti = q2t[qn.id()];
      int tj = src_nei(0, depth), tk = src_nei(1, depth);
      if (tk >= 0) {
        ABSL_DCHECK_NE(ti, tj);
        vf2pp_sub_pair(target, q2t, conn, qn.id(), ti);
        ++tk;
      } else {
        int qk = 0;
        if (ti < 0) {
          qk = absl::c_find_if(
                   qn, [&](auto nei) { return q2t[nei.dst().id()] >= 0; })
               - qn.begin();
        }

        if (qk == qn.degree() || ti >= 0) {
          int ti2 = 0;
          if (ti >= 0) {
            ti2 = ti + 1;
            vf2pp_sub_pair(target, q2t, conn, qn.id(), ti);
          }

          if constexpr (kMt != MappingType::kEdgeSubgraph) {
            ti2 = std::find_if(target.begin() + ti2, target.end(),
                               [&](auto tn) {
                                 return conn[tn.id()] == 0
                                        && vf2pp_feas<kMt, NT, ET>(
                                            qn, tn, qlbl, tlbl, q2t, conn,
                                            label_tmp1, label_tmp2, r_inout,
                                            r_new);
                               })
                  - target.begin();
          } else {
            ti2 = std::find_if(target.begin() + ti2, target.end(),
                               [&](auto tn) {
                                 return conn[tn.id()] >= 0
                                        && vf2pp_feas<kMt, NT, ET>(
                                            qn, tn, qlbl, tlbl, q2t, conn,
                                            label_tmp1, label_tmp2, r_inout,
                                            r_new);
                               })
                  - target.begin();
          }

          if (ti2 < target.size()) {
            vf2pp_add_pair(target, q2t, conn, qn.id(), ti2);
            ++depth;
          } else {
            --depth;
          }

          continue;
        }

        tj = q2t[qn[qk].dst().id()];
        tk = 0;
      }

      for (; tk < target.degree(tj); ++tk) {
        auto tn = target.node(tj)[tk].dst();
        if (conn[tn.id()] > 0
            && vf2pp_feas<kMt, NT, ET>(qn, tn, qlbl, tlbl, q2t, conn,
                                       label_tmp1, label_tmp2, r_inout,
                                       r_new)) {
          vf2pp_add_pair(target, q2t, conn, qn.id(), tn.id());
          break;
        }
      }

      if (tk < target.degree(tj)) {
        src_nei(1, depth++) = tk;
      } else {
        src_nei.col(depth--).fill(-1);
      }
    }

    return { q2t, false };
  }
}  // namespace internal

template <class NT, class ET>
std::pair<ArrayXi, bool> vf2pp(const Graph<NT, ET> &query,
                               const Graph<NT, ET> &target, const ArrayXi &qlbl,
                               const ArrayXi &tlbl, MappingType mt) {
  const int nlabel = nuri::max(qlbl.maxCoeff(), tlbl.maxCoeff()) + 1;

  ArrayXi label_tmp1(nlabel), label_tmp2(nlabel);

  ArrayXi query_tmp(query.size());
  ArrayXb visited(query.size());

  ArrayXi target_tmp(target.size());

  // _order
  ArrayXi order = internal::vf2pp_init_order(query, qlbl, tlbl, label_tmp1,
                                             query_tmp, visited);
  // _rInOutLabels1, _rNewLabels1
  auto [r_inout, r_new] = internal::vf2pp_init_r_new_r_inout(
      query, qlbl, order, label_tmp1, label_tmp2, query_tmp);

  switch (mt) {
  case MappingType::kEdgeSubgraph:
    return internal::vf2pp_ext_match<MappingType::kEdgeSubgraph>(
        query, target, qlbl, tlbl, order, target_tmp, label_tmp1, label_tmp2,
        r_inout, r_new);
  case MappingType::kNodeSubgraph:
    return internal::vf2pp_ext_match<MappingType::kNodeSubgraph>(
        query, target, qlbl, tlbl, order, target_tmp, label_tmp1, label_tmp2,
        r_inout, r_new);
  case MappingType::kIsomorphism:
    return internal::vf2pp_ext_match<MappingType::kIsomorphism>(
        query, target, qlbl, tlbl, order, target_tmp, label_tmp1, label_tmp2,
        r_inout, r_new);
  }

  ABSL_UNREACHABLE();
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_VF2PP_H_ */

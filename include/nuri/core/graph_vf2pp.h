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

  template <class NT, class ET, class AL1, class AL2, class AL3>
  ArrayXi vf2pp_init_order(const Graph<NT, ET> &query, const AL1 &qlbl,
                           const AL2 &tlbl, ArrayXi &target_lcnt,
                           AL3 &curr_conn) {
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

  template <class NT, class ET, class AL1, class AL2>
  std::pair<Vf2ppLabelMap, Vf2ppLabelMap>
  vf2pp_init_r_new_r_inout(const Graph<NT, ET> &query, const AL1 &qlbl,
                           const ArrayXi &order, ArrayXi &r_inout,
                           ArrayXi &r_new, AL2 &visit_count) {
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

  template <class NT, class ET, class AL>
  void vf2pp_add_pair(const Graph<NT, ET> &target, ArrayXi &q2t, AL &conn,
                      const int qi, const int ti) {
    ABSL_DCHECK_GE(ti, 0);
    ABSL_DCHECK_LT(ti, target.size());

    q2t[qi] = ti;
    conn[ti] = -1;

    for (auto nei: target.node(ti)) {
      if (conn[nei.dst().id()] != -1)
        ++conn[nei.dst().id()];
    }
  }

  template <class NT, class ET, class AL>
  void vf2pp_sub_pair(const Graph<NT, ET> &target, ArrayXi &q2t, AL &conn,
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

  template <MappingType kMt>
  bool vf2pp_r_matches(const std::vector<std::pair<int, int>> &r_node,
                       const ArrayXi &label_tmp) {
    return absl::c_none_of(r_node, [&](std::pair<int, int> p) {
      return kMt == MappingType::kIsomorphism ? label_tmp[p.first] != 0
                                              : label_tmp[p.first] > 0;
    });
  }

  template <MappingType kMt, class NT, class ET, class AL1, class AL2>
  bool vf2pp_cut_by_labels(typename Graph<NT, ET>::ConstNodeRef tn,
                           const AL1 &tlbl, ArrayXi &r_inout_cnt,
                           ArrayXi &r_new_cnt, const Vf2ppLabels &r_inout,
                           const Vf2ppLabels &r_new, const AL2 &conn) {
    // zero init
    r_inout_cnt(tlbl)(as_index(tn)).setZero();
    for (auto [lbl, cnt]: r_inout)
      r_inout_cnt[lbl] = 0;
    if constexpr (kMt != MappingType::kSubgraph) {
      r_new_cnt(tlbl)(as_index(tn)).setZero();
      for (auto [lbl, cnt]: r_new)
        r_new_cnt[lbl] = 0;
    }

    for (auto tnei: tn) {
      const int curr = tnei.dst().id();
      if (conn[curr] > 0)
        --r_inout_cnt[tlbl[curr]];
      else if constexpr (kMt != MappingType::kSubgraph) {
        if (conn[curr] == 0)
          --r_new_cnt[tlbl[curr]];
      }
    }

    for (auto [lbl, cnt]: r_inout)
      r_inout_cnt[lbl] += cnt;
    if constexpr (kMt != MappingType::kSubgraph) {
      for (auto [lbl, cnt]: r_new)
        r_new_cnt[lbl] += cnt;
    }

    const bool r_inout_match = vf2pp_r_matches<kMt>(r_inout, r_inout_cnt);
    if constexpr (kMt == MappingType::kSubgraph)
      return r_inout_match;

    return r_inout_match && vf2pp_r_matches<kMt>(r_new, r_new_cnt);
  }

  template <MappingType kMt, class BinaryPred, class N1, class E1, class N2,
            class E2, class AL1, class AL2, class AL3>
  bool vf2pp_feas(typename Graph<N1, E1>::ConstNodeRef qn,
                  typename Graph<N2, E2>::ConstNodeRef tn,
                  const BinaryPred &match, const AL1 &qlbl, const AL2 &tlbl,
                  ArrayXi &q2t, AL3 &conn, ArrayXi &r_inout_cnt,
                  ArrayXi &r_new_cnt, const Vf2ppLabelMap &r_inout,
                  const Vf2ppLabelMap &r_new) {
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
      else if constexpr (kMt != MappingType::kSubgraph) {
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
      if constexpr (kMt == MappingType::kSubgraph) {
        if (curr_conn < -1)
          return false;
      } else {
        return false;
      }
    }

    return match(qn, tn)
           && vf2pp_cut_by_labels<kMt, N2, E2>(tn, tlbl, r_inout_cnt, r_new_cnt,
                                               r_inout[qn.id()], r_new[qn.id()],
                                               conn);
  }

  template <MappingType kMt, class BinaryPred, class N1, class E1, class N2,
            class E2, class AL1, class AL2, class AL3>
  std::pair<ArrayXi, bool>
  vf2pp_ext_match(const Graph<N1, E1> &query, const Graph<N2, E2> &target,
                  const BinaryPred &match, const AL1 &qlbl, const AL2 &tlbl,
                  const ArrayXi &order, AL3 &conn, ArrayXi &r_inout_cnt,
                  ArrayXi &r_new_cnt, const Vf2ppLabelMap &r_inout,
                  const Vf2ppLabelMap &r_new) {
    ArrayXi q2t = ArrayXi::Constant(query.size(), -1);

    Array2Xi src_nei = Array2Xi::Constant(2, target.size(), -1);
    for (int depth = 0; depth >= 0;) {
      if (depth == order.size())
        return { q2t, true };

      const auto qn = query.node(order[depth]);
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
        } else {
          vf2pp_sub_pair(target, q2t, conn, qn.id(), ti);
        }

        if (qk == qn.degree() || ti >= 0) {
          const int ti2 =
              std::find_if(
                  target.begin() + ti + 1, target.end(),
                  [&](auto tn) {
                    bool candidate = kMt == MappingType::kSubgraph
                                         ? conn[tn.id()] >= 0
                                         : conn[tn.id()] == 0;
                    return candidate
                           && vf2pp_feas<kMt, BinaryPred, N1, E1, N2, E2>(
                               qn, tn, match, qlbl, tlbl, q2t, conn,
                               r_inout_cnt, r_new_cnt, r_inout, r_new);
                  })
              - target.begin();

          if (ti2 < target.size()) {
            vf2pp_add_pair(target, q2t, conn, qn.id(), ti2);
            ++depth;
          } else {
            --depth;
          }

          continue;
        }

        src_nei(0, depth) = tj = q2t[qn[qk].dst().id()];
        tk = 0;
      }

      for (; tk < target.degree(tj); ++tk) {
        auto tn = target.node(tj)[tk].dst();
        if (conn[tn.id()] > 0
            && vf2pp_feas<kMt, BinaryPred, N1, E1, N2, E2>(
                qn, tn, match, qlbl, tlbl, q2t, conn, r_inout_cnt, r_new_cnt,
                r_inout, r_new)) {
          vf2pp_add_pair(target, q2t, conn, qn.id(), tn.id());
          break;
        }
      }

      if (tk < target.degree(tj)) {
        src_nei(1, depth++) = tk;
      } else {
        src_nei(1, depth--) = -1;
      }
    }

    return { q2t, false };
  }
}  // namespace internal

template <class N1, class E1, class N2, class E2, class BinaryPred, class AL1,
          class AL2>
std::pair<ArrayXi, bool> vf2pp(const Graph<N1, E1> &query,
                               const Graph<N2, E2> &target,
                               const BinaryPred &match, const AL1 &qlbl,
                               const AL2 &tlbl, MappingType mt) {
  const int nlabel = nuri::max(qlbl.maxCoeff(), tlbl.maxCoeff()) + 1;

  ArrayXi label_tmp1 = ArrayXi::Zero(nlabel),
          label_tmp2 = ArrayXi::Zero(nlabel);
  ArrayXi node_tmp = ArrayXi::Zero(nuri::max(query.size(), target.size()));

  auto query_tmp = node_tmp.head(query.size());
  // _order
  ArrayXi order =
      internal::vf2pp_init_order(query, qlbl, tlbl, label_tmp1, query_tmp);
  // _rInOutLabels1, _rNewLabels1
  auto [r_inout, r_new] = internal::vf2pp_init_r_new_r_inout(
      query, qlbl, order, label_tmp1, label_tmp2, query_tmp);

  node_tmp.head(nuri::min(query.size(), target.size())).setZero();
  auto target_tmp = node_tmp.head(target.size());
  switch (mt) {
  case MappingType::kSubgraph:
    return internal::vf2pp_ext_match<MappingType::kSubgraph>(
        query, target, match, qlbl, tlbl, order, target_tmp, label_tmp1,
        label_tmp2, r_inout, r_new);
  case MappingType::kInduced:
    return internal::vf2pp_ext_match<MappingType::kInduced>(
        query, target, match, qlbl, tlbl, order, target_tmp, label_tmp1,
        label_tmp2, r_inout, r_new);
  case MappingType::kIsomorphism:
    return internal::vf2pp_ext_match<MappingType::kIsomorphism>(
        query, target, match, qlbl, tlbl, order, target_tmp, label_tmp1,
        label_tmp2, r_inout, r_new);
  }

  ABSL_UNREACHABLE();
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_VF2PP_H_ */

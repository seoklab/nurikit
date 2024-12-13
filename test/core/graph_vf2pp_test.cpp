//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/graph_vf2pp.h"

#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/graph.h"

namespace nuri {
namespace {
using GT = Graph<int, int>;

int lid(const GT &g, int i) {
  return g[i].data();
}

auto lemon_add_edge(GT &g, int src, int dst, int data) {
  return g.add_edge(lid(g, src), lid(g, dst), data);
}

GT lemon_c_n(int n) {
  GT g;
  for (int i = n - 1; i >= 0; --i)
    g.add_node(i);

  for (int i = n - 1; i >= 0; --i)
    lemon_add_edge(g, i, (i + 1) % n, i);

  return g;
}

GT lemon_p_n(int n) {
  GT g;
  for (int i = n - 1; i >= 0; --i)
    g.add_node(i);

  for (int i = n - 2; i >= 0; --i)
    lemon_add_edge(g, i, i + 1, i);

  return g;
}

GT lemon_petersen() {
  GT g;
  for (int i = 9; i >= 0; --i)
    g.add_node(i);

  lemon_add_edge(g, 6, 8, 14);
  lemon_add_edge(g, 9, 7, 13);
  lemon_add_edge(g, 9, 6, 12);
  lemon_add_edge(g, 5, 7, 11);
  lemon_add_edge(g, 5, 8, 10);
  lemon_add_edge(g, 4, 9, 9);
  lemon_add_edge(g, 3, 8, 8);
  lemon_add_edge(g, 2, 7, 7);
  lemon_add_edge(g, 1, 6, 6);
  lemon_add_edge(g, 0, 5, 5);
  lemon_add_edge(g, 4, 0, 4);
  lemon_add_edge(g, 3, 4, 3);
  lemon_add_edge(g, 2, 3, 2);
  lemon_add_edge(g, 1, 2, 1);
  lemon_add_edge(g, 0, 1, 0);

  return g;
}

GT lemon_c5_petersen() {
  GT g;
  for (int i = 14; i >= 0; --i)
    g.add_node(i);

  lemon_add_edge(g, 11, 13, 19);
  lemon_add_edge(g, 14, 12, 18);
  lemon_add_edge(g, 14, 11, 17);
  lemon_add_edge(g, 10, 12, 16);
  lemon_add_edge(g, 10, 13, 15);
  lemon_add_edge(g, 9, 14, 14);
  lemon_add_edge(g, 8, 13, 13);
  lemon_add_edge(g, 7, 12, 12);
  lemon_add_edge(g, 6, 11, 11);
  lemon_add_edge(g, 5, 10, 10);
  lemon_add_edge(g, 9, 5, 9);
  lemon_add_edge(g, 8, 9, 8);
  lemon_add_edge(g, 7, 8, 7);
  lemon_add_edge(g, 6, 7, 6);
  lemon_add_edge(g, 5, 6, 5);

  lemon_add_edge(g, 4, 0, 4);
  lemon_add_edge(g, 3, 4, 3);
  lemon_add_edge(g, 2, 3, 2);
  lemon_add_edge(g, 1, 2, 1);
  lemon_add_edge(g, 0, 1, 0);

  return g;
}

TEST(VF2ppComponentTest, ProcessBfsTreePetersenSingleLabel) {
  GT g = lemon_petersen();

  ArrayXi order = ArrayXi::Constant(g.size(), -1);
  ArrayXb visited = ArrayXb::Zero(g.size());
  ArrayXi curr_conn = ArrayXi::Zero(g.size());

  ArrayXi lcnt(1);
  lcnt[0] = 10;
  ArrayXi lbl = ArrayXi::Zero(g.size());
  auto query_cnts = lcnt(lbl);

  internal::Vf2ppDegreeHelper<GT> degrees(g);
  const int ret = internal::vf2pp_process_bfs_tree(g, degrees, order, visited,
                                                   curr_conn, query_cnts, 0, 0);
  EXPECT_EQ(ret, 10);

  ArrayXi expected_order(10);
  expected_order << 9, 7, 6, 4, 5, 2, 8, 1, 0, 3;
  for (int i = 0; i < order.size(); ++i)
    EXPECT_EQ(lid(g, order[i]), expected_order[i]);
}

TEST(VF2ppComponentTest, ProcessBfsTreePetersenMultiLabel) {
  GT g = lemon_petersen();

  ArrayXi order = ArrayXi::Constant(g.size(), -1);
  ArrayXb visited = ArrayXb::Zero(g.size());
  ArrayXi curr_conn = ArrayXi::Zero(g.size());

  ArrayXi lcnt(2);
  lcnt[0] = 4;
  lcnt[1] = 6;

  ArrayXi qlbl(g.size());
  qlbl.head(6).setOnes();
  qlbl.tail(4).setZero();
  auto query_cnts = lcnt(qlbl);

  internal::Vf2ppDegreeHelper<GT> degrees(g);
  const int ret = internal::vf2pp_process_bfs_tree(g, degrees, order, visited,
                                                   curr_conn, query_cnts, 6, 0);
  EXPECT_EQ(ret, 10);

  ArrayXi expected_order(10);
  expected_order << 3, 2, 4, 8, 6, 5, 9, 0, 7, 1;
  for (int i = 0; i < order.size(); ++i)
    EXPECT_EQ(lid(g, order[i]), expected_order[i]);
}

TEST(VF2ppComponentTest, InitOrderC5Petersen) {
  GT g = lemon_c5_petersen();

  ArrayXi lbl(g.size());
  lbl << 1, 1, 1, 1, 1,  //
      1, 0, 0, 0, 0,     //
      4, 3, 2, 1, 0;

  ArrayXi lcnt = ArrayXi::Zero(5), curr_conn = ArrayXi::Zero(g.size());
  ArrayXi order = internal::vf2pp_init_order(g, lbl, lbl, lcnt, curr_conn);

  ArrayXi expected_order(15);
  expected_order << 4, 3, 0, 1, 2, 8, 7, 9, 13, 11, 10, 14, 5, 12, 6;
  for (int i = 0; i < order.size(); ++i)
    EXPECT_EQ(lid(g, order[i]), expected_order[i]);
}

TEST(VF2ppComponentTest, InitRNewRInoutC5Petersen) {
  GT g = lemon_c5_petersen();
  ArrayXi lbl(g.size());
  lbl << 1, 1, 1, 1, 1,  //
      1, 0, 0, 0, 0,     //
      4, 3, 2, 1, 0;

  ArrayXi ltmp1(5), ltmp2(5);
  ArrayXi visit_count(g.size());
  ArrayXi order(15);
  order << 10, 11, 14, 13, 12, 6, 7, 5, 1, 3, 4, 0, 9, 2, 8;

  auto [r_inout, r_new] = internal::vf2pp_init_r_new_r_inout(
      g, lbl, order, ltmp1, ltmp2, visit_count);

  internal::Vf2ppLabelMap r_new_expected {
    { { 1, 1 } },
    {},
    {},
    { { 2, 1 } },
    { { 0, 1 }, { 3, 1 } },
    {},
    {},
    { { 1, 1 }, { 0, 1 } },
    { { 1, 2 }, { 0, 1 } },
    { { 1, 1 }, { 0, 1 } },
    {},
    {},
    {},
    { { 1, 2 } },
    {},
  };
  internal::Vf2ppLabelMap r_inout_expected {
    {},
    { { 2, 1 } },
    {},
    {},
    {},
    { { 0, 1 } },
    {},
    {},
    {},
    {},
    { { 1, 1 }, { 0, 1 } },
    { { 1, 1 }, { 0, 1 } },
    {},
    {},
    { { 1, 1 } },
  };

  constexpr auto vector_eq = absl::c_equal<std::vector<std::pair<int, int>>,
                                           std::vector<std::pair<int, int>>>;

  for (int i = 0; i < g.size(); ++i) {
    int j = lid(g, i);
    EXPECT_PRED2(vector_eq, r_new[i], r_new_expected[j]);
    EXPECT_PRED2(vector_eq, r_inout[i], r_inout_expected[j]);
  }
}

constexpr auto dummy_match = [](auto, auto) { return true; };

TEST(VF2ppSingleLabelSimpleMatchTest, PetersenSubgraph) {
  GT target = lemon_petersen();
  ArrayXi expected_map(15);

  auto run_vf2pp = [&](const GT &query) {
    return vf2pp(query, target, dummy_match, dummy_match,
                 IsoMapType::kSubgraph);
  };

  {
    GT c5 = lemon_c_n(5);
    auto [nmap, emap, ok] = run_vf2pp(c5);
    EXPECT_TRUE(ok);

    expected_map.head(5) << 7, 5, 8, 6, 9;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(c5, i)]), expected_map[i]);

    // lemon: 4-0, 3-4, 2-3, 1-2, 0-1
    expected_map.head(5) << 1, 2, 0, 4, 3;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }

  {
    GT c7 = lemon_c_n(7);
    auto [_1, _2, ok] = run_vf2pp(c7);
    EXPECT_FALSE(ok);
  }

  {
    GT p10 = lemon_p_n(10);
    auto [nmap, emap, ok] = run_vf2pp(p10);
    EXPECT_TRUE(ok);

    expected_map.head(10) << 2, 1, 0, 4, 3, 8, 5, 7, 9, 6;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(p10, i)]), expected_map[i]);

    // lemon: 8-9, 7-8, ..., 0-1
    expected_map.head(9) << 2, 1, 3, 4, 6, 11, 10, 14, 13;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }

  {
    GT c10 = lemon_c_n(10);
    auto [_1, _2, ok] = run_vf2pp(c10);
    EXPECT_FALSE(ok);
  }

  {
    auto [nmap, emap, ok] = run_vf2pp(target);
    EXPECT_TRUE(ok);

    absl::c_iota(expected_map, 0);

    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(target, i)]), expected_map[i]);

    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }
}

TEST(VF2ppSingleLabelSimpleMatchTest, PetersenInduced) {
  GT target = lemon_petersen();
  ArrayXi expected_map(15);

  auto run_vf2pp = [&](const GT &query) {
    return vf2pp(query, target, dummy_match, dummy_match, IsoMapType::kInduced);
  };

  {
    GT c5 = lemon_c_n(5);
    auto [nmap, emap, ok] = run_vf2pp(c5);
    EXPECT_TRUE(ok);

    expected_map.head(5) << 7, 5, 8, 6, 9;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(c5, i)]), expected_map[i]);

    // lemon: 4-0, 3-4, 2-3, 1-2, 0-1
    expected_map.head(5) << 1, 2, 0, 4, 3;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }

  {
    GT c7 = lemon_c_n(7);
    auto [_1, _2, ok] = run_vf2pp(c7);
    EXPECT_FALSE(ok);
  }

  {
    GT p10 = lemon_p_n(10);
    auto [_1, _2, ok] = run_vf2pp(p10);
    EXPECT_FALSE(ok);
  }

  {
    GT c10 = lemon_c_n(10);
    auto [_1, _2, ok] = run_vf2pp(c10);
    EXPECT_FALSE(ok);
  }

  {
    auto [nmap, emap, ok] = run_vf2pp(target);
    EXPECT_TRUE(ok);

    absl::c_iota(expected_map, 0);

    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(target, i)]), expected_map[i]);

    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }
}

TEST(VF2ppSingleLabelSimpleMatchTest, PetersenGraph) {
  GT target = lemon_petersen();
  ArrayXi expected_map(15);

  auto run_vf2pp = [&](const GT &query) {
    return vf2pp(query, target, dummy_match, dummy_match, IsoMapType::kGraph);
  };

  {
    GT c5 = lemon_c_n(5);
    auto [_1, _2, ok] = run_vf2pp(c5);
    EXPECT_FALSE(ok);
  }

  {
    GT c7 = lemon_c_n(7);
    auto [_1, _2, ok] = run_vf2pp(c7);
    EXPECT_FALSE(ok);
  }

  {
    GT p10 = lemon_p_n(10);
    auto [_1, _2, ok] = run_vf2pp(p10);
    EXPECT_FALSE(ok);
  }

  {
    GT c10 = lemon_c_n(10);
    auto [_1, _2, ok] = run_vf2pp(c10);
    EXPECT_FALSE(ok);
  }

  {
    auto [nmap, emap, ok] = run_vf2pp(target);
    EXPECT_TRUE(ok);

    absl::c_iota(expected_map, 0);

    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(target, i)]), expected_map[i]);

    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }
}

TEST(VF2ppSingleLabelSimpleMatchTest, C10P10) {
  GT petersen = lemon_petersen(), p10 = lemon_p_n(10), target = lemon_c_n(10);
  ArrayXi expected_map(10);

  auto run_vf2pp = [&](const GT &query, IsoMapType mt) {
    return vf2pp(query, target, dummy_match, dummy_match, mt);
  };

  {
    auto [_1, _2, ok] = run_vf2pp(petersen, IsoMapType::kSubgraph);
    EXPECT_FALSE(ok);
  }

  {
    auto [nmap, emap, ok] = run_vf2pp(p10, IsoMapType::kSubgraph);
    EXPECT_TRUE(ok);

    expected_map << 7, 6, 5, 4, 3, 2, 1, 0, 9, 8;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(p10, i)]), expected_map[i]);

    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], nmap[i]);
  }

  {
    auto [_1, _2, ok] = run_vf2pp(p10, IsoMapType::kInduced);
    EXPECT_FALSE(ok);
  }
  {
    auto [_1, _2, ok] = run_vf2pp(p10, IsoMapType::kGraph);
    EXPECT_FALSE(ok);
  }

  {
    auto [_1, _2, ok] =
        // NOLINTNEXTLINE(*-suspicious-call-argument)
        vf2pp(target, p10, dummy_match, dummy_match, IsoMapType::kGraph);
    EXPECT_FALSE(ok);
  }

  {
    auto [nmap, emap, ok] = run_vf2pp(target, IsoMapType::kGraph);
    EXPECT_TRUE(ok);

    absl::c_iota(expected_map, 0);

    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(target, nmap[lid(target, i)]), expected_map[i]);

    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }
}

TEST(VF2ppMultiLabelSimpleMatchTest, C5Petersen) {
  GT c5 = lemon_c_n(5), petersen = lemon_petersen();

  ArrayXi c5_lbl(5);
  c5_lbl << 4, 3, 2, 1, 0;

  ArrayXi petersen_lbl1(10);
  petersen_lbl1.head(6).fill(1);
  petersen_lbl1.tail(4).fill(0);

  ArrayXi petersen_lbl2(10);
  petersen_lbl2.head(5) = c5_lbl;
  petersen_lbl2.tail(5) = c5_lbl;

  {
    auto [_1, _2, ok] = vf2pp(c5, petersen, c5_lbl, petersen_lbl1, dummy_match,
                              dummy_match, IsoMapType::kSubgraph);
    EXPECT_FALSE(ok);
  }

  {
    auto [nmap, emap, ok] = vf2pp(c5, petersen, c5_lbl, petersen_lbl2,
                                  dummy_match, dummy_match,
                                  IsoMapType::kSubgraph);
    ASSERT_TRUE(ok);

    ArrayXi expected_map(5);

    expected_map << 0, 1, 2, 3, 4;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(lid(petersen, nmap[lid(c5, i)]), expected_map[i]);

    expected_map << 10, 11, 12, 13, 14;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]);
  }
}

TEST(VF2ppMultiLabelSimpleMatchTest, PetersenPetersen) {
  GT petersen = lemon_petersen();

  ArrayXi petersen_lbl1(10);
  petersen_lbl1.head(6).fill(1);
  petersen_lbl1.tail(4).fill(0);

  ArrayXi petersen_lbl2(10);
  petersen_lbl2.head(5) << 4, 3, 2, 1, 0;
  petersen_lbl2.tail(5) = petersen_lbl2.head(5);

  ArrayXi expected_map(15);
  absl::c_iota(expected_map, 0);

  for (IsoMapType mt:
       { IsoMapType::kSubgraph, IsoMapType::kInduced, IsoMapType::kGraph }) {
    {
      auto [nmap, emap, ok] = vf2pp(petersen, petersen, petersen_lbl1,
                                    petersen_lbl1, dummy_match, dummy_match,
                                    mt);
      ASSERT_TRUE(ok) << static_cast<int>(mt);

      for (int i = 0; i < nmap.size(); ++i)
        EXPECT_EQ(lid(petersen, nmap[lid(petersen, i)]), expected_map[i])
            << static_cast<int>(mt);

      for (int i = 0; i < emap.size(); ++i)
        EXPECT_EQ(emap[i], expected_map[i]) << static_cast<int>(mt);
    }

    {
      auto [nmap, emap, ok] = vf2pp(petersen, petersen, petersen_lbl2,
                                    petersen_lbl2, dummy_match, dummy_match,
                                    mt);
      ASSERT_TRUE(ok) << static_cast<int>(mt);

      for (int i = 0; i < nmap.size(); ++i)
        EXPECT_EQ(lid(petersen, nmap[lid(petersen, i)]), expected_map[i])
            << static_cast<int>(mt);

      for (int i = 0; i < emap.size(); ++i)
        EXPECT_EQ(emap[i], expected_map[i]) << static_cast<int>(mt);
    }

    {
      auto [_1, _2, ok] = vf2pp(petersen, petersen, petersen_lbl1,
                                petersen_lbl2, dummy_match, dummy_match, mt);
      EXPECT_FALSE(ok) << static_cast<int>(mt);
    }

    {
      auto [_1, _2, ok] = vf2pp(petersen, petersen, petersen_lbl2,
                                petersen_lbl1, dummy_match, dummy_match, mt);
      EXPECT_FALSE(ok) << static_cast<int>(mt);
    }
  }
}

TEST(VF2ppPetersenSubgraphSimpleMatchTest, SingleLabel) {
  GT petersen = lemon_petersen();
  ArrayXi idxs(5), expected_map(5);

  //                                1   3   9   10
  //                              0   1   2   4   3
  // {0, 2, 4, 5, 9}, connection: 0 - 2 - 4 - 9 - 5
  auto p5 = subgraph_from_edges(petersen, { 1, 3, 9, 10 });
  //                                4   3   7   12  6
  //                              0   2   1   4   3   0
  // {1, 2, 4, 6, 7}, connection: 1 - 4 - 2 - 7 - 6 - 1
  auto c5 = subgraph_from_edges(petersen, { 3, 4, 6, 7, 12 });

  idxs << 0, 1, 2, 4, 3;
  {
    auto [nmap, emap, ok] =
        vf2pp(p5, c5, dummy_match, dummy_match, IsoMapType::kSubgraph);
    EXPECT_TRUE(ok);

    expected_map << 3, 0, 2, 1, 4;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(nmap[idxs[i]], expected_map[i]) << i;

    expected_map.head(4) << 2, 1, 0, 3;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]) << i;
  }

  {
    auto [_1, _2, ok] =
        vf2pp(p5, c5, dummy_match, dummy_match, IsoMapType::kInduced);
    EXPECT_FALSE(ok);
  }

  {
    auto [_1, _2, ok] =
        vf2pp(c5, p5, dummy_match, dummy_match, IsoMapType::kSubgraph);
    EXPECT_FALSE(ok);
  }

  {
    auto [nmap, emap, ok] =
        vf2pp(c5, c5, dummy_match, dummy_match, IsoMapType::kGraph);
    EXPECT_TRUE(ok);

    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(nmap[i], i) << i;

    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], i) << i;
  }

  {
    auto [nmap, emap, ok] =
        vf2pp(p5, petersen, dummy_match, dummy_match, IsoMapType::kSubgraph);
    EXPECT_TRUE(ok);

    expected_map << 3, 0, 2, 4, 1;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(nmap[idxs[i]], expected_map[i]) << i;

    expected_map.head(4) << 2, 1, 3, 4;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]) << i;
  }

  {
    auto [nmap, emap, ok] =
        vf2pp(c5, petersen, dummy_match, dummy_match, IsoMapType::kSubgraph);
    EXPECT_TRUE(ok);

    idxs << 0, 2, 1, 4, 3;
    expected_map << 0, 2, 4, 1, 3;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(nmap[idxs[i]], expected_map[i]) << i;

    idxs << 1, 0, 3, 4, 2;
    expected_map << 1, 3, 4, 0, 2;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[idxs[i]], expected_map[i]) << i;
  }
}

TEST(VF2ppPetersenSubgraphSimpleMatchTest, MultiLabel) {
  GT petersen = lemon_petersen();
  ArrayXi qlbl(5), tlbl(10), idxs(5), expected_map(5);

  //                                1   3   9   10
  //  0  1  2  1  1               0   1   2   4   3
  // {0, 2, 4, 5, 9}, connection: 0 - 2 - 4 - 9 - 5
  auto p5 = subgraph_from_edges(petersen, { 1, 3, 9, 10 });
  //                                4   3   7   12  6
  //  0  2  1  1  1               0   2   1   4   3   0
  // {1, 2, 4, 6, 7}, connection: 1 - 4 - 2 - 7 - 6 - 1
  auto c5 = subgraph_from_edges(petersen, { 3, 4, 6, 7, 12 });

  {
    qlbl << 0, 1, 2, 1, 1;
    tlbl.head(5) << 0, 2, 1, 1, 1;
    auto [nmap, emap, ok] = vf2pp(p5, c5, qlbl, tlbl.head(5), dummy_match,
                                  dummy_match, IsoMapType::kSubgraph);
    EXPECT_TRUE(ok);

    idxs << 0, 1, 2, 4, 3;
    expected_map << 0, 2, 1, 4, 3;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(nmap[idxs[i]], expected_map[i]) << i;

    expected_map.head(4) << 1, 0, 3, 4;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]) << i;
  }

  {
    qlbl << 0, 0, 1, 1, 1;
    auto [_1, _2, ok] = vf2pp(p5, c5, qlbl, qlbl, dummy_match, dummy_match,
                              IsoMapType::kSubgraph);
    EXPECT_FALSE(ok);
  }

  tlbl.head(4).setZero();
  tlbl.tail(6).setOnes();

  {
    auto matcher = make_vf2pp<IsoMapType::kSubgraph>(p5, petersen,
                                                     tlbl(p5.node_ids()), tlbl);

    bool ok = matcher.next(dummy_match, dummy_match);
    EXPECT_TRUE(ok);
    auto map = matcher.node_map();
    //  0  0  1  1  1               0   1   2   4   3
    // {0, 2, 4, 5, 9}, connection: 0 - 2 - 4 - 9 - 5
    idxs << 0, 1, 2, 4, 3;
    expected_map.head(5) << 2, 0, 5, 9, 4;
    for (int i = 0; i < map.size(); ++i)
      EXPECT_EQ(map[idxs[i]], expected_map[i]) << i;

    bool found_original = false;
    while (matcher.next(dummy_match, dummy_match)) {
      if (absl::c_equal(p5.node_ids(), matcher.node_map()))
        found_original = true;

      for (int i = 0; i < p5.num_edges(); ++i) {
        int src = matcher.node_map()[p5.edge(i).src().id()],
            dst = matcher.node_map()[p5.edge(i).dst().id()];

        auto edge = petersen.edge(matcher.edge_map()[i]);

        EXPECT_TRUE((edge.src().id() == src && edge.dst().id() == dst)
                    || (edge.src().id() == dst && edge.dst().id() == src));
      }
    }
    EXPECT_TRUE(found_original);
  }

  {
    auto [nmap, emap, ok] = vf2pp(c5, petersen, tlbl(c5.node_ids()), tlbl,
                                  dummy_match, dummy_match,
                                  IsoMapType::kSubgraph);
    EXPECT_TRUE(ok);

    //  0  0  1  1  1               0   2   1   4   3   0
    // {1, 2, 4, 6, 7}, connection: 1 - 4 - 2 - 7 - 6 - 1
    idxs.head(5) << 0, 2, 1, 4, 3;
    expected_map.head(5) << 1, 4, 2, 7, 6;
    for (int i = 0; i < nmap.size(); ++i)
      EXPECT_EQ(nmap[idxs[i]], expected_map[i]) << i;

    expected_map.head(5) << 3, 4, 6, 7, 12;
    for (int i = 0; i < emap.size(); ++i)
      EXPECT_EQ(emap[i], expected_map[i]) << i;
  }
}

TEST(VF2ppPetersenSubgraphSingleLabelTest, NodeNoOverlap) {
  GT petersen = lemon_petersen();
  auto p2 = subgraph_from_edges(petersen, { 1 });

  ArrayXi qlbl = ArrayXi::Zero(2), tlbl = ArrayXi::Zero(10);
  ArrayXi node_cnt = ArrayXi::Zero(10);

  auto vf2 = make_vf2pp<IsoMapType::kSubgraph>(p2, petersen, qlbl, tlbl);
  while (vf2.next([&](auto, auto tn) { return node_cnt[tn.id()] == 0; },
                  dummy_match)) {
    node_cnt(vf2.node_map()) += 1;
  }

  for (auto cnt: node_cnt)
    EXPECT_EQ(cnt, 1);
}

TEST(VF2ppPetersenSubgraphSingleLabelTest, EdgeNoOverlap) {
  GT petersen = lemon_petersen();
  auto p4 = subgraph_from_edges(petersen, { 1, 7, 12 });

  ArrayXi qlbl = ArrayXi::Zero(4), tlbl = ArrayXi::Zero(10);
  ArrayXi edge_cnt = ArrayXi::Zero(15);

  auto vf2 = make_vf2pp<IsoMapType::kSubgraph>(p4, petersen, qlbl, tlbl);
  while (vf2.next(dummy_match,
                  [&](auto, auto te) { return edge_cnt[te.id()] < 2; })) {
    edge_cnt(vf2.edge_map()) += 1;
  }

  for (auto edge: petersen.edges())
    EXPECT_EQ(edge_cnt[edge.id()], 2) << edge.id();
}

TEST(VF2ppPetersenSubgraphSingleLabelTest, NodeEdgeNoOverlap) {
  GT petersen = lemon_petersen();
  auto p4 = subgraph_from_edges(petersen, { 1, 7, 12 });

  ArrayXi qlbl = ArrayXi::Zero(4), tlbl = ArrayXi::Zero(10);
  ArrayXi node_cnt = ArrayXi::Zero(10), edge_cnt = ArrayXi::Zero(15);

  auto vf2 = make_vf2pp<IsoMapType::kSubgraph>(p4, petersen, qlbl, tlbl);
  while (vf2.next(  //
      [&](auto, auto tn) { return node_cnt[tn.id()] < tn.degree() * 2; },
      [&](auto, auto te) { return edge_cnt[te.id()] < 2; })) {
    node_cnt(vf2.node_map()) += 1;
    edge_cnt(vf2.edge_map()) += 1;
  }

  for (auto node: petersen)
    EXPECT_LE(node_cnt[node.id()], node.degree() * 2) << node.id();

  for (auto edge: petersen.edges())
    EXPECT_EQ(edge_cnt[edge.id()], 2) << edge.id();
}
}  // namespace
}  // namespace nuri

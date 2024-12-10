//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/graph_vf2pp.h"

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/graph.h"

namespace nuri {
namespace {
using GT = Graph<int, int>;

// lemon node id
int lid(const GT &g, int i) {
  return g[i].data();
}

auto lemon_add_edge(GT &g, int src, int dst, int data) {
  return g.add_edge(lid(g, src), lid(g, dst), data);
}

Graph<int, int> lemon_petersen() {
  Graph<int, int> g;
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

TEST(VF2PPComponentTest, ProcessBfsTreePetersenSingleLabel) {
  Graph<int, int> g = lemon_petersen();

  ArrayXi order = ArrayXi::Constant(g.size(), -1);
  ArrayXb visited = ArrayXb::Zero(g.size());
  ArrayXi curr_conn = ArrayXi::Zero(g.size());

  ArrayXi target_lcnt(1);
  target_lcnt[0] = 10;
  ArrayXi qlbl = ArrayXi::Zero(g.size());
  auto query_cnts = target_lcnt(qlbl);

  const int ret = internal::vf2pp_process_bfs_tree(g, order, visited, curr_conn,
                                                   query_cnts, 0, 0);
  EXPECT_EQ(ret, 10);

  ArrayXi expected_order(10);
  expected_order << 9, 7, 6, 4, 5, 2, 8, 1, 0, 3;
  for (int i = 0; i < order.size(); ++i)
    EXPECT_EQ(lid(g, order[i]), expected_order[i]);
}

TEST(VF2PPComponentTest, ProcessBfsTreePetersenMultiLabel) {
  Graph<int, int> g = lemon_petersen();

  ArrayXi order = ArrayXi::Constant(g.size(), -1);
  ArrayXb visited = ArrayXb::Zero(g.size());
  ArrayXi curr_conn = ArrayXi::Zero(g.size());

  ArrayXi target_lcnt(2);
  target_lcnt[0] = 4;
  target_lcnt[1] = 6;

  ArrayXi qlbl(g.size());
  qlbl.head(6).setOnes();
  qlbl.tail(4).setZero();

  auto query_cnts = target_lcnt(qlbl);

  const int ret = internal::vf2pp_process_bfs_tree(g, order, visited, curr_conn,
                                                   query_cnts, 6, 0);
  EXPECT_EQ(ret, 10);

  ArrayXi expected_order(10);
  expected_order << 3, 2, 4, 8, 6, 5, 9, 0, 7, 1;
  for (int i = 0; i < order.size(); ++i)
    EXPECT_EQ(lid(g, order[i]), expected_order[i]);
}

// TEST(VF2PPTest, Playground) {
//   Graph<int, int> g1;
//   g1.add_node(1);
//   g1.add_node(2);
//   g1.add_node(3);
//   g1.add_node(4);
//   g1.add_node(5);
//   g1.add_edge(0, 1, 1);
//   g1.add_edge(1, 2, 2);
//   g1.add_edge(2, 3, 3);
//   g1.add_edge(3, 4, 4);
//   g1.add_edge(4, 0, 5);

//   Graph<int, int> g2;
//   g2.add_node(1);
//   g2.add_node(2);
//   g2.add_node(3);
//   g2.add_node(4);
//   g2.add_node(5);
//   g2.add_node(1);
//   g2.add_node(2);
//   g2.add_node(3);
//   g2.add_node(4);
//   g2.add_node(5);
//   g2.add_edge(0, 1, 1);
//   g2.add_edge(1, 2, 2);
//   g2.add_edge(2, 3, 3);
//   g2.add_edge(3, 4, 4);
//   g2.add_edge(4, 0, 5);
//   g2.add_edge(0, 5, 6);
//   g2.add_edge(1, 6, 7);
//   g2.add_edge(2, 7, 8);
//   g2.add_edge(3, 8, 9);
//   g2.add_edge(4, 9, 10);
//   g2.add_edge(5, 8, 11);
//   g2.add_edge(5, 7, 12);
//   g2.add_edge(9, 6, 13);
//   g2.add_edge(9, 7, 14);
//   g2.add_edge(6, 8, 15);

//   ArrayXi qlbl = ArrayXi::Zero(g1.size()), tlbl = ArrayXi::Zero(g2.size());

//   auto [map, ok] = vf2pp(g1, g2, qlbl, tlbl, MappingType::kEdgeSubgraph);
//   EXPECT_TRUE(ok);
// }
}  // namespace
}  // namespace nuri

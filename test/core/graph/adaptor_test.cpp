//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/graph/adaptor.h"  // IWYU pragma: keep

#include <absl/algorithm/container.h>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/graph/properties.hpp>

#include <gtest/gtest.h>

#include "nuri/core/graph/graph.h"
#include "nuri/core/graph/subgraph.h"

namespace nuri {
namespace {
using Graph = nuri::Graph<int, int>;
using Subgraph = nuri::Subgraph<int, int, false>;
using ConstSubgraph = nuri::Subgraph<int, int, true>;

Graph c5_petersen_graph() {
  Graph g;
  for (int i = 0; i < 15; ++i)
    g.add_node(i);

  g.add_edge(0, 1, 0);
  g.add_edge(1, 2, 1);
  g.add_edge(2, 3, 2);
  g.add_edge(3, 4, 3);
  g.add_edge(4, 0, 4);

  g.add_edge(5, 6, 5);
  g.add_edge(6, 7, 6);
  g.add_edge(7, 8, 7);
  g.add_edge(8, 9, 8);
  g.add_edge(9, 5, 9);
  g.add_edge(5, 10, 10);
  g.add_edge(6, 11, 11);
  g.add_edge(7, 12, 12);
  g.add_edge(8, 13, 13);
  g.add_edge(9, 14, 14);
  g.add_edge(10, 13, 15);
  g.add_edge(10, 12, 16);
  g.add_edge(14, 11, 17);
  g.add_edge(14, 12, 18);
  g.add_edge(11, 13, 19);

  return g;
}

template <class G>
struct GraphAdaptorConcept: virtual boost::concepts::BidirectionalGraph<G>,
                            virtual boost::concepts::VertexAndEdgeListGraph<G>,
                            virtual boost::concepts::AdjacencyGraph<G> { };

BOOST_CONCEPT_ASSERT((GraphAdaptorConcept<Graph>));
BOOST_CONCEPT_ASSERT((GraphAdaptorConcept<Subgraph>));
BOOST_CONCEPT_ASSERT((GraphAdaptorConcept<ConstSubgraph>));

TEST(BoostGraphTest, MaximumCardinalityMatching) {
  using IPMap =
      boost::iterator_property_map<int *,
                                   boost::typed_identity_property_map<int>>;

  Graph g = c5_petersen_graph();

  ArrayXi mates(g.size());
  boost::edmonds_maximum_cardinality_matching(g, IPMap(mates.data()));

  ASSERT_TRUE(boost::is_a_matching(g, IPMap(mates.data())));
  EXPECT_EQ(boost::matching_size(g, IPMap(mates.data())), 7);

  for (auto node: g) {
    auto cnt = absl::c_count_if(mates, [&](int d) { return d == node.id(); });
    EXPECT_LE(cnt, 1) << node.id();
  }
}

TEST(BoostGraphTest, SubgraphMaximumCardinalityMatching) {
  using IPMap =
      boost::iterator_property_map<int *,
                                   boost::typed_identity_property_map<int>>;

  Graph g = c5_petersen_graph();
  Subgraph sg = Subgraph::from_nodes(g, { 0, 1, 5, 6, 7, 8, 9 });

  ArrayXi mates(sg.size());
  boost::edmonds_maximum_cardinality_matching(sg, IPMap(mates.data()));

  ASSERT_TRUE(boost::is_a_matching(sg, IPMap(mates.data())));
  EXPECT_EQ(boost::matching_size(sg, IPMap(mates.data())), 3);

  for (auto node: sg) {
    auto cnt = absl::c_count_if(mates, [&](int d) { return d == node.id(); });
    EXPECT_LE(cnt, 1) << node.id() << " (" << node.as_parent().id() << ")";
  }
}

TEST(BoostGraphTest, GraphIndexMaps) {
  Graph g = c5_petersen_graph();

  auto vmap = get(boost::vertex_index, g);
  EXPECT_EQ(vmap[0], 0);
  EXPECT_EQ(vmap[5], 5);

  auto emap = get(boost::edge_index, g);
  BoostEdgeDesc e0 { 0, 0, 1 }, e1 { 1, 1, 2 };
  EXPECT_EQ(emap[e0], 0);
  EXPECT_EQ(emap[e1], 1);
}

TEST(BoostGraphTest, SubgraphIndexMaps) {
  Graph g = c5_petersen_graph();
  Subgraph sg = Subgraph::from_nodes(g, { 0, 1, 5, 6, 7, 8, 9 });

  auto vmap = get(boost::vertex_index, sg);
  EXPECT_EQ(vmap[0], 0);
  EXPECT_EQ(vmap[5], 5);

  auto emap = get(boost::edge_index, sg);
  BoostEdgeDesc e0 { 0, 0, 1 }, e1 { 1, 1, 2 };
  EXPECT_EQ(emap[e0], 0);
  EXPECT_EQ(emap[e1], 1);
}
}  // namespace
}  // namespace nuri

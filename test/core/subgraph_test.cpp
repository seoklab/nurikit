//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "nuri/core/graph.h"

#define NURI_TEST_CONVERTIBILITY(from_name, from_expr, to_name, to_expr,       \
                                 allowed)                                      \
  static_assert(std::is_convertible_v<from_expr, to_expr> == allowed,          \
                #from_name " to " #to_name " convertibility != " #allowed);    \
  static_assert(std::is_constructible_v<to_expr, from_expr> == allowed,        \
                #from_name " to " #to_name " constructibility != " #allowed);  \
  static_assert(std::is_assignable_v<to_expr, from_expr> == allowed,           \
                #from_name " to " #to_name " assignability != " #allowed)

#define NURI_ASSERT_ONEWAY_CONVERTIBLE(from_name, from_expr, to_name, to_expr) \
  NURI_TEST_CONVERTIBILITY(from_name, from_expr, to_name, to_expr, true);      \
  NURI_TEST_CONVERTIBILITY(to_name, to_expr, from_name, from_expr, false)
#define NURI_ASSERT_TWOWAY_CONVERTIBLE(from_name, from_expr, to_name, to_expr) \
  NURI_TEST_CONVERTIBILITY(from_name, from_expr, to_name, to_expr, true);      \
  NURI_TEST_CONVERTIBILITY(to_name, to_expr, from_name, from_expr, true)

namespace nuri {
namespace {
using internal::IndexSet;

class IndexSetTest: public testing::Test {
protected:
  void SetUp() override {
    si_ = { 3, 5, 7, 5, 1 };

    ASSERT_EQ(si_.size(), 4);
    ASSERT_EQ(si_[0], 1);
    ASSERT_EQ(si_[1], 3);
    ASSERT_EQ(si_[2], 5);
    ASSERT_EQ(si_[3], 7);
  }

  IndexSet si_;
};

TEST_F(IndexSetTest, EraseCond) {
  si_.erase_if([](int i) { return i < 5; });
  ASSERT_EQ(si_.size(), 2);
  EXPECT_EQ(si_[0], 5);
  EXPECT_EQ(si_[1], 7);
}

TEST_F(IndexSetTest, Difference) {
  IndexSet si2 = { 1, 4, 7, 8 };
  si_.difference(si2);
  ASSERT_EQ(si_.size(), 2);
  EXPECT_EQ(si_[0], 3);
  EXPECT_EQ(si_[1], 5);
}

TEST_F(IndexSetTest, FindIndex) {
  int found = si_.find_index(5);
  EXPECT_EQ(found, 2);

  int not_found = si_.find_index(100);
  EXPECT_EQ(not_found, 4);
}

using Graph = Graph<int, int>;

static_assert(std::is_same_v<SubgraphOf<Graph>, Subgraph<int, int>>,
              "SubGraphType<Graph> must be a Subgraph (mutable)");
static_assert(std::is_same_v<SubgraphOf<Graph &>, Subgraph<int, int>>,
              "SubGraphType<Graph> must be a Subgraph (mutable)");
static_assert(std::is_same_v<SubgraphOf<const Graph>, Subgraph<int, int, true>>,
              "SubGraphType<Graph> must be a ConstSubGraph");
static_assert(
    std::is_same_v<SubgraphOf<const Graph &>, Subgraph<int, int, true>>,
    "SubGraphType<Graph> must be a ConstSubGraph");

NURI_ASSERT_ONEWAY_CONVERTIBLE(subgraph, SubgraphOf<Graph>, const subgraph,
                               SubgraphOf<const Graph>);

NURI_ASSERT_ONEWAY_CONVERTIBLE(iterator, SubgraphOf<Graph>::iterator,
                               const_iterator,
                               SubgraphOf<Graph>::const_iterator);

TEST(BasicSubgraphTest, CreateAndAssign) {
  Graph g;

  Subgraph sg(g);
  ASSERT_TRUE(sg.empty());
  ASSERT_EQ(sg.size(), 0);
  ASSERT_EQ(sg.num_nodes(), 0);

  Subgraph sg_copy(sg);
  ASSERT_TRUE(sg_copy.empty());
  ASSERT_EQ(sg_copy.size(), 0);
  ASSERT_EQ(sg_copy.num_nodes(), 0);

  Subgraph sg_move(std::move(sg));
  ASSERT_TRUE(sg_move.empty());
  ASSERT_EQ(sg_move.size(), 0);
  ASSERT_EQ(sg_move.num_nodes(), 0);

  sg_copy = sg_move;
  sg_move = std::move(sg_copy);

  Subgraph csg(std::as_const(g));
  ASSERT_TRUE(csg.empty());
  ASSERT_EQ(csg.size(), 0);
  ASSERT_EQ(csg.num_nodes(), 0);

  Subgraph csg_copy(csg);
  ASSERT_TRUE(csg_copy.empty());
  ASSERT_EQ(csg_copy.size(), 0);
  ASSERT_EQ(csg_copy.num_nodes(), 0);

  Subgraph csg_move(std::move(csg));
  ASSERT_TRUE(csg_move.empty());
  ASSERT_EQ(csg_move.size(), 0);
  ASSERT_EQ(csg_move.num_nodes(), 0);

  csg_copy = csg_move;
  csg_move = std::move(csg_copy);

  auto deduced = make_subgraph(g);
  static_assert(std::is_same_v<decltype(deduced), Subgraph<int, int, false>>,
                "make_subgraph(mutable graph) must return Subgraph");

  auto cdeduced = make_subgraph(std::as_const(g));
  static_assert(std::is_same_v<decltype(cdeduced), Subgraph<int, int, true>>,
                "make_subgraph(const graph) must return ConstSubGraph");
}

TEST(BasicSubgraphTest, IterateNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg1 = subgraph_from_nodes(g, { 2, 3 });
  int i = 2;
  for (auto it = sg1.begin(); it != sg1.end(); it++, i++) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
    static_assert(std::is_assignable_v<decltype(it->data()), int>,
                  "Subgraph iterator must be mutable");
  }

  for (auto rit = std::make_reverse_iterator(sg1.end());
       rit != std::make_reverse_iterator(sg1.begin()); rit++) {
    --i;
    EXPECT_EQ((*rit).as_parent().id(), i);
    EXPECT_EQ((*rit).data(), i);
    static_assert(std::is_assignable_v<decltype((*rit).data()), int>,
                  "Subgraph iterator must be mutable");
  }

  const auto &csg1 = sg1;
  i = 2;
  for (auto it = csg1.begin(); it != csg1.end(); it++, i++) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
    static_assert(!std::is_assignable_v<decltype(it->data()), int>,
                  "Subgraph const_iterator must be immutable");
  }

  for (auto rit = std::make_reverse_iterator(csg1.end());
       rit != std::make_reverse_iterator(csg1.begin()); rit++) {
    --i;
    EXPECT_EQ((*rit).as_parent().id(), i);
    EXPECT_EQ((*rit).data(), i);
    static_assert(!std::is_assignable_v<decltype((*rit).data()), int>,
                  "Subgraph const_iterator must be immutable");
  }

  i = 2;
  for (auto it = sg1.cbegin(); it != sg1.cend(); it++, i++) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
    static_assert(!std::is_assignable_v<decltype(it->data()), int>,
                  "Subgraph const_iterator must be immutable");
  }

  EXPECT_TRUE(sg1.begin() == sg1.cbegin());

  EXPECT_TRUE(sg1.begin() == csg1.begin());
  EXPECT_TRUE(sg1.begin() == csg1.cbegin());

  EXPECT_TRUE(sg1.cbegin() == csg1.begin());
  EXPECT_TRUE(sg1.cbegin() == csg1.cbegin());

  Subgraph csg2 = subgraph_from_nodes(std::as_const(g), { 4, 3 });
  static_assert(
      std::is_same_v<decltype(csg2)::iterator, decltype(csg2)::const_iterator>,
      "const_subgraph::iterator and const_subgraph::const_iterator must be "
      "same");

  i = 3;
  for (auto it = csg2.begin(); it != csg2.end(); it++, i++) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
    static_assert(!std::is_assignable_v<decltype(it->data()), int>,
                  "ConstSubGraph iterator must be immutable");
  }
  for (auto rit = std::make_reverse_iterator(csg2.end());
       rit != std::make_reverse_iterator(csg2.begin()); ++rit) {
    --i;
    EXPECT_EQ((*rit).as_parent().id(), i);
    EXPECT_EQ((*rit).data(), i);
    static_assert(!std::is_assignable_v<decltype((*rit).data()), int>,
                  "ConstSubGraph iterator must be immutable");
  }

  Subgraph sg2 = subgraph_from_nodes(g, { 4, 3 });
  EXPECT_TRUE(sg2.begin() == sg2.cbegin());

  EXPECT_TRUE(sg2.begin() == csg2.begin());
  EXPECT_TRUE(sg2.begin() == csg2.cbegin());

  EXPECT_TRUE(sg2.cbegin() == csg2.begin());
  EXPECT_TRUE(sg2.cbegin() == csg2.cbegin());
}

TEST(BasicSubgraphTest, AddNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg(g);
  sg.add_node(2);
  sg.add_node(3);

  int i = 2;
  for (auto it = sg.begin(); it != sg.end(); ++it, ++i) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
  }

  i = 2;
  for (auto it = sg.cbegin(); it != sg.cend(); ++it, ++i) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
  }

  Subgraph csg(std::as_const(g));
  csg.add_node(4);
  csg.add_node(3);
  i = 3;
  for (auto it = csg.begin(); it != csg.end(); ++it, ++i) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
  }
}

class AdvancedSubgraphTest: public testing::Test {
public:
protected:
  void SetUp() override {
    graph_.clear();
    graph_.reserve(11);

    for (int i = 0; i < 11; ++i) {
      graph_.add_node({ i });
    }

    graph_.add_edge(3, 1, { 100 });
    graph_.add_edge(3, 2, { 101 });
    graph_.add_edge(5, 3, { 102 });
    graph_.add_edge(6, 3, { 103 });
    graph_.add_edge(1, 2, { 104 });
    graph_.add_edge(7, 1, { 105 });
    graph_.add_edge(8, 1, { 106 });
    graph_.add_edge(2, 0, { 107 });
    graph_.add_edge(9, 2, { 108 });
    graph_.add_edge(0, 4, { 109 });

    sg_ = subgraph_from_nodes(graph_, { 3, 10, 1, 2 });
    ASSERT_EQ(sg_.size(), 4);
    ASSERT_EQ(sg_.num_edges(), 3);
    csg_ = sg_;
    ASSERT_EQ(csg_.size(), 4);
    ASSERT_EQ(csg_.num_edges(), 3);
  }

  /**
   * The resulting graph (somewhat looks like a cyclopropane derivative):
   *
   *            0 - 4
   *            |
   *        9 - 2          10 (<-intentionally unconnected)
   *          /   \
   *     6 - 3 --- 1 - 8
   *         |     |
   *         5     7
   */
  Graph graph_;
  SubgraphOf<Graph> sg_ = Subgraph(graph_);
  SubgraphOf<const Graph> csg_ = sg_;
};

TEST_F(AdvancedSubgraphTest, Clear) {
  ASSERT_FALSE(sg_.empty());
  sg_.clear();
  EXPECT_TRUE(sg_.empty());

  ASSERT_FALSE(csg_.empty());
  csg_.clear();
  EXPECT_TRUE(csg_.empty());
}

TEST_F(AdvancedSubgraphTest, ClearEdges) {
  ASSERT_FALSE(sg_.empty());
  sg_.clear_edges();
  EXPECT_FALSE(sg_.empty());
  EXPECT_EQ(sg_.num_edges(), 0);

  ASSERT_FALSE(csg_.empty());
  csg_.clear_edges();
  EXPECT_FALSE(csg_.empty());
  EXPECT_EQ(sg_.num_edges(), 0);
}

TEST_F(AdvancedSubgraphTest, ContainsNodes) {
  EXPECT_TRUE(sg_.contains_node(2));
  EXPECT_TRUE(sg_.contains_node(graph_.node(2)));

  EXPECT_FALSE(sg_.contains_node(4));
  EXPECT_FALSE(sg_.contains_node(graph_.node(4)));
}

TEST_F(AdvancedSubgraphTest, ContainsEdges) {
  EXPECT_TRUE(sg_.contains_edge(4));
  EXPECT_TRUE(sg_.contains_edge(graph_.edge(4)));

  EXPECT_FALSE(sg_.contains_edge(2));
  EXPECT_FALSE(sg_.contains_edge(graph_.edge(2)));
}

TEST_F(AdvancedSubgraphTest, AccessNodes) {
  EXPECT_EQ(sg_[0].id(), 0);
  EXPECT_EQ(sg_[0].as_parent().id(), 1);
  EXPECT_EQ(sg_[1].id(), 1);
  EXPECT_EQ(sg_[1].as_parent().id(), 2);

  const auto &sgc = sg_;
  EXPECT_EQ(sgc[0].id(), 0);
  EXPECT_EQ(sgc[0].as_parent().id(), 1);
  EXPECT_EQ(sgc[1].id(), 1);
  EXPECT_EQ(sgc[1].as_parent().id(), 2);

  EXPECT_EQ(csg_[0].id(), 0);
  EXPECT_EQ(csg_[0].as_parent().id(), 1);
  EXPECT_EQ(csg_[1].id(), 1);
  EXPECT_EQ(csg_[1].as_parent().id(), 2);
}

TEST_F(AdvancedSubgraphTest, UpdateNodes) {
  sg_.update_nodes({ 0, 2, 4 });
  EXPECT_EQ(sg_.size(), 3);
  EXPECT_EQ(sg_.num_edges(), 2);

  EXPECT_TRUE(sg_.contains_edge(7));
  EXPECT_TRUE(sg_.contains_edge(9));

  EXPECT_FALSE(sg_.contains_edge(0));
  EXPECT_FALSE(sg_.contains_edge(1));
  EXPECT_FALSE(sg_.contains_edge(4));
}

TEST_F(AdvancedSubgraphTest, AddNodesWithEdges) {
  auto sg = subgraph_from_nodes(graph_, { 3, 1 });
  sg.add_nodes_with_edges({ 2, 0, 4 });

  EXPECT_EQ(sg.size(), 5);
  EXPECT_EQ(sg.num_edges(), 3);

  EXPECT_NE(sg.find_edge(graph_.node(0), graph_.node(2)), sg.edge_end());
  EXPECT_NE(sg.find_edge(graph_.node(0), graph_.node(4)), sg.edge_end());

  EXPECT_EQ(sg.find_edge(graph_.node(1), graph_.node(2)), sg.edge_end());
}

TEST_F(AdvancedSubgraphTest, FindNodes) {
  EXPECT_NE(sg_.find_node(2), sg_.end());
  EXPECT_EQ(sg_.find_node(2), sg_.find_node(graph_.node(2)));
  EXPECT_EQ(sg_.find_node(2), std::as_const(sg_).find_node(graph_.node(2)));
  EXPECT_EQ(sg_.find_node(2)->as_parent().id(), 2);
  EXPECT_EQ(sg_.find_node(4), sg_.end());

  EXPECT_NE(csg_.find_node(2), csg_.end());
  EXPECT_EQ(csg_.find_node(2), csg_.find_node(graph_.node(2)));
  EXPECT_EQ(csg_.find_node(2), std::as_const(csg_).find_node(graph_.node(2)));
  EXPECT_EQ(csg_.find_node(2)->as_parent().id(), 2);
  EXPECT_EQ(csg_.find_node(4), csg_.end());
}

TEST_F(AdvancedSubgraphTest, EraseNode) {
  // Start with 1, 2, 3, 10

  // 2, 3, 10
  sg_.erase_node(0);
  EXPECT_EQ(sg_.size(), 3);
  EXPECT_EQ(sg_.num_edges(), 1);
  EXPECT_TRUE(sg_.contains_edge(1));
  // 3, 10
  sg_.erase_node_of(2);
  EXPECT_EQ(sg_.size(), 2);

  // 3, 5, 10
  sg_.add_node(5);
  // 3, 10
  sg_.erase_node(*++sg_.begin());
  EXPECT_EQ(sg_.size(), 2);

  // 3, 5, 10
  sg_.add_node(5);
  // 3, 10
  sg_.erase_node_of(graph_.node(5));
  EXPECT_EQ(sg_.size(), 2);
  EXPECT_FALSE(sg_.contains_node(5));
}

TEST_F(AdvancedSubgraphTest, EraseNodeRange) {
  // Start with 1, 2, 3, 7, 10
  sg_.add_edge(5);

  // 1, 7, 10
  sg_.erase_nodes(sg_.begin() + 1, sg_.begin() + 3);
  EXPECT_EQ(sg_.size(), 3);
  EXPECT_EQ(sg_.num_edges(), 1);
  EXPECT_TRUE(sg_.contains_edge(5));
}

TEST_F(AdvancedSubgraphTest, EraseNodePred) {
  // Start with 1, 2, 3, 7, 10
  sg_.add_edge(5);

  // 1, 3, 7
  sg_.erase_nodes_if([](int i) { return i % 2 == 0; });
  EXPECT_EQ(sg_.size(), 3);
  EXPECT_EQ(sg_.num_edges(), 2);
  EXPECT_TRUE(sg_.contains_edge(0));
  EXPECT_TRUE(sg_.contains_edge(5));
}

TEST_F(AdvancedSubgraphTest, CreateFromEdges) {
  auto sg = subgraph_from_edges(graph_, { 3, 4, 6 });
  EXPECT_EQ(sg.size(), 5);
  EXPECT_EQ(sg.num_edges(), 3);

  EXPECT_TRUE(sg.contains_node(1));
  EXPECT_TRUE(sg.contains_node(2));
  EXPECT_TRUE(sg.contains_node(3));
  EXPECT_TRUE(sg.contains_node(6));
  EXPECT_TRUE(sg.contains_node(8));
}

TEST_F(AdvancedSubgraphTest, UpdateEdges) {
  IndexSet edges = { 3, 4, 6 };

  sg_.update_edges(std::move(edges));
  EXPECT_EQ(sg_.size(), 5);
  EXPECT_EQ(sg_.num_edges(), 3);

  EXPECT_TRUE(sg_.contains_node(1));
  EXPECT_TRUE(sg_.contains_node(2));
  EXPECT_TRUE(sg_.contains_node(3));
  EXPECT_TRUE(sg_.contains_node(6));
  EXPECT_TRUE(sg_.contains_node(8));
}

TEST_F(AdvancedSubgraphTest, RefreshEdges) {
  graph_.add_edge(2, 10, { 110 });

  sg_.refresh_edges();
  EXPECT_EQ(sg_.size(), 4);
  EXPECT_EQ(sg_.num_edges(), 4);
  EXPECT_TRUE(sg_.contains_edge(10));
}

TEST_F(AdvancedSubgraphTest, AddEdges) {
  sg_.add_edge(2);
  EXPECT_EQ(sg_.num_nodes(), 5);
  EXPECT_EQ(sg_.num_edges(), 4);
  EXPECT_TRUE(sg_.contains_edge(2));
  EXPECT_TRUE(sg_.contains_node(5));

  // 4 is duplicated
  IndexSet edges = { 3, 4, 9 };
  sg_.add_edges(edges);
  EXPECT_EQ(sg_.num_nodes(), 8);
  EXPECT_EQ(sg_.num_edges(), 6);
  EXPECT_TRUE(sg_.contains_edge(3));
  EXPECT_TRUE(sg_.contains_edge(4));
  EXPECT_TRUE(sg_.contains_edge(9));
  EXPECT_TRUE(sg_.contains_node(0));
  EXPECT_TRUE(sg_.contains_node(4));
  EXPECT_TRUE(sg_.contains_node(6));
}

TEST_F(AdvancedSubgraphTest, AccessEdges) {
  auto it = sg_.find_edge(4);
  EXPECT_NE(it, sg_.edge_end());

  EXPECT_EQ(it->src().id(), 0);
  EXPECT_EQ(it->src().as_parent().id(), 1);
  EXPECT_EQ(it->dst().id(), 1);
  EXPECT_EQ(it->dst().as_parent().id(), 2);
  EXPECT_EQ(it->as_parent().id(), 4);

  auto cit = std::as_const(sg_).find_edge(4);
  EXPECT_NE(cit, sg_.edge_end());

  EXPECT_EQ(cit->src().id(), 0);
  EXPECT_EQ(cit->src().as_parent().id(), 1);
  EXPECT_EQ(cit->dst().id(), 1);
  EXPECT_EQ(cit->dst().as_parent().id(), 2);
  EXPECT_EQ(cit->as_parent().id(), 4);
}

TEST_F(AdvancedSubgraphTest, FindEdges) {
  EXPECT_NE(sg_.find_edge(4), sg_.edge_end());
  EXPECT_EQ(sg_.find_edge(4), sg_.find_edge(graph_.edge(4)));
  EXPECT_EQ(sg_.find_edge(4), std::as_const(sg_).find_edge(graph_.edge(4)));
  EXPECT_EQ(sg_.find_edge(4)->as_parent().id(), 4);
  EXPECT_EQ(sg_.find_edge(2), sg_.edge_end());

  EXPECT_NE(csg_.find_edge(4), csg_.edge_end());
  EXPECT_EQ(csg_.find_edge(4), csg_.find_edge(graph_.edge(4)));
  EXPECT_EQ(csg_.find_edge(4), std::as_const(csg_).find_edge(graph_.edge(4)));
  EXPECT_EQ(csg_.find_edge(4)->as_parent().id(), 4);
  EXPECT_EQ(csg_.find_edge(2), csg_.edge_end());

  EXPECT_NE(sg_.find_edge(sg_.node(0), sg_.node(1)), sg_.edge_end());
  EXPECT_EQ(sg_.find_edge(sg_.node(0), sg_.node(1))->as_parent().id(), 4);

  EXPECT_EQ(sg_.find_edge(sg_.node(0), sg_.node(3)), sg_.edge_end());

  const auto &sgc = sg_;
  EXPECT_NE(sgc.find_edge(sgc.node(0), sgc.node(1)), sgc.edge_end());
  EXPECT_EQ(sgc.find_edge(sgc.node(0), sgc.node(1))->as_parent().id(), 4);

  EXPECT_EQ(sgc.find_edge(sgc.node(0), sgc.node(3)), sgc.edge_end());
}

TEST_F(AdvancedSubgraphTest, EraseEdges) {
  auto s1 = sg_;
  s1.erase_edge(2);
  EXPECT_EQ(s1.size(), 4);
  EXPECT_EQ(s1.num_edges(), 2);
  EXPECT_FALSE(s1.contains_edge(4));

  auto s2 = sg_;
  s2.erase_edge(sg_.edge(2));
  EXPECT_EQ(s2.size(), 4);
  EXPECT_EQ(s2.num_edges(), 2);
  EXPECT_FALSE(s2.contains_edge(4));

  auto s3 = sg_;
  s3.erase_edges(sg_.edge_begin(), sg_.edge_begin() + 2);
  EXPECT_EQ(s3.size(), 4);
  EXPECT_EQ(s3.num_edges(), 1);
  EXPECT_TRUE(s3.contains_edge(4));

  auto s4 = sg_;
  s4.erase_edge_of(4);
  EXPECT_EQ(s4.size(), 4);
  EXPECT_EQ(s4.num_edges(), 2);
  EXPECT_FALSE(s4.contains_edge(4));

  auto s5 = sg_;
  s5.erase_edge_of(graph_.edge(4));
  EXPECT_EQ(s5.size(), 4);
  EXPECT_EQ(s5.num_edges(), 2);
  EXPECT_FALSE(s5.contains_edge(4));

  sg_.erase_edges_if([](int i) { return i % 2 == 0; });
  EXPECT_EQ(sg_.size(), 4);
  EXPECT_EQ(sg_.num_edges(), 1);
  EXPECT_TRUE(sg_.contains_edge(1));
}

TEST_F(AdvancedSubgraphTest, CountDegrees) {
  sg_.add_node(4);
  sg_.erase_node_of(10);

  for (auto node: sg_) {
    EXPECT_EQ(node.degree(), node.as_parent().id() == 4 ? 0 : 2)
        << node.as_parent().id();
  }
}

TEST_F(AdvancedSubgraphTest, IterateAdjacency) {
  int total = 0;
  for (auto node: sg_) {
    for (auto it = node.begin(); it != node.end(); it++) {
      ++total;
    }
  }
  EXPECT_EQ(total, 6);

  total = 0;
  for (auto node: sg_) {
    for (auto rit = std::make_reverse_iterator(node.end());
         rit != std::make_reverse_iterator(node.begin()); rit++) {
      ++total;
    }
  }
  EXPECT_EQ(total, 6);

  total = 0;
  for (auto node: csg_) {
    for (auto it = node.begin(); it != node.end(); it++) {
      ++total;
    }
  }
  EXPECT_EQ(total, 6);

  total = 0;
  for (auto node: csg_) {
    for (auto rit = std::make_reverse_iterator(node.end());
         rit != std::make_reverse_iterator(node.begin()); rit++) {
      ++total;
    }
  }
  EXPECT_EQ(total, 6);
}

TEST_F(AdvancedSubgraphTest, FindAdjacency) {
  auto mmit = sg_.find_adjacent(sg_.node(0), sg_.node(3));
  EXPECT_TRUE(mmit.end());
  EXPECT_EQ(mmit, sg_.node(0).find_adjacent(sg_.node(3)));

  mmit = sg_.find_adjacent(sg_.node(0), sg_.node(1));
  EXPECT_FALSE(mmit.end());
  EXPECT_EQ(mmit->as_parent().eid(), 4);
  EXPECT_EQ(mmit, sg_.node(0).find_adjacent(sg_.node(1)));

  auto mcit = std::as_const(sg_).find_adjacent(std::as_const(sg_).node(0),
                                               std::as_const(sg_).node(3));
  EXPECT_TRUE(mcit.end());
  EXPECT_EQ(mcit, std::as_const(sg_).node(0).find_adjacent(
                      std::as_const(sg_).node(3)));

  mcit = std::as_const(sg_).find_adjacent(std::as_const(sg_).node(0),
                                          std::as_const(sg_).node(1));
  EXPECT_FALSE(mcit.end());
  EXPECT_EQ(mcit->as_parent().eid(), 4);
  EXPECT_EQ(mcit, std::as_const(sg_).node(0).find_adjacent(
                      std::as_const(sg_).node(1)));

  auto cmit = csg_.find_adjacent(csg_.node(0), csg_.node(3));
  EXPECT_TRUE(cmit.end());
  EXPECT_EQ(cmit, csg_.node(0).find_adjacent(csg_.node(3)));

  cmit = csg_.find_adjacent(csg_.node(0), csg_.node(1));
  EXPECT_FALSE(cmit.end());
  EXPECT_EQ(cmit->as_parent().eid(), 4);
  EXPECT_EQ(cmit, csg_.node(0).find_adjacent(csg_.node(1)));

  auto ccit = std::as_const(csg_).find_adjacent(std::as_const(csg_).node(0),
                                                std::as_const(csg_).node(3));
  EXPECT_TRUE(ccit.end());
  EXPECT_EQ(ccit, std::as_const(csg_).node(0).find_adjacent(
                      std::as_const(csg_).node(3)));

  ccit = std::as_const(csg_).find_adjacent(std::as_const(csg_).node(0),
                                           std::as_const(csg_).node(1));
  EXPECT_FALSE(ccit.end());
  EXPECT_EQ(ccit->as_parent().eid(), 4);
  EXPECT_EQ(ccit, std::as_const(csg_).node(0).find_adjacent(
                      std::as_const(csg_).node(1)));
}

TEST_F(AdvancedSubgraphTest, AddSubgraph) {
  sg_.update_nodes({ 0, 2, 4 });

  graph_.merge(sg_);

  ASSERT_EQ(graph_.size(), 14);
  ASSERT_EQ(graph_.num_edges(), 12);

  EXPECT_EQ(graph_.node(11).data(), 0);
  EXPECT_EQ(graph_.node(12).data(), 2);
  EXPECT_EQ(graph_.node(13).data(), 4);

  EXPECT_EQ(graph_.find_edge(11, 12)->data(), 107);
  EXPECT_EQ(graph_.find_edge(11, 13)->data(), 109);
}
}  // namespace

template class Subgraph<int, int, false>;
template class Subgraph<int, int, true>;

namespace internal {
using GraphType = nuri::Graph<int, int>;

// Other repetitive instantiations
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define NURI_INSTANTIATE_ALL_TEMPLATES(SGT, is_const)                           \
  template class SubEdgeWrapper<SGT, (is_const)>;                               \
  static_assert(std::is_trivially_copyable_v<SubEdgeWrapper<SGT, (is_const)>>,  \
                "SubEdgeWrapper must be trivially copyable");                   \
                                                                                \
  template class DataIteratorBase<SubEdgeIterator<SGT, (is_const)>, SGT,        \
                                  SubEdgeWrapper<SGT, (is_const)>, (is_const)>; \
  template class SubEdgeIterator<SGT, (is_const)>;                              \
  static_assert(                                                                \
      std::is_trivially_copyable_v<SubEdgeIterator<SGT, (is_const)>>,           \
      "SubEdgeIterator must be trivially copyable");                            \
                                                                                \
  template class SubNodeWrapper<SGT, (is_const)>;                               \
  static_assert(std::is_trivially_copyable_v<SubNodeWrapper<SGT, (is_const)>>,  \
                "SubNodeWrapper must be trivially copyable");                   \
                                                                                \
  template class DataIteratorBase<SubNodeIterator<SGT, (is_const)>, SGT,        \
                                  SubNodeWrapper<SGT, (is_const)>, (is_const)>; \
  template class SubNodeIterator<SGT, (is_const)>;                              \
  static_assert(                                                                \
      std::is_trivially_copyable_v<SubNodeIterator<SGT, (is_const)>>,           \
      "SubNodeIterator must be trivially copyable");                            \
                                                                                \
  template class SubAdjWrapper<SGT, (is_const)>;                                \
  static_assert(std::is_trivially_copyable_v<SubAdjWrapper<SGT, (is_const)>>,   \
                "AdjWrapper must be trivially copyable");                       \
                                                                                \
  template class SubAdjIterator<SGT, (is_const)>;                               \
  static_assert(std::is_trivially_copyable_v<SubAdjIterator<SGT, (is_const)>>,  \
                "SubAdjIterator must be trivially copyable")

NURI_INSTANTIATE_ALL_TEMPLATES(SubgraphOf<GraphType>, false);
NURI_INSTANTIATE_ALL_TEMPLATES(SubgraphOf<GraphType>, true);
NURI_INSTANTIATE_ALL_TEMPLATES(SubgraphOf<const GraphType>, true);
// NOLINTEND(cppcoreguidelines-macro-usage)
}  // namespace internal
}  // namespace nuri

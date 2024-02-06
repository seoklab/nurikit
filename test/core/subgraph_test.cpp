//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

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

TEST(BasicSubgraphTest, Clear) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg(g, { 2, 3 });
  ASSERT_FALSE(sg.empty());
  sg.clear();
  EXPECT_TRUE(sg.empty());

  Subgraph csg(std::as_const(g), { 0, 5 });
  ASSERT_FALSE(csg.empty());
  csg.clear();
  EXPECT_TRUE(csg.empty());
}

TEST(BasicSubgraphTest, FindNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg(g, { 2, 3 });
  EXPECT_NE(sg.find_node(2), sg.end());
  EXPECT_EQ(sg.find_node(2), sg.find_node(g.node(2)));
  EXPECT_EQ(sg.find_node(2), std::as_const(sg).find_node(g.node(2)));
  EXPECT_EQ(sg.find_node(2)->as_parent().id(), 2);
  EXPECT_EQ(sg.find_node(4), sg.end());

  Subgraph csg(std::as_const(g), { 2, 3 });
  EXPECT_NE(csg.find_node(2), csg.end());
  EXPECT_EQ(csg.find_node(2), csg.find_node(g.node(2)));
  EXPECT_EQ(csg.find_node(2), std::as_const(csg).find_node(g.node(2)));
  EXPECT_EQ(csg.find_node(2)->as_parent().id(), 2);
  EXPECT_EQ(csg.find_node(4), csg.end());
}

TEST(BasicSubgraphTest, AccessNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg(g, { 2, 3 });
  EXPECT_EQ(sg[0].id(), 0);
  EXPECT_EQ(sg[0].as_parent().id(), 2);
  EXPECT_EQ(sg[1].id(), 1);
  EXPECT_EQ(sg[1].as_parent().id(), 3);

  const auto &csg = sg;
  EXPECT_EQ(csg[0].id(), 0);
  EXPECT_EQ(csg[0].as_parent().id(), 2);
  EXPECT_EQ(csg[1].id(), 1);
  EXPECT_EQ(csg[1].as_parent().id(), 3);

  std::vector ids = { 2, 3 };
  Subgraph ccsg(std::as_const(g), ids);
  EXPECT_EQ(ccsg[0].id(), 0);
  EXPECT_EQ(ccsg[0].as_parent().id(), 2);
  EXPECT_EQ(ccsg[1].id(), 1);
  EXPECT_EQ(ccsg[1].as_parent().id(), 3);

  EXPECT_TRUE(
      std::is_permutation(ids.begin(), ids.end(), ccsg.node_ids().begin()));
}

TEST(BasicSubgraphTest, ContainsNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg(g, { 2, 3 });
  EXPECT_TRUE(sg.contains(2));
  EXPECT_TRUE(sg.contains(g.node(2)));

  EXPECT_FALSE(sg.contains(4));
  EXPECT_FALSE(sg.contains(g.node(4)));
}

TEST(BasicSubgraphTest, IterateNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg1(g, { 2, 3 });
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
    EXPECT_EQ(rit->as_parent().id(), i);
    EXPECT_EQ(rit->data(), i);
    static_assert(std::is_assignable_v<decltype(rit->data()), int>,
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
    EXPECT_EQ(rit->as_parent().id(), i);
    EXPECT_EQ(rit->data(), i);
    static_assert(!std::is_assignable_v<decltype(rit->data()), int>,
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

  std::vector<int> nodes { 4, 3 };
  Subgraph csg2(std::as_const(g), nodes);
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
    EXPECT_EQ(rit->as_parent().id(), i);
    EXPECT_EQ(rit->data(), i);
    static_assert(!std::is_assignable_v<decltype(rit->data()), int>,
                  "ConstSubGraph iterator must be immutable");
  }

  Subgraph sg2(g, nodes);
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

TEST(BasicSubgraphTest, EraseNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg(g, { 2, 3 });
  sg.erase_node(0);
  EXPECT_EQ(sg.size(), 1);
  sg.erase_node_of(2);
  EXPECT_EQ(sg.size(), 1);

  sg.add_node(5);
  sg.erase_node(*++sg.begin());
  EXPECT_EQ(sg.size(), 1);

  sg.add_node(5);
  sg.erase_node_of(g.node(5));
  EXPECT_EQ(sg.size(), 1);

  int i = 3;
  for (auto it = sg.begin(); it != sg.end(); ++it, ++i) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
  }

  i = 3;
  for (auto it = sg.cbegin(); it != sg.cend(); ++it, ++i) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
  }

  sg.add_node(5);
  sg.add_node(2);
  EXPECT_EQ(sg.size(), 3);

  sg.erase_nodes_if([](int id) { return id > 3; });
  EXPECT_EQ(sg.size(), 2);
  EXPECT_FALSE(sg.contains(5));

  sg.erase_nodes(sg.begin(), sg.end());
  EXPECT_TRUE(sg.empty());

  Subgraph csg(std::as_const(g), { 4, 5, 3 });
  csg.erase_node_of(5);
  EXPECT_EQ(csg.size(), 2);

  i = 3;
  for (auto it = csg.begin(); it != csg.end(); ++it, ++i) {
    EXPECT_EQ(it->as_parent().id(), i);
    EXPECT_EQ(it->data(), i);
  }

  csg.add_node(5);
  csg.add_node(2);
  EXPECT_EQ(csg.size(), 4);

  csg.erase_nodes_if([](int id) { return id > 3; });
  EXPECT_EQ(csg.size(), 2);
  EXPECT_FALSE(csg.contains(4));
  EXPECT_FALSE(csg.contains(5));

  csg.erase_nodes(csg.begin(), csg.end());
  EXPECT_TRUE(csg.empty());
}

TEST(BasicSubgraphTest, UpdateNodes) {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_node(3);
  g.add_node(4);
  g.add_node(5);

  Subgraph sg(g, { 2, 3 });
  for (auto node: sg) {
    node.data() = node.as_parent().id() * 2;
  }

  for (auto node: g) {
    EXPECT_EQ(node.data(), sg.contains(node.id()) ? node.id() * 2 : node.id());
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

    edges_.push_back(graph_.add_edge(0, 1, { 100 }));
    edges_.push_back(graph_.add_edge(0, 2, { 101 }));
    edges_.push_back(graph_.add_edge(5, 0, { 102 }));
    edges_.push_back(graph_.add_edge(6, 0, { 103 }));

    edges_.push_back(graph_.add_edge(1, 2, { 104 }));
    edges_.push_back(graph_.add_edge(7, 1, { 105 }));
    edges_.push_back(graph_.add_edge(8, 1, { 106 }));

    edges_.push_back(graph_.add_edge(2, 3, { 107 }));
    edges_.push_back(graph_.add_edge(9, 2, { 108 }));

    edges_.push_back(graph_.add_edge(3, 4, { 109 }));

    sg_.add_node(0);
    sg_.add_node(10);
    sg_.add_node(1);
    sg_.add_node(2);
    ASSERT_EQ(sg_.size(), 4);
    csg_ = sg_;
    ASSERT_EQ(csg_.size(), 4);
  }

  /**
   * The resulting graph (somewhat looks like a cyclopropane derivative):
   *
   *            3 - 4
   *            |
   *        9 - 2          10 (<-intentionally unconnected)
   *          /   \
   *     6 - 0 --- 1 - 8
   *         |     |
   *         5     7
   */
  Graph graph_;
  SubgraphOf<Graph> sg_ = Subgraph(graph_);
  SubgraphOf<const Graph> csg_ = sg_;
  std::vector<Graph::edge_iterator> edges_;
};

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
  auto it = sg_.find_adjacent(0, 5);
  EXPECT_TRUE(it.end());

  it = sg_.find_adjacent(0, 1);
  EXPECT_FALSE(it.end());
  EXPECT_EQ(it->eid(), edges_[0]->id());

  auto cit = std::as_const(sg_).find_adjacent(0, 5);
  EXPECT_TRUE(cit.end());

  cit = std::as_const(sg_).find_adjacent(0, 1);
  EXPECT_FALSE(cit.end());
  EXPECT_EQ(cit->eid(), edges_[0]->id());

  auto ccit = csg_.find_adjacent(0, 5);
  EXPECT_TRUE(ccit.end());

  ccit = csg_.find_adjacent(0, 1);
  EXPECT_FALSE(ccit.end());
  EXPECT_EQ(ccit->eid(), edges_[0]->id());

  ccit = csg_.find_adjacent(0, 5);
  EXPECT_TRUE(ccit.end());

  ccit = csg_.find_adjacent(0, 1);
  EXPECT_FALSE(ccit.end());
  EXPECT_EQ(ccit->eid(), edges_[0]->id());
}

TEST_F(AdvancedSubgraphTest, FindEdges) {
  auto test_iter = [this](const auto &sg, auto &ef, int line) {
    std::vector<int> visit_count(graph_.size(), 0);

    for (auto it = ef.begin(); it != ef.end(); it++) {
      visit_count[it->src().id()]++;
      visit_count[it->dst().id()]++;
    }

    for (int i = 0; i < visit_count.size(); ++i) {
      auto sit = sg.find_node(i);
      if (sit != sg.end() && sit->degree() > 0) {
        EXPECT_EQ(visit_count[i], 2)
            << "i: " << i << " callsite: line " << line;
      } else {
        EXPECT_EQ(visit_count[i], 0)
            << "i: " << i << " callsite: line " << line;
      }
    }
  };

  auto edges = sg_.edges();
  EXPECT_EQ(edges.size(), 3);
  test_iter(sg_, edges, __LINE__);

  auto cedges = std::as_const(sg_).edges();
  EXPECT_EQ(cedges.size(), 3);
  test_iter(sg_, cedges, __LINE__);

  auto ccedges = csg_.edges();
  EXPECT_EQ(ccedges.size(), 3);
  test_iter(csg_, ccedges, __LINE__);
}

TEST_F(AdvancedSubgraphTest, AddSubgraph) {
  sg_.update({ 2, 3, 4 });

  graph_.merge(sg_);

  ASSERT_EQ(graph_.size(), 14);
  ASSERT_EQ(graph_.num_edges(), 12);

  EXPECT_EQ(graph_.node(11).data(), 2);
  EXPECT_EQ(graph_.node(12).data(), 3);
  EXPECT_EQ(graph_.node(13).data(), 4);

  EXPECT_EQ(graph_.find_edge(11, 12)->data(), 107);
  EXPECT_EQ(graph_.find_edge(12, 13)->data(), 109);
}
}  // namespace

template class Subgraph<int, int, false>;
template class Subgraph<int, int, true>;

namespace internal {
using GraphType = nuri::Graph<int, int>;

template class SubEdgeWrapper<SubgraphOf<GraphType>, false>;
template class SubEdgeWrapper<SubgraphOf<GraphType>, true>;
static_assert(
    std::is_trivially_copyable_v<SubEdgeWrapper<SubgraphOf<GraphType>, false>>,
    "SubEdgeWrapper must be trivially copyable");
static_assert(
    std::is_trivially_copyable_v<SubEdgeWrapper<SubgraphOf<GraphType>, true>>,
    "ConstSubEdgeWrapper must be trivially copyable");

template class DataIteratorBase<
    SubEdgeIterator<SubEdgesFinder<SubgraphOf<GraphType>, false>, true>,
    SubEdgesFinder<SubgraphOf<GraphType>, false>,
    SubEdgeWrapper<SubgraphOf<GraphType>, true>, true>;
template class SubEdgeIterator<SubEdgesFinder<SubgraphOf<GraphType>, false>,
                               true>;
static_assert(
    std::is_trivially_copyable_v<
        SubEdgeIterator<SubEdgesFinder<SubgraphOf<GraphType>, false>, true>>,
    "SubEdgeIterator must be trivially copyable");

// Other repetitive instantiations
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define NURI_INSTANTIATE_ALL_TEMPLATES(SGT, is_const)                          \
  template class SubEdgesFinder<SGT, is_const>;                                \
                                                                               \
  template class DataIteratorBase<                                             \
      SubEdgeIterator<SubEdgesFinder<SGT, is_const>, is_const>,                \
      SubEdgesFinder<SGT, is_const>, SubEdgeWrapper<SGT, is_const>, is_const>; \
  template class SubEdgeIterator<SubEdgesFinder<SGT, is_const>, is_const>;     \
  static_assert(                                                               \
      std::is_trivially_copyable_v<                                            \
          SubEdgeIterator<SubEdgesFinder<SGT, is_const>, (is_const)>>,         \
      "SubEdgeIterator must be trivially copyable");                           \
                                                                               \
  template class SubNodeWrapper<SGT, is_const>;                                \
  static_assert(std::is_trivially_copyable_v<SubNodeWrapper<SGT, (is_const)>>, \
                "SubNodeWrapper must be trivially copyable");                  \
                                                                               \
  template class DataIteratorBase<SubNodeIterator<SGT, is_const>, SGT,         \
                                  SubNodeWrapper<SGT, is_const>, is_const>;    \
  template class SubNodeIterator<SGT, is_const>;                               \
  static_assert(                                                               \
      std::is_trivially_copyable_v<SubNodeIterator<SGT, (is_const)>>,          \
      "SubNodeIterator must be trivially copyable");                           \
                                                                               \
  template class AdjWrapper<SGT, is_const>;                                    \
  static_assert(std::is_trivially_copyable_v<AdjWrapper<SGT, (is_const)>>,     \
                "AdjWrapper must be trivially copyable");                      \
                                                                               \
  template class SubAdjIterator<SGT, is_const>;                                \
  static_assert(std::is_trivially_copyable_v<SubAdjIterator<SGT, (is_const)>>, \
                "SubAdjIterator must be trivially copyable")

NURI_INSTANTIATE_ALL_TEMPLATES(SubgraphOf<GraphType>, false);
NURI_INSTANTIATE_ALL_TEMPLATES(SubgraphOf<GraphType>, true);
NURI_INSTANTIATE_ALL_TEMPLATES(SubgraphOf<const GraphType>, true);
// NOLINTEND(cppcoreguidelines-macro-usage)
}  // namespace internal
}  // namespace nuri

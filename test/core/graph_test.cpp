//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/graph.h"

#include <iterator>
#include <type_traits>

#include <gtest/gtest.h>

#define NURIKIT_TEST_CONVERTIBILITY(from_name, from_expr, to_name, to_expr,    \
                                    allowed)                                   \
  static_assert(std::is_convertible_v<from_expr, to_expr> == allowed,          \
                #from_name " to " #to_name " convertibility != " #allowed);    \
  static_assert(std::is_constructible_v<to_expr, from_expr> == allowed,        \
                #from_name " to " #to_name " constructibility != " #allowed);  \
  static_assert(std::is_assignable_v<to_expr, from_expr> == allowed,           \
                #from_name " to " #to_name " assignability != " #allowed)

#define NURIKIT_ASSERT_ONEWAY_CONVERTIBLE(from_name, from_expr, to_name,       \
                                          to_expr)                             \
  NURIKIT_TEST_CONVERTIBILITY(from_name, from_expr, to_name, to_expr, true);   \
  NURIKIT_TEST_CONVERTIBILITY(to_name, to_expr, from_name, from_expr, false)

namespace {
// NOLINTBEGIN(clang-diagnostic-unused-member-function)
class NonTrivial {
public:
  NonTrivial(): data_(new int(0)) { }

  NonTrivial(int data): data_(new int(data)) { }

  NonTrivial(const NonTrivial &other): data_(new int(other.data())) { }

  NonTrivial(NonTrivial &&other) noexcept: data_(other.data_) {
    other.data_ = nullptr;
  }

  NonTrivial &operator=(const NonTrivial &other) {
    if (this == &other) {
      return *this;
    }

    if (data_ == nullptr) {
      data_ = new int(other.data());
      return *this;
    }

    *data_ = other.data();
    return *this;
  }

  NonTrivial &operator=(NonTrivial &&other) noexcept {
    std::swap(data_, other.data_);
    return *this;
  }

  ~NonTrivial() noexcept { delete data_; }

  int data() const { return *data_; }

  bool operator==(int i) const { return data() == i; }

private:
  int *data_;
};
// NOLINTEND(clang-diagnostic-unused-member-function)

using TrivialGraph = nuri::Graph<int, int>;
using NonTrivialGraph = nuri::Graph<NonTrivial, NonTrivial>;

using testing::Types;
using Implementations = Types<int, NonTrivial>;

template <class T>
class BasicGraphTest: public testing::Test { };

TYPED_TEST_SUITE(BasicGraphTest, Implementations);

TYPED_TEST(BasicGraphTest, CreationTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;

  Graph g1;
  ASSERT_EQ(g1.size(), 0);
  ASSERT_EQ(g1.num_nodes(), 0);
  ASSERT_EQ(g1.num_edges(), 0);

  Graph g2(10);
  ASSERT_EQ(g2.size(), 10);
  ASSERT_EQ(g2.num_nodes(), 10);
  ASSERT_EQ(g2.num_edges(), 0);

  Graph g3(10, { 10 });
  ASSERT_EQ(g3.size(), 10);
  ASSERT_EQ(g3.num_nodes(), 10);
  ASSERT_EQ(g3.num_edges(), 0);
  for (int i = 0; i < g3.num_nodes(); ++i) {
    ASSERT_EQ(g3[i].data(), 10);
    ASSERT_EQ(g3.node(i).data(), 10);
  }

  const Graph &const_graph = g3;
  for (int i = 0; i < const_graph.num_nodes(); ++i) {
    ASSERT_EQ(const_graph[i].data(), 10);
    ASSERT_EQ(const_graph.node(i).data(), 10);
  }

  {
    g3.add_edge(0, 1, { 100 });

    Graph g4(g3);
    ASSERT_EQ(g4.size(), 10);
    ASSERT_EQ(g4.num_nodes(), 10);
    ASSERT_EQ(g4.num_edges(), 1);
    for (int i = 0; i < g4.num_nodes(); ++i) {
      ASSERT_EQ(g4[i].data(), 10);
      ASSERT_EQ(g4.node(i).data(), 10);
    }

    g4.add_edge(0, 2, { 200 });
    ASSERT_EQ(g3.num_edges(), 1);

    g3.erase_edge_between(0, 1);
    ASSERT_EQ(g4.num_edges(), 2);
    ASSERT_EQ(g4.find_edge(0, 1)->data(), 100);
  }

  static_assert(std::is_move_constructible_v<Graph>, "graph is not movable");

  Graph g5(std::move(g3));
  ASSERT_EQ(g5.size(), 10);
  ASSERT_EQ(g5.num_nodes(), 10);
  ASSERT_EQ(g5.num_edges(), 0);
  for (int i = 0; i < g5.num_nodes(); ++i) {
    ASSERT_EQ(g5[i].data(), 10);
    ASSERT_EQ(g5.node(i).data(), 10);
  }
}

TYPED_TEST(BasicGraphTest, AssignmentTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;

  Graph g1(10, { 10 });
  g1.add_edge(0, 1, { 100 });

  {
    Graph g2(20, { 20 });
    g2 = g1;

    ASSERT_EQ(g1.num_nodes(), 10);
    ASSERT_EQ(g1.num_edges(), 1);
    for (int i = 0; i < g1.num_nodes(); ++i) {
      ASSERT_EQ(g1.node(i).data(), 10);
    }

    ASSERT_EQ(g2.num_nodes(), 10);
    ASSERT_EQ(g2.num_edges(), 1);
    for (int i = 0; i < g2.num_nodes(); ++i) {
      ASSERT_EQ(g2.node(i).data(), 10);
    }

    g2.add_edge(0, 2, { 200 });
    ASSERT_EQ(g1.num_edges(), 1);
    ASSERT_EQ(g2.num_edges(), 2);
  }

  static_assert(std::is_move_assignable_v<Graph>, "graph is not movable");

  Graph g3(10, { 10 });
  Graph g4(20, { 20 });
  g3 = std::move(g4);
  ASSERT_EQ(g3.num_nodes(), 20);
  ASSERT_EQ(g3.num_edges(), 0);
  for (int i = 0; i < g3.num_nodes(); ++i) {
    ASSERT_EQ(g3.node(i).data(), 20);
  }
}

TYPED_TEST(BasicGraphTest, AddNodeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;

  Graph graph;
  graph.add_node({ 0 });
  ASSERT_EQ(graph.num_nodes(), 1);
  ASSERT_EQ(graph.num_edges(), 0);
  ASSERT_EQ(graph.node(0).data(), 0);

  graph.add_node({ 1 });
  ASSERT_EQ(graph.num_nodes(), 2);
  ASSERT_EQ(graph.num_edges(), 0);
  ASSERT_EQ(graph.node(1).data(), 1);
}

TYPED_TEST(BasicGraphTest, AddEdgeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;

  Graph graph;
  graph.add_node({ 0 });
  graph.add_node({ 1 });
  graph.add_node({ 2 });
  graph.add_node({ 3 });
  graph.add_node({ 4 });
  graph.add_node({ 5 });

  TypeParam new_data { 6 };
  auto e0 = graph.add_edge(0, 1, new_data);
  auto e1 = graph.add_edge(2, 0, { 7 });
  auto e2 = graph.add_edge(3, 0, { 8 });

  ASSERT_EQ(graph.num_nodes(), 6);
  ASSERT_EQ(graph.num_edges(), 3);

  ASSERT_EQ(graph.edge(e0).src(), 0);
  ASSERT_EQ(graph.edge(e0).dst(), 1);
  ASSERT_EQ(graph.edge(e1).src(), 2);
  ASSERT_EQ(graph.edge(e1).dst(), 0);
  ASSERT_EQ(graph.edge(e2).src(), 3);
  ASSERT_EQ(graph.edge(e2).dst(), 0);

  for (int i = 0; i < graph.num_nodes(); ++i) {
    ASSERT_EQ(graph.node(i).data(), i);
  }

  std::vector<typename Graph::edge_id_type> edges { e0, e1, e2 };
  for (int i = 0; i < graph.num_edges(); ++i) {
    ASSERT_EQ(graph.edge(edges[i]).data(), i + graph.num_nodes());
  }

  auto it = graph.find_edge(0, 1);
  ASSERT_EQ(it->id(), e0);
  ASSERT_EQ(graph.find_edge(0, 2)->id(), e1);
  ASSERT_EQ(graph.find_edge(0, 3)->id(), e2);

  ASSERT_EQ(graph.find_edge(0, 4), graph.edge_end());

  it->data() = 100;

  const Graph &const_graph = graph;
  auto cit = const_graph.find_edge(0, 1);
  ASSERT_EQ(cit->data(), 100);
  ASSERT_EQ(const_graph.edge(cit->id()).data(), 100);
  static_assert(!std::is_assignable_v<decltype(cit->data()), TypeParam>,
                "const edge iterator should not be assignable");
}

template <class T>
class AdvancedGraphTest: public testing::Test {
public:
  using Graph = nuri::Graph<T, T>;

protected:
  void SetUp() override {
    graph_ = Graph();
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
  }

  Graph graph_;
  std::vector<typename Graph::edge_id_type> edges_;
};

TYPED_TEST_SUITE(AdvancedGraphTest, Implementations);

TYPED_TEST(AdvancedGraphTest, ClearEmptyTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  {
    Graph tmp(graph);
    tmp.clear_edge();

    ASSERT_EQ(tmp.num_nodes(), 11);
    ASSERT_TRUE(tmp.edge_empty());
  }

  ASSERT_EQ(graph.num_nodes(), 11);
  ASSERT_EQ(graph.num_edges(), 10);

  graph.clear();
  ASSERT_TRUE(graph.empty());
  ASSERT_TRUE(graph.edge_empty());
}

TYPED_TEST(AdvancedGraphTest, UpdateNodeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  typename Graph::node_data_type new_data = { 1000 };
  graph.update_node(0, new_data);
  graph.update_node(1, { 1001 });

  ASSERT_EQ(graph.node(0).data(), 1000);
  ASSERT_EQ(graph.node(1).data(), 1001);

  static_assert(!std::is_assignable_v<decltype(graph.node(0).as_const().data()),
                                      typename Graph::node_data_type>,
                "const node wrapper should not be assignable");
  ASSERT_EQ(graph.node(0).as_const().data(), 1000);
  ASSERT_EQ(graph.node(1).as_const().data(), 1001);
}

TYPED_TEST(AdvancedGraphTest, NodeIteratorTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  int cnt = 0;
  for (auto it = graph.begin(); it != graph.end(); ++it) {
    it->data() = { -1 };
    cnt++;

    NURIKIT_ASSERT_ONEWAY_CONVERTIBLE(node_iterator, decltype(it),
                                      const_node_iterator,
                                      typename Graph::const_node_iterator);
  }
  ASSERT_EQ(cnt, graph.num_nodes());

  const auto &const_graph = graph;
  for (auto it = const_graph.begin(); it != const_graph.end(); ++it) {
    static_assert(!std::is_assignable_v<decltype(it->data()),
                                        typename Graph::node_data_type>,
                  "const_iterator should be read-only");
    ASSERT_EQ(it->data(), -1);
  }

  cnt = 0;
  for (auto it = graph.begin(); it != graph.end(); it++) {
    it->data() = { -10 };
    cnt++;
  }
  ASSERT_EQ(cnt, graph.num_nodes());
  for (auto it = const_graph.begin(); it != const_graph.end(); it++) {
    ASSERT_EQ(it->data(), -10);
  }

  cnt = 0;
  for (auto it = --graph.end(); it >= graph.begin(); it--) {
    it->data() = { -100 };
    cnt++;
  }
  ASSERT_EQ(cnt, graph.num_nodes());
  for (auto it = const_graph.begin(); it != const_graph.end(); it++) {
    ASSERT_EQ(it->data(), -100);
  }

  for (auto it = graph.begin(); it < graph.end(); ++it) {
    it->data() = { -1000 };
  }
  cnt = 0;
  for (auto it = --const_graph.end(); it >= const_graph.begin(); it--) {
    ASSERT_EQ(it->data(), -1000);
    cnt++;
  }
  ASSERT_EQ(cnt, graph.num_nodes());
}

TYPED_TEST(AdvancedGraphTest, UpdateEdgeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;
  std::vector<typename Graph::edge_id_type> &edges = this->edges_;

  typename Graph::edge_data_type new_data = { 1000 };
  graph.update_edge(edges[0], new_data);
  graph.update_edge(edges[1], { 1001 });

  ASSERT_EQ(graph.edge(edges[0]).data(), 1000);
  ASSERT_EQ(graph.edge(edges[1]).data(), 1001);
  static_assert(
    !std::is_assignable_v<decltype(graph.edge(edges[0]).as_const().data()),
                          typename Graph::edge_data_type>,
    "const edge wrapper should not be assignable");
  ASSERT_EQ(graph.edge(edges[0]).as_const().data(), 1000);
  ASSERT_EQ(graph.edge(edges[1]).as_const().data(), 1001);
}

TYPED_TEST(AdvancedGraphTest, EdgeIteratorTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  int cnt = 0;
  for (auto it = graph.edge_begin(); it != graph.edge_end(); ++it) {
    it->data() = { -1 };
    cnt++;

    NURIKIT_ASSERT_ONEWAY_CONVERTIBLE(edge_iterator, decltype(it),
                                      const_edge_iterator,
                                      typename Graph::const_edge_iterator);
  }
  ASSERT_EQ(cnt, graph.num_edges());

  const Graph &const_graph = graph;
  for (auto it = const_graph.edge_begin(); it != const_graph.edge_end(); ++it) {
    static_assert(!std::is_assignable_v<decltype(it->data()),
                                        typename Graph::edge_data_type>,
                  "const_iterator should be read-only");
    ASSERT_EQ(it->data(), -1);
  }

  cnt = 0;
  for (auto it = std::make_reverse_iterator(graph.edge_end());
       it != std::make_reverse_iterator(graph.edge_begin()); ++it) {
    it->data() = { -10 };
    cnt++;
  }
  ASSERT_EQ(cnt, graph.num_edges());

  for (auto it = --const_graph.edge_end(); it != --const_graph.edge_begin();
       it--) {
    static_assert(!std::is_assignable_v<decltype(it->data()),
                                        typename Graph::edge_data_type>,
                  "const_iterator should be read-only");
    ASSERT_EQ(it->data(), -10);
  }
}

TYPED_TEST(AdvancedGraphTest, AdjacencyTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  ASSERT_EQ(graph.degree(0), 4);
  ASSERT_EQ(graph.degree(1), 4);
  ASSERT_EQ(graph.degree(2), 4);
  ASSERT_EQ(graph.degree(3), 2);
  ASSERT_EQ(graph.degree(4), 1);
  ASSERT_EQ(graph.degree(5), 1);
  ASSERT_EQ(graph.degree(6), 1);
  ASSERT_EQ(graph.degree(7), 1);
  ASSERT_EQ(graph.degree(8), 1);
  ASSERT_EQ(graph.degree(9), 1);
  ASSERT_EQ(graph.degree(10), 0);

  auto adj01 = *graph.find_adjacent(0, 1);
  ASSERT_EQ(adj01.src(), 0);
  ASSERT_EQ(adj01.dst(), 1);
  ASSERT_EQ(adj01.edge_data(), 100);
  adj01.edge_data() = 1000;

  auto adj10 = *graph.find_adjacent(1, 0);
  ASSERT_EQ(adj10.src(), 1);
  ASSERT_EQ(adj10.dst(), 0);
  ASSERT_EQ(adj10.edge_data(), 1000);
  ASSERT_EQ(adj10.as_const().edge_data(), 1000);

  ASSERT_TRUE(graph.find_adjacent(0, 10).end());

  const Graph &const_graph = graph;
  auto cadj01 = *const_graph.find_adjacent(0, 1);
  static_assert(!std::is_assignable_v<decltype(cadj01.edge_data()),
                                      typename Graph::edge_data_type>,
                "const_iterator should be read-only");
  ASSERT_EQ(cadj01.edge_data(), 1000);
}

TYPED_TEST(AdvancedGraphTest, AdjIteratorTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  ASSERT_EQ(graph.node(0).adj_begin(), graph.adj_begin(0));
  ASSERT_EQ(graph.node(0).adj_end(), graph.adj_end(0));

  int cnt = 0;
  for (auto it = graph.adj_begin(0); !it.end(); ++it) {
    ASSERT_EQ(it->src(), 0);
    it->edge_data() = { -1 };
    cnt++;

    NURIKIT_ASSERT_ONEWAY_CONVERTIBLE(adj_iterator, decltype(it),
                                      const_adjacency_iterator,
                                      typename Graph::const_adjacency_iterator);
  }
  ASSERT_EQ(cnt, graph.degree(0));

  const Graph &const_graph = graph;
  for (auto it = const_graph.adj_begin(0); !it.end(); ++it) {
    static_assert(!std::is_assignable_v<decltype(it->edge_data()),
                                        typename Graph::edge_data_type>,
                  "const_iterator should be read-only");
    ASSERT_EQ(it->edge_data(), -1);
  }

  auto it1 = graph.adj_begin(0);
  ASSERT_EQ(it1 + graph.degree(0), graph.adj_end(0));
  ASSERT_EQ(it1, graph.adj_end(0) - graph.degree(0));

  it1 += 2;
  ASSERT_EQ(it1, graph.adj_begin(0) + 2);
  it1 -= 2;
  ASSERT_EQ(it1, graph.adj_begin(0));

  ASSERT_EQ(it1[2].edge_data(), -1);
}

TYPED_TEST(AdvancedGraphTest, PopNodeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  auto data = graph.pop_node(0);
  ASSERT_EQ(data, 0);
  ASSERT_EQ(graph.num_nodes(), 10);
  ASSERT_EQ(graph.num_edges(), 6);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(graph.node(i).id(), i);
    ASSERT_EQ(graph.node(i).data(), i + 1);
  }

  ASSERT_NE(graph.find_adjacent(0, 1), graph.adj_end(0));
  ASSERT_NE(graph.find_adjacent(0, 6), graph.adj_end(0));
  ASSERT_NE(graph.find_adjacent(0, 7), graph.adj_end(0));
  ASSERT_NE(graph.find_adjacent(1, 2), graph.adj_end(1));
  ASSERT_NE(graph.find_adjacent(1, 8), graph.adj_end(1));
  ASSERT_NE(graph.find_adjacent(2, 3), graph.adj_end(2));

  data = graph.pop_node(1);
  ASSERT_EQ(data, 2);
  ASSERT_EQ(graph.num_nodes(), 9);
  ASSERT_EQ(graph.num_edges(), 3);
  ASSERT_NE(graph.find_adjacent(0, 5), graph.adj_end(0));
  ASSERT_NE(graph.find_adjacent(0, 6), graph.adj_end(0));
  ASSERT_NE(graph.find_adjacent(1, 2), graph.adj_end(1));
}

TYPED_TEST(AdvancedGraphTest, EraseNoNodeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  graph.erase_nodes(graph.end(), graph.begin());
  graph.erase_nodes(graph.begin(), graph.end(), [](auto) { return false; });

  ASSERT_EQ(graph.num_nodes(), 11);
  ASSERT_EQ(graph.num_edges(), 10);
}

TYPED_TEST(AdvancedGraphTest, EraseAllNodesTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  graph.erase_nodes(graph.begin(), graph.end());

  ASSERT_EQ(graph.num_edges(), 0);
}

TYPED_TEST(AdvancedGraphTest, EraseExceptOneNodeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  graph.erase_nodes(graph.begin(), graph.end() - 1);

  ASSERT_EQ(graph.num_nodes(), 1);
  ASSERT_TRUE(graph.edge_empty());
}

TYPED_TEST(AdvancedGraphTest, EraseLeadingNodesTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  graph.erase_nodes(graph.begin(), graph.begin() + 3);

  ASSERT_EQ(graph.num_nodes(), 8);
  ASSERT_EQ(graph.num_edges(), 1);

  ASSERT_EQ(graph.find_adjacent(0, 1)->edge_data(), 109);
}

TYPED_TEST(AdvancedGraphTest, EraseTrailingNodesTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  graph.erase_nodes(graph.end() - 3, graph.end());

  ASSERT_EQ(graph.num_nodes(), 8);
  ASSERT_EQ(graph.num_edges(), 8);

  ASSERT_EQ(graph.find_adjacent(0, 1)->edge_data(), 100);
  ASSERT_EQ(graph.find_adjacent(0, 2)->edge_data(), 101);
  ASSERT_EQ(graph.find_adjacent(0, 5)->edge_data(), 102);
  ASSERT_EQ(graph.find_adjacent(0, 6)->edge_data(), 103);
  ASSERT_EQ(graph.find_adjacent(1, 2)->edge_data(), 104);
  ASSERT_EQ(graph.find_adjacent(1, 7)->edge_data(), 105);
  ASSERT_EQ(graph.find_adjacent(2, 3)->edge_data(), 107);
  ASSERT_EQ(graph.find_adjacent(3, 4)->edge_data(), 109);
}

TYPED_TEST(AdvancedGraphTest, EraseMixedNodesTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  graph.erase_nodes(graph.begin(), graph.end(),
                    [](auto node) { return node.id() % 2 == 0; });

  ASSERT_EQ(graph.num_nodes(), 5);
  ASSERT_EQ(graph.num_edges(), 1);

  ASSERT_EQ(graph.find_adjacent(0, 3)->edge_data(), 105);
}

TYPED_TEST(AdvancedGraphTest, PopEdgeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;
  std::vector<typename Graph::edge_id_type> &edges = this->edges_;

  auto data = graph.pop_edge(edges[0]);
  ASSERT_EQ(data, 100);
  ASSERT_EQ(graph.num_edges(), 9);
  ASSERT_EQ(graph.find_adjacent(0, 1), graph.adj_end(0));

  data = graph.pop_edge(edges[2]);
  ASSERT_EQ(data, 102);
  ASSERT_EQ(graph.num_edges(), 8);
  ASSERT_EQ(graph.find_adjacent(0, 5), graph.adj_end(0));

  ASSERT_EQ(graph.num_nodes(), 11);
}

TYPED_TEST(AdvancedGraphTest, EraseEdgeTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;
  std::vector<typename Graph::edge_id_type> &edges = this->edges_;

  graph.erase_edge(edges[0]);
  ASSERT_EQ(graph.num_edges(), 9);
  ASSERT_EQ(graph.find_adjacent(0, 1), graph.adj_end(0));

  graph.erase_edge(edges[2]);
  ASSERT_EQ(graph.num_edges(), 8);
  ASSERT_EQ(graph.find_adjacent(0, 5), graph.adj_end(0));

  ASSERT_FALSE(graph.erase_edge_between(0, 1));
  ASSERT_TRUE(graph.erase_edge_between(2, 0));
  ASSERT_EQ(graph.find_adjacent(0, 2), graph.adj_end(0));
  ASSERT_EQ(graph.find_adjacent(2, 0), graph.adj_end(2));

  ASSERT_EQ(graph.num_nodes(), 11);
}

TYPED_TEST(AdvancedGraphTest, EraseAllEdgesTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  for (auto it = graph.edge_begin(); it != graph.edge_end();) {
    graph.erase_edge(it++);
  }
  ASSERT_EQ(graph.num_nodes(), 11);
  ASSERT_TRUE(graph.edge_empty());
}

TYPED_TEST(AdvancedGraphTest, EraseMixedEdgesTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  int i = 0;
  for (auto it = graph.edge_begin(); it != graph.edge_end();) {
    if (i++ % 2 == 0) {
      graph.erase_edge(it++);
    } else {
      ++it;
    }
  }

  ASSERT_EQ(graph.num_nodes(), 11);
  ASSERT_EQ(graph.num_edges(), 5);

  ASSERT_EQ(graph.find_adjacent(0, 1), graph.adj_end(0));
  ASSERT_EQ(graph.find_adjacent(0, 2)->edge_data(), 101);
  ASSERT_EQ(graph.find_adjacent(0, 5), graph.adj_end(0));
  ASSERT_EQ(graph.find_adjacent(0, 6)->edge_data(), 103);
  ASSERT_EQ(graph.find_adjacent(1, 2), graph.adj_end(1));
  ASSERT_EQ(graph.find_adjacent(1, 7)->edge_data(), 105);
  ASSERT_EQ(graph.find_adjacent(1, 8), graph.adj_end(1));
  ASSERT_EQ(graph.find_adjacent(2, 3)->edge_data(), 107);
  ASSERT_EQ(graph.find_adjacent(2, 9), graph.adj_end(2));
  ASSERT_EQ(graph.find_adjacent(3, 4)->edge_data(), 109);
}

TYPED_TEST(AdvancedGraphTest, EraseAddTest) {
  using Graph = nuri::Graph<TypeParam, TypeParam>;
  Graph &graph = this->graph_;

  graph.pop_node(1);
  graph.add_node({ 11 });
  ASSERT_EQ(graph.num_nodes(), 11);

  graph.erase_edge_between(0, 1);
  auto e = graph.add_edge(7, 10, 110);
  ASSERT_EQ(graph.num_edges(), 6);
  ASSERT_EQ(graph.find_adjacent(7, 10)->eid(), e);
  ASSERT_EQ(graph.find_adjacent(7, 10)->edge_data(), 110);
}
}  // namespace

// Explicit instantiation of few template classes for coverage report.
namespace nuri {
template class Graph<int, int>;
template class Graph<NonTrivial, NonTrivial>;

namespace internal {
  // NOLINTBEGIN(cppcoreguidelines-macro-usage)
  // Non-const version of wrappers could not be instantiated due to the
  // converting constructor. Non-const iterators could be instantiated.
#define NURIKIT_INSTANTIATE_TEMPLATES_WITH_BASE(GraphType, iterator, RefType)  \
  template class RefType##Wrapper<GraphType, true>;                            \
  template class DataIteratorBase<GraphType::iterator, GraphType,              \
                                  GraphType::RefType##Ref, false>;             \
  template class RefType##Iterator<GraphType, false>;

#define NURIKIT_INSTANTIATE_ALL_TEMPLATES(GraphType)                           \
  NURIKIT_INSTANTIATE_TEMPLATES_WITH_BASE(GraphType, iterator, Node)           \
  NURIKIT_INSTANTIATE_TEMPLATES_WITH_BASE(GraphType, adjacency_iterator, Adj)  \
  template class EdgeWrapper<GraphType, true>;                                 \
  template class EdgeIterator<GraphType, true>;

  NURIKIT_INSTANTIATE_ALL_TEMPLATES(TrivialGraph)
  NURIKIT_INSTANTIATE_ALL_TEMPLATES(NonTrivialGraph)
  // NOLINTEND(cppcoreguidelines-macro-usage)
}  // namespace internal
}  // namespace nuri

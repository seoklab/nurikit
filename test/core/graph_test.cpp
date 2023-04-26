//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/graph.h"

#include <type_traits>

#include "gtest/gtest.h"

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
using Implementations = Types<TrivialGraph, NonTrivialGraph>;

template <class T>
class BasicGraphTest: public testing::Test { };

TYPED_TEST_SUITE(BasicGraphTest, Implementations);

TYPED_TEST(BasicGraphTest, CreationTest) {
  TypeParam g1;
  ASSERT_EQ(g1.size(), 0);
  ASSERT_EQ(g1.num_nodes(), 0);
  ASSERT_EQ(g1.num_edges(), 0);

  TypeParam g2(10);
  ASSERT_EQ(g2.size(), 10);
  ASSERT_EQ(g2.num_nodes(), 10);
  ASSERT_EQ(g2.num_edges(), 0);

  TypeParam g3(10, { 10 });
  ASSERT_EQ(g3.size(), 10);
  ASSERT_EQ(g3.num_nodes(), 10);
  ASSERT_EQ(g3.num_edges(), 0);
  for (int i = 0; i < g3.num_nodes(); ++i) {
    ASSERT_EQ(g3[i].data(), 10);
    ASSERT_EQ(g3.node(i).data(), 10);
  }

  const TypeParam &const_graph = g3;
  for (int i = 0; i < const_graph.num_nodes(); ++i) {
    ASSERT_EQ(const_graph[i].data(), 10);
    ASSERT_EQ(const_graph.node(i).data(), 10);
  }

  TypeParam g4(g3);
  ASSERT_EQ(g4.size(), 10);
  ASSERT_EQ(g4.num_nodes(), 10);
  ASSERT_EQ(g4.num_edges(), 0);
  for (int i = 0; i < g4.num_nodes(); ++i) {
    ASSERT_EQ(g4[i].data(), 10);
    ASSERT_EQ(g4.node(i).data(), 10);
  }

  static_assert(std::is_move_constructible_v<TypeParam>,
                "graph is not movable");

  TypeParam g5(std::move(g3));
  ASSERT_EQ(g5.size(), 10);
  ASSERT_EQ(g5.num_nodes(), 10);
  ASSERT_EQ(g5.num_edges(), 0);
  for (int i = 0; i < g5.num_nodes(); ++i) {
    ASSERT_EQ(g5[i].data(), 10);
    ASSERT_EQ(g5.node(i).data(), 10);
  }
}

TYPED_TEST(BasicGraphTest, AssignmentTest) {
  TypeParam g1(10, { 10 });
  TypeParam g2(20, { 20 });
  g1 = g2;
  ASSERT_EQ(g1.num_nodes(), 20);
  ASSERT_EQ(g1.num_edges(), 0);
  for (int i = 0; i < g1.num_nodes(); ++i) {
    ASSERT_EQ(g1.node(i).data(), 20);
  }
  ASSERT_EQ(g2.num_nodes(), 20);
  ASSERT_EQ(g2.num_edges(), 0);
  for (int i = 0; i < g2.num_nodes(); ++i) {
    ASSERT_EQ(g2.node(i).data(), 20);
  }

  static_assert(std::is_move_assignable_v<TypeParam>, "graph is not movable");

  TypeParam g3(10, { 10 });
  TypeParam g4(20, { 20 });
  g3 = std::move(g4);
  ASSERT_EQ(g3.num_nodes(), 20);
  ASSERT_EQ(g3.num_edges(), 0);
  for (int i = 0; i < g3.num_nodes(); ++i) {
    ASSERT_EQ(g3.node(i).data(), 20);
  }
}

TYPED_TEST(BasicGraphTest, AddNodeTest) {
  TypeParam graph;
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
  TypeParam graph;
  graph.add_node({ 0 });
  graph.add_node({ 1 });
  graph.add_node({ 2 });
  graph.add_node({ 3 });
  graph.add_node({ 4 });
  graph.add_node({ 5 });

  typename TypeParam::edge_data_type new_data { 6 };
  graph.add_edge(0, 1, new_data);
  graph.add_edge(2, 0, { 7 });
  graph.add_edge(3, 0, { 8 });

  ASSERT_EQ(graph.num_nodes(), 6);
  ASSERT_EQ(graph.num_edges(), 3);

  ASSERT_EQ(graph.edge(0).src(), 0);
  ASSERT_EQ(graph.edge(0).dst(), 1);
  ASSERT_EQ(graph.edge(1).src(), 2);
  ASSERT_EQ(graph.edge(1).dst(), 0);
  ASSERT_EQ(graph.edge(2).src(), 3);
  ASSERT_EQ(graph.edge(2).dst(), 0);

  for (int i = 0; i < graph.num_nodes(); ++i) {
    ASSERT_EQ(graph.node(i).data(), i);
  }
  for (int i = 0; i < graph.num_edges(); ++i) {
    ASSERT_EQ(graph.edge(i).data(), i + graph.num_nodes());
  }

  auto it = graph.find_edge(0, 1);
  ASSERT_EQ(it->id(), 0);
  ASSERT_EQ(graph.find_edge(0, 2)->id(), 1);
  ASSERT_EQ(graph.find_edge(0, 3)->id(), 2);

  ASSERT_EQ(graph.find_edge(0, 4), graph.edge_end());

  it->data() = 100;

  const auto &const_graph = graph;
  auto cit = const_graph.find_edge(0, 1);
  ASSERT_EQ(cit->data(), 100);
  static_assert(!std::is_assignable_v<decltype(cit->data()),
                                      typename TypeParam::edge_data_type>,
                "const edge iterator should not be assignable");
}

template <class T>
class AdvancedGraphTest: public testing::Test {
protected:
  void SetUp() override {
    graph_ = T();
    graph_.reserve(11);

    for (int i = 0; i < 11; ++i) {
      graph_.add_node({ i });
    }

    graph_.add_edge(0, 1, { 100 });
    graph_.add_edge(0, 2, { 101 });
    graph_.add_edge(5, 0, { 102 });
    graph_.add_edge(6, 0, { 103 });

    graph_.add_edge(1, 2, { 104 });
    graph_.add_edge(7, 1, { 105 });
    graph_.add_edge(8, 1, { 106 });

    graph_.add_edge(2, 3, { 107 });
    graph_.add_edge(9, 2, { 108 });

    graph_.add_edge(3, 4, { 109 });
  }

  T graph_;
};

TYPED_TEST_SUITE(AdvancedGraphTest, Implementations);

TYPED_TEST(AdvancedGraphTest, UpdateNodeTest) {
  auto &graph = this->graph_;

  typename TypeParam::node_data_type new_data = { 1000 };
  graph.update_node(0, new_data);
  graph.update_node(1, { 1001 });

  ASSERT_EQ(graph.node(0).data(), 1000);
  ASSERT_EQ(graph.node(1).data(), 1001);

  static_assert(!std::is_assignable_v<decltype(graph.node(0).as_const().data()),
                                      typename TypeParam::node_data_type>,
                "const node wrapper should not be assignable");
  ASSERT_EQ(graph.node(0).as_const().data(), 1000);
  ASSERT_EQ(graph.node(1).as_const().data(), 1001);
}

TYPED_TEST(AdvancedGraphTest, NodeIteratorTest) {
  auto &graph = this->graph_;
  int cnt = 0;
  for (auto it = graph.begin(); it != graph.end(); ++it) {
    it->data() = { -1 };
    cnt++;
  }
  ASSERT_EQ(cnt, graph.num_nodes());

  const auto &const_graph = graph;
  for (auto it = const_graph.begin(); it != const_graph.end(); ++it) {
    static_assert(!std::is_assignable_v<decltype(it->data()),
                                        typename TypeParam::node_data_type>,
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
  auto &graph = this->graph_;

  typename TypeParam::edge_data_type new_data = { 1000 };
  graph.update_edge(0, new_data);
  graph.update_edge(1, { 1001 });

  ASSERT_EQ(graph.edge(0).data(), 1000);
  ASSERT_EQ(graph.edge(1).data(), 1001);
  static_assert(!std::is_assignable_v<decltype(graph.edge(0).as_const().data()),
                                      typename TypeParam::edge_data_type>,
                "const edge wrapper should not be assignable");
  ASSERT_EQ(graph.edge(0).as_const().data(), 1000);
  ASSERT_EQ(graph.edge(1).as_const().data(), 1001);
}

TYPED_TEST(AdvancedGraphTest, EdgeIteratorTest) {
  auto &graph = this->graph_;
  int cnt = 0;
  for (auto it = graph.edge_begin(); it != graph.edge_end(); ++it) {
    it->data() = { -1 };
    cnt++;

    static_assert(
      std::is_convertible_v<decltype(it),
                            typename TypeParam::const_edge_iterator>,
      "edge_iterator should be convertible to const_edge_iterator");
    static_assert(
      std::is_constructible_v<decltype(it),
                              typename TypeParam::const_edge_iterator>,
      "const_edge_iterator should be constructible from edge_iterator");
    static_assert(std::is_assignable_v<decltype(it),
                                       typename TypeParam::const_edge_iterator>,
                  "edge_iterator should be assignable to const_edge_iterator");
  }
  ASSERT_EQ(cnt, graph.num_edges());

  const auto &const_graph = graph;
  for (auto it = const_graph.edge_begin(); it != const_graph.edge_end(); ++it) {
    static_assert(!std::is_assignable_v<decltype(it->data()),
                                        typename TypeParam::edge_data_type>,
                  "const_iterator should be read-only");
    ASSERT_EQ(it->data(), -1);
  }
}

TYPED_TEST(AdvancedGraphTest, AdjacencyTest) {
  auto &graph = this->graph_;

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

  ASSERT_EQ(graph.find_adjacent(0, 10), graph.adj_end(0));

  const TypeParam &const_graph = graph;
  auto cadj01 = *const_graph.find_adjacent(0, 1);
  static_assert(!std::is_assignable_v<decltype(cadj01.edge_data()),
                                      typename TypeParam::edge_data_type>,
                "const_iterator should be read-only");
  ASSERT_EQ(cadj01.edge_data(), 1000);
}

TYPED_TEST(AdvancedGraphTest, AdjIteratorTest) {
  auto &graph = this->graph_;
  ASSERT_EQ(graph.node(0).adj_begin(), graph.adj_begin(0));
  ASSERT_EQ(graph.node(0).adj_end(), graph.adj_end(0));

  int cnt = 0;
  for (auto it = graph.adj_begin(0); it != graph.adj_end(0); ++it) {
    ASSERT_EQ(it->src(), 0);

    ASSERT_EQ(it->src(), it->src_node().id());
    ASSERT_EQ(it->dst(), it->dst_node().id());

    it->edge_data() = { -1 };
    cnt++;
  }
  ASSERT_EQ(cnt, graph.degree(0));

  const auto &const_graph = graph;
  for (auto it = const_graph.adj_begin(0); it != const_graph.adj_end(0); ++it) {
    static_assert(!std::is_assignable_v<decltype(it->edge_data()),
                                        typename TypeParam::edge_data_type>,
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
  NURIKIT_INSTANTIATE_TEMPLATES_WITH_BASE(GraphType, edge_iterator, Edge)      \
  NURIKIT_INSTANTIATE_TEMPLATES_WITH_BASE(GraphType, adjacency_iterator, Adj)

  NURIKIT_INSTANTIATE_ALL_TEMPLATES(TrivialGraph);
  NURIKIT_INSTANTIATE_ALL_TEMPLATES(NonTrivialGraph);
  // NOLINTEND(cppcoreguidelines-macro-usage)
}  // namespace internal
}  // namespace nuri

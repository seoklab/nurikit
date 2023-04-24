//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.h"

#include <type_traits>

#include "gtest/gtest.h"

namespace {
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

using TrivialGraph = nuri::Graph<int, int>;
using NonTrivialGraph = nuri::Graph<NonTrivial, NonTrivial>;

using testing::Types;
using Implementations = Types<TrivialGraph, NonTrivialGraph>;

template <class T>
class GraphTest: public testing::Test {
protected:
  void SetUp() override { graph_ = T(); }

  T graph_;
};

TYPED_TEST_SUITE(GraphTest, Implementations);

TYPED_TEST(GraphTest, CreationTest) {
  TypeParam g1;
  ASSERT_EQ(g1.num_nodes(), 0);
  ASSERT_EQ(g1.num_edges(), 0);

  TypeParam g2(10);
  ASSERT_EQ(g2.num_nodes(), 10);
  ASSERT_EQ(g2.num_edges(), 0);

  TypeParam g3(10, { 10 });
  ASSERT_EQ(g3.num_nodes(), 10);
  ASSERT_EQ(g3.num_edges(), 0);
  for (int i = 0; i < g3.num_nodes(); ++i) {
    ASSERT_EQ(g3.node(i).data(), 10);
  }
}

TYPED_TEST(GraphTest, AssignmentTest) {
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

TYPED_TEST(GraphTest, NodeIteratorTest) {
  auto &graph = this->graph_;
  graph.add_node({ 0 });
  ASSERT_EQ(graph.node(0).data(), 0);

  int cnt = 0;
  for (auto it = graph.begin(); it != graph.end(); ++it) {
    it->data() = { 1 };
    cnt++;
  }
  ASSERT_EQ(graph.node(0).data(), 1);
  ASSERT_EQ(cnt, 1);

  const auto &const_graph = graph;
  for (auto it = const_graph.begin(); it != const_graph.end(); ++it) {
    static_assert(!std::is_assignable<decltype(it->data()), int>::value,
                  "const_iterator should be read-only");
  }
}
}  // namespace

namespace nuri {
// Explicit instantiation of few template classes for testing.
template class nuri::Graph<int, int>;
template class nuri::Graph<NonTrivial, NonTrivial>;
}  // namespace nuri

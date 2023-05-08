//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_H_
#define NURI_CORE_GRAPH_H_

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/utils.h"

namespace nuri {
template <class NT, class ET>
class Graph;

namespace internal {
  template <class Parent>
  class ArrowHelper {
  public:
    constexpr ArrowHelper(Parent &&p) noexcept: p_(p) { }

    constexpr Parent *operator->() noexcept { return &p_; }
    constexpr const Parent *operator->() const noexcept { return &p_; }

  private:
    Parent p_;
  };

  template <class Derived, class GT, class DT, bool is_const>
  class DataIteratorBase {
  public:
    using parent_type = const_if_t<is_const, GT>;

    using difference_type = int;
    using value_type = DT;
    using pointer = ArrowHelper<value_type>;
    using reference = value_type &;
    using iterator_category = std::random_access_iterator_tag;

    constexpr DataIteratorBase(parent_type *graph,
                               difference_type index) noexcept
      : graph_(graph), index_(index) { }

    template <class Other,
              class = std::enable_if_t<!std::is_same_v<Derived, Other>>>
    constexpr DataIteratorBase(const Other &other) noexcept
      : graph_(other.graph_), index_(other.index_) { }

    template <class Other,
              class = std::enable_if_t<!std::is_same_v<Derived, Other>>>
    constexpr DataIteratorBase &operator=(const Other &other) noexcept {
      graph_ = other.graph_;
      index_ = other.index_;
      return *this;
    }

    constexpr Derived &operator++() noexcept {
      ++index_;
      return *derived();
    }

    constexpr Derived operator++(int) noexcept {
      Derived tmp(*derived());
      ++index_;
      return tmp;
    }

    constexpr Derived &operator--() noexcept {
      --index_;
      return *derived();
    }

    constexpr Derived operator--(int) noexcept {
      Derived tmp(*derived());
      --index_;
      return tmp;
    }

    constexpr Derived &operator+=(difference_type n) noexcept {
      index_ += n;
      return *derived();
    }

    constexpr Derived operator+(difference_type n) const noexcept {
      return derived()->advance(n);
    }

    constexpr Derived &operator-=(difference_type n) noexcept {
      index_ -= n;
      return *derived();
    }

    constexpr Derived operator-(difference_type n) const noexcept {
      return derived()->advance(-n);
    }

    template <class Other,
              class = std::enable_if_t<std::is_convertible_v<Other, Derived>>>
    constexpr difference_type operator-(const Other &other) const noexcept {
      return index_ - other.index_;
    }

    constexpr value_type operator*() const noexcept {
      return derived()->deref(graph_, index_);
    }

    constexpr value_type operator[](difference_type n) const noexcept {
      return *(*derived() + n);
    }

    constexpr pointer operator->() const noexcept { return **derived(); }

    template <class Other,
              class = std::enable_if_t<std::is_convertible_v<Other, Derived>>>
    constexpr bool operator<(const Other &other) const noexcept {
      return index_ < other.index_;
    }

    template <class Other,
              class = std::enable_if_t<std::is_convertible_v<Other, Derived>>>
    constexpr bool operator>(const Other &other) const noexcept {
      return index_ > other.index_;
    }

    template <class Other,
              class = std::enable_if_t<std::is_convertible_v<Other, Derived>>>
    constexpr bool operator<=(const Other &other) const noexcept {
      return index_ <= other.index_;
    }

    template <class Other,
              class = std::enable_if_t<std::is_convertible_v<Other, Derived>>>
    constexpr bool operator>=(const Other &other) const noexcept {
      return index_ >= other.index_;
    }

    template <class Other,
              class = std::enable_if_t<std::is_convertible_v<Other, Derived>>>
    constexpr bool operator==(const Other &other) const noexcept {
      return index_ == other.index_;
    }

    template <class Other,
              class = std::enable_if_t<std::is_convertible_v<Other, Derived>>>
    constexpr bool operator!=(const Other &other) const noexcept {
      return !(*this == other);
    }

  protected:
    template <class, class, class, bool>
    friend class DataIteratorBase;

    constexpr Derived *derived() noexcept {
      return static_cast<Derived *>(this);
    }

    constexpr const Derived *derived() const noexcept {
      return static_cast<const Derived *>(this);
    }

    template <class... Args>
    constexpr Derived advance(difference_type n,
                              Args &&...args) const noexcept {
      return Derived(graph_, index_ + n, std::forward<Args>(args)...);
    }

  private:
    parent_type *graph_;
    difference_type index_;
  };

  template <class GT, bool is_const>
  class AdjIterator;

  template <class GT, bool is_const>
  class NodeWrapper {
  public:
    using DT = typename GT::node_data_type;

    using parent_type = const_if_t<is_const, GT>;
    using value_type = const_if_t<is_const, DT>;

    template <bool other_const>
    using Other = NodeWrapper<GT, other_const>;

    constexpr NodeWrapper(int nid, DT &data, parent_type &graph) noexcept
      : nid_(nid), data_(&data), graph_(&graph) { }

    constexpr NodeWrapper(int nid, const DT &data, parent_type &graph) noexcept
      : nid_(nid), data_(&data), graph_(&graph) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr NodeWrapper(const Other<other_const> &other) noexcept
      : nid_(other.nid_), data_(other.data_), graph_(other.graph_) { }

    constexpr int id() const noexcept { return nid_; }

    constexpr value_type &data() const noexcept {
      return *const_cast<value_type *>(data_);
    }

    constexpr AdjIterator<GT, is_const> adj_begin() const noexcept {
      return graph_->adj_begin(nid_);
    }

    constexpr AdjIterator<GT, is_const> adj_end() const noexcept {
      return graph_->adj_end(nid_);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class NodeWrapper;

    int nid_;
    value_type *data_;
    parent_type *graph_;
  };

  template <class GT, bool is_const>
  class NodeIterator
    : public DataIteratorBase<NodeIterator<GT, is_const>, GT,
                              NodeWrapper<GT, is_const>, is_const> {
  public:
    using Base = DataIteratorBase<NodeIterator<GT, is_const>, GT,
                                  NodeWrapper<GT, is_const>, is_const>;

    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    using Base::Base;

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr NodeIterator(const NodeIterator<GT, other_const> &other) noexcept
      : Base(other) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr NodeIterator &
    operator=(const NodeIterator<GT, other_const> &other) noexcept {
      Base::operator=(other);
      return *this;
    }

  private:
    friend Base;
    template <class, bool other_const>
    friend class NodeIterator;

    constexpr value_type deref(typename Base::parent_type *graph,
                               difference_type index) const noexcept {
      return graph->node(index);
    }
  };

  template <class GT, bool is_const>
  class EdgeWrapper {
  public:
    using ET = typename GT::edge_type;
    using DT = typename GT::edge_data_type;

    using parent_type = const_if_t<is_const, GT>;
    using value_type = const_if_t<is_const, DT>;

    template <bool other_const>
    using Other = EdgeWrapper<GT, other_const>;

    constexpr EdgeWrapper(int eid, ET &edge, parent_type &graph) noexcept
      : eid_(eid), edge_(&edge), graph_(&graph) { }

    constexpr EdgeWrapper(int eid, const ET &edge, parent_type &graph) noexcept
      : eid_(eid), edge_(&edge), graph_(&graph) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr EdgeWrapper(const Other<other_const> &other) noexcept
      : eid_(other.eid_), edge_(other.edge_), graph_(other.graph_) { }

    constexpr int id() const noexcept { return eid_; }
    constexpr int src() const noexcept { return edge_->src; }
    constexpr int dst() const noexcept { return edge_->dst; }

    constexpr value_type &data() const noexcept {
      return const_cast<value_type &>(edge_->data);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class EdgeWrapper;

    int eid_;
    const_if_t<is_const, ET> *edge_;
    parent_type *graph_;
  };

  template <class GT, bool is_const>
  class EdgeIterator
    : public DataIteratorBase<EdgeIterator<GT, is_const>, GT,
                              EdgeWrapper<GT, is_const>, is_const> {
  public:
    using Base = DataIteratorBase<EdgeIterator<GT, is_const>, GT,
                                  EdgeWrapper<GT, is_const>, is_const>;

    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    using Base::Base;

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr EdgeIterator(const EdgeIterator<GT, other_const> &other) noexcept
      : Base(other) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr EdgeIterator &
    operator=(const EdgeIterator<GT, other_const> &other) noexcept {
      Base::operator=(other);
      return *this;
    }

  private:
    friend Base;
    template <class, bool other_const>
    friend class EdgeIterator;

    constexpr value_type deref(typename Base::parent_type *graph,
                               difference_type index) const noexcept {
      return graph->edge(index);
    }
  };

  template <class GT, bool is_const>
  class AdjWrapper {
  public:
    using DT = typename GT::adj_data_type;
    using NodeRef = NodeWrapper<GT, is_const>;

    using value_type = const_if_t<is_const, DT>;
    using parent_type = const_if_t<is_const, GT>;

    template <bool other_const>
    using Other = AdjWrapper<GT, other_const>;

    constexpr AdjWrapper(int src, DT &adj, parent_type &graph) noexcept
      : src_(src), adj_(&adj), graph_(&graph) { }

    constexpr AdjWrapper(int src, const DT &adj, parent_type &graph) noexcept
      : src_(src), adj_(&adj), graph_(&graph) { }

    template <bool other_const,
              typename = std::enable_if_t<is_const && !other_const>>
    constexpr AdjWrapper(const Other<other_const> &other) noexcept
      : src_(other.src_), adj_(other.adj_), graph_(other.graph_) { }

    constexpr int src() const noexcept { return src_; }
    constexpr int dst() const noexcept { return adj_->dst; }

    constexpr NodeRef src_node() const noexcept {
      return const_cast<parent_type *>(graph_)->node(src());
    }

    constexpr NodeRef dst_node() const noexcept {
      return const_cast<parent_type *>(graph_)->node(dst());
    }

    constexpr int eid() const noexcept { return adj_->eid; }
    constexpr auto &edge_data() const noexcept {
      return const_cast<parent_type *>(graph_)->edge(eid()).data();
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class AdjWrapper;

    int src_;
    value_type *adj_;
    parent_type *graph_;
  };

  template <class GT, bool is_const>
  class AdjIterator
    : public DataIteratorBase<AdjIterator<GT, is_const>, GT,
                              AdjWrapper<GT, is_const>, is_const> {
  public:
    using Base = DataIteratorBase<AdjIterator<GT, is_const>, GT,
                                  AdjWrapper<GT, is_const>, is_const>;

    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    using typename Base::parent_type;

    template <bool other_const>
    using Other = AdjIterator<GT, other_const>;

    constexpr AdjIterator(parent_type *graph, difference_type idx,
                          difference_type nid) noexcept
      : Base(graph, idx), nid_(nid) { }

    template <bool other_const,
              typename = std::enable_if_t<is_const && !other_const>>
    constexpr AdjIterator(const Other<other_const> &other) noexcept
      : Base(other), nid_(other.nid_) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr AdjIterator &
    operator=(const AdjIterator<GT, other_const> &other) noexcept {
      Base::operator=(other);
      nid_ = other.nid_;
      return *this;
    }

  private:
    friend Base;
    template <class, bool other_const>
    friend class AdjIterator;

    constexpr AdjIterator advance(difference_type n) const noexcept {
      return Base::advance(n, nid_);
    }

    constexpr value_type deref(parent_type *graph,
                               difference_type index) const noexcept {
      return graph->adjacent(nid_, index);
    }

    int nid_;
  };
}  // namespace internal

/**
 * @brief Class for \e very sparse graphs, especially designed for the molecular
 *        graphs.
 *
 * @tparam NT node data type.
 * @tparam ET edge data type.
 */
template <class NT, class ET>
class Graph {
public:
  using node_data_type = NT;
  using edge_data_type = ET;

  using iterator = internal::NodeIterator<Graph, false>;
  using node_iterator = iterator;
  using const_iterator = internal::NodeIterator<Graph, true>;
  using const_node_iterator = const_iterator;
  using NodeRef = typename iterator::value_type;
  using ConstNodeRef = typename const_iterator::value_type;

  using edge_iterator = internal::EdgeIterator<Graph, false>;
  using const_edge_iterator = internal::EdgeIterator<Graph, true>;
  using EdgeRef = typename edge_iterator::value_type;
  using ConstEdgeRef = typename const_edge_iterator::value_type;

  using adjacency_iterator = internal::AdjIterator<Graph, false>;
  using const_adjacency_iterator = internal::AdjIterator<Graph, true>;
  using AdjRef = typename adjacency_iterator::value_type;
  using ConstAdjRef = typename const_adjacency_iterator::value_type;

  Graph() = default;
  Graph(const Graph &) = default;
  Graph(Graph &&) noexcept = default;
  Graph &operator=(const Graph &) = default;
  Graph &operator=(Graph &&) noexcept = default;
  ~Graph() noexcept = default;

  Graph(int num_nodes): adj_list_(num_nodes), nodes_(num_nodes) { }
  Graph(int num_nodes, const NT &data)
    : adj_list_(num_nodes), nodes_(num_nodes, data) { }

  bool empty() const {
    // GCOV_EXCL_START
    ABSL_DCHECK(num_nodes() > 0 || num_edges() == 0)
      << "The graph is empty (num_nodes() == 0) but num_edges() == "
      << num_edges();
    // GCOV_EXCL_STOP
    return size() == 0;
  }
  int size() const { return num_nodes(); }
  int num_nodes() const { return nodes_.size(); }

  bool edge_empty() const {
    // GCOV_EXCL_START
    ABSL_DCHECK(num_nodes() > 0 || num_edges() == 0)
      << "The graph is empty (num_nodes() == 0) but num_edges() == "
      << num_edges();
    // GCOV_EXCL_STOP
    return num_edges() == 0;
  }
  int num_edges() const { return edges_.size(); }

  int degree(int id) const { return adj_list_[id].size(); }

  void reserve(int num_nodes) {
    nodes_.reserve(num_nodes);
    adj_list_.reserve(num_nodes);
  }

  int add_node(const NT &node) {
    int id = num_nodes();
    nodes_.push_back(node);
    adj_list_.push_back({});
    return id;
  }

  int add_node(NT &&node) noexcept {
    int id = num_nodes();
    nodes_.push_back(std::move(node));
    adj_list_.push_back({});
    return id;
  }

  int add_edge(int src, int dst, const ET &edge) {
    int id = num_edges();
    adj_list_[src].push_back({ dst, id });
    adj_list_[dst].push_back({ src, id });
    edges_.push_back({ src, dst, edge });
    return id;
  }

  int add_edge(int src, int dst, ET &&edge) noexcept {
    int id = num_edges();
    adj_list_[src].push_back({ dst, id });
    adj_list_[dst].push_back({ src, id });
    edges_.push_back({ src, dst, std::move(edge) });
    return id;
  }

  NodeRef operator[](int id) { return node(id); }
  ConstNodeRef operator[](int id) const { return node(id); }

  NodeRef node(int id) { return { id, nodes_[id], *this }; }
  ConstNodeRef node(int id) const { return { id, nodes_[id], *this }; }
  void update_node(int id, const NT &data) { nodes_[id] = data; }
  void update_node(int id, NT &&data) noexcept { nodes_[id] = std::move(data); }

  void clear() {
    nodes_.clear();
    edges_.clear();
    adj_list_.clear();
  }

  /**
   * @brief Remove a node and all its associated edge(s) from the graph.
   *
   * @param id The id of the node to be removed.
   * @return The data of the removed node.
   * @sa erase_nodes()
   * @note Time complexity: \f$O(V+E)\f$ if the node has any neighbor, and
   *       \f$O(E)\f$ if the node has no neighbors. If \p id \f$\ge\f$
   *       `num_nodes()` or \p id \f$\lt 0\f$, the behavior is undefined.
   */
  NT pop_node(int id) {
    NT ret = std::move(nodes_[id]);
    erase_nodes(begin() + id, begin() + id + 1);
    return ret;
  }

  /**
   * @brief Remove nodes and all its associated edge(s) from the graph.
   *
   * @param begin The beginning of the range of nodes to be removed.
   * @param end The end of the range of nodes to be removed.
   * @sa pop_node()
   * @note Time complexity: \f$O(V+E)\f$. If \p begin or \p end is out of range,
   *       the behavior is undefined.
   */
  void erase_nodes(const_iterator begin, const_iterator end) {
    erase_nodes(begin, end, [](auto /* ref */) { return true; });
  }

  /**
   * @brief Remove matching nodes and all its associated edge(s) from the graph.
   *
   * @tparam UnaryPred A unary predicate that takes a `ConstNodeRef` and returns
   *        `bool`.
   * @param begin The beginning of the range of nodes to be removed.
   * @param end The end of the range of nodes to be removed.
   * @param pred A unary predicate that takes a `ConstNodeRef` and returns
   *        `true` if the node should be removed.
   * @sa pop_node()
   * @note Time complexity: \f$O(V+E)\f$. If \p begin or \p end is out of range,
   *       the behavior is undefined.
   */
  template <class UnaryPred>
  void erase_nodes(const_iterator begin, const_iterator end, UnaryPred pred);

  iterator begin() { return { this, 0 }; }
  iterator end() { return { this, num_nodes() }; }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  const_iterator cbegin() const { return { this, 0 }; }
  const_iterator cend() const { return { this, num_nodes() }; }

  EdgeRef edge(int id) { return { id, edges_[id], *this }; }
  ConstEdgeRef edge(int id) const { return { id, edges_[id], *this }; }
  void update_edge(int id, const ET &data) { edges_[id].data = data; }
  void update_edge(int id, ET &&data) noexcept {
    edges_[id].data = std::move(data);
  }

  edge_iterator find_edge(int src, int dst) {
    return find_edge_helper(*this, src, dst);
  }

  const_edge_iterator find_edge(int src, int dst) const {
    return find_edge_helper(*this, src, dst);
  }

  void clear_edge() {
    edges_.clear();
    for (std::vector<AdjEntry> &adj: adj_list_) {
      adj.clear();
    }
  }

  /**
   * @brief Remove an edge from the graph.
   *
   * @param id The id of the edge to be removed.
   * @return The data of the removed edge.
   * @sa erase_edges()
   * @note Time complexity: \f$O(V+E)\f$. If \p id \f$\ge\f$ `num_edges()` or
   *       \p id \f$\lt 0\f$, the behavior is undefined.
   */
  ET pop_edge(int id) {
    ET ret = std::move(edges_[id].data);
    erase_edges(edge_begin() + id, edge_begin() + id + 1);
    return ret;
  }

  /**
   * @brief Remove edges from the graph.
   *
   * @param begin The beginning of the range of edges to be removed.
   * @param end The end of the range of edges to be removed.
   * @sa pop_edge()
   * @note Time complexity: \f$O(V+E)\f$. If \p begin or \p end is out of range,
   *       the behavior is undefined.
   */
  void erase_edges(const_edge_iterator begin, const_edge_iterator end) {
    erase_edges(begin, end, [](auto /* ref */) { return true; });
  }

  /**
   * @brief Remove matching edges from the graph.
   *
   * @tparam UnaryPred A unary predicate that takes a `ConstEdgeRef` and returns
   *         `bool`.
   * @param begin The beginning of the range of edges to be removed.
   * @param end The end of the range of edges to be removed.
   * @param pred A unary predicate that takes a `ConstEdgeRef` and returns
   *        `true` if the edge should be removed.
   * @sa pop_edge()
   * @note Time complexity: \f$O(V+E)\f$. If \p begin or \p end is out of range,
   *       the behavior is undefined.
   */
  template <class UnaryPred>
  void erase_edges(const_edge_iterator begin, const_edge_iterator end,
                   UnaryPred pred);

  edge_iterator edge_begin() { return { this, 0 }; }
  edge_iterator edge_end() { return { this, num_edges() }; }
  const_edge_iterator edge_begin() const { return edge_cbegin(); }
  const_edge_iterator edge_end() const { return edge_cend(); }
  const_edge_iterator edge_cbegin() const { return { this, 0 }; }
  const_edge_iterator edge_cend() const { return { this, num_edges() }; }

  adjacency_iterator find_adjacent(int src, int dst) {
    return find_adj_helper(*this, src, dst);
  }

  const_adjacency_iterator find_adjacent(int src, int dst) const {
    return find_adj_helper(*this, src, dst);
  }

  adjacency_iterator adj_begin(int nid) { return { this, 0, nid }; }
  adjacency_iterator adj_end(int nid) { return { this, degree(nid), nid }; }
  const_adjacency_iterator adj_begin(int nid) const { return adj_cbegin(nid); }
  const_adjacency_iterator adj_end(int nid) const { return adj_cend(nid); }
  const_adjacency_iterator adj_cbegin(int nid) const {
    return { this, 0, nid };
  }
  const_adjacency_iterator adj_cend(int nid) const {
    return { this, degree(nid), nid };
  }

private:
  struct StoredEdge {
    int src;
    int dst;
    ET data;
  };

  struct AdjEntry {
    int dst;
    int eid;
  };

  using edge_type = StoredEdge;
  using adj_data_type = AdjEntry;

  friend EdgeRef;
  friend ConstEdgeRef;
  friend adjacency_iterator;
  friend const_adjacency_iterator;
  friend AdjRef;
  friend ConstAdjRef;

  template <class GT>
  static internal::AdjIterator<Graph, std::is_const_v<GT>>
  find_adj_helper(GT &graph, int src, int dst) {
    auto ret = graph.adj_begin(src);
    for (; ret != graph.adj_end(src); ++ret) {
      if (ret->dst() == dst) {
        break;
      }
    }
    return ret;
  }

  template <class GT>
  static internal::EdgeIterator<Graph, std::is_const_v<GT>>
  find_edge_helper(GT &graph, int src, int dst) {
    if (graph.degree(src) > graph.degree(dst)) {
      return find_edge_helper(graph, dst, src);
    }

    auto ait = find_adj_helper(graph, src, dst);
    return ait == graph.adj_end(src) ? graph.edge_end()
                                     : graph.edge_begin() + ait->eid();
  }

  AdjRef adjacent(int nid, int idx) {
    return { nid, adj_list_[nid][idx], *this };
  }

  ConstAdjRef adjacent(int nid, int idx) const {
    return { nid, adj_list_[nid][idx], *this };
  }

  std::vector<std::vector<AdjEntry>> adj_list_;
  std::vector<NT> nodes_;
  std::vector<StoredEdge> edges_;
};

namespace internal {
  template <class Iterator, class UnaryPred>
  void item_mask(Iterator begin, Iterator end, UnaryPred pred,
                 std::vector<int> &mask, bool &remove, bool &remove_trailing) {
    for (auto it = begin; it != end; ++it) {
      if (pred(*it)) {
        remove = true;
        mask[it->id()] = 0;
      } else if (remove) {
        remove_trailing = false;
      }
    }
  }
}  // namespace internal

/* Out-of-line definitions */

template <class NT, class ET>
template <class UnaryPred>
void Graph<NT, ET>::erase_nodes(const const_iterator begin,
                                const const_iterator end, UnaryPred pred) {
  // Note: the time complexity notations are only for very sparse graphs, i.e.,
  // E = O(V).

  // This will also handle size() == 0 case correctly.
  if (begin >= end) {
    return;
  }

  // Phase I: mark nodes & edges for removal, O(V+E)
  std::vector<int> node_keep(num_nodes(), 1);
  bool remove = false, remove_trailing = end == this->end();
  internal::item_mask(begin, end, pred, node_keep, remove, remove_trailing);
  // Fast path 1: if no nodes are removed, return early.
  if (!remove) {
    return;
  }

  // Phase II: re-number the nodes, O(V)
  // The map contains new node id.
  std::vector<int> node_map = mask_to_map(node_keep);
  // Fast path 2: if all nodes are removed, clear and return.
  if (node_map.back() == -1) {
    ABSL_DLOG(INFO) << "clearing graph";
    clear();
    return;
  }

  // Phase III: remove unused adjacencies, O(1) or O(V)
  if (remove_trailing) {
    ABSL_DLOG(INFO) << "resizing adjacency & node list";
    nodes_.resize(node_map.back() + 1);
    adj_list_.resize(nodes_.size());
  } else {
    int i = 0;
    auto new_nodes_end =
      std::remove_if(nodes_.begin(), nodes_.end(),
                     [&](const NT &) { return node_keep[i++] == 0; });
    nodes_.erase(new_nodes_end, nodes_.end());

    i = 0;
    auto new_adj_end = std::remove_if(adj_list_.begin(), adj_list_.end(),
                                      [&](const std::vector<AdjEntry> &) {
                                        return node_keep[i++] == 0;
                                      });
    adj_list_.erase(new_adj_end, adj_list_.end());
  }

  // Phase IV: remove corresponding edges, O(V+E)
  erase_edges(edge_begin(), edge_end(), [&](const ConstEdgeRef &e) {
    return (node_keep[e.src()] & node_keep[e.dst()]) == 0;
  });
  // Fast path 3: if all edges are removed, return.
  if (edge_empty()) {
    return;
  }

  // Phase V: update node index of edges, O(E)
  for (StoredEdge &edge: edges_) {
    edge.src = node_map[edge.src];
    edge.dst = node_map[edge.dst];
  }

  // Phase VI: update adjacencies, ~O(V)
  for (std::vector<AdjEntry> &adjs: adj_list_) {
    for (AdjEntry &adj: adjs) {
      adj.dst = node_map[adj.dst];
    }
  }

  // GCOV_EXCL_START
  ABSL_DCHECK(num_nodes() == adj_list_.size())
    << "node count mismatch: " << num_nodes() << " vs " << adj_list_.size();
  // GCOV_EXCL_STOP
}

template <class NT, class ET>
template <class UnaryPred>
void Graph<NT, ET>::erase_edges(const const_edge_iterator begin,
                                const const_edge_iterator end, UnaryPred pred) {
  // This will also handle size() == 0 case correctly.
  if (begin >= end) {
    return;
  }

  // Phase I: Mark edges for removal, O(E)
  std::vector<int> edge_keep(num_edges(), 1);
  bool remove = false, remove_trailing = end == this->edge_end();
  internal::item_mask(begin, end, pred, edge_keep, remove, remove_trailing);
  // Fast path 1: if no edges are removed, return early.
  if (!remove) {
    return;
  }

  std::vector<int> edge_map = mask_to_map(edge_keep);
  // Fast path 2: if all edges are removed, clear and return.
  if (edge_map.back() == -1) {
    ABSL_DLOG(INFO) << "clearing edges";
    clear_edge();
  }

  // Phase II: update adjacencies, ~O(V)
  for (std::vector<AdjEntry> &old_adj: adj_list_) {
    std::vector<AdjEntry> new_adj;
    for (auto [dst, eid]: old_adj) {
      if (edge_keep[eid] != 0) {
        new_adj.push_back({ dst, edge_map[eid] });
      }
    }
    std::swap(new_adj, old_adj);
  }

  // Phase III: remove unused edges, O(E)
  if (remove_trailing) {
    // Fast path 2: if only trailing edges are removed, resize.
    ABSL_DLOG(INFO) << "resizing edge list";
    edges_.resize(edge_map.back() + 1);
  } else {
    int i = 0;
    auto new_end = std::remove_if(edges_.begin(), edges_.end(),
                                  [&](auto &) { return edge_keep[i++] == 0; });
    edges_.erase(new_end, edges_.end());
  }
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_H_ */

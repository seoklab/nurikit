//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_H_
#define NURI_CORE_GRAPH_H_

#include <algorithm>
#include <iterator>
#include <queue>
#include <stack>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>
#include <Eigen/Dense>

#include <absl/base/optimization.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/utils.h"

namespace nuri {
namespace internal {
  template <class Derived, class GT, class DT, bool is_const>
  class DataIteratorBase
      : public ProxyIterator<Derived, DT, std::random_access_iterator_tag, int> {
    using Traits =
        std::iterator_traits<typename DataIteratorBase::iterator_facade_>;

  public:
    using parent_type = const_if_t<is_const, GT>;

    using iterator_category = typename Traits::iterator_category;
    using value_type = typename Traits::value_type;
    using difference_type = typename Traits::difference_type;
    using pointer = typename Traits::pointer;
    using reference = typename Traits::reference;

    constexpr DataIteratorBase() noexcept = default;

    constexpr DataIteratorBase(parent_type *graph,
                               difference_type index) noexcept
        : graph_(graph), index_(index) { }

  protected:
    using Parent = DataIteratorBase;

    template <class Other,
              std::enable_if_t<!std::is_same_v<Derived, Other>, int> = 0>
    constexpr DataIteratorBase(const Other &other) noexcept
        : graph_(other.graph_), index_(other.index_) { }

    template <class Other,
              std::enable_if_t<!std::is_same_v<Derived, Other>, int> = 0>
    constexpr DataIteratorBase &operator=(const Other &other) noexcept {
      graph_ = other.graph_;
      index_ = other.index_;
      return *this;
    }

    constexpr parent_type *graph() const noexcept { return graph_; }

    constexpr difference_type index() const noexcept { return index_; }

  private:
    template <class, class, class, bool>
    friend class DataIteratorBase;

    friend class boost::iterator_core_access;

    template <class Other,
              std::enable_if_t<std::is_convertible_v<Other, Derived>, int> = 0>
    constexpr bool equal(const Other &other) const noexcept {
      return index_ == other.index_;
    }

    template <class Other,
              std::enable_if_t<std::is_convertible_v<Other, Derived>, int> = 0>
    constexpr difference_type distance_to(const Other &other) const noexcept {
      return other.index_ - index_;
    }

    void increment() noexcept { ++index_; }
    void decrement() noexcept { --index_; }
    void advance(difference_type n) noexcept { index_ += n; }

    parent_type *graph_;
    difference_type index_;
  };

  template <class GT, bool is_const>
  class NodeWrapper;

  template <class GT, bool is_const>
  class AdjWrapper {
  public:
    using edge_value_type = const_if_t<is_const, typename GT::edge_data_type>;
    using parent_type = const_if_t<is_const, GT>;

    template <bool other_const>
    using Other = AdjWrapper<GT, other_const>;

    constexpr AdjWrapper(parent_type &graph, int src, int dst, int eid) noexcept
        : src_(src), dst_(dst), eid_(eid), graph_(&graph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr AdjWrapper(const Other<other_const> &other) noexcept
        : src_(other.src_), dst_(other.dst_), eid_(other.eid_),
          graph_(other.graph_) { }

    constexpr auto src() const noexcept { return graph_->node(src_); }
    constexpr auto dst() const noexcept { return graph_->node(dst_); }

    constexpr int eid() const noexcept { return eid_; }
    constexpr edge_value_type &edge_data() const noexcept {
      return graph_->edge(eid_).data();
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class AdjWrapper;

    template <class, bool>
    friend class SubAdjIterator;
    template <class, bool>
    friend class SubEdgeWrapper;
    template <class, bool>
    friend class SubEdgesFinder;

    friend GT;

    int src_;
    int dst_;
    int eid_;
    parent_type *graph_;
  };

  template <class GT, bool is_const>
  class AdjIterator
      : public DataIteratorBase<AdjIterator<GT, is_const>, GT,
                                AdjWrapper<GT, is_const>, is_const> {
    using Base = typename AdjIterator::Parent;

  public:
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
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr AdjIterator(const Other<other_const> &other) noexcept
        : Base(other), nid_(other.nid_) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr AdjIterator &operator=(const Other<other_const> &other) noexcept {
      Base::operator=(other);
      nid_ = other.nid_;
      return *this;
    }

    constexpr bool begin() const noexcept {
      return *this == this->graph()->adj_begin(nid_);
    }

    constexpr bool end() const noexcept {
      return *this == this->graph()->adj_end(nid_);
    }

  private:
    friend Base;

    friend class boost::iterator_core_access;

    template <class, bool>
    friend class AdjIterator;

    constexpr reference dereference() const noexcept {
      return this->graph()->adjacent(nid_, this->index());
    }

    int nid_;
  };

  template <class GT, bool is_const>
  class NodeWrapper {
  public:
    using DT = typename GT::node_data_type;

    using parent_type = const_if_t<is_const, GT>;
    using value_type = const_if_t<is_const, DT>;

    using adjacency_iterator =
        std::conditional_t<is_const, typename GT::const_adjacency_iterator,
                           typename GT::adjacency_iterator>;

    template <bool other_const>
    using Other = NodeWrapper<GT, other_const>;

    constexpr NodeWrapper(int nid, DT &data, parent_type &graph) noexcept
        : nid_(nid), data_(&data), graph_(&graph) { }

    template <bool this_const = is_const,
              std::enable_if_t<is_const && this_const, int> = 0>
    constexpr NodeWrapper(int nid, const DT &data, parent_type &graph) noexcept
        : nid_(nid), data_(&data), graph_(&graph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr NodeWrapper(const Other<other_const> &other) noexcept
        : nid_(other.nid_), data_(other.data_), graph_(other.graph_) { }

    constexpr int id() const noexcept { return nid_; }

    constexpr value_type &data() const noexcept { return *data_; }

    constexpr int degree() const noexcept { return graph_->degree(nid_); }

    adjacency_iterator begin() const noexcept {
      return graph_->adj_begin(nid_);
    }

    adjacency_iterator end() const noexcept { return graph_->adj_end(nid_); }

    AdjWrapper<GT, is_const> neighbor(int idx) const noexcept {
      return graph_->adjacent(nid_, idx);
    }

    AdjWrapper<GT, is_const> operator[](int idx) const noexcept {
      return neighbor(idx);
    }

    adjacency_iterator find_adjacent(int aid) const noexcept {
      return graph_->find_adjacent(nid_, aid);
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
    using Base = typename NodeIterator::Parent;

  public:
    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    using Base::Base;

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr NodeIterator(const NodeIterator<GT, other_const> &other) noexcept
        : Base(other) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr NodeIterator &
    operator=(const NodeIterator<GT, other_const> &other) noexcept {
      Base::operator=(other);
      return *this;
    }

  private:
    friend Base;

    friend class boost::iterator_core_access;

    template <class, bool other_const>
    friend class NodeIterator;

    constexpr reference dereference() const noexcept {
      return this->graph()->node(this->index());
    }
  };

  template <class GT, bool is_const>
  class EdgeWrapper {
  public:
    using DT = typename GT::edge_data_type;

    using parent_type = const_if_t<is_const, GT>;
    using value_type = const_if_t<is_const, DT>;

    template <bool other_const>
    using Other = EdgeWrapper<GT, other_const>;

    constexpr EdgeWrapper(int eid, value_type &data,
                          parent_type &graph) noexcept
        : eid_(eid), data_(&data), graph_(&graph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr EdgeWrapper(const Other<other_const> &other) noexcept
        : eid_(other.eid_), data_(other.data_), graph_(other.graph_) { }

    constexpr int id() const noexcept { return eid_; }

    constexpr NodeWrapper<GT, is_const> src() const noexcept {
      return graph_->node(graph_->edges_[eid_].src);
    }

    constexpr NodeWrapper<GT, is_const> dst() const noexcept {
      return graph_->node(graph_->edges_[eid_].dst);
    }

    constexpr value_type &data() const noexcept { return *data_; }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class EdgeWrapper;

    int eid_;
    value_type *data_;
    parent_type *graph_;
  };

  template <class GT, bool is_const>
  class EdgeIterator
      : public DataIteratorBase<EdgeIterator<GT, is_const>, GT,
                                EdgeWrapper<GT, is_const>, is_const> {
    using Base = typename EdgeIterator::Parent;

  public:
    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    using Base::Base;

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr EdgeIterator(const EdgeIterator<GT, other_const> &other) noexcept
        : Base(other) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr EdgeIterator &
    operator=(const EdgeIterator<GT, other_const> &other) noexcept {
      Base::operator=(other);
      return *this;
    }

  private:
    friend Base;

    friend class boost::iterator_core_access;

    template <class, bool other_const>
    friend class EdgeIterator;

    constexpr reference dereference() const noexcept {
      return this->graph()->edge(this->index());
    }
  };

  template <class GT, bool is_const>
  class EdgesWrapper {
  public:
    template <class GU = GT,
              std::enable_if_t<std::is_same_v<GU, GT> && !is_const, int> = 0>
    EdgesWrapper(GT &graph): graph_(&graph) { }

    EdgesWrapper(const GT &graph): graph_(&graph) { }

    template <bool other_const>
    using Other = EdgesWrapper<GT, other_const>;

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    EdgesWrapper(const Other<other_const> &other) noexcept
        : graph_(other.graph_) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    EdgesWrapper &operator=(const Other<other_const> &other) noexcept {
      graph_ = other.graph_;
      return *this;
    }

    EdgeWrapper<GT, is_const> operator[](int id) const {
      return graph_->edge(id);
    }

    EdgeIterator<GT, is_const> begin() const { return graph_->edge_begin(); }
    EdgeIterator<GT, is_const> end() const { return graph_->edge_end(); }

    EdgeIterator<GT, true> cbegin() const { return graph_->edge_cbegin(); }
    EdgeIterator<GT, true> cend() const { return graph_->edge_cend(); }

    EdgesWrapper<GT, true> as_const() const { return *this; }

    int size() const { return graph_->num_edges(); }

  private:
    const_if_t<is_const, GT> *graph_;
  };

  template <class, bool>
  class SubEdgeWrapper;
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
private:
  struct StoredEdge {
    int src;
    int dst;
    ET data;
  };
  using edge_type = StoredEdge;

  struct AdjEntry {
    int dst;
    int eid;
  };

  template <class, bool>
  friend class internal::NodeWrapper;

  template <class, bool>
  friend class internal::EdgeWrapper;
  template <class, bool>
  friend class internal::EdgeIterator;

  template <class, bool>
  friend class internal::AdjWrapper;
  template <class, bool>
  friend class internal::AdjIterator;

  template <class, class, bool>
  friend class Subgraph;
  template <class, bool>
  friend class internal::SubEdgeWrapper;

public:
  using node_data_type = NT;
  using edge_data_type = ET;

  using iterator = internal::NodeIterator<Graph, false>;
  using node_iterator = iterator;
  using const_iterator = internal::NodeIterator<Graph, true>;
  using const_node_iterator = const_iterator;
  using NodeRef = internal::NodeWrapper<Graph, false>;
  using ConstNodeRef = internal::NodeWrapper<Graph, true>;

  using edge_iterator = internal::EdgeIterator<Graph, false>;
  using const_edge_iterator = internal::EdgeIterator<Graph, true>;
  using EdgeRef = internal::EdgeWrapper<Graph, false>;
  using ConstEdgeRef = internal::EdgeWrapper<Graph, true>;

  using adjacency_iterator = internal::AdjIterator<Graph, false>;
  using const_adjacency_iterator = internal::AdjIterator<Graph, true>;
  using AdjRef = internal::AdjWrapper<Graph, false>;
  using ConstAdjRef = internal::AdjWrapper<Graph, true>;

  static_assert(std::is_same_v<typename iterator::reference, NodeRef>);
  static_assert(
      std::is_same_v<typename const_iterator::reference, ConstNodeRef>);
  static_assert(std::is_same_v<typename edge_iterator::reference, EdgeRef>);
  static_assert(
      std::is_same_v<typename const_edge_iterator::reference, ConstEdgeRef>);
  static_assert(std::is_same_v<typename adjacency_iterator::reference, AdjRef>);
  static_assert(
      std::is_same_v<typename const_adjacency_iterator::reference, ConstAdjRef>);

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
  int num_nodes() const { return static_cast<int>(nodes_.size()); }

  bool edge_empty() const {
    // GCOV_EXCL_START
    ABSL_DCHECK(num_nodes() > 0 || num_edges() == 0)
        << "The graph is empty (num_nodes() == 0) but num_edges() == "
        << num_edges();
    // GCOV_EXCL_STOP
    return num_edges() == 0;
  }
  int num_edges() const { return static_cast<int>(edges_.size()); }

  int degree(int id) const { return static_cast<int>(adj_list_[id].size()); }

  void reserve(int num_nodes) {
    nodes_.reserve(num_nodes);
    adj_list_.reserve(num_nodes);
  }

  void reserve_edges(int num_edges) { edges_.reserve(num_edges); }

  int add_node(const NT &data) {
    int id = num_nodes();
    nodes_.push_back(data);
    adj_list_.push_back({});
    return id;
  }

  int add_node(NT &&data) noexcept {
    int id = num_nodes();
    nodes_.push_back(std::move(data));
    adj_list_.push_back({});
    return id;
  }

  template <class Iterator,
            internal::enable_if_compatible_iter_t<Iterator, NT> = 0>
  int add_node(Iterator begin, Iterator end) noexcept {
    auto it = nodes_.insert(nodes_.end(), begin, end);
    adj_list_.resize(num_nodes());
    return static_cast<int>(it - nodes_.begin());
  }

  edge_iterator add_edge(int src, int dst, const ET &data) {
    int eid = num_edges();
    edges_.push_back({ src, dst, data });
    add_adjacency_entry(src, dst, eid);
    return { this, eid };
  }

  edge_iterator add_edge(int src, int dst, ET &&data) noexcept {
    int eid = num_edges();
    edges_.push_back({ src, dst, std::move(data) });
    add_adjacency_entry(src, dst, eid);
    return { this, eid };
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
   * @brief Erase a node and all its associated edge(s) from the graph.
   *
   * @param id The id of the node to be erased.
   * @return The data of the erased node.
   * @sa erase_nodes()
   * @note Time complexity: \f$O(V)\f$ if only trailing node is erased,
   *       \f$O(V+E)\f$ otherwise. If \p id \f$\ge\f$ `num_nodes()` or \p id
   *       \f$\lt 0\f$, the behavior is undefined.
   */
  NT pop_node(int id) {
    NT ret = std::move(nodes_[id]);
    erase_nodes(begin() + id, begin() + id + 1);
    return ret;
  }

  /**
   * @brief Erase nodes and all its associated edge(s) from the graph.
   *
   * @param begin The beginning of the range of nodes to be erased.
   * @param end The end of the range of nodes to be erased.
   * @return A pair of (`new end id`, mapping of `old node id -> new node id`).
   *         If only trailing nodes are erased, `new end id` will be set to the
   *         first erased node id, and the mapping will be in a valid but
   *         unspecified state. If no nodes are erased (special case of
   *         trailing node removal), `new end id` will be equal to the size of
   *         the graph before this operation. Otherwise, `new end id` will be
   *         set to -1 and erased nodes will be marked as -1 in the mapping.
   * @sa pop_node()
   * @note Time complexity: \f$O(V)\f$ if only trailing nodes and edges are
   *       erased, \f$O(V+E)\f$ otherwise. If \p begin or \p end is out of
   *       range, the behavior is undefined.
   */
  std::pair<int, std::vector<int>> erase_nodes(const_iterator begin,
                                               const_iterator end) {
    return erase_nodes(begin, end, [](auto /* ref */) { return true; });
  }

  /**
   * @brief Erase matching nodes and all its associated edge(s) from the graph.
   *
   * @tparam UnaryPred A unary predicate that takes a `ConstNodeRef` and returns
   *        `bool`.
   * @param begin The beginning of the range of nodes to be erased.
   * @param end The end of the range of nodes to be erased.
   * @param pred A unary predicate that takes a `ConstNodeRef` and returns
   *        `true` if the node should be erased.
   * @return A pair of (`new end id`, mapping of `old node id -> new node id`).
   *         If only trailing nodes are erased, `new end id` will be set to the
   *         first erased node id, and the mapping will be in a valid but
   *         unspecified state. If no nodes are erased (special case of
   *         trailing node removal), `new end id` will be equal to the size of
   *         the graph before this operation. Otherwise, `new end id` will be
   *         set to -1 and erased nodes will be marked as -1 in the mapping.
   * @sa pop_node()
   * @note Time complexity: \f$O(V)\f$ if only trailing nodes are erased,
   *       \f$O(V+E)\f$ otherwise. If \p begin or \p end is out of range, the
   *       behavior is undefined.
   */
  template <class UnaryPred>
  std::pair<int, std::vector<int>>
  erase_nodes(const_iterator begin, const_iterator end, UnaryPred pred);

  /**
   * @brief Erase nodes and all its associated edge(s) from the graph.
   *
   * @tparam Iterator An iterator type that dereferences to a value compatible
   *         with `int`.
   * @param begin The beginning of the range of node ids to be erased.
   * @param end The end of the range of node ids to be erased.
   * @return A pair of (`new end id`, mapping of `old node id -> new node id`).
   *         If only trailing nodes are erased, `new end id` will be set to the
   *         first erased node id, and the mapping will be in a valid but
   *         unspecified state. If no nodes are erased (special case of
   *         trailing node removal), `new end id` will be equal to the size of
   *         the graph before this operation. Otherwise, `new end id` will be
   *         set to -1 and erased nodes will be marked as -1 in the mapping.
   * @sa pop_node()
   * @note Time complexity: \f$O(V)\f$ if only trailing nodes are erased,
   *       \f$O(V+E)\f$ otherwise. If any iterator in range `[`\p begin,
   *       \p end`)` references an invalid node id, the behavior is undefined.
   */
  template <class Iterator,
            class = internal::enable_if_compatible_iter_t<Iterator, int>>
  std::pair<int, std::vector<int>> erase_nodes(Iterator begin, Iterator end);

  iterator begin() { return { this, 0 }; }
  iterator end() { return { this, num_nodes() }; }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  const_iterator cbegin() const { return { this, 0 }; }
  const_iterator cend() const { return { this, num_nodes() }; }

  EdgeRef edge(int id) { return { id, edges_[id].data, *this }; }
  ConstEdgeRef edge(int id) const { return { id, edges_[id].data, *this }; }
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
    for (std::vector<AdjEntry> &adj: adj_list_)
      adj.clear();
  }

  /**
   * @brief Erase an edge from the graph.
   *
   * @param id The id of the edge to be erased.
   * @return The data of the erased edge.
   * @sa erase_edges()
   * @note Time complexity: \f$O(E/V)\f$. If \p id is out of range, the behavior
   *       is undefined.
   */
  ET pop_edge(int id) {
    ET ret = std::move(edges_[id].data);
    erase_edges(edge_begin() + id, edge_begin() + id + 1);
    return ret;
  }

  /**
   * @brief Erase an edge from the graph between two nodes.
   *
   * @param src The id of the source node.
   * @param dst The id of the destination node.
   * @return Whether the edge is erased.
   * @note Time complexity: \f$O(E/V)\f$ if the edge is the last edge of the
   *       graph or is not found, \f$O(V+E)\f$ otherwise. If \p src or \p dst is
   *       out of range, the behavior is undefined. \p src and \p dst are
   *       interchangeable.
   */
  bool erase_edge_between(int src, int dst);

  /**
   * @brief Erase edges from the graph.
   *
   * @param begin The beginning of the range of nodes to be erased.
   * @param end The end of the range of nodes to be erased.
   * @return A pair of (`new end id`, mapping of `old edge id -> new edge id`).
   *         If only trailing edges are erased, `new end id` will be set to the
   *         first erased edge id, and the mapping will be in a valid but
   *         unspecified state. If no edges are erased (special case of trailing
   *         edge removal), `new end id` will be equal to the size of the graph
   *         before this operation. Otherwise, `new end id` will be set to -1
   *         and erased edges will be marked as -1 in the mapping.
   * @sa pop_edge()
   * @note Time complexity: \f$O(E)\f$. If \p begin or \p end is out of range,
   *       the behavior is undefined.
   */
  std::pair<int, std::vector<int>> erase_edges(const_edge_iterator begin,
                                               const_edge_iterator end) {
    return erase_edges(begin, end, [](auto /* ref */) { return true; });
  }

  /**
   * @brief Erase matching edges from the graph.
   *
   * @tparam UnaryPred A unary predicate that takes a `ConstEdgeRef` and returns
   *        `bool`.
   * @param begin The beginning of the range of edges to be erased.
   * @param end The end of the range of edges to be erased.
   * @param pred A unary predicate that takes a `ConstEdgeRef` and returns
   *        `true` if the edge should be erased.
   * @return A pair of (`new end id`, mapping of `old edge id -> new edge id`).
   *         If only trailing edges are erased, `new end id` will be set to the
   *         first erased edge id, and the mapping will be in a valid but
   *         unspecified state. If no edges are erased (special case of
   *         trailing edge removal), `new end id` will be equal to the size of
   *         the graph before this operation. Otherwise, `new end id` will be
   *         set to -1 and erased edges will be marked as -1 in the mapping.
   * @sa pop_edge()
   * @note Time complexity: \f$O(E)\f$. If \p begin or \p end is out of range,
   *       the behavior is undefined.
   */
  template <class UnaryPred>
  std::pair<int, std::vector<int>> erase_edges(const_edge_iterator begin,
                                               const_edge_iterator end,
                                               UnaryPred pred);

  /**
   * @brief Erase edges from the graph.
   *
   * @tparam Iterator An iterator type that dereferences to a value compatible
   *         with `int`.
   * @param begin The beginning of the range of edge ids to be erased.
   * @param end The end of the range of edge ids to be erased.
   * @return A pair of (`new end id`, mapping of `old edge id -> new edge id`).
   *         If only trailing edges are erased, `new end id` will be set to the
   *         first erased edge id, and the mapping will be in a valid but
   *         unspecified state. If no edges are erased (special case of
   *         trailing edge removal), `new end id` will be equal to the size of
   *         the graph before this operation. Otherwise, `new end id` will be
   *         set to -1 and erased edges will be marked as -1 in the mapping.
   * @sa pop_edge()
   * @note Time complexity: \f$O(E)\f$. If any iterator in range `[`\p begin,
   *       \p end`)` references an invalid edge id, the behavior is undefined.
   */
  template <class Iterator,
            class = internal::enable_if_compatible_iter_t<Iterator, int>>
  std::pair<int, std::vector<int>> erase_edges(Iterator begin, Iterator end);

  internal::EdgesWrapper<Graph, false> edges() { return { *this }; }
  internal::EdgesWrapper<Graph, true> edges() const { return { *this }; }
  internal::EdgesWrapper<Graph, true> cedges() const { return { *this }; }

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

  template <class GraphLike>
  void merge(const GraphLike &other) {
    const int offset = size();
    reserve(offset + other.size());

    for (auto node: other)
      add_node(node.data());

    for (auto edge: other.edges())
      add_edge(edge.src().id() + offset, edge.dst().id() + offset, edge.data());
  }

private:
  template <class GT>
  static internal::AdjIterator<Graph, std::is_const_v<GT>>
  find_adj_helper(GT &graph, int src, int dst) {
    auto ret = graph.adj_begin(src);

    for (; ret != graph.adj_end(src); ++ret)
      if (ret->dst().id() == dst)
        break;

    return ret;
  }

  template <class GT>
  static internal::EdgeIterator<Graph, std::is_const_v<GT>>
  find_edge_helper(GT &graph, int src, int dst) {
    if (graph.degree(src) > graph.degree(dst))
      return find_edge_helper(graph, dst, src);

    auto ait = find_adj_helper(graph, src, dst);
    if (ait.end())
      return graph.edge_end();

    return { &graph, ait->eid_ };
  }

  void erase_nodes_common(std::vector<int> &node_keep, int first_erased_id,
                          bool erase_trailing);

  void add_adjacency_entry(int src, int dst, int eid) {
    adj_list_[src].push_back({ dst, eid });
    adj_list_[dst].push_back({ src, eid });
  }

  int erase_adjacent(const int src, const int dst) {
    int ret = num_edges();

    auto pred_gen = [](int other) {
      return [other](const AdjEntry &adj) { return adj.dst == other; };
    };

    std::vector<AdjEntry> &src_adjs = adj_list_[src];
    auto it = std::find_if(src_adjs.begin(), src_adjs.end(), pred_gen(dst));
    if (it != src_adjs.end()) {
      ret = it->eid;
      src_adjs.erase(it);

      std::vector<AdjEntry> &dst_adjs = adj_list_[dst];
      dst_adjs.erase(
          std::find_if(dst_adjs.begin(), dst_adjs.end(), pred_gen(src)));
    }

    return ret;
  }

  void erase_edges_common(std::vector<int> &edge_keep, int first_erased_id,
                          bool erase_trailing);

  AdjRef adjacent(int nid, int idx) {
    AdjEntry &adj = adj_list_[nid][idx];
    return { *this, nid, adj.dst, adj.eid };
  }

  ConstAdjRef adjacent(int nid, int idx) const {
    const AdjEntry &adj = adj_list_[nid][idx];
    return { *this, nid, adj.dst, adj.eid };
  }

  std::vector<std::vector<AdjEntry>> adj_list_;
  std::vector<NT> nodes_;
  std::vector<StoredEdge> edges_;
};

/* Out-of-line definitions */

template <class NT, class ET>
template <class Iterator, class>
std::pair<int, std::vector<int>> Graph<NT, ET>::erase_nodes(Iterator begin,
                                                            Iterator end) {
  // Note: the time complexity notations are only for very sparse graphs, i.e.,
  // E = O(V).
  if (begin == end)
    return { size(), {} };

  // Phase I: mark nodes for removal, O(V)
  std::vector<int> node_keep(num_nodes(), 1);
  int first_erased_id = num_nodes();
  for (auto it = begin; it != end; ++it) {
    int nid = *it;
    node_keep[nid] = 0;
    first_erased_id = std::min(first_erased_id, nid);
  }

  bool erase_trailing = true;
  for (int i = num_nodes() - 1; i >= first_erased_id; --i) {
    if (node_keep[i] == 1) {
      erase_trailing = false;
      break;
    }
  }

  erase_nodes_common(node_keep, first_erased_id, erase_trailing);
  return { erase_trailing ? first_erased_id : -1, std::move(node_keep) };
}

template <class NT, class ET>
template <class UnaryPred>
std::pair<int, std::vector<int>>
Graph<NT, ET>::erase_nodes(const const_iterator begin, const const_iterator end,
                           UnaryPred pred) {
  // Note: the time complexity notations are only for very sparse graphs, i.e.,
  // E = O(V).

  // This will also handle size() == 0 case correctly.
  if (begin >= end)
    return { size(), {} };

  // Phase I: mark nodes for removal, O(V)
  std::vector<int> node_keep(num_nodes(), 1);
  int first_erased_id = -1;
  bool erase_trailing = end == this->end();
  for (auto it = begin; it != end; ++it) {
    if (pred(*it)) {
      int nid = it->id();

      // GCOV_EXCL_START
      ABSL_ASSUME(nid >= 0);
      // GCOV_EXCL_STOP

      if (first_erased_id < 0)
        first_erased_id = nid;

      node_keep[nid] = 0;
    } else if (first_erased_id >= 0) {
      erase_trailing = false;
    }
  }

  erase_nodes_common(node_keep, first_erased_id, erase_trailing);
  return { erase_trailing ? first_erased_id : -1, std::move(node_keep) };
}

template <class NT, class ET>
void Graph<NT, ET>::erase_nodes_common(std::vector<int> &node_keep,
                                       const int first_erased_id,
                                       const bool erase_trailing) {
  // Fast path 1: no node is erased
  if (first_erased_id < 0 || first_erased_id >= num_nodes())
    return;

  // Phase I: erase the edges
  erase_edges(edge_begin(), edge_end(), [&](ConstEdgeRef edge) {
    return node_keep[edge.src().id()] == 0 || node_keep[edge.dst().id()] == 0;
  });

  // Phase II: erase the nodes & adjacencies
  if (erase_trailing) {
    // Fast path 2: if only trailing nodes are erased, no node number needs to
    // be updated.
    ABSL_DLOG(INFO) << "resizing adjacency & node list";
    // O(1) operations
    nodes_.resize(first_erased_id);
    adj_list_.resize(nodes_.size());
    return;
  }

  // Erase unused nodes and adjacencies, O(V)
  int i = 0;
  erase_if(nodes_, [&](const NT &) { return node_keep[i++] == 0; });
  i = 0;
  erase_if(adj_list_,
           [&](const std::vector<AdjEntry> &) { return node_keep[i++] == 0; });

  // Phase III: update the node numbers in adjacencies and edges, O(V+E)
  mask_to_map(node_keep);

  for (std::vector<AdjEntry> &adjs: adj_list_)
    for (AdjEntry &adj: adjs)
      adj.dst = node_keep[adj.dst];

  for (StoredEdge &edge: edges_) {
    edge.src = node_keep[edge.src];
    edge.dst = node_keep[edge.dst];
  }

  // GCOV_EXCL_START
  ABSL_DCHECK(num_nodes() == adj_list_.size())
      << "node count mismatch: " << num_nodes() << " vs " << adj_list_.size();
  // GCOV_EXCL_STOP
}

template <class NT, class ET>
bool Graph<NT, ET>::erase_edge_between(int src, int dst) {
  int eid = erase_adjacent(src, dst), orig_edges = num_edges();
  if (eid >= orig_edges)
    return false;

  edges_.erase(edges_.begin() + eid);

  if (eid != orig_edges - 1) {
    for (std::vector<AdjEntry> &adjs: adj_list_) {
      for (AdjEntry &adj: adjs) {
        if (adj.eid > eid)
          --adj.eid;
      }
    }
  }

  return true;
}

template <class NT, class ET>
template <class UnaryPred>
std::pair<int, std::vector<int>>
Graph<NT, ET>::erase_edges(const_edge_iterator begin, const_edge_iterator end,
                           UnaryPred pred) {
  // This will also handle size() == 0 case correctly.
  if (begin >= end)
    return { num_edges(), {} };

  // Phase I: mark edges for removal, O(E)
  std::vector<int> edge_keep(num_edges(), 1);
  int first_erased_id = -1;
  bool erase_trailing = end == this->edge_end();
  for (auto it = begin; it != end; ++it) {
    if (pred(*it)) {
      int eid = it->id();
      // GCOV_EXCL_START
      ABSL_ASSUME(eid >= 0);
      // GCOV_EXCL_STOP
      if (first_erased_id < 0)
        first_erased_id = eid;

      edge_keep[eid] = 0;
    } else if (first_erased_id >= 0) {
      erase_trailing = false;
    }
  }

  erase_edges_common(edge_keep, first_erased_id, erase_trailing);
  return { erase_trailing ? first_erased_id : -1, std::move(edge_keep) };
}

template <class NT, class ET>
template <class Iterator, class>
std::pair<int, std::vector<int>> Graph<NT, ET>::erase_edges(Iterator begin,
                                                            Iterator end) {
  // This will also handle size() == 0 case correctly.
  if (begin >= end)
    return { num_edges(), {} };

  // Phase I: mark edges for removal, O(E)
  std::vector<int> edge_keep(num_edges(), 1);
  int first_erased_id = num_edges();
  for (auto it = begin; it != end; ++it) {
    int eid = *it;
    edge_keep[eid] = 0;
    first_erased_id = std::min(first_erased_id, eid);
  }

  bool erase_trailing = true;
  for (int i = num_edges() - 1; i >= first_erased_id; --i) {
    if (edge_keep[i] == 1) {
      erase_trailing = false;
      break;
    }
  }

  erase_edges_common(edge_keep, first_erased_id, erase_trailing);
  return { erase_trailing ? first_erased_id : -1, std::move(edge_keep) };
}

template <class NT, class ET>
void Graph<NT, ET>::erase_edges_common(std::vector<int> &edge_keep,
                                       int first_erased_id,
                                       bool erase_trailing) {
  // Fast path 1: no edge is erased
  if (first_erased_id < 0 || first_erased_id >= num_edges())
    return;

  // Phase II: erase unused adjacencies
  for (std::vector<AdjEntry> &adjs: adj_list_)
    erase_if(adjs,
             [&](const AdjEntry &adj) { return edge_keep[adj.eid] == 0; });

  // Fast path 2: if only trailing edges are erased, no edge number needs to
  // be updated.
  if (erase_trailing) {
    ABSL_DLOG(INFO) << "resizing edge list";
    // O(1) operation
    edges_.resize(first_erased_id);
    return;
  }

  // Phase III: erase the edges
  int i = 0;
  erase_if(edges_, [&](const StoredEdge &) { return edge_keep[i++] == 0; });

  // Phase IV: update the edge numbers in adjacencies, O(V+E)
  mask_to_map(edge_keep);

  for (std::vector<AdjEntry> &adjs: adj_list_)
    for (AdjEntry &adj: adjs)
      adj.eid = edge_keep[adj.eid];
}

namespace internal {
  template <class GT, bool is_const>
  class EigenNeighborIndexer {
  public:
    EigenNeighborIndexer(NodeWrapper<GT, is_const> node): node_(node) { }

    int size() const { return node_.degree(); }

    int operator[](int i) const { return node_[i].dst().id(); }

  private:
    NodeWrapper<GT, is_const> node_;
  };
}  // namespace internal

template <class GT, bool is_const>
internal::EigenNeighborIndexer<GT, is_const>
as_index(internal::NodeWrapper<GT, is_const> node) {
  return { node };
}

template <class, class, bool>
class Subgraph;

namespace internal {
  template <class, bool>
  class SubEdgesFinder;

  template <class GT>
  struct GraphTraits;

  // NOLINTBEGIN(readability-identifier-naming)

  template <class NT, class ET>
  struct GraphTraits<Graph<NT, ET>> { };

  template <class NT, class ET, bool subg_const>
  struct GraphTraits<Subgraph<NT, ET, subg_const>> {
    constexpr static bool is_const = subg_const;
  };

  template <class SGT, bool subg_const>
  struct GraphTraits<SubEdgesFinder<SGT, subg_const>> {
    constexpr static bool is_const = subg_const;
  };

  // NOLINTEND(readability-identifier-naming)

  template <class SGT, bool is_const>
  class SubNodeWrapper {
  public:
    using DT = typename SGT::node_data_type;

    using parent_type = const_if_t<is_const, SGT>;
    using value_type = const_if_t<is_const, DT>;

    using adjacency_iterator =
        std::conditional_t<is_const, typename SGT::const_adjacency_iterator,
                           typename SGT::adjacency_iterator>;

    template <bool other_const>
    using Other = SubNodeWrapper<SGT, other_const>;

    static_assert(!GraphTraits<SGT>::is_const || is_const,
                  "Cannot create non-const SubNodeWrapper from const "
                  "Subgraph");

    constexpr SubNodeWrapper(int idx, DT &data, parent_type &subgraph) noexcept
        : idx_(idx), data_(&data), subgraph_(&subgraph) { }

    template <bool this_const = is_const,
              std::enable_if_t<is_const && this_const, int> = 0>
    constexpr SubNodeWrapper(int idx, const DT &data,
                             parent_type &subgraph) noexcept
        : idx_(idx), data_(&data), subgraph_(&subgraph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr SubNodeWrapper(const Other<other_const> &other) noexcept
        : idx_(other.idx_), data_(other.data_), subgraph_(other.subgraph_) { }

    constexpr int id() const noexcept { return idx_; }

    constexpr value_type &data() const noexcept { return *data_; }

    constexpr int degree() const noexcept { return subgraph_->degree(idx_); }

    adjacency_iterator begin() const noexcept {
      return subgraph_->adj_begin(idx_);
    }

    adjacency_iterator end() const noexcept { return subgraph_->adj_end(idx_); }

    adjacency_iterator find_adjacent(int aid) const noexcept {
      return subgraph_->find_adjacent(idx_, aid);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

    NodeWrapper<typename SGT::graph_type, is_const> as_parent() const noexcept {
      return subgraph_->parent_node(idx_);
    }

  private:
    template <class, bool>
    friend class SubNodeWrapper;

    int idx_;
    value_type *data_;
    parent_type *subgraph_;
  };

  template <class SGT, bool is_const>
  class SubNodeIterator
      : public DataIteratorBase<SubNodeIterator<SGT, is_const>, SGT,
                                SubNodeWrapper<SGT, is_const>, is_const> {
    using Base = typename SubNodeIterator::Parent;

  public:
    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    static_assert(!GraphTraits<SGT>::is_const || is_const,
                  "Cannot create non-const SubNodeIterator from const "
                  "Subgraph");

    using Base::Base;

    // Might look strange, but this is to provide all comparisons between
    // all types of iterators (due to boost implementation). However, they
    // could not be really constructed, so we static_assert to prevent misuse.

    template <
        class SGU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<SGT, SGU> || (is_const && !other_const), int> = 0>
    constexpr SubNodeIterator(
        const SubNodeIterator<SGU, other_const> &other) noexcept
        : Base(other) {
      static_assert(std::is_same_v<SGT, SGU>,
                    "Cannot copy-construct SubNodeIterator from different "
                    "Subgraphs");
    }

    template <
        class SGU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<SGT, SGU> || (is_const && !other_const), int> = 0>
    constexpr SubNodeIterator &
    operator=(const SubNodeIterator<SGU, other_const> &other) noexcept {
      static_assert(std::is_same_v<SGT, SGU>,
                    "Cannot copy-assign SubNodeIterator from different "
                    "Subgraphs");

      Base::operator=(other);
      return *this;
    }

  private:
    friend Base;

    friend class boost::iterator_core_access;

    template <class, bool>
    friend class SubNodeIterator;

    constexpr reference dereference() const noexcept {
      return this->graph()->node(this->index());
    }

    template <class SGU, bool other_const,
              std::enable_if_t<std::is_same_v<typename SGT::graph_type,
                                              typename SGU::graph_type>,
                               int> = 0>
    constexpr bool
    equal(const SubNodeIterator<SGU, other_const> &other) const noexcept {
      return this->index() == other.index();
    }

    template <class SGU, bool other_const,
              std::enable_if_t<std::is_same_v<typename SGT::graph_type,
                                              typename SGU::graph_type>,
                               int> = 0>
    constexpr difference_type
    distance_to(const SubNodeIterator<SGU, other_const> &other) const noexcept {
      return other.index() - this->index();
    }
  };

  template <class SGT, bool is_const>
  class SubAdjIterator
      : public ProxyIterator<SubAdjIterator<SGT, is_const>,
                             AdjWrapper<SGT, is_const>,
                             std::bidirectional_iterator_tag, int> {
    using parent_nonconst_adjacency_iterator =
        typename SGT::graph_type::adjacency_iterator;
    using parent_const_adjacency_iterator =
        typename SGT::graph_type::const_adjacency_iterator;

    using parent_adjacency_iterator =
        std::conditional_t<is_const, parent_const_adjacency_iterator,
                           parent_nonconst_adjacency_iterator>;

    using Traits =
        std::iterator_traits<typename SubAdjIterator::iterator_facade_>;

  public:
    using parent_type = const_if_t<is_const, SGT>;

    using iterator_category = typename Traits::iterator_category;
    using value_type = typename Traits::value_type;
    using difference_type = typename Traits::difference_type;
    using pointer = typename Traits::pointer;
    using reference = typename Traits::reference;

    static_assert(!GraphTraits<SGT>::is_const || is_const,
                  "Cannot create non-const SubAdjIterator from const "
                  "Subgraph");

    constexpr SubAdjIterator(parent_type &subgraph, int src,
                             parent_adjacency_iterator ait) noexcept
        : subgraph_(&subgraph), src_(src), ait_(ait) {
      find_next();
    }

    // Might look strange, but this is to provide all comparisons between
    // all types of iterators (due to boost implementation). However, they
    // could not be really constructed, so we static_assert to prevent misuse.

    template <
        class SGU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<SGT, SGU> || (is_const && !other_const), int> = 0>
    constexpr SubAdjIterator(
        const SubAdjIterator<SGU, other_const> &other) noexcept
        : subgraph_(other.subgraph_), src_(other.src_), dst_(other.dst_),
          ait_(other.ait_) {
      static_assert(std::is_same_v<SGT, SGU>,
                    "Cannot copy-construct SubAdjIterator from different "
                    "Subgraphs");
    }

    template <
        class SGU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<SGT, SGU> || (is_const && !other_const), int> = 0>
    constexpr SubAdjIterator &
    operator=(const SubAdjIterator<SGU, other_const> &other) noexcept {
      static_assert(std::is_same_v<SGT, SGU>,
                    "Cannot copy-assign SubAdjIterator from different "
                    "Subgraphs");

      subgraph_ = other.subgraph_;
      src_ = other.src_;
      dst_ = other.dst_;
      ait_ = other.ait_;
      return *this;
    }

    bool end() const noexcept { return ait_.end(); }

  private:
    template <class, bool>
    friend class SubAdjIterator;

    friend class boost::iterator_core_access;

    // NOT an iterator_facade interface; see constructor for why we need this
    // as a separate function.
    void find_next() {
      for (; !ait_.end(); ++ait_) {
        auto it = subgraph_->find_node(ait_->dst());
        if (it != subgraph_->end()) {
          dst_ = it->id();
          return;
        }
      }
    }

    reference dereference() const noexcept {
      return { *subgraph_, src_, dst_, ait_->eid_ };
    }

    template <class SGU, bool other_const>
    bool equal(const SubAdjIterator<SGU, other_const> &other) const noexcept {
      return ait_ == other.ait_;
    }

    void increment() {
      ++ait_;
      find_next();
    }

    void decrement() {
      for (; !ait_--.begin();) {
        auto it = subgraph_->find_node(ait_->dst());
        if (it != subgraph_->end()) {
          dst_ = it->id();
          return;
        }
      }
    }

    parent_type *subgraph_;
    int src_;
    int dst_;
    parent_adjacency_iterator ait_;
  };

  template <class SGT, bool is_const>
  class SubEdgeWrapper {
  public:
    using DT = typename SGT::edge_data_type;

    using parent_type = const_if_t<is_const, SGT>;
    using value_type = const_if_t<is_const, DT>;

    template <bool other_const>
    using Other = SubEdgeWrapper<SGT, other_const>;

    static_assert(!GraphTraits<SGT>::is_const || is_const,
                  "Cannot create non-const SubEdgeWrapper from const "
                  "Subgraph");

    constexpr SubEdgeWrapper(int src, int dst, int eid, value_type &data,
                             parent_type &subgraph) noexcept
        : src_(src), dst_(dst), eid_(eid), data_(&data), subgraph_(&subgraph) {
    }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr SubEdgeWrapper(const Other<other_const> &other) noexcept
        : src_(other.src_), dst_(other.dst_), eid_(other.eid_),
          data_(other.data_), subgraph_(other.subgraph_) { }

    constexpr auto src() const noexcept { return subgraph_->node(src_); }
    constexpr auto dst() const noexcept { return subgraph_->node(dst_); }

    constexpr value_type &data() const noexcept { return *data_; }

    constexpr Other<true> as_const() const noexcept { return *this; }

    EdgeWrapper<typename SGT::graph_type, is_const> as_parent() const noexcept {
      return subgraph_->parent().edge(eid_);
    }

  private:
    template <class, bool>
    friend class SubEdgeWrapper;

    template <class, bool>
    friend class SubEdgesFinder;

    template <class, bool>
    friend class SubEdgeIterator;

    int src_;
    int dst_;
    int eid_;
    value_type *data_;
    parent_type *subgraph_;
  };

  template <class EFT, bool is_const>
  class SubEdgeIterator
      : public DataIteratorBase<
            SubEdgeIterator<EFT, is_const>, EFT,
            SubEdgeWrapper<typename EFT::graph_type, is_const>, is_const> {
    using Base = typename SubEdgeIterator::Parent;

  public:
    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    static_assert(!GraphTraits<EFT>::is_const || is_const,
                  "Cannot create non-const SubEdgeIterator from const "
                  "SubEdgesFinder");

    using Base::Base;

    // Might look strange, but this is to provide all comparisons between
    // all types of iterators (due to boost implementation). However, they
    // could not be really constructed, so we static_assert to prevent misuse.

    template <
        class EFU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<EFT, EFU> || (is_const && !other_const), int> = 0>
    constexpr SubEdgeIterator(
        const SubEdgeIterator<EFU, other_const> &other) noexcept
        : Base(other) {
      static_assert(std::is_same_v<EFT, EFU>,
                    "Cannot copy-construct SubEdgeIterator from different "
                    "SubEdgesFinder");
    }

    template <
        class EFU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<EFT, EFU> || (is_const && !other_const), int> = 0>
    constexpr SubEdgeIterator &
    operator=(const SubEdgeIterator<EFU, other_const> &other) noexcept {
      static_assert(std::is_same_v<EFT, EFU>,
                    "Cannot copy-assign SubEdgeIterator from different "
                    "SubEdgesFinder");

      Base::operator=(other);
      return *this;
    }

  private:
    friend Base;

    friend class boost::iterator_core_access;

    template <class, bool>
    friend class SubEdgeIterator;

    template <class EFU, bool other_const,
              std::enable_if_t<std::is_same_v<typename EFT::graph_type,
                                              typename EFU::graph_type>,
                               int> = 0>
    constexpr bool
    equal(const SubEdgeIterator<EFU, other_const> &other) const noexcept {
      return this->index() == other.index();
    }

    template <class EFU, bool other_const,
              std::enable_if_t<std::is_same_v<typename EFT::graph_type,
                                              typename EFU::graph_type>,
                               int> = 0>
    constexpr difference_type
    distance_to(const SubEdgeIterator<EFU, other_const> &other) const noexcept {
      return other.index() - this->index();
    }

    reference dereference() const noexcept {
      return (*this->graph())[this->index()];
    }
  };

  template <class SGT, bool is_const>
  class SubEdgesFinder {
  public:
    using graph_type = SGT;
    using parent_type = const_if_t<is_const, SGT>;

    using edge_data_type = typename SGT::edge_data_type;

    using iterator = SubEdgeIterator<SubEdgesFinder, is_const>;
    using const_iterator = SubEdgeIterator<SubEdgesFinder, true>;

    using EdgeRef = typename iterator::value_type;
    using ConstEdgeRef = typename const_iterator::value_type;

    static_assert(!GraphTraits<SGT>::is_const || is_const,
                  "Cannot create non-const SubEdgesFinder from const "
                  "Subgraph");

    template <class SGU, bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubEdgesFinder(const SubEdgesFinder<SGU, other_const> &other)
        : subgraph_(other.subgraph_), edges_(other.edges_) { }

    template <class SGU, bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubEdgesFinder(SubEdgesFinder<SGU, other_const> &&other) noexcept
        : subgraph_(other.subgraph_), edges_(std::move(other.edges_)) { }

    SubEdgesFinder(SGT &subgraph)
        : subgraph_(&subgraph), edges_(find_edges(subgraph)) { }

    template <bool this_const = is_const, std::enable_if_t<this_const, int> = 0>
    SubEdgesFinder(const SGT &subgraph)
        : subgraph_(&subgraph), edges_(find_edges(subgraph)) { }

    SubEdgesFinder(SGT &&subgraph) = delete;

    template <class SGU, bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubEdgesFinder &operator=(const SubEdgesFinder<SGU, other_const> &other) {
      subgraph_ = other.subgraph_;
      edges_ = other.edges_;
      return *this;
    }

    template <class SGU, bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubEdgesFinder &
    operator=(SubEdgesFinder<SGU, other_const> &&other) noexcept {
      subgraph_ = other.subgraph_;
      edges_ = std::move(other.edges_);
      return *this;
    }

    int size() const { return static_cast<int>(edges_.size()) - 1; }

    EdgeRef operator[](int idx) {
      auto [src, dst, eid] = edges_[idx];
      return { src, dst, eid, subgraph_->edge(eid).data(), *subgraph_ };
    }

    ConstEdgeRef operator[](int idx) const {
      auto [src, dst, eid] = edges_[idx];
      return { src, dst, eid, subgraph_->edge(eid).data(), *subgraph_ };
    }

    iterator begin() { return { this, 0 }; }
    iterator end() { return { this, size() }; }
    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    const_iterator cbegin() const { return { this, 0 }; }
    const_iterator cend() const { return { this, size() }; }

  private:
    struct SubEdgeInfo {
      int src;
      int dst;
      int eid;
    };

    static std::vector<SubEdgeInfo>
    find_edges(const_if_t<is_const, SGT> &subgraph) {
      std::vector<SubEdgeInfo> edges;
      std::stack<int, std::vector<int>> stack;
      std::vector<int> visited(subgraph.num_nodes(), 0);

      for (int i = 0; i < subgraph.size(); ++i) {
        if (visited[i] != 0)
          continue;

        visited[i] = 1;
        stack.push(i);

        while (!stack.empty()) {
          int u = stack.top();
          stack.pop();

          for (auto nei: subgraph[u]) {
            int v = nei.dst().id();
            if (u < v)
              edges.push_back({ u, v, nei.eid_ });

            if (visited[v] == 0) {
              visited[v] = 1;
              stack.push(v);
            }
          }
        }
      }

      edges.push_back({ -1, -1, subgraph.parent().num_edges() });

      return edges;
    }

    const_if_t<is_const, SGT> *subgraph_;
    std::vector<SubEdgeInfo> edges_;
  };

  class SortedNodes {
  public:
    SortedNodes() = default;

    SortedNodes(const std::vector<int> &nodes): nodes_(nodes) { init(); }

    SortedNodes(std::vector<int> &&nodes) noexcept: nodes_(std::move(nodes)) {
      init();
    }

    template <class Iter, internal::enable_if_compatible_iter_t<Iter, int> = 0>
    SortedNodes(Iter begin, Iter end): nodes_(begin, end) {
      init();
    }

    bool empty() const { return nodes_.empty(); }

    int size() const { return static_cast<int>(nodes_.size()); }

    void clear() noexcept { nodes_.clear(); }

    void reserve(int num_nodes) { nodes_.reserve(num_nodes); }

    void insert(int id) { insert_sorted(nodes_, id); }

    void erase(int id) {
      int pos = find(id);
      if (pos < size()) {
        nodes_.erase(nodes_.begin() + pos);
      }
    }

    template <class Iter>
    void erase(Iter begin, Iter end) {
      nodes_.erase(begin, end);
    }

    template <class UnaryPred,
              std::enable_if_t<!std::is_convertible_v<UnaryPred, int>, int> = 0>
    void erase(UnaryPred &&pred) {
      erase_if(nodes_, std::forward<UnaryPred>(pred));
    }

    int operator[](int idx) const { return nodes_[idx]; }

    bool contains(int id) const {
      return std::binary_search(nodes_.begin(), nodes_.end(), id);
    }

    int find(int id) const {
      auto it = std::lower_bound(nodes_.begin(), nodes_.end(), id);
      if (it == nodes_.end() || *it != id) {
        return static_cast<int>(nodes_.size());
      }
      return static_cast<int>(it - nodes_.begin());
    }

    auto begin() const { return nodes_.begin(); }
    auto end() const { return nodes_.end(); }

    void replace(const std::vector<int> &nodes) {
      nodes_ = nodes;
      init();
    }

    void replace(std::vector<int> &&nodes) noexcept {
      nodes_ = std::move(nodes);
      init();
    }

    void remap(const std::vector<int> &old_to_new) {
      auto first = std::find_if(nodes_.begin(), nodes_.end(),
                                [&](int id) { return old_to_new[id] < 0; });

      for (auto it = nodes_.begin(); it < first; ++it)
        *it = old_to_new[*it];

      for (auto it = first; it++ < nodes_.end() - 1;) {
        int new_id = old_to_new[*it];
        *first = new_id;
        first += value_if(new_id >= 0);
      }

      nodes_.erase(first, nodes_.end());
    }

    const std::vector<int> &ids() const { return nodes_; }

  private:
    void init() { std::sort(nodes_.begin(), nodes_.end()); }

    std::vector<int> nodes_;
  };
}  // namespace internal

/**
 * @brief A subgraph of a graph.
 *
 * @tparam NT node data type
 * @tparam ET edge data type
 * @tparam is_const whether the subgraph is const. This is to support creating
 *         subgraphs of const graphs.
 * @note Removing nodes/edges on the parent graph will invalidate the
 * subgraph.
 *
 * The subgraph is a non-owning view of a graph. A subgraph could be
 * constructed by selecting nodes of the parent graph. The resulting subgraph
 * will contain only the nodes and edges that are incident to the selected
 * nodes.
 *
 * The subgraph has very similar API to the graph. Here, we note major
 * differences:
 *   - Adding/removing nodes will just mark the nodes as selected/unselected.
 *   - Edges could not be added/removed.
 *   - Iterating edges require graph traversal, so there are no direct edge
 *     iterators in the subgraph.
 *   - Some methods have different time complexity requirements. Refer to the
 *     documentation of each method for details.
 *
 * In all time complexity specifications, \f$V\f$ and \f$E\f$ are the number
 * of nodes and edges in the parent graph, and \f$V'\f$ and \f$E'\f$ are the
 * number of nodes and edges in the subgraph, respectively.
 */
template <class NT, class ET, bool is_const = false>
class Subgraph {
public:
  using graph_type = Graph<NT, ET>;
  using parent_type = internal::const_if_t<is_const, graph_type>;

  using node_data_type = NT;
  using edge_data_type = ET;

  using iterator = internal::SubNodeIterator<Subgraph, is_const>;
  using node_iterator = iterator;
  using const_iterator = internal::SubNodeIterator<Subgraph, true>;
  using const_node_iterator = const_iterator;
  using NodeRef = internal::SubNodeWrapper<Subgraph, is_const>;
  using ConstNodeRef = internal::SubNodeWrapper<Subgraph, true>;

  using adjacency_iterator = internal::SubAdjIterator<Subgraph, is_const>;
  using const_adjacency_iterator = internal::SubAdjIterator<Subgraph, true>;
  using AdjRef = internal::AdjWrapper<Subgraph, is_const>;
  using ConstAdjRef = internal::AdjWrapper<Subgraph, true>;

  static_assert(std::is_same_v<typename iterator::reference, NodeRef>);
  static_assert(
      std::is_same_v<typename const_iterator::reference, ConstNodeRef>);
  static_assert(std::is_same_v<typename adjacency_iterator::reference, AdjRef>);
  static_assert(
      std::is_same_v<typename const_adjacency_iterator::reference, ConstAdjRef>);

  template <bool other_const>
  using Other = Subgraph<NT, ET, other_const>;

  Subgraph(graph_type &&graph) = delete;

  /**
   * @brief Construct an empty subgraph
   *
   * @param graph The parent graph
   */
  Subgraph(parent_type &graph): parent_(&graph) { }

  /**
   * @brief Construct a new Subgraph object
   *
   * @param graph The parent graph
   * @param nodes The set of nodes in the subgraph. If any of the node ids are
   *        not in the parent graph, or if there are duplicates, the behavior
   *        is undefined.
   */
  Subgraph(parent_type &graph, const std::vector<int> &nodes)
      : parent_(&graph), nodes_(nodes) { }

  /**
   * @brief Construct a new Subgraph object
   *
   * @param graph The parent graph
   * @param nodes The set of nodes in the subgraph. If any of the node ids are
   *        not in the parent graph, or if there are duplicates, the behavior
   *        is undefined.
   */
  Subgraph(parent_type &graph, std::vector<int> &&nodes) noexcept
      : parent_(&graph), nodes_(std::move(nodes)) { }

  /**
   * @brief Converting constructor to allow implicit conversion from a
   * non-const subgraph to a const subgraph.
   *
   * @param other The non-const subgraph
   */
  template <bool other_const,
            std::enable_if_t<is_const && !other_const, int> = 0>
  Subgraph(const Other<other_const> &other)
      : parent_(other.parent_), nodes_(other.nodes_) { }

  /**
   * @brief Converting constructor to allow implicit conversion from a
   * non-const subgraph to a const subgraph.
   *
   * @param other The non-const subgraph
   */
  template <bool other_const,
            std::enable_if_t<is_const && !other_const, int> = 0>
  Subgraph(Other<other_const> &&other) noexcept
      : parent_(other.parent_), nodes_(std::move(other.nodes_)) { }

  /**
   * @brief Converting assignment operator to allow implicit conversion from a
   *        non-const subgraph to a const subgraph.
   *
   * @param other The non-const subgraph
   */
  template <bool other_const,
            std::enable_if_t<is_const && !other_const, int> = 0>
  Subgraph &operator=(const Other<other_const> &other) {
    parent_ = other.parent_;
    nodes_ = other.nodes_;
    return *this;
  }

  /**
   * @brief Converting assignment operator to allow implicit conversion from a
   *        non-const subgraph to a const subgraph.
   *
   * @param other The non-const subgraph
   */
  template <bool other_const,
            std::enable_if_t<is_const && !other_const, int> = 0>
  Subgraph &operator=(Other<other_const> &&other) noexcept {
    parent_ = other.parent_;
    nodes_ = std::move(other.nodes_);
    return *this;
  }

  /**
   * @brief Get the parent graph
   *
   * @return Reference (might be const) to the parent graph
   */
  parent_type &parent() { return *parent_; }

  /**
   * @brief Get the parent graph
   *
   * @return const-reference to the parent graph
   */
  const graph_type &parent() const { return *parent_; }

  /**
   * @brief Whether the subgraph is empty
   *
   * @return true if the subgraph is empty, false otherwise
   */
  bool empty() const { return nodes_.empty(); }

  /**
   * @brief Count number of nodes in the subgraph
   *
   * @return The number of nodes in the subgraph
   */
  int size() const { return num_nodes(); }

  /**
   * @brief Count number of nodes in the subgraph
   *
   * @return The number of nodes in the subgraph
   */
  int num_nodes() const { return nodes_.size(); }

  /**
   * @brief Change the set of nodes in the subgraph
   *
   * @param nodes The new set of nodes
   * @note If any of the node ids are not in the parent graph, or if there are
   *       duplicates, the behavior is undefined.
   * @note Time complexity: \f$O(V' \log V')\f$
   */
  void update(const std::vector<int> &nodes) { nodes_.replace(nodes); }

  /**
   * @brief Change the set of nodes in the subgraph
   *
   * @param nodes The new set of nodes
   * @note If any of the node ids are not in the parent graph, or if there are
   *       duplicates, the behavior is undefined.
   * @note Time complexity: \f$O(V' \log V')\f$
   */
  void update(std::vector<int> &&nodes) noexcept {
    nodes_.replace(std::move(nodes));
  }

  /**
   * @brief Clear the subgraph
   *
   * @note This does not affect the parent graph.
   */
  void clear() noexcept { nodes_.clear(); }

  /**
   * @brief Reserve space for a number of nodes
   *
   * @param num_nodes The number of nodes to reserve space for
   */
  void reserve(int num_nodes) { nodes_.reserve(num_nodes); }

  /**
   * @brief Add a node to the subgraph
   *
   * @param id The id of the node to add
   * @note If the node is already in the subgraph, this is a no-op. If the
   * node id is out of range, the behavior is undefined.
   */
  void add_node(int id) { return nodes_.insert(id); }

  /**
   * @brief Check if a node is in the subgraph
   *
   * @param id the id of the node to check
   * @return true if the node is in the subgraph, false otherwise
   * @note Time complexity: \f$O(\log V')\f$.
   */
  bool contains(int id) const { return nodes_.contains(id); }

  /**
   * @brief Check if a node is in the subgraph
   *
   * @param node The node to check
   * @return true if the node is in the subgraph, false otherwise
   * @note This is equivalent to calling contains(node.id()).
   * @note Time complexity: \f$O(\log V')\f$.
   */
  bool contains(typename graph_type::ConstNodeRef node) const {
    return contains(node.id());
  }

  /**
   * @brief Get a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A reference wrapper to the node (might be const)
   */
  NodeRef operator[](int idx) { return node(idx); }

  /**
   * @brief Get a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A const-reference wrapper to the node
   */
  ConstNodeRef operator[](int idx) const { return node(idx); }

  /**
   * @brief Get a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A reference wrapper to the node (might be const)
   */
  NodeRef node(int idx) {
    auto pnode = parent_->node(nodes_[idx]);
    return { idx, pnode.data(), *this };
  }

  /**
   * @brief Get a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A const-reference wrapper to the node
   */
  ConstNodeRef node(int idx) const {
    auto pnode = parent_->node(nodes_[idx]);
    return { idx, pnode.data(), *this };
  }

  /**
   * @brief Get a parent node of a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A reference wrapper to the node (might be const)
   */
  std::conditional_t<is_const, typename parent_type::ConstNodeRef,
                     typename parent_type::NodeRef>
  parent_node(int idx) {
    return parent_->node(nodes_[idx]);
  }

  /**
   * @brief Get a parent node of a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A const-reference wrapper to the node.
   */
  typename parent_type::ConstNodeRef parent_node(int idx) const {
    return parent_->node(nodes_[idx]);
  }

  /**
   * @brief Find a node with the given id.
   *
   * @param id The id of the node to find
   * @return An iterator to the node if found, end() otherwise.
   * @note Time complexity: \f$O(\log V')\f$.
   */
  iterator find_node(int id) { return begin() + nodes_.find(id); }

  /**
   * @brief Find a node with the given id.
   *
   * @param node The node to find
   * @return An iterator to the node if found, end() otherwise.
   * @note This is equivalent to calling find_node(node.id()).
   * @note Time complexity: \f$O(\log V')\f$.
   */
  iterator find_node(typename graph_type::ConstNodeRef node) {
    return find_node(node.id());
  }

  /**
   * @brief Find a node with the given id.
   *
   * @param id The id of the node to find
   * @return A const_iterator to the node if found, end() otherwise
   * @note Time complexity: \f$O(\log V')\f$.
   */
  const_iterator find_node(int id) const { return begin() + nodes_.find(id); }

  /**
   * @brief Find a node with the given id.
   *
   * @param node The node to find
   * @return A const_iterator to the node if found, end() otherwise.
   * @note This is equivalent to calling find_node(node.id()).
   * @note Time complexity: \f$O(\log V')\f$.
   */
  const_iterator find_node(typename graph_type::ConstNodeRef node) const {
    return find_node(node.id());
  }

  /**
   * @brief Erase a node from the subgraph
   *
   * @param idx The index of the node to erase
   * @note The behavior is undefined if idx >= num_nodes().
   * @note Time complexity: \f$O(V')\f$ in worst case.
   */
  void erase_node(int idx) {
    nodes_.erase(nodes_.begin() + idx, nodes_.begin() + idx + 1);
  }

  /**
   * @brief Erase a node from the subgraph
   *
   * @param node The node to erase
   * @note This is equivalent to calling erase_node(node.id()).
   * @note Time complexity: \f$O(V')\f$ in worst case.
   */
  void erase_node(ConstNodeRef node) { erase_node(node.id()); }

  /**
   * @brief Erase range of nodes from the subgraph
   *
   * @param begin Iterator pointing to the first node to erase
   * @param end Iterator pointing to the node after the last node to erase
   * @note Time complexity: \f$O(V')\f$ in worst case.
   */
  void erase_nodes(const_iterator begin, const_iterator end) {
    nodes_.erase(begin - this->begin() + nodes_.begin(),
                 end - this->begin() + nodes_.begin());
  }

  /**
   * @brief Erase a node with given id from the subgraph
   *
   * @param id The id of the node to erase
   * @note This is a no-op if the node is not in the subgraph.
   * @note Time complexity: \f$O(V')\f$ in worst case.
   */
  void erase_node_of(int id) { nodes_.erase(id); }

  /**
   * @brief Erase a node with given id from the subgraph
   *
   * @param node The parent node to erase
   * @note This is a no-op if the node is not in the subgraph.
   * @note Time complexity: \f$O(V')\f$ in worst case.
   */
  void erase_node_of(typename graph_type::ConstNodeRef node) {
    erase_node_of(node.id());
  }

  /**
   * @brief Erase matching nodes from the subgraph
   *
   * @tparam UnaryPred Type of the unary predicate
   * @param pred Unary predicate that returns true for nodes to erase
   * @note Time complexity: \f$O(V')\f$ in worst case.
   */
  template <class UnaryPred>
  void erase_nodes_if(UnaryPred &&pred) {
    nodes_.erase(std::forward<UnaryPred>(pred));
  }

  /**
   * @brief Re-map node ids
   *
   * @param old_to_new A vector that maps old node ids to new node ids, so
   * that old_to_new[old_id] = new_id. If old_to_new[old_id] < 0, then the
   *        node is removed from the subgraph.
   * @note Time complexity: \f$O(V')\f$.
   */
  void remap_nodes(const std::vector<int> &old_to_new) {
    nodes_.remap(old_to_new);
  }

  iterator begin() { return { this, 0 }; }
  iterator end() { return { this, num_nodes() }; }

  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

  const_iterator cbegin() const { return { this, 0 }; }
  const_iterator cend() const { return { this, num_nodes() }; }

  /**
   * @brief Get all node ids in the subgraph
   *
   * @return The node ids in the subgraph, in an unspecified order
   */
  const std::vector<int> &node_ids() const { return nodes_.ids(); }

  /**
   * @brief Find all edges in the subgraph
   *
   * @return A wrapper that can be used to iterate over all edges in the
   *         subgraph (might be const).
   * @note Time complexity: \f$O(V'+E')\f$.
   */
  internal::SubEdgesFinder<Subgraph, is_const> edges() { return { *this }; }

  /**
   * @brief Find all edges in the subgraph
   *
   * @return A const-wrapper that can be used to iterate over all edges in the
   *         subgraph.
   * @note Time complexity: \f$O(V'+E')\f$.
   */
  internal::SubEdgesFinder<Subgraph, true> edges() const { return { *this }; }

  /**
   * @brief Count in-subgraph neighbors of a node
   *
   * @param idx The index of the node
   * @return The number of neighbors of the node that are in the subgraph
   * @note The behavior is undefined if the node is not in the subgraph.
   * @note Time complexity: \f$O(V/E)\f$.
   */
  int degree(int idx) const {
    return std::distance(adj_cbegin(idx), adj_cend(idx));
  }

  /**
   * @brief Find adjacent node of a node
   *
   * @param src The index of one node
   * @param dst The index of the other node
   * @return An iterator to the adjacent node if found, adj_end(src)
   * otherwise.
   * @note This will only find edges that are in the subgraph.
   * @note Time complexity: \f$O(V/E)\f$.
   */
  adjacency_iterator find_adjacent(int src, int dst) {
    return find_adj_helper(*this, src, dst);
  }

  /**
   * @brief Find adjacent node of a node
   *
   * @param src The index of one node
   * @param dst The index of the other node
   * @return A const-iterator to the adjacent node if found, adj_end(src)
   *         otherwise.
   * @note This will only find edges that are in the subgraph.
   * @note Time complexity: \f$O(V/E)\f$.
   */
  const_adjacency_iterator find_adjacent(int src, int dst) const {
    return find_adj_helper(*this, src, dst);
  }

  adjacency_iterator adj_begin(int idx) {
    return { *this, idx, parent_->adj_begin(nodes_[idx]) };
  }
  adjacency_iterator adj_end(int idx) {
    return { *this, idx, parent_->adj_end(nodes_[idx]) };
  }

  const_adjacency_iterator adj_begin(int idx) const { return adj_cbegin(idx); }
  const_adjacency_iterator adj_end(int idx) const { return adj_cend(idx); }

  const_adjacency_iterator adj_cbegin(int idx) const {
    return { *this, idx, parent_->adj_cbegin(nodes_[idx]) };
  }
  const_adjacency_iterator adj_cend(int idx) const {
    return { *this, idx, parent_->adj_cend(nodes_[idx]) };
  }

  /**
   * @brief Re-bind the subgraph to a new parent graph.
   *
   * @param parent The new parent graph to bind.
   * @warning This method does no bookkeeping, so it is the caller's
   *          responsibility to ensure that the new parent graph is compatible
   *          with the subgraph. The behavior is undefined otherwise.
   */
  void rebind(parent_type &parent) { parent_ = &parent; }

private:
  template <class, class, bool>
  friend class Subgraph;

  template <class, bool>
  friend class internal::AdjWrapper;

  template <class, bool>
  friend class internal::SubEdgesFinder;

  // Only for AdjWrapper!

  auto edge(int eid) { return parent().edge(eid); }

  auto edge(int eid) const { return parent().edge(eid); }

  template <class SGT>
  static auto find_adj_helper(SGT &graph, int src, int dst) {
    auto ret = graph.adj_begin(src);

    for (; ret != graph.adj_end(src); ++ret)
      if (ret->dst().id() == dst)
        break;

    return ret;
  }

  parent_type *parent_;
  internal::SortedNodes nodes_;
};

/* Deduction guides */

template <class NT, class ET>
Subgraph(Graph<NT, ET> &graph) -> Subgraph<NT, ET, false>;

template <class NT, class ET>
Subgraph(Graph<NT, ET> &graph, const std::vector<int> &nodes)
    -> Subgraph<NT, ET, false>;

template <class NT, class ET>
Subgraph(Graph<NT, ET> &graph, std::vector<int> &&nodes) noexcept
    -> Subgraph<NT, ET, false>;

template <class NT, class ET>
Subgraph(const Graph<NT, ET> &graph) -> Subgraph<NT, ET, true>;

template <class NT, class ET>
Subgraph(const Graph<NT, ET> &graph, const std::vector<int> &nodes)
    -> Subgraph<NT, ET, true>;

template <class NT, class ET>
Subgraph(const Graph<NT, ET> &graph, std::vector<int> &&nodes) noexcept
    -> Subgraph<NT, ET, true>;

/* Helper templates */

namespace internal {
  template <class T>
  struct SubgraphTypeHelper;

  template <class T>
  struct SubgraphTypeHelper<T &> {
    using type = typename SubgraphTypeHelper<T>::type;
  };

  template <template <class, class> class GT, class NT, class ET>
  struct SubgraphTypeHelper<GT<NT, ET>> {
    using type = Subgraph<NT, ET, false>;
  };

  template <template <class, class> class GT, class NT, class ET>
  struct SubgraphTypeHelper<const GT<NT, ET>> {
    using type = Subgraph<NT, ET, true>;
  };
}  // namespace internal

template <class GT>
using SubgraphOf = typename internal::SubgraphTypeHelper<GT>::type;

/**
 * @brief Make a subgraph from a graph.
 * @tparam GT The type of the graph
 * @tparam Args Extra arguments to pass to the subgraph constructor
 * @param graph The graph to make a subgraph of
 * @param args Extra arguments to pass to the subgraph constructor
 * @return A subgraph of the graph
 */
template <class GT, class... Args>
SubgraphOf<GT> make_subgraph(GT &&graph, Args &&...args) {
  return { std::forward<GT>(graph), std::forward<Args>(args)... };
}

namespace internal {
  template <class NT, class ET>
  void connected_components_impl(const Graph<NT, ET> &g,
                                 absl::flat_hash_set<int> &visited,
                                 std::queue<int> &q) {
    while (!q.empty()) {
      int atom = q.front();
      q.pop();

      auto [_, inserted] = visited.insert(atom);
      if (!inserted)
        continue;

      for (auto ait = g.adj_begin(atom); !ait.end(); ++ait) {
        int dst = ait->dst().id();
        q.push(dst);
      }
    }
  }
}  // namespace internal

template <class NT, class ET>
absl::flat_hash_set<int> connected_components(const Graph<NT, ET> &g,
                                              int begin) {
  absl::flat_hash_set<int> visited;
  std::queue<int> q { { begin } };

  internal::connected_components_impl(g, visited, q);

  return visited;
}

template <class NT, class ET>
absl::flat_hash_set<int> connected_components(const Graph<NT, ET> &g, int begin,
                                              int exclude) {
  absl::flat_hash_set<int> visited { begin };
  std::queue<int> q;

  for (auto ait = g.adj_begin(begin); !ait.end(); ++ait) {
    int dst = ait->dst().id();
    if (dst != exclude)
      q.push(dst);
  }

  internal::connected_components_impl(g, visited, q);

  if (visited.contains(exclude))
    return {};

  return visited;
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_H_ */

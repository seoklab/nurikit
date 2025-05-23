//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_GRAPH_H_
#define NURI_CORE_GRAPH_GRAPH_H_

//! @cond
#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <boost/iterator/iterator_facade.hpp>
//! @endcond

#include "nuri/core/container/container_ext.h"
#include "nuri/core/graph/traits.h"
#include "nuri/iterator.h"
#include "nuri/meta.h"
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

    constexpr DataIteratorBase(parent_type &graph,
                               difference_type index) noexcept
        : graph_(&graph), index_(index) { }

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

    constexpr parent_type &graph() const noexcept { return *graph_; }

    constexpr difference_type index() const noexcept { return index_; }

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

  private:
    template <class, class, class, bool>
    friend class DataIteratorBase;

    friend class boost::iterator_core_access;

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
      return graph_->edge_data(eid_);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class AdjWrapper;

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

    constexpr AdjIterator() noexcept = default;

    constexpr AdjIterator(parent_type &graph, difference_type idx,
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
      return Base::equal(this->graph().adj_begin(nid_));
    }

    constexpr bool end() const noexcept {
      return Base::equal(this->graph().adj_end(nid_));
    }

  private:
    friend Base;

    friend class boost::iterator_core_access;

    template <class, bool>
    friend class AdjIterator;

    template <
        class Other,
        std::enable_if_t<std::is_convertible_v<Other, AdjIterator>, int> = 0>
    constexpr bool equal(const Other &other) const noexcept {
      return nid_ == other.nid_ && Base::equal(other);
    }

    constexpr reference dereference() const noexcept {
      return this->graph().adjacent(nid_, this->index());
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

    constexpr NodeWrapper(int nid, parent_type &graph) noexcept
        : nid_(nid), graph_(&graph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr NodeWrapper(const Other<other_const> &other) noexcept
        : nid_(other.nid_), graph_(other.graph_) { }

    constexpr int id() const noexcept { return nid_; }

    constexpr value_type &data() const noexcept {
      return graph_->node_data(nid_);
    }

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

    adjacency_iterator
    find_adjacent(NodeWrapper<GT, true> node) const noexcept {
      return find_adjacent(node.id());
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class NodeWrapper;

    int nid_;
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
      return this->graph().node(this->index());
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

    constexpr EdgeWrapper(int eid, parent_type &graph) noexcept
        : eid_(eid), graph_(&graph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr EdgeWrapper(const Other<other_const> &other) noexcept
        : eid_(other.eid_), graph_(other.graph_) { }

    constexpr int id() const noexcept { return eid_; }

    constexpr NodeWrapper<GT, is_const> src() const noexcept {
      return graph_->edge_src(eid_);
    }

    constexpr NodeWrapper<GT, is_const> dst() const noexcept {
      return graph_->edge_dst(eid_);
    }

    constexpr value_type &data() const noexcept {
      return graph_->edge_data(eid_);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class EdgeWrapper;

    int eid_;
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
      return this->graph().edge(this->index());
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

    auto operator[](int id) const { return graph_->edge(id); }

    auto begin() const { return graph_->edge_begin(); }
    auto end() const { return graph_->edge_end(); }

    auto cbegin() const { return graph_->edge_cbegin(); }
    auto cend() const { return graph_->edge_cend(); }

    EdgesWrapper<GT, true> as_const() const { return *this; }

    int size() const { return graph_->num_edges(); }

  private:
    const_if_t<is_const, GT> *graph_;
  };

  template <class, bool>
  class SubEdgeWrapper;
}  // namespace internal

using NodesErased = std::pair<std::pair<int, std::vector<int>>,
                              std::pair<int, std::vector<int>>>;

template <class NT, class ET, bool is_const>
class Subgraph;

/**
 * @brief Class for \e very sparse graphs, especially designed for the molecular
 *        graphs.
 *
 * @tparam NT node data type.
 * @tparam ET edge data type.
 *
 * For all time complexity specifications, \f$V\f$ denotes the number of nodes
 * (vertices) and \f$E\f$ denotes the number of edges. If present, \f$N\f$ is
 * the size of the user-supplied range (such as an iterator range; see, e.g.,
 * erase_nodes()).
 */
template <class NT, class ET>
class Graph {
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

  /**
   * @brief Create a graph with \p num_nodes nodes.
   * @param num_nodes The number of nodes in the graph. All node data will be
   *        default-constructed.
   */
  Graph(int num_nodes): adj_list_(num_nodes), nodes_(num_nodes) { }

  /**
   * @brief Create a graph with \p num_nodes nodes, each initialized with
   *        \p data.
   * @param num_nodes The number of nodes in the graph.
   * @param data The data to copy-initialize each node with.
   */
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

  /**
   * @brief Add a node to the graph.
   * @param data The data to copy-construct the node with.
   * @return The id of the newly added node.
   * @note Time complexity: \f$O(1)\f$ amortized.
   */
  int add_node(const NT &data) {
    int id = num_nodes();
    nodes_.push_back(data);
    adj_list_.push_back({});
    return id;
  }

  /**
   * @brief Add a node to the graph.
   * @param data The data to move-construct the node with.
   * @return The id of the newly added node.
   * @note Time complexity: \f$O(1)\f$ amortized.
   */
  int add_node(NT &&data) noexcept {
    int id = num_nodes();
    nodes_.push_back(std::move(data));
    adj_list_.push_back({});
    return id;
  }

  /**
   * @brief Add multiple nodes to the graph.
   * @tparam Iterator The type of the iterator. Must be dereferenceable to
   *         a value type implicitly convertible to `NT`.
   * @param begin The beginning of the range of nodes to be added.
   * @param end The end of the range of nodes to be added.
   * @note Time complexity: \f$O(N)\f$.
   */
  template <class Iterator,
            internal::enable_if_compatible_iter_t<Iterator, NT> = 0>
  void add_nodes(Iterator begin, Iterator end) {
    nodes_.insert(nodes_.end(), begin, end);
    adj_list_.resize(num_nodes());
  }

  /**
   * @brief Add an edge to the graph.
   * @param src The source node id.
   * @param dst The destination node id.
   * @param data The data to copy-construct the edge with.
   * @return The id of the newly added edge.
   * @note Time complexity: \f$O(1)\f$ amortized.
   * @note If \p src or \p dst is out of range, \p src equals \p dst, or an edge
   *       between \p src and \p dst already exists, the behavior is undefined.
   */
  int add_edge(int src, int dst, const ET &data) {
    ABSL_DCHECK_NE(src, dst) << "self-loop is not allowed";

    int eid = num_edges();
    edges_.push_back({ src, dst, data });
    add_adjacency_entry(src, dst, eid);
    return eid;
  }

  /**
   * @brief Add an edge to the graph.
   * @param src The source node id.
   * @param dst The destination node id.
   * @param data The data to move-construct the edge with.
   * @return The id of the newly added edge.
   * @note Time complexity: \f$O(1)\f$ amortized.
   * @note If \p src or \p dst is out of range, \p src equals \p dst, or an edge
   *       between \p src and \p dst already exists, the behavior is undefined.
   */
  int add_edge(int src, int dst, ET &&data) noexcept {
    ABSL_DCHECK_NE(src, dst) << "self-loop is not allowed";

    int eid = num_edges();
    edges_.push_back({ src, dst, std::move(data) });
    add_adjacency_entry(src, dst, eid);
    return eid;
  }

  NodeRef operator[](int id) { return node(id); }
  ConstNodeRef operator[](int id) const { return node(id); }

  NodeRef node(int id) { return { id, *this }; }
  ConstNodeRef node(int id) const { return { id, *this }; }

  NT &node_data(int id) { return nodes_[id]; }
  const NT &node_data(int id) const { return nodes_[id]; }

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
   *       \f$O(V+E)\f$ otherwise.
   * @note If \p id is out of range, the behavior is undefined.
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
   * @return A pair for node and edge erasure result. Each pair contains a
   *         pair of (`new end id`, mapping of `old id -> new id`). For nodes,
   *         if only trailing nodes are erased, `new end id` will be set to the
   *         first erased node id, and the mapping will be in a valid but
   *         unspecified state. If no nodes are erased (special case of trailing
   *         node removal), `new end id` will be equal to the size of the graph
   *         before this operation. Otherwise, `new end id` will be set to -1
   *         and erased nodes will be marked as -1 in the mapping. The same rule
   *         applies to edges.
   * @sa pop_node()
   * @note Time complexity:
   *         1. \f$O(N)\f$ if no nodes are erased,
   *         2. \f$O(V)\f$ if only trailing nodes are erased and no edges are
   *            erased,
   *         3. \f$O(V+E)\f$ otherwise.
   * @note If any of the iterators in range `[`\p begin, \p end`)` is out of
   *       range, the behavior is undefined.
   */
  NodesErased erase_nodes(const_iterator begin, const_iterator end) {
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
   * @return A pair for node and edge erasure result. Each pair contains a
   *         pair of (`new end id`, mapping of `old id -> new id`). For nodes,
   *         if only trailing nodes are erased, `new end id` will be set to the
   *         first erased node id, and the mapping will be in a valid but
   *         unspecified state. If no nodes are erased (special case of trailing
   *         node removal), `new end id` will be equal to the size of the graph
   *         before this operation. Otherwise, `new end id` will be set to -1
   *         and erased nodes will be marked as -1 in the mapping. The same rule
   *         applies to edges.
   * @sa pop_node()
   * @note Time complexity: same as erase_nodes(const_iterator, const_iterator).
   * @note If any of the iterators in range `[`\p begin, \p end`)` is out of
   *       range, the behavior is undefined.
   */
  template <class UnaryPred>
  NodesErased erase_nodes(const_iterator begin, const_iterator end,
                          UnaryPred pred);

  /**
   * @brief Erase nodes and all its associated edge(s) from the graph.
   *
   * @tparam Iterator An iterator  type that dereferences to a value compatible
   *         with `int`.
   * @param begin The beginning of the range of node ids to be erased.
   * @param end The end of the range of node ids to be erased.
   * @return A pair for node and edge erasure result. Each pair contains a
   *         pair of (`new end id`, mapping of `old id -> new id`). For nodes,
   *         if only trailing nodes are erased, `new end id` will be set to the
   *         first erased node id, and the mapping will be in a valid but
   *         unspecified state. If no nodes are erased (special case of trailing
   *         node removal), `new end id` will be equal to the size of the graph
   *         before this operation. Otherwise, `new end id` will be set to -1
   *         and erased nodes will be marked as -1 in the mapping. The same rule
   *         applies to edges.
   * @sa pop_node()
   * @note Time complexity: same as erase_nodes(const_iterator, const_iterator).
   * @note If any of the iterators in range `[`\p begin, \p end`)` points to an
   *       invalid node id, the behavior is undefined.
   */
  template <class Iterator,
            class = internal::enable_if_compatible_iter_t<Iterator, int>>
  NodesErased erase_nodes(Iterator begin, Iterator end);

  iterator begin() { return { *this, 0 }; }
  iterator end() { return { *this, num_nodes() }; }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  const_iterator cbegin() const { return { *this, 0 }; }
  const_iterator cend() const { return { *this, num_nodes() }; }

  EdgeRef edge(int id) { return { id, *this }; }
  ConstEdgeRef edge(int id) const { return { id, *this }; }

  NodeRef edge_src(int id) { return this->node(edges_[id].src); }
  ConstNodeRef edge_src(int id) const { return this->node(edges_[id].src); }

  NodeRef edge_dst(int id) { return this->node(edges_[id].dst); }
  ConstNodeRef edge_dst(int id) const { return this->node(edges_[id].dst); }

  ET &edge_data(int id) { return edges_[id].data; }
  const ET &edge_data(int id) const { return edges_[id].data; }

  /**
   * @brief Find an edge between two nodes.
   * @param src The source node id.
   * @param dst The destination node id.
   * @return Iterator to the edge if found, otherwise the end iterator.
   * @note Time complexity: \f$O(V/E)\f$.
   * @note If \p src or \p dst is out of range, the behavior is undefined.
   */
  edge_iterator find_edge(int src, int dst) {
    return find_edge_helper(*this, src, dst);
  }

  /**
   * @brief Find an edge between two nodes.
   * @param src The source node id.
   * @param dst The destination node id.
   * @return Iterator to the edge if found, otherwise the end iterator.
   * @note Time complexity: same as find_edge(int, int).
   * @note If \p src or \p dst is out of range, the behavior is undefined.
   */
  const_edge_iterator find_edge(int src, int dst) const {
    return find_edge_helper(*this, src, dst);
  }

  /**
   * @brief Find an edge between two nodes.
   * @param src The source node.
   * @param dst The destination node.
   * @return Iterator to the edge if found, otherwise the end iterator.
   * @note Time complexity: same as find_edge(int, int).
   * @note If \p src or \p dst does not belong to this graph, the behavior is
   *       undefined.
   */
  edge_iterator find_edge(ConstNodeRef src, ConstNodeRef dst) {
    return find_edge(src.id(), dst.id());
  }

  /**
   * @brief Find an edge between two nodes.
   * @param src The source node.
   * @param dst The destination node.
   * @return Iterator to the edge if found, otherwise the end iterator.
   * @note Time complexity: same as find_edge(int, int).
   * @note If \p src or \p dst does not belong to this graph, the behavior is
   *       undefined.
   */
  const_edge_iterator find_edge(ConstNodeRef src, ConstNodeRef dst) const {
    return find_edge(src.id(), dst.id());
  }

  /**
   * @brief Remove all edges from the graph.
   * @note Time complexity: \f$O(V)\f$.
   */
  void clear_edges() {
    edges_.clear();
    for (std::vector<AdjEntry> &adj: adj_list_)
      adj.clear();
  }

  /**
   * @brief Erase an edge from the graph.
   *
   * @param id The id of the edge to be erased.
   * @return The data of the erased edge.
   * @sa erase_edge(), erase_edge_between(), erase_edges()
   * @note Time complexity: same as erase_edge().
   * @note If \p id is out of range, the behavior is undefined.
   */
  ET pop_edge(int id) {
    ET ret = std::move(edges_[id].data);
    erase_edge(id);
    return ret;
  }

  /**
   * @brief Erase an edge from the graph.
   *
   * @param id The id of the edge to be erased.
   * @sa pop_edge(), erase_edge_between(), erase_edges()
   * @note Time complexity: \f$O(E/V)\f$ if the edge is the last edge,
   *       \f$O(V+E)\f$ otherwise.
   * @note If \p id is out of range, the behavior is undefined.
   */
  void erase_edge(int id) {
    const StoredEdge &edge = edges_[id];
    auto srcit = find_adjacency_entry(edge.src, edge.dst),
         dstit = find_adjacency_entry(edge.dst, edge.src);
    erase_adjacency_entry(edge.src, srcit, edge.dst, dstit);

    erase_edge_common(id);
  }

  /**
   * @brief Erase an edge from the graph between two nodes.
   *
   * @param src The source node, interchangeable with \p dst.
   * @param dst The destination node, interchangeable with \p src.
   * @return Whether the edge is erased.
   * @sa pop_edge(), erase_edge(), erase_edges()
   * @note Time complexity: same as erase_edge().
   * @note If \p src or \p dst is out of range, the behavior is undefined.
   */
  bool erase_edge_between(int src, int dst);

  /**
   * @brief Erase an edge from the graph between two nodes.
   *
   * @param src The source node, interchangeable with \p dst.
   * @param dst The destination node, interchangeable with \p src.
   * @return Whether the edge is erased.
   * @sa pop_edge(), erase_edge(), erase_edges()
   * @note Time complexity: same as erase_edge().
   * @note If \p src or \p dst does not belong to this graph, the behavior is
   *       undefined.
   */
  bool erase_edge_between(ConstNodeRef src, ConstNodeRef dst) {
    return erase_edge_between(src.id(), dst.id());
  }

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
   * @sa pop_edge(), erase_edge(), erase_edge_between()
   * @note Time complexity: \f$O(N)\f$ if no edges were removed, \f$O(V+E)\f$
   *       otherwise.
   * @note If any of the iterators in range `[`\p begin, \p end`)` is out of
   *       range, the behavior is undefined.
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
   * @sa pop_edge(), erase_edge(), erase_edge_between()
   * @note Time complexity: same as
   *       erase_edges(const_edge_iterator, const_edge_iterator).
   * @note If any of the iterators in range `[`\p begin, \p end`)` is out of
   *       range, the behavior is undefined.
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
   * @sa pop_edge(), erase_edge(), erase_edge_between()
   * @note Time complexity: same as
   *       erase_edges(const_edge_iterator, const_edge_iterator).
   * @note If any iterator in range `[`\p begin, \p end`)` references an invalid
   *       edge id, the behavior is undefined.
   */
  template <class Iterator,
            class = internal::enable_if_compatible_iter_t<Iterator, int>>
  std::pair<int, std::vector<int>> erase_edges(Iterator begin, Iterator end);

  internal::EdgesWrapper<Graph, false> edges() { return { *this }; }
  internal::EdgesWrapper<Graph, true> edges() const { return { *this }; }
  internal::EdgesWrapper<Graph, true> cedges() const { return { *this }; }

  edge_iterator edge_begin() { return { *this, 0 }; }
  edge_iterator edge_end() { return { *this, num_edges() }; }
  const_edge_iterator edge_begin() const { return edge_cbegin(); }
  const_edge_iterator edge_end() const { return edge_cend(); }
  const_edge_iterator edge_cbegin() const { return { *this, 0 }; }
  const_edge_iterator edge_cend() const { return { *this, num_edges() }; }

  /**
   * @brief Find an adjacency entry between two nodes.
   * @param src The source node id.
   * @param dst The destination node id.
   * @return Iterator to the adjacency entry if found, otherwise the end
   *         iterator of the adjacency list of \p src.
   * @note Time complexity: \f$O(V/E)\f$.
   * @note If \p src or \p dst is out of range, the behavior is undefined.
   */
  adjacency_iterator find_adjacent(int src, int dst) {
    return find_adj_helper(*this, src, dst);
  }

  /**
   * @brief Find an adjacency entry between two nodes.
   * @param src The source node id.
   * @param dst The destination node id.
   * @return Iterator to the adjacency entry if found, otherwise the end
   *         iterator of the adjacency list of \p src.
   * @note Time complexity: \f$O(V/E)\f$.
   * @note If \p src or \p dst is out of range, the behavior is undefined.
   */
  const_adjacency_iterator find_adjacent(int src, int dst) const {
    return find_adj_helper(*this, src, dst);
  }

  /**
   * @brief Find an adjacency entry between two nodes.
   * @param src The source node.
   * @param dst The destination node.
   * @return Iterator to the adjacency entry if found, otherwise the end
   *         iterator of the adjacency list of \p src.
   * @note Time complexity: \f$O(V/E)\f$.
   * @note If \p src or \p dst does not belong to this graph, the behavior is
   *       undefined.
   */
  adjacency_iterator find_adjacent(ConstNodeRef src, ConstNodeRef dst) {
    return find_adjacent(src.id(), dst.id());
  }

  /**
   * @brief Find an adjacency entry between two nodes.
   * @param src The source node.
   * @param dst The destination node.
   * @return Iterator to the adjacency entry if found, otherwise the end
   *         iterator of the adjacency list of \p src.
   * @note Time complexity: \f$O(V/E)\f$.
   * @note If \p src or \p dst does not belong to this graph, the behavior is
   *       undefined.
   */
  const_adjacency_iterator find_adjacent(ConstNodeRef src,
                                         ConstNodeRef dst) const {
    return find_adjacent(src.id(), dst.id());
  }

  adjacency_iterator adj_begin(int nid) { return { *this, 0, nid }; }
  adjacency_iterator adj_end(int nid) { return { *this, degree(nid), nid }; }
  const_adjacency_iterator adj_begin(int nid) const { return adj_cbegin(nid); }
  const_adjacency_iterator adj_end(int nid) const { return adj_cend(nid); }
  const_adjacency_iterator adj_cbegin(int nid) const {
    return { *this, 0, nid };
  }
  const_adjacency_iterator adj_cend(int nid) const {
    return { *this, degree(nid), nid };
  }

  /**
   * @brief Merge another graph-like object into this graph.
   * @tparam GraphLike The type of the graph-like object to be merged.
   * @param other The graph to be merged.
   * @note Time complexity: \f$O(V'+E')\f$ addition cost, where \f$V'\f$ and
   *       \f$E'\f$ are the number of nodes and edges in \p other, respectively.
   *       The actual time complexity may vary depending on the implementation
   *       of the graph-like object.
   */
  template <class GraphLike>
  void merge(const GraphLike &other) {
    const int offset = size();
    reserve(offset + other.size());

    for (auto node: other)
      add_node(node.data());

    auto edges = other.edges();
    reserve_edges(num_edges() + edges.size());

    for (auto edge: edges)
      add_edge(edge.src().id() + offset, edge.dst().id() + offset, edge.data());
  }

private:
  template <class, bool>
  friend class internal::NodeWrapper;

  template <class, bool>
  friend class internal::EdgeIterator;

  template <class, bool>
  friend class internal::AdjIterator;

  friend class Subgraph<NT, ET, false>;
  friend class Subgraph<NT, ET, true>;

  template <class GT>
  static internal::AdjIterator<Graph, std::is_const_v<GT>>
  find_adj_helper(GT &graph, int src, int dst) {
    auto ret = graph.adj_begin(src);

    for (; !ret.end(); ++ret)
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

    return { graph, ait->eid_ };
  }

  std::pair<int, std::vector<int>>
  erase_nodes_common(std::vector<int> &node_keep, int first_erased_id,
                     bool erase_trailing);

  void erase_edge_common(int id);

  void erase_edges_common(std::vector<int> &edge_keep, int first_erased_id,
                          bool erase_trailing);

  void add_adjacency_entry(int src, int dst, int eid) {
    adj_list_[src].push_back({ dst, eid });
    adj_list_[dst].push_back({ src, eid });
  }

  auto find_adjacency_entry(int src, int dst) {
    return std::find_if(adj_list_[src].begin(), adj_list_[src].end(),
                        [dst](const AdjEntry &adj) { return adj.dst == dst; });
  }

  void erase_adjacency_entry(
      int src, typename std::vector<AdjEntry>::const_iterator srcit, int dst,
      typename std::vector<AdjEntry>::const_iterator dstit) {
    adj_list_[src].erase(srcit);
    adj_list_[dst].erase(dstit);
  }

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
NodesErased Graph<NT, ET>::erase_nodes(Iterator begin, Iterator end) {
  // Note: the time complexity notations are only for very sparse graphs, i.e.,
  // E = O(V).
  if (begin == end) {
    return {
      { num_nodes(), {} },
      { num_edges(), {} },
    };
  }

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

  auto edge_info =
      erase_nodes_common(node_keep, first_erased_id, erase_trailing);
  return {
    { erase_trailing ? first_erased_id : -1, std::move(node_keep) },
    std::move(edge_info),
  };
}

template <class NT, class ET>
template <class UnaryPred>
NodesErased Graph<NT, ET>::erase_nodes(const const_iterator begin,
                                       const const_iterator end,
                                       UnaryPred pred) {
  // Note: the time complexity notations are only for very sparse graphs, i.e.,
  // E = O(V).

  // This will also handle size() == 0 case correctly.
  if (begin >= end) {
    return {
      { num_nodes(), {} },
      { num_edges(), {} },
    };
  }

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

  auto edge_info =
      erase_nodes_common(node_keep, first_erased_id, erase_trailing);
  return {
    { erase_trailing ? first_erased_id : -1, std::move(node_keep) },
    std::move(edge_info),
  };
}

template <class NT, class ET>
std::pair<int, std::vector<int>>
Graph<NT, ET>::erase_nodes_common(std::vector<int> &node_keep,
                                  const int first_erased_id,
                                  const bool erase_trailing) {
  // Fast path 1: no node is erased
  if (first_erased_id < 0 || first_erased_id >= num_nodes())
    return { num_edges(), {} };

  // Phase II: erase the edges, O(V+E)
  auto ret = erase_edges(edge_begin(), edge_end(), [&](ConstEdgeRef edge) {
    return node_keep[edge.src().id()] == 0 || node_keep[edge.dst().id()] == 0;
  });

  // Phase III: erase the nodes & adjacencies
  if (erase_trailing) {
    // Fast path 2: if only trailing nodes are erased, no node number needs to
    // be updated.
    ABSL_DLOG(INFO) << "resizing adjacency & node list";
    // O(1) operations
    nodes_.resize(first_erased_id);
    adj_list_.resize(nodes_.size());
    return ret;
  }

  // Erase unused nodes and adjacencies, O(V)
  int i = 0;
  erase_if(nodes_,
           [&](const NT & /* unused */) { return node_keep[i++] == 0; });
  i = 0;
  erase_if(adj_list_, [&](const std::vector<AdjEntry> & /* unused */) {
    return node_keep[i++] == 0;
  });

  // Phase IV: update the node numbers in adjacencies and edges, O(V+E)
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

  return ret;
}

template <class NT, class ET>
bool Graph<NT, ET>::erase_edge_between(int src, int dst) {
  if (degree(src) > degree(dst))
    return erase_edge_between(dst, src);

  auto srcit = find_adjacency_entry(src, dst);
  if (srcit == adj_list_[src].end())
    return false;

  int eid = srcit->eid;
  // NOLINTNEXTLINE(readability-suspicious-call-argument)
  auto dstit = find_adjacency_entry(dst, src);
  erase_adjacency_entry(src, srcit, dst, dstit);

  erase_edge_common(eid);

  return true;
}

template <class NT, class ET>
void Graph<NT, ET>::erase_edge_common(int id) {
  int orig_edges = num_edges();
  edges_.erase(edges_.begin() + id);

  if (id == orig_edges - 1)
    return;

  for (std::vector<AdjEntry> &adjs: adj_list_) {
    for (AdjEntry &adj: adjs) {
      if (adj.eid > id)
        --adj.eid;
    }
  }
}

template <class NT, class ET>
template <class UnaryPred>
std::pair<int, std::vector<int>>
Graph<NT, ET>::erase_edges(const_edge_iterator begin, const_edge_iterator end,
                           UnaryPred pred) {
  // This will also handle num_edges() == 0 case correctly.
  if (begin >= end)
    return { num_edges(), {} };

  // Phase I: mark edges for removal, O(N)
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
  if (begin == end)
    return { num_edges(), {} };

  // Phase I: mark edges for removal, O(N)
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

  // Phase II: erase unused adjacencies, O(V+E)
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

  // Phase III: erase the edges, O(E)
  int i = 0;
  erase_if(edges_, [&](const StoredEdge & /* unused */) {
    return edge_keep[i++] == 0;
  });

  // Phase IV: update the edge numbers in adjacencies, O(V+E)
  mask_to_map(edge_keep);

  for (std::vector<AdjEntry> &adjs: adj_list_)
    for (AdjEntry &adj: adjs)
      adj.eid = edge_keep[adj.eid];
}

namespace internal {
  // NOLINTBEGIN(readability-identifier-naming)
  template <class NT, class ET>
  struct GraphTraits<Graph<NT, ET>> {
    constexpr static bool is_degree_constant_time = true;
  };
  // NOLINTEND(readability-identifier-naming)

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

/**
 * @brief Find connected components of a graph starting from a node
 *
 * @tparam NT The type of the node data
 * @tparam ET The type of the edge data
 * @param g The graph to find connected components of
 * @param begin The node to start from
 * @return A set of node ids that are in the same connected component as the
 *         starting node.
 *
 * @note Time complexity: \f$O(V + E)\f$.
 */
template <class NT, class ET>
absl::flat_hash_set<int> connected_components(const Graph<NT, ET> &g,
                                              int begin) {
  absl::flat_hash_set<int> visited;
  std::queue<int> q { { begin } };

  internal::connected_components_impl(g, visited, q);

  return visited;
}

/**
 * @brief Find connected components of a graph starting from a node, excluding
 *        an edge between the starting node and the excluded node (if present).
 *
 * @tparam NT The type of the node data
 * @tparam ET The type of the edge data
 * @param g The graph to find connected components of
 * @param begin The node to start from
 * @param exclude The node to exclude
 * @return A set of node ids that are in the same connected component as the
 *         starting node, when the edge between the starting node and the
 *         excluded node is removed. If the excluded node is in the same
 *         connected component as the starting node even after removing the
 *         edge, an empty set is returned.
 *
 * @note Time complexity: \f$O(V + E)\f$.
 */
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

#endif /* NURI_CORE_GRAPH_GRAPH_H_ */

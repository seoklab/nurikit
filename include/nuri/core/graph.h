//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_H_
#define NURI_CORE_GRAPH_H_

//! @cond
#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <queue>
#include <stack>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <boost/container/flat_set.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
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

  template <class, class, bool>
  friend class Subgraph;

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
  struct GraphTraits<Graph<NT, ET>> {
    constexpr static bool is_degree_constant_time = true;
  };

  template <class NT, class ET, bool subg_const>
  struct GraphTraits<Subgraph<NT, ET, subg_const>> {
    constexpr static bool is_const = subg_const;
    constexpr static bool is_degree_constant_time = false;
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

    constexpr SubNodeWrapper(int idx, parent_type &subgraph) noexcept
        : idx_(idx), subgraph_(&subgraph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr SubNodeWrapper(const Other<other_const> &other) noexcept
        : idx_(other.idx_), subgraph_(other.subgraph_) { }

    constexpr int id() const noexcept { return idx_; }

    constexpr value_type &data() const noexcept {
      return subgraph_->node_data(idx_);
    }

    constexpr int degree() const noexcept { return subgraph_->degree(idx_); }

    adjacency_iterator begin() const noexcept {
      return subgraph_->adj_begin(idx_);
    }

    adjacency_iterator end() const noexcept { return subgraph_->adj_end(idx_); }

    adjacency_iterator
    find_adjacent(SubNodeWrapper<SGT, true> node) const noexcept {
      return subgraph_->find_adjacent(*this, node);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

    NodeWrapper<typename SGT::graph_type, is_const> as_parent() const noexcept {
      return subgraph_->parent_node(idx_);
    }

  private:
    template <class, bool>
    friend class SubNodeWrapper;

    int idx_;
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
      return this->graph().node(this->index());
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
  class SubAdjWrapper {
  public:
    using edge_value_type = const_if_t<is_const, typename SGT::edge_data_type>;
    using parent_type = const_if_t<is_const, SGT>;

    template <bool other_const>
    using Other = SubAdjWrapper<SGT, other_const>;

    constexpr SubAdjWrapper(parent_type &graph, int src, int dst,
                            int eid) noexcept
        : src_(src), dst_(dst), eid_(eid), graph_(&graph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr SubAdjWrapper(const Other<other_const> &other) noexcept
        : src_(other.src_), dst_(other.dst_), eid_(other.eid_),
          graph_(other.graph_) { }

    constexpr auto src() const noexcept { return graph_->node(src_); }
    constexpr auto dst() const noexcept { return graph_->node(dst_); }

    constexpr int eid() const noexcept { return eid_; }
    constexpr edge_value_type &edge_data() const noexcept {
      return graph_->edge_data(eid_);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

    constexpr auto as_parent() const noexcept {
      return internal::AdjWrapper<typename SGT::graph_type, is_const>(
          graph_->parent(), graph_->parent_node(src_).id(),
          graph_->parent_node(dst_).id(), graph_->parent_edge(eid_).id());
    }

  private:
    template <class, bool>
    friend class SubAdjWrapper;

    template <class, bool>
    friend class SubAdjIterator;

    friend SGT;

    int src_;
    int dst_;
    int eid_;
    parent_type *graph_;
  };

  template <class SGT, bool is_const>
  class SubAdjIterator
      : public ProxyIterator<SubAdjIterator<SGT, is_const>,
                             SubAdjWrapper<SGT, is_const>,
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
                             int parent_idx) noexcept
        : subgraph_(&subgraph), src_(src), parent_idx_(parent_idx) {
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
          eid_(other.eid_), parent_idx_(other.parent_idx_) {
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
      eid_ = other.eid_;
      parent_idx_ = other.parent_idx_;
      return *this;
    }

    bool begin() const noexcept { return parent_idx_ <= 0; }

    bool end() const noexcept {
      return parent_idx_ >= subgraph_->parent_node(src_).degree();
    }

  private:
    friend SGT;

    template <class, bool>
    friend class SubAdjIterator;

    friend class boost::iterator_core_access;

    constexpr SubAdjIterator(parent_type &subgraph, int src, int dst, int eid,
                             int parent_idx) noexcept
        : subgraph_(&subgraph), src_(src), dst_(dst), eid_(eid),
          parent_idx_(parent_idx) { }

    auto parent_iter() const {
      return subgraph_->parent_node(src_).begin() + parent_idx_;
    }

    // NOT an iterator_facade interface; see constructor for why we need this
    // as a separate function.
    void find_next() {
      for (auto ait = parent_iter(); !ait.end(); ++ait, ++parent_idx_) {
        auto nit = subgraph_->find_node(ait->dst());
        auto eit = subgraph_->find_edge(ait->eid());
        if (nit != subgraph_->node_end() && eit != subgraph_->edge_end()) {
          dst_ = nit->id();
          eid_ = eit->id();
          return;
        }
      }
    }

    reference dereference() const noexcept {
      return { *subgraph_, src_, dst_, eid_ };
    }

    template <class SGU, bool other_const>
    bool equal(const SubAdjIterator<SGU, other_const> &other) const noexcept {
      return parent_idx_ == other.parent_idx_;
    }

    void increment() {
      ++parent_idx_;
      find_next();
    }

    void decrement() {
      --parent_idx_;
      for (auto ait = parent_iter(); parent_idx_ >= 0; --ait, --parent_idx_) {
        auto nit = subgraph_->find_node(ait->dst());
        auto eit = subgraph_->find_edge(ait->eid());
        if (nit != subgraph_->node_end() && eit != subgraph_->edge_end()) {
          dst_ = nit->id();
          eid_ = eit->id();
          return;
        }
      }
    }

    parent_type *subgraph_;
    int src_;
    int dst_;
    int eid_;
    int parent_idx_;
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

    constexpr SubEdgeWrapper(int src, int dst, int eid,
                             parent_type &subgraph) noexcept
        : src_(src), dst_(dst), eid_(eid), subgraph_(&subgraph) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    constexpr SubEdgeWrapper(const Other<other_const> &other) noexcept
        : src_(other.src_), dst_(other.dst_), eid_(other.eid_),
          subgraph_(other.subgraph_) { }

    constexpr auto id() const noexcept { return eid_; }

    constexpr auto src() const noexcept { return subgraph_->node(src_); }
    constexpr auto dst() const noexcept { return subgraph_->node(dst_); }

    constexpr value_type &data() const noexcept {
      return subgraph_->edge_data(eid_);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

    EdgeWrapper<typename SGT::graph_type, is_const> as_parent() const noexcept {
      return subgraph_->parent_edge(eid_);
    }

  private:
    template <class, bool>
    friend class SubEdgeWrapper;

    template <class, bool>
    friend class SubEdgeIterator;

    int src_;
    int dst_;
    int eid_;
    parent_type *subgraph_;
  };

  template <class SGT, bool is_const>
  class SubEdgeIterator
      : public DataIteratorBase<SubEdgeIterator<SGT, is_const>, SGT,
                                SubEdgeWrapper<SGT, is_const>, is_const> {
    using Base = typename SubEdgeIterator::Parent;

  public:
    using typename Base::difference_type;
    using typename Base::iterator_category;
    using typename Base::pointer;
    using typename Base::reference;
    using typename Base::value_type;

    static_assert(!GraphTraits<SGT>::is_const || is_const,
                  "Cannot create non-const SubEdgeIterator from const "
                  "Subgraph");

    using Base::Base;

    // Might look strange, but this is to provide all comparisons between
    // all types of iterators (due to boost implementation). However, they
    // could not be really constructed, so we static_assert to prevent misuse.

    template <
        class SGU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<SGT, SGU> || (is_const && !other_const), int> = 0>
    constexpr SubEdgeIterator(
        const SubEdgeIterator<SGU, other_const> &other) noexcept
        : Base(other) {
      static_assert(std::is_same_v<SGT, SGU>,
                    "Cannot copy-construct SubEdgeIterator from different "
                    "Subgraph");
    }

    template <
        class SGU, bool other_const,
        std::enable_if_t<
            !std::is_same_v<SGT, SGU> || (is_const && !other_const), int> = 0>
    constexpr SubEdgeIterator &
    operator=(const SubEdgeIterator<SGU, other_const> &other) noexcept {
      static_assert(std::is_same_v<SGT, SGU>,
                    "Cannot copy-assign SubEdgeIterator from different "
                    "Subgraph");

      Base::operator=(other);
      return *this;
    }

  private:
    friend Base;

    friend class boost::iterator_core_access;

    template <class, bool>
    friend class SubEdgeIterator;

    template <class SGU, bool other_const,
              std::enable_if_t<std::is_same_v<typename SGT::graph_type,
                                              typename SGU::graph_type>,
                               int> = 0>
    constexpr bool
    equal(const SubEdgeIterator<SGU, other_const> &other) const noexcept {
      return this->index() == other.index();
    }

    template <class SGU, bool other_const,
              std::enable_if_t<std::is_same_v<typename SGT::graph_type,
                                              typename SGU::graph_type>,
                               int> = 0>
    constexpr difference_type
    distance_to(const SubEdgeIterator<SGU, other_const> &other) const noexcept {
      return other.index() - this->index();
    }

    reference dereference() const noexcept {
      return this->graph().edge(this->index());
    }
  };

  class IndexSet
      : public boost::container::flat_set<int, std::less<>, std::vector<int>> {
  private:
    using Base = boost::container::flat_set<int, std::less<>, std::vector<int>>;

  public:
    using Base::Base;

    explicit IndexSet(std::vector<int> &&vec) noexcept {
      adopt_sequence(std::move(vec));
    }

    IndexSet(boost::container::ordered_unique_range_t tag,
             std::vector<int> &&vec) noexcept {
      adopt_sequence(tag, std::move(vec));
    }

    template <class UnaryPred>
    void erase_if(UnaryPred &&pred) {
      std::vector<int> work = extract_sequence();
      nuri::erase_if(work, std::forward<UnaryPred>(pred));
      adopt_sequence(boost::container::ordered_unique_range, std::move(work));
    }

    void union_with(const IndexSet &other) {
      std::vector<int> result;
      result.reserve(size() + other.size());
      absl::c_set_union(*this, other, std::back_inserter(result));
      adopt_sequence(boost::container::ordered_unique_range, std::move(result));
    }

    void difference(const IndexSet &other) {
      std::vector<int> result;
      result.reserve(size());
      absl::c_set_difference(*this, other, std::back_inserter(result));
      adopt_sequence(boost::container::ordered_unique_range, std::move(result));
    }

    int operator[](int idx) const { return sequence()[idx]; }

    int find_index(int id) const {
      auto it = find(id);
      return static_cast<int>(it - begin());
    }

    void remap(const std::vector<int> &old_to_new);
  };

  template <class GT>
  IndexSet find_nodes(GT &parent, const IndexSet &edges) {
    std::vector<int> nodes;
    nodes.reserve(static_cast<size_t>(edges.size()) * 2);

    for (auto i: edges) {
      nodes.push_back(parent.edge(i).src().id());
      nodes.push_back(parent.edge(i).dst().id());
    }

    return IndexSet(std::move(nodes));
  }

  template <class GT>
  IndexSet find_edges(GT &parent, const IndexSet &nodes) {
    std::vector<int> edges;
    std::stack<int, std::vector<int>> stack;
    ArrayXb visited = ArrayXb::Zero(static_cast<int>(nodes.size()));

    for (int i = 0; i < nodes.size(); ++i) {
      if (visited[i])
        continue;

      visited[i] = true;
      stack.push(i);

      do {
        int u = nodes[stack.top()];
        stack.pop();

        for (auto nei: parent[u]) {
          int v = nei.dst().id();
          int j = nodes.find_index(v);
          if (j >= nodes.size())
            continue;

          if (u < v)
            edges.push_back(nei.eid());

          if (!visited[j]) {
            visited[j] = true;
            stack.push(j);
          }
        }
      } while (!stack.empty());
    }

    return IndexSet(std::move(edges));
  }
}  // namespace internal

/**
 * @brief A subgraph of a graph.
 *
 * @tparam NT node data type
 * @tparam ET edge data type
 * @tparam is_const whether the subgraph is const. This is to support
 *         creating subgraphs of const graphs.
 * @note Removing nodes/edges on the parent graph will invalidate the
 *       subgraph.
 *
 * The subgraph is a non-owning view of a graph. A subgraph could be
 * constructed by selecting nodes of the parent graph. The resulting
 * subgraph will contain only the nodes and edges that are incident to the
 * selected nodes.
 *
 * The subgraph has very similar API to the graph. Here, we note major
 * differences:
 *   - Adding/removing nodes/edges will just mark the nodes/edges as
 *     selected/unselected.
 *   - Some methods have different time complexity requirements. Refer to
 *     the documentation of each method for details.
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

  using edge_iterator = internal::SubEdgeIterator<Subgraph, is_const>;
  using const_edge_iterator = internal::SubEdgeIterator<Subgraph, true>;
  using EdgeRef = internal::SubEdgeWrapper<Subgraph, is_const>;
  using ConstEdgeRef = internal::SubEdgeWrapper<Subgraph, true>;
  using EdgesWrapper = internal::EdgesWrapper<Subgraph, is_const>;
  using ConstEdgesWrapper = internal::EdgesWrapper<Subgraph, true>;

  using adjacency_iterator = internal::SubAdjIterator<Subgraph, is_const>;
  using const_adjacency_iterator = internal::SubAdjIterator<Subgraph, true>;
  using AdjRef = internal::SubAdjWrapper<Subgraph, is_const>;
  using ConstAdjRef = internal::SubAdjWrapper<Subgraph, true>;

  static_assert(std::is_same_v<typename iterator::reference, NodeRef>);
  static_assert(
      std::is_same_v<typename const_iterator::reference, ConstNodeRef>);
  static_assert(std::is_same_v<typename edge_iterator::reference, EdgeRef>);
  static_assert(
      std::is_same_v<typename const_edge_iterator::reference, ConstEdgeRef>);
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
   * @brief Construct a subgraph with the given nodes and edges
   * @param graph The parent graph
   * @param nodes The set of nodes
   * @param edges The set of edges
   * @return The created subgraph
   * @note The nodes connected to the edges will be automatically added even if
   *       they are not in the nodes set.
   * @note Time complexity:
   *         1. Building node set: \f$O(V' \log V')\f$
   *         2. Building edge set: \f$O(E' \log E')\f$
   *         3. Finding nodes connected to edges: \f$O(E')\f$
   * @note The behavior is undefined if any of the node or edge ids are not in
   *       the parent graph.
   */
  static Subgraph from_indices(parent_type &graph, internal::IndexSet &&nodes,
                               internal::IndexSet &&edges) {
    Subgraph subgraph(graph, std::move(nodes), {});
    subgraph.add_edges(edges);
    return subgraph;
  }

  /**
   * @brief Construct a subgraph with the given nodes and all edges connecting
   *        the nodes.
   * @param graph The parent graph
   * @param nodes The set of nodes
   * @return The created subgraph
   * @note Time complexity:
   *         1. Building node set: \f$O(V' \log V')\f$
   *         2. Finding edges connecting the nodes: \f$O(V' E/V \log V')\f$
   *         3. Building edge set: \f$O(E' \log E')\f$
   * @note The behavior is undefined if any of the node ids are not in the
   *       parent graph.
   */
  static Subgraph from_nodes(parent_type &graph, internal::IndexSet &&nodes) {
    internal::IndexSet edges = internal::find_edges(graph, nodes);
    Subgraph subgraph(graph, std::move(nodes), std::move(edges));
    return subgraph;
  }

  /**
   * @brief Construct a subgraph with the given edges and all nodes connected
   *        to the edges.
   * @param graph The parent graph
   * @param edges The set of edges
   * @return The created subgraph
   * @note Time complexity: \f$O(E' \log E')\f$
   * @note The behavior is undefined if any of the edge ids are not in the
   *       parent graph.
   */
  static Subgraph from_edges(parent_type &graph, internal::IndexSet &&edges) {
    internal::IndexSet nodes = internal::find_nodes(graph, edges);
    Subgraph subgraph(graph, std::move(nodes), std::move(edges));
    return subgraph;
  }

  /**
   * @brief Converting constructor to allow implicit conversion from a
   * non-const subgraph to a const subgraph.
   *
   * @param other The non-const subgraph
   */
  template <bool other_const,
            std::enable_if_t<is_const && !other_const, int> = 0>
  Subgraph(const Other<other_const> &other)
      : parent_(other.parent_), nodes_(other.nodes_), edges_(other.edges_) { }

  /**
   * @brief Converting constructor to allow implicit conversion from a
   * non-const subgraph to a const subgraph.
   *
   * @param other The non-const subgraph
   */
  template <bool other_const,
            std::enable_if_t<is_const && !other_const, int> = 0>
  Subgraph(Other<other_const> &&other) noexcept
      : parent_(other.parent_), nodes_(std::move(other.nodes_)),
        edges_(std::move(other.edges_)) { }

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
    edges_ = other.edges_;
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
    edges_ = std::move(other.edges_);
    return *this;
  }

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
  int num_nodes() const { return static_cast<int>(nodes_.size()); }

  /**
   * @brief Count number of edges in the subgraph
   *
   * @return The number of edges in the subgraph
   */
  int num_edges() const { return static_cast<int>(edges_.size()); }

  /**
   * @brief Change the set of nodes and edges in the subgraph
   *
   * @param nodes The new set of nodes.
   * @param edges The new set of edges.
   * @note The nodes connected to the edges will be automatically added even if
   *       they are not in the nodes set.
   * @note Time complexity: same as from_indices().
   * @note If any of the node ids are not in the parent graph, the behavior is
   *       undefined.
   */
  void update(internal::IndexSet &&nodes, internal::IndexSet &&edges) {
    nodes_ = std::move(nodes);

    edges_.clear();
    add_edges(edges);
  }

  /**
   * @brief Change the set of nodes in the subgraph
   *
   * @param nodes The new set of nodes. Edges will be automatically updated.
   * @note Time complexity: same as from_nodes().
   * @note If any of the node ids are not in the parent graph, the behavior is
   *       undefined.
   */
  void update_nodes(internal::IndexSet &&nodes) noexcept {
    nodes_ = std::move(nodes);
    edges_ = internal::find_edges(*parent_, nodes_);
  }

  /**
   * @brief Change the set of edges in the subgraph
   *
   * @param edges The new set of edges. All nodes connected to the edges will be
   *        automatically added.
   * @note Time complexity: same as from_edges().
   * @note If any of the edge ids are not in the parent graph, the behavior is
   *       undefined.
   */
  void update_edges(internal::IndexSet &&edges) noexcept {
    edges_ = std::move(edges);
    nodes_ = internal::find_nodes(*parent_, edges_);
  }

  /**
   * @brief Make this graph an induced subgraph of the parent graph
   *
   * Replace the current set of edges with the set of edges connecting the
   * current set of nodes.
   *
   * @note Time complexity: same as from_edges().
   */
  void refresh_edges() { edges_ = internal::find_edges(*parent_, nodes_); }

  /**
   * @brief Clear the subgraph
   *
   * @note This does not affect the parent graph.
   * @note There is no clear_nodes() member function because it is equivalent to
   *       this function.
   */
  void clear() noexcept {
    nodes_.clear();
    edges_.clear();
  }

  /**
   * @brief Clear the edges
   *
   * @note This does not affect the parent graph.
   */
  void clear_edges() noexcept { edges_.clear(); }

  /**
   * @brief Reserve space for a number of nodes
   *
   * @param num_nodes The number of nodes to reserve space for
   */
  void reserve_nodes(int num_nodes) { nodes_.reserve(num_nodes); }

  /**
   * @brief Reserve space for a number of edges
   *
   * @param num_edges The number of edges to reserve space for
   */
  void reserve_edges(int num_edges) { edges_.reserve(num_edges); }

  /**
   * @brief Add a node to the subgraph
   *
   * @param id The id of the node to add
   * @note Time complexity: \f$O(V')\f$ when the node is not in the graph,
   *       \f$O(\log V')\f$ otherwise.
   * @note If the node is already in the subgraph, this is a no-op. If the
   *       node id is out of range, the behavior is undefined.
   */
  void add_node(int id) { nodes_.insert(id); }

  /**
   * @brief Add an edge to the subgraph. Also adds the incident nodes.
   *
   * @param id The id of the edge to add
   * @note Time complexity: \f$O(E' + V')\f$ when the edge is not in the graph,
   *       \f$O(\log E')\f$ otherwise.
   * @note If the edge is already in the subgraph, this is a no-op. If the
   *       edge id is out of range, the behavior is undefined.
   */
  void add_edge(int id) {
    auto [_, added] = edges_.insert(id);
    if (added) {
      auto edge = parent_->edge(id);
      nodes_.insert(edge.src().id());
      nodes_.insert(edge.dst().id());
    }
  }

  /**
   * @brief Add nodes to the subgraph
   *
   * @param nodes The new set of nodes
   * @note Time complexity: \f$O(V')\f$.
   * @note If any of the node id is out of range, the behavior is undefined.
   */
  void add_nodes(const internal::IndexSet &nodes) { nodes_.union_with(nodes); }

  /**
   * @brief Add edges to the subgraph. Also adds the incident nodes.
   *
   * @param edges The new set of edges
   * @note Time complexity: \f$O(V' + E' + N \log N)\f$, where \f$N\f$ is the
   *       number of new edges.
   * @note If any of the edge id is out of range, the behavior is undefined.
   */
  void add_edges(const internal::IndexSet &edges) {
    edges_.union_with(edges);

    auto new_nodes = internal::find_nodes(*parent_, edges);
    nodes_.union_with(new_nodes);
  }

  /**
   * @brief Add nodes to the subgraph, and update the edges
   *
   * @param nodes The new set of nodes
   * @note Time complexity: same as from_nodes().
   * @note If any of the node id is out of range, the behavior is undefined.
   */
  void add_nodes_with_edges(const internal::IndexSet &nodes) {
    add_nodes(nodes);
    auto new_edges = internal::find_edges(*parent_, nodes);
    edges_.union_with(new_edges);
  }

  /**
   * @brief Check if a node is in the subgraph
   *
   * @param id the id of the node to check
   * @return true if the node is in the subgraph, false otherwise
   * @note Time complexity: \f$O(\log V')\f$.
   */
  bool contains_node(int id) const { return nodes_.contains(id); }

  /**
   * @brief Check if a node is in the subgraph
   *
   * @param node The node to check
   * @return true if the node is in the subgraph, false otherwise
   * @note This is equivalent to calling contains_node(node.id()).
   * @note Time complexity: \f$O(\log V')\f$.
   */
  bool contains_node(typename graph_type::ConstNodeRef node) const {
    return contains_node(node.id());
  }

  /**
   * @brief Check if an edge is in the subgraph
   *
   * @param id the id of the edge to check
   * @return true if the edge is in the subgraph, false otherwise
   * @note Time complexity: \f$O(\log E')\f$.
   */
  bool contains_edge(int id) const { return edges_.contains(id); }

  /**
   * @brief Check if an edge is in the subgraph
   *
   * @param edge The edge to check
   * @return true if the edge is in the subgraph, false otherwise
   * @note This is equivalent to calling contains_edge(edge.id()).
   * @note Time complexity: \f$O(\log E')\f$.
   */
  bool contains_edge(typename graph_type::ConstEdgeRef edge) const {
    return contains_edge(edge.id());
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
  NodeRef node(int idx) { return { idx, *this }; }

  /**
   * @brief Get a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A const-reference wrapper to the node
   */
  ConstNodeRef node(int idx) const { return { idx, *this }; }

  /**
   * @brief Get an edge in the subgraph
   *
   * @param idx The index of the edge to get
   * @return A reference wrapper to the edge (might be const)
   *
   * @note Time complexity: \f$O(\log V')\f$.
   */
  EdgeRef edge(int idx) {
    auto pedge = parent_edge(idx);
    int src = nodes_.find_index(pedge.src().id()),
        dst = nodes_.find_index(pedge.dst().id());
    return { src, dst, idx, *this };
  }

  /**
   * @brief Get an edge in the subgraph
   *
   * @param idx The index of the edge to get
   * @return A const-reference wrapper to the edge
   *
   * @note Time complexity: \f$O(\log V')\f$.
   */
  ConstEdgeRef edge(int idx) const {
    auto pedge = parent_edge(idx);
    int src = nodes_.find_index(pedge.src().id()),
        dst = nodes_.find_index(pedge.dst().id());
    return { src, dst, idx, *this };
  }

  /**
   * @brief Get data of a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A reference to the node data (might be const)
   */
  internal::const_if_t<is_const, NT> &node_data(int idx) {
    return parent_->node_data(nodes_[idx]);
  }

  /**
   * @brief Get data of a node in the subgraph
   *
   * @param idx The index of the node to get
   * @return A const-reference to the node data
   */
  const NT &node_data(int idx) const { return parent_->node_data(nodes_[idx]); }

  /**
   * @brief Get data of an edge in the subgraph
   *
   * @param idx The index of the edge to get
   * @return A reference to the edge data (might be const)
   */
  internal::const_if_t<is_const, ET> &edge_data(int idx) {
    return parent_->edge_data(edges_[idx]);
  }

  /**
   * @brief Get data of an edge in the subgraph
   *
   * @param idx The index of the edge to get
   * @return A const-reference to the edge data
   */
  const ET &edge_data(int idx) const { return parent_->edge_data(edges_[idx]); }

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
   * @brief Get a parent edge of an edge in the subgraph
   *
   * @param idx The index of the edge to get
   * @return A reference wrapper to the edge (might be const)
   */
  std::conditional_t<is_const, typename parent_type::ConstEdgeRef,
                     typename parent_type::EdgeRef>
  parent_edge(int idx) {
    return parent_->edge(edges_[idx]);
  }

  /**
   * @brief Get a parent edge of an edge in the subgraph
   *
   * @param idx The index of the edge to get
   * @return A const-reference wrapper to the edge.
   */
  typename parent_type::ConstEdgeRef parent_edge(int idx) const {
    return parent_->edge(edges_[idx]);
  }

  /**
   * @brief Find a node with the given id.
   *
   * @param id The id of the node to find
   * @return An iterator to the node if found, end() otherwise.
   * @note Time complexity: \f$O(\log V')\f$.
   */
  iterator find_node(int id) { return begin() + nodes_.find_index(id); }

  /**
   * @brief Find a node with the given id.
   *
   * @param node The node to find
   * @return An iterator to the node if found, end() otherwise.
   * @note This is equivalent to calling find_node(node.id()).
   * @note Time complexity: same as find_node(int).
   */
  iterator find_node(typename graph_type::ConstNodeRef node) {
    return find_node(node.id());
  }

  /**
   * @brief Find a node with the given id.
   *
   * @param id The id of the node to find
   * @return A const_iterator to the node if found, end() otherwise
   * @note Time complexity: same as find_node(int).
   */
  const_iterator find_node(int id) const {
    return begin() + nodes_.find_index(id);
  }

  /**
   * @brief Find a node with the given id.
   *
   * @param node The node to find
   * @return A const_iterator to the node if found, end() otherwise.
   * @note This is equivalent to calling find_node(node.id()).
   * @note Time complexity: same as find_node(int).
   */
  const_iterator find_node(typename graph_type::ConstNodeRef node) const {
    return find_node(node.id());
  }

  /**
   * @brief Find an edge with the given id.
   *
   * @param id The id of the edge to find
   * @return An iterator to the edge if found, edge_end() otherwise.
   * @note Time complexity: \f$O(\log E')\f$.
   */
  edge_iterator find_edge(int id) {
    return edge_begin() + edges_.find_index(id);
  }

  /**
   * @brief Find an edge with the given id.
   *
   * @param edge The edge to find
   * @return An iterator to the edge if found, edge_end() otherwise.
   * @note This is equivalent to calling find_edge(edge.id()).
   * @note Time complexity: same as find_edge(int).
   */
  edge_iterator find_edge(typename graph_type::ConstEdgeRef edge) {
    return find_edge(edge.id());
  }

  /**
   * @brief Find an edge with the given id.
   *
   * @param id The id of the edge to find
   * @return A const_iterator to the edge if found, edge_end() otherwise
   * @note Time complexity: same as find_edge(int).
   */
  const_edge_iterator find_edge(int id) const {
    return edge_begin() + edges_.find_index(id);
  }

  /**
   * @brief Find an edge with the given id.
   *
   * @param edge The edge to find
   * @return A const_iterator to the edge if found, edge_end() otherwise.
   * @note This is equivalent to calling find_edge(edge.id()).
   * @note Time complexity: same as find_edge(int).
   */
  const_edge_iterator find_edge(typename graph_type::ConstEdgeRef edge) const {
    return find_edge(edge.id());
  }

  /**
   * @brief Erase a node from the subgraph. Corresponding edges will also be
   *        removed.
   *
   * @param idx The index of the node to erase.
   * @note The behavior is undefined if the index is out of range.
   * @note Time complexity: \f$O(E/V (\log V' + \log E') + V' + E')\f$.
   */
  void erase_node(int idx) {
    std::vector<int> erased_edges;
    for (auto nei: node(idx))
      erased_edges.push_back(nei.as_parent().eid());
    nodes_.erase(nodes_.begin() + idx);
    edges_.difference(internal::IndexSet(std::move(erased_edges)));
  }

  /**
   * @brief Erase a node from the subgraph
   *
   * @param node The node to erase
   * @note This is equivalent to calling erase_node(node.id()).
   * @note Time complexity: same as erase_node(int).
   */
  void erase_node(ConstNodeRef node) { erase_node(node.id()); }

  /**
   * @brief Erase an edge from the subgraph
   *
   * @param idx The index of the edge to erase
   * @note The behavior is undefined if the index is out of range.
   * @note Time complexity: \f$O(E')\f$ in worst case.
   */
  void erase_edge(int idx) { edges_.erase(edges_.begin() + idx); }

  /**
   * @brief Erase an edge from the subgraph
   *
   * @param edge The edge to erase
   * @note This is equivalent to calling erase_edge(edge.id()).
   * @note Time complexity: same as erase_edge(int).
   */
  void erase_edge(ConstEdgeRef edge) { erase_edge(edge.id()); }

  /**
   * @brief Erase range of nodes from the subgraph. Corresponding edges will
   *        also be removed.
   *
   * @param begin Iterator pointing to the first node to erase
   * @param end Iterator pointing to the node after the last node to erase
   * @note Time complexity: \f$O(N E/V (\log V' + \log E') + V' + E')\f$, where
   *       \f$N\f$ is the distance between begin and end.
   */
  void erase_nodes(const_iterator begin, const_iterator end) {
    std::vector<int> erased_edges;
    for (auto it = begin; it != end; ++it)
      for (auto nei: *it)
        erased_edges.push_back(nei.as_parent().eid());

    nodes_.erase(begin - this->begin() + nodes_.begin(),
                 end - this->begin() + nodes_.begin());
    edges_.difference(internal::IndexSet(std::move(erased_edges)));
  }

  /**
   * @brief Erase range of edges from the subgraph
   *
   * @param begin Iterator pointing to the first edge to erase
   * @param end Iterator pointing to the edge after the last edge to erase
   * @note Time complexity: \f$O(E')\f$ in worst case.
   */
  void erase_edges(const_edge_iterator begin, const_edge_iterator end) {
    edges_.erase(begin - this->edge_begin() + edges_.begin(),
                 end - this->edge_begin() + edges_.begin());
  }

  /**
   * @brief Erase a node with given id from the subgraph. Corresponding edges
   *        will also be removed.
   *
   * @param id The id of the node to erase
   * @note This is a no-op if the node is not in the subgraph.
   * @note Time complexity: same as erase_node().
   */
  void erase_node_of(int id) {
    int idx = nodes_.find_index(id);
    if (idx < num_nodes())
      erase_node(idx);
  }

  /**
   * @brief Erase a node with given id from the subgraph. Corresponding edges
   *        will also be removed.
   *
   * @param node The parent node to erase
   * @note This is a no-op if the node is not in the subgraph.
   * @note Time complexity: same as erase_node().
   */
  void erase_node_of(typename graph_type::ConstNodeRef node) {
    erase_node_of(node.id());
  }

  /**
   * @brief Erase an edge with given id from the subgraph
   *
   * @param id The id of the edge to erase
   * @note This is a no-op if the edge is not in the subgraph.
   * @note Time complexity: same as erase_edge().
   */
  void erase_edge_of(int id) { edges_.erase(id); }

  /**
   * @brief Erase an edge with given id from the subgraph
   *
   * @param edge The parent edge to erase
   * @note This is a no-op if the edge is not in the subgraph.
   * @note Time complexity: same as erase_edge().
   */
  void erase_edge_of(typename graph_type::ConstEdgeRef edge) {
    erase_edge_of(edge.id());
  }

  /**
   * @brief Erase matching nodes from the subgraph. Corresponding edges will
   *        also be removed.
   *
   * @tparam UnaryPred Type of the unary predicate
   * @param pred Unary predicate that returns true for nodes to erase
   * @note Time complexity: same as erase_nodes().
   */
  template <class UnaryPred>
  void erase_nodes_if(UnaryPred pred) {
    std::vector<int> erased_nodes, erased_edges;

    for (int i = 0; i < num_nodes(); ++i) {
      if (!pred(nodes_[i]))
        continue;

      erased_nodes.push_back(nodes_[i]);
      for (auto nei: node(i))
        erased_edges.push_back(nei.as_parent().eid());
    }

    nodes_.difference(internal::IndexSet(std::move(erased_nodes)));
    edges_.difference(internal::IndexSet(std::move(erased_edges)));
  }

  /**
   * @brief Erase matching edges from the subgraph
   *
   * @tparam UnaryPred Type of the unary predicate
   * @param pred Unary predicate that returns true for edges to erase
   * @note Time complexity: \f$O(E')\f$ in worst case.
   */
  template <class UnaryPred>
  void erase_edges_if(UnaryPred &&pred) {
    edges_.erase_if(std::forward<UnaryPred>(pred));
  }

  /**
   * @brief Re-map node/edge ids
   *
   * @param node_map A vector that maps old node ids to new node ids, so that
   *        old_to_new[old_id] = new_id. If old_to_new[old_id] < 0, then the
   *        node is removed from the subgraph.
   * @param edge_map A vector that maps old edge ids to new edge ids, so that
   *        old_to_new[old_id] = new_id. If old_to_new[old_id] < 0, then the
   *        edge is removed from the subgraph.
   * @note Time complexity: \f$O(V' + E')\f$.
   */
  void remap(const std::vector<int> &node_map,
             const std::vector<int> &edge_map) {
    nodes_.remap(node_map);
    edges_.remap(edge_map);
  }

  /**
   * @brief Re-map node ids
   *
   * @param node_map A vector that maps old node ids to new node ids, so that
   *        old_to_new[old_id] = new_id. If old_to_new[old_id] < 0, then the
   *        node is removed from the subgraph.
   * @note The caller is responsible for ensuring that the new node ids are
   *       compatible with the selected edges; otherwise, the behavior is
   *       undefined.
   * @note Time complexity: \f$O(V')\f$.
   */
  void remap_nodes(const std::vector<int> &node_map) { nodes_.remap(node_map); }

  /**
   * @brief Re-map edge ids
   *
   * @param old_to_new A vector that maps old edge ids to new edge ids, so that
   *        old_to_new[old_id] = new_id. If old_to_new[old_id] < 0, then the
   *        edge is removed from the subgraph.
   * @note Time complexity: \f$O(E')\f$.
   */
  void remap_edges(const std::vector<int> &old_to_new) {
    edges_.remap(old_to_new);
  }

  iterator begin() { return { *this, 0 }; }
  iterator end() { return { *this, num_nodes() }; }

  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

  const_iterator cbegin() const { return { *this, 0 }; }
  const_iterator cend() const { return { *this, num_nodes() }; }

  node_iterator node_begin() { return { *this, 0 }; }
  node_iterator node_end() { return { *this, num_nodes() }; }

  const_node_iterator node_begin() const { return node_cbegin(); }
  const_node_iterator node_end() const { return node_cend(); }

  const_node_iterator node_cbegin() const { return { *this, 0 }; }
  const_node_iterator node_cend() const { return { *this, num_nodes() }; }

  EdgesWrapper edges() { return { *this }; }

  ConstEdgesWrapper edges() const { return { *this }; }

  edge_iterator edge_begin() { return { *this, 0 }; }
  edge_iterator edge_end() { return { *this, num_edges() }; }

  const_edge_iterator edge_begin() const { return edge_cbegin(); }
  const_edge_iterator edge_end() const { return edge_cend(); }

  const_edge_iterator edge_cbegin() const { return { *this, 0 }; }
  const_edge_iterator edge_cend() const { return { *this, num_edges() }; }

  /**
   * @brief Get all node ids in the subgraph
   *
   * @return The node ids in the subgraph, in an unspecified order
   */
  const std::vector<int> &node_ids() const { return nodes_.sequence(); }

  /**
   * @brief Get all edge ids in the subgraph
   *
   * @return The edge ids in the subgraph, in an unspecified order
   */
  const std::vector<int> &edge_ids() const { return edges_.sequence(); }

  /**
   * @brief Count in-subgraph neighbors of a node
   *
   * @param idx The index of the node
   * @return The number of neighbors of the node that are in the subgraph
   * @note The behavior is undefined if the node is not in the subgraph.
   * @note Time complexity: \f$O(E/V (\log V' + \log E'))\f$.
   */
  int degree(int idx) const {
    return std::distance(adj_cbegin(idx), adj_cend(idx));
  }

  /**
   * @brief Find edge between two nodes
   *
   * @param src The source atom
   * @param dst The destination atom
   * @return An iterator to the edge if found, edge_end() otherwise.
   * @note This will only find edges that are in the subgraph.
   * @note Time complexity: \f$O(E/V + \log E')\f$.
   */
  edge_iterator find_edge(ConstNodeRef src, ConstNodeRef dst) {
    return find_edge(src.id(), dst.id());
  }

  /**
   * @brief Find edge between two nodes
   *
   * @param src The source atom
   * @param dst The destination atom
   * @return An iterator to the edge if found, edge_end() otherwise.
   * @note This will only find edges that are in the subgraph.
   * @note Time complexity: \f$O(E/V + \log E')\f$.
   */
  const_edge_iterator find_edge(ConstNodeRef src, ConstNodeRef dst) const {
    return find_edge(src.id(), dst.id());
  }

  /**
   * @brief Find edge between two nodes
   *
   * @param src The parent node of one node
   * @param dst The parent node of the other node
   * @return An iterator to the edge if found, edge_end() otherwise.
   * @note This will only find edges that are in the subgraph. If any of the
   *       nodes are not in the subgraph, returns edge_end().
   * @note Time complexity: \f$O(\log V' + E/V + \log E')\f$.
   */
  edge_iterator find_edge(typename graph_type::ConstNodeRef src,
                          typename graph_type::ConstNodeRef dst) {
    auto sit = find_node(src), dit = find_node(dst);
    if (sit == end() || dit == end())
      return edge_end();

    return find_edge(*sit, *dit);
  }

  /**
   * @brief Find edge between two nodes
   *
   * @param src The index of one node
   * @param dst The index of the other node
   * @return An iterator to the edge if found, edge_end() otherwise.
   * @note This will only find edges that are in the subgraph. If any of the
   *       nodes are not in the subgraph, returns edge_end().
   * @note Time complexity: \f$O(\log V' + E/V + \log E')\f$.
   */
  const_edge_iterator find_edge(typename graph_type::ConstNodeRef src,
                                typename graph_type::ConstNodeRef dst) const {
    auto sit = find_node(src), dit = find_node(dst);
    if (sit == end() || dit == end())
      return edge_end();

    return find_edge(*sit, *dit);
  }

  /**
   * @brief Find adjacent node of a node
   *
   * @param src The source atom
   * @param dst The destination atom
   * @return An iterator to the adjacent node if found, adj_end(src) otherwise.
   * @note This will only find edges that are in the subgraph.
   * @note Time complexity: same as find_edge().
   */
  adjacency_iterator find_adjacent(ConstNodeRef src, ConstNodeRef dst) {
    return find_adjacent(src.id(), dst.id());
  }

  /**
   * @brief Find adjacent node of a node
   *
   * @param src The source atom
   * @param dst The destination atom
   * @return A const-iterator to the adjacent node if found, adj_end(src)
   *         otherwise.
   * @note This will only find edges that are in the subgraph.
   * @note Time complexity: same as find_edge().
   */
  const_adjacency_iterator find_adjacent(ConstNodeRef src,
                                         ConstNodeRef dst) const {
    return find_adjacent(src.id(), dst.id());
  }

  adjacency_iterator adj_begin(int idx) { return { *this, idx, 0 }; }
  adjacency_iterator adj_end(int idx) {
    return { *this, idx, parent_node(idx).degree() };
  }

  const_adjacency_iterator adj_begin(int idx) const { return adj_cbegin(idx); }
  const_adjacency_iterator adj_end(int idx) const { return adj_cend(idx); }

  const_adjacency_iterator adj_cbegin(int idx) const {
    return { *this, idx, 0 };
  }
  const_adjacency_iterator adj_cend(int idx) const {
    return { *this, idx, parent_node(idx).degree() };
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
  friend class internal::SubAdjWrapper;
  template <class, bool>
  friend class internal::SubEdgeWrapper;

  template <class, bool>
  friend class internal::SubEdgesFinder;

  Subgraph(parent_type &graph, internal::IndexSet &&nodes,
           internal::IndexSet &&edges) noexcept
      : parent_(&graph), nodes_(std::move(nodes)), edges_(std::move(edges)) { }

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

  edge_iterator find_edge(int src, int dst) {
    return find_edge_helper(*this, src, dst);
  }

  const_edge_iterator find_edge(int src, int dst) const {
    return find_edge_helper(*this, src, dst);
  }

  adjacency_iterator find_adjacent(int src, int dst) {
    return find_adj_helper<adjacency_iterator>(*this, src, dst);
  }

  const_adjacency_iterator find_adjacent(int src, int dst) const {
    return find_adj_helper<const_adjacency_iterator>(*this, src, dst);
  }

  template <class SGT>
  static auto find_edge_helper(SGT &graph, int src, int dst) {
    if (graph.parent_node(src).degree() > graph.parent_node(dst).degree())
      std::swap(src, dst);

    auto ait = graph.find_adjacent(src, dst);
    if (ait.end())
      return graph.edge_end();

    return graph.edge_begin() + ait->eid();
  }

  template <class AIT, class SGT>
  static AIT find_adj_helper(SGT &graph, int src, int dst) {
    auto pait = graph.parent_node(src).find_adjacent(graph.parent_node(dst));
    if (pait.end())
      return graph.adj_end(src);

    auto eit = graph.find_edge(pait->eid());
    if (eit == graph.edge_end())
      return graph.adj_end(src);

    return { graph, src, dst, eit->id(),
             pait - graph.parent_node(src).begin() };
  }

  parent_type *parent_;
  internal::IndexSet nodes_;
  internal::IndexSet edges_;
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

namespace internal {
  /* Helper templates */

  template <class T>
  struct SubgraphTypeHelper;

  template <class T>
  struct SubgraphTypeHelper<T &&> { };

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
 * @brief Make an empty subgraph from a graph.
 * @tparam GT The type of the graph
 * @param graph The graph to make a subgraph of
 * @return A subgraph of the graph.
 */
template <class GT>
SubgraphOf<GT> make_subgraph(GT &&graph) {
  return { std::forward<GT>(graph) };
}

/**
 * @brief Make a subgraph from list of nodes and edges.
 * @tparam GT The type of the graph
 * @param graph The graph to make a subgraph of
 * @param nodes The nodes to include in the subgraph
 * @param edges The edges to include in the subgraph
 * @return A subgraph of the graph. All nodes connected by the edges will be
 *         included in the subgraph even if they are not explicitly listed in
 *         \p nodes.
 *
 * @note Time complexity: same as Subgraph::from_indices().
 */
template <class GT>
SubgraphOf<GT> make_subgraph(GT &&graph, internal::IndexSet &&nodes,
                             internal::IndexSet &&edges) {
  return SubgraphOf<GT>::from_indices(std::forward<GT>(graph), std::move(nodes),
                                      std::move(edges));
}

/**
 * @brief Make a subgraph from list of nodes.
 * @tparam GT The type of the graph
 * @param graph The graph to make a subgraph of
 * @param nodes The nodes to include in the subgraph
 * @return A subgraph of the graph. All edges between the nodes will be included
 *         in the subgraph.
 *
 * @note Time complexity: same as Subgraph::from_nodes().
 */
template <class GT>
SubgraphOf<GT> subgraph_from_nodes(GT &&graph, internal::IndexSet &&nodes) {
  return SubgraphOf<GT>::from_nodes(std::forward<GT>(graph), std::move(nodes));
}

/**
 * @brief Make a subgraph from list of edges.
 * @tparam GT The type of the graph
 * @param graph The graph to make a subgraph of
 * @param edges The edges to include in the subgraph
 * @return A subgraph of the graph. All nodes connected by the edges will be
 *         included in the subgraph.
 *
 * @note Time complexity: same as Subgraph::from_edges().
 */
template <class GT>
SubgraphOf<GT> subgraph_from_edges(GT &&graph, internal::IndexSet &&edges) {
  return SubgraphOf<GT>::from_edges(std::forward<GT>(graph), std::move(edges));
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

#endif /* NURI_CORE_GRAPH_H_ */

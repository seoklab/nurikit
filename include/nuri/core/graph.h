//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_H_
#define NURI_CORE_GRAPH_H_

#include <algorithm>
#include <iterator>
#include <list>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/flat_hash_set.h>
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
    constexpr ArrowHelper(Parent &&p) noexcept: p_(std::move(p)) { }

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
    using reference = value_type;
    using iterator_category = std::random_access_iterator_tag;

    constexpr DataIteratorBase(parent_type *graph,
                               difference_type index) noexcept
      : graph_(graph), index_(index) { }

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

    constexpr reference operator*() const noexcept {
      return derived()->deref(graph_, index_);
    }

    constexpr reference operator[](difference_type n) const noexcept {
      return *(*derived() + n);
    }

    constexpr pointer operator->() const noexcept { return { **derived() }; }

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

    constexpr parent_type *graph() const noexcept {
      return const_cast<parent_type *>(graph_);
    }

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
  class NodeWrapper;

  template <class GT, bool is_const>
  class AdjWrapper {
  public:
    using DT = typename GT::adj_data_type;
    using edge_id_type = typename GT::edge_id_type;
    using value_type = const_if_t<is_const, DT>;
    using edge_value_type = const_if_t<is_const, typename GT::edge_data_type>;

    using parent_type = const_if_t<is_const, GT>;

    template <bool other_const>
    using Other = AdjWrapper<GT, other_const>;

    constexpr AdjWrapper(GT &graph, int src, DT &adj) noexcept
      : src_(src), adj_(&adj), graph_(&graph) { }

    constexpr AdjWrapper(const GT &graph, int src, const DT &adj) noexcept
      : src_(src), adj_(&adj), graph_(&graph) { }

    template <bool other_const,
              typename = std::enable_if_t<is_const && !other_const>>
    constexpr AdjWrapper(const Other<other_const> &other) noexcept
      : src_(other.src_), adj_(other.adj_), graph_(other.graph_) { }

    constexpr NodeWrapper<GT, is_const> src() const noexcept {
      return graph_->node(src_);
    }

    constexpr NodeWrapper<GT, is_const> dst() const noexcept {
      return graph_->node(adj_->dst);
    }

    constexpr edge_id_type eid() const noexcept { return adj_->eid; }
    constexpr edge_value_type &edge_data() const noexcept {
      return adj_->eid->data;
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    template <class, bool>
    friend class AdjWrapper;
    friend GT;

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

    constexpr bool begin() const noexcept {
      return *this == this->graph()->adj_begin(nid_);
    }

    constexpr bool end() const noexcept {
      return *this == this->graph()->adj_end(nid_);
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

    constexpr int degree() const noexcept { return graph_->degree(nid_); }

    adjacency_iterator begin() const noexcept {
      return graph_->adj_begin(nid_);
    }

    adjacency_iterator end() const noexcept { return graph_->adj_end(nid_); }

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
    using DT = typename GT::edge_data_type;

    using edge_id_type = typename GT::edge_id_type;
    using value_type = const_if_t<is_const, DT>;

    template <bool other_const>
    using Other = EdgeWrapper<GT, other_const>;

    constexpr EdgeWrapper(edge_id_type eid) noexcept: eid_(eid) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    constexpr EdgeWrapper(const Other<other_const> &other) noexcept
      : eid_(other.eid_) { }

    constexpr edge_id_type id() const noexcept { return eid_; }
    constexpr int src() const noexcept { return eid_->src; }
    constexpr int dst() const noexcept { return eid_->dst; }

    constexpr value_type &data() const noexcept {
      return const_cast<value_type &>(eid_->data);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    using stored_edge_id_type = typename GT::stored_edge_id_type;
    using EIT = std::conditional_t<is_const, edge_id_type, stored_edge_id_type>;

    friend GT;
    template <class, bool>
    friend class EdgeWrapper;
    template <class, bool>
    friend class EdgeIterator;

    constexpr EdgeWrapper(stored_edge_id_type eid) noexcept: eid_(eid) { }

    EIT eid_;
  };

  template <class GT, bool is_const>
  class EdgeIterator {
  public:
    using difference_type = int;
    using iterator_category = std::bidirectional_iterator_tag;
    using pointer = ArrowHelper<EdgeWrapper<GT, is_const>>;
    using reference = EdgeWrapper<GT, is_const>;
    using value_type = EdgeWrapper<GT, is_const>;

    using edge_id_type = typename GT::edge_id_type;

    template <bool other_const>
    using Other = EdgeIterator<GT, other_const>;

    EdgeIterator(edge_id_type eid) noexcept: eid_(eid) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    EdgeIterator(const Other<other_const> &other) noexcept: eid_(other.eid_) { }

    template <bool other_const,
              class = std::enable_if_t<is_const && !other_const>>
    EdgeIterator &operator=(const Other<other_const> &other) noexcept {
      eid_ = other.eid_;
      return *this;
    }

    EdgeIterator &operator++() noexcept {
      ++eid_;
      return *this;
    }

    EdgeIterator operator++(int) noexcept {
      EdgeIterator tmp(*this);
      ++eid_;
      return tmp;
    }

    EdgeIterator &operator--() noexcept {
      --eid_;
      return *this;
    }

    EdgeIterator operator--(int) noexcept {
      EdgeIterator tmp(*this);
      --eid_;
      return tmp;
    }

    reference operator*() const noexcept { return { eid_ }; }
    pointer operator->() const noexcept { return { **this }; }

    template <bool other_const>
    bool operator==(const Other<other_const> &other) const noexcept {
      return eid_ == other.eid_;
    }

    template <bool other_const>
    bool operator!=(const Other<other_const> &other) const noexcept {
      return !(*this == other);
    }

  private:
    using stored_edge_id_type = typename GT::stored_edge_id_type;
    using EIT = std::conditional_t<is_const, edge_id_type, stored_edge_id_type>;

    friend GT;
    template <class, bool>
    friend class EdgeIterator;

    EdgeIterator(stored_edge_id_type eid) noexcept: eid_(eid) { }

    EIT eid_;
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

    EdgeIterator<GT, is_const> begin() const { return graph_->edge_begin(); }
    EdgeIterator<GT, is_const> end() const { return graph_->edge_end(); }

    EdgeIterator<GT, true> cbegin() const { return graph_->edge_cbegin(); }
    EdgeIterator<GT, true> cend() const { return graph_->edge_cend(); }

    EdgesWrapper<GT, true> as_const() const { return *this; }

    int size() const { return graph_->num_edges(); }

  private:
    const_if_t<is_const, GT> *graph_;
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
private:
  struct StoredEdge {
    int src;
    int dst;
    ET data;
  };
  using edge_type = StoredEdge;
  using stored_edge_id_type = typename std::list<StoredEdge>::iterator;

  struct AdjEntry {
    int dst;
    stored_edge_id_type eid;
  };
  using adj_data_type = AdjEntry;

  template <class, bool>
  friend class internal::EdgeWrapper;
  template <class, bool>
  friend class internal::EdgeIterator;

  template <class, bool>
  friend class internal::AdjWrapper;
  template <class, bool>
  friend class internal::AdjIterator;

public:
  using node_data_type = NT;
  using edge_data_type = ET;
  using edge_id_type = typename std::list<StoredEdge>::const_iterator;

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
  Graph(const Graph & /* other */);
  Graph(Graph &&) noexcept = default;
  Graph &operator=(const Graph & /* other */);
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
            class = internal::enable_if_compatible_iter_t<Iterator, NT>>
  int add_node(Iterator begin, Iterator end) noexcept {
    auto it = nodes_.insert(nodes_.end(), begin, end);
    adj_list_.resize(num_nodes());
    return static_cast<int>(it - nodes_.begin());
  }

  edge_id_type add_edge(int src, int dst, const ET &data) {
    stored_edge_id_type eid = edges_.insert(edges_.end(), { src, dst, data });
    add_adjacency_entry(eid);
    return eid;
  }

  edge_id_type add_edge(int src, int dst, ET &&data) noexcept {
    stored_edge_id_type eid =
      edges_.insert(edges_.end(), { src, dst, std::move(data) });
    add_adjacency_entry(eid);
    return eid;
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
   * @note Time complexity: \f$O(V)\f$ if only trailing nodes are erased,
   *       \f$O(V+E)\f$ otherwise. If \p begin or \p end is out of range, the
   *       behavior is undefined.
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

  EdgeRef edge(edge_id_type id) { return { promote_const_eid(id) }; }
  ConstEdgeRef edge(edge_id_type id) const { return { id }; }
  void update_edge(edge_id_type id, const ET &data) {
    promote_const_eid(id)->data = data;
  }
  void update_edge(edge_id_type id, ET &&data) noexcept {
    promote_const_eid(id)->data = std::move(data);
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
   * @brief Erase an edge from the graph.
   *
   * @param it The iterator of the edge to be erased.
   * @return The data of the erased edge.
   * @sa erase_edge()
   * @note Time complexity: \f$O(E/V)\f$. If \p it is out of range, the behavior
   *       is undefined.
   */
  ET pop_edge(const_edge_iterator it) {
    ET ret = std::move(it->id()->data);
    erase_edge(it);
    return ret;
  }

  /**
   * @brief Erase an edge from the graph.
   *
   * @param it The iterator of the edge to be erased.
   * @return The edge iterator following the erased edge.
   * @sa pop_edge()
   * @note Time complexity: \f$O(E/V)\f$. If \p id is out of range, the behavior
   *       is undefined.
   */
  edge_iterator erase_edge(const_edge_iterator it);

  /**
   * @brief Erase an edge from the graph between two nodes.
   *
   * @param src The id of the source node.
   * @param dst The id of the destination node.
   * @return Whether the edge is erased.
   * @note Time complexity: \f$O(E/V)\f$. If \p src or \p dst is out of range,
   *       the behavior is undefined. \p src and \p dst is interchangeable.
   */
  bool erase_edge_between(int src, int dst);

  internal::EdgesWrapper<Graph, false> edges() { return { *this }; }
  internal::EdgesWrapper<Graph, true> edges() const { return { *this }; }
  internal::EdgesWrapper<Graph, true> cedges() const { return { *this }; }

  edge_iterator edge_begin() { return { edges_.begin() }; }
  edge_iterator edge_end() { return { edges_.end() }; }
  const_edge_iterator edge_begin() const { return edge_cbegin(); }
  const_edge_iterator edge_end() const { return edge_cend(); }
  const_edge_iterator edge_cbegin() const { return { edges_.cbegin() }; }
  const_edge_iterator edge_cend() const { return { edges_.cend() }; }

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
  template <class GT>
  static internal::AdjIterator<Graph, std::is_const_v<GT>>
  find_adj_helper(GT &graph, int src, int dst) {
    auto ret = graph.adj_begin(src);
    for (; ret != graph.adj_end(src); ++ret) {
      if (ret->dst().id() == dst) {
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
    if (ait == graph.adj_end(src)) {
      return graph.edge_end();
    }
    return { ait->adj_->eid };
  }

  void erase_nodes_common(std::vector<int> &node_keep, int first_erased_id,
                          bool erase_trailing);

  void add_adjacency_entry(stored_edge_id_type eid) {
    adj_list_[eid->src].push_back({ eid->dst, eid });
    adj_list_[eid->dst].push_back({ eid->src, eid });
  }

  edge_id_type erase_adjacent(const int src, const int dst) {
    edge_id_type ret = edge_end()->id();

    auto pred_gen = [](int other) {
      return [=](const AdjEntry &adj) { return adj.dst == other; };
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

  stored_edge_id_type promote_const_eid(edge_id_type eid) {
    return edges_.erase(eid, eid);
  }

  AdjRef adjacent(int nid, int idx) {
    return { *this, nid, adj_list_[nid][idx] };
  }

  ConstAdjRef adjacent(int nid, int idx) const {
    return { *this, nid, adj_list_[nid][idx] };
  }

  std::vector<std::vector<AdjEntry>> adj_list_;
  std::vector<NT> nodes_;
  std::list<StoredEdge> edges_;
};

/* Out-of-line definitions */

template <class NT, class ET>
Graph<NT, ET>::Graph(const Graph &other)
  : adj_list_(other.num_nodes()), nodes_(other.nodes_), edges_(other.edges_) {
  for (auto eit = edges_.begin(); eit != edges_.end(); ++eit) {
    add_adjacency_entry(eit);
  }
}

template <class NT, class ET>
Graph<NT, ET> &Graph<NT, ET>::operator=(const Graph &other) {
  if (this != &other) {
    adj_list_ = std::vector<std::vector<AdjEntry>>(other.num_nodes());
    nodes_ = other.nodes_;
    edges_ = other.edges_;

    for (auto eit = edges_.begin(); eit != edges_.end(); ++eit) {
      add_adjacency_entry(eit);
    }
  }
  return *this;
}

template <class NT, class ET>
template <class Iterator, class>
std::pair<int, std::vector<int>> Graph<NT, ET>::erase_nodes(Iterator begin,
                                                            Iterator end) {
  // Note: the time complexity notations are only for very sparse graphs, i.e.,
  // E = O(V).
  if (begin == end) {
    return { size(), {} };
  }

  // Phase I: mark nodes for removal, O(V)
  std::vector<int> node_keep(num_nodes(), 1);
  int first_erased_id = num_nodes();
  for (auto it = begin; it != end; ++it) {
    int nid = *it;
    node_keep[nid] = 0;
    first_erased_id = std::min(first_erased_id, nid);
    for (AdjEntry &adj: adj_list_[nid]) {
      erase_first(adj_list_[adj.dst],
                  [&](const auto &neigh) { return neigh.dst == nid; });
      edges_.erase(adj.eid);
    }
    adj_list_[nid].clear();
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
  if (begin >= end) {
    return { size(), {} };
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
      if (first_erased_id < 0) {
        first_erased_id = nid;
      }

      node_keep[nid] = 0;
      for (AdjEntry &adj: adj_list_[nid]) {
        erase_first(adj_list_[adj.dst],
                    [&](const auto &neigh) { return neigh.dst == nid; });
        edges_.erase(adj.eid);
      }
      adj_list_[nid].clear();
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
  if (first_erased_id < 0 || first_erased_id >= num_nodes()) {
    return;
  }

  // Phase II: erase unused adjacencies
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
  {
    int i = 0;
    erase_if(nodes_, [&](const NT &) { return node_keep[i++] == 0; });
    i = 0;
    erase_if(adj_list_, [&](const std::vector<AdjEntry> &) {
      return node_keep[i++] == 0;
    });
  }

  // Phase III: update the node numbers in adjacencies and edges, O(V+E)
  mask_to_map(node_keep);

  for (std::vector<AdjEntry> &adjs: adj_list_) {
    for (AdjEntry &adj: adjs) {
      adj.dst = node_keep[adj.dst];
    }
  }

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
typename Graph<NT, ET>::edge_iterator
Graph<NT, ET>::erase_edge(const const_edge_iterator it) {
  auto pred = [&](const auto &adj) { return adj.eid == it->id(); };

  erase_first(adj_list_[it->src()], pred);
  erase_first(adj_list_[it->dst()], pred);
  return edges_.erase(it->id());
}

template <class NT, class ET>
bool Graph<NT, ET>::erase_edge_between(int src, int dst) {
  auto id = erase_adjacent(src, dst);
  if (id != edge_end()->id()) {
    edges_.erase(id);
    return true;
  }
  return false;
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
      if (!inserted) {
        continue;
      }

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
    if (dst != exclude) {
      q.push(dst);
    }
  }

  internal::connected_components_impl(g, visited, q);

  if (visited.contains(exclude)) {
    return {};
  }

  return visited;
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_H_ */

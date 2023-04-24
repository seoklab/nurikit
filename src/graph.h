//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURIKIT_GRAPH_H_
#define NURIKIT_GRAPH_H_

#include <type_traits>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include "utils.h"

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
  struct DataIteratorBase {
  public:
    using parent_type = const_if_t<is_const, GT>;

    using difference_type = int;
    using value_type = DT;
    using pointer = ArrowHelper<value_type>;
    using reference = value_type &;
    using iterator_category = std::random_access_iterator_tag;

    template <bool other_const>
    using Other = DataIteratorBase<Derived, GT, DT, other_const>;

    constexpr DataIteratorBase(parent_type *graph,
                               difference_type index) noexcept
      : graph_(graph), index_(index) { }

    template <bool other_const,
              typename = std::enable_if_t<is_const && !other_const>>
    constexpr DataIteratorBase(const Other<other_const> &other) noexcept
      : graph_(other.graph_), index_(other.index_) { }

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
      return Derived(graph_, index_ + n);
    }

    constexpr Derived &operator-=(difference_type n) noexcept {
      index_ -= n;
      return *derived();
    }

    constexpr Derived operator-(difference_type n) const noexcept {
      return Derived(graph_, index_ - n);
    }

    template <bool other_const>
    constexpr difference_type
    operator-(const Other<other_const> &other) const noexcept {
      return index_ - other.index_;
    }

    constexpr value_type operator*() const noexcept {
      return derived()->deref(graph_, index_);
    }

    constexpr value_type operator[](difference_type n) const noexcept {
      return *(*derived() + n);
    }

    constexpr pointer operator->() const noexcept { return **derived(); }

    template <bool other_const>
    constexpr bool operator<(const Other<other_const> &other) const noexcept {
      return index_ < other.index_;
    }

    template <bool other_const>
    constexpr bool operator>(const Other<other_const> &other) const noexcept {
      return index_ > other.index_;
    }

    template <bool other_const>
    constexpr bool operator==(const Other<other_const> &other) const noexcept {
      return index_ == other.index_;
    }

    template <bool other_const>
    constexpr bool operator!=(const Other<other_const> &other) const noexcept {
      return !(*this == other);
    }

  protected:
    template <class, class, class, bool other_const>
    friend struct DataIteratorBase;

    constexpr Derived *derived() noexcept {
      return static_cast<Derived *>(this);
    }

    constexpr const Derived *derived() const noexcept {
      return static_cast<const Derived *>(this);
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
    using NT = typename GT::node_property_type;

    using parent_type = const_if_t<is_const, GT>;
    using value_type = const_if_t<is_const, NT>;

    template <bool other_const>
    using Other = NodeWrapper<GT, other_const>;

    constexpr NodeWrapper(int nid, NT &data, parent_type &graph) noexcept
      : nid_(nid), data_(&data), graph_(&graph) { }

    constexpr NodeWrapper(int nid, const NT &data, parent_type &graph) noexcept
      : nid_(nid), data_(&data), graph_(&graph) { }

    template <bool other_const,
              typename = std::enable_if_t<is_const && !other_const>>
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

  private:
    using parent_type = const_if_t<is_const, GT>;

    friend GT;
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
    using ET = typename GT::edge_property_type;

    using parent_type = const_if_t<is_const, GT>;
    using value_type = const_if_t<is_const, ET>;

    template <bool other_const>
    using Other = EdgeWrapper<GT, other_const>;

    constexpr EdgeWrapper(int eid, ET &data, parent_type &graph) noexcept
      : eid_(eid), data_(&data), graph_(&graph) { }

    constexpr EdgeWrapper(int eid, const ET &data, parent_type &graph) noexcept
      : eid_(eid), data_(&data), graph_(&graph) { }

    template <bool other_const,
              typename = std::enable_if_t<is_const && !other_const>>
    constexpr EdgeWrapper(const Other<other_const> &other) noexcept
      : eid_(other.eid_), data_(other.data_), graph_(other.graph_) { }

    constexpr int id() const noexcept { return eid_; }

    constexpr value_type &data() const noexcept {
      return *const_cast<value_type *>(data_);
    }

    constexpr Other<true> as_const() const noexcept { return *this; }

  private:
    int eid_;
    value_type *data_;
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

  private:
    using parent_type = const_if_t<is_const, GT>;

    friend GT;
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
    using AT = typename GT::adj_property_type;

    using value_type = const_if_t<is_const, AT>;
    using parent_type = const_if_t<is_const, GT>;

    using NodeRef = NodeWrapper<GT, is_const>;

    constexpr AdjWrapper(int src, AT &adj, parent_type &graph) noexcept
      : src_(src), adj_(&adj), graph_(&graph) { }

    constexpr AdjWrapper(int src, const AT &adj, parent_type &graph) noexcept
      : src_(src), adj_(&adj), graph_(&graph) { }

    template <bool other_const,
              typename = std::enable_if_t<is_const && !other_const>>
    constexpr AdjWrapper(const AdjWrapper<GT, other_const> &other) noexcept
      : src_(other.src_), adj_(other.adj_), graph_(other.graph_) { }

    constexpr int eid() const noexcept { return adj_->eid; }
    constexpr int src() const noexcept { return src_; }
    constexpr int dst() const noexcept { return adj_->dst; }

    constexpr NodeRef src_node() const noexcept {
      return const_cast<parent_type *>(graph_)->node(src());
    }

    constexpr NodeRef dst_node() const noexcept {
      return const_cast<parent_type *>(graph_)->node(dst());
    }

    constexpr auto &edge_data() const noexcept {
      return const_cast<parent_type *>(graph_)->edge(eid()).data();
    }

    constexpr AdjWrapper<GT, true> as_const() const noexcept { return *this; }

  private:
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
      : Base(other), nid_(other.nid) { }

  private:
    friend GT;
    friend Base;
    template <class, bool other_const>
    friend class AdjIterator;

    constexpr value_type deref(parent_type *graph,
                               difference_type index) const noexcept {
      return graph->adjacent(nid_, index);
    }

    int nid_;
  };
}  // namespace internal

template <class NT, class ET>
class Graph {
public:
  struct StoredEdge {
    int src;
    int dst;
    ET data;
  };

  struct AdjEntry {
    int dst;
    int eid;
  };

  using node_property_type = NT;
  using edge_property_type = ET;
  using adj_property_type = AdjEntry;

  using iterator = internal::NodeIterator<Graph, false>;
  using const_iterator = internal::NodeIterator<Graph, true>;
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

  Graph(int num_nodes): nodes_(num_nodes), adj_list_(num_nodes) { }
  Graph(int num_nodes, const NT &data)
    : nodes_(num_nodes, data), adj_list_(num_nodes) { }

  int size() const { return num_nodes(); }
  int num_nodes() const { return nodes_.size(); }
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

  int add_edge(int src, int dst) {
    int id = num_edges();
    adj_list_[src].push_back({ dst, id });
    adj_list_[dst].push_back({ src, id });
    edges_.push_back({ src, dst });
    return id;
  }

  NodeRef operator[](int id) { return node(id); }
  ConstNodeRef operator[](int id) const { return node(id); }

  NodeRef node(int id) { return { id, nodes_[id], *this }; }
  ConstNodeRef node(int id) const { return { id, nodes_[id], *this }; }
  void update_node(int id, const NT &data) { nodes_[id] = data; }
  void update_node(int id, NT &&data) noexcept { nodes_[id] = std::move(data); }

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

  iterator edge_begin() { return { this, 0 }; }
  iterator edge_end() { return { this, num_edges() }; }
  const_iterator edge_begin() const { return edge_cbegin(); }
  const_iterator edge_end() const { return edge_cend(); }
  const_iterator edge_cbegin() const { return { this, 0 }; }
  const_iterator edge_cend() const { return { this, num_edges() }; }

  AdjRef adjacent(int nid, int idx) {
    return { nid, adj_list_[nid][idx], *this };
  }

  ConstAdjRef adjacent(int nid, int idx) const {
    return { nid, adj_list_[nid][idx], *this };
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
  std::vector<std::vector<AdjEntry>> adj_list_;
  std::vector<NT> nodes_;
  std::vector<StoredEdge> edges_;
};
}  // namespace nuri

namespace boost {
template <class NT, class ET>
struct graph_traits<nuri::Graph<NT, ET>> {
  using vertex_descriptor = int;
  using edge_descriptor = int;
  using directed_category = undirected_tag;
  using edge_parallel_category = allow_parallel_edge_tag;
  using traversal_category = boost::undir_adj_list_traversal_tag;

  using vertices_size_type = int;
  using edges_size_type = int;
  using degree_size_type = int;
};

template <class NT, class ET>
typename graph_traits<nuri::Graph<NT, ET>>::vertex_descriptor
source(typename graph_traits<nuri::Graph<NT, ET>>::edge_descriptor e,
       const nuri::Graph<NT, ET> &g) {
  return g.edge(e).src;
}

template <class NT, class ET>
typename graph_traits<nuri::Graph<NT, ET>>::vertex_descriptor
target(typename graph_traits<nuri::Graph<NT, ET>>::edge_descriptor e,
       const nuri::Graph<NT, ET> &g) {
  return g.edge(e).dst;
}
}  // namespace boost

#endif /* NURIKIT_GRAPH_H_ */

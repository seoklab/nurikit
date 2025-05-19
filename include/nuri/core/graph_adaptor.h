//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_GRAPH_ADAPTOR_H_
#define NURI_CORE_GRAPH_ADAPTOR_H_

//! @cond
#include <utility>

#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
//! @endcond

#include "nuri/core/graph.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  struct nuri_graph_traversal_tag: virtual boost::incidence_graph_tag,
                                   virtual boost::adjacency_graph_tag,
                                   virtual boost::bidirectional_graph_tag,
                                   virtual boost::vertex_list_graph_tag,
                                   virtual boost::edge_list_graph_tag { };

  template <class Impl, class VT, class Iterator>
  class BoostGraphIteratorAdaptorBase
      : public ProxyIterator<Impl, VT, typename Iterator::iterator_category,
                             typename Iterator::difference_type> {
  public:
    BoostGraphIteratorAdaptorBase() = default;

    BoostGraphIteratorAdaptorBase(Iterator it): it_(it) { }

  protected:
    using Base = BoostGraphIteratorAdaptorBase;

  private:
    VT dereference() const {
      return static_cast<const Impl &>(*this).dereference_impl(it_);
    }

    bool equal(const BoostGraphIteratorAdaptorBase &rhs) const {
      return it_ == rhs.it_;
    }

    typename Iterator::difference_type
    distance_to(const BoostGraphIteratorAdaptorBase &rhs) const {
      return rhs.it_ - it_;
    }

    void increment() { ++it_; }
    void decrement() { --it_; }

    void advance(int n) { it_ += n; }

    Iterator it_;

    friend class boost::iterator_core_access;
  };

  template <class GT>
  struct BoostGraphTraits {
  private:
    using Graph = GT;
    using nuri_cnode_iter = typename Graph::const_node_iterator;
    using nuri_cedge_iter = typename Graph::const_edge_iterator;
    using nuri_cadj_iter = typename Graph::const_adjacency_iterator;

  public:
    using vertex_descriptor = int;
    using edge_descriptor = int;

    class adjacency_iterator
        : public BoostGraphIteratorAdaptorBase<
              adjacency_iterator, vertex_descriptor, nuri_cadj_iter> {
      using Base = typename adjacency_iterator::Base;

    public:
      using Base::Base;

    private:
      static vertex_descriptor dereference_impl(nuri_cadj_iter it) {
        return it->dst().id();
      }

      friend Base;
    };

    class out_edge_iterator
        : public BoostGraphIteratorAdaptorBase<
              out_edge_iterator, edge_descriptor, nuri_cadj_iter> {
      using Base = typename out_edge_iterator::Base;

    public:
      using Base::Base;

    private:
      static edge_descriptor dereference_impl(nuri_cadj_iter it) {
        return it->eid();
      }

      friend Base;
    };
    using in_edge_iterator = out_edge_iterator;

    class vertex_iterator
        : public BoostGraphIteratorAdaptorBase<
              vertex_iterator, vertex_descriptor, nuri_cnode_iter> {
      using Base = typename vertex_iterator::Base;

    public:
      using Base::Base;

    private:
      static vertex_descriptor dereference_impl(nuri_cnode_iter it) {
        return it->id();
      }

      friend Base;
    };

    class edge_iterator
        : public BoostGraphIteratorAdaptorBase<edge_iterator, edge_descriptor,
                                               nuri_cedge_iter> {
      using Base = typename edge_iterator::Base;

    public:
      using Base::Base;

    private:
      static edge_descriptor dereference_impl(nuri_cedge_iter it) {
        return it->id();
      }

      friend Base;
    };

    using directed_category = boost::undirected_tag;
    using edge_parallel_category = boost::allow_parallel_edge_tag;
    using traversal_category = nuri_graph_traversal_tag;

    using vertices_size_type = int;
    using edges_size_type = int;
    using degree_size_type = int;

    static vertex_descriptor null_vertex() { return -1; }
  };
}  // namespace internal
}  // namespace nuri

namespace boost {
template <class NT, class ET>
struct graph_traits<nuri::Graph<NT, ET>>
    : nuri::internal::BoostGraphTraits<nuri::Graph<NT, ET>> { };

template <class NT, class ET, bool is_const>
struct graph_traits<nuri::Subgraph<NT, ET, is_const>>
    : nuri::internal::BoostGraphTraits<nuri::Subgraph<NT, ET, is_const>> { };
}  // namespace boost

// These functions must be put here for ADL
namespace nuri {
//! @section Concept requirements for nuri::Graph
//! @subsection IncidenceGraph concept requirements

template <class NT, class ET>
auto out_edges(typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
               const Graph<NT, ET> &g) {
  using Iter = typename boost::graph_traits<Graph<NT, ET>>::out_edge_iterator;
  return std::make_pair(Iter(g[v].begin()), Iter(g[v].end()));
}

template <class NT, class ET>
typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor
source(typename boost::graph_traits<Graph<NT, ET>>::edge_descriptor e,
       const Graph<NT, ET> &g) {
  return g.edge(e).src().id();
}

template <class NT, class ET>
typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor
target(typename boost::graph_traits<Graph<NT, ET>>::edge_descriptor e,
       const Graph<NT, ET> &g) {
  return g.edge(e).dst().id();
}

template <class NT, class ET>
auto out_degree(typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
                const Graph<NT, ET> &g) {
  return g.degree(v);
}

//! @subsection BidirectionalGraph concept requirements

template <class NT, class ET>
auto in_edges(typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
              const Graph<NT, ET> &g) {
  using Iter = typename boost::graph_traits<Graph<NT, ET>>::in_edge_iterator;
  return std::make_pair(Iter(g[v].begin()), Iter(g[v].end()));
}

template <class NT, class ET>
auto in_degree(typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
               const Graph<NT, ET> &g) {
  return g.degree(v);
}

template <class NT, class ET>
auto degree(typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
            const Graph<NT, ET> &g) {
  return g.degree(v);
}

//! @subsection AdjacencyGraph concept requirements

template <class NT, class ET>
auto adjacent_vertices(
    typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
    const Graph<NT, ET> &g) {
  using Iter = typename boost::graph_traits<Graph<NT, ET>>::adjacency_iterator;
  return std::make_pair(Iter(g[v].begin()), Iter(g[v].end()));
}

//! @subsection VertexListGraph concept requirements

template <class NT, class ET>
auto vertices(const Graph<NT, ET> &g) {
  using Iter = typename boost::graph_traits<Graph<NT, ET>>::vertex_iterator;
  return std::make_pair(Iter(g.begin()), Iter(g.end()));
}

template <class NT, class ET>
auto num_vertices(const Graph<NT, ET> &g) {
  return g.size();
}

//! @subsection EdgeListGraph concept requirements

template <class NT, class ET>
auto edges(const Graph<NT, ET> &g) {
  using Iter = typename boost::graph_traits<Graph<NT, ET>>::edge_iterator;
  return std::make_pair(Iter(g.edge_begin()), Iter(g.edge_end()));
}

template <class NT, class ET>
auto num_edges(const Graph<NT, ET> &g) {
  return g.num_edges();
}

//! @subsection MutableGraph concept requirements

template <class NT, class ET>
auto add_vertex(Graph<NT, ET> &g) {
  return g.add_node({});
}

template <class NT, class ET>
void clear_vertex(
    typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
    Graph<NT, ET> &g) {
  auto extract_eid = [](auto adj) { return adj.eid(); };
  g.erase_edges(internal::make_transform_iterator(g[v].begin(), extract_eid),
                internal::make_transform_iterator(g[v].end(), extract_eid));
}

template <class NT, class ET>
void remove_vertex(
    typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
    Graph<NT, ET> &g) {
  g.erase_nodes(g.begin() + v, g.begin() + v + 1);
}

template <class NT, class ET>
auto add_edge(typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor u,
              typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
              Graph<NT, ET> &g) {
  int e = g.add_edge(u, v, {});
  return std::make_pair(e, true);
}

template <class NT, class ET>
void remove_edge(
    typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor u,
    typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v,
    Graph<NT, ET> &g) {
  g.erase_edge_between(u, v);
}

template <class NT, class ET>
void remove_edge(typename boost::graph_traits<Graph<NT, ET>>::edge_descriptor e,
                 Graph<NT, ET> &g) {
  g.erase_edge(e);
}

template <class NT, class ET>
void remove_edge(typename boost::graph_traits<Graph<NT, ET>>::edge_iterator it,
                 Graph<NT, ET> &g) {
  remove_edge(*it, g);
}

//! @subsection PropertyGraph concept requirements (partial)

template <class NT, class ET>
auto get(boost::vertex_index_t /* tag */, const Graph<NT, ET> & /* g */) {
  return boost::typed_identity_property_map<
      typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor> {};
}

template <class NT, class ET>
auto get(boost::vertex_index_t /* tag */, const Graph<NT, ET> & /* g */,
         typename boost::graph_traits<Graph<NT, ET>>::vertex_descriptor v) {
  return v;
}

template <class NT, class ET>
auto get(boost::edge_index_t /* tag */, const Graph<NT, ET> & /* g */) {
  return boost::typed_identity_property_map<
      typename boost::graph_traits<Graph<NT, ET>>::edge_descriptor> {};
}

template <class NT, class ET>
auto get(boost::edge_index_t /* tag */, const Graph<NT, ET> & /* g */,
         typename boost::graph_traits<Graph<NT, ET>>::edge_descriptor e) {
  return e;
}

//! @section Concept requirements for nuri::Subgraph
//! @subsection IncidenceGraph concept requirements

template <class NT, class ET, bool is_const>
auto out_edges(typename boost::graph_traits<
                   Subgraph<NT, ET, is_const>>::vertex_descriptor v,
               const Subgraph<NT, ET, is_const> &g) {
  using Iter =
      typename boost::graph_traits<Subgraph<NT, ET, is_const>>::out_edge_iterator;
  return std::make_pair(Iter(g[v].begin()), Iter(g[v].end()));
}

template <class NT, class ET, bool is_const>
typename boost::graph_traits<Subgraph<NT, ET, is_const>>::vertex_descriptor
source(
    typename boost::graph_traits<Subgraph<NT, ET, is_const>>::edge_descriptor e,
    const Subgraph<NT, ET, is_const> &g) {
  return g.edge(e).src().id();
}

template <class NT, class ET, bool is_const>
typename boost::graph_traits<Subgraph<NT, ET, is_const>>::vertex_descriptor
target(
    typename boost::graph_traits<Subgraph<NT, ET, is_const>>::edge_descriptor e,
    const Subgraph<NT, ET, is_const> &g) {
  return g.edge(e).dst().id();
}

template <class NT, class ET, bool is_const>
auto out_degree(typename boost::graph_traits<
                    Subgraph<NT, ET, is_const>>::vertex_descriptor v,
                const Subgraph<NT, ET, is_const> &g) {
  return g.degree(v);
}

//! @subsection BidirectionalGraph concept requirements

template <class NT, class ET, bool is_const>
auto in_edges(typename boost::graph_traits<
                  Subgraph<NT, ET, is_const>>::vertex_descriptor v,
              const Subgraph<NT, ET, is_const> &g) {
  using Iter =
      typename boost::graph_traits<Subgraph<NT, ET, is_const>>::in_edge_iterator;
  return std::make_pair(Iter(g[v].begin()), Iter(g[v].end()));
}

template <class NT, class ET, bool is_const>
auto in_degree(typename boost::graph_traits<
                   Subgraph<NT, ET, is_const>>::vertex_descriptor v,
               const Subgraph<NT, ET, is_const> &g) {
  return g.degree(v);
}

template <class NT, class ET, bool is_const>
auto degree(typename boost::graph_traits<
                Subgraph<NT, ET, is_const>>::vertex_descriptor v,
            const Subgraph<NT, ET, is_const> &g) {
  return g.degree(v);
}

//! @subsection AdjacencyGraph concept requirements

template <class NT, class ET, bool is_const>
auto adjacent_vertices(typename boost::graph_traits<
                           Subgraph<NT, ET, is_const>>::vertex_descriptor v,
                       const Subgraph<NT, ET, is_const> &g) {
  using Iter = typename boost::graph_traits<
      Subgraph<NT, ET, is_const>>::adjacency_iterator;
  return std::make_pair(Iter(g[v].begin()), Iter(g[v].end()));
}

//! @subsection VertexListGraph concept requirements

template <class NT, class ET, bool is_const>
auto vertices(const Subgraph<NT, ET, is_const> &g) {
  using Iter =
      typename boost::graph_traits<Subgraph<NT, ET, is_const>>::vertex_iterator;
  return std::make_pair(Iter(g.begin()), Iter(g.end()));
}

template <class NT, class ET, bool is_const>
auto num_vertices(const Subgraph<NT, ET, is_const> &g) {
  return g.size();
}

//! @subsection EdgeListGraph concept requirements

template <class NT, class ET, bool is_const>
auto edges(const Subgraph<NT, ET, is_const> &g) {
  using Iter =
      typename boost::graph_traits<Subgraph<NT, ET, is_const>>::edge_iterator;
  return std::make_pair(Iter(g.edge_begin()), Iter(g.edge_end()));
}

template <class NT, class ET, bool is_const>
auto num_edges(const Subgraph<NT, ET, is_const> &g) {
  return g.num_edges();
}

//! @subsection PropertyGraph concept requirements (partial)

template <class NT, class ET, bool is_const>
auto get(boost::vertex_index_t /* tag */,
         const Subgraph<NT, ET, is_const> & /* g */) {
  return boost::typed_identity_property_map<typename boost::graph_traits<
      Subgraph<NT, ET, is_const>>::vertex_descriptor> {};
}

template <class NT, class ET, bool is_const>
auto get(
    boost::vertex_index_t /* tag */, const Subgraph<NT, ET, is_const> & /* g */,
    typename boost::graph_traits<Subgraph<NT, ET, is_const>>::vertex_descriptor
        v) {
  return v;
}

template <class NT, class ET, bool is_const>
auto get(boost::edge_index_t /* tag */,
         const Subgraph<NT, ET, is_const> & /* g */) {
  return boost::typed_identity_property_map<typename boost::graph_traits<
      Subgraph<NT, ET, is_const>>::edge_descriptor> {};
}

template <class NT, class ET, bool is_const>
auto get(
    boost::edge_index_t /* tag */, const Subgraph<NT, ET, is_const> & /* g */,
    typename boost::graph_traits<Subgraph<NT, ET, is_const>>::edge_descriptor e) {
  return e;
}
}  // namespace nuri

#endif /* NURI_CORE_GRAPH_ADAPTOR_H_ */

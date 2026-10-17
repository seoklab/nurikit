/*
 *  Copyright (c) 2000-2022 Inria
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *  * Neither the name of the ALICE Project-Team nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Contact: Bruno Levy
 *
 *     https://www.inria.fr/fr/bruno-levy
 *
 *     Inria,
 *     Domaine de Voluceau,
 *     78150 Le Chesnay - Rocquencourt
 *     FRANCE
 *
 */

#ifndef GEOGRAM_DELAUNAY_DELAUNAY
#define GEOGRAM_DELAUNAY_DELAUNAY

#include <geogram/basic/common.h>
#include <geogram/basic/memory.h>

/**
 * \file geogram/delaunay/delaunay.h
 * \brief Abstract interface for Delaunay
 */

namespace GEO {

    /************************************************************************/

    /**
     * \brief Abstract interface for Delaunay triangulation in Nd.
     * \details
     * Delaunay objects are created using method create() which
     * uses the Factory service. New Delaunay triangulations can be
     * implemented and registered to the factory using
     * geo_register_Delaunay_creator().
     * \see DelaunayFactory
     * \see geo_register_Delaunay_creator
     */
    class GEOGRAM_API Delaunay {
    public:


        /**
         * \brief Gets the dimension of this Delaunay.
         * \return the dimension of this Delauna
         */
        coord_index_t dimension() const {
            return dimension_;
        }

        /**
         * \brief Gets the number of vertices in each cell
         * \details Cell_size =  dimension + 1
         * \return the number of vertices in each cell
         */
        index_t cell_size() const {
            return cell_size_;
        }

        /**
         * \brief Sets the vertices of this Delaunay, and recomputes the cells.
         * \param[in] nb_vertices number of vertices
         * \param[in] vertices a pointer to the coordinates of the vertices, as
         *  a contiguous array of doubles
         */
        void set_vertices(index_t nb_vertices, const double* vertices);

        /**
         * \brief Specifies whether vertices should be reordered.
         * \details Reordering is activated by default. Some special
         *  usages of Delaunay3d may require to deactivate it (for
         *  instance if vertices are already known to be ordered).
         * \param[in] x if true, then vertices are reordered using
         *  BRIO-Hilbert ordering. This improves speed significantly
         *  (enabled by default).
         */
        void set_reorder(bool x) {
            do_reorder_ = x;
        }

        /**
         * \brief Gets a pointer to the array of vertices.
         * \return A const pointer to the array of vertices.
         */
        const double* vertices_ptr() const {
            return vertices_;
        }

        /**
         * \brief Gets a pointer to a vertex by its global index.
         * \param[in] i global index of the vertex
         * \return a pointer to vertex \p i
         */
        const double* vertex_ptr(index_t i) const {
            geo_debug_assert(i < nb_vertices());
            return vertices_ + vertex_stride_ * i;
        }

        /**
         * \brief Gets the number of vertices.
         * \return the number of vertices in this Delaunay
         */
        index_t nb_vertices() const {
            return nb_vertices_;
        }

        /**
         * \brief Gets the number of cells.
         * \return the number of cells in this Delaunay
         */
        index_t nb_cells() const {
            return nb_cells_;
        }

        /**
         * \brief Gets the number of finite cells.
         * \pre this function can only be called if
         *   keep_finite is set
         * \details Finite cells have indices 0..nb_finite_cells()-1
         *  and infinite cells have indices nb_finite_cells()..nb_cells()-1
         * \see set_keeps_infinite(), keeps_infinite()
         */
        index_t nb_finite_cells() const {
            geo_debug_assert(keeps_infinite());
            return nb_finite_cells_;
        }

        /**
         * \brief Gets a pointer to the cell-to-vertex incidence array.
         * \return a const pointer to the cell-to-vertex incidence array
         */
        const index_t* cell_to_v() const {
            return cell_to_v_;
        }

        /**
         * \brief Gets a pointer to the cell-to-cell adjacency array.
         * \return a const pointer to the cell-to-cell adjacency array
         */
        const index_t* cell_to_cell() const {
            return cell_to_cell_;
        }

        /**
         * \brief Computes the nearest vertex from a query point.
         * \param[in] p query point
         * \return the index of the nearest vertex
         */
        index_t nearest_vertex(const double* p) const;

        /**
         * \brief Gets a vertex index by cell index and local vertex index.
         * \param[in] c cell index
         * \param[in] lv local vertex index in cell \p c
         * \return the index of the lv-th vertex of cell c.
         */
        index_t cell_vertex(index_t c, index_t lv) const {
            geo_debug_assert(c < nb_cells());
            geo_debug_assert(lv < cell_size());
            return cell_to_v_[c * cell_v_stride_ + lv];
        }

        /**
         * \brief Gets an adjacent cell index by cell index and
         *  local facet index.
         * \param[in] c cell index
         * \param[in] lf local facet index
         * \return the index of the cell adjacent to \p c accros
         *  facet \p lf if it exists, or NO_INDEX if on border
         */
        index_t cell_adjacent(index_t c, index_t lf) const {
            geo_debug_assert(c < nb_cells());
            geo_debug_assert(lf < cell_size());
            return cell_to_cell_[c * cell_neigh_stride_ + lf];
        }

        /**
         * \brief Tests whether a cell is infinite.
         * \retval true if cell \p c is infinite
         * \retval false otherwise
         * \see keeps_infinite(), set_keeps_infinite()
         */
        bool cell_is_infinite(index_t c) const;

        /**
         * \brief Tests whether a cell is finite.
         * \retval true if cell \p c is finite
         * \retval false otherwise
         * \see keeps_infinite(), set_keeps_infinite()
         */
        bool cell_is_finite(index_t c) const {
            return !cell_is_infinite(c);
        }

        /**
         * \brief Retrieves a local vertex index from cell index
         *  and global vertex index.
         * \param[in] c cell index
         * \param[in] v global vertex index
         * \return the local index of vertex \p v in cell \p c
         * \pre cell \p c is incident to vertex \p v
         */
        index_t index(index_t c, index_t v) const {
            geo_debug_assert(c < nb_cells());
            geo_debug_assert(v == NO_INDEX || v < nb_vertices());
            for(index_t iv = 0; iv < cell_size(); iv++) {
                if(cell_vertex(c, iv) == v) {
                    return iv;
                }
            }
            geo_assert_not_reached;
        }

        /**
         * \brief Retrieves a local facet index from two adacent
         *  cell global indices.
         * \param[in] c1 global index of first cell
         * \param[in] c2 global index of second cell
         * \return the local index of the face accros which
         *  \p c2 is adjacent to \p c1
         * \pre cell \p c1 and cell \p c2 are adjacent
         */
        index_t adjacent_index(index_t c1, index_t c2) const {
            geo_debug_assert(c1 < nb_cells());
            geo_debug_assert(c2 < nb_cells());
            for(index_t f = 0; f < cell_size(); f++) {
                if(cell_adjacent(c1, f) == c2) {
                    return f;
                }
            }
            geo_assert_not_reached;
        }

        /**
         * \brief Gets an incident cell index by a vertex index.
         * \details Can only be used if set_stores_cicl(true) was called.
         * \param[in] v a vertex index
         * \return the index of a cell incident to vertex \p v
         * \see stores_cicl(), set_store_cicl()
         */
        index_t vertex_cell(index_t v) const {
            geo_debug_assert(v < nb_vertices());
            geo_debug_assert(v < v_to_cell_.size());
            return v_to_cell_[v];
        }


        /**
         * \brief Traverses the list of cells incident to a vertex.
         * \details Can only be used if set_stores_cicl(true) was called.
         * \param[in] c cell index
         * \param[in] lv local vertex index
         * \return the index of the next cell around vertex \p c or NO_INDEX if
         *  \p c was the last one in the list
         * \see stores_cicl(), set_store_cicl()
         */
        index_t next_around_vertex(index_t c, index_t lv) const {
            geo_debug_assert(c < nb_cells());
            geo_debug_assert(lv < cell_size());
            return cicl_[cell_size() * c + lv];
        }

        /**
         * \brief Gets the one-ring neighbors of vertex v.
         * \details Depending on store_neighbors_ internal flag, the
         *  neighbors are computed or copied from the previously computed
         *  list.
         * \param[in] v vertex index
         * \param[out] neighbors indices of the one-ring neighbors of
         *  vertex \p v
         * \see stores_neighbors(), set_stores_neighbors()
         */
        void get_neighbors(index_t v, vector<index_t>& neighbors) const {
            geo_debug_assert(v < nb_vertices());
            get_neighbors_internal(v, neighbors);
        }

        /**
         * \brief Tests whether incident tetrahedra lists
         *   are stored.
         * \retval true if incident tetrahedra lists are stored.
         * \retval false otherwise.
         */
        bool stores_cicl() const {
            return store_cicl_;
        }

        /**
         * \brief Specifies whether incident tetrahedra lists
         *   should be stored.
         * \param[in] x if true, incident trahedra lists are stored,
         *   else they are not.
         */
        void set_stores_cicl(bool x) {
            store_cicl_ = x;
        }


        /**
         * \brief Tests whether infinite elements are kept.
         * \retval true if infinite elements are kept
         * \retval false otherwise
         */
        bool keeps_infinite() const {
            return keep_infinite_;
        }

        /**
         * \brief Sets whether infinite elements should be kept.
         * \details Internally, Delaunay implementation uses an
         *  infinite vertex and infinite simplices indicent to it.
         *  By default they are discarded at the end of set_vertices().
         *  \param[in] x true if infinite elements should be kept,
         *   false otherwise
         */
        void set_keeps_infinite(bool x) {
            keep_infinite_ = x;
        }


    protected:
        /**
         * \brief Creates a new Delaunay triangulation
         * \details This creates a new Delaunay triangulation for the
         * specified \p dimension. Specific implementations of the Delaunay
         * triangulation may not support the specified \p dimension and will
         * throw a InvalidDimension exception.
         * \param[in] dimension dimension of the triangulation
         * \throw InvalidDimension This exception is thrown if the specified
         * \p dimension is not supported by the Delaunay implementation.
         * \note This function is never called directly, use create()
         */
        Delaunay(coord_index_t dimension);

        /**
         * \brief Internal implementation for get_neighbors (with vector).
         * \param[in] v index of the Delaunay vertex
         * \param[in,out] neighbors the computed neighbors of vertex \p v.
         *    Its size is used to determine the number of queried neighbors.
         */
        void get_neighbors_internal(
            index_t v, vector<index_t>& neighbors
        ) const;

        /**
         * \brief Sets the arrays that represent the combinatorics
         *  of this Delaunay.
         * \param[in] nb_cells number of cells
         * \param[in] cell_to_v the cell-to-vertex incidence array
         * \param[in] cell_to_cell the cell-to-cell adjacency array
         */
        void set_arrays(
            index_t nb_cells,
            const index_t* cell_to_v, const index_t* cell_to_cell
        );

        /**
         * \brief Stores for each vertex v a cell incident to v.
         */
        void update_v_to_cell();

        /**
         * \brief Updates the circular incident cell lists.
         * \details Used by next_around_vertex().
         */
        void update_cicl();

        /**
         * \brief Sets the circular incident edge list.
         * \param[in] c1 index of a cell
         * \param[in] lv local index of a vertex of \p c1
         * \param[in] c2 index of the next cell around \p c1%'s vertex \p lv
         */
        void set_next_around_vertex(
            index_t c1, index_t lv, index_t c2
        ) {
            geo_debug_assert(c1 < nb_cells());
            geo_debug_assert(c2 < nb_cells());
            geo_debug_assert(lv < cell_size());
            cicl_[cell_size() * c1 + lv] = c2;
        }

        /**
         * \brief Sets the dimension of this Delaunay.
         * \details Updates all the parameters related with
         *  the dimension. This includes vertex_stride (number
         *  of doubles between two consecutive vertices),
         *  cell size (number of vertices in a cell),
         *  cell_v_stride (number of integers between two
         *  consecutive cell vertex arrays),
         *  cell_neigh_stride (number of integers
         *  between two consecutive cell adjacency arrays).
         * \param[in] dim the dimension
         */
        void set_dimension(coord_index_t dim) {
            dimension_ = dim;
            vertex_stride_ = dim;
            cell_size_ = index_t(dim) + 1;
            cell_v_stride_ = cell_size_;
            cell_neigh_stride_ = cell_size_;
        }

        coord_index_t dimension_;
        index_t vertex_stride_;
        index_t cell_size_;
        index_t cell_v_stride_;
        index_t cell_neigh_stride_;

        const double* vertices_;
        index_t nb_vertices_;
        index_t nb_cells_;
        const index_t* cell_to_v_;
        const index_t* cell_to_cell_;
        vector<index_t> v_to_cell_;
        vector<index_t> cicl_;
        bool is_locked_;
        /**
         * \brief If true, uses BRIO reordering
         * (in some implementations)
         */
        bool do_reorder_;

        /**
         * \brief It true, circular incident tet
         * lists are stored.
         */
        bool store_cicl_;

        /**
         * \brief If true, infinite vertex and
         * infinite simplices are kept.
         */
        bool keep_infinite_;

        /**
         * \brief If keep_infinite_ is true, then
         *  finite cells are 0..nb_finite_cells_-1
         *  and infinite cells are
         *  nb_finite_cells_ ... nb_cells_
         */
        index_t nb_finite_cells_;

    };

}

#endif

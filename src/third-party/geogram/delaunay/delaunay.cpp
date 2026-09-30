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

#include <geogram/delaunay/delaunay.h>
#include <geogram/basic/geometry_nd.h>
#include <geogram/basic/algorithm.h>

namespace GEO {

    Delaunay::Delaunay(coord_index_t dimension) {
        set_dimension(dimension);
        vertices_ = nullptr;
        nb_vertices_ = 0;
        nb_cells_ = 0;
        cell_to_v_ = nullptr;
        cell_to_cell_ = nullptr;
        is_locked_ = false;
        do_reorder_ = true;
        store_cicl_ = false;
        keep_infinite_ = false;
        nb_finite_cells_ = 0;
    }

    void Delaunay::set_vertices(index_t nb_vertices, const double* vertices) {
        nb_vertices_ = nb_vertices;
        vertices_ = vertices;
    }

    void Delaunay::set_arrays(
        index_t nb_cells,
        const index_t* cell_to_v, const index_t* cell_to_cell
    ) {
        nb_cells_ = nb_cells;
        cell_to_v_ = cell_to_v;
        cell_to_cell_ = cell_to_cell;

        if(cell_to_cell != nullptr) {
            if(store_cicl_) {
                update_v_to_cell();
                update_cicl();
            }
        }
    }

    index_t Delaunay::nearest_vertex(const double* p) const {
        // Unefficient implementation (but at least it works).
        // Derived classes are supposed to overload.
        geo_assert(nb_vertices() > 0);
        index_t result = 0;
        double d = Geom::distance2(vertex_ptr(0), p, dimension());
        for(index_t i = 1; i < nb_vertices(); i++) {
            double cur_d = Geom::distance2(vertex_ptr(i), p, dimension());
            if(cur_d < d) {
                d = cur_d;
                result = i;
            }
        }
        return result;
    }

    void Delaunay::get_neighbors_internal(
        index_t v, vector<index_t>& neighbors
    ) const {
        neighbors.resize(0);

        index_t vt = v_to_cell_[v];
        if(vt != NO_INDEX) { // Happens when there are duplicated vertices.
            index_t t = vt;
            do {
                index_t lvit = index(t, v);
                for(index_t lv = 0; lv < cell_size(); lv++) {
                    if(lvit != lv) {
                        index_t neigh = cell_vertex(t, lv);
                        geo_debug_assert(neigh != NO_INDEX);
                        neighbors.push_back(neigh);
                    }
                }
                t = next_around_vertex(t, index(t, v));
            } while(t != vt);
        }

        // Remove duplicates by sorting the neighbors
        sort_unique(neighbors);
    }

    void Delaunay::update_v_to_cell() {
        geo_assert(!is_locked_);  // Not thread-safe
        is_locked_ = true;

        // Note: if keeps_infinite is set, then infinite vertex
        // appears in the v_to_cell_ array.
        if(keeps_infinite()) {
            v_to_cell_.assign(nb_vertices()+1, NO_INDEX);
            for(index_t c = 0; c < nb_cells(); c++) {
                for(index_t lv = 0; lv < cell_size(); lv++) {
                    index_t v = cell_vertex(c, lv);
                    if(v == NO_INDEX) {
                        v = nb_vertices();
                    }
                    v_to_cell_[v] = c;
                }
            }
        } else {
            v_to_cell_.assign(nb_vertices(), NO_INDEX);
            for(index_t c = 0; c < nb_cells(); c++) {
                for(index_t lv = 0; lv < cell_size(); lv++) {
                    v_to_cell_[cell_vertex(c, lv)] = c;
                }
            }
        }
        is_locked_ = false;
    }

    void Delaunay::update_cicl() {
        geo_assert(!is_locked_);  // Not thread-safe
        is_locked_ = true;
        cicl_.resize(cell_size() * nb_cells());

        for(index_t v = 0; v < nb_vertices(); ++v) {
            index_t t = v_to_cell_[v];
            if(t != NO_INDEX) {
                index_t lv = index(t, v);
                set_next_around_vertex(t, lv, t);
            }
        }

        if(keeps_infinite()) {

            {
                // Process the infinite vertex at index nb_vertices().
                index_t t = v_to_cell_[nb_vertices()];
                if(t != NO_INDEX) {
                    index_t lv = index(t, NO_INDEX);
                    set_next_around_vertex(t, lv, t);
                }
            }

            for(index_t t = 0; t < nb_cells(); ++t) {
                for(index_t lv = 0; lv < cell_size(); ++lv) {
                    index_t v = cell_vertex(t, lv);
                    index_t vv = (v == NO_INDEX) ? nb_vertices() : v;
                    if(v_to_cell_[vv] != t) {
                        index_t t1 = v_to_cell_[vv];
                        index_t lv1 = index(t1, v);
                        index_t t2 = next_around_vertex(t1, lv1);
                        set_next_around_vertex(t1, lv1, t);
                        set_next_around_vertex(t, lv, t2);
                    }
                }
            }


        } else {
            for(index_t t = 0; t < nb_cells(); ++t) {
                for(index_t lv = 0; lv < cell_size(); ++lv) {
                    index_t v = cell_vertex(t, lv);
                    if(v_to_cell_[v] != t) {
                        index_t t1 = v_to_cell_[v];
                        index_t lv1 = index(t1, v);
                        index_t t2 = next_around_vertex(t1, lv1);
                        set_next_around_vertex(t1, lv1, t);
                        set_next_around_vertex(t, lv, t2);
                    }
                }
            }
        }

        is_locked_ = false;
    }

    bool Delaunay::cell_is_infinite(index_t c) const {
        geo_debug_assert(c < nb_cells());
        for(index_t lv=0; lv < cell_size(); ++lv) {
            if(cell_vertex(c,lv) == NO_INDEX) {
                return true;
            }
        }
        return false;
    }
}

//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_SURFACE_H_
#define NURI_DESC_SURFACE_H_

#include <cmath>
#include <optional>
#include <utility>

#include <absl/log/absl_check.h>
#include <absl/types/span.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  enum class SrSasaMethod {
    kAuto = 0,
    kDirect,
    kOctree,
  };

  extern ArrayXd sr_sasa_impl(const Matrix3Xd &pts, const ArrayXd &radii,
                              int nprobe, SrSasaMethod method);
}  // namespace internal

/**
 * @brief Calculate the Solvent-Accessible Surface Area (SASA) of a molecule
 *        conformation using the Shrake-Rupley algorithm.
 *
 * @param mol The input molecule.
 * @param conf The conformation matrix.
 * @param nprobe The number of probe spheres. Default is 92.
 * @param rprobe The radius of the probe spheres. Default is 1.4 angstroms.
 * @param method Whether prefer direct or octree method. Default is auto.
 *               This is mainly for testing purpose; in most cases, the auto
 *               method will choose the optimal method.
 * @return The calculated SASA values per atom (in angstroms squared).
 */
extern ArrayXd shrake_rupley_sasa(
    const Molecule &mol, const Matrix3Xd &conf, int nprobe = 92,
    double rprobe = 1.4,
    internal::SrSasaMethod method = internal::SrSasaMethod::kAuto);

namespace internal {
  constexpr double kSurfaceLengthEps = 1e-6;
  constexpr double kSurfaceAngleEps = 1e-9;

  class OffsetTable {
  public:
    OffsetTable(): off_(ArrayXi::Zero(1)) { }

    explicit OffsetTable(int n): off_(n + 1) { }

    explicit OffsetTable(ArrayXi &&off) noexcept: off_(std::move(off)) { }

    int operator[](int i) const { return offset(i); }

    int offset(int i) const { return off_[i]; }

    int degree(int i) const { return off_[i + 1] - off_[i]; }

    int max_deg() const {
      return (off_.tail(size()) - off_.head(size())).maxCoeff();
    }

    int size() const { return static_cast<int>(off_.size()) - 1; }

    ArrayXi &off() { return off_; }

    const ArrayXi &off() const { return off_; }

  private:
    ArrayXi off_;
  };

  /**
   * Union-find with path halving; the smaller root absorbs, so every root is
   * the smallest member of its set.
   */
  class UnionFind {
  public:
    explicit UnionFind(int n): parent_(ArrayXi::LinSpaced(n, 0, n - 1)) { }

    int find(int x) {
      while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
      }
      return x;
    }

    void merge(int a, int b) {
      auto [lo, hi] = nuri::minmax(find(a), find(b));
      parent_[hi] = lo;
    }

    int n_sets() const {
      int n = 0;
      for (int x = 0; x < parent_.size(); ++x)
        n += static_cast<int>(parent_[x] == x);
      return n;
    }

    /**
     * Relabel every element with its set id, numbered by smallest member;
     * returns the number of sets.
     */
    int relabel() {
      const int n = static_cast<int>(parent_.size());
      for (int x = 0; x < n; ++x)
        parent_[x] = find(x);

      int k = 0;
      for (int x = 0; x < n; ++x)
        parent_[x] = parent_[x] == x ? k++ : parent_[parent_[x]];
      return k;
    }

    ArrayXi &labels() { return parent_; }

  private:
    ArrayXi parent_;
  };

  class CSR {
  public:
    using const_iterator = ArrayXi::const_iterator;

    CSR() = default;

    CSR(ArrayXi &&adj, OffsetTable &&off) noexcept
        : adj_(std::move(adj)), tbl_(std::move(off)) { }

    const_iterator begin(int i) const { return adj_.begin() + tbl_[i]; }

    const_iterator end(int i) const { return adj_.begin() + tbl_[i + 1]; }

    int offset(int i) const { return tbl_[i]; }

    int degree(int i) const { return tbl_.degree(i); }

    int max_deg() const { return tbl_.max_deg(); }

    auto nbrs(int i) const { return adj_.segment(tbl_[i], degree(i)); }

    int eid(const_iterator it) const {
      return static_cast<int>(it - adj_.begin());
    }

    int m() const { return static_cast<int>(adj_.size()); }

    int n() const { return tbl_.size(); }

    template <class F, class B>
    void for_each_triangle(int n_rows, const F &on_match,
                           const B &before_row) const {
      for (int i = 0; i < n_rows; ++i) {
        if (degree(i) < 2)
          continue;

        before_row(i);
        const auto ei = end(i);
        for (auto pij = begin(i); pij < ei; ++pij) {
          const int j = *pij;
          const auto ej = end(j);
          for (auto pik = pij + 1, pjk = begin(j); pik < ei && pjk < ej;) {
            const int ki = *pik, kj = *pjk;
            if (ki == kj)
              on_match(i, j, ki, pij, pik, pjk);
            pik += value_if(ki <= kj);
            pjk += value_if(kj <= ki);
          }
        }
      }
    }

    ArrayXi &adj() { return adj_; }

    const ArrayXi &adj() const { return adj_; }

    ArrayXi &off() { return tbl_.off(); }

    const ArrayXi &off() const { return tbl_.off(); }

  private:
    ArrayXi adj_;
    OffsetTable tbl_;
  };

  /**
   * Atoms ordered `[active | need | shell | occluders]` with contained spheres
   * dropped: `order` maps new to old indices, spheres `< n_active` own
   * surface, `< n_solve` get arrangements, `< n_enum` get caps. `g` is the
   * forward overlap graph on new indices (`row(i)` = sorted `j > i`) over all
   * kept spheres; pair `q` carries a circle iff `i < n_enum`, and `d[q]` is
   * its centre distance.
   */
  struct SaPrep {
    Matrix3Xd pts;
    ArrayXd sar;
    ArrayXi order;
    CSR g;
    ArrayXd d;
    int n_active;
    int n_solve;
    int n_enum;
  };

  extern std::optional<SaPrep> prepare(const Matrix3Xd &pts, const ArrayXd &sar,
                                       const ArrayXb &active, double rp);

  struct SasCircle {
    Vector3d axis, cntr;
    double a, rl;
    int i, j;
  };

  /**
   * `phi` is measured in the circle frame `e1 = any_perpendicular(axis)`,
   * `e2 = axis x e1`.
   */
  struct SasArc {
    double phi, dphi;
    int circ, beg, end;
  };

  struct SasCaps {
    CSR h;
    Matrix3Xd axis;
    ArrayXd cosa, sina;
  };

  struct SasProbes {
    CSR atoms;
    Matrix3Xd pos, tan;
    OffsetTable tan_off;
    int n_active;
  };

  struct SasGeometry {
    std::vector<SasCircle> circles;

    SasCaps caps;
    SasProbes probes;

    std::vector<SasArc> arcs;
    int n_active_arcs;

    ArrayXd area;
  };

  extern SasGeometry build_sas(const SaPrep &sa);

  /**
   * Arrangement of caps on one sphere. Fill with `begin`, the `add_*` calls
   * (ids are assigned in call order), then `solve`; buffers persist across
   * problems.
   */
  class ArrangementSolver {
  public:
    /**
     * Sizes every buffer once for at most `mcap` caps, `kcap` vertices and
     * `ecap` crossing pairs per problem; a vertex may sit on both caps of a
     * pair, so incidences are bounded by `4 ecap`.
     */
    ArrangementSolver(int mcap, int kcap, int ecap);

    void begin(double radius);

    int add_cap(const Vector3d &axis, double cosa, double sina);

    int add_vertex(const Vector3d &rep, bool accessible);

    void add_crossing(int a, int b) { edges_.push_back({ a, b }); }

    void add_incidence(int cap, int v) { incs_.push_back({ cap, v }); }

    /**
     * Appends the accessible arcs: `circ` is the cap, `beg` and `end` the
     * vertices, both `k` on a full circle, `phi` in the cap frame
     * `e1 = any_perpendicular(axis)`, `e2 = axis x e1`. Returns the
     * accessible area.
     */
    double solve(std::vector<SasArc> &arcs);

  private:
    struct RingVertex {
      double phi;
      int v;
    };

    struct Dart {
      double angle, kappa;
      int arc;
      bool is_in;
    };

    double cap_arcs(std::vector<SasArc> &arcs,
                    const E::Map<ArrayXX<bool>> &crossing);
    std::pair<int, double> walk(const std::vector<SasArc> &arcs, int a0);
    static void snap_ring(absl::Span<Dart> ring);

    double radius_ = 0;
    int m_ = 0, k_ = 0;

    Matrix3Xd axis_, e1_, e2_;
    ArrayXd cosa_, sina_;

    Matrix3Xd reps_, ea_, eb_;
    ArrayXb accessible_;

    std::vector<std::pair<int, int>> edges_, incs_;

    ArrayXX<bool> crossing_;
    OffsetTable off_;
    ArrayXi order_, succ_;
    ArrayXb seen_;
    std::vector<int> keys_;
    std::vector<RingVertex> ring_;
    std::vector<Dart> darts_, dring_;
  };
}  // namespace internal

template <class Key, class Map>
void argsort_bucket(ArrayXi &idxs, internal::OffsetTable &off, const Key &key,
                    const Map &map) {
  return argsort_bucket(idxs, off.off(), key, map);
}
}  // namespace nuri

#endif /* NURI_DESC_SURFACE_H_ */

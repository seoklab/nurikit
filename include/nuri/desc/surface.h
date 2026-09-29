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
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"

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

  struct SasArc {
    double dphi;
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
    ArrayXi tan_off;
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
   * Caps on one sphere with the vertices where their circles cross: the
   * first `m` caps and `k` vertices are live, the rest is capacity. Vertex
   * `v` excuses the caps whose crossing points merged into it.
   */
  struct ArrangementProblem {
    double radius;
    int m, k;
    Matrix3Xd axis;
    ArrayXd cosa, sina;
    ArrayXX<bool> crossing;
    Matrix3Xd reps;
    ArrayXX<bool> excused;
    ArrayXb accessible;

    void reserve(int mcap, int kcap) {
      axis.resize(3, mcap);
      cosa.resize(mcap);
      sina.resize(mcap);
      crossing.resize(mcap, mcap);
      reps.resize(3, kcap);
      excused.resize(kcap, mcap);
      accessible.resize(kcap);
    }
  };

  /**
   * Appends the accessible arcs: `circ` is the cap, `beg` and `end` the
   * vertices, both `k` on a full circle. Returns the accessible area.
   */
  extern double solve_arrangement(const ArrangementProblem &prob,
                                  std::vector<SasArc> &arcs);
}  // namespace internal

template <class Key, class Map>
void argsort_bucket(ArrayXi &idxs, internal::OffsetTable &off, const Key &key,
                    const Map &map) {
  return argsort_bucket(idxs, off.off(), key, map);
}
}  // namespace nuri

#endif /* NURI_DESC_SURFACE_H_ */

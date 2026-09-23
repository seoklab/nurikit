//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_SURFACE_H_
#define NURI_DESC_SURFACE_H_

#include <optional>
#include <utility>

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

  class CSR {
  public:
    using const_iterator = ArrayXi::const_iterator;

    CSR(): off_(ArrayXi::Zero(1)) { }

    CSR(ArrayXi &&adj, ArrayXi &&off) noexcept
        : adj_(std::move(adj)), off_(std::move(off)) { }

    const_iterator begin(int i) const { return adj_.begin() + off_[i]; }

    const_iterator end(int i) const { return adj_.begin() + off_[i + 1]; }

    int offset(int i) const { return off_[i]; }

    int degree(int i) const { return off_[i + 1] - off_[i]; }

    int max_deg() const { return (off_.tail(n()) - off_.head(n())).maxCoeff(); }

    auto nbrs(int i) const { return adj_.segment(off_[i], degree(i)); }

    int eid(const_iterator it) const {
      return static_cast<int>(it - adj_.begin());
    }

    int m() const { return static_cast<int>(adj_.size()); }

    int n() const { return static_cast<int>(off_.size()) - 1; }

  private:
    ArrayXi adj_;
    ArrayXi off_;
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
}  // namespace internal
}  // namespace nuri

#endif /* NURI_DESC_SURFACE_H_ */

//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    ArrayXd convex_areas(const SaPrep &sa, const SasGeometry &geo,
                         const double rp) {
      const auto sar = sa.sar.head(sa.n_active);
      return geo.area.head(sa.n_active) * ((sar - rp) / sar).square();
    }

    struct Beta {
      double angle, sin;
    };

    Beta pick(const bool cond, const Beta &x, const Beta &y) {
      return cond ? x : y;
    }

    struct SaddleRanges {
      Array4d beta;
      E::Array2d integral;
    };

    /**
     * Valid ranges `(lo0, hi0, lo1, hi1)` of the generating arc angle `β`
     * on circle `(i, j)` and the integral `rl (hi − lo) − rp (sin hi − sin
     * lo)` of each. `β` runs from `−θ_i` to `θ_j` with `sin θ = a / R`; on
     * a spindle (`rl < rp`) the part `|β| < β0` lies beyond the axis and is
     * cut out, `β0 = 0` otherwise. Absent parts are zero-width. Endpoints
     * are selected together with their sines, so no sine of a selected
     * angle is ever evaluated.
     */
    SaddleRanges saddle_ranges(const double rl, const double rp,
                               const double a_i, const double a_j,
                               const double sas_i, const double sas_j) {
      const Beta lo { -std::atan2(a_i, rl), -a_i / sas_i },
          hi { std::atan2(a_j, rl), a_j / sas_j };
      const double root = std::sqrt(nuri::max(rp * rp - rl * rl, 0.0));
      const Beta b0 { std::atan2(root, rl), root / rp },
          nb0 { -b0.angle, -b0.sin };

      Beta m = pick(hi.angle < nb0.angle, hi, nb0);
      const Beta below_hi = pick(lo.angle > m.angle, lo, m);
      m = pick(lo.angle > b0.angle, lo, b0);
      const Beta above_lo = pick(hi.angle < m.angle, hi, m);

      SaddleRanges out;
      out.beta << lo.angle, below_hi.angle, above_lo.angle, hi.angle;
      out.integral
          << rl * (below_hi.angle - lo.angle) - rp * (below_hi.sin - lo.sin),
          rl * (hi.angle - above_lo.angle) - rp * (hi.sin - above_lo.sin);
      return out;
    }

    void saddles(SesGeometry &ses, const SaPrep &sa, const SasGeometry &geo,
                 const double rp) {
      const int n_circ = sa.g.offset(sa.n_active);
      ses.saddle_beta.resize(4, n_circ);
      ses.saddle_integral.resize(2, n_circ);
      for (int q = 0; q < n_circ; ++q) {
        const SasCircle &c = geo.circles[q];
        const SaddleRanges sr = saddle_ranges(c.rl, rp, c.a, sa.d[q] - c.a,
                                              sa.sar[c.i], sa.sar[c.j]);
        ses.saddle_beta.col(q) = sr.beta;
        ses.saddle_integral.col(q) = sr.integral;
      }

      ses.saddle_area.resize(geo.n_active_arcs);
      for (int r = 0; r < geo.n_active_arcs; ++r) {
        const SasArc &arc = geo.arcs[r];
        ses.saddle_area[r] =
            arc.dphi * rp * ses.saddle_integral.col(arc.circ).sum();
      }
    }
  }  // namespace

  SesGeometry build_ses(const SaPrep &sa, const SasGeometry &geo,
                        const double rp) {
    SesGeometry ses;
    ses.convex_area = convex_areas(sa, geo, rp);
    saddles(ses, sa, geo, rp);
    ses.face_area = ArrayXd::Constant(geo.probes.n_active,
                                      std::numeric_limits<double>::quiet_NaN());
    ses.face_off = OffsetTable(ArrayXi::Zero(geo.probes.n_active + 1));
    return ses;
  }
}  // namespace internal
}  // namespace nuri

//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    using constants::kTwoPi;

    int lattice_count(const double area, const double density) {
      return static_cast<int>(nuri::max(std::lround(area * density), 1L));
    }

    class DotBuffer {
    public:
      void add(const Vector3d &pt, const Vector3d &nrm, const double area,
               const int atom) {
        pts_.push_back(pt);
        nrm_.push_back(nrm);
        area_.push_back(area);
        atom_.push_back(atom);
      }

      int size() const { return static_cast<int>(area_.size()); }

      void finish(SesDots &out) const {
        const int n = size();
        out.pts.resize(3, n);
        out.nrm.resize(3, n);
        out.area.resize(n);
        out.atom.resize(n);
        for (int k = 0; k < n; ++k) {
          out.pts.col(k) = pts_[k];
          out.nrm.col(k) = nrm_[k];
          out.area[k] = area_[k];
          out.atom[k] = atom_[k];
        }
      }

    private:
      std::vector<Vector3d> pts_, nrm_;
      std::vector<double> area_;
      std::vector<int> atom_;
    };

    /**
     * Lattice directions outside every cap `(axis_k, cosa_k)`: `axis_k · d
     * > cosa_k` for no `k`.
     */
    template <class Axes, class Cosa>
    ArrayXb outside_caps(const Matrix3Xd &dirs, const Axes &axes,
                         const Cosa &cosa) {
      if (axes.cols() == 0)
        return ArrayXb::Constant(dirs.cols(), true);

      const ArrayXXd margin =
          (axes.transpose() * dirs).array().colwise() - cosa;
      return !(margin > 0).colwise().any().transpose();
    }

    class Sampler {
    public:
      Sampler(const SaPrep &sa, const SasGeometry &geo, const SesGeometry &ses,
              const double density)
          : sa_(&sa), geo_(&geo), ses_(&ses), rp_(ses.rp), density_(density) { }

      SesDots run() {
        SesDots out;
        out.rp = rp_;
        out.kind = OffsetTable(3);
        ArrayXi &off = out.kind.off();

        off[0] = 0;
        convex();
        off[1] = buf_.size();
        toroidal();
        off[2] = buf_.size();
        concave();
        off[3] = buf_.size();

        buf_.finish(out);
        out.atom = sa_->order(out.atom);
        out.dropped_area = dropped_;
        return out;
      }

    private:
      const Matrix3Xd &lattice(const int count) {
        auto [it, inserted] = lattices_.try_emplace(count);
        if (inserted)
          it->second = canonical_fibonacci_lattice(count);
        return it->second;
      }

      void convex() {
        const SasCaps &caps = geo_->caps;
        for (int i = 0; i < sa_->n_active; ++i) {
          const double area = ses_->convex_area[i];
          if (area <= 0)
            continue;

          const double r = sa_->sar[i] - rp_;
          const Matrix3Xd &dirs =
              lattice(lattice_count(2 * kTwoPi * r * r, density_));
          const int beg = caps.h.offset(i), m = caps.h.degree(i);
          const ArrayXb keep = outside_caps(dirs, caps.axis.middleCols(beg, m),
                                            caps.cosa.segment(beg, m));
          const int n_kept = static_cast<int>(keep.count());
          if (n_kept == 0) {
            dropped_ += area;
            continue;
          }

          const Vector3d c = sa_->pts.col(i);
          const double w = area / n_kept;
          for (int k = 0; k < dirs.cols(); ++k) {
            if (keep[k])
              buf_.add(c + r * dirs.col(k), dirs.col(k), w, i);
          }
        }
      }

      struct SaddleRow {
        double lo, hi, integral;
      };

      void toroidal() {
        for (int r = 0; r < geo_->n_active_arcs; ++r) {
          const SasArc &arc = geo_->arcs[r];
          const SasCircle &c = geo_->circles[arc.circ];
          const Array4d beta = ses_->saddle_beta.col(arc.circ);
          const E::Array2d integral = ses_->saddle_integral.col(arc.circ);

          if (c.rl >= rp_) {
            saddle_row(arc, c, { beta[0], beta[3], integral.sum() });
          } else {
            saddle_row(arc, c, { beta[0], beta[1], integral[0] });
            saddle_row(arc, c, { beta[2], beta[3], integral[1] });
          }
        }
      }

      /**
       * Rings of equal `β` width, each with `round(a_m ρ)` dots; a row whose
       * rings all round to zero is sampled as one ring, and a row that still
       * rounds to zero is dropped.
       */
      void saddle_row(const SasArc &arc, const SasCircle &c,
                      const SaddleRow &row) {
        const double width = row.hi - row.lo,
                     total = rp_ * arc.dphi * row.integral;
        if (width <= 0)
          return;

        const int k_beta = static_cast<int>(
            nuri::max(std::lround(rp_ * width * std::sqrt(density_)), 1L));
        const double dbeta = width / k_beta;

        ArrayXd sin_edge(k_beta + 1);
        for (int m = 0; m <= k_beta; ++m)
          sin_edge[m] = std::sin(row.lo + m * dbeta);

        ArrayXd beta =
                    row.lo
                    + (ArrayXd::LinSpaced(k_beta, 0, k_beta - 1) + 0.5) * dbeta,
                area =
                    rp_ * arc.dphi
                    * (c.rl * dbeta
                       - rp_ * (sin_edge.tail(k_beta) - sin_edge.head(k_beta)));
        ArrayXi k_phi = area.unaryExpr([&](double a) {
          return static_cast<int>(std::lround(a * density_));
        });
        if (!(k_phi > 0).any()) {
          beta = ArrayXd::Constant(1, 0.5 * (row.lo + row.hi));
          area = ArrayXd::Constant(1, total);
          k_phi = ArrayXi::Constant(
              1, static_cast<int>(std::lround(total * density_)));
        }

        double kept_area = 0;
        for (int m = 0; m < k_phi.size(); ++m)
          kept_area += k_phi[m] > 0 ? area[m] : 0;
        if (!(k_phi > 0).any()) {
          dropped_ += total;
          return;
        }
        const double scale = total / kept_area;

        const int i = c.i, j = c.j;
        const double a_i = c.a, a_j = sa_->d[arc.circ] - c.a,
                     sas_i = sa_->sar[i], sas_j = sa_->sar[j],
                     r_i = sas_i - rp_, r_j = sas_j - rp_;

        for (int m = 0; m < k_phi.size(); ++m) {
          const int k = k_phi[m];
          if (k <= 0)
            continue;

          const double cos_b = std::cos(beta[m]), sin_b = std::sin(beta[m]);
          const double depth_i =
              std::sqrt(r_i * r_i
                        + 2 * rp_ * (sas_i - c.rl * cos_b + a_i * sin_b))
              - r_i;
          const double depth_j =
              std::sqrt(r_j * r_j
                        + 2 * rp_ * (sas_j - c.rl * cos_b - a_j * sin_b))
              - r_j;
          const int owner = depth_i <= depth_j ? i : j;

          const double w = area[m] * scale / k,
                       offset = (0.25 + 0.5 * (m % 2)) / k;
          for (int n = 0; n < k; ++n) {
            const double phi =
                arc.phi + (static_cast<double>(n) / k + offset) * arc.dphi;
            const Vector3d radial = std::cos(phi) * c.e1 + std::sin(phi) * c.e2,
                           q = c.cntr + c.rl * radial,
                           inward = -cos_b * radial + sin_b * c.axis;
            buf_.add(q + rp_ * inward, -inward, w, owner);
          }
        }
      }

      void concave() {
        const SasProbes &probes = geo_->probes;
        if (probes.n_active == 0)
          return;

        const Matrix3Xd dirs = canonical_fibonacci_lattice(
            lattice_count(2 * kTwoPi * rp_ * rp_, density_));
        for (int p = 0; p < probes.n_active; ++p) {
          const double area = ses_->face_area[p];
          if (area <= 0)
            continue;

          const int beg = ses_->face_off[p], m = ses_->face_off.degree(p);
          const ArrayXb keep = outside_caps(dirs,
                                            ses_->face_axis.middleCols(beg, m),
                                            ses_->face_cosa.segment(beg, m));
          const int n_kept = static_cast<int>(keep.count());
          if (n_kept == 0) {
            dropped_ += area;
            continue;
          }

          const Vector3d pos = probes.pos.col(p);
          const auto atoms = probes.atoms.nbrs(p);
          const int na = static_cast<int>(atoms.size());
          Matrix3Xd contacts(3, na);
          ArrayXd big(na), small(na);
          for (int a = 0; a < na; ++a) {
            contacts.col(a) = (sa_->pts.col(atoms[a]) - pos).normalized();
            big[a] = sa_->sar[atoms[a]];
            small[a] = big[a] - rp_;
          }

          const double w = area / n_kept;
          for (int k = 0; k < dirs.cols(); ++k) {
            if (!keep[k])
              continue;

            const Vector3d d = dirs.col(k);
            const ArrayXd depth =
                (small.square()
                 + 2 * rp_ * big * (1 - (contacts.transpose() * d).array()))
                    .sqrt()
                - small;
            int best;
            depth.minCoeff(&best);
            buf_.add(pos + rp_ * d, -d, w, atoms[best]);
          }
        }
      }

      const SaPrep *sa_;
      const SasGeometry *geo_;
      const SesGeometry *ses_;
      double rp_, density_;

      absl::flat_hash_map<int, Matrix3Xd> lattices_;
      DotBuffer buf_;
      double dropped_ = 0;
    };
  }  // namespace

  SesDots SesDots::subset(const ArrayXb &keep) const {
    ABSL_DCHECK_EQ(keep.size(), n());

    SesDots out;
    out.rp = rp;
    out.dropped_area = dropped_area;

    const int m = static_cast<int>(keep.count());
    out.pts.resize(3, m);
    out.nrm.resize(3, m);
    out.area.resize(m);
    out.atom.resize(m);
    out.kind = OffsetTable(kind.size());

    int w = 0;
    out.kind.off()[0] = 0;
    for (int b = 0; b < kind.size(); ++b) {
      for (int i = kind[b]; i < kind[b + 1]; ++i) {
        if (!keep[i])
          continue;
        out.pts.col(w) = pts.col(i);
        out.nrm.col(w) = nrm.col(i);
        out.area[w] = area[i];
        out.atom[w] = atom[i];
        ++w;
      }
      out.kind.off()[b + 1] = w;
    }
    return out;
  }

  SesDots sample_ses(const SaPrep &sa, const SasGeometry &geo,
                     const SesGeometry &ses, const double density) {
    ABSL_DCHECK_GT(density, 0);
    return Sampler(sa, geo, ses, density).run();
  }
}  // namespace internal
}  // namespace nuri

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
      void reserve(const int n) {
        pts_.reserve(n);
        nrm_.reserve(n);
        area_.reserve(n);
        atom_.reserve(n);
      }

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
     * Buffers of one `sample_ses` call, sized once for the largest lattice,
     * cap list, probe contact list and saddle row; `lattices` caches the
     * Fibonacci lattice per dot count.
     */
    struct SampleScratch {
      absl::flat_hash_map<int, Matrix3Xd> lattices;
      ArrayXb outside;
      MatrixX3d axt;
      Matrix3Xd contacts;
      ArrayXd big, r, depth;
      ArrayXd sin_edge, beta, area;
      ArrayXi k_phi;
      DotBuffer dots;
      double dropped = 0;
    };

    SampleScratch sample_scratch(const SurfaceSpheres &ss,
                                 const SasGeometry &geo, const SesGeometry &ses,
                                 const double density) {
      const double rp = ss.rp, rmax = ss.n_active == 0
                                          ? 0
                                          : ss.r.head(ss.n_active).maxCoeff();
      const int ndir =
                    nuri::max(lattice_count(2 * kTwoPi * rmax * rmax, density),
                              lattice_count(2 * kTwoPi * rp * rp, density)),
                mcap = nuri::max(geo.caps.h.max_deg(), ses.face_off.max_deg()),
                ncontact = geo.probes.atoms.max_deg(),
                k_beta = static_cast<int>(nuri::max(
                    std::lround(rp * constants::kPi * std::sqrt(density)), 1L));
      const double total_area =
          ses.convex_area.sum() + ses.saddle_area.sum() + ses.face_area.sum();

      SampleScratch scratch;
      scratch.outside.resize(ndir);
      scratch.axt.resize(mcap, 3);
      scratch.contacts.resize(3, ncontact);
      scratch.big.resize(ncontact);
      scratch.r.resize(ncontact);
      scratch.depth.resize(ncontact);
      scratch.sin_edge.resize(k_beta + 1);
      scratch.beta.resize(k_beta);
      scratch.area.resize(k_beta);
      scratch.k_phi.resize(k_beta);
      scratch.dots.reserve(
          static_cast<int>(nuri::max(std::lround(total_area * density), 0L)));
      return scratch;
    }

    /**
     * The cached lattice of `count` dots; valid until the next call.
     */
    const Matrix3Xd &lattice(SampleScratch &scratch, const int count) {
      auto [it, inserted] = scratch.lattices.try_emplace(count);
      if (inserted)
        it->second = canonical_fibonacci_lattice(count);
      return it->second;
    }

    /**
     * Lattice directions outside every cap `(axis_c, cosa_c)`: `axis_c · d
     * > cosa_c` for no `c`.
     */
    auto outside_caps(SampleScratch &scratch, const Matrix3Xd &dirs,
                      ConstBlock<Matrix3Xd, 3, E::Dynamic, true> axes,
                      ConstVecBlock<ArrayXd> cosa) {
      const int n = static_cast<int>(dirs.cols()),
                m = static_cast<int>(axes.cols());
      auto outside = scratch.outside.head(n);
      if (m == 0) {
        outside.setConstant(true);
        return outside;
      }

      auto axm = take_buffer(scratch.axt, m, 3);
      axm = axes.transpose();
      const auto ax = axm.col(0), ay = axm.col(1), az = axm.col(2);
      for (int k = 0; k < n; ++k) {
        const double dx = dirs(0, k), dy = dirs(1, k), dz = dirs(2, k);
        bool covered = false;
        for (int c = 0; c < m && !covered; ++c)
          covered = ax[c] * dx + ay[c] * dy + az[c] * dz - cosa[c] > 0;
        outside[k] = !covered;
      }
      return outside;
    }

    void sample_convex(const SurfaceSpheres &ss, const SasGeometry &geo,
                       const SesGeometry &ses, const double density,
                       SampleScratch &scratch) {
      const SasCaps &caps = geo.caps;
      for (int i = 0; i < ss.n_active; ++i) {
        const double area = ses.convex_area[i];
        if (area <= 0)
          continue;

        const double r = ss.r[i];
        const Matrix3Xd &dirs =
            lattice(scratch, lattice_count(2 * kTwoPi * r * r, density));
        const int beg = caps.h.offset(i), m = caps.h.degree(i);
        const auto keep = outside_caps(scratch, dirs,
                                       caps.axis.middleCols(beg, m),
                                       caps.cosa.segment(beg, m));
        const int n_kept = static_cast<int>(keep.count());
        if (n_kept == 0) {
          scratch.dropped += area;
          continue;
        }

        const Vector3d c = ss.pts.col(i);
        const double w = area / n_kept;
        for (int k = 0; k < dirs.cols(); ++k) {
          if (keep[k])
            scratch.dots.add(c + r * dirs.col(k), dirs.col(k), w, i);
        }
      }
    }

    struct SaddleRow {
      double lo, hi, integral;
    };

    /**
     * Rings of equal `β` width, each with `round(a_m ρ)` dots; a row whose
     * rings all round to zero is sampled as one ring, and a row that still
     * rounds to zero is dropped.
     */
    void saddle_row(const SurfaceSpheres &ss, const SasArc &arc,
                    const SasCircle &c, const SaddleRow &row,
                    const double density, SampleScratch &scratch) {
      const double rp = ss.rp;
      const double width = row.hi - row.lo,
                   total = rp * arc.dphi * row.integral;
      if (width <= 0)
        return;

      const int k_beta = static_cast<int>(
          nuri::max(std::lround(rp * width * std::sqrt(density)), 1L));
      const double dbeta = width / k_beta;

      auto sin_edge = scratch.sin_edge.head(k_beta + 1);
      for (int m = 0; m <= k_beta; ++m)
        sin_edge[m] = std::sin(row.lo + m * dbeta);

      auto beta = scratch.beta.head(k_beta), area = scratch.area.head(k_beta);
      auto k_phi = scratch.k_phi.head(k_beta);
      beta = row.lo + (ArrayXd::LinSpaced(k_beta, 0, k_beta - 1) + 0.5) * dbeta;
      area = rp * arc.dphi
             * (c.rl * dbeta
                - rp * (sin_edge.segment(1, k_beta) - sin_edge.head(k_beta)));
      k_phi = area.unaryExpr(
          [&](double a) { return static_cast<int>(std::lround(a * density)); });
      int rows = k_beta;
      if (!(k_phi > 0).any()) {
        rows = 1;
        beta[0] = 0.5 * (row.lo + row.hi);
        area[0] = total;
        k_phi[0] = static_cast<int>(std::lround(total * density));
      }

      double kept_area = 0;
      for (int m = 0; m < rows; ++m)
        kept_area += k_phi[m] > 0 ? area[m] : 0;
      if (!(k_phi.head(rows) > 0).any()) {
        scratch.dropped += total;
        return;
      }
      const double scale = total / kept_area;

      const int i = c.i, j = c.j;
      const double a_i = c.a, a_j = ss.d[arc.circ] - c.a, sas_i = ss.sar[i],
                   sas_j = ss.sar[j], r_i = sas_i - rp, r_j = sas_j - rp;

      for (int m = 0; m < rows; ++m) {
        const int k = k_phi[m];
        if (k <= 0)
          continue;

        const double cos_b = std::cos(beta[m]), sin_b = std::sin(beta[m]);
        const double depth_i =
            std::sqrt(r_i * r_i + 2 * rp * (sas_i - c.rl * cos_b + a_i * sin_b))
            - r_i;
        const double depth_j =
            std::sqrt(r_j * r_j + 2 * rp * (sas_j - c.rl * cos_b - a_j * sin_b))
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
          scratch.dots.add(q + rp * inward, -inward, w, owner);
        }
      }
    }

    void sample_toroidal(const SurfaceSpheres &ss, const SasGeometry &geo,
                         const SesGeometry &ses, const double density,
                         SampleScratch &scratch) {
      for (int r = 0; r < geo.n_active_arcs; ++r) {
        const SasArc &arc = geo.arcs[r];
        const SasCircle &c = geo.circles[arc.circ];
        const Array4d beta = ses.saddle_beta.col(arc.circ);
        const E::Array2d integral = ses.saddle_integral.col(arc.circ);

        if (beta[1] < beta[2]) {
          saddle_row(ss, arc, c, { beta[0], beta[1], integral[0] }, density,
                     scratch);
          saddle_row(ss, arc, c, { beta[2], beta[3], integral[1] }, density,
                     scratch);
        } else {
          saddle_row(ss, arc, c, { beta[0], beta[3], integral.sum() }, density,
                     scratch);
        }
      }
    }

    void sample_concave(const SurfaceSpheres &ss, const SasGeometry &geo,
                        const SesGeometry &ses, const double density,
                        SampleScratch &scratch) {
      const SasProbes &probes = geo.probes;
      if (probes.n_active == 0)
        return;

      const double rp = ss.rp;
      const Matrix3Xd dirs = canonical_fibonacci_lattice(
          lattice_count(2 * kTwoPi * rp * rp, density));
      for (int p = 0; p < probes.n_active; ++p) {
        const double area = ses.face_area[p];
        if (area <= 0)
          continue;

        const int beg = ses.face_off[p], m = ses.face_off.degree(p);
        const auto keep = outside_caps(scratch, dirs,
                                       ses.face_axis.middleCols(beg, m),
                                       ses.face_cosa.segment(beg, m));
        const int n_kept = static_cast<int>(keep.count());
        if (n_kept == 0) {
          scratch.dropped += area;
          continue;
        }

        const Vector3d pos = probes.pos.col(p);
        const auto atoms = probes.atoms.nbrs(p);
        const int na = static_cast<int>(atoms.size());
        auto contacts = scratch.contacts.leftCols(na);
        auto big = scratch.big.head(na), small = scratch.r.head(na),
             depth = scratch.depth.head(na);
        for (int a = 0; a < na; ++a) {
          contacts.col(a) = (ss.pts.col(atoms[a]) - pos).normalized();
          big[a] = ss.sar[atoms[a]];
          small[a] = ss.r[atoms[a]];
        }

        const double w = area / n_kept;
        for (int k = 0; k < dirs.cols(); ++k) {
          if (!keep[k])
            continue;

          const Vector3d d = dirs.col(k);
          depth = (small.square()
                   + 2 * rp * big * (1 - (contacts.transpose() * d).array()))
                      .sqrt()
                  - small;
          int best;
          depth.minCoeff(&best);
          scratch.dots.add(pos + rp * d, -d, w, atoms[best]);
        }
      }
    }
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

  SesDots sample_ses(const SurfaceSpheres &ss, const SasGeometry &geo,
                     const SesGeometry &ses, const double density) {
    ABSL_DCHECK_GT(density, 0);

    SampleScratch scratch = sample_scratch(ss, geo, ses, density);

    SesDots out;
    out.rp = ss.rp;
    out.kind = OffsetTable(3);
    ArrayXi &off = out.kind.off();

    off[0] = 0;
    sample_convex(ss, geo, ses, density, scratch);
    off[1] = scratch.dots.size();
    sample_toroidal(ss, geo, ses, density, scratch);
    off[2] = scratch.dots.size();
    sample_concave(ss, geo, ses, density, scratch);
    off[3] = scratch.dots.size();

    scratch.dots.finish(out);
    out.atom = ss.order(out.atom);
    out.dropped_area = scratch.dropped;
    return out;
  }
}  // namespace internal
}  // namespace nuri

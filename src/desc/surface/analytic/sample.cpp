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

    void resize_dots(SesDots &out, const int n) {
      out.pts.conservativeResize(3, n);
      out.nrm.conservativeResize(3, n);
      out.area.conservativeResize(n);
      out.atom.conservativeResize(n);
    }

    /**
     * Room for `need` dots past the first `n`; the capacity doubles when the
     * lattice subsets overshoot the area estimate.
     */
    void room(SesDots &out, const int n, const int need) {
      const int cap = out.n();
      if (n + need > cap)
        resize_dots(out, nuri::max(2 * cap, n + need));
    }

    /**
     * The expected dot count plus one per patch row, the margin of a patch
     * that rounds up.
     */
    int dot_capacity(const SasGeometry &geo, const SesGeometry &ses,
                     const double density) {
      const double total_area =
          ses.convex_area.sum() + ses.saddle_area.sum() + ses.face_area.sum();
      const int rows = static_cast<int>(ses.convex_area.size())
                       + 2 * geo.n_active_arcs + geo.probes.n_active;
      return static_cast<int>(nuri::max(std::lround(total_area * density), 0L))
             + rows;
    }

    /**
     * Buffers of one `sample_ses` call, sized once for the largest lattice,
     * cap list, probe contact list and saddle row; `lattices` caches the
     * Fibonacci lattice per dot count and `n_dots` is the write cursor into
     * the output.
     */
    struct SampleScratch {
      absl::flat_hash_map<int, Matrix3Xd> lattices;
      ArrayXb outside;
      MatrixX3d axt;
      Matrix3Xd kept, contacts;
      ArrayXd big, r;
      MatrixXd depth;
      ArrayXd sin_edge, beta, area;
      ArrayXi k_phi;
      int n_dots = 0;
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

      SampleScratch scratch;
      scratch.outside.resize(ndir);
      scratch.axt.resize(mcap, 3);
      scratch.kept.resize(3, ndir);
      scratch.contacts.resize(3, ncontact);
      scratch.big.resize(ncontact);
      scratch.r.resize(ncontact);
      scratch.depth.resize(ncontact, ndir);
      scratch.sin_edge.resize(k_beta + 1);
      scratch.beta.resize(k_beta);
      scratch.area.resize(k_beta);
      scratch.k_phi.resize(k_beta);
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
                       SampleScratch &scratch, SesDots &out) {
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

        room(out, scratch.n_dots, n_kept);
        const int b = scratch.n_dots;
        auto nrm = out.nrm.middleCols(b, n_kept);
        for (int k = 0, w = 0; k < dirs.cols(); ++k) {
          if (keep[k])
            nrm.col(w++) = dirs.col(k);
        }
        out.pts.middleCols(b, n_kept) = (r * nrm).colwise() + ss.pts.col(i);
        out.area.segment(b, n_kept).setConstant(area / n_kept);
        out.atom.segment(b, n_kept).setConstant(i);
        scratch.n_dots += n_kept;
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
                    const double density, SampleScratch &scratch,
                    SesDots &out) {
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

        room(out, scratch.n_dots, k);
        const int b = scratch.n_dots;
        out.area.segment(b, k).setConstant(area[m] * scale / k);
        out.atom.segment(b, k).setConstant(owner);
        const double offset = (0.25 + 0.5 * (m % 2)) / k;
        for (int n = 0; n < k; ++n) {
          const double phi =
              arc.phi + (static_cast<double>(n) / k + offset) * arc.dphi;
          const Vector3d radial = std::cos(phi) * c.e1 + std::sin(phi) * c.e2,
                         q = c.cntr + c.rl * radial,
                         inward = -cos_b * radial + sin_b * c.axis;
          out.pts.col(b + n) = q + rp * inward;
          out.nrm.col(b + n) = -inward;
        }
        scratch.n_dots += k;
      }
    }

    void sample_toroidal(const SurfaceSpheres &ss, const SasGeometry &geo,
                         const SesGeometry &ses, const double density,
                         SampleScratch &scratch, SesDots &out) {
      for (int r = 0; r < geo.n_active_arcs; ++r) {
        const SasArc &arc = geo.arcs[r];
        const SasCircle &c = geo.circles[arc.circ];
        const Array4d beta = ses.saddle_beta.col(arc.circ);
        const E::Array2d integral = ses.saddle_integral.col(arc.circ);

        if (beta[1] < beta[2]) {
          saddle_row(ss, arc, c, { beta[0], beta[1], integral[0] }, density,
                     scratch, out);
          saddle_row(ss, arc, c, { beta[2], beta[3], integral[1] }, density,
                     scratch, out);
        } else {
          saddle_row(ss, arc, c, { beta[0], beta[3], integral.sum() }, density,
                     scratch, out);
        }
      }
    }

    void sample_concave(const SurfaceSpheres &ss, const SasGeometry &geo,
                        const SesGeometry &ses, const double density,
                        SampleScratch &scratch, SesDots &out) {
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

        auto kept = take_buffer(scratch.kept, 3, n_kept);
        for (int k = 0, w = 0; k < dirs.cols(); ++k) {
          if (keep[k])
            kept.col(w++) = dirs.col(k);
        }

        const Vector3d pos = probes.pos.col(p);
        const auto atoms = probes.atoms.nbrs(p);
        const int na = static_cast<int>(atoms.size());
        auto contacts = scratch.contacts.leftCols(na);
        auto big = scratch.big.head(na), small = scratch.r.head(na);
        for (int a = 0; a < na; ++a) {
          contacts.col(a) = (ss.pts.col(atoms[a]) - pos).normalized();
          big[a] = ss.sar[atoms[a]];
          small[a] = ss.r[atoms[a]];
        }

        auto depth = take_buffer(scratch.depth, na, n_kept);
        depth.noalias() = contacts.transpose() * kept;
        depth.array() =
            (((1 - depth.array()).colwise() * (2 * rp * big)).colwise()
             + small.square())
                .sqrt()
                .colwise()
            - small;

        room(out, scratch.n_dots, n_kept);
        const int b = scratch.n_dots;
        out.pts.middleCols(b, n_kept) = (rp * kept).colwise() + pos;
        out.nrm.middleCols(b, n_kept) = -kept;
        out.area.segment(b, n_kept).setConstant(area / n_kept);
        for (int k = 0; k < n_kept; ++k) {
          int best;
          depth.col(k).minCoeff(&best);
          out.atom[b + k] = atoms[best];
        }
        scratch.n_dots += n_kept;
      }
    }
  }  // namespace

  void SesDots::keep(const ArrayXb &mask) {
    ABSL_DCHECK_EQ(mask.size(), n());

    int w = 0;
    for (int b = 0, beg = 0; b < kind.size(); ++b) {
      const int end = kind[b + 1];
      for (int i = beg; i < end; ++i) {
        if (!mask[i])
          continue;
        pts.col(w) = pts.col(i);
        nrm.col(w) = nrm.col(i);
        area[w] = area[i];
        atom[w] = atom[i];
        ++w;
      }
      kind.off()[b + 1] = w;
      beg = end;
    }
    resize_dots(*this, w);
  }

  SesDots SesDots::subset(const ArrayXb &mask) const {
    ABSL_DCHECK_EQ(mask.size(), n());

    SesDots out;
    out.rp = rp;
    out.dropped_area = dropped_area;

    const int m = static_cast<int>(mask.count());
    out.pts.resize(3, m);
    out.nrm.resize(3, m);
    out.area.resize(m);
    out.atom.resize(m);
    out.kind = OffsetTable(kind.size());

    int w = 0;
    out.kind.off()[0] = 0;
    for (int b = 0; b < kind.size(); ++b) {
      for (int i = kind[b]; i < kind[b + 1]; ++i) {
        if (!mask[i])
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
    resize_dots(out, dot_capacity(geo, ses, density));
    ArrayXi &off = out.kind.off();

    off[0] = 0;
    sample_convex(ss, geo, ses, density, scratch, out);
    off[1] = scratch.n_dots;
    sample_toroidal(ss, geo, ses, density, scratch, out);
    off[2] = scratch.n_dots;
    sample_concave(ss, geo, ses, density, scratch, out);
    off[3] = scratch.n_dots;

    resize_dots(out, scratch.n_dots);
    out.atom = ss.order(out.atom);
    out.dropped_area = scratch.dropped;
    return out;
  }
}  // namespace internal
}  // namespace nuri

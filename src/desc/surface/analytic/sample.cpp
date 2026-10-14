//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <utility>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/utils.h"

namespace nuri {
SurfaceDots::SurfaceDots(const SurfaceKind kind, const double rp,
                         Matrix3Xd &&pts, Matrix3Xd &&nrm, ArrayXd &&area,
                         ArrayXi &&atom, internal::OffsetTable &&patch,
                         const double dropped_area)
    : pts_(std::move(pts)), nrm_(std::move(nrm)), area_(std::move(area)),
      atom_(std::move(atom)), patch_(std::move(patch)), kind_(kind), rp_(rp),
      dropped_area_(dropped_area) { }

Matrix3Xd SurfaceDots::probes() const {
  if (kind_ == SurfaceKind::kSas)
    return pts_;
  return pts_ + rp_ * nrm_;
}

void SurfaceDots::keep(const ArrayXb &mask) {
  ABSL_DCHECK_EQ(mask.size(), n());

  int w = 0;
  for (int b = 0, beg = 0; b < patch_.size(); ++b) {
    const int end = patch_[b + 1];
    for (int i = beg; i < end; ++i) {
      if (!mask[i])
        continue;
      pts_.col(w) = pts_.col(i);
      nrm_.col(w) = nrm_.col(i);
      area_[w] = area_[i];
      atom_[w] = atom_[i];
      ++w;
    }
    patch_.off()[b + 1] = w;
    beg = end;
  }
  pts_.conservativeResize(3, w);
  nrm_.conservativeResize(3, w);
  area_.conservativeResize(w);
  atom_.conservativeResize(w);
}

SurfaceDots SurfaceDots::subset(const ArrayXb &mask) const {
  ABSL_DCHECK_EQ(mask.size(), n());

  const int m = static_cast<int>(mask.count());
  Matrix3Xd pts(3, m), nrm(3, m);
  ArrayXd area(m);
  ArrayXi atom(m);
  internal::OffsetTable patch(patch_.size());

  int w = 0;
  patch.off()[0] = 0;
  for (int b = 0; b < patch_.size(); ++b) {
    for (int i = patch_[b]; i < patch_[b + 1]; ++i) {
      if (!mask[i])
        continue;
      pts.col(w) = pts_.col(i);
      nrm.col(w) = nrm_.col(i);
      area[w] = area_[i];
      atom[w] = atom_[i];
      ++w;
    }
    patch.off()[b + 1] = w;
  }
  return { kind_,
           rp_,
           std::move(pts),
           std::move(nrm),
           std::move(area),
           std::move(atom),
           std::move(patch),
           dropped_area_ };
}

namespace {
  using constants::kTwoPi;
  using internal::RingArc;
  using internal::SampleScratch;
  using internal::SasCircle;
  using internal::SurfaceGeometry;

  int lattice_count(const double area, const double density) {
    return static_cast<int>(nuri::max(std::lround(area * density), 1L));
  }

  struct DotArrays {
    Matrix3Xd pts, nrm;
    ArrayXd area;
    ArrayXi atom;
  };

  void resize_dots(DotArrays &out, const int n) {
    out.pts.conservativeResize(3, n);
    out.nrm.conservativeResize(3, n);
    out.area.conservativeResize(n);
    out.atom.conservativeResize(n);
  }

  /**
   * Room for `need` dots past the first `n`; the capacity doubles when the
   * lattice subsets overshoot the area estimate.
   */
  void room(DotArrays &out, const int n, const int need) {
    const int cap = static_cast<int>(out.area.size());
    if (n + need > cap)
      resize_dots(out, nuri::max(2 * cap, n + need));
  }

  /**
   * The expected dot count plus one per patch row, the margin of a patch
   * that rounds up.
   */
  int dot_capacity(const SurfaceGeometry &geo, const double density) {
    const int rows = static_cast<int>(geo.convex().area.size())
                     + 2 * static_cast<int>(geo.toroidal().arcs.size())
                     + static_cast<int>(geo.concave().area.size());
    return static_cast<int>(
               nuri::max(std::lround(geo.total_area() * density), 0L))
           + rows;
  }

  template <class AL>
  void grow(AL &buf, const int n) {
    if (buf.size() < n)
      buf.resize(n);
  }

  template <class ML>
  void grow(ML &buf, const int rows, const int cols) {
    if (buf.rows() < rows || buf.cols() < cols)
      buf.resize(nuri::max(static_cast<int>(buf.rows()), rows),
                 nuri::max(static_cast<int>(buf.cols()), cols));
  }

  /**
   * Grows every buffer to the largest lattice, cap list, probe contact list
   * and saddle row of `geo`; storage only grows across calls.
   */
  void size_scratch(SampleScratch &scratch, const SurfaceGeometry &geo,
                    const double density) {
    const int n_active = static_cast<int>(geo.convex().area.size());
    const double rp = geo.rp(),
                 rmax = n_active == 0 ? 0
                                      : geo.radii().head(n_active).maxCoeff();
    const int ndir = nuri::max(lattice_count(2 * kTwoPi * rmax * rmax, density),
                               lattice_count(2 * kTwoPi * rp * rp, density)),
              mcap = nuri::max(geo.convex().caps.off.max_deg(),
                               geo.concave().caps.off.max_deg()),
              ncontact = geo.concave().atoms.max_deg(),
              k_beta = static_cast<int>(nuri::max(
                  std::lround(rp * constants::kPi * std::sqrt(density)), 1L));

    grow(scratch.outside, ndir);
    grow(scratch.axt, mcap, 3);
    grow(scratch.kept, 3, ndir);
    grow(scratch.contacts, 3, ncontact);
    grow(scratch.big, ncontact);
    grow(scratch.r, ncontact);
    grow(scratch.depth, ncontact, ndir);
    grow(scratch.sin_edge, k_beta + 1);
    grow(scratch.beta, k_beta);
    grow(scratch.area, k_beta);
    grow(scratch.k_phi, k_beta);
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
   * Lattice directions outside every cap `(axis_c, cosa_c)`: `dot(axis_c, d)
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

  void sample_convex(const SurfaceGeometry &geo, const double density,
                     SampleScratch &scratch, DotArrays &out, int &n_dots,
                     double &dropped) {
    const internal::ConvexPatches &convex = geo.convex();
    const internal::CapTable &caps = convex.caps;
    const ArrayXd &radii = geo.radii();
    for (int i = 0; i < convex.area.size(); ++i) {
      const double area = convex.area[i];
      if (area <= 0)
        continue;

      const double r = radii[i];
      const Matrix3Xd &dirs =
          lattice(scratch, lattice_count(2 * kTwoPi * r * r, density));
      const int beg = caps.off[i], m = caps.off.degree(i);
      const auto keep = outside_caps(scratch, dirs,
                                     caps.axis.middleCols(beg, m),
                                     caps.cosa.segment(beg, m));
      const int n_kept = static_cast<int>(keep.count());
      if (n_kept == 0) {
        dropped += area;
        continue;
      }

      room(out, n_dots, n_kept);
      const int b = n_dots;
      auto nrm = out.nrm.middleCols(b, n_kept);
      for (int k = 0, w = 0; k < dirs.cols(); ++k) {
        if (keep[k])
          nrm.col(w++) = dirs.col(k);
      }
      out.pts.middleCols(b, n_kept) = (r * nrm).colwise() + geo.pts().col(i);
      out.area.segment(b, n_kept).setConstant(area / n_kept);
      out.atom.segment(b, n_kept).setConstant(i);
      n_dots += n_kept;
    }
  }

  struct SaddleRow {
    double lo, hi, integral;
  };

  /**
   * Rings of equal `beta` width, each with `round(a_m rho)` dots; a row whose
   * rings all round to zero is sampled as one ring, and a row that still
   * rounds to zero is dropped.
   */
  void saddle_row(const SurfaceGeometry &geo, const RingArc &arc,
                  const SasCircle &c, const SaddleRow &row,
                  const double density, SampleScratch &scratch, DotArrays &out,
                  int &n_dots, double &dropped) {
    const double rp = geo.rp();
    const double width = row.hi - row.lo, total = rp * arc.dphi * row.integral;
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
      dropped += total;
      return;
    }
    const double scale = total / kept_area;

    const int i = c.i, j = c.j;
    const ArrayXd &sar = geo.sar();
    const double a_i = c.aij[0], a_j = c.aij[1], sas_i = sar[i], sas_j = sar[j],
                 r_i = sas_i - rp, r_j = sas_j - rp;

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

      room(out, n_dots, k);
      const int b = n_dots;
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
      n_dots += k;
    }
  }

  void sample_toroidal(const SurfaceGeometry &geo, const double density,
                       SampleScratch &scratch, DotArrays &out, int &n_dots,
                       double &dropped) {
    const internal::ToroidalPatches &tor = geo.toroidal();
    for (const RingArc &arc: tor.arcs) {
      const SasCircle &c = tor.circles[arc.circ];
      const Array4d beta = tor.beta.col(arc.circ);
      const E::Array2d integral = tor.integral.col(arc.circ);

      if (beta[1] < beta[2]) {
        saddle_row(geo, arc, c, { beta[0], beta[1], integral[0] }, density,
                   scratch, out, n_dots, dropped);
        saddle_row(geo, arc, c, { beta[2], beta[3], integral[1] }, density,
                   scratch, out, n_dots, dropped);
      } else {
        saddle_row(geo, arc, c, { beta[0], beta[3], integral.sum() }, density,
                   scratch, out, n_dots, dropped);
      }
    }
  }

  void sample_concave(const SurfaceGeometry &geo, const double density,
                      SampleScratch &scratch, DotArrays &out, int &n_dots,
                      double &dropped) {
    const internal::ConcavePatches &concave = geo.concave();
    const int n_probes = static_cast<int>(concave.area.size());
    if (n_probes == 0)
      return;

    const double rp = geo.rp();
    const ArrayXd &sar = geo.sar(), &r = geo.r();
    const Matrix3Xd &dirs =
        lattice(scratch, lattice_count(2 * kTwoPi * rp * rp, density));
    for (int p = 0; p < n_probes; ++p) {
      const double area = concave.area[p];
      if (area <= 0)
        continue;

      const int beg = concave.caps.off[p], m = concave.caps.off.degree(p);
      const auto keep = outside_caps(scratch, dirs,
                                     concave.caps.axis.middleCols(beg, m),
                                     concave.caps.cosa.segment(beg, m));
      const int n_kept = static_cast<int>(keep.count());
      if (n_kept == 0) {
        dropped += area;
        continue;
      }

      auto kept = take_buffer(scratch.kept, 3, n_kept);
      for (int k = 0, w = 0; k < dirs.cols(); ++k) {
        if (keep[k])
          kept.col(w++) = dirs.col(k);
      }

      const Vector3d pos = concave.pos.col(p);
      const auto atoms = concave.atoms.nbrs(p);
      const int na = static_cast<int>(atoms.size());
      auto contacts = scratch.contacts.leftCols(na);
      auto big = scratch.big.head(na), small = scratch.r.head(na);
      for (int a = 0; a < na; ++a) {
        contacts.col(a) = (geo.pts().col(atoms[a]) - pos).normalized();
        big[a] = sar[atoms[a]];
        small[a] = r[atoms[a]];
      }

      auto depth = take_buffer(scratch.depth, na, n_kept);
      depth.noalias() = contacts.transpose() * kept;
      depth.array() =
          (((1 - depth.array()).colwise() * (2 * rp * big)).colwise()
           + small.square())
              .sqrt()
              .colwise()
          - small;

      room(out, n_dots, n_kept);
      const int b = n_dots;
      out.pts.middleCols(b, n_kept) = (rp * kept).colwise() + pos;
      out.nrm.middleCols(b, n_kept) = -kept;
      out.area.segment(b, n_kept).setConstant(area / n_kept);
      for (int k = 0; k < n_kept; ++k) {
        int best;
        depth.col(k).minCoeff(&best);
        out.atom[b + k] = atoms[best];
      }
      n_dots += n_kept;
    }
  }
}  // namespace

SurfaceSampler::SurfaceSampler(const double density): density_(density) {
  ABSL_DCHECK_GT(density, 0);
}

SurfaceDots SurfaceSampler::sample(const SurfaceGeometry &geo) {
  size_scratch(scratch_, geo, density_);

  DotArrays out;
  resize_dots(out, dot_capacity(geo, density_));
  internal::OffsetTable patch(3);
  ArrayXi &off = patch.off();

  int n_dots = 0;
  double dropped = 0;
  off[0] = 0;
  sample_convex(geo, density_, scratch_, out, n_dots, dropped);
  off[1] = n_dots;
  sample_toroidal(geo, density_, scratch_, out, n_dots, dropped);
  off[2] = n_dots;
  sample_concave(geo, density_, scratch_, out, n_dots, dropped);
  off[3] = n_dots;

  resize_dots(out, n_dots);
  out.atom = geo.order()(out.atom);
  return { geo.kind(),          geo.rp(),
           std::move(out.pts),  std::move(out.nrm),
           std::move(out.area), std::move(out.atom),
           std::move(patch),    dropped };
}
}  // namespace nuri

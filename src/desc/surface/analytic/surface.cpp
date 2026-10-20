//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/desc/surface.h"

#include <optional>
#include <utility>

#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "stages.h"
#include "nuri/core/container/csr.h"

namespace nuri {
namespace internal {
  SurfaceGeometry::SurfaceGeometry(const SurfaceKind kind, const double rp,
                                   Matrix3Xd &&pts, ArrayXd &&sar, ArrayXd &&r,
                                   ArrayXi &&order, ConvexPatches &&convex,
                                   ToroidalPatches &&toroidal,
                                   ConcavePatches &&concave)
      : kind_(kind), rp_(rp), pts_(std::move(pts)), sar_(std::move(sar)),
        r_(std::move(r)), order_(std::move(order)), convex_(std::move(convex)),
        toroidal_(std::move(toroidal)), concave_(std::move(concave)) { }

  const ArrayXd &SurfaceGeometry::area(const SurfacePatch patch) const {
    switch (patch) {
    case SurfacePatch::kConvex:
      return convex_.area;
    case SurfacePatch::kToroidal:
      return toroidal_.area;
    case SurfacePatch::kConcave:
      return concave_.area;
    }
    ABSL_UNREACHABLE();
  }

  double SurfaceGeometry::total_area(const SurfacePatch patch) const {
    return area(patch).sum();
  }

  double SurfaceGeometry::total_area() const {
    return convex_.area.sum() + toroidal_.area.sum() + concave_.area.sum();
  }
}  // namespace internal

namespace {
  using internal::CapTable;
  using internal::ConcavePatches;
  using internal::ConvexPatches;
  using internal::CSR;
  using internal::OffsetTable;
  using internal::SasGeometry;
  using internal::SesGeometry;
  using internal::SurfaceGeometry;
  using internal::SurfaceSpheres;
  using internal::ToroidalPatches;

  /**
   * The first `n` rows of a cap table whose rows are ordered by sphere.
   */
  CapTable cap_prefix(const CSR &h, const Matrix3Xd &axis, const ArrayXd &cosa,
                      const int n) {
    const int m = h.offset(n);
    return { OffsetTable(ArrayXi(h.off().head(n + 1))), axis.leftCols(m),
             cosa.head(m) };
  }

  ConvexPatches convex_patches(const SurfaceSpheres &ss, const SasGeometry &sas,
                               ArrayXd &&area) {
    return { std::move(area), cap_prefix(sas.caps.h, sas.caps.axis,
                                         sas.caps.cosa, ss.n_active) };
  }

  /**
   * Takes the circles and arcs of `sas` and the saddle tables of `ses`.
   */
  ToroidalPatches toroidal_patches(const SurfaceSpheres &ss, SasGeometry &sas,
                                   SesGeometry &ses) {
    ToroidalPatches tor { std::move(sas.circles), std::move(ses.saddle_beta),
                          std::move(ses.saddle_integral), std::move(sas.arcs),
                          std::move(ses.saddle_area) };
    tor.circles.resize(ss.g.offset(ss.n_active));
    tor.arcs.resize(sas.n_active_arcs);
    return tor;
  }

  /**
   * Takes the face tables of `ses`.
   */
  ConcavePatches concave_patches(const SasGeometry &sas, SesGeometry &ses) {
    const int n_probes = sas.probes.n_active;
    return {
      sas.probes.pos.leftCols(n_probes),
      sas.probes.atoms.prefix(n_probes),
      std::move(ses.face_area),
      { std::move(ses.face_off), std::move(ses.face_axis),
                             std::move(ses.face_cosa) }
    };
  }

  SurfaceGeometry sas_geometry(const SurfaceSpheres &ss,
                               const SasGeometry &sas) {
    return { SurfaceKind::kSas,
             ss.rp,
             Matrix3Xd(ss.pts),
             ArrayXd(ss.sar),
             ArrayXd(ss.r),
             ArrayXi(ss.order),
             convex_patches(ss, sas, sas.area.head(ss.n_active)),
             {},
             {} };
  }

  /**
   * Takes the circles and arcs of `sas` and every table of `ses`.
   */
  SurfaceGeometry ses_geometry(const SurfaceSpheres &ss, SasGeometry &sas,
                               SesGeometry &ses) {
    ConvexPatches convex = convex_patches(ss, sas, std::move(ses.convex_area));
    ToroidalPatches toroidal = toroidal_patches(ss, sas, ses);
    ConcavePatches concave = concave_patches(sas, ses);
    return { SurfaceKind::kSes, ss.rp,
             Matrix3Xd(ss.pts), ArrayXd(ss.sar),
             ArrayXd(ss.r),     ArrayXi(ss.order),
             std::move(convex), std::move(toroidal),
             std::move(concave) };
  }
}  // namespace

AnalyticSurface::AnalyticSurface(ConstRef<Matrix3Xd> pts,
                                 ConstRef<ArrayXd> radii)
    : pts_(pts), radii_(radii) {
  ABSL_DCHECK_EQ(pts.cols(), radii.size());
}

bool AnalyticSurface::initialize(const double rp) {
  return initialize(rp, ArrayXb::Constant(radii_.size(), true));
}

bool AnalyticSurface::initialize(const double rp, ConstRef<ArrayXb> active) {
  ABSL_DCHECK_EQ(active.size(), radii_.size());

  std::optional<SurfaceSpheres> ss =
      internal::prepare(pts_, radii_, active, rp);
  if (!ss) {
    ss_ = {};
    return false;
  }

  ss_ = std::move(*ss);
  return true;
}

SurfaceGeometry AnalyticSurface::build_sas() const {
  ABSL_DCHECK(initialized());

  const internal::SasDelaunay del = internal::triangulate(ss_);
  const SasGeometry sas = internal::build_sas(ss_, del);
  return sas_geometry(ss_, sas);
}

SurfaceGeometry AnalyticSurface::build_ses() const {
  ABSL_DCHECK(initialized());

  const internal::SasDelaunay del = internal::triangulate(ss_);
  SasGeometry sas = internal::build_sas(ss_, del);
  SesGeometry ses = internal::build_ses(ss_, del, sas);
  return ses_geometry(ss_, sas, ses);
}

std::pair<SurfaceGeometry, SurfaceGeometry>
AnalyticSurface::build_both() const {
  ABSL_DCHECK(initialized());

  const internal::SasDelaunay del = internal::triangulate(ss_);
  SasGeometry sas = internal::build_sas(ss_, del);
  SesGeometry ses = internal::build_ses(ss_, del, sas);

  SurfaceGeometry gs = sas_geometry(ss_, sas);
  return { std::move(gs), ses_geometry(ss_, sas, ses) };
}
}  // namespace nuri

//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "desc_test_common.h"
#include "desc/surface/analytic/stages.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/random.h"

namespace nuri {
namespace internal {
namespace {
using constants::kPi;
using constants::kTwoPi;

constexpr double kRp = 1.0;

struct Sas {
  SurfaceSpheres ss;
  SasGeometry geo;
};

Sas solve(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXb &active,
          const double rp = kRp) {
  std::optional<SurfaceSpheres> ss = prepare(pts, sar - rp, active, rp);
  EXPECT_TRUE(ss);
  SasDelaunay del = triangulate(*ss);
  SasGeometry geo = build_sas(*ss, del);
  return { std::move(*ss), std::move(geo) };
}

Sas solve(const Matrix3Xd &pts, const ArrayXd &sar, const double rp = kRp) {
  return solve(pts, sar, ArrayXb::Constant(sar.size(), true), rp);
}

struct Surface {
  SurfaceGeometry sas, ses;
};

Surface surface(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXb &active,
                const double rp = kRp) {
  const ArrayXd r = sar - rp;
  AnalyticSurface surf(pts, r);
  EXPECT_TRUE(surf.initialize(rp, active));
  auto [sas, ses] = surf.build_both();
  return { std::move(sas), std::move(ses) };
}

Surface surface(const Matrix3Xd &pts, const ArrayXd &sar,
                const double rp = kRp) {
  return surface(pts, sar, ArrayXb::Constant(sar.size(), true), rp);
}

double sr_total(const Matrix3Xd &pts, const ArrayXd &sar) {
  return sr_sasa_impl(pts, sar, 5000, SrSasaMethod::kDirect).sum();
}

int probes_at(const Matrix3Xd &pos, const Vector3d &x,
              const double radius = 1e-9) {
  int n = 0;
  for (int p = 0; p < pos.cols(); ++p)
    n += static_cast<int>((pos.col(p) - x).norm() <= radius);
  return n;
}

double merge_tol(const Sas &sas) {
  return probe_merge_tol(sas.ss.sar.maxCoeff());
}

/**
 * On an exact fixture coincident roots merge and distinct roots lie far
 * apart, so no two probes are within the merge band.
 */
void expect_no_unmerged_pairs(const Matrix3Xd &pos, const double tau) {
  const int np = static_cast<int>(pos.cols());
  for (int p = 0; p < np; ++p) {
    for (int q = p + 1; q < np; ++q) {
      EXPECT_GT((pos.col(p) - pos.col(q)).norm(), tau)
          << "probes " << p << ", " << q;
    }
  }
}

/**
 * No two probes are one exact point.
 */
void expect_no_coincident_pairs(const SasGeometry &geo, const BallExact &ex) {
  const SasProbes &pr = geo.probes;
  const int np = static_cast<int>(pr.pos.cols());
  for (int p = 0; p < np; ++p) {
    for (int q = p + 1; q < np; ++q) {
      EXPECT_FALSE(ex.coincide(ball_triple(pr.face.col(p)), pr.plus[p],
                               ball_triple(pr.face.col(q)), pr.plus[q]))
          << "probes " << p << ", " << q;
    }
  }
}

void expect_no_coincident_pairs(const Matrix3Xd &pts, const ArrayXd &sar) {
  std::optional<SurfaceSpheres> ss =
      prepare(pts, sar - kRp, ArrayXb::Constant(sar.size(), true), kRp);
  ASSERT_TRUE(ss);
  const SasDelaunay del = triangulate(*ss);
  expect_no_coincident_pairs(build_sas(*ss, del), del.ex);
}

/**
 * The probes within `radius` of `x` (the roots of one rounded point)
 * exist and together touch exactly the `k` atoms.
 */
void expect_atoms_within(const SurfaceSpheres &ss, const SasGeometry &geo,
                         const Vector3d &x, const double radius, const int k) {
  const SasProbes &pr = geo.probes;
  std::vector<int> atoms;
  int n = 0;
  for (int p = 0; p < pr.pos.cols(); ++p) {
    if ((pr.pos.col(p) - x).norm() > radius)
      continue;
    ++n;
    for (const int a: pr.atoms.nbrs(p))
      atoms.push_back(ss.order[a]);
  }
  std::sort(atoms.begin(), atoms.end());
  atoms.erase(std::unique(atoms.begin(), atoms.end()), atoms.end());
  EXPECT_GE(n, 1);
  EXPECT_EQ(static_cast<int>(atoms.size()), k);
}

TEST(BuildSasTest, RejectsNonFiniteOrFarCoordinates) {
  Matrix3Xd pts = Matrix3Xd::Zero(3, 2);
  pts.col(1) << 3, 0, 0;
  ArrayXd sar = ArrayXd::Constant(2, 1.5);
  const ArrayXb all = ArrayXb::Constant(2, true);
  ASSERT_TRUE(prepare(pts, sar - kRp, all, kRp));

  pts(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(prepare(pts, sar - kRp, all, kRp));
  pts(0, 0) = 0;

  sar[1] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(prepare(pts, sar - kRp, all, kRp));
  sar[1] = 1.5;

  pts.col(1) << 2 * kSurfaceMaxCoord, 0, 0;
  EXPECT_FALSE(prepare(pts, sar - kRp, all, kRp));
}

TEST(BuildSasTest, RejectsBadRadii) {
  Matrix3Xd pts = Matrix3Xd::Zero(3, 2);
  pts.col(1) << 3, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(2, 1.5);
  ArrayXd r = sar - kRp;

  AnalyticSurface surf(pts, r);
  EXPECT_FALSE(surf.initialize(0));
  EXPECT_FALSE(surf.initialized());
  EXPECT_FALSE(surf.initialize(-1));
  EXPECT_FALSE(surf.initialized());

  r.setConstant(0.1);
  EXPECT_FALSE(surf.initialize(1.0));
  EXPECT_FALSE(surf.initialized());

  const double threshold = std::sqrt(2.0) * kRp;
  r.setConstant(threshold * (1 - 1e-3) - kRp);
  EXPECT_FALSE(surf.initialize(kRp));
  EXPECT_FALSE(surf.initialized());

  r.setConstant(threshold * (1 + 1e-3) - kRp);
  EXPECT_TRUE(surf.initialize(kRp));
  EXPECT_TRUE(surf.initialized());

  r.setConstant(-5.0);
  EXPECT_FALSE(surf.initialize(kRp));
  EXPECT_FALSE(surf.initialized());

  r = sar - kRp;
  EXPECT_TRUE(surf.initialize(kRp));
  EXPECT_TRUE(surf.initialized());
}

/**
 * Three standard deviations of a Shrake-Rupley estimate with `n` points at
 * the worst-case accessible fraction 1/2.
 */
double sr_tol(const double radius, const int n) {
  return 3 * 4 * kPi * radius * radius * 0.5 / std::sqrt(n);
}

void expect_areas(const Sas &sas, const ArrayXd &sr, const ArrayXd &sar,
                  const int n) {
  for (int p = 0; p < sas.ss.n_solve; ++p) {
    const int o = sas.ss.order[p];
    EXPECT_NEAR(sas.geo.area[p], sr[o], sr_tol(sar[o], n)) << "atom " << o;
  }
  for (int p = 0; p < sas.geo.probes.pos.cols(); ++p)
    EXPECT_GE(sas.geo.probes.atoms.degree(p), 3);
}

void expect_sr(const Sas &sas, const Matrix3Xd &pts, const ArrayXd &sar,
               const int n = 20000) {
  expect_areas(sas, sr_sasa_impl(pts, sar, n, SrSasaMethod::kDirect), sar, n);
}

void expect_sr(const Surface &s, const Matrix3Xd &pts, const ArrayXd &sar,
               const int n = 20000) {
  const ArrayXd sr = sr_sasa_impl(pts, sar, n, SrSasaMethod::kDirect);
  const ArrayXd &area = s.sas.convex().area;
  for (int p = 0; p < area.size(); ++p) {
    const int o = s.sas.order()[p];
    EXPECT_NEAR(area[p], sr[o], sr_tol(sar[o], n)) << "atom " << o;
  }
  const ConcavePatches &con = s.ses.concave();
  for (int p = 0; p < con.pos.cols(); ++p)
    EXPECT_GE(con.atoms.degree(p), 3);
}

/**
 * The SAS geometry of `s` over the spheres `sar`: the convex patches of the
 * kept atoms and nothing else, and no probe on its SES.
 */
void expect_sas_only(const Surface &s, const ArrayXd &sar) {
  const SurfaceGeometry &sas = s.sas;
  EXPECT_EQ(sas.kind(), SurfaceKind::kSas);
  EXPECT_EQ(sas.rp(), kRp);
  ASSERT_EQ(sas.sar().size(), sas.n_atoms());
  ASSERT_EQ(sas.r().size(), sas.n_atoms());
  for (int p = 0; p < sas.n_atoms(); ++p) {
    const int o = sas.order()[p];
    EXPECT_EQ(sas.sar()[p], sar[o]) << p;
    EXPECT_EQ(sas.r()[p], sar[o] - kRp) << p;
  }
  EXPECT_TRUE(sas.toroidal().arcs.empty());
  EXPECT_EQ(sas.concave().pos.cols(), 0);
  EXPECT_EQ(sas.total_area(SurfacePatch::kToroidal), 0);
  EXPECT_EQ(s.ses.concave().pos.cols(), 0);
}

/**
 * One sphere, two overlapping spheres, an exactly tangent pair, a ball
 * inside another and one poking out of it by 1e-9: closed-form areas, one
 * full circle for the overlap only.
 */
TEST(BuildSasTest, SpherePairs) {
  {
    const Matrix3Xd pts = Matrix3Xd::Zero(3, 1);
    const ArrayXd sar = ArrayXd::Constant(1, 2.5);

    const Surface s = surface(pts, sar);
    expect_sas_only(s, sar);
    ASSERT_EQ(s.sas.convex().area.size(), 1);
    EXPECT_DOUBLE_EQ(s.sas.convex().area[0], 4 * kPi * 2.5 * 2.5);
    EXPECT_TRUE(s.ses.toroidal().arcs.empty());
  }
  {
    Matrix3Xd pts(3, 2);
    pts << 0, 2.0, 0, 0, 0, 0;
    ArrayXd sar(2);
    sar << 1.8, 1.5;

    const Surface s = surface(pts, sar);
    expect_sas_only(s, sar);
    const ArrayXd &area = s.sas.convex().area;
    ASSERT_EQ(area.size(), 2);

    const double d = 2.0,
                 a0 = (d * d + sar[0] * sar[0] - sar[1] * sar[1]) / (2 * d);
    EXPECT_NEAR(area[s.sas.order()[0] == 0 ? 0 : 1],
                4 * kPi * sar[0] * sar[0] - kTwoPi * sar[0] * (sar[0] - a0),
                1e-12);
    EXPECT_NEAR(area[s.sas.order()[0] == 1 ? 0 : 1],
                4 * kPi * sar[1] * sar[1]
                    - kTwoPi * sar[1] * (sar[1] - (d - a0)),
                1e-12);

    const std::vector<RingArc> &arcs = s.ses.toroidal().arcs;
    ASSERT_EQ(arcs.size(), 1);
    EXPECT_EQ(arcs[0].beg, -1);
    EXPECT_EQ(arcs[0].end, -1);
    EXPECT_DOUBLE_EQ(arcs[0].dphi, kTwoPi);
  }
  {
    Matrix3Xd pts(3, 2);
    pts.col(0) << 0, 0, 0;
    pts.col(1) << 3, 0, 0;
    const ArrayXd sar = ArrayXd::Constant(2, 1.5);

    const Surface s = surface(pts, sar);
    expect_sas_only(s, sar);
    ASSERT_EQ(s.sas.convex().area.size(), 2);
    for (int p = 0; p < 2; ++p)
      EXPECT_NEAR(s.sas.convex().area[p], 4 * kPi * 1.5 * 1.5, 1e-9);
    EXPECT_TRUE(s.ses.toroidal().arcs.empty());
  }
  for (const int inner: { 0, 1 }) {
    Matrix3Xd pts(3, 2);
    ArrayXd sar(2);
    pts.col(inner) << 2, 0, 0;
    sar[inner] = 3;
    pts.col(1 - inner) << 0, 0, 0;
    sar[1 - inner] = 5;

    const Surface s = surface(pts, sar);
    expect_sas_only(s, sar);
    ASSERT_EQ(s.sas.n_atoms(), 1);
    EXPECT_EQ(s.sas.order()[0], 1 - inner);
    EXPECT_NEAR(s.sas.convex().area[0], 4 * kPi * 25, 1e-9);
    EXPECT_TRUE(s.ses.toroidal().arcs.empty());
  }
  for (const int inner: { 0, 1 }) {
    Matrix3Xd pts(3, 2);
    ArrayXd sar(2);
    pts.col(inner) << 2 + 1e-9, 0, 0;
    sar[inner] = 3;
    pts.col(1 - inner) << 0, 0, 0;
    sar[1 - inner] = 5;

    const Surface s = surface(pts, sar);
    expect_sas_only(s, sar);
    ASSERT_EQ(s.sas.n_atoms(), 2);
    EXPECT_TRUE((s.sas.convex().area > 0).all());
    expect_sr(s, pts, sar);
  }
}

TEST(BuildSasTest, CoplanarTriangle) {
  Matrix3Xd pts(3, 3);
  pts << 0, 2.4, 1.2, 0, 0, 2.1, 0, 0, 0;
  ArrayXd sar = ArrayXd::Constant(3, 1.9);

  const auto [sas, ses] = surface(pts, sar);
  const ArrayXd &area = sas.convex().area;
  ASSERT_EQ(area.size(), 3);
  EXPECT_NEAR(area.sum(), sr_total(pts, sar), 1e-2 * area.sum());

  const ConcavePatches &con = ses.concave();
  ASSERT_EQ(con.pos.cols(), 2);
  EXPECT_NEAR(std::abs(con.pos(2, 0)), std::abs(con.pos(2, 1)), 1e-12);
  EXPECT_NEAR(con.pos(2, 0) + con.pos(2, 1), 0, 1e-12);
  for (int p = 0; p < 2; ++p) {
    EXPECT_EQ(con.atoms.degree(p), 3);
    EXPECT_EQ(hemispheres(ses, p).cols(), 3);
  }

  const std::vector<RingArc> &arcs = ses.toroidal().arcs;
  EXPECT_EQ(arcs.size(), 3);
  for (const RingArc &arc: arcs) {
    EXPECT_GE(arc.beg, 0);
    EXPECT_GE(arc.end, 0);
    EXPECT_NE(arc.beg, arc.end);
    EXPECT_GT(arc.dphi, 0);
    EXPECT_LT(arc.dphi, kTwoPi);
  }
}

TEST(BuildSasTest, CoveredSphere) {
  Matrix3Xd pts(3, 7);
  pts << 0, 1, -1, 0, 0, 0, 0,  //
      0, 0, 0, 1, -1, 0, 0,     //
      0, 0, 0, 0, 0, 1, -1;
  pts *= 2.2;
  ArrayXd sar = ArrayXd::Constant(7, 2.1);
  sar[0] = 1.5;

  const Surface s = surface(pts, sar);
  const ArrayXd &area = s.sas.convex().area;
  ASSERT_EQ(area.size(), 7);
  const ArrayXi &order = s.sas.order();
  const int center = static_cast<int>(std::find(order.begin(), order.end(), 0)
                                      - order.begin());
  ASSERT_LT(center, 7);
  EXPECT_NEAR(area[center], 0, 1e-12);
  EXPECT_NEAR(area.sum(), sr_total(pts, sar), 1e-2 * area.sum());
}

/**
 * Trapped point of a rounded tetrahedron: its four roots are distinct
 * points, each accessible or not by the rounded geometry; the area does
 * not depend on it.
 */
TEST(BuildSasTest, FourSphereVertex) {
  const double reach = 2.0;
  Matrix3Xd pts(3, 4);
  pts << 1, 1, -1, -1,  //
      1, -1, 1, -1,     //
      1, -1, -1, 1;
  pts *= reach / std::sqrt(3.0);
  ArrayXd sar = ArrayXd::Constant(4, reach);

  const Surface s = surface(pts, sar);
  EXPECT_LE(probes_at(s.ses.concave().pos, Vector3d::Zero(), 1e-6), 4);
  expect_no_coincident_pairs(pts, sar);
  const double total = s.sas.convex().area.sum();
  EXPECT_NEAR(total, sr_total(pts, sar), 1e-2 * total);
}

TEST(BuildSasTest, FiveSphereVertex) {
  const Matrix3Xd pts = star(4, 110, 2.0);
  const ArrayXd sar = ArrayXd::Constant(5, 2.0);

  const Sas sas = solve(pts, sar);
  EXPECT_LE(probes_at(sas.geo.probes.pos, Vector3d::Zero()), 1);
  expect_no_unmerged_pairs(sas.geo.probes.pos, merge_tol(sas));
  EXPECT_NEAR(sas.geo.area.sum(), sr_total(pts, sar),
              1e-2 * sas.geo.area.sum());
}

TEST(BuildSasTest, CoplanarSquareVertex) {
  Matrix3Xd pts(3, 4);
  pts << 1, 1, -1, -1,  //
      1, -1, 1, -1,     //
      0, 0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(4, 1.5);

  const Sas sas = solve(pts, sar);
  EXPECT_EQ(sas.geo.probes.pos.cols(), 2);
  for (const double z: { 0.5, -0.5 }) {
    const int p = probe_at(sas.geo.probes.pos, Vector3d(0, 0, z));
    ASSERT_GE(p, 0) << "z " << z;
    EXPECT_EQ(sas.geo.probes.atoms.degree(p), 4);
    EXPECT_EQ(sas.geo.probes.tan_off.degree(p), 4);
  }
  expect_no_unmerged_pairs(sas.geo.probes.pos, merge_tol(sas));
  EXPECT_NEAR(sas.geo.area.sum(), sr_total(pts, sar),
              1e-2 * sas.geo.area.sum());
}

struct TangentApex {
  Matrix3Xd pts;
  ArrayXd sar;
  Vector3d x;
};

TangentApex tangent_apex(const double gap) {
  const double reach = 2.0, circum = 1.5, rl = 1.8;
  TangentApex t { Matrix3Xd(3, 4), ArrayXd(4), Vector3d() };
  for (int k = 0; k < 3; ++k) {
    const double az = kTwoPi * k / 3;
    t.pts.col(k) << circum * std::cos(az), circum * std::sin(az), 0;
  }
  t.x << 0, 0, std::sqrt(reach * reach - circum * circum);
  t.pts.col(3) = t.x + (rl + gap) * (t.x - t.pts.col(0)).normalized();
  t.sar << reach, reach, reach, rl;
  return t;
}

/**
 * Shrake-Rupley with 20000 points agrees to 0.05 here and on the fixtures
 * of similar size below, `(sar / 2)^2` times that elsewhere: empirical.
 * `sr_tol` is the random-sampling bound, 0.3 to 0.5 at these radii, which
 * the Fibonacci lattice beats by an order of magnitude.
 */
TEST(BuildSasTest, NearBandPinch) {
  for (const double gap: { 1e-7, 5e-7, 2e-6 }) {
    const TangentApex t = tangent_apex(gap);
    auto [ss, geo] = solve(t.pts, t.sar);
    const ArrayXd sr = sr_sasa_impl(t.pts, t.sar, 20000, SrSasaMethod::kDirect);
    for (int p = 0; p < 4; ++p)
      EXPECT_NEAR(geo.area[p], sr[ss.order[p]], 0.05)
          << "gap " << gap << " atom " << p;
  }
}

/**
 * Pairs below `touch - 2eps` are overlaps without the predicate, the tangency
 * band is decided exactly, and a pair beyond `touch` is near but no overlap.
 */
TEST(BuildSasTest, OverlapBandsAroundTouch) {
  constexpr double eps = kSurfaceLengthEps, touch = 3.0;
  const std::array<double, 4> gaps { -3 * eps, -eps, eps, 3 * eps };
  Matrix3Xd pts(3, 8);
  for (int k = 0; k < 4; ++k) {
    pts.col(2 * k) << 0, 0, 20.0 * k;
    pts.col(2 * k + 1) << touch + gaps[k], 0, 20.0 * k;
  }
  const ArrayXd sar = ArrayXd::Constant(8, touch / 2);

  const Surface s = surface(pts, sar);
  ASSERT_EQ(s.sas.n_atoms(), 8);

  std::vector<std::pair<int, int>> edges;
  for (const SasCircle &c: s.ses.toroidal().circles)
    edges.push_back(nuri::minmax(s.ses.order()[c.i], s.ses.order()[c.j]));
  std::sort(edges.begin(), edges.end());
  const std::vector<std::pair<int, int>> expected {
    { 0, 1 },
    { 2, 3 }
  };
  EXPECT_EQ(edges, expected);
}

TEST(BuildSasTest, SliverBetweenNearCrossings) {
  const Matrix3Xd pts = star(4, 110, 2.0);
  ArrayXd sar = ArrayXd::Constant(5, 2.0);
  sar[2] -= 1e-5;

  auto [ss, geo] = solve(pts, sar);
  const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
  for (int p = 0; p < 5; ++p)
    EXPECT_NEAR(geo.area[p], sr[ss.order[p]], 0.05) << "atom " << p;
}

/**
 * A fourth sphere grazing the triple point within kSurfaceLengthEps, its
 * surface nearly along circle (a, b) so that its own cuts on that circle land
 * farther away: the cluster is a 4-fold vertex, decided against the apexes.
 */
TEST(BuildSasTest, GrazingApexWithinTolerance) {
  for (const double depth: { 5e-7, -5e-7 }) {
    const TangentApex t = tangent_apex(0);
    const Vector3d na = (t.x - t.pts.col(0)).normalized(),
                   nb = (t.x - t.pts.col(1)).normalized(),
                   tan = na.cross(nb).normalized(),
                   m = (na + 0.1 * tan).normalized();
    Matrix3Xd pts = t.pts;
    pts.col(3) = t.x - (t.sar[3] - depth) * m;

    auto [ss, geo] = solve(pts, t.sar);
    EXPECT_GE(probes_at(geo.probes.pos, t.x, kSurfaceLengthEps), 1)
        << "depth " << depth;
    const ArrayXd sr = sr_sasa_impl(pts, t.sar, 20000, SrSasaMethod::kDirect);
    for (int p = 0; p < 4; ++p)
      EXPECT_NEAR(geo.area[p], sr[ss.order[p]], 0.05)
          << "depth " << depth << " atom " << p;
  }
}

/**
 * Flat cell: the apex sphere's center lies 1e-3 above the plane of the face
 * and grazes the triple point, while a steeper fifth sphere contains the
 * point by 5e-5; the point must be rejected.
 */
TEST(BuildSasTest, FlatCellApexDoesNotHideCover) {
  const TangentApex t = tangent_apex(0);
  const double eta = 1e-3, depth = 3e-7, cover = 5e-5, rl = t.sar[3], rm = 1.7,
               phi = 100 * kPi / 180;
  const double rho =
      std::sqrt((rl - depth) * (rl - depth) - (t.x[2] - eta) * (t.x[2] - eta));
  Matrix3Xd pts(3, 5);
  pts.leftCols(3) = t.pts.leftCols(3);
  pts.col(3) << rho * std::cos(phi), rho * std::sin(phi), eta;
  pts.col(4) = t.x + (rm - cover) * Vector3d(0.3, -0.2, 1.0).normalized();
  ArrayXd sar(5);
  sar << t.sar[0], t.sar[1], t.sar[2], rl, rm;

  auto [ss, geo] = solve(pts, sar);
  for (int p = 0; p < geo.probes.pos.cols(); ++p)
    EXPECT_GT((geo.probes.pos.col(p) - t.x).norm(), 1e-5) << p;
  const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
  for (int p = 0; p < 5; ++p)
    EXPECT_NEAR(geo.area[p], sr[ss.order[p]], 0.05) << "atom " << p;
}

/**
 * Ideal benzene: 6-fold axis vertices and 4-fold C-C-H-H vertices.
 */
Matrix3Xd benzene_pts(ArrayXd &sar, const double rp) {
  Matrix3Xd pts(3, 12);
  sar.resize(12);
  for (int k = 0; k < 6; ++k) {
    const double az = kPi / 3 * k;
    pts.col(k) << 1.39 * std::cos(az), 1.39 * std::sin(az), 0;
    pts.col(6 + k) << 2.47 * std::cos(az), 2.47 * std::sin(az), 0;
    sar[k] = 1.7 + rp;
    sar[6 + k] = 1.2 + rp;
  }
  return pts;
}

/**
 * A 1e-7 jitter scatters the raw points of one benzene vertex beyond
 * kSurfaceLengthEps along their circles while every sphere still passes
 * within the tolerance. Deterministic jitter, rp 1.4; Shrake-Rupley to 0.1,
 * the 0.05 of NearBandPinch for these larger spheres.
 */
TEST(BuildSasTest, JitteredBenzeneVertices) {
  const double rp = 1.4;
  ArrayXd sar;
  Matrix3Xd pts = benzene_pts(sar, rp);
  for (int i = 0; i < pts.size(); ++i)
    pts.data()[i] += 1e-7 * std::sin(1000.0 * (i + 1));

  std::optional<SurfaceSpheres> ss =
      prepare(pts, sar - rp, ArrayXb::Constant(12, true), rp);
  ASSERT_TRUE(ss);
  const SasDelaunay del = triangulate(*ss);
  const SasGeometry geo = build_sas(*ss, del);

  const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
  for (int p = 0; p < 12; ++p)
    EXPECT_NEAR(geo.area[p], sr[ss->order[p]], 0.1) << "atom " << p;
  for (int p = 0; p < geo.probes.pos.cols(); ++p)
    EXPECT_GE(geo.probes.atoms.degree(p), 3);
}

/**
 * Spheres j and k are tangent (or a near pair) at a point T of the host
 * sphere, and sphere l passes through the point of circle (host, j) at
 * arc offset `delta` from T. The three raw points merge into one vertex
 * whose two pinch darts are `delta (kappa_j + kappa_k)` apart, far more
 * than any angle tolerance; the corner must be the signed dart angle.
 */
TEST(BuildSasTest, NearPairPinchWithThirdSphere) {
  const double rs = 2.0, rj = 1.8, rk = 1.7, rl = 1.9, beta = kPi / 3;
  const Vector3d top(0, 0, rs), nj(std::sin(beta), 0, std::cos(beta));
  for (const double gap: { 0.0, 5e-7 }) {
    for (const double delta: { 3e-5, 1e-4, 3e-4, 1e-3 }) {
      Matrix3Xd pts(3, 4);
      pts.col(0).setZero();
      pts.col(1) = top + rj * nj;
      pts.col(2) = top - (rk + gap) * nj;
      const Vector3d uj = pts.col(1).normalized();
      const double circ = top.cross(uj).norm();
      const Vector3d p = AngleAxisd(delta / circ, uj) * top,
                     ph = p.normalized(), tj = uj.cross(p).normalized(),
                     w = ph.cross(tj),
                     m = (0.6 * ph + 0.57 * tj + 0.57 * w).normalized();
      pts.col(3) = p + rl * m;
      ArrayXd sar(4);
      sar << rs, rj, rk, rl;

      auto [ss, geo] = solve(pts, sar);
      const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
      for (int a = 0; a < 4; ++a) {
        EXPECT_NEAR(geo.area[a], sr[ss.order[a]], 0.05)
            << "gap " << gap << " delta " << delta << " atom " << a;
      }
    }
  }
}

/**
 * The two cut points of `near_tangent_face` without its cutter are `2h`
 * apart: distinct vertices at any `h > 0`, far below the merge band
 * included, since only exact coincidence merges two roots.
 */
TEST(BuildSasTest, TwoCutPointsOfOneFaceStayDistinct) {
  for (const double h: { 5e-6, 1e-13 }) {
    const auto [pts4, sar4] = near_tangent_face(h);
    const Matrix3Xd pts = pts4.leftCols(3);
    const ArrayXd sar = sar4.head(3);

    auto [ss, geo] = solve(pts, sar);
    ASSERT_EQ(geo.probes.pos.cols(), 2) << "h " << h;
    const double dist = (geo.probes.pos.col(0) - geo.probes.pos.col(1)).norm();
    if (h > probe_merge_tol(sar.maxCoeff())) {
      EXPECT_NEAR(dist, 2 * h, 1e-9);
    } else {
      EXPECT_LT(dist, probe_merge_tol(sar.maxCoeff()));
      expect_no_coincident_pairs(pts, sar);
    }
    for (int p = 0; p < 2; ++p)
      EXPECT_EQ(geo.probes.atoms.degree(p), 3) << "h " << h;
  }
}

Matrix3Xd lattice24_pts() {
  const int n = 24;
  Matrix3Xd pts(3, n);
  for (int i = 0; i < n; ++i) {
    const int ix = i % 4, iy = (i / 4) % 3, iz = i / 12;
    pts.col(i) << 1.7 * ix, 1.6 * iy + 0.3 * (i % 2), 1.5 * iz + 0.2 * (i % 3);
  }
  return pts;
}

/**
 * One object initialized again: the same mask gives the same surface, a
 * half mask the same areas on its active atoms, a rejected probe radius
 * empties the object and the next good call brings the surface back.
 */
TEST(BuildSasTest, Reinitialize) {
  const Matrix3Xd pts = lattice24_pts();
  const int n = static_cast<int>(pts.cols());
  const ArrayXd sar =
      ArrayXd::Constant(n, 1.5) + 0.05 * ArrayXd::LinSpaced(n, 0, 1);
  const ArrayXd r = sar - kRp;

  ArrayXb all = ArrayXb::Constant(n, true), half = all;
  half.tail(n / 2).setConstant(false);

  AnalyticSurface surf(pts, r);
  ASSERT_TRUE(surf.initialize(kRp));
  const ArrayXi order = surf.order();
  ASSERT_EQ(order.size(), n);
  seed_thread(0);
  const ArrayXd area = surf.build_sas().convex().area;
  ASSERT_EQ(area.size(), n);

  ASSERT_TRUE(surf.initialize(kRp, all));
  EXPECT_TRUE((surf.order() == order).all());
  seed_thread(0);
  EXPECT_TRUE((surf.build_sas().convex().area == area).all());

  ArrayXd area_all(n);
  for (int p = 0; p < n; ++p)
    area_all[order[p]] = area[p];

  ASSERT_TRUE(surf.initialize(kRp, half));
  EXPECT_EQ(surf.n_active(), n / 2);
  const SurfaceGeometry part = surf.build_sas();
  ASSERT_EQ(part.convex().area.size(), surf.n_active());
  for (int p = 0; p < surf.n_active(); ++p)
    EXPECT_DOUBLE_EQ(part.convex().area[p], area_all[part.order()[p]]) << p;

  EXPECT_FALSE(surf.initialize(-kRp));
  EXPECT_FALSE(surf.initialized());
  EXPECT_EQ(surf.n_atoms(), 0);
  EXPECT_EQ(surf.n_active(), 0);

  ASSERT_TRUE(surf.initialize(kRp));
  EXPECT_TRUE((surf.order() == order).all());
  seed_thread(0);
  EXPECT_TRUE((surf.build_sas().convex().area == area).all());
}

struct Masked {
  Matrix3Xd pts;
  ArrayXd sar;
  ArrayXb active;
};

ArrayXi all_but(const int n, std::initializer_list<int> drop) {
  ArrayXi rest(n - static_cast<int>(drop.size()));
  for (int q = 0, p = 0; q < n; ++q) {
    if (std::find(drop.begin(), drop.end(), q) == drop.end())
      rest[p++] = q;
  }
  return rest;
}

Surface surface_only(const Matrix3Xd &pts, const ArrayXd &sar,
                     const ArrayXb &active, const ArrayXi &rest) {
  return surface(pts(E::all, rest), sar(rest), active(rest));
}

/**
 * `full` over every atom equals `ctrl` over the atoms `rest` only: the same
 * atoms kept in the same order with the same patches (seeded builds).
 */
void expect_same_surface(const Surface &full, const Surface &ctrl,
                         const ArrayXi &rest) {
  ASSERT_EQ(full.sas.n_atoms(), ctrl.sas.n_atoms());
  for (int p = 0; p < full.sas.n_atoms(); ++p)
    EXPECT_EQ(full.sas.order()[p], rest[ctrl.sas.order()[p]]) << p;

  ASSERT_EQ(full.sas.convex().area.size(), ctrl.sas.convex().area.size());
  for (int p = 0; p < full.sas.convex().area.size(); ++p)
    EXPECT_NEAR(full.sas.convex().area[p], ctrl.sas.convex().area[p], 1e-12)
        << p;

  for (const SurfacePatch patch:
       { SurfacePatch::kConvex, SurfacePatch::kToroidal,
         SurfacePatch::kConcave }) {
    const ArrayXd &a = full.ses.area(patch), &b = ctrl.ses.area(patch);
    ASSERT_EQ(a.size(), b.size()) << static_cast<int>(patch);
    for (int k = 0; k < a.size(); ++k)
      EXPECT_NEAR(a[k], b[k], 1e-12) << static_cast<int>(patch) << " " << k;
  }

  ASSERT_EQ(full.ses.concave().pos.cols(), ctrl.ses.concave().pos.cols());
  EXPECT_EQ(full.ses.concave().pos, ctrl.ses.concave().pos);
  EXPECT_EQ(full.ses.toroidal().arcs.size(), ctrl.ses.toroidal().arcs.size());
}

/**
 * A chain active, need, shell, occluder, occluder: the last overlaps only
 * the first occluder and is dropped.
 */
Masked occluder_chain() {
  Masked m { Matrix3Xd(3, 6), ArrayXd(6), ArrayXb(6) };
  m.pts.col(0) << 0, 0, 0;
  m.pts.col(1) << 0, 2.5, 0;
  for (int i = 2; i < 6; ++i)
    m.pts.col(i) << 2.5 * (i - 1), 0, 0;
  m.sar.setConstant(1.5);
  m.active.setConstant(false);
  m.active.head(2).setConstant(true);
  return m;
}

TEST(BuildSasTest, OccluderBeyondShellBandDropped) {
  const Masked m = occluder_chain();
  const ArrayXi rest = all_but(6, { 5 });

  seed_thread(0);
  const Surface full = surface(m.pts, m.sar, m.active);
  seed_thread(0);
  const Surface part = surface_only(m.pts, m.sar, m.active, rest);
  ASSERT_EQ(full.sas.n_atoms(), 5);
  EXPECT_TRUE((full.sas.order() != 5).all());
  EXPECT_TRUE((full.sas.order() == 4).any());
  EXPECT_EQ(full.sas.convex().area.size(), 2);
  expect_same_surface(full, part, rest);
}

/**
 * Three spheres of radius 1.5 stacked 2.5 apart above `foot`, the top one
 * the only active atom, so the atoms near `foot` sit at hop 3.
 */
ArrayXb append_chain(Matrix3Xd &pts, ArrayXd &sar, const Vector3d &foot) {
  const int m = static_cast<int>(pts.cols());
  pts.conservativeResize(E::NoChange, m + 3);
  sar.conservativeResize(m + 3);
  for (int q = 0; q < 3; ++q) {
    pts.col(m + q) = foot + Vector3d(0, 0, 2.5 * q);
    sar[m + q] = 1.5;
  }
  ArrayXb active = ArrayXb::Constant(m + 3, false);
  active[m + 2] = true;
  return active;
}

/**
 * Atoms beyond the reach of every active atom change nothing: a cloud
 * with two active atoms and, at every sixth index, an atom 300 A away.
 */
TEST(BuildSasTest, PrepareIgnoresAtomsBeyondReach) {
  constexpr int n = 240;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0, 6), ur(1.5, 2.5);
  Matrix3Xd pts(3, n);
  ArrayXd sar(n);
  ArrayXi rest(n - n / 6);
  for (int i = 0, w = 0; i < n; ++i) {
    pts.col(i) << u(rng), u(rng), u(rng);
    if (i % 6 == 5)
      pts(2, i) += 300;
    else
      rest[w++] = i;
    sar[i] = ur(rng);
  }
  ArrayXb active = ArrayXb::Constant(n, false);
  active[0] = active[7] = true;

  seed_thread(0);
  const Surface full = surface(pts, sar, active);
  seed_thread(0);
  const Surface part = surface_only(pts, sar, active, rest);
  EXPECT_GT(full.sas.n_atoms(), 2);
  for (const int i: full.sas.order())
    EXPECT_NE(i % 6, 5);
  expect_same_surface(full, part, rest);
}

void expect_empty(const SurfaceGeometry &g) {
  EXPECT_EQ(g.n_atoms(), 0);
  EXPECT_EQ(g.convex().area.size(), 0);
  EXPECT_EQ(g.total_area(), 0);
  EXPECT_TRUE(g.toroidal().arcs.empty());
  EXPECT_TRUE(g.toroidal().circles.empty());
  EXPECT_EQ(g.concave().pos.cols(), 0);
}

void expect_empty(AnalyticSurface &surf) {
  EXPECT_TRUE(surf.initialized());
  EXPECT_EQ(surf.n_atoms(), 0);
  EXPECT_EQ(surf.n_active(), 0);
  EXPECT_EQ(surf.order().size(), 0);

  const auto [sas, ses] = surf.build_both();
  expect_empty(sas);
  expect_empty(ses);

  const SurfaceDots dots = SurfaceSampler(15).sample(ses);
  EXPECT_EQ(dots.n(), 0);
  for (const SurfacePatch patch:
       { SurfacePatch::kConvex, SurfacePatch::kToroidal,
         SurfacePatch::kConcave })
    EXPECT_EQ(dots.count(patch), 0);
}

/**
 * Without an active atom nothing is kept, and without an atom at all.
 */
TEST(BuildSasTest, EmptySurfaces) {
  {
    Matrix3Xd pts(3, 3);
    pts << 0, 2.5, 5, 0, 0, 0, 0, 0, 0;
    const ArrayXd r = ArrayXd::Constant(3, 1.5 - kRp);
    const ArrayXb none = ArrayXb::Constant(3, false);

    AnalyticSurface surf(pts, r);
    ASSERT_TRUE(surf.initialize(kRp, none));
    expect_empty(surf);
  }
  {
    const Matrix3Xd pts(3, 0);
    const ArrayXd r(0);

    AnalyticSurface surf(pts, r);
    ASSERT_TRUE(surf.initialize(kRp));
    expect_empty(surf);
  }
}

/**
 * A chain active, need, shell, then the hop-3 sphere 3 contains the hop-3
 * sphere 4 and the hop-4 sphere 5, which is near nothing else. A container
 * is near every neighbor of the ball it contains, so the container of a
 * hop-3 atom is within three hops itself and the band sees both pairs.
 */
Masked contained_partner() {
  Masked m { Matrix3Xd(3, 6), ArrayXd(6), ArrayXb(6) };
  for (int i = 0; i < 3; ++i)
    m.pts.col(i) << 2.5 * i, 0, 0;
  m.pts.col(3) << 8, 0, 0;
  m.pts.col(4) << 7.5, 0, 0;
  m.pts.col(5) << 8.9, 0, 0;
  m.sar.setConstant(1.5);
  m.sar[3] = 2.5;
  m.active.setConstant(false);
  m.active[0] = true;
  return m;
}

TEST(BuildSasTest, BandContainedPartnerAtHopFour) {
  const Masked m = contained_partner();
  const ArrayXi rest = all_but(6, { 4, 5 });

  seed_thread(0);
  const Surface full = surface(m.pts, m.sar, m.active);
  seed_thread(0);
  const Surface part = surface_only(m.pts, m.sar, m.active, rest);
  expect_same_surface(full, part, rest);
}

/**
 * The hop-4 sphere of a chain with a partner of radius `r` on its center at
 * `x`: an identical twin, a larger ball around it or a smaller ball inside
 * it, within reach of the hop-3 sphere at 10 and out of its reach at 10.8,
 * where only a larger partner stays at hop 4.
 */
Masked twin_at_hop_four(const double x, const double r) {
  Masked m { Matrix3Xd(3, 6), ArrayXd(6), ArrayXb(6) };
  for (int i = 0; i < 4; ++i)
    m.pts.col(i) << 2.5 * i, 0, 0;
  m.pts.col(4) << x, 0, 0;
  m.pts.col(5) << x, 0, 0;
  m.sar.setConstant(1.5);
  m.sar[5] = r;
  m.active.setConstant(false);
  m.active[0] = true;
  return m;
}

/**
 * No zero distance reaches the overlap graph and the chain without the
 * pair is the control.
 */
TEST(BuildSasTest, BandTwinAtHopFour) {
  const ArrayXi rest = all_but(6, { 4, 5 });
  const Masked base = twin_at_hop_four(10.0, 1.5);
  seed_thread(0);
  const Surface ref = surface_only(base.pts, base.sar, base.active, rest);
  ASSERT_EQ(ref.sas.n_atoms(), 4);

  for (const double x: { 10.0, 10.8 }) {
    for (const double r: { 1.5, 2.0, 1.45 }) {
      SCOPED_TRACE(testing::Message() << x << " " << r);
      const Masked m = twin_at_hop_four(x, r);
      seed_thread(0);
      const Surface full = surface(m.pts, m.sar, m.active);
      expect_same_surface(full, ref, rest);
    }
  }
}

TEST(BuildSasTest, TriangulationTables) {
  const Matrix3Xd pts = lattice24_pts();
  const int n = static_cast<int>(pts.cols());
  const ArrayXd sar = ArrayXd::Constant(n, 1.5);
  std::optional<SurfaceSpheres> ss =
      prepare(pts, sar - kRp, ArrayXb::Constant(n, true), kRp);
  ASSERT_TRUE(ss);
  const SasDelaunay del = triangulate(*ss);

  const int nf = static_cast<int>(del.tets.cols());
  ArrayXi seen = ArrayXi::Zero(del.n_faces);
  for (int c = 0; c < nf; ++c) {
    for (int lf = 0; lf < 4; ++lf) {
      const int f = del.face(lf, c), c2 = del.adj(lf, c);
      ASSERT_GE(f, 0);
      ASSERT_LT(f, del.n_faces);
      ++seen[f];
      if (c2 >= 0) {
        int lf2 = 0;
        while (del.adj(lf2, c2) != c)
          ++lf2;
        EXPECT_EQ(del.face(lf2, c2), f);
      }
    }
  }
  EXPECT_TRUE((seen >= 1).all() && (seen <= 2).all());
  EXPECT_EQ(seen.sum(), 4 * nf);

  for (int v = 0; v < del.nbrs.n(); ++v) {
    for (auto it = del.nbrs.begin(v), e = del.nbrs.end(v); it < e; ++it) {
      if (it + 1 < e) {
        EXPECT_LT(*it, *(it + 1)) << v;
      }

      const int c = del.edge_cell[del.nbrs.eid(it)];
      const Array4i tv = del.tets.col(c);
      EXPECT_TRUE((tv == v).any() && (tv == *it).any()) << v << " " << *it;
    }
  }

  EXPECT_EQ(del.ex.n(), n + 4);
  for (int v = 0; v < n; ++v)
    EXPECT_NEAR(del.ex.rho2(v), 2.25, 1e-12);
}

TEST(BuildSasTest, EdgeTablesMatchBruteForce) {
  const int n = 80;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> ud(0.0, 10.0), ur(1.5, 2.5);
  Matrix3Xd pts(3, n);
  ArrayXd sar(n);
  ArrayXb active(n);
  for (int i = 0; i < n; ++i) {
    pts.col(i) << ud(rng), ud(rng), ud(rng);
    sar[i] = ur(rng);
    active[i] = i % 3 == 0;
  }
  std::optional<SurfaceSpheres> ss = prepare(pts, sar - kRp, active, kRp);
  ASSERT_TRUE(ss);
  const SasDelaunay del = triangulate(*ss);
  const SasEdgeTables tbl = edge_tables(*ss, del);

  const int nk = ss->g.n();
  ArrayXi v_of_new(nk), new_of_v = ArrayXi::Constant(del.ex.n(), -1);
  for (int p = 0; p < nk; ++p) {
    v_of_new[p] = del.vertex[ss->order[p]];
    new_of_v[v_of_new[p]] = p;
  }
  EXPECT_TRUE((v_of_new != ArrayXi::LinSpaced(nk, 0, nk - 1)).any());

  ASSERT_EQ(tbl.pair_of_edge.size(), del.nbrs.m());
  int n_corner = 0, n_apart = 0, n_overlap = 0;
  for (int v = 0; v < nk; ++v) {
    for (auto it = del.nbrs.begin(v), e = del.nbrs.end(v); it < e; ++it) {
      const int s = new_of_v[v], t = new_of_v[*it];
      const auto [a, b] = nuri::minmax(s, t);
      int expected = -1;
      if (t >= 0) {
        const auto gb = ss->g.begin(a), ge = ss->g.end(a),
                   found = std::find(gb, ge, b);
        expected = found != ge ? ss->g.eid(found) : -1;
      }
      EXPECT_EQ(tbl.pair_of_edge[del.nbrs.eid(it)], expected)
          << v << " " << *it;
      n_corner += t < 0;
      n_apart += t >= 0 && expected < 0;
      n_overlap += expected >= 0;
    }
  }
  EXPECT_GT(n_corner, 0);
  EXPECT_GT(n_apart, 0);
  EXPECT_GT(n_overlap, 0);

  ASSERT_EQ(tbl.edge_of_pair.size(), ss->g.m());
  int n_adjacent = 0, n_far = 0;
  for (int i = 0; i < nk; ++i) {
    for (auto it = ss->g.begin(i), e = ss->g.end(i); it < e; ++it) {
      const int va = v_of_new[i], vb = v_of_new[*it];
      const auto rb = del.nbrs.begin(va), re = del.nbrs.end(va),
                 found = std::find(rb, re, vb);
      const int expected = found != re ? del.nbrs.eid(found) : -1;
      EXPECT_EQ(tbl.edge_of_pair[ss->g.eid(it)], expected) << i << " " << *it;
      n_adjacent += expected >= 0;
      n_far += expected < 0;
    }
  }
  EXPECT_GT(n_adjacent, 0);
  EXPECT_GT(n_far, 0);

  int slots = 0;
  for (int a = 0; a < 4; ++a) {
    for (int b = a + 1; b < 4; ++b) {
      EXPECT_EQ(kCellPairSlot[a][b], kCellPairSlot[b][a]);
      slots |= 1 << kCellPairSlot[a][b];
    }
  }
  EXPECT_EQ(slots, 0b111111);

  int n_both_corners = 0;
  for (int c = 0; c < del.tets.cols(); ++c) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        const auto [lo, hi] = nuri::minmax(del.tets(a, c), del.tets(b, c));
        const int k = del.cell_edge(kCellPairSlot[a][b], c);
        if (lo >= nk) {
          EXPECT_EQ(k, -1) << c;
          ++n_both_corners;
          continue;
        }
        ASSERT_GE(k, del.nbrs.offset(lo)) << c;
        ASSERT_LT(k, del.nbrs.offset(lo + 1)) << c;
        EXPECT_EQ(del.nbrs.adj()[k], hi) << c;
      }
    }
  }
  EXPECT_GT(n_both_corners, 0);
}

/**
 * Probes strictly inside a sphere that is not one of their atoms.
 */
int spurious_probes(const Sas &sas, const Matrix3Xd &pts, const ArrayXd &sar) {
  int n = 0;
  for (int p = 0; p < sas.geo.probes.pos.cols(); ++p) {
    const Vector3d x = sas.geo.probes.pos.col(p);
    for (int m = 0; m < sar.size(); ++m) {
      bool atom = false;
      for (const int a: sas.geo.probes.atoms.nbrs(p))
        atom |= sas.ss.order[a] == m;
      n += !atom && sar[m] - (x - pts.col(m)).norm() > kSurfaceLengthEps;
    }
  }
  return n;
}

int flat_cells(const SasDelaunay &del, const int n_coplanar) {
  int n = 0;
  for (int c = 0; c < del.tets.cols(); ++c) {
    bool all = true;
    for (int lv = 0; lv < 4; ++lv) {
      const int v = del.tets(lv, c);
      all &= (del.vertex.head(n_coplanar) == v).any();
    }
    n += all;
  }
  return n;
}

/**
 * Four coplanar spheres through two points and a fifth covering the upper
 * one: under a rigid motion the coplanar four form a cell of rounding-level
 * height whose apex the old slack excused, leaving a probe 0.4 angstrom inside
 * the cover.
 */
TEST(BuildSasTest, FlatSquareApexNoSpuriousProbe) {
  Matrix3Xd base(3, 5);
  base << 1, 1, -1, -1, 0,  //
      1, -1, 1, -1, 0,      //
      0, 0, 0, 0, 1.6;
  const ArrayXd sar = ArrayXd::Constant(5, 1.5);

  int with_flat = 0;
  for (int seed = 0; seed < 40; ++seed) {
    const Matrix3Xd pts = rigid(base, seed);
    std::optional<SurfaceSpheres> ss =
        prepare(pts, sar - kRp, ArrayXb::Constant(5, true), kRp);
    ASSERT_TRUE(ss);
    const SasDelaunay del = triangulate(*ss);
    with_flat += flat_cells(del, 4) > 0;
    const Sas sas { *ss, build_sas(*ss, del) };
    EXPECT_EQ(spurious_probes(sas, pts, sar), 0) << "seed " << seed;
    expect_sr(sas, pts, sar, 5000);
  }
  EXPECT_GT(with_flat, 0);
}

TEST(BuildSasTest, BenzeneCoverNoSpuriousProbe) {
  const double rp = 1.4;
  ArrayXd sar;
  Matrix3Xd base = benzene_pts(sar, rp);
  base.conservativeResize(3, 13);
  sar.conservativeResize(13);
  Vector3d v(1.93, 1.1147, 0);
  v[2] = std::sqrt(sar[0] * sar[0]
                   - (v.head(2) - base.col(0).head(2)).squaredNorm());
  base.col(12) = v + Vector3d(0, 0, 1.0);
  sar[12] = 1.4 + rp;

  for (int seed = 0; seed < 10; ++seed) {
    const Matrix3Xd pts = rigid(base, seed);
    const Sas sas = solve(pts, sar, rp);
    EXPECT_EQ(spurious_probes(sas, pts, sar), 0) << "seed " << seed;
    expect_sr(sas, pts, sar, 5000);
  }
}

/**
 * The third center lies `amp` from the axis of circle `(a, b)`: the radical
 * line form of the cut points loses `amp^2` of precision, the circle form is
 * exact there, and the builder keeps whichever fits its spheres better.
 */
TEST(BuildSasTest, CoaxialCutPointsStayOnTheirSpheres) {
  const double amp = 3e-6, ra = 2, rb = 2, xb = 3, xc = 5;
  const double a_ab = (xb * xb + ra * ra - rb * rb) / (2 * xb),
               rl = std::sqrt(ra * ra - a_ab * a_ab);
  const Vector3d cntr(a_ab, 0, 0), cc(xc, amp, 0), w = cntr - cc;
  const double g = 0.5 * amp,
               rc = std::sqrt(w.squaredNorm() + rl * rl + 2 * rl * g);
  Matrix3Xd base(3, 3);
  base.col(0) << 0, 0, 0;
  base.col(1) << xb, 0, 0;
  base.col(2) = cc;
  ArrayXd sar(3);
  sar << ra, rb, rc;

  for (int seed = 0; seed < 5; ++seed) {
    const Matrix3Xd pts = rigid(base, seed);
    const Sas sas = solve(pts, sar);
    ASSERT_EQ(sas.geo.probes.pos.cols(), 2);
    for (int p = 0; p < 2; ++p) {
      for (const int a: sas.geo.probes.atoms.nbrs(p)) {
        const int o = sas.ss.order[a];
        EXPECT_NEAR((sas.geo.probes.pos.col(p) - pts.col(o)).norm(), sar[o],
                    1e-9)
            << "seed " << seed;
      }
    }
    expect_sr(sas, pts, sar, 5000);
  }
}

/**
 * Genuine k-fold points with radii jittered by up to 0.9 TAU_C: every probe
 * has three atoms and the areas do not depend on how the jitter splits them.
 */
TEST(BuildSasTest, JitteredKFoldPoints) {
  const double reach = 2.0, s = 1.5, rl = std::sqrt(reach * reach - s * s),
               rc = 1.8, rlg = 1.9, graze = 5 * kPi / 180;
  Matrix3Xd four(3, 4);
  four.col(0) << -s, 0, 0;
  four.col(1) << s, 0, 0;
  four.col(2) << 0, rl, -rc;
  const Vector3d nl(std::sin(0.7) * std::cos(graze),
                    std::cos(0.7) * std::cos(graze), std::sin(graze));
  four.col(3) = Vector3d(0, rl, 0) - rlg * nl;
  ArrayXd four_r(4);
  four_r << reach, reach, rc, rlg;

  const std::array<std::pair<Matrix3Xd, ArrayXd>, 3> cases {
    std::pair { star(4, 110, reach), ArrayXd::Constant(5, reach) },
    std::pair { star(5, 110, reach), ArrayXd::Constant(6, reach) },
    std::pair { four, four_r },
  };
  for (const auto &[pts, base_r]: cases) {
    for (int seed = 0; seed < 10; ++seed) {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> u(-0.9 * kSurfaceLengthEps,
                                               0.9 * kSurfaceLengthEps);
      ArrayXd sar = base_r;
      for (int i = 0; i < sar.size(); ++i)
        sar[i] += u(rng);
      expect_sr(surface(pts, sar), pts, sar);
    }
  }
}

/**
 * Coplanar a, b, c through x and x', and l nearly tangent to a with x just
 * inside it: the triangulation keeps only the faces through the near pair,
 * which the old overlap graph refused, leaving crossings without vertices.
 */
TEST(BuildSasTest, NearPairInCoplanarQuad) {
  const double h = 5e-4, delta = 5e-7, a = 1.5, b = 1.2, c = 1.3, rl = 1.5,
               beta = 100 * kPi / 180, gamma = -110 * kPi / 180;
  Matrix3Xd pts(3, 4);
  ArrayXd sar(4);
  pts.col(1) << -a, 0, 0;
  pts.col(2) << b * std::cos(beta), b * std::sin(beta), 0;
  pts.col(3) << c * std::cos(gamma), c * std::sin(gamma), 0;
  sar[1] = std::sqrt(a * a + h * h);
  sar[2] = std::sqrt(b * b + h * h);
  sar[3] = std::sqrt(c * c + h * h);
  pts.col(0) = pts.col(1) + (sar[1] + rl - delta) * Vector3d::UnitX();
  sar[0] = rl;

  const Sas sas = solve(pts, sar, 0.5);
  expect_sr(sas, pts, sar, 50000);
  for (const RingArc &arc: sas.geo.arcs)
    EXPECT_LE(arc.dphi, kTwoPi);
}

Vector3d cap_center(const Vector3d &n, const double a, const double rc,
                    const double rs) {
  return (a + std::sqrt(a * a - rs * rs + rc * rc)) * n;
}

/**
 * Cap t nested in cap u and internally tangent to it at X, with a great
 * circle cap c through X: whether t is hidden was a rounding coin flip.
 */
TEST(BuildSasTest, InternallyTangentNestedCaps) {
  const double rs = 1.5, au = 40 * kPi / 180, at = 20 * kPi / 180;
  const Vector3d nu(0, 0, 1), x = rs * Vector3d(std::sin(au), 0, std::cos(au)),
                              nt(std::sin(au - at), 0, std::cos(au - at)),
                              m(std::cos(au), 0, -std::sin(au)),
                              nc = (x / rs).cross(m).normalized();
  Matrix3Xd pts(3, 4);
  ArrayXd sar(4);
  pts.col(0).setZero();
  sar[0] = rs;
  pts.col(1) = cap_center(nu, rs * std::cos(au), 2.0, rs);
  sar[1] = 2.0;
  pts.col(2) = cap_center(nt, rs * std::cos(at), 1.6, rs);
  sar[2] = 1.6;
  pts.col(3) = cap_center(nc, 0.0, 2.2, rs);
  sar[3] = 2.2;

  const Sas sas = solve(pts, sar, 0.5);
  expect_sr(sas, pts, sar, 50000);
}

/**
 * A tiny cap j crossed steeply by a great-circle cap c, a second tiny cap k
 * linked to that vertex through c, and a cap e cutting circle j just past
 * the vertex: the representative of the old clustering slid along j past
 * the distinct vertex and the dart rings stopped alternating.
 */
TEST(BuildSasTest, TinyCapSteepCrossing) {
  const double rs = 1.5, rt = 1.5, rbig = 2.2, delta = 6e-5, eps = 4e-6,
               beta = 80 * kPi / 180, betak = 45 * kPi / 180,
               betae = 80 * kPi / 180, rho = 1e-2, alc = kPi / 2,
               ale = 60 * kPi / 180;
  const double alpha = std::asin(rho / rs), atiny = rs * std::cos(alpha);
  const Vector3d nj(0, 0, 1),
      xj = rs * Vector3d(std::sin(alpha), 0, std::cos(alpha)), uj = xj / rs,
      yhat(0, 1, 0), mhat(std::cos(alpha), 0, -std::sin(alpha));
  const Vector3d tc = std::cos(beta) * yhat + std::sin(beta) * mhat;
  Vector3d ic = uj.cross(tc).normalized();
  if (ic.dot(yhat) > 0)
    ic = -ic;
  const Vector3d nc = (std::cos(alc) * uj + std::sin(alc) * ic).normalized();
  const double dphi_e = eps / rho;
  const Vector3d xe = rs
                      * Vector3d(std::sin(alpha) * std::cos(dphi_e),
                                 std::sin(alpha) * std::sin(dphi_e),
                                 std::cos(alpha));
  const Vector3d uk = std::cos(delta / rs) * uj + std::sin(delta / rs) * tc;
  const Vector3d perp = uk.cross(tc).normalized(),
                 w = std::cos(betak) * tc + std::sin(betak) * perp,
                 nk = (std::cos(alpha) * uk + std::sin(alpha) * w).normalized();
  const Vector3d ue = xe / rs, ye(-std::sin(dphi_e), std::cos(dphi_e), 0);
  Vector3d me = ue.cross(ye).normalized();
  if (me.dot(mhat) < 0)
    me = -me;
  const Vector3d te = std::cos(betae) * ye + std::sin(betae) * me;
  Vector3d ie = ue.cross(te).normalized();
  if (ie.dot(yhat) < 0)
    ie = -ie;
  const Vector3d ne = (std::cos(ale) * ue + std::sin(ale) * ie).normalized();

  Matrix3Xd pts(3, 5);
  ArrayXd sar(5);
  pts.col(0).setZero();
  sar[0] = rs;
  pts.col(1) = cap_center(nj, atiny, rt, rs);
  sar[1] = rt;
  pts.col(2) = cap_center(nc, rs * std::cos(alc), rbig, rs);
  sar[2] = rbig;
  pts.col(3) = cap_center(nk, atiny, rt, rs);
  sar[3] = rt;
  pts.col(4) = cap_center(ne, rs * std::cos(ale), rbig, rs);
  sar[4] = rbig;

  const Sas sas = solve(pts, sar, 0.5);
  expect_sr(sas, pts, sar, 50000);
}

/**
 * A ball inside another, internally tangent, below a chain.
 */
Masked tangent_ball_chain(const int inner) {
  Masked m { Matrix3Xd(3, 2), ArrayXd(2), {} };
  m.pts.col(inner) << 2, 0, 0;
  m.sar[inner] = 3;
  m.pts.col(1 - inner) << 0, 0, 0;
  m.sar[1 - inner] = 5;
  m.active = append_chain(m.pts, m.sar, { 2, 0, 4.4 });
  return m;
}

TEST(BuildSasTest, BandInternallyTangentBallDropped) {
  for (const int inner: { 0, 1 }) {
    SCOPED_TRACE(inner);
    const Masked m = tangent_ball_chain(inner);
    const ArrayXi rest = all_but(5, { inner });

    seed_thread(0);
    const Surface full = surface(m.pts, m.sar, m.active);
    seed_thread(0);
    const Surface part = surface_only(m.pts, m.sar, m.active, rest);
    expect_same_surface(full, part, rest);
  }
}

/**
 * Identical twins carry one surface: the active twin keeps it when exactly
 * one is active, else the lower index does; with no active atom nothing is
 * kept.
 */
TEST(BuildSasTest, IdenticalTwinsKeepActive) {
  Matrix3Xd pts(3, 3);
  pts.col(0) << 0, 0, 0;
  pts.col(1) << 0, 0, 0;
  pts.col(2) << 4, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(3, 3.0);
  const Sas lone = solve(pts.rightCols(2), sar.head(2));

  for (const int alive: { 0, 1 }) {
    ArrayXb active = ArrayXb::Constant(3, true);
    active[1 - alive] = false;
    auto [ss, geo] = solve(pts, sar, active);
    ASSERT_EQ(ss.n_active, 2);
    EXPECT_EQ(ss.order[0], alive);
    EXPECT_EQ(geo.area[0], lone.geo.area[0]);
  }

  {
    auto [ss, geo] = solve(pts, sar, ArrayXb::Constant(3, true));
    ASSERT_EQ(ss.order.size(), 2);
    EXPECT_TRUE((ss.order != 1).all());
  }
  {
    auto [ss, geo] = solve(pts, sar, ArrayXb::Constant(3, false));
    EXPECT_EQ(ss.order.size(), 0);
  }
}

/**
 * The exact triple drops its middle; shifted by 1e-9 it keeps all three;
 * rigid motions keep the areas.
 */
TEST(BuildSasTest, SharedCircleMiddle) {
  for (int mid = 0; mid < 3; ++mid) {
    ArrayXd sar;
    const Matrix3Xd pts = shared_circle_pts(mid, sar);
    const Surface s = surface(pts, sar);
    ASSERT_EQ(s.sas.n_atoms(), 2);
    EXPECT_TRUE((s.sas.order() != mid).all());
    EXPECT_EQ(s.ses.concave().pos.cols(), 0);
    expect_sr(s, pts, sar);
  }

  for (const double shift: { 1e-9, -1e-9 }) {
    ArrayXd sar;
    Matrix3Xd pts = shared_circle_pts(1, sar);
    pts(2, 1) += shift;
    const Surface s = surface(pts, sar);
    ASSERT_EQ(s.sas.n_atoms(), 3);
    expect_sr(s, pts, sar);
  }

  for (int seed = 0; seed < 8; ++seed) {
    ArrayXd sar;
    const Matrix3Xd pts = rigid(shared_circle_pts(seed % 3, sar), seed);
    expect_sr(surface(pts, sar), pts, sar);
  }
}

/**
 * A fourth sphere containing one outer sphere of the shared circle: the
 * middle still has no surface and is dropped although its outer partner is.
 */
TEST(BuildSasTest, SharedCircleMiddleUnderContainedOuter) {
  for (int mid = 0; mid < 3; ++mid) {
    ArrayXd sar3;
    const Matrix3Xd pts3 = shared_circle_pts(mid, sar3);
    Matrix3Xd pts(3, 4);
    pts.leftCols(3) = pts3;
    pts.col(3) << 0, 0, 1.140625;
    ArrayXd sar(4);
    sar << sar3, 3.5;

    const Sas sas = solve(pts, sar);
    ASSERT_EQ(sas.ss.order.size(), 2);
    EXPECT_TRUE((sas.ss.order != mid).all());
    EXPECT_EQ(sas.geo.probes.pos.cols(), 0);
    expect_sr(sas, pts, sar);
  }
}

/**
 * Append `n` spheres of radius 3 off the z axis, overlapping every sphere
 * within 2 of the origin without containment.
 */
void append_off_axis(Matrix3Xd &pts, ArrayXd &sar, const int n,
                     const int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> rho(1.5, 3.0), phi(0, kTwoPi),
      zd(-2.0, 2.0);
  const int m = static_cast<int>(pts.cols());
  pts.conservativeResize(E::NoChange, m + n);
  sar.conservativeResize(m + n);
  for (int q = m; q < m + n; ++q) {
    const double r = rho(rng), ang = phi(rng);
    pts.col(q) << r * std::cos(ang), r * std::sin(ang), zd(rng);
    sar[q] = 3.0;
  }
}

/**
 * The shared-circle triple among enough off-axis neighbors that its row
 * holds several unrelated circles on either side of the matching keys.
 */
TEST(BuildSasTest, SharedCircleMiddleAmongNeighbors) {
  for (int mid = 0; mid < 3; ++mid) {
    ArrayXd sar;
    Matrix3Xd pts = shared_circle_pts(mid, sar);
    append_off_axis(pts, sar, 6, mid);

    const Surface s = surface(pts, sar);
    ASSERT_EQ(s.sas.n_atoms(), 8);
    EXPECT_EQ(s.sas.convex().area.size(), 8);
    EXPECT_TRUE((s.sas.order() != mid).all());
    expect_sr(s, pts, sar);
  }
}

/**
 * The shared-circle triple below a chain whose foot is near the middle and
 * the upper outer sphere, hop 3, but not the lower outer sphere, hop 4,
 * which for `mid == 1` holds the smallest index of the triple, so the
 * middles pass reads its row from outside the three-hop set.
 */
Masked shared_circle_chain(const int mid) {
  Masked m;
  m.pts = shared_circle_pts(mid, m.sar);
  m.active = append_chain(m.pts, m.sar, { 0, 0, 5.078125 });
  return m;
}

TEST(BuildSasTest, BandSharedCircleRowAtHopFour) {
  for (int mid = 0; mid < 3; ++mid) {
    SCOPED_TRACE(mid);
    const Masked m = shared_circle_chain(mid);
    const ArrayXi rest = all_but(6, { mid });

    seed_thread(0);
    const Surface full = surface(m.pts, m.sar, m.active);
    seed_thread(0);
    const Surface part = surface_only(m.pts, m.sar, m.active, rest);
    expect_same_surface(full, part, rest);
  }
}

struct FullRank {
  ArrayXi rank, hop;
};

/**
 * Ranks (0 active, 1 need, 2 shell, 3 occluder, 4 dropped) and near hops
 * from the active set, decided by the exact predicates over every pair.
 */
FullRank full_rank(const Matrix3Xd &pts, const ArrayXd &sar,
                   const ArrayXb &active) {
  const int n = static_cast<int>(sar.size());
  const double rmax = sar.maxCoeff(), wmax = rmax * rmax;
  Matrix4Xd lifted(4, n);
  lifted.topRows(3) = pts;
  lifted.row(3) = (wmax - sar.square()).sqrt().transpose();
  const BallExact ex = BallExact::make(lifted, wmax);

  ArrayXi keep = ArrayXi::Ones(n);
  std::vector<std::vector<int>> near(n), over(n);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double d = (pts.col(j) - pts.col(i)).norm();
      if (d <= sar[i] + sar[j] + kPairLengthEps) {
        near[i].push_back(j);
        near[j].push_back(i);
      }

      const int in = ex.contained(i, j);
      if (in >= 0) {
        keep[in == 0 || (in == 2 && !active[i] && active[j]) ? i : j] = 0;
        continue;
      }
      if (ex.overlap(i, j) == Sgn::kPos)
        over[i].push_back(j);
    }
  }
  for (int i = 0; i < n; ++i) {
    for (const int j: over[i]) {
      for (const int k: over[i]) {
        if (k <= j || !std::binary_search(over[j].begin(), over[j].end(), k))
          continue;
        const int mid = ex.shared_circle({ i, j, k });
        if (mid >= 0)
          keep[mid == 0 ? i : mid == 1 ? j : k] = 0;
      }
    }
  }

  FullRank ref { ArrayXi::Constant(n, 4), ArrayXi::Constant(n, n) };
  std::vector<int> frontier, reach;
  for (int i = 0; i < n; ++i) {
    if (!active[i])
      continue;
    ref.hop[i] = 0;
    reach.push_back(i);
    if (keep[i] != 0) {
      ref.rank[i] = 0;
      frontier.push_back(i);
    }
  }
  for (int h = 1; h <= 3; ++h) {
    std::vector<int> next;
    for (const int i: frontier) {
      for (const int j: near[i]) {
        if (keep[j] == 0 || ref.rank[j] < 4)
          continue;
        ref.rank[j] = h;
        next.push_back(j);
      }
    }
    frontier = std::move(next);
  }
  for (int h = 1; !reach.empty(); ++h) {
    std::vector<int> next;
    for (const int i: reach) {
      for (const int j: near[i]) {
        if (ref.hop[j] <= h)
          continue;
        ref.hop[j] = h;
        next.push_back(j);
      }
    }
    reach = std::move(next);
  }
  return ref;
}

ArrayXi prepared_rank(const SurfaceSpheres &ss, const int n) {
  ArrayXi got = ArrayXi::Constant(n, 4);
  for (int p = 0; p < ss.order.size(); ++p) {
    got[ss.order[p]] = p < ss.n_active  ? 0
                       : p < ss.n_solve ? 1
                       : p < ss.n_enum  ? 2
                                        : 3;
  }
  return got;
}

/**
 * `prepare` against the exact oracle over every pair: the same ranks, and
 * no zero distance in the overlap graph.
 */
SurfaceSpheres expect_ranks(const Matrix3Xd &pts, const ArrayXd &sar,
                            const ArrayXb &active) {
  std::optional<SurfaceSpheres> ss = prepare(pts, sar - kRp, active, kRp);
  EXPECT_TRUE(ss);
  const int n = static_cast<int>(sar.size());
  EXPECT_TRUE(
      (prepared_rank(*ss, n) == full_rank(pts, sar, active).rank).all());
  EXPECT_TRUE((ss->d > 0).all());
  return std::move(*ss);
}

/**
 * Keep flags and ranks of a sparse mask over an elongated random set and
 * over the hand-built band fixtures, against `prepare`.
 */
TEST(BuildSasTest, PrepareRanksMatchOracle) {
  for (const int n: { 60, 200 }) {
    for (int seed = 0; seed < 3; ++seed) {
      SCOPED_TRACE(testing::Message() << n << " " << seed);
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> uxy(0, 6), uz(0, 100),
          ur(1.5, 2.5);
      std::uniform_int_distribution<int> ui(0, n - 1);
      Matrix3Xd pts(3, n);
      ArrayXd sar(n);
      for (int i = 0; i < n; ++i) {
        pts.col(i) << uxy(rng), uxy(rng), uz(rng);
        sar[i] = ur(rng);
      }
      ArrayXb active = ArrayXb::Constant(n, false);
      active[ui(rng)] = true;
      active[ui(rng)] = true;

      ASSERT_TRUE((full_rank(pts, sar, active).hop > 4).any());
      expect_ranks(pts, sar, active);
    }
  }

  {
    const Masked m = occluder_chain();
    const SurfaceSpheres ss = expect_ranks(m.pts, m.sar, m.active);
    EXPECT_EQ(ss.order.size(), 5);
    EXPECT_EQ(ss.n_active, 2);
    EXPECT_EQ(ss.n_solve, 3);
    EXPECT_EQ(ss.n_enum, 4);
  }
  {
    const Masked m = contained_partner();
    const SurfaceSpheres ss = expect_ranks(m.pts, m.sar, m.active);
    EXPECT_EQ(ss.n_active, 1);
    EXPECT_EQ(ss.n_solve, 2);
    EXPECT_EQ(ss.n_enum, 3);
  }
  for (const double x: { 10.0, 10.8 }) {
    for (const double r: { 1.5, 2.0, 1.45 }) {
      SCOPED_TRACE(testing::Message() << x << " " << r);
      const Masked m = twin_at_hop_four(x, r);
      expect_ranks(m.pts, m.sar, m.active);
    }
  }
  for (const int inner: { 0, 1 }) {
    SCOPED_TRACE(inner);
    const Masked m = tangent_ball_chain(inner);
    expect_ranks(m.pts, m.sar, m.active);
  }
  for (int mid = 0; mid < 3; ++mid) {
    SCOPED_TRACE(mid);
    const Masked m = shared_circle_chain(mid);
    expect_ranks(m.pts, m.sar, m.active);
  }
  {
    const Matrix3Xd ring = star(4, 110, 2.0).rightCols(4);
    Matrix3Xd pts(3, 5);
    pts.leftCols(4) = ring;
    pts.col(4) = ring.col(3) + Vector3d(0, -3.0, 0);
    ArrayXd sar(5);
    sar << 2.0, 2.0, 2.0, 2.0, 1.5;
    ArrayXb active = ArrayXb::Constant(5, false);
    active[4] = true;

    const SurfaceSpheres ss = expect_ranks(pts, sar, active);
    EXPECT_EQ(ss.n_active, 1);
    EXPECT_EQ(ss.n_solve, 2);
    EXPECT_EQ(ss.n_enum, 5);
  }
}

/**
 * Identical twins through one circle of a third, collinear sphere: the
 * contained filter already settles which twin stays, the shared-circle pass
 * must not drop the other.
 */
TEST(BuildSasTest, SharedCircleSparesTwin) {
  for (const int alive: { 1, 2 }) {
    Matrix3Xd pts(3, 3);
    pts.col(0) << 0, 0, 0;
    pts.col(1) << 0, 0, 2;
    pts.col(2) << 0, 0, 2;
    ArrayXd sar(3);
    sar << 3.5, 3, 3;
    append_off_axis(pts, sar, 5, alive);
    ArrayXb active = ArrayXb::Constant(8, true);
    active[3 - alive] = false;

    const Sas sas = solve(pts, sar, active);
    ASSERT_EQ(sas.ss.order.size(), 7);
    EXPECT_EQ(sas.ss.n_active, 7);
    EXPECT_TRUE((sas.ss.order != 3 - alive).all());
    EXPECT_TRUE((sas.ss.order == alive).any());

    const ArrayXi rest = all_but(8, { 3 - alive });
    const Sas ref = solve(pts(E::all, rest), sar(rest));
    EXPECT_NEAR(sas.geo.area.sum(), ref.geo.area.sum(), 1e-9);
  }
}

/**
 * A third sphere through the point of circle (a, b) at frame angle pi: the
 * vertex's atan2 is +-pi to rounding and must fold with its exact class, not
 * toward the ray.
 */
TEST(BuildSasTest, VertexAtFrameAnglePi) {
  const Vector3d ca(0, 0, 0), cb(2, 0, 0);
  const double ra = 1.5, rb = 1.5;
  const Vector3d d = cb - ca;
  const Vector3d e1 =
      d.cross(Vector3d::Unit(BallExact::reference_axis(d))).normalized();
  const double a = (d.squaredNorm() + ra * ra - rb * rb) / (2 * d.norm()),
               rl = std::sqrt(ra * ra - a * a);
  const Vector3d p = ca + a * d.normalized() - rl * e1;

  std::mt19937 rng(1);
  std::normal_distribution<double> nd;
  for (int seed = 0; seed < 40; ++seed) {
    Matrix3Xd pts(3, 3);
    pts.col(0) = ca;
    pts.col(1) = cb;
    pts.col(2) << 1 + 0.3 * nd(rng), 1 + 0.6 * nd(rng), -2 + 0.5 * nd(rng);
    ArrayXd sar(3);
    sar << ra, rb, (pts.col(2) - p).norm();
    if (sar[2] < 1.0)
      continue;

    const Sas sas = solve(pts, sar, 0.5);
    expect_sr(sas, pts, sar);
  }
}

template <class F>
void for_each_axis_variant(const Matrix3Xd &pts, const F &f) {
  std::array<int, 3> perm { 0, 1, 2 };
  do {
    for (int signs = 0; signs < 8; ++signs) {
      Matrix3Xd p(3, pts.cols());
      for (int k = 0; k < 3; ++k)
        p.row(k) = ((signs >> k & 1) != 0 ? -1.0 : 1.0) * pts.row(perm[k]);
      f(p);
    }
  } while (std::next_permutation(perm.begin(), perm.end()));
}

/**
 * Over every axis permutation, sign flip and index order of
 * `tangent_through_pts`: the tangent pair carries no cap, the third sphere
 * sees two tangent discs, no probe sits at the point.
 */
TEST(BuildSasTest, TangentPairThirdSphereThroughPoint) {
  ArrayXd sar;
  const Matrix3Xd base = tangent_through_pts(sar);
  const double rp = 0.5;
  int n_variants = 0;

  for_each_axis_variant(base, [&](const Matrix3Xd &pts) {
    const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
    for_each_index_order(pts, sar,
                         [&](const Matrix3Xd &p, const ArrayXd &r,
                             const std::vector<int> &order) {
                           const Sas sas = solve(p, r, rp);
                           ASSERT_EQ(sas.ss.order.size(), 3);
                           EXPECT_EQ(sas.geo.probes.pos.cols(), 0);
                           ArrayXd sr_p(3);
                           for (int i = 0; i < 3; ++i)
                             sr_p[i] = sr[order[i]];
                           expect_areas(sas, sr_p, r, 20000);
                           ++n_variants;
                         });
  });
  EXPECT_EQ(n_variants, 288);
}

/**
 * The tangent pair plus two spheres through the tangency point: a 4-fold
 * point where the perturbation would reject the cut points of the two real
 * circles against the tangent apex, the limit accepts them.
 */
TEST(BuildSasTest, TangentPairFourFoldPoint) {
  ArrayXd sar3;
  const Matrix3Xd base3 = tangent_through_pts(sar3);
  Matrix3Xd base(3, 4);
  base.leftCols(3) = base3;
  base.col(3) << 1.5, 0, 1.5;
  const ArrayXd sar = ArrayXd::Constant(4, 1.5);
  const Vector3d x(1.5, 0, 0);
  const ArrayXd sr = sr_sasa_impl(base, sar, 20000, SrSasaMethod::kDirect);

  for_each_index_order(base, sar,
                       [&](const Matrix3Xd &p, const ArrayXd &r,
                           const std::vector<int> &order) {
                         const Sas sas = solve(p, r, 0.5);
                         ASSERT_EQ(sas.ss.order.size(), 4);
                         const int c = probe_at(sas.geo.probes.pos, x);
                         ASSERT_GE(c, 0);
                         EXPECT_EQ(sas.geo.probes.atoms.degree(c), 4);
                         ArrayXd sr_p(4);
                         for (int i = 0; i < 4; ++i)
                           sr_p[i] = sr[order[i]];
                         expect_areas(sas, sr_p, r, 20000);
                       });
}

TEST(BuildSasTest, ApexTangentAtVertexIndexOrders) {
  const TangentApex t = tangent_apex(0);
  const ArrayXd sr = sr_sasa_impl(t.pts, t.sar, 20000, SrSasaMethod::kDirect);

  for_each_index_order(t.pts, t.sar,
                       [&](const Matrix3Xd &p, const ArrayXd &r,
                           const std::vector<int> &order) {
                         const Sas sas = solve(p, r);
                         expect_atoms_within(sas.ss, sas.geo, t.x, 1e-6, 4);
                         expect_no_coincident_pairs(p, r);
                         ArrayXd sr_p(4);
                         for (int i = 0; i < 4; ++i)
                           sr_p[i] = sr[order[i]];
                         expect_areas(sas, sr_p, r, 20000);
                       });
}

/**
 * Two circles on sphere a exactly tangent (face discriminant zero on the
 * stored heights, radii all 5 so the lift is exact): internally at
 * (1,0,7), externally at (7,0,-1) and (1,0,-1), with and without a fourth
 * sphere through the tangency point. The corner at an internal tangency is
 * exactly pi, the arcs between coincident probes have zero length, and the
 * result must not depend on the index order or a rigid motion.
 */
TEST(BuildSasTest, ExactCircleTangencies) {
  const std::vector<std::vector<Vector3d>> cases {
    { { 0, 0, 0 }, { 0, 0, 6 }, { 1, 0, 7 } },
    { { 0, 0, 0 }, { 0, 0, 6 }, { 7, 0, -1 } },
    { { 0, 0, 0 }, { 0, 0, 6 }, { 1, 0, -1 } },
    { { 0, 0, 0 }, { 0, 0, 6 }, { 1, 0, 7 }, { 4, 5, 3 } },
    { { 0, 0, 0 }, { 0, 0, 6 }, { 1, 0, 7 }, { 4, -5, 3 } },
  };
  for (const auto &centers: cases) {
    const int n = static_cast<int>(centers.size());
    Matrix3Xd pts(3, n);
    for (int i = 0; i < n; ++i)
      pts.col(i) = centers[i];
    const ArrayXd sar = ArrayXd::Constant(n, 5.0);
    const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
    const ArrayXd ref = solve(pts, sar).geo.area;

    for_each_index_order(pts, sar,
                         [&](const Matrix3Xd &p, const ArrayXd &r,
                             const std::vector<int> &order) {
                           const Sas sas = solve(p, r);
                           ArrayXd sr_p(n);
                           for (int i = 0; i < n; ++i) {
                             sr_p[i] = sr[order[i]];
                             EXPECT_NEAR(sas.geo.area[i], ref[order[i]], 1e-12);
                           }
                           expect_areas(sas, sr_p, r, 20000);
                         });

    for (int seed = 0; seed < 16; ++seed) {
      const Matrix3Xd q = rigid(pts, seed);
      expect_sr(solve(q, sar), q, sar);
    }
  }
}

/**
 * Per-atom areas of every index order agree with the first order to 1e-9
 * and with Shrake-Rupley; returns the first order's areas by atom.
 */
ArrayXd expect_order_invariant_areas(const Matrix3Xd &pts, const ArrayXd &sar) {
  const int n = static_cast<int>(sar.size());
  const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
  const Sas first = solve(pts, sar);
  ArrayXd ref(n);
  for (int p = 0; p < n; ++p)
    ref[first.ss.order[p]] = first.geo.area[p];

  for_each_index_order(pts, sar,
                       [&](const Matrix3Xd &p, const ArrayXd &r,
                           const std::vector<int> &order) {
                         const Sas sas = solve(p, r);
                         ArrayXd sr_p(n);
                         std::string tag;
                         for (int i = 0; i < n; ++i) {
                           sr_p[i] = sr[order[i]];
                           tag += std::to_string(order[i]);
                         }
                         expect_areas(sas, sr_p, r, 20000);
                         for (int i = 0; i < n; ++i) {
                           const int o = order[sas.ss.order[i]];
                           EXPECT_NEAR(sas.geo.area[i], ref[o], 1e-9)
                               << "atom " << o << " order " << tag;
                         }
                       });
  return ref;
}

/**
 * A tangent pair `p`, `q` and two spheres `b`, `c` through their touching
 * point, all of radius 3: on every sphere the two caps of the others touch
 * at the origin, their centers coplanar with it. Only some orders resolve
 * the touch into a 4-atom probe; the others see two caps that merely
 * touch, which are not joined (analytic_surface page, SAS step 5), so every
 * order gives the same areas. The areas are pinned from this
 * implementation's optdebug build at 1e-5; reldeb agrees.
 */
TEST(BuildSasTest, TouchingCapsWithoutLensVertex) {
  Matrix3Xd pts(3, 4);
  pts << 0, 0, 2, -2,  //
      0, 0, 1, -1,     //
      -3, 3, 2, 2;
  const ArrayXd sar = ArrayXd::Constant(4, 3.0);

  const ArrayXd ref = expect_order_invariant_areas(pts, sar);
  EXPECT_NEAR(ref[0], 103.243270, 1e-5);
  EXPECT_NEAR(ref[1], 46.171794, 1e-5);
  EXPECT_NEAR(ref[2], 74.707532, 1e-5);
  EXPECT_NEAR(ref[3], 74.707532, 1e-5);
}

/**
 * The tangent pair of radius 7 with four spheres through the touching
 * point whose centers are coplanar with it: a 6-fold touch with three
 * tangent pairs, whose deleted circles leave the roots of `{0, 2, 4}` and
 * `{0, 3, 5}` at the origin on no common circle.
 */
TEST(BuildSasTest, TouchingCapsSixFold) {
  Matrix3Xd pts(3, 6);
  pts << 0, 0, 6, -6, 6, -6,  //
      0, 0, 3, -3, 3, -3,     //
      -7, 7, 2, 2, -2, -2;
  const ArrayXd sar = ArrayXd::Constant(6, 7.0);

  const ArrayXd ref = expect_order_invariant_areas(pts, sar);

  int k = 0;
  for_each_index_order(
      pts, sar,
      [&](const Matrix3Xd &p, const ArrayXd &r, const std::vector<int> &order) {
        if (k++ % 37 != 0)
          return;
        const Sas sas = solve(p, r);
        expect_no_unmerged_pairs(sas.geo.probes.pos, merge_tol(sas));
        const int c = probe_at(sas.geo.probes.pos, Vector3d::Zero());
        ASSERT_GE(c, 0) << "order " << order[0] << order[1] << order[2]
                        << order[3] << order[4] << order[5];
      });
}

/**
 * Rigid motions of the two tangent fixtures: rounding turns the exact
 * tangency into a circle of radius ~1e-8 with two real cut points whose
 * floating positions do not resolve the circle.
 */
TEST(BuildSasTest, RoundingTinyCirclesUnderRigidMotion) {
  ArrayXd sar3;
  const Matrix3Xd base3 = tangent_through_pts(sar3);
  const TangentApex t = tangent_apex(0);

  for (int seed = 0; seed < 128; ++seed) {
    const Matrix3Xd p3 = rigid(base3, seed);
    expect_sr(solve(p3, sar3, 0.5), p3, sar3);

    const Matrix3Xd p4 = rigid(t.pts, seed);
    expect_sr(solve(p4, t.sar), p4, t.sar);
  }
}

TEST(BuildSasTest, TangentApexUnderRigidMotion) {
  for (const double d: { -5e-7, 0.0, 1e-7, 5e-7, 2e-6 }) {
    for (int seed = -1; seed < 3; ++seed) {
      TangentApex t = tangent_apex(d);
      const Matrix3Xd pts = seed < 0 ? t.pts : rigid(t.pts, seed);
      expect_sr(surface(pts, t.sar), pts, t.sar);
    }
  }
}

/**
 * Genuine k-fold points: `k` equal spheres with centers on a cone through
 * the origin. Under rigid motions the coincident probes round apart into
 * distinct points whose probes together touch the `k` atoms and whose
 * areas are those of the exact point.
 */
TEST(BuildSasTest, MergedKFoldUnderRigidMotion) {
  for (const int k: { 4, 5 }) {
    const Matrix3Xd base = star(k, 110, 2.0).rightCols(k);
    const ArrayXd sar = ArrayXd::Constant(k, 2.0);
    const Surface ref = surface(base, sar);
    expect_sr(ref, base, sar);
    ASSERT_EQ(ref.sas.convex().area.size(), k);
    ArrayXd ref_area(k);
    for (int p = 0; p < k; ++p)
      ref_area[ref.sas.order()[p]] = ref.sas.convex().area[p];

    for (int seed = 0; seed < 16; ++seed) {
      const auto [rot, t] = rigid_motion(seed);
      const Matrix3Xd pts = (rot * base).colwise() + t;
      const Surface s = surface(pts, sar);
      const Sas sas = solve(pts, sar);
      expect_atoms_within(sas.ss, sas.geo, t, 1e-6, k);
      expect_no_coincident_pairs(sas.geo, triangulate(sas.ss).ex);
      EXPECT_NEAR(s.ses.total_area(), ref.ses.total_area(), 1e-9)
          << "k " << k << " seed " << seed;
      ASSERT_EQ(s.sas.convex().area.size(), k);
      for (int p = 0; p < k; ++p) {
        EXPECT_NEAR(s.sas.convex().area[p], ref_area[s.sas.order()[p]], 1e-9)
            << "k " << k << " seed " << seed << " atom " << p;
      }
    }
  }
}

/**
 * Sphere c through exactly one point P of circle (a, b), the tangency
 * perturbed to a cut with both roots at P. The three circles through P
 * share the tangent `y`; (a, b) and (b, c) are accessible full loops back
 * to P (an hourglass on b), (a, c) is buried. One probe, whose tangents are
 * the loops' `+-y`, hemispheres that cancel in pairs, so no face.
 */
TEST(BuildSasTest, GrazingVertex) {
  Matrix3Xd pts(3, 3);
  pts.col(0) << 0, 0, 0;
  pts.col(1) << 0, 0, 6;
  pts.col(2) << 7, 0, 7;
  const ArrayXd sar = ArrayXd::Constant(3, 5.0);
  const Vector3d x(4, 0, 3);

  const SurfaceGeometry s = surface(pts, sar).ses;
  ASSERT_EQ(s.concave().pos.cols(), 1);
  const int c = probe_at(s.concave().pos, x);
  ASSERT_EQ(c, 0);
  const int buried = circle_of(s, 0, 2);
  ASSERT_GE(buried, 0);

  int n_loops = 0;
  for (const RingArc &arc: s.toroidal().arcs) {
    ASSERT_EQ(arc.beg, c);
    ASSERT_EQ(arc.end, c);
    if (arc.circ == buried) {
      EXPECT_NEAR(arc.dphi, 0, 1e-9);
      continue;
    }
    EXPECT_NEAR(arc.dphi, kTwoPi, 1e-9);
    ++n_loops;
  }
  EXPECT_EQ(n_loops, 2);

  const auto tan = hemispheres(s, c);
  ASSERT_EQ(tan.cols(), 2 * n_loops);
  EXPECT_EQ(s.concave().caps.off.degree(c), tan.cols());
  for (int i = 0; i < tan.cols(); ++i)
    EXPECT_NEAR(std::abs(tan(1, i)), 1, 1e-12) << i;
  EXPECT_NEAR(tan.rowwise().sum().norm(), 0, 1e-12);

  ASSERT_EQ(s.concave().area.size(), 1);
  EXPECT_EQ(s.concave().area[0], 0);
}

/**
 * Active `a` overlaps only `b`; `b, c, d, e` pass through one point off `a`
 * and `c, d, e` overlap nothing active, so probe `(c, d, e)` is owned by a
 * shell sphere. Its circles are ringed all the same and the point is one
 * probe on `{b, c, d, e}`.
 */
TEST(BuildSasTest, CutterOnlyClusterMerged) {
  const Matrix3Xd ring = star(4, 110, 2.0).rightCols(4);
  Matrix3Xd pts(3, 5);
  pts.leftCols(4) = ring;
  pts.col(4) = ring.col(3) + Vector3d(0, -3.0, 0);
  ArrayXd sar(5);
  sar << 2.0, 2.0, 2.0, 2.0, 1.5;
  ArrayXb active = ArrayXb::Constant(5, false);
  active[4] = true;

  const Sas sas = solve(pts, sar, active);
  ASSERT_EQ(sas.ss.n_active, 1);
  ASSERT_EQ(sas.ss.n_solve, 2);
  ASSERT_EQ(sas.ss.n_enum, 5);

  const int c = probe_at(sas.geo.probes.pos, Vector3d::Zero());
  ASSERT_GE(c, 0);
  std::vector<int> atoms;
  for (const int p: sas.geo.probes.atoms.nbrs(c))
    atoms.push_back(sas.ss.order[p]);
  std::sort(atoms.begin(), atoms.end());
  EXPECT_EQ(atoms, (std::vector<int> { 0, 1, 2, 3 }));
  expect_no_unmerged_pairs(sas.geo.probes.pos, merge_tol(sas));
}

/**
 * The two shell probes are one probe of four atoms in every atom order;
 * the all-active run is the control.
 */
TEST(BuildSasTest, CoincidentShellProbesAcrossOccluderCircle) {
  const Matrix3Xd base = occluder_shell_pts();
  const ArrayXd sar = ArrayXd::Constant(6, 1.5);
  const Vector3d x(1.5, 0, 0);

  std::array<int, 6> perm { 0, 1, 2, 3, 4, 5 };
  do {
    Matrix3Xd pts(3, 6);
    ArrayXb shell(6);
    for (int i = 0; i < 6; ++i) {
      pts.col(i) = base.col(perm[i]);
      shell[i] = perm[i] == 5;
    }
    for (const bool all: { false, true }) {
      const std::optional<SurfaceSpheres> ss = prepare(
          pts, sar - 0.5, all ? ArrayXb::Constant(6, true) : shell, 0.5);
      ASSERT_TRUE(ss);
      ASSERT_EQ(ss->n_enum, all ? 6 : 4);
      const SasDelaunay del = triangulate(*ss);
      const SasGeometry geo = build_sas(*ss, del);

      const int c = probe_at(geo.probes.pos, x);
      ASSERT_GE(c, 0) << "all " << all << " perm " << perm[0] << perm[1]
                      << perm[2] << perm[3] << perm[4] << perm[5];
      EXPECT_EQ(geo.probes.atoms.degree(c), 4);
      expect_no_unmerged_pairs(geo.probes.pos, probe_merge_tol(sar.maxCoeff()));
    }
  } while (std::next_permutation(perm.begin(), perm.end()));
}

/**
 * Under a rigid motion the two shell roots round apart by a few 1e-15 into
 * distinct points, no longer one probe. The tangent pair `(0, 1)` rounds
 * to a gap of either sign: apart or exactly tangent, the probes within the
 * ~1e-8 rounding of the tangency root together touch its four atoms;
 * overlapping, its circle of radius ~1e-8 has two cut points that stay
 * distinct probes.
 */
TEST(BuildSasTest, CoincidentShellProbesUnderRigidMotion) {
  const Matrix3Xd base = occluder_shell_pts();
  const ArrayXd sar = ArrayXd::Constant(6, 1.5);
  ArrayXb shell = ArrayXb::Constant(6, false);
  shell[5] = true;

  for (int seed = 0; seed < 32; ++seed) {
    const auto [rot, t] = rigid_motion(seed);
    const Matrix3Xd pts = (rot * base).colwise() + t;
    const std::optional<SurfaceSpheres> ss =
        prepare(pts, sar - 0.5, shell, 0.5);
    ASSERT_TRUE(ss);
    const SasDelaunay del = triangulate(*ss);
    const SasGeometry geo = build_sas(*ss, del);

    SCOPED_TRACE(seed);
    expect_no_coincident_pairs(geo, del.ex);
    if ((pts.col(0) - pts.col(1)).norm() < 3.0)
      continue;

    expect_atoms_within(*ss, geo, rot * Vector3d(1.5, 0, 0) + t, 1e-6, 4);
  }
}

/**
 * Random dense sets: every probe has three tangents, a probe ends at most
 * one arc per circle, and the areas match Shrake-Rupley.
 */
TEST(BuildSasTest, RandomDenseSets) {
  for (const int n: { 60, 200 }) {
    for (int seed = 0; seed < 2; ++seed) {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> u(0, std::cbrt(n) * 1.9),
          ur(1.5, 2.5);
      Matrix3Xd pts(3, n);
      ArrayXd sar(n);
      for (int i = 0; i < n; ++i) {
        pts.col(i) << u(rng), u(rng), u(rng);
        sar[i] = ur(rng);
      }
      const Surface s = surface(pts, sar);

      const ConcavePatches &con = s.ses.concave();
      for (int p = 0; p < con.pos.cols(); ++p) {
        EXPECT_EQ(con.atoms.degree(p), 3);
        EXPECT_EQ(hemispheres(s.ses, p).cols(), 3) << "probe " << p;
      }
      std::vector<std::pair<int, int>> ends;
      for (const RingArc &arc: s.ses.toroidal().arcs) {
        if (arc.beg < 0)
          continue;
        ends.emplace_back(arc.circ, arc.beg);
        ends.emplace_back(arc.circ, arc.end);
      }
      std::sort(ends.begin(), ends.end());
      EXPECT_EQ(std::adjacent_find(ends.begin(), ends.end()), ends.end());

      const double total = s.sas.convex().area.sum();
      EXPECT_NEAR(total, sr_total(pts, sar), 1e-2 * total);
    }
  }
}

/**
 * `n` spheres of radius 1.5 to 2.5 in a box of side 10, every `stride`th
 * active; stride 3 at seed 11 is the fixture of EdgeTablesMatchBruteForce.
 */
Masked random_masked(const int n, const int seed, const int stride) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ud(0.0, 10.0), ur(1.5, 2.5);
  Masked m { Matrix3Xd(3, n), ArrayXd(n), ArrayXb(n) };
  for (int i = 0; i < n; ++i) {
    m.pts.col(i) << ud(rng), ud(rng), ud(rng);
    m.sar[i] = ur(rng);
    m.active[i] = i % stride == 0;
  }
  return m;
}

template <class Lhs, class Rhs>
void expect_same_array(const Lhs &a, const Rhs &b) {
  ASSERT_EQ(a.size(), b.size());
  EXPECT_TRUE((a == b).all());
}

template <class Lhs, class Rhs>
void expect_same_matrix(const Lhs &a, const Rhs &b) {
  ASSERT_EQ(a.rows(), b.rows());
  ASSERT_EQ(a.cols(), b.cols());
  EXPECT_EQ(a, b);
}

void expect_same_caps(const CapTable &a, const CapTable &b) {
  expect_same_array(a.off.off(), b.off.off());
  expect_same_matrix(a.axis, b.axis);
  expect_same_array(a.cosa, b.cosa);
}

void expect_same_circle(const SasCircle &a, const SasCircle &b) {
  EXPECT_EQ(a.axis, b.axis);
  EXPECT_EQ(a.cntr, b.cntr);
  EXPECT_EQ(a.e1, b.e1);
  EXPECT_EQ(a.e2, b.e2);
  EXPECT_EQ(a.aij[0], b.aij[0]);
  EXPECT_EQ(a.aij[1], b.aij[1]);
  EXPECT_EQ(a.rl, b.rl);
  EXPECT_EQ(a.i, b.i);
  EXPECT_EQ(a.j, b.j);
}

void expect_same_arc(const RingArc &a, const RingArc &b) {
  EXPECT_EQ(a.phi, b.phi);
  EXPECT_EQ(a.dphi, b.dphi);
  EXPECT_EQ(a.circ, b.circ);
  EXPECT_EQ(a.beg, b.beg);
  EXPECT_EQ(a.end, b.end);
}

void expect_same_geometry(const SurfaceGeometry &a, const SurfaceGeometry &b) {
  EXPECT_EQ(a.kind(), b.kind());
  EXPECT_EQ(a.rp(), b.rp());
  ASSERT_EQ(a.n_atoms(), b.n_atoms());
  expect_same_array(a.order(), b.order());
  expect_same_matrix(a.pts(), b.pts());
  expect_same_array(a.sar(), b.sar());
  expect_same_array(a.r(), b.r());

  expect_same_array(a.convex().area, b.convex().area);
  expect_same_caps(a.convex().caps, b.convex().caps);

  const ToroidalPatches &ta = a.toroidal(), &tb = b.toroidal();
  ASSERT_EQ(ta.circles.size(), tb.circles.size());
  for (int q = 0; q < static_cast<int>(ta.circles.size()); ++q)
    expect_same_circle(ta.circles[q], tb.circles[q]);
  expect_same_matrix(ta.beta, tb.beta);
  expect_same_matrix(ta.integral, tb.integral);
  ASSERT_EQ(ta.arcs.size(), tb.arcs.size());
  for (int k = 0; k < static_cast<int>(ta.arcs.size()); ++k)
    expect_same_arc(ta.arcs[k], tb.arcs[k]);
  expect_same_array(ta.area, tb.area);

  const ConcavePatches &ca = a.concave(), &cb = b.concave();
  expect_same_matrix(ca.pos, cb.pos);
  expect_same_array(ca.atoms.off(), cb.atoms.off());
  expect_same_array(ca.atoms.adj(), cb.atoms.adj());
  expect_same_array(ca.area, cb.area);
  expect_same_caps(ca.caps, cb.caps);
}

/**
 * `build_both` gives the geometries of `build_sas` and `build_ses`, table
 * for table, from the same triangulation seed.
 */
TEST(BuildSasTest, BuildBothMatchesSingle) {
  const std::array<Masked, 2> fixtures { random_masked(80, 11, 3),
                                         random_masked(60, 3, 1) };
  for (const Masked &m: fixtures) {
    const ArrayXd r = m.sar - kRp;
    AnalyticSurface surf(m.pts, r);
    ASSERT_TRUE(surf.initialize(kRp, m.active));

    seed_thread(0);
    const SurfaceGeometry sas = surf.build_sas();
    seed_thread(0);
    const SurfaceGeometry ses = surf.build_ses();
    seed_thread(0);
    const auto [both_sas, both_ses] = surf.build_both();

    EXPECT_GT(ses.concave().pos.cols(), 0);
    expect_same_geometry(sas, both_sas);
    expect_same_geometry(ses, both_ses);
  }
}

/**
 * The public geometries are the active prefixes of the stage tables.
 */
TEST(BuildSasTest, PublicGeometryMatchesStages) {
  const std::array<Masked, 2> fixtures { random_masked(80, 11, 3),
                                         random_masked(60, 3, 1) };
  for (const Masked &m: fixtures) {
    const ArrayXd r = m.sar - kRp;

    seed_thread(0);
    const std::optional<SurfaceSpheres> ss = prepare(m.pts, r, m.active, kRp);
    ASSERT_TRUE(ss);
    const SasDelaunay del = triangulate(*ss);
    const SasGeometry sas = build_sas(*ss, del);
    const SesGeometry ses = build_ses(*ss, del, sas);

    AnalyticSurface surf(m.pts, r);
    ASSERT_TRUE(surf.initialize(kRp, m.active));
    seed_thread(0);
    const auto [gs, ge] = surf.build_both();

    const int na = ss->n_active, nq = ss->g.offset(na),
              np = sas.probes.n_active, nc = sas.caps.h.offset(na);
    if (!m.active.all()) {
      EXPECT_LT(nq, static_cast<int>(sas.circles.size()));
      EXPECT_LT(sas.n_active_arcs, static_cast<int>(sas.arcs.size()));
      EXPECT_LT(np, sas.probes.pos.cols());
    }

    for (const SurfaceGeometry *g: { &gs, &ge }) {
      EXPECT_EQ(g->rp(), kRp);
      expect_same_array(g->order(), ss->order);
      expect_same_matrix(g->pts(), ss->pts);
      expect_same_array(g->sar(), ss->sar);
      expect_same_array(g->r(), ss->r);
      expect_same_array(g->convex().caps.off.off(),
                        sas.caps.h.off().head(na + 1));
      expect_same_matrix(g->convex().caps.axis, sas.caps.axis.leftCols(nc));
      expect_same_array(g->convex().caps.cosa, sas.caps.cosa.head(nc));
    }

    EXPECT_EQ(gs.kind(), SurfaceKind::kSas);
    expect_same_array(gs.convex().area, sas.area.head(na));
    EXPECT_TRUE(gs.toroidal().circles.empty());
    EXPECT_EQ(gs.toroidal().beta.cols(), 0);
    EXPECT_EQ(gs.toroidal().integral.cols(), 0);
    EXPECT_TRUE(gs.toroidal().arcs.empty());
    EXPECT_EQ(gs.toroidal().area.size(), 0);
    EXPECT_EQ(gs.concave().pos.cols(), 0);
    EXPECT_EQ(gs.concave().atoms.n(), 0);
    EXPECT_EQ(gs.concave().area.size(), 0);
    EXPECT_EQ(gs.concave().caps.off.size(), 0);

    EXPECT_EQ(ge.kind(), SurfaceKind::kSes);
    expect_same_array(ge.convex().area, ses.convex_area);

    const ToroidalPatches &tor = ge.toroidal();
    ASSERT_EQ(static_cast<int>(tor.circles.size()), nq);
    for (int q = 0; q < nq; ++q)
      expect_same_circle(tor.circles[q], sas.circles[q]);
    expect_same_matrix(tor.beta, ses.saddle_beta);
    expect_same_matrix(tor.integral, ses.saddle_integral);
    ASSERT_EQ(static_cast<int>(tor.arcs.size()), sas.n_active_arcs);
    for (int k = 0; k < sas.n_active_arcs; ++k)
      expect_same_arc(tor.arcs[k], sas.arcs[k]);
    expect_same_array(tor.area, ses.saddle_area);

    const ConcavePatches &con = ge.concave();
    expect_same_matrix(con.pos, sas.probes.pos.leftCols(np));
    expect_same_array(con.atoms.off(), sas.probes.atoms.off().head(np + 1));
    expect_same_array(con.atoms.adj(),
                      sas.probes.atoms.adj().head(sas.probes.atoms.offset(np)));
    expect_same_array(con.area, ses.face_area);
    expect_same_array(con.caps.off.off(), ses.face_off.off());
    expect_same_matrix(con.caps.axis, ses.face_axis);
    expect_same_array(con.caps.cosa, ses.face_cosa);
  }
}
}  // namespace
}  // namespace internal
}  // namespace nuri

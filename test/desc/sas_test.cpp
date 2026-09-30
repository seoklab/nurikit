//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"

namespace nuri {
namespace internal {
namespace {
using constants::kPi;
using constants::kTwoPi;

constexpr double kRp = 1.0;

struct Sas {
  SaPrep sa;
  SasGeometry geo;
};

Sas solve(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXb &active) {
  std::optional<SaPrep> sa = prepare(pts, sar, active, kRp);
  EXPECT_TRUE(sa);
  SasDelaunay del = triangulate(*sa);
  SasGeometry geo = build_sas(*sa, del);
  return { std::move(*sa), std::move(geo) };
}

Sas solve(const Matrix3Xd &pts, const ArrayXd &sar) {
  return solve(pts, sar, ArrayXb::Constant(sar.size(), true));
}

double sr_total(const Matrix3Xd &pts, const ArrayXd &sar) {
  return sr_sasa_impl(pts, sar, 5000, SrSasaMethod::kDirect).sum();
}

/**
 * Probes within `radius` of `x`; every probe has exactly three atoms, so a
 * k-fold point shows up as several coincident probes.
 */
int probes_at(const SasGeometry &geo, const Vector3d &x,
              const double radius = 1e-9) {
  int n = 0;
  for (int p = 0; p < geo.probes.pos.cols(); ++p) {
    EXPECT_EQ(geo.probes.atoms.degree(p), 3);
    n += static_cast<int>((geo.probes.pos.col(p) - x).norm() <= radius);
  }
  return n;
}

TEST(BuildSasTest, SingleSphere) {
  Matrix3Xd pts = Matrix3Xd::Zero(3, 1);
  ArrayXd sar = ArrayXd::Constant(1, 2.5);

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 1);
  EXPECT_DOUBLE_EQ(geo.area[0], 4 * kPi * 2.5 * 2.5);
  EXPECT_EQ(geo.probes.pos.cols(), 0);
  EXPECT_TRUE(geo.arcs.empty());
}

TEST(BuildSasTest, TwoSpheresFullCircle) {
  Matrix3Xd pts(3, 2);
  pts << 0, 2.0, 0, 0, 0, 0;
  ArrayXd sar(2);
  sar << 1.8, 1.5;

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 2);

  const double d = 2.0,
               a0 = (d * d + sar[0] * sar[0] - sar[1] * sar[1]) / (2 * d);
  EXPECT_NEAR(geo.area[sa.order[0] == 0 ? 0 : 1],
              4 * kPi * sar[0] * sar[0] - kTwoPi * sar[0] * (sar[0] - a0),
              1e-12);
  EXPECT_NEAR(geo.area[sa.order[0] == 1 ? 0 : 1],
              4 * kPi * sar[1] * sar[1] - kTwoPi * sar[1] * (sar[1] - (d - a0)),
              1e-12);

  EXPECT_EQ(geo.probes.pos.cols(), 0);
  ASSERT_EQ(geo.arcs.size(), 1);
  EXPECT_EQ(geo.arcs[0].beg, -1);
  EXPECT_EQ(geo.arcs[0].end, -1);
  EXPECT_DOUBLE_EQ(geo.arcs[0].dphi, kTwoPi);
}

TEST(BuildSasTest, CoplanarTriangle) {
  Matrix3Xd pts(3, 3);
  pts << 0, 2.4, 1.2, 0, 0, 2.1, 0, 0, 0;
  ArrayXd sar = ArrayXd::Constant(3, 1.9);

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 3);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());

  ASSERT_EQ(geo.probes.pos.cols(), 2);
  EXPECT_NEAR(std::abs(geo.probes.pos(2, 0)), std::abs(geo.probes.pos(2, 1)),
              1e-12);
  EXPECT_NEAR(geo.probes.pos(2, 0) + geo.probes.pos(2, 1), 0, 1e-12);
  for (int p = 0; p < 2; ++p) {
    EXPECT_EQ(geo.probes.atoms.degree(p), 3);
    EXPECT_EQ(geo.probes.tan_off.degree(p), 3);
  }

  EXPECT_EQ(geo.arcs.size(), 3);
  for (const SasArc &arc: geo.arcs) {
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

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.area.size(), 7);
  const int center = static_cast<int>(
      std::find(sa.order.begin(), sa.order.end(), 0) - sa.order.begin());
  ASSERT_LT(center, sa.n_solve);
  EXPECT_NEAR(geo.area[center], 0, 1e-12);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

TEST(BuildSasTest, FourSphereVertex) {
  const double reach = 2.0;
  Matrix3Xd pts(3, 4);
  pts << 1, 1, -1, -1,  //
      1, -1, 1, -1,     //
      1, -1, -1, 1;
  pts *= reach / std::sqrt(3.0);
  ArrayXd sar = ArrayXd::Constant(4, reach);

  auto [sa, geo] = solve(pts, sar);
  // trapped point: the perturbation decides how many coincident probes are
  // accessible there; the area does not depend on it
  probes_at(geo, Vector3d::Zero());
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

Matrix3Xd star(const int n_ring, const double polar_deg, const double reach) {
  const double polar = polar_deg * kPi / 180;
  Matrix3Xd pts(3, n_ring + 1);
  pts.col(0) << 0, 0, reach;
  for (int k = 0; k < n_ring; ++k) {
    const double az = kTwoPi * k / n_ring;
    pts.col(k + 1) << reach * std::sin(polar) * std::cos(az),
        reach * std::sin(polar) * std::sin(az), reach * std::cos(polar);
  }
  return pts;
}

TEST(BuildSasTest, FiveSphereVertex) {
  const Matrix3Xd pts = star(4, 110, 2.0);
  const ArrayXd sar = ArrayXd::Constant(5, 2.0);

  auto [sa, geo] = solve(pts, sar);
  probes_at(geo, Vector3d::Zero());
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
}

TEST(BuildSasTest, CoplanarSquareVertex) {
  Matrix3Xd pts(3, 4);
  pts << 1, 1, -1, -1,  //
      1, -1, 1, -1,     //
      0, 0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(4, 1.5);

  auto [sa, geo] = solve(pts, sar);
  EXPECT_EQ(geo.probes.pos.cols(), 4);
  EXPECT_EQ(probes_at(geo, Vector3d(0, 0, 0.5)), 2);
  EXPECT_EQ(probes_at(geo, Vector3d(0, 0, -0.5)), 2);
  EXPECT_NEAR(geo.area.sum(), sr_total(pts, sar), 1e-2 * geo.area.sum());
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

TEST(BuildSasTest, ApexTangentAtVertex) {
  const TangentApex t = tangent_apex(0);

  auto [sa, geo] = solve(t.pts, t.sar);
  EXPECT_EQ(geo.probes.pos.cols(), 4);
  EXPECT_EQ(probes_at(geo, t.x), 2);
  EXPECT_NEAR(geo.area.sum(), sr_total(t.pts, t.sar), 1e-2 * geo.area.sum());
}

TEST(BuildSasTest, NearBandPinch) {
  for (const double gap: { 1e-7, 5e-7, 2e-6 }) {
    const TangentApex t = tangent_apex(gap);
    auto [sa, geo] = solve(t.pts, t.sar);
    const ArrayXd sr = sr_sasa_impl(t.pts, t.sar, 20000, SrSasaMethod::kDirect);
    for (int p = 0; p < 4; ++p)
      EXPECT_NEAR(geo.area[p], sr[sa.order[p]], 0.05)
          << "gap " << gap << " atom " << p;
  }
}

TEST(BuildSasTest, SliverBetweenNearCrossings) {
  const Matrix3Xd pts = star(4, 110, 2.0);
  ArrayXd sar = ArrayXd::Constant(5, 2.0);
  sar[2] -= 1e-5;

  auto [sa, geo] = solve(pts, sar);
  const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
  for (int p = 0; p < 5; ++p)
    EXPECT_NEAR(geo.area[p], sr[sa.order[p]], 0.05) << "atom " << p;
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

    auto [sa, geo] = solve(pts, t.sar);
    EXPECT_GE(probes_at(geo, t.x, kSurfaceLengthEps), 1) << "depth " << depth;
    const ArrayXd sr = sr_sasa_impl(pts, t.sar, 20000, SrSasaMethod::kDirect);
    for (int p = 0; p < 4; ++p)
      EXPECT_NEAR(geo.area[p], sr[sa.order[p]], 0.05)
          << "depth " << depth << " atom " << p;
  }
}

/**
 * Flat cell: the apex sphere's centre lies 1e-3 above the plane of the face
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

  auto [sa, geo] = solve(pts, sar);
  for (int p = 0; p < geo.probes.pos.cols(); ++p)
    EXPECT_GT((geo.probes.pos.col(p) - t.x).norm(), 1e-5) << p;
  const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
  for (int p = 0; p < 5; ++p)
    EXPECT_NEAR(geo.area[p], sr[sa.order[p]], 0.05) << "atom " << p;
}

/**
 * Ideal benzene has 6-fold axis vertices and 4-fold C-C-H-H vertices; a
 * 1e-7 jitter scatters the raw points of one vertex beyond
 * kSurfaceLengthEps along their circles while every sphere still passes
 * within the tolerance. Deterministic jitter, rp 1.4.
 */
TEST(BuildSasTest, JitteredBenzeneVertices) {
  Matrix3Xd pts(3, 12);
  ArrayXd sar(12);
  for (int k = 0; k < 6; ++k) {
    const double az = kPi / 3 * k;
    pts.col(k) << 1.39 * std::cos(az), 1.39 * std::sin(az), 0;
    pts.col(6 + k) << 2.47 * std::cos(az), 2.47 * std::sin(az), 0;
    sar[k] = 1.7 + 1.4;
    sar[6 + k] = 1.2 + 1.4;
  }
  for (int i = 0; i < pts.size(); ++i)
    pts.data()[i] += 1e-7 * std::sin(1000.0 * (i + 1));

  const double rp = 1.4;
  std::optional<SaPrep> sa = prepare(pts, sar, ArrayXb::Constant(12, true), rp);
  ASSERT_TRUE(sa);
  const SasDelaunay del = triangulate(*sa);
  const SasGeometry geo = build_sas(*sa, del);

  const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
  for (int p = 0; p < 12; ++p)
    EXPECT_NEAR(geo.area[p], sr[sa->order[p]], 0.1) << "atom " << p;
  for (int p = 0; p < geo.probes.pos.cols(); ++p)
    EXPECT_EQ(geo.probes.atoms.degree(p), 3);
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

      auto [sa, geo] = solve(pts, sar);
      const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
      for (int a = 0; a < 4; ++a) {
        EXPECT_NEAR(geo.area[a], sr[sa.order[a]], 0.05)
            << "gap " << gap << " delta " << delta << " atom " << a;
      }
    }
  }
}

/**
 * Spheres j, k overlap by 1e-4 in a circle of radius 0.014; the host cuts
 * that circle nearly tangentially so the face's two cut points are 1e-5
 * apart. They are distinct vertices: only a tangency within
 * kSurfaceLengthEps merges the two cut points of one face.
 */
TEST(BuildSasTest, TwoCutPointsOfOneFaceStayDistinct) {
  const double r = 2.0, d = 4.0 - 1e-4, h = 5e-6, zs = 3.0;
  const double rc = std::sqrt(r * r - d * d / 4),
               zc = std::sqrt(rc * rc - h * h);
  Matrix3Xd pts(3, 3);
  pts.col(0) << 0, 0, zs;
  pts.col(1) << -d / 2, 0, 0;
  pts.col(2) << d / 2, 0, 0;
  ArrayXd sar(3);
  sar << std::sqrt(h * h + (zs - zc) * (zs - zc)), r, r;

  auto [sa, geo] = solve(pts, sar);
  ASSERT_EQ(geo.probes.pos.cols(), 2);
  EXPECT_NEAR((geo.probes.pos.col(0) - geo.probes.pos.col(1)).norm(), 2 * h,
              1e-9);
  for (int p = 0; p < 2; ++p)
    EXPECT_EQ(geo.probes.atoms.degree(p), 3);
}

TEST(BuildSasTest, TriangulationSharedAcrossMasks) {
  const int n = 24;
  Matrix3Xd pts(3, n);
  for (int i = 0; i < n; ++i) {
    const int ix = i % 4, iy = (i / 4) % 3, iz = i / 12;
    pts.col(i) << 1.7 * ix, 1.6 * iy + 0.3 * (i % 2), 1.5 * iz + 0.2 * (i % 3);
  }
  ArrayXd sar = ArrayXd::Constant(n, 1.5) + 0.05 * ArrayXd::LinSpaced(n, 0, 1);

  ArrayXb all = ArrayXb::Constant(n, true), half = all;
  half.tail(n / 2).setConstant(false);

  std::optional<SaPrep> sa_all = prepare(pts, sar, all, kRp),
                        sa_half = prepare(pts, sar, half, kRp);
  ASSERT_TRUE(sa_all && sa_half);

  const SasDelaunay del = triangulate(*sa_all);
  const SasGeometry geo_all = build_sas(*sa_all, del),
                    geo_half = build_sas(*sa_half, del);

  ArrayXd area_all(n);
  for (int p = 0; p < sa_all->n_solve; ++p)
    area_all[sa_all->order[p]] = geo_all.area[p];

  EXPECT_EQ(sa_half->n_active, n / 2);
  for (int p = 0; p < sa_half->n_active; ++p)
    EXPECT_DOUBLE_EQ(geo_half.area[p], area_all[sa_half->order[p]]) << p;
}

TEST(BuildSasTest, TriangulationTables) {
  const int n = 24;
  Matrix3Xd pts(3, n);
  for (int i = 0; i < n; ++i) {
    const int ix = i % 4, iy = (i / 4) % 3, iz = i / 12;
    pts.col(i) << 1.7 * ix, 1.6 * iy + 0.3 * (i % 2), 1.5 * iz + 0.2 * (i % 3);
  }
  const ArrayXd sar = ArrayXd::Constant(n, 1.5);
  std::optional<SaPrep> sa = prepare(pts, sar, ArrayXb::Constant(n, true), kRp);
  ASSERT_TRUE(sa);
  const SasDelaunay del = triangulate(*sa);

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
      const int c = del.edge_cell[del.nbrs.eid(it)];
      const Array4i tv = del.tets.col(c);
      EXPECT_TRUE((tv == v).any() && (tv == *it).any()) << v << " " << *it;
    }
  }

  EXPECT_EQ(del.ex.n(), n + 4);
  for (int v = 0; v < n; ++v)
    EXPECT_NEAR(del.ex.rho2(v), 2.25, 1e-12);
}

/**
 * Three standard deviations of a Shrake–Rupley estimate with `n` points at
 * the worst-case accessible fraction 1/2.
 */
double sr_tol(const double radius, const int n) {
  return 3 * 4 * kPi * radius * radius * 0.5 / std::sqrt(n);
}

void expect_sr(const Sas &sas, const Matrix3Xd &pts, const ArrayXd &sar,
               const int n = 20000) {
  const ArrayXd sr = sr_sasa_impl(pts, sar, n, SrSasaMethod::kDirect);
  for (int p = 0; p < sas.sa.n_solve; ++p) {
    const int o = sas.sa.order[p];
    EXPECT_NEAR(sas.geo.area[p], sr[o], sr_tol(sar[o], n)) << "atom " << o;
  }
  for (int p = 0; p < sas.geo.probes.pos.cols(); ++p)
    EXPECT_EQ(sas.geo.probes.atoms.degree(p), 3);
}

Sas solve_rp(const Matrix3Xd &pts, const ArrayXd &sar, const double rp) {
  std::optional<SaPrep> sa =
      prepare(pts, sar, ArrayXb::Constant(sar.size(), true), rp);
  EXPECT_TRUE(sa);
  SasDelaunay del = triangulate(*sa);
  SasGeometry geo = build_sas(*sa, del);
  return { std::move(*sa), std::move(geo) };
}

Matrix3Xd rigid(const Matrix3Xd &pts, const int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  const Vector3d ax(nd(rng), nd(rng), nd(rng));
  const Matrix3d rot = AngleAxisd(nd(rng), ax.normalized()).toRotationMatrix();
  const Vector3d t(nd(rng), nd(rng), nd(rng));
  return (rot * pts).colwise() + 3.0 * t;
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
        atom |= sas.sa.order[a] == m;
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
 * height whose apex the old slack excused, leaving a probe 0.4 Å inside the
 * cover.
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
    std::optional<SaPrep> sa =
        prepare(pts, sar, ArrayXb::Constant(5, true), kRp);
    ASSERT_TRUE(sa);
    const SasDelaunay del = triangulate(*sa);
    with_flat += flat_cells(del, 4) > 0;
    const Sas sas { *sa, build_sas(*sa, del) };
    EXPECT_EQ(spurious_probes(sas, pts, sar), 0) << "seed " << seed;
    expect_sr(sas, pts, sar, 5000);
  }
  EXPECT_GT(with_flat, 0);
}

TEST(BuildSasTest, BenzeneCoverNoSpuriousProbe) {
  const double rp = 1.4;
  Matrix3Xd base(3, 13);
  ArrayXd sar(13);
  for (int k = 0; k < 6; ++k) {
    const double az = kPi / 3 * k;
    base.col(k) << 1.39 * std::cos(az), 1.39 * std::sin(az), 0;
    base.col(6 + k) << 2.47 * std::cos(az), 2.47 * std::sin(az), 0;
    sar[k] = 1.7 + rp;
    sar[6 + k] = 1.2 + rp;
  }
  Vector3d v(1.93, 1.1147, 0);
  v[2] = std::sqrt(sar[0] * sar[0]
                   - (v.head(2) - base.col(0).head(2)).squaredNorm());
  base.col(12) = v + Vector3d(0, 0, 1.0);
  sar[12] = 1.4 + rp;

  for (int seed = 0; seed < 10; ++seed) {
    const Matrix3Xd pts = rigid(base, seed);
    const Sas sas = solve_rp(pts, sar, rp);
    EXPECT_EQ(spurious_probes(sas, pts, sar), 0) << "seed " << seed;
    expect_sr(sas, pts, sar, 5000);
  }
}

/**
 * The third centre lies `amp` from the axis of circle `(a, b)`: the radical
 * line form of the cut points loses `amp²` of precision, the circle form is
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
        const int o = sas.sa.order[a];
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
      const Sas sas = solve(pts, sar);
      expect_sr(sas, pts, sar);
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

  const Sas sas = solve_rp(pts, sar, 0.5);
  expect_sr(sas, pts, sar, 50000);
  for (const SasArc &arc: sas.geo.arcs)
    EXPECT_LE(arc.dphi, kTwoPi);
}

Vector3d cap_centre(const Vector3d &n, const double a, const double rc,
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
  pts.col(1) = cap_centre(nu, rs * std::cos(au), 2.0, rs);
  sar[1] = 2.0;
  pts.col(2) = cap_centre(nt, rs * std::cos(at), 1.6, rs);
  sar[2] = 1.6;
  pts.col(3) = cap_centre(nc, 0.0, 2.2, rs);
  sar[3] = 2.2;

  const Sas sas = solve_rp(pts, sar, 0.5);
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
  pts.col(1) = cap_centre(nj, atiny, rt, rs);
  sar[1] = rt;
  pts.col(2) = cap_centre(nc, rs * std::cos(alc), rbig, rs);
  sar[2] = rbig;
  pts.col(3) = cap_centre(nk, atiny, rt, rs);
  sar[3] = rt;
  pts.col(4) = cap_centre(ne, rs * std::cos(ale), rbig, rs);
  sar[4] = rbig;

  const Sas sas = solve_rp(pts, sar, 0.5);
  expect_sr(sas, pts, sar, 50000);
}

TEST(BuildSasTest, ExactlyTangentPair) {
  Matrix3Xd pts(3, 2);
  pts.col(0) << 0, 0, 0;
  pts.col(1) << 3, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(2, 1.5);

  auto [sa, geo] = solve(pts, sar);
  EXPECT_EQ(geo.probes.pos.cols(), 0);
  for (int p = 0; p < 2; ++p)
    EXPECT_NEAR(geo.area[p], 4 * kPi * 1.5 * 1.5, 1e-9);
}

TEST(BuildSasTest, TangentApexUnderRigidMotion) {
  for (const double d: { -5e-7, 0.0, 1e-7, 5e-7, 2e-6 }) {
    for (int seed = -1; seed < 3; ++seed) {
      TangentApex t = tangent_apex(d);
      const Matrix3Xd pts = seed < 0 ? t.pts : rigid(t.pts, seed);
      const Sas sas = solve(pts, t.sar);
      expect_sr(sas, pts, t.sar);
    }
  }
}

/**
 * Random dense sets: every probe has three tangents, a probe ends at most
 * one arc per circle, and the areas match Shrake–Rupley.
 */
TEST(BuildSasTest, RandomDenseSets) {
  for (const int n: { 60, 200 }) {
    for (int seed = 0; seed < 3; ++seed) {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> u(0, std::cbrt(n) * 1.9),
          ur(1.5, 2.5);
      Matrix3Xd pts(3, n);
      ArrayXd sar(n);
      for (int i = 0; i < n; ++i) {
        pts.col(i) << u(rng), u(rng), u(rng);
        sar[i] = ur(rng);
      }
      const Sas sas = solve(pts, sar);

      for (int p = 0; p < sas.geo.probes.pos.cols(); ++p) {
        EXPECT_EQ(sas.geo.probes.atoms.degree(p), 3);
        EXPECT_EQ(sas.geo.probes.tan_off.degree(p), 3) << "probe " << p;
      }
      std::vector<std::pair<int, int>> ends;
      for (const SasArc &arc: sas.geo.arcs) {
        if (arc.beg < 0)
          continue;
        ends.emplace_back(arc.circ, arc.beg);
        ends.emplace_back(arc.circ, arc.end);
      }
      std::sort(ends.begin(), ends.end());
      EXPECT_EQ(std::adjacent_find(ends.begin(), ends.end()), ends.end());

      EXPECT_NEAR(sas.geo.area.sum(), sr_total(pts, sar),
                  1e-2 * sas.geo.area.sum());
    }
  }
}
}  // namespace
}  // namespace internal
}  // namespace nuri

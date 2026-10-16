//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cmath>
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

namespace nuri {
namespace internal {
namespace {
using constants::kPi;
using constants::kTwoPi;

constexpr double kRp = 1.4;

SurfaceGeometry solve(const Matrix3Xd &pts, const ArrayXd &sar,
                      const ArrayXb &active, const double rp = kRp) {
  const ArrayXd r = sar - rp;
  AnalyticSurface surf(pts, r);
  EXPECT_TRUE(surf.initialize(rp, active));
  return surf.build_ses();
}

SurfaceGeometry solve(const Matrix3Xd &pts, const ArrayXd &sar,
                      const double rp = kRp) {
  return solve(pts, sar, ArrayXb::Constant(sar.size(), true), rp);
}

std::pair<SurfaceGeometry, SurfaceGeometry>
solve_both(const Matrix3Xd &pts, const ArrayXd &sar, const double rp) {
  const ArrayXd r = sar - rp;
  AnalyticSurface surf(pts, r);
  EXPECT_TRUE(surf.initialize(rp));
  return surf.build_both();
}

struct Ses {
  SurfaceSpheres ss;
  SasDelaunay del;
  SasGeometry geo;
  SesGeometry ses;
};

Ses solve_stages(const Matrix3Xd &pts, const ArrayXd &sar,
                 const double rp = kRp) {
  std::optional<SurfaceSpheres> ss =
      prepare(pts, sar - rp, ArrayXb::Constant(sar.size(), true), rp);
  EXPECT_TRUE(ss);
  SasDelaunay del = triangulate(*ss);
  SasGeometry geo = build_sas(*ss, del);
  SesGeometry ses = build_ses(*ss, del, geo);
  return { std::move(*ss), std::move(del), std::move(geo), std::move(ses) };
}

struct TwoSpheres {
  double a1, a2, rl;
};

TwoSpheres two_spheres(const double sar1, const double sar2, const double d) {
  const double a = (d * d + sar1 * sar1 - sar2 * sar2) / (2 * d);
  return { a, d - a, std::sqrt(sar1 * sar1 - a * a) };
}

template <class F>
double simpson(const F &f, const double lo, const double hi, const int n) {
  const double h = (hi - lo) / (2 * n);
  double odd = 0, even = 0;
  for (int k = 1; k < 2 * n; k += 2)
    odd += f(lo + k * h);
  for (int k = 2; k < 2 * n; k += 2)
    even += f(lo + k * h);
  return h / 3 * (f(lo) + 4 * odd + 2 * even + f(hi));
}


struct RandomCluster {
  Matrix3Xd pts;
  ArrayXd sar;
};

RandomCluster random_cluster(const int seed, const int n,
                             const double box = 6.0) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0, box), ur(1.0, 1.4);
  RandomCluster c { Matrix3Xd(3, n), ArrayXd(n) };
  for (int i = 0; i < n; ++i) {
    c.pts.col(i) << u(rng), u(rng), u(rng);
    c.sar[i] = ur(rng) + kRp;
  }
  return c;
}

constexpr int kLattice = 30000;

const Matrix3Xd &lattice() {
  static const Matrix3Xd dirs = canonical_fibonacci_lattice(kLattice);
  return dirs;
}

/**
 * Lattice estimate of face `p`: directions outside every departure
 * hemisphere whose point on the probe sphere is outside every other probe
 * ball (`|rp d - w|^2 >= rp^2` as `2 rp dot(d, w) <= |w|^2`, exact enough
 * for neighbors at rounding distance), brute force over the active probes;
 * complete for all-active runs.
 */
double area_within(const SurfaceGeometry &s, const Vector3d &x,
                   const double radius) {
  const Matrix3Xd &pos = s.concave().pos;
  double total = 0;
  for (int p = 0; p < pos.cols(); ++p) {
    if ((pos.col(p) - x).norm() < radius)
      total += s.concave().area[p];
  }
  return total;
}

double face_from_dots(const SurfaceGeometry &s, const int p, const double rp) {
  const Matrix3Xd &pos = s.concave().pos;
  const Matrix3Xd &dirs = lattice();
  const Vector3d x = pos.col(p);
  const auto t = hemispheres(s, p);
  const int nt = static_cast<int>(t.cols());

  std::vector<Vector3d> near;
  for (int y = 0; y < pos.cols(); ++y) {
    const Vector3d w = pos.col(y) - x;
    if (y != p && w.norm() < 2 * rp)
      near.push_back(w);
  }

  int count = 0;
  for (int k = 0; k < dirs.cols(); ++k) {
    const Vector3d d = dirs.col(k);
    if (nt > 0 && (t.transpose() * d).maxCoeff() > 0)
      continue;
    bool outside = true;
    for (const Vector3d &w: near)
      outside &= 2 * rp * d.dot(w) <= w.squaredNorm();
    count += outside;
  }
  return 2 * kTwoPi * rp * rp * count / dirs.cols();
}

double area_within(const Ses &s, const Vector3d &x, const double radius) {
  const Matrix3Xd &pos = s.geo.probes.pos;
  double total = 0;
  for (int p = 0; p < s.geo.probes.n_active; ++p) {
    if ((pos.col(p) - x).norm() < radius)
      total += s.ses.face_area[p];
  }
  return total;
}

/**
 * `face_from_dots` on the stages: a neighbor closer than the rounded
 * positions resolve (the roots of one rounded point) is placed by the exact
 * kernel's difference, as the pair stage places it.
 */
double face_from_dots(const Ses &s, const int p, const double rp) {
  const SasProbes &pr = s.geo.probes;
  const Matrix3Xd &dirs = lattice();
  const Vector3d x = pr.pos.col(p);
  const auto t = pr.tan.middleCols(pr.tan_off[p], pr.tan_off.degree(p));
  const int nt = static_cast<int>(t.cols());

  std::vector<Vector3d> near;
  for (int y = 0; y < pr.pos.cols(); ++y) {
    Vector3d w = pr.pos.col(y) - x;
    if (y == p || w.norm() >= 2 * rp)
      continue;
    if (w.norm() < 1e-6) {
      const auto [dir, len] =
          s.del.ex.difference(ball_triple(pr.face.col(y)), pr.plus[y],
                              ball_triple(pr.face.col(p)), pr.plus[p]);
      w = len * dir;
    }
    near.push_back(w);
  }

  int count = 0;
  for (int k = 0; k < dirs.cols(); ++k) {
    const Vector3d d = dirs.col(k);
    if (nt > 0 && (t.transpose() * d).maxCoeff() > 0)
      continue;
    bool outside = true;
    for (const Vector3d &w: near)
      outside &= 2 * rp * d.dot(w) <= w.squaredNorm();
    count += outside;
  }
  return 2 * kTwoPi * rp * rp * count / dirs.cols();
}

double convex_from_dots(const SurfaceGeometry &s, const int i,
                        const double rp) {
  const Matrix3Xd &dirs = lattice();
  const Matrix3Xd &pts = s.pts();
  const ArrayXd &sar = s.sar();
  const Vector3d c = pts.col(i);
  const double sas = sar[i], r = sas - rp;

  int count = 0;
  for (int k = 0; k < dirs.cols(); ++k) {
    const Vector3d y = c + sas * dirs.col(k);
    bool outside = true;
    for (int j = 0; j < pts.cols(); ++j) {
      outside &= j == i || (y - pts.col(j)).squaredNorm() >= sar[j] * sar[j];
    }
    count += outside;
  }
  return 2 * kTwoPi * r * r * count / dirs.cols();
}

/**
 * Torus patches by quadrature of `rp (rl - rp cos beta)` over each arc's full
 * generating range; spindles excluded (`rl >= rp` asserted).
 */
double saddle_from_quadrature(const SurfaceGeometry &s, const double rp) {
  const ToroidalPatches &tor = s.toroidal();
  double total = 0;
  for (const RingArc &arc: tor.arcs) {
    const SasCircle &c = tor.circles[arc.circ];
    EXPECT_GE(c.rl, rp);
    const double lo = -std::atan2(c.aij[0], c.rl),
                 hi = std::atan2(c.aij[1], c.rl);
    total += arc.dphi * rp
             * simpson([&](double b) { return c.rl - rp * std::cos(b); }, lo,
                       hi, 2000);
  }
  return total;
}

/**
 * Every active face against the lattice to `1e-3` relative, floored at
 * `1.5e-3` of the probe sphere: four times the worst lattice error over
 * these fixtures, below their smallest face (`2.2e-3` of its sphere).
 */
void expect_faces_match_dots(const SurfaceGeometry &s, const double rp,
                             const char *tag = "") {
  const double sphere = 2 * kTwoPi * rp * rp;
  const ArrayXd &face_area = s.concave().area;
  for (int p = 0; p < face_area.size(); ++p) {
    const double area = face_area[p];
    ASSERT_TRUE(std::isfinite(area)) << tag << " probe " << p;
    EXPECT_GE(area, -1e-9) << tag << " probe " << p;
    const double dots = face_from_dots(s, p, rp);
    EXPECT_NEAR(area, dots, nuri::max(1e-3 * dots, 1.5e-3 * sphere))
        << tag << " probe " << p;
  }
}

void expect_faces_match_dots(const Ses &s, const double rp,
                             const char *tag = "") {
  const double sphere = 2 * kTwoPi * rp * rp;
  const ArrayXd &face_area = s.ses.face_area;
  for (int p = 0; p < face_area.size(); ++p) {
    const double area = face_area[p];
    ASSERT_TRUE(std::isfinite(area)) << tag << " probe " << p;
    EXPECT_GE(area, -1e-9) << tag << " probe " << p;
    const double dots = face_from_dots(s, p, rp);
    EXPECT_NEAR(area, dots, nuri::max(1e-3 * dots, 1.5e-3 * sphere))
        << tag << " probe " << p;
  }
}

/**
 * Quan & Stamm (2016) eqs. 5.26-5.27 for two spheres on an ordinary torus.
 */
void expect_two_sphere_closed_form(const double r1, const double r2,
                                   const double d) {
  Matrix3Xd pts(3, 2);
  pts << 0, d, 0, 0, 0, 0;
  ArrayXd sar(2);
  sar << r1 + kRp, r2 + kRp;

  const auto [a1, a2, rl] = two_spheres(sar[0], sar[1], d);
  ASSERT_GE(rl, kRp);

  const SurfaceGeometry s = solve(pts, sar);
  ASSERT_EQ(s.convex().area.size(), 2);
  for (int k = 0; k < 2; ++k) {
    const bool first = s.order()[k] == 0;
    const double r = first ? r1 : r2, a = first ? a1 : a2;
    EXPECT_NEAR(s.convex().area[k], kTwoPi * r * r * (1 + a / (r + kRp)),
                1e-10);
  }

  const ToroidalPatches &tor = s.toroidal();
  ASSERT_EQ(tor.arcs.size(), 1);
  ASSERT_EQ(tor.beta.cols(), 1);
  ASSERT_EQ(tor.area.size(), 1);
  EXPECT_DOUBLE_EQ(tor.beta(1, 0), 0);
  EXPECT_DOUBLE_EQ(tor.beta(2, 0), 0);

  const double torus = kTwoPi * kRp
                       * (rl * (std::atan2(a1, rl) + std::atan2(a2, rl))
                          - kRp * (a1 / sar[0] + a2 / sar[1]));
  EXPECT_NEAR(tor.area[0], torus, 1e-10);

  EXPECT_EQ(s.concave().pos.cols(), 0);
  EXPECT_EQ(s.concave().area.size(), 0);
  EXPECT_EQ(s.concave().caps.off.size(), 0);
}

TEST(BuildSesTest, TwoSpheresClosedForm) {
  expect_two_sphere_closed_form(1.5, 1.5, 2.5);
  expect_two_sphere_closed_form(1.2, 1.9, 2.5);
}

/**
 * Spindle torus: the arc `[-theta_1, theta_2]` loses
 * `|beta| < beta0 = acos(rl / rp)`.
 */
void expect_two_sphere_cusp(const double r1, const double r2, const double d,
                            const bool lower_part, const bool upper_part) {
  Matrix3Xd pts(3, 2);
  pts << 0, d, 0, 0, 0, 0;
  ArrayXd sar(2);
  sar << r1 + kRp, r2 + kRp;

  const SurfaceGeometry s = solve(pts, sar);
  const ToroidalPatches &tor = s.toroidal();
  ASSERT_EQ(tor.beta.cols(), 1);
  ASSERT_EQ(tor.area.size(), 1);
  const SasCircle &c = tor.circles[0];
  const double rl = c.rl, a_i = c.aij[0], a_j = c.aij[1];
  ASSERT_LT(rl, kRp);
  const double b0 = std::acos(rl / kRp), lo = -std::atan2(a_i, rl),
               hi = std::atan2(a_j, rl);

  const Array4d beta = tor.beta.col(0);
  EXPECT_NEAR(beta[0], lo, 1e-12);
  EXPECT_NEAR(beta[3], hi, 1e-12);
  EXPECT_EQ(beta[1] > beta[0], lower_part);
  EXPECT_EQ(beta[3] > beta[2], upper_part);
  if (lower_part) {
    EXPECT_NEAR(beta[1], nuri::min(hi, -b0), 1e-12);
  }
  if (upper_part) {
    EXPECT_NEAR(beta[2], nuri::max(lo, b0), 1e-12);
  }
  EXPECT_GE(tor.integral(0, 0), 0);
  EXPECT_GE(tor.integral(1, 0), 0);

  auto rho = [&](double b) { return kRp * (rl - kRp * std::cos(b)); };
  double integral = 0;
  if (lo < -b0)
    integral += simpson(rho, lo, nuri::min(hi, -b0), 2000);
  if (hi > b0)
    integral += simpson(rho, nuri::max(lo, b0), hi, 2000);
  EXPECT_NEAR(tor.area[0], kTwoPi * integral, 1e-8);
}

TEST(BuildSesTest, TwoSpheresCusp) {
  expect_two_sphere_cusp(1.0, 1.0, 4.2, true, true);
  expect_two_sphere_cusp(1.0, 2.0, 1.1, false, true);
}

TEST(BuildSesTest, TwoSpheresNoActiveNoSaddle) {
  Matrix3Xd pts(3, 2);
  pts << 0, 2.5, 0, 0, 0, 0;
  ArrayXd sar(2);
  sar << 1.2 + kRp, 1.9 + kRp;
  ArrayXb active(2);
  active << false, true;

  const SurfaceGeometry s = solve(pts, sar, active);
  ASSERT_EQ(s.convex().area.size(), 1);
  EXPECT_EQ(s.order()[0], 1);
  const auto [a1, a2, rl] = two_spheres(sar[0], sar[1], 2.5);
  EXPECT_NEAR(s.convex().area[0], kTwoPi * 1.9 * 1.9 * (1 + a2 / sar[1]),
              1e-10);
  EXPECT_EQ(s.toroidal().beta.cols(), 1);
  EXPECT_EQ(s.toroidal().integral.cols(), 1);
  EXPECT_EQ(s.toroidal().area.size(), 1);
  EXPECT_EQ(s.concave().area.size(), 0);

  Matrix3Xd tet(3, 4);
  tet << 0, 2.4, 1.2, 1.2,  //
      0, 0, 2.1, 0.7,       //
      0, 0, 0, 2.0;
  ArrayXd sar4 = ArrayXd::Constant(4, 1.9 + kRp);
  ArrayXb one = ArrayXb::Constant(4, false);
  one[0] = true;

  const SurfaceGeometry full = solve(tet, sar4);
  const SurfaceGeometry part = solve(tet, sar4, one);
  ASSERT_EQ(part.convex().area.size(), 1);
  const ToroidalPatches &tor = part.toroidal();
  const int n_circ = static_cast<int>(tor.circles.size());
  EXPECT_EQ(n_circ, 3);
  EXPECT_LT(n_circ, static_cast<int>(full.toroidal().circles.size()));
  EXPECT_EQ(tor.beta.cols(), n_circ);
  EXPECT_EQ(tor.integral.cols(), n_circ);
  EXPECT_EQ(tor.area.size(), tor.arcs.size());
  EXPECT_LT(tor.arcs.size(), full.toroidal().arcs.size());
  const int n_probes = static_cast<int>(part.concave().pos.cols());
  EXPECT_GT(n_probes, 0);
  EXPECT_LT(n_probes, full.concave().pos.cols());
  EXPECT_EQ(part.concave().area.size(), n_probes);
  EXPECT_EQ(part.concave().caps.off.size(), n_probes);
}

/**
 * `rp^2 (2pi - sum angles)` over the planes through the probe and each pair of
 * its atom centers, oriented away from the third atom.
 */
double excess_from_planes(const SurfaceGeometry &s, const int p) {
  const Vector3d x = s.concave().pos.col(p);
  const Matrix3d tri = s.pts()(E::all, s.concave().atoms.nbrs(p));
  Matrix3d normals;
  for (int m = 0; m < 3; ++m) {
    const Vector3d u = tri.col(m) - x, v = tri.col((m + 1) % 3) - x,
                   w = tri.col((m + 2) % 3) - x;
    Vector3d n = u.cross(v).normalized();
    if (n.dot(w) > 0)
      n = -n;
    normals.col(m) = n;
  }
  double total = 0;
  for (int m = 0; m < 3; ++m) {
    const Vector3d n1 = normals.col(m), n2 = normals.col((m + 1) % 3);
    total += std::atan2(n1.cross(n2).norm(), n1.dot(n2));
  }
  return kRp * kRp * (kTwoPi - total);
}

double excess_from_tangents(const SurfaceGeometry &s, const int p) {
  const auto t = hemispheres(s, p);
  double total = 0;
  for (int m = 0; m < 3; ++m) {
    const Vector3d t1 = t.col(m), t2 = t.col((m + 1) % 3);
    total += std::atan2(t1.cross(t2).norm(), t1.dot(t2));
  }
  return kRp * kRp * (kTwoPi - total);
}

TEST(BuildSesTest, HighFaceSphericalExcess) {
  const double side = 3.0;
  Matrix3Xd pts(3, 3);
  pts << 0, side, side / 2,             //
      0, 0, side * std::sqrt(3.0) / 2,  //
      0, 0, 0;
  ArrayXd sar = ArrayXd::Constant(3, 3.0);

  const SurfaceGeometry s = solve(pts, sar);
  const ConcavePatches &con = s.concave();
  ASSERT_EQ(con.pos.cols(), 2);
  ASSERT_EQ(con.area.size(), 2);
  ASSERT_EQ(con.caps.off.size(), 2);

  for (int p = 0; p < 2; ++p) {
    ASSERT_GE(std::abs(con.pos(2, p)), kRp);
    ASSERT_EQ(hemispheres(s, p).cols(), 3);

    const double area = con.area[p];
    EXPECT_GT(area, 0);
    EXPECT_LT(area, kTwoPi * kRp * kRp);
    EXPECT_NEAR(area, excess_from_tangents(s, p), 1e-12);
    EXPECT_NEAR(area, excess_from_planes(s, p), 1e-10);

    ASSERT_EQ(con.caps.off.degree(p), 3);
    for (int k = con.caps.off[p]; k < con.caps.off[p + 1]; ++k) {
      EXPECT_EQ(con.caps.cosa[k], 0);
      EXPECT_NEAR(con.caps.axis.col(k).norm(), 1, 1e-12);
    }
  }
}

struct Cutter {
  int probe;
  Vector3d axis;
  double cosa;
  bool kept;
};

/**
 * Every other probe strictly within `2 rp` of active probe `p`, flagged
 * whether its cap is in the stored cap list of `p`.
 */
std::vector<Cutter> cutters(const SurfaceGeometry &s, const int p) {
  const Matrix3Xd &pos = s.concave().pos;
  const CapTable &caps = s.concave().caps;
  const Vector3d x = pos.col(p);
  const int nt = static_cast<int>(hemispheres(s, p).cols()),
            beg = caps.off[p] + nt, end = caps.off[p + 1];

  std::vector<Cutter> out;
  for (int y = 0; y < pos.cols(); ++y) {
    if (y == p)
      continue;
    const Vector3d w = pos.col(y) - x;
    const double dist = w.norm();
    if (dist >= 2 * kRp)
      continue;

    Cutter c { y, w / dist, dist / (2 * kRp), false };
    for (int k = beg; k < end; ++k) {
      c.kept |= (caps.axis.col(k) - c.axis).norm() < 1e-9
                && std::abs(caps.cosa[k] - c.cosa) < 1e-12;
    }
    out.push_back(c);
  }
  EXPECT_EQ(
      static_cast<int>(std::count_if(out.begin(), out.end(),
                                     [](const Cutter &c) { return c.kept; })),
      end - beg)
      << "probe " << p;
  return out;
}

/**
 * Random clusters: a cutter whose cap removes a sampled face direction
 * that no other cutter removes is kept, and every sampled face direction a
 * dropped cutter removes is removed by a kept one too.
 */
TEST(BuildSesTest, FiltersAreNecessary) {
  const Matrix3Xd dirs = canonical_fibonacci_lattice(2000);
  int n_kept = 0, n_dropped = 0, n_cut = 0;

  for (const int seed: { 1, 2, 3, 4, 5, 6, 7, 8 }) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 6.0), ur(1.0, 1.4);
    const int n = 8;
    Matrix3Xd pts(3, n);
    ArrayXd sar(n);
    for (int i = 0; i < n; ++i) {
      pts.col(i) << u(rng), u(rng), u(rng);
      sar[i] = ur(rng) + kRp;
    }

    const SurfaceGeometry s = solve(pts, sar);
    const ConcavePatches &con = s.concave();
    for (int p = 0; p < con.pos.cols(); ++p) {
      const auto t = hemispheres(s, p);
      const int nt = static_cast<int>(t.cols());
      const std::vector<Cutter> cs = cutters(s, p);
      n_cut += con.caps.off.degree(p) > nt;
      for (const Cutter &c: cs) {
        n_kept += c.kept;
        n_dropped += !c.kept;
      }

      ArrayXb inside(cs.size());
      for (int k = 0; k < dirs.cols(); ++k) {
        const Vector3d d = dirs.col(k);
        if ((t.transpose() * d).maxCoeff() > 0)
          continue;

        for (int y = 0; y < cs.size(); ++y)
          inside[y] = d.dot(cs[y].axis) > cs[y].cosa;
        const int n_inside = inside.count();
        bool kept_inside = false;
        for (int y = 0; y < cs.size(); ++y)
          kept_inside |= inside[y] && cs[y].kept;

        for (int y = 0; y < cs.size(); ++y) {
          if (!inside[y] || cs[y].kept)
            continue;
          EXPECT_TRUE(kept_inside)
              << "seed " << seed << " probe " << p << " dropped cutter "
              << cs[y].probe << " removes direction " << k << " alone";
          EXPECT_GT(n_inside, 1);
        }
      }
    }
  }
  EXPECT_GT(n_cut, 0);
  EXPECT_GT(n_kept, 0);
  EXPECT_GT(n_dropped, 0);
}

TEST(BuildSesTest, ThreeSpheresVsDots) {
  Matrix3Xd pts(3, 3);
  pts << 0, 4.8, 2.6,  //
      0, 0.2, 4.16,    //
      0, 0.3, -0.4;
  ArrayXd sar(3);
  sar << 1.5 + kRp, 1.9 + kRp, 1.7 + kRp;

  const SurfaceGeometry s = solve(pts, sar);
  const int n_probes = static_cast<int>(s.concave().pos.cols());
  ASSERT_GT(n_probes, 0);
  for (int p = 0; p < n_probes; ++p)
    ASSERT_GT(s.concave().caps.off.degree(p), 3) << "probe " << p << " is high";
  expect_faces_match_dots(s, kRp);

  double convex = 0;
  for (int i = 0; i < s.convex().area.size(); ++i)
    convex += convex_from_dots(s, i, kRp);
  double faces = 0;
  for (int p = 0; p < n_probes; ++p)
    faces += face_from_dots(s, p, kRp);
  const double total = convex + saddle_from_quadrature(s, kRp) + faces,
               analytic = s.convex().area.sum() + s.toroidal().area.sum()
                          + s.concave().area.sum();
  EXPECT_NEAR(analytic, total, 1e-3 * total);
}

TEST(BuildSesTest, RandomClustersVsDots) {
  for (const int seed: { 1, 2, 3, 4, 5, 6 }) {
    const RandomCluster c = random_cluster(seed, 8 + seed % 5);
    const SurfaceGeometry s = solve(c.pts, c.sar);
    expect_faces_match_dots(s, kRp, "seed");
  }

  const RandomCluster dense = random_cluster(1, 32, 8.0);
  const SurfaceGeometry s = solve(dense.pts, dense.sar);
  EXPECT_GE(s.concave().caps.off.max_deg(), 8);
  expect_faces_match_dots(s, kRp, "dense");
}

/**
 * Every face solved again on the unfiltered cap list (all probes strictly
 * within `2 rp`) through the same solver; returns how many faces had a
 * filtered cap.
 */
int expect_filters_keep_area(const Matrix3Xd &pts, const ArrayXd &sar,
                             const std::string &where) {
  const Ses s = solve_stages(pts, sar);
  const SasProbes &probes = s.geo.probes;

  int n_extra = 0;
  for (int p = 0; p < probes.n_active; ++p) {
    const Vector3d x = probes.pos.col(p);
    const int nt = probes.tan_off.degree(p);
    std::vector<int> near;
    for (int y = 0; y < probes.pos.cols(); ++y) {
      if (y != p && (probes.pos.col(y) - x).norm() < 2 * kRp)
        near.push_back(y);
    }

    const int m = nt + static_cast<int>(near.size());
    n_extra += m > s.ses.face_off.degree(p);
    Matrix3Xd n(3, m);
    ArrayXd h(m), cosa(m);
    n.leftCols(nt) = probes.tan.middleCols(probes.tan_off[p], nt);
    h.head(nt).setZero();
    cosa.head(nt).setZero();
    for (int k = 0; k < static_cast<int>(near.size()); ++k) {
      n.col(nt + k) = probes.pos.col(near[k]) - x;
      h[nt + k] = n.col(nt + k).squaredNorm();
      cosa[nt + k] = std::sqrt(h[nt + k]) / (2 * kRp);
    }

    ArrayXb live;
    const double area = solve_ses_face(n, h, cosa, kRp, live);
    EXPECT_TRUE(live.all()) << where << " probe " << p;
    EXPECT_NEAR(area, s.ses.face_area[p], 1e-9) << where << " probe " << p;
  }
  return n_extra;
}

/**
 * Spheres 1, 2 overlap by 1e-4 in a circle of radius 0.014 that sphere 0
 * cuts nearly tangentially, so the face's two roots sit `h` on either side
 * of the plane of the three centers with nearly coplanar departure
 * tangents; sphere 3 supplies cutters within `2 rp` of them.
 */
std::pair<Matrix3Xd, ArrayXd> near_tangent_face(const double h) {
  const double r = 2.0, d = 4.0 - 1e-4, zs = 3.0;
  const double rc = std::sqrt(r * r - d * d / 4),
               zc = std::sqrt(rc * rc - h * h);
  Matrix3Xd pts(3, 4);
  pts.col(0) << 0, 0, zs;
  pts.col(1) << -d / 2, 0, 0;
  pts.col(2) << d / 2, 0, 0;
  pts.col(3) << 1.5, 1.5, 1.0;
  ArrayXd sar(4);
  sar << std::sqrt(h * h + (zs - zc) * (zs - zc)), r, r, 2.2;
  return { pts, sar };
}

TEST(BuildSesTest, FiltersDoNotChangeArea) {
  int n_extra = 0;
  for (const int seed: { 1, 2, 3, 4, 5, 6 }) {
    const RandomCluster c = random_cluster(seed, 8 + seed % 5);
    n_extra +=
        expect_filters_keep_area(c.pts, c.sar, "seed " + std::to_string(seed));
  }
  EXPECT_GT(n_extra, 0);

  for (const double h: { 5e-6, 2e-7, 1e-13 }) {
    const auto [pts, sar] = near_tangent_face(h);
    EXPECT_GT(expect_filters_keep_area(pts, sar, "h " + std::to_string(h)), 0);
  }
}

/**
 * The two roots of the nearly tangent face, `(0, +-h, zc)`, each keep the
 * other's cap and are cut by it to the zone between the contact plane and
 * its mirror, of area at most `2 pi rp h` up to the vertex certificate
 * (a wrong side keeps the hemisphere, `2 pi rp^2`): the side of the plane
 * comes from the root sign, which the rounded height cannot give below
 * the position certificate.
 */
TEST(BuildSesTest, NearTangentFacesAreSlivers) {
  for (const double h: { 5e-6, 2e-7, 1e-13 }) {
    SCOPED_TRACE(h);
    const auto [pts, sar] = near_tangent_face(h);
    const double r = 2.0, d = 4.0 - 1e-4,
                 zc = std::sqrt(r * r - d * d / 4 - h * h);
    const SurfaceGeometry s = solve(pts, sar);
    const ConcavePatches &con = s.concave();

    int n = 0;
    for (int p = 0; p < con.pos.cols(); ++p) {
      if ((con.pos.col(p) - Vector3d(0, 0, zc)).norm() > 1e-5)
        continue;
      ++n;
      EXPECT_GT(con.caps.off.degree(p), hemispheres(s, p).cols()) << p;
      EXPECT_LE(con.area[p],
                4 * kPi * kRp * h + kTwoPi * kRp * kRp * kOffsetRelTol)
          << p;
    }
    EXPECT_EQ(n, 2);
  }
}

/**
 * A neighbor probe at exactly `2 rp` is tangent to the probe sphere and
 * carries no cap; one an ulp closer carries a cap of angular radius
 * `2^-26`, cut or buried exactly, of area below `1e-14`.
 */
TEST(BuildSesTest, TangentNeighborCapIsDropped) {
  const double rp = 1.0;
  Matrix3Xd n(3, 4);
  n.col(0) = Vector3d(1, 0, -1).normalized();
  n.col(1) = Vector3d(-1, 1, -1).normalized();
  n.col(2) = Vector3d(-1, -1, -1).normalized();
  ArrayXd h = ArrayXd::Zero(4), cosa = ArrayXd::Zero(4);

  ArrayXb live;
  const double uncut =
      solve_ses_face(n.leftCols(3), h.head(3), cosa.head(3), rp, live);
  EXPECT_TRUE(live.all());
  EXPECT_GT(uncut, 0);

  for (const double scale: { 1.0, 1 - 0x1p-52 }) {
    n.col(3) = Vector3d(0, 0, 2 * rp * scale);
    h[3] = n.col(3).squaredNorm();
    cosa[3] = std::sqrt(h[3]) / (2 * rp);
    const double area = solve_ses_face(n, h, cosa, rp, live);
    EXPECT_TRUE(live.head(3).all()) << scale;
    if (scale == 1.0) {
      EXPECT_FALSE(live[3]);
      EXPECT_EQ(area, uncut);
    } else {
      EXPECT_NEAR(area, uncut, 1e-12);
    }
  }
}

/**
 * Dyadic triangle with circumcenter `(4, 0, 0)` and circumradius exactly 4.
 */
Matrix3Xd circumradius_four_triangle() {
  Matrix3Xd pts(3, 3);
  pts << 0, 8, 4,  //
      0, 0, 4,     //
      0, 0, 0;
  return pts;
}

/**
 * Circumcenter `(4, 0, 0)`, circumradius 4, `sar` 5, `rp` 3: the probes at
 * `(4, 0, +-3)` are exactly `2 rp` apart, so neither caps the other; a rigid
 * motion rounds the pair to either side of tangency.
 */
TEST(BuildSesTest, TangentNeighborProbe) {
  const double rp = 3.0;
  const Matrix3Xd base = circumradius_four_triangle();
  const ArrayXd sar = ArrayXd::Constant(3, 5.0);

  const SurfaceGeometry ref = solve(base, sar, rp);
  const ConcavePatches &con = ref.concave();
  ASSERT_EQ(con.pos.cols(), 2);
  for (const double z: { 3.0, -3.0 }) {
    const int p = probe_at(con.pos, Vector3d(4, 0, z), 1e-12);
    ASSERT_GE(p, 0) << "z " << z;
    EXPECT_EQ(hemispheres(ref, p).cols(), 3) << "z " << z;
    EXPECT_EQ(con.caps.off.degree(p), 3) << "z " << z;
  }
  EXPECT_NEAR(con.area[0], con.area[1], 1e-12);
  expect_faces_match_dots(ref, rp, "tangent");

  for (int seed = 0; seed < 16; ++seed) {
    SCOPED_TRACE(seed);
    const auto [rot, t] = rigid_motion(seed);
    const Matrix3Xd pts = (rot * base).colwise() + t;
    const SurfaceGeometry s = solve(pts, sar, rp);
    ASSERT_EQ(s.concave().pos.cols(), 2);
    for (int p = 0; p < 2; ++p) {
      EXPECT_NEAR(s.concave().area[p], con.area[0], 1e-9) << "probe " << p;
      const int m = s.concave().caps.off.degree(p);
      EXPECT_TRUE(m == 3 || m == 4) << "probe " << p << " caps " << m;
    }
    expect_faces_match_dots(s, rp, "tangent");
  }
}

/**
 * The merged probe at `x` of a planar layout: `k` atoms, hemispheres in
 * antipodal pairs, exactly zero area. Returns its index, -1 if absent.
 */
int expect_grazing_cluster(const SurfaceGeometry &s, const Vector3d &x,
                           const int k) {
  const ConcavePatches &con = s.concave();
  const int c = probe_at(con.pos, x);
  EXPECT_GE(c, 0);
  if (c < 0)
    return c;

  EXPECT_EQ(con.atoms.degree(c), k);
  const auto hs = hemispheres(s, c);
  EXPECT_GE(hs.cols(), 2);
  bool antipodal = false;
  for (int i = 0; i < hs.cols(); ++i) {
    for (int j = i + 1; j < hs.cols(); ++j)
      antipodal |= hs.col(i) == -hs.col(j);
  }
  EXPECT_TRUE(antipodal);
  EXPECT_EQ(con.area[c], 0);
  return c;
}

/**
 * Acute dyadic triangle inscribed in the circle of radius 5 about the origin
 * with `sar` 5: probe height exactly 0, the two probes coincide at the
 * circumcenter. Shrake-Rupley to 0.05 of NearBandPinch times `(sar / 2)^2`.
 */
TEST(BuildSesTest, PlanarProbe) {
  const double rp = 2.0, sr_tol = 0.05 * 2.5 * 2.5;
  Matrix3Xd base(3, 3);
  base << 5, -3, -3,  //
      0, 4, -4,       //
      0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(3, 5.0);

  const auto [ref_sas, ref_ses] = solve_both(base, sar, rp);
  ASSERT_EQ(ref_sas.n_atoms(), 3);
  ArrayXd ref_area(3);
  for (int i = 0; i < 3; ++i)
    ref_area[ref_sas.order()[i]] = ref_sas.convex().area[i];
  const double ref_total = ref_ses.total_area();

  auto expect_sr = [&](const SurfaceGeometry &sas, const Matrix3Xd &pts) {
    const ArrayXd sr = sr_sasa_impl(pts, sar, 20000, SrSasaMethod::kDirect);
    ASSERT_EQ(sas.n_atoms(), 3);
    for (int i = 0; i < 3; ++i) {
      const int o = sas.order()[i];
      EXPECT_NEAR(sas.convex().area[i], sr[o], sr_tol) << "atom " << o;
    }
  };

  for_each_index_order(base, sar,
                       [&](const Matrix3Xd &pts, const ArrayXd &r,
                           const std::vector<int> &order) {
                         SCOPED_TRACE(testing::Message()
                                      << "order " << order[0] << order[1]
                                      << order[2]);
                         const auto [sas, ses] = solve_both(pts, r, rp);
                         expect_sr(sas, pts);
                         for (int i = 0; i < sas.n_atoms(); ++i) {
                           const int o = order[sas.order()[i]];
                           EXPECT_NEAR(sas.convex().area[i], ref_area[o], 1e-9)
                               << "atom " << o;
                         }
                         EXPECT_NEAR(ses.total_area(), ref_total, 1e-9);
                         expect_faces_match_dots(ses, rp, "planar");
                         expect_grazing_cluster(ses, Vector3d::Zero(), 3);
                       });

  const Ses stages = solve_stages(base, sar, rp);
  const int c = probe_at(stages.geo.probes.pos, Vector3d::Zero());
  ASSERT_GE(c, 0);
  EXPECT_EQ(stages.geo.probes.members[c], 2);
  EXPECT_EQ(stages.ses.face_area[c], 0);

  for (int seed = 0; seed < 8; ++seed) {
    SCOPED_TRACE(seed);
    const auto [rot, t] = rigid_motion(seed);
    const Matrix3Xd pts = (rot * base).colwise() + t;
    const auto [sas, ses] = solve_both(pts, sar, rp);
    expect_sr(sas, pts);
    expect_faces_match_dots(ses, rp, "planar");
  }
}

/**
 * The planar triangle with a fourth atom above it whose probes cut the
 * circumcenter's probe sphere: the grazing loops end the face before the
 * cutters are looked at, so its area stays exactly 0.
 */
TEST(BuildSesTest, PlanarProbeWithCutter) {
  const double rp = 2.0;
  Matrix3Xd pts(3, 4);
  pts << 5, -3, -3, 0,  //
      0, 4, -4, 0,      //
      0, 0, 0, 7;
  ArrayXd sar(4);
  sar << 5, 5, 5, 5;

  const SurfaceGeometry ses = solve(pts, sar, rp);
  const int c = expect_grazing_cluster(ses, Vector3d::Zero(), 3);
  ASSERT_GE(c, 0);
  int n_near = 0;
  for (int p = 0; p < ses.concave().pos.cols(); ++p) {
    const double dist = ses.concave().pos.col(p).norm();
    n_near += p != c && dist < 2 * rp;
  }
  EXPECT_GT(n_near, 0);
  expect_faces_match_dots(ses, rp, "cutter");
}

/**
 * Coincident roots with covered circles (Lemma 13 of the analytic_surface
 * page): the obtuse triangle `(-3, 0), (3, 0), (0, 1)` of `sar` 5 meets at
 * `(0, -4, 0)` where the circle of the first two is covered by the third;
 * the pair `(-3, 4), (3, 4)` of `sar` 5 and `(0, 3)` of `sar` 3 meet at
 * the origin beyond a corner, two circles covered. Under the perturbation
 * the tie is a cut iff the first-listed atom's barycentric coordinate is
 * positive: one order gives the zero-area cluster with one antipodal pair,
 * the other no probe at all, and the surface is the same.
 */
TEST(BuildSesTest, CoincidentRootsWithCoveredCircles) {
  const double rp = 2.0;
  struct Layout {
    Matrix3Xd pts;
    ArrayXd sar;
    Vector3d x;
    std::vector<int> cut, miss;
  };
  Matrix3Xd obtuse(3, 3), corner(3, 3);
  obtuse << -3, 3, 0,  //
      0, 0, 1,         //
      0, 0, 0;
  corner << -3, 3, 0,  //
      4, 4, 3,         //
      0, 0, 0;
  ArrayXd sar(3);
  sar << 5, 5, 5;
  ArrayXd mixed(3);
  mixed << 5, 5, 3;
  const std::array<Layout, 2> layouts {
    Layout { obtuse, sar, Vector3d(0, -4, 0), { 0, 1, 2 }, { 2, 0, 1 } },
    Layout { corner, mixed, Vector3d::Zero(), { 2, 0, 1 }, { 0, 1, 2 } },
  };

  for (const Layout &l: layouts) {
    SCOPED_TRACE(l.x.transpose());
    auto reorder = [&](const std::vector<int> &order) {
      Matrix3Xd p(3, 3);
      ArrayXd r(3);
      for (int i = 0; i < 3; ++i) {
        p.col(i) = l.pts.col(order[i]);
        r[i] = l.sar[order[i]];
      }
      return solve(p, r, rp);
    };

    const SurfaceGeometry cut = reorder(l.cut);
    const int c = expect_grazing_cluster(cut, l.x, 3);
    ASSERT_GE(c, 0);
    expect_faces_match_dots(cut, rp, "cut");

    const SurfaceGeometry miss = reorder(l.miss);
    EXPECT_EQ(probe_at(miss.concave().pos, l.x), -1);
    expect_faces_match_dots(miss, rp, "miss");
    EXPECT_NEAR(miss.total_area(), cut.total_area(), 1e-9);
  }
}

/**
 * Probes `P = (i, j, k, k')` at the origin and `P' = (i, j, l, l')` at
 * `(3, 0, 9)` lie on the circle `(i, j)` of radius exactly `rp`, a cusp at
 * `(0, 0, 5)`. On `P` the cap of `P'` touches the hemisphere of the `(i, j)`
 * arc at the cusp point from inside, so its ring is two coincident roots
 * there and the loop is decided by the antipode `(3, 0, 4)`, which the
 * hemisphere covers: the face is the spherical quadrilateral `i, j, k', k`.
 * Four atoms per probe keep both off the planar and triangular prefilters.
 * Under a rigid motion each 4-fold point rounds into distinct roots whose
 * faces partition its face, with the same areas.
 */
TEST(BuildSesTest, SinglePointWindow) {
  const double rp = 5.0;
  Matrix3Xd base(3, 6);
  base.col(0) << 0, -12, 5;
  base.col(1) << 0, 12, 5;
  base.col(2) << -12, 0, -5;
  base.col(3) << -9, 0, 14;
  base.col(4) << -4, 12, -3;
  base.col(5) << -1, 12, 12;
  const ArrayXd sar = ArrayXd::Constant(6, 13.0);
  const Vector3d x0(0, 0, 0), w(3, 0, 9);
  const double quad = 50 * (kPi / 2 + std::atan2(19.0, 91.0));

  const SurfaceGeometry ref = solve(base, sar, rp);
  ASSERT_EQ(ref.concave().pos.cols(), 6);
  const int p = probe_at(ref.concave().pos, x0),
            q = probe_at(ref.concave().pos, w);
  ASSERT_GE(p, 0);
  ASSERT_GE(q, 0);
  EXPECT_EQ(ref.concave().pos.col(p), x0);
  EXPECT_EQ(ref.concave().pos.col(q), w);
  EXPECT_EQ(ref.concave().atoms.degree(p), 4);
  EXPECT_EQ(ref.concave().atoms.degree(q), 4);

  const auto tan = hemispheres(ref, p);
  ASSERT_EQ(tan.cols(), 4);
  int n_exact = 0;
  for (int i = 0; i < tan.cols(); ++i)
    n_exact += static_cast<int>(tan.col(i) == Vector3d(1, 0, 0));
  EXPECT_EQ(n_exact, 1);

  const CapTable &caps = ref.concave().caps;
  ASSERT_EQ(caps.off.degree(p), 5);
  const int kcap = caps.off[p] + 4;
  EXPECT_EQ(caps.cosa[kcap], w.norm() / (2 * rp));
  EXPECT_EQ(caps.axis.col(kcap), Vector3d(w / w.norm()));

  EXPECT_NEAR(ref.concave().area[p], quad, 1e-12);
  expect_faces_match_dots(ref, rp);

  const double ref_total = ref.concave().area.sum();
  for (int seed = 0; seed < 8; ++seed) {
    SCOPED_TRACE(seed);
    const auto [rot, t] = rigid_motion(seed);
    const Matrix3Xd pts = (rot * base).colwise() + t;
    const SurfaceGeometry s = solve(pts, sar, rp);
    EXPECT_NEAR(area_within(s, rot * x0 + t, 1e-6), quad, 1e-9);
    EXPECT_NEAR(s.concave().area.sum(), ref_total, 1e-9);
  }
}

struct KFold {
  Matrix3Xd pts;
  ArrayXd sar;
  int k;
  bool cut;
};

/**
 * The ring of the k-star alone, and the 4-ring with a fifth sphere whose
 * probe with two ring atoms lies within `2 rp` of the 4-fold probe and cuts
 * its face. The 4-ring's coordinates round symmetrically, so its point is
 * exact; the 5-ring's do not, so even the reference is a rounded point
 * whose roots are distinct probes.
 */
std::vector<KFold> k_fold_fixtures() {
  std::vector<KFold> fixtures;
  for (const int k: { 4, 5 }) {
    fixtures.push_back({ Matrix3Xd(star(k, 110, 2.0).rightCols(k)),
                         ArrayXd::Constant(k, 2.0), k, false });
  }
  Matrix3Xd cut(3, 5);
  cut.leftCols(4) = star(4, 110, 2.0).rightCols(4);
  cut.col(4) << 1.5, 1.5, 2.5;
  fixtures.push_back({ cut, ArrayXd::Constant(5, 2.0), 4, true });
  return fixtures;
}

TEST(BuildSesTest, KFoldFaceUnderRigidMotion) {
  for (const KFold &f: k_fold_fixtures()) {
    SCOPED_TRACE(testing::Message() << "k " << f.k << " cut " << f.cut);
    const Ses ref = solve_stages(f.pts, f.sar, 1.0);
    std::vector<int> atoms;
    int n_cut = 0;
    for (int p = 0; p < ref.geo.probes.n_active; ++p) {
      if (ref.geo.probes.pos.col(p).norm() > 1e-6)
        continue;
      for (const int a: ref.geo.probes.atoms.nbrs(p))
        atoms.push_back(ref.ss.order[a]);
      n_cut += ref.ses.face_off.degree(p) > ref.geo.probes.tan_off.degree(p);
    }
    std::sort(atoms.begin(), atoms.end());
    atoms.erase(std::unique(atoms.begin(), atoms.end()), atoms.end());
    EXPECT_EQ(static_cast<int>(atoms.size()), f.k);
    if (f.cut)
      EXPECT_GT(n_cut, 0);
    const double ref_face = area_within(ref, Vector3d::Zero(), 1e-6),
                 ref_total = ref.ses.face_area.sum();
    EXPECT_GT(ref_face, 0);
    expect_faces_match_dots(ref, 1.0, "k-fold");

    for (int seed = 0; seed < 16; ++seed) {
      const auto [rot, t] = rigid_motion(seed);
      const Matrix3Xd pts = (rot * f.pts).colwise() + t;
      const Ses s = solve_stages(pts, f.sar, 1.0);
      EXPECT_NEAR(area_within(s, t, 1e-6), ref_face, 1e-9) << "seed " << seed;
      EXPECT_NEAR(s.ses.face_area.sum(), ref_total, 1e-9) << "seed " << seed;
      expect_faces_match_dots(s, 1.0, "rounded k-fold");
    }
  }
}

/**
 * The occluder-shell fixture of sas_test, all active, under rigid motions:
 * where the tangent pair `(0, 1)` rounds into an overlap, the roots on its
 * circle of radius ~1e-8 are neighbor probes 1e-8 to 1e-7 apart whose
 * cap axes their positions cannot resolve, so the lattice oracle places
 * them by the exact difference. The faces at the point sum to the quarter
 * sphere of the exact 4-fold point and match the lattice.
 */
TEST(BuildSesTest, RoundedTangentPairFaces) {
  Matrix3Xd base(3, 6);
  base << 0, 3, 1.5, 1.5, 1.5, 1.5,  //
      0, 0, 1.5, 0, -1.5, -3.5,      //
      0, 0, 0, 1.5, -1.5, -3.5;
  const ArrayXd sar = ArrayXd::Constant(6, 1.5);
  const double rp = 0.5, quarter = kPi * rp * rp;
  const Vector3d x0(1.5, 0, 0);

  const SurfaceGeometry ref = solve(base, sar, rp);
  const int c0 = probe_at(ref.concave().pos, x0);
  ASSERT_GE(c0, 0);
  EXPECT_NEAR(ref.concave().area[c0], quarter, 1e-12);

  for (int seed = 0; seed < 32; ++seed) {
    SCOPED_TRACE(seed);
    const auto [rot, t] = rigid_motion(seed);
    const Matrix3Xd pts = (rot * base).colwise() + t;
    const Ses s = solve_stages(pts, sar, rp);

    EXPECT_NEAR(area_within(s, rot * x0 + t, 1e-6), quarter, 1e-6);
    EXPECT_NEAR(s.ses.face_area.sum(), ref.concave().area.sum(), 1e-6);
    expect_faces_match_dots(s, rp);
  }
}

/**
 * Spindle circle (a, b) of radius 3 under `rp = 3.25` with four probes at
 * dyadic points; the two axis points `(0, 0, +-1.25)` lie on every probe
 * sphere, so on each face the hemisphere of the `(a, b)` arc and the caps
 * of the other three probes pass through the same two points exactly.
 */
TEST(BuildSesTest, SpindleAxisCoincidentRoots) {
  const double rp = 3.25;
  Matrix3Xd pts(3, 4);
  pts.col(0) << 0, 0, -4;
  pts.col(1) << 0, 0, 4;
  pts.col(2) << 3, 3, 4;
  pts.col(3) << -3, -3, 4;
  const ArrayXd sar = ArrayXd::Constant(4, 5.0);

  const SurfaceGeometry s = solve(pts, sar, rp);
  const int q = circle_of(s, 0, 1);
  ASSERT_GE(q, 0);
  EXPECT_LT(s.toroidal().circles[q].rl, rp);

  const std::array<Vector3d, 4> on_circle {
    Vector3d(0, 3, 0), Vector3d(3, 0, 0), Vector3d(0, -3, 0), Vector3d(-3, 0, 0)
  };
  for (const Vector3d &x: on_circle) {
    const int p = probe_at(s.concave().pos, x);
    ASSERT_GE(p, 0) << x.transpose();
    EXPECT_EQ(s.concave().pos.col(p), x);
    EXPECT_GE(s.concave().caps.off.degree(p) - hemispheres(s, p).cols(), 2);
  }
  expect_faces_match_dots(s, rp);
}

/**
 * Active faces of a masked run equal the same faces of the all-active run:
 * the shells supply every neighbor probe within `2 rp`.
 */
TEST(BuildSesTest, MaskedFacesMatchFull) {
  const RandomCluster c = random_cluster(11, 12);
  const SurfaceGeometry full = solve(c.pts, c.sar);

  for (const int stride: { 2, 6 }) {
    ArrayXb active(12);
    for (int i = 0; i < 12; ++i)
      active[i] = i % stride == 0;

    const SurfaceGeometry part = solve(c.pts, c.sar, active);
    const int n_probes = static_cast<int>(part.concave().pos.cols());
    ASSERT_GT(n_probes, 0);
    ASSERT_LT(n_probes, full.concave().pos.cols());

    for (int p = 0; p < n_probes; ++p) {
      const int q = probe_at(full.concave().pos, part.concave().pos.col(p));
      ASSERT_GE(q, 0) << "stride " << stride << " probe " << p;
      EXPECT_NEAR(part.concave().area[p], full.concave().area[q], 1e-12)
          << "stride " << stride << " probe " << p;
    }
  }
}
}  // namespace
}  // namespace internal
}  // namespace nuri

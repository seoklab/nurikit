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

constexpr double kRp = 1.4;

struct Ses {
  SaPrep sa;
  SasGeometry geo;
  SesGeometry ses;
};

Ses solve(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXb &active,
          const double rp = kRp) {
  std::optional<SaPrep> sa = prepare(pts, sar, active, rp);
  EXPECT_TRUE(sa);
  SasDelaunay del = triangulate(*sa);
  SasGeometry geo = build_sas(*sa, del);
  SesGeometry ses = build_ses(*sa, geo, rp);
  return { std::move(*sa), std::move(geo), std::move(ses) };
}

Ses solve(const Matrix3Xd &pts, const ArrayXd &sar, const double rp = kRp) {
  return solve(pts, sar, ArrayXb::Constant(sar.size(), true), rp);
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

struct RigidMotion {
  Matrix3d rot;
  Vector3d t;
};

RigidMotion rigid_motion(const int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  const Vector3d ax(nd(rng), nd(rng), nd(rng));
  const Matrix3d rot = AngleAxisd(nd(rng), ax.normalized()).toRotationMatrix();
  const Vector3d t(nd(rng), nd(rng), nd(rng));
  return { rot, 3.0 * t };
}

/**
 * Circle of the original atoms `a`, `b`, -1 if none.
 */
int circle_of(const Ses &s, const int a, const int b) {
  const auto [lo, hi] = nuri::minmax(a, b);
  for (int q = 0; q < static_cast<int>(s.geo.circles.size()); ++q) {
    const SasCircle &c = s.geo.circles[q];
    const auto [ci, cj] = nuri::minmax(s.sa.order[c.i], s.sa.order[c.j]);
    if (ci == lo && cj == hi)
      return q;
  }
  return -1;
}

/**
 * The one probe within `radius` of `x`, -1 if none or several.
 */
int probe_at(const SasGeometry &geo, const Vector3d &x,
             const double radius = 1e-9) {
  int found = -1;
  for (int p = 0; p < geo.probes.pos.cols(); ++p) {
    if ((geo.probes.pos.col(p) - x).norm() > radius)
      continue;
    if (found >= 0)
      return -1;
    found = p;
  }
  return found;
}

struct RandomCluster {
  Matrix3Xd pts;
  ArrayXd sar;
};

RandomCluster random_cluster(const int seed, const int n) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0, 6.0), ur(1.0, 1.4);
  RandomCluster c { Matrix3Xd(3, n), ArrayXd(n) };
  for (int i = 0; i < n; ++i) {
    c.pts.col(i) << u(rng), u(rng), u(rng);
    c.sar[i] = ur(rng) + kRp;
  }
  return c;
}

constexpr int kLattice = 1000000;

const Matrix3Xd &lattice() {
  static const Matrix3Xd dirs = canonical_fibonacci_lattice(kLattice);
  return dirs;
}

/**
 * Lattice estimate of face `p`: directions outside every departure
 * hemisphere whose point on the probe sphere is outside every other probe
 * ball, brute force over all probes.
 */
double face_from_dots(const Ses &s, const int p, const double rp) {
  const SasProbes &probes = s.geo.probes;
  const Matrix3Xd &dirs = lattice();
  const Vector3d x = probes.pos.col(p);
  const int nt = probes.tan_off.degree(p);
  const auto t = probes.tan.middleCols(probes.tan_off[p], nt);

  std::vector<Vector3d> near;
  for (int y = 0; y < probes.pos.cols(); ++y) {
    const Vector3d w = probes.pos.col(y) - x;
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
      outside &= (rp * d - w).squaredNorm() >= rp * rp;
    count += outside;
  }
  return 2 * kTwoPi * rp * rp * count / dirs.cols();
}

double convex_from_dots(const Ses &s, const int i, const double rp) {
  const Matrix3Xd &dirs = lattice();
  const Vector3d c = s.sa.pts.col(i);
  const double sas = s.sa.sar[i], r = sas - rp;

  int count = 0;
  for (int k = 0; k < dirs.cols(); ++k) {
    const Vector3d y = c + sas * dirs.col(k);
    bool outside = true;
    for (int j = 0; j < s.sa.pts.cols(); ++j) {
      outside &= j == i
                 || (y - s.sa.pts.col(j)).squaredNorm()
                        >= s.sa.sar[j] * s.sa.sar[j];
    }
    count += outside;
  }
  return 2 * kTwoPi * r * r * count / dirs.cols();
}

/**
 * Torus patches by quadrature of `rp (rl − rp cos β)` over each arc's full
 * generating range; spindles excluded (`rl ≥ rp` asserted).
 */
double saddle_from_quadrature(const Ses &s, const double rp) {
  double total = 0;
  for (int r = 0; r < s.geo.n_active_arcs; ++r) {
    const SasArc &arc = s.geo.arcs[r];
    const SasCircle &c = s.geo.circles[arc.circ];
    EXPECT_GE(c.rl, rp);
    const double lo = -std::atan2(c.a, c.rl),
                 hi = std::atan2(s.sa.d[arc.circ] - c.a, c.rl);
    total += arc.dphi * rp
             * simpson([&](double b) { return c.rl - rp * std::cos(b); }, lo,
                       hi, 2000);
  }
  return total;
}

/**
 * Every active face against the lattice to `1e-3` relative, floored at
 * `1e-4` of the probe sphere (four times the worst lattice error seen on
 * faces below 0.1 Å²).
 */
void expect_faces_match_dots(const Ses &s, const double rp,
                             const char *tag = "") {
  const double sphere = 2 * kTwoPi * rp * rp;
  for (int p = 0; p < s.geo.probes.n_active; ++p) {
    const double area = s.ses.face_area[p];
    ASSERT_TRUE(std::isfinite(area)) << tag << " probe " << p;
    EXPECT_GE(area, -1e-9) << tag << " probe " << p;
    const double dots = face_from_dots(s, p, rp);
    EXPECT_NEAR(area, dots, nuri::max(1e-3 * dots, 1e-4 * sphere))
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

  const Ses s = solve(pts, sar);
  ASSERT_EQ(s.ses.convex_area.size(), 2);
  for (int k = 0; k < 2; ++k) {
    const bool first = s.sa.order[k] == 0;
    const double r = first ? r1 : r2, a = first ? a1 : a2;
    EXPECT_NEAR(s.ses.convex_area[k], kTwoPi * r * r * (1 + a / (r + kRp)),
                1e-10);
  }

  ASSERT_EQ(s.geo.arcs.size(), 1);
  ASSERT_EQ(s.geo.n_active_arcs, 1);
  ASSERT_EQ(s.ses.saddle_beta.cols(), 1);
  ASSERT_EQ(s.ses.saddle_area.size(), 1);
  EXPECT_DOUBLE_EQ(s.ses.saddle_beta(1, 0), 0);
  EXPECT_DOUBLE_EQ(s.ses.saddle_beta(2, 0), 0);

  const double torus = kTwoPi * kRp
                       * (rl * (std::atan2(a1, rl) + std::atan2(a2, rl))
                          - kRp * (a1 / sar[0] + a2 / sar[1]));
  EXPECT_NEAR(s.ses.saddle_area[0], torus, 1e-10);

  EXPECT_EQ(s.geo.probes.pos.cols(), 0);
  EXPECT_EQ(s.ses.face_area.size(), 0);
  EXPECT_EQ(s.ses.face_off.size(), 0);
}

TEST(BuildSesTest, TwoSpheresClosedForm) {
  expect_two_sphere_closed_form(1.5, 1.5, 2.5);
  expect_two_sphere_closed_form(1.2, 1.9, 2.5);
}

/**
 * Spindle torus: the arc `[−θ_1, θ_2]` loses `|β| < β0 = acos(rl / rp)`.
 */
void expect_two_sphere_cusp(const double r1, const double r2, const double d,
                            const bool lower_part, const bool upper_part) {
  Matrix3Xd pts(3, 2);
  pts << 0, d, 0, 0, 0, 0;
  ArrayXd sar(2);
  sar << r1 + kRp, r2 + kRp;

  const Ses s = solve(pts, sar);
  ASSERT_EQ(s.ses.saddle_beta.cols(), 1);
  ASSERT_EQ(s.ses.saddle_area.size(), 1);
  const SasCircle &c = s.geo.circles[0];
  const double rl = c.rl, a_i = c.a, a_j = s.sa.d[0] - c.a;
  ASSERT_LT(rl, kRp);
  const double b0 = std::acos(rl / kRp), lo = -std::atan2(a_i, rl),
               hi = std::atan2(a_j, rl);

  const Array4d beta = s.ses.saddle_beta.col(0);
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
  EXPECT_GE(s.ses.saddle_integral(0, 0), 0);
  EXPECT_GE(s.ses.saddle_integral(1, 0), 0);

  auto rho = [&](double b) { return kRp * (rl - kRp * std::cos(b)); };
  double integral = 0;
  if (lo < -b0)
    integral += simpson(rho, lo, nuri::min(hi, -b0), 2000);
  if (hi > b0)
    integral += simpson(rho, nuri::max(lo, b0), hi, 2000);
  EXPECT_NEAR(s.ses.saddle_area[0], kTwoPi * integral, 1e-8);
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

  const Ses s = solve(pts, sar, active);
  ASSERT_EQ(s.sa.n_active, 1);
  EXPECT_EQ(s.sa.order[0], 1);
  ASSERT_EQ(s.ses.convex_area.size(), 1);
  const auto [a1, a2, rl] = two_spheres(sar[0], sar[1], 2.5);
  EXPECT_NEAR(s.ses.convex_area[0], kTwoPi * 1.9 * 1.9 * (1 + a2 / sar[1]),
              1e-10);
  EXPECT_EQ(s.ses.saddle_beta.cols(), 1);
  EXPECT_EQ(s.ses.saddle_integral.cols(), 1);
  EXPECT_EQ(s.ses.saddle_area.size(), 1);
  EXPECT_EQ(s.ses.face_area.size(), 0);

  Matrix3Xd tet(3, 4);
  tet << 0, 2.4, 1.2, 1.2,  //
      0, 0, 2.1, 0.7,       //
      0, 0, 0, 2.0;
  ArrayXd sar4 = ArrayXd::Constant(4, 1.9 + kRp);
  ArrayXb one = ArrayXb::Constant(4, false);
  one[0] = true;

  const Ses t = solve(tet, sar4, one);
  ASSERT_EQ(t.sa.n_active, 1);
  EXPECT_EQ(t.ses.convex_area.size(), 1);
  const int n_circ = t.sa.g.offset(1);
  EXPECT_EQ(n_circ, 3);
  EXPECT_LT(n_circ, t.sa.g.m());
  EXPECT_EQ(t.ses.saddle_beta.cols(), n_circ);
  EXPECT_EQ(t.ses.saddle_integral.cols(), n_circ);
  EXPECT_EQ(t.ses.saddle_area.size(), t.geo.n_active_arcs);
  EXPECT_LT(t.geo.n_active_arcs, static_cast<int>(t.geo.arcs.size()));
  EXPECT_GT(t.geo.probes.n_active, 0);
  EXPECT_LT(t.geo.probes.n_active, t.geo.probes.pos.cols());
  EXPECT_EQ(t.ses.face_area.size(), t.geo.probes.n_active);
  EXPECT_EQ(t.ses.face_off.size(), t.geo.probes.n_active);
}

/**
 * `rp² (2π − Σ angles)` over the planes through the probe and each pair of
 * its atom centres, oriented away from the third atom.
 */
double excess_from_planes(const Ses &s, const int p) {
  const Vector3d x = s.geo.probes.pos.col(p);
  const Matrix3d tri = s.sa.pts(E::all, s.geo.probes.atoms.nbrs(p));
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

double excess_from_tangents(const Ses &s, const int p) {
  const auto t = s.geo.probes.tan.middleCols<3>(s.geo.probes.tan_off[p]);
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

  const Ses s = solve(pts, sar);
  ASSERT_EQ(s.geo.probes.pos.cols(), 2);
  ASSERT_EQ(s.geo.probes.n_active, 2);
  ASSERT_EQ(s.ses.face_area.size(), 2);
  ASSERT_EQ(s.ses.face_off.size(), 2);

  for (int p = 0; p < 2; ++p) {
    ASSERT_GE(std::abs(s.geo.probes.pos(2, p)), kRp);
    ASSERT_EQ(s.geo.probes.tan_off.degree(p), 3);

    const double area = s.ses.face_area[p];
    EXPECT_GT(area, 0);
    EXPECT_LT(area, kTwoPi * kRp * kRp);
    EXPECT_NEAR(area, excess_from_tangents(s, p), 1e-12);
    EXPECT_NEAR(area, excess_from_planes(s, p), 1e-10);

    ASSERT_EQ(s.ses.face_off.degree(p), 3);
    for (int k = s.ses.face_off[p]; k < s.ses.face_off[p + 1]; ++k) {
      EXPECT_EQ(s.ses.face_cosa[k], 0);
      EXPECT_EQ(s.ses.face_sina[k], 1);
      EXPECT_NEAR(s.ses.face_axis.col(k).norm(), 1, 1e-12);
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
std::vector<Cutter> cutters(const Ses &s, const int p) {
  const SasProbes &probes = s.geo.probes;
  const Vector3d x = probes.pos.col(p);
  const int nt = probes.tan_off.degree(p), beg = s.ses.face_off[p] + nt,
            end = s.ses.face_off[p + 1];

  std::vector<Cutter> out;
  for (int y = 0; y < probes.pos.cols(); ++y) {
    if (y == p)
      continue;
    const Vector3d w = probes.pos.col(y) - x;
    const double dist = w.norm();
    if (dist >= 2 * kRp)
      continue;

    Cutter c { y, w / dist, dist / (2 * kRp), false };
    for (int k = beg; k < end; ++k) {
      c.kept |= (s.ses.face_axis.col(k) - c.axis).norm() < 1e-9
                && std::abs(s.ses.face_cosa[k] - c.cosa) < 1e-12;
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

    const Ses s = solve(pts, sar);
    const SasProbes &probes = s.geo.probes;
    for (int p = 0; p < probes.n_active; ++p) {
      const int nt = probes.tan_off.degree(p);
      const auto t = probes.tan.middleCols(probes.tan_off[p], nt);
      const std::vector<Cutter> cs = cutters(s, p);
      n_cut += s.ses.face_off.degree(p) > nt;
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

  const Ses s = solve(pts, sar);
  ASSERT_GT(s.geo.probes.n_active, 0);
  for (int p = 0; p < s.geo.probes.n_active; ++p)
    ASSERT_GT(s.ses.face_off.degree(p), 3) << "probe " << p << " is high";
  expect_faces_match_dots(s, kRp);

  double convex = 0;
  for (int i = 0; i < s.sa.n_active; ++i)
    convex += convex_from_dots(s, i, kRp);
  double faces = 0;
  for (int p = 0; p < s.geo.probes.n_active; ++p)
    faces += face_from_dots(s, p, kRp);
  const double total = convex + saddle_from_quadrature(s, kRp) + faces,
               analytic = s.ses.convex_area.sum() + s.ses.saddle_area.sum()
                          + s.ses.face_area.sum();
  EXPECT_NEAR(analytic, total, 1e-3 * total);
}

TEST(BuildSesTest, RandomClustersVsDots) {
  for (const int seed: { 1, 2, 3, 4, 5, 6 }) {
    const RandomCluster c = random_cluster(seed, 8 + seed % 5);
    const Ses s = solve(c.pts, c.sar);
    expect_faces_match_dots(s, kRp, "seed");
  }
}

/**
 * Every face solved again on the unfiltered cap list (all probes strictly
 * within `2 rp`) through the same solver.
 */
TEST(BuildSesTest, FiltersDoNotChangeArea) {
  int n_extra = 0;
  for (const int seed: { 1, 2, 3, 4, 5, 6 }) {
    const RandomCluster c = random_cluster(seed, 8 + seed % 5);
    const Ses s = solve(c.pts, c.sar);
    const SasProbes &probes = s.geo.probes;

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
      EXPECT_TRUE(live.all()) << "seed " << seed << " probe " << p;
      EXPECT_NEAR(area, s.ses.face_area[p], 1e-9)
          << "seed " << seed << " probe " << p;
    }
  }
  EXPECT_GT(n_extra, 0);
}

/**
 * Sphere c grazes circle (a, b): the one merged probe has two full loops
 * with tangents `±y`, two exactly complementary hemispheres, no face.
 */
TEST(BuildSesTest, GrazingVertexFaceIsZero) {
  Matrix3Xd pts(3, 3);
  pts.col(0) << 0, 0, 0;
  pts.col(1) << 0, 0, 6;
  pts.col(2) << 7, 0, 7;
  const ArrayXd sar = ArrayXd::Constant(3, 5.0);

  const Ses s = solve(pts, sar, 1.0);
  ASSERT_EQ(s.geo.probes.pos.cols(), 1);
  ASSERT_EQ(s.ses.face_area.size(), 1);
  EXPECT_EQ(s.ses.face_area[0], 0);
  EXPECT_EQ(s.ses.face_off.degree(0), s.geo.probes.tan_off.degree(0));
}

/**
 * A neighbour probe at exactly `2 rp` is tangent to the probe sphere and
 * carries no cap; one an ulp closer carries a cap of angular radius
 * `2^-26`, cut or buried exactly, of area below `1e-14`.
 */
TEST(BuildSesTest, TangentNeighbourCapIsDropped) {
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

TEST(BuildSesTest, KFoldFaceUnderRigidMotion) {
  for (const int k: { 4, 5 }) {
    const Matrix3Xd base = star(k, 110, 2.0).rightCols(k);
    const ArrayXd sar = ArrayXd::Constant(k, 2.0);
    const Ses ref = solve(base, sar, 1.0);
    const int c0 = probe_at(ref.geo, Vector3d::Zero());
    ASSERT_GE(c0, 0) << "k " << k;
    ASSERT_EQ(ref.geo.probes.atoms.degree(c0), k);
    const double ref_face = ref.ses.face_area[c0],
                 ref_total = ref.ses.face_area.sum();
    EXPECT_GT(ref_face, 0);
    expect_faces_match_dots(ref, 1.0, "k-fold");

    for (int seed = 0; seed < 16; ++seed) {
      const auto [rot, t] = rigid_motion(seed);
      const Matrix3Xd pts = (rot * base).colwise() + t;
      const Ses s = solve(pts, sar, 1.0);
      const int c = probe_at(s.geo, t);
      ASSERT_GE(c, 0) << "k " << k << " seed " << seed;
      EXPECT_NEAR(s.ses.face_area[c], ref_face, 1e-9)
          << "k " << k << " seed " << seed;
      EXPECT_NEAR(s.ses.face_area.sum(), ref_total, 1e-9)
          << "k " << k << " seed " << seed;
    }
  }
}

/**
 * Spindle circle (a, b) of radius 3 under `rp = 3.25` with four probes at
 * dyadic points; the two axis points `(0, 0, ±1.25)` lie on every probe
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

  const Ses s = solve(pts, sar, rp);
  const int q = circle_of(s, 0, 1);
  ASSERT_GE(q, 0);
  EXPECT_LT(s.geo.circles[q].rl, rp);

  const std::array<Vector3d, 4> on_circle {
    Vector3d(0, 3, 0), Vector3d(3, 0, 0), Vector3d(0, -3, 0), Vector3d(-3, 0, 0)
  };
  for (const Vector3d &x: on_circle) {
    const int p = probe_at(s.geo, x);
    ASSERT_GE(p, 0) << x.transpose();
    EXPECT_EQ(s.geo.probes.pos.col(p), x);
    EXPECT_GE(s.ses.face_off.degree(p) - s.geo.probes.tan_off.degree(p), 2);
  }
  expect_faces_match_dots(s, rp);
}

/**
 * Active faces of a masked run equal the same faces of the all-active run:
 * the shells supply every neighbour probe within `2 rp`.
 */
TEST(BuildSesTest, MaskedFacesMatchFull) {
  const RandomCluster c = random_cluster(11, 12);
  const Ses full = solve(c.pts, c.sar);

  for (const int stride: { 2, 6 }) {
    ArrayXb active(12);
    for (int i = 0; i < 12; ++i)
      active[i] = i % stride == 0;

    const Ses part = solve(c.pts, c.sar, active);
    ASSERT_GT(part.geo.probes.n_active, 0);
    ASSERT_LT(part.geo.probes.n_active, full.geo.probes.n_active);

    for (int p = 0; p < part.geo.probes.n_active; ++p) {
      const int q = probe_at(full.geo, part.geo.probes.pos.col(p));
      ASSERT_GE(q, 0) << "stride " << stride << " probe " << p;
      ASSERT_LT(q, full.geo.probes.n_active);
      EXPECT_NEAR(part.ses.face_area[p], full.ses.face_area[q], 1e-12)
          << "stride " << stride << " probe " << p;
    }
  }
}
}  // namespace
}  // namespace internal
}  // namespace nuri

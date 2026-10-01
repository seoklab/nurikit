//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <optional>
#include <utility>

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
}  // namespace
}  // namespace internal
}  // namespace nuri

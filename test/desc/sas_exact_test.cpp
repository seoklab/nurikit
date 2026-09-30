//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include <geogram/delaunay/delaunay_3d.h>
#include <geogram/numerics/predicates.h>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"

namespace nuri {
namespace internal {
namespace {
using Q = __float128;

Q q_sqrt(const Q x) {
  if (x <= 0)
    return 0;
  Q s = std::sqrt(static_cast<double>(x));
  for (int i = 0; i < 4; ++i)
    s = (s + x / s) / 2;
  return s;
}

int q_sgn(const Q x, const Q tol = 0) {
  return x > tol ? 1 : x < -tol ? -1 : 0;
}

struct QVec {
  Q x, y, z;
};

QVec operator-(const QVec &a, const QVec &b) {
  return { a.x - b.x, a.y - b.y, a.z - b.z };
}
QVec operator+(const QVec &a, const QVec &b) {
  return { a.x + b.x, a.y + b.y, a.z + b.z };
}
QVec operator*(const Q s, const QVec &a) {
  return { s * a.x, s * a.y, s * a.z };
}
Q dot(const QVec &a, const QVec &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
QVec cross(const QVec &a, const QVec &b) {
  return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
           a.x * b.y - a.y * b.x };
}

/**
 * Reference geometry in quad precision on squared radii `rho2` (which the
 * tie tests perturb one at a time).
 */
struct Ref {
  Matrix3Xd c;
  std::vector<Q> rho2;

  QVec cen(int i) const { return { c(0, i), c(1, i), c(2, i) }; }

  Q power(int i, const QVec &x) const {
    const QVec y = x - cen(i);
    return dot(y, y) - rho2[i];
  }

  // pi_i - pi_a = 2 (v_i - y . d_i), y = x - c_a
  Q v(int a, int i) const {
    const QVec d = cen(i) - cen(a);
    return (dot(d, d) + rho2[a] - rho2[i]) / 2;
  }

  struct Face {
    QVec db, dc, u, yperp;
    Q u2, disc;
  };

  Face face(SasFace f) const {
    Face r;
    r.db = cen(f.b) - cen(f.a);
    r.dc = cen(f.c) - cen(f.a);
    r.u = cross(r.db, r.dc);
    r.u2 = dot(r.u, r.u);
    const Q gbb = dot(r.db, r.db), gbc = dot(r.db, r.dc), gcc = dot(r.dc, r.dc),
            vb = v(f.a, f.b), vc = v(f.a, f.c);
    const Q lam = gcc * vb - gbc * vc, mu = gbb * vc - gbc * vb;
    r.yperp = (1 / r.u2) * (lam * r.db + mu * r.dc);
    r.disc = r.u2 * (rho2[f.a] - dot(r.yperp, r.yperp));
    return r;
  }

  QVec root(SasFace f, bool plus) const {
    const Face fc = face(f);
    const Q s = q_sqrt(fc.disc > 0 ? fc.disc : 0) / fc.u2;
    return cen(f.a) + fc.yperp + ((plus ? s : -s) * fc.u);
  }

  QVec cntr(int a, int b) const {
    const QVec d = cen(b) - cen(a);
    return cen(a) + (v(a, b) / dot(d, d)) * d;
  }

  // sign of pi_c on circle (a, b): constant when c does not cut it
  Q side(int a, int b, int c) const {
    const QVec d = cen(b) - cen(a), dc = cen(c) - cen(a);
    return dot(d, d) * v(a, c) - dot(d, dc) * v(a, b);
  }

  Q overlap(int a, int b) const {
    const QVec d = cen(b) - cen(a);
    const Q t = rho2[a] + rho2[b] - dot(d, d);
    return t >= 0 ? Q(1) : 4 * rho2[a] * rho2[b] - t * t;
  }
};

struct Fixture {
  Matrix4Xd lifted;
  double wmax;
  SasExact ex;
  Ref ref;
};

Fixture setup(const Matrix3Xd &pts, const ArrayXd &sar) {
  Fixture s;
  const int n = static_cast<int>(sar.size());
  s.wmax = sar.square().maxCoeff();
  s.lifted.resize(4, n);
  s.lifted.topRows(3) = pts;
  for (int i = 0; i < n; ++i)
    s.lifted(3, i) = std::sqrt(s.wmax - sar[i] * sar[i]);
  s.ex = SasExact::make(s.lifted, s.wmax);
  s.ref.c = pts;
  s.ref.rho2.resize(n);
  for (int i = 0; i < n; ++i) {
    const Q cc = Q(pts(0, i)) * pts(0, i) + Q(pts(1, i)) * pts(1, i)
                 + Q(pts(2, i)) * pts(2, i);
    s.ref.rho2[i] = Q(s.wmax) + cc - Q(s.ex.h()[i]);
  }
  return s;
}

Matrix3Xd star(const int n_ring, const double polar_deg, const double reach) {
  const double polar = polar_deg * constants::kPi / 180;
  Matrix3Xd pts(3, n_ring + 1);
  pts.col(0) << 0, 0, reach;
  for (int k = 0; k < n_ring; ++k) {
    const double az = constants::kTwoPi * k / n_ring;
    pts.col(k + 1) << reach * std::sin(polar) * std::cos(az),
        reach * std::sin(polar) * std::sin(az), reach * std::cos(polar);
  }
  return pts;
}

Matrix3Xd square_apex() {
  Matrix3Xd pts(3, 5);
  pts.col(0) << 1, 1, 0;
  pts.col(1) << 1, -1, 0;
  pts.col(2) << -1, 1, 0;
  pts.col(3) << -1, -1, 0;
  pts.col(4) << 0, 0, 1.6;
  return pts;
}

Matrix3Xd tetra_through_origin() {
  Matrix3Xd pts(3, 4);
  pts.col(0) << 1, 1, 1;
  pts.col(1) << 1, -1, -1;
  pts.col(2) << -1, 1, -1;
  pts.col(3) << -1, -1, 1;
  return pts / std::sqrt(3.0);
}

Matrix3Xd random_pts(std::mt19937 &rng, const int n, const double box) {
  std::uniform_real_distribution<double> u(-box, box);
  Matrix3Xd pts(3, n);
  for (int i = 0; i < n; ++i)
    pts.col(i) << u(rng), u(rng), u(rng);
  return pts;
}

ArrayXd random_radii(std::mt19937 &rng, const int n, const double lo,
                     const double hi) {
  std::uniform_real_distribution<double> u(lo, hi);
  ArrayXd r(n);
  for (int i = 0; i < n; ++i)
    r[i] = u(rng);
  return r;
}

TEST(SasExactTest, Selftest) {
  EXPECT_TRUE(SasExact::selftest());
}

void expect_geogram_agrees(const Fixture &s) {
  const int n = static_cast<int>(s.lifted.cols());
  GEO::Delaunay3d del(4);
  del.set_keeps_infinite(true);
  del.set_vertices(n, s.lifted.data());

  const double *p = s.lifted.data();
  int checked = 0;
  for (GEO::index_t c = 0; c < del.nb_finite_cells(); ++c) {
    std::array<int, 4> v;
    for (int lv = 0; lv < 4; ++lv)
      v[lv] = static_cast<int>(del.cell_vertex(c, lv));
    for (int q = 0; q < n; ++q) {
      if (q == v[0] || q == v[1] || q == v[2] || q == v[3])
        continue;
      const GEO::Sign sg = GEO::PCK::orient_3dlifted_SOS(
          p + 4 * v[0], p + 4 * v[1], p + 4 * v[2], p + 4 * v[3], p + 4 * q,
          s.ex.h()[v[0]], s.ex.h()[v[1]], s.ex.h()[v[2]], s.ex.h()[v[3]],
          s.ex.h()[q]);
      ASSERT_EQ(sg, GEO::NEGATIVE) << "cell " << c << " conflicts with " << q;
      ++checked;
    }
  }
  EXPECT_GE(checked + 4, n);
}

TEST(SasExactTest, HeightsMatchGeogram) {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 5; ++trial) {
    const Matrix3Xd pts = random_pts(rng, 40, 5.0);
    const ArrayXd sar = random_radii(rng, 40, 1.4, 2.6);
    expect_geogram_agrees(setup(pts, sar));
  }

  expect_geogram_agrees(setup(star(4, 110, 2), ArrayXd::Constant(5, 2.0)));
  expect_geogram_agrees(setup(star(5, 100, 2), ArrayXd::Constant(6, 2.0)));
  expect_geogram_agrees(setup(square_apex(), ArrayXd::Constant(5, 1.5)));
  expect_geogram_agrees(
      setup(tetra_through_origin(), ArrayXd::Constant(4, 1.0)));

  Matrix3Xd lattice(3, 27);
  for (int i = 0; i < 27; ++i)
    lattice.col(i) << i % 3, (i / 3) % 3, i / 9;
  expect_geogram_agrees(setup(lattice, ArrayXd::Constant(27, 0.8)));
}

constexpr Q kQTol = 1e-26;

int expect_root_sign(const Ref &ref, const SasFace f, const bool plus,
                     const int l) {
  return q_sgn(ref.power(l, ref.root(f, plus)), kQTol);
}

void check_random_agreement(const bool force) {
  SasExact::force_exact(force);
  std::mt19937 rng(force ? 11 : 13);
  int n_accept = 0, n_cuts = 0, n_side = 0, n_hp = 0, n_ccw = 0, n_anti = 0;

  for (int trial = 0; trial < 40; ++trial) {
    const int n = 8;
    const Matrix3Xd pts = random_pts(rng, n, 1.6);
    const ArrayXd sar = random_radii(rng, n, 1.5, 2.5);
    const Fixture s = setup(pts, sar);
    const Ref &ref = s.ref;

    for (int a = 0; a < n; ++a) {
      for (int b = a + 1; b < n; ++b) {
        const int ov = q_sgn(ref.overlap(a, b), kQTol);
        if (ov != 0) {
          EXPECT_EQ(static_cast<int>(s.ex.overlap(a, b)), ov);
          EXPECT_EQ(static_cast<int>(
                        SasExact::overlap(pts.col(a), s.lifted(3, a),
                                          pts.col(b), s.lifted(3, b), s.wmax)),
                    ov);
        }
        for (int c = b + 1; c < n; ++c) {
          const SasFace f { a, b, c };
          const Ref::Face rf = ref.face(f);
          const int cut = q_sgn(rf.disc, kQTol);
          if (cut == 0)
            continue;
          EXPECT_EQ(static_cast<int>(s.ex.cuts(f)), cut);
          ++n_cuts;

          if (cut < 0) {
            const int sd = q_sgn(ref.side(a, b, c), kQTol);
            if (sd != 0) {
              EXPECT_EQ(static_cast<int>(s.ex.side(a, b, c)), sd);
              ++n_side;
            }
            continue;
          }

          const auto [xp, xm] = s.ex.roots(f);
          const QVec qp = ref.root(f, true), qm = ref.root(f, false);
          EXPECT_LT(std::abs(xp[0] - static_cast<double>(qp.x)), 1e-9);
          EXPECT_LT(std::abs(xm[1] - static_cast<double>(qm.y)), 1e-9);

          for (int l = 0; l < n; ++l) {
            if (l == a || l == b || l == c)
              continue;
            for (const bool plus: { true, false }) {
              const int want = expect_root_sign(ref, f, plus, l);
              if (want == 0)
                continue;
              EXPECT_EQ(static_cast<int>(s.ex.accept(f, plus, l)), want)
                  << a << b << c << " root " << plus << " apex " << l;
              ++n_accept;

              const QVec q = (Q(2) * ref.cntr(a, b)) - ref.root(f, plus);
              const int anti = q_sgn(ref.power(l, q), kQTol);
              if (anti != 0) {
                EXPECT_EQ(static_cast<int>(s.ex.antipode(a, b, f, plus, l)),
                          anti);
                ++n_anti;
              }
            }
          }

          const Vector3d dfl = pts.col(b) - pts.col(a);
          const int k = SasExact::reference_axis(dfl);
          Vector3d ek = Vector3d::Zero();
          ek[k] = 1;
          const Vector3d rfl = dfl.cross(ek);
          const QVec r { rfl[0], rfl[1], rfl[2] };
          const QVec d = ref.cen(b) - ref.cen(a), r2 = cross(d, r);
          for (const bool plus: { true, false }) {
            const QVec y = ref.root(f, plus) - ref.cntr(a, b);
            const int sc = q_sgn(dot(r, y), kQTol),
                      ss = q_sgn(dot(r2, y), kQTol);
            if (ss == 0 || sc == 0)
              continue;
            EXPECT_EQ(s.ex.half_plane(a, b, f, plus), ss > 0 ? 1 : 3);
            ++n_hp;
          }

          for (int c2 = c + 1; c2 < n; ++c2) {
            const SasFace f2 { a, b, c2 };
            if (q_sgn(ref.face(f2).disc, kQTol) <= 0)
              continue;
            const QVec yi = ref.root(f, true) - ref.cntr(a, b),
                       yj = ref.root(f2, false) - ref.cntr(a, b);
            const int want = q_sgn(dot(cross(yi, yj), d), kQTol);
            if (want == 0)
              continue;
            EXPECT_EQ(static_cast<int>(s.ex.ccw(a, b, f, true, f2, false)),
                      want);
            ++n_ccw;
          }
        }
      }
    }
  }
  SasExact::force_exact(false);

  EXPECT_GT(n_accept, 1000);
  EXPECT_GT(n_cuts, 300);
  EXPECT_GT(n_side, 30);
  EXPECT_GT(n_hp, 100);
  EXPECT_GT(n_ccw, 50);
  EXPECT_GT(n_anti, 500);
}

TEST(SasExactTest, FilteredMatchesQuad) {
  check_random_agreement(false);
}

TEST(SasExactTest, ExactMatchesQuad) {
  check_random_agreement(true);
}

/**
 * Perturbation oracle: grow the squared radii one at a time by 1e-20, in
 * index order, and take the first sign that emerges.
 */
template <class F>
int perturbed_sign(Ref ref, const std::vector<int> &participants, F &&value) {
  std::vector<int> order = participants;
  std::sort(order.begin(), order.end());
  order.erase(std::unique(order.begin(), order.end()), order.end());
  for (const int j: order) {
    Ref pert = ref;
    pert.rho2[j] += Q(1e-20);
    const int s = q_sgn(value(pert), kQTol);
    if (s != 0)
      return s;
  }
  return 0;
}

TEST(SasExactTest, TangentPairOverlaps) {
  Matrix3Xd pts(3, 3);
  pts.col(0) << 0, 0, 0;
  pts.col(1) << 3, 0, 0;
  pts.col(2) << 1.5, 1.5, 0;
  const ArrayXd sar = ArrayXd::Constant(3, 1.5);
  const Fixture s = setup(pts, sar);

  EXPECT_EQ(s.ex.overlap(0, 1), Sgn::kPos);
  EXPECT_EQ(SasExact::overlap(pts.col(0), s.lifted(3, 0), pts.col(1),
                              s.lifted(3, 1), s.wmax),
            Sgn::kPos);
  // sphere 2 passes exactly through the tangency point (1.5, 0, 0)
  const SasFace f { 0, 1, 2 };
  const int want = perturbed_sign(s.ref, { 0, 1, 2 },
                                  [&](const Ref &r) { return r.face(f).disc; });
  ASSERT_NE(want, 0);
  EXPECT_EQ(static_cast<int>(s.ex.cuts(f)), want);
}

void check_tie_fixture(const Matrix3Xd &pts, const ArrayXd &sar,
                       const Vector3d &point) {
  const Fixture s = setup(pts, sar);
  const int n = static_cast<int>(sar.size());
  int n_ties = 0;

  for (int a = 0; a < n; ++a) {
    for (int b = a + 1; b < n; ++b) {
      for (int c = b + 1; c < n; ++c) {
        const SasFace f { a, b, c };
        const int cut = static_cast<int>(s.ex.cuts(f));
        const int want_cut = perturbed_sign(
            s.ref, { a, b, c }, [&](const Ref &r) { return r.face(f).disc; });
        if (want_cut != 0) {
          EXPECT_EQ(cut, want_cut) << a << b << c;
        }
        if (cut <= 0)
          continue;

        for (const bool plus: { true, false }) {
          const auto [xp, xm] = s.ex.roots(f);
          const Vector3d x = plus ? xp : xm;
          for (int l = 0; l < n; ++l) {
            if (l == a || l == b || l == c)
              continue;
            const int got = static_cast<int>(s.ex.accept(f, plus, l));
            const int want =
                perturbed_sign(s.ref, { a, b, c, l }, [&](const Ref &r) {
                  return r.power(l, r.root(f, plus));
                });
            ASSERT_NE(want, 0);
            EXPECT_EQ(got, want)
                << a << b << c << " root " << plus << " apex " << l;
            n_ties += (x - point).norm() < 1e-9;
          }
        }
      }
    }
  }
  EXPECT_GT(n_ties, 0);
}

TEST(SasExactTest, TetraThroughOrigin) {
  check_tie_fixture(tetra_through_origin(), ArrayXd::Constant(4, 1.0),
                    Vector3d::Zero());
}

TEST(SasExactTest, FiveFoldStar) {
  check_tie_fixture(star(4, 110, 2), ArrayXd::Constant(5, 2.0),
                    Vector3d::Zero());
}

TEST(SasExactTest, CoplanarSquareWithApex) {
  check_tie_fixture(square_apex(), ArrayXd::Constant(5, 1.5),
                    Vector3d(0, 0, 0.5));
}

TEST(SasExactTest, CoincidentRootsTie) {
  // two 4-fold points from two faces of the square: coincident roots
  const Fixture s = setup(square_apex(), ArrayXd::Constant(5, 1.5));
  const SasFace f1 { 0, 1, 2 }, f2 { 0, 1, 3 };
  ASSERT_EQ(s.ex.cuts(f1), Sgn::kPos);
  ASSERT_EQ(s.ex.cuts(f2), Sgn::kPos);
  const auto [p1, m1] = s.ex.roots(f1);
  const auto [p2, m2] = s.ex.roots(f2);
  const bool same = (p1 - p2).norm() < 1e-12;
  EXPECT_EQ(s.ex.ccw(0, 1, f1, true, f2, same), Sgn::kZero);
  EXPECT_EQ(s.ex.half_plane(0, 1, f1, true), s.ex.half_plane(0, 1, f2, same));
}
}  // namespace
}  // namespace internal
}  // namespace nuri

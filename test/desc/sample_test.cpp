//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <utility>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "desc_test_common.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/fmt/pdb.h"
#include "nuri/random.h"

namespace nuri {
namespace internal {
namespace {
using constants::kTwoPi;

constexpr double kRp = 1.4;

constexpr SurfacePatch kPatch[] = { SurfacePatch::kConvex,
                                    SurfacePatch::kToroidal,
                                    SurfacePatch::kConcave };

struct Sampled {
  internal::SurfaceGeometry geo;
  SurfaceDots dots;
};

struct Spheres {
  Matrix3Xd pts;
  ArrayXd sar;
  ArrayXb active;
};

Spheres three_spheres() {
  Matrix3Xd pts(3, 3);
  pts << 0, 4.8, 2.6,  //
      0, 0.2, 4.16,    //
      0, 0.3, -0.4;
  ArrayXd sar(3);
  sar << 1.5 + kRp, 1.9 + kRp, 1.7 + kRp;
  return { std::move(pts), std::move(sar), ArrayXb::Constant(3, true) };
}

Spheres random_cluster() {
  const int n = 12;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0, 6.0), ur(1.0, 1.4);
  Matrix3Xd pts(3, n);
  ArrayXd sar(n);
  for (int i = 0; i < n; ++i) {
    pts.col(i) << u(rng), u(rng), u(rng);
    sar[i] = ur(rng) + kRp;
  }
  ArrayXb active = ArrayXb::Constant(n, false);
  active.head(n / 2).setConstant(true);
  return { std::move(pts), std::move(sar), std::move(active) };
}

Sampled solve(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXb &active,
              const double rp, const double density) {
  const ArrayXd r = sar - rp;
  AnalyticSurface surf(pts, r);
  EXPECT_TRUE(surf.initialize(rp, active));
  internal::SurfaceGeometry geo = surf.build_ses();
  SurfaceSampler sampler(density);
  SurfaceDots dots = sampler.sample(geo);
  return { std::move(geo), std::move(dots) };
}

Sampled solve(const Matrix3Xd &pts, const ArrayXd &sar, const double rp,
              const double density) {
  return solve(pts, sar, ArrayXb::Constant(sar.size(), true), rp, density);
}

internal::SurfaceGeometry seeded_ses(const Spheres &s, const double rp) {
  const ArrayXd r = s.sar - rp;
  AnalyticSurface surf(s.pts, r);
  EXPECT_TRUE(surf.initialize(rp, s.active));
  internal::seed_thread(0);
  return surf.build_ses();
}

double block_area(const SurfaceDots &dots, const int kind) {
  const SurfacePatch p = kPatch[kind];
  return dots.area().segment(dots.offset(p), dots.count(p)).sum();
}

Array3d analytic_areas(const Sampled &s) {
  return { s.geo.total_area(SurfacePatch::kConvex),
           s.geo.total_area(SurfacePatch::kToroidal),
           s.geo.total_area(SurfacePatch::kConcave) };
}

/**
 * Nothing dropped: every block carries exactly its analytic area.
 */
void expect_block_sums(const Sampled &s) {
  const Array3d ref = analytic_areas(s);
  EXPECT_EQ(s.dots.dropped_area(), 0);
  for (int b = 0; b < 3; ++b)
    EXPECT_NEAR(block_area(s.dots, b), ref[b], 1e-9) << "block " << b;
}

/**
 * Patches may drop: no block exceeds its analytic area and the drops make
 * up the difference.
 */
void expect_block_sums_with_drops(const Sampled &s) {
  const Array3d ref = analytic_areas(s);
  EXPECT_GE(s.dots.dropped_area(), 0);
  double sampled = 0;
  for (int b = 0; b < 3; ++b) {
    sampled += block_area(s.dots, b);
    EXPECT_LE(block_area(s.dots, b), ref[b] + 1e-9) << "block " << b;
  }
  EXPECT_NEAR(sampled + s.dots.dropped_area(), ref.sum(), 1e-9 * ref.sum());
}

/**
 * Every probe center `pts + rp * nrm` is outside every SAS sphere, and every
 * normal is unit.
 */
void expect_on_ses(const Sampled &s, const Matrix3Xd &pts, const ArrayXd &sar,
                   const double rp) {
  const SurfaceDots &d = s.dots;
  EXPECT_EQ(d.probes(), Matrix3Xd(d.pts() + rp * d.nrm()));
  for (int k = 0; k < d.n(); ++k) {
    EXPECT_NEAR(d.nrm().col(k).norm(), 1, 1e-12) << "dot " << k;
    const Vector3d probe = d.pts().col(k) + rp * d.nrm().col(k);
    for (int a = 0; a < pts.cols(); ++a) {
      EXPECT_GE((probe - pts.col(a)).norm(), sar[a] - 1e-9)
          << "dot " << k << " atom " << a;
    }
  }
}

/**
 * `subset` keeps the masked dots in block order and `keep` agrees with it.
 */
SurfaceDots expect_keep_subset(const SurfaceDots &d, const ArrayXb &keep) {
  SurfaceDots sub = d.subset(keep);
  EXPECT_EQ(sub.n(), keep.count());
  EXPECT_EQ(sub.rp(), d.rp());
  EXPECT_EQ(sub.dropped_area(), d.dropped_area());
  EXPECT_EQ(sub.offset(SurfacePatch::kConvex), 0);

  int w = 0;
  for (int b = 0; b < 3; ++b) {
    int kept = 0;
    const int beg = d.offset(kPatch[b]), end = beg + d.count(kPatch[b]);
    for (int k = beg; k < end; ++k) {
      if (!keep[k])
        continue;
      ++kept;
      EXPECT_EQ(sub.pts().col(w), d.pts().col(k));
      EXPECT_EQ(sub.nrm().col(w), d.nrm().col(k));
      EXPECT_EQ(sub.area()[w], d.area()[k]);
      EXPECT_EQ(sub.atom()[w], d.atom()[k]);
      ++w;
    }
    EXPECT_EQ(sub.count(kPatch[b]), kept) << "block " << b;
  }

  SurfaceDots inplace = d;
  inplace.keep(keep);
  EXPECT_EQ(inplace.n(), sub.n());
  EXPECT_EQ(inplace.pts(), sub.pts());
  EXPECT_EQ(inplace.nrm(), sub.nrm());
  EXPECT_TRUE((inplace.area() == sub.area()).all());
  EXPECT_TRUE((inplace.atom() == sub.atom()).all());
  for (int b = 0; b < 3; ++b) {
    EXPECT_EQ(inplace.offset(kPatch[b]), sub.offset(kPatch[b]));
    EXPECT_EQ(inplace.count(kPatch[b]), sub.count(kPatch[b]));
  }
  EXPECT_EQ(inplace.rp(), sub.rp());
  EXPECT_EQ(inplace.dropped_area(), sub.dropped_area());
  return sub;
}

ArrayXb every_other(const int n) {
  ArrayXb keep(n);
  for (int k = 0; k < n; ++k)
    keep[k] = k % 2 == 0;
  return keep;
}

TEST(SampleSesTest, SingleSphere) {
  const double r = 1.5, rp = 1.7, density = 15;
  const Matrix3Xd pts = Matrix3Xd::Zero(3, 1);
  const ArrayXd sar = ArrayXd::Constant(1, r + rp);

  const Sampled s = solve(pts, sar, rp, density);
  const SurfaceDots &d = s.dots;
  const double sphere = 2 * kTwoPi * r * r;
  EXPECT_NEAR(d.area().sum(), sphere, 1e-9);
  EXPECT_EQ(d.n(), static_cast<int>(std::lround(sphere * density)));
  EXPECT_EQ(d.count(kPatch[0]), d.n());
  EXPECT_EQ(d.count(kPatch[1]), 0);
  EXPECT_EQ(d.count(kPatch[2]), 0);
  EXPECT_EQ(d.rp(), rp);

  for (int k = 0; k < d.n(); ++k) {
    EXPECT_NEAR(d.pts().col(k).norm(), r, 1e-12);
    EXPECT_NEAR((d.nrm().col(k) - d.pts().col(k) / r).norm(), 0, 1e-12);
    EXPECT_EQ(d.atom()[k], 0);
  }
}

TEST(SampleSesTest, TwoSpheresStammTable1) {
  const double rp = 1.2;
  Matrix3Xd pts(3, 2);
  pts << 0, 2, 0, 0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(2, 1.2 + rp);

  const Sampled s = solve(pts, sar, rp, 200);
  expect_block_sums(s);
  EXPECT_EQ(s.dots.count(kPatch[2]), 0);
  EXPECT_NEAR(block_area(s.dots, 0) + block_area(s.dots, 1), 32.23514, 1e-4);
}

TEST(SampleSesTest, TwoSpheresCusp) {
  const double rp = 1.2;
  Matrix3Xd pts(3, 2);
  pts << 0, 4.4, 0, 0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(2, 1.2 + rp);

  const Sampled s = solve(pts, sar, rp, 200);
  ASSERT_LT(std::sqrt(sar[0] * sar[0] - 2.2 * 2.2), rp);
  expect_block_sums(s);
  EXPECT_GT(s.dots.count(kPatch[1]), 0);
  expect_on_ses(s, pts, sar, rp);
}

TEST(SampleSesTest, ThreeSpheresConcave) {
  const Spheres c = three_spheres();
  const Matrix3Xd &pts = c.pts;
  const ArrayXd &sar = c.sar;

  const Sampled s = solve(pts, sar, kRp, 15);
  ASSERT_GT(s.geo.area(SurfacePatch::kConcave).size(), 0);
  ASSERT_GT(s.dots.count(kPatch[2]), 0);
  expect_block_sums(s);
  expect_on_ses(s, pts, sar, kRp);

  const SurfaceDots &d = s.dots;
  for (int k = 0; k < d.n(); ++k) {
    EXPECT_GE(d.atom()[k], 0);
    EXPECT_LT(d.atom()[k], 3);
  }
  for (int k = d.offset(SurfacePatch::kConcave); k < d.n(); ++k) {
    const Vector3d probe = d.pts().col(k) + kRp * d.nrm().col(k);
    EXPECT_NEAR((probe - pts.col(d.atom()[k])).norm(), sar[d.atom()[k]], 1e-9)
        << "concave dot " << k << " owner " << d.atom()[k] << " is no contact";
  }
}

TEST(SampleSesTest, MaskedRunAndSubset) {
  const Spheres c = random_cluster();
  const int n = static_cast<int>(c.sar.size());

  const Sampled s = solve(c.pts, c.sar, c.active, kRp, 15);
  const SurfaceDots &d = s.dots;
  expect_block_sums_with_drops(s);
  ASSERT_GT(d.count(kPatch[0]), 0);
  for (int k = 0; k < d.count(SurfacePatch::kConvex); ++k)
    EXPECT_TRUE(c.active[d.atom()[k]]) << "convex dot " << k;
  for (int k = 0; k < d.n(); ++k) {
    EXPECT_GE(d.atom()[k], 0);
    EXPECT_LT(d.atom()[k], n);
  }

  expect_keep_subset(d, every_other(d.n()));
}

TEST(SampleSesTest, SasDots) {
  const Spheres c = random_cluster();
  const ArrayXd r = c.sar - kRp;
  AnalyticSurface surf(c.pts, r);
  ASSERT_TRUE(surf.initialize(kRp, c.active));
  const internal::SurfaceGeometry geo = surf.build_sas();
  const SurfaceDots d = SurfaceSampler(15).sample(geo);

  EXPECT_EQ(d.kind(), SurfaceKind::kSas);
  EXPECT_EQ(d.rp(), kRp);
  ASSERT_GT(d.n(), 0);
  EXPECT_EQ(d.count(SurfacePatch::kConvex), d.n());
  EXPECT_EQ(d.count(SurfacePatch::kToroidal), 0);
  EXPECT_EQ(d.count(SurfacePatch::kConcave), 0);
  EXPECT_EQ(d.offset(SurfacePatch::kConvex), 0);
  EXPECT_EQ(d.offset(SurfacePatch::kToroidal), d.n());
  EXPECT_EQ(d.offset(SurfacePatch::kConcave), d.n());
  EXPECT_NEAR(d.area().sum() + d.dropped_area(), geo.total_area(),
              1e-9 * geo.total_area());
  EXPECT_EQ(d.probes(), d.pts());

  for (int k = 0; k < d.n(); ++k) {
    const int a = d.atom()[k];
    ASSERT_GE(a, 0);
    ASSERT_LT(a, c.sar.size());
    EXPECT_TRUE(c.active[a]) << "dot " << k;
    const Vector3d radial = d.pts().col(k) - c.pts.col(a);
    EXPECT_NEAR(radial.norm(), c.sar[a], 1e-12) << "dot " << k;
    EXPECT_NEAR((d.nrm().col(k) - radial / c.sar[a]).norm(), 0, 1e-12)
        << "dot " << k;
  }

  expect_keep_subset(d, every_other(d.n()));
  const SurfaceDots none =
      expect_keep_subset(d, ArrayXb::Constant(d.n(), false));
  EXPECT_EQ(none.n(), 0);
  for (const SurfacePatch p: kPatch) {
    EXPECT_EQ(none.offset(p), 0);
    EXPECT_EQ(none.count(p), 0);
  }
}

TEST(SampleSesTest, ProteinDensity) {
  const double rp = 1.7, density = 15;
  const Protein prot = heavy_atoms("1brs.pdb");
  ASSERT_GT(prot.pts.cols(), 500);
  const ArrayXd sar = prot.radii + rp;

  const Sampled s = solve(prot.pts, sar, rp, density);
  expect_block_sums_with_drops(s);
  EXPECT_LT(s.dots.dropped_area(), 1e-3 * analytic_areas(s).sum());
  for (int b = 0; b < 3; ++b) {
    const double area = block_area(s.dots, b);
    ASSERT_GT(area, 0) << "block " << b;
    EXPECT_NEAR(s.dots.count(kPatch[b]) / area, density, 0.03 * density)
        << "block " << b;
  }
}

void expect_same_dots(const SurfaceDots &lhs, const SurfaceDots &rhs) {
  ASSERT_EQ(lhs.n(), rhs.n());
  EXPECT_EQ(lhs.pts(), rhs.pts());
  EXPECT_EQ(lhs.nrm(), rhs.nrm());
  EXPECT_TRUE((lhs.area() == rhs.area()).all());
  EXPECT_TRUE((lhs.atom() == rhs.atom()).all());
  for (const SurfacePatch p: kPatch)
    EXPECT_EQ(lhs.offset(p), rhs.offset(p));
}

TEST(SampleSesTest, SamplerReuse) {
  const double rp = 1.7, density = 15;
  Protein prot = heavy_atoms("1brs.pdb");
  ASSERT_GT(prot.pts.cols(), 500);
  const Spheres protein { std::move(prot.pts), prot.radii + rp,
                          ArrayXb::Constant(prot.radii.size(), true) };
  const internal::SurfaceGeometry big = seeded_ses(protein, rp);
  const internal::SurfaceGeometry small = seeded_ses(three_spheres(), kRp);

  SurfaceSampler sampler(density);
  const SurfaceDots first = sampler.sample(small);
  const SurfaceDots second = sampler.sample(small);
  expect_same_dots(first, second);

  const SurfaceDots grown = sampler.sample(big);
  ASSERT_GT(grown.n(), first.n());
  expect_same_dots(sampler.sample(small),
                   SurfaceSampler(density).sample(small));
}
}  // namespace
}  // namespace internal
}  // namespace nuri

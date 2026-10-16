//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "desc_test_common.h"
#include "nuri/desc/surface.h"
#include "nuri/fmt/pdb.h"
#include "nuri/random.h"

namespace nuri {
namespace {
constexpr double kRadius = 1.7;

Matrix3Xd slab(const double z, const double shift = 0, const int n = 7,
               const double spacing = 3.0) {
  Matrix3Xd pts(3, n * n);
  for (int i = 0, w = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j, ++w) {
      pts.col(w) << (i - (n - 1) / 2.0) * spacing + shift,
          (j - (n - 1) / 2.0) * spacing + shift, z;
    }
  }
  return pts;
}

TEST(ShapeComplementarityTest, FacingSlabs) {
  const Matrix3Xd a = slab(0.0), b = slab(3.4, 1.5), b_far = slab(4.4, 1.5);
  const ArrayXd radii = ArrayXd::Constant(a.cols(), kRadius);

  const std::optional<ScResult> tight =
      shape_complementarity(a, radii, b, radii);
  ASSERT_TRUE(tight);
  EXPECT_GT(tight->sc, 0.6);
  EXPECT_LE(tight->sc, 0.999);
  EXPECT_LT(tight->distance, 1.0);
  EXPECT_GT(tight->area, 0.0);
  for (const ScSide &s: tight->sides) {
    EXPECT_EQ(s.n_atoms, 49);
    EXPECT_EQ(s.n_active, 49);
    EXPECT_GT(s.n_dots, s.n_buried);
    EXPECT_GE(s.n_buried, s.n_trimmed);
    EXPECT_GT(s.n_trimmed, 0);
    EXPECT_GT(s.trimmed_area, 0.0);
  }

  const std::optional<ScResult> loose =
      shape_complementarity(a, radii, b_far, radii);
  ASSERT_TRUE(loose);
  EXPECT_LT(loose->sc, tight->sc);
  EXPECT_GT(loose->distance, tight->distance);
}

TEST(ShapeComplementarityTest, FarPairHasNoInterface) {
  Matrix3Xd a = Matrix3Xd::Zero(3, 1), b = Matrix3Xd::Zero(3, 1);
  b(0, 0) = 30.0;
  const ArrayXd radii = ArrayXd::Constant(1, kRadius);

  EXPECT_FALSE(shape_complementarity(a, radii, b, radii));
}

void expect_same_side(const ScSide &lhs, const ScSide &rhs) {
  EXPECT_EQ(lhs.n_atoms, rhs.n_atoms);
  EXPECT_EQ(lhs.n_active, rhs.n_active);
  EXPECT_EQ(lhs.n_dots, rhs.n_dots);
  EXPECT_EQ(lhs.n_buried, rhs.n_buried);
  EXPECT_EQ(lhs.n_trimmed, rhs.n_trimmed);
  EXPECT_DOUBLE_EQ(lhs.trimmed_area, rhs.trimmed_area);
  EXPECT_DOUBLE_EQ(lhs.d_median, rhs.d_median);
  EXPECT_DOUBLE_EQ(lhs.s_median, rhs.s_median);
}

TEST(ShapeComplementarityTest, SwapIsSymmetric) {
  const Matrix3Xd a = slab(0.0), b = slab(3.4, 1.5);
  const ArrayXd radii = ArrayXd::Constant(a.cols(), kRadius);

  internal::seed_thread(42);
  const std::optional<ScResult> ab = shape_complementarity(a, radii, b, radii);
  internal::seed_thread(42);
  const std::optional<ScResult> ba = shape_complementarity(b, radii, a, radii);
  ASSERT_TRUE(ab);
  ASSERT_TRUE(ba);

  EXPECT_DOUBLE_EQ(ab->sc, ba->sc);
  EXPECT_DOUBLE_EQ(ab->distance, ba->distance);
  EXPECT_DOUBLE_EQ(ab->area, ba->area);
  expect_same_side(ab->sides[0], ba->sides[1]);
  expect_same_side(ab->sides[1], ba->sides[0]);
}

void expect_same_result(const ScResult &lhs, const ScResult &rhs) {
  EXPECT_DOUBLE_EQ(lhs.sc, rhs.sc);
  EXPECT_DOUBLE_EQ(lhs.distance, rhs.distance);
  EXPECT_DOUBLE_EQ(lhs.area, rhs.area);
  for (int i = 0; i < 2; ++i) {
    const ScSide &l = lhs.sides[i], &r = rhs.sides[i];
    expect_same_side(l, r);
    EXPECT_EQ(l.dots.pts(), r.dots.pts());
    EXPECT_EQ(l.dots.nrm(), r.dots.nrm());
    EXPECT_TRUE((l.dots.area() == r.dots.area()).all());
    EXPECT_TRUE((l.dots.atom() == r.dots.atom()).all());
    EXPECT_TRUE((l.distance == r.distance).all());
    EXPECT_TRUE((l.score == r.score).all());
  }
}

TEST(ShapeComplementarityTest, ClassMatchesFreeFunction) {
  const Matrix3Xd a = slab(0.0), b = slab(3.4, 1.5);
  const ArrayXd radii = ArrayXd::Constant(a.cols(), kRadius);

  ScParams params;
  params.rp = 1.4;
  params.density = 10;
  params.weight = 0.4;
  params.band = 1.0;
  params.sep = 6;
  params.clamp = 0.9;

  ShapeComplementarity sc(params);
  EXPECT_DOUBLE_EQ(sc.params().rp, params.rp);
  EXPECT_DOUBLE_EQ(sc.params().density, params.density);
  EXPECT_DOUBLE_EQ(sc.params().weight, params.weight);
  EXPECT_DOUBLE_EQ(sc.params().band, params.band);
  EXPECT_DOUBLE_EQ(sc.params().sep, params.sep);
  EXPECT_DOUBLE_EQ(sc.params().clamp, params.clamp);

  internal::seed_thread(7);
  const std::optional<ScResult> first = sc.compute(a, radii, b, radii);
  internal::seed_thread(7);
  const std::optional<ScResult> direct =
      shape_complementarity(a, radii, b, radii, params);
  internal::seed_thread(7);
  const std::optional<ScResult> second = sc.compute(a, radii, b, radii);
  ASSERT_TRUE(first);
  ASSERT_TRUE(direct);
  ASSERT_TRUE(second);

  expect_same_result(*first, *direct);
  expect_same_result(*first, *second);
}

TEST(ShapeComplementarityTest, FullMasksMatchUnmasked) {
  const Matrix3Xd a = slab(0.0), b = slab(3.4, 1.5);
  const ArrayXd radii = ArrayXd::Constant(a.cols(), kRadius);
  const ArrayXb all = ArrayXb::Constant(a.cols(), true);

  internal::seed_thread(3);
  const std::optional<ScResult> plain =
      shape_complementarity(a, radii, b, radii);
  internal::seed_thread(3);
  const std::optional<ScResult> masked =
      shape_complementarity(a, radii, all, b, radii, all);
  ASSERT_TRUE(plain);
  ASSERT_TRUE(masked);
  expect_same_result(*plain, *masked);
}

TEST(ShapeComplementarityTest, MaskedAtomsOnlyOcclude) {
  const Matrix3Xd a = slab(0.0), b = slab(3.4, 1.5), extra = slab(10.9, 1.5);
  Matrix3Xd a_ext(3, a.cols() + extra.cols());
  a_ext << a, extra;
  const ArrayXd radii = ArrayXd::Constant(a.cols(), kRadius),
                radii_ext = ArrayXd::Constant(a_ext.cols(), kRadius);
  ArrayXb mask = ArrayXb::Constant(a_ext.cols(), true);
  mask.tail(extra.cols()).setConstant(false);

  internal::seed_thread(5);
  const std::optional<ScResult> plain =
      shape_complementarity(a, radii, b, radii);
  internal::seed_thread(5);
  const std::optional<ScResult> masked = shape_complementarity(
      a_ext, radii_ext, mask, b, radii, ArrayXb::Constant(b.cols(), true));
  internal::seed_thread(5);
  const std::optional<ScResult> unmasked =
      shape_complementarity(a_ext, radii_ext, b, radii);
  ASSERT_TRUE(plain);
  ASSERT_TRUE(masked);
  ASSERT_TRUE(unmasked);

  EXPECT_DOUBLE_EQ(masked->sc, plain->sc);
  EXPECT_DOUBLE_EQ(masked->distance, plain->distance);
  EXPECT_DOUBLE_EQ(masked->area, plain->area);
  EXPECT_EQ(masked->sides[0].n_atoms, a_ext.cols());
  EXPECT_EQ(masked->sides[0].n_active, plain->sides[0].n_active);
  EXPECT_EQ(masked->sides[0].n_dots, plain->sides[0].n_dots);
  EXPECT_EQ(masked->sides[0].n_trimmed, plain->sides[0].n_trimmed);
  expect_same_side(masked->sides[1], plain->sides[1]);

  EXPECT_EQ(unmasked->sides[0].n_active, a_ext.cols());
  EXPECT_GT(unmasked->sides[0].n_dots, masked->sides[0].n_dots);
  EXPECT_DOUBLE_EQ(unmasked->sc, masked->sc);

  const std::optional<ScResult> none =
      shape_complementarity(a, radii, ArrayXb::Constant(a.cols(), false), b,
                            radii, ArrayXb::Constant(b.cols(), true));
  EXPECT_FALSE(none);
}

double median(const ArrayXd &arr) {
  std::vector<double> v(arr.begin(), arr.end());
  std::sort(v.begin(), v.end());
  const auto mid = v.size() / 2;
  if (v.size() % 2 == 1)
    return v[mid];
  return 0.5 * (v[mid - 1] + v[mid]);
}

std::optional<ScResult> seeded_slabs(const ScParams &params = {}) {
  const Matrix3Xd a = slab(0.0), b = slab(3.4, 1.5);
  const ArrayXd radii = ArrayXd::Constant(a.cols(), kRadius);
  internal::seed_thread(0);
  return shape_complementarity(a, radii, b, radii, params);
}

TEST(ShapeComplementarityTest, SideArrays) {
  const ScParams params;
  const std::optional<ScResult> res = seeded_slabs(params);
  ASSERT_TRUE(res);

  for (const ScSide &s: res->sides) {
    const SurfaceDots &dots = s.dots;
    EXPECT_EQ(dots.n(), s.n_trimmed);
    EXPECT_EQ(s.distance.size(), s.n_trimmed);
    EXPECT_EQ(s.score.size(), s.n_trimmed);
    EXPECT_NEAR(s.trimmed_area, dots.area().sum(), 1e-9);
    EXPECT_EQ(dots.kind(), SurfaceKind::kSes);
    EXPECT_DOUBLE_EQ(dots.rp(), params.rp);
    EXPECT_EQ(dots.count(SurfacePatch::kConvex)
                  + dots.count(SurfacePatch::kToroidal)
                  + dots.count(SurfacePatch::kConcave),
              dots.n());
    EXPECT_TRUE((s.score.abs() <= params.clamp).all());
    EXPECT_TRUE((s.distance > 0).all());
    EXPECT_DOUBLE_EQ(s.d_median, median(s.distance));
    EXPECT_DOUBLE_EQ(s.s_median, median(s.score));
  }

  EXPECT_DOUBLE_EQ(res->area,
                   res->sides[0].trimmed_area + res->sides[1].trimmed_area);
  EXPECT_DOUBLE_EQ(res->sc,
                   0.5 * (res->sides[0].s_median + res->sides[1].s_median));
  EXPECT_DOUBLE_EQ(res->distance,
                   0.5 * (res->sides[0].d_median + res->sides[1].d_median));
}

TEST(ShapeComplementarityTest, ParamsMatter) {
  const std::optional<ScResult> base = seeded_slabs();
  ASSERT_TRUE(base);

  const Matrix3Xd wide = slab(0.0, 0, 9), b = slab(3.4, 1.5);
  const ArrayXd wide_radii = ArrayXd::Constant(wide.cols(), kRadius),
                radii = ArrayXd::Constant(b.cols(), kRadius);
  ScParams params;
  internal::seed_thread(0);
  const std::optional<ScResult> sep_default =
      shape_complementarity(wide, wide_radii, b, radii, params);
  params.sep = 4.5;
  internal::seed_thread(0);
  const std::optional<ScResult> sep_short =
      shape_complementarity(wide, wide_radii, b, radii, params);
  ASSERT_TRUE(sep_default);
  ASSERT_TRUE(sep_short);
  EXPECT_EQ(sep_default->sides[0].n_active, 81);
  EXPECT_EQ(sep_default->sides[1].n_active, 49);
  EXPECT_LT(sep_short->sides[0].n_active, 81);
  EXPECT_GT(sep_short->sides[0].n_active, 0);
  EXPECT_EQ(sep_short->sides[1].n_active, 49);

  params = {};
  params.density = 30;
  const std::optional<ScResult> dense = seeded_slabs(params);
  ASSERT_TRUE(dense);
  for (int i = 0; i < 2; ++i) {
    const double expected = 2.0 * base->sides[i].n_dots;
    EXPECT_NEAR(dense->sides[i].n_dots, expected, 0.1 * expected);
  }

  params = {};
  params.band = 0.1;
  const std::optional<ScResult> untrimmed = seeded_slabs(params);
  ASSERT_TRUE(untrimmed);
  for (const ScSide &s: untrimmed->sides)
    EXPECT_EQ(s.n_trimmed, s.n_buried);

  params = {};
  params.clamp = 0.5;
  const std::optional<ScResult> clamped = seeded_slabs(params);
  ASSERT_TRUE(clamped);
  for (const ScSide &s: clamped->sides) {
    EXPECT_LE(s.s_median, 0.5);
    EXPECT_TRUE((s.score.abs() <= 0.5).all());
  }
  EXPECT_LT(clamped->sc, base->sc);
}

/**
 * Heavy atoms with element vdW radii, seeded. The values are pinned from
 * this implementation's optdebug build at 1e-6; reldeb agrees.
 */
TEST(ShapeComplementarityTest, BarnaseBarstar) {
  const internal::Protein a = internal::heavy_atoms("1brs.pdb", 'A'),
                          b = internal::heavy_atoms("1brs.pdb", 'D');
  ASSERT_GT(a.pts.cols(), 500);
  ASSERT_GT(b.pts.cols(), 500);

  internal::seed_thread(0);
  const std::optional<ScResult> res =
      shape_complementarity(a.pts, a.radii, b.pts, b.radii);
  ASSERT_TRUE(res);
  EXPECT_NEAR(res->sc, 0.660846137477, 1e-6);
  EXPECT_NEAR(res->distance, 0.639743237709, 1e-6);
  EXPECT_NEAR(res->area, 906.024644525, 1e-6);

  EXPECT_EQ(res->sides[0].n_atoms, a.pts.cols());
  EXPECT_EQ(res->sides[1].n_atoms, b.pts.cols());
  for (const ScSide &s: res->sides) {
    EXPECT_GT(s.n_active, 0);
    EXPECT_LT(s.n_active, s.n_atoms);
    EXPECT_GT(s.n_dots, 0);
    EXPECT_GT(s.n_buried, 0);
    EXPECT_GT(s.n_trimmed, 0);
    EXPECT_GT(s.trimmed_area, 0.0);
    EXPECT_GT(s.d_median, 0.0);
    EXPECT_GT(s.s_median, 0.0);
    EXPECT_LE(s.s_median, 0.999);
  }
}
}  // namespace
}  // namespace nuri

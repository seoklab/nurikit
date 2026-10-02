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
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/fmt/pdb.h"

namespace nuri {
namespace internal {
namespace {
using constants::kTwoPi;

constexpr double kRp = 1.4;

struct Sampled {
  SaPrep sa;
  SasGeometry geo;
  SesGeometry ses;
  SesDots dots;
};

Sampled solve(const Matrix3Xd &pts, const ArrayXd &sar, const ArrayXb &active,
              const double rp, const double density) {
  std::optional<SaPrep> sa = prepare(pts, sar, active, rp);
  EXPECT_TRUE(sa);
  SasDelaunay del = triangulate(*sa);
  SasGeometry geo = build_sas(*sa, del);
  SesGeometry ses = build_ses(*sa, del, geo, rp);
  SesDots dots = sample_ses(*sa, geo, ses, density);
  return { std::move(*sa), std::move(geo), std::move(ses), std::move(dots) };
}

Sampled solve(const Matrix3Xd &pts, const ArrayXd &sar, const double rp,
              const double density) {
  return solve(pts, sar, ArrayXb::Constant(sar.size(), true), rp, density);
}

double block_area(const SesDots &dots, const int kind) {
  return dots.area.segment(dots.kind[kind], dots.kind.degree(kind)).sum();
}

Array3d analytic_areas(const Sampled &s) {
  return { s.ses.convex_area.sum(), s.ses.saddle_area.sum(),
           s.ses.face_area.head(s.geo.probes.n_active).sum() };
}

/**
 * Nothing dropped: every block carries exactly its analytic area.
 */
void expect_block_sums(const Sampled &s) {
  const Array3d ref = analytic_areas(s);
  EXPECT_EQ(s.dots.kind.size(), 3);
  EXPECT_EQ(s.dots.kind[3], s.dots.n());
  EXPECT_EQ(s.dots.dropped_area, 0);
  for (int b = 0; b < 3; ++b)
    EXPECT_NEAR(block_area(s.dots, b), ref[b], 1e-9) << "block " << b;
}

/**
 * Patches may drop: no block exceeds its analytic area and the drops make
 * up the difference.
 */
void expect_block_sums_with_drops(const Sampled &s) {
  const Array3d ref = analytic_areas(s);
  EXPECT_EQ(s.dots.kind.size(), 3);
  EXPECT_EQ(s.dots.kind[3], s.dots.n());
  EXPECT_GE(s.dots.dropped_area, 0);
  double sampled = 0;
  for (int b = 0; b < 3; ++b) {
    sampled += block_area(s.dots, b);
    EXPECT_LE(block_area(s.dots, b), ref[b] + 1e-9) << "block " << b;
  }
  EXPECT_NEAR(sampled + s.dots.dropped_area, ref.sum(), 1e-9 * ref.sum());
}

/**
 * Every probe centre `pts + rp·nrm` is outside every SAS sphere, and every
 * normal is unit.
 */
void expect_on_ses(const Sampled &s, const Matrix3Xd &pts, const ArrayXd &sar,
                   const double rp) {
  const SesDots &d = s.dots;
  for (int k = 0; k < d.n(); ++k) {
    EXPECT_NEAR(d.nrm.col(k).norm(), 1, 1e-12) << "dot " << k;
    const Vector3d probe = d.pts.col(k) + rp * d.nrm.col(k);
    for (int a = 0; a < pts.cols(); ++a) {
      EXPECT_GE((probe - pts.col(a)).norm(), sar[a] - 1e-9)
          << "dot " << k << " atom " << a;
    }
  }
}

TEST(SampleSesTest, SingleSphere) {
  const double r = 1.5, rp = 1.7, density = 15;
  const Matrix3Xd pts = Matrix3Xd::Zero(3, 1);
  const ArrayXd sar = ArrayXd::Constant(1, r + rp);

  const Sampled s = solve(pts, sar, rp, density);
  const SesDots &d = s.dots;
  const double sphere = 2 * kTwoPi * r * r;
  EXPECT_NEAR(d.area.sum(), sphere, 1e-9);
  EXPECT_EQ(d.n(), static_cast<int>(std::lround(sphere * density)));
  EXPECT_EQ(d.kind.degree(0), d.n());
  EXPECT_EQ(d.kind.degree(1), 0);
  EXPECT_EQ(d.kind.degree(2), 0);
  EXPECT_EQ(d.rp, rp);

  for (int k = 0; k < d.n(); ++k) {
    EXPECT_NEAR(d.pts.col(k).norm(), r, 1e-12);
    EXPECT_NEAR((d.nrm.col(k) - d.pts.col(k) / r).norm(), 0, 1e-12);
    EXPECT_EQ(d.atom[k], 0);
  }
}

TEST(SampleSesTest, TwoSpheresStammTable1) {
  const double rp = 1.2;
  Matrix3Xd pts(3, 2);
  pts << 0, 2, 0, 0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(2, 1.2 + rp);

  const Sampled s = solve(pts, sar, rp, 200);
  expect_block_sums(s);
  EXPECT_EQ(s.dots.kind.degree(2), 0);
  EXPECT_NEAR(block_area(s.dots, 0) + block_area(s.dots, 1), 32.23514, 1e-4);
}

TEST(SampleSesTest, TwoSpheresCusp) {
  const double rp = 1.2;
  Matrix3Xd pts(3, 2);
  pts << 0, 4.4, 0, 0, 0, 0;
  const ArrayXd sar = ArrayXd::Constant(2, 1.2 + rp);

  const Sampled s = solve(pts, sar, rp, 200);
  ASSERT_LT(s.geo.circles[0].rl, rp);
  expect_block_sums(s);
  EXPECT_GT(s.dots.kind.degree(1), 0);
  expect_on_ses(s, pts, sar, rp);
}

TEST(SampleSesTest, ThreeSpheresConcave) {
  Matrix3Xd pts(3, 3);
  pts << 0, 4.8, 2.6,  //
      0, 0.2, 4.16,    //
      0, 0.3, -0.4;
  ArrayXd sar(3);
  sar << 1.5 + kRp, 1.9 + kRp, 1.7 + kRp;

  const Sampled s = solve(pts, sar, kRp, 15);
  ASSERT_GT(s.geo.probes.n_active, 0);
  ASSERT_GT(s.dots.kind.degree(2), 0);
  expect_block_sums(s);
  expect_on_ses(s, pts, sar, kRp);

  const SesDots &d = s.dots;
  for (int k = 0; k < d.n(); ++k) {
    EXPECT_GE(d.atom[k], 0);
    EXPECT_LT(d.atom[k], 3);
  }
  for (int k = d.kind[2]; k < d.kind[3]; ++k) {
    const Vector3d probe = d.pts.col(k) + kRp * d.nrm.col(k);
    EXPECT_NEAR((probe - pts.col(d.atom[k])).norm(), sar[d.atom[k]], 1e-9)
        << "concave dot " << k << " owner " << d.atom[k] << " is no contact";
  }
}

TEST(SampleSesTest, MaskedRunAndSubset) {
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

  const Sampled s = solve(pts, sar, active, kRp, 15);
  const SesDots &d = s.dots;
  expect_block_sums_with_drops(s);
  ASSERT_GT(d.kind.degree(0), 0);
  for (int k = 0; k < d.kind[1]; ++k)
    EXPECT_TRUE(active[d.atom[k]]) << "convex dot " << k;
  for (int k = 0; k < d.n(); ++k) {
    EXPECT_GE(d.atom[k], 0);
    EXPECT_LT(d.atom[k], n);
  }

  ArrayXb keep(d.n());
  for (int k = 0; k < d.n(); ++k)
    keep[k] = k % 2 == 0;
  const SesDots sub = d.subset(keep);
  ASSERT_EQ(sub.n(), keep.count());
  EXPECT_EQ(sub.rp, d.rp);
  EXPECT_EQ(sub.dropped_area, d.dropped_area);
  ASSERT_EQ(sub.kind.size(), 3);
  EXPECT_EQ(sub.kind[0], 0);
  EXPECT_EQ(sub.kind[3], sub.n());

  int w = 0;
  for (int b = 0; b < 3; ++b) {
    int kept = 0;
    for (int k = d.kind[b]; k < d.kind[b + 1]; ++k) {
      if (!keep[k])
        continue;
      ++kept;
      EXPECT_EQ(sub.pts.col(w), d.pts.col(k));
      EXPECT_EQ(sub.nrm.col(w), d.nrm.col(k));
      EXPECT_EQ(sub.area[w], d.area[k]);
      EXPECT_EQ(sub.atom[w], d.atom[k]);
      ++w;
    }
    EXPECT_EQ(sub.kind.degree(b), kept) << "block " << b;
  }
}

struct Protein {
  Matrix3Xd pts;
  ArrayXd radii;
};

Protein heavy_atoms(const char *name) {
  std::ifstream ifs(std::filesystem::path(NURI_TEST_DATA_DIR) / name);
  EXPECT_TRUE(ifs) << name;
  PDBReader reader(ifs);
  PDBRecord record;
  EXPECT_TRUE(reader.getnext(record));
  ParseResult<PDBModel> model = read_pdb_model(record.text());
  EXPECT_TRUE(model) << model.error_msg();

  const auto &atoms = model->atoms();
  const int n = static_cast<int>(atoms.size());
  Protein out { Matrix3Xd(3, n), ArrayXd(n) };
  int w = 0;
  for (int i = 0; i < n; ++i) {
    const PDBAtom &a = atoms[i];
    if (a.hetero() || a.element().atomic_number() <= 1)
      continue;
    out.pts.col(w) = model->major_conf().col(i);
    out.radii[w] = a.element().vdw_radius();
    ++w;
  }
  out.pts.conservativeResize(3, w);
  out.radii.conservativeResize(w);
  return out;
}

TEST(SampleSesTest, ProteinDensity) {
  const double rp = 1.7, density = 15;
  const Protein prot = heavy_atoms("1brs.pdb");
  ASSERT_GT(prot.pts.cols(), 500);
  const ArrayXd sar = prot.radii + rp;

  const Sampled s = solve(prot.pts, sar, rp, density);
  expect_block_sums_with_drops(s);
  EXPECT_LT(s.dots.dropped_area, 1e-3 * analytic_areas(s).sum());
  for (int b = 0; b < 3; ++b) {
    const double area = block_area(s.dots, b);
    ASSERT_GT(area, 0) << "block " << b;
    EXPECT_NEAR(s.dots.kind.degree(b) / area, density, 0.03 * density)
        << "block " << b;
  }
}
}  // namespace
}  // namespace internal
}  // namespace nuri

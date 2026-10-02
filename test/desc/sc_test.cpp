//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <fstream>
#include <optional>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
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

struct Protein {
  Matrix3Xd pts;
  ArrayXd radii;
};

Protein heavy_atoms(const char *name, const char chain) {
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
    if (a.hetero() || a.element().atomic_number() <= 1
        || a.rid().chain_id != chain)
      continue;
    out.pts.col(w) = model->major_conf().col(i);
    out.radii[w] = a.element().vdw_radius();
    ++w;
  }
  out.pts.conservativeResize(3, w);
  out.radii.conservativeResize(w);
  return out;
}

TEST(ShapeComplementarityTest, BarnaseBarstar) {
  const Protein a = heavy_atoms("1brs.pdb", 'A'),
                b = heavy_atoms("1brs.pdb", 'D');
  ASSERT_GT(a.pts.cols(), 500);
  ASSERT_GT(b.pts.cols(), 500);

  const std::optional<ScResult> res =
      shape_complementarity(a.pts, a.radii, b.pts, b.radii);
  ASSERT_TRUE(res);
  EXPECT_GT(res->sc, 0.0);
  EXPECT_LT(res->sc, 1.0);
  EXPECT_GT(res->distance, 0.0);
  EXPECT_LT(res->distance, 3.0);
  EXPECT_GT(res->area, 100.0);

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

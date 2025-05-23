//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/tools/galign.h"

#include <initializer_list>
#include <vector>

#include <absl/algorithm/container.h>

#include <gtest/gtest.h>

#include "nuri/core/molecule.h"
#include "nuri/fmt/smiles.h"

namespace nuri {
namespace {
testing::AssertionResult
verify_rotation_info(const internal::GARotationInfo &ri, int origin,
                     const std::initializer_list<int> &expected_moving) {
  if (ri.origin != origin) {
    return testing::AssertionFailure()
           << "origin mismatch: " << ri.origin << " != " << origin;
  }

  if (!absl::c_equal(ri.moving, expected_moving)) {
    return testing::AssertionFailure()
           << "moving atoms: " << ri.moving.transpose();
  }

  return testing::AssertionSuccess();
}

TEST(GAlign, ModelAsTree) {
  Molecule mol = read_smiles({
      "O=C(C1CCC(CC(C)OC)CC1)c2cc3c(CC)ccc(C(C)C)c3cc2.Cl",
  });

  std::vector ri = internal::ga_resolve_rotation_info(mol);
  ASSERT_EQ(ri.size(), 7);

  for (auto &r: ri) {
    auto bit = mol.find_bond(r.pivot, r.origin);
    ASSERT_NE(bit, mol.bond_end());

    EXPECT_TRUE(bit->data().is_rotatable());

    switch (r.pivot) {
    case 2:
      EXPECT_TRUE(
          verify_rotation_info(r, 1, { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
      break;
    case 1:
      EXPECT_TRUE(verify_rotation_info(
          r, 13, { 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
      break;
    case 6:
      EXPECT_TRUE(verify_rotation_info(r, 5, { 7, 8, 9, 10 }));
      break;
    case 17:
      EXPECT_TRUE(verify_rotation_info(r, 16, { 18 }));
      break;
    case 7:
      EXPECT_TRUE(verify_rotation_info(r, 6, { 8, 9, 10 }));
      break;
    case 22:
      EXPECT_TRUE(verify_rotation_info(r, 21, { 23, 24 }));
      break;
    case 9:
      EXPECT_TRUE(verify_rotation_info(r, 7, { 10 }));
      break;
    default:
      FAIL() << "Unexpected pivot/origin pair: " << r.pivot << ", " << r.origin;
    }
  }
}
}  // namespace
}  // namespace nuri

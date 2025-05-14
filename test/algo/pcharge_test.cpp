//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/pcharge.h"

#include <gtest/gtest.h>

#include "nuri/core/molecule.h"
#include "nuri/fmt/smiles.h"

namespace nuri {
namespace {
// From the original Gasteiger paper
TEST(ChargeGasteiger, Ethane) {
  Molecule mol = read_smiles({ "CC" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  ASSERT_TRUE(assign_charges_gasteiger(mol));

  EXPECT_NEAR(mol[0].data().partial_charge(), -0.068, 5e-3);
  EXPECT_NEAR(mol[1].data().partial_charge(), -0.068, 5e-3);
}
}  // namespace
}  // namespace nuri

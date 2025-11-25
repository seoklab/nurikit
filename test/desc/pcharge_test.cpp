//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/desc/pcharge.h"

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

  EXPECT_EQ(internal::get_key(mol.props(), "mol2_charge_type"), "GASTEIGER");
}

TEST(ChargeGasteiger, AcetateIon) {
  Molecule mol = read_smiles({ "CC(=O)[O-]" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  ASSERT_TRUE(assign_charges_gasteiger(mol));

  EXPECT_NEAR(mol[2].data().partial_charge(), mol[3].data().partial_charge(),
              1e-6);
}

TEST(ChargeGasteiger, GuanidiniumChloride) {
  Molecule mol = read_smiles({ "NC(=[NH2+])N.[Cl-]" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  ASSERT_TRUE(assign_charges_gasteiger(mol));

  EXPECT_NEAR(mol[0].data().partial_charge(), mol[2].data().partial_charge(),
              1e-6);
  EXPECT_NEAR(mol[0].data().partial_charge(), mol[3].data().partial_charge(),
              1e-6);
  EXPECT_NEAR(mol[4].data().partial_charge(), -1, 1e-6);
}
}  // namespace
}  // namespace nuri

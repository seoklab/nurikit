//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/mmcif.h"

#include <vector>

#include <gtest/gtest.h>

#include "fmt_test_common.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/cif.h"

namespace nuri {
namespace {
struct MmcifTest: ::testing::Test {
  // NOLINTBEGIN(readability-identifier-naming)
  std::ifstream ifs_;
  // NOLINTEND(readability-identifier-naming)

  void set_test_file(std::string_view name) {
    ifs_.open(internal::test_data(name));
    ASSERT_TRUE(ifs_) << "Failed to open file: " << name;
  }
};

TEST_F(MmcifTest, BasicParsing) {
  set_test_file("1a8o.cif");

  CifParser parser(ifs_);
  std::vector mols = mmcif_read_next_block(parser);
  ASSERT_EQ(mols.size(), 1);

  const Molecule &mol = mols[0];
  EXPECT_EQ(mol.name(), "1A8O");
  EXPECT_EQ(mol.num_atoms(), 644);
  EXPECT_EQ(mol.num_bonds(), 7);

  // 70 residues + 88 water molecules + 1 chain
  ASSERT_EQ(mol.substructures().size(), 70 + 88 + 1);

  EXPECT_EQ(mol.substructures()[0].name(), "MSE");
  EXPECT_EQ(mol.substructures()[0].id(), 151);
  EXPECT_EQ(mol.substructures()[0].size(), 8);
  EXPECT_TRUE(mol.substructures()[0].category() == SubstructCategory::kResidue);

  EXPECT_EQ(mol.substructures().back().name(), "A");
  EXPECT_TRUE(mol.substructures().back().category()
              == SubstructCategory::kChain);
}
}  // namespace
}  // namespace nuri

//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/mmcif.h"

#include <vector>

#include <boost/iterator/filter_iterator.hpp>

#include <gtest/gtest.h>

#include "fmt_test_common.h"
#include "test_utils.h"
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

TEST_F(MmcifTest, HandleMultipleModels) {
  set_test_file("3cye_part.cif");

  CifParser parser(ifs_);
  std::vector mols = mmcif_read_next_block(parser);
  ASSERT_EQ(mols.size(), 2);

  EXPECT_EQ(mols[0].name(), "3CYE");
  EXPECT_EQ(internal::get_key(mols[0].props(), "model"), "1");

  EXPECT_EQ(mols[0].num_atoms(), 67);
  EXPECT_EQ(mols[0].num_bonds(), 1);

  // Disulfide bond
  EXPECT_NE(mols[0].find_bond(28, 34), mols[0].bond_end());

  ASSERT_EQ(mols[0].confs().size(), 2);
  NURI_EXPECT_EIGEN_EQ(mols[0].confs()[0].col(0), mols[0].confs()[1].col(0));
  NURI_EXPECT_EIGEN_NE(mols[0].confs()[0].col(43), mols[0].confs()[1].col(43));

  // 9 residues + 1 chain
  ASSERT_EQ(mols[0].substructures().size(), 10);
  EXPECT_EQ(mols[0].substructures()[0].name(), "VAL");
  EXPECT_EQ(mols[0].substructures()[0].num_atoms(), 7);

  EXPECT_EQ(internal::get_key(mols[0].substructures()[8].props(), "icode"),
            "A");

  EXPECT_EQ(mols[1].name(), "3CYE");
  EXPECT_EQ(internal::get_key(mols[1].props(), "model"), "2");

  EXPECT_EQ(mols[1].num_atoms(), 36);
  EXPECT_EQ(mols[1].num_bonds(), 0);

  ASSERT_EQ(mols[1].confs().size(), 2);
  NURI_EXPECT_EIGEN_EQ(mols[1].confs()[0].col(0), mols[1].confs()[1].col(0));
  NURI_EXPECT_EIGEN_NE(mols[1].confs()[0].col(31), mols[1].confs()[1].col(31));

  // 5 residues + 1 chain
  ASSERT_EQ(mols[1].substructures().size(), 6);
  EXPECT_EQ(mols[1].substructures()[0].name(), "VAL");
  EXPECT_EQ(mols[1].substructures()[0].num_atoms(), 7);
}

class PDB1alxTest: public testing::Test {
protected:
  // NOLINTNEXTLINE(clang-diagnostic-unused-member-function)
  static void SetUpTestSuite() {
    std::ifstream ifs(internal::test_data("1alx.cif"));
    ASSERT_TRUE(ifs) << "Failed to open file: 1alx.cif";

    CifParser parser(ifs);
    std::vector<Molecule> mols = mmcif_read_next_block(parser);
    ASSERT_EQ(mols.size(), 1);

    mol_ = std::move(mols[0]);
    ASSERT_EQ(mol_.name(), "1ALX");
    ASSERT_EQ(mol_.num_atoms(), 669);
    ASSERT_EQ(mol_.num_bonds(), 28);
    ASSERT_EQ(mol_.substructures().size(), 54);
  }

  static const Molecule &mol() { return mol_; }

private:
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline static Molecule mol_;
};

TEST_F(PDB1alxTest, HandleMultipleAltlocs) {
  ASSERT_EQ(mol().confs().size(), 5);

  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(0), mol().confs()[1].col(0));
  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(0), mol().confs()[2].col(0));

  NURI_EXPECT_EIGEN_NE(mol().confs()[0].col(85), mol().confs()[1].col(85));
  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(85), mol().confs()[2].col(85));

  NURI_EXPECT_EIGEN_NE(mol().confs()[0].col(263), mol().confs()[1].col(263));
  NURI_EXPECT_EIGEN_NE(mol().confs()[1].col(263), mol().confs()[2].col(263));
}

TEST_F(PDB1alxTest, HandleInconsistentResidues) {
  const Substructure &res = mol().substructures()[10];
  EXPECT_EQ(res.name(), "TYR");
  EXPECT_EQ(res.id(), 11);
  EXPECT_EQ(res.num_atoms(), 21);
  EXPECT_EQ(res.count_heavy_atoms(), 12);

  auto find_11 = [](const Substructure &sub) { return sub.id() == 11; };
  std::vector<Substructure> res_11(
      boost::make_filter_iterator(find_11, mol().substructures().begin(),
                                  mol().substructures().end()),
      boost::make_filter_iterator(find_11, mol().substructures().end(),
                                  mol().substructures().end()));

  // A/B chain
  EXPECT_EQ(res_11.size(), 2);
  EXPECT_EQ(res_11[0].name(), "TYR");
}
}  // namespace
}  // namespace nuri

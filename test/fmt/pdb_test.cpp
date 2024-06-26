//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/pdb.h"

#include <fstream>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/strings/match.h>
#include <gtest/gtest.h>

#include "fmt_test_common.h"
#include "test_utils.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
namespace {
using PDBTest = internal::FileFormatTest<PDBReader>;

TEST_F(PDBTest, BasicParsing) {
  set_test_file("dummy.pdb");

  ASSERT_TRUE(advance());

  EXPECT_EQ(mol().num_atoms(), 88);
  EXPECT_EQ(mol().num_bonds(), 87);

  // Peptide + H2O
  EXPECT_EQ(mol().num_fragments(), 2);
  // Ring from PHE
  EXPECT_EQ(mol().ring_groups().size(), 1);

  // 11 residues + 1 chain
  EXPECT_EQ(mol().num_substructures(), 12);

  // First residue is MET
  const Substructure &sub = mol().get_substructure(0);
  EXPECT_EQ(sub.name(), "MET");
  EXPECT_EQ(sub.id(), 1);
  EXPECT_EQ(sub.num_atoms(), 8);

  // Phenylalanine N
  auto atom = mol().atom(25);
  EXPECT_EQ(atom.data().get_name(), "N");
  EXPECT_EQ(atom.data().atomic_number(), 7);
  EXPECT_EQ(atom.data().hybridization(), constants::kSP2);
  EXPECT_TRUE(atom.data().is_conjugated());
}

TEST_F(PDBTest, HandleInvalidCoordinates) {
  set_test_file("invalid.pdb");

  ASSERT_TRUE(advance());

  EXPECT_EQ(mol().num_atoms(), 90);
  EXPECT_EQ(mol().num_bonds(), 89);

  // Peptide + H2O
  EXPECT_EQ(mol().num_fragments(), 2);
  // Ring from PHE
  EXPECT_EQ(mol().ring_groups().size(), 1);

  // 11 residues + 1 chain
  EXPECT_EQ(mol().num_substructures(), 12);

  // First residue is MET
  const Substructure &sub = mol().get_substructure(0);
  EXPECT_EQ(sub.name(), "MET");
  EXPECT_EQ(sub.id(), 1);
  EXPECT_EQ(sub.num_atoms(), 8);

  // Phenylalanine N
  auto atom = mol().atom(25);
  EXPECT_EQ(atom.data().get_name(), "N");
  EXPECT_EQ(atom.data().atomic_number(), 7);
  EXPECT_EQ(atom.data().hybridization(), constants::kSP2);
  EXPECT_TRUE(atom.data().is_conjugated());

  EXPECT_NE(mol().find_bond(87, 88), mol().bond_end());
  EXPECT_NE(mol().find_bond(87, 89), mol().bond_end());
}

TEST_F(PDBTest, HandleCleanPDB) {
  set_test_file("1ubq.pdb");

  ASSERT_TRUE(advance());

  EXPECT_EQ(mol().name(), "1UBQ");
  EXPECT_EQ(mol().num_atoms(), 660);
  EXPECT_EQ(mol().count_heavy_atoms(), 660);

  Molecule::Atom atom = mol().atom(0);
  EXPECT_EQ(atom.data().get_name(), "N");
  EXPECT_EQ(atom.data().hybridization(), constants::kSP3);
  EXPECT_EQ(atom.data().implicit_hydrogens(), 2);
  EXPECT_FALSE(atom.data().is_conjugated());

  atom = mol().atom(2);
  EXPECT_EQ(atom.data().get_name(), "C");
  EXPECT_EQ(atom.data().hybridization(), constants::kSP2);
  EXPECT_TRUE(atom.data().is_conjugated());

  ASSERT_EQ(mol().confs().size(), 1);
  // 134 residues + 1 chains
  ASSERT_EQ(mol().num_substructures(), 135);
}

TEST_F(PDBTest, HandleMultipleModels) {
  set_test_file("3cye_part.pdb");

  ASSERT_TRUE(advance());
  EXPECT_EQ(mol().name(), "3CYE");

  EXPECT_EQ(mol().num_atoms(), 55);
  EXPECT_EQ(mol().num_bonds(), 54);

  // Disulfide bond
  EXPECT_NE(mol().find_bond(28, 34), mol().bond_end());
  EXPECT_EQ(mol().atom(28).data().implicit_hydrogens(), 0);
  EXPECT_EQ(mol().atom(34).data().implicit_hydrogens(), 0);

  ASSERT_EQ(mol().confs().size(), 2);
  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(0), mol().confs()[1].col(0));
  NURI_EXPECT_EIGEN_NE(mol().confs()[0].col(43), mol().confs()[1].col(43));

  // 8 residues + 1 chain
  ASSERT_EQ(mol().num_substructures(), 9);
  EXPECT_EQ(mol().get_substructure(0).name(), "VAL");
  EXPECT_EQ(mol().get_substructure(0).num_atoms(), 7);

  ASSERT_TRUE(advance());
  EXPECT_EQ(mol().name(), "3CYE");

  EXPECT_EQ(mol().num_atoms(), 36);
  EXPECT_EQ(mol().num_bonds(), 34);

  ASSERT_EQ(mol().confs().size(), 2);
  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(0), mol().confs()[1].col(0));
  NURI_EXPECT_EIGEN_NE(mol().confs()[0].col(31), mol().confs()[1].col(31));

  // 5 residues + 1 chain
  ASSERT_EQ(mol().num_substructures(), 6);
  EXPECT_EQ(mol().get_substructure(0).name(), "VAL");
  EXPECT_EQ(mol().get_substructure(0).num_atoms(), 7);
}

class PDB1alxTest: public testing::Test {
protected:
  // NOLINTNEXTLINE(clang-diagnostic-unused-member-function)
  static void SetUpTestSuite() {
    std::ifstream ifs(internal::test_data("1alx_part.pdb"));
    ASSERT_TRUE(ifs) << "Failed to open file: 1alx_part.pdb";
    PDBReader reader(ifs);
    MoleculeStream<PDBReader> ms(reader);

    ASSERT_TRUE(ms.advance());

    mol_ = std::move(ms.current());
    ASSERT_EQ(mol_.name(), "1ALX");
    ASSERT_EQ(mol_.num_atoms(), 73);
    ASSERT_EQ(mol_.num_bonds(), 70);
    ASSERT_EQ(mol_.num_fragments(), 4);
    // 5 residues + 1 chain
    ASSERT_EQ(mol_.num_substructures(), 6);

    ASSERT_FALSE(ms.advance());
  }

  static const Molecule &mol() { return mol_; }

private:
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline static Molecule mol_;
};

TEST_F(PDB1alxTest, HandleMultipleAltlocs) {
  ASSERT_EQ(mol().confs().size(), 3);

  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(0), mol().confs()[1].col(0));
  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(0), mol().confs()[2].col(0));

  NURI_EXPECT_EIGEN_NE(mol().confs()[0].col(31), mol().confs()[1].col(31));
  NURI_EXPECT_EIGEN_EQ(mol().confs()[0].col(31), mol().confs()[2].col(31));

  NURI_EXPECT_EIGEN_NE(mol().confs()[0].col(72), mol().confs()[1].col(72));
  NURI_EXPECT_EIGEN_NE(mol().confs()[1].col(72), mol().confs()[2].col(72));
}

TEST_F(PDB1alxTest, HandleInconsistentResidues) {
  const Substructure &res = mol().get_substructure(3);
  EXPECT_EQ(res.name(), "TYR");
  EXPECT_EQ(res.id(), 11);
  EXPECT_EQ(res.num_atoms(), 21);
  EXPECT_EQ(res.count_heavy_atoms(), 12);

  auto subs = mol().find_substructures(11);
  std::vector<Substructure> res_11(subs.begin(), subs.end());
  EXPECT_EQ(res_11.size(), 1);
  EXPECT_EQ(res_11[0].name(), "TYR");
}
}  // namespace
}  // namespace nuri

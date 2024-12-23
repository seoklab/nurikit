//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/pdb.h"

#include <fstream>
#include <sstream>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/strings/match.h>

#include <gtest/gtest.h>

#include "fmt_test_common.h"
#include "test_utils.h"
#include "nuri/algo/guess.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
class PDBTest: public internal::FileFormatTest<PDBReader> {
  using Parent = internal::FileFormatTest<PDBReader>;

public:
  bool advance_and_guess() {
    return advance() && internal::guess_update_subs(mol());
  }
};

TEST_F(PDBTest, BasicParsing) {
  set_test_file("dummy.pdb");

  ASSERT_TRUE(advance_and_guess());

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
  EXPECT_NO_FATAL_FAILURE(
      static_cast<void>(internal::guess_update_subs(mol())));

  EXPECT_EQ(mol().num_atoms(), 90);

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
}

TEST_F(PDBTest, HandleCleanPDB) {
  set_test_file("1ubq.pdb");

  ASSERT_TRUE(advance_and_guess());

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

  ASSERT_TRUE(advance_and_guess());
  EXPECT_EQ(mol().name(), "3CYE");
  EXPECT_EQ(internal::get_key(mol().props(), "model"), "1");

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

  ASSERT_TRUE(advance_and_guess());
  EXPECT_EQ(mol().name(), "3CYE");
  EXPECT_EQ(internal::get_key(mol().props(), "model"), "2");

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

TEST(PDBErrorTest, HandleSyntaxErrors) {
  // parser ignores contents after model index, use remaining for comments
  std::istringstream iss(R"pdb(
MODEL        1  // Missing coordinates
ATOM      1  N   ALA A   1
ENDMDL
MODEL        2  // Invalid serial number
ATOM      A  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ENDMDL
MODEL        3  // Invalid residue number
ATOM      1  N   ALA A   A      11.104   6.134  -6.504  1.00  0.00           N
ENDMDL
MODEL        4  // Missing atom name
ATOM      1      ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ENDMDL
)pdb");
  PDBReader reader(iss);
  auto ms = reader.stream();

  int cnt = 0;
  while (ms.advance()) {
    const Molecule &mol = ms.current();
    EXPECT_TRUE(mol.empty());
    ++cnt;
  }
  EXPECT_EQ(cnt, 4);
}

// GH-402
TEST(PDBErrorTest, HandleSemanticErrors) {
  // parser ignores contents after model index, use remaining for comments
  std::istringstream iss(R"pdb(
MODEL        1  // Duplicate atom records (one with same serial, other with same id)
ATOM   9006  H   SER B 153      30.485  52.658  25.676  1.00 58.61           H
ATOM   9006  H   SER B 153      30.485  52.658  25.676  1.00 58.61           H
ATOM   9007  H   SER B 153      30.485  52.658  25.676  0.00 58.61           H
ENDMDL
)pdb");
  PDBReader reader(iss);
  auto ms = reader.stream();

  while (ms.advance()) {
    const Molecule &mol = ms.current();
    EXPECT_EQ(mol.num_atoms(), 1);
  }
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
    EXPECT_TRUE(internal::guess_update_subs(ms.current()));

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

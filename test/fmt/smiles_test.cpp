//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/smiles.h"

#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "fmt_test_common.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
namespace {
using SmilesTest = internal::FormatTest<SmilesStream>;

TEST_F(SmilesTest, SingleAtomTest) {
  set_test_string(
    // Taken from opensmiles spec
    "[U] uranium\n"
    "[Pb] lead\n"
    "[He] helium\n"
    "[*] unknown atom\n");

  test_next_mol("uranium", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().atomic_number(), 92);

  test_next_mol("lead", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().atomic_number(), 82);

  test_next_mol("helium", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().atomic_number(), 2);

  test_next_mol("unknown atom", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().atomic_number(), 0);
}

TEST_F(SmilesTest, SingleHeavyAtomTest) {
  set_test_string(
    // Taken from opensmiles spec
    "[CH4] methane\n"
    "[ClH] hydrogen chloride\n"
    "[ClH1] hydrogen chloride\n");

  test_next_mol("methane", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 4);

  test_next_mol("hydrogen chloride", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 1);

  test_next_mol("hydrogen chloride", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 1);
}

TEST_F(SmilesTest, ChargeTest) {
  set_test_string(
    // Taken from opensmiles spec
    "[Cl-] chloride anion\n"
    "[OH-] hydroxide anion\n"
    "[NH4+] ammonium cation\n"
    "[Cu+2] copper(II) cation\n"
    "[Cu++] copper(II) cation\n");

  test_next_mol("chloride anion", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().formal_charge(), -1);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 0);

  test_next_mol("hydroxide anion", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().formal_charge(), -1);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 1);

  test_next_mol("ammonium cation", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().formal_charge(), 1);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 4);

  test_next_mol("copper(II) cation", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().formal_charge(), 2);

  test_next_mol("copper(II) cation", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().formal_charge(), 2);
}

TEST_F(SmilesTest, IsotopeTest) {
  set_test_string(
    // Taken from opensmiles spec
    "[13CH4] 13C methane\n"
    "[2H+] deuterium ion\n");

  test_next_mol("13C methane", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().isotope().mass_number, 13);

  test_next_mol("deuterium ion", 1, 0);
  EXPECT_EQ(mol_.atom(0).data().isotope().mass_number, 2);
  EXPECT_EQ(mol_.atom(0).data().formal_charge(), 1);
}

TEST_F(SmilesTest, WildcardAtomTest) {
  set_test_string(
    // Taken from opensmiles spec
    // 01 2 3 4567
    "Oc1c(*)cccc1 ortho-substituted phenol\n");

  test_next_mol("ortho-substituted phenol", 8, 8);
  EXPECT_EQ(mol_.atom(3).data().implicit_hydrogens(), 0);
}

TEST_F(SmilesTest, AtomClassTest) {
  set_test_string(
    // Taken from opensmiles spec
    "[CH4:005] methane with atom class\n");

  test_next_mol("methane with atom class", 1, 0);
}

TEST_F(SmilesTest, BasicBondsTest) {
  set_test_string(
    // Taken from opensmiles spec
    "CC ethane\n"
    "C=C ethene\n"
    "C#N hydrogen cyanide\n"
    "[Rh-](Cl)(Cl)(Cl)(Cl)$[Rh-](Cl)(Cl)(Cl)Cl octachlorodirhenate (III)\n"
    "c:1:c:c:c:c:c:1 benzene\n");

  test_next_mol("ethane", 2, 1);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 3);
  EXPECT_EQ(mol_.atom(1).data().implicit_hydrogens(), 3);
  EXPECT_EQ(mol_.find_bond(0, 1)->data().order(), constants::kSingleBond);

  test_next_mol("ethene", 2, 1);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 2);
  EXPECT_EQ(mol_.atom(1).data().implicit_hydrogens(), 2);
  EXPECT_EQ(mol_.find_bond(0, 1)->data().order(), constants::kDoubleBond);

  test_next_mol("hydrogen cyanide", 2, 1);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 1);
  EXPECT_EQ(mol_.atom(1).data().implicit_hydrogens(), 0);
  EXPECT_EQ(mol_.find_bond(0, 1)->data().order(), constants::kTripleBond);

  test_next_mol("octachlorodirhenate (III)", 10, 9);
  EXPECT_EQ(mol_.find_bond(0, 5)->data().order(), constants::kQuadrupleBond);

  test_next_mol("benzene", 6, 6);
  for (auto bond: mol_.bonds()) {
    EXPECT_EQ(bond.data().order(), constants::kAromaticBond);
  }
}

TEST_F(SmilesTest, BranchTest) {
  set_test_string(
    // Taken from opensmiles spec
    "OS(=O)(=S)O thiosulfate\n"
    "[O-]P(=O)([O-])[O-] phosphate\n"
    "C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C))))))))))))))))))))C C22H46");

  test_next_mol("thiosulfate", 5, 4);
  EXPECT_EQ(mol_.find_bond(0, 1)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol_.find_bond(1, 2)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol_.find_bond(1, 3)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol_.find_bond(1, 4)->data().order(), constants::kSingleBond);

  test_next_mol("phosphate", 5, 4);
  EXPECT_EQ(mol_.find_bond(0, 1)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol_.find_bond(1, 2)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol_.find_bond(1, 3)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol_.find_bond(1, 4)->data().order(), constants::kSingleBond);

  test_next_mol("C22H46", 22, 21);
  EXPECT_EQ(std::accumulate(mol_.begin(), mol_.end(), 0,
                            [](int sum, Molecule::Atom atom) {
                              return sum + atom.data().implicit_hydrogens();
                            }),
            46);
}

TEST_F(SmilesTest, RingsTest) {
  set_test_string(
    // Taken from opensmiles spec
    "N1CC2CCCCC2CC1 perhydroisoquinoline\n"
    "C=1CCCCC=1 cyclohexene\n"
    "C=1CCCCC1 cyclohexene\n"
    "C1CCCCC=1 cyclohexene\n"
    "C-1CCCCC=1 cyclohexene error\n"
    "C1CCCCC1C1CCCCC1 bicyclohexyl\n"
    "C12(CCCCC1)CCCCC2 spiro[5.5]undecane\n"
    "C%12CCCCC%12 cyclohexane\n"
    "C12CCCCC12 error\n"
    "C12C2CCC1 error\n"
    "C11 error\n");

  test_next_mol("perhydroisoquinoline", 10, 11);
  for (auto atom: mol_) {
    if (atom.data().atomic_number() == 7 || atom.degree() == 3) {
      EXPECT_EQ(atom.data().implicit_hydrogens(), 1);
    } else {
      EXPECT_EQ(atom.data().implicit_hydrogens(), 2);
    }
  }
  for (auto bond: mol_.bonds()) {
    EXPECT_EQ(bond.data().order(), constants::kSingleBond);
  }
  EXPECT_EQ(mol_.num_sssr(), 2);

  test_next_mol("cyclohexene", 6, 6);
  test_next_mol("cyclohexene", 6, 6);
  test_next_mol("cyclohexene", 6, 6);
  test_parse_fail();

  test_next_mol("bicyclohexyl", 12, 13);
  EXPECT_EQ(mol_.num_sssr(), 2);

  test_next_mol("spiro[5.5]undecane", 11, 12);
  EXPECT_EQ(mol_.num_sssr(), 2);

  test_next_mol("cyclohexane", 6, 6);
  EXPECT_EQ(mol_.num_sssr(), 1);

  test_parse_fail();
  test_parse_fail();
  test_parse_fail();
}

TEST_F(SmilesTest, AromaticityTest) {
  set_test_string(
    // Taken from opensmiles spec
    "c1ccc2CCCc2c1 indane\n"
    "C1=CC=C2CCCC2=C1 indane\n"
    "c1occc1 furan\n"
    "C1OC=CC=1 furan\n"
    "c1ccc1 cyclobutadiene\n"
    "C1=CC=C1 cyclobutadiene\n"
    "c1ccccc1-c2ccccc2 biphenyl\n"
    "c1ccccc1c2ccccc2 biphenyl error\n"
    // Taken from rdkit documentation
    "C1=CC2=C(C=C1)C1=CC=CC=C21 aromatic test\n"
    "O=C1C=CC(=O)C2=C1OC=CO2 aromatic test\n");

  auto test_indane = [&]() {
    test_next_mol("indane", 9, 10);
    EXPECT_EQ(mol_.num_sssr(), 2);
    for (int i: { 0, 1, 2, 3, 7, 8 }) {
      EXPECT_TRUE(mol_.atom(i).data().is_aromatic());
    }
    for (int i: { 4, 5, 6 }) {
      EXPECT_FALSE(mol_.atom(i).data().is_aromatic());
    }
  };
  test_indane();
  test_indane();

  auto test_furan = [&]() {
    test_next_mol("furan", 5, 5);
    EXPECT_EQ(mol_.num_sssr(), 1);
    for (auto atom: mol_) {
      EXPECT_TRUE(atom.data().is_aromatic());
    }
  };
  test_furan();
  test_furan();

  auto test_cbd = [&]() {
    test_next_mol("cyclobutadiene", 4, 4);
    EXPECT_EQ(mol_.num_sssr(), 1);
    for (auto atom: mol_) {
      EXPECT_FALSE(atom.data().is_aromatic());
    }
  };
  test_cbd();
  test_cbd();

  test_next_mol("biphenyl", 12, 13);
  EXPECT_EQ(mol_.num_sssr(), 2);
  for (auto atom: mol_) {
    EXPECT_TRUE(atom.data().is_aromatic());
  }
  for (auto bond: mol_.bonds()) {
    if (bond.src() == 5 && bond.dst() == 6) {
      EXPECT_EQ(bond.data().order(), constants::kSingleBond);
      EXPECT_FALSE(bond.data().is_aromatic());
    } else {
      EXPECT_EQ(bond.data().order(), constants::kAromaticBond);
      EXPECT_TRUE(bond.data().is_aromatic());
    }
  }

  test_error_mol();

  test_next_mol("aromatic test", 12, 14);
  EXPECT_EQ(mol_.num_sssr(), 3);
  for (auto atom: mol_) {
    EXPECT_TRUE(atom.data().is_aromatic());
  }
  EXPECT_FALSE(mol_.find_bond(3, 6)->data().is_aromatic());

  test_next_mol("aromatic test", 12, 13);
  EXPECT_EQ(mol_.num_sssr(), 2);
  for (auto atom: mol_) {
    EXPECT_FALSE(atom.data().is_aromatic());
  }
}

TEST_F(SmilesTest, MoreHydrogensTest) {
  // Taken from opensmiles spec
  set_test_string("[H]C([H])([H])[H] explicit hydrogen methane");

  test_next_mol("explicit hydrogen methane", 5, 4);
  EXPECT_EQ(mol_.atom(0).data().implicit_hydrogens(), 0);
}

TEST_F(SmilesTest, DotBondTest) {
  set_test_string(
    // Taken from opensmiles spec
    "[Na+].[Cl-] sodium chloride\n"
    "[NH4+].[NH4+].[O-]S(=O)(=O)[S-] diammonium thiosulfate\n"
    "c1c2c3c4cc1.Br2.Cl3.Cl4 1-bromo-2,3-dichlorobenzene\n");

  test_next_mol("sodium chloride", 2, 0);
  test_next_mol("diammonium thiosulfate", 7, 4);
  test_next_mol("1-bromo-2,3-dichlorobenzene", 9, 9);
}

TEST_F(SmilesTest, BondGeometryTest) {
  set_test_string(  // Taken from opensmiles spec
    "F/C=C/F trans-difluoride\n"
    "C(/F)=C/F cis-difluoride\n");

  test_next_mol("trans-difluoride", 4, 3);
  test_next_mol("cis-difluoride", 4, 3);
}

TEST_F(SmilesTest, EnamineRealExamplesTest) {
  // clang-format off
  set_test_string(
// Taken from Enamine REAL database
// 01 2 34 5  6 789  0 1  23 4 5 67890  1  2    3  4    5  6 789  0   1 2
  "CC(C)CN(C(=O)COC(=O)C1=CC=C(N2CCCCC2)C([N+](=O)[O-])=C1)C1CCS(=O)(=O)C1 Z19788751\n"
// 012  34 5 6 78 9 01  2 3 4   5 6789  0  1234  5  6
  "COC1=CC=C2C=CC=C(CC(=O)N=S3(=O)CCCC4(C3)OCCO4)C2=C1	test molecule Z3640991685\n"
// 012  34 5  6 7  8 90 1  2 34567  8 90 1  2 3 4 5  6
  "COC1=CC(F)=C(C(=O)NC(C)(C)CCCNC(=O)OC(C)(C)C)C(F)=C1 Z3085457096\n"
// 01 2 3  4  5     6789 0  1 2  34 5 678  90 1 2  3 4  5 6
  "CN(C)C(=O)[C@@H]1CCCN1C(=O)C1=CC=C(COC2=CC=C(Cl)C=C2)C=C1 Z2719008285\n"
);
  // clang-format on

  print_ = true;

  test_next_mol("Z19788751", 33, 35);
  test_next_mol("test molecule Z3640991685", 27, 30);
  test_next_mol("Z3085457096", 27, 27);
  test_next_mol("Z2719008285", 27, 29);
}

TEST(SmilesFactoryTest, CreationTest) {
  std::istringstream iss("C methane");
  const MoleculeStreamFactory *smiles_factory =
    MoleculeStreamFactory::find_factory("smi");
  ASSERT_TRUE(smiles_factory != nullptr);

  std::unique_ptr<MoleculeStream> ss = smiles_factory->from_stream(iss);
  ASSERT_TRUE(ss);
  ASSERT_TRUE(ss->advance());

  Molecule mol = ss->current();
  EXPECT_FALSE(mol.empty());

  MoleculeSanitizer sanitizer(mol);
  EXPECT_TRUE(sanitizer.sanitize_all());

  EXPECT_EQ(mol.num_atoms(), 1);
  EXPECT_EQ(mol.num_bonds(), 0);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 4);

  ASSERT_FALSE(ss->advance());
}
}  // namespace
}  // namespace nuri

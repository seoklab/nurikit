//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/smiles.h"

#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "fmt_test_common.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
namespace {
using SmilesTest = internal::StringFormatTest<SmilesReader>;

TEST_F(SmilesTest, SingleAtomTest) {
  set_test_string(
      // Taken from opensmiles spec
      "[U] uranium\n"
      "[Pb] lead\n"
      "[He] helium\n"
      "[*] unknown atom\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("uranium", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 92);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("lead", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 82);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("helium", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 2);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("unknown atom", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 0);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("uranium", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 92);

  NURI_FMT_TEST_NEXT_MOL("lead", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 82);

  NURI_FMT_TEST_NEXT_MOL("helium", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 2);

  NURI_FMT_TEST_NEXT_MOL("unknown atom", 1, 0);
  EXPECT_EQ(mol().atom(0).data().atomic_number(), 0);
}

TEST_F(SmilesTest, SingleHeavyAtomTest) {
  set_test_string(
      // Taken from opensmiles spec
      "[CH4] methane\n"
      "[ClH] hydrogen chloride\n"
      "[ClH1] hydrogen chloride\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("methane", 1, 0);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 4);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("hydrogen chloride", 1, 0);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("hydrogen chloride", 1, 0);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("methane", 1, 0);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 4);

  NURI_FMT_TEST_NEXT_MOL("hydrogen chloride", 1, 0);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);

  NURI_FMT_TEST_NEXT_MOL("hydrogen chloride", 1, 0);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);
}

TEST_F(SmilesTest, ChargeTest) {
  set_test_string(
      // Taken from opensmiles spec
      "[Cl-] chloride anion\n"
      "[OH-] hydroxide anion\n"
      "[NH4+] ammonium cation\n"
      "[Cu+2] copper(II) cation\n"
      "[Cu++] copper(II) cation\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("chloride anion", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), -1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 0);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("hydroxide anion", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), -1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("ammonium cation", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 4);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("copper(II) cation", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 2);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("copper(II) cation", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 2);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("chloride anion", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), -1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 0);

  NURI_FMT_TEST_NEXT_MOL("hydroxide anion", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), -1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);

  NURI_FMT_TEST_NEXT_MOL("ammonium cation", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 4);

  NURI_FMT_TEST_NEXT_MOL("copper(II) cation", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 2);

  NURI_FMT_TEST_NEXT_MOL("copper(II) cation", 1, 0);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 2);
}

TEST_F(SmilesTest, IsotopeTest) {
  set_test_string(
      // Taken from opensmiles spec
      "[13CH4] 13C methane\n"
      "[2H+] deuterium ion\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("13C methane", 1, 0);
  EXPECT_EQ(mol().atom(0).data().isotope().mass_number, 13);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("deuterium ion", 1, 0);
  EXPECT_EQ(mol().atom(0).data().isotope().mass_number, 2);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 1);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("13C methane", 1, 0);
  EXPECT_EQ(mol().atom(0).data().isotope().mass_number, 13);

  NURI_FMT_TEST_NEXT_MOL("deuterium ion", 1, 0);
  EXPECT_EQ(mol().atom(0).data().isotope().mass_number, 2);
  EXPECT_EQ(mol().atom(0).data().formal_charge(), 1);
}

TEST_F(SmilesTest, WildcardAtomTest) {
  set_test_string(
      // Taken from opensmiles spec
      // 01 2 3 4567
      "Oc1c(*)cccc1 ortho-substituted phenol\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("ortho-substituted phenol", 8, 8);
  EXPECT_EQ(mol().atom(3).data().implicit_hydrogens(), 0);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("ortho-substituted phenol", 8, 8);
  EXPECT_EQ(mol().atom(3).data().implicit_hydrogens(), 0);
}

TEST_F(SmilesTest, AtomClassTest) {
  set_test_string(
      // Taken from opensmiles spec
      "[CH4:005] methane with atom class\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("methane with atom class", 1, 0);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("methane with atom class", 1, 0);
}

TEST_F(SmilesTest, BasicBondsTest) {
  set_test_string(
      // Taken from opensmiles spec
      "CC ethane\n"
      "C=C ethene\n"
      "C#N hydrogen cyanide\n"
      "[Rh-](Cl)(Cl)(Cl)(Cl)$[Rh-](Cl)(Cl)(Cl)Cl octachlorodirhenate (III)\n"
      "c:1:c:c:c:c:c:1 benzene\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("ethane", 2, 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 3);
  EXPECT_EQ(mol().atom(1).data().implicit_hydrogens(), 3);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kSingleBond);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("ethene", 2, 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 2);
  EXPECT_EQ(mol().atom(1).data().implicit_hydrogens(), 2);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kDoubleBond);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("hydrogen cyanide", 2, 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);
  EXPECT_EQ(mol().atom(1).data().implicit_hydrogens(), 0);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kTripleBond);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("octachlorodirhenate (III)", 10, 9);
  EXPECT_EQ(mol().find_bond(0, 5)->data().order(), constants::kQuadrupleBond);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("benzene", 6, 6);
  for (auto bond: mol().bonds()) {
    EXPECT_EQ(bond.data().order(), constants::kAromaticBond);
  }
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("ethane", 2, 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 3);
  EXPECT_EQ(mol().atom(1).data().implicit_hydrogens(), 3);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kSingleBond);

  NURI_FMT_TEST_NEXT_MOL("ethene", 2, 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 2);
  EXPECT_EQ(mol().atom(1).data().implicit_hydrogens(), 2);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kDoubleBond);

  NURI_FMT_TEST_NEXT_MOL("hydrogen cyanide", 2, 1);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 1);
  EXPECT_EQ(mol().atom(1).data().implicit_hydrogens(), 0);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kTripleBond);

  NURI_FMT_TEST_NEXT_MOL("octachlorodirhenate (III)", 10, 9);
  EXPECT_EQ(mol().find_bond(0, 5)->data().order(), constants::kQuadrupleBond);

  NURI_FMT_TEST_NEXT_MOL("benzene", 6, 6);
  for (auto bond: mol().bonds()) {
    EXPECT_EQ(bond.data().order(), constants::kAromaticBond);
  }
}

TEST_F(SmilesTest, BranchTest) {
  set_test_string(
      // Taken from opensmiles spec
      "OS(=O)(=S)O thiosulfate\n"
      "[O-]P(=O)([O-])[O-] phosphate\n"
      "C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C))))))))))))))))))))C C22H46");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("thiosulfate", 5, 4);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol().find_bond(1, 2)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol().find_bond(1, 3)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol().find_bond(1, 4)->data().order(), constants::kSingleBond);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("phosphate", 5, 4);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol().find_bond(1, 2)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol().find_bond(1, 3)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol().find_bond(1, 4)->data().order(), constants::kSingleBond);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("C22H46", 22, 21);
  EXPECT_EQ(std::accumulate(mol().begin(), mol().end(), 0,
                            [](int sum, Molecule::Atom atom) {
                              return sum + atom.data().implicit_hydrogens();
                            }),
            46);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("thiosulfate", 5, 4);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol().find_bond(1, 2)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol().find_bond(1, 3)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol().find_bond(1, 4)->data().order(), constants::kSingleBond);

  NURI_FMT_TEST_NEXT_MOL("phosphate", 5, 4);
  EXPECT_EQ(mol().find_bond(0, 1)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol().find_bond(1, 2)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol().find_bond(1, 3)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol().find_bond(1, 4)->data().order(), constants::kSingleBond);

  NURI_FMT_TEST_NEXT_MOL("C22H46", 22, 21);
  EXPECT_EQ(std::accumulate(mol().begin(), mol().end(), 0,
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
      "C1CCCCC1C1CCCCC1 bicyclohexyl\n"
      "C12(CCCCC1)CCCCC2 spiro[5.5]undecane\n"
      "C%12CCCCC%12 cyclohexane\n"
      "C-1CCCCC=1 cyclohexene error\n"
      "C12CCCCC12 error\n"
      "C12C2CCC1 error\n"
      "C11 error\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("perhydroisoquinoline", 10, 11);
  for (auto atom: mol()) {
    if (atom.data().atomic_number() == 7 || atom.degree() == 3) {
      EXPECT_EQ(atom.data().implicit_hydrogens(), 1);
    } else {
      EXPECT_EQ(atom.data().implicit_hydrogens(), 2);
    }
  }
  for (auto bond: mol().bonds()) {
    EXPECT_EQ(bond.data().order(), constants::kSingleBond);
  }
  EXPECT_EQ(mol().num_sssr(), 2);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("cyclohexene", 6, 6);
  write_smiles(smi, mol());
  NURI_FMT_TEST_NEXT_MOL("cyclohexene", 6, 6);
  write_smiles(smi, mol());
  NURI_FMT_TEST_NEXT_MOL("cyclohexene", 6, 6);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("bicyclohexyl", 12, 13);
  EXPECT_EQ(mol().num_sssr(), 2);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("spiro[5.5]undecane", 11, 12);
  EXPECT_EQ(mol().num_sssr(), 2);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("cyclohexane", 6, 6);
  EXPECT_EQ(mol().num_sssr(), 1);
  write_smiles(smi, mol());

  NURI_FMT_TEST_PARSE_FAIL();
  NURI_FMT_TEST_PARSE_FAIL();
  NURI_FMT_TEST_PARSE_FAIL();
  NURI_FMT_TEST_PARSE_FAIL();

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("perhydroisoquinoline", 10, 11);
  for (auto atom: mol()) {
    if (atom.data().atomic_number() == 7 || atom.degree() == 3) {
      EXPECT_EQ(atom.data().implicit_hydrogens(), 1);
    } else {
      EXPECT_EQ(atom.data().implicit_hydrogens(), 2);
    }
  }
  for (auto bond: mol().bonds()) {
    EXPECT_EQ(bond.data().order(), constants::kSingleBond);
  }
  EXPECT_EQ(mol().num_sssr(), 2);

  NURI_FMT_TEST_NEXT_MOL("cyclohexene", 6, 6);
  NURI_FMT_TEST_NEXT_MOL("cyclohexene", 6, 6);
  NURI_FMT_TEST_NEXT_MOL("cyclohexene", 6, 6);

  NURI_FMT_TEST_NEXT_MOL("bicyclohexyl", 12, 13);
  EXPECT_EQ(mol().num_sssr(), 2);

  NURI_FMT_TEST_NEXT_MOL("spiro[5.5]undecane", 11, 12);
  EXPECT_EQ(mol().num_sssr(), 2);

  NURI_FMT_TEST_NEXT_MOL("cyclohexane", 6, 6);
  EXPECT_EQ(mol().num_sssr(), 1);
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
      // Taken from rdkit documentation
      "C1=CC2=C(C=C1)C1=CC=CC=C21 aromatic test\n"
      "O=C1C=CC(=O)C2=C1OC=CO2 aromatic test\n"
      // Extra
      "c1[cH-]ccc1 cyclopenadienyl anion\n"
      "c1ccccc1:c2ccccc2 biphenyl error\n"
      // GH-364
      "c1cc[nH]c1 pyrrole\n"
      "C1C=CNC=1 pyrrole\n"
      "c1c[nH]cn1 imidazole\n"
      "C1=CNC=N1 imidazole\n");

  std::string smi;

  auto test_indane = [&]() {
    NURI_FMT_TEST_NEXT_MOL("indane", 9, 10);
    EXPECT_EQ(mol().num_sssr(), 2);
    for (int i: { 0, 1, 2, 3, 7, 8 }) {
      EXPECT_TRUE(mol().atom(i).data().is_aromatic());
    }
    for (int i: { 4, 5, 6 }) {
      EXPECT_FALSE(mol().atom(i).data().is_aromatic());
    }
  };

  auto test_furan = [&]() {
    NURI_FMT_TEST_NEXT_MOL("furan", 5, 5);
    EXPECT_EQ(mol().num_sssr(), 1);
    for (auto atom: mol()) {
      EXPECT_TRUE(atom.data().is_aromatic());
    }
  };

  auto test_cbd = [&]() {
    NURI_FMT_TEST_NEXT_MOL("cyclobutadiene", 4, 4);
    EXPECT_EQ(mol().num_sssr(), 1);
    for (auto atom: mol()) {
      EXPECT_FALSE(atom.data().is_aromatic());
    }
  };

  auto test_biphenyl = [&]() {
    NURI_FMT_TEST_NEXT_MOL("biphenyl", 12, 13);
    EXPECT_EQ(mol().num_sssr(), 2);
    for (auto atom: mol()) {
      EXPECT_TRUE(atom.data().is_aromatic());
    }
    for (auto bond: mol().bonds()) {
      if (bond.src().id() == 5 && bond.dst().id() == 6) {
        EXPECT_EQ(bond.data().order(), constants::kSingleBond);
        EXPECT_FALSE(bond.data().is_aromatic());
      } else {
        EXPECT_EQ(bond.data().order(), constants::kAromaticBond);
        EXPECT_TRUE(bond.data().is_aromatic());
      }
    }
  };

  auto test_rdkit_aromatic1 = [&]() {
    NURI_FMT_TEST_NEXT_MOL("aromatic test", 12, 14);
    EXPECT_EQ(mol().num_sssr(), 3);
    for (auto atom: mol()) {
      EXPECT_TRUE(atom.data().is_aromatic());
    }
    EXPECT_FALSE(mol().find_bond(3, 6)->data().is_aromatic());
  };

  auto test_rdkit_aromatic2 = [&]() {
    NURI_FMT_TEST_NEXT_MOL("aromatic test", 12, 13);
    EXPECT_EQ(mol().num_sssr(), 2);
    for (auto atom: mol()) {
      EXPECT_FALSE(atom.data().is_aromatic());
    }
  };

  auto test_cyclopentadienyl = [&]() {
    NURI_FMT_TEST_NEXT_MOL("cyclopenadienyl anion", 5, 5);
    EXPECT_EQ(mol().num_sssr(), 1);

    int total_nh = 0;
    for (auto atom: mol()) {
      EXPECT_TRUE(atom.data().is_aromatic());
      total_nh += atom.data().implicit_hydrogens();
    }
    EXPECT_EQ(total_nh, 5);
  };

  auto test_pyrrole = [&]() {
    NURI_FMT_TEST_NEXT_MOL("pyrrole", 5, 5);
    EXPECT_EQ(mol().num_sssr(), 1);

    int total_nh = 0;
    for (auto atom: mol()) {
      EXPECT_TRUE(atom.data().is_aromatic());
      total_nh += atom.data().implicit_hydrogens();
    }
    EXPECT_EQ(total_nh, 5);
  };

  auto test_imidazole = [&]() {
    NURI_FMT_TEST_NEXT_MOL("imidazole", 5, 5);
    EXPECT_EQ(mol().num_sssr(), 1);

    int total_nh = 0;
    for (auto atom: mol()) {
      EXPECT_TRUE(atom.data().is_aromatic());
      total_nh += atom.data().implicit_hydrogens();
    }
    EXPECT_EQ(total_nh, 4);
  };

  {
    SCOPED_TRACE("Initial read");

    test_indane();
    write_smiles(smi, mol());
    test_indane();
    write_smiles(smi, mol());

    test_furan();
    write_smiles(smi, mol());
    test_furan();
    write_smiles(smi, mol());

    test_cbd();
    write_smiles(smi, mol());
    test_cbd();
    write_smiles(smi, mol());

    test_biphenyl();
    write_smiles(smi, mol());

    test_rdkit_aromatic1();
    write_smiles(smi, mol());

    test_rdkit_aromatic2();
    write_smiles(smi, mol());

    test_cyclopentadienyl();
    write_smiles(smi, mol());

    NURI_FMT_TEST_ERROR_MOL();

    test_pyrrole();
    write_smiles(smi, mol());
    test_pyrrole();
    write_smiles(smi, mol());

    test_imidazole();
    write_smiles(smi, mol());
    test_imidazole();
    write_smiles(smi, mol());
  }

  set_test_string(smi);

  {
    SCOPED_TRACE("Re-read");

    test_indane();
    test_indane();

    test_furan();
    test_furan();

    test_cbd();
    test_cbd();

    test_biphenyl();

    test_rdkit_aromatic1();
    test_rdkit_aromatic2();

    test_cyclopentadienyl();

    test_pyrrole();
    test_pyrrole();

    test_imidazole();
    test_imidazole();
  }
}

TEST_F(SmilesTest, MoreHydrogensTest) {
  // Taken from opensmiles spec
  set_test_string("[H]C([H])([H])[H] explicit hydrogen methane");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("explicit hydrogen methane", 5, 4);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 0);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("explicit hydrogen methane", 5, 4);
  EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 0);
}

TEST_F(SmilesTest, DotBondTest) {
  set_test_string(
      // Taken from opensmiles spec
      "[Na+].[Cl-] sodium chloride\n"
      "[NH4+].[NH4+].[O-]S(=O)(=O)[S-] diammonium thiosulfate\n"
      "c1c2c3c4cc1.Br2.Cl3.Cl4 1-bromo-2,3-dichlorobenzene\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("sodium chloride", 2, 0);
  EXPECT_EQ(mol().num_fragments(), 2);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("diammonium thiosulfate", 7, 4);
  EXPECT_EQ(mol().num_fragments(), 3);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("1-bromo-2,3-dichlorobenzene", 9, 9);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 1);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("sodium chloride", 2, 0);
  EXPECT_EQ(mol().num_fragments(), 2);

  NURI_FMT_TEST_NEXT_MOL("diammonium thiosulfate", 7, 4);
  EXPECT_EQ(mol().num_fragments(), 3);

  NURI_FMT_TEST_NEXT_MOL("1-bromo-2,3-dichlorobenzene", 9, 9);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 1);
}

TEST_F(SmilesTest, BondGeometryTest) {
  set_test_string(  // Taken from opensmiles spec
      "F/C=C/F trans-difluoride\n"
      "C(/F)=C/F cis-difluoride\n"
      "C/1=C/C=C\\C=C/C=C\\1 cyclooctatetraene\n"
      "C/C=C/C(/C=C(CC)\\C)=C(/C=C/C)\\C(\\C)=C\\CC sample1\n"
      "C/C=C/C(/C=C/C)(/C=C/C)/C=C/C sample2\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("trans-difluoride", 4, 3);
  EXPECT_TRUE(mol().bond(1).data().has_config());
  EXPECT_TRUE(mol().bond(1).data().is_trans());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("cis-difluoride", 4, 3);
  EXPECT_TRUE(mol().bond(1).data().has_config());
  EXPECT_FALSE(mol().bond(1).data().is_trans());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("cyclooctatetraene", 8, 8);
  for (auto bond: mol().bonds()) {
    if (bond.data().order() == constants::kDoubleBond) {
      EXPECT_TRUE(bond.data().has_config());
      EXPECT_FALSE(bond.data().is_trans());
    } else {
      EXPECT_FALSE(bond.data().has_config());
    }
  }
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("sample1", 18, 17);
  for (auto bond: mol().bonds()) {
    if (bond.data().order() == constants::kDoubleBond) {
      EXPECT_TRUE(bond.data().has_config());
      EXPECT_TRUE(bond.data().is_trans());
    } else {
      EXPECT_FALSE(bond.data().has_config());
    }
  }
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("sample2", 13, 12);
  for (auto bond: mol().bonds()) {
    if (bond.data().order() == constants::kDoubleBond) {
      EXPECT_TRUE(bond.data().has_config());
      EXPECT_TRUE(bond.data().is_trans());
    } else {
      EXPECT_FALSE(bond.data().has_config());
    }
  }
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("trans-difluoride", 4, 3);
  EXPECT_TRUE(mol().bond(1).data().has_config());
  EXPECT_TRUE(mol().bond(1).data().is_trans());

  NURI_FMT_TEST_NEXT_MOL("cis-difluoride", 4, 3);
  EXPECT_TRUE(mol().bond(1).data().has_config());
  EXPECT_FALSE(mol().bond(1).data().is_trans());

  NURI_FMT_TEST_NEXT_MOL("cyclooctatetraene", 8, 8);
  for (auto bond: mol().bonds()) {
    if (bond.data().order() == constants::kDoubleBond) {
      EXPECT_TRUE(bond.data().has_config());
      EXPECT_FALSE(bond.data().is_trans());
    } else {
      EXPECT_FALSE(bond.data().has_config());
    }
  }

  NURI_FMT_TEST_NEXT_MOL("sample1", 18, 17);
  for (auto bond: mol().bonds()) {
    if (bond.data().order() == constants::kDoubleBond) {
      EXPECT_TRUE(bond.data().has_config());
      EXPECT_TRUE(bond.data().is_trans());
    } else {
      EXPECT_FALSE(bond.data().has_config());
    }
  }

  NURI_FMT_TEST_NEXT_MOL("sample2", 13, 12);
  for (auto bond: mol().bonds()) {
    if (bond.data().order() == constants::kDoubleBond) {
      EXPECT_TRUE(bond.data().has_config());
      EXPECT_TRUE(bond.data().is_trans());
    } else {
      EXPECT_FALSE(bond.data().has_config());
    }
  }
}

TEST_F(SmilesTest, InvalidBondGeometry) {
  // Found with fuzzing
  set_test_string("c1ccc2CCCc2/c=1 error\n");

  NURI_FMT_TEST_NEXT_MOL("error", 9, 10);
  for (auto bond: mol().bonds())
    EXPECT_FALSE(bond.data().has_config());
}

TEST_F(SmilesTest, ChiralityTest) {
  set_test_string(  // Taken from opensmiles spec
      "C[C@@H](C(=O)O)N alanine\n"
      "C[C@H](N)C(=O)O alanine-reversed\n"
      "[C@@H](F)(Cl)Br test first\n"
      "F[C@H](Cl)Br test not-first\n"
      "C[C@@H]1C[C@@]1(F)C test ring closing\n"
      "CC[C@@H]1C.[C@@H]1(F)C test ring closing disconnected\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("alanine", 6, 5);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_TRUE(mol().atom(1).data().is_clockwise());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("alanine-reversed", 6, 5);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("test first", 4, 3);
  EXPECT_TRUE(mol().atom(0).data().is_chiral());
  EXPECT_FALSE(mol().atom(0).data().is_clockwise());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("test not-first", 4, 3);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("test ring closing", 6, 6);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
  EXPECT_TRUE(mol().atom(3).data().is_chiral());
  EXPECT_TRUE(mol().atom(3).data().is_clockwise());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("test ring closing disconnected", 7, 6);
  EXPECT_TRUE(mol().atom(2).data().is_chiral());
  EXPECT_FALSE(mol().atom(2).data().is_clockwise());
  EXPECT_TRUE(mol().atom(4).data().is_chiral());
  EXPECT_FALSE(mol().atom(4).data().is_clockwise());
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("alanine", 6, 5);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_TRUE(mol().atom(1).data().is_clockwise());

  NURI_FMT_TEST_NEXT_MOL("alanine-reversed", 6, 5);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());

  NURI_FMT_TEST_NEXT_MOL("test first", 4, 3);
  EXPECT_TRUE(mol().atom(0).data().is_chiral());
  EXPECT_FALSE(mol().atom(0).data().is_clockwise());

  NURI_FMT_TEST_NEXT_MOL("test not-first", 4, 3);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());

  NURI_FMT_TEST_NEXT_MOL("test ring closing", 6, 6);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
  EXPECT_TRUE(mol().atom(3).data().is_chiral());
  EXPECT_TRUE(mol().atom(3).data().is_clockwise());

  NURI_FMT_TEST_NEXT_MOL("test ring closing disconnected", 7, 6);
  EXPECT_TRUE(mol().atom(2).data().is_chiral());
  EXPECT_FALSE(mol().atom(2).data().is_clockwise());
  EXPECT_TRUE(mol().atom(4).data().is_chiral());
  EXPECT_FALSE(mol().atom(4).data().is_clockwise());
}

TEST_F(SmilesTest, InvalidChiralityTest) {
  set_test_string(  //
      "C[C@H]=C too few explicit neighbors\n"
      "C[P@H2](F)Cl too many implicit hydrogens\n"
      "C[B@](F)Cl too few total neighbors\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("too few explicit neighbors", 3, 2);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("too many implicit hydrogens", 4, 3);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("too few total neighbors", 4, 3);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("too few explicit neighbors", 3, 2);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());

  NURI_FMT_TEST_NEXT_MOL("too many implicit hydrogens", 4, 3);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());

  NURI_FMT_TEST_NEXT_MOL("too few total neighbors", 4, 3);
  EXPECT_TRUE(mol().atom(1).data().is_chiral());
  EXPECT_FALSE(mol().atom(1).data().is_clockwise());
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

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("Z19788751", 33, 35);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 3);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("test molecule Z3640991685", 27, 30);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 4);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("Z3085457096", 27, 27);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 1);
  write_smiles(smi, mol());

  NURI_FMT_TEST_NEXT_MOL("Z2719008285", 27, 29);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 3);
  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("Z19788751", 33, 35);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 3);

  NURI_FMT_TEST_NEXT_MOL("test molecule Z3640991685", 27, 30);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 4);

  NURI_FMT_TEST_NEXT_MOL("Z3085457096", 27, 27);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 1);

  NURI_FMT_TEST_NEXT_MOL("Z2719008285", 27, 29);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 3);
}

TEST_F(SmilesTest, DUDEExamplesTest) {
  set_test_string(  // Taken from DUD-E set
      "Cc1ccc(cc1)n2c(=O)c3c4c(sc3nc2SCC=C)CCC[C@@H]4C C02302104\n");

  std::string smi;

  NURI_FMT_TEST_NEXT_MOL("C02302104", 26, 29);

  // GH-298
  EXPECT_EQ(mol()[13].data().implicit_hydrogens(), 0);
  EXPECT_EQ(mol()[13].data().formal_charge(), 0);

  write_smiles(smi, mol());

  set_test_string(smi);

  NURI_FMT_TEST_NEXT_MOL("C02302104", 26, 29);
}

TEST_F(SmilesTest, ManyRings) {
  set_test_string(
      "C12C(C34C(C56C(C78CC79C8%10C9%11C%10%12C%11%13C%12%14C%13%15C%14%16C%15%"
      "17C%16%18C%17%19C%18%20C%19%21C%20%22C%21%23C%22%24C%25%26C%23%24C%25%"
      "27C%26%28C%27%29C%28%30C%29%31C%30%32C%31%33C%32%34C%33%35C%34%36C%35%"
      "37C%36%38C%37C%38%39)C5%40C6%41C%40%42C%41%43C%42%44C%43%45C%44%46C%45%"
      "47C%46%48C%47%49C%48%50C%49%51C%50%52C%51%53C%52%54C%53%55C%56%57C%54%"
      "55C%56%58C%57%59C%58%60C%59%61C%60%62C%61%63C%62%64C%63%65C%64%66C%65%"
      "67C%66%68C%67%69C%68%39C%69%70)C3%71C4%72C%71%73C%72%74C%73%75C%74%76C%"
      "75%77C%76%78C%77%79C%78%80C%79%81C%80%82C%81%83C%82%84C%83%85C%84%86C%"
      "87%88C%85%86C%87%89C%88%90C%89%91C%90%92C%91%93C%92%94C%93%95C%94%96C%"
      "95%97C%96%98C%97%99C%983C%99%70C33)"
      "C11C22C11C22C11C22C11C22C11C22C11C22C11C22C11C22C45C12C41C52C11C22C11C22"
      "C11C22C11C22C11C22C13C2 many rings");

  std::string smi;

  ASSERT_TRUE(advance());
  EXPECT_EQ(mol().name(), "many rings");
  EXPECT_EQ(mol().num_atoms(), 136);
  EXPECT_EQ(mol().num_bonds(), 266);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 131);
  write_smiles(smi, mol());

  set_test_string(smi);

  ASSERT_TRUE(advance());
  EXPECT_EQ(mol().name(), "many rings");
  EXPECT_EQ(mol().num_atoms(), 136);
  EXPECT_EQ(mol().num_bonds(), 266);
  EXPECT_EQ(mol().num_fragments(), 1);
  EXPECT_EQ(mol().num_sssr(), 131);
}

TEST(SmilesFactoryTest, CreationTest) {
  std::istringstream iss("C methane");
  const MoleculeReaderFactory *smiles_factory =
      MoleculeReaderFactory::find_factory("smi");
  ASSERT_TRUE(smiles_factory != nullptr);

  std::unique_ptr<MoleculeReader> sr = smiles_factory->from_stream(iss);
  ASSERT_TRUE(sr);
  std::vector<std::string> block = sr->next();
  ASSERT_FALSE(block.empty());

  Molecule mol = sr->parse(block);
  EXPECT_FALSE(mol.empty());

  MoleculeSanitizer sanitizer(mol);
  EXPECT_TRUE(sanitizer.sanitize_all());

  EXPECT_EQ(mol.num_atoms(), 1);
  EXPECT_EQ(mol.num_bonds(), 0);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 4);

  ASSERT_TRUE(sr->next().empty());
}
}  // namespace
}  // namespace nuri

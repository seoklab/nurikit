//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/smiles.h"

#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include "nuri/core/molecule.h"

namespace {
void print_mol(const nuri::Molecule &mol) {
  for (auto atom: mol) {
    std::cout << atom.data().element_symbol() << " ";
  }
  std::cout << '\n';
  for (auto bond: mol.bonds()) {
    std::cout << bond.src() << " -> " << bond.dst() << ' '
              << bond.data().order() << '\n';
  }
  std::cout << "---\n";
}

TEST(SmilesTest, BasicTest) {
  // clang-format off
  std::istringstream iss(
// 01 2 34 5  6 789  0 1  23 4 5 67890  1  2    3  4    5  6 789  0   1 2
  "CC(C)CN(C(=O)COC(=O)C1=CC=C(N2CCCCC2)C([N+](=O)[O-])=C1)C1CCS(=O)(=O)C1 Z19788751\n"
// 012  34 5 6 78 9 01  2 3 4   5 6789  0  1234  5  6
  "COC1=CC=C2C=CC=C(CC(=O)N=S3(=O)CCCC4(C3)OCCO4)C2=C1	test molecule Z3640991685\n"
);
  // clang-format on

  nuri::SmilesStream ss(iss);
  nuri::Molecule mol;

  EXPECT_TRUE(ss.advance());
  mol = ss.current();
  ASSERT_TRUE(mol.was_valid());
  EXPECT_EQ(mol.name(), "Z19788751");

  print_mol(mol);

  EXPECT_TRUE(ss.advance());
  mol = ss.current();
  ASSERT_TRUE(mol.was_valid());
  EXPECT_EQ(mol.name(), "test molecule Z3640991685");

  print_mol(mol);

  EXPECT_FALSE(ss.advance());
}
}  // namespace

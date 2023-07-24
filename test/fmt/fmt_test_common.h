//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TEST_FMT_FMT_TEST_COMMON_H_
#define NURI_TEST_FMT_FMT_TEST_COMMON_H_

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "nuri/core/molecule.h"

namespace nuri {
namespace internal {
inline void print_mol(const Molecule &mol) {
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

template <class MoleculeStream>
class FormatTest: public ::testing::Test {
public:
  // NOLINTBEGIN(readability-identifier-naming)
  std::istringstream iss_;
  MoleculeStream ms_;
  Molecule mol_;
  bool print_;
  // NOLINTEND(readability-identifier-naming)

  void set_test_string(const std::string &str) { iss_.str(str); }

  void test_parse_fail() {
    ASSERT_TRUE(ms_.advance());
    mol_ = ms_.current();
    EXPECT_TRUE(mol_.empty());
  }

  void test_error_mol() {
    ASSERT_TRUE(ms_.advance());
    mol_ = ms_.current();

    MoleculeSanitizer sanitizer(mol_);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  void test_next_mol(std::string_view name, int natoms, int nbonds) {
    ASSERT_TRUE(ms_.advance());
    mol_ = ms_.current();

    MoleculeSanitizer sanitizer(mol_);
    ASSERT_TRUE(sanitizer.sanitize_all());

    EXPECT_EQ(mol_.name(), name);
    EXPECT_EQ(mol_.num_atoms(), natoms);
    EXPECT_EQ(mol_.num_bonds(), nbonds);

    if (print_) {
      print_mol(mol_);
    }
  }

protected:
  void SetUp() override {
    iss_.clear();
    ms_ = MoleculeStream(iss_);
    print_ = false;
  }

  void TearDown() override { EXPECT_FALSE(ms_.advance()); }
};
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TEST_FMT_FMT_TEST_COMMON_H_ */

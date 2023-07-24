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

#define NURI_FMT_TEST_PARSE_FAIL()                                             \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    this->mol_ = this->ms_.current();                                          \
    EXPECT_TRUE(this->mol_.empty()) << "Molecule index: " << this->idx_;       \
  } while (false)

#define NURI_FMT_TEST_ERROR_MOL()                                              \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    this->mol_ = this->ms_.current();                                          \
    MoleculeSanitizer sanitizer(this->mol_);                                   \
    EXPECT_FALSE(sanitizer.sanitize_all())                                     \
      << "Molecule index: " << this->idx_;                                     \
  } while (false)

#define NURI_FMT_TEST_NEXT_MOL(mol_name, natoms, nbonds)                       \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    this->mol_ = this->ms_.current();                                          \
                                                                               \
    MoleculeSanitizer sanitizer(this->mol_);                                   \
    EXPECT_TRUE(sanitizer.sanitize_all()) << "Molecule index: " << this->idx_; \
                                                                               \
    EXPECT_EQ(this->mol_.name(), mol_name)                                     \
      << "Molecule index: " << this->idx_;                                     \
    EXPECT_EQ(this->mol_.num_atoms(), natoms)                                  \
      << "Molecule index: " << this->idx_;                                     \
    EXPECT_EQ(this->mol_.num_bonds(), nbonds)                                  \
      << "Molecule index: " << this->idx_;                                     \
                                                                               \
    if (print_) {                                                              \
      internal::print_mol(this->mol_);                                         \
    }                                                                          \
  } while (false)

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
  int idx_;
  bool print_;
  // NOLINTEND(readability-identifier-naming)

  void set_test_string(const std::string &str) { iss_.str(str); }

  bool advance() {
    ++idx_;
    return ms_.advance();
  }

protected:
  void SetUp() override {
    iss_.clear();
    ms_ = MoleculeStream(iss_);
    idx_ = 0;
    print_ = false;
  }

  void TearDown() override {
    EXPECT_FALSE(advance()) << "Molecule index: " << idx_;
  }
};
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TEST_FMT_FMT_TEST_COMMON_H_ */

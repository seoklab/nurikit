//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TEST_FMT_FMT_TEST_COMMON_H_
#define NURI_TEST_FMT_FMT_TEST_COMMON_H_

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

#define NURI_FMT_TEST_PARSE_FAIL()                                             \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    EXPECT_FALSE(this->ok()) << "Molecule index: " << this->idx_;              \
  } while (false)

#define NURI_FMT_TEST_PARSE_FAIL_MSG(substr)                                   \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    ASSERT_FALSE(this->ok()) << "Molecule index: " << this->idx_;              \
    EXPECT_THAT(this->error_msg(), ::testing::HasSubstr(substr))               \
        << "Molecule index: " << this->idx_;                                   \
  } while (false)

#define NURI_FMT_TEST_NEXT_EMPTY_MOL(mol_name)                                 \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    ASSERT_TRUE(this->ok()) << this->error_msg();                              \
                                                                               \
    MoleculeSanitizer sanitizer(this->mol());                                  \
    EXPECT_TRUE(sanitizer.sanitize_all()) << "Molecule index: " << this->idx_; \
                                                                               \
    EXPECT_EQ(this->mol().name(), mol_name)                                    \
        << "Molecule index: " << this->idx_;                                   \
    EXPECT_TRUE(this->mol().empty()) << "Molecule index: " << this->idx_;      \
    EXPECT_EQ(this->mol().num_bonds(), 0) << "Molecule index: " << this->idx_; \
  } while (false)

#define NURI_FMT_TEST_ERROR_MOL()                                              \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    ASSERT_TRUE(this->ok()) << this->error_msg();                              \
    MoleculeSanitizer sanitizer(this->mol());                                  \
    EXPECT_FALSE(sanitizer.sanitize_all())                                     \
        << "Molecule index: " << this->idx_;                                   \
  } while (false)

#define NURI_FMT_TEST_NEXT_MOL(mol_name, natoms, nbonds)                       \
  do {                                                                         \
    ASSERT_TRUE(this->advance()) << "Molecule index: " << this->idx_;          \
    ASSERT_TRUE(this->ok()) << this->error_msg();                              \
                                                                               \
    MoleculeSanitizer sanitizer(this->mol());                                  \
    EXPECT_TRUE(sanitizer.sanitize_all()) << "Molecule index: " << this->idx_; \
                                                                               \
    EXPECT_EQ(this->mol().name(), mol_name)                                    \
        << "Molecule index: " << this->idx_;                                   \
    ASSERT_EQ(this->mol().num_atoms(), natoms)                                 \
        << "Molecule index: " << this->idx_;                                   \
    ASSERT_EQ(this->mol().num_bonds(), nbonds)                                 \
        << "Molecule index: " << this->idx_;                                   \
                                                                               \
    if (print_) {                                                              \
      internal::print_mol(this->mol());                                        \
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
    std::cout << bond.src().id() << " -> " << bond.dst().id() << ' '
              << bond.data().order() << '\n';
  }
  std::cout << "---\n";
}

inline std::filesystem::path test_data(std::string_view name) {
  return std::filesystem::path(NURI_TEST_DATA_DIR) / name;
}

template <class MoleculeReader>
class StringFormatTest: public ::testing::Test {
public:
  // NOLINTBEGIN(readability-identifier-naming)
  std::istringstream iss_;
  MoleculeReader mr_ = { iss_ };
  MoleculeStream<MoleculeReader> ms_ = { mr_ };
  int idx_;
  bool print_;
  // NOLINTEND(readability-identifier-naming)

  void set_test_string(const std::string &str) {
    iss_.clear();
    iss_.str(str);
  }

  bool advance() {
    ++idx_;
    return ms_.advance();
  }

  bool ok() const { return ms_.ok(); }

  std::string_view error_msg() const { return ms_.error_msg(); }

  Molecule &mol() { return ms_.current(); }

protected:
  void SetUp() override {
    idx_ = 0;
    print_ = false;
  }

  void TearDown() override {
    EXPECT_FALSE(advance()) << "Molecule index: " << idx_;
  }
};

template <class MoleculeReader>
class FileFormatTest: public ::testing::Test {
public:
  // NOLINTBEGIN(readability-identifier-naming)
  std::ifstream ifs_;
  MoleculeReader mr_ = { ifs_ };
  MoleculeStream<MoleculeReader> ms_ = { mr_ };
  int idx_;
  bool print_;
  // NOLINTEND(readability-identifier-naming)

  void set_test_file(std::string_view name) {
    ifs_.open(test_data(name));
    ASSERT_TRUE(ifs_) << "Failed to open file: " << name;
  }

  bool advance() {
    ++idx_;
    return ms_.advance();
  }

  bool ok() const { return ms_.ok(); }

  std::string_view error_msg() const { return ms_.error_msg(); }

  Molecule &mol() { return ms_.current(); }

protected:
  void SetUp() override {
    ifs_.clear();
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

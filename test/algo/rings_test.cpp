//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/rings.h"

#include <algorithm>
#include <vector>

#include <absl/container/flat_hash_set.h>
#include <gtest/gtest.h>

#include "nuri/core/molecule.h"

namespace nuri {
namespace {
// NOLINTNEXTLINE(*-using-namespace)
using namespace constants;

const PeriodicTable &pt = PeriodicTable::get();

template <class T>
bool is_permutation_no_dup(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size()) {
    return false;
  }

  absl::flat_hash_set<T> a_set(a.begin(), a.end());
  return std::all_of(b.begin(), b.end(),
                     [&a_set](const T &x) { return a_set.contains(x); });
}

TEST(MoleculeAlgorithmTest, IndoleTest) {
  Molecule mol;
  {
    auto m = mol.mutator();

    for (int i = 0; i < 8; ++i) {
      m.add_atom(pt[6]);
    }
    m.add_atom(pt[7]);
    for (int i = 0; i < 9; ++i) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
    for (int i: { 4, 5 }) {
      mol.atom(i).data().set_implicit_hydrogens(0);
    }

    m.add_bond(0, 1, BondData(kAromaticBond));
    m.add_bond(0, 5, BondData(kAromaticBond));
    m.add_bond(1, 2, BondData(kAromaticBond));
    m.add_bond(2, 3, BondData(kAromaticBond));
    m.add_bond(3, 4, BondData(kAromaticBond));
    m.add_bond(4, 5, BondData(kAromaticBond));
    m.add_bond(4, 8, BondData(kAromaticBond));
    m.add_bond(5, 6, BondData(kAromaticBond));
    m.add_bond(6, 7, BondData(kAromaticBond));
    m.add_bond(7, 8, BondData(kAromaticBond));
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  RingSetsFinder finder(mol);

  auto sssr = finder.find_sssr();
  EXPECT_EQ(sssr.size(), 2);
  for (const auto &ring: sssr) {
    for (int i: ring) {
      std::cout << i << ' ';
    }
    std::cout << '\n';
  }

  auto relevant = finder.find_relevant_rings();
  EXPECT_EQ(relevant.size(), 2);
  EXPECT_TRUE(std::is_permutation(sssr.begin(), sssr.end(), relevant.begin(),
                                  relevant.end(), is_permutation_no_dup<int>));
}

TEST(MoleculeAlgorithmTest, CubaneTest) {
  Molecule mol;
  {
    auto m = mol.mutator();
    for (int i = 0; i < 8; ++i) {
      m.add_atom(pt[6]);
    }
    for (int i = 0; i < 8; ++i) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
    m.add_bond(0, 1, BondData(kSingleBond));
    m.add_bond(0, 3, BondData(kSingleBond));
    m.add_bond(0, 5, BondData(kSingleBond));
    m.add_bond(1, 2, BondData(kSingleBond));
    m.add_bond(1, 6, BondData(kSingleBond));
    m.add_bond(2, 3, BondData(kSingleBond));
    m.add_bond(2, 7, BondData(kSingleBond));
    m.add_bond(3, 4, BondData(kSingleBond));
    m.add_bond(4, 5, BondData(kSingleBond));
    m.add_bond(4, 7, BondData(kSingleBond));
    m.add_bond(5, 6, BondData(kSingleBond));
    m.add_bond(6, 7, BondData(kSingleBond));
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  auto sssr = find_sssr(mol);
  EXPECT_EQ(sssr.size(), 5);
  for (const auto &ring: sssr) {
    EXPECT_EQ(ring.size(), 4);
    for (int i: ring) {
      std::cout << i << ' ';
    }
    std::cout << '\n';
  }

  auto relevant = find_relevant_rings(mol);
  EXPECT_EQ(relevant.size(), 6);
  for (const auto &ring: relevant) {
    EXPECT_EQ(ring.size(), 4);
    for (int i: ring) {
      std::cout << i << ' ';
    }
    std::cout << '\n';
  }
}
}  // namespace
}  // namespace nuri

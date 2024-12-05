//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/rings.h"

#include <algorithm>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/strings/str_join.h>

#include <gtest/gtest.h>

#include "nuri/core/molecule.h"

namespace nuri {
namespace {
// NOLINTNEXTLINE(*-using-namespace)
using namespace constants;

class IndoleRingTest: public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    {
      auto m = mol_.mutator();

      for (int i = 0; i < 8; ++i) {
        m.add_atom(kPt[6]);
      }
      m.add_atom(kPt[7]);
      for (int i = 0; i < 9; ++i) {
        mol_.atom(i).data().set_implicit_hydrogens(1);
      }
      for (int i: { 4, 5 }) {
        mol_.atom(i).data().set_implicit_hydrogens(0);
      }

      // 0-5-0 is a ring, 4-8-4 is a ring
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
  }

  static const Molecule &mol() { return mol_; }

private:
  inline static Molecule mol_;
};

class IndoleRingVerifier {
public:
  IndoleRingVerifier() { reset(); }

  void verify(const std::vector<int> &ring) {
    switch (ring.size()) {
    case 5:
      ASSERT_TRUE(five_ok_) << "5-membered ring is not allowed";
      EXPECT_TRUE(absl::c_all_of(ring, [](int i) { return 4 <= i && i < 9; }));
      disallow_five();
      break;
    case 6:
      ASSERT_TRUE(six_ok_) << "6-membered ring is not allowed";
      EXPECT_TRUE(absl::c_all_of(ring, [](int i) { return i < 6; }));
      disallow_six();
      break;
    case 9:
      ASSERT_TRUE(nine_ok_) << "9-membered ring is not allowed";
      EXPECT_TRUE(absl::c_all_of(ring, [](int i) { return i < 9; }));
      disallow_nine();
      break;
    default:
      FAIL() << "Unexpected ring size: " << ring.size();
    }
  }

  IndoleRingVerifier &reset() {
    five_ok_ = six_ok_ = nine_ok_ = true;
    return *this;
  }

  IndoleRingVerifier &disallow_five() {
    five_ok_ = false;
    return *this;
  }

  IndoleRingVerifier &disallow_six() {
    six_ok_ = false;
    return *this;
  }

  IndoleRingVerifier &disallow_nine() {
    nine_ok_ = false;
    return *this;
  }

private:
  bool five_ok_;
  bool six_ok_;
  bool nine_ok_;
};

TEST_F(IndoleRingTest, FindAll) {
  auto [rings, ok] = find_all_rings(mol());
  ASSERT_TRUE(ok);
  EXPECT_EQ(rings.size(), 3);

  IndoleRingVerifier verifier;
  for (const auto &ring: rings)
    verifier.verify(ring);

  std::tie(rings, ok) = find_all_rings(mol(), 6);
  ASSERT_TRUE(ok);
  EXPECT_EQ(rings.size(), 2);

  verifier.reset().disallow_nine();
  for (const auto &ring: rings)
    verifier.verify(ring);

  std::tie(rings, ok) = find_all_rings(mol(), 5);
  ASSERT_TRUE(ok);
  EXPECT_EQ(rings.size(), 1);

  verifier.reset().disallow_six().disallow_nine();
  verifier.verify(rings[0]);

  std::tie(rings, ok) = find_all_rings(mol(), 3);
  ASSERT_TRUE(ok);
  EXPECT_EQ(rings.size(), 0);
}

TEST_F(IndoleRingTest, FindRingSets) {
  RingSetsFinder finder(mol());

  auto sssr = finder.find_sssr();
  EXPECT_EQ(sssr.size(), 2);

  IndoleRingVerifier verifier;
  verifier.disallow_nine();
  for (const auto &ring: sssr)
    verifier.verify(ring);

  auto relevant = finder.find_relevant_rings();
  EXPECT_EQ(relevant.size(), 2);

  verifier.reset().disallow_nine();
  for (const auto &ring: sssr)
    verifier.verify(ring);
}

TEST_F(IndoleRingTest, MaxFiveRingSets) {
  RingSetsFinder finder(mol(), 5);

  auto sssr = finder.find_sssr();
  EXPECT_EQ(sssr.size(), 1);
  EXPECT_TRUE(absl::c_all_of(sssr, [](const std::vector<int> &ring) {
    return ring.size() <= 5;
  }));

  IndoleRingVerifier verifier;
  verifier.disallow_nine().disallow_six();
  verifier.verify(sssr[0]);

  auto relevant = finder.find_relevant_rings();
  EXPECT_EQ(relevant.size(), 1);
  verifier.reset().disallow_nine().disallow_six();
  verifier.verify(sssr[0]);
}

class CubaneRingTest: public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    auto m = mol_.mutator();
    for (int i = 0; i < 8; ++i) {
      m.add_atom(kPt[6]);
    }
    for (int i = 0; i < 8; ++i) {
      mol_.atom(i).data().set_implicit_hydrogens(1);
    }

    // Six relevant rings:
    // 0-1-2-3-0, 0-3-4-5-0, 0-1-6-5-0, 1-2-7-6-1, 2-3-4-7-2, 4-5-6-7-4
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

    sub_.update_atoms({ 1, 2, 4, 5, 6, 7 });
  }

  static const Molecule &mol() { return mol_; }

  static const ConstSubstructure &sub() { return sub_; }

private:
  inline static Molecule mol_;
  inline static ConstSubstructure sub_ = mol().substructure();
};

bool is_permutation_no_dup(const std::vector<int> &a,
                           const std::vector<int> &b) {
  if (a.size() != b.size())
    return false;

  absl::flat_hash_set<int> a_set(a.begin(), a.end());
  return std::all_of(b.begin(), b.end(),
                     [&a_set](int x) { return a_set.contains(x); });
}

class CubaneSubringVerifier {
public:
  CubaneSubringVerifier() { reset(); }

  void verify(const std::vector<int> &ring) {
    switch (ring.size()) {
    case 4:
      break;
    case 6:
      ASSERT_TRUE(six_ok_) << "6-membered ring is not allowed";
      EXPECT_TRUE(absl::c_all_of(ring, [](int i) { return i < 6; }));
      disallow_six();
      return;
    default:
      FAIL() << "Unexpected ring size: " << ring.size();
    }

    if (is_permutation_no_dup(ring, { 0, 1, 4, 5 })) {
      ASSERT_TRUE(first_ok_) << "First subring is not allowed";
      disallow_first();
    } else if (is_permutation_no_dup(ring, { 2, 3, 4, 5 })) {
      ASSERT_TRUE(second_ok_) << "Second subring is not allowed";
      disallow_second();
    } else {
      FAIL() << "Unexpected subring: [" << absl::StrJoin(ring, ", ") << "]";
    }
  }

  CubaneSubringVerifier &reset() {
    first_ok_ = second_ok_ = six_ok_ = true;
    return *this;
  }

  CubaneSubringVerifier &disallow_first() {
    first_ok_ = false;
    return *this;
  }

  CubaneSubringVerifier &disallow_second() {
    second_ok_ = false;
    return *this;
  }

  CubaneSubringVerifier &disallow_six() {
    six_ok_ = false;
    return *this;
  }

private:
  bool first_ok_;
  bool second_ok_;
  bool six_ok_;
};

TEST_F(CubaneRingTest, FindFromMolecule) {
  auto sssr = find_sssr(mol());
  EXPECT_EQ(sssr.size(), 5);
  for (const auto &ring: sssr) {
    EXPECT_EQ(ring.size(), 4);
  }

  auto relevant = find_relevant_rings(mol());
  EXPECT_EQ(relevant.size(), 6);
  for (const auto &ring: relevant) {
    EXPECT_EQ(ring.size(), 4);
  }

  sssr = find_sssr(mol(), 5);
  EXPECT_EQ(sssr.size(), 5);

  for (const auto &ring: sssr) {
    EXPECT_EQ(ring.size(), 4);
  }

  relevant = find_relevant_rings(mol(), 4);
  EXPECT_EQ(relevant.size(), 6);
  for (const auto &ring: relevant) {
    EXPECT_EQ(ring.size(), 4);
  }
}

TEST_F(CubaneRingTest, FindAllFromSubstructure) {
  auto [rings, ok] = find_all_rings(sub());
  ASSERT_TRUE(ok);
  EXPECT_EQ(rings.size(), 3);

  CubaneSubringVerifier verifier;
  for (const auto &ring: rings)
    verifier.verify(ring);

  std::tie(rings, ok) = find_all_rings(sub(), 5);
  ASSERT_TRUE(ok);
  EXPECT_EQ(rings.size(), 2);

  verifier.reset().disallow_six();
  for (const auto &ring: rings)
    verifier.verify(ring);
}

TEST_F(CubaneRingTest, FindSetsFromSubstructure) {
  RingSetsFinder finder(sub());

  auto sssr = finder.find_sssr();
  EXPECT_EQ(sssr.size(), 2);

  CubaneSubringVerifier verifier;
  verifier.disallow_six();
  for (const auto &ring: sssr)
    verifier.verify(ring);

  auto relevant = finder.find_relevant_rings();
  EXPECT_EQ(relevant.size(), 2);

  verifier.reset().disallow_six();
  for (const auto &ring: relevant)
    verifier.verify(ring);
}

TEST_F(CubaneRingTest, FindMaxFourSetsFromSubstructure) {
  RingSetsFinder finder(sub(), 4);

  auto sssr = finder.find_sssr();
  EXPECT_EQ(sssr.size(), 2);

  CubaneSubringVerifier verifier;
  verifier.disallow_six();
  for (const auto &ring: sssr)
    verifier.verify(ring);

  auto relevant = finder.find_relevant_rings();
  EXPECT_EQ(relevant.size(), 2);

  verifier.reset().disallow_six();
  for (const auto &ring: relevant)
    verifier.verify(ring);
}
}  // namespace
}  // namespace nuri

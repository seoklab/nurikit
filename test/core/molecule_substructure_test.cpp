//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>

#include <gtest/gtest.h>

#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"

#define NURI_TEST_CONVERTIBILITY(from_name, from_expr, to_name, to_expr,       \
                                 allowed)                                      \
  static_assert(std::is_convertible_v<from_expr, to_expr> == allowed,          \
                #from_name " to " #to_name " convertibility != " #allowed);    \
  static_assert(std::is_constructible_v<to_expr, from_expr> == allowed,        \
                #from_name " to " #to_name " constructibility != " #allowed);  \
  static_assert(std::is_assignable_v<to_expr, from_expr> == allowed,           \
                #from_name " to " #to_name " assignability != " #allowed)

#define NURI_ASSERT_ONEWAY_CONVERTIBLE(from_name, from_expr, to_name, to_expr) \
  NURI_TEST_CONVERTIBILITY(from_name, from_expr, to_name, to_expr, true);      \
  NURI_TEST_CONVERTIBILITY(to_name, to_expr, from_name, from_expr, false)

namespace nuri {
namespace {
class SubstructureTest: public ::testing::Test {
public:
  // NOLINTBEGIN(readability-identifier-naming)
  Molecule mol_;
  // Required for default-constructing Test class
  Substructure sub_ = mol_.substructure();
  // NOLINTEND(readability-identifier-naming)

protected:
  /**
   * This will build:
   *
   *         H (8)  C (0) H (9) H (10) -- N- (4) -- H (5)
   *             \   |
   *               C (2)              Na+ (11) (<-intentionally unconnected)
   *             /       \
   *  H (6) --  C (3) == C (1) -- H(7)
   *
   */
  void SetUp() override {
    mol_ = Molecule();

    {
      auto mutator = mol_.mutator();

      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[7], 0, -1 });
      for (int i = 5; i < 11; ++i) {
        mutator.add_atom({ kPt[1] });
      }
      mutator.add_atom({ kPt[11], 0, +1 });

      mutator.add_bond(0, 4, BondData { constants::kSingleBond });
      mutator.add_bond(3, 2, BondData { constants::kSingleBond });
      mutator.add_bond(3, 1, BondData { constants::kDoubleBond });
      mutator.add_bond(1, 2, BondData { constants::kSingleBond });
      mutator.add_bond(2, 0, BondData { constants::kSingleBond });
      mutator.add_bond(4, 5, BondData { constants::kSingleBond });
      mutator.add_bond(3, 6, BondData { constants::kSingleBond });
      mutator.add_bond(1, 7, BondData { constants::kSingleBond });
      mutator.add_bond(2, 8, BondData { constants::kSingleBond });
      mutator.add_bond(0, 9, BondData { constants::kSingleBond });
      mutator.add_bond(0, 10, BondData { constants::kSingleBond });
    }

    ASSERT_EQ(mol_.num_atoms(), 12);
    ASSERT_EQ(mol_.num_bonds(), 11);

    sub_ = mol_.atom_substructure({ 1, 2, 3, 10 });
    ASSERT_FALSE(sub_.empty());
  }
};

TEST_F(SubstructureTest, CreateSubstructure) {
  EXPECT_EQ(sub_.size(), 4);
  EXPECT_EQ(sub_.num_atoms(), 4);
  EXPECT_EQ(sub_.count_heavy_atoms(), 3);

  Substructure sub = mol_.atom_substructure({ 0, 1, 2, 10 });
  EXPECT_EQ(sub.size(), 4);
  EXPECT_EQ(sub.num_atoms(), 4);
  EXPECT_EQ(sub.count_heavy_atoms(), 3);

  const Molecule &cmol = mol_;
  ConstSubstructure csub1 = cmol.atom_substructure({ 0, 1, 2, 10 });
  EXPECT_EQ(csub1.size(), 4);
  EXPECT_EQ(csub1.num_atoms(), 4);
  EXPECT_EQ(csub1.count_heavy_atoms(), 3);
}

TEST_F(SubstructureTest, ClearAll) {
  sub_.name() = "test";
  sub_.add_prop("key", "val");
  sub_.set_id(10);

  sub_.clear();

  EXPECT_EQ(sub_.size(), 0);
  EXPECT_TRUE(sub_.empty());
  EXPECT_TRUE(sub_.name().empty());
  EXPECT_TRUE(sub_.props().empty());
  EXPECT_EQ(sub_.id(), 0);
}

TEST_F(SubstructureTest, ClearAtoms) {
  using std::string_literals::operator""s;

  sub_.name() = "test";
  sub_.add_prop("key", "val");
  sub_.set_id(10);

  sub_.clear_atoms();

  EXPECT_EQ(sub_.size(), 0);
  EXPECT_TRUE(sub_.empty());

  EXPECT_EQ(sub_.name(), "test");

  ASSERT_FALSE(sub_.props().empty());
  EXPECT_EQ(sub_.props().begin()[0], std::pair("key"s, "val"s));

  EXPECT_EQ(sub_.id(), 10);
}

TEST_F(SubstructureTest, UpdateAtoms) {
  std::vector<int> prev { 1, 2, 3, 10 };
  EXPECT_TRUE(absl::c_is_permutation(sub_.atom_ids(), prev));

  prev.push_back(4);
  sub_.update_atoms(internal::IndexSet(std::vector<int> { prev }));
  EXPECT_TRUE(absl::c_is_permutation(sub_.atom_ids(), prev));

  sub_.update_atoms({ 0 });
  EXPECT_TRUE(absl::c_is_permutation(sub_.atom_ids(), std::vector { 0 }));
}

TEST_F(SubstructureTest, ContainAtoms) {
  EXPECT_TRUE(sub_.contains_atom(1));
  EXPECT_TRUE(sub_.contains_atom(mol_.atom(1)));
  EXPECT_TRUE(sub_.contains_atom(2));
  EXPECT_TRUE(sub_.contains_atom(mol_.atom(2)));
  EXPECT_TRUE(sub_.contains_atom(3));
  EXPECT_TRUE(sub_.contains_atom(mol_.atom(3)));
  EXPECT_TRUE(sub_.contains_atom(10));
  EXPECT_TRUE(sub_.contains_atom(mol_.atom(10)));

  EXPECT_FALSE(sub_.contains_atom(4));
  EXPECT_FALSE(sub_.contains_atom(mol_.atom(4)));
}

TEST_F(SubstructureTest, GetAtoms) {
  auto a1 = sub_.atom(0), a2 = sub_[0];
  EXPECT_EQ(a1.id(), a2.id());
  static_assert(!std::is_const_v<std::remove_reference_t<decltype(a1.data())>>,
                "data of mutable atom should not be const");

  const Substructure &csub = sub_;
  auto a3 = csub.atom(0), a4 = csub[0];
  EXPECT_EQ(a3.id(), a4.id());
  static_assert(std::is_const_v<std::remove_reference_t<decltype(a3.data())>>,
                "data of const atom should be const");

  NURI_ASSERT_ONEWAY_CONVERTIBLE(mutable atom, decltype(a1), immutable atom,
                                 decltype(a3));
}

TEST_F(SubstructureTest, FindAtoms) {
  auto it = sub_.find_atom(1);
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->as_parent().id(), 1);

  it = sub_.find_atom(10);
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->as_parent().id(), 10);

  it = sub_.find_atom(4);
  EXPECT_EQ(it, sub_.end());

  const Substructure &csub = sub_;
  auto cit = csub.find_atom(1);
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->as_parent().id(), 1);

  cit = csub.find_atom(10);
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->as_parent().id(), 10);

  cit = csub.find_atom(4);
  EXPECT_EQ(cit, csub.end());

  it = sub_.find_atom(mol_.atom(1));
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->as_parent().id(), 1);

  it = sub_.find_atom(mol_.atom(10));
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->as_parent().id(), 10);

  it = sub_.find_atom(mol_.atom(4));
  EXPECT_EQ(it, sub_.end());

  cit = csub.find_atom(mol_.atom(1));
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->as_parent().id(), 1);

  cit = csub.find_atom(mol_.atom(10));
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->as_parent().id(), 10);

  cit = csub.find_atom(mol_.atom(4));
  EXPECT_EQ(cit, csub.end());
}

TEST_F(SubstructureTest, EraseAtoms) {
  auto sub1 = sub_;
  EXPECT_TRUE(sub1.contains_atom(1));
  sub1.erase_atom_of(1);
  EXPECT_FALSE(sub1.contains_atom(1));
  sub1.erase_atom_of(1);
  EXPECT_FALSE(sub1.contains_atom(1));
  EXPECT_EQ(sub1.size(), 3);

  sub1.erase_atoms(sub1.begin(), --sub1.end());
  EXPECT_EQ(sub1.size(), 1);

  sub1.erase_atoms_if([](const auto &) { return false; });
  EXPECT_EQ(sub1.size(), 1);

  sub1.erase_atoms_if([](const auto &) { return true; });
  EXPECT_TRUE(sub1.empty());

  auto sub2 = sub_;
  int first_id = sub_[0].as_parent().id();
  EXPECT_TRUE(sub2.contains_atom(first_id));
  sub2.erase_atom(sub_[0]);
  EXPECT_FALSE(sub2.contains_atom(first_id));
  EXPECT_EQ(sub2.size(), 3);

  auto sub3 = sub_;
  sub3.erase_atom_of(mol_.atom(1));
  EXPECT_FALSE(sub3.contains_atom(mol_.atom(1)));
  EXPECT_EQ(sub3.size(), 3);
}

TEST_F(SubstructureTest, IterateAtoms) {
  int count = 0;
  for (auto atom: sub_) {
    EXPECT_TRUE(sub_.contains_atom(atom.as_parent().id()));
    ++count;
  }
  EXPECT_EQ(count, sub_.size());

  count = 0;
  for (auto atom: std::as_const(sub_)) {
    EXPECT_TRUE(sub_.contains_atom(atom.as_parent().id()));
    ++count;
  }
  EXPECT_EQ(count, sub_.size());

  count = 0;
  for (auto rit = std::make_reverse_iterator(sub_.end()),
            rend = std::make_reverse_iterator(sub_.begin());
       rit != rend; ++rit) {
    EXPECT_TRUE(sub_.contains_atom((*rit).as_parent().id()));
    ++count;
  }
  EXPECT_EQ(count, sub_.size());

  count = 0;
  for (auto rit = std::make_reverse_iterator(sub_.cend()),
            rend = std::make_reverse_iterator(sub_.cbegin());
       rit != rend; ++rit) {
    EXPECT_TRUE(sub_.contains_atom((*rit).as_parent().id()));
    ++count;
  }
  EXPECT_EQ(count, sub_.size());
}

TEST_F(SubstructureTest, ListBonds) {
  auto bonds = sub_.bonds();
  EXPECT_EQ(bonds.size(), 3);

  auto cbonds = std::as_const(sub_).bonds();
  EXPECT_EQ(cbonds.size(), 3);
}

TEST_F(SubstructureTest, CountDegrees) {
  for (auto atom: sub_) {
    EXPECT_EQ(atom.degree(), atom.as_parent().id() == 10 ? 0 : 2)
        << atom.as_parent().id();
    EXPECT_EQ(sub_.degree(atom.id()), atom.as_parent().id() == 10 ? 0 : 2)
        << atom.as_parent().id();
  }
}

TEST_F(SubstructureTest, FindNeighbors) {
  EXPECT_FALSE(sub_.find_neighbor(sub_.atom(0), sub_.atom(1)).end());
  EXPECT_FALSE(sub_.find_neighbor(sub_.atom(1), sub_.atom(2)).end());
  EXPECT_FALSE(sub_.find_neighbor(sub_.atom(2), sub_.atom(0)).end());

  EXPECT_TRUE(sub_.find_neighbor(sub_.atom(0), sub_.atom(3)).end());

  const Substructure &csub = sub_;
  EXPECT_FALSE(csub.find_neighbor(sub_.atom(0), sub_.atom(1)).end());
  EXPECT_FALSE(csub.find_neighbor(sub_.atom(1), sub_.atom(2)).end());
  EXPECT_FALSE(csub.find_neighbor(sub_.atom(2), sub_.atom(0)).end());

  EXPECT_TRUE(csub.find_neighbor(sub_.atom(0), sub_.atom(3)).end());
}

TEST_F(SubstructureTest, IterateNeighbors) {
  int cnt = 0;
  for (auto nit = sub_.neighbor_begin(0); nit != sub_.neighbor_end(0);
       ++nit, ++cnt) {
    EXPECT_FALSE(mol_.find_neighbor(nit->src().as_parent().id(),
                                    nit->dst().as_parent().id())
                     .end());
    EXPECT_TRUE(sub_.contains_atom(nit->src().as_parent()));
    EXPECT_TRUE(sub_.contains_atom(nit->dst().as_parent()));
  }
  EXPECT_EQ(cnt, sub_.degree(0));

  cnt = 0;
  for (auto nit = sub_.neighbor_cbegin(0); nit != sub_.neighbor_cend(0);
       ++nit, ++cnt) {
    EXPECT_FALSE(mol_.find_neighbor(nit->src().as_parent().id(),
                                    nit->dst().as_parent().id())
                     .end());
    EXPECT_TRUE(sub_.contains_atom(nit->src().as_parent()));
    EXPECT_TRUE(sub_.contains_atom(nit->dst().as_parent()));
  }
  EXPECT_EQ(cnt, sub_.degree(0));

  cnt = 0;
  const Substructure &csub = sub_;
  for (auto nit = csub.neighbor_begin(0); nit != csub.neighbor_end(0);
       ++nit, ++cnt) {
    EXPECT_FALSE(mol_.find_neighbor(nit->src().as_parent().id(),
                                    nit->dst().as_parent().id())
                     .end());
    EXPECT_TRUE(csub.contains_atom(nit->src().as_parent()));
    EXPECT_TRUE(csub.contains_atom(nit->dst().as_parent()));
  }
  EXPECT_EQ(cnt, csub.degree(0));
}

TEST_F(SubstructureTest, SetProperties) {
  sub_.name() = "test";
  EXPECT_EQ(sub_.name(), "test");
  sub_.set_id(10);
  EXPECT_EQ(sub_.id(), 10);
}

class MolSubstructureTest: public ::testing::Test {
public:
  // NOLINTBEGIN(readability-identifier-naming)
  Molecule mol_;
  // NOLINTEND(readability-identifier-naming)

protected:
  /**
   * This will build:
   *
   *         H (8)  C (0) H (9) H (10) -- N- (4) -- H (5)
   *             \   |
   *               C (2)              Na+ (11) (<-intentionally unconnected)
   *             /       \
   *  H (6) --  C (3) == C (1) -- H(7)
   *
   */
  void SetUp() override {
    mol_ = Molecule();

    {
      auto mutator = mol_.mutator();

      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[6] });
      mutator.add_atom({ kPt[7], 0, -1 });
      for (int i = 5; i < 11; ++i) {
        mutator.add_atom({ kPt[1] });
      }
      mutator.add_atom({ kPt[11], 0, +1 });

      mutator.add_bond(0, 4, BondData { constants::kSingleBond });
      mutator.add_bond(3, 2, BondData { constants::kSingleBond });
      mutator.add_bond(3, 1, BondData { constants::kDoubleBond });
      mutator.add_bond(1, 2, BondData { constants::kSingleBond });
      mutator.add_bond(2, 0, BondData { constants::kSingleBond });
      mutator.add_bond(4, 5, BondData { constants::kSingleBond });
      mutator.add_bond(3, 6, BondData { constants::kSingleBond });
      mutator.add_bond(1, 7, BondData { constants::kSingleBond });
      mutator.add_bond(2, 8, BondData { constants::kSingleBond });
      mutator.add_bond(0, 9, BondData { constants::kSingleBond });
      mutator.add_bond(0, 10, BondData { constants::kSingleBond });
    }

    ASSERT_EQ(mol_.num_atoms(), 12);
    ASSERT_EQ(mol_.num_bonds(), 11);

    mol_.substructures().push_back(mol_.atom_substructure({ 1, 2, 3, 10 }));
  }
};

TEST_F(MolSubstructureTest, EraseAtomsFromMolecule) {
  Substructure &sub = mol_.substructures()[0];
  EXPECT_TRUE(sub.contains_atom(10));

  {
    auto mut = mol_.mutator();
    mut.mark_atom_erase(10);
  }

  EXPECT_EQ(sub.size(), 3);
  EXPECT_EQ(sub.num_bonds(), 3);
  EXPECT_FALSE(sub.contains_atom(10));

  {
    auto mut = mol_.mutator();
    mut.mark_atom_erase(2);
  }

  EXPECT_EQ(sub.size(), 2);
  EXPECT_TRUE(sub.contains_atom(1));
  EXPECT_TRUE(sub.contains_atom(2));

  EXPECT_EQ(sub.num_bonds(), 1);
  EXPECT_NE(sub.find_bond(sub.atom(0), sub.atom(1)), sub.bond_end());
  EXPECT_EQ(sub.bond(0).as_parent().id(), 1);
  EXPECT_EQ(sub.bond(0).data().order(), constants::kDoubleBond);
}

TEST_F(MolSubstructureTest, MergeSubstructure) {
  Substructure &sub = mol_.substructures()[0];
  mol_.merge(sub);

  ASSERT_EQ(mol_.size(), 16);
  ASSERT_EQ(mol_.num_bonds(), 14);

  EXPECT_EQ(mol_.atom(12).data().atomic_number(), 6);
  EXPECT_EQ(mol_.atom(13).data().atomic_number(), 6);
  EXPECT_EQ(mol_.atom(14).data().atomic_number(), 6);
  EXPECT_EQ(mol_.atom(15).data().atomic_number(), 1);

  EXPECT_EQ(mol_.find_bond(12, 14)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol_.find_bond(13, 12)->data().order(), constants::kSingleBond);
  EXPECT_EQ(mol_.find_bond(13, 14)->data().order(), constants::kSingleBond);

  EXPECT_EQ(mol_.num_sssr(), 2);
}

TEST_F(MolSubstructureTest, Properties) {
  Substructure &sub = mol_.substructures()[0];
  sub.add_prop("test", "1");
  auto it = absl::c_find_if(sub.props(), [](const auto &p) {
    return p == std::pair<std::string, std::string>("test", "1");
  });
  EXPECT_NE(it, sub.props().end());
}
}  // namespace

namespace internal {
template class Substructure<true>;
template class Substructure<false>;
}  // namespace internal
}  // namespace nuri

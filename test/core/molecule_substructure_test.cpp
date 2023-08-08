//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
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
const PeriodicTable &pt = PeriodicTable::get();

class SubstructureTest: public ::testing::Test {
public:
  // NOLINTBEGIN(readability-identifier-naming)
  Molecule mol_;
  // Required for default-constructing Test class
  Substructure sub_ = mol_.substructure({});
  // NOLINTEND(readability-identifier-naming)

protected:
  /**
   * This will build:
   *
   *         H (8)  C (3) H (9) H (10) -- N- (4) -- H (5)
   *             \   |
   *               C (2)              Na+ (11) (<-intentionally unconnected)
   *             /       \
   *  H (6) --  C (0) == C (1) -- H(7)
   *
   */
  void SetUp() override {
    mol_ = Molecule();

    {
      auto mutator = mol_.mutator();

      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[7], 0, -1 });
      for (int i = 5; i < 11; ++i) {
        mutator.add_atom({ pt[1] });
      }
      mutator.add_atom({ pt[11], 0, +1 });

      mutator.add_bond(0, 1, BondData { constants::kDoubleBond });
      mutator.add_bond(0, 2, BondData { constants::kSingleBond });
      mutator.add_bond(1, 2, BondData { constants::kSingleBond });
      mutator.add_bond(2, 3, BondData { constants::kSingleBond });
      mutator.add_bond(3, 4, BondData { constants::kSingleBond });
      mutator.add_bond(4, 5, BondData { constants::kSingleBond });
      mutator.add_bond(0, 6, BondData { constants::kSingleBond });
      mutator.add_bond(1, 7, BondData { constants::kSingleBond });
      mutator.add_bond(2, 8, BondData { constants::kSingleBond });
      mutator.add_bond(3, 9, BondData { constants::kSingleBond });
      mutator.add_bond(3, 10, BondData { constants::kSingleBond });
    }

    ASSERT_EQ(mol_.num_atoms(), 12);
    ASSERT_EQ(mol_.num_bonds(), 11);

    sub_ = mol_.substructure({ 0, 1, 2, 10 });
  }
};

TEST_F(SubstructureTest, CreateSubstructure) {
  EXPECT_EQ(sub_.size(), 4);
  EXPECT_EQ(sub_.num_atoms(), 4);

  std::vector<int> tmp { 0, 1, 2, 10 };
  Substructure sub = mol_.substructure(tmp);
  EXPECT_EQ(sub.size(), 4);
  EXPECT_EQ(sub.num_atoms(), 4);

  const Molecule &cmol = mol_;
  ConstSubstructure csub1 = cmol.substructure({ 0, 1, 2, 10 });
  EXPECT_EQ(csub1.size(), 4);
  EXPECT_EQ(csub1.num_atoms(), 4);

  ConstSubstructure csub2 = cmol.substructure(tmp);
  EXPECT_EQ(csub2.size(), 4);
  EXPECT_EQ(csub2.num_atoms(), 4);
}

TEST_F(SubstructureTest, ClearSubstructure) {
  EXPECT_FALSE(sub_.empty());

  sub_.clear();
  EXPECT_EQ(sub_.size(), 0);
  EXPECT_TRUE(sub_.empty());
}

TEST_F(SubstructureTest, UpdateAtoms) {
  std::vector<int> prev { 0, 1, 2, 10 };
  EXPECT_TRUE(absl::c_is_permutation(sub_.atom_ids(), prev));

  prev.push_back(3);
  sub_.update(prev);
  EXPECT_TRUE(absl::c_is_permutation(sub_.atom_ids(), prev));

  sub_.update({ 0 });
  EXPECT_TRUE(absl::c_is_permutation(sub_.atom_ids(), std::vector { 0 }));
}

TEST_F(SubstructureTest, ContainAtoms) {
  EXPECT_TRUE(sub_.contains(0));
  EXPECT_TRUE(sub_.contains(mol_.atom(0)));
  EXPECT_TRUE(sub_.contains(1));
  EXPECT_TRUE(sub_.contains(mol_.atom(1)));
  EXPECT_TRUE(sub_.contains(2));
  EXPECT_TRUE(sub_.contains(mol_.atom(2)));
  EXPECT_TRUE(sub_.contains(10));
  EXPECT_TRUE(sub_.contains(mol_.atom(10)));

  EXPECT_FALSE(sub_.contains(4));
  EXPECT_FALSE(sub_.contains(mol_.atom(4)));
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
  auto it = sub_.find_atom(0);
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->id(), 0);

  it = sub_.find_atom(10);
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->id(), 10);

  it = sub_.find_atom(4);
  EXPECT_EQ(it, sub_.end());

  const Substructure &csub = sub_;
  auto cit = csub.find_atom(0);
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->id(), 0);

  cit = csub.find_atom(10);
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->id(), 10);

  cit = csub.find_atom(4);
  EXPECT_EQ(cit, csub.end());

  it = sub_.find_atom(mol_.atom(0));
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->id(), 0);

  it = sub_.find_atom(mol_.atom(10));
  EXPECT_NE(it, sub_.end());
  EXPECT_EQ(it->id(), 10);

  it = sub_.find_atom(mol_.atom(4));
  EXPECT_EQ(it, sub_.end());

  cit = csub.find_atom(mol_.atom(0));
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->id(), 0);

  cit = csub.find_atom(mol_.atom(10));
  EXPECT_NE(cit, csub.end());
  EXPECT_EQ(cit->id(), 10);

  cit = csub.find_atom(mol_.atom(4));
  EXPECT_EQ(cit, csub.end());
}

TEST_F(SubstructureTest, EraseAtoms) {
  auto sub1 = sub_;
  EXPECT_TRUE(sub1.contains(0));
  sub1.erase_atom(0);
  EXPECT_FALSE(sub1.contains(0));
  sub1.erase_atom(0);
  EXPECT_FALSE(sub1.contains(0));
  EXPECT_EQ(sub1.size(), 3);

  sub1.erase_atoms(sub1.begin(), --sub1.end());
  EXPECT_EQ(sub1.size(), 1);

  sub1.erase_atoms([](const auto &) { return false; });
  EXPECT_EQ(sub1.size(), 1);

  sub1.erase_atoms([](const auto &) { return true; });
  EXPECT_TRUE(sub1.empty());

  auto sub2 = sub_;
  int first_id = sub_[0].id();
  EXPECT_TRUE(sub2.contains(first_id));
  sub2.erase_atom(sub_[0]);
  EXPECT_FALSE(sub2.contains(first_id));
  EXPECT_EQ(sub2.size(), 3);

  auto sub3 = sub_;
  sub3.erase_atom(mol_.atom(0));
  EXPECT_FALSE(sub3.contains(mol_.atom(0)));
  EXPECT_EQ(sub3.size(), 3);
}

TEST_F(SubstructureTest, IterateAtoms) {
  int count = 0;
  for (auto atom: sub_) {
    EXPECT_TRUE(sub_.contains(atom.id()));
    ++count;
  }
  EXPECT_EQ(count, sub_.size());

  count = 0;
  for (auto atom: std::as_const(sub_)) {
    EXPECT_TRUE(sub_.contains(atom.id()));
    ++count;
  }
  EXPECT_EQ(count, sub_.size());

  count = 0;
  for (auto rit = std::make_reverse_iterator(sub_.end()),
            rend = std::make_reverse_iterator(sub_.begin());
       rit != rend; ++rit) {
    EXPECT_TRUE(sub_.contains(rit->id()));
    ++count;
  }
  EXPECT_EQ(count, sub_.size());

  count = 0;
  for (auto rit = std::make_reverse_iterator(sub_.cend()),
            rend = std::make_reverse_iterator(sub_.cbegin());
       rit != rend; ++rit) {
    EXPECT_TRUE(sub_.contains(rit->id()));
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
    EXPECT_EQ(atom.degree(), atom.id() == 10 ? 0 : 2) << atom.id();
    EXPECT_EQ(sub_.degree(atom.id()), atom.id() == 10 ? 0 : 2) << atom.id();
  }
}

TEST_F(SubstructureTest, FindNeighbors) {
  EXPECT_FALSE(sub_.find_neighbor(0, 1).end());
  EXPECT_FALSE(sub_.find_neighbor(1, 2).end());
  EXPECT_FALSE(sub_.find_neighbor(2, 0).end());

  EXPECT_TRUE(sub_.find_neighbor(0, 6).end());
  EXPECT_TRUE(sub_.find_neighbor(0, 10).end());

  const Substructure &csub = sub_;
  EXPECT_FALSE(csub.find_neighbor(0, 1).end());
  EXPECT_FALSE(csub.find_neighbor(1, 2).end());
  EXPECT_FALSE(csub.find_neighbor(2, 0).end());

  EXPECT_TRUE(csub.find_neighbor(0, 6).end());
  EXPECT_TRUE(csub.find_neighbor(0, 10).end());
}

TEST_F(SubstructureTest, IterateNeighbors) {
  int cnt = 0;
  for (auto nit = sub_.neighbor_begin(0); nit != sub_.neighbor_end(0);
       ++nit, ++cnt) {
    EXPECT_FALSE(mol_.find_neighbor(nit->src().id(), nit->dst().id()).end());
    EXPECT_TRUE(sub_.contains(nit->src().id()));
    EXPECT_TRUE(sub_.contains(nit->dst().id()));
  }
  EXPECT_EQ(cnt, sub_.degree(0));

  cnt = 0;
  for (auto nit = sub_.neighbor_cbegin(0); nit != sub_.neighbor_cend(0);
       ++nit, ++cnt) {
    EXPECT_FALSE(mol_.find_neighbor(nit->src().id(), nit->dst().id()).end());
    EXPECT_TRUE(sub_.contains(nit->src().id()));
    EXPECT_TRUE(sub_.contains(nit->dst().id()));
  }
  EXPECT_EQ(cnt, sub_.degree(0));

  cnt = 0;
  const Substructure &csub = sub_;
  for (auto nit = csub.neighbor_begin(0); nit != csub.neighbor_end(0);
       ++nit, ++cnt) {
    EXPECT_FALSE(mol_.find_neighbor(nit->src().id(), nit->dst().id()).end());
    EXPECT_TRUE(csub.contains(nit->src().id()));
    EXPECT_TRUE(csub.contains(nit->dst().id()));
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
   *         H (8)  C (3) H (9) H (10) -- N- (4) -- H (5)
   *             \   |
   *               C (2)              Na+ (11) (<-intentionally unconnected)
   *             /       \
   *  H (6) --  C (0) == C (1) -- H(7)
   *
   */
  void SetUp() override {
    mol_ = Molecule();

    {
      auto mutator = mol_.mutator();

      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[7], 0, -1 });
      for (int i = 5; i < 11; ++i) {
        mutator.add_atom({ pt[1] });
      }
      mutator.add_atom({ pt[11], 0, +1 });

      mutator.add_bond(0, 1, BondData { constants::kDoubleBond });
      mutator.add_bond(0, 2, BondData { constants::kSingleBond });
      mutator.add_bond(1, 2, BondData { constants::kSingleBond });
      mutator.add_bond(2, 3, BondData { constants::kSingleBond });
      mutator.add_bond(3, 4, BondData { constants::kSingleBond });
      mutator.add_bond(4, 5, BondData { constants::kSingleBond });
      mutator.add_bond(0, 6, BondData { constants::kSingleBond });
      mutator.add_bond(1, 7, BondData { constants::kSingleBond });
      mutator.add_bond(2, 8, BondData { constants::kSingleBond });
      mutator.add_bond(3, 9, BondData { constants::kSingleBond });
      mutator.add_bond(3, 10, BondData { constants::kSingleBond });
    }

    ASSERT_EQ(mol_.num_atoms(), 12);
    ASSERT_EQ(mol_.num_bonds(), 11);

    mol_.add_substructure({ 0, 1, 2, 10 });
  }
};

TEST_F(MolSubstructureTest, AddSubstructures) {
  EXPECT_TRUE(mol_.has_substructure());
  EXPECT_EQ(mol_.num_substructures(), 1);

  Substructure &sub1 = mol_.add_substructure();
  EXPECT_EQ(mol_.num_substructures(), 2);
  EXPECT_TRUE(sub1.empty());
  sub1.add_atom(10);
  EXPECT_EQ(sub1.size(), 1);

  Substructure &sub2 = mol_.add_substructure({ 0, 1, 2, 10 });
  EXPECT_EQ(mol_.num_substructures(), 3);
  EXPECT_EQ(sub2.size(), 4);

  std::vector<int> ids = { 0, 1, 2, 10 };
  Substructure &sub3 = mol_.add_substructure(ids);
  EXPECT_EQ(mol_.num_substructures(), 4);
  EXPECT_EQ(sub3.size(), 4);
}

TEST_F(MolSubstructureTest, ClearSubstructures) {
  EXPECT_TRUE(mol_.has_substructure());
  mol_.clear_substructures();
  EXPECT_FALSE(mol_.has_substructure());
}

TEST_F(MolSubstructureTest, GetSubstructures) {
  Substructure &sub = mol_.get_substructure(0);
  EXPECT_EQ(sub.size(), 4);

  const Substructure &csub = std::as_const(mol_).get_substructure(0);
  EXPECT_EQ(csub.size(), 4);

  EXPECT_EQ(mol_.substructures().size(), mol_.num_substructures());
}

TEST_F(MolSubstructureTest, EraseSubstructures) {
  Substructure &sub = mol_.get_substructure(0);
  EXPECT_EQ(sub.size(), 4);

  mol_.erase_substructure(0);
  EXPECT_FALSE(mol_.has_substructure());

  mol_.add_substructure({ 0, 1, 2, 10 });
  mol_.erase_substructures([](const auto &) { return false; });
  EXPECT_TRUE(mol_.has_substructure());
  mol_.erase_substructures([](const auto &) { return true; });
  EXPECT_FALSE(mol_.has_substructure());
}

TEST_F(MolSubstructureTest, FindSubstructures) {
  mol_.get_substructure(0).set_id(100);
  mol_.add_substructure().set_id(200);
  mol_.add_substructure({ 10 }).set_id(200);

  {
    auto empty = mol_.find_substructures(0);
    EXPECT_EQ(empty.begin(), empty.end());
    EXPECT_EQ(std::as_const(empty).begin(), std::as_const(empty).end());

    auto has_one = mol_.find_substructures(100);
    EXPECT_EQ(std::distance(has_one.begin(), has_one.end()), 1);
    EXPECT_EQ(std::distance(std::as_const(has_one).begin(),
                            std::as_const(has_one).end()),
              1);
    EXPECT_EQ(has_one.begin()->size(), mol_.get_substructure(0).size());

    auto has_two = mol_.find_substructures(200);
    EXPECT_EQ(std::distance(has_two.begin(), has_two.end()), 2);
    EXPECT_EQ(std::distance(std::as_const(has_two).begin(),
                            std::as_const(has_two).end()),
              2);

    auto all = mol_.find_substructures("");
    EXPECT_EQ(std::distance(all.begin(), all.end()), 3);
  }

  {
    const Molecule &cmol = mol_;

    auto empty = cmol.find_substructures(0);
    EXPECT_EQ(empty.begin(), empty.end());

    auto has_one = cmol.find_substructures(100);
    EXPECT_EQ(std::distance(has_one.begin(), has_one.end()), 1);
    EXPECT_EQ(has_one.begin()->size(), cmol.get_substructure(0).size());

    auto has_two = cmol.find_substructures(200);
    EXPECT_EQ(std::distance(has_two.begin(), has_two.end()), 2);

    auto all = cmol.find_substructures("");
    EXPECT_EQ(std::distance(all.begin(), all.end()), 3);
  }
}

NURI_ASSERT_ONEWAY_CONVERTIBLE(substurcture iterator, Substructure::iterator,
                               const substurcture iterator,
                               Substructure::const_iterator);

TEST_F(MolSubstructureTest, EraseNodesFromMolecule) {
  Substructure &sub = mol_.get_substructure(0);
  EXPECT_TRUE(sub.contains(10));

  {
    auto mut = mol_.mutator();
    mut.mark_atom_erase(10);
  }

  EXPECT_EQ(sub.size(), 3);
  EXPECT_FALSE(sub.contains(10));

  {
    auto mut = mol_.mutator();
    mut.mark_atom_erase(1);
  }

  EXPECT_EQ(sub.size(), 2);
  EXPECT_TRUE(sub.contains(1));
  EXPECT_FALSE(sub.contains(2));
}
}  // namespace

namespace internal {
template class Substructure<true>;
template class Substructure<false>;
}  // namespace internal
}  // namespace nuri

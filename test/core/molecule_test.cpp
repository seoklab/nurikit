//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"

namespace {
using nuri::AtomData;
using nuri::BondData;
using nuri::Molecule;

// NOLINTNEXTLINE(*-using-namespace)
using namespace nuri::constants;

const nuri::PeriodicTable &pt = nuri::PeriodicTable::get();

TEST(Basic2DMoleculeTest, CreationTest) {
  Molecule empty;

  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.num_atoms(), 0);
  EXPECT_EQ(empty.num_bonds(), 0);

  std::vector<nuri::AtomData> atoms(1);
  atoms.reserve(10);
  for (int i = 1; i < 10; ++i) {
    atoms.push_back(nuri::AtomData(pt[i], kSP3, 0, 0, i));
  }

  Molecule ten(atoms.begin(), atoms.end());

  EXPECT_EQ(ten.size(), 10);
  EXPECT_EQ(ten.num_atoms(), 10);
  EXPECT_EQ(ten.num_bonds(), 0);

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(ten.atom(i).data(), atoms[i]);
  }
}

TEST(Basic2DMoleculeTest, AddAtomsTest) {
  Molecule m;
  {
    auto mutator = m.mutator();
    for (int i = 0; i < 10; ++i) {
      mutator.add_atom(nuri::AtomData(pt[i], kSP3, 0, 0, i * 2));
    }
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(m.atom(i).data().atomic_number(), i);
  }
}

TEST(Basic2DMoleculeTest, AddBondsTest) {
  std::vector<nuri::AtomData> atoms(1);
  atoms.reserve(10);
  for (int i = 1; i < 10; ++i) {
    atoms.push_back(nuri::AtomData(pt[i], kSP3, 0, 0, i));
  }

  Molecule ten(atoms.begin(), atoms.end());
  {
    auto mutator = ten.mutator();
    EXPECT_TRUE(mutator.add_bond(0, 1, BondData(kSingleBond)));

    EXPECT_FALSE(mutator.add_bond(0, 0, BondData(kSingleBond)));
    EXPECT_FALSE(mutator.add_bond(1, 0, BondData(kDoubleBond)));
  }
  {
    auto mutator = ten.mutator();
    EXPECT_FALSE(mutator.add_bond(1, 0, BondData(kDoubleBond)));
  }

  EXPECT_EQ(ten.num_bonds(), 1);

  auto b = ten.find_bond(1, 0);
  EXPECT_NE(b, ten.bond_end());
  EXPECT_EQ(b->data().order(), kSingleBond);
}

class MoleculeTest: public ::testing::Test {
public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  Molecule mol_;

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

      mutator.add_atom({ pt[6], kSP2 });
      mutator.add_atom({ pt[6], kSP2 });
      mutator.add_atom({ pt[6], kSP3 });
      mutator.add_atom({ pt[6], kSP3 });
      mutator.add_atom({ pt[7], kSP3, -1 });
      for (int i = 5; i < 11; ++i) {
        mutator.add_atom({ pt[1], kTerminal });
      }
      mutator.add_atom({ pt[11], kTerminal, +1 });

      mutator.add_bond(0, 1, { kDoubleBond });
      mutator.add_bond(0, 2, { kSingleBond });
      mutator.add_bond(1, 2, { kSingleBond });
      mutator.add_bond(2, 3, { kSingleBond });
      mutator.add_bond(3, 4, { kSingleBond });
      mutator.add_bond(4, 5, { kSingleBond });
      mutator.add_bond(0, 6, { kSingleBond });
      mutator.add_bond(1, 7, { kSingleBond });
      mutator.add_bond(2, 8, { kSingleBond });
      mutator.add_bond(3, 9, { kSingleBond });
      mutator.add_bond(3, 10, { kSingleBond });
    }

    ASSERT_EQ(mol_.num_atoms(), 12);
    ASSERT_EQ(mol_.num_bonds(), 11);

    // The coordinates are generated by corina
    nuri::MatrixX3d coords1(12, 3);
    coords1 << -1.1913, 0.5880, -2.2656,  //
      -0.1528, 1.3191, -2.5866,           //
      -0.7527, 1.5927, -1.2315,           //
      -0.0127, 1.0858, 0.0080,            //
      -0.7003, 1.5596, 1.2166,            //
      -0.2304, 1.2378, 2.0496,            //
      -1.8821, -0.1932, -2.5467,          //
      0.6062, 1.5595, -3.3164,            //
      -1.3450, 2.4981, -1.0997,           //
      0.0021, -0.0041, 0.0020,            //
      1.0099, 1.4631, 0.0003,             //
      6.0099, 7.4981, 7.0496;
    mol_.add_pos(coords1);

    mol_.erase_pos(0);
    mol_.add_pos(std::move(coords1));

    nuri::MatrixX3d coords2(12, 3);
    coords2 << -2.8265, 2.3341, -0.0040,  //
      -2.4325, 1.6004, 1.0071,            //
      -1.3392, 2.1190, 0.1089,            //
      -0.4723, 3.2756, 0.6106,            //
      -1.2377, 4.5275, 0.5399,            //
      -0.6865, 5.3073, 0.8661,            //
      -3.6475, 2.7827, -0.5437,           //
      -2.7033, 1.0234, 1.8791,            //
      -0.8591, 1.4229, -0.5788,           //
      -0.1776, 3.0877, 1.6430,            //
      0.4184, 3.3588, -0.0122,            //
      10.1054, 2.3061, 4.3717;
    mol_.add_pos(std::move(coords2));

    ASSERT_EQ(mol_.npos(), 2);
  }
};

TEST_F(MoleculeTest, TransformTest) {
  nuri::Affine3d trs = nuri::Affine3d::Identity();

  {
    Molecule mol(mol_);
    ASSERT_EQ(mol.npos(), mol_.npos());
    mol.transform(trs);
    for (int i = 0; i < mol.npos(); ++i) {
      EXPECT_TRUE(mol.pos(i).isApprox(mol_.pos(i)));
    }
  }

  trs.translation() << 5, 0, 0;
  trs.linear() = nuri::AngleAxisd(90, { 0, 0, 1 }).to_matrix();

  {
    Molecule mol(mol_);
    mol.transform(trs);
    for (int i = 0; i < mol.npos(); ++i) {
      EXPECT_TRUE(
        mol.pos(i).transpose().isApprox(trs * mol_.pos(i).transpose()));
    }
  }

  {
    Molecule mol(mol_);
    mol.transform(1, trs);
    EXPECT_TRUE(mol.pos(0).isApprox(mol_.pos(0)));
    EXPECT_TRUE(mol.pos(1).transpose().isApprox(trs * mol_.pos(1).transpose()));
  }
}

TEST_F(MoleculeTest, RotateBondTest) {
  Molecule mol_all(mol_);

  // Unconnected
  ASSERT_FALSE(mol_all.rotate_bond(0, 7, 90));
  // Not a rotable bond
  auto bid = mol_all.find_bond(0, 1)->id();
  ASSERT_FALSE(mol_all.rotate_bond(bid, 90));

  // Now real rotation
  ASSERT_TRUE(mol_all.rotate_bond(2, 3, 90));

  std::vector<int> fixed { 0, 1, 2, 3, 6, 7, 8, 11 };
  for (int i = 0; i < mol_all.npos(); ++i) {
    EXPECT_TRUE(mol_all.pos(i)(fixed, Eigen::all)
                  .isApprox(mol_.pos(i)(fixed, Eigen::all)));
  }

  std::vector<int> rotated { 4, 5, 9, 10 };
  Eigen::MatrixX3d result_coords(4, 3);

  // Rotated with UCSF Chimera
  result_coords << -0.5602, -0.2180, 0.4060,  //
    -0.0876, -0.5744, 1.2232,                 //
    1.0480, 0.9803, -0.2199,                  //
    -0.1401, 1.7976, 0.8236;
  EXPECT_TRUE(mol_all.pos(0)(rotated, Eigen::all).isApprox(result_coords, 1e-3))
    << mol_all.pos(0)(rotated, Eigen::all);

  // Rotated with UCSF Chimera
  result_coords << -0.6589, 3.4347, 2.0590,  //
    -0.0965, 4.1932, 2.4150,                 //
    0.5757, 3.0620, 0.4005,                  //
    -0.7645, 4.1952, 0.1036;

  EXPECT_TRUE(mol_all.pos(1)(rotated, Eigen::all).isApprox(result_coords, 1e-3))
    << mol_all.pos(1)(rotated, Eigen::all);

  Molecule mol_one(mol_all);

  // Not rotable
  bid = mol_one.find_bond(0, 1)->id();
  ASSERT_FALSE(mol_one.rotate_bond(1, bid, 90));
  // Rotate reverse!
  ASSERT_TRUE(mol_one.rotate_bond(0, 2, 3, -90));

  EXPECT_TRUE(mol_one.pos(0).isApprox(mol_.pos(0)));
  EXPECT_TRUE(mol_one.pos(1).isApprox(mol_all.pos(1)));
}

TEST_F(MoleculeTest, RemoveAtomsTest) {
  int predicted_size;

  Molecule mol1(mol_);
  {
    auto m = mol1.mutator();
    m.erase_atom(9);
    m.erase_atom(10);
    m.erase_atom(11);

    predicted_size = m.num_atoms();
  }
  EXPECT_EQ(mol1.num_atoms(), predicted_size);
  EXPECT_EQ(mol1.num_atoms(), 9);
  EXPECT_EQ(mol1.num_bonds(), 9);

  for (int i = 0; i < mol1.npos(); ++i) {
    EXPECT_TRUE(mol1.pos(i).isApprox(mol_.pos(i).topRows<9>()));
  }

  Molecule mol2(mol_);
  {
    auto m = mol2.mutator();
    m.erase_atom(0);
    m.erase_atom(1);
    m.erase_atom(2);

    predicted_size = m.num_atoms();
  }
  EXPECT_EQ(mol2.num_atoms(), predicted_size);
  EXPECT_EQ(mol2.num_atoms(), 9);
  EXPECT_EQ(mol2.num_bonds(), 4);

  for (int i = 0; i < mol2.npos(); ++i) {
    EXPECT_TRUE(mol2.pos(i).isApprox(mol_.pos(i).bottomRows<9>()));
  }

  Molecule mol3(mol_);
  {
    auto m = mol3.mutator();
    m.erase_atom(0);
    m.erase_atom(4);
    m.erase_atom(9);

    predicted_size = m.num_atoms();
  }
  EXPECT_EQ(mol3.num_atoms(), predicted_size);
  EXPECT_EQ(mol3.num_atoms(), 9);
  EXPECT_EQ(mol3.num_bonds(), 5);

  std::vector<int> keep { 1, 2, 3, 5, 6, 7, 8, 10, 11 };
  for (int i = 0; i < mol3.npos(); ++i) {
    EXPECT_TRUE(mol3.pos(i).isApprox(mol_.pos(i)(keep, Eigen::all)));
  }
}

TEST_F(MoleculeTest, RemoveBondsTest) {
  Molecule mol1(mol_);
  {
    // All nop
    auto m = mol1.mutator();
    m.erase_bond(8, 8);
    m.erase_bond(8, 9);
  }
  EXPECT_EQ(mol1.num_atoms(), 12);
  EXPECT_EQ(mol1.num_bonds(), 11);

  Molecule mol2(mol_);
  {
    auto m = mol2.mutator();
    m.erase_bond(0, 1);
  }
  EXPECT_EQ(mol2.num_atoms(), 12);
  EXPECT_EQ(mol2.num_bonds(), 10);

  Molecule mol3(std::move(mol_));
  {
    auto m = mol3.mutator();
    m.erase_bond(0, 1);
    m.erase_bond(8, 9);  // nop
    m.erase_bond(3, 2);
  }
  EXPECT_EQ(mol3.num_atoms(), 12);
  EXPECT_EQ(mol3.num_bonds(), 9);
}
}  // namespace

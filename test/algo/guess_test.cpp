//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/guess.h"

#include <absl/container/flat_hash_set.h>
#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"

namespace nuri {
namespace {
TEST(GuessTypesTest, Histidine) {
  Molecule mol;
  mol.reserve(11);

  {
    auto mut = mol.mutator();

    mut.add_atom(kPt[7]);  // N   (0)
    mut.add_atom(kPt[6]);  // CA  (1)
    mut.add_atom(kPt[6]);  // C   (2)
    mut.add_atom(kPt[8]);  // O   (3)
    mut.add_atom(kPt[6]);  // CB  (4)
    mut.add_atom(kPt[6]);  // CG  (5)
    mut.add_atom(kPt[7]);  // ND1 (6)
    mut.add_atom(kPt[6]);  // CD2 (7)
    mut.add_atom(kPt[6]);  // CE1 (8)
    mut.add_atom(kPt[7]);  // NE2 (9)
    mut.add_atom(kPt[8]);  // OXT (10)
    mut.add_bond(0, 1, {});
    mut.add_bond(1, 2, {});
    mut.add_bond(1, 4, {});
    mut.add_bond(2, 3, {});
    mut.add_bond(2, 10, {});
    mut.add_bond(4, 5, {});
    mut.add_bond(5, 6, {});
    mut.add_bond(5, 7, {});
    mut.add_bond(6, 8, {});
    mut.add_bond(7, 9, {});
    mut.add_bond(8, 9, {});
  }

  // Taken from pdb standard AA data
  Matrix3Xd pos(3, 11);
  pos.transpose() << 33.472, 42.685, -4.610,  //
      33.414, 41.686, -5.673,                 //
      33.773, 42.279, -7.040,                 //
      33.497, 43.444, -7.337,                 //
      32.005, 41.080, -5.734,                 //
      31.888, 39.902, -6.651,                 //
      32.539, 38.710, -6.414,                 //
      31.199, 39.734, -7.804,                 //
      32.251, 37.857, -7.382,                 //
      31.439, 38.453, -8.237,                 //
      34.382, 41.455, -7.879;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_all_types(mol));

  // N
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 2);

  // CA
  EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(1).data().implicit_hydrogens(), 1);

  // C
  EXPECT_EQ(mol.atom(2).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(2).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(2).data().is_conjugated());

  // O
  EXPECT_EQ(mol.atom(3).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(3).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(3).data().is_conjugated());

  // OXT
  EXPECT_EQ(mol.atom(10).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(10).data().implicit_hydrogens(), 1);
  EXPECT_TRUE(mol.atom(10).data().is_conjugated());

  // CB
  EXPECT_EQ(mol.atom(4).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(4).data().implicit_hydrogens(), 2);

  // imidazole ring
  int implicit_hydrogens = 0;
  for (int i = 5; i < 10; ++i) {
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2);
    EXPECT_TRUE(mol.atom(i).data().is_aromatic());
    implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(implicit_hydrogens, 3);
}

TEST(GuessTypesTest, Tyrosine) {
  Molecule mol;
  mol.reserve(13);

  {
    auto mut = mol.mutator();

    mut.add_atom(kPt[7]);  // N   (0)
    mut.add_atom(kPt[6]);  // CA  (1)
    mut.add_atom(kPt[6]);  // C   (2)
    mut.add_atom(kPt[8]);  // O   (3)
    mut.add_atom(kPt[6]);  // CB  (4)
    mut.add_atom(kPt[6]);  // CG  (5)
    mut.add_atom(kPt[6]);  // CD1 (6)
    mut.add_atom(kPt[6]);  // CD2 (7)
    mut.add_atom(kPt[6]);  // CE1 (8)
    mut.add_atom(kPt[6]);  // CE2 (9)
    mut.add_atom(kPt[6]);  // CZ  (10)
    mut.add_atom(kPt[8]);  // OH  (11)
    mut.add_atom(kPt[8]);  // OXT (12)
    mut.add_bond(0, 1, {});
    mut.add_bond(1, 2, {});
    mut.add_bond(1, 4, {});
    mut.add_bond(2, 3, {});
    mut.add_bond(2, 12, {});
    mut.add_bond(4, 5, {});
    mut.add_bond(5, 6, {});
    mut.add_bond(5, 7, {});
    mut.add_bond(6, 8, {});
    mut.add_bond(7, 9, {});
    mut.add_bond(8, 10, {});
    mut.add_bond(9, 10, {});
    mut.add_bond(10, 11, {});
  }

  // Taken from pdb standard AA data
  Matrix3Xd pos(3, mol.size());
  pos.transpose() << 5.005, 5.256, 15.563,  //
      5.326, 6.328, 16.507,                 //
      4.742, 7.680, 16.116,                 //
      4.185, 8.411, 16.947,                 //
      6.836, 6.389, 16.756,                 //
      7.377, 5.438, 17.795,                 //
      6.826, 5.370, 19.075,                 //
      8.493, 4.624, 17.565,                 //
      7.308, 4.536, 20.061,                 //
      9.029, 3.816, 18.552,                 //
      8.439, 3.756, 19.805,                 //
      8.954, 2.936, 20.781,                 //
      4.840, 8.051, 14.829;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_all_types(mol));

  // N
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 2);

  // CA
  EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(1).data().implicit_hydrogens(), 1);

  // C
  EXPECT_EQ(mol.atom(2).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(2).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(2).data().is_conjugated());

  // O
  EXPECT_EQ(mol.atom(3).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(3).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(3).data().is_conjugated());

  // OXT
  EXPECT_EQ(mol.atom(12).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(12).data().implicit_hydrogens(), 1);
  EXPECT_TRUE(mol.atom(12).data().is_conjugated());

  // CB
  EXPECT_EQ(mol.atom(4).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(4).data().implicit_hydrogens(), 2);

  // benzene ring
  int implicit_hydrogens = 0;
  for (int i = 5; i < 11; ++i) {
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2);
    EXPECT_TRUE(mol.atom(i).data().is_aromatic());
    implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(implicit_hydrogens, 4);

  // OH
  EXPECT_EQ(mol.atom(11).data().hybridization(), constants::kSP2);
  EXPECT_TRUE(mol.atom(11).data().is_conjugated());
}

TEST(GuessTypesTest, Tryptophan) {
  Molecule mol;
  mol.reserve(15);

  {
    auto mut = mol.mutator();

    mut.add_atom(kPt[7]);  // N   (0)
    mut.add_atom(kPt[6]);  // CA  (1)
    mut.add_atom(kPt[6]);  // C   (2)
    mut.add_atom(kPt[8]);  // O   (3)
    mut.add_atom(kPt[6]);  // CB  (4)
    mut.add_atom(kPt[6]);  // CG  (5)
    mut.add_atom(kPt[6]);  // CD1 (6)
    mut.add_atom(kPt[6]);  // CD2 (7)
    mut.add_atom(kPt[7]);  // NE1 (8)
    mut.add_atom(kPt[6]);  // CE2 (9)
    mut.add_atom(kPt[6]);  // CE3 (10)
    mut.add_atom(kPt[6]);  // CZ2 (11)
    mut.add_atom(kPt[6]);  // CZ3 (12)
    mut.add_atom(kPt[6]);  // CH2 (13)
    mut.add_atom(kPt[8]);  // OXT (14)
    mut.add_bond(0, 1, {});
    mut.add_bond(1, 2, {});
    mut.add_bond(1, 4, {});
    mut.add_bond(2, 3, {});
    mut.add_bond(2, 14, {});
    mut.add_bond(4, 5, {});
    mut.add_bond(5, 6, {});
    mut.add_bond(5, 7, {});
    mut.add_bond(6, 8, {});
    mut.add_bond(7, 9, {});
    mut.add_bond(7, 10, {});
    mut.add_bond(8, 9, {});
    mut.add_bond(9, 11, {});
    mut.add_bond(10, 12, {});
    mut.add_bond(11, 13, {});
    mut.add_bond(12, 13, {});
  }

  // Taken from pdb standard AA data
  Matrix3Xd pos(3, mol.size());
  pos.transpose() << 1.278, 1.121, 2.059,  //
      -0.008, 0.417, 1.970,                //
      -0.490, 0.076, 3.357,                //
      0.308, -0.130, 4.240,                //
      0.168, -0.868, 1.161,                //
      0.650, -0.526, -0.225,               //
      1.928, -0.418, -0.622,               //
      -0.186, -0.256, -1.396,              //
      1.978, -0.095, -1.951,               //
      0.701, 0.014, -2.454,                //
      -1.564, -0.210, -1.615,              //
      0.190, 0.314, -3.712,                //
      -2.044, 0.086, -2.859,               //
      -1.173, 0.348, -3.907,               //
      -1.806, 0.001, 3.610;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_all_types(mol));

  // N
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 2);

  // CA
  EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(1).data().implicit_hydrogens(), 1);

  // C
  EXPECT_EQ(mol.atom(2).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(2).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(2).data().is_conjugated());

  // O
  EXPECT_EQ(mol.atom(3).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(3).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(3).data().is_conjugated());

  // OXT
  EXPECT_EQ(mol.atom(12).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(12).data().implicit_hydrogens(), 1);
  EXPECT_TRUE(mol.atom(12).data().is_conjugated());

  // CB
  EXPECT_EQ(mol.atom(4).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(4).data().implicit_hydrogens(), 2);

  // imidazole ring
  int implicit_hydrogens = 0;
  for (int i = 5; i < 10; ++i) {
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2);
    EXPECT_TRUE(mol.atom(i).data().is_aromatic());
    implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(implicit_hydrogens, 2);

  // benzene ring (remainder)
  implicit_hydrogens = 0;
  for (int i = 10; i < 14; ++i) {
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2);
    EXPECT_TRUE(mol.atom(i).data().is_aromatic());
    implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(implicit_hydrogens, 4);
}

TEST(GuessBondsTest, Histidine) {
  Molecule mol;
  mol.reserve(11);

  auto mut = mol.mutator();
  mut.add_atom(kPt[7]);  // N   (0)
  mut.add_atom(kPt[6]);  // CA  (1)
  mut.add_atom(kPt[6]);  // C   (2)
  mut.add_atom(kPt[8]);  // O   (3)
  mut.add_atom(kPt[6]);  // CB  (4)
  mut.add_atom(kPt[6]);  // CG  (5)
  mut.add_atom(kPt[7]);  // ND1 (6)
  mut.add_atom(kPt[6]);  // CD2 (7)
  mut.add_atom(kPt[6]);  // CE1 (8)
  mut.add_atom(kPt[7]);  // NE2 (9)
  mut.add_atom(kPt[8]);  // OXT (10)

  // Taken from pdb standard AA data
  Matrix3Xd pos(3, 11);
  pos.transpose() << 33.472, 42.685, -4.610,  //
      33.414, 41.686, -5.673,                 //
      33.773, 42.279, -7.040,                 //
      33.497, 43.444, -7.337,                 //
      32.005, 41.080, -5.734,                 //
      31.888, 39.902, -6.651,                 //
      32.539, 38.710, -6.414,                 //
      31.199, 39.734, -7.804,                 //
      32.251, 37.857, -7.382,                 //
      31.439, 38.453, -8.237,                 //
      34.382, 41.455, -7.879;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_bonds(mut));

  EXPECT_EQ(mol.num_bonds(), 11);

  // N
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 2);

  // CA
  EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(1).data().implicit_hydrogens(), 1);

  // C
  EXPECT_EQ(mol.atom(2).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(2).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(2).data().is_conjugated());

  // O
  EXPECT_EQ(mol.atom(3).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(3).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(3).data().is_conjugated());

  // OXT
  EXPECT_EQ(mol.atom(10).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(10).data().implicit_hydrogens(), 1);
  EXPECT_TRUE(mol.atom(10).data().is_conjugated());

  // CB
  EXPECT_EQ(mol.atom(4).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(4).data().implicit_hydrogens(), 2);

  // imidazole ring
  int implicit_hydrogens = 0;
  for (int i = 5; i < 10; ++i) {
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2);
    EXPECT_TRUE(mol.atom(i).data().is_aromatic());
    implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(implicit_hydrogens, 3);
}

TEST(GuessBondsTest, Tryptophan) {
  Molecule mol;
  mol.reserve(15);

  auto mut = mol.mutator();

  mut.add_atom(kPt[7]);  // N   (0)
  mut.add_atom(kPt[6]);  // CA  (1)
  mut.add_atom(kPt[6]);  // C   (2)
  mut.add_atom(kPt[8]);  // O   (3)
  mut.add_atom(kPt[6]);  // CB  (4)
  mut.add_atom(kPt[6]);  // CG  (5)
  mut.add_atom(kPt[6]);  // CD1 (6)
  mut.add_atom(kPt[6]);  // CD2 (7)
  mut.add_atom(kPt[7]);  // NE1 (8)
  mut.add_atom(kPt[6]);  // CE2 (9)
  mut.add_atom(kPt[6]);  // CE3 (10)
  mut.add_atom(kPt[6]);  // CZ2 (11)
  mut.add_atom(kPt[6]);  // CZ3 (12)
  mut.add_atom(kPt[6]);  // CH2 (13)
  mut.add_atom(kPt[8]);  // OXT (14)

  // Taken from pdb standard AA data
  Matrix3Xd pos(3, mol.size());
  pos.transpose() << 1.278, 1.121, 2.059,  //
      -0.008, 0.417, 1.970,                //
      -0.490, 0.076, 3.357,                //
      0.308, -0.130, 4.240,                //
      0.168, -0.868, 1.161,                //
      0.650, -0.526, -0.225,               //
      1.928, -0.418, -0.622,               //
      -0.186, -0.256, -1.396,              //
      1.978, -0.095, -1.951,               //
      0.701, 0.014, -2.454,                //
      -1.564, -0.210, -1.615,              //
      0.190, 0.314, -3.712,                //
      -2.044, 0.086, -2.859,               //
      -1.173, 0.348, -3.907,               //
      -1.806, 0.001, 3.610;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_bonds(mut));

  // N
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 2);

  // CA
  EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(1).data().implicit_hydrogens(), 1);

  // C
  EXPECT_EQ(mol.atom(2).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(2).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(2).data().is_conjugated());

  // O
  EXPECT_EQ(mol.atom(3).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(3).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(3).data().is_conjugated());

  // OXT
  EXPECT_EQ(mol.atom(12).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(12).data().implicit_hydrogens(), 1);
  EXPECT_TRUE(mol.atom(12).data().is_conjugated());

  // CB
  EXPECT_EQ(mol.atom(4).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(4).data().implicit_hydrogens(), 2);

  // imidazole ring
  int implicit_hydrogens = 0;
  for (int i = 5; i < 10; ++i) {
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2);
    EXPECT_TRUE(mol.atom(i).data().is_aromatic());
    implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(implicit_hydrogens, 2);

  // benzene ring (remainder)
  implicit_hydrogens = 0;
  for (int i = 10; i < 14; ++i) {
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2);
    EXPECT_TRUE(mol.atom(i).data().is_aromatic());
    implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(implicit_hydrogens, 4);
}

// beta-L-Fucosylazide
TEST(GuessBondsTest, FUY) {
  Molecule mol;
  mol.reserve(13);

  auto mut = mol.mutator();

  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);

  // Taken from rcsb
  Matrix3Xd pos(3, mol.size());
  pos.transpose() << -4.2030, -27.2040, 56.8430,  //
      -4.6330, -26.5100, 58.0570,                 //
      -2.6810, -27.3580, 56.8270,                 //
      -5.3960, -27.0960, 58.8540,                 //
      -2.1160, -27.7280, 58.0920,                 //
      -2.3990, -28.4080, 55.7610,                 //
      -6.1620, -27.7360, 59.6050,                 //
      -1.0020, -28.7280, 55.7270,                 //
      -2.8920, -27.8720, 54.4110,                 //
      -1.9860, -26.8680, 53.9410,                 //
      -4.3070, -27.2710, 54.4690,                 //
      -4.5320, -26.4830, 55.6530,                 //
      -4.5690, -26.4120, 53.2330;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_bonds(mut));
  ASSERT_EQ(mol.num_bonds(), 13);

  EXPECT_EQ(mol.atom(3).data().hybridization(), constants::kSP);
  EXPECT_EQ(mol.atom(3).data().formal_charge(), 1);
  EXPECT_EQ(mol.atom(6).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(6).data().formal_charge(), -1);

  int total_implicit_hydrogens = 0;
  for (auto atom: mol) {
    if (atom.data().atomic_number() == 7)
      continue;

    EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
    EXPECT_EQ(atom.data().hybridization(), constants::kSP3) << atom.id();
    total_implicit_hydrogens += atom.data().implicit_hydrogens();
  }
  EXPECT_EQ(total_implicit_hydrogens, 11);

  for (auto bond: mol.bonds()) {
    EXPECT_EQ(bond.data().order(), bond.src() == 3 || bond.dst() == 3
                                       ? constants::kDoubleBond
                                       : constants::kSingleBond);
  }
}

// 4-Nitro-2-phenoxymethanesulfonanilide
TEST(GuessBondsTest, NIM) {
  Molecule mol;
  mol.reserve(21);

  auto mut = mol.mutator();

  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[16]);
  mut.add_atom(kPt[8]);

  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);

  mut.add_atom(kPt[6]);

  // Taken from rcsb
  Matrix3Xd pos(3, mol.size());
  pos.transpose() << 8.1410, 27.7780, -6.9650,  //
      8.1570, 26.5560, -6.8040,                 //
      8.0500, 26.1580, -5.6500,                 //
      8.2710, 25.6740, -7.9090,                 //
      8.2060, 24.2770, -7.8230,                 //
      8.3150, 23.4770, -8.9690,                 //
      8.5430, 24.0800, -10.2140,                //
      8.4920, 23.4960, -11.4270,                //
      7.2690, 22.5120, -11.7500,                //
      7.3990, 21.2210, -11.1070,                //
      7.2090, 22.3730, -13.1820,                //
      5.9130, 23.3120, -11.2360,                //
      8.4450, 26.2430, -9.1680,                 //
      8.5590, 25.4500, -10.3030,                //
      8.7130, 25.9910, -11.5540,                //
      9.2750, 27.2030, -11.7810,                //
      10.5570, 27.4740, -11.3250,               //
      11.1050, 28.7320, -11.5760,               //
      10.3580, 29.6920, -12.2730,               //
      9.0760, 29.3960, -12.7290,                //
      8.5430, 28.1400, -12.4810;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_bonds(mut));
  ASSERT_EQ(mol.num_bonds(), 22);

  EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(1).data().formal_charge(), 1);
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(2).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(0).data().formal_charge()
                + mol.atom(2).data().formal_charge(),
            -1);
  EXPECT_EQ(mol.find_bond(1, 0)->data().order()
                + mol.find_bond(1, 2)->data().order(),
            3);

  for (int i: { 3, 4, 5, 6, 12, 13, 15, 16, 17, 18, 19, 20 }) {
    const AtomData &data = mol.atom(i).data();
    EXPECT_EQ(data.hybridization(), constants::kSP2) << i;
    EXPECT_EQ(data.implicit_hydrogens(), 3 - mol.atom(i).degree()) << i;
    EXPECT_EQ(data.formal_charge(), 0) << i;
    EXPECT_TRUE(data.is_aromatic()) << i;
  }

  EXPECT_EQ(mol.atom(7).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(7).data().formal_charge(), 0);
  EXPECT_EQ(mol.atom(7).data().implicit_hydrogens(), 1);

  EXPECT_EQ(mol.atom(14).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(14).data().formal_charge(), 0);
  EXPECT_EQ(mol.atom(14).data().implicit_hydrogens(), 0);

  EXPECT_EQ(mol.find_bond(8, 9)->data().order(), constants::kDoubleBond);
  EXPECT_EQ(mol.find_bond(8, 10)->data().order(), constants::kDoubleBond);
}

// 2'-Fluoroguanylyl-(3'-5')-phosphocytidine
TEST(GuessBondsTest, GPC) {
  Molecule mol;
  mol.reserve(40);

  auto mut = mol.mutator();

  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);

  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[9]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);

  mut.add_atom(kPt[15]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);

  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[8]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[7]);
  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[6]);

  // Taken from pdb 1rds
  Matrix3Xd pos(3, mol.size());
  pos.transpose() << 26.275, 37.320, 33.197,  //
      27.383, 38.106, 33.276,                 //
      27.408, 38.983, 34.275,                 //
      28.355, 37.939, 32.372,                 //
      28.217, 36.988, 31.399,                 //
      27.079, 36.174, 31.300,                 //
      26.058, 36.316, 32.270,                 //
      24.970, 35.774, 32.406,                 //
      27.216, 35.337, 30.249,                 //
      28.414, 35.598, 29.655,                 //
      29.021, 36.613, 30.350,                 //
      30.361, 37.192, 30.044,                 //
      30.266, 38.609, 29.489,                 //
      30.809, 38.687, 28.212,                 //
      30.873, 39.414, 30.601,                 //
      31.759, 40.469, 30.051,                 //
      31.718, 38.412, 31.393,                 //
      31.040, 37.197, 31.293,                 //
      31.683, 38.844, 32.867,                 //
      32.474, 38.056, 33.727,                 //
      31.101, 41.893, 29.823,                 //
      30.901, 42.148, 28.397,                 //
      29.851, 41.882, 30.633,                 //
      32.063, 42.925, 30.508,                 //
      33.444, 43.126, 30.080,                 //
      33.750, 44.635, 30.192,                 //
      32.824, 45.283, 31.040,                 //
      35.153, 45.045, 30.568,                 //
      35.870, 45.856, 29.633,                 //
      34.928, 45.916, 31.766,                 //
      35.965, 46.901, 31.791,                 //
      33.508, 46.460, 31.504,                 //
      32.943, 46.841, 32.841,                 //
      32.592, 48.142, 33.139,                 //
      32.684, 49.083, 32.375,                 //
      32.091, 48.360, 34.404,                 //
      31.975, 47.390, 35.333,                 //
      31.473, 47.686, 36.540,                 //
      32.337, 46.094, 35.043,                 //
      32.832, 45.818, 33.770;
  mol.add_conf(std::move(pos));

  ASSERT_TRUE(guess_bonds(mut));
  ASSERT_EQ(mol.num_bonds(), 44);

  int total_implicit_hydrogens = 0;
  for (int i: { 0, 1, 3, 4, 5, 6 }) {
    const AtomData &data = mol.atom(i).data();
    EXPECT_EQ(data.hybridization(), constants::kSP2) << i;
    EXPECT_TRUE(data.is_aromatic()) << i;
    total_implicit_hydrogens += data.implicit_hydrogens();
  }
  EXPECT_EQ(total_implicit_hydrogens, 1);

  EXPECT_EQ(mol.atom(2).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(2).data().implicit_hydrogens(), 2);
  EXPECT_TRUE(mol.atom(2).data().is_conjugated());
  EXPECT_EQ(mol.atom(7).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(7).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(7).data().is_conjugated());

  for (int i: { 4, 5, 8, 9, 10 }) {
    const AtomData &data = mol.atom(i).data();
    EXPECT_EQ(data.hybridization(), constants::kSP2);
    EXPECT_TRUE(data.is_aromatic());
    EXPECT_EQ(data.implicit_hydrogens(), i == 9);
  }

  EXPECT_EQ(mol.atom(20).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(20).data().implicit_hydrogens(), 0);
  EXPECT_EQ(sum_bond_order(mol.atom(20)), 5);

  total_implicit_hydrogens = 0;
  for (int i: { 15, 21, 22, 23 }) {
    EXPECT_FALSE(mol.find_neighbor(20, i).end());
    total_implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
  }
  EXPECT_EQ(total_implicit_hydrogens, 1);

  for (int i: { 32, 33, 35, 36, 38, 39 }) {
    const AtomData &data = mol.atom(i).data();
    EXPECT_EQ(data.hybridization(), constants::kSP2);
    EXPECT_TRUE(data.is_aromatic());
    EXPECT_EQ(data.implicit_hydrogens(), i == 38 || i == 39);
  }
  EXPECT_EQ(mol.atom(34).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(34).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(34).data().is_conjugated());
  EXPECT_EQ(mol.atom(37).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(37).data().implicit_hydrogens(), 2);
  EXPECT_TRUE(mol.atom(37).data().is_conjugated());

  for (int i:
       { 11, 12, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31 })
    EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP3) << i;

  for (auto bond: mol.bonds()) {
    if (bond.data().is_aromatic())
      continue;

    // Checked already in the previous steps
    if (bond.src() == 20 || bond.dst() == 20)
      continue;
    if (bond.src() == 7 || bond.dst() == 7 || bond.src() == 34
        || bond.dst() == 34)
      continue;

    EXPECT_EQ(bond.data().order(), 1);
  }
}
}  // namespace
}  // namespace nuri

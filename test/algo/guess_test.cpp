//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/guess.h"

#include <fstream>
#include <string>
#include <string_view>

#include <absl/algorithm/container.h>
#include <absl/strings/ascii.h>
#include <absl/strings/str_split.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "fmt_test_common.h"
#include "test_utils.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/pdb.h"
#include "nuri/fmt/smiles.h"

namespace nuri {
namespace {
class AATestParam {
public:
  virtual ~AATestParam() noexcept = default;

  virtual const char *name() const = 0;

  virtual Molecule templ() const = 0;

  virtual std::vector<std::pair<int, int>> bonds() const = 0;

  virtual void verify_types(const Molecule &mol) const = 0;
};

void verify_types_cooh(Molecule::Atom carbon) {
  // C
  EXPECT_EQ(carbon.data().hybridization(), constants::kSP2);
  EXPECT_EQ(carbon.data().implicit_hydrogens(), 0);
  EXPECT_TRUE(carbon.data().is_conjugated());

  // =O
  auto oxo = absl::c_find_if(carbon, [](Molecule::Neighbor nei) {
    return nei.dst().data().atomic_number() == 8
           && nei.edge_data().order() == constants::kDoubleBond;
  });
  ASSERT_FALSE(oxo.end());
  EXPECT_EQ(oxo->dst().data().hybridization(), constants::kTerminal);
  EXPECT_EQ(oxo->dst().data().implicit_hydrogens(), 0);
  EXPECT_TRUE(oxo->dst().data().is_conjugated());

  // -OH
  auto oh = absl::c_find_if(carbon, [](Molecule::Neighbor nei) {
    return nei.dst().data().atomic_number() == 8
           && nei.edge_data().order() == constants::kSingleBond;
  });
  ASSERT_FALSE(oh.end());
  EXPECT_EQ(oh->dst().data().hybridization(), constants::kSP2);
  EXPECT_EQ(oh->dst().data().implicit_hydrogens(), 1);
  EXPECT_TRUE(oh->dst().data().is_conjugated());
}

void verify_types_common(const Molecule &mol) {
  // N
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 2);
  EXPECT_FALSE(mol.atom(0).data().is_conjugated());

  // CA
  EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(1).data().implicit_hydrogens(), 1);
  EXPECT_FALSE(mol.atom(1).data().is_conjugated());

  verify_types_cooh(mol.atom(2));
}

class ARGTestParam: public AATestParam {
  const char *name() const override { return "ARG"; }

  Molecule templ() const override {
    Molecule mol;

    auto mut = mol.mutator();
    mol.reserve(12);
    mol.reserve_bonds(11);

    mut.add_atom(kPt[7]);  // N   (0)
    mut.add_atom(kPt[6]);  // CA  (1)
    mut.add_atom(kPt[6]);  // C   (2)
    mut.add_atom(kPt[8]);  // O   (3)
    mut.add_atom(kPt[6]);  // CB  (4)
    mut.add_atom(kPt[6]);  // CG  (5)
    mut.add_atom(kPt[6]);  // CD  (6)
    mut.add_atom(kPt[7]);  // NE  (7)
    mut.add_atom(kPt[6]);  // CZ  (8)
    mut.add_atom(kPt[7]);  // NH1 (9)
    mut.add_atom(kPt[7]);  // NH2 (10)
    mut.add_atom(kPt[8]);  // OXT (11)

    // Taken from pdb standard AA data
    Matrix3Xd pos(3, mol.size());
    pos.transpose() << 69.812, 14.685, 89.810,  //
        70.052, 14.573, 91.280,                 //
        71.542, 14.389, 91.604,                 //
        72.354, 14.342, 90.659,                 //
        69.227, 13.419, 91.854,                 //
        67.722, 13.607, 91.686,                 //
        66.952, 12.344, 92.045,                 //
        67.307, 11.224, 91.178,                 //
        66.932, 9.966, 91.380,                  //
        66.176, 9.651, 92.421,                  //
        67.344, 9.015, 90.554,                  //
        71.901, 14.320, 92.798;
    mol.confs().push_back(std::move(pos));

    return mol;
  }

  std::vector<std::pair<int, int>> bonds() const override {
    return {
      { 0,  1 },
      { 1,  2 },
      { 1,  4 },
      { 2,  3 },
      { 2, 11 },
      { 4,  5 },
      { 5,  6 },
      { 6,  7 },
      { 7,  8 },
      { 8,  9 },
      { 8, 10 },
    };
  }

  void verify_types(const Molecule &mol) const override {
    verify_types_common(mol);

    // CB-CD
    for (int i: { 4, 5, 6 }) {
      EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP3) << i;
      EXPECT_EQ(mol.atom(i).data().implicit_hydrogens(), 2) << i;
    }

    // NE
    EXPECT_EQ(mol.atom(7).data().hybridization(), constants::kSP2);
    EXPECT_EQ(mol.atom(7).data().implicit_hydrogens(), 1);
    EXPECT_TRUE(mol.atom(7).data().is_conjugated());

    // CZ
    EXPECT_EQ(mol.atom(8).data().hybridization(), constants::kSP2);
    EXPECT_EQ(mol.atom(8).data().implicit_hydrogens(), 0);
    EXPECT_TRUE(mol.atom(8).data().is_conjugated());

    // NH1-2
    int implicit_hydrogens = 0;
    for (int i: { 9, 10 }) {
      EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP2) << i;
      EXPECT_TRUE(mol.atom(i).data().is_conjugated()) << i;
      implicit_hydrogens += mol.atom(i).data().implicit_hydrogens();
    }
    EXPECT_EQ(implicit_hydrogens, 3);
  }
};

class HISTestParam: public AATestParam {
  const char *name() const override { return "HIS"; }

  Molecule templ() const override {
    Molecule mol;

    auto mut = mol.mutator();
    mol.reserve(11);
    mol.reserve_bonds(11);

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
    Matrix3Xd pos(3, mol.size());
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
    mol.confs().push_back(std::move(pos));

    return mol;
  }

  std::vector<std::pair<int, int>> bonds() const override {
    return {
      { 0,  1 },
      { 1,  2 },
      { 1,  4 },
      { 2,  3 },
      { 2, 10 },
      { 4,  5 },
      { 5,  6 },
      { 5,  7 },
      { 6,  8 },
      { 7,  9 },
      { 8,  9 },
    };
  }

  void verify_types(const Molecule &mol) const override {
    verify_types_common(mol);

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
};

class PROTestParam: public AATestParam {
  const char *name() const override { return "PRO"; }

  Molecule templ() const override {
    Molecule mol;

    auto mut = mol.mutator();
    mol.reserve(8);
    mol.reserve_bonds(8);

    mut.add_atom(kPt[7]);  // N   (0)
    mut.add_atom(kPt[6]);  // CA  (1)
    mut.add_atom(kPt[6]);  // C   (2)
    mut.add_atom(kPt[8]);  // O   (3)
    mut.add_atom(kPt[6]);  // CB  (4)
    mut.add_atom(kPt[6]);  // CG  (5)
    mut.add_atom(kPt[6]);  // CD  (6)
    mut.add_atom(kPt[8]);  // OXT (7)

    // Taken from pdb standard AA data
    Matrix3Xd pos(3, mol.size());
    pos.transpose() << 39.165, 37.768, 82.966,  //
        38.579, 38.700, 82.008,                 //
        37.217, 39.126, 82.515,                 //
        36.256, 38.332, 82.370,                 //
        38.491, 37.874, 80.720,                 //
        38.311, 36.445, 81.200,                 //
        38.958, 36.358, 82.579,                 //
        37.131, 40.263, 83.047;
    mol.confs().push_back(std::move(pos));

    return mol;
  }

  std::vector<std::pair<int, int>> bonds() const override {
    return {
      { 0, 1 },
      { 0, 6 },
      { 1, 2 },
      { 1, 4 },
      { 2, 3 },
      { 2, 7 },
      { 4, 5 },
      { 5, 6 },
    };
  }

  void verify_types(const Molecule &mol) const override {
    // N
    EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);
    EXPECT_EQ(mol.atom(0).data().implicit_hydrogens(), 1);
    EXPECT_FALSE(mol.atom(0).data().is_conjugated());

    // CA
    EXPECT_EQ(mol.atom(1).data().hybridization(), constants::kSP3);
    EXPECT_EQ(mol.atom(1).data().implicit_hydrogens(), 1);
    EXPECT_FALSE(mol.atom(1).data().is_conjugated());

    verify_types_cooh(mol.atom(2));

    // CB-CD
    for (int i: { 4, 5, 6 }) {
      EXPECT_EQ(mol.atom(i).data().hybridization(), constants::kSP3) << i;
      EXPECT_EQ(mol.atom(i).data().implicit_hydrogens(), 2) << i;
      EXPECT_FALSE(mol.atom(i).data().is_conjugated()) << i;
    }
  }
};

class TYRTestParam: public AATestParam {
  const char *name() const override { return "TYR"; }

  Molecule templ() const override {
    Molecule mol;

    auto mut = mol.mutator();
    mol.reserve(13);
    mol.reserve_bonds(13);

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
    mol.confs().push_back(std::move(pos));

    return mol;
  }

  std::vector<std::pair<int, int>> bonds() const override {
    return {
      {  0,  1 },
      {  1,  2 },
      {  1,  4 },
      {  2,  3 },
      {  2, 12 },
      {  4,  5 },
      {  5,  6 },
      {  5,  7 },
      {  6,  8 },
      {  7,  9 },
      {  8, 10 },
      {  9, 10 },
      { 10, 11 },
    };
  }

  void verify_types(const Molecule &mol) const override {
    verify_types_common(mol);

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
};

class TRPTestParam: public AATestParam {
  const char *name() const override { return "TRP"; }

  Molecule templ() const override {
    Molecule mol;

    auto mut = mol.mutator();
    mol.reserve(15);
    mol.reserve_bonds(16);

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
    pos.transpose() << 74.708, 60.512, 32.843,  //
        74.400, 61.735, 32.114,                 //
        73.588, 61.411, 30.840,                 //
        72.939, 62.292, 30.277,                 //
        75.684, 62.473, 31.706,                 //
        76.675, 62.727, 32.832,                 //
        77.753, 61.964, 33.157,                 //
        76.646, 63.805, 33.777,                 //
        78.403, 62.494, 34.247,                 //
        77.741, 63.625, 34.650,                 //
        75.796, 64.902, 33.974,                 //
        78.014, 64.499, 35.709,                 //
        76.065, 65.776, 35.031,                 //
        77.168, 65.565, 35.884,                 //
        73.495, 60.470, 30.438;
    mol.confs().push_back(std::move(pos));

    return mol;
  }

  std::vector<std::pair<int, int>> bonds() const override {
    return {
      {  0,  1 },
      {  1,  2 },
      {  1,  4 },
      {  2,  3 },
      {  2, 14 },
      {  4,  5 },
      {  5,  6 },
      {  5,  7 },
      {  6,  8 },
      {  7,  9 },
      {  7, 10 },
      {  8,  9 },
      {  9, 11 },
      { 10, 12 },
      { 11, 13 },
      { 12, 13 },
    };
  }

  void verify_types(const Molecule &mol) const override {
    verify_types_common(mol);

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
};

template <class T>
class GuessAATest: public ::testing::Test {
public:
  void SetUp() override {
    param_ = T();

    const AATestParam &param = param_;
    mol_ = param.templ();
    mol_.name() = param.name();
    bonds_ = param.bonds();
  }

  Molecule &mol() { return mol_; }

  void add_bonds() {
    auto mut = mol_.mutator();
    for (auto [src, dst]: bonds_)
      mut.add_bond(src, dst, {});
  }

  void verify_connectivity() const {
    EXPECT_EQ(mol_.num_bonds(), bonds_.size());
    for (auto [src, dst]: bonds_)
      EXPECT_NE(mol_.find_bond(src, dst), mol_.bond_end())
          << "Bond not found: " << src << " - " << dst;
  }

  void verify_types() const {
    static_cast<const AATestParam &>(param_).verify_types(mol_);
  }

private:
  T param_;
  Molecule mol_;
  std::vector<std::pair<int, int>> bonds_;
};

using AATestParams = ::testing::Types<ARGTestParam, HISTestParam, PROTestParam,
                                      TYRTestParam, TRPTestParam>;
TYPED_TEST_SUITE(GuessAATest, AATestParams);

TYPED_TEST(GuessAATest, GuessConnectivity) {
  Molecule &mol = this->mol();
  {
    auto mut = mol.mutator();
    guess_connectivity(mut);
  }
  this->verify_connectivity();
}

TYPED_TEST(GuessAATest, GuessTypes) {
  this->add_bonds();
  ASSERT_TRUE(guess_all_types(this->mol()));
  this->verify_types();
}

TYPED_TEST(GuessAATest, GuessEverything) {
  Molecule &mol = this->mol();
  {
    auto mut = mol.mutator();
    ASSERT_TRUE(guess_everything(mut));
  }
  this->verify_types();
}

// MET-GLN, taken from 1UBQ (last oxygen was ILE N in the original file)
TEST(GuessSelectedMolecules, MetGln) {
  Molecule mol;
  mol.reserve(18);

  auto mut = mol.mutator();
  for (int z: { 7, 6, 6, 8, 6, 6, 16, 6, 7, 6, 6, 8, 6, 6, 6, 8, 7, 8 })
    mut.add_atom(kPt[z]);

  Matrix3Xd pos(3, mol.size());
  pos.transpose() << 27.340, 24.430, 2.614,  //
      26.266, 25.413, 2.842,                 //
      26.913, 26.639, 3.531,                 //
      27.886, 26.463, 4.263,                 //
      25.112, 24.880, 3.649,                 //
      25.353, 24.860, 5.134,                 //
      23.930, 23.959, 5.904,                 //
      24.447, 23.984, 7.620,                 //
      26.335, 27.770, 3.258,                 //
      26.850, 29.021, 3.898,                 //
      26.100, 29.253, 5.202,                 //
      24.865, 29.024, 5.330,                 //
      26.733, 30.148, 2.905,                 //
      26.882, 31.546, 3.409,                 //
      26.786, 32.562, 2.270,                 //
      27.783, 33.160, 1.870,                 //
      25.562, 32.733, 1.806,                 //
      26.849, 29.656, 6.217;
  mol.confs().push_back(std::move(pos));

  ASSERT_TRUE(guess_everything(mut));
  ASSERT_EQ(mol.num_bonds(), 17);

  // MET
  // N-term
  EXPECT_EQ(mol.atom(0).data().hybridization(), constants::kSP3);

  // O
  EXPECT_EQ(mol.atom(3).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(3).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(3).data().is_conjugated());

  // SD
  EXPECT_EQ(mol.atom(6).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(6).data().implicit_hydrogens(), 0);
  EXPECT_FALSE(mol.atom(6).data().is_conjugated());

  // CE
  EXPECT_EQ(mol.atom(7).data().hybridization(), constants::kSP3);
  EXPECT_EQ(mol.atom(7).data().implicit_hydrogens(), 3);
  EXPECT_FALSE(mol.atom(7).data().is_conjugated());

  // GLN
  // N
  EXPECT_EQ(mol.atom(8).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(8).data().implicit_hydrogens(), 1);
  EXPECT_TRUE(mol.atom(8).data().is_conjugated());

  // CD
  EXPECT_EQ(mol.atom(14).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(14).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(14).data().is_conjugated());

  // OE1
  EXPECT_EQ(mol.atom(15).data().hybridization(), constants::kTerminal);
  EXPECT_EQ(mol.atom(15).data().implicit_hydrogens(), 0);
  EXPECT_TRUE(mol.atom(15).data().is_conjugated());

  // NE2
  EXPECT_EQ(mol.atom(16).data().hybridization(), constants::kSP2);
  EXPECT_EQ(mol.atom(16).data().implicit_hydrogens(), 2);
  EXPECT_TRUE(mol.atom(16).data().is_conjugated());

  // C-term
  verify_types_cooh(mol.atom(10));
}

// beta-L-Fucosylazide
TEST(GuessSelectedMolecules, FUY) {
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
  mol.confs().push_back(std::move(pos));

  ASSERT_TRUE(guess_everything(mut));
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
    EXPECT_EQ(bond.data().order(), bond.src().id() == 3 || bond.dst().id() == 3
                                       ? constants::kDoubleBond
                                       : constants::kSingleBond);
  }
}

// 4-Nitro-2-phenoxymethanesulfonanilide
TEST(GuessSelectedMolecules, NIM) {
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
  mol.confs().push_back(std::move(pos));

  ASSERT_TRUE(guess_everything(mut));
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
TEST(GuessSelectedMolecules, GPC) {
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
  mol.confs().push_back(std::move(pos));

  ASSERT_TRUE(guess_everything(mut));
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
    EXPECT_EQ(data.implicit_hydrogens(), i == 38 || i == 39) << i;
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
    if (bond.src().id() == 20 || bond.dst().id() == 20)
      continue;
    if (bond.src().id() == 7 || bond.dst().id() == 7 || bond.src().id() == 34
        || bond.dst().id() == 34)
      continue;

    EXPECT_EQ(bond.data().order(), 1);
  }
}

// GH-302
TEST(GuessSelectedMolecules, PDB6xmk) {
  Molecule mol;
  mol.reserve(33);

  auto mut = mol.mutator();

  mut.add_atom(kPt[6]);
  mut.add_atom(kPt[16]);
  for (int i = 0; i < 21; ++i)
    mut.add_atom(kPt[6]);
  for (int i = 0; i < 2; ++i)
    mut.add_atom(kPt[9]);
  for (int i = 0; i < 3; ++i)
    mut.add_atom(kPt[7]);
  for (int i = 0; i < 5; ++i)
    mut.add_atom(kPt[8]);

  Matrix3Xd pos(3, mol.size());
  pos.transpose() << 10.290, 15.829, 27.600,  //
      10.374, 17.435, 26.821,                 //
      10.655, 30.154, 23.992,                 //
      9.358, 29.459, 23.448,                  //
      9.109, 28.075, 24.025,                  //
      10.325, 27.117, 24.014,                 //
      10.035, 25.811, 24.817,                 //
      11.408, 24.001, 25.646,                 //
      12.440, 22.085, 26.767,                 //
      13.590, 21.827, 27.785,                 //
      14.951, 22.333, 27.321,                 //
      16.072, 22.070, 28.371,                 //
      15.321, 21.714, 25.972,                 //
      11.126, 21.566, 27.314,                 //
      9.317, 19.931, 27.372,                  //
      8.223, 19.796, 26.192,                  //
      7.777, 21.146, 25.652,                  //
      6.832, 21.040, 24.469,                  //
      5.849, 22.685, 25.853,                  //
      6.982, 21.976, 26.669,                  //
      9.568, 18.547, 28.025,                  //
      11.582, 27.781, 24.564,                 //
      11.877, 29.222, 24.125,                 //
      10.975, 31.209, 23.207,                 //
      10.425, 30.759, 25.194,                 //
      12.332, 23.532, 26.492,                 //
      10.536, 20.420, 26.876,                 //
      5.784, 21.883, 24.588,                  //
      11.169, 25.347, 25.570,                 //
      7.086, 20.253, 23.535,                  //
      8.348, 18.039, 28.498,                  //
      10.543, 22.261, 28.153,                 //
      10.753, 23.234, 24.926;
  mol.confs().push_back(std::move(pos));

  ASSERT_TRUE(guess_everything(mut));

  EXPECT_EQ(mol.num_bonds(), 34);

  std::vector<int> sp2_atoms = { 7, 25, 28, 17, 27, 13, 26 };
  absl::c_sort(sp2_atoms);

  for (auto atom: mol) {
    EXPECT_EQ(atom.data().formal_charge(), 0);

    if (all_neighbors(atom) == 1) {
      EXPECT_EQ(atom.data().hybridization(), constants::kTerminal);
    } else if (absl::c_binary_search(sp2_atoms, atom.id())) {
      EXPECT_EQ(atom.data().hybridization(), constants::kSP2);
    } else {
      EXPECT_EQ(atom.data().hybridization(), constants::kSP3);
    }
  }
}

TEST(GuessSelectedMolecules, GH358) {
  std::vector<std::string_view> smiles_answers {
    // clang-format off
    "CSC(CS(c1ccccc1)(=O)=O)C(NC(C(NC(N2CCNCC2)=O)CC(C)C)=O)CCc3ccccc3	1mem_lig",
    "CSC(=N)C(COCc1ccccc1)NC(C(NC(=O)N2CCOCC2)CC(C)C)=O	1ms6_lig",
    "CSC(C(CCCC)NC(=O)OC1CN(CC1(C)C)C(=O)Nc2c(C)noc2C)(O)C(=O)NC(C)c3ccccc3	2bdl_lig",
    "N1(CCOCC1)C(=O)NC(C(=O)NC(CCc2ccccc2)CCS(c3ccccc3)(=O)=O)CS(CC=4C(C)ONC=4C)(=O)=O	2fye_lig-bcq",
    "CSC(CS(c1ccccc1)(=O)=O)C(NC(C(NC(N2CCOCC2)=O)CCS(c3ccccc3)(=O)=O)=O)CCc4ccccc4	2g6d_lig",
    "CSC=1CC(OC(c2c(cc(cc2O)OC)C=CCC(C(C(=O)C=1)O)O)=O)C	4gs6_lig",
    "CSC1C2(C(C(C(O)=C1C#N)C)CCc3c(-c4cccnc4)n(C)nc32)C	5daf_lig",
    "COC(C(C1SCC(C)=C(C(O)=O)N1)NC(=O)Cc2cccs2)=O	5zg6_lig",
    "C(=O)(N1C(CCC1)C(O)=CC(=O)O)O	6m06_lig-bwf",
    "C(=C1N=C(C(=C1CCC(=O)O)C)C=C2NC(C(=C2C)CC)=O)c3[nH]c(C=C4NC(C(C4CC)C)=O)c(c3CCC(=O)O)C	6oap_lig-cyc",
    "COC(C(NC(=O)C(c1csc(N)n1)=NOC)C2SCC(=C(N2)C(O)=O)CSC=3N(C)NC(C(N=3)=O)=O)=O	6p54_lig",
    "COC(C(NC(C(=NOCC(=O)O)c1csc(N)n1)=O)C2NC(=C(CS2)C=C)C(=O)O)=O	6p55_lig",
    "CSCCC(Nc1c(Nc2cc(N(CCCN3CCN(C)CC3)C(=O)Nc4c(Cl)c(cc(c4Cl)OC)OC)ncn2)cccc1)=O	6p69_lig",
    "COC(=O)C(C(C)O)C1NC(C(O)=O)=C(SC2CNC(CNS(N)(=O)=O)C2)C1C	6p9c_lig",
    "COC(C(C1NC(=C(SC2CC(NC2)C(N(C)C)=O)C1C)C(=O)O)C(O)C)=O	6pt1_lig",
    "O=C1c2cc(O)c(O)cc2C(=O)N1CCNC(=O)C=3CN(NC=3C(=O)O)CC(C=O)NC(=O)C(=NOC(C)(C)C(=O)O)c4nc(N)sc4	6vot_lig-r7g",
    "CSC(c1c(CN2C(=O)CN(C)CC2)cc3c(n1)N(CCC3)C(=O)Nc4cc(c(cn4)C#N)NCCOC)O	6yi8_lig",
    "COC(C(C1C(C(SC2CC(C(N(C)C)=O)NC2)C(C(=O)O)=N1)C)C(C)O)=O	6zrp_lig",
    "COC(=O)C=CNCC(=O)CCO	7l5t_lig",
    "COC(C(C1N=C(C(SCCNC=N)C1)C(=O)O)C(O)C)=O	7rp8_lig",
    "COC(C(C1N=C(C(SCC)C1)C(O)=O)C(O)C)=O	7rp9_lig",
    "CSC(C(NC(C(NC(N1CCN(C)CC1)=O)Cc2ccccc2)=O)CCc3ccccc3)CS(=O)(=O)c4ccccc4	8qkb_lig",
    // clang-format on
  };

  std::ifstream ifs(internal::test_data("gh-358.pdb"));
  PDBReader reader(ifs);
  std::vector<std::string> blk;

  int i = 0;
  for (; i < smiles_answers.size() && reader.getnext(blk); ++i) {
    Molecule mol = reader.parse(blk);
    EXPECT_TRUE(internal::guess_update_subs(mol));

    std::string smi = NURI_WRITE_ONCE(write_smiles, mol);
    absl::StripAsciiWhitespace(&smi);

    std::pair<std::string_view, std::string_view> ans_split =
        absl::StrSplit(smiles_answers[i], '\t');
    if (ans_split.second == "6oap_lig-cyc") {
      // Due to tautomerization, any of the two SMILES is correct
      EXPECT_THAT(  //
          smi,      //
          testing::AnyOf(testing::Eq(ans_split.first),
                         testing::Eq(
                             // clang-format off
"C(c1[nH]c(c(c1CCC(=O)O)C)C=C2NC(C(=C2C)CC)=O)=C3N=C(C=C4NC(C(C4CC)C)=O)C(=C3CCC(=O)O)C"
                             // clang-format on
                             ))  //
      );
    } else {
      EXPECT_EQ(smi, ans_split.first)
          << "Different SMILES for " << ans_split.second;
    }
  }

  EXPECT_EQ(i, smiles_answers.size());
}

TEST(GuessSelectedMolecules, GH367) {
  std::vector<std::string_view> smiles_answers {
    // clang-format off
    "COC(C(CN1CC(C(NCCN2C(=O)c3cc(O)c(O)cc3C2=O)=O)=C(C(O)=O)N1)NC(=O)C(=NOC(C)(C)C(O)=O)c4nc(N)sc4)=O	6vot_lig",
    // clang-format on
  };

  std::ifstream ifs(internal::test_data("gh-367.pdb"));
  PDBReader reader(ifs);
  std::vector<std::string> blk;

  int i = 0;
  for (; i < smiles_answers.size() && reader.getnext(blk); ++i) {
    Molecule mol = reader.parse(blk);
    EXPECT_TRUE(internal::guess_update_subs(mol));

    std::string smi = NURI_WRITE_ONCE(write_smiles, mol);
    absl::StripAsciiWhitespace(&smi);

    std::pair<std::string_view, std::string_view> ans_split =
        absl::StrSplit(smiles_answers[i], '\t');
    EXPECT_EQ(smi, ans_split.first)
        << "Different SMILES for " << ans_split.second;
  }

  EXPECT_EQ(i, smiles_answers.size());
}

TEST(GuessFchargeOnly, ChargedPhosphorus) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(kPt[15]);
  }

  mol.atom(0).data().set_implicit_hydrogens(4);

  guess_fcharge_2d(mol);

  EXPECT_EQ(mol[0].data().formal_charge(), +1);
}

TEST(GuessFchargeOnly, Sulfonyl) {
  Molecule mol;
  {
    auto mut = mol.mutator();

    mut.add_atom(kPt[16]);
    mut.add_atom(kPt[8]);
    mut.add_atom(kPt[8]);
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[7]);

    mut.add_bond(0, 1, BondData(constants::kDoubleBond));
    mut.add_bond(0, 2, BondData(constants::kDoubleBond));
    mut.add_bond(0, 3, BondData(constants::kSingleBond));
    mut.add_bond(0, 4, BondData(constants::kSingleBond));
  }

  mol.atom(3).data().set_implicit_hydrogens(3);
  mol.atom(4).data().set_implicit_hydrogens(3);

  guess_fcharge_2d(mol);

  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(mol[i].data().formal_charge(), 0);

  EXPECT_EQ(mol[4].data().formal_charge(), +1);
}

TEST(GuessFchargeOnly, Thiophene) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(kPt[16]);
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);

    mut.add_bond(0, 1,
                 BondData(constants::kSingleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(1, 2,
                 BondData(constants::kDoubleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(2, 3,
                 BondData(constants::kSingleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(3, 4,
                 BondData(constants::kDoubleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(4, 0,
                 BondData(constants::kSingleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
  }

  for (int i = 1; i < 4; ++i)
    mol.atom(i).data().set_implicit_hydrogens(1);

  guess_fcharge_2d(mol);

  for (auto atom: mol)
    EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();

  for (auto bond: mol.bonds())
    bond.data().set_order(constants::kAromaticBond);

  guess_fcharge_2d(mol);

  for (auto atom: mol)
    EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
}

TEST(GuessFchargeOnly, ChargedThiophene) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(kPt[16]);
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);

    mut.add_bond(0, 1,
                 BondData(constants::kSingleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(1, 2,
                 BondData(constants::kDoubleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(2, 3,
                 BondData(constants::kSingleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(3, 4,
                 BondData(constants::kDoubleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
    mut.add_bond(4, 0,
                 BondData(constants::kSingleBond)
                     .add_flags(BondFlags::kAromatic | BondFlags::kConjugated));
  }

  for (auto atom: mol)
    atom.data().set_implicit_hydrogens(1);

  guess_fcharge_2d(mol);

  EXPECT_EQ(mol[0].data().formal_charge(), +1);
  for (int i = 1; i < 4; ++i)
    EXPECT_EQ(mol[i].data().formal_charge(), 0);

  mol[0].data().set_formal_charge(0);
  for (auto bond: mol.bonds())
    bond.data().set_order(constants::kAromaticBond);

  guess_fcharge_2d(mol);

  EXPECT_EQ(mol[0].data().formal_charge(), +1);
  for (int i = 1; i < 4; ++i)
    EXPECT_EQ(mol[i].data().formal_charge(), 0);
}
}  // namespace
}  // namespace nuri

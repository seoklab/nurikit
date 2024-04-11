//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"

namespace {
using std::string_literals::operator""s;

using nuri::AtomData;
using nuri::BondData;
using nuri::Molecule;
using nuri::MoleculeSanitizer;

// NOLINTNEXTLINE(*-using-namespace)
using namespace nuri::constants;

const nuri::PeriodicTable &pt = nuri::PeriodicTable::get();

TEST(Basic2DMoleculeTest, CreationTest) {
  Molecule empty;

  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.num_atoms(), 0);
  EXPECT_EQ(empty.num_bonds(), 0);

  std::vector<AtomData> atoms(1);
  atoms.reserve(10);
  for (int i = 1; i < 10; ++i) {
    atoms.push_back(AtomData(pt[i], 0, 0, kSP3, 0, i));
  }

  Molecule ten(atoms.begin(), atoms.end());

  EXPECT_EQ(ten.size(), 10);
  EXPECT_EQ(ten.num_atoms(), 10);
  EXPECT_EQ(ten.num_bonds(), 0);

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(ten.atom(i).data(), atoms[i]);
  }
}

TEST(Basic2DMoleculeTest, AddBondsTest) {
  std::vector<AtomData> atoms(1);
  atoms.reserve(10);
  for (int i = 1; i < 10; ++i) {
    atoms.push_back(AtomData(pt[i], 0, 0, kSP3, 0, i));
  }

  Molecule ten(atoms.begin(), atoms.end());
  {
    Molecule::bond_iterator bit1, bit2;
    bool success;

    auto mutator = ten.mutator();
    std::tie(bit1, success) = mutator.add_bond(0, 1, BondData(kSingleBond));
    EXPECT_TRUE(success);

    std::tie(bit2, success) = mutator.add_bond(1, 0, BondData(kDoubleBond));
    EXPECT_EQ(bit1, bit2);
    EXPECT_FALSE(success);
  }
  {
    auto mutator = ten.mutator();
    EXPECT_FALSE(mutator.add_bond(1, 0, BondData(kDoubleBond)).second);
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

      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[6] });
      mutator.add_atom({ pt[7], 0, -1 });
      for (int i = 5; i < 11; ++i) {
        mutator.add_atom({ pt[1] });
      }
      mutator.add_atom({ pt[11], 0, +1 });

      mutator.add_bond(0, 1, BondData { kDoubleBond });
      mutator.add_bond(0, 2, BondData { kSingleBond });
      mutator.add_bond(1, 2, BondData { kSingleBond });
      mutator.add_bond(2, 3, BondData { kSingleBond });
      mutator.add_bond(3, 4, BondData { kSingleBond });
      mutator.add_bond(4, 5, BondData { kSingleBond });
      mutator.add_bond(0, 6, BondData { kSingleBond });
      mutator.add_bond(1, 7, BondData { kSingleBond });
      mutator.add_bond(2, 8, BondData { kSingleBond });
      mutator.add_bond(3, 9, BondData { kSingleBond });
      mutator.add_bond(3, 10, BondData { kSingleBond });
    }

    ASSERT_EQ(mol_.num_atoms(), 12);
    ASSERT_EQ(mol_.num_bonds(), 11);

    auto &confs = mol_.confs();

    // The coordinates are generated by corina
    nuri::Matrix3Xd coords1(3, 12);
    coords1.transpose() << -1.1913, 0.5880, -2.2656,  //
        -0.1528, 1.3191, -2.5866,                     //
        -0.7527, 1.5927, -1.2315,                     //
        -0.0127, 1.0858, 0.0080,                      //
        -0.7003, 1.5596, 1.2166,                      //
        -0.2304, 1.2378, 2.0496,                      //
        -1.8821, -0.1932, -2.5467,                    //
        0.6062, 1.5595, -3.3164,                      //
        -1.3450, 2.4981, -1.0997,                     //
        0.0021, -0.0041, 0.0020,                      //
        1.0099, 1.4631, 0.0003,                       //
        6.0099, 7.4981, 7.0496;
    confs.push_back(std::move(coords1));

    nuri::Matrix3Xd coords2(3, 12);
    coords2.transpose() << -2.8265, 2.3341, -0.0040,  //
        -2.4325, 1.6004, 1.0071,                      //
        -1.3392, 2.1190, 0.1089,                      //
        -0.4723, 3.2756, 0.6106,                      //
        -1.2377, 4.5275, 0.5399,                      //
        -0.6865, 5.3073, 0.8661,                      //
        -3.6475, 2.7827, -0.5437,                     //
        -2.7033, 1.0234, 1.8791,                      //
        -0.8591, 1.4229, -0.5788,                     //
        -0.1776, 3.0877, 1.6430,                      //
        0.4184, 3.3588, -0.0122,                      //
        10.1054, 2.3061, 4.3717;
    confs.push_back(std::move(coords2));
  }

  void add_extra_data() {
    mol_.name() = "test molecule";
    mol_.add_prop("key", "val");
    nuri::Substructure &sub =
        mol_.add_substructure(mol_.atom_substructure({ 0, 1, 2 }));
    sub.name() = "test substructure";
  }
};

TEST_F(MoleculeTest, AddAtomsTest) {
  {
    auto mutator = mol_.mutator();
    mutator.add_atom(AtomData(pt[1], 0, 0, kSP3, 0, 2));
  }

  EXPECT_EQ(mol_.num_atoms(), 13);
  EXPECT_EQ(mol_.count_heavy_atoms(), 6);
  for (const nuri::Matrix3Xd &conf: mol_.confs()) {
    EXPECT_EQ(conf.cols(), 13);
  }

  EXPECT_EQ(mol_.atom(mol_.size() - 1).data().atomic_number(), 1);
}

TEST_F(MoleculeTest, AddBonds) {
  {
    auto mutator = mol_.mutator();
    mutator.add_bond(4, 11, {});
  }

  EXPECT_EQ(mol_.num_bonds(), 12);

  auto bit = mol_.find_bond(4, 11);
  ASSERT_NE(bit, mol_.bond_end());
}

TEST_F(MoleculeTest, TransformTest) {
  Eigen::Affine3d trs = Eigen::Affine3d::Identity();

  {
    Molecule mol(mol_);
    ASSERT_EQ(mol.confs().size(), mol_.confs().size());
    mol.transform(trs);
    for (int i = 0; i < mol.confs().size(); ++i) {
      EXPECT_TRUE(mol.confs()[i].isApprox(mol_.confs()[i]));
    }
  }

  trs.translation() << 5, 0, 0;
  trs.linear() = Eigen::AngleAxisd(nuri::deg2rad(90), Eigen::Vector3d::UnitZ())
                     .toRotationMatrix();

  {
    Molecule mol(mol_);
    mol.transform(trs);
    for (int i = 0; i < mol.confs().size(); ++i) {
      EXPECT_TRUE(mol.confs()[i].isApprox(trs * mol_.confs()[i]));
    }
  }

  {
    Molecule mol(mol_);
    mol.transform(1, trs);
    EXPECT_TRUE(mol.confs()[0].isApprox(mol_.confs()[0]));
    EXPECT_TRUE(mol.confs()[1].isApprox(trs * mol_.confs()[1]));
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
  const auto &confs = mol_all.confs();
  for (int i = 0; i < confs.size(); ++i) {
    EXPECT_TRUE(
        confs[i](Eigen::all, fixed).isApprox(confs[i](Eigen::all, fixed)));
  }

  std::vector<int> rotated { 4, 5, 9, 10 };
  Eigen::Matrix3Xd result_coords(3, 4);

  // Rotated with UCSF Chimera
  result_coords.transpose() << -0.5602, -0.2180, 0.4060,  //
      -0.0876, -0.5744, 1.2232,                           //
      1.0480, 0.9803, -0.2199,                            //
      -0.1401, 1.7976, 0.8236;
  EXPECT_TRUE(
      mol_all.confs()[0](Eigen::all, rotated).isApprox(result_coords, 1e-3))
      << mol_all.confs()[0](Eigen::all, rotated);

  // Rotated with UCSF Chimera
  result_coords.transpose() << -0.6589, 3.4347, 2.0590,  //
      -0.0965, 4.1932, 2.4150,                           //
      0.5757, 3.0620, 0.4005,                            //
      -0.7645, 4.1952, 0.1036;

  EXPECT_TRUE(
      mol_all.confs()[1](Eigen::all, rotated).isApprox(result_coords, 1e-3))
      << mol_all.confs()[1](Eigen::all, rotated);

  Molecule mol_one(mol_all);

  // Not rotable
  bid = mol_one.find_bond(0, 1)->id();
  ASSERT_FALSE(mol_one.rotate_bond_conf(1, bid, 90));
  // Rotate reverse!
  ASSERT_TRUE(mol_one.rotate_bond_conf(0, 2, 3, -90));

  EXPECT_TRUE(mol_one.confs()[0].isApprox(mol_.confs()[0]));
  EXPECT_TRUE(mol_one.confs()[1].isApprox(mol_all.confs()[1]));
}

TEST_F(MoleculeTest, EraseAtomsTest) {
  Molecule mol1(mol_);
  {
    auto m = mol1.mutator();
    m.mark_atom_erase(9);
    m.mark_atom_erase(10);
    m.mark_atom_erase(11);
  }
  EXPECT_EQ(mol1.num_atoms(), 9);
  EXPECT_EQ(mol1.num_bonds(), 9);

  for (int i = 0; i < mol1.confs().size(); ++i) {
    EXPECT_TRUE(mol1.confs()[i].isApprox(mol_.confs()[i].leftCols<9>()));
    EXPECT_EQ(mol1.confs()[i].cols(), 9);
  }

  Molecule mol2(mol_);
  {
    auto m = mol2.mutator();
    m.mark_atom_erase(0);
    m.mark_atom_erase(1);
    m.mark_atom_erase(2);
  }
  EXPECT_EQ(mol2.num_atoms(), 9);
  EXPECT_EQ(mol2.num_bonds(), 4);

  for (int i = 0; i < mol2.confs().size(); ++i) {
    EXPECT_TRUE(mol2.confs()[i].isApprox(mol_.confs()[i].rightCols<9>()));
    EXPECT_EQ(mol2.confs()[i].cols(), 9);
  }

  Molecule mol3(mol_);
  {
    auto m = mol3.mutator();
    m.mark_atom_erase(0);
    m.mark_atom_erase(4);
    m.mark_atom_erase(9);
  }
  EXPECT_EQ(mol3.num_atoms(), 9);
  EXPECT_EQ(mol3.num_bonds(), 5);

  std::vector<int> keep { 1, 2, 3, 5, 6, 7, 8, 10, 11 };
  for (int i = 0; i < mol3.confs().size(); ++i) {
    EXPECT_TRUE(mol3.confs()[i].isApprox(mol_.confs()[i](Eigen::all, keep)));
    EXPECT_EQ(mol3.confs()[i].cols(), 9);
  }
}

TEST_F(MoleculeTest, EraseBondsTest) {
  Molecule mol1(mol_);
  {
    // All nop
    auto m = mol1.mutator();
    m.mark_bond_erase(8, 8);
    m.mark_bond_erase(8, 9);
  }
  EXPECT_EQ(mol1.num_atoms(), 12);
  EXPECT_EQ(mol1.num_bonds(), 11);

  Molecule mol2(mol_);
  {
    auto m = mol2.mutator();
    m.mark_bond_erase(0, 1);
  }
  EXPECT_EQ(mol2.num_atoms(), 12);
  EXPECT_EQ(mol2.num_bonds(), 10);

  Molecule mol3(std::move(mol_));
  {
    auto m = mol3.mutator();
    m.mark_bond_erase(0, 1);
    m.mark_bond_erase(8, 9);  // nop
    m.mark_bond_erase(3, 2);
  }
  EXPECT_EQ(mol3.num_atoms(), 12);
  EXPECT_EQ(mol3.num_bonds(), 9);
}

TEST_F(MoleculeTest, EraseHydrogensTest) {
  mol_.erase_hydrogens();

  EXPECT_EQ(mol_.size(), 6);
  for (auto atom: mol_) {
    EXPECT_NE(atom.data().atomic_number(), 1);
  }

  EXPECT_EQ(mol_.num_bonds(), 5);
  for (const auto &c: mol_.confs()) {
    EXPECT_EQ(c.cols(), mol_.size());
  }
}

TEST(EraseNontrivialHydrogensTest, ExplicitH2) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[1]);
    mut.add_atom(pt[1]);
    mut.add_bond(0, 1, BondData(kSingleBond));
  }

  mol.erase_hydrogens();
  EXPECT_EQ(mol.size(), 2);
  EXPECT_EQ(mol.num_bonds(), 1);
}

TEST(EraseNontrivialHydrogensTest, ImplicitH2) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[1]);
  }

  mol[0].data().set_implicit_hydrogens(1);

  mol.erase_hydrogens();
  EXPECT_EQ(mol.size(), 1);
}

TEST(EraseNontrivialHydrogensTest, BridgingH) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[5]);
    mut.add_atom(pt[1]);
    mut.add_atom(pt[5]);
    mut.add_bond(0, 1, BondData(kSingleBond));
    mut.add_bond(1, 2, BondData(kSingleBond));
  }

  mol.erase_hydrogens();
  EXPECT_EQ(mol.size(), 3);
  EXPECT_EQ(mol.num_bonds(), 2);
}

void verify_clear_all(const Molecule &mol) {
  EXPECT_EQ(mol.size(), 0);
  EXPECT_EQ(mol.num_atoms(), 0);
  EXPECT_EQ(mol.num_bonds(), 0);
  EXPECT_EQ(mol.confs().size(), 0);
  EXPECT_EQ(mol.num_fragments(), 0);
  EXPECT_TRUE(mol.name().empty());
  EXPECT_TRUE(mol.props().empty());
  EXPECT_TRUE(mol.substructures().empty());
  EXPECT_TRUE(mol.ring_groups().empty());
}

TEST_F(MoleculeTest, ClearAll) {
  add_extra_data();
  mol_.clear();
  verify_clear_all(mol_);
}

TEST_F(MoleculeTest, ClearAllWithMutator) {
  add_extra_data();

  {
    auto mut = mol_.mutator();
    mut.mark_atom_erase(5);
    mut.mark_bond_erase(5);
    mut.clear();
  }

  verify_clear_all(mol_);
}

void verify_clear_atoms(const Molecule &mol) {
  EXPECT_EQ(mol.size(), 0);
  EXPECT_EQ(mol.num_atoms(), 0);
  EXPECT_EQ(mol.num_bonds(), 0);
  EXPECT_EQ(mol.num_fragments(), 0);
  EXPECT_TRUE(mol.ring_groups().empty());

  EXPECT_EQ(mol.confs().size(), 2);
  for (const auto &c: mol.confs())
    EXPECT_EQ(c.cols(), 0);

  EXPECT_EQ(mol.name(), "test molecule");

  ASSERT_EQ(mol.props().size(), 1);
  EXPECT_EQ(mol.props()[0], std::pair("key"s, "val"s));

  ASSERT_EQ(mol.substructures().size(), 1);
  EXPECT_TRUE(mol.substructures()[0].empty());
  EXPECT_EQ(mol.substructures()[0].name(), "test substructure");
}

TEST_F(MoleculeTest, ClearAtoms) {
  add_extra_data();
  mol_.clear_atoms();
  verify_clear_atoms(mol_);
}

TEST_F(MoleculeTest, ClearAtomsWithMutator) {
  add_extra_data();

  {
    auto mut = mol_.mutator();
    mut.mark_atom_erase(5);
    mut.mark_bond_erase(5);
    mut.clear_atoms();
  }

  verify_clear_atoms(mol_);
}

void verify_clear_bonds(const Molecule &mol, int num_atoms) {
  EXPECT_EQ(mol.num_bonds(), 0);

  EXPECT_EQ(mol.size(), num_atoms);
  EXPECT_EQ(mol.num_atoms(), num_atoms);
  EXPECT_EQ(mol.num_fragments(), num_atoms);
  EXPECT_TRUE(mol.ring_groups().empty());

  EXPECT_EQ(mol.confs().size(), 2);
  for (const auto &c: mol.confs())
    EXPECT_EQ(c.cols(), num_atoms);

  EXPECT_EQ(mol.name(), "test molecule");

  ASSERT_EQ(mol.props().size(), 1);
  EXPECT_EQ(mol.props()[0], std::pair("key"s, "val"s));

  ASSERT_EQ(mol.substructures().size(), 1);
  EXPECT_EQ(mol.substructures()[0].atom_ids(), std::vector<int>({ 0, 1, 2 }));
  EXPECT_EQ(mol.substructures()[0].name(), "test substructure");
}

TEST_F(MoleculeTest, ClearBonds) {
  add_extra_data();
  mol_.clear_bonds();
  verify_clear_bonds(mol_, 12);
}

TEST_F(MoleculeTest, ClearBondsWithMutator) {
  add_extra_data();

  {
    auto mut = mol_.mutator();
    mut.mark_atom_erase(5);
    mut.mark_bond_erase(5);
    mut.clear_bonds();
  }

  verify_clear_bonds(mol_, 11);
}

TEST_F(MoleculeTest, SanitizeTest) {
  MoleculeSanitizer sanitizer(mol_);
  ASSERT_TRUE(sanitizer.sanitize_all());

  EXPECT_EQ(mol_.atom(0).data().hybridization(), kSP2);
  EXPECT_EQ(mol_.atom(1).data().hybridization(), kSP2);
  EXPECT_EQ(mol_.atom(2).data().hybridization(), kSP3);
  EXPECT_EQ(mol_.atom(3).data().hybridization(), kSP3);
  EXPECT_EQ(mol_.atom(4).data().hybridization(), kSP3);
  for (int i = 5; i < 11; ++i) {
    EXPECT_EQ(mol_.atom(i).data().hybridization(), kTerminal);
  }
  EXPECT_EQ(mol_.atom(11).data().hybridization(), kUnbound);

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(mol_.atom(i).data().is_ring_atom());
  }
  for (int i = 3; i < mol_.num_atoms(); ++i) {
    EXPECT_FALSE(mol_.atom(i).data().is_ring_atom());
  }

  for (auto atom: mol_) {
    EXPECT_FALSE(atom.data().is_aromatic());
  }
}

TEST_F(MoleculeTest, MergeOther) {
  Molecule mol;

  {
    auto mut = mol.mutator();
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);

    mut.add_bond(0, 1, BondData(kSingleBond));
  }

  mol_.merge(mol);

  ASSERT_EQ(mol_.num_atoms(), 14);
  ASSERT_EQ(mol_.num_bonds(), 12);

  EXPECT_EQ(mol_.atom(12).data().atomic_number(), 6);
  EXPECT_EQ(mol_.atom(13).data().atomic_number(), 6);

  EXPECT_EQ(mol_.find_bond(12, 13)->data().order(), kSingleBond);
}

TEST_F(MoleculeTest, GetFragments) {
  std::vector fragments = nuri::fragments(mol_);
  EXPECT_EQ(fragments.size(), 2);
  EXPECT_EQ(fragments.size(), mol_.num_fragments());

  bool found_1 = false, found_11 = false;
  for (const auto &f: fragments) {
    if (f.size() == 11) {
      EXPECT_FALSE(found_11) << "Duplicate fragment";
      EXPECT_TRUE(absl::c_all_of(f, [](int i) { return i < 11; }));
      found_11 = true;
    } else if (f.size() == 1) {
      EXPECT_FALSE(found_1) << "Duplicate fragment";
      EXPECT_EQ(f[0], 11);
      found_1 = true;
    } else {
      FAIL() << "Unexpected fragment size: " << f.size();
    }
  }
  EXPECT_TRUE(found_1 && found_11);
}

TEST_F(MoleculeTest, Properties) {
  mol_.add_prop("test", "1");
  auto it = absl::c_find_if(mol_.props(), [](const auto &p) {
    return p == std::pair<std::string, std::string>("test", "1");
  });
  EXPECT_NE(it, mol_.props().end());

  mol_.atom(0).data().add_prop("test", "2");
  it = absl::c_find_if(mol_.atom(0).data().props(), [](const auto &p) {
    return p == std::pair<std::string, std::string>("test", "2");
  });
  EXPECT_NE(it, mol_.atom(0).data().props().end());

  mol_.bond_begin()->data().add_prop("test", "3");
  it = absl::c_find_if(mol_.bond_begin()->data().props(), [](const auto &p) {
    return p == std::pair<std::string, std::string>("test", "3");
  });
  EXPECT_NE(it, mol_.bond_begin()->data().props().end());

  mol_.atom(0).data().set_name("test");
  EXPECT_EQ(mol_.atom(0).data().get_name(), "test");

  mol_.bond_begin()->data().set_name("test");
  EXPECT_EQ(mol_.bond_begin()->data().get_name(), "test");
}

TEST(SanitizeTest, FindRingsTest) {
  Molecule mol;

  // fused cyclopronane - methylcyclopropane
  {
    auto mut = mol.mutator();

    for (int i = 0; i < 8; ++i) {
      mut.add_atom(pt[6]);
    }

    mut.add_bond(0, 3, BondData(kSingleBond));
    mut.add_bond(0, 6, BondData(kSingleBond));
    mut.add_bond(1, 4, BondData(kSingleBond));
    mut.add_bond(1, 5, BondData(kSingleBond));
    mut.add_bond(1, 7, BondData(kSingleBond));
    mut.add_bond(2, 5, BondData(kSingleBond));
    mut.add_bond(3, 4, BondData(kSingleBond));
    mut.add_bond(3, 6, BondData(kSingleBond));
    mut.add_bond(4, 6, BondData(kSingleBond));
    mut.add_bond(5, 7, BondData(kSingleBond));

    for (int i: { 1, 3, 4, 5, 6 }) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
    for (int i: { 0, 7 }) {
      mol.atom(i).data().set_implicit_hydrogens(2);
    }
    mol.atom(2).data().set_implicit_hydrogens(3);
  }
}

TEST(SanitizeTest, ConjugatedTest) {
  Molecule mol;

  // acetic acid
  {
    auto mut = mol.mutator();

    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);

    mut.add_bond(0, 1, BondData(kSingleBond));
    mut.add_bond(0, 2, BondData(kDoubleBond));
    mut.add_bond(0, 3, BondData(kSingleBond));

    mol.atom(1).data().set_implicit_hydrogens(3);
    mol.atom(3).data().set_implicit_hydrogens(1);
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  EXPECT_EQ(mol.atom(2).data().hybridization(), kTerminal);
  EXPECT_EQ(mol.atom(2).data().is_conjugated(), true);
  for (auto atom: mol) {
    if (atom.id() == 2) {
      continue;
    }

    EXPECT_EQ(atom.data().hybridization(), atom.id() == 1 ? kSP3 : kSP2);
    EXPECT_EQ(atom.data().is_conjugated(), atom.id() != 1);
  }

  mol.clear();

  // pyrrole
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[7]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);

    for (int i = 0; i < 5; ++i) {
      mut.add_bond(i, (i + 1) % 5, BondData(kAromaticBond));
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_EQ(atom.data().hybridization(), kSP2);
    EXPECT_TRUE(atom.data().is_conjugated());
  }

  for (int i = 0; i < 5; ++i) {
    mol.find_bond(i, (i + 1) % 5)->data().order() = kAromaticBond;
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_EQ(atom.data().hybridization(), kSP2);
    EXPECT_TRUE(atom.data().is_conjugated());
  }

  mol.clear();

  // HC(2) # C(3) - CH(1) = C(0) = CH(4) - CH3(5)
  {
    auto mut = mol.mutator();
    for (int i = 0; i < 6; ++i) {
      mut.add_atom(pt[6]);
    }
    mut.add_bond(0, 4, BondData(kDoubleBond));
    mut.add_bond(0, 1, BondData(kDoubleBond));
    mut.add_bond(1, 3, BondData(kSingleBond));
    mut.add_bond(3, 2, BondData(kTripleBond));
    mut.add_bond(4, 5, BondData(kSingleBond));

    for (int i: { 1, 2, 4 }) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
    mol.atom(5).data().set_implicit_hydrogens(3);
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_EQ(atom.data().is_conjugated(), atom.id() < 4) << atom.id();

    if (atom.id() == 1 || atom.id() == 4) {
      EXPECT_EQ(atom.data().hybridization(), kSP2);
    } else if (atom.id() == 5) {
      EXPECT_EQ(atom.data().hybridization(), kSP3);
    } else {
      EXPECT_EQ(atom.data().hybridization(), kSP);
    }
  }

  mol.clear();

  // H2N(2) - NH(1) - NH(4) - N(0) = CH2(3)
  {
    auto mut = mol.mutator();

    mut.add_atom(pt[7]);
    mut.add_atom(pt[7]);
    mut.add_atom(pt[7]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[7]);
    mut.add_bond(0, 3, BondData(kDoubleBond));
    mut.add_bond(0, 4, BondData(kSingleBond));
    mut.add_bond(1, 2, BondData(kSingleBond));
    mut.add_bond(1, 4, BondData(kSingleBond));

    for (int i: { 1, 4 }) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
    for (int i: { 2, 3 }) {
      mol.atom(i).data().set_implicit_hydrogens(2);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_EQ(atom.data().is_conjugated(), atom.id() != 1 && atom.id() != 2)
        << atom.id();
    EXPECT_EQ(atom.data().hybridization(),
              atom.id() == 1 || atom.id() == 2 ? kSP3 : kSP2);
  }
}

TEST(SanitizeTest, AromaticTest) {
  Molecule mol;

  // pyrrole
  {
    auto mut = mol.mutator();

    mut.add_atom(pt[7]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);

    mut.add_bond(0, 1, BondData(kSingleBond));
    mut.add_bond(1, 3, BondData(kDoubleBond));
    mut.add_bond(3, 2, BondData(kSingleBond));
    mut.add_bond(2, 4, BondData(kDoubleBond));
    mut.add_bond(4, 0, BondData(kSingleBond));

    for (int i = 0; i < 5; ++i) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_TRUE(atom.data().is_aromatic());
    EXPECT_EQ(atom.data().hybridization(), kSP2);
  }
  for (auto bond: mol.bonds()) {
    EXPECT_TRUE(bond.data().is_aromatic());
  }

  for (int i = 0; i < 5; ++i) {
    mol.find_bond(0, 1)->data().order() = kAromaticBond;
    mol.find_bond(1, 3)->data().order() = kAromaticBond;
    mol.find_bond(3, 2)->data().order() = kAromaticBond;
    mol.find_bond(2, 4)->data().order() = kAromaticBond;
    mol.find_bond(4, 0)->data().order() = kAromaticBond;
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_TRUE(atom.data().is_aromatic());
    EXPECT_EQ(atom.data().hybridization(), kSP2);
  }
  for (auto bond: mol.bonds()) {
    EXPECT_TRUE(bond.data().is_aromatic());
  }

  mol.clear();

  // [bH]1cccc1, NOT aromatic
  {
    auto mut = mol.mutator();

    mut.add_atom(pt[5]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);

    mut.add_bond(0, 1, BondData(kSingleBond));
    mut.add_bond(1, 2, BondData(kDoubleBond));
    mut.add_bond(2, 3, BondData(kSingleBond));
    mut.add_bond(3, 4, BondData(kDoubleBond));
    mut.add_bond(4, 0, BondData(kSingleBond));

    for (int i = 0; i < 5; ++i) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_FALSE(atom.data().is_aromatic());
    EXPECT_EQ(atom.data().hybridization(), kSP2);
  }
  for (auto bond: mol.bonds()) {
    EXPECT_FALSE(bond.data().is_aromatic());
  }

  for (int i = 0; i < 5; ++i) {
    mol.find_bond(i, (i + 1) % 5)->data() = BondData(kAromaticBond);
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  mol.clear();

  // Benzoquinone, not aromatic
  {
    auto mut = mol.mutator();

    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);

    mut.add_bond(0, 1, BondData(kSingleBond));
    mut.add_bond(1, 2, BondData(kDoubleBond));
    mut.add_bond(2, 3, BondData(kSingleBond));
    mut.add_bond(3, 4, BondData(kSingleBond));
    mut.add_bond(4, 5, BondData(kDoubleBond));
    mut.add_bond(5, 0, BondData(kSingleBond));
    mut.add_bond(0, 6, BondData(kDoubleBond));
    mut.add_bond(3, 7, BondData(kDoubleBond));

    for (int i: { 1, 2, 4, 5 }) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_FALSE(atom.data().is_aromatic());
    EXPECT_EQ(atom.data().hybridization(),
              atom.data().atomic_number() == 8 ? kTerminal : kSP2);
  }

  for (int i = 0; i < 6; ++i) {
    mol.find_bond(i, (i + 1) % 6)->data() = BondData(kAromaticBond);
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }
}

TEST(SanitizeTest, FusedAromaticTest) {
  Molecule mol;

  // benzene-cyclobutadiene-benzene fused ring
  const auto verify_bcb = [&mol]() {
    {
      MoleculeSanitizer sanitizer(mol);
      ASSERT_TRUE(sanitizer.sanitize_all());
    }

    for (auto atom: mol) {
      EXPECT_TRUE(atom.data().is_aromatic());
    }
    for (auto bond: mol.bonds()) {
      auto [src, dst] = nuri::minmax(bond.src().id(), bond.dst().id());
      if ((src == 0 && dst == 11) || (src == 5 && dst == 6)) {
        EXPECT_FALSE(bond.data().is_aromatic());
      } else {
        EXPECT_TRUE(bond.data().is_aromatic());
      }
    }
  };

  {
    auto mut = mol.mutator();
    for (int i = 0; i < 12; ++i) {
      mut.add_atom(pt[6]);
      if (i != 0 && i != 5 && i != 6 && i != 11) {
        mol.atom(i).data().set_implicit_hydrogens(1);
      }
    }
    for (int i = 0; i < 6; ++i) {
      mut.add_bond(i, (i + 1) % 6, BondData(kAromaticBond));
      mut.add_bond(i + 6, (i + 1) % 6 + 6, BondData(kAromaticBond));
    }
    mut.add_bond(0, 11, BondData(kSingleBond));
    mut.add_bond(5, 6, BondData(kSingleBond));
  }
  verify_bcb();

  // kekulized version

  mol.find_bond(1, 2)->data().order() = mol.find_bond(3, 4)->data().order() =
      mol.find_bond(5, 0)->data().order() =
          mol.find_bond(6, 7)->data().order() =
              mol.find_bond(8, 9)->data().order() =
                  mol.find_bond(10, 11)->data().order() = kDoubleBond;

  mol.find_bond(0, 1)->data().order() = mol.find_bond(2, 3)->data().order() =
      mol.find_bond(4, 5)->data().order() =
          mol.find_bond(7, 8)->data().order() =
              mol.find_bond(9, 10)->data().order() =
                  mol.find_bond(11, 6)->data().order() = kSingleBond;

  verify_bcb();

  mol.clear();

  const auto verify_azulene = [&mol]() {
    {
      MoleculeSanitizer sanitizer(mol);
      ASSERT_TRUE(sanitizer.sanitize_all());
    }

    for (auto atom: mol) {
      EXPECT_TRUE(atom.data().is_aromatic());
    }
    for (auto bond: mol.bonds()) {
      auto [src, dst] = nuri::minmax(bond.src().id(), bond.dst().id());
      if (src == 0 && dst == 6) {
        EXPECT_FALSE(bond.data().is_aromatic());
      } else {
        EXPECT_TRUE(bond.data().is_aromatic());
      }
    }
  };

  // Azulene
  {
    auto mut = mol.mutator();
    for (int i = 0; i < 10; ++i) {
      mut.add_atom(pt[6]);
      if (i != 0 && i != 6) {
        mol.atom(i).data().set_implicit_hydrogens(1);
      }
    }
    for (int i = 0; i < 6; ++i) {
      ASSERT_TRUE(mut.add_bond(i, i + 1, BondData(kAromaticBond)).second);
    }
    ASSERT_TRUE(mut.add_bond(6, 0, BondData(kSingleBond)).second);
    for (int i = 6; i < 10; ++i) {
      ASSERT_TRUE(
          mut.add_bond(i, (i + 1) % 10, BondData(kAromaticBond)).second);
    }
  }
  verify_azulene();

  // kekulized version
  mol.find_bond(1, 2)->data().order() = mol.find_bond(3, 4)->data().order() =
      mol.find_bond(5, 6)->data().order() =
          mol.find_bond(7, 8)->data().order() =
              mol.find_bond(9, 0)->data().order() = kDoubleBond;

  mol.find_bond(0, 1)->data().order() = mol.find_bond(2, 3)->data().order() =
      mol.find_bond(4, 5)->data().order() =
          mol.find_bond(6, 7)->data().order() =
              mol.find_bond(8, 9)->data().order() = kSingleBond;
  verify_azulene();
}

TEST(SanitizeTest, NonstandardTest) {
  // Not main group element, skipped verification
  Molecule mol;

  // MnO4-
  {
    auto mut = mol.mutator();
    mut.add_atom({ pt[25], 0, -1 });
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);

    mut.add_bond(0, 1, BondData(kDoubleBond));
    mut.add_bond(0, 2, BondData(kDoubleBond));
    mut.add_bond(0, 3, BondData(kDoubleBond));
    mut.add_bond(0, 4, BondData(kDoubleBond));
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  EXPECT_EQ(mol.atom(0).data().hybridization(), kSP3);

  mol.clear();

  // pyrrole, but with radical at N: not aromatic
  {
    auto mut = mol.mutator();
    mut.add_atom({ pt[7], 0, 1 });
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);

    for (int i = 0; i < 5; ++i) {
      mut.add_bond(i, (i + 1) % 5, BondData(kAromaticBond));
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_FALSE(atom.data().is_aromatic());
  }

  mol.clear();

  // pyrrole-like, with dummy atom: aromatic
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[0]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);

    for (int i = 0; i < 5; ++i) {
      mut.add_bond(i, (i + 1) % 5, BondData(kAromaticBond));
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    ASSERT_TRUE(sanitizer.sanitize_all());
  }

  for (auto atom: mol) {
    EXPECT_TRUE(atom.data().is_aromatic());
  }
}

TEST(SanitizeTest, ErrorMolTest) {
  Molecule mol;

  // > 4 bonds for period 2 atom
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[6]);
    mol.atom(0).data().set_implicit_hydrogens(5);
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  mol.clear();

  // O3 with SO2 like bond
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);
    mut.add_bond(0, 1, BondData(kDoubleBond));
    mut.add_bond(0, 2, BondData(kDoubleBond));
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  mol.clear();

  // Linear molecule has aromatic bonds
  {
    auto mut = mol.mutator();
    for (int i = 0; i < 4; ++i) {
      mut.add_atom(pt[6]);
    }
    mut.add_bond(0, 1, BondData(kAromaticBond));
    mut.add_bond(1, 2, BondData(kAromaticBond));
    mut.add_bond(2, 3, BondData(kAromaticBond));
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  mol.clear();

  // CH4+
  {
    auto mut = mol.mutator();
    mut.add_atom({ pt[6], 4, 1 });
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  mol.clear();

  // CH+12, invalid
  {
    auto mut = mol.mutator();
    mut.add_atom({ pt[6], 1, 12 });
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  mol.clear();

  // [cH+12]1ccccc1, invalid
  {
    auto mut = mol.mutator();
    mut.add_atom({ pt[6], 1, 12 });
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);
    mut.add_atom(pt[6]);

    mut.add_bond(0, 1, BondData(kAromaticBond));
    mut.add_bond(1, 2, BondData(kAromaticBond));
    mut.add_bond(2, 3, BondData(kAromaticBond));
    mut.add_bond(3, 4, BondData(kAromaticBond));
    mut.add_bond(4, 5, BondData(kAromaticBond));
    mut.add_bond(5, 0, BondData(kAromaticBond));

    for (int i = 0; i < 6; ++i) {
      mol.atom(i).data().set_implicit_hydrogens(1);
    }
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }

  mol.clear();

  // Mn(=O)4, too many bonds for Mn
  {
    auto mut = mol.mutator();
    mut.add_atom(pt[25]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);
    mut.add_atom(pt[8]);

    mut.add_bond(0, 1, BondData(kDoubleBond));
    mut.add_bond(0, 2, BondData(kDoubleBond));
    mut.add_bond(0, 3, BondData(kDoubleBond));
    mut.add_bond(0, 4, BondData(kDoubleBond));
  }

  {
    MoleculeSanitizer sanitizer(mol);
    EXPECT_FALSE(sanitizer.sanitize_all());
  }
}
}  // namespace

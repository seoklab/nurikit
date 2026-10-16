//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/param/radii.h"

#include <gtest/gtest.h>

#include "test_utils.h"
#include "nuri/core/molecule.h"

namespace nuri {
namespace {
using internal::read_first;

constexpr ConnollyRadii kRadii {};

TEST(ConnollyRadiiTest, Backbone) {
  EXPECT_DOUBLE_EQ(kRadii("ALA", "N"), 1.65);
  EXPECT_DOUBLE_EQ(kRadii("ALA", "CA"), 1.85);
  EXPECT_DOUBLE_EQ(kRadii("ALA", "C"), 1.80);
  EXPECT_DOUBLE_EQ(kRadii("ALA", "O"), 1.60);
  EXPECT_DOUBLE_EQ(kRadii("ALA", "OXT"), 1.60);
  EXPECT_DOUBLE_EQ(kRadii("GLY", "CA"), 1.90);
  EXPECT_DOUBLE_EQ(kRadii("PRO", "N"), 1.65);
}

TEST(ConnollyRadiiTest, Sidechain) {
  EXPECT_DOUBLE_EQ(kRadii("ALA", "CB"), 1.95);
  EXPECT_DOUBLE_EQ(kRadii("VAL", "CB"), 1.85);
  EXPECT_DOUBLE_EQ(kRadii("LYS", "CE"), 1.90);
  EXPECT_DOUBLE_EQ(kRadii("LYS", "NZ"), 1.75);
  EXPECT_DOUBLE_EQ(kRadii("ARG", "NE"), 1.65);
  EXPECT_DOUBLE_EQ(kRadii("ARG", "NH1"), 1.70);
  EXPECT_DOUBLE_EQ(kRadii("ASP", "OD2"), 1.60);
  EXPECT_DOUBLE_EQ(kRadii("SER", "OG"), 1.70);
  EXPECT_DOUBLE_EQ(kRadii("CYS", "SG"), 1.90);
  EXPECT_DOUBLE_EQ(kRadii("MET", "SD"), 1.90);
  EXPECT_DOUBLE_EQ(kRadii("MSE", "SE"), 1.90);
}

TEST(ConnollyRadiiTest, AromaticRing) {
  EXPECT_DOUBLE_EQ(kRadii("PHE", "CG"), 1.80);
  EXPECT_DOUBLE_EQ(kRadii("PHE", "CZ"), 1.90);
  EXPECT_DOUBLE_EQ(kRadii("TYR", "CZ"), 1.80);
  EXPECT_DOUBLE_EQ(kRadii("TYR", "OH"), 1.70);
  EXPECT_DOUBLE_EQ(kRadii("TRP", "CD2"), 1.80);
  EXPECT_DOUBLE_EQ(kRadii("TRP", "CH2"), 1.90);
  EXPECT_DOUBLE_EQ(kRadii("TRP", "NE1"), 1.65);
  EXPECT_DOUBLE_EQ(kRadii("HIS", "CE1"), 1.90);
  EXPECT_DOUBLE_EQ(kRadii("HIS", "ND1"), 1.65);
}

TEST(ConnollyRadiiTest, UnknownNames) {
  EXPECT_LT(kRadii("HOH", "O"), 0);
  EXPECT_LT(kRadii("ALA", "CG"), 0);
  EXPECT_LT(kRadii("ALA", "HA"), 0);
  EXPECT_LT(kRadii("", ""), 0);
}

TEST(ConnollyRadiiTest, MoleculeCarbon) {
  const Molecule mol = read_first("smi", "Cc1ccccc1");
  EXPECT_DOUBLE_EQ(kRadii(mol[0]), 1.95);
  EXPECT_DOUBLE_EQ(kRadii(mol[1]), 1.80);
  EXPECT_DOUBLE_EQ(kRadii(mol[2]), 1.90);

  const Molecule ethane = read_first("smi", "[H]C([H])([H])C");
  EXPECT_LT(kRadii(ethane[0]), 0);
  EXPECT_DOUBLE_EQ(kRadii(ethane[1]), 1.95);
  EXPECT_DOUBLE_EQ(kRadii(ethane[4]), 1.95);

  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "CC(C)C")[1]), 1.85);
  EXPECT_LT(kRadii(read_first("smi", "C")[0]), 0);
}

TEST(ConnollyRadiiTest, MoleculeHeteroatoms) {
  const Molecule mol = read_first("smi", "NCC(=O)NC(=O)[O-]");
  EXPECT_DOUBLE_EQ(kRadii(mol[0]), 1.70);
  EXPECT_DOUBLE_EQ(kRadii(mol[1]), 1.90);
  EXPECT_DOUBLE_EQ(kRadii(mol[3]), 1.60);
  EXPECT_DOUBLE_EQ(kRadii(mol[4]), 1.65);

  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "C[NH3+]")[1]), 1.75);
  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "CO")[1]), 1.70);
  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "CS")[1]), 1.90);
  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "CSC")[1]), 1.90);
  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "C[Se]C")[1]), 1.90);
  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "OP(=O)(O)O")[1]), 1.80);
  EXPECT_DOUBLE_EQ(kRadii(read_first("smi", "c1cc[nH]c1")[3]), 1.65);

  EXPECT_LT(kRadii(read_first("smi", "O")[0]), 0);
  EXPECT_LT(kRadii(read_first("smi", "[NH4+]")[0]), 0);
  EXPECT_LT(kRadii(read_first("smi", "[Na+]")[0]), 0);
}
}  // namespace
}  // namespace nuri

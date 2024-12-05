//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/crdgen.h"

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/smiles.h"

namespace nuri {
namespace {
TEST(Crdgen, CHEMBL2228334) {
  Molecule mol = read_smiles({ "CC(=O)OC1CCCC2COC(=O)C21" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  Matrix3Xd &conf = mol.confs().emplace_back(3, mol.num_atoms());
  ASSERT_TRUE(generate_coords(mol, conf));

  // XXX: The algorithm is basically deterministic on first pass, so just
  // compare it with the reference output. Should be changed later on?
  MatrixX3d ans {
    {  -3.47601, -0.0353863,   -0.915114 },
    {  -2.68436,   0.614118,    0.208193 },
    {  -3.18419,   0.779729,     1.32576 },
    {  -1.50221,    1.07873,   0.0670281 },
    { -0.477185,   0.435378,     -0.6758 },
    {   0.47463,     1.5616,    -1.04135 },
    {   1.27018,    2.05112,    0.159791 },
    {    1.9832,   0.912623,    0.871179 },
    {   1.05167,  -0.226333,     1.23773 },
    {    2.0347,   -1.38704,     1.33505 },
    {    2.1559,   -1.84728, -0.00663011 },
    {   1.16153,   -1.48779,   -0.732103 },
    {   1.02281,   -1.77099,     -1.9257 },
    {  0.169328,  -0.678476,   0.0919661 },
  };

  NURI_EXPECT_EIGEN_EQ_TOL(conf.transpose(), ans, 5e-2);
}

TEST(Crdgen, CHEMBL2228334Chiral) {
  Molecule mol = read_smiles({ "CC(=O)O[C@H]1CCC[C@@H]2COC(=O)[C@@H]21" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  Matrix3Xd &conf = mol.confs().emplace_back(3, mol.num_atoms());
  ASSERT_TRUE(generate_coords(mol, conf));

  // XXX: The algorithm is basically deterministic on first pass, so just
  // compare it with the reference output. Should be changed later on?
  MatrixX3d ans {
    {  -3.78139,   1.52978, -0.509079 },
    {  -2.73636,  0.564952, 0.0270455 },
    {  -2.94067, -0.630358,  0.263047 },
    {  -1.55534,  0.979773,  0.284738 },
    { -0.471054,  0.501119, -0.497313 },
    {  0.478089,   1.65228, -0.783571 },
    {   1.31437,   2.02058,  0.433091 },
    {   2.03983,  0.815629,   1.00967 },
    {   1.11091, -0.348428,   1.29502 },
    {   2.08704,  -1.51812,   1.25001 },
    {   2.16303,  -1.84824, -0.132682 },
    {   1.14966,    -1.416,  -0.78892 },
    {  0.951933,  -1.61857,  -1.99079 },
    {  0.189955, -0.684386,  0.139736 },
  };

  NURI_EXPECT_EIGEN_EQ_TOL(conf.transpose(), ans, 5e-2);
}
}  // namespace
}  // namespace nuri
